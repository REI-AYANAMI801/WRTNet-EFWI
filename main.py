# %%
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple
import random
from decimal import Decimal
import deepwave
import warnings
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import time
import shutil
import pywt
import ptwt
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Select one model definition and keep the other import commented out.
from WRTNet import Physics_deepwave, Transfomerdecoder  # WRTNet with wavelet refinement
# from networks_transformer import Physics_deepwave, Transfomerdecoder  # Transformer baseline with plain fusion
# %%
gpu_count = torch.cuda.device_count()
print(f"The number of available GPUs is: {gpu_count}")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"The selected GPU device is: {torch.cuda.get_device_name(DEVICE)}")
else:
    DEVICE = torch.device("cpu")
    print("No available GPUs detected, switched to using CPU")


# %%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


def wavelet_multiscale_l1_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    levels=(1, 2),
    wA=1.0,
    wD=0.3,
    per_level_decay=0.7,
    normalize=True,
    eps=1e-8,
):
    """
    pred, obs: [Ns, Nt, Nrec]
    """
    if pred.dim() != 3 or obs.dim() != 3:
        raise ValueError(f"need 3D [Ns,Nt,Nrec], got {pred.shape} and {obs.shape}")
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, obs {obs.shape}")

    Ns = pred.shape[0]
    total = pred.new_tensor(0.0)
    denom = (torch.norm(obs, p=2) + eps) if normalize else pred.new_tensor(1.0)

    for s in range(Ns):
        g_pred = pred[s]  # [Nt, Nrec]
        g_obs  = obs[s]

        for li, L in enumerate(levels):
            coeffs_p = ptwt.wavedec2(g_pred, pywt.Wavelet("haar"), level=L, mode="zero")
            coeffs_o = ptwt.wavedec2(g_obs,  pywt.Wavelet("haar"), level=L, mode="zero")

            cA_p = coeffs_p[0].squeeze(0) if coeffs_p[0].dim() == 3 else coeffs_p[0]
            cA_o = coeffs_o[0].squeeze(0) if coeffs_o[0].dim() == 3 else coeffs_o[0]

            (cH_p, cV_p, cD_p) = coeffs_p[1]
            (cH_o, cV_o, cD_o) = coeffs_o[1]

            if cH_p.dim() == 3:
                cH_p, cV_p, cD_p = cH_p.squeeze(0), cV_p.squeeze(0), cD_p.squeeze(0)
            if cH_o.dim() == 3:
                cH_o, cV_o, cD_o = cH_o.squeeze(0), cV_o.squeeze(0), cD_o.squeeze(0)

            wl = (per_level_decay ** li)

            loss_A = F.l1_loss(cA_p, cA_o, reduction="mean")
            loss_D = (
                F.l1_loss(cH_p, cH_o, reduction="mean")
                + F.l1_loss(cV_p, cV_o, reduction="mean")
                + F.l1_loss(cD_p, cD_o, reduction="mean")
            )

            total = total + wl * (wA * loss_A + wD * loss_D)

    total = total / Ns
    total = total / denom
    return total
def masked_wavelet_loss(pred, obs, keep_r,
                        normalize=True, eps=1e-8,
                        **wavelet_kwargs):
    """
    pred/obs: [Ns, Nt, Nr]
    keep_r: [1, 1, 1, Nr] or [Ns, 1, Nr]
    normalize: normalize by the energy of the masked observations
    """
    if keep_r.dim() == 4:
        w = keep_r.squeeze(0)          # [1,1,Nr]
    elif keep_r.dim() == 3:
        w = keep_r                     # [Ns,1,Nr]
    else:
        raise ValueError(f"keep_r dim must be 3 or 4, got {keep_r.shape}")

    # broadcast -> [Ns,Nt,Nr]
    w = w.expand(pred.shape[0], 1, pred.shape[2]).expand_as(pred)

    # Compute residuals only at selected receiver locations.
    res = (pred - obs) * w

    # Penalize the masked residual in the wavelet domain.
    loss = wavelet_multiscale_l1_loss(
        res, torch.zeros_like(res),
        normalize=False,
        **wavelet_kwargs
    )

    # Optionally normalize by the masked observed-data energy.
    if normalize:
        denom = torch.norm(obs * w, p=2) + eps
        loss = loss / denom

    return loss


def make_receiver_mask(nr: int, drop_ratio: float, device, same_for_all_shots=True):
    """
    Returns keep_r with shape [1, 1, 1, Nr].
    A value of 1 includes a receiver in the misfit; 0 excludes it.
    """
    keep = (torch.rand(1, 1, 1, nr, device=device) > drop_ratio).float()
    return keep

def Downsample(img, aim_height, aim_width):
    """
    img: [C, H, W]
    """
    channel, height, width = img.shape
    empty_img = torch.zeros((channel, aim_height, aim_width), device=img.device, dtype=img.dtype)
    transform_h = aim_height / height
    transform_w = aim_width / width
    for i in range(aim_height):
        for j in range(aim_width):
            x = int(i / transform_h)
            y = int(j / transform_w)
            empty_img[:, i, j] = img[:, x, y]
    return empty_img


def resample_time(x_shot_t_rec: torch.Tensor, out_nt: int):
    """
    x_shot_t_rec: [Ns, Nt, Nrec]
    Linearly resamples the time dimension to produce [Ns, out_nt, Nrec].
    """
    Ns, Nt, Nrec = x_shot_t_rec.shape
    x = x_shot_t_rec.permute(0, 2, 1).contiguous()   # [Ns, Nrec, Nt]
    x = x.unsqueeze(1)                               # [Ns, 1, Nrec, Nt]
    x = F.interpolate(x, size=(Nrec, out_nt), mode="bilinear", align_corners=False)
    x = x.squeeze(1).permute(0, 2, 1).contiguous()   # [Ns, out_nt, Nrec]
    return x


def get_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


# %%
def train_engine(
    transfomerdecoder, physics, optim_transfomerdecoder,
    rho_initial, vx_initial, vy_initial,
    d_obs_vx, d_obs_vy, input_vx, input_vy,
    batch, mini_batches,
    submarine, submarine_deep, submarine_vp, submarine_vs, submarine_rho,
    vp_scale, vs_scale, rho_scale,
    loss_nt: int,
    keep_r=None,
    do_backward=True,
):

    earth_model_vp, earth_model_vs, earth_model_rho = transfomerdecoder(input_vx, input_vy)
    device = earth_model_vp.device

    vp_pred = earth_model_vp.squeeze(0).squeeze(0)  # [H,W]
    vs_pred = earth_model_vs.squeeze(0).squeeze(0)
    rho_pred = earth_model_rho.squeeze(0).squeeze(0)

    if vp_pred.shape != vx_initial.shape:
        raise RuntimeError(f"vp pred {vp_pred.shape} != init {vx_initial.shape}")

    vp = vp_pred * vp_scale + vx_initial
    vs = vs_pred * vs_scale + vy_initial
    rho = rho_pred * rho_scale + rho_initial

    if submarine == "yes":
        vp[:submarine_deep, :]  = submarine_vp
        vs[:submarine_deep, :]  = submarine_vs
        rho[:submarine_deep, :] = submarine_rho

    vp  = vp.to(device).requires_grad_(True)
    vs  = vs.to(device).requires_grad_(True)
    rho = rho.to(device).requires_grad_(True)
    # Enforce physically valid parameter ranges before forward modeling.
    vp = torch.clamp(vp, min=500.0, max=8000.0)
    vs = torch.clamp(vs, min=0.0, max=5000.0)
    rho = torch.clamp(rho, min=500.0, max=3500.0)
    # physics output: each [1, Ns, NT, Nrec]
    vx_pred, vy_pred = physics(vp, vs, rho)  # vx_pred: [5, NT, 199], vy_pred: [5, NT, 199]
    # Combine both predicted components along the shot dimension.
    pred_all = torch.cat([vx_pred, vy_pred], dim=0)
    # Select the shots assigned to the current minibatch.
    d_obs_vx_filtered = d_obs_vx[0, batch::mini_batches]  # [5, 401, 199]
    d_obs_vy_filtered = d_obs_vy[0, batch::mini_batches]  # [5, 401, 199]
    # Combine observed components in the same order as the predictions.
    obs_all = torch.cat([d_obs_vx_filtered, d_obs_vy_filtered], dim=0)
    # Match the predicted time dimension to the loss sampling.
    pred_all = resample_time(pred_all, out_nt=loss_nt)
    # --- build minibatch receiver mask (current minibatch shots) ---
    if keep_r is None:
        # Use all receivers when no mask is provided.
        keep_r_mb = torch.ones(1, 1, 1, pred_all.shape[-1], device=pred_all.device)
    else:
        keep_r_mb = keep_r  # [1,1,1,Nr]

    # wavelet misfit on masked residual
    loss = masked_wavelet_loss(
        pred_all, obs_all, keep_r_mb,
        levels=(1, 2),
        wA=1.0, wD=0.3,
        per_level_decay=0.7,
        normalize=True
    )

    if do_backward:
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            transfomerdecoder.parameters(),
            max_norm=1.0,
            error_if_nonfinite=False,
        )

        if torch.isfinite(grad_norm):
            optim_transfomerdecoder.step()
        else:
            print(
                "[WARNING] Non-finite gradient detected. "
                "Skip this optimizer step."
            )
            optim_transfomerdecoder.zero_grad(set_to_none=True)

    return loss.item(), vp, vs, rho


def train_deepwave(
    Physics,
    transfomerdecoder,
    deepwave_size,
    rho_initial,
    vx_initial,
    vy_initial,
    d_obs_vx,
    d_obs_vy,
    input_vx,
    input_vy,
    optim_transfomerdecoder,
    mini_batches,
    src_loc,
    rec_loc,
    src,
    inpa,
    submarine,
    vp_scale,
    vs_scale,
    rho_scale,
    submarine_deep,
    submarine_vp,
    submarine_vs,
    submarine_rho,
    loss_nt: int,
    keep_r=None,
    do_backward=True,
):

    submarine_deep = int(submarine_deep)
    loss_data_minibatch = []

    for batch in range(mini_batches):
        if do_backward:
            optim_transfomerdecoder.zero_grad()

        src_loc_batch = src_loc[batch::mini_batches]
        rec_loc_batch = rec_loc[batch::mini_batches]
        src_batch = src[batch::mini_batches]

        physics = Physics(
            inpa['dh'], inpa['dt'], inpa['fdom'],
            size=deepwave_size,     # Use the full time axis for wave propagation.
            src=src_batch,
            src_loc=src_loc_batch,
            rec_loc=rec_loc_batch
        ).to(DEVICE)

        loss_data, mp, ms, mrho = train_engine(
            transfomerdecoder, physics, optim_transfomerdecoder,
            rho_initial, vx_initial, vy_initial,
            d_obs_vx, d_obs_vy, input_vx, input_vy,
            batch, mini_batches,
            submarine, submarine_deep, submarine_vp, submarine_vs, submarine_rho,
            vp_scale, vs_scale, rho_scale,
            loss_nt=loss_nt,
            keep_r=keep_r,
            do_backward=do_backward,
        )

        loss_data_minibatch.append(loss_data)

    return float(np.mean(loss_data_minibatch)), mp, ms, mrho, transfomerdecoder


# %%
# =========================
# Parameters
# =========================
Physics = Physics_deepwave
BATCH_SIZE = 1

model_shape = [180, 400]
submarine = "no"
submarine_deep = 16
submarine_vp = 1500
submarine_vs = 0
submarine_rho = 1009

BASE_NPY_DIR = os.path.join("data", "model")

VP_NPY  = os.path.join(BASE_NPY_DIR, "vp_arid_shallow.npy")
VS_NPY  = os.path.join(BASE_NPY_DIR, "vs_arid_shallow.npy")
RHO_NPY = os.path.join(BASE_NPY_DIR, "rho_arid_shallow.npy")

time_tag = time.strftime("%Y%m%d-%H%M%S")
model_name = "arid_shallow"
result_dir_name = f"clean_TEST_{time_tag}"

T = 6.0
DT = 0.003
DH = 20
F_PEAK = 10
N_SHOTS = 20
N_SOURCE_PER_SHOT = 1

vp_scale = 1.74
vs_scale = 1.0
rho_scale = 1

# The network and loss use 401 samples; physics uses the full time axis.
MODEL_INPUT_SIZE = 401     # network input time samples
LOSS_NT = 401              # loss time samples
learn = 1e-4

N_BLOCKS_DECODER = 4
MINI_BATCHES = 8
LR_MILESTONE = 1000
ITERATION = 2000
PRINT_FREQ = 1
SAVE_FREQ = 500
BOARDSHOW_FREQ = 50

DECODER_INITIAL_SHAPE = tuple((torch.tensor(model_shape) // (2 ** (N_BLOCKS_DECODER - 1))).tolist())

inpa = {
    'ns': N_SHOTS,
    'sdo': 4,
    'fdom': F_PEAK,
    'dh': DH,
    'dt': DT,
    'acq_type': 1,
    't': T,
    'npml': 20,
    'pmlR': 1e-5,
    'pml_dir': 2,
    'device': 1,
    'seimogram_shape': '3d',
    'energy_balancing': False,
    "chpr": 70,
}

t_in = str(inpa['t'])
dt_in = str(inpa["dt"])
NT = int(Decimal(t_in) // Decimal(dt_in) + 1)
print("NT(full physics):", NT)


# %%
# =========================
# Geometry aligned with Siamese
# =========================
num_shots = N_SHOTS
num_sources_per_shot = N_SOURCE_PER_SHOT
num_receivers_per_shot = 199

dx = DH

source_depth_m   = 2 * dx
receiver_depth_m = 2 * dx

d_source_m      = np.floor(dx * (model_shape[1] / (num_shots + 1)))
first_source_m  = 25 * dx

d_receiver_m     = np.floor(dx * (model_shape[1] / (num_receivers_per_shot)))
first_receiver_m = 1 * dx

# Deepwave locations follow the [z, x] coordinate order.
src_loc = torch.zeros(num_shots, num_sources_per_shot, 2, dtype=torch.long, device=DEVICE)
src_loc[..., 0] = int(source_depth_m / dx)  # z
src_loc[:, 0, 1] = (torch.arange(num_shots, device=DEVICE) * int(d_source_m / dx) + int(first_source_m / dx))  # x

rec_loc = torch.zeros(num_shots, num_receivers_per_shot, 2, dtype=torch.long, device=DEVICE)
rec_loc[..., 0] = int(receiver_depth_m / dx)  # z
rec_x = (torch.arange(num_receivers_per_shot, device=DEVICE) * int(d_receiver_m / dx) + int(first_receiver_m / dx))  # x
rec_loc[:, :, 1] = rec_x.repeat(num_shots, 1)

N_RECEIVERS = num_receivers_per_shot
print("N_RECEIVERS:", N_RECEIVERS)

peak_source_time = 1.0 / F_PEAK
src = (
    deepwave.wavelets.ricker(F_PEAK, NT, DT, peak_source_time)
    .repeat(N_SHOTS, N_SOURCE_PER_SHOT, 1)
    .to(DEVICE)
)
print('wavelets shape:', src.shape)


# %%
# =========================
# Save paths
# =========================
variable_value = f'{N_SHOTS}_{F_PEAK}_{T}_{DT}_{MODEL_INPUT_SIZE}_{LOSS_NT}'
save_path = f"result/{model_name}/{result_dir_name}/"
get_dir(save_path)
model_path = f"result/{model_name}/{result_dir_name}/model"
get_dir(model_path)
summary_path = f"result/{model_name}/{result_dir_name}/summary"
get_dir(summary_path)
Fig_path = f"result/{model_name}/{result_dir_name}/Fig"
get_dir(Fig_path)
constant_path = f"result/{model_name}/{result_dir_name}/constant"
get_dir(constant_path)


# %%
# =========================
# LOAD TRUE MODELS (.npy)
# =========================
def load_arid_npy(path, reshape_hw=(400, 180)):
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(*reshape_hw)  # (400,180)
    if arr.shape == (400, 180):
        arr = arr.T  # -> (180,400)
    return arr

vp_np  = load_arid_npy(VP_NPY)
vs_np  = load_arid_npy(VS_NPY)
rho_np = load_arid_npy(RHO_NPY)

vp_true  = torch.from_numpy(vp_np).float().to(DEVICE)
vs_true  = torch.from_numpy(vs_np).float().to(DEVICE)
rho_true = torch.from_numpy(rho_np).float().to(DEVICE)
rho_true = rho_true * 1000.0  # g/cm^3 -> kg/m^3

def resize2d(t2d, out_hw):
    x = t2d[None, None, ...]
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return x[0, 0]

vp  = resize2d(vp_true,  (model_shape[0], model_shape[1]))
vs  = resize2d(vs_true,  (model_shape[0], model_shape[1]))
rho = resize2d(rho_true, (model_shape[0], model_shape[1]))

vp_initial  = torch.from_numpy(gaussian_filter(vp.detach().cpu().numpy(),  sigma=[10, 20])).float().to(DEVICE)
vs_initial  = torch.from_numpy(gaussian_filter(vs.detach().cpu().numpy(),  sigma=[10, 20])).float().to(DEVICE)
rho_initial = torch.from_numpy(gaussian_filter(rho.detach().cpu().numpy(), sigma=[10, 20])).float().to(DEVICE)

print("True model (vp/vs/rho):", vp.shape, vs.shape, rho.shape)
print("Init model (vp/vs/rho):", vp_initial.shape, vs_initial.shape, rho_initial.shape)
print("=== TRUE MODEL RANGE ===")
print("vp true   min/max:", vp.min().item(), vp.max().item())
print("vs true   min/max:", vs.min().item(), vs.max().item())
print("rho true  min/max:", rho.min().item(), rho.max().item())


# %%
# =========================
# Forward modeling -> observed data (full NT)
# =========================
with torch.no_grad():
    physics_full = Physics_deepwave(
        dh=inpa["dh"], dt=inpa["dt"], F_PEAK=inpa["fdom"],
        size=NT,                      # Use the full time axis for forward modeling.
        src=src, src_loc=src_loc, rec_loc=rec_loc
    ).to(DEVICE)

    d_obs_vx_full, d_obs_vy_full = physics_full(vp, vs, rho)  # [1, Nshots, NT, Nrec]
    print("Forward raw d_obs_vx:", d_obs_vx_full.shape, "d_obs_vy:", d_obs_vy_full.shape)

# inputs = observed (clean)
input_vx_full = d_obs_vx_full.clone()
input_vy_full = d_obs_vy_full.clone()

# squeeze to [Nshots, NT, Nrec]
d_obs_vx_full = d_obs_vx_full.squeeze(0)
d_obs_vy_full = d_obs_vy_full.squeeze(0)
input_vx_full = input_vx_full.squeeze(0)
input_vy_full = input_vy_full.squeeze(0)

# Resample the time axis for consistent network inputs and loss computation.
d_obs_vx = resample_time(d_obs_vx_full, out_nt=LOSS_NT)          # [Ns, 401, Nrec]
d_obs_vy = resample_time(d_obs_vy_full, out_nt=LOSS_NT)
input_vx = resample_time(input_vx_full, out_nt=MODEL_INPUT_SIZE) # [Ns, 401, Nrec]
input_vy = resample_time(input_vy_full, out_nt=MODEL_INPUT_SIZE)

print("After Downsample d_obs_vx:", d_obs_vx.shape, "d_obs_vy:", d_obs_vy.shape)
print("After Downsample input_vx:", input_vx.shape, "input_vy:", input_vy.shape)

# restore to [1, Nshots, Nt, Nrec]
d_obs_vx = d_obs_vx.unsqueeze(0)
d_obs_vy = d_obs_vy.unsqueeze(0)
input_vx = input_vx.unsqueeze(0)
input_vy = input_vy.unsqueeze(0)


# %%
# Quick plot
fig, ax = plt.subplots(2, 3, figsize=(9, 12))
ax[0, 0].imshow(vp.detach().cpu().numpy(), cmap='RdBu_r'); ax[0, 0].set_title("vp true")
ax[0, 1].imshow(vs.detach().cpu().numpy(), cmap='RdBu_r'); ax[0, 1].set_title("vs true")
ax[0, 2].imshow(rho.detach().cpu().numpy(), cmap='RdBu_r'); ax[0, 2].set_title("rho true")
ax[1, 0].imshow(vp_initial.detach().cpu().numpy(), cmap='RdBu_r'); ax[1, 0].set_title("vp init")
ax[1, 1].imshow(vs_initial.detach().cpu().numpy(), cmap='RdBu_r'); ax[1, 1].set_title("vs init")
ax[1, 2].imshow(rho_initial.detach().cpu().numpy(), cmap='RdBu_r'); ax[1, 2].set_title("rho init")
plt.tight_layout()
plt.savefig(os.path.join(Fig_path, "models_true_init.png"), dpi=200)
plt.show()


# %%
# Network / Optim
criteria = torch.nn.L1Loss(reduction='sum')
transdecoder = Transfomerdecoder(
    batch_size=BATCH_SIZE,
    in_channels=N_SHOTS,
    nt=MODEL_INPUT_SIZE,
    nr=N_RECEIVERS,
    patch_size=(8, 8),
    embed_dim=256,
    transddepth=12,
    n_blocks_decoder=N_BLOCKS_DECODER,
    final_size_encoder=DECODER_INITIAL_SHAPE[0] * DECODER_INITIAL_SHAPE[1],
    initial_shape_decoder=list(DECODER_INITIAL_SHAPE),
    final_spatial_shape=model_shape,
    num_heads=8,
    mlp_ratio=4.0,
    drop_ratio=0.0,
    attn_drop_ratio=0.0,
    drop_path_ratio=0.1,

    # WRTNet-only options: remove these two arguments when using the baseline import.
    use_wave_refine=True,
    wave_base_ch=24,
).to(DEVICE)
optim_transfomerdecoder = torch.optim.Adam(transdecoder.parameters(), lr=learn, betas=(0.5, 0.9))
scheduler_transfomerdecoder = torch.optim.lr_scheduler.StepLR(optim_transfomerdecoder, LR_MILESTONE, gamma=0.5)

all_loss_data = []
all_loss_vp_model = []
all_loss_vs_model = []
all_loss_rho_model = []
all_loss_model = []
all_loss_eval = []   # held-out receiver loss
all_gap = []         # eval - train
all_eval_iters = []  # Iterations corresponding to held-out evaluations

log_dir = summary_path + "/" + variable_value
writer = SummaryWriter(log_dir=log_dir)


def plotimg(a, b, c):
    fig, ax = plt.subplots(1, 3, figsize=(10,4))
    ax[0].imshow(a.detach().cpu(), cmap="RdBu_r"); ax[0].set_title("vp est")
    ax[1].imshow(b.detach().cpu(), cmap="RdBu_r"); ax[1].set_title("vs est")
    ax[2].imshow(c.detach().cpu(), cmap="RdBu_r"); ax[2].set_title("rho est")
    plt.tight_layout()
    return fig

drop_r = 0
keep_r_train = make_receiver_mask(N_RECEIVERS, drop_r, DEVICE)  # Receivers used for training
keep_r_eval = 1.0 - keep_r_train                                # Complementary held-out receivers


# %%
for it in range(ITERATION):

    # Train with the receivers selected by keep_r_train.
    transdecoder.train()
    loss_data, mp_sq, ms_sq, mrho_sq,  transdecoder = train_deepwave(
        Physics=Physics,
        transfomerdecoder=transdecoder,
        deepwave_size=NT,
        rho_initial=rho_initial,
        vx_initial=vp_initial,
        vy_initial=vs_initial,
        d_obs_vx=d_obs_vx,
        d_obs_vy=d_obs_vy,
        input_vx=input_vx,
        input_vy=input_vy,
        optim_transfomerdecoder=optim_transfomerdecoder,
        mini_batches=MINI_BATCHES,
        src_loc=src_loc,
        rec_loc=rec_loc,
        src=src,
        inpa=inpa,
        submarine=submarine,
        vp_scale=vp_scale,
        vs_scale=vs_scale,
        rho_scale=rho_scale,
        submarine_deep=submarine_deep,
        submarine_vp=submarine_vp,
        submarine_vs=submarine_vs,
        submarine_rho=submarine_rho,
        loss_nt=LOSS_NT,
        keep_r=keep_r_train,
        do_backward=True,
    )

    all_loss_data.append(loss_data)

    # Track model-space errors for evaluation only.
    with torch.no_grad():
        all_loss_vp_model.append(criteria(mp_sq, vp).item())
        all_loss_vs_model.append(criteria(ms_sq, vs).item())
        all_loss_rho_model.append(criteria(mrho_sq, rho).item())
        all_loss_model.append(all_loss_vp_model[-1] + all_loss_vs_model[-1] + all_loss_rho_model[-1])

    if (it + 1) % PRINT_FREQ == 0:
        print(f"Iteration {it + 1} ===== train data loss: {all_loss_data[-1]} | model loss: {all_loss_model[-1]}")

    # Evaluate on held-out receivers without backpropagation.
    if (it + 1) % BOARDSHOW_FREQ == 0:
        transdecoder.eval()
        with torch.no_grad():
            loss_eval, _, _, _, _ = train_deepwave(
                Physics=Physics,
                transfomerdecoder=transdecoder,
                deepwave_size=NT,
                rho_initial=rho_initial,
                vx_initial=vp_initial,
                vy_initial=vs_initial,
                d_obs_vx=d_obs_vx,
                d_obs_vy=d_obs_vy,
                input_vx=input_vx,
                input_vy=input_vy,
                optim_transfomerdecoder=optim_transfomerdecoder,
                mini_batches=MINI_BATCHES,
                src_loc=src_loc,
                rec_loc=rec_loc,
                src=src,
                inpa=inpa,
                submarine=submarine,
                vp_scale=vp_scale,
                vs_scale=vs_scale,
                rho_scale=rho_scale,
                submarine_deep=submarine_deep,
                submarine_vp=submarine_vp,
                submarine_vs=submarine_vs,
                submarine_rho=submarine_rho,
                loss_nt=LOSS_NT,
                keep_r=keep_r_eval,
                do_backward=False,
            )

        print(f"Iteration {it + 1} ===== held-out receiver loss: {loss_eval}")
        # Store held-out loss and the train-evaluation gap.
        all_loss_eval.append(loss_eval)
        all_gap.append(loss_eval - loss_data)
        all_eval_iters.append(it + 1)

        writer.add_scalar('Loss/gap_eval_minus_train', loss_eval - loss_data, it + 1)

        writer.add_scalar('Loss/Data_train_receivers', loss_data, it + 1)
        writer.add_scalar('Loss/Data_eval_holdout_receivers', loss_eval, it + 1)

        # Log model-space errors and current inversion results.
        writer.add_scalar('Loss/model vp', all_loss_vp_model[-1], it + 1)
        writer.add_scalar('Loss/model vs', all_loss_vs_model[-1], it + 1)
        writer.add_scalar('Loss/model rho', all_loss_rho_model[-1], it + 1)
        writer.add_scalar('Loss/model', all_loss_model[-1], it + 1)
        writer.add_figure("compare/train", plotimg(mp_sq, ms_sq, mrho_sq), it + 1, close=True)

    if (it + 1) % SAVE_FREQ == 0:

        # Save estimates, losses, and the receiver mask.
        estimatedv = {
            "vp_est": mp_sq,
            "vs_est": ms_sq,
            "rho_est": mrho_sq,
            "loss_train": all_loss_data,
            "loss_eval": all_loss_eval,
            "gap": all_gap,
            "eval_iters": all_eval_iters,
            "keep_r_train": keep_r_train.detach().cpu(),
        }
        torch.save(estimatedv, f"{model_path}/{it + 1}.pth")

        # Save the held-out receiver loss curve.
        if len(all_loss_eval) > 1:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(all_eval_iters, all_loss_eval, 'r-', lw=2, label='Held-out receivers')
            plt.plot(all_eval_iters,
                     [all_loss_data[i - 1] for i in all_eval_iters],
                     'b--', lw=1.5, label='Train receivers')
            plt.xlabel("Iteration")
            plt.ylabel("Wavelet loss")
            plt.title(f"Held-out receiver loss (drop_r={drop_r})")
            plt.legend()
            plt.grid(alpha=0.3)

            fig_path = os.path.join(Fig_path, f"heldout_curve_iter_{it + 1}.png")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()

    scheduler_transfomerdecoder.step()



# %%
# final plot
fig, ax = plt.subplots(1, 3, figsize=(12,4))
ax[0].imshow(mp_sq.detach().cpu(), cmap="RdBu_r"); ax[0].set_title("vp est")
ax[1].imshow(ms_sq.detach().cpu(), cmap="RdBu_r"); ax[1].set_title("vs est")
ax[2].imshow(mrho_sq.detach().cpu(), cmap="RdBu_r"); ax[2].set_title("rho est")
plt.tight_layout()
plt.savefig(os.path.join(Fig_path, "inverse_result.png"), dpi=200)
plt.show()