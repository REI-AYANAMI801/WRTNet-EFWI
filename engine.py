# %%
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import random
from decimal import Decimal
import deepwave
import warnings
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from functools import partial
import time

from normaltrans import *
import shutil
import pywt
import ptwt
warnings.filterwarnings('ignore')
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# %%
gpu_count = torch.cuda.device_count()
print(f"The number of available GPUs is: {gpu_count}")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"The selected GPU device is: {torch.cuda.get_device_name(DEVICE)}")
else:
    DEVICE = torch.device("cpu")
    print("No available GPUs detected, switched to using CPU")
# %% md
# Define functions
# %%
# export CUBLAS_WORKSPACE_CONFIG=:16:8
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def wavelet_multiscale_l1_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    levels=(1, 2),          # 多尺度：建议先 (1,2)，稳定后再加 3
    wA=1.0,                 # 近似系数权重（低频/大尺度）
    wD=0.3,                 # 细节系数权重（高频/小尺度）
    per_level_decay=0.5,    # 越高层（越粗尺度）权重越大/或越小，你可调
    normalize=True,
    eps=1e-8,
):
    """
    pred, obs: [Ns, Nt, Nrec]  或 [B, Nt, Nrec]
    返回：标量 loss
    """
    if pred.dim() != 3 or obs.dim() != 3:
        raise ValueError(f"need 3D [Ns,Nt,Nrec], got {pred.shape} and {obs.shape}")
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, obs {obs.shape}")

    Ns = pred.shape[0]
    total = pred.new_tensor(0.0)

    # 归一化（可选，避免尺度漂）
    if normalize:
        denom = torch.norm(obs, p=2) + eps
    else:
        denom = pred.new_tensor(1.0)

    for s in range(Ns):
        g_pred = pred[s]  # [Nt, Nrec]
        g_obs  = obs[s]

        # 多尺度累计
        for li, L in enumerate(levels):
            # ptwt.wavedec2: coeffs = [cA_L, (cH_L,cV_L,cD_L), (cH_{L-1},...), ...]
            coeffs_p = ptwt.wavedec2(g_pred, pywt.Wavelet("haar"), level=L, mode="zero")
            coeffs_o = ptwt.wavedec2(g_obs,  pywt.Wavelet("haar"), level=L, mode="zero")

            # 这里取“最深层”的一组（对应尺度 L）
            cA_p = coeffs_p[0].squeeze(0) if coeffs_p[0].dim() == 3 else coeffs_p[0]
            cA_o = coeffs_o[0].squeeze(0) if coeffs_o[0].dim() == 3 else coeffs_o[0]

            (cH_p, cV_p, cD_p) = coeffs_p[1]
            (cH_o, cV_o, cD_o) = coeffs_o[1]

            # 有些版本 ptwt 会多一个 batch 维 [1,H,W]，这里统一 squeeze
            if cH_p.dim() == 3: cH_p, cV_p, cD_p = cH_p.squeeze(0), cV_p.squeeze(0), cD_p.squeeze(0)
            if cH_o.dim() == 3: cH_o, cV_o, cD_o = cH_o.squeeze(0), cV_o.squeeze(0), cD_o.squeeze(0)

            # 每一层一个权重（越高层权重按 decay 调整）
            wl = (per_level_decay ** li)

            loss_A = F.l1_loss(cA_p, cA_o, reduction="mean")
            loss_D = (
                F.l1_loss(cH_p, cH_o, reduction="mean")
                + F.l1_loss(cV_p, cV_o, reduction="mean")
                + F.l1_loss(cD_p, cD_o, reduction="mean")
            )

            total = total + wl * (wA * loss_A + wD * loss_D)

    # 按 shot 数平均 + 可选归一化
    total = total / Ns
    total = total / denom
    return total

def seed_everything(seed=42):
    """
    Random seeds in fixed code are easy to reproduce
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def Downsample(img, aim_height, aim_width):
    channel, height, width = img.shape
    empty_img = torch.zeros((channel, aim_height, aim_width))
    transform_h = aim_height / height
    transform_w = aim_width / width
    for i in range(aim_height):
        for j in range(aim_width):
            x = int(i / transform_h)
            y = int(j / transform_w)
            empty_img[:, i, j] = img[:, x, y]
    return empty_img


def awgn(x_volt, snr):
    """

    https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python

    """
    if snr != 0:
        x_watts = x_volt ** 2
        sig_avg_watts = torch.mean(x_watts)

        sig_avg_db = 10 * torch.log10(sig_avg_watts)

        noise_avg_db = sig_avg_db - snr
        noise_avg_watts = 10 ** (noise_avg_db / 10)

        mean_noise = 0

        noise = torch.normal(mean_noise, torch.sqrt(noise_avg_watts), x_watts.shape)
        x_volt += noise

    return x_volt


def get_dir(directory):
    """
    Creates the given directory if it does not exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def clear_dir(directory):
    """
    Removes all files in the given directory.
    """
    if not os.path.isdir(directory): raise Exception("%s is not a directory" % (directory))
    if type(directory) != str: raise Exception("string type required for directory: %s" % (directory))
    if directory in ["..", ".", "", "/", "./", "../", "*"]: raise Exception(
        "trying to delete current directory, probably bad idea?!")

    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

def train_deepwave(Physics,
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
                   criteria,
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
                   ):
    submarine_deep = int(submarine_deep)
    loss_data_minibatch = []

    for batch in range(mini_batches):
        loss_freqs = []

        optim_transfomerdecoder.zero_grad()

        src_loc_batch = src_loc[batch::mini_batches]
        rec_loc_batch = rec_loc[batch::mini_batches]
        src_batch = src[batch::mini_batches]

        physics = Physics(inpa['dh'], inpa['dt'], inpa['fdom'], size=deepwave_size, src=src_batch,
                          src_loc=src_loc_batch, rec_loc=rec_loc_batch
                          )
        loss_data, mp_sq, ms_sq, mrho_sq = train_engine(transfomerdecoder, physics,
                                                        criteria, optim_transfomerdecoder,
                                                        rho_initial,
                                                        vx_initial,
                                                        vy_initial,
                                                        d_obs_vx, d_obs_vy,
                                                        input_vx, input_vy,
                                                        batch, mini_batches,
                                                        )
        loss_freqs.append(loss_data)
        loss_data_minibatch.append(np.mean(loss_freqs))
    return np.mean(loss_data_minibatch), mp_sq, ms_sq, mrho_sq, transfomerdecoder


def train_engine(transfomerdecoder, physics, criteria, optim_transfomerdecoder, rho_initial, vx_initial, vy_initial,
                 d_obs_vx, d_obs_vy, input_vx, input_vy, batch, mini_batches):
    earth_model_vp, earth_model_vs, earth_model_rho = transfomerdecoder(input_vx, input_vy)
    device = earth_model_vp.device
    size_a = int(vx_initial.shape[0])
    size_b = int(vx_initial.shape[1])
    if earth_model_vp.view(size_a, size_b).shape == vx_initial.squeeze().shape and earth_model_vs.view(size_a,
                                                                                                       size_b).shape == vy_initial.squeeze().shape:
        vp = earth_model_vp.view(size_a, size_b) * vp_scale + vx_initial
        vs = earth_model_vs.view(size_a, size_b) * vs_scale + vy_initial
        rho = (earth_model_rho.view(size_a, size_b)) * rho_scale + rho_initial
        if submarine == "yes":
            vp[:submarine_deep, :] = submarine_vp
            vs[:submarine_deep, :] = submarine_vs
            rho[:submarine_deep, :] = submarine_rho
        elif submarine == "no":
            vp = vp
            vs = vs
            rho = rho
        vp = vp.requires_grad_(True)
        vs = vs.requires_grad_(True)
        rho = rho.requires_grad_(True)

    else:
        print('The initial velocity model and the expected velocity model are of different sizes.')
    model_shape_x = vp.shape[0]
    model_shape_y = vp.shape[1]
    vp = vp.to(device)
    vs = vs.to(device)
    rho = rho.to(device)

    mp = vp
    ms = vs
    mrho = rho

    taux_est = physics(mp, ms, mrho)
    taux_vx = taux_est[0].squeeze(0)  # -> [Ns, Nt, Nrec]
    taux_vy = taux_est[1].squeeze(0)  # -> [Ns, Nt, Nrec]

    # obs 同样先 squeeze 到 3 维
    d_obs_vx_mb = d_obs_vx[:, batch::mini_batches].squeeze(0)  # [Ns, Nt, Nrec]
    d_obs_vy_mb = d_obs_vy[:, batch::mini_batches].squeeze(0)  # [Ns, Nt, Nrec]

    # 在 shot 维拼接：dim=0
    pred_all = torch.cat([taux_vx, taux_vy], dim=0)  # [2*Ns, Nt, Nrec]
    obs_all = torch.cat([d_obs_vx_mb, d_obs_vy_mb], dim=0)  # [2*Ns, Nt, Nrec]

    # 可选：调试打印一次
    # print("pred_all", pred_all.shape, "obs_all", obs_all.shape)

    loss = wavelet_multiscale_l1_loss(
        pred_all,
        obs_all,
        levels=(1, 2),
        wA=1.0,
        wD=0.3,
        per_level_decay=0.7,
        normalize=True,
    )
    # 你要返回的模型
    mp_sq, ms_sq, mrho_sq = mp, ms, mrho

    loss.backward()
    optim_transfomerdecoder.step()
    return loss.item(), mp_sq, ms_sq, mrho_sq


# %% md
# Parameters for ViT-EFWI
# %%
Physics = Physics_deepwave
train_fun = train_deepwave
BATCH_SIZE = 1
rp_properties = None

model_shape = [128, 256]
submarine = "yes"  # If on the seabed, "yes"; on land, "no"
submarine_deep = 16  # water layer deep
submarine_vp = 1500  # water layer vp
submarine_vs = 0  # water layer vs
submarine_rho = 1009  # water layer rho

VPPATH = './data/model/marmousi2_vp'
VSPATH = './data/model/marmousi2_vs'
rhoPATH = './data/model/marmousi2_rho'

VPinitialPATH = "./data/model/marmousi2_vpinit"
VSinitialPATH = "./data/model/marmousi2_vsinit"
rhoinitialPATH = "./data/model/marmousi2_rhoinit"

d_obs_vx_PATH = "./data/observed/marmousi2_gather/noise0.5/clean_obs_vx_20_10_2.4_0.006_401"
d_obs_vy_PATH = "./data/observed/marmousi2_gather/noise0.5/clean_obs_vy_20_10_2.4_0.006_401"
input_vx_PATH = "./data/observed/marmousi2_gather/noise0.5/clean_input_vx_20_10_2.4_0.006_401"
input_vy_PATH = "./data/observed/marmousi2_gather/noise0.5/clean_input_vy_20_10_2.4_0.006_401"
time_tag = time.strftime("%Y%m%d-%H%M%S")

model_name = 'remake_marmousi2'  # use "toy" , "marmousi2" or "bp"  the set setting is only related to naming
result_dir_name = f"clean_TEST_{time_tag}"
  # The inversion result will be saved with the name in folder model_name

NOISE: int = 0
T = 2.4
DT = 0.006
F_PEAK = 10
DH = 10
N_SHOTS = 20
N_SOURCE_PER_SHOT = 1

vp_scale = 1.74  # the scale factor to balance the weights of vp
vs_scale = 1  # the scale factor to balance the weights of vs
rho_scale = 0.025  # the scale factor to balance the weights of rho

MODEL_INPUT_SIZE = 401  # Control the size of network input seismic records
DEEPWAVE_SIZE = 401  # Used to calculate the size of seismic records for data loss
learn = 1e-3
N_BLOCKS_DECODER = 4
PATCH_SIZE = (20, 20)
EMBED_DIM = 12
NUM_HEADS = 12
MPL_RATION = 4.0
TRANSDDEPTH = 12
MINI_BATCHES = 4
LR_MILESTONE = 2000  # Learning rate regulation
ITERATION = 2000
PRINT_FREQ = 1  # Loss printing frequency
SAVE_FREQ = 500  # Model saving frequency
BOARDSHOW_FREQ = 50  # tensorboard display frequency

DECODER_INITIAL_SHAPE = torch.div(torch.tensor(model_shape), (2 ** (N_BLOCKS_DECODER - 1)),
                                  rounding_mode='floor')  # 解码器初始形状
FINAL_SIZE_ENCODER = BATCH_SIZE * DECODER_INITIAL_SHAPE[0] * DECODER_INITIAL_SHAPE[1]  # 全连接输出

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
print("NT:", NT)
inpa['rec_dis'] = 1 * inpa['dh']  # Define the receivers' distance

offsetx = inpa['dh'] * model_shape[1]
print("offsetx:", offsetx)
depth = inpa['dh'] * model_shape[0]
print("depth:", depth)
surface_loc_x = np.arange(13 * inpa["dh"], offsetx - 13 * inpa["dh"], inpa['dh'], np.float32)

n_surface_rec = len(surface_loc_x)

surface_loc_z = 17 * inpa["dh"] * np.ones(n_surface_rec, np.float32)

surface_loc = np.vstack((surface_loc_x, surface_loc_z)).T

rec_loc_temp = surface_loc

src_loc_temp = np.vstack((
    np.linspace(13 * inpa["dh"], offsetx - 13 * inpa["dh"], N_SHOTS, np.float32),
    2 * inpa["dh"] * np.ones(N_SHOTS, np.float32)
)).T

src_loc_temp[:, 1] -= 2 * inpa['dh']
# Create the source
N_RECEIVERS = n_surface_rec
print('N_RECEIVERS:', N_RECEIVERS)

# Shot 1 source located at cell [0, 1], shot 2 at [0, 2], shot 3 at [0, 3]
src_loc = torch.zeros(N_SHOTS, N_SOURCE_PER_SHOT, 2,
                      dtype=torch.int, device=DEVICE)

src_loc[:, 0, :] = torch.Tensor(np.flip(src_loc_temp) // DH)

src_loc[:, :, 0] = 1

# Receivers located at [0, 1], [0, 2], ... for every shot
rec_loc = torch.zeros(N_SHOTS, N_RECEIVERS, 2,
                      dtype=torch.long, device=DEVICE)
rec_loc[:, :, :] = (
    torch.Tensor(np.flip(rec_loc_temp) / DH)
)
src = (
    deepwave.wavelets.ricker(F_PEAK, NT, DT, 1.5 / F_PEAK)
    .repeat(N_SHOTS, N_SOURCE_PER_SHOT, 1)
    .to(DEVICE)
)
print('wavelets shape:', src.shape)

######### 保存路径设置 #########
variable_value = f'{N_SHOTS}_{F_PEAK}_{T}_{DT}_{MODEL_INPUT_SIZE}_{DEEPWAVE_SIZE}'  # 变量名称    需修改
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
######### constant保存 #########
constant_path = constant_path + '/constant_' + "_" + variable_value + '.txt'
with open(constant_path, 'w') as file:
    file.write(f"model_shape: {model_shape}\nVPPATH: {VPPATH}\nVSPATH: {VSPATH}\nmodel_name: {model_name}"
               f"\nNOISE: {NOISE}\nT: {T}\nDT: {DT}\nF_PEAK: {F_PEAK}\nDH: {DH}\nN_SHOTS: {N_SHOTS}\nN_SOURCE_PER_SHOT: {N_SOURCE_PER_SHOT}"
               f"\nMODEL_INPUT_SIZE: {MODEL_INPUT_SIZE}\nDEEPWAVE_SIZE: {DEEPWAVE_SIZE}\nLR_MILESTONE: {LR_MILESTONE}\nITERATION: {ITERATION}"
               f"\nTRANSDDEPTH: {TRANSDDEPTH}\nN_BLOCKS_DECODER: {N_BLOCKS_DECODER}"
               f"\nPRINT_FREQ: {PRINT_FREQ}\nSAVE_FREQ: {SAVE_FREQ}\nMINI_BATCHES: {MINI_BATCHES}"
               f"\nlearn: {learn}\nN_RECEIVERS: {N_RECEIVERS}\nsrc_loc: {src_loc}\nrec_loc: {rec_loc}")
# %%
######### load data #########
vp = torch.load(VPPATH).to(DEVICE)
vs = torch.load(VSPATH).to(DEVICE)
rho = torch.load(rhoPATH).to(DEVICE)

vp_initial = torch.load(VPinitialPATH).to(DEVICE)
vs_initial = torch.load(VSinitialPATH).to(DEVICE)
rho_initial = torch.load(rhoinitialPATH).to(DEVICE)

d_obs_vx = torch.load(d_obs_vx_PATH).to(DEVICE)
print('aaaa', d_obs_vx.shape)
d_obs_vx = Downsample(d_obs_vx, DEEPWAVE_SIZE, N_RECEIVERS).to(DEVICE)
print(d_obs_vx.shape)

d_obs_vy = torch.load(d_obs_vy_PATH).to(DEVICE)
d_obs_vy = Downsample(d_obs_vy, DEEPWAVE_SIZE, N_RECEIVERS).to(DEVICE)
print(d_obs_vy.shape)

input_vx = torch.load(input_vx_PATH).to(DEVICE)
input_vx = Downsample(input_vx, MODEL_INPUT_SIZE, N_RECEIVERS).to(DEVICE)
print(input_vx.shape)
input_vy = torch.load(input_vy_PATH).to(DEVICE)
input_vy = Downsample(input_vy, MODEL_INPUT_SIZE, N_RECEIVERS).to(DEVICE)
print(input_vy.shape)
# print(d_obs_vx.device)
######### polt data #########
VP_MIN = vp.min().item()
VP_MAX = vp.max().item()
VS_MIN = vs.min().item()
VS_MAX = vs.max().item()

fig, ax = plt.subplots(2, 3, figsize=(9, 12))
fig0 = ax[0, 0].imshow(vp.cpu().numpy(), cmap='RdBu_r')
ax[0, 0].plot(rec_loc_temp[:, 0] / DH, rec_loc_temp[:, 1] / DH, 'k*', markersize=1)
ax[0, 0].plot(src_loc_temp[:, 0] / DH, src_loc_temp[:, 1] / DH, 'rv', markersize=4)
ax[0, 0].tick_params(axis='x', which='both', labelleft=True, labelbottom=False)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks(np.arange(0, 128, 20), (DH * np.arange(0, 128, 20)) / 1000)
ax[0, 0].tick_params(axis='y', which='both', length=2.5, labelsize=9)
ax[0, 0].set_ylabel("Depth (km)\n", fontsize=9)

fig1 = ax[0, 1].imshow(vs.cpu().numpy(), cmap='RdBu_r')
ax[0, 1].plot(rec_loc_temp[:, 0] / DH, rec_loc_temp[:, 1] / DH, 'k*', markersize=1)
ax[0, 1].plot(src_loc_temp[:, 0] / DH, src_loc_temp[:, 1] / DH, 'rv', markersize=4)
ax[0, 1].tick_params(axis='x', which='both', labelleft=False, labelbottom=False)
ax[0, 1].set_xticks([])
ax[0, 1].tick_params(axis='y', which='both', labelleft=False, labelbottom=False)
ax[0, 1].set_yticks([])

fig1 = ax[0, 2].imshow(rho.cpu().numpy(), cmap='RdBu_r')
ax[0, 2].plot(rec_loc_temp[:, 0] / DH, rec_loc_temp[:, 1] / DH, 'k*', markersize=1)
ax[0, 2].plot(src_loc_temp[:, 0] / DH, src_loc_temp[:, 1] / DH, 'rv', markersize=4)
ax[0, 2].tick_params(axis='x', which='both', labelleft=False, labelbottom=False)
ax[0, 2].set_xticks([])
ax[0, 2].tick_params(axis='y', which='both', labelleft=False, labelbottom=False)
ax[0, 2].set_yticks([])

fig2 = ax[1, 0].imshow(vp_initial.cpu().numpy(), cmap='RdBu_r')
ax[1, 0].plot(rec_loc_temp[:, 0] / DH, rec_loc_temp[:, 1] / DH, 'k*', markersize=1)
ax[1, 0].plot(src_loc_temp[:, 0] / DH, src_loc_temp[:, 1] / DH, 'rv', markersize=4)
ax[1, 0].set_yticks(np.arange(0, 128, 20), (DH * np.arange(0, 128, 20)) / 1000)
ax[1, 0].tick_params(axis='y', which='both', length=2.5, labelsize=9)
ax[1, 0].set_xticks(np.arange(0, 256, 40), (DH * np.arange(0, 256, 40)) / 1000)
ax[1, 0].tick_params(axis='x', which='both', length=2.5, labelsize=9)
ax[1, 0].set_xlabel("\nDistance (km)", fontsize=9)
ax[1, 0].set_ylabel("Depth (km)\n", fontsize=9)

fig3 = ax[1, 1].imshow(vs_initial.cpu().numpy(), cmap='RdBu_r')
ax[1, 1].plot(rec_loc_temp[:, 0] / DH, rec_loc_temp[:, 1] / DH, 'k*', markersize=1)
ax[1, 1].plot(src_loc_temp[:, 0] / DH, src_loc_temp[:, 1] / DH, 'rv', markersize=4)
ax[1, 1].set_yticks([])
ax[1, 1].tick_params(axis='x', which='both', labelbottom=True)
ax[1, 1].tick_params(axis='y', which='both', labelleft=False)
ax[1, 1].set_xticks(np.arange(0, 256, 40), (DH * np.arange(0, 256, 40)) / 1000)
ax[1, 1].tick_params(axis='x', which='both', length=2.5, labelsize=9)
ax[1, 1].set_xlabel("\nDistance (km)", fontsize=9)

fig3 = ax[1, 2].imshow(rho_initial.cpu().numpy(), cmap='RdBu_r')
ax[1, 1].plot(rec_loc_temp[:, 0] / DH, rec_loc_temp[:, 1] / DH, 'k*', markersize=1)
ax[1, 1].plot(src_loc_temp[:, 0] / DH, src_loc_temp[:, 1] / DH, 'rv', markersize=4)
ax[1, 2].set_yticks([])
ax[1, 2].tick_params(axis='x', which='both', labelbottom=True)
ax[1, 2].tick_params(axis='y', which='both', labelleft=False)
ax[1, 2].set_xticks(np.arange(0, 256, 40), (DH * np.arange(0, 256, 40)) / 1000)
ax[1, 2].tick_params(axis='x', which='both', length=2.5, labelsize=9)
ax[1, 2].set_xlabel("\nDistance (km)", fontsize=9)

cbar = fig.colorbar(fig1, ax=ax.ravel().tolist())
# cbar.set_label('Label')
cbar.ax.set_position([0.92, 0.395, 3, 0.2])
plt.subplots_adjust(hspace=-0.84, wspace=0.08)
# fig1.set_clim(500, 3000)
# cbar.ax.set_aspect(1)
plt.savefig(Fig_path + f"/velocity_model$initial_model" + ".pdf", bbox_inches='tight', dpi=900)
plt.show()

vpmin, vpmax = torch.quantile(d_obs_vx[N_SHOTS // 2],
                              torch.tensor([0.01, 0.99]).to(DEVICE))
vsmin, vsmax = torch.quantile(d_obs_vy[N_SHOTS // 2],
                              torch.tensor([0.01, 0.99]).to(DEVICE))

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax[0, 0].imshow(d_obs_vx[N_SHOTS // 2].cpu().detach().numpy(), aspect='auto', cmap='gray', vmin=vpmin, vmax=vpmax)
ax[0, 0].set_xlabel("Receiver")
ax[0, 0].set_ylabel("Time sample")
ax[0, 1].imshow(d_obs_vy[N_SHOTS // 2].cpu().detach().numpy(), aspect='auto', cmap='gray', vmin=vsmin, vmax=vsmax)
ax[0, 1].set_xlabel("Receiver")
ax[1, 0].imshow(input_vx[N_SHOTS // 2].cpu().detach().numpy(), aspect='auto', cmap='gray', vmin=vpmin, vmax=vpmax)
ax[1, 0].set_xlabel("Receiver")
ax[1, 0].set_ylabel("Time sample")
ax[1, 1].imshow(input_vy[N_SHOTS // 2].cpu().detach().numpy(), aspect='auto', cmap='gray', vmin=vsmin, vmax=vsmax)
ax[1, 1].set_xlabel("Receiver")
ax[0, 0].set_title("Vx Label")
ax[0, 1].set_title("Vy Label ")
ax[1, 0].set_title("Vx Network Input Image")
ax[1, 1].set_title("Vy Network Input Image")
# ax[1].set_ylabel("Time sample")
plt.subplots_adjust(hspace=0.6)
plt.savefig(Fig_path + f"/obs&input_gather" + ".jpg", dpi=900)
plt.show()

d_obs_vx = d_obs_vx.unsqueeze(0)
d_obs_vy = d_obs_vy.unsqueeze(0)
input_vx = input_vx.unsqueeze(0)
input_vy = input_vy.unsqueeze(0)
# %%
criteria = torch.nn.L1Loss(reduction='sum')
PATCH_SIZE = (4, 4)

transfomerdecoder = Transfomerdecoder(
    batch_size=BATCH_SIZE,
    in_channels=N_SHOTS,
    nt=MODEL_INPUT_SIZE, nr=N_RECEIVERS,

    patch_size=PATCH_SIZE,
    # fusion_mode="mix",          # ✅ 强制用 MAE cross-attn

    embed_dim=EMBED_DIM,
    transddepth=TRANSDDEPTH,
    n_blocks_decoder=N_BLOCKS_DECODER,
    final_size_encoder=FINAL_SIZE_ENCODER,
    initial_shape_decoder=DECODER_INITIAL_SHAPE,
    final_spatial_shape=model_shape,
    num_heads=NUM_HEADS,
    mlp_ratio=MPL_RATION
).to(DEVICE)
# %%
optim_transfomerdecoder = torch.optim.Adam(transfomerdecoder.parameters(), lr=learn, betas=(0.5, 0.9))
scheduler_transfomerdecoder = torch.optim.lr_scheduler.StepLR(optim_transfomerdecoder, LR_MILESTONE, gamma=0.5)
all_loss_data = []
all_loss_vx_model = []
all_loss_vy_model = []
all_loss_rho_model = []
all_loss_model = []
# %%
# %%time
log_dir = summary_path + "/" + variable_value
# clear_dir(log_dir)
writer = SummaryWriter(log_dir=log_dir)


def plotimg(a, b, c):
    fig, ax = plt.subplots(1, 3)
    im0 = ax[0].imshow(a.squeeze(0).detach().cpu(), cmap="RdBu_r")
    im1 = ax[1].imshow(b.squeeze(0).detach().cpu(), cmap="RdBu_r")
    im2 = ax[2].imshow(c.squeeze(0).detach().cpu(), cmap="RdBu_r")
    points = ax[2].get_position().get_points()
    dy = points[1, 1] - points[0, 1]

    cax = fig.add_axes([0.91, points[0, 1], 0.02, dy])
    cax.yaxis.set_ticks_position("right")
    cbar = fig.colorbar(im1, cax=cax, orientation="vertical", extend="neither", label="$Velocity (m/s)$")
    # plt.show()
    return fig


for iter in range(ITERATION):
    loss_data, mp_sq, ms_sq, mrho_sq, transfomerdecoder = train_fun(
        Physics=Physics,
        transfomerdecoder=transfomerdecoder,
        deepwave_size=DEEPWAVE_SIZE,
        rho_initial=rho_initial,
        vx_initial=vp_initial,
        vy_initial=vs_initial,
        d_obs_vx=d_obs_vx,
        d_obs_vy=d_obs_vy,
        input_vx=input_vx,
        input_vy=input_vy,
        optim_transfomerdecoder=optim_transfomerdecoder,
        criteria=criteria,
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
        submarine_rho=submarine_rho)

    all_loss_data.append(loss_data)

    with torch.no_grad():
        all_loss_vx_model.append(
            criteria(mp_sq, vp).item()
        )
        all_loss_vy_model.append(
            criteria(ms_sq, vs).item()
        )
        all_loss_rho_model.append(
            criteria(mrho_sq, rho).item()
        )
        all_loss_model.append(
            criteria(mp_sq, vp).item() + criteria(ms_sq, vs).item() + criteria(mrho_sq, rho).item()
        )

    if (iter + 1) % PRINT_FREQ == 0:
        print(f"Iteration {iter + 1} ===== loss: {all_loss_data[-1]} for data and {all_loss_model[-1]} for model")
    if (iter + 1) % BOARDSHOW_FREQ == 0:
        writer.add_scalar('Loss/Data', loss_data, iter + 1)
        writer.add_scalar('Loss/model vx', all_loss_vx_model[-1], iter + 1)
        writer.add_scalar('Loss/model vy', all_loss_vy_model[-1], iter + 1)
        writer.add_scalar('Loss/model rho', all_loss_rho_model[-1], iter + 1)
        writer.add_scalar('Loss/model', all_loss_model[-1], iter + 1)
        writer.add_figure("compare/train", plotimg(mp_sq, ms_sq, mrho_sq), iter + 1, close=True)
        fig, ax = plt.subplots(1, 3)
        im0 = ax[0].imshow(mp_sq.squeeze(0).detach().cpu(),
                           cmap="RdBu_r")
        im1 = ax[1].imshow(ms_sq.squeeze(0).detach().cpu(),
                           cmap="RdBu_r")
        im2 = ax[2].imshow(mrho_sq.squeeze(0).detach().cpu(),
                           cmap="RdBu_r")
        points = ax[1].get_position().get_points()
        dy = points[1, 1] - points[0, 1]

        cax = fig.add_axes([0.91, points[0, 1], 0.02, dy])
        cax.yaxis.set_ticks_position("right")
        cbar = fig.colorbar(im1, cax=cax, orientation="vertical",
                            extend="neither", label="$Velocity (m/s)$"
                            )
        plt.show()
    if (iter + 1) % SAVE_FREQ == 0:
        estimatedv = {"vp_est": mp_sq, "vs_est": ms_sq, "rho_est": mrho_sq, "data_loss": all_loss_data}
        estimatedv_name = f"{model_path}/{iter + 1}.pth"
        torch.save(estimatedv, estimatedv_name)
    scheduler_transfomerdecoder.step()
# %%
fig, ax = plt.subplots(1, 3)
im0 = ax[0].imshow(mp_sq.squeeze(0).detach().cpu(),
                   cmap="RdBu_r", vmin=vp.min(),
                   vmax=vp.max())
im1 = ax[1].imshow(ms_sq.squeeze(0).detach().cpu(),
                   cmap="RdBu_r", vmin=vs.min(),
                   vmax=vs.max())
im2 = ax[2].imshow(mrho_sq.squeeze(0).detach().cpu(),
                   cmap="RdBu_r", vmin=rho.min(),
                   vmax=rho.max())
points = ax[1].get_position().get_points()
dy = points[1, 1] - points[0, 1]

cax = fig.add_axes([0.91, points[0, 1], 0.02, dy])
cax.yaxis.set_ticks_position("right")
cbar = fig.colorbar(im1, cax=cax, orientation="vertical",
                    extend="neither", label="$Velocity (m/s)$")
plt.savefig(Fig_path + f"/inverse_result.pdf", bbox_inches='tight', dpi=900)
plt.show()

