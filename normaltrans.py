# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from functools import partial
import deepwave

# ========== Conv blocks ==========
class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SubBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Eblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, operation, final_shape=None):
        super(Eblock, self).__init__()
        layers = [
            SubBlock(in_channels=in_channels, out_channels=out_channels, stride=stride),
            SubBlock(in_channels=out_channels, out_channels=out_channels, stride=stride)
        ]
        if operation == "down":
            layers.append(nn.MaxPool2d(kernel_size=2))
        elif operation == "up":
            if not final_shape:
                layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            else:
                layers.append(nn.Upsample(final_shape, mode="bilinear"))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Fusion(nn.Module):
    def __init__(self, in_channels):
        super(Fusion, self).__init__()
        self.vconv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)
        self.fconv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1)

    def forward(self, xp, xs):
        if xp.shape != xs.shape:
            print("Vp=%s is not equal to Vs=%s" % (xp.shape, xs.shape))
        else:
            xp = self.vconv(xp)
            xs = self.vconv(xs)
            v = torch.cat((xp, xs), dim=1)
            x = self.fconv(v)
            return x


# ========== Transformer encoder bits ==========
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, nt, nr, patch_size=(16, 16), embed_dim=768, norm_layer=None):
        super().__init__()
        self.nt = nt
        self.nr = nr
        self.patch_size = patch_size
        self.grid_size = (nt // patch_size[0], nr // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ========== Decoders ==========
class Decoder_vp(nn.Module):
    def __init__(self, batch_size, initial_shape: Tuple, final_shape: Tuple, n_blocks, final_out_channels=1):
        super(Decoder_vp, self).__init__()
        self.initial_shape = initial_shape
        self.batch_size = batch_size

        layers = []
        self.out_channels = [8 * (2 ** i) for i in range(n_blocks)]
        self.out_channels = sorted(self.out_channels, reverse=True)

        layers.append(Eblock(in_channels=1, out_channels=self.out_channels[0], stride=1, operation="up"))
        for layer_idx in range(n_blocks - 1):
            finalize = final_shape if layer_idx == n_blocks - 2 else None
            layers.append(Eblock(in_channels=self.out_channels[layer_idx],
                                 out_channels=self.out_channels[layer_idx + 1],
                                 stride=1, operation="up", final_shape=finalize))
        self.conv_layers = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels[-1], out_channels=final_out_channels,
                      kernel_size=3, padding=1, stride=1, bias=True)
        )

    def forward(self, x):
        x = x.reshape(self.batch_size, 1, self.initial_shape[0], self.initial_shape[1])
        out_vp = self.conv_layers(x)
        out_vp = self.final(out_vp)
        return out_vp


class Decoder_vs(nn.Module):
    def __init__(self, batch_size, initial_shape: Tuple, final_shape: Tuple, n_blocks, final_out_channels=1):
        super(Decoder_vs, self).__init__()
        self.initial_shape = initial_shape
        self.batch_size = batch_size

        layers = []
        self.out_channels = [8 * (2 ** i) for i in range(n_blocks)]
        self.out_channels = sorted(self.out_channels, reverse=True)

        layers.append(Eblock(in_channels=1, out_channels=self.out_channels[0], stride=1, operation="up"))
        for layer_idx in range(n_blocks - 1):
            finalize = final_shape if layer_idx == n_blocks - 2 else None
            layers.append(Eblock(in_channels=self.out_channels[layer_idx],
                                 out_channels=self.out_channels[layer_idx + 1],
                                 stride=1, operation="up", final_shape=finalize))
        self.conv_layers = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels[-1], out_channels=final_out_channels,
                      kernel_size=3, padding=1, stride=1, bias=True)
        )

    def forward(self, x):
        x = x.reshape(self.batch_size, 1, self.initial_shape[0], self.initial_shape[1])
        out_vs = self.conv_layers(x)
        out_vs = self.final(out_vs)
        return out_vs


class Decoder_rho(nn.Module):
    def __init__(self, batch_size, initial_shape: Tuple, final_shape: Tuple, n_blocks, final_out_channels=1):
        super(Decoder_rho, self).__init__()
        self.initial_shape = initial_shape
        self.batch_size = batch_size

        layers = []
        self.out_channels = [8 * (2 ** i) for i in range(n_blocks)]
        self.out_channels = sorted(self.out_channels, reverse=True)

        layers.append(Eblock(in_channels=1, out_channels=self.out_channels[0], stride=1, operation="up"))
        for layer_idx in range(n_blocks - 1):
            finalize = final_shape if layer_idx == n_blocks - 2 else None
            layers.append(Eblock(in_channels=self.out_channels[layer_idx],
                                 out_channels=self.out_channels[layer_idx + 1],
                                 stride=1, operation="up", final_shape=finalize))
        self.conv_layers = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channels[-1], out_channels=final_out_channels,
                      kernel_size=3, padding=1, stride=1, bias=True)
        )

    def forward(self, x):
        x = x.reshape(self.batch_size, 1, self.initial_shape[0], self.initial_shape[1])
        out_rho = self.conv_layers(x)
        out_rho = self.final(out_rho)
        return out_rho
import pywt
import ptwt


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x: [B,C,H,W]
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LightBlock(nn.Module):
    """
    轻量局部增强块，替代你原来依赖 FFCResnetBlock 的版本
    """
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, bias=True),
            SimpleGate(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            LayerNorm2d(dim),
            nn.Conv2d(dim, dim * 2, kernel_size=1, padding=0, bias=True),
            SimpleGate(),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x + self.block(x)


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            bias=True
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class WaveletTransform2D(nn.Module):
    """
    用 ptwt 做二维小波分解/重建
    输入输出都是 torch tensor，可反传
    """
    def __init__(self, wave='haar', mode='zero'):
        super().__init__()
        self.wave = pywt.Wavelet(wave)
        self.mode = mode

    def dwt(self, x):
        """
        x: [B,C,H,W]
        返回:
            ll, hl, lh, hh : [B,C,H/2,W/2]
        """
        B, C, H, W = x.shape
        ll_list, hl_list, lh_list, hh_list = [], [], [], []

        for c in range(C):
            xc = x[:, c, :, :]  # [B,H,W]
            coeffs = ptwt.wavedec2(xc, self.wave, level=1, mode=self.mode)
            cA = coeffs[0]
            cH, cV, cD = coeffs[1]

            if cA.dim() == 2:
                cA = cA.unsqueeze(0)
                cH = cH.unsqueeze(0)
                cV = cV.unsqueeze(0)
                cD = cD.unsqueeze(0)

            ll_list.append(cA.unsqueeze(1))
            hl_list.append(cH.unsqueeze(1))
            lh_list.append(cV.unsqueeze(1))
            hh_list.append(cD.unsqueeze(1))

        ll = torch.cat(ll_list, dim=1)
        hl = torch.cat(hl_list, dim=1)
        lh = torch.cat(lh_list, dim=1)
        hh = torch.cat(hh_list, dim=1)
        return ll, hl, lh, hh

    def idwt(self, ll, hl, lh, hh):
        """
        ll/hl/lh/hh: [B,C,h,w]
        return: [B,C,H,W]
        """
        B, C, h, w = ll.shape
        out_list = []

        for c in range(C):
            coeffs = [
                ll[:, c, :, :],
                (
                    hl[:, c, :, :],
                    lh[:, c, :, :],
                    hh[:, c, :, :]
                )
            ]
            xc = ptwt.waverec2(coeffs, self.wave)
            if xc.dim() == 2:
                xc = xc.unsqueeze(0)
            out_list.append(xc.unsqueeze(1))

        x = torch.cat(out_list, dim=1)
        return x


class WaveProcessBlock(nn.Module):
    """
    适配你任务的小波处理块：
    - ll 走低频增强
    - hl/lh/hh 分别增强
    - 用方向卷积从 ll 提供额外高频先验
    - 最后 idwt 回来
    """
    def __init__(self, dim, n_l_block=1, n_h_block=1):
        super().__init__()
        self.dim = dim
        self.wt = WaveletTransform2D(wave='haar', mode='zero')

        self.low_blocks = nn.ModuleList([LightBlock(dim) for _ in range(max(1, n_l_block))])

        self.high_pre = DepthConv(dim * 3, dim * 3)

        self.high_blocks = nn.ModuleList([
            nn.Sequential(
                LayerNorm2d(dim * 3),
                nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=3, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim * 3, dim * 3, 1, padding=0, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for _ in range(max(1, n_h_block))
        ])

        self.h_fuse = nn.Conv2d(dim * 6, dim * 3, kernel_size=1, bias=True)

        self.post = nn.ModuleList([LightBlock(dim) for _ in range(max(1, n_l_block))])

        self.horizontal_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.vertical_conv   = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.diagonal_conv   = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)

        self._init_wave_kernels()

    def _init_wave_kernels(self):
        hk = torch.tensor([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]], dtype=torch.float32)
        vk = torch.tensor([[1, 1, 1],
                           [0, 0, 0],
                           [-1, -1, -1]], dtype=torch.float32)
        dk = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32)

        hk = hk.view(1, 1, 3, 3).repeat(self.dim, 1, 1, 1)
        vk = vk.view(1, 1, 3, 3).repeat(self.dim, 1, 1, 1)
        dk = dk.view(1, 1, 3, 3).repeat(self.dim, 1, 1, 1)

        with torch.no_grad():
            self.horizontal_conv.weight.copy_(hk)
            self.vertical_conv.weight.copy_(vk)
            self.diagonal_conv.weight.copy_(dk)

        self.horizontal_conv.weight.requires_grad = False
        self.vertical_conv.weight.requires_grad = False
        self.diagonal_conv.weight.requires_grad = False

    def forward(self, x):
        x0 = x
        H0, W0 = x0.shape[-2], x0.shape[-1]

        ll, hl, lh, hh = self.wt.dwt(x)

        for blk in self.low_blocks:
            ll = blk(ll)

        ll_hl = self.horizontal_conv(ll)
        ll_lh = self.vertical_conv(ll)
        ll_hh = self.diagonal_conv(ll)

        high = torch.cat([hl, lh, hh], dim=1)
        guide = torch.cat([ll_hl, ll_lh, ll_hh], dim=1)

        high = self.high_pre(high)
        high = self.h_fuse(torch.cat([high, guide], dim=1))

        for blk in self.high_blocks:
            high = high + blk(high)

        hl, lh, hh = torch.chunk(high, chunks=3, dim=1)

        x = self.wt.idwt(ll, hl, lh, hh)

        # 裁回输入尺寸，避免奇数尺寸导致的 1 像素偏差
        x = x[:, :, :H0, :W0]

        for blk in self.post:
            x = blk(x)

        return x + x0


class CWRefineNet(nn.Module):
    """
    专门给 vp/vs/rho correction map 用的小波 refinement 网络
    输入输出都是 [B,3,H,W]
    """
    def __init__(self, in_ch=3, base_ch=24, n_l_blocks=(1,1,1,1,1), n_h_blocks=(1,1,1,1,1)):
        super().__init__()

        self.in_proj = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True)

        self.conv1 = WaveProcessBlock(base_ch,   n_l_block=n_l_blocks[0], n_h_block=n_h_blocks[0])
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=2, stride=2)

        self.conv2 = WaveProcessBlock(base_ch * 2, n_l_block=n_l_blocks[1], n_h_block=n_h_blocks[1])
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 3, kernel_size=2, stride=2)

        self.conv3 = WaveProcessBlock(base_ch * 3, n_l_block=n_l_blocks[2], n_h_block=n_h_blocks[2])

        self.up1 = nn.Conv2d(base_ch * 5, base_ch * 2, kernel_size=1, bias=True)
        self.conv4 = WaveProcessBlock(base_ch * 2, n_l_block=n_l_blocks[3], n_h_block=n_h_blocks[3])

        self.up2 = nn.Conv2d(base_ch * 3, base_ch, kernel_size=1, bias=True)
        self.conv5 = WaveProcessBlock(base_ch, n_l_block=n_l_blocks[4], n_h_block=n_h_blocks[4])

        self.out_proj = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        """
        x: [B,3,H,W]
        """
        x0 = x

        x = self.in_proj(x)
        x1 = self.conv1(x)

        x2 = self.down1(x1)
        x2 = self.conv2(x2)

        x3 = self.down2(x2)
        x3 = self.conv3(x3)

        x4 = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.up1(torch.cat([x4, x2], dim=1))
        x4 = self.conv4(x4)

        x5 = F.interpolate(x4, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x5 = self.up2(torch.cat([x5, x1], dim=1))
        x5 = self.conv5(x5)

        out = self.out_proj(x5)

        # refinement residual
        return x0 + out

class Transfomerdecoder(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channels,
        nt,
        nr,
        patch_size=(16, 16),
        embed_dim=256,
        transddepth=8,              # ✅ Transformer depth
        n_blocks_decoder=4,          # CNN decoder upsample blocks
        final_size_encoder=98,
        initial_shape_decoder=(14, 28),
        final_spatial_shape=(116, 227),
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_ratio=0.,
        attn_drop_ratio=0.,
        drop_path_ratio=0.,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        use_wave_refine=True,
        wave_base_ch=24,
    ):
        super().__init__()

        # --- basic sizes ---
        self.batch_size = batch_size
        self.nt = nt
        self.nr = nr
        self.patch_size = patch_size

        self.H_v = nt // patch_size[0]
        self.W_v = nr // patch_size[1]
        self.num_patches = self.H_v * self.W_v
        self.embed_dim = embed_dim

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # --- fusion: [B,Ns,Nt,Nr] -> [B,1,Nt,Nr] ---
        self.fusion = Fusion(in_channels)

        # --- patch embed: [B,1,Nt,Nr] -> [B,N,C] ---
        self.patch_embed = embed_layer(nt, nr, patch_size=patch_size, embed_dim=embed_dim)

        # --- positional embedding: [1,N,C] ---
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # --- transformer blocks ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, transddepth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(transddepth)   # ✅ 用 transddepth，不是 n_blocks_decoder
        ])
        self.norm = norm_layer(embed_dim)

        # --- flatten tokens -> latent vector ---
        # tokens: [B,N,C] => flatten => [B, N*C]
        self.fc_in_features = embed_dim * num_patches
        self.final = nn.Linear(self.fc_in_features, final_size_encoder)

        # --- CNN decoders ---
        # ✅ 推荐 tuple/int，别用 list（虽然 list 也能跑，但更干净）
        if isinstance(initial_shape_decoder, list):
            initial_shape_decoder = (int(initial_shape_decoder[0]), int(initial_shape_decoder[1]))
        if isinstance(final_spatial_shape, list):
            final_spatial_shape = (int(final_spatial_shape[0]), int(final_spatial_shape[1]))

        self.decoder_vp = Decoder_vp(
            batch_size,
            initial_shape=initial_shape_decoder,
            final_shape=final_spatial_shape,
            n_blocks=n_blocks_decoder,
            final_out_channels=1
        )
        self.decoder_vs = Decoder_vs(
            batch_size,
            initial_shape=initial_shape_decoder,
            final_shape=final_spatial_shape,
            n_blocks=n_blocks_decoder,
            final_out_channels=1
        )
        self.decoder_rho = Decoder_rho(
            batch_size,
            initial_shape=initial_shape_decoder,
            final_shape=final_spatial_shape,
            n_blocks=n_blocks_decoder,
            final_out_channels=1
        )
        self.use_wave_refine = use_wave_refine
        if self.use_wave_refine:
            self.wave_refine = CWRefineNet(
                in_ch=3,
                base_ch=wave_base_ch,
                n_l_blocks=(1, 1, 1, 1, 1),
                n_h_blocks=(1, 1, 1, 1, 1),
            )
        # init pos_embed (可选)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, xx, yy):
        """
        xx, yy: [B, Ns, Nt, Nr]
        returns: vp, vs, rho: [B,1,H_model,W_model]
        """
        x = self.fusion(xx, yy)             # [B,1,Nt,Nr]

        x = self.patch_embed(x)             # [B,N,C]
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)                  # [B,N,C]
        x = self.norm(x)                    # [B,N,C]

        x = x.reshape(x.shape[0], -1)       # [B, N*C]
        x = self.final(x)                   # [B, final_size_encoder]

        vp = self.decoder_vp(x)             # [B,1,H,W]
        vs = self.decoder_vs(x)
        rho = self.decoder_rho(x)

        if self.use_wave_refine:
            model_3c = torch.cat([vp, vs, rho], dim=1)   # [B,3,H,W]
            model_3c = self.wave_refine(model_3c)
            vp = model_3c[:, 0:1, :, :]
            vs = model_3c[:, 1:2, :, :]
            rho = model_3c[:, 2:3, :, :]

        return vp, vs, rho


# ========== Physics (deepwave) ==========
class Physics_deepwave(nn.Module):
    def __init__(self, dh, dt, F_PEAK, size, src, src_loc, rec_loc, rp_properties=None):
        super(Physics_deepwave, self).__init__()
        self.dh = dh
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
        self.F_PEAK = F_PEAK
        self.size = size
        rp_properties = rp_properties

    def forward(self, vp, vs, rho):
        out = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
            self.dh, self.dt,
            source_amplitudes_y=self.src,
            source_amplitudes_x=self.src,
            source_locations_y=self.src_loc,
            source_locations_x=self.src_loc,
            receiver_locations_y=self.rec_loc,
            receiver_locations_x=self.rec_loc,
            pml_freq=self.F_PEAK
        )
        vx = out[15]
        vy = out[14]
        return vx.permute(0, 2, 1), vy.permute(0, 2, 1)
