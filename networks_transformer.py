# networks_transformer.py
import torch
import torch.nn as nn
from typing import Tuple
from functools import partial
import deepwave

# ============================================================
# 1) Plain multi-component fusion: (vx, vy) -> [B, 1, Nt, Nr]
# ============================================================

class PlainFusionToMap(nn.Module):
    """
    Compresses the shot dimension of each component and combines the two
    components with 1x1 convolutions. No attention or token masking is used.
    """
    def __init__(self, in_shots: int):
        super().__init__()
        self.vconv = nn.Conv2d(
            in_channels=in_shots,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=True
        )
        self.fconv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=True
        )

    def forward(self, vx, vy):
        if vx.shape != vy.shape:
            raise RuntimeError(f"vx shape {vx.shape} != vy shape {vy.shape}")

        vx = self.vconv(vx)                 # [B, Ns, Nt, Nr] -> [B, 1, Nt, Nr]
        vy = self.vconv(vy)
        return self.fconv(torch.cat([vx, vy], dim=1))


# ============================================================
# 2) Transformer (ViT-style) encoder blocks
# ============================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
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
        # x: [B,1,Nt,Nr] -> [B,N,C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x: [B,N,C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)

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
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
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


# ============================================================
# 3) CNN decoders (your original)
# ============================================================

class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)

class Eblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, operation, final_shape=None):
        super().__init__()
        layers = [
            SubBlock(in_channels, out_channels, stride),
            SubBlock(out_channels, out_channels, stride)
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

class Decoder_vp(nn.Module):
    def __init__(self, batch_size, initial_shape: Tuple, final_shape: Tuple, n_blocks, final_out_channels=1):
        super().__init__()
        self.initial_shape = initial_shape
        self.batch_size = batch_size
        out_channels = [8 * (2 ** i) for i in range(n_blocks)]
        out_channels = sorted(out_channels, reverse=True)

        layers = [Eblock(1, out_channels[0], stride=1, operation="up")]
        for i in range(n_blocks - 1):
            finalize = final_shape if i == n_blocks - 2 else None
            layers.append(Eblock(out_channels[i], out_channels[i+1], stride=1, operation="up", final_shape=finalize))
        self.conv_layers = nn.Sequential(*layers)
        self.final = nn.Conv2d(out_channels[-1], final_out_channels, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        x = x.reshape(self.batch_size, 1, self.initial_shape[0], self.initial_shape[1])
        x = self.conv_layers(x)
        return self.final(x)

class Decoder_vs(Decoder_vp):
    pass

class Decoder_rho(Decoder_vp):
    pass


# ============================================================
# 4) Transformerdecoder (kept) + FIXED reshape
# ============================================================

class Transfomerdecoder(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channels,
        nt,
        nr,
        patch_size=(16, 16),
        embed_dim=256,          # Token embedding dimension
        transddepth=8,          # Number of Transformer encoder blocks
        n_blocks_decoder=4,     # Number of CNN upsampling blocks
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
    ):
        super().__init__()
        self.batch_size = batch_size
        self.H_v = nt // patch_size[0]
        self.W_v = nr // patch_size[1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Plain 1x1-convolution fusion without cross-attention.
        self.fusion = PlainFusionToMap(in_shots=in_channels)

        self.patch_embed = embed_layer(nt, nr, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # transformer blocks: use transddepth (not n_blocks_decoder)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, transddepth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(transddepth)
        ])
        self.norm = norm_layer(embed_dim)

        # flatten tokens -> latent vector for CNN decoder heads
        self.fc_in_features = embed_dim * self.H_v * self.W_v
        self.final = nn.Linear(self.fc_in_features, final_size_encoder)

        self.decoder_vp = Decoder_vp(batch_size, initial_shape=initial_shape_decoder,
                                     final_shape=final_spatial_shape, n_blocks=n_blocks_decoder, final_out_channels=1)
        self.decoder_vs = Decoder_vs(batch_size, initial_shape=initial_shape_decoder,
                                     final_shape=final_spatial_shape, n_blocks=n_blocks_decoder, final_out_channels=1)
        self.decoder_rho = Decoder_rho(batch_size, initial_shape=initial_shape_decoder,
                                       final_shape=final_spatial_shape, n_blocks=n_blocks_decoder, final_out_channels=1)

    def forward(self, xx, yy):
        # xx, yy: [B, Ns, Nt, Nr]
        x = self.fusion(xx, yy)             # [B,1,Nt,Nr]
        x = self.patch_embed(x)             # [B,N,C]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)                  # [B,N,C]
        x = self.norm(x)                    # [B,N,C]

        # Flatten all tokens before the latent projection.
        B, N, C = x.shape
        x = x.reshape(B, N * C)             # [B, N*C] == [B, embed_dim*H_v*W_v]
        x = self.final(x)                   # [B, final_size_encoder]

        vp = self.decoder_vp(x)
        vs = self.decoder_vs(x)
        rho = self.decoder_rho(x)
        return vp, vs, rho


# ============================================================
# 5) Physics (deepwave)
# ============================================================

class Physics_deepwave(nn.Module):
    def __init__(self, dh, dt, F_PEAK, size, src, src_loc, rec_loc, rp_properties=None):
        super().__init__()
        self.dh = dh
        self.dt = dt
        self.src = src
        self.src_loc = src_loc
        self.rec_loc = rec_loc
        self.F_PEAK = F_PEAK
        self.size = size
        _ = rp_properties

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