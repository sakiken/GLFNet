# rmtse.py
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------
# Utilities
# ---------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, dim: int, r: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, max(8, dim // r), kernel_size=1)
        self.fc2 = nn.Conv2d(max(8, dim // r), dim, kernel_size=1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class LePE(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim, bias=False)

    def forward(self, x, H: int, W: int):
        # x: (B, N, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def _qkv(x: torch.Tensor, qkv: nn.Module, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, C, H, W = x.shape
    N = H * W
    qkv_out = qkv(x).flatten(2).transpose(1, 2)  # (B, N, 3C)
    q, k, v = qkv_out.chunk(3, dim=-1)
    head_dim = C // num_heads
    q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)  # (B,h,N,hd)
    k = k.reshape(B, N, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, N, num_heads, head_dim).transpose(1, 2)
    return q, k, v

# ----------------------
# Manhattan Self-Attn
# ----------------------

class DecomposedMaSA(nn.Module):
    """
    分解式曼哈顿注意力：分别沿 H/W 轴做注意力（线性复杂度），并叠加基于 |i-j| 的衰减核。
    """
    def __init__(self, dim: int, num_heads: int = 4, gamma_init: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.lepe = LePE(dim)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = _qkv(x, self.qkv, self.num_heads)  # (B,h,N,hd)

        # reshape to axis-wise sequences
        q_h = q.view(B, self.num_heads, H, W, self.head_dim)
        k_h = k.view(B, self.num_heads, H, W, self.head_dim)
        v_h = v.view(B, self.num_heads, H, W, self.head_dim)

        # Horizontal attention (per row)
        qh = q_h * self.scale
        attn_h = torch.einsum('bhijd,bhizd->bhijz', qh, k_h)  # (B,h,H,W,W)
        device = x.device
        idx_w = torch.arange(W, device=device).float()
        dist_w = torch.abs(idx_w[None, :] - idx_w[:, None])  # (W,W)
        decay_w = torch.exp(-self.gamma * dist_w)[None, None, None, :, :]  # (1,1,1,W,W)
        attn_h = F.softmax(attn_h, dim=-1) * decay_w
        out_h = torch.einsum('bhijz,bhizd->bhijd', attn_h, v_h)  # (B,h,H,W,hd)

        # Vertical attention (per column)
        q_w = q_h.permute(0,1,3,2,4)  # (B,h,W,H,hd)
        k_w = k_h.permute(0,1,3,2,4)
        v_w = v_h.permute(0,1,3,2,4)

        qw = q_w * self.scale
        attn_w = torch.einsum('bhwjd,bhwzd->bhwjz', qw, k_w)  # (B,h,W,H,H)
        idx_h = torch.arange(H, device=device).float()
        dist_h = torch.abs(idx_h[None, :] - idx_h[:, None])  # (H,H)
        decay_h = torch.exp(-self.gamma * dist_h)[None, None, None, :, :]  # (1,1,1,H,H)
        attn_w = F.softmax(attn_w, dim=-1) * decay_h
        out_w = torch.einsum('bhwjz,bhwzd->bhwjd', attn_w, v_w)  # (B,h,W,H,hd)
        out_w = out_w.permute(0,1,3,2,4)  # (B,h,H,W,hd)

        out = out_h + out_w
        out = out.reshape(B, self.num_heads, H*W, self.head_dim).transpose(1, 2).reshape(B, H*W, self.dim)
        out = self.lepe(out, H, W)
        out = out.transpose(1, 2).view(B, self.dim, H, W)
        out = self.proj(out)
        return out

class OriginalMaSA(nn.Module):
    """
    原始（2D 全局）曼哈顿注意力：O(N^2)。为避免显存爆炸，超阈值时自动回退到分解式实现。
    """
    def __init__(self, dim: int, num_heads: int = 4, gamma_init: float = 0.1, max_tokens_sq: int = 4096*4096):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.lepe = LePE(dim)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.max_tokens_sq = max_tokens_sq
        self.fallback = DecomposedMaSA(dim, num_heads, gamma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        if N * N > self.max_tokens_sq:
            return self.fallback(x)

        q, k, v = _qkv(x, self.qkv, self.num_heads)  # (B,h,N,hd)
        q = q * self.scale
        # Manhattan decay on grid
        device = x.device
        xs = torch.arange(W, device=device)
        ys = torch.arange(H, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # (W,H)
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).float()  # (N,2)
        dx = torch.abs(coords[:, 0:1] - coords[None, :, 0:1])
        dy = torch.abs(coords[:, 1:2] - coords[None, :, 1:2])
        dist = dx + dy
        decay = torch.exp(-self.gamma * dist)  # (N,N)

        attn = torch.einsum('bhnd,bhmd->bhnm', q, k)  # (B,h,N,N)
        attn = F.softmax(attn, dim=-1) * decay
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)  # (B,h,N,hd)

        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.lepe(out, H, W)
        out = out.transpose(1, 2).view(B, self.dim, H, W)
        out = self.proj(out)
        return out

class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MaSA_SE_Block(nn.Module):
    """
    DwConv → (MaSA + LePE) → FFN → SE，并在三处加残差。
    """
    def __init__(self, dim: int, heads: int, decomposed: bool, drop_path: float = 0.0):
        super().__init__()
        self.patch_embed = DepthwiseSeparableConv(dim)
        self.attn = DecomposedMaSA(dim, heads) if decomposed else OriginalMaSA(dim, heads)
        self.ffn = FFN(dim)
        self.se = SqueezeExcite(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embed + residual
        shortcut = x
        x = self.patch_embed(x)
        x = shortcut + self.drop_path(x)

        # Attention + residual
        shortcut = x
        x = self.attn(x)
        x = shortcut + self.drop_path(x)

        # FFN + residual
        shortcut = x
        x = self.ffn(x)
        x = shortcut + self.drop_path(x)

        # SE
        x = self.se(x)
        return x

class Downsample(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvStem(nn.Module):
    """
    四层 3×3 conv：224→112→56→56→56，输出 dim（默认 64）。
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 64):
        super().__init__()
        layers = []
        c = in_ch
        for (ks, st, p, oc) in [(3,2,1,out_ch//2), (3,2,1,out_ch), (3,1,1,out_ch), (3,1,1,out_ch)]:
            layers += [nn.Conv2d(c, oc, ks, st, p, bias=False),
                       nn.BatchNorm2d(oc),
                       nn.GELU()]
            c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Stage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, decomposed: bool, drop_path: float):
        super().__init__()
        self.blocks = nn.Sequential(*[MaSA_SE_Block(dim, heads, decomposed, drop_path) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)

class RMTSE(nn.Module):
    """
    Conv Stem → [Stage1(decomp) → Down] → [Stage2(decomp) → Down]
              → [Stage3(original) → Down] → [Stage4(original)] → BN → GAP → FC
    """
    def __init__(self,
                 in_chans: int = 3,
                 num_classes: int = 10,
                 dims: List[int] = [64, 128, 256, 512],
                 depths: List[int] = [2, 2, 6, 2],
                 heads: List[int] = [2, 4, 8, 8],
                 drop_path: float = 0.1,
                 final_pool: str = "avg"):
        super().__init__()
        assert len(dims) == len(depths) == len(heads) == 4
        self.stem = ConvStem(in_chans, dims[0])
        dpr = drop_path

        self.stage1 = Stage(dims[0], depths[0], heads[0], decomposed=True, drop_path=dpr)
        self.down1 = Downsample(dims[0], dims[1])

        self.stage2 = Stage(dims[1], depths[1], heads[1], decomposed=True, drop_path=dpr)
        self.down2 = Downsample(dims[1], dims[2])

        self.stage3 = Stage(dims[2], depths[2], heads[2], decomposed=False, drop_path=dpr)
        self.down3 = Downsample(dims[2], dims[3])

        self.stage4 = Stage(dims[3], depths[3], heads[3], decomposed=False, drop_path=dpr)

        self.norm = nn.BatchNorm2d(dims[-1])
        self.pool = nn.AdaptiveAvgPool2d(1) if final_pool == "avg" else nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)          # (B, 64, 56, 56)
        x = self.stage1(x)
        x = self.down1(x)         # (B, 128, 28, 28)

        x = self.stage2(x)
        x = self.down2(x)         # (B, 256, 14, 14)

        x = self.stage3(x)
        x = self.down3(x)         # (B, 512, 7, 7)

        x = self.stage4(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x

if __name__ == "__main__":
    # 快速自检 (32,3,224,224)
    model = RMTSE(in_chans=3, num_classes=10)
    x = torch.randn(32,3,224,224)
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)  # (32, 10)
