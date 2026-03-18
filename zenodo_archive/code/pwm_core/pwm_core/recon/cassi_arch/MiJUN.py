"""MiJUN: Mamba-inspired Joint Unfolding Network for Snapshot Spectral Compressive Imaging.

AAAI 2025. Reference: Meng et al., 2025.
Official repo: https://github.com/Mengjie-s/MiJUN

Pure-PyTorch implementation (no mamba_ssm dependency).
Forward logic matches the official repository exactly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Utilities (matching official MiJUN repo exactly) ────────────────────────

def A(x, Phi):
    """Forward model: sum of modulated spectral bands. x,Phi: (B,nC,H,W)."""
    return torch.sum(x * Phi, 1)  # (B, H, W)


def At(y, Phi):
    """Adjoint: broadcast measurement and modulate. y:(B,H,W), Phi:(B,nC,H,W)."""
    return y.unsqueeze(1).repeat(1, Phi.shape[1], 1, 1) * Phi  # (B,nC,H,W)


def shift_3d(inputs, step=2):
    """Zero-pad then circular-shift: (B,C,H,W) -> (B,C,H,W+(C-1)*step)."""
    B, C, H, W = inputs.shape
    temp = torch.zeros((B, C, H, W + (C - 1) * step), device=inputs.device,
                        dtype=inputs.dtype)
    temp[:, :, :, :W] = inputs
    for i in range(C):
        temp[:, i, :, :] = torch.roll(temp[:, i, :, :], shifts=step * i, dims=2)
    return temp


def shift_back_3d(inputs, step=2):
    """Reverse circular shift (in-place). Input/output: (B,nC,H,W_meas)."""
    bs, nC, row, col = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=-step * i, dims=2)
    return inputs


def window_partition(x, window_size):
    """Partition (B, H, W, C) into (B*nW, ws*ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_unpartition(windows, window_size, H, W):
    """Merge (B*nW, ws*ws, C) back to (B, H, W, C)."""
    nH = H // window_size
    nW = W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_reverse(windows, original_size, window_size=(8, 8)):
    """Reverse window partition: (B*nW, ws*ws, C) -> (B, H*W, C).

    Matches official MiJUN repo window_reverse.
    """
    H, W = original_size
    ws_h, ws_w = window_size
    B = int(windows.shape[0] / (H * W / ws_h / ws_w))
    output = windows.view(B, H // ws_h, W // ws_w, ws_h, ws_w, -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output


def get_relative_position_index(window_size):
    """Compute relative position index for window attention."""
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, ws, ws)
    coords_flat = coords.view(2, -1)  # (2, ws*ws)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws*ws, ws*ws)
    rel = rel.permute(1, 2, 0).contiguous()  # (ws*ws, ws*ws, 2)
    rel[:, :, 0] += window_size - 1
    rel[:, :, 1] += window_size - 1
    rel[:, :, 0] *= 2 * window_size - 1
    return rel.sum(-1).view(-1)  # (ws*ws * ws*ws,) flattened


# ─── Layer Norm (with .body wrapper to match checkpoint) ─────────────────────

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """LayerNorm wrapper with .body attribute (matches checkpoint naming)."""
    def __init__(self, dim):
        super().__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            x = self.body(x)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return x
        return self.body(x)


# ─── Mamba S6 — prefer CUDA mamba_ssm, fallback to pure-PyTorch ──────────────

# Try to import the official CUDA Mamba for best accuracy
try:
    from mamba_ssm import Mamba as _CudaMamba
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


class Mamba(nn.Module):
    """Mamba selective state space model.

    Uses the official mamba_ssm CUDA kernel when available (pip install mamba_ssm).
    Falls back to a pure-PyTorch sequential scan otherwise. The CUDA kernel is
    strongly recommended for best accuracy — trained weights are sensitive to the
    exact numerical behavior of the scan implementation.

    Parameter names match the official mamba_ssm.Mamba module exactly.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        if _HAS_MAMBA_SSM:
            # Delegate to official CUDA implementation (best accuracy)
            self._cuda_mamba = _CudaMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            # Expose sub-module parameters with correct names for checkpoint loading
            self.in_proj = self._cuda_mamba.in_proj
            self.conv1d = self._cuda_mamba.conv1d
            self.x_proj = self._cuda_mamba.x_proj
            self.dt_proj = self._cuda_mamba.dt_proj
            self.A_log = self._cuda_mamba.A_log
            self.D = self._cuda_mamba.D
            self.out_proj = self._cuda_mamba.out_proj
        else:
            # Pure PyTorch fallback
            self._cuda_mamba = None
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                    bias=True, groups=self.d_inner,
                                    padding=d_conv - 1)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        if self._cuda_mamba is not None:
            return self._cuda_mamba(x)

        # Pure PyTorch fallback
        B, L, _ = x.shape

        # 1. Input projection -> x, z
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 2. Causal conv1d
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: take first L
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        # 3. SSM parameters
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # ensure positive

        A = -torch.exp(self.A_log)  # (d_inner, d_state) — negative for stability

        # 4. Selective scan (sequential for correctness)
        y = self._selective_scan(x_conv, dt, A, B_ssm, C_ssm, self.D)

        # 5. Gate and project
        y = y * F.silu(z)  # (B, L, d_inner)
        return self.out_proj(y)  # (B, L, d_model)

    def _selective_scan(self, u, delta, A, B, C, D):
        """Selective scan (S6) — sequential pure-PyTorch implementation.

        u: (batch, L, d_inner) — input
        delta: (batch, L, d_inner) — discretization step
        A: (d_inner, d_state) — state matrix (negative)
        B: (batch, L, d_state) — input-dependent B
        C: (batch, L, d_state) — input-dependent C
        D: (d_inner,) — skip connection
        """
        batch, L, d_inner = u.shape
        d_state = A.shape[1]
        orig_dtype = u.dtype

        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=torch.float32)
        ys = torch.empty(batch, L, d_inner, device=u.device, dtype=orig_dtype)

        for t in range(L):
            dt = delta[:, t]  # (batch, d_inner)
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dBu = dt.unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            h = dA * h + dBu
            ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)

        return ys + u * D.unsqueeze(0).unsqueeze(0)


# ─── Mamba Layer (3-mode SSM) ────────────────────────────────────────────────

class MambaLayer(nn.Module):
    """Three-mode Mamba processing matching official MiJUN repo.

    Mode 0: spatial scan — (B, H*W, C) with Mamba(d_model=C)
    Mode 1: height scan — (B, C*W, H) with Mamba2(d_model=H)
    Mode 2: width scan  — (B, C*H, W) with Mamba2(d_model=W)

    All modes process full spectral-spatial data together (not per-channel).
    """
    def __init__(self, dim, dim_m):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(dim_m)
        self.mamba2 = Mamba(d_model=dim_m, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape

        # Mode 0: spatial scan — (B, H*W, C) — d_model=C, L=H*W
        x0 = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)
        x0 = self.mamba(self.norm(x0))
        x0 = x0.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Mode 1: height scan — permute each (C,H,W) → (H,C,W), then
        # flatten to (B, C*W, H) — d_model=H, L=C*W
        x1_in = x.permute(0, 2, 1, 3).reshape(B, H, C * W)  # (B, H, C*W)
        x1_flat = x1_in.transpose(1, 2)  # (B, C*W, H)
        x1_out = self.mamba2(self.norm2(x1_flat))  # (B, C*W, H)
        x1 = x1_out.transpose(1, 2).reshape(B, H, C, W).permute(0, 2, 1, 3)  # (B,C,H,W)

        # Mode 2: width scan — permute each (C,H,W) → (W,C,H), then
        # flatten to (B, C*H, W) — d_model=W, L=C*H
        x2_in = x.permute(0, 3, 1, 2).reshape(B, W, C * H)  # (B, W, C*H)
        x2_flat = x2_in.transpose(1, 2)  # (B, C*H, W)
        x2_out = self.mamba2(self.norm2(x2_flat))  # (B, C*H, W)
        x2 = x2_out.transpose(1, 2).reshape(B, W, C, H).permute(0, 2, 3, 1)  # (B,C,H,W)

        return (x0 + x1 + x2) / 3.0


# ─── Attention (matches official MiJUN Attention class) ──────────────────────

class WindowAttn(nn.Module):
    """Multi-head attention with local windows + global aggregation/broadcast.

    Matches the official MiJUN Attention class exactly:
    1. QKV on concatenated [global_token, image_tokens]
    2. Local windowed attention on image tokens
    3. Global aggregation: global token queries all image tokens
    4. Global broadcast: image tokens attend to global kv
    5. proj on concatenated result

    Parameter names match checkpoint keys (qkv, kv_global, proj,
    relative_position_bias_table, relative_position_index).
    """
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_tokens = 1  # global token count
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.kv_global = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        self.register_buffer('relative_position_index',
                             get_relative_position_index(window_size))

    def _get_relative_positional_bias(self):
        rpb = self.relative_position_bias_table[
            self.relative_position_index].view(
                self.attn_area, self.attn_area, -1)
        return rpb.permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward_global_aggregation(self, q, k, v):
        """Global token aggregates from all image tokens."""
        B, _, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, q, k, v, H, W):
        """Local windowed attention on image tokens."""
        B, num_heads, N, C = q.shape
        ws = self.window_size
        h_group, w_group = H // ws, W // ws

        # Partition into windows
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(
            0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, num_heads, ws * ws, C)
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(
            0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, num_heads, ws * ws, C)
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(
            0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, num_heads, ws * ws, v.shape[-1])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        # Reverse windows
        x = window_reverse(x, (H, W), (ws, ws))
        return x

    def forward_global_broadcast(self, q, k, v):
        """Image tokens attend to global tokens (broadcast)."""
        B, num_heads, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward(self, x, H, W):
        """x: (B, 1+H*W, C) — global token concatenated with image tokens.

        Returns: (B, 1+H*W, C) after proj.
        """
        B, N, C = x.shape
        NC = self.num_tokens  # 1
        ws = self.window_size

        # Separate global and image tokens
        x_img, x_global = x[:, NC:], x[:, :NC]
        x_img = x_img.view(B, H, W, C)

        # Pad for windowing
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_img = F.pad(x_img, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x_img.shape[1], x_img.shape[2]
        x_img = x_img.view(B, -1, C)
        x = torch.cat([x_global, x_img], dim=1)

        # QKV
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, 3, self.num_heads,
                            C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # Split global / image
        q_img, k_img, v_img = q[:, :, NC:], k[:, :, NC:], v[:, :, NC:]
        q_cls = q[:, :, :NC]

        # Local windowed attention on image tokens
        x_img = self.forward_local(q_img, k_img, v_img, Hp, Wp)

        # Remove padding
        x_img = x_img.view(B, Hp, Wp, -1)[:, :H, :W].reshape(B, H * W, -1)
        q_img = q_img.reshape(B, self.num_heads, Hp, Wp, -1)[
            :, :, :H, :W].reshape(B, self.num_heads, H * W, -1)
        k_img = k_img.reshape(B, self.num_heads, Hp, Wp, -1)[
            :, :, :H, :W].reshape(B, self.num_heads, H * W, -1)
        v_img = v_img.reshape(B, self.num_heads, Hp, Wp, -1)[
            :, :, :H, :W].reshape(B, self.num_heads, H * W, -1)

        # Global aggregation: global token queries all image tokens
        x_cls = self.forward_global_aggregation(q_cls, k_img, v_img)
        k_cls, v_cls = self.kv_global(x_cls).view(
            B, -1, 2, self.num_heads,
            C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # Global broadcast: image tokens attend to global tokens
        x_img = x_img + self.forward_global_broadcast(q_img, k_cls, v_cls)

        x = torch.cat([x_cls, x_img], dim=1)
        x = self.proj(x)
        return x


# ─── Local-NonLocal Attention ────────────────────────────────────────────────

class LocalNonLocalAttention(nn.Module):
    """Local-NonLocal attention matching official MiJUN l_nl_attn.

    The official forward path is:
      norm1 -> in_proj -> dwc -> SiLU -> cat global_token -> Attention -> return 4D
    Note: act_proj, out_proj, cpe1, cpe2, norm2 exist as parameters
    (loaded from checkpoint) but are NOT used in the forward path
    (dead code in official repo).
    """
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.in_proj = nn.Linear(dim, dim, bias=True)
        self.act_proj = nn.Linear(dim, dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.global_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn = WindowAttn(dim, num_heads, window_size)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)

        Matches official l_nl_attn.forward exactly.
        """
        B, C, H, W = x.shape

        # Flatten to (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        global_token = self.global_token.expand(x.shape[0], -1, -1)

        # norm1
        x = self.norm1(x)

        # in_proj -> reshape -> dwc -> SiLU
        x = self.in_proj(x).view(B, H, W, C)
        x = F.silu(self.dwc(x.permute(0, 3, 1, 2))).permute(
            0, 2, 3, 1).view(B, H * W, C)

        # Cat global token
        x_att = torch.cat((global_token, x), dim=1)  # (B, 1+H*W, C)

        # Attention (local + global aggregation/broadcast)
        x = self.attn(x_att, H, W)  # (B, 1+H*W, C)

        # Take only image tokens, reshape to 4D
        x = x[:, -H * W:]
        out = x.view(B, H, W, self.dim).permute(0, 3, 1, 2).contiguous()

        return out


# ─── Gated Depthwise Feed-Forward ───────────────────────────────────────────

class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expand=2.66):
        super().__init__()
        hidden = int(dim * ffn_expand)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=True)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, C, H, W)"""
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ─── PreNorm wrapper ────────────────────────────────────────────────────────

class PreNorm(nn.Module):
    """PreNorm with LayerNorm(.body) wrapper matching checkpoint naming."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


# ─── Local-NonLocal Block ───────────────────────────────────────────────────

class LocalNonLocalBlock(nn.Module):
    """Block with attention + mamba + FFN sub-blocks."""
    def __init__(self, dim, dim_m, num_heads, num_blocks=1,
                 window_size=8, ffn_expand=2.66):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleList([
                PreNorm(dim, LocalNonLocalAttention(dim, num_heads, window_size)),
                PreNorm(dim, MambaLayer(dim, dim_m)),
                PreNorm(dim, Gated_Dconv_FeedForward(dim, ffn_expand)),
            ])
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            for module in block:
                x = x + module(x)
        return x


# ─── Down/Up sample ─────────────────────────────────────────────────────────

class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 2, stride=2, bias=True)
        )

    def forward(self, x):
        return self.up(x)


# ─── LNLT Denoiser ──────────────────────────────────────────────────────────

class LNLT(nn.Module):
    """Local-NonLocal Transformer denoiser (U-Net with Mamba + attention)."""
    def __init__(self, in_dim=29, dim=28, out_dim=28,
                 num_blocks=(1, 1, 1, 1, 1), window_size=8,
                 ffn_expand=2.66, spatial_sizes=(256, 128, 64)):
        super().__init__()
        dims = [dim, dim * 2, dim * 4]  # [28, 56, 112]
        heads = [max(1, d // 28) for d in dims]  # [1, 2, 4]
        dim_m = list(spatial_sizes)  # [256, 128, 64]

        # Input/output
        self.embedding = nn.Conv2d(in_dim, dim, 3, padding=1, bias=False)
        self.mapping = nn.Conv2d(dim, out_dim, 3, padding=1, bias=False)

        # Encoder (2 levels)
        self.Encoder = nn.ModuleList([
            LocalNonLocalBlock(dims[0], dim_m[0], heads[0], num_blocks[0],
                               window_size, ffn_expand),
            LocalNonLocalBlock(dims[1], dim_m[1], heads[1], num_blocks[1],
                               window_size, ffn_expand),
        ])

        # Downsampling
        self.Downs = nn.ModuleList([
            DownSample(dims[0], dims[1]),
            DownSample(dims[1], dims[2]),
        ])

        # Bottleneck
        self.BottleNeck = LocalNonLocalBlock(
            dims[2], dim_m[2], heads[2], num_blocks[2],
            window_size, ffn_expand)

        # Upsampling
        self.Ups = nn.ModuleList([
            UpSample(dims[2], dims[1]),
            UpSample(dims[1], dims[0]),
        ])

        # Skip-connection fusion (concat then 1x1 conv)
        self.fusions = nn.ModuleList([
            nn.Conv2d(dims[1] * 2, dims[1], 1, bias=False),
            nn.Conv2d(dims[0] * 2, dims[0], 1, bias=False),
        ])

        # Decoder (2 levels)
        self.Decoder = nn.ModuleList([
            LocalNonLocalBlock(dims[1], dim_m[1], heads[1], num_blocks[3],
                               window_size, ffn_expand),
            LocalNonLocalBlock(dims[0], dim_m[0], heads[0], num_blocks[4],
                               window_size, ffn_expand),
        ])

    def forward(self, x):
        """x: (B, in_dim, H, W) -> (B, out_dim, H, W)

        Matches official MiJUN repo exactly:
        - Pad input to multiples of 16
        - Explicit encoder-decoder with skip connections
        - Residual: mapping(dec) + x[:, 1:, :, :]  (skip noise-level channel)
        - Crop back to original size
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        x1 = self.embedding(x)
        res1 = self.Encoder[0](x1)

        x2 = self.Downs[0](res1)
        res2 = self.Encoder[1](x2)

        x4 = self.Downs[1](res2)
        res4 = self.BottleNeck(x4)

        dec_res2 = self.Ups[0](res4)
        dec_res2 = torch.cat([dec_res2, res2], dim=1)
        dec_res2 = self.fusions[0](dec_res2)
        dec_res2 = self.Decoder[0](dec_res2)

        dec_res1 = self.Ups[1](dec_res2)
        dec_res1 = torch.cat([dec_res1, res1], dim=1)
        dec_res1 = self.fusions[1](dec_res1)
        dec_res1 = self.Decoder[1](dec_res1)

        # Residual: skip the noise-level channel (channel 0 of input)
        out = self.mapping(dec_res1) + x[:, 1:, :, :]

        return out[:, :, :h_inp, :w_inp]


# ─── Degradation Estimation ─────────────────────────────────────────────────

class DegradationEstimation(nn.Module):
    """Estimates refined Phi, mu (regularization), and noise level.

    Matches official MiJUN DegradationEstimation exactly.
    Input: concatenated [phi, z] (56 channels) -> DL -> phi refinement.
    The down_sample has stride=2 for spatial downsampling before global pool.
    Both mu and noise_level go through Softplus (final mlp layer).
    """
    def __init__(self, in_c=28, mid_c=56):
        super().__init__()
        # DL: ModuleList of two PWDWPWConv blocks (56->56, 56->28)
        # Key naming: DL.0.{0,2,4}, DL.1.{0,2,4} — matches checkpoint
        # DL: nn.Sequential of two PWDWPWConv blocks with GELU (matching official)
        self.DL = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(mid_c, 64, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True),
                nn.GELU(),
                nn.Conv2d(64, mid_c, 1, bias=False),
            ),
            nn.Sequential(
                nn.Conv2d(mid_c, 64, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True),
                nn.GELU(),
                nn.Conv2d(64, in_c, 1, bias=False),
            ),
        )

        # stride=2 downsampling for mu/noise_level estimation
        self.down_sample = nn.Conv2d(in_c, mid_c, 3, stride=2, padding=1, bias=True)

        self.mlp = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, mid_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, 2, 1, bias=True),
            nn.Softplus(),  # both outputs are positive
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z, phi):
        """
        z: (B, 28, H, W_meas) current spectral estimate (shifted domain)
        phi: (B, 28, H, W_meas) shifted mask3d

        Returns: (phi_refined, mu, noise_level)
            mu: (B, H_ds, W_ds) — regularization parameter (used in denominator)
            noise_level: (B, 1, H_ds, W_ds)
        """
        # Concatenate phi and z -> 56 channels
        inp = torch.cat([phi, z], dim=1)  # (B, 56, H, W_meas)
        phi_r = self.DL(inp)  # Sequential: PWDWPWConv(56->56) then PWDWPWConv(56->28)
        phi = phi + phi_r

        # Estimate mu and noise_level from phi refinement
        x = self.down_sample(self.relu(phi_r))  # (B, 56, H/2, W_meas/2)
        x = F.adaptive_avg_pool2d(x, 1)  # (B, 56, 1, 1)
        x = self.mlp(x) + 1e-6  # (B, 2, 1, 1) — positive via Softplus
        mu = x[:, 0, :, :]  # (B, 1, 1)
        noise_level = x[:, 1:2, :, :]  # (B, 1, 1, 1)

        return phi, mu, noise_level


# ─── MiJUN Main Model ───────────────────────────────────────────────────────

class MiJUN(nn.Module):
    """Mamba-inspired Joint Unfolding Network.

    5-stage unrolled optimization with shared parameters (SHARE_PARAMS=True):
    - DP: DegradationEstimation (shared across stages)
    - PP: LNLT denoiser (shared across stages)
    - fusion: learned initialization from [y_shift, Phi]

    All computation in the SHIFTED domain (B, 28, H, W_meas=310).
    Denoiser operates in image domain (B, 28, H, W=256) via shift_back/shift.

    Matches official repo: https://github.com/Mengjie-s/MiJUN
    """
    def __init__(self, stage=5, n_bands=28, step=2):
        super().__init__()
        self.stage = stage
        self.n_bands = n_bands
        self.step = step

        self.fusion = nn.Conv2d(n_bands * 2, n_bands, 1, bias=True)
        self.DP = DegradationEstimation(in_c=n_bands, mid_c=n_bands * 2)
        self.PP = LNLT(
            in_dim=n_bands + 1,  # 28 bands + 1 noise level
            dim=n_bands,
            out_dim=n_bands,
            num_blocks=(1, 1, 1, 1, 1),
            window_size=8,
            ffn_expand=2.66,
            spatial_sizes=(256, 128, 64),
        )

    def initial(self, y, Phi):
        """Learned initialization via fusion layer.

        y: (B, H, W_meas) measurement
        Phi: (B, nC, H, W_meas) shifted mask3d

        Returns: z (B, nC, H, W_meas) initial estimate in shifted domain
        """
        nC = self.n_bands
        step = self.step
        B, _, H, W_meas = Phi.shape
        W = W_meas - (nC - 1) * step  # image width

        # Distribute measurement y into spectral bands with circular shift offsets
        y_shift = torch.zeros(B, nC, H, W_meas, device=y.device, dtype=y.dtype)
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + W] = y[:, :, step * i:step * i + W]

        # Learned fusion: [y_shift, Phi] -> z
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z

    def forward_test(self, data):
        """Run unrolled reconstruction.

        data: dict with keys:
            'Y': (B, H, W_meas) measurement in shifted domain (W_meas = W + (nC-1)*step)
            'mask': (B, nC, H, W_meas) shifted mask3d (mask_3d_shift)
            'H': (B, nC, H, W) initial estimate from shift_back(Y/nC*2) [optional, for shape]
        """
        y = data['Y']        # (B, H, W_meas)
        phi = data['mask']   # (B, nC, H, W_meas)
        B, nC, H, W_meas = phi.shape
        W = W_meas - (nC - 1) * self.step  # image width (256)

        # Use H from data if available (for W_ reference), else compute W
        if 'H' in data:
            W_ = data['H'].shape[-1]
        else:
            W_ = W

        # Learned initialization
        z = self.initial(y, phi)

        z_hat = z
        z_list = [z]
        beta = 0.5 * torch.ones((self.stage, 1), device=y.device)

        for i in range(self.stage):
            # 1. Degradation estimation
            Phi, mu, noise_level = self.DP(z, phi)

            # 2. Data consistency (mu in denominator = regularization)
            Phi_s = torch.sum(Phi ** 2, 1)  # (B, H, W_meas)
            Phi_s[Phi_s == 0] = 1
            Phi_z = A(z_hat, Phi)  # (B, H, W_meas) — forward model
            x = z + At(torch.div(y - Phi_z, mu + Phi_s), Phi)

            # 3. Convert to image domain for denoiser
            x = shift_back_3d(x)[:, :, :, :W_]  # (B, nC, H, W)
            noise_level_repeat = noise_level.repeat(1, 1, x.shape[2], x.shape[3])

            # 4. Denoise with LNLT (with noise level channel)
            z = self.PP(torch.cat([noise_level_repeat, x], dim=1))

            # 5. Back to shifted domain
            z = shift_3d(z)  # (B, nC, H, W_meas)

            z_list.append(z)

            # 6. Nesterov-like momentum
            z_hat = z + beta[i] * (z_list[-1] - z_list[-2])

        # Final: convert to image domain
        z = shift_back_3d(z)[:, :, :, :W_]
        return z

    def forward(self, *args, **kwargs):
        return self.forward_test(*args, **kwargs)
