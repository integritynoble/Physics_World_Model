"""MST (Mask-aware Spectral Transformer) for CASSI reconstruction.

Based on: "Mask-guided Spectral-wise Transformer for Efficient Hyperspectral
Image Reconstruction" (Cai et al., CVPR 2022).
Reference: https://github.com/caiyuanhao1998/MST

This module provides:
- MST model architecture (CASSI variant with 28-band input/output)
- Torch-based CASSI shift/shift_back helpers
- High-level mst_recon_cassi() function for use in benchmarks
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from einops import rearrange

    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("MST requires PyTorch. Install with: pip install torch")


def _rearrange_bhnd(t, num_heads):
    """Fallback for einops rearrange 'b n (h d) -> b h n d'."""
    if HAS_EINOPS:
        return rearrange(t, "b n (h d) -> b h n d", h=num_heads)
    b, n, hd = t.shape
    d = hd // num_heads
    return t.reshape(b, n, num_heads, d).permute(0, 2, 1, 3)


# ============================================================================
# CASSI shift operations (torch)
# ============================================================================


def shift_torch(inputs: "torch.Tensor", step: int = 2) -> "torch.Tensor":
    """Shift spectral bands for CASSI dispersion (forward).

    Args:
        inputs: [B, nC, H, W]
        step: dispersion step (pixels per band)

    Returns:
        [B, nC, H, W + (nC-1)*step]
    """
    bs, nC, row, col = inputs.shape
    output = torch.zeros(
        bs, nC, row, col + (nC - 1) * step, device=inputs.device, dtype=inputs.dtype
    )
    for i in range(nC):
        output[:, i, :, step * i : step * i + col] = inputs[:, i, :, :]
    return output


def shift_back_meas_torch(
    meas: "torch.Tensor", step: int = 2, nC: int = 28
) -> "torch.Tensor":
    """Convert 2D CASSI measurement to initial spectral estimate via shift_back.

    Args:
        meas: [B, H, W_ext] where W_ext = W + (nC-1)*step
        step: dispersion step
        nC: number of spectral bands

    Returns:
        [B, nC, H, W]
    """
    bs, row, col = meas.shape
    W = col - (nC - 1) * step
    output = torch.zeros(bs, nC, row, W, device=meas.device, dtype=meas.dtype)
    for i in range(nC):
        output[:, i, :, :] = meas[:, :, step * i : step * i + W]
    return output


def _shift_back_feature(
    inputs: "torch.Tensor", base_resolution: int, step: int = 2
) -> "torch.Tensor":
    """Shift back for feature maps inside model (handles multi-scale).

    Used by MaskGuidedMechanism to align shifted mask features back to
    square spatial dimensions at each resolution level in the U-Net.

    Args:
        inputs: [B, nC, row, col] feature map
        base_resolution: full spatial resolution (e.g., 256)
        step: base dispersion step

    Returns:
        [B, nC, row, row] aligned features
    """
    bs, nC, row, col = inputs.shape
    down_sample = max(base_resolution // row, 1)
    effective_step = float(step) / float(down_sample * down_sample)
    out_col = row
    output = torch.zeros(
        bs, nC, row, out_col, device=inputs.device, dtype=inputs.dtype
    )
    for i in range(nC):
        offset = int(effective_step * i)
        end = offset + out_col
        if end <= col:
            output[:, i, :, :] = inputs[:, i, :, offset:end]
        else:
            valid = col - offset
            if valid > 0:
                output[:, i, :, :valid] = inputs[:, i, :, offset:col]
    return output


# ============================================================================
# Model components
# ============================================================================


if HAS_TORCH:

    class GELU(nn.Module):
        def forward(self, x):
            return F.gelu(x)

    class PreNorm(nn.Module):
        def __init__(self, dim, fn):
            super().__init__()
            self.fn = fn
            self.norm = nn.LayerNorm(dim)

        def forward(self, x, *args, **kwargs):
            x = self.norm(x)
            return self.fn(x, *args, **kwargs)

    class MaskGuidedMechanism(nn.Module):
        """Process shifted mask to produce attention modulation weights."""

        def __init__(
            self, n_feat: int, base_resolution: int = 256, step: int = 2
        ):
            super().__init__()
            self.base_resolution = base_resolution
            self.step = step
            self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
            self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True)
            self.depth_conv = nn.Conv2d(
                n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat
            )

        def forward(self, mask_shift):
            mask_shift = self.conv1(mask_shift)
            attn_map = torch.sigmoid(self.depth_conv(self.conv2(mask_shift)))
            res = mask_shift * attn_map
            mask_shift = res + mask_shift
            mask_emb = _shift_back_feature(
                mask_shift, self.base_resolution, self.step
            )
            return mask_emb

    class MS_MSA(nn.Module):
        """Mask-guided Spectral-wise Multi-head Self-Attention."""

        def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            base_resolution=256,
            step=2,
        ):
            super().__init__()
            self.num_heads = heads
            self.dim_head = dim_head
            self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
            self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
            self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
            self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
            self.proj = nn.Linear(dim_head * heads, dim, bias=True)
            self.pos_emb = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
                GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            )
            self.mm = MaskGuidedMechanism(dim, base_resolution, step)
            self.dim = dim

        def forward(self, x_in, mask=None):
            """
            x_in: [B, H, W, C]
            mask: [B, H, W_mask, C] (shifted mask, wider than x)
            """
            b, h, w, c = x_in.shape
            x = x_in.reshape(b, h * w, c)
            q_inp = self.to_q(x)
            k_inp = self.to_k(x)
            v_inp = self.to_v(x)

            # Process mask through mask-guided mechanism
            # mask comes as [B, H, W_mask, C], permute to [B, C, H, W_mask]
            mask_attn = self.mm(mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            if b != 0:
                mask_attn = mask_attn[0, :, :, :].expand([b, h, w, c])

            q, k, v, mask_attn = map(
                lambda t: _rearrange_bhnd(t, self.num_heads),
                (q_inp, k_inp, v_inp, mask_attn.flatten(1, 2)),
            )
            v = v * mask_attn

            # Spectral-wise attention: transpose to [B, heads, dim_head, HW]
            q = q.transpose(-2, -1)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)
            q = F.normalize(q, dim=-1, p=2)
            k = F.normalize(k, dim=-1, p=2)
            attn = k @ q.transpose(-2, -1)
            attn = attn * self.rescale
            attn = attn.softmax(dim=-1)
            x = attn @ v

            x = x.permute(0, 3, 1, 2)
            x = x.reshape(b, h * w, self.num_heads * self.dim_head)
            out_c = self.proj(x).view(b, h, w, c)
            out_p = self.pos_emb(
                v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
            return out_c + out_p

    class FeedForward(nn.Module):
        def __init__(self, dim, mult=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
                GELU(),
                nn.Conv2d(
                    dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult
                ),
                GELU(),
                nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
            )

        def forward(self, x):
            """x: [B, H, W, C]"""
            return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    class MSAB(nn.Module):
        """Multi-Scale Attention Block."""

        def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
            base_resolution=256,
            step=2,
        ):
            super().__init__()
            self.blocks = nn.ModuleList([])
            for _ in range(num_blocks):
                self.blocks.append(
                    nn.ModuleList(
                        [
                            MS_MSA(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                base_resolution=base_resolution,
                                step=step,
                            ),
                            PreNorm(dim, FeedForward(dim=dim)),
                        ]
                    )
                )

        def forward(self, x, mask):
            """x: [B, C, H, W], mask: [B, C, H, W_mask]"""
            x = x.permute(0, 2, 3, 1)
            for attn, ff in self.blocks:
                x = attn(x, mask=mask.permute(0, 2, 3, 1)) + x
                x = ff(x) + x
            return x.permute(0, 3, 1, 2)

    class MST(nn.Module):
        """Mask-aware Spectral Transformer for CASSI reconstruction.

        Args:
            dim: base feature dimension (typically = number of spectral bands)
            stage: number of U-Net encoder/decoder stages
            num_blocks: attention block counts per stage (length = stage + 1)
            in_channels: input spectral channels (28 for CASSI)
            out_channels: output spectral channels (28 for CASSI)
            base_resolution: spatial resolution of input (for shift_back scaling)
            step: CASSI dispersion step in pixels per band
        """

        def __init__(
            self,
            dim: int = 28,
            stage: int = 2,
            num_blocks: List[int] = None,
            in_channels: int = 28,
            out_channels: int = 28,
            base_resolution: int = 256,
            step: int = 2,
        ):
            super().__init__()
            if num_blocks is None:
                num_blocks = [2, 4, 2]
            self.dim = dim
            self.stage = stage
            self.base_resolution = base_resolution
            self.step = step

            # Input projection
            self.embedding = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=False)

            # Encoder
            self.encoder_layers = nn.ModuleList([])
            dim_stage = dim
            for i in range(stage):
                self.encoder_layers.append(
                    nn.ModuleList(
                        [
                            MSAB(
                                dim=dim_stage,
                                num_blocks=num_blocks[i],
                                dim_head=dim,
                                heads=dim_stage // dim,
                                base_resolution=base_resolution,
                                step=step,
                            ),
                            nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                            nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                        ]
                    )
                )
                dim_stage *= 2

            # Bottleneck
            self.bottleneck = MSAB(
                dim=dim_stage,
                dim_head=dim,
                heads=dim_stage // dim,
                num_blocks=num_blocks[-1],
                base_resolution=base_resolution,
                step=step,
            )

            # Decoder
            self.decoder_layers = nn.ModuleList([])
            for i in range(stage):
                self.decoder_layers.append(
                    nn.ModuleList(
                        [
                            nn.ConvTranspose2d(
                                dim_stage,
                                dim_stage // 2,
                                stride=2,
                                kernel_size=2,
                                padding=0,
                                output_padding=0,
                            ),
                            nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                            MSAB(
                                dim=dim_stage // 2,
                                num_blocks=num_blocks[stage - 1 - i],
                                dim_head=dim,
                                heads=(dim_stage // 2) // dim,
                                base_resolution=base_resolution,
                                step=step,
                            ),
                        ]
                    )
                )
                dim_stage //= 2

            # Output projection
            self.mapping = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=False)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        def forward(
            self, x: "torch.Tensor", mask: "torch.Tensor" = None
        ) -> "torch.Tensor":
            """
            Args:
                x: initial estimate [B, in_channels, H, W]
                mask: shifted mask [B, in_channels, H, W + (nC-1)*step].
                      If None, uses zero mask (no mask guidance).

            Returns:
                Reconstructed HSI [B, out_channels, H, W]
            """
            b, c, h_inp, w_inp = x.shape

            if mask is None:
                w_mask = w_inp + (c - 1) * self.step
                mask = torch.zeros(
                    1, c, h_inp, w_mask, device=x.device, dtype=x.dtype
                )

            # Pad to multiple of 2^stage for U-Net
            pad_factor = 2**self.stage
            pad_h = (pad_factor - h_inp % pad_factor) % pad_factor
            pad_w = (pad_factor - w_inp % pad_factor) % pad_factor
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
                mask_pad_w = (pad_factor - mask.shape[3] % pad_factor) % pad_factor
                mask = F.pad(mask, [0, mask_pad_w, 0, pad_h], mode="reflect")

            # Embedding (input only; mask goes directly to attention blocks)
            fea = self.lrelu(self.embedding(x))

            # Encoder
            fea_encoder = []
            masks = []
            for msab, fea_down, mask_down in self.encoder_layers:
                fea = msab(fea, mask)
                masks.append(mask)
                fea_encoder.append(fea)
                fea = fea_down(fea)
                mask = mask_down(mask)

            # Bottleneck
            fea = self.bottleneck(fea, mask)

            # Decoder
            for i, (fea_up, fusion, msab) in enumerate(self.decoder_layers):
                fea = fea_up(fea)
                fea = fusion(
                    torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1)
                )
                mask = masks[self.stage - 1 - i]
                fea = msab(fea, mask)

            # Output mapping + residual connection
            out = self.mapping(fea) + x

            # Remove padding
            return out[:, :, :h_inp, :w_inp]


# ============================================================================
# High-level reconstruction function
# ============================================================================


MST_CONFIGS = {
    "mst_s": {
        "dim": 28, "stage": 2, "num_blocks": [2, 2, 2],
        "description": "MST-S: Small variant (~0.93M params)",
    },
    "mst_m": {
        "dim": 28, "stage": 2, "num_blocks": [2, 4, 2],
        "description": "MST-M: Medium variant (~1.5M params)",
    },
    "mst_l": {
        "dim": 28, "stage": 2, "num_blocks": [4, 7, 5],
        "description": "MST-L: Large variant (default, ~2.03M params)",
    },
    "mst_plus_plus": {
        "dim": 28, "stage": 2, "num_blocks": [4, 7, 5],
        "description": "MST++: NTIRE 2022 winner (~1.33M params, 35.99 dB)",
    },
}


def create_mst(
    variant: str = "mst_l",
    in_channels: int = 28,
    out_channels: int = 28,
    base_resolution: int = 256,
    step: int = 2,
) -> "MST":
    """Create MST model with a named variant configuration.

    Supported variants: mst_s, mst_m, mst_l, mst_plus_plus

    Args:
        variant: Model variant name
        in_channels: Number of input spectral channels
        out_channels: Number of output spectral channels
        base_resolution: Spatial resolution
        step: CASSI dispersion step

    Returns:
        MST model instance
    """
    _require_torch()
    cfg = MST_CONFIGS.get(variant, MST_CONFIGS["mst_l"])
    return MST(
        dim=cfg["dim"],
        stage=cfg["stage"],
        num_blocks=cfg["num_blocks"],
        in_channels=in_channels,
        out_channels=out_channels,
        base_resolution=base_resolution,
        step=step,
    )


def _find_mst_weights(variant: str = "mst_l", weights_path: Optional[str] = None) -> Optional[str]:
    """Search for MST weights in standard locations.

    Args:
        variant: Model variant name
        weights_path: Explicit path (takes priority)

    Returns:
        Path to weights file or None
    """
    if weights_path is not None and Path(weights_path).exists():
        return weights_path

    pkg_root = Path(__file__).resolve().parent.parent
    search_paths = [
        pkg_root / f"weights/mst/{variant}.pth",
        pkg_root / "weights/mst/mst_l.pth",  # fallback to mst_l
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    return None


def mst_recon_cassi(
    measurement: np.ndarray,
    mask_2d: np.ndarray,
    nC: int = 28,
    step: int = 2,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    variant: str = "mst_l",
) -> np.ndarray:
    """Reconstruct CASSI hyperspectral cube using MST.

    Args:
        measurement: 2D measurement [H, W_ext] where W_ext = W + (nC-1)*step
        mask_2d: 2D coded aperture [H, W]
        nC: number of spectral bands
        step: dispersion step
        weights_path: path to pretrained weights (optional)
        device: torch device string
        variant: MST variant ('mst_s', 'mst_m', 'mst_l', 'mst_plus_plus')

    Returns:
        Reconstructed cube [H, W, nC]
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    H, W = mask_2d.shape

    # Create model from variant config
    model = create_mst(
        variant=variant,
        in_channels=nC,
        out_channels=nC,
        base_resolution=H,
        step=step,
    ).to(device)

    # Load pretrained weights if available
    resolved_path = _find_mst_weights(variant, weights_path)
    if resolved_path is not None:
        checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Prepare mask: [H, W] -> [1, nC, H, W] -> shifted [1, nC, H, W_ext]
    mask_3d = np.tile(mask_2d[:, :, np.newaxis], (1, 1, nC))
    mask_3d_t = (
        torch.from_numpy(mask_3d.transpose(2, 0, 1).copy())
        .unsqueeze(0)
        .float()
        .to(device)
    )
    mask_3d_shift = shift_torch(mask_3d_t, step=step)

    # Prepare initial estimate from measurement
    meas_t = torch.from_numpy(measurement.copy()).unsqueeze(0).float().to(device)
    x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
    x_init = x_init / nC * 2  # Scaling from original MST code

    # Forward pass
    with torch.no_grad():
        recon = model(x_init, mask_3d_shift)

    # Convert to numpy [H, W, nC]
    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(recon, 0, 1).astype(np.float32)
