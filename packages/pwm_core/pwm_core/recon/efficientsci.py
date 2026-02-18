"""EfficientSCI: Video Snapshot Compressive Imaging Reconstruction.

Space-time factorized Transformer for CACTI/SCI reconstruction.

References:
- Wang, L. et al. (2023). "EfficientSCI: Densely Connected Network with
  Space-time Factorization for Large-scale Video Snapshot Compressive Imaging",
  CVPR 2023.

Benchmark (6 videos, 256x256x8):
- EfficientSCI-T: 33.53 dB (3.78M params)
- EfficientSCI-S: 35.20 dB (5.16M params)
- EfficientSCI-B: 36.78 dB (12.05M params)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "EfficientSCI requires PyTorch. Install with: pip install torch"
        )


# ============================================================================
# Variant configurations
# ============================================================================

VARIANT_CONFIGS = {
    "tiny": {"dim": 32, "num_blocks": 2},
    "small": {"dim": 48, "num_blocks": 4},
    "base": {"dim": 64, "num_blocks": 6},
}


# ============================================================================
# Model components
# ============================================================================


if HAS_TORCH:

    class ReversibleConv2d(nn.Module):
        """Convolution block with optional residual connection.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
            kernel_size: Convolution kernel size.
            residual: Whether to add a skip connection.
        """

        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int = 3,
            residual: bool = True,
        ):
            super().__init__()
            self.residual = residual and (in_ch == out_ch)
            padding = kernel_size // 2
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, 1, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, 1, padding, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.act = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.conv(x)
            if self.residual:
                out = out + x
            return self.act(out)

    class DenseBlock(nn.Module):
        """Dense connection block.

        Each layer receives concatenated outputs from all previous layers.

        Args:
            in_ch: Input channels.
            growth_rate: Channels added per dense layer.
            num_layers: Number of dense layers.
        """

        def __init__(self, in_ch: int, growth_rate: int, num_layers: int = 4):
            super().__init__()
            self.layers = nn.ModuleList()
            ch = in_ch
            for _ in range(num_layers):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(ch, growth_rate, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(growth_rate),
                        nn.LeakyReLU(0.1, inplace=True),
                    )
                )
                ch += growth_rate
            # 1x1 bottleneck to compress back
            self.bottleneck = nn.Conv2d(ch, in_ch, 1, 1, 0, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = [x]
            for layer in self.layers:
                out = layer(torch.cat(features, dim=1))
                features.append(out)
            return self.bottleneck(torch.cat(features, dim=1))

    class ResDNet(nn.Module):
        """Initial reconstruction network using dense blocks.

        Takes concatenated measurement and mask frames as input and produces
        an initial per-frame estimate.

        Args:
            n_frames: Number of compressed frames (B).
            dim: Feature dimension.
        """

        def __init__(self, n_frames: int = 8, dim: int = 32):
            super().__init__()
            in_ch = 1 + n_frames  # measurement + mask frames
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, dim, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.dense1 = DenseBlock(dim, growth_rate=dim // 2, num_layers=4)
            self.conv_mid = ReversibleConv2d(dim, dim, kernel_size=3, residual=True)
            self.dense2 = DenseBlock(dim, growth_rate=dim // 2, num_layers=4)
            self.tail = nn.Conv2d(dim, n_frames, 3, 1, 1, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, 1+B, H, W) concatenated measurement and mask frames.

            Returns:
                (batch, B, H, W) initial frame estimates.
            """
            fea = self.head(x)
            fea = self.dense1(fea) + fea
            fea = self.conv_mid(fea)
            fea = self.dense2(fea) + fea
            return self.tail(fea)

    class SpaceTimeSeparableAttention(nn.Module):
        """Factorized attention: spatial within frames, temporal across frames.

        Args:
            dim: Feature dimension per token.
            num_heads: Number of attention heads.
            n_frames: Number of temporal frames.
        """

        def __init__(self, dim: int, num_heads: int = 4, n_frames: int = 8):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.n_frames = n_frames
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5

            # Spatial self-attention projections
            self.spatial_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.spatial_proj = nn.Linear(dim, dim, bias=False)

            # Temporal self-attention projections
            self.temporal_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.temporal_proj = nn.Linear(dim, dim, bias=False)

            self.norm_s = nn.LayerNorm(dim)
            self.norm_t = nn.LayerNorm(dim)

        def _spatial_attention(self, x: "torch.Tensor") -> "torch.Tensor":
            """Spatial self-attention within each frame.

            Args:
                x: (batch*B, H*W, dim)

            Returns:
                (batch*B, H*W, dim)
            """
            b, n, c = x.shape
            qkv = self.spatial_qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, n, c)
            return self.spatial_proj(out)

        def _temporal_attention(self, x: "torch.Tensor") -> "torch.Tensor":
            """Temporal self-attention across frames at each spatial position.

            Args:
                x: (batch*H*W, B, dim)

            Returns:
                (batch*H*W, B, dim)
            """
            b, n, c = x.shape
            qkv = self.temporal_qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, n, c)
            return self.temporal_proj(out)

        def forward(self, x: "torch.Tensor", h: int, w: int) -> "torch.Tensor":
            """
            Args:
                x: (batch, B*H*W, dim) flattened space-time tokens.
                h: Spatial height.
                w: Spatial width.

            Returns:
                (batch, B*H*W, dim)
            """
            batch = x.shape[0]
            hw = h * w
            B = self.n_frames

            # Spatial attention: reshape to (batch*B, H*W, dim)
            xs = x.reshape(batch, B, hw, self.dim)
            xs = self.norm_s(xs)
            xs = xs.reshape(batch * B, hw, self.dim)
            xs = self._spatial_attention(xs)
            xs = xs.reshape(batch, B, hw, self.dim)
            x = x.reshape(batch, B, hw, self.dim) + xs

            # Temporal attention: reshape to (batch*H*W, B, dim)
            xt = x.permute(0, 2, 1, 3).reshape(batch * hw, B, self.dim)
            xt = self.norm_t(
                xt.reshape(batch, hw, B, self.dim)
            ).reshape(batch * hw, B, self.dim)
            xt = self._temporal_attention(xt)
            xt = xt.reshape(batch, hw, B, self.dim).permute(0, 2, 1, 3)
            x = x + xt

            return x.reshape(batch, B * hw, self.dim)

    class CFormerBlock(nn.Module):
        """Transformer block with space-time separable attention and FFN.

        Args:
            dim: Feature dimension.
            num_heads: Number of attention heads.
            n_frames: Number of temporal frames.
            mlp_ratio: FFN expansion ratio.
        """

        def __init__(
            self,
            dim: int,
            num_heads: int = 4,
            n_frames: int = 8,
            mlp_ratio: float = 2.0,
        ):
            super().__init__()
            self.attn = SpaceTimeSeparableAttention(
                dim=dim, num_heads=num_heads, n_frames=n_frames
            )
            self.norm = nn.LayerNorm(dim)
            hidden = int(dim * mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )

        def forward(
            self, x: "torch.Tensor", h: int, w: int
        ) -> "torch.Tensor":
            """
            Args:
                x: (batch, B*H*W, dim) space-time tokens.
                h: Spatial height.
                w: Spatial width.

            Returns:
                (batch, B*H*W, dim)
            """
            x = x + self.attn(x, h, w)
            x = x + self.ffn(self.norm(x))
            return x

    class CFormer(nn.Module):
        """Stack of CFormerBlocks for video refinement.

        Args:
            dim: Feature dimension.
            num_blocks: Number of transformer blocks.
            num_heads: Number of attention heads.
            n_frames: Number of temporal frames.
            mlp_ratio: FFN expansion ratio.
        """

        def __init__(
            self,
            dim: int = 32,
            num_blocks: int = 2,
            num_heads: int = 4,
            n_frames: int = 8,
            mlp_ratio: float = 2.0,
        ):
            super().__init__()
            self.n_frames = n_frames
            self.embed = nn.Conv2d(n_frames, dim, 3, 1, 1, bias=False)
            self.blocks = nn.ModuleList(
                [
                    CFormerBlock(
                        dim=dim,
                        num_heads=num_heads,
                        n_frames=n_frames,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(num_blocks)
                ]
            )
            self.head = nn.Conv2d(dim, n_frames, 3, 1, 1, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (batch, B, H, W) initial frame estimates from stage 1.

            Returns:
                (batch, B, H, W) refined frame estimates.
            """
            batch, B, h, w = x.shape
            # Embed: (batch, B, H, W) -> (batch, dim, H, W)
            fea = self.embed(x)
            dim = fea.shape[1]

            # Downsample spatially to keep attention tractable.
            # Spatial attention is O((H'*W')^2) per frame — cap at 1024 tokens.
            ds = 1
            while (h // (ds * 2)) * (w // (ds * 2)) >= 1024 and ds < 16:
                ds *= 2

            if ds > 1:
                fea_ds = F.avg_pool2d(fea, kernel_size=ds, stride=ds)
            else:
                fea_ds = fea
            h_ds, w_ds = fea_ds.shape[2], fea_ds.shape[3]

            # Expand to B space-time copies for factorized attention.
            # (batch, dim, h_ds, w_ds) -> (batch, B*h_ds*w_ds, dim)
            fea_exp = fea_ds.unsqueeze(1).expand(-1, B, -1, -1, -1)
            fea_tokens = fea_exp.reshape(batch, B, dim, h_ds * w_ds)
            fea_tokens = fea_tokens.permute(0, 1, 3, 2).reshape(
                batch, B * h_ds * w_ds, dim
            )

            for block in self.blocks:
                fea_tokens = block(fea_tokens, h_ds, w_ds)

            # Collapse B dimension (mean), reshape back to spatial
            fea_tokens = fea_tokens.reshape(batch, B, h_ds * w_ds, dim)
            fea_out = fea_tokens.mean(dim=1)  # (batch, h_ds*w_ds, dim)
            fea_out = fea_out.permute(0, 2, 1).reshape(batch, dim, h_ds, w_ds)

            if ds > 1:
                fea_out = F.interpolate(
                    fea_out, size=(h, w), mode="bilinear", align_corners=False
                )

            return self.head(fea_out) + x

    class EfficientSCI(nn.Module):
        """EfficientSCI: two-stage video SCI reconstruction.

        Stage 1: ResDNet for initial reconstruction from measurement + mask.
        Stage 2: CFormer (space-time factorized Transformer) for refinement.

        Args:
            n_frames: Number of compressed video frames.
            variant: Model size variant ('tiny', 'small', 'base').
            dim: Override feature dimension (uses variant default if None).
            num_blocks: Override CFormer block count (uses variant default if None).
        """

        def __init__(
            self,
            n_frames: int = 8,
            variant: str = "tiny",
            dim: Optional[int] = None,
            num_blocks: Optional[int] = None,
        ):
            super().__init__()
            vcfg = VARIANT_CONFIGS.get(variant, VARIANT_CONFIGS["tiny"])
            self.dim = dim if dim is not None else vcfg["dim"]
            self.num_blocks = (
                num_blocks if num_blocks is not None else vcfg["num_blocks"]
            )
            self.n_frames = n_frames
            self.variant = variant

            # Stage 1: initial reconstruction
            self.stage1 = ResDNet(n_frames=n_frames, dim=self.dim)

            # Stage 2: Transformer refinement
            num_heads = max(self.dim // 8, 1)
            self.stage2 = CFormer(
                dim=self.dim,
                num_blocks=self.num_blocks,
                num_heads=num_heads,
                n_frames=n_frames,
            )

        def forward(
            self,
            meas: "torch.Tensor",
            mask: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Args:
                meas: (batch, 1, H, W) snapshot measurement.
                mask: (batch, B, H, W) temporal mask frames.

            Returns:
                Tuple of (stage2_output, stage1_output), each (batch, B, H, W).
            """
            # Concatenate measurement and mask: (batch, 1+B, H, W)
            x_in = torch.cat([meas, mask], dim=1)
            x1 = self.stage1(x_in)
            x2 = self.stage2(x1)
            return x2, x1


# ============================================================================
# Weight loading utilities
# ============================================================================


def _find_weights(weights_path: Optional[str], variant: str) -> Optional[Path]:
    """Search for pretrained weights.

    Search order:
    1. Direct path (if provided and exists).
    2. {pkg_root}/weights/efficientsci/efficientsci_{variant}.pth

    Returns:
        Path to weights file, or None if not found.
    """
    if weights_path is not None:
        p = Path(weights_path)
        if p.exists():
            return p

    pkg_root = Path(__file__).parent.parent
    candidate = (
        pkg_root / "weights" / "efficientsci" / f"efficientsci_{variant}.pth"
    )
    if candidate.exists():
        return candidate

    return None


# ============================================================================
# High-level reconstruction function
# ============================================================================


def efficientsci_recon(
    meas: np.ndarray,
    mask: np.ndarray,
    weights_path: Optional[str] = None,
    variant: str = "tiny",
    device: Optional[str] = None,
) -> np.ndarray:
    """Reconstruct video from CACTI snapshot measurement using EfficientSCI.

    Args:
        meas: 2D snapshot measurement (H, W).
        mask: 3D mask tensor (H, W, B) with B temporal frames.
        weights_path: Path to pretrained weights (optional).
        variant: Model variant ('tiny', 'small', 'base').
        device: Torch device string.

    Returns:
        Reconstructed video frames (B, H, W).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    H, W = meas.shape[:2]
    n_frames = mask.shape[2]

    # Create model
    model = EfficientSCI(
        n_frames=n_frames,
        variant=variant,
    ).to(device)

    # Load pretrained weights if available
    wpath = _find_weights(weights_path, variant)
    if wpath is not None:
        checkpoint = torch.load(
            str(wpath), map_location=device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        warnings.warn(
            f"No pretrained weights found for EfficientSCI-{variant}. "
            "Running with random initialization.",
            stacklevel=2,
        )

    model.eval()

    # Prepare measurement: (H, W) -> (1, 1, H, W)
    meas_t = (
        torch.from_numpy(meas.copy())
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Prepare mask: (H, W, B) -> (1, B, H, W)
    mask_t = (
        torch.from_numpy(mask.transpose(2, 0, 1).copy())
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Forward pass
    with torch.no_grad():
        x2, x1 = model(meas_t, mask_t)

    # Convert to numpy: (1, B, H, W) -> (B, H, W)
    recon = x2.squeeze(0).cpu().numpy()
    return np.clip(recon, 0, 1).astype(np.float32)


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_efficientsci(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio wrapper for EfficientSCI reconstruction.

    Args:
        y: Snapshot measurement (H, W).
        physics: Physics operator (must have .masks attribute or info()).
        cfg: Configuration with:
            - variant: Model variant ('tiny', 'small', 'base'), default 'tiny'.
            - weights_path: Path to pretrained weights (optional).
            - device: Torch device string (optional).

    Returns:
        Tuple of (reconstructed video (B, H, W), info_dict).
    """
    variant = cfg.get("variant", "tiny")
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)

    info = {
        "solver": "efficientsci",
        "variant": variant,
    }

    try:
        # Extract masks from physics operator
        masks = None

        if hasattr(physics, "masks"):
            masks = physics.masks
        elif hasattr(physics, "info"):
            op_info = physics.info()
            masks = op_info.get("masks", None)

        if masks is None:
            info["error"] = "no masks found on physics operator"
            return y.astype(np.float32), info

        result = efficientsci_recon(
            meas=y,
            mask=masks,
            weights_path=weights_path,
            variant=variant,
            device=device,
        )
        return result, info

    except Exception as e:
        info["error"] = str(e)
        # Fallback: adjoint or identity
        if hasattr(physics, "adjoint"):
            result = physics.adjoint(y)
            return result.astype(np.float32), info
        return y.astype(np.float32), info
