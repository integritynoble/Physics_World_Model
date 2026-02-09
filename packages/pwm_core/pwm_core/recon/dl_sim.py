"""DL-SIM: Deep Learning for Structured Illumination Microscopy.

Supervised CNN for SIM reconstruction with 2x super-resolution.

References:
- Jin, L. et al. (2020). "Deep learning enables structured illumination
  microscopy with low light levels and enhanced speed", Nature Communications.

Benchmark:
- Comparable to traditional SIM at much lower light levels
- Params: ~3M, VRAM: <1GB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "dl_sim" / "dl_sim.pth"


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "DL-SIM requires PyTorch. Install with: pip install torch"
        )


# ============================================================================
# Model components
# ============================================================================

if HAS_TORCH:

    class ResBlock(nn.Module):
        """Residual block: Conv-BN-ReLU-Conv-BN + skip connection."""

        def __init__(self, channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return F.relu(self.block(x) + x, inplace=True)

    class DLSIMNet(nn.Module):
        """U-Net with residual blocks for SIM super-resolution.

        Input:  (B, n_angles * n_phases, H, W)  e.g. (B, 9, H, W)
        Output: (B, 1, 2H, 2W)

        Architecture:
            - Conv stem
            - 3-level encoder with ResBlocks and max-pool downsampling
            - 3-level decoder with bilinear upsampling and skip connections
            - PixelShuffle 2x upsampling
            - Conv head

        Params: ~3M
        """

        def __init__(self, in_channels: int = 9, base_dim: int = 64):
            super().__init__()
            d = base_dim  # 64

            # Stem
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, d, 3, padding=1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )

            # Encoder
            self.enc1 = nn.Sequential(ResBlock(d), ResBlock(d))
            self.down1 = nn.MaxPool2d(2)

            self.enc2 = nn.Sequential(ResBlock(d * 1), ResBlock(d * 1))
            self.proj2 = nn.Sequential(
                nn.Conv2d(d, d * 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.ReLU(inplace=True),
            )
            self.down2 = nn.MaxPool2d(2)

            self.enc3 = nn.Sequential(ResBlock(d * 2), ResBlock(d * 2))
            self.proj3 = nn.Sequential(
                nn.Conv2d(d * 2, d * 4, 1, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.ReLU(inplace=True),
            )
            self.down3 = nn.MaxPool2d(2)

            # Bottleneck
            self.bottleneck = nn.Sequential(ResBlock(d * 4), ResBlock(d * 4))

            # Decoder
            self.up3 = nn.Sequential(
                nn.Conv2d(d * 4, d * 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.ReLU(inplace=True),
            )
            self.dec3 = nn.Sequential(ResBlock(d * 2), ResBlock(d * 2))

            self.up2 = nn.Sequential(
                nn.Conv2d(d * 2, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
            self.dec2 = nn.Sequential(ResBlock(d), ResBlock(d))

            self.up1 = nn.Sequential(
                nn.Conv2d(d, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True),
            )
            self.dec1 = nn.Sequential(ResBlock(d), ResBlock(d))

            # PixelShuffle for 2x super-resolution: d -> d*4 -> d, scale=2
            self.upsample = nn.Sequential(
                nn.Conv2d(d, d * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
            )

            # Output head
            self.head = nn.Conv2d(d, 1, 3, padding=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                x: (B, in_channels, H, W) raw SIM pattern images.

            Returns:
                (B, 1, 2H, 2W) super-resolved image.
            """
            # Pad to multiple of 8 for 3 levels of 2x downsampling
            _, _, h, w = x.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

            # Encoder
            s1 = self.stem(x)
            e1 = self.enc1(s1)

            d1 = self.down1(e1)
            e2 = self.proj2(self.enc2(d1))

            d2 = self.down2(e2)
            e3 = self.proj3(self.enc3(d2))

            d3 = self.down3(e3)

            # Bottleneck
            b = self.bottleneck(d3)

            # Decoder with skip connections (additive)
            u3 = self.up3(F.interpolate(
                b, size=e3.shape[2:], mode="bilinear", align_corners=False
            ))
            o3 = self.dec3(u3 + e3)

            u2 = self.up2(F.interpolate(
                o3, size=e2.shape[2:], mode="bilinear", align_corners=False
            ))
            o2 = self.dec2(u2 + e2)

            u1 = self.up1(F.interpolate(
                o2, size=e1.shape[2:], mode="bilinear", align_corners=False
            ))
            o1 = self.dec1(u1 + e1)

            # 2x super-resolution via PixelShuffle
            up = self.upsample(o1)
            out = self.head(up)

            # Remove padding (output is 2x input)
            out_h = h * 2
            out_w = w * 2
            return out[:, :, :out_h, :out_w]


# ============================================================================
# High-level reconstruction function
# ============================================================================


def dl_sim_reconstruct(
    raw_images: np.ndarray,
    weights_path: Optional[str] = None,
    n_angles: int = 3,
    n_phases: int = 3,
    device: Optional[str] = None,
) -> np.ndarray:
    """Reconstruct super-resolved SIM image using DL-SIM.

    Args:
        raw_images: Raw SIM data (n_angles * n_phases, H, W) as numpy float.
        weights_path: Path to pretrained weights. If None, uses default
            location ``{pkg_root}/weights/dl_sim/dl_sim.pth``.
        n_angles: Number of illumination angles (default 3).
        n_phases: Number of phase shifts per angle (default 3).
        device: Torch device string (default: auto-detect).

    Returns:
        Super-resolved image (2H, 2W) as float32 numpy array.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n_total = n_angles * n_phases
    if raw_images.ndim != 3 or raw_images.shape[0] != n_total:
        raise ValueError(
            f"Expected raw_images shape ({n_total}, H, W), "
            f"got {raw_images.shape}"
        )

    in_channels = n_total
    model = DLSIMNet(in_channels=in_channels).to(device)

    # Resolve weights path
    if weights_path is None:
        weights_path = str(_DEFAULT_WEIGHTS)

    if Path(weights_path).exists():
        checkpoint = torch.load(
            weights_path, map_location=device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded DL-SIM weights from %s", weights_path)
    else:
        logger.warning(
            "DL-SIM weights not found at %s; using random init", weights_path
        )

    model.eval()

    # Prepare input tensor (B, C, H, W)
    x = torch.from_numpy(raw_images.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    # Convert to numpy (2H, 2W)
    result = out.squeeze(0).squeeze(0).cpu().numpy()
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_dl_sim(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for DL-SIM reconstruction.

    Args:
        y: Raw SIM images (n_images, H, W).
        physics: SIM physics operator (used for metadata only).
        cfg: Configuration dict with optional keys:
            - n_angles (int): Number of angles (default 3).
            - n_phases (int): Number of phases (default 3).
            - weights_path (str): Path to model weights.
            - device (str): Torch device.

    Returns:
        Tuple of (super-resolved image, info dict).
    """
    n_angles = cfg.get("n_angles", 3)
    n_phases = cfg.get("n_phases", 3)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)

    info: Dict[str, Any] = {
        "solver": "dl_sim",
        "n_angles": n_angles,
        "n_phases": n_phases,
    }

    try:
        result = dl_sim_reconstruct(
            raw_images=y,
            weights_path=weights_path,
            n_angles=n_angles,
            n_phases=n_phases,
            device=device,
        )
        return result, info
    except Exception as e:
        info["error"] = str(e)
        logger.error("DL-SIM reconstruction failed: %s", e)
        # Fallback: average raw images
        if y.ndim >= 3:
            fallback = np.mean(y, axis=0).astype(np.float32)
            return fallback, info
        return y.astype(np.float32), info
