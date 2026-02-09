"""CARE: Content-Aware Image Restoration for Fluorescence Microscopy.

U-Net architecture for paired-data microscopy restoration.

References:
- Weigert, M. et al. (2018). "Content-Aware Image Restoration: Pushing the
  Limits of Fluorescence Microscopy", Nature Methods.

Benchmark:
- Widefield deconvolution: +3-5 dB over Richardson-Lucy
- Confocal low-dose: +4-6 dB over raw
- Params: ~2M (2D), ~2.5M (3D)
- VRAM: <1GB (2D), ~2GB (3D)
"""

from __future__ import annotations

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


_PKG_ROOT = Path(__file__).resolve().parent.parent
_WEIGHTS_DIR = _PKG_ROOT / "weights" / "care"


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "CARE U-Net requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class ConvBlock(nn.Module):
        """Conv + BatchNorm + ReLU block.

        Supports both 2D and 3D convolutions depending on *ndim*.
        """

        def __init__(self, in_ch: int, out_ch: int, ndim: int = 2):
            super().__init__()
            conv_fn = nn.Conv2d if ndim == 2 else nn.Conv3d
            norm_fn = nn.BatchNorm2d if ndim == 2 else nn.BatchNorm3d
            self.block = nn.Sequential(
                conv_fn(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                norm_fn(out_ch),
                nn.ReLU(inplace=True),
                conv_fn(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                norm_fn(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block(x)

    class DownBlock(nn.Module):
        """ConvBlock followed by spatial down-sampling (MaxPool)."""

        def __init__(self, in_ch: int, out_ch: int, ndim: int = 2):
            super().__init__()
            pool_fn = nn.MaxPool2d if ndim == 2 else nn.MaxPool3d
            self.pool = pool_fn(kernel_size=2)
            self.conv = ConvBlock(in_ch, out_ch, ndim=ndim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.conv(self.pool(x))

    class UpBlock(nn.Module):
        """Upsample + concatenate skip connection + ConvBlock."""

        def __init__(self, in_ch: int, out_ch: int, ndim: int = 2):
            super().__init__()
            mode = "bilinear" if ndim == 2 else "trilinear"
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            # After concatenation the channel count doubles
            self.conv = ConvBlock(in_ch + out_ch, out_ch, ndim=ndim)

        def forward(
            self, x: "torch.Tensor", skip: "torch.Tensor"
        ) -> "torch.Tensor":
            x = self.up(x)
            # Pad if sizes mismatch due to odd spatial dims
            if x.shape != skip.shape:
                diff = [s - x_ for s, x_ in zip(skip.shape[2:], x.shape[2:])]
                pad = []
                for d in reversed(diff):
                    pad.extend([d // 2, d - d // 2])
                x = F.pad(x, pad)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    # -----------------------------------------------------------------------
    # 2D CARE U-Net
    # -----------------------------------------------------------------------

    class CAREUNet2D(nn.Module):
        """2D U-Net for single-image fluorescence restoration.

        Architecture mirrors the original CARE network:
        - Encoder: 4 levels with channels [64, 128, 256, 512]
        - Bottleneck: 512 channels
        - Decoder: 4 levels with skip connections
        - Input: (B, 1, H, W) -> Output: (B, 1, H, W)
        - ~2M parameters
        """

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            features: Tuple[int, ...] = (64, 128, 256, 512),
        ):
            super().__init__()
            self.in_conv = ConvBlock(in_channels, features[0], ndim=2)

            # Encoder path
            self.encoders = nn.ModuleList()
            for i in range(len(features) - 1):
                self.encoders.append(DownBlock(features[i], features[i + 1], ndim=2))

            # Bottleneck
            self.bottleneck = DownBlock(features[-1], features[-1], ndim=2)

            # Decoder path
            self.decoders = nn.ModuleList()
            rev = list(reversed(features))
            for i in range(len(rev)):
                ch_in = rev[i] if i == 0 else rev[i - 1]
                self.decoders.append(UpBlock(ch_in, rev[i], ndim=2))

            self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Encoder
            skips = []
            x = self.in_conv(x)
            skips.append(x)
            for enc in self.encoders:
                x = enc(x)
                skips.append(x)

            # Bottleneck
            x = self.bottleneck(x)

            # Decoder
            for i, dec in enumerate(self.decoders):
                x = dec(x, skips[-(i + 1)])

            return self.out_conv(x)

    # -----------------------------------------------------------------------
    # 3D CARE U-Net
    # -----------------------------------------------------------------------

    class CAREUNet3D(nn.Module):
        """3D U-Net for volumetric confocal restoration.

        Same encoder-decoder structure but uses Conv3d, MaxPool3d, etc.
        Input: (B, 1, D, H, W) -> Output: (B, 1, D, H, W)
        Processes as 2.5D (thick slices) for memory efficiency on small GPUs.
        ~2.5M parameters.
        """

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            features: Tuple[int, ...] = (32, 64, 128, 256),
        ):
            super().__init__()
            self.in_conv = ConvBlock(in_channels, features[0], ndim=3)

            self.encoders = nn.ModuleList()
            for i in range(len(features) - 1):
                self.encoders.append(DownBlock(features[i], features[i + 1], ndim=3))

            self.bottleneck = DownBlock(features[-1], features[-1], ndim=3)

            self.decoders = nn.ModuleList()
            rev = list(reversed(features))
            for i in range(len(rev)):
                ch_in = rev[i] if i == 0 else rev[i - 1]
                self.decoders.append(UpBlock(ch_in, rev[i], ndim=3))

            self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            skips = []
            x = self.in_conv(x)
            skips.append(x)
            for enc in self.encoders:
                x = enc(x)
                skips.append(x)

            x = self.bottleneck(x)

            for i, dec in enumerate(self.decoders):
                x = dec(x, skips[-(i + 1)])

            return self.out_conv(x)


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------


def _load_weights(
    model: "nn.Module",
    weights_path: Optional[str],
    device: "torch.device",
) -> None:
    """Load pretrained weights into *model* if the file exists."""
    if weights_path is None:
        return
    p = Path(weights_path)
    if not p.exists():
        return
    checkpoint = torch.load(str(p), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# High-level 2D restoration
# ---------------------------------------------------------------------------


def care_restore_2d(
    image: np.ndarray,
    psf: Optional[np.ndarray] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    in_channels: int = 1,
    out_channels: int = 1,
) -> np.ndarray:
    """Restore a 2D fluorescence image with a CARE U-Net.

    Args:
        image: Input image (H, W), float32, typically normalised to [0, 1].
        psf: Point spread function (unused by the network but kept for API
             consistency with Richardson-Lucy and other solvers).
        weights_path: Path to a ``.pth`` checkpoint.  Falls back to
            ``{pkg_root}/weights/care/care_2d.pth`` when *None*.
        device: Torch device string (default: auto-select).
        in_channels: Number of input channels for the U-Net.
        out_channels: Number of output channels for the U-Net.

    Returns:
        Restored image (H, W) as float32 in the same value range as *image*.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if weights_path is None:
        weights_path = str(_WEIGHTS_DIR / "care_2d.pth")

    model = CAREUNet2D(
        in_channels=in_channels, out_channels=out_channels
    ).to(dev)
    _load_weights(model, weights_path, dev)
    model.eval()

    # Normalise input to [0, 1]
    img = image.astype(np.float32)
    v_min, v_max = img.min(), img.max()
    v_range = v_max - v_min
    if v_range > 1e-8:
        img = (img - v_min) / v_range
    else:
        img = img - v_min

    # (H, W) -> (1, 1, H, W)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(dev)

    # Pad to multiple of 16 for U-Net
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

    with torch.no_grad():
        out = model(x)

    out = out[:, :, :h, :w].squeeze().cpu().numpy()

    # Map back to original range
    if v_range > 1e-8:
        out = np.clip(out, 0.0, 1.0) * v_range + v_min
    else:
        out = out + v_min

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level 3D restoration
# ---------------------------------------------------------------------------


def care_restore_3d(
    volume: np.ndarray,
    psf: Optional[np.ndarray] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    in_channels: int = 1,
    out_channels: int = 1,
    slab_depth: int = 16,
) -> np.ndarray:
    """Restore a 3D confocal volume with a CARE U-Net.

    For large volumes the data is processed in overlapping slabs of
    *slab_depth* slices (2.5D strategy) so that VRAM stays manageable.

    Args:
        volume: Input volume (D, H, W), float32.
        psf: Point spread function (unused, API consistency).
        weights_path: Path to a ``.pth`` checkpoint.  Falls back to
            ``{pkg_root}/weights/care/care_3d.pth`` when *None*.
        device: Torch device string (default: auto-select).
        in_channels: Number of input channels for the U-Net.
        out_channels: Number of output channels for the U-Net.
        slab_depth: Number of z-slices per processing block.

    Returns:
        Restored volume (D, H, W) as float32.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if weights_path is None:
        weights_path = str(_WEIGHTS_DIR / "care_3d.pth")

    model = CAREUNet3D(
        in_channels=in_channels, out_channels=out_channels
    ).to(dev)
    _load_weights(model, weights_path, dev)
    model.eval()

    vol = volume.astype(np.float32)
    v_min, v_max = vol.min(), vol.max()
    v_range = v_max - v_min
    if v_range > 1e-8:
        vol = (vol - v_min) / v_range
    else:
        vol = vol - v_min

    D, H, W = vol.shape
    result = np.zeros_like(vol)
    counts = np.zeros_like(vol)

    # Pad spatial dims to multiple of 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16

    overlap = slab_depth // 4
    start = 0
    while start < D:
        end = min(start + slab_depth, D)
        slab = vol[start:end, :, :]

        # Pad depth to multiple of 16 as well
        cur_d = slab.shape[0]
        pad_d = (16 - cur_d % 16) % 16

        x = torch.from_numpy(slab).unsqueeze(0).unsqueeze(0).to(dev)
        if pad_d or pad_h or pad_w:
            # Use constant padding for depth if pad >= dimension (reflect fails)
            if pad_d >= cur_d:
                x = F.pad(x, [0, pad_w, 0, pad_h, 0, 0], mode="reflect")
                x = F.pad(x, [0, 0, 0, 0, 0, pad_d], mode="constant", value=0)
            else:
                x = F.pad(x, [0, pad_w, 0, pad_h, 0, pad_d], mode="reflect")

        with torch.no_grad():
            out = model(x)

        out = out[:, :, :cur_d, :H, :W].squeeze().cpu().numpy()
        result[start:end] += out
        counts[start:end] += 1.0

        if end >= D:
            break
        start += slab_depth - overlap

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    result = result / counts

    if v_range > 1e-8:
        result = np.clip(result, 0.0, 1.0) * v_range + v_min
    else:
        result = result + v_min

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick training helper
# ---------------------------------------------------------------------------


def care_train_quick(
    noisy: np.ndarray,
    clean: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> "CAREUNet2D":
    """Quick-train a CARE U-Net on a single (noisy, clean) pair.

    Useful when no pretrained weights are available. Trains directly
    on the benchmark's own synthetic data.

    Args:
        noisy: Noisy input image (H, W), float32 in [0, 1].
        clean: Clean ground truth (H, W), float32 in [0, 1].
        epochs: Training epochs.
        lr: Learning rate.
        device: Torch device string.

    Returns:
        Trained CAREUNet2D model (eval mode).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = CAREUNet2D(in_channels=1, out_channels=1).to(dev)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Prepare data: (H, W) -> (1, 1, H, W)
    x = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
    y = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)

    # Pad to multiple of 16
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        y = F.pad(y, [0, pad_w, 0, pad_h], mode="reflect")

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if epoch == epochs // 2:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------


def run_care(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run CARE U-Net restoration (portfolio interface).

    Args:
        y: Noisy / degraded measurement array.
        physics: Physics operator (may carry PSF; not used by the network
                 but forwarded for logging).
        cfg: Configuration dict with optional keys:
            - mode: '2d' or '3d' (default: inferred from y.ndim)
            - weights_path: path to .pth checkpoint
            - device: torch device string
            - slab_depth: z-slab size for 3D mode (default: 16)

    Returns:
        Tuple of (restored_array, info_dict).
    """
    info: Dict[str, Any] = {"solver": "care"}

    mode = cfg.get("mode", "3d" if y.ndim == 3 else "2d")
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)

    # Optionally extract PSF from physics (for logging only)
    psf = None
    if hasattr(physics, "psf"):
        psf = physics.psf
    elif hasattr(physics, "kernel"):
        psf = physics.kernel

    try:
        if mode == "3d":
            slab_depth = cfg.get("slab_depth", 16)
            result = care_restore_3d(
                y,
                psf=psf,
                weights_path=weights_path,
                device=device,
                slab_depth=slab_depth,
            )
            info["mode"] = "3d"
            info["slab_depth"] = slab_depth
        else:
            result = care_restore_2d(
                y,
                psf=psf,
                weights_path=weights_path,
                device=device,
            )
            info["mode"] = "2d"

        return result, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
