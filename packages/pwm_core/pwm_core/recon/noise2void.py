"""Noise2Void: Self-Supervised Denoising from Single Noisy Images.

Blind-spot network for microscopy denoising without clean ground truth.

References:
- Krull, A. et al. (2019). "Noise2Void - Learning Denoising from Single
  Noisy Images", CVPR 2019.

Benchmark:
- Comparable to supervised denoising when no GT available
- Self-supervised: trains on the input image itself
- Params: ~1M, VRAM: <0.5GB
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_HAS_TORCH = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    pass

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "noise2void" / "n2v.pth"


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "Noise2Void requires PyTorch. Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class BlindSpotConv(nn.Module):
        """Convolution that excludes the center pixel.

        Uses a masked kernel so the center weight is forced to zero,
        preventing the network from learning the identity mapping during
        self-supervised training.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
        ):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            )
            # Register a persistent mask that zeros out the center weight
            mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
            center = kernel_size // 2
            mask[:, :, center, center] = 0.0
            self.register_buffer("mask", mask)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.conv.weight.data *= self.mask
            return self.conv(x)

    class _DoubleConv(nn.Module):
        """Two convolutions + BatchNorm + ReLU."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class Noise2VoidUNet(nn.Module):
        """U-Net with blind-spot constraint for Noise2Void.

        Encoder: 3 levels [32, 64, 128]
        Decoder: 3 levels with skip connections
        Input:  (B, 1, H, W)
        Output: (B, 1, H, W) denoised
        ~1M params
        """

        def __init__(self, in_channels: int = 1, out_channels: int = 1):
            super().__init__()
            features = [32, 64, 128]

            # Encoder
            self.enc1 = _DoubleConv(in_channels, features[0])
            self.enc2 = _DoubleConv(features[0], features[1])
            self.enc3 = _DoubleConv(features[1], features[2])

            self.pool = nn.MaxPool2d(2)

            # Bottleneck
            self.bottleneck = _DoubleConv(features[2], features[2])

            # Decoder
            self.up3 = nn.ConvTranspose2d(features[2], features[2], 2, stride=2)
            self.dec3 = _DoubleConv(features[2] * 2, features[1])

            self.up2 = nn.ConvTranspose2d(features[1], features[1], 2, stride=2)
            self.dec2 = _DoubleConv(features[1] * 2, features[0])

            self.up1 = nn.ConvTranspose2d(features[0], features[0], 2, stride=2)
            self.dec1 = _DoubleConv(features[0] * 2, features[0])

            # Blind-spot output head
            self.head = BlindSpotConv(features[0], out_channels, kernel_size=3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pad to multiple of 8 for three pooling levels
            _, _, h, w = x.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))

            b = self.bottleneck(self.pool(e3))

            d3 = self.up3(b)
            d3 = self._match_and_cat(d3, e3)
            d3 = self.dec3(d3)

            d2 = self.up2(d3)
            d2 = self._match_and_cat(d2, e2)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = self._match_and_cat(d1, e1)
            d1 = self.dec1(d1)

            out = self.head(d1)

            # Remove padding
            return out[:, :, :h, :w]

        @staticmethod
        def _match_and_cat(
            upsampled: torch.Tensor, skip: torch.Tensor
        ) -> torch.Tensor:
            """Crop or pad upsampled tensor to match skip, then concatenate."""
            dh = skip.shape[2] - upsampled.shape[2]
            dw = skip.shape[3] - upsampled.shape[3]
            if dh != 0 or dw != 0:
                upsampled = F.pad(upsampled, [0, dw, 0, dh], mode="reflect")
            return torch.cat([upsampled, skip], dim=1)


# ---------------------------------------------------------------------------
# Blind-spot masking helpers
# ---------------------------------------------------------------------------


def _create_blind_spot_mask(
    shape: Tuple[int, int],
    ratio: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create a random blind-spot mask for N2V training.

    Args:
        shape: (H, W) spatial dimensions.
        ratio: Fraction of pixels to mask (default 2%).
        rng: Numpy random generator for reproducibility.

    Returns:
        Boolean mask of shape (H, W), True at blind-spot locations.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_pixels = int(shape[0] * shape[1] * ratio)
    mask = np.zeros(shape, dtype=bool)
    indices = rng.choice(shape[0] * shape[1], size=n_pixels, replace=False)
    mask.ravel()[indices] = True
    return mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def n2v_train_single(
    noisy_image: np.ndarray,
    model: Optional[Any] = None,
    epochs: int = 200,
    lr: float = 1e-3,
    mask_ratio: float = 0.02,
    device: Optional[str] = None,
    seed: int = 42,
) -> Any:
    """Train a Noise2Void model on a single noisy image.

    Creates random blind-spot masks and runs a self-supervised
    training loop where the network predicts masked pixels from
    their neighbours.

    Args:
        noisy_image: 2D noisy input (H, W), float.
        model: Optional pre-initialised Noise2VoidUNet.
        epochs: Number of training epochs.
        lr: Learning rate.
        mask_ratio: Fraction of pixels masked per epoch.
        device: Torch device string (auto-detected if None).
        seed: Random seed.

    Returns:
        Trained Noise2VoidUNet model (on *device*).
    """
    _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    rng = np.random.default_rng(seed)

    # Normalise to [0, 1]
    img = noisy_image.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 1e-8:
        img = (img - vmin) / (vmax - vmin)

    # Prepare tensor (B=1, C=1, H, W)
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    if model is None:
        model = Noise2VoidUNet(in_channels=1, out_channels=1)
    model = model.to(device)
    model.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    h, w = img.shape
    for _epoch in range(epochs):
        mask = _create_blind_spot_mask((h, w), ratio=mask_ratio, rng=rng)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

        pred = model(img_t)

        # Loss only on masked (blind-spot) pixels
        loss = F.mse_loss(pred[mask_t], img_t[mask_t])

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def n2v_denoise(
    image: np.ndarray,
    weights_path: Optional[str] = None,
    pretrained_model: Optional[Any] = None,
    epochs: int = 200,
    device: Optional[str] = None,
) -> np.ndarray:
    """Denoise an image using Noise2Void.

    If a pretrained model or weights are available the image is denoised
    in a single forward pass.  Otherwise a quick self-supervised training
    loop is run on the input image itself (the core N2V idea).

    Args:
        image: 2D noisy image (H, W), float or uint.
        weights_path: Path to saved model weights (.pth).
        pretrained_model: An already-loaded Noise2VoidUNet instance.
        epochs: Training epochs when no pretrained model is available.
        device: Torch device string.

    Returns:
        Denoised image as numpy float32 array (H, W).
    """
    _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    img = image.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 1e-8:
        img_norm = (img - vmin) / (vmax - vmin)
    else:
        img_norm = img - vmin

    img_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    # Resolve model ----------------------------------------------------------
    model = None

    if pretrained_model is not None:
        model = pretrained_model
        model = model.to(device)

    elif weights_path is not None and Path(weights_path).exists():
        model = Noise2VoidUNet(in_channels=1, out_channels=1)
        state = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        model = model.to(device)

    elif _DEFAULT_WEIGHTS.exists():
        model = Noise2VoidUNet(in_channels=1, out_channels=1)
        state = torch.load(
            str(_DEFAULT_WEIGHTS), map_location=device, weights_only=False
        )
        model.load_state_dict(state, strict=False)
        model = model.to(device)

    # If no pretrained model, self-supervise on the input image itself
    if model is None:
        model = n2v_train_single(img, epochs=epochs, device=str(device))

    model.eval()

    with torch.no_grad():
        denoised = model(img_t)

    out = denoised.squeeze().cpu().numpy()

    # Denormalise
    if vmax - vmin > 1e-8:
        out = out * (vmax - vmin) + vmin

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------


def run_noise2void(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Noise2Void denoising (portfolio-compatible interface).

    Args:
        y: Noisy measurement (H, W).
        physics: Physics operator (unused for pure denoising).
        cfg: Configuration dict with optional keys:
            - weights_path: str, path to pretrained weights
            - epochs: int, self-supervised training epochs (default 200)
            - device: str, torch device

    Returns:
        Tuple of (denoised_image, info_dict).
    """
    info: Dict[str, Any] = {"solver": "noise2void"}

    try:
        weights = cfg.get("weights_path", None)
        epochs = cfg.get("epochs", 200)
        device = cfg.get("device", None)

        denoised = n2v_denoise(
            image=y,
            weights_path=weights,
            epochs=epochs,
            device=device,
        )

        info["epochs"] = epochs
        return denoised, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
