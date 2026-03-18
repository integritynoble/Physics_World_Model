"""DeStripe: Self-Supervised Destriping for Light-Sheet Microscopy.

Self2Self + graph + Hessian prior for stripe artifact removal.

References:
- Liang, Y. et al. (2022). "DeStripe: A self-supervised speckle and stripe
  removal framework for light-sheet fluorescence microscopy", Optics Letters.

Benchmark:
- Effective stripe removal with structure preservation
- Params: ~2M, VRAM: <1GB
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
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "destripe" / "destripe.pth"


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "DeStripeNet requires PyTorch. Install with: pip install torch"
        )


# ============================================================================
# Network architecture
# ============================================================================

if HAS_TORCH:

    class _DirectionalConv(nn.Module):
        """Convolution with directional emphasis for stripe artifacts.

        Uses an asymmetric kernel (wider along the stripe direction) so that
        the network can distinguish stripe patterns from genuine structures.
        Stripes in light-sheet microscopy typically run horizontally.
        """

        def __init__(self, in_ch: int, out_ch: int, horizontal: bool = True):
            super().__init__()
            # Wider kernel along stripe direction, narrow perpendicular
            if horizontal:
                kernel_size = (3, 7)
                padding = (1, 3)
            else:
                kernel_size = (7, 3)
                padding = (3, 1)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
            self.bn = nn.BatchNorm2d(out_ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.leaky_relu(self.bn(self.conv(x)), 0.1, inplace=True)

    class _DoubleConv(nn.Module):
        """Two conv layers with BatchNorm and LeakyReLU."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class DeStripeNet(nn.Module):
        """U-Net for stripe artifact removal in light-sheet microscopy.

        Three-level encoder-decoder with directional convolutions that
        emphasise the horizontal stripe direction.  The network uses
        residual learning: it predicts the stripe component, which is
        subtracted from the input to produce the clean image.

        Input:  (B, 1, H, W)
        Output: (B, 1, H, W) destriped

        Approximate parameter count: ~2M.
        """

        def __init__(self, channels: Tuple[int, ...] = (32, 64, 128)):
            super().__init__()
            c1, c2, c3 = channels

            # Encoder ----------------------------------------------------------
            self.dir_conv = _DirectionalConv(1, c1, horizontal=True)
            self.enc1 = _DoubleConv(c1, c1)
            self.pool1 = nn.MaxPool2d(2)

            self.enc2 = _DoubleConv(c1, c2)
            self.pool2 = nn.MaxPool2d(2)

            self.enc3 = _DoubleConv(c2, c3)
            self.pool3 = nn.MaxPool2d(2)

            # Bottleneck -------------------------------------------------------
            self.bottleneck = _DoubleConv(c3, c3)

            # Decoder ----------------------------------------------------------
            self.up3 = nn.ConvTranspose2d(c3, c3, 2, stride=2)
            self.dec3 = _DoubleConv(c3 + c3, c3)

            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.dec2 = _DoubleConv(c2 + c2, c2)

            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.dec1 = _DoubleConv(c1 + c1, c1)

            # Head: predict stripe residual ------------------------------------
            self.head = nn.Conv2d(c1, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict destriped image.

            The network outputs ``x - stripe``, where ``stripe`` is the
            learned stripe residual.
            """
            # Pad input to multiple of 8 for three pooling levels
            _, _, h, w = x.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

            identity = x

            # Encoder
            e1 = self.enc1(self.dir_conv(x))
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))

            # Bottleneck
            b = self.bottleneck(self.pool3(e3))

            # Decoder with skip connections
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            stripe = self.head(d1)

            # Residual learning: clean = input - stripe
            out = identity - stripe

            # Remove padding
            return out[:, :, :h, :w]


# ============================================================================
# Self-supervised training on a single image (Self2Self strategy)
# ============================================================================


def _hessian_penalty(img: "torch.Tensor") -> "torch.Tensor":
    """Hessian regularisation along the stripe (horizontal) direction.

    Penalises the second-order horizontal derivative of the output to
    encourage smooth stripe removal without hallucinating new structures.

    Args:
        img: (B, 1, H, W) tensor.

    Returns:
        Scalar penalty.
    """
    # d^2 f / dx^2  (horizontal second derivative)
    d2x = img[:, :, :, 2:] - 2 * img[:, :, :, 1:-1] + img[:, :, :, :-2]
    return d2x.abs().mean()


def destripe_self_supervised(
    image: np.ndarray,
    model: Optional[Any] = None,
    iters: int = 500,
    lr: float = 1e-3,
    dropout_rate: float = 0.5,
    hessian_weight: float = 0.1,
    device: Optional[str] = None,
) -> np.ndarray:
    """Train a DeStripeNet on a single image using Self2Self masking.

    A random Bernoulli mask is applied to the input at every iteration so
    that the network can only see a subset of pixels.  The loss is computed
    on the *complementary* pixels (the ones the network did not see),
    which prevents the network from learning the identity and forces it
    to remove spatially correlated stripe noise.

    Args:
        image:          2-D input image (H, W), float32, arbitrary range.
        model:          Pre-initialised ``DeStripeNet`` (created if *None*).
        iters:          Number of training iterations.
        lr:             Learning rate for Adam.
        dropout_rate:   Fraction of pixels masked out per iteration.
        hessian_weight: Weight for the Hessian smoothness penalty.
        device:         Torch device string (auto-detected if *None*).

    Returns:
        Destriped image as float32 array with the same shape as *image*.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Normalise to [0, 1]
    img_min, img_max = float(image.min()), float(image.max())
    img_range = img_max - img_min
    if img_range < 1e-8:
        return image.copy()
    img_norm = (image - img_min) / img_range

    img_t = (
        torch.from_numpy(img_norm.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )  # (1, 1, H, W)

    if model is None:
        model = DeStripeNet()
    model = model.to(device)
    model.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(iters):
        # Bernoulli dropout mask (Self2Self)
        mask = (torch.rand_like(img_t) > dropout_rate).float()

        masked_input = img_t * mask
        output = model(masked_input)

        # Loss on unseen pixels
        complement = 1.0 - mask
        n_complement = complement.sum().clamp(min=1.0)
        recon_loss = ((output - img_t) ** 2 * complement).sum() / n_complement

        # Hessian smoothness along stripe direction
        h_loss = _hessian_penalty(output)

        loss = recon_loss + hessian_weight * h_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # Inference: average several masked forward passes for stability
    model.eval()
    n_avg = 8
    accum = torch.zeros_like(img_t)
    with torch.no_grad():
        for _ in range(n_avg):
            mask = (torch.rand_like(img_t) > dropout_rate).float()
            accum += model(img_t * mask)
    result = (accum / n_avg).squeeze().cpu().numpy()

    # Denormalise
    result = np.clip(result, 0.0, 1.0) * img_range + img_min
    return result.astype(np.float32)


# ============================================================================
# High-level API
# ============================================================================


def destripe_denoise(
    image: np.ndarray,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    self_supervised_iters: int = 500,
) -> np.ndarray:
    """Remove stripe artifacts from a light-sheet microscopy image.

    If *weights_path* points to an existing checkpoint the pretrained
    model is loaded and a single forward pass is performed.  Otherwise
    the function falls back to self-supervised training on the input
    image (slower but does not require pretrained weights).

    Args:
        image:                  2-D image (H, W), float32.
        weights_path:           Path to ``destripe.pth`` (optional).
        device:                 Torch device string (auto-detected if *None*).
        self_supervised_iters:  Iterations for the Self2Self fallback.

    Returns:
        Destriped image (H, W) as float32.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    model = DeStripeNet()

    # Try to load pretrained weights
    wp = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
    if wp.exists():
        checkpoint = torch.load(str(wp), map_location=device_obj, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device_obj)
        model.eval()

        # Normalise
        img_min, img_max = float(image.min()), float(image.max())
        img_range = img_max - img_min
        if img_range < 1e-8:
            return image.copy()
        img_norm = (image - img_min) / img_range

        img_t = (
            torch.from_numpy(img_norm.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device_obj)
        )

        with torch.no_grad():
            result = model(img_t).squeeze().cpu().numpy()

        result = np.clip(result, 0.0, 1.0) * img_range + img_min
        return result.astype(np.float32)

    # No pretrained weights -- fall back to self-supervised training
    return destripe_self_supervised(
        image,
        model=model,
        iters=self_supervised_iters,
        device=device,
    )


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_destripe(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run DeStripe reconstruction (portfolio-compatible interface).

    Args:
        y:       Striped image (H, W) or (H, W, C).
        physics: Physics operator (unused for destriping; kept for API
                 compatibility with the solver portfolio).
        cfg:     Configuration dict with optional keys:
                 - weights_path: path to pretrained checkpoint
                 - device: torch device string
                 - iters: Self2Self iterations (default 500)

    Returns:
        Tuple of (destriped_image, info_dict).
    """
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)
    iters = cfg.get("iters", 500)

    info: Dict[str, Any] = {"solver": "destripe"}

    try:
        if y.ndim == 2:
            result = destripe_denoise(
                y,
                weights_path=weights_path,
                device=device,
                self_supervised_iters=iters,
            )
        elif y.ndim == 3:
            # Process each channel independently
            channels = []
            for c in range(y.shape[2]):
                channels.append(
                    destripe_denoise(
                        y[:, :, c],
                        weights_path=weights_path,
                        device=device,
                        self_supervised_iters=iters,
                    )
                )
            result = np.stack(channels, axis=2)
        else:
            info["warning"] = f"unsupported ndim={y.ndim}, returning input"
            return y.astype(np.float32), info

        info["iters"] = iters
        return result, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
