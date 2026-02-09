"""HSI-SDeCNN: Single Model CNN for Hyperspectral Image Denoising.

DnCNN-style architecture adapted for hyperspectral denoising.
Used as plug-in denoiser for GAP-based CASSI reconstruction.

References:
- Maffei, A. et al. (2020). "A Single Model CNN for Hyperspectral Image
  Denoising", IEEE TGRS.

Benchmark (KAIST, GAP+HSI-SDeCNN):
- PSNR: 29.22 dB
- Params: 0.56M
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Default weight path relative to package root
_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "hsi_sdecnn" / "hsi_sdecnn.pth"

_NUM_LAYERS = 17
_NUM_FEATURES = 64


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "HSI-SDeCNN requires PyTorch. Install with: pip install torch"
        )


# ============================================================================
# Model
# ============================================================================


if HAS_TORCH:

    class HSI_SDeCNN(nn.Module):
        """Single Model CNN for Hyperspectral Image Denoising.

        DnCNN-style residual network with 17 Conv2d layers (64 channels).
        Treats each spectral band independently via shared 2D convolutions,
        then predicts the noise residual.

        Args:
            num_features: Number of intermediate feature channels (default: 64).
            num_layers: Total number of convolutional layers (default: 17).

        Input:
            (B, nC, H, W) noisy HSI cube (nC spectral bands).

        Output:
            (B, nC, H, W) denoised HSI cube.
        """

        def __init__(
            self,
            num_features: int = _NUM_FEATURES,
            num_layers: int = _NUM_LAYERS,
        ):
            super().__init__()
            layers = []

            # First layer: Conv2d (no BN, no ReLU at input)
            layers.append(nn.Conv2d(1, num_features, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))

            # Middle layers: Conv2d + BN + ReLU
            for _ in range(num_layers - 2):
                layers.append(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False)
                )
                layers.append(nn.BatchNorm2d(num_features))
                layers.append(nn.ReLU(inplace=True))

            # Last layer: Conv2d producing single-channel noise estimate
            layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1, bias=True))

            self.body = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass with residual learning.

            The network predicts the noise component; the clean image is
            obtained by subtracting the predicted noise from the input.

            Each spectral band is processed independently through shared
            weights by reshaping (B, nC) into the batch dimension.

            Args:
                x: Noisy HSI cube [B, nC, H, W].

            Returns:
                Denoised HSI cube [B, nC, H, W].
            """
            B, nC, H, W = x.shape
            # Merge spectral bands into batch: (B*nC, 1, H, W)
            x_in = x.reshape(B * nC, 1, H, W)
            noise = self.body(x_in)
            denoised = x_in - noise
            return denoised.reshape(B, nC, H, W)


# ============================================================================
# Functional API
# ============================================================================


def hsi_sdecnn_denoise(
    hsi_cube: np.ndarray,
    sigma: Optional[float] = None,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Denoise a hyperspectral image cube using HSI-SDeCNN.

    Args:
        hsi_cube: HSI cube as numpy array.  Accepted layouts:
            - (H, W, nC) -- will be transposed internally
            - (nC, H, W)
        sigma: Noise level hint (unused by current model, reserved for
            future sigma-conditional variants).
        weights_path: Path to pretrained ``.pth`` weights.  Falls back to
            the default package location when *None*.
        device: Torch device string (default: auto-detect CUDA/CPU).

    Returns:
        Denoised HSI cube with the **same shape** as input.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Determine layout and normalise to (nC, H, W)
    transposed = False
    if hsi_cube.ndim == 3 and hsi_cube.shape[2] < hsi_cube.shape[0]:
        # (H, W, nC) -> (nC, H, W)
        hsi_cube = hsi_cube.transpose(2, 0, 1)
        transposed = True

    x = torch.from_numpy(hsi_cube.copy()).unsqueeze(0).float().to(device)

    # Build model and load weights
    model = HSI_SDeCNN().to(device)

    wp = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
    if wp.exists():
        state = torch.load(str(wp), map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state, strict=False)

    model.eval()
    with torch.no_grad():
        out = model(x)

    result = out.squeeze(0).cpu().numpy()
    result = np.clip(result, 0, 1).astype(np.float32)

    if transposed:
        result = result.transpose(1, 2, 0)

    return result


# ============================================================================
# GAP + HSI-SDeCNN CASSI reconstruction
# ============================================================================


def gap_sdecnn_cassi(
    meas: np.ndarray,
    mask: np.ndarray,
    n_bands: int = 28,
    weights_path: Optional[str] = None,
    iters: int = 50,
    acc: float = 1.0,
    device: Optional[str] = None,
) -> np.ndarray:
    """GAP reconstruction with HSI-SDeCNN as the PnP denoiser.

    Replaces the TV denoising step in gap_tv_cassi with a learned
    HSI-SDeCNN denoiser, yielding better spectral fidelity.

    Args:
        meas: 2D CASSI measurement (H, W + n_bands - 1).
        mask: 2D coded aperture (H, W).
        n_bands: Number of spectral bands.
        weights_path: Path to HSI-SDeCNN weights (optional).
        iters: Number of GAP iterations.
        acc: Acceleration parameter.
        device: Torch device string.

    Returns:
        Reconstructed spectral cube (H, W, n_bands).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build and load denoiser once
    model = HSI_SDeCNN().to(dev)
    wp = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
    if wp.exists():
        state = torch.load(str(wp), map_location=dev, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state, strict=False)
    model.eval()

    h, w_meas = meas.shape
    w = w_meas - n_bands + 1

    # Initialise estimate via back-projection
    x = np.zeros((h, w, n_bands), dtype=np.float32)
    mask_sum = n_bands * mask ** 2 + 1e-10

    for k in range(n_bands):
        x[:, :, k] = mask * meas[:, k : k + w] / mask_sum

    for _it in range(iters):
        # Forward model
        y_est = np.zeros_like(meas)
        for k in range(n_bands):
            y_est[:, k : k + w] += mask * x[:, :, k]

        # Back-project residual
        residual = meas - y_est
        x_update = np.zeros_like(x)
        for k in range(n_bands):
            x_update[:, :, k] = mask * residual[:, k : k + w]

        x = x + acc * x_update / mask_sum[:, :, np.newaxis]

        # HSI-SDeCNN denoising step (nC, H, W) layout
        x_t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            x_t = model(x_t)
        x = x_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Non-negativity
        x = np.maximum(x, 0)

    return np.clip(x, 0, 1).astype(np.float32)


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_hsi_sdecnn(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for GAP + HSI-SDeCNN.

    Args:
        y: 2D CASSI measurement.
        physics: Physics operator (must expose mask and n_bands).
        cfg: Configuration dict with optional keys:
            - iters: GAP iterations (default: 50)
            - acc: acceleration (default: 1.0)
            - weights_path: path to HSI-SDeCNN weights

    Returns:
        Tuple of (reconstructed cube, info dict).
    """
    iters = cfg.get("iters", 50)
    acc = cfg.get("acc", 1.0)
    weights_path = cfg.get("weights_path", None)

    info: Dict[str, Any] = {
        "solver": "gap_hsi_sdecnn",
        "iters": iters,
    }

    try:
        mask = getattr(physics, "mask", None)
        n_bands = getattr(physics, "n_bands", 28)

        if hasattr(physics, "info"):
            op_info = physics.info()
            if mask is None:
                mask = op_info.get("mask", None)
            n_bands = op_info.get("n_bands", op_info.get("n_channels", n_bands))

        if mask is None:
            info["error"] = "physics has no mask attribute"
            return y.astype(np.float32), info

        result = gap_sdecnn_cassi(
            y, mask, n_bands=n_bands,
            weights_path=weights_path, iters=iters, acc=acc,
        )
        return result, info

    except Exception as e:
        info["error"] = str(e)
        # Graceful fallback to adjoint if available
        if hasattr(physics, "adjoint"):
            return physics.adjoint(y).astype(np.float32), info
        return y.astype(np.float32), info
