"""HSI-SDeCNN: Single Model CNN for Hyperspectral Image Denoising.

DnCNN-style architecture adapted for hyperspectral denoising.
Used as plug-in denoiser for GAP-based CASSI reconstruction.

Original architecture (Maffei et al.):
- Processes 25-band spectral windows with SubP (PixelUnshuffle/PixelShuffle)
- 14 Conv-ReLU layers with 128 features
- Noise-level (sigma) concatenation as extra input channel
- Input: (25, H, W) → PixelUnshuffle → (101, H/2, W/2) → convs → (4, H/2, W/2) → PixelShuffle → (1, H, W)

References:
- Maffei, A. et al. (2020). "A Single Model CNN for Hyperspectral Image
  Denoising", IEEE TGRS.

Benchmark (KAIST, GAP+HSI-SDeCNN):
- PSNR: 29.22 dB
- Params: 0.56M
"""

from __future__ import annotations

import warnings
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

# Default weight paths relative to package root
_PKG_ROOT = Path(__file__).resolve().parent.parent
_PKG_TOP = _PKG_ROOT.parent
_DEFAULT_WEIGHTS = _PKG_TOP / "weights" / "hsi_sdecnn" / "hsi_sdecnn.pth"
_DEFAULT_WEIGHTS_ALT = _PKG_ROOT / "weights" / "hsi_sdecnn" / "hsi_sdecnn.pth"

_NCH = 25  # Number of spectral bands in sliding window
_NUM_FEATURES = 128
_NUM_CONV_LAYERS = 14


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "HSI-SDeCNN requires PyTorch. Install with: pip install torch"
        )


def _find_weights(weights_path: Optional[str] = None) -> Optional[Path]:
    """Search standard locations for HSI-SDeCNN weights."""
    if weights_path is not None:
        p = Path(weights_path)
        if p.exists():
            return p
    for candidate in [_DEFAULT_WEIGHTS, _DEFAULT_WEIGHTS_ALT]:
        if candidate.exists():
            return candidate
    return None


# ============================================================================
# Original HSI-SDeCNN model (matches pretrained MATLAB weights)
# ============================================================================

if HAS_TORCH:

    class HSI_SDeCNN(nn.Module):
        """Original HSI-SDeCNN with SubP layers and sigma concatenation.

        Processes 25-band spectral windows through:
        1. PixelUnshuffle(2): (25, H, W) → (100, H/2, W/2)
        2. Concat sigma map: (101, H/2, W/2)
        3. 14 Conv-ReLU layers (128 features)
        4. PixelShuffle(2): (4, H/2, W/2) → (1, H, W)
        5. Residual: output = center_band - predicted_noise
        """

        def __init__(
            self,
            nch: int = _NCH,
            num_features: int = _NUM_FEATURES,
            num_conv_layers: int = _NUM_CONV_LAYERS,
        ):
            super().__init__()
            self.nch = nch
            in_ch = nch * 4 + 1  # SubP output + sigma channel

            layers = []
            # First conv
            layers.append(nn.Conv2d(in_ch, num_features, 3, 1, 1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            # Middle convs
            for _ in range(num_conv_layers - 2):
                layers.append(nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True))
                layers.append(nn.ReLU(inplace=True))
            # Last conv → 4 channels for PixelShuffle
            layers.append(nn.Conv2d(num_features, 4, 3, 1, 1, bias=True))

            self.body = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor", sigma: float = 0.1) -> "torch.Tensor":
            """Denoise center band of a spectral window.

            The network directly predicts the clean center band (no residual).

            Args:
                x: [B, nch, H, W] spectral window (nch=25 neighboring bands).
                sigma: Noise level (0-1 scale).

            Returns:
                Denoised center band [B, 1, H, W].
            """
            B, C, H_orig, W_orig = x.shape
            # Pad to even dimensions
            pad_h = H_orig % 2
            pad_w = W_orig % 2
            if pad_h or pad_w:
                x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
            _, _, H, W = x.shape

            # PixelUnshuffle (space-to-depth)
            x_down = F.pixel_unshuffle(x, 2)  # [B, C*4, H/2, W/2]

            # Concat sigma map
            sigma_map = torch.full(
                (B, 1, H // 2, W // 2), sigma,
                device=x.device, dtype=x.dtype,
            )
            x_in = torch.cat([x_down, sigma_map], dim=1)

            # Predict clean band in SubP space
            out_down = self.body(x_in)  # [B, 4, H/2, W/2]

            # PixelShuffle (depth-to-space) → clean center band
            out = F.pixel_shuffle(out_down, 2)  # [B, 1, H, W]

            # Remove padding
            if pad_h or pad_w:
                out = out[:, :, :H_orig, :W_orig]

            return out

    def _denoise_cube(
        model: "HSI_SDeCNN",
        cube: "torch.Tensor",
        sigma: float = 0.1,
    ) -> "torch.Tensor":
        """Denoise full HSI cube using sliding 25-band window.

        Args:
            model: Loaded HSI_SDeCNN model (eval mode).
            cube: [B, nC, H, W] full HSI cube.
            sigma: Noise level.

        Returns:
            Denoised cube [B, nC, H, W].
        """
        B, nC, H, W = cube.shape
        nch = model.nch
        half = nch // 2  # 12

        # Pad spectrally with mirror reflection
        pad_bands = cube.flip(dims=[1])
        # Left padding: bands half..1 (mirror)
        left_pad = cube[:, 1:half + 1].flip(dims=[1])
        # Right padding: bands nC-2..nC-half-1 (mirror)
        right_pad = cube[:, nC - half - 1:nC - 1].flip(dims=[1])
        padded = torch.cat([left_pad, cube, right_pad], dim=1)  # [B, nC + 2*half, H, W]

        output = torch.zeros(B, nC, H, W, device=cube.device, dtype=cube.dtype)
        for z in range(nC):
            window = padded[:, z:z + nch]  # [B, 25, H, W]
            output[:, z:z + 1] = model(window, sigma=sigma)

        return output


# ============================================================================
# Functional API
# ============================================================================


def hsi_sdecnn_denoise(
    hsi_cube: np.ndarray,
    sigma: float = 0.1,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Denoise a hyperspectral image cube using HSI-SDeCNN.

    Args:
        hsi_cube: HSI cube (H, W, nC) or (nC, H, W).
        sigma: Noise level (0-1 scale).
        weights_path: Path to pretrained weights.
        device: Torch device string.

    Returns:
        Denoised HSI cube with same shape as input.
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    transposed = False
    if hsi_cube.ndim == 3 and hsi_cube.shape[2] < hsi_cube.shape[0]:
        hsi_cube = hsi_cube.transpose(2, 0, 1)
        transposed = True

    x = torch.from_numpy(hsi_cube.copy()).unsqueeze(0).float().to(dev)

    model = HSI_SDeCNN().to(dev)
    wp = _find_weights(weights_path)
    if wp is not None:
        state = torch.load(str(wp), map_location=dev, weights_only=False)
        model.load_state_dict(state, strict=True)
    else:
        warnings.warn("HSI-SDeCNN: no pretrained weights found", stacklevel=2)

    model.eval()
    with torch.no_grad():
        out = _denoise_cube(model, x, sigma=sigma)

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
    step: int = 2,
    weights_path: Optional[str] = None,
    iters: int = 20,
    sigma: float = 0.39,
    sigma_end: float = 0.1,
    device: Optional[str] = None,
) -> np.ndarray:
    """GAP reconstruction with HSI-SDeCNN as the PnP denoiser.

    Uses plain GAP (no Nesterov acceleration) with sigma annealing from
    sigma to sigma_end over iterations.

    Args:
        meas: 2D CASSI measurement (H, W_ext) where W_ext = W + (n_bands-1)*step.
        mask: 2D coded aperture (H, W).
        n_bands: Number of spectral bands.
        step: CASSI dispersion step (pixels per band).
        weights_path: Path to HSI-SDeCNN weights.
        iters: Number of GAP iterations (default 20).
        sigma: Initial noise level for denoiser (default 0.39).
        sigma_end: Final noise level after annealing (default 0.1).
        device: Torch device string.

    Returns:
        Reconstructed spectral cube (H, W, n_bands).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build and load denoiser
    model = HSI_SDeCNN().to(dev)
    wp = _find_weights(weights_path)
    if wp is not None:
        state = torch.load(str(wp), map_location=dev, weights_only=False)
        model.load_state_dict(state, strict=True)
    else:
        warnings.warn("HSI-SDeCNN: no pretrained weights found", stacklevel=2)
    model.eval()

    h, w = mask.shape
    w_ext = w + (n_bands - 1) * step

    meas = meas[:, :w_ext].astype(np.float32)
    mask = mask.astype(np.float32)

    # Precompute Phi^T Phi diagonal
    Phi_sum = np.zeros((h, w_ext), dtype=np.float32)
    for k in range(n_bands):
        Phi_sum[:, k * step:k * step + w] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    # Initialize via back-projection
    x = np.zeros((h, w, n_bands), dtype=np.float32)
    for k in range(n_bands):
        x[:, :, k] = (mask * meas[:, k * step:k * step + w]
                       / np.maximum(Phi_sum[:, k * step:k * step + w], 1e-6))

    for it in range(iters):
        # Forward model
        y_est = np.zeros((h, w_ext), dtype=np.float32)
        for k in range(n_bands):
            y_est[:, k * step:k * step + w] += mask * x[:, :, k]

        # Plain GAP back-projection (no Nesterov — stabler with learned denoiser)
        residual = meas - y_est
        for k in range(n_bands):
            x[:, :, k] += (mask * residual[:, k * step:k * step + w]
                           / np.maximum(Phi_sum[:, k * step:k * step + w], 1e-6))

        # Sigma annealing
        frac = it / max(iters - 1, 1)
        sig = sigma * (1 - frac) + sigma_end * frac

        # HSI-SDeCNN denoising (sliding 25-band window)
        x_t = torch.from_numpy(
            x.transpose(2, 0, 1).copy()
        ).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            x_t = _denoise_cube(model, x_t, sigma=sig)
        x = x_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

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
    """Portfolio-compatible wrapper for GAP + HSI-SDeCNN."""
    iters = cfg.get("iters", 50)
    acc = cfg.get("acc", 1.0)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)
    step = cfg.get("step", 2)

    info: Dict[str, Any] = {"solver": "gap_hsi_sdecnn", "iters": iters}

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
            y, mask, n_bands=n_bands, step=step,
            weights_path=weights_path, iters=iters,
            device=device,
        )
        return result, info

    except Exception as e:
        info["error"] = str(e)
        if hasattr(physics, "adjoint"):
            return physics.adjoint(y).astype(np.float32), info
        return y.astype(np.float32), info
