"""HSI-SDeCNN: Single Model CNN for Hyperspectral Image Denoising.

Used as plug-in denoiser for GAP-based PnP-CASSI reconstruction.

Architecture (from reference hsi.py / deep_denoiser.pth):
- in_nc=7 (7-band spectral context window: 3 before + current + 3 after)
- PixelUnShuffle(2): (7, H, W) -> (28, H/2, W/2)
- Concat sigma map: (29, H/2, W/2)
- 15 Conv-ReLU layers (128 features)
- PixelShuffle(2): (4, H/2, W/2) -> (1, H, W)

PnP-HSICNN (GAP + HSI-SDeCNN):
- 124-iter GAP with Nesterov acceleration
- Iters 0-82: Chambolle TV denoiser
- Iters 83-123: HSI-SDeCNN deep denoiser (3/4 schedule: 3 HSICNN, 1 TV)
- Reference: 25.12 dB mean on KAIST 10 scenes (InverseNet ECCV)

References:
- Maffei, A. et al. (2020). "A Single Model CNN for Hyperspectral Image
  Denoising", IEEE TGRS.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
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
_DEFAULT_WEIGHTS = [
    _PKG_ROOT / "weights" / "hsi_sdecnn" / "deep_denoiser.pth",
    _PKG_TOP / "weights" / "hsi_sdecnn" / "deep_denoiser.pth",
    _PKG_ROOT / "weights" / "hsi_sdecnn" / "hsi_sdecnn.pth",
    _PKG_TOP / "weights" / "hsi_sdecnn" / "hsi_sdecnn.pth",
]


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
    for candidate in _DEFAULT_WEIGHTS:
        if candidate.exists():
            return candidate
    return None


# ============================================================================
# Reference HSI-SDeCNN model (matches deep_denoiser.pth from PnP-CASSI)
# Architecture: in_nc=7, out_nc=1, nc=128, nb=15
# ============================================================================

if HAS_TORCH:

    def _pixel_unshuffle(input, upscale_factor):
        batch_size, channels, in_height, in_width = input.size()
        out_height = in_height // upscale_factor
        out_width = in_width // upscale_factor
        input_view = input.contiguous().view(
            batch_size, channels, out_height, upscale_factor,
            out_width, upscale_factor)
        channels *= upscale_factor ** 2
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(batch_size, channels, out_height, out_width)

    class _PixelUnShuffle(nn.Module):
        def __init__(self, upscale_factor):
            super().__init__()
            self.upscale_factor = upscale_factor

        def forward(self, input):
            return _pixel_unshuffle(input, self.upscale_factor)

    def _sequential(*args):
        if len(args) == 1:
            if isinstance(args[0], OrderedDict):
                raise NotImplementedError('sequential does not support OrderedDict input.')
            return args[0]
        modules = []
        for module in args:
            if isinstance(module, nn.Sequential):
                for submodule in module.children():
                    modules.append(submodule)
            elif isinstance(module, nn.Module):
                modules.append(module)
        return nn.Sequential(*modules)

    def _conv_rule(in_channels=128, out_channels=128, kernel_size=3,
                   stride=1, padding=1, bias=True, if_relu=True):
        L = []
        L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=bias))
        if if_relu:
            L.append(nn.ReLU(inplace=True))
        return _sequential(*L)

    class HSI_SDeCNN(nn.Module):
        """Reference HSI-SDeCNN architecture matching deep_denoiser.pth.

        Uses 7-channel spectral context window (3 before + current + 3 after).
        Architecture: PixelUnShuffle(2) -> 15 Conv layers (128 features) -> PixelShuffle(2)
        """

        def __init__(self, in_nc=7, out_nc=1, nc=128, nb=15):
            super().__init__()
            sf = 2
            self.m_down = _PixelUnShuffle(upscale_factor=sf)
            m_head = _conv_rule(in_nc * sf * sf + 1, nc)
            m_body = [_conv_rule(nc, nc) for _ in range(nb - 2)]
            m_tail = _conv_rule(nc, out_nc * sf * sf, if_relu=False)
            self.model = _sequential(m_head, *m_body, m_tail)
            self.m_up = nn.PixelShuffle(upscale_factor=sf)

        def forward(self, x, sigma):
            h, w = x.size()[-2:]
            paddingBottom = int(np.ceil(h / 2) * 2 - h)
            paddingRight = int(np.ceil(w / 2) * 2 - w)
            x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
            x = self.m_down(x)
            m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
            x = torch.cat((x, m), 1)
            x = self.model(x)
            x = self.m_up(x)
            x = x[..., :h, :w]
            return x


# ============================================================================
# Band-by-band denoising with 7-channel context window
# ============================================================================


def _apply_hsicnn_denoiser(
    x: np.ndarray,
    model: "HSI_SDeCNN",
    dev: "torch.device",
    nC: int = 28,
    sigma_val: float = 10.0,
) -> np.ndarray:
    """Apply HSI-SDeCNN band-by-band with 7-channel context window.

    For each band i, creates a 7-channel input:
    - Middle bands (3 <= i <= nC-4): x[:,:,i-3:i+4]
    - Edge bands: mirror-padded to maintain 7 channels

    Args:
        x: (H, W, nC) spectral cube.
        model: Loaded HSI_SDeCNN model (eval mode).
        dev: Torch device.
        nC: Number of spectral bands.
        sigma_val: Noise level (0-255 scale).

    Returns:
        Denoised cube (H, W, nC).
    """
    import torch as _torch

    result = np.zeros_like(x)
    sigma_t = _torch.full((1, 1, 1, 1), sigma_val / 255.0, device=dev)

    for i in range(nC):
        if i < 3:
            if i == 0:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 0],
                                    x[:, :, 0:4]))
            elif i == 1:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 0],
                                    x[:, :, 1:5]))
            else:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 1],
                                    x[:, :, 2:6]))
        elif i > nC - 4:
            if i == nC - 3:
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1],
                                    x[:, :, i + 2], x[:, :, i + 2]))
            elif i == nC - 2:
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1],
                                    x[:, :, i + 1], x[:, :, i + 1]))
            else:
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i],
                                    x[:, :, i], x[:, :, i]))
        else:
            net_in = x[:, :, i - 3:i + 4]

        net_t = (_torch.from_numpy(np.ascontiguousarray(net_in))
                 .permute(2, 0, 1).float().unsqueeze(0).to(dev))
        with _torch.no_grad():
            out = model(net_t, sigma_t)
        result[:, :, i] = out.squeeze().cpu().numpy()

    return result


def _is_hsicnn_iter(k: int) -> bool:
    """True for iterations 83+ with 3/4 HSICNN schedule.

    Pattern: 3 iterations of HSICNN then 1 iteration of TV, repeating.
    """
    if k < 83:
        return False
    return (k - 83) % 4 < 3


# ============================================================================
# Functional API
# ============================================================================


def hsi_sdecnn_denoise(
    hsi_cube: np.ndarray,
    sigma: float = 10.0,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Denoise a hyperspectral image cube using HSI-SDeCNN.

    Args:
        hsi_cube: HSI cube (H, W, nC). Must have nC >= 4.
        sigma: Noise level (0-255 scale, default 10).
        weights_path: Path to pretrained weights.
        device: Torch device string.

    Returns:
        Denoised HSI cube (H, W, nC).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if hsi_cube.ndim == 3 and hsi_cube.shape[0] < hsi_cube.shape[2]:
        hsi_cube = hsi_cube.transpose(1, 2, 0)

    nC = hsi_cube.shape[2]

    model = HSI_SDeCNN(in_nc=7, out_nc=1, nc=128, nb=15).to(dev)
    wp = _find_weights(weights_path)
    if wp is not None:
        state = torch.load(str(wp), map_location=dev, weights_only=False)
        model.load_state_dict(state, strict=True)
    else:
        warnings.warn("HSI-SDeCNN: no pretrained weights found", stacklevel=2)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    result = _apply_hsicnn_denoiser(hsi_cube, model, dev, nC=nC, sigma_val=sigma)
    return np.clip(result, 0, 1).astype(np.float32)


# ============================================================================
# GAP + HSI-SDeCNN PnP-CASSI reconstruction (InverseNet reference algorithm)
# ============================================================================


def gap_sdecnn_cassi(
    meas: np.ndarray,
    mask: np.ndarray,
    n_bands: int = 28,
    step: int = 2,
    weights_path: Optional[str] = None,
    iters: int = 124,
    tv_weight: float = 0.05,
    sigma_val: float = 10.0,
    device: Optional[str] = None,
) -> np.ndarray:
    """PnP-HSICNN: GAP reconstruction with hybrid TV + HSI-SDeCNN denoiser.

    124-iter GAP with Nesterov acceleration:
    - Iters 0-82: Chambolle TV denoiser (weight=tv_weight)
    - Iters 83-123: HSI-SDeCNN deep denoiser (3/4 schedule)
    Reference: 25.12 dB mean on KAIST 10 scenes (InverseNet ECCV)

    Args:
        meas: 2D CASSI measurement (H, W_ext).
        mask: 2D coded aperture (H, W).
        n_bands: Number of spectral bands.
        step: CASSI dispersion step.
        weights_path: Path to deep_denoiser.pth weights.
        iters: Total GAP iterations (default 124).
        tv_weight: TV denoising weight (default 0.05 = 12.75/255).
        sigma_val: Noise level for HSI-SDeCNN (0-255 scale, default 10).
        device: Torch device string.

    Returns:
        Reconstructed spectral cube (H, W, n_bands).
    """
    _require_torch()
    from skimage.restoration import denoise_tv_chambolle

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build and load denoiser
    model = HSI_SDeCNN(in_nc=7, out_nc=1, nc=128, nb=15).to(dev)
    wp = _find_weights(weights_path)
    if wp is not None:
        state = torch.load(str(wp), map_location=dev, weights_only=False)
        model.load_state_dict(state, strict=True)
    else:
        warnings.warn("HSI-SDeCNN: no pretrained weights found", stacklevel=2)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    h, w = mask.shape
    nC = n_bands
    w_ext = w + (nC - 1) * step

    meas = meas[:, :w_ext].astype(np.float32)
    mask = mask.astype(np.float32)

    # Precompute Phi^T Phi diagonal
    Phi_sum = np.zeros((h, w_ext), dtype=np.float32)
    for k in range(nC):
        Phi_sum[:, k * step:k * step + w] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1.0)

    # Initialize with adjoint (no normalization)
    x = np.zeros((h, w, nC), dtype=np.float32)
    for k in range(nC):
        x[:, :, k] = mask * meas[:, k * step:k * step + w]

    # Nesterov accumulated residual
    y1 = np.zeros((h, w_ext), dtype=np.float32)

    for k_iter in range(iters):
        # Forward model
        y_est = np.zeros((h, w_ext), dtype=np.float32)
        for k in range(nC):
            y_est[:, k * step:k * step + w] += mask * x[:, :, k]

        # Nesterov GAP update
        y1 += (meas - y_est)
        norm_r = (y1 - y_est) / Phi_sum
        for k in range(nC):
            x[:, :, k] += mask * norm_r[:, k * step:k * step + w]

        # Denoiser selection
        if _is_hsicnn_iter(k_iter):
            x = _apply_hsicnn_denoiser(
                np.clip(x, 0, None), model, dev, nC=nC, sigma_val=sigma_val)
        else:
            x = denoise_tv_chambolle(
                np.clip(x, 0, None), weight=tv_weight,
                max_num_iter=5, channel_axis=2).astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)


# ============================================================================
# Portfolio wrapper
# ============================================================================


def run_hsi_sdecnn(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for PnP-HSICNN (GAP + HSI-SDeCNN)."""
    iters = cfg.get("iters", 124)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)
    step = cfg.get("step", 2)

    info: Dict[str, Any] = {"solver": "pnp_hsicnn", "iters": iters}

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
