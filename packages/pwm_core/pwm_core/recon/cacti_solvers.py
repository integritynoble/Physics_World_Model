"""Coded Aperture Compressive Temporal Imaging (CACTI) Reconstruction Solvers.

Uses original pretrained model implementations for benchmark-grade results:

  GAP-TV          – skimage.denoise_tv_chambolle  (classical baseline, ~27 dB)
  PnP-FFDNet      – GAP + FFDNet deep denoiser    (~29 dB with pretrained)
  ELP-Unfolding   – original ECCV 2022 model       (~34.6 dB with pretrained)
  EfficientSCI    – original CVPR 2023 model       (~35.8 dB with pretrained)

References
----------
- Yuan, X. (2016). "Generalized alternating projection based total variation
  minimization for compressive sensing"
- Zhang, K. et al. (2018). "FFDNet: Toward a Fast and Flexible Solution for
  CNN-Based Image Denoising"
- Yang, C. et al. (2022). ELP-Unfolding, ECCV 2022
- Wang, L. et al. (2023). EfficientSCI, CVPR 2023

Benchmark: SCI Video Benchmark (256x256x8, 8:1 compression)
Expected PSNR (with pretrained weights):
- GAP-TV:        ~27 dB
- PnP-FFDNet:    ~29 dB
- ELP-Unfolding: ~34.6 dB
- EfficientSCI:  ~35.8 dB
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# External repository paths (for original pretrained models)
# ---------------------------------------------------------------------------
_ELP_REPO = "/home/spiritai/ELP-Unfolding-master"
_ELP_CKPT = "/home/spiritai/ELP-Unfolding-master/trained_dataset/ckptall.pth"
_ESCI_REPO = "/home/spiritai/EfficientSCI-main"
_ESCI_CKPT = "/home/spiritai/EfficientSCI-main/checkpoints/efficientsci_base.pth"
_FFDNET_PKG = "/home/spiritai/PnP-SCI_python-master/packages"
_FFDNET_WEIGHTS = "/home/spiritai/PnP-SCI_python-master/packages/ffdnet/models/net_gray.pth"

# Model cache (singleton — load once, reuse across calls)
_cached_models: Dict[str, Any] = {}


# ============================================================================
# Core GAP-denoise engine  (from benchmarks/run_all.py)
# ============================================================================

def _gap_denoise_core(
    y: np.ndarray,
    Phi: np.ndarray,
    max_iter: int = 100,
    lam: float = 1.0,
    accelerate: bool = True,
    tv_weight: float = 0.15,
    tv_iter: int = 5,
    x_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generalized Alternating Projection with TV denoising for video SCI.

    Parameters
    ----------
    y : (H, W) measurement.
    Phi : (H, W, T) coded aperture masks.
    max_iter : outer GAP iterations.
    lam : step-size multiplier (keep at 1.0).
    accelerate : use acceleration (recommended).
    tv_weight : per-frame TV Chambolle weight.
    tv_iter : inner TV iterations per frame.
    x_init : optional warm-start estimate (H, W, T).
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None

    h, w, nF = Phi.shape
    Phi_sum = np.sum(Phi, axis=2)
    Phi_sum[Phi_sum == 0] = 1

    if x_init is not None:
        x = x_init.copy().astype(np.float32)
    else:
        x = y[:, :, np.newaxis] * Phi / Phi_sum[:, :, np.newaxis]

    y1 = y.copy()

    for _ in range(max_iter):
        yb = np.sum(x * Phi, axis=2)

        if accelerate:
            y1 = y1 + (y - yb)
            residual = y1 - yb
        else:
            residual = y - yb

        x = x + lam * (residual / Phi_sum)[:, :, np.newaxis] * Phi

        if denoise_tv_chambolle is not None:
            for f in range(nF):
                x[:, :, f] = denoise_tv_chambolle(
                    x[:, :, f], weight=tv_weight, max_num_iter=tv_iter,
                )
        else:
            from scipy.ndimage import gaussian_filter
            for f in range(nF):
                x[:, :, f] = gaussian_filter(x[:, :, f], sigma=0.5)

        x = np.clip(x, 0, 1)

    return x.astype(np.float32)


# ============================================================================
# Method 1: GAP-TV  (classical baseline)
# ============================================================================

def gap_tv_cacti(
    y: np.ndarray,
    mask: np.ndarray,
    iterations: int = 100,
    tv_weight: float = 0.1,
    tv_iter: int = 5,
    verbose: bool = False,
    **_kw,
) -> np.ndarray:
    """GAP-TV: Generalized Alternating Projection with Total Variation.

    Expected PSNR on SCI benchmark: ~26-27 dB (Scenario I).
    """
    return _gap_denoise_core(
        y, mask,
        max_iter=iterations, lam=1.0, accelerate=True,
        tv_weight=tv_weight, tv_iter=tv_iter,
    )


# ============================================================================
# Method 2: PnP-FFDNet  (GAP + FFDNet deep denoiser)
# ============================================================================

def _load_ffdnet(device_str: str):
    """Load FFDNet grayscale denoiser from PnP-SCI repository."""
    cache_key = f"ffdnet_{device_str}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]

    if not HAS_TORCH or not os.path.isfile(_FFDNET_WEIGHTS):
        return None

    try:
        if _FFDNET_PKG not in sys.path:
            sys.path.insert(0, _FFDNET_PKG)
        from ffdnet.models import FFDNet

        dev = torch.device(device_str)
        net = FFDNet(num_input_channels=1)
        state_dict = torch.load(_FFDNET_WEIGHTS, map_location=dev, weights_only=False)
        # Strip DataParallel wrapper prefix if present
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(cleaned, strict=True)
        net = net.to(dev)
        net.eval()
        _cached_models[cache_key] = net
        logger.info("FFDNet loaded (%d params)", sum(p.numel() for p in net.parameters()))
        return net
    except Exception as e:
        logger.warning("FFDNet load failed: %s", e)
        return None


def _gap_ffdnet_core(
    y: np.ndarray,
    Phi: np.ndarray,
    ffdnet_model,
    device_str: str = "cpu",
    sigma_list=None,
    iter_list=None,
) -> np.ndarray:
    """GAP with FFDNet denoiser (replaces TV with learned denoiser)."""
    if sigma_list is None:
        sigma_list = [50 / 255, 25 / 255, 12 / 255]
    if iter_list is None:
        iter_list = [10, 10, 10]

    h, w, nF = Phi.shape
    Phi_sum = np.sum(Phi, axis=2)
    Phi_sum[Phi_sum == 0] = 1

    x = y[:, :, np.newaxis] * Phi / Phi_sum[:, :, np.newaxis]
    y1 = np.zeros_like(y)

    dev = torch.device(device_str)
    use_gpu = "cuda" in device_str

    for sigma, n_iter in zip(sigma_list, iter_list):
        for _ in range(n_iter):
            yb = np.sum(x * Phi, axis=2)
            y1 = y1 + (y - yb)
            x = x + ((y1 - yb) / Phi_sum)[:, :, np.newaxis] * Phi

            # FFDNet per-frame denoising
            for f in range(nF):
                frame_t = torch.from_numpy(x[:, :, f].copy()).unsqueeze(0).unsqueeze(0).float()
                sigma_t = torch.FloatTensor([sigma])
                if use_gpu:
                    frame_t = frame_t.to(dev)
                    sigma_t = sigma_t.to(dev)
                with torch.no_grad():
                    noise_est = ffdnet_model(frame_t, sigma_t)
                x[:, :, f] = (frame_t - noise_est).squeeze().cpu().numpy()

            x = np.clip(x, 0, 1)

    return x.astype(np.float32)


def pnp_ffdnet_cacti(
    y: np.ndarray,
    mask: np.ndarray,
    device: str = "cpu",
    iterations: int = 120,
    tv_weight: float = 0.15,
    tv_iter: int = 8,
    verbose: bool = False,
    **_kw,
) -> np.ndarray:
    """PnP-FFDNet: GAP + FFDNet deep denoiser.

    Falls back to GAP-TV with heavier regularisation if FFDNet unavailable.
    Expected PSNR: ~29 dB with pretrained FFDNet, ~27 dB fallback.
    """
    dev_str = _resolve_device(device)
    ffdnet = _load_ffdnet(dev_str)
    if ffdnet is not None:
        return _gap_ffdnet_core(y, mask, ffdnet, device_str=dev_str)
    # Fallback: stronger GAP-TV
    return _gap_denoise_core(
        y, mask, max_iter=iterations, lam=1.0, accelerate=True,
        tv_weight=tv_weight, tv_iter=tv_iter,
    )


# ============================================================================
# Method 3: ELP-Unfolding  (deep unfolded, ECCV 2022)
# ============================================================================

def _load_elp(device_str: str):
    """Load original ELP-Unfolding model with pretrained weights.

    The pretrained checkpoint uses init_channels=512 (565M params).
    """
    cache_key = f"elp_{device_str}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]

    if not HAS_TORCH or not os.path.isfile(_ELP_CKPT):
        return None

    try:
        if _ELP_REPO not in sys.path:
            sys.path.insert(0, _ELP_REPO)
        from SCI_Modelcollect import SCI_backwardcollect

        dev = torch.device(device_str)
        argdict = {
            "init_channels": 512,
            "pres_channels": 512,
            "init_input": 8,
            "pres_input": 8,
            "priors": 6,
            "iter__number": 8,
        }
        model = SCI_backwardcollect(argdict).to(dev)
        ckpt = torch.load(_ELP_CKPT, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["color_SCI_backward_dict"], strict=False)
        model.eval()
        _cached_models[cache_key] = model
        logger.info("ELP-Unfolding loaded (%d params)", sum(p.numel() for p in model.parameters()))
        return model
    except Exception as e:
        logger.warning("ELP-Unfolding load failed: %s", e)
        return None


def elp_unfolding_cacti(
    y: np.ndarray,
    mask: np.ndarray,
    device: str = "cpu",
    verbose: bool = False,
    **_kw,
) -> np.ndarray:
    """ELP-Unfolding: deep unfolded ADMM with ensemble priors (ECCV 2022).

    Uses the original pretrained model (init_channels=512, 565M params).
    Falls back to two-pass GAP-denoise if model unavailable.
    Expected PSNR: ~34.6 dB with pretrained, ~27 dB fallback.
    """
    dev_str = _resolve_device(device)
    model = _load_elp(dev_str)
    if model is None:
        x = _gap_denoise_core(y, mask, max_iter=100, tv_weight=0.12, tv_iter=5)
        x = _gap_denoise_core(y, mask, max_iter=80, tv_weight=0.15, tv_iter=8, x_init=x)
        return x

    nF = mask.shape[2]
    H, W = y.shape[:2]
    dev = torch.device(dev_str)

    # mask: (H,W,T) -> (1,T,H,W)
    mask_t = torch.from_numpy(mask.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(dev)
    # meas: (H,W) -> (1,1,H,W)
    meas_t = torch.from_numpy(y.copy()).unsqueeze(0).unsqueeze(0).float().to(dev)
    # initial estimate: ones (matching original training code)
    img_out_ori = torch.ones(1, nF, H, W, device=dev)

    with torch.no_grad():
        x_list, _ = model(mask_t, meas_t, img_out_ori)

    recon = x_list[-1].squeeze(0).clamp(0, 1).cpu().numpy()  # (T, H, W)
    return recon.transpose(1, 2, 0).astype(np.float32)  # -> (H, W, T)


# ============================================================================
# Method 4: EfficientSCI  (end-to-end learned, CVPR 2023)
# ============================================================================

def _load_efficientsci(device_str: str):
    """Load original EfficientSCI-base model with pretrained weights."""
    cache_key = f"esci_{device_str}"
    if cache_key in _cached_models:
        return _cached_models[cache_key]

    if not HAS_TORCH or not os.path.isfile(_ESCI_CKPT):
        return None

    try:
        if _ESCI_REPO not in sys.path:
            sys.path.insert(0, _ESCI_REPO)
        from cacti.models.efficientsci import EfficientSCI as OrigEfficientSCI

        dev = torch.device(device_str)
        model = OrigEfficientSCI(in_ch=64, units=8, group_num=4, color_ch=1).to(dev)
        ckpt = torch.load(_ESCI_CKPT, map_location=dev, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        _cached_models[cache_key] = model
        logger.info("EfficientSCI-base loaded (%d params)", sum(p.numel() for p in model.parameters()))
        return model
    except Exception as e:
        logger.warning("EfficientSCI load failed: %s", e)
        return None


def efficient_sci_cacti(
    y: np.ndarray,
    mask: np.ndarray,
    device: str = "cpu",
    variant: str = "base",
    verbose: bool = False,
    **_kw,
) -> np.ndarray:
    """EfficientSCI: two-stage ResDNet + CFormer transformer (CVPR 2023).

    Uses the original pretrained EfficientSCI-base model.
    Falls back to triple-pass GAP-denoise if model unavailable.
    Expected PSNR: ~35.8 dB with pretrained, ~27 dB fallback.
    """
    dev_str = _resolve_device(device)
    model = _load_efficientsci(dev_str)
    if model is None:
        x = _gap_denoise_core(y, mask, max_iter=100, tv_weight=0.10, tv_iter=5)
        x = _gap_denoise_core(y, mask, max_iter=80, tv_weight=0.15, tv_iter=8, x_init=x)
        x = _gap_denoise_core(y, mask, max_iter=60, tv_weight=0.18, tv_iter=10, x_init=x)
        return x

    nF = mask.shape[2]
    H, W = y.shape[:2]
    dev = torch.device(dev_str)

    # Phi: (1, T, H, W)
    Phi = torch.from_numpy(mask.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(dev)
    # Phi_s: (1, 1, H, W) — sum over temporal dim
    Phi_s = Phi.sum(dim=1, keepdim=True)
    Phi_s[Phi_s == 0] = 1
    # meas: (1, 1, H, W)
    meas_t = torch.from_numpy(y.copy()).unsqueeze(0).unsqueeze(0).float().to(dev)

    with torch.no_grad():
        outputs = model(meas_t, Phi, Phi_s)

    recon = outputs[-1].squeeze(0).clamp(0, 1).cpu().numpy()  # (T, H, W)
    return recon.transpose(1, 2, 0).astype(np.float32)  # -> (H, W, T)


# ============================================================================
# Helpers
# ============================================================================

def _resolve_device(device: str) -> str:
    """Resolve device string, preferring GPU when available."""
    if HAS_TORCH and torch.cuda.is_available():
        if device and device != "cpu":
            return device
        return "cuda:0"
    return "cpu"


# ============================================================================
# Public API
# ============================================================================

SOLVERS = {
    "gap_tv": gap_tv_cacti,
    "pnp_ffdnet": pnp_ffdnet_cacti,
    "elp_unfolding": elp_unfolding_cacti,
    "efficient_sci": efficient_sci_cacti,
}


def solve_cacti(
    y: np.ndarray,
    mask: np.ndarray,
    method: str = "gap_tv",
    **kwargs,
) -> np.ndarray:
    """Unified interface for CACTI reconstruction.

    Args:
        y: Measurement (H, W)
        mask: Temporal masks (H, W, T)
        method: Solver name
        **kwargs: Method-specific parameters

    Returns:
        x: Reconstructed video (H, W, T)
    """
    if method not in SOLVERS:
        raise ValueError(f"Unknown method: {method}. Available: {list(SOLVERS.keys())}")
    return SOLVERS[method](y, mask, **kwargs)
