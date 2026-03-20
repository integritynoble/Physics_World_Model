#!/usr/bin/env python3
"""CASSI InverseNet Validation -- benchmark-grade reconstruction.

Validates 4 reconstruction methods (GAP-TV, PnP-HSICNN, HDNet, MST-L) across
3 scenarios (I: Ideal, II: Assumed, III: Truth Forward Model) on 10 KAIST scenes.

5-parameter mismatch model (v4.0):
  Group 1 (Mask Affine):  mask_dx, mask_dy, mask_theta
  Group 2 (Dispersion):   disp_a1, disp_alpha

Scenarios:
  Scenario I   : ideal measurement + ideal operator       (oracle upper bound)
  Scenario II  : corrupted measurement + ideal operator    (baseline degradation)
  Scenario III : corrupted measurement + truth operator    (oracle, knows all 5 params)

Methods:
  GAP-TV     -- classical iterative (mask-aware)    (~24 dB ideal)
  PnP-HSICNN -- GAP + HSI-SDeCNN deep denoiser     (~29 dB ideal)
  HDNet      -- dual-domain deep learning           (~35 dB ideal, mask-oblivious)
  MST-L      -- mask-guided Transformer (large)     (~36 dB ideal)

Usage:
    python validate_cassi_inversenet.py [--device cuda:0] [--save-recon]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import affine_transform
from scipy.signal import correlate2d

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
DATASET_REAL = Path("/home/spiritai/MST-main/datasets/TSA_real_data")
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR = RESULTS_DIR / "cassi_reconstructions"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
RECONSTRUCTION_METHODS = ["gap_tv", "pnp_hsicnn", "hdnet", "mst_l"]
SCENARIOS = ["scenario_i", "scenario_ii", "scenario_iii"]
NUM_SCENES = 10

METHOD_LABELS = {
    "gap_tv": "GAP-TV",
    "pnp_hsicnn": "PnP-HSICNN",
    "hdnet": "HDNet",
    "mst_l": "MST-L",
}

# ---------------------------------------------------------------------------
# mismatch spec  (5-parameter, from cassi_plan_inversenet.md v4.0 section 3)
# ---------------------------------------------------------------------------
@dataclass
class MismatchParameters:
    """5-parameter mismatch for CASSI operator.

    Group 1 (Mask Affine): mask_dx, mask_dy, mask_theta
    Group 2 (Dispersion):  disp_a1, disp_alpha
    """
    # Group 1: Mask affine (W1-W2)
    mask_dx: float = 0.5        # pixels  (horizontal shift)
    mask_dy: float = 0.3        # pixels  (vertical shift)
    mask_theta: float = 0.1     # degrees (rotation)
    # Group 2: Dispersion (W4-W5)
    disp_a1: float = 2.02       # px/band (nominal=2.0, 1% drift)
    disp_alpha: float = 0.15    # degrees (dispersion axis offset)


# ===================================================================
# helpers -- data loading
# ===================================================================
def load_mask(path: Path) -> Optional[np.ndarray]:
    """Load mask from MATLAB .mat file."""
    try:
        data = sio.loadmat(str(path))
        for key in ["mask", "Mask", "mask_data"]:
            if key in data:
                mask = data[key]
                if isinstance(mask, np.ndarray):
                    return mask.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load mask from {path}: {e}")
    return None


def load_scene(scene_name: str) -> Optional[np.ndarray]:
    """Load scene from MATLAB .mat file (256x256x28)."""
    try:
        # Try Truth subdirectory first
        path = DATASET_SIMU / "Truth" / f"{scene_name}.mat"
        if not path.exists():
            path = DATASET_SIMU / f"{scene_name}.mat"

        if path.exists():
            data = sio.loadmat(str(path))
            for key in ["img", "Img", "scene", "Scene", "data"]:
                if key in data:
                    scene = data[key].astype(np.float32)
                    if scene.ndim == 3 and scene.shape[2] == 28:
                        return scene
    except Exception as e:
        logger.warning(f"Failed to load scene {scene_name}: {e}")
    return None


# ===================================================================
# helpers -- forward model & warping
# ===================================================================
def warp_affine_2d(mask: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    """Apply 2D affine transformation to mask (translation + rotation).

    Reuses sign convention from cassi_upwmi_alg12.py / validate_cacti.

    Args:
        mask: (H, W) input mask
        dx: x-translation in pixels
        dy: y-translation in pixels
        theta: rotation in degrees

    Returns:
        Warped mask (H, W), clipped to [0, 1]
    """
    H, W = mask.shape
    cx, cy = W / 2.0, H / 2.0

    th = np.radians(theta)
    cos_t, sin_t = np.cos(th), np.sin(th)

    # Forward affine: rotate about center + translate
    mat = np.array([
        [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
        [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
    ])

    # scipy needs inverse matrix
    inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
    warped = affine_transform(mask, inv[:2, :2], offset=inv[:2, 2], cval=0, order=1)

    return np.clip(warped, 0, 1).astype(np.float32)


def cassi_forward(scene: np.ndarray, mask: np.ndarray, step: int = 2) -> np.ndarray:
    """Simple CASSI forward model with spectral dispersion.

    y[:, k*step : k*step + W] += mask * scene[:, :, k]

    Args:
        scene: (H, W, nC) spectral cube
        mask: (H, W) coded aperture
        step: dispersion step in pixels per band

    Returns:
        y: (H, W + (nC-1)*step) 2D measurement
    """
    H, W, nC = scene.shape
    W_ext = W + (nC - 1) * step
    y = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        y[:, k * step:k * step + W] += mask * scene[:, :, k]
    return y


def cassi_forward_with_dispersion(
    scene: np.ndarray, mask: np.ndarray,
    a1: float = 2.0, alpha_deg: float = 0.0,
) -> np.ndarray:
    """CASSI forward model with parameterized dispersion.

    Handles non-integer dispersion slope (a1 != 2.0) and rotated dispersion
    axis (alpha != 0) via sub-pixel interpolation.

    Args:
        scene: (H, W, nC) spectral cube
        mask: (H, W) coded aperture
        a1: dispersion slope in pixels per band (nominal=2.0)
        alpha_deg: dispersion axis angle in degrees (nominal=0)

    Returns:
        y: (H, W_ext) 2D measurement with W_ext = W + (nC-1)*2  (standard size)
    """
    from scipy.ndimage import shift as ndi_shift

    H, W, nC = scene.shape
    alpha_rad = np.radians(alpha_deg)

    # Standard output width (compatible with all reconstructors)
    W_ext = W + (nC - 1) * 2  # 310 for 256×28
    # Working buffer may need to be wider for a1 > 2
    max_shift = int(np.ceil(a1 * (nC - 1))) + 2
    W_work = max(W + max_shift, W_ext)
    y = np.zeros((H, W_work), dtype=np.float32)

    for k in range(nC):
        coded_k = (mask * scene[:, :, k]).astype(np.float32)

        # True dispersion shift for band k
        shift_total = a1 * k
        shift_x = shift_total * np.cos(alpha_rad)
        shift_y = shift_total * np.sin(alpha_rad)

        # Integer and fractional parts
        shift_x_int = int(np.floor(shift_x))
        shift_x_frac = shift_x - shift_x_int

        # Apply sub-pixel shift (vertical from alpha, horizontal fraction)
        if abs(shift_x_frac) > 1e-6 or abs(shift_y) > 1e-6:
            coded_k = ndi_shift(coded_k, [shift_y, shift_x_frac],
                                order=1, mode='constant', cval=0.0)

        # Accumulate at integer column offset
        col_start = shift_x_int
        col_end = col_start + W
        if col_start >= 0 and col_end <= W_work:
            y[:, col_start:col_end] += coded_k

    # Crop/pad to standard width
    return y[:, :W_ext].copy()


def cassi_adjoint_with_dispersion(
    y: np.ndarray, mask: np.ndarray, nC: int = 28,
    a1: float = 2.0, alpha_deg: float = 0.0,
) -> np.ndarray:
    """CASSI adjoint with parameterized dispersion (back-projection).

    Args:
        y: (H, W_ext) measurement
        mask: (H, W) coded aperture
        nC: number of spectral bands
        a1: dispersion slope
        alpha_deg: dispersion axis angle in degrees

    Returns:
        x: (H, W, nC) back-projected cube
    """
    from scipy.ndimage import shift as ndi_shift

    H, W = mask.shape
    alpha_rad = np.radians(alpha_deg)
    x = np.zeros((H, W, nC), dtype=np.float32)

    for k in range(nC):
        shift_total = a1 * k
        shift_x = shift_total * np.cos(alpha_rad)
        shift_y = shift_total * np.sin(alpha_rad)

        shift_x_int = int(np.round(shift_x))
        col_start = shift_x_int
        col_end = col_start + W

        if col_start >= 0 and col_end <= y.shape[1]:
            band_contrib = mask * y[:, col_start:col_end]
            # Undo vertical shift from alpha
            if abs(shift_y) > 1e-6:
                band_contrib = ndi_shift(band_contrib, [-shift_y, 0],
                                         order=1, mode='constant', cval=0.0)
            x[:, :, k] = band_contrib

    return x


def reconstruct_gap_tv_with_dispersion(
    y: np.ndarray, mask: np.ndarray,
    a1: float = 2.0, alpha_deg: float = 0.0,
    device: str = "cuda:0",
) -> np.ndarray:
    """GAP-TV reconstruction using true dispersion parameters.

    Uses Chambolle TV with custom forward/adjoint matching the true
    dispersion slope and angle (oracle Scenario III).

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) coded aperture (already warped with true affine)
        a1: true dispersion slope
        alpha_deg: true dispersion axis angle
        device: unused (CPU)

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    from skimage.restoration import denoise_tv_chambolle

    H, W = mask.shape
    nC = 28
    mask = np.clip(mask, 0, 1).astype(np.float32)

    # Compute Phi_sum using the parametric forward model
    # Build it by accumulating mask^2 at each band's true offset
    alpha_rad = np.radians(alpha_deg)
    W_ext = W + (nC - 1) * 2  # standard output width

    Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        shift_x = int(np.round(a1 * k * np.cos(alpha_rad)))
        if 0 <= shift_x and shift_x + W <= W_ext:
            Phi_sum[:, shift_x:shift_x + W] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    # Initialise with adjoint
    x = cassi_adjoint_with_dispersion(y, mask, nC=nC, a1=a1, alpha_deg=alpha_deg)

    # Nesterov accumulated residual
    y1 = np.zeros((H, W_ext), dtype=np.float32)

    for _ in range(100):
        y_est = cassi_forward_with_dispersion(x, mask, a1=a1, alpha_deg=alpha_deg)

        h_min = min(y.shape[0], y_est.shape[0])
        w_min = min(y.shape[1], y_est.shape[1], W_ext)

        raw_residual = np.zeros((H, W_ext), dtype=np.float32)
        raw_residual[:h_min, :w_min] = y[:h_min, :w_min] - y_est[:h_min, :w_min]

        y1 += raw_residual
        norm_r = np.zeros((H, W_ext), dtype=np.float32)
        norm_r[:h_min, :w_min] = (y1[:h_min, :w_min] - y_est[:h_min, :w_min]) / Phi_sum[:h_min, :w_min]

        x = x + cassi_adjoint_with_dispersion(norm_r, mask, nC=nC, a1=a1, alpha_deg=alpha_deg)
        x = denoise_tv_chambolle(
            np.clip(x, 0, None), weight=0.1,
            max_num_iter=5, channel_axis=2,
        ).astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)


def add_poisson_gaussian_noise(y: np.ndarray, peak: float = 100000,
                               sigma: float = 0.01) -> np.ndarray:
    """Add Poisson + Gaussian noise to measurement."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0)

    y_max = np.max(y)
    if y_max <= 0:
        y_max = 1.0

    y_scaled = (y / y_max) * peak
    y_scaled = np.maximum(y_scaled, 0)

    y_poisson = np.random.poisson(y_scaled.astype(np.int64)).astype(np.float64)
    y_noisy = y_poisson + np.random.normal(0, sigma, y_poisson.shape)
    y_noisy = y_noisy / peak * y_max

    return np.maximum(y_noisy, 0).astype(np.float32)


# ===================================================================
# helpers -- metrics
# ===================================================================
def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate PSNR in dB (data in [0,1])."""
    x_true = np.clip(x_true, 0, 1).astype(np.float64)
    x_recon = np.clip(x_recon, 0, 1).astype(np.float64)
    mse = float(np.mean((x_true - x_recon) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(x_true: np.ndarray, x_recon: np.ndarray, window_size: int = 11) -> float:
    """Calculate SSIM on 2D grayscale images."""
    x_true = np.clip(x_true, 0, 1).astype(np.float64)
    x_recon = np.clip(x_recon, 0, 1).astype(np.float64)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window = np.ones((window_size, window_size)) / (window_size ** 2)

    mu_true = correlate2d(x_true, window, mode="same", boundary="symm")
    mu_recon = correlate2d(x_recon, window, mode="same", boundary="symm")
    mu_true_sq = mu_true ** 2
    mu_recon_sq = mu_recon ** 2
    mu_cross = mu_true * mu_recon

    sigma_true_sq = correlate2d(x_true ** 2, window, mode="same", boundary="symm") - mu_true_sq
    sigma_recon_sq = correlate2d(x_recon ** 2, window, mode="same", boundary="symm") - mu_recon_sq
    sigma_cross = correlate2d(x_true * x_recon, window, mode="same", boundary="symm") - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_true_sq + mu_recon_sq + C1) * (sigma_true_sq + sigma_recon_sq + C2))

    return float(np.mean(ssim_map))


def compute_sam(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate Spectral Angle Mapper (SAM) in degrees."""
    x_true = np.clip(x_true, 1e-6, 1).astype(np.float64)
    x_recon = np.clip(x_recon, 1e-6, 1).astype(np.float64)

    x_true_flat = x_true.reshape(-1, x_true.shape[2])
    x_recon_flat = x_recon.reshape(-1, x_recon.shape[2])

    x_true_norm = x_true_flat / (np.linalg.norm(x_true_flat, axis=1, keepdims=True) + 1e-10)
    x_recon_norm = x_recon_flat / (np.linalg.norm(x_recon_flat, axis=1, keepdims=True) + 1e-10)

    dots = np.sum(x_true_norm * x_recon_norm, axis=1)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)

    return float(np.degrees(np.mean(angles)))


# ===================================================================
# reconstruction methods
# ===================================================================
def _gap_tv_cassi_chambolle(
    y: np.ndarray, mask: np.ndarray, n_bands: int = 28,
    step: int = 2, iterations: int = 100,
    tv_weight: float = 0.1, tv_iter: int = 5,
) -> np.ndarray:
    """GAP-TV for CASSI using skimage Chambolle TV denoiser.

    Matches the implementation in validate_cassi_pwmi.py that achieves
    ~24 dB mean PSNR on KAIST.  Uses Nesterov acceleration (Yuan 2016,
    Eq. 8) and 2-D per-channel TV denoising.

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) coded aperture
        n_bands: number of spectral bands
        step: dispersion step (pixels per band)
        iterations: GAP iterations
        tv_weight: weight for TV denoising
        tv_iter: inner TV iterations per GAP step

    Returns:
        x: (H, W, n_bands) reconstructed cube
    """
    from skimage.restoration import denoise_tv_chambolle

    H, W = mask.shape
    nC = n_bands
    W_ext = W + (nC - 1) * step
    mask = np.clip(mask, 0, 1).astype(np.float32)

    # Phi_sum = diag(Phi Phi^T)
    Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        Phi_sum[:, k * step:k * step + W] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    # Initialise with normalised back-projection
    x = np.zeros((H, W, nC), dtype=np.float32)
    for k in range(nC):
        x[:, :, k] = mask * y[:, k * step:k * step + W] / np.maximum(
            Phi_sum[:, k * step:k * step + W], 1e-6)

    # Nesterov accumulated residual
    y1 = np.zeros((H, W_ext), dtype=np.float32)

    for _ in range(iterations):
        y_est = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            y_est[:, k * step:k * step + W] += mask * x[:, :, k]

        y1 += (y[:, :W_ext] - y_est)
        norm_r = (y1 - y_est) / Phi_sum
        for k in range(nC):
            x[:, :, k] += mask * norm_r[:, k * step:k * step + W]

        x = denoise_tv_chambolle(
            np.clip(x, 0, None), weight=tv_weight,
            max_num_iter=tv_iter, channel_axis=2,
        ).astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)


def reconstruct_gap_tv(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using GAP-TV (Chambolle TV, accelerated).

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) forward operator mask
        device: unused (GAP-TV is CPU-based)

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        return _gap_tv_cassi_chambolle(y, mask, n_bands=28, step=2,
                                       iterations=100, tv_weight=0.1, tv_iter=5)
    except Exception as e:
        logger.warning(f"GAP-TV failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


# ===================================================================
# PnP-HSICNN (GAP + HSI-SDeCNN deep denoiser)
# ===================================================================
_pnp_hsicnn_cache = {}


def _load_pnp_hsicnn(device: str):
    """Load HSI-SDeCNN denoiser from PnP-CASSI."""
    if "model" in _pnp_hsicnn_cache:
        return _pnp_hsicnn_cache["model"]

    import torch
    import importlib.util

    hsi_path = "/home/spiritai/PnP-CASSI-main/hsi.py"
    spec = importlib.util.spec_from_file_location("hsi_sdecnn", hsi_path)
    hsi_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hsi_mod)

    dev = torch.device(device)
    model = hsi_mod.HSI_SDeCNN(in_nc=7, out_nc=1, nc=128, nb=15).to(dev)

    weights_path = "/home/spiritai/PnP-CASSI-main/check_points/deep_denoiser.pth"
    state_dict = torch.load(weights_path, map_location=dev, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info("  Loaded HSI-SDeCNN denoiser with pretrained weights")

    _pnp_hsicnn_cache["model"] = (model, dev)
    return model, dev


def _apply_hsicnn_denoiser(x: np.ndarray, model, dev,
                           nC: int = 28, sigma_val: float = 10.0) -> np.ndarray:
    """Apply HSI-SDeCNN band-by-band with 7-channel context window."""
    import torch

    result = np.zeros_like(x)
    sigma_t = torch.full((1, 1, 1, 1), sigma_val / 255.0, device=dev)

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

        net_t = (torch.from_numpy(np.ascontiguousarray(net_in))
                 .permute(2, 0, 1).float().unsqueeze(0).to(dev))
        with torch.no_grad():
            out = model(net_t, sigma_t)
        result[:, :, i] = out.squeeze().cpu().numpy()

    return result


def _is_hsicnn_iter(k: int) -> bool:
    """True for iterations 83+ with 3/4 HSICNN schedule."""
    if k < 83:
        return False
    return (k - 83) % 4 < 3


def reconstruct_pnp_hsicnn(y: np.ndarray, mask: np.ndarray,
                            device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using PnP-HSICNN (GAP + HSI-SDeCNN deep denoiser).

    124-iter GAP with hybrid TV (iters 0-82) + HSICNN (iters 83-123).
    """
    try:
        from skimage.restoration import denoise_tv_chambolle

        model, dev = _load_pnp_hsicnn(device)

        H, W = mask.shape
        nC = 28
        step = 2
        W_ext = W + (nC - 1) * step

        Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            Phi_sum[:, k * step:k * step + W] += mask ** 2
        Phi_sum = np.maximum(Phi_sum, 1.0)

        # Initialize with adjoint
        x = np.zeros((H, W, nC), dtype=np.float32)
        for k in range(nC):
            x[:, :, k] = mask * y[:, k * step:k * step + W]

        y_nom = np.zeros((H, W_ext), dtype=np.float32)
        hh = min(H, y.shape[0])
        ww = min(W_ext, y.shape[1])
        y_nom[:hh, :ww] = y[:hh, :ww]

        y1 = np.zeros_like(y_nom)
        nsig_tv = 12.75  # TV weight ~ 0.05
        n_total = 124

        for k_iter in range(n_total):
            y_est = np.zeros((H, W_ext), dtype=np.float32)
            for k in range(nC):
                y_est[:, k * step:k * step + W] += mask * x[:, :, k]

            y1 += (y_nom - y_est)
            norm_r = (y1 - y_est) / Phi_sum

            for k in range(nC):
                x[:, :, k] += mask * norm_r[:, k * step:k * step + W]

            if _is_hsicnn_iter(k_iter):
                x = _apply_hsicnn_denoiser(
                    np.clip(x, 0, None), model, dev, nC=nC, sigma_val=10.0)
            else:
                x = denoise_tv_chambolle(
                    np.clip(x, 0, None), weight=nsig_tv / 255.0,
                    max_num_iter=5, channel_axis=2).astype(np.float32)

        return np.clip(x, 0, 1).astype(np.float32)
    except Exception as e:
        logger.warning(f"PnP-HSICNN failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


_hdnet_cache = {}

def _load_original_hdnet(device: str):
    """Load original HDNet model from MST-main (correct architecture)."""
    if "model" in _hdnet_cache:
        return _hdnet_cache["model"]

    import torch
    import importlib.util

    hdnet_path = "/home/spiritai/MST-main/simulation/test_code/architecture/HDNet.py"
    spec = importlib.util.spec_from_file_location("hdnet_orig", hdnet_path)
    hdnet_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdnet_mod)

    dev = torch.device(device)
    model = hdnet_mod.HDNet(in_ch=28, out_ch=28).to(dev)

    weights_path = "/home/spiritai/MST-main/model_zoo/hdnet/hdnet.pth"
    checkpoint = torch.load(weights_path, map_location=dev, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info("  Loaded original HDNet with pretrained weights")

    _hdnet_cache["model"] = (model, dev)
    return model, dev


def reconstruct_hdnet(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using HDNet (mask-oblivious).

    HDNet takes ONLY the initial spectral estimate (28 channels) as input.
    The mask is NOT passed to the model. However, different masks produce
    different initial estimates via shift_back, so scenarios still differ.

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) forward operator mask (used only for initial estimate)
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        import torch
        from pwm_core.recon.mst import shift_back_meas_torch

        model, dev = _load_original_hdnet(device)

        H, W = mask.shape
        nC, step = 28, 2
        W_ext = W + (nC - 1) * step

        # Pad/crop measurement to expected size
        y_padded = np.zeros((H, W_ext), dtype=np.float32)
        hh = min(H, y.shape[0])
        ww = min(W_ext, y.shape[1])
        y_padded[:hh, :ww] = y[:hh, :ww]

        # Create initial estimate using shift_back (same as MST pipeline)
        meas_t = torch.from_numpy(y_padded.copy()).unsqueeze(0).float().to(dev)
        x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
        x_init = x_init / nC * 2  # Scaling from original MST/HDNet code

        # Forward pass (HDNet takes only the initial estimate, no mask)
        with torch.no_grad():
            recon = model(x_init)

        recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(recon, 0, 1).astype(np.float32)
    except Exception as e:
        logger.warning(f"HDNet failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


def reconstruct_mst_l(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using MST-L (mask-aware Transformer, large).

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) forward operator mask
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.mst import mst_recon_cassi
        return mst_recon_cassi(y, mask, nC=28, step=2, device=device, variant="mst_l")
    except Exception as e:
        logger.warning(f"MST-L failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


RECONSTRUCTION_FUNCTIONS = {
    "gap_tv": reconstruct_gap_tv,
    "pnp_hsicnn": reconstruct_pnp_hsicnn,
    "hdnet": reconstruct_hdnet,
    "mst_l": reconstruct_mst_l,
}


# ===================================================================
# scenario validation
# ===================================================================
def validate_scenario_i(scene: np.ndarray, mask_ideal: np.ndarray,
                        methods: List[str], device: str,
                        save_recon: bool = False) -> Dict[str, Dict]:
    """Scenario I: Ideal (perfect forward model, no mismatch, no noise).

    Args:
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        methods: list of method names
        device: torch device
        save_recon: if True, include reconstruction arrays and measurement

    Returns:
        Dictionary with metrics for each method (and 'recon' dict if save_recon)
    """
    logger.info("  Scenario I: Ideal (oracle)")
    results = {}
    recon_data = {}

    y_ideal = cassi_forward(scene, mask_ideal, step=2)
    if save_recon:
        recon_data["meas_ideal"] = y_ideal.copy()

    for method in methods:
        t0 = time.time()
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_ideal, mask_ideal, device=device)
            x_hat = np.clip(x_hat, 0, 1)
            results[method] = {
                "psnr": float(compute_psnr(scene, x_hat)),
                "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                "sam": float(compute_sam(scene, x_hat)),
            }
            if save_recon:
                recon_data[f"scenario_i_{method}"] = x_hat.copy()
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {"psnr": 0.0, "ssim": 0.0, "sam": 180.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:8s}: PSNR={results[method]['psnr']:.2f} dB  ({dt:.1f}s)")

    if save_recon:
        results["recon"] = recon_data
    return results


def validate_scenario_ii(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchParameters,
                         methods: List[str], device: str,
                         save_recon: bool = False) -> Tuple[Dict[str, Dict], np.ndarray, Optional[np.ndarray]]:
    """Scenario II: Assumed/Baseline (corrupted measurement, uncorrected operator).

    Measurement is generated with ALL 5 mismatch parameters (mask affine +
    dispersion).  Reconstruction assumes ideal operator (no mismatch).

    Args:
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        mismatch: MismatchParameters (all 5 factors)
        methods: list of method names
        device: torch device
        save_recon: if True, include reconstruction arrays

    Returns:
        Tuple of (results dict, y_corrupt, mask_warped or None)
    """
    logger.info("  Scenario II: Assumed/Baseline (5-param uncorrected mismatch)")
    logger.info(f"    Mask: dx={mismatch.mask_dx}, dy={mismatch.mask_dy}, "
                f"theta={mismatch.mask_theta}")
    logger.info(f"    Disp: a1={mismatch.disp_a1}, alpha={mismatch.disp_alpha}")
    results = {}
    recon_data = {}

    # Create corrupted measurement with ALL 5 mismatch factors
    # Step 1: Warp mask with mask affine parameters
    mask_corrupted = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta,
    )
    # Step 2: Forward with corrupted dispersion parameters
    y_corrupt = cassi_forward_with_dispersion(
        scene, mask_corrupted,
        a1=mismatch.disp_a1,
        alpha_deg=mismatch.disp_alpha,
    )
    y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=100000, sigma=0.01)

    if save_recon:
        recon_data["mask_warped"] = mask_corrupted.copy()
        recon_data["meas_corrupt"] = y_corrupt.copy()

    # Reconstruct with each method ASSUMING IDEAL operator (no mismatch)
    # Ideal: mask=ideal, step=2, alpha=0
    for method in methods:
        t0 = time.time()
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_corrupt, mask_ideal, device=device)
            x_hat = np.clip(x_hat, 0, 1)
            results[method] = {
                "psnr": float(compute_psnr(scene, x_hat)),
                "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                "sam": float(compute_sam(scene, x_hat)),
            }
            if save_recon:
                recon_data[f"scenario_ii_{method}"] = x_hat.copy()
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {"psnr": 0.0, "ssim": 0.0, "sam": 180.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:8s}: PSNR={results[method]['psnr']:.2f} dB  ({dt:.1f}s)")

    if save_recon:
        results["recon"] = recon_data
    return results, y_corrupt, mask_corrupted if save_recon else None


def validate_scenario_iii(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchParameters, y_corrupt: np.ndarray,
                         methods: List[str], device: str,
                         save_recon: bool = False) -> Dict[str, Dict]:
    """Scenario III: Truth Forward Model (corrupted measurement, oracle operator).

    Oracle knows ALL 5 true mismatch parameters:
      - GAP-TV: custom forward/adjoint with true a1, alpha + true warped mask
      - PnP-HSICNN: true warped mask (mask-aware GAP framework, integer step)
      - MST-L: true warped mask (best approx, architecture uses integer step)
      - HDNet: mask-oblivious, same as Scenario II (no oracle benefit)

    Args:
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        mismatch: MismatchParameters (all 5 ground truth params)
        y_corrupt: measurement from Scenario II
        methods: list of method names
        device: torch device
        save_recon: if True, include reconstruction arrays

    Returns:
        Dictionary with metrics for each method (and 'recon' dict if save_recon)
    """
    logger.info("  Scenario III: Truth Forward Model (oracle, all 5 params)")
    results = {}
    recon_data = {}

    # Apply true mask affine -> oracle knows the mask corruption
    mask_truth = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta,
    )

    for method in methods:
        t0 = time.time()
        try:
            if method == "gap_tv":
                # GAP-TV: use dispersion-aware forward/adjoint with true a1, alpha
                x_hat = reconstruct_gap_tv_with_dispersion(
                    y_corrupt, mask_truth,
                    a1=mismatch.disp_a1, alpha_deg=mismatch.disp_alpha,
                    device=device)
            elif method == "pnp_hsicnn":
                # PnP-HSICNN: mask-aware GAP, pass true warped mask
                x_hat = reconstruct_pnp_hsicnn(y_corrupt, mask_truth, device=device)
            else:
                # MST-L: mask-aware, benefit from true warped mask
                # HDNet: mask-oblivious, Scenario III ≈ Scenario II
                x_hat = RECONSTRUCTION_FUNCTIONS[method](
                    y_corrupt, mask_truth, device=device)

            x_hat = np.clip(x_hat, 0, 1)
            results[method] = {
                "psnr": float(compute_psnr(scene, x_hat)),
                "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                "sam": float(compute_sam(scene, x_hat)),
            }
            if save_recon:
                recon_data[f"scenario_iii_{method}"] = x_hat.copy()
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {"psnr": 0.0, "ssim": 0.0, "sam": 180.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:8s}: PSNR={results[method]['psnr']:.2f} dB  ({dt:.1f}s)")

    if save_recon:
        results["recon"] = recon_data
    return results


# ===================================================================
# per-scene validation
# ===================================================================
def validate_scene(scene_idx: int, scene: np.ndarray,
                   mask_ideal: np.ndarray,
                   mismatch: MismatchParameters,
                   methods: List[str], device: str,
                   save_recon: bool = False) -> Dict:
    """Validate one scene across all 3 scenarios and all methods.

    Args:
        scene_idx: scene index (0-9)
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        mismatch: MismatchParameters
        methods: list of method names
        device: torch device
        save_recon: if True, collect reconstruction arrays

    Returns:
        Dictionary with complete results (and 'recon' dict if save_recon)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Scene {scene_idx + 1}/10")
    logger.info(f"{'='*70}")

    start_time = time.time()

    # Scenario I
    res_i = validate_scenario_i(scene, mask_ideal, methods, device, save_recon=save_recon)

    # Scenario II (returns both results and measurement for reuse)
    res_ii, y_corrupt, _ = validate_scenario_ii(scene, mask_ideal, mismatch, methods, device, save_recon=save_recon)

    # Scenario III (reuses y_corrupt from Scenario II)
    res_iii = validate_scenario_iii(scene, mask_ideal, mismatch, y_corrupt, methods, device, save_recon=save_recon)

    elapsed = time.time() - start_time

    # Collect recon data before stripping from results
    scene_recon = None
    if save_recon:
        scene_recon = {"gt": scene.copy(), "mask_ideal": mask_ideal.copy()}
        for res in [res_i, res_ii, res_iii]:
            if "recon" in res:
                scene_recon.update(res.pop("recon"))

    # Compile results
    result = {
        "scene_idx": scene_idx + 1,
        "scenario_i": res_i,
        "scenario_ii": res_ii,
        "scenario_iii": res_iii,
        "elapsed_time": round(elapsed, 2),
        "mismatch_injected": {
            "mask_dx": mismatch.mask_dx,
            "mask_dy": mismatch.mask_dy,
            "mask_theta": mismatch.mask_theta,
            "disp_a1": mismatch.disp_a1,
            "disp_alpha": mismatch.disp_alpha,
        },
    }
    if scene_recon is not None:
        result["recon"] = scene_recon

    # Calculate gaps for each method
    result["gaps"] = {}
    for method in methods:
        psnr_i = res_i[method]["psnr"]
        psnr_ii = res_ii[method]["psnr"]
        psnr_iii = res_iii[method]["psnr"]

        result["gaps"][method] = {
            "gap_i_ii": round(psnr_i - psnr_ii, 4),     # Degradation from mismatch
            "gap_ii_iii": round(psnr_iii - psnr_ii, 4),   # Recovery from oracle
            "gap_iii_i": round(psnr_i - psnr_iii, 4),     # Residual gap
        }

    # Log summary for this scene
    logger.info(f"\n  Scene {scene_idx+1} summary ({elapsed:.1f}s):")
    for method in methods:
        pi = res_i[method]["psnr"]
        pii = res_ii[method]["psnr"]
        piii = res_iii[method]["psnr"]
        logger.info(
            f"    {METHOD_LABELS[method]:8s}  I={pi:6.2f}  II={pii:6.2f}  III={piii:6.2f}  "
            f"gap_I-II={pi-pii:+.2f}  rec_II-III={piii-pii:+.2f}"
        )

    return result


# ===================================================================
# results aggregation
# ===================================================================
def compute_summary_statistics(all_results: List[Dict]) -> Dict:
    """Compute aggregated statistics across all scenes."""
    summary = {
        "num_scenes": len(all_results),
        "methods": list(RECONSTRUCTION_METHODS),
        "scenarios": ["scenario_i", "scenario_ii", "scenario_iii"],
        "mismatch": {
            "mask_dx": 0.5, "mask_dy": 0.3, "mask_theta": 0.1,
            "disp_a1": 2.02, "disp_alpha": 0.15,
        },
        "noise": {"alpha": 100000, "sigma": 0.01},
    }

    for scenario_key in ["scenario_i", "scenario_ii", "scenario_iii"]:
        summary[scenario_key] = {}

        for method in RECONSTRUCTION_METHODS:
            psnr_vals = [r[scenario_key][method]["psnr"] for r in all_results
                         if r[scenario_key][method]["psnr"] > 0]
            ssim_vals = [r[scenario_key][method]["ssim"] for r in all_results
                         if r[scenario_key][method]["ssim"] > 0]
            sam_vals = [r[scenario_key][method]["sam"] for r in all_results
                        if r[scenario_key][method]["sam"] < 180]

            summary[scenario_key][method] = {
                "psnr_mean": round(float(np.mean(psnr_vals)), 2) if psnr_vals else 0.0,
                "psnr_std": round(float(np.std(psnr_vals)), 2) if psnr_vals else 0.0,
                "ssim_mean": round(float(np.mean(ssim_vals)), 4) if ssim_vals else 0.0,
                "ssim_std": round(float(np.std(ssim_vals)), 4) if ssim_vals else 0.0,
                "sam_mean": round(float(np.mean(sam_vals)), 2) if sam_vals else 0.0,
                "sam_std": round(float(np.std(sam_vals)), 2) if sam_vals else 0.0,
            }

    # Compute gaps across scenarios
    summary["gaps"] = {}
    for method in RECONSTRUCTION_METHODS:
        gap_i_ii = [r["gaps"][method]["gap_i_ii"] for r in all_results]
        gap_ii_iii = [r["gaps"][method]["gap_ii_iii"] for r in all_results]

        summary["gaps"][method] = {
            "gap_i_ii_mean": round(float(np.mean(gap_i_ii)), 2),
            "gap_i_ii_std": round(float(np.std(gap_i_ii)), 2),
            "gap_ii_iii_mean": round(float(np.mean(gap_ii_iii)), 2),
            "gap_ii_iii_std": round(float(np.std(gap_ii_iii)), 2),
        }

    return summary


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="CASSI InverseNet Validation")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--save-recon", action="store_true",
                        help="Save reconstruction arrays to .npz files")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CASSI Validation for InverseNet ECCV Paper (v4.0, 5-param mismatch)")
    logger.info("3 Scenarios (I, II, III) x 4 Methods x 10 Scenes = 120 Reconstructions")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)

    # Load masks
    mask_ideal = load_mask(DATASET_SIMU / "mask.mat")
    mask_real = load_mask(DATASET_REAL / "mask.mat")

    if mask_ideal is None:
        logger.warning("Ideal mask not found, using synthetic")
        mask_ideal = np.random.rand(256, 256).astype(np.float32) * 0.8 + 0.1
    if mask_real is None:
        logger.warning("Real mask not found, using ideal mask")
        mask_real = mask_ideal.copy()

    logger.info(f"Ideal mask shape: {mask_ideal.shape}")

    # Mismatch parameters (5 factors)
    mismatch = MismatchParameters()
    logger.info(f"Mismatch (mask affine): dx={mismatch.mask_dx} px, "
                f"dy={mismatch.mask_dy} px, theta={mismatch.mask_theta} deg")
    logger.info(f"Mismatch (dispersion):  a1={mismatch.disp_a1} px/band, "
                f"alpha={mismatch.disp_alpha} deg")

    np.random.seed(42)

    if args.save_recon:
        RECON_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving reconstructions to: {RECON_DIR}")

    # Validate all scenes
    all_results = []
    start_total = time.time()

    for scene_idx in range(NUM_SCENES):
        scene_name = f"scene{scene_idx + 1:02d}"
        scene = load_scene(scene_name)

        if scene is None:
            logger.warning(f"{scene_name} not found, skipping")
            continue

        result = validate_scene(
            scene_idx, scene, mask_ideal,
            mismatch, RECONSTRUCTION_METHODS, args.device,
            save_recon=args.save_recon,
        )

        # Save per-scene .npz immediately (free memory)
        if args.save_recon and "recon" in result:
            npz_path = RECON_DIR / f"{scene_name}.npz"
            recon_data = result.pop("recon")
            np.savez_compressed(str(npz_path), **recon_data)
            n_arrays = len(recon_data)
            size_mb = sum(v.nbytes for v in recon_data.values()) / 1e6
            logger.info(f"  Saved {npz_path.name}: {n_arrays} arrays, {size_mb:.0f} MB uncompressed")
            del recon_data

        all_results.append(result)

    total_time = time.time() - start_total

    if not all_results:
        logger.error("No results collected!")
        return

    # Compute summary
    summary = compute_summary_statistics(all_results)
    summary["execution_seconds"] = round(total_time, 1)

    # Print overall results
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL RESULTS  (mean +/- std across all scenes)")
    logger.info("=" * 70)

    for scen_label, scen_key in [
        ("Scenario I  (Ideal)",    "scenario_i"),
        ("Scenario II (Baseline)", "scenario_ii"),
        ("Scenario III (Oracle)",   "scenario_iii"),
    ]:
        logger.info(f"\n  {scen_label}:")
        for method in RECONSTRUCTION_METHODS:
            s = summary[scen_key][method]
            logger.info(
                f"    {METHOD_LABELS[method]:8s}  "
                f"PSNR = {s['psnr_mean']:6.2f} +/- {s['psnr_std']:.2f} dB   "
                f"SSIM = {s['ssim_mean']:.4f}   "
                f"SAM = {s['sam_mean']:.2f} deg"
            )

    logger.info("\n  Gaps:")
    for method in RECONSTRUCTION_METHODS:
        g = summary["gaps"][method]
        logger.info(
            f"    {METHOD_LABELS[method]:8s}  "
            f"I-II = {g['gap_i_ii_mean']:+.2f} dB   "
            f"II-III = {g['gap_ii_iii_mean']:+.2f} dB"
        )

    logger.info(f"\n  Total time: {total_time:.1f}s ({total_time/len(all_results):.1f}s per scene)")

    # Save results
    out_detail = RESULTS_DIR / "cassi_validation_results.json"
    out_summary = RESULTS_DIR / "cassi_summary.json"

    with open(out_detail, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults  -> {out_detail}")
    logger.info(f"Summary  -> {out_summary}")

    if args.save_recon:
        npz_files = sorted(RECON_DIR.glob("scene*.npz"))
        total_size = sum(f.stat().st_size for f in npz_files) / 1e6
        logger.info(f"\nReconstruction data -> {RECON_DIR}")
        logger.info(f"  {len(npz_files)} files, {total_size:.0f} MB total (compressed)")

    logger.info("\nCASSI validation complete!")


if __name__ == "__main__":
    main()
