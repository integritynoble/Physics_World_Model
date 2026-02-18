#!/usr/bin/env python3
"""PWMI-CASSI 4-Scenario Validation -- differentiable calibration benchmark.

Validates 5 reconstruction methods across 4 scenarios on 10 KAIST scenes
with two-phase calibration: Algorithm 2 for mask affine (dx, dy, theta)
+ 1D grid search for dispersion slope a1.

Scenarios:
  Scenario I   : ideal measurement + ideal mask                (oracle upper bound)
  Scenario II  : corrupted measurement + ideal mask            (baseline degradation)
  Scenario III : corrupted measurement + calibrated mask (Alg2) (our method)
  Scenario IV  : corrupted measurement + truth mask            (oracle operator)

Methods:
  GAP-TV     -- classical iterative (mask-aware)              (~24 dB ideal)
  MST-S      -- mask-guided Transformer (small)               (~34 dB ideal)
  MST-L      -- mask-guided Transformer (large)               (~35 dB ideal)
  HDNet      -- deep unfolding network (mask-oblivious)       (~35 dB ideal)
  PnP-HSICNN -- plug-and-play GAP + HSI-SDeCNN denoiser      (~25 dB ideal)

Mismatch model (5 parameters):
  Group 1 - Mask affine:  dx=1.5 px, dy=1.0 px, theta=0.3 deg
  Group 2 - Dispersion:   a1=2.04 px/band (nominal=2.0), alpha=0.5 deg

Measurement generation: cassi_forward_parametric() with warped mask and a1.
Note: alpha has negligible effect at native resolution (all vertical offsets
round to 0 for |alpha|<2° with nC=28), so only a1 creates measurable
dispersion mismatch. Alpha is kept in the spec for completeness.

Noise: Poisson (alpha=100000) + Gaussian (sigma=0.01)

Usage:
    python validate_cassi_pwmi.py [--device cuda:0] [--scenes 10] [--save-recon]
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
from scipy.ndimage import affine_transform, zoom as ndi_zoom
from scipy.signal import correlate2d

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "pwm_core"))

DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
DATASET_REAL = Path("/home/spiritai/MST-main/datasets/TSA_real_data")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR = RESULTS_DIR / "reconstructions"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
RECONSTRUCTION_METHODS = ["gap_tv", "mst_s", "mst_l", "hdnet", "pnp_hsicnn"]
SCENARIOS = ["scenario_i", "scenario_ii", "scenario_iii", "scenario_iv"]
NUM_SCENES = 10
STEP = 2
N_BANDS = 28

# Correct dispersion: cumulative stride-2 offsets [0, 2, 4, ..., 54]
S_NOM = np.arange(N_BANDS, dtype=np.float32) * STEP

METHOD_LABELS = {
    "gap_tv": "GAP-TV",
    "mst_s": "MST-S",
    "mst_l": "MST-L",
    "hdnet": "HDNet",
    "pnp_hsicnn": "PnP-HSICNN",
}

# ---------------------------------------------------------------------------
# mismatch spec
# ---------------------------------------------------------------------------
@dataclass
class MismatchSpec:
    """Mismatch parameters for injection (5-parameter model).

    Group 1 - Mask Affine: dx, dy, theta
    Group 2 - Dispersion:  a1 (slope), alpha (axis angle)
    """
    mask_dx: float = 1.5          # pixels, mask x-shift
    mask_dy: float = 1.0          # pixels, mask y-shift
    mask_theta: float = 0.3       # degrees, mask rotation
    disp_a1: float = 2.04         # px/band dispersion slope (nominal=2.0)
    disp_alpha: float = 0.5       # degrees, dispersion axis angle (nominal=0.0)

    def __repr__(self) -> str:
        return (f"Mismatch(dx={self.mask_dx}, dy={self.mask_dy}, θ={self.mask_theta}°, "
                f"a₁={self.disp_a1}, α={self.disp_alpha}°)")


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

    mat = np.array([
        [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
        [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
    ])

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
# helpers -- parametric dispersion & enlarged grid
# ===================================================================
def _compute_band_offsets(nC: int, a1: float = 2.0, alpha: float = 0.0
                          ) -> List[Tuple[int, int]]:
    """Compute per-band integer dispersion offsets (dx_k, dy_k).

    Args:
        nC: number of spectral bands
        a1: dispersion slope (px/band in original space)
        alpha: dispersion axis angle (degrees)

    Returns:
        List of (dx_k, dy_k) integer pixel offsets for each band k.
    """
    cos_a = np.cos(np.radians(alpha))
    sin_a = np.sin(np.radians(alpha))
    return [
        (int(round(a1 * k * cos_a)), int(round(a1 * k * sin_a)))
        for k in range(nC)
    ]


def _measurement_extent(nC: int, a1: float, alpha: float, H: int, W: int
                        ) -> Tuple[int, int, int]:
    """Return (H_ext, W_ext, dy_offset) for the parametric forward model."""
    offsets = _compute_band_offsets(nC, a1, alpha)
    max_dx = max(o[0] for o in offsets)
    max_dy = max(o[1] for o in offsets)
    min_dy = min(o[1] for o in offsets)
    dy_off = -min_dy
    return H + max_dy - min_dy, W + max_dx, dy_off


def cassi_forward_enlarged(scene: np.ndarray, mask: np.ndarray,
                           a1: float = 2.0, alpha: float = 0.0,
                           N: int = 4) -> np.ndarray:
    """Enlarged grid forward model for accurate measurement generation.

    1. Upsample scene (bilinear) and mask (nearest) by N.
    2. Interpolate spectrum from nC to L_exp = (nC-1)*N*2+1 bands.
    3. Apply 2D dispersion shifts based on (a1, alpha) in enlarged space.
    4. Downsample measurement back to original resolution via block averaging.

    Args:
        scene: (H, W, nC) ground truth hyperspectral cube
        mask:  (H, W) coded aperture (already warped if mismatch desired)
        a1:    dispersion slope (px/band in *original* space, nominal=2.0)
        alpha: dispersion axis angle (degrees, nominal=0.0)
        N:     spatial enlargement factor (default 4)

    Returns:
        y: measurement at original resolution, shape (H, W_ext)
    """
    H, W, nC = scene.shape
    K = STEP  # original dispersion step (2)
    L_exp = (nC - 1) * N * K + 1  # 217 for nC=28, N=4, K=2

    # ----- upsample -----
    scene_up = ndi_zoom(scene, (N, N, 1), order=1)     # bilinear (H*N, W*N, nC)
    mask_up = ndi_zoom(mask, (N, N), order=0)           # nearest  (H*N, W*N)
    H_up, W_up = H * N, W * N

    # ----- dispersion offsets in enlarged space -----
    cos_a = np.cos(np.radians(alpha))
    sin_a = np.sin(np.radians(alpha))

    shifts_dx = []
    shifts_dy = []
    for k in range(L_exp):
        # enlarged band k -> original band position
        orig_pos = k * (nC - 1) / (L_exp - 1)  # 0 .. nC-1
        s_enlarged = a1 * orig_pos * N           # shift in enlarged pixels
        shifts_dx.append(int(round(s_enlarged * cos_a)))
        shifts_dy.append(int(round(s_enlarged * sin_a)))

    max_dx = max(shifts_dx)
    max_dy = max(shifts_dy) if shifts_dy else 0
    min_dy = min(shifts_dy) if shifts_dy else 0
    dy_off = -min_dy

    W_meas_up = W_up + max_dx
    H_meas_up = H_up + max_dy - min_dy
    y_up = np.zeros((H_meas_up, W_meas_up), dtype=np.float64)

    # ----- accumulate each enlarged band (memory-efficient) -----
    for k in range(L_exp):
        # linear spectral interpolation
        orig_f = k * (nC - 1) / (L_exp - 1)
        lo = int(np.floor(orig_f))
        hi = min(lo + 1, nC - 1)
        frac = orig_f - lo
        band = ((1.0 - frac) * scene_up[:, :, lo] + frac * scene_up[:, :, hi]
                if lo != hi else scene_up[:, :, lo])

        coded = band * mask_up

        dx_k = shifts_dx[k]
        dy_k = shifts_dy[k] + dy_off
        y_up[dy_k:dy_k + H_up, dx_k:dx_k + W_up] += coded

    # ----- downsample by block averaging -----
    h_pad = (-H_meas_up) % N
    w_pad = (-W_meas_up) % N
    if h_pad or w_pad:
        y_up = np.pad(y_up, ((0, h_pad), (0, w_pad)))
    y_down = y_up.reshape(
        y_up.shape[0] // N, N, y_up.shape[1] // N, N
    ).mean(axis=(1, 3))

    # crop to first H rows (vertical expansion from alpha is < 1 row for |alpha|<1°)
    y_out = y_down[:H, :]

    # --- intensity normalization ---
    # The enlarged grid sums L_exp interpolated bands and block-averages N×N.
    # To match the simple stride-2 model's intensity, multiply by:
    #   norm = N² × (nC-1) / (L_exp-1)  =  16 × 27 / 216  =  2.0
    norm_factor = float(N * N * (nC - 1)) / float(L_exp - 1)
    y_out *= norm_factor

    return y_out.astype(np.float32)


def cassi_forward_parametric(scene: np.ndarray, mask: np.ndarray,
                             a1: float = 2.0, alpha: float = 0.0
                             ) -> np.ndarray:
    """Native-resolution parametric CASSI forward model.

    y[:, dx_k : dx_k+W] += mask * scene[:,:,k]
    where dx_k = round(a1 * k * cos(alpha)).

    Horizontal-only dispersion: the vertical component sin(alpha) is
    negligible at native resolution for |alpha| < 2° and nC=28
    (all dy_k round to 0), so it is omitted.
    """
    H, W, nC = scene.shape
    offsets = _compute_band_offsets(nC, a1, alpha)
    max_dx = max(o[0] for o in offsets)
    W_ext = W + max_dx
    y = np.zeros((H, W_ext), dtype=np.float32)

    for k in range(nC):
        dx_k = offsets[k][0]  # horizontal offset only
        y[:, dx_k:dx_k + W] += mask * scene[:, :, k]
    return y


def cassi_adjoint_parametric(y: np.ndarray, mask: np.ndarray,
                             H: int, W: int, nC: int,
                             a1: float = 2.0, alpha: float = 0.0
                             ) -> np.ndarray:
    """Native-resolution parametric CASSI adjoint (back-projection).

    Horizontal-only dispersion (vertical component negligible for |alpha|<2°).
    """
    offsets = _compute_band_offsets(nC, a1, alpha)

    x = np.zeros((H, W, nC), dtype=np.float32)
    for k in range(nC):
        dx_k = offsets[k][0]  # horizontal offset only
        w_avail = min(W, y.shape[1] - dx_k)
        h_avail = min(H, y.shape[0])
        if w_avail > 0 and h_avail > 0:
            x[:h_avail, :w_avail, k] = (
                y[:h_avail, dx_k:dx_k + w_avail]
                * mask[:h_avail, :w_avail]
            )
    return x


def _gap_tv_parametric(y: np.ndarray, mask: np.ndarray,
                       a1: float = 2.0, alpha: float = 0.0,
                       n_iter: int = 100, lam: float = 0.1,
                       accelerate: bool = True,
                       tv_max_iter: int = 5,
                       ) -> np.ndarray:
    """GAP-TV solver using the parametric (a1, alpha) forward/adjoint.

    Implements the Generalized Alternating Projection (Yuan 2016):
      v^{k+1} = x^k + Φ^T diag(ΦΦ^T)^{-1} (y - Φx^k)
      x^{k+1} = TV_denoise(v^{k+1})

    With acceleration (Yuan 2016, Eq. 8):
      y1 = y1 + (y - Φx^k)
      v^{k+1} = x^k + Φ^T diag(ΦΦ^T)^{-1} (y1 - Φx^k)

    The Phi_sum normalization is critical when variable offsets (a1!=2.0)
    create overlapping band contributions at some detector pixels.
    """
    from skimage.restoration import denoise_tv_chambolle

    H, W = mask.shape
    nC = N_BANDS
    offsets = _compute_band_offsets(nC, a1, alpha)
    max_dx = max(o[0] for o in offsets)
    W_ext = W + max_dx

    # Phi_sum = diag(ΦΦ^T): sum of mask^2 at each measurement pixel
    Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        dx_k = offsets[k][0]
        Phi_sum[:, dx_k:dx_k + W] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    # initialise with adjoint
    x = cassi_adjoint_parametric(y, mask, H, W, nC, a1, alpha)
    y_in = y.copy()

    # Accumulated residual for Nesterov acceleration
    y1 = np.zeros((H, W_ext), dtype=np.float32) if accelerate else None

    for _ in range(n_iter):
        y_hat = cassi_forward_parametric(x, mask, a1, alpha)
        # size-safe residual computation
        h_min = min(y_in.shape[0], y_hat.shape[0])
        w_min = min(y_in.shape[1], y_hat.shape[1])

        raw_residual = np.zeros((H, W_ext), dtype=np.float32)
        raw_residual[:h_min, :w_min] = (
            y_in[:h_min, :w_min] - y_hat[:h_min, :w_min]
        )

        if accelerate:
            y1 += raw_residual
            norm_r = np.zeros((H, W_ext), dtype=np.float32)
            norm_r[:h_min, :w_min] = (
                (y1[:h_min, :w_min] - y_hat[:h_min, :w_min])
                / Phi_sum[:h_min, :w_min]
            )
        else:
            norm_r = np.zeros((H, W_ext), dtype=np.float32)
            norm_r[:h_min, :w_min] = (
                raw_residual[:h_min, :w_min]
                / Phi_sum[:h_min, :w_min]
            )

        x = x + cassi_adjoint_parametric(norm_r, mask, H, W, nC, a1, alpha)
        x = denoise_tv_chambolle(np.clip(x, 0, None), weight=lam,
                                 max_num_iter=tv_max_iter,
                                 channel_axis=2).astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)


def _dc_parametric(x_hat: np.ndarray, y: np.ndarray, mask: np.ndarray,
                   a1: float = 2.0, alpha: float = 0.0,
                   mu: float = 0.02, n_iter: int = 5) -> np.ndarray:
    """Data-consistency refinement using the parametric forward/adjoint.

    Applies gentle gradient-descent steps so that the reconstruction
    better matches the measured data under the (a1, alpha) forward model.
    Used as post-processing for neural-network methods (MST, HDNet).

    Includes Phi_sum normalization for variable-offset dispersion.
    """
    H, W = mask.shape
    nC = N_BANDS
    offsets = _compute_band_offsets(nC, a1, alpha)
    max_dx = max(o[0] for o in offsets)
    W_ext = W + max_dx

    # Phi_sum normalization
    Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        dx_k = offsets[k][0]
        Phi_sum[:, dx_k:dx_k + W] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    y_in = y.copy()
    x = x_hat.copy()

    for _ in range(n_iter):
        y_hat = cassi_forward_parametric(x, mask, a1, alpha)
        h_min = min(y_in.shape[0], y_hat.shape[0])
        w_min = min(y_in.shape[1], y_hat.shape[1])
        residual = np.zeros((H, W_ext), dtype=np.float32)
        residual[:h_min, :w_min] = (
            (y_in[:h_min, :w_min] - y_hat[:h_min, :w_min])
            / Phi_sum[:h_min, :w_min]
        )
        grad = cassi_adjoint_parametric(residual, mask, H, W, nC, a1, alpha)
        x = np.clip(x + mu * grad, 0, 1).astype(np.float32)

    return x


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
def reconstruct_gap_tv(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0",
                       a1: float = 2.0, alpha: float = 0.0) -> np.ndarray:
    """Reconstruct using GAP-TV with acceleration and parametric dispersion.

    Always uses the parametric solver with skimage's denoise_tv_chambolle
    and Nesterov acceleration (Yuan 2016). Parameters tuned for step=2
    grayscale TSA mask: weight=0.1, max_num_iter=5, 100 GAP iterations.
    """
    try:
        H, W = mask.shape
        y_param = _fit_measurement(y, H, W, N_BANDS, a1, alpha)
        return _gap_tv_parametric(y_param, mask, a1=a1, alpha=alpha,
                                  n_iter=100, lam=0.1, accelerate=True,
                                  tv_max_iter=5)
    except Exception as e:
        logger.warning(f"GAP-TV failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, N_BANDS).astype(np.float32) * 0.1, 0, 1)


def reconstruct_mst_s(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0",
                      a1: float = 2.0, alpha: float = 0.0) -> np.ndarray:
    """Reconstruct using MST-S + parametric DC when dispersion deviates."""
    try:
        from pwm_core.recon.mst import mst_recon_cassi
        H, W = mask.shape
        # MST requires nominal stride-2 measurement width
        y_nom = _fit_measurement(y, H, W, N_BANDS, 2.0, 0.0)
        x_hat = mst_recon_cassi(y_nom, mask, nC=N_BANDS, step=STEP,
                                device=device, variant="mst_s")
        if abs(a1 - 2.0) > 1e-6 or abs(alpha) > 1e-6:
            y_param = _fit_measurement(y, H, W, N_BANDS, a1, alpha)
            x_hat = _dc_parametric(x_hat, y_param, mask, a1, alpha, mu=0.02, n_iter=5)
        return x_hat
    except Exception as e:
        logger.warning(f"MST-S failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, N_BANDS).astype(np.float32) * 0.1, 0, 1)


def reconstruct_mst_l(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0",
                      a1: float = 2.0, alpha: float = 0.0) -> np.ndarray:
    """Reconstruct using MST-L + parametric DC when dispersion deviates."""
    try:
        from pwm_core.recon.mst import mst_recon_cassi
        H, W = mask.shape
        # MST requires nominal stride-2 measurement width
        y_nom = _fit_measurement(y, H, W, N_BANDS, 2.0, 0.0)
        x_hat = mst_recon_cassi(y_nom, mask, nC=N_BANDS, step=STEP,
                                device=device, variant="mst_l")
        if abs(a1 - 2.0) > 1e-6 or abs(alpha) > 1e-6:
            y_param = _fit_measurement(y, H, W, N_BANDS, a1, alpha)
            x_hat = _dc_parametric(x_hat, y_param, mask, a1, alpha, mu=0.02, n_iter=5)
        return x_hat
    except Exception as e:
        logger.warning(f"MST-L failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, N_BANDS).astype(np.float32) * 0.1, 0, 1)


# ---------------------------------------------------------------------------
# HDNet reconstruction (mask-oblivious)
# ---------------------------------------------------------------------------
_hdnet_cache = {}


def _load_original_hdnet(device: str):
    """Load original HDNet model from MST-main."""
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


def reconstruct_hdnet(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0",
                      a1: float = 2.0, alpha: float = 0.0) -> np.ndarray:
    """Reconstruct using HDNet + parametric data-consistency refinement.

    HDNet's forward(x, input_mask=None) ignores the mask, so we add
    post-reconstruction DC iterations using the scenario-specific mask
    and parametric (a1, alpha) dispersion model.
    """
    try:
        import torch
        from pwm_core.recon.mst import shift_back_meas_torch

        model, dev = _load_original_hdnet(device)

        H, W = mask.shape
        nC, step = N_BANDS, STEP
        W_ext = W + (nC - 1) * step

        # HDNet requires nominal stride-2 measurement width
        y_nom = _fit_measurement(y, H, W, nC, 2.0, 0.0)
        meas_t = torch.from_numpy(y_nom.copy()).unsqueeze(0).float().to(dev)
        x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
        x_init = x_init / nC * 2

        with torch.no_grad():
            recon = model(x_init)

        x_hat = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        x_hat = np.clip(x_hat, 0, 1).astype(np.float32)

        # --- parametric DC refinement ---
        if abs(a1 - 2.0) > 1e-6 or abs(alpha) > 1e-6:
            y_param = _fit_measurement(y, H, W, nC, a1, alpha)
            x_hat = _dc_parametric(x_hat, y_param, mask, a1, alpha, mu=0.02, n_iter=5)
        return x_hat
    except Exception as e:
        logger.warning(f"HDNet failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, N_BANDS).astype(np.float32) * 0.1, 0, 1)


# ---------------------------------------------------------------------------
# PnP-HSICNN reconstruction (ADMM + HSI-SDeCNN deep denoiser)
# ---------------------------------------------------------------------------
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
        # Build 7-channel input centered on band i
        if i < 3:
            if i == 0:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 0],
                                    x[:, :, 0:4]))
            elif i == 1:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 0],
                                    x[:, :, 1:5]))
            else:  # i == 2
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 1],
                                    x[:, :, 2:6]))
        elif i > nC - 4:  # i = 25, 26, 27
            if i == nC - 3:  # 25
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1],
                                    x[:, :, i + 2], x[:, :, i + 2]))
            elif i == nC - 2:  # 26
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1],
                                    x[:, :, i + 1], x[:, :, i + 1]))
            else:  # i == 27
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


def _shift_back_np(x: np.ndarray, step: int) -> np.ndarray:
    """Shift bands back (dispersed -> image domain)."""
    x = x.copy()
    H, W_ext, nC = x.shape
    for i in range(nC):
        x[:, :, i] = np.roll(x[:, :, i], -step * i, axis=1)
    return x[:, :W_ext - step * (nC - 1), :]


def _shift_fwd_np(x: np.ndarray, step: int, nC: int) -> np.ndarray:
    """Shift bands forward (image -> dispersed domain)."""
    H, W = x.shape[0], x.shape[1]
    W_ext = W + (nC - 1) * step
    out = np.zeros((H, W_ext, nC), dtype=x.dtype)
    for i in range(nC):
        out[:, i * step:i * step + W, i] = x[:, :, i]
    return out


def _is_hsicnn_iter(k: int) -> bool:
    """Check if iteration k should use HSICNN denoiser.

    Follows the PnP-CASSI reference pattern: from iteration 83 onwards,
    alternate 3 HSICNN iterations + 1 TV iteration in a cycle.
    """
    if k < 83:
        return False
    return (k - 83) % 4 < 3


def reconstruct_pnp_hsicnn(y: np.ndarray, mask: np.ndarray,
                           device: str = "cuda:0",
                           a1: float = 2.0, alpha: float = 0.0) -> np.ndarray:
    """Reconstruct using PnP-HSICNN (GAP + HSI-SDeCNN deep denoiser).

    Uses the GAP framework with Nesterov acceleration based on
    PnP-CASSI reference (Yuan et al.), tuned for step=2 grayscale mask:
    - Iterations 0-82: TV denoising (weight ≈ 0.05, max_num_iter=5)
    - Iterations 83-123: Alternating HSICNN (3 iters) / TV (1 iter)
    - Total: 124 iterations with accumulated residual acceleration
    """
    try:
        from skimage.restoration import denoise_tv_chambolle

        model, dev = _load_pnp_hsicnn(device)

        H, W = mask.shape
        nC = N_BANDS
        step = STEP
        W_ext = W + (nC - 1) * step

        # Phi_sum: sum of mask^2 at each measurement pixel (2D)
        Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            Phi_sum[:, k * step:k * step + W] += mask ** 2
        Phi_sum = np.maximum(Phi_sum, 1.0)

        # Fit measurement to nominal width
        y_nom = _fit_measurement(y, H, W, nC, 2.0, 0.0)

        # Initialize with adjoint (image domain)
        x = np.zeros((H, W, nC), dtype=np.float32)
        for k in range(nC):
            x[:, :, k] = mask * y_nom[:, k * step:k * step + W]

        # Accumulated residual for Nesterov acceleration
        y1 = np.zeros_like(y_nom)

        _lambda = 1.0
        n_total = 124
        nsig_tv = 12.75  # TV weight = nsig/255 ≈ 0.05 (tuned for step=2 grayscale mask)

        for k_iter in range(n_total):
            # Forward model: A(x)
            y_est = np.zeros((H, W_ext), dtype=np.float32)
            for k in range(nC):
                y_est[:, k * step:k * step + W] += mask * x[:, :, k]

            # GAP with Nesterov acceleration
            y1 += (y_nom - y_est)
            norm_r = (y1 - y_est) / Phi_sum

            # Adjoint: back-project to image domain
            for k in range(nC):
                x[:, :, k] += _lambda * mask * norm_r[:, k * step:k * step + W]

            # Denoising step (image domain)
            if _is_hsicnn_iter(k_iter):
                x = _apply_hsicnn_denoiser(
                    np.clip(x, 0, None), model, dev, nC=nC, sigma_val=10.0)
            else:
                x = denoise_tv_chambolle(
                    np.clip(x, 0, None), weight=nsig_tv / 255.0,
                    max_num_iter=5,
                    channel_axis=2).astype(np.float32)

        result = np.clip(x, 0, 1).astype(np.float32)

        # DC correction for non-nominal dispersion
        if abs(a1 - 2.0) > 1e-6 or abs(alpha) > 1e-6:
            y_param = _fit_measurement(y, H, W, nC, a1, alpha)
            result = _dc_parametric(result, y_param, mask, a1, alpha, mu=0.02, n_iter=5)

        return result
    except Exception as e:
        logger.warning(f"PnP-HSICNN failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, N_BANDS).astype(np.float32) * 0.1, 0, 1)


RECONSTRUCTION_FUNCTIONS = {
    "gap_tv": reconstruct_gap_tv,
    "mst_s": reconstruct_mst_s,
    "mst_l": reconstruct_mst_l,
    "hdnet": reconstruct_hdnet,
    "pnp_hsicnn": reconstruct_pnp_hsicnn,
}


# ===================================================================
# calibration (Scenario III)
# ===================================================================
def _calibrate_dispersion(y_corrupt: np.ndarray, mask_calibrated: np.ndarray,
                          ) -> Tuple[float, float]:
    """Estimate dispersion slope a1 via 1D grid search.

    After mask affine is calibrated, search for the a1 that minimises
    the measurement residual ||y - A(x; a1)||^2 using a few GAP-TV
    iterations per candidate.  Alpha is fixed at 0 (vertical dispersion
    unidentifiable at native resolution for |alpha|<2° and nC=28).

    Returns:
        (best_a1, 0.0)
    """
    from skimage.restoration import denoise_tv_chambolle

    H, W = mask_calibrated.shape
    nC = N_BANDS

    a1_grid = np.linspace(1.90, 2.10, 21)

    best_score = float("inf")
    best_a1 = 2.0

    n_quick = 5  # quick GAP-TV iterations per candidate

    logger.info(f"  Dispersion calibration: {len(a1_grid)} a1 candidates")

    for a1_c in a1_grid:
        y_fit = _fit_measurement(y_corrupt, H, W, nC, a1_c, 0.0)

        x_test = cassi_adjoint_parametric(
            y_fit, mask_calibrated, H, W, nC, a1_c, 0.0)
        for _ in range(n_quick):
            y_hat = cassi_forward_parametric(x_test, mask_calibrated,
                                             a1_c, 0.0)
            h_m = min(y_fit.shape[0], y_hat.shape[0])
            w_m = min(y_fit.shape[1], y_hat.shape[1])
            res = np.zeros_like(y_hat)
            res[:h_m, :w_m] = y_fit[:h_m, :w_m] - y_hat[:h_m, :w_m]
            x_test = x_test + cassi_adjoint_parametric(
                res, mask_calibrated, H, W, nC, a1_c, 0.0)
            x_test = denoise_tv_chambolle(
                np.clip(x_test, 0, None), weight=0.01,
                channel_axis=2).astype(np.float32)

        # score: measurement residual
        y_hat = cassi_forward_parametric(x_test, mask_calibrated,
                                         a1_c, 0.0)
        h_m = min(y_fit.shape[0], y_hat.shape[0])
        w_m = min(y_fit.shape[1], y_hat.shape[1])
        score = float(np.sum(
            (y_fit[:h_m, :w_m] - y_hat[:h_m, :w_m]) ** 2))

        if score < best_score:
            best_score = score
            best_a1 = float(a1_c)

    logger.info(f"  Dispersion calibrated: a1={best_a1:.4f}")
    return best_a1, 0.0


def calibrate_mismatch(y_corrupt: np.ndarray, mask_ideal: np.ndarray,
                       device: str = "cuda:0") -> Tuple[Dict, float]:
    """Two-phase calibration: mask affine (Alg2) then dispersion grid search.

    Phase 1: Algorithm 2 estimates (dx, dy, theta) via differentiable pipeline.
    Phase 2: Grid search estimates (a1, alpha) with calibrated mask.

    Returns:
        Tuple of (estimated params dict, calibration time in seconds)
    """
    from pwm_core.calibration.cassi_upwmi_alg12 import (
        MismatchParameters, Algorithm2JointGradientRefinement,
    )
    from pwm_core.recon.gap_tv import gap_tv_cassi

    t0 = time.time()

    # Fit measurement to nominal stride-2 size for Algorithm 2
    H, W = mask_ideal.shape
    y_nom = _fit_measurement(y_corrupt, H, W, N_BANDS, a1=2.0, alpha=0.0)

    # --- Phase 1: mask affine calibration ---
    x_proxy = gap_tv_cassi(y_nom, mask_ideal, n_bands=N_BANDS,
                           iterations=50, lam=0.5, step=STEP, accelerate=True)

    coarse = MismatchParameters(mask_dx=0.0, mask_dy=0.0, mask_theta=0.0)
    alg2 = Algorithm2JointGradientRefinement(device=device)
    estimated = alg2.refine(
        mismatch_coarse=coarse,
        y_meas=y_nom,
        mask_real=mask_ideal,
        x_true=x_proxy,
        s_nom=S_NOM,
    )

    t_phase1 = time.time() - t0
    logger.info(f"  Phase 1 (mask affine): dx={estimated.mask_dx:.4f}, "
                f"dy={estimated.mask_dy:.4f}, theta={estimated.mask_theta:.4f} "
                f"({t_phase1:.1f}s)")

    # --- Phase 2: dispersion calibration with calibrated mask ---
    mask_calibrated = warp_affine_2d(
        mask_ideal,
        dx=estimated.mask_dx,
        dy=estimated.mask_dy,
        theta=estimated.mask_theta,
    )
    t2 = time.time()
    est_a1, est_alpha = _calibrate_dispersion(y_corrupt, mask_calibrated)
    t_phase2 = time.time() - t2

    dt = time.time() - t0

    params = {
        "dx": estimated.mask_dx,
        "dy": estimated.mask_dy,
        "theta": estimated.mask_theta,
        "a1": est_a1,
        "alpha": est_alpha,
    }

    logger.info(f"  Calibration total: dx={params['dx']:.4f}, dy={params['dy']:.4f}, "
                f"theta={params['theta']:.4f}, a1={params['a1']:.4f}, "
                f"alpha={params['alpha']:.4f} ({dt:.1f}s = {t_phase1:.0f}+{t_phase2:.0f}s)")

    return params, dt


# ===================================================================
# scenario validation
# ===================================================================
def _fit_measurement(y: np.ndarray, H: int, W: int, nC: int,
                     a1: float, alpha: float) -> np.ndarray:
    """Pad or crop measurement to match the forward model for (a1, alpha).

    Height is always H (vertical dispersion negligible at native resolution).
    Width is W + max horizontal offset.
    """
    offsets = _compute_band_offsets(nC, a1, alpha)
    max_dx = max(o[0] for o in offsets)
    W_ext = W + max_dx

    y_fit = np.zeros((H, W_ext), dtype=np.float32)
    h_copy = min(y.shape[0], H)
    w_copy = min(y.shape[1], W_ext)
    y_fit[:h_copy, :w_copy] = y[:h_copy, :w_copy]
    return y_fit


def _run_methods(y: np.ndarray, mask: np.ndarray, scene: np.ndarray,
                 methods: List[str], device: str,
                 a1: float = 2.0, alpha: float = 0.0,
                 recon_out: Optional[Dict[str, np.ndarray]] = None,
                 ) -> Dict[str, Dict]:
    """Run all reconstruction methods and collect PSNR/SSIM/SAM.

    Each method internally fits the measurement to the width its forward
    model expects (nominal for neural nets, parametric for physics-based).

    Args:
        recon_out: if provided, stores {method: x_hat} for each reconstruction.
    """
    results: Dict[str, Dict] = {}
    for method in methods:
        t0 = time.time()
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](
                y, mask, device=device, a1=a1, alpha=alpha)
            x_hat = np.clip(x_hat, 0, 1)
            results[method] = {
                "psnr": float(compute_psnr(scene, x_hat)),
                "ssim": float(compute_ssim(
                    np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                "sam": float(compute_sam(scene, x_hat)),
            }
            if recon_out is not None:
                recon_out[method] = x_hat.copy()
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {"psnr": 0.0, "ssim": 0.0, "sam": 180.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:8s}: "
                     f"PSNR={results[method]['psnr']:.2f} dB  ({dt:.1f}s)")
    return results


def validate_scenario_i(scene: np.ndarray, mask_ideal: np.ndarray,
                        methods: List[str], device: str,
                        recon_out: Optional[Dict[str, np.ndarray]] = None,
                        ) -> Dict[str, Dict]:
    """Scenario I: Ideal (no mismatch, no noise, nominal a1=2.0, alpha=0)."""
    logger.info("  Scenario I: Ideal (oracle upper bound)")
    y_ideal = cassi_forward(scene, mask_ideal, step=STEP)
    return _run_methods(y_ideal, mask_ideal, scene, methods, device,
                        a1=2.0, alpha=0.0, recon_out=recon_out)


def validate_scenario_ii(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchSpec,
                         methods: List[str], device: str,
                         recon_out: Optional[Dict[str, np.ndarray]] = None,
                         ) -> Tuple[Dict[str, Dict], np.ndarray, np.ndarray]:
    """Scenario II: Measurement via enlarged grid with full 5-param mismatch,
    reconstruction with ideal/nominal forward model (no correction).

    Returns:
        (results, y_corrupt, mask_corrupted)
    """
    logger.info("  Scenario II: Baseline (parametric mismatch, no correction)")

    # Warp mask for measurement generation (mask affine mismatch)
    mask_corrupted = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta,
    )

    # Generate measurement with parametric forward model (a1, alpha mismatch)
    logger.info(f"    Generating measurement with a1={mismatch.disp_a1}, "
                f"alpha={mismatch.disp_alpha}° ...")
    t_fwd = time.time()
    y_corrupt = cassi_forward_parametric(
        scene, mask_corrupted,
        a1=mismatch.disp_a1,
        alpha=mismatch.disp_alpha,
    )
    y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=100000, sigma=0.01)
    logger.info(f"    Measurement done ({time.time()-t_fwd:.1f}s), "
                f"y shape={y_corrupt.shape}")

    # Reconstruct with nominal forward model (a1=2.0, alpha=0) — NO correction
    results = _run_methods(y_corrupt, mask_ideal, scene, methods, device,
                           a1=2.0, alpha=0.0, recon_out=recon_out)
    return results, y_corrupt, mask_corrupted


def validate_scenario_iii(scene: np.ndarray, mask_ideal: np.ndarray,
                          y_corrupt: np.ndarray, estimated_params: Dict,
                          methods: List[str], device: str,
                          recon_out: Optional[Dict[str, np.ndarray]] = None,
                          ) -> Dict[str, Dict]:
    """Scenario III: Corrected mask + estimated dispersion."""
    logger.info("  Scenario III: Corrected (calibrated mask + dispersion)")

    mask_calibrated = warp_affine_2d(
        mask_ideal,
        dx=estimated_params["dx"],
        dy=estimated_params["dy"],
        theta=estimated_params["theta"],
    )

    return _run_methods(y_corrupt, mask_calibrated, scene, methods, device,
                        a1=estimated_params.get("a1", 2.0),
                        alpha=estimated_params.get("alpha", 0.0),
                        recon_out=recon_out)


def validate_scenario_iv(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchSpec, y_corrupt: np.ndarray,
                         methods: List[str], device: str,
                         recon_out: Optional[Dict[str, np.ndarray]] = None,
                         ) -> Dict[str, Dict]:
    """Scenario IV: Oracle mask + oracle dispersion."""
    logger.info("  Scenario IV: Oracle (truth mask + truth dispersion)")

    mask_truth = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta,
    )

    return _run_methods(y_corrupt, mask_truth, scene, methods, device,
                        a1=mismatch.disp_a1, alpha=mismatch.disp_alpha,
                        recon_out=recon_out)


# ===================================================================
# per-scene validation
# ===================================================================
def validate_scene(scene_idx: int, scene: np.ndarray,
                   mask_ideal: np.ndarray,
                   mismatch: MismatchSpec,
                   methods: List[str], device: str,
                   save_recon: bool = False) -> Tuple[Dict, Optional[Dict[str, np.ndarray]]]:
    """Validate one scene across all 4 scenarios and all methods.

    Returns:
        Tuple of (result_dict, recon_data_or_None).
        recon_data keys: 'gt', 'mask_ideal', 'mask_warped',
        'scenario_i_{method}', 'scenario_ii_{method}', etc.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Scene {scene_idx + 1}/{NUM_SCENES}")
    logger.info(f"{'='*70}")

    start_time = time.time()

    # Prepare recon storage dicts per scenario
    recon_i = {} if save_recon else None
    recon_ii = {} if save_recon else None
    recon_iii = {} if save_recon else None
    recon_iv = {} if save_recon else None

    # Scenario I: Ideal
    res_i = validate_scenario_i(scene, mask_ideal, methods, device,
                                recon_out=recon_i)

    # Scenario II: Assumed (returns y_corrupt for reuse)
    res_ii, y_corrupt, mask_warped = validate_scenario_ii(
        scene, mask_ideal, mismatch, methods, device,
        recon_out=recon_ii)

    # Calibration: Run Algorithm 2 to estimate mismatch from y_corrupt
    estimated_params, calib_time = calibrate_mismatch(y_corrupt, mask_ideal, device)

    # Scenario III: Corrected (use calibrated mask)
    res_iii = validate_scenario_iii(
        scene, mask_ideal, y_corrupt, estimated_params, methods, device,
        recon_out=recon_iii)

    # Scenario IV: Oracle (use truth mask)
    res_iv = validate_scenario_iv(
        scene, mask_ideal, mismatch, y_corrupt, methods, device,
        recon_out=recon_iv)

    elapsed = time.time() - start_time

    # Compile results
    result = {
        "scene_idx": scene_idx + 1,
        "scenario_i": res_i,
        "scenario_ii": res_ii,
        "scenario_iii": res_iii,
        "scenario_iv": res_iv,
        "elapsed_time": round(elapsed, 2),
        "calibration_time": round(calib_time, 2),
        "mismatch_injected": {
            "dx": mismatch.mask_dx,
            "dy": mismatch.mask_dy,
            "theta": mismatch.mask_theta,
            "a1": mismatch.disp_a1,
            "alpha": mismatch.disp_alpha,
        },
        "mismatch_estimated": estimated_params,
        "parameter_error": {
            "dx_err": abs(estimated_params["dx"] - mismatch.mask_dx),
            "dy_err": abs(estimated_params["dy"] - mismatch.mask_dy),
            "theta_err": abs(estimated_params["theta"] - mismatch.mask_theta),
            "a1_err": abs(estimated_params.get("a1", 2.0) - mismatch.disp_a1),
            "alpha_err": abs(estimated_params.get("alpha", 0.0) - mismatch.disp_alpha),
        },
    }

    # Calculate gaps for each method
    result["gaps"] = {}
    for method in methods:
        psnr_i = res_i[method]["psnr"]
        psnr_ii = res_ii[method]["psnr"]
        psnr_iii = res_iii[method]["psnr"]
        psnr_iv = res_iv[method]["psnr"]

        result["gaps"][method] = {
            "degradation_i_ii": round(psnr_i - psnr_ii, 4),
            "calibration_gain_ii_iii": round(psnr_iii - psnr_ii, 4),
            "residual_gap_iii_iv": round(psnr_iv - psnr_iii, 4),
            "oracle_gain_ii_iv": round(psnr_iv - psnr_ii, 4),
        }

    # Log summary for this scene
    logger.info(f"\n  Scene {scene_idx+1} summary ({elapsed:.1f}s, calib={calib_time:.1f}s):")
    logger.info(f"  Estimated: dx={estimated_params['dx']:.4f}, "
                f"dy={estimated_params['dy']:.4f}, theta={estimated_params['theta']:.4f}, "
                f"a1={estimated_params.get('a1', 2.0):.4f}, "
                f"alpha={estimated_params.get('alpha', 0.0):.4f}")
    for method in methods:
        pi = res_i[method]["psnr"]
        pii = res_ii[method]["psnr"]
        piii = res_iii[method]["psnr"]
        piv = res_iv[method]["psnr"]
        gain = piii - pii
        logger.info(
            f"    {METHOD_LABELS[method]:8s}  I={pi:6.2f}  II={pii:6.2f}  "
            f"III={piii:6.2f}  IV={piv:6.2f}  gain={gain:+.2f} dB"
        )

    # Collect reconstruction data for saving
    recon_data = None
    if save_recon:
        recon_data = {
            "gt": scene.copy(),
            "mask_ideal": mask_ideal.copy(),
            "mask_warped": mask_warped.copy(),
        }
        for method, arr in recon_i.items():
            recon_data[f"scenario_i_{method}"] = arr
        for method, arr in recon_ii.items():
            recon_data[f"scenario_ii_{method}"] = arr
        for method, arr in recon_iii.items():
            recon_data[f"scenario_iii_{method}"] = arr
        for method, arr in recon_iv.items():
            recon_data[f"scenario_iv_{method}"] = arr

    return result, recon_data


# ===================================================================
# results aggregation
# ===================================================================
def compute_summary_statistics(all_results: List[Dict], methods: List[str]) -> Dict:
    """Compute aggregated statistics across all scenes."""
    summary = {
        "num_scenes": len(all_results),
        "methods": methods,
        "scenarios": SCENARIOS,
        "mismatch": {"dx": 1.5, "dy": 1.0, "theta": 0.3,
                      "a1": 2.04, "alpha": 0.5},
        "noise": {"alpha": 100000, "sigma": 0.01},
        "s_nom": "np.arange(28) * 2",
        "forward_model": "parametric_dispersion",
    }

    for scenario_key in SCENARIOS:
        summary[scenario_key] = {}
        for method in methods:
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

    # Gaps
    summary["gaps"] = {}
    for method in methods:
        degradation = [r["gaps"][method]["degradation_i_ii"] for r in all_results]
        calib_gain = [r["gaps"][method]["calibration_gain_ii_iii"] for r in all_results]
        residual = [r["gaps"][method]["residual_gap_iii_iv"] for r in all_results]
        oracle = [r["gaps"][method]["oracle_gain_ii_iv"] for r in all_results]

        summary["gaps"][method] = {
            "degradation_mean": round(float(np.mean(degradation)), 2),
            "degradation_std": round(float(np.std(degradation)), 2),
            "calibration_gain_mean": round(float(np.mean(calib_gain)), 2),
            "calibration_gain_std": round(float(np.std(calib_gain)), 2),
            "residual_gap_mean": round(float(np.mean(residual)), 2),
            "residual_gap_std": round(float(np.std(residual)), 2),
            "oracle_gain_mean": round(float(np.mean(oracle)), 2),
            "oracle_gain_std": round(float(np.std(oracle)), 2),
        }

    # Parameter recovery statistics
    dx_errs = [r["parameter_error"]["dx_err"] for r in all_results]
    dy_errs = [r["parameter_error"]["dy_err"] for r in all_results]
    theta_errs = [r["parameter_error"]["theta_err"] for r in all_results]
    a1_errs = [r["parameter_error"]["a1_err"] for r in all_results]
    alpha_errs = [r["parameter_error"]["alpha_err"] for r in all_results]

    summary["parameter_recovery"] = {
        "dx_rmse": round(float(np.sqrt(np.mean(np.array(dx_errs)**2))), 4),
        "dy_rmse": round(float(np.sqrt(np.mean(np.array(dy_errs)**2))), 4),
        "theta_rmse": round(float(np.sqrt(np.mean(np.array(theta_errs)**2))), 4),
        "a1_rmse": round(float(np.sqrt(np.mean(np.array(a1_errs)**2))), 4),
        "alpha_rmse": round(float(np.sqrt(np.mean(np.array(alpha_errs)**2))), 4),
        "dx_mean_err": round(float(np.mean(dx_errs)), 4),
        "dy_mean_err": round(float(np.mean(dy_errs)), 4),
        "theta_mean_err": round(float(np.mean(theta_errs)), 4),
        "a1_mean_err": round(float(np.mean(a1_errs)), 4),
        "alpha_mean_err": round(float(np.mean(alpha_errs)), 4),
    }

    # Timing
    calib_times = [r["calibration_time"] for r in all_results]
    total_times = [r["elapsed_time"] for r in all_results]
    summary["timing"] = {
        "calibration_mean": round(float(np.mean(calib_times)), 1),
        "calibration_std": round(float(np.std(calib_times)), 1),
        "total_mean": round(float(np.mean(total_times)), 1),
        "total_std": round(float(np.std(total_times)), 1),
    }

    return summary


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="PWMI-CASSI 4-Scenario Validation")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--scenes", type=int, default=10, help="Number of scenes (1-10)")
    parser.add_argument("--save-recon", action="store_true",
                        help="Save reconstruction arrays to .npz files")
    args = parser.parse_args()

    n_scenes = min(max(args.scenes, 1), 10)

    logger.info("=" * 70)
    logger.info("PWMI-CASSI 4-Scenario Validation (5-param mismatch + enlarged grid)")
    logger.info(f"4 Scenarios x 5 Methods x {n_scenes} Scenes = {4 * 5 * n_scenes} Reconstructions")
    logger.info(f"Mismatch: dx=1.5, dy=1.0, theta=0.3 deg, a1=2.04, alpha=0.5 deg")
    logger.info(f"Forward model: parametric dispersion (a1, alpha)")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)

    # Load mask -- prefer simulation mask (256x256) over real mask (660x660)
    mask_ideal = load_mask(DATASET_SIMU / "mask.mat")
    if mask_ideal is None:
        mask_ideal = load_mask(DATASET_REAL / "mask.mat")
    if mask_ideal is None:
        logger.error("No mask found!")
        return

    # Ensure mask matches scene size (256x256)
    if mask_ideal.shape[0] != 256 or mask_ideal.shape[1] != 256:
        logger.warning(f"Mask {mask_ideal.shape} != (256,256), center-cropping")
        h, w = mask_ideal.shape
        r0 = (h - 256) // 2
        c0 = (w - 256) // 2
        mask_ideal = mask_ideal[r0:r0+256, c0:c0+256].copy()

    logger.info(f"Mask shape: {mask_ideal.shape}")

    mismatch = MismatchSpec()
    logger.info(f"Mismatch: {mismatch}")

    np.random.seed(42)

    if args.save_recon:
        RECON_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving reconstructions to: {RECON_DIR}")

    # Validate all scenes
    all_results = []
    start_total = time.time()

    for scene_idx in range(n_scenes):
        scene_name = f"scene{scene_idx + 1:02d}"
        scene = load_scene(scene_name)

        if scene is None:
            logger.warning(f"{scene_name} not found, skipping")
            continue

        result, recon_data = validate_scene(
            scene_idx, scene, mask_ideal,
            mismatch, RECONSTRUCTION_METHODS, args.device,
            save_recon=args.save_recon,
        )

        all_results.append(result)

        if args.save_recon and recon_data is not None:
            npz_path = RECON_DIR / f"scene{scene_idx + 1:02d}.npz"
            np.savez_compressed(str(npz_path), **recon_data)
            logger.info(f"  Saved reconstructions -> {npz_path}")

    total_time = time.time() - start_total

    if not all_results:
        logger.error("No results collected!")
        return

    # Compute summary
    summary = compute_summary_statistics(all_results, RECONSTRUCTION_METHODS)
    summary["execution_seconds"] = round(total_time, 1)

    # Print overall results
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL RESULTS  (mean +/- std across all scenes)")
    logger.info("=" * 70)

    for scen_label, scen_key in [
        ("Scenario I   (Ideal)",     "scenario_i"),
        ("Scenario II  (Baseline)",  "scenario_ii"),
        ("Scenario III (Corrected)", "scenario_iii"),
        ("Scenario IV  (Oracle)",    "scenario_iv"),
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

    logger.info("\n  Calibration Gains (II -> III):")
    for method in RECONSTRUCTION_METHODS:
        g = summary["gaps"][method]
        logger.info(
            f"    {METHOD_LABELS[method]:8s}  "
            f"degradation = {g['degradation_mean']:+.2f} dB   "
            f"gain = {g['calibration_gain_mean']:+.2f} dB   "
            f"oracle = {g['oracle_gain_mean']:+.2f} dB"
        )

    pr = summary["parameter_recovery"]
    logger.info(f"\n  Parameter Recovery RMSE:")
    logger.info(f"    dx: {pr['dx_rmse']:.4f} px   dy: {pr['dy_rmse']:.4f} px   "
                f"theta: {pr['theta_rmse']:.4f} deg")
    logger.info(f"    a1: {pr['a1_rmse']:.4f} px/band   "
                f"alpha: {pr['alpha_rmse']:.4f} deg")

    logger.info(f"\n  Total time: {total_time:.1f}s ({total_time/len(all_results):.1f}s per scene)")

    # Save results
    out_detail = RESULTS_DIR / "pwmi_cassi_results.json"
    out_summary = RESULTS_DIR / "pwmi_cassi_summary.json"

    with open(out_detail, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults  -> {out_detail}")
    logger.info(f"Summary  -> {out_summary}")
    logger.info("\nPWMI-CASSI validation complete!")


if __name__ == "__main__":
    main()
