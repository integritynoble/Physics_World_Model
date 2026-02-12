#!/usr/bin/env python3
"""CASSI Real-Data Benchmark — 4 Solvers on 10 Hyperspectral Scenes.

Runs GAP-TV, HDNet, MST-S, and MST-L on the TSA simulation benchmark
(scene01–scene10, 256×256×28, step=2 dispersion).

W1: 4-solver comparison on all 10 scenes
W2: 5-scenario realistic mismatch model (mask translation, rotation,
    dispersion slope, dispersion axis, PSF blur) on scene01 (GAP-TV)

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cassi_benchmark.py
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cassi_benchmark.py --scenes scene01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import scipy.io as sio

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

# External codebases (paths only — added to sys.path on demand to avoid conflicts)
PNPCASSI_ROOT = "/home/spiritai/PnP-CASSI-main"
MST_TEST_ROOT = "/home/spiritai/MST-main/simulation/test_code"

from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)

# ── Constants ────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "cassi"
DATASET_DIR = "/home/spiritai/MST-main/datasets/TSA_simu_data"
MASK_PATH = DATASET_DIR
TRUTH_PATH = os.path.join(DATASET_DIR, "Truth")
ALL_SCENES = [f"scene{i:02d}" for i in range(1, 11)]
PIXEL_MAX = 1.0       # truth is in [0, ~0.91]
NC = 28               # spectral bands
STEP = 2              # dispersion step
NOISE_SIGMA = 0.01    # for NLL computation
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
DET_H = 256                              # fixed detector rows
DET_W = 256 + (NC - 1) * STEP            # fixed detector cols = 310

# Model weight paths
HDNET_WEIGHTS = "/home/spiritai/MST-main/model_zoo/hdnet/hdnet.pth"
MST_S_WEIGHTS = "/home/spiritai/MST-main/model_zoo/mst/mst_s.pth"
MST_L_WEIGHTS = "/home/spiritai/MST-main/model_zoo/mst/mst_l.pth"


# ── Helpers ──────────────────────────────────────────────────────────────

def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def psnr_band(gt: np.ndarray, rec: np.ndarray, maxval: float = 1.0) -> float:
    mse = np.mean((gt.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(maxval ** 2 / mse))


def ssim_band(gt: np.ndarray, rec: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        gt.astype(np.float64), rec.astype(np.float64), data_range=1.0
    ))


def compute_metrics_allbands(orig: np.ndarray, recon: np.ndarray) -> dict:
    """Compute per-band and mean PSNR/SSIM for (H,W,L) arrays in [0,1]."""
    nbands = orig.shape[2]
    psnrs, ssims = [], []
    for b in range(nbands):
        psnrs.append(psnr_band(orig[:, :, b], recon[:, :, b]))
        ssims.append(ssim_band(orig[:, :, b], recon[:, :, b]))
    return {
        "mean_psnr": round(mean(psnrs), 2),
        "mean_ssim": round(mean(ssims), 4),
        "per_band_psnr": [round(p, 2) for p in psnrs],
        "per_band_ssim": [round(s, 4) for s in ssims],
    }


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def load_mask():
    """Load 2D mask (256,256) and pre-shifted 3D mask (256,310,28)."""
    mask2d = sio.loadmat(os.path.join(MASK_PATH, "mask.mat"))["mask"].astype(np.float32)
    mask3d_shift = sio.loadmat(
        os.path.join(MASK_PATH, "mask_3d_shift.mat")
    )["mask_3d_shift"].astype(np.float32)
    return mask2d, mask3d_shift


def load_truth_scene(scene: str) -> np.ndarray:
    """Load single scene truth (256,256,28) float32 in [0,~0.91]."""
    path = os.path.join(TRUTH_PATH, f"{scene}.mat")
    return sio.loadmat(path)["img"].astype(np.float32)


def make_mask3d_from_2d(mask2d: np.ndarray, step: int = STEP) -> np.ndarray:
    """Construct shifted 3D mask (H, W+(NC-1)*step, NC) from 2D mask."""
    H, W = mask2d.shape
    mask3d = np.tile(mask2d[:, :, np.newaxis], (1, 1, NC))
    return shift_np(mask3d, step=step)


def shift_np(inputs: np.ndarray, step: int = STEP) -> np.ndarray:
    """Shift each band along columns: band i is shifted by i*step pixels."""
    row, col, nC = inputs.shape
    output = np.zeros((row, col + (nC - 1) * step, nC), dtype=inputs.dtype)
    for i in range(nC):
        output[:, i * step:i * step + col, i] = inputs[:, :, i]
    return output


def shift_back_np(inputs: np.ndarray, step: int = STEP) -> np.ndarray:
    """Inverse shift: extract band i from column offset i*step."""
    row, col, nC = inputs.shape
    W = col - (nC - 1) * step
    output = np.zeros((row, W, nC), dtype=inputs.dtype)
    for i in range(nC):
        output[:, :, i] = inputs[:, i * step:i * step + W, i]
    return output


# ── Parametric forward model ────────────────────────────────────────────

def warp_mask(mask2d, dx, dy, theta):
    """Affine-transform 2D mask: translate (dx,dy) + rotate theta (radians).
    Uses scipy.ndimage.affine_transform with bilinear interpolation."""
    from scipy.ndimage import affine_transform
    H, W = mask2d.shape
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotation matrix (inverse for pull-based affine_transform)
    R_inv = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    # Center of rotation
    cy, cx = H / 2, W / 2
    # Offset: translate center, apply inverse rotation, translate back + shift
    offset = np.array([cy, cx]) - R_inv @ np.array([cy - dy, cx - dx])
    return affine_transform(mask2d, R_inv, offset=offset, order=1,
                            mode='constant', cval=0.0).astype(np.float32)


def compute_dispersion_shifts(nC, a1, a2=0.0, alpha=0.0):
    """Compute per-band (col, row) shifts from dispersion curve.
    delta(l) = a1*l + a2*l^2, direction = (cos alpha, sin alpha).
    Returns: col_shifts, row_shifts (arrays of nC floats)."""
    bands = np.arange(nC, dtype=np.float64)
    delta = a1 * bands + a2 * bands ** 2
    col_shifts = delta * np.cos(alpha)
    row_shifts = delta * np.sin(alpha)
    return col_shifts, row_shifts


def shift_np_parametric(inputs, col_shifts, row_shifts=None):
    """Shift each band by fractional (col, row) offsets within fixed detector.
    Integer col shifts with zero row shifts use fast direct placement;
    sub-pixel or row shifts use scipy.ndimage.shift for interpolation."""
    from scipy.ndimage import shift as ndi_shift
    H, W, nC = inputs.shape
    output = np.zeros((DET_H, DET_W, nC), dtype=inputs.dtype)
    for i in range(nC):
        cs = col_shifts[i]
        rs = row_shifts[i] if row_shifts is not None else 0.0
        if cs == int(cs) and rs == 0.0:
            c = int(cs)
            w_copy = min(W, DET_W - c)
            h_copy = min(H, DET_H)
            if c >= 0 and w_copy > 0:
                output[:h_copy, c:c + w_copy, i] = inputs[:h_copy, :w_copy, i]
        else:
            canvas = np.zeros((DET_H, DET_W), dtype=inputs.dtype)
            h_copy = min(H, DET_H)
            w_copy = min(W, DET_W)
            canvas[:h_copy, :w_copy] = inputs[:h_copy, :w_copy, i]
            output[:, :, i] = ndi_shift(canvas, [rs, cs], order=1, mode='constant')
    return output


def shift_back_np_parametric(inputs, col_shifts, row_shifts=None,
                             orig_H=256, orig_W=256):
    """Inverse of shift_np_parametric: extract each band from its offset."""
    from scipy.ndimage import shift as ndi_shift
    _, _, nC = inputs.shape
    output = np.zeros((orig_H, orig_W, nC), dtype=inputs.dtype)
    for i in range(nC):
        cs = col_shifts[i]
        rs = row_shifts[i] if row_shifts is not None else 0.0
        if cs == int(cs) and rs == 0.0:
            c = int(cs)
            w_copy = min(orig_W, DET_W - c)
            if c >= 0 and w_copy > 0:
                output[:, :w_copy, i] = inputs[:orig_H, c:c + w_copy, i]
        else:
            shifted = ndi_shift(inputs[:, :, i], [-rs, -cs], order=1,
                                mode='constant')
            output[:, :, i] = shifted[:orig_H, :orig_W]
    return output


def apply_psf(y, sigma):
    """Apply Gaussian PSF blur to 2D measurement."""
    if sigma <= 0:
        return y
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(y, sigma=sigma).astype(np.float32)


def apply_poisson_read_noise(y_clean, photon_gain=1000.0, read_sigma=5.0,
                             rng=None):
    """Poisson shot noise + Gaussian read noise.
    y_photon = Poisson(gain * y_clean) / gain
    y_noisy = y_photon + N(0, read_sigma/gain)"""
    if rng is None:
        rng = np.random.RandomState(SEED)
    y_photon = rng.poisson(
        photon_gain * np.clip(y_clean, 0, None)
    ) / photon_gain
    y_noisy = y_photon + rng.randn(*y_clean.shape) * (read_sigma / photon_gain)
    return y_noisy.astype(np.float32)


def cassi_forward(truth, mask2d, a1=2.0, a2=0.0, alpha=0.0,
                  mask_dx=0.0, mask_dy=0.0, mask_theta=0.0,
                  psf_sigma=0.0, photon_gain=0, read_sigma=0.0, rng=None):
    """Full parametric CASSI forward model.
    Returns y (2D measurement) and Phi_shifted (3D shifted mask)."""
    # 1. Warp mask
    if mask_dx != 0 or mask_dy != 0 or mask_theta != 0:
        mask_warped = warp_mask(mask2d, mask_dx, mask_dy, mask_theta)
    else:
        mask_warped = mask2d
    # 2. Mask the spectral cube
    mask3d = np.tile(mask_warped[:, :, np.newaxis], (1, 1, NC))
    masked = truth * mask3d
    # 3. Dispersion shifts
    col_shifts, row_shifts = compute_dispersion_shifts(NC, a1, a2, alpha)
    # 4. Shift + integrate
    shifted = shift_np_parametric(masked, col_shifts, row_shifts)
    y = np.sum(shifted, axis=2)
    # 5. PSF blur
    y = apply_psf(y, psf_sigma)
    # 6. Noise
    if photon_gain > 0:
        y = apply_poisson_read_noise(y, photon_gain, read_sigma, rng)
    # Shifted mask for reconstruction
    Phi_shifted = shift_np_parametric(mask3d, col_shifts, row_shifts)
    return y, Phi_shifted


def spectral_angle_mapper(x_true, x_hat):
    """SAM: mean spectral angle (degrees) between truth and recon."""
    dot = np.sum(x_true * x_hat, axis=2)
    norm_true = np.sqrt(np.sum(x_true ** 2, axis=2))
    norm_hat = np.sqrt(np.sum(x_hat ** 2, axis=2))
    cos_angle = dot / (norm_true * norm_hat + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    angles = np.arccos(cos_angle) * 180 / np.pi
    return float(np.mean(angles))


def measurement_residual(y_meas, y_pred):
    """||y_meas - y_pred||_2 / ||y_meas||_2 (relative residual)."""
    return float(np.linalg.norm(y_meas - y_pred) / (np.linalg.norm(y_meas) + 1e-10))


def wiener_deblur(y, sigma, noise_power=1e-3):
    """Frequency-domain Wiener deblurring for Gaussian PSF."""
    from numpy.fft import fft2, ifft2
    H, W = y.shape
    ax_r = np.arange(H) - H // 2
    ax_c = np.arange(W) - W // 2
    rr, cc = np.meshgrid(ax_r, ax_c, indexing='ij')
    psf = np.exp(-(rr ** 2 + cc ** 2) / (2 * sigma ** 2))
    psf /= psf.sum()
    psf = np.roll(np.roll(psf, -H // 2, axis=0), -W // 2, axis=1)
    PSF = fft2(psf)
    Y = fft2(y.astype(np.float64))
    Wiener = np.conj(PSF) / (np.abs(PSF) ** 2 + noise_power)
    return np.real(ifft2(Y * Wiener)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# sys.path management
# ══════════════════════════════════════════════════════════════════════════

def _enter_pnpcassi():
    """Set sys.path for PnP-CASSI utils (A, At, shift, TV_denoiser)."""
    if MST_TEST_ROOT in sys.path:
        sys.path.remove(MST_TEST_ROOT)
    if PNPCASSI_ROOT not in sys.path:
        sys.path.insert(0, PNPCASSI_ROOT)
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if hasattr(mod, "__file__") and mod.__file__ and "PnP-CASSI" not in (mod.__file__ or ""):
            del sys.modules["utils"]


def _enter_mst():
    """Set sys.path for MST test_code (model_generator, utils)."""
    if PNPCASSI_ROOT in sys.path:
        sys.path.remove(PNPCASSI_ROOT)
    if MST_TEST_ROOT not in sys.path:
        sys.path.insert(0, MST_TEST_ROOT)
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if hasattr(mod, "__file__") and mod.__file__ and "MST-main" not in (mod.__file__ or ""):
            del sys.modules["utils"]


# ══════════════════════════════════════════════════════════════════════════
# Solver 1: GAP-TV (self-contained, no PnP-CASSI import needed)
# ══════════════════════════════════════════════════════════════════════════

def _A_cassi(x, Phi):
    """Forward model: collapse shifted 3D cube via element-wise mask."""
    return np.sum(x * Phi, axis=2)


def _At_cassi(y, Phi):
    """Transpose: expand 2D measurement back to 3D."""
    return np.multiply(np.repeat(y[:, :, np.newaxis], Phi.shape[2], axis=2), Phi)


def _tv_denoiser(x, _lambda, n_iter_max):
    """Isotropic TV denoiser using dual variable (Chambolle 2004)."""
    dt = 0.25
    N = x.shape
    idx = np.arange(1, N[0] + 1); idx[-1] = N[0] - 1
    iux = np.arange(-1, N[0] - 1); iux[0] = 0
    ir = np.arange(1, N[1] + 1); ir[-1] = N[1] - 1
    il = np.arange(-1, N[1] - 1); il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)
    for _ in range(n_iter_max):
        z = divp - x * _lambda
        z1 = z[:, ir, :] - z
        z2 = z[idx, :, :] - z
        denom_2d = 1 + dt * np.sqrt(np.sum(z1 ** 2 + z2 ** 2, 2))
        denom_3d = np.tile(denom_2d[:, :, np.newaxis], (1, 1, N[2]))
        p1 = (p1 + dt * z1) / denom_3d
        p2 = (p2 + dt * z2) / denom_3d
        divp = p1 - p1[:, il, :] + p2 - p2[iux, :, :]
    return x - divp / _lambda


def gap_tv_cassi(y, Phi, step=STEP, iter_max=50, tv_weight=0.1,
                 tv_iter_max=5, _lambda=1, accelerate=True,
                 X_orig=None, show_iqa=True):
    """GAP-TV reconstruction for CASSI (step=2, numpy, CPU).

    Operates in the shifted domain.  Phi is the 3D shifted mask.
    Returns (x_unshifted, psnr_list, ssim_list).
    """
    x = _At_cassi(y, Phi)
    y1 = np.zeros_like(y)
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum == 0] = 1
    psnr_list, ssim_list = [], []

    for it in range(iter_max):
        yb = _A_cassi(x, Phi)
        if accelerate:
            y1 = y1 + (y - yb)
            x = x + _lambda * _At_cassi((y1 - yb) / Phi_sum, Phi)
        else:
            x = x + _lambda * _At_cassi((y - yb) / Phi_sum, Phi)

        # Shift back for TV denoising
        x_unshifted = shift_back_np(x, step=step)
        x_unshifted = _tv_denoiser(x_unshifted, tv_weight, n_iter_max=tv_iter_max)
        # Shift forward again
        x = shift_np(x_unshifted, step=step)

        if show_iqa and X_orig is not None:
            from skimage.metrics import structural_similarity
            p = psnr_band(X_orig, x_unshifted, maxval=1.0)
            s = float(np.mean([
                structural_similarity(
                    X_orig[:, :, b].astype(np.float64),
                    x_unshifted[:, :, b].astype(np.float64),
                    data_range=1.0
                ) for b in range(X_orig.shape[2])
            ]))
            psnr_list.append(p)
            ssim_list.append(s)
            if (it + 1) % 10 == 0:
                print(f"      GAP-TV iter {it+1:3d}: PSNR {p:.2f} dB, SSIM {s:.4f}")

    x_final = shift_back_np(x, step=step)
    return x_final, psnr_list, ssim_list


def gap_tv_cassi_parametric(y, Phi, col_shifts, row_shifts=None,
                            orig_H=256, orig_W=256,
                            iter_max=50, tv_weight=0.1, tv_iter_max=5,
                            _lambda=1, accelerate=True,
                            X_orig=None, show_iqa=False):
    """GAP-TV for parametric CASSI with potentially non-integer dispersion."""
    x = _At_cassi(y, Phi)
    y1 = np.zeros_like(y)
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum == 0] = 1

    for it in range(iter_max):
        yb = _A_cassi(x, Phi)
        if accelerate:
            y1 = y1 + (y - yb)
            x = x + _lambda * _At_cassi((y1 - yb) / Phi_sum, Phi)
        else:
            x = x + _lambda * _At_cassi((y - yb) / Phi_sum, Phi)

        x_unshifted = shift_back_np_parametric(
            x, col_shifts, row_shifts, orig_H=orig_H, orig_W=orig_W)
        x_unshifted = _tv_denoiser(x_unshifted, tv_weight,
                                   n_iter_max=tv_iter_max)
        x = shift_np_parametric(x_unshifted, col_shifts, row_shifts)

        if show_iqa and X_orig is not None:
            from skimage.metrics import structural_similarity
            p = psnr_band(X_orig, x_unshifted, maxval=1.0)
            s = float(np.mean([
                structural_similarity(
                    X_orig[:, :, b].astype(np.float64),
                    x_unshifted[:, :, b].astype(np.float64),
                    data_range=1.0
                ) for b in range(X_orig.shape[2])
            ]))

    x_final = shift_back_np_parametric(
        x, col_shifts, row_shifts, orig_H=orig_H, orig_W=orig_W)
    return x_final, [], []


def run_gap_tv_scene(truth, mask2d, mask3d_shift):
    """Run GAP-TV on one scene.  Returns (recon, wall_time)."""
    # Generate measurement in shifted domain
    truth_shift = shift_np(truth, step=STEP)  # (256, 310, 28)
    y = _A_cassi(truth_shift, mask3d_shift)   # (256, 310)

    t0 = time.time()
    x_hat, psnr_list, ssim_list = gap_tv_cassi(
        y, mask3d_shift, step=STEP, iter_max=50,
        tv_weight=0.1, tv_iter_max=5, _lambda=1, accelerate=True,
        X_orig=truth, show_iqa=True,
    )
    wall = time.time() - t0
    return np.clip(x_hat, 0, 1).astype(np.float32), wall


# ══════════════════════════════════════════════════════════════════════════
# Deep solver loading (MST codebase)
# ══════════════════════════════════════════════════════════════════════════

def load_deep_models():
    """Load HDNet, MST-S, MST-L from the MST codebase. Returns dict of models."""
    _enter_mst()
    import torch
    from architecture import model_generator

    models = {}

    # HDNet — model_generator returns (model, fdl_loss) for hdnet
    print("    Loading HDNet ...")
    hdnet_model, _ = model_generator("hdnet", HDNET_WEIGHTS)
    hdnet_model.eval()
    models["hdnet"] = hdnet_model

    # MST-S
    print("    Loading MST-S ...")
    mst_s = model_generator("mst_s", MST_S_WEIGHTS)
    mst_s.eval()
    models["mst_s"] = mst_s

    # MST-L
    print("    Loading MST-L ...")
    mst_l = model_generator("mst_l", MST_L_WEIGHTS)
    mst_l.eval()
    models["mst_l"] = mst_l

    return models


def run_deep_solvers(models, truths, mask2d, mask3d_shift):
    """Run all 3 deep solvers on 10 scenes (batch GPU inference).

    Args:
        models: dict with 'hdnet', 'mst_s', 'mst_l'
        truths: list of 10 truth arrays (256,256,28)
        mask2d: (256,256) float32
        mask3d_shift: (256,310,28) float32

    Returns:
        dict solver_name -> list of 10 recon arrays (256,256,28)
        dict solver_name -> wall_time
    """
    _enter_mst()
    import torch
    from utils import init_mask, init_meas, shift as mst_shift

    n_scenes = len(truths)

    # Prepare mask tensors
    mask3d_batch = torch.from_numpy(
        np.tile(mask2d[np.newaxis, :, :], (NC, 1, 1))  # (28, 256, 256)
    ).unsqueeze(0).expand(n_scenes, NC, 256, 256).cuda().float()

    # Shifted mask for MST models: (nscenes, 28, 256, 310)
    Phi_batch = torch.from_numpy(
        mask3d_shift.transpose(2, 0, 1)  # (28, 256, 310)
    ).unsqueeze(0).expand(n_scenes, NC, 256, 310).cuda().float()

    # Prepare truth tensor: (N, 28, 256, 256) — CHW format
    test_gt = np.stack(truths, axis=0)  # (10, 256, 256, 28)
    test_gt = torch.from_numpy(
        test_gt.transpose(0, 3, 1, 2)  # (10, 28, 256, 256)
    ).cuda().float()

    # Generate measurement H (unfolded): init_meas with Y2H=True
    # H = shift_back(sum(shift(mask3d * x, 2), dim=1) / nC * 2)
    # Using MST's gen_meas_torch via init_meas
    temp = mst_shift(mask3d_batch * test_gt, STEP)           # (10, 28, 256, 310)
    meas = torch.sum(temp, 1)                                 # (10, 256, 310)
    H = meas / NC * 2
    # shift_back: [bs, 256, 310] -> [bs, 28, 256, 256]
    from utils import shift_back as mst_shift_back
    H = mst_shift_back(H, step=STEP)                          # (10, 28, 256, 256)

    results = {}
    times = {}

    # HDNet: input = H only (no mask)
    print("  [2/4] HDNet (batch GPU) ...")
    hdnet = models["hdnet"]
    hdnet.eval()
    t0 = time.time()
    with torch.no_grad():
        out_hdnet = hdnet(H)
    torch.cuda.synchronize()
    times["hdnet"] = time.time() - t0
    results["hdnet"] = out_hdnet.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    print(f"    HDNet done in {times['hdnet']:.1f}s")

    # MST-S: input = H + Phi (shifted mask)
    print("  [3/4] MST-S (batch GPU) ...")
    mst_s = models["mst_s"]
    mst_s.eval()
    t0 = time.time()
    with torch.no_grad():
        out_mst_s = mst_s(H, Phi_batch)
    torch.cuda.synchronize()
    times["mst_s"] = time.time() - t0
    results["mst_s"] = out_mst_s.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    print(f"    MST-S done in {times['mst_s']:.1f}s")

    # MST-L: input = H + Phi (shifted mask)
    print("  [4/4] MST-L (batch GPU) ...")
    mst_l = models["mst_l"]
    mst_l.eval()
    t0 = time.time()
    with torch.no_grad():
        out_mst_l = mst_l(H, Phi_batch)
    torch.cuda.synchronize()
    times["mst_l"] = time.time() - t0
    results["mst_l"] = out_mst_l.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    print(f"    MST-L done in {times['mst_l']:.1f}s")

    return results, times


# ══════════════════════════════════════════════════════════════════════════
# W2: Realistic mismatch model — 5 scenarios (scene01, GAP-TV)
# ══════════════════════════════════════════════════════════════════════════

def _make_grid(base_kw, param_name, values):
    """Generate search grid: list of dicts, varying one param."""
    grid = []
    for v in values:
        trial = dict(base_kw)
        trial[param_name] = v
        grid.append(trial)
    return grid


def _make_grid_2d(base_kw, p1_name, p1_vals, p2_name, p2_vals):
    """Generate 2D search grid."""
    grid = []
    for v1 in p1_vals:
        for v2 in p2_vals:
            trial = dict(base_kw)
            trial[p1_name] = v1
            trial[p2_name] = v2
            grid.append(trial)
    return grid


def _recon_parametric(y, Phi, kw, truth, orig_H=256, orig_W=256):
    """Run GAP-TV reconstruction using parametric shifts from kw."""
    col_shifts, row_shifts = compute_dispersion_shifts(
        NC, kw.get('a1', 2.0), kw.get('a2', 0.0), kw.get('alpha', 0.0))
    x_hat, _, _ = gap_tv_cassi_parametric(
        y, Phi, col_shifts, row_shifts,
        orig_H=orig_H, orig_W=orig_W,
        iter_max=50, tv_weight=0.1, tv_iter_max=5,
        _lambda=1, accelerate=True,
        X_orig=truth, show_iqa=False,
    )
    return np.clip(x_hat, 0, 1).astype(np.float32)


def run_w2_scenario(name, truth, mask2d, nominal_kw, perturbed_kw,
                    search_grid, rng):
    """Generic W2 mismatch scenario runner.

    Parameters
    ----------
    name : str — scenario label
    truth : (H,W,L) array
    mask2d : (H,W) array
    nominal_kw : dict — cassi_forward kwargs for nominal (no noise)
    perturbed_kw : dict — cassi_forward kwargs with mismatch + noise params
    search_grid : list of dicts — trial params (no noise) for grid search
    rng : np.random.RandomState

    Returns dict of metrics.
    """
    H, W = truth.shape[:2]
    print(f"    [{name}] Generating measurements ...")

    # 1. Nominal measurement (clean)
    y_nom, Phi_nom = cassi_forward(truth, mask2d, **nominal_kw)

    # 2. Perturbed measurement (with noise)
    y_pert, Phi_pert = cassi_forward(truth, mask2d, **perturbed_kw, rng=rng)

    # 3. Nominal reconstruction (baseline: clean operator, clean measurement)
    print(f"    [{name}] Nominal reconstruction ...")
    t0 = time.time()
    x_nom = _recon_parametric(y_nom, Phi_nom, nominal_kw, truth, H, W)
    t_nom = time.time() - t0
    m_nom = compute_metrics_allbands(truth, x_nom)

    # 4. Uncorrected reconstruction (nominal operator, perturbed measurement)
    print(f"    [{name}] Uncorrected reconstruction ...")
    t0 = time.time()
    x_uncorr = _recon_parametric(y_pert, Phi_nom, nominal_kw, truth, H, W)
    t_uncorr = time.time() - t0
    m_uncorr = compute_metrics_allbands(truth, x_uncorr)

    # 5. Grid search: minimize ||y_pert - y_trial||^2
    print(f"    [{name}] Grid search ({len(search_grid)} trials) ...")
    t0 = time.time()
    nll_nom = compute_nll_gaussian(y_pert, y_nom, sigma=NOISE_SIGMA)
    best_nll, best_trial = np.inf, dict(nominal_kw)
    for trial_kw in search_grid:
        y_trial, _ = cassi_forward(truth, mask2d, **trial_kw)
        nll = compute_nll_gaussian(y_pert, y_trial, sigma=NOISE_SIGMA)
        if nll < best_nll:
            best_nll, best_trial = nll, dict(trial_kw)
    t_grid = time.time() - t0
    nll_after = best_nll
    nll_decrease_pct = (nll_nom - nll_after) / (nll_nom + 1e-12) * 100

    print(f"    [{name}] Best params: {best_trial}")
    print(f"    [{name}] NLL: {nll_nom:.1f} -> {nll_after:.1f} "
          f"({nll_decrease_pct:.1f}% decrease, grid took {t_grid:.1f}s)")

    # 6. Corrected reconstruction
    print(f"    [{name}] Corrected reconstruction ...")
    _, Phi_corr = cassi_forward(truth, mask2d, **best_trial)
    y_recon = y_pert

    # If PSF was estimated, Wiener-deblur the measurement
    psf_sigma_best = best_trial.get('psf_sigma', 0.0)
    if psf_sigma_best > 0:
        y_recon = wiener_deblur(y_pert, psf_sigma_best)

    t0 = time.time()
    x_corr = _recon_parametric(y_recon, Phi_corr, best_trial, truth, H, W)
    t_corr = time.time() - t0
    m_corr = compute_metrics_allbands(truth, x_corr)

    # 7. SAM + residual
    sam_uncorr = spectral_angle_mapper(truth, x_uncorr)
    sam_corr = spectral_angle_mapper(truth, x_corr)

    y_pred_nom, _ = cassi_forward(truth, mask2d, **nominal_kw)
    y_pred_corr, _ = cassi_forward(truth, mask2d, **best_trial)
    res_uncorr = measurement_residual(y_pert, y_pred_nom)
    res_corr = measurement_residual(y_pert, y_pred_corr)

    print(f"    [{name}] PSNR: nom={m_nom['mean_psnr']:.2f}, "
          f"uncorr={m_uncorr['mean_psnr']:.2f}, corr={m_corr['mean_psnr']:.2f} "
          f"(delta={m_corr['mean_psnr'] - m_uncorr['mean_psnr']:+.2f})")
    print(f"    [{name}] SAM:  uncorr={sam_uncorr:.2f}°, corr={sam_corr:.2f}°")

    return {
        "scenario": name,
        "nll_before": round(nll_nom, 1),
        "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "psnr_nominal": m_nom["mean_psnr"],
        "ssim_nominal": m_nom["mean_ssim"],
        "psnr_uncorrected": m_uncorr["mean_psnr"],
        "ssim_uncorrected": m_uncorr["mean_ssim"],
        "psnr_corrected": m_corr["mean_psnr"],
        "ssim_corrected": m_corr["mean_ssim"],
        "psnr_delta": round(m_corr["mean_psnr"] - m_uncorr["mean_psnr"], 2),
        "ssim_delta": round(m_corr["mean_ssim"] - m_uncorr["mean_ssim"], 4),
        "sam_uncorrected": round(sam_uncorr, 2),
        "sam_corrected": round(sam_corr, 2),
        "sam_delta": round(sam_corr - sam_uncorr, 2),
        "residual_uncorrected": round(res_uncorr, 4),
        "residual_corrected": round(res_corr, 4),
        "injected_params": {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in perturbed_kw.items()
                            if k not in ('photon_gain', 'read_sigma')},
        "recovered_params": {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in best_trial.items()},
        "time_nominal": round(t_nom, 1),
        "time_uncorrected": round(t_uncorr, 1),
        "time_corrected": round(t_corr, 1),
        "time_grid": round(t_grid, 1),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CASSI Real-Data Benchmark")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                        help="Scenes to benchmark (default: all 10)")
    args = parser.parse_args()
    scenes = args.scenes

    results = {}
    print("=" * 70)
    print(f"CASSI Real-Data Benchmark — {len(scenes)} scene(s)")
    print("=" * 70)

    # ── Create RunBundle ─────────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "cassi_benchmark")
    rb_name = os.path.basename(rb_dir)
    art_dir = os.path.join(rb_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    print(f"RunBundle: {rb_name}")

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[0] Loading dataset and models ...")
    mask2d, mask3d_shift = load_mask()
    print(f"  Mask: {mask2d.shape}, Shift mask: {mask3d_shift.shape}")

    truths = []
    for sc in scenes:
        t = load_truth_scene(sc)
        truths.append(t)
        print(f"  {sc}: {t.shape}, range [{t.min():.3f}, {t.max():.3f}]")

    # ── Load deep models once ────────────────────────────────────────────
    t0_load = time.time()
    deep_models = load_deep_models()
    print(f"    All models loaded in {time.time() - t0_load:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # W1: 4-Solver Comparison — all scenes
    # ══════════════════════════════════════════════════════════════════════
    all_scene_results = {}

    # GAP-TV runs per-scene on CPU
    print(f"\n{'=' * 70}")
    print("W1: GAP-TV (per-scene, CPU)")
    print("=" * 70)
    gap_tv_recons = {}
    for si, sc in enumerate(scenes):
        print(f"\n  [{si+1}/{len(scenes)}] {sc} ...")
        scene_dir = os.path.join(art_dir, sc)
        os.makedirs(scene_dir, exist_ok=True)

        x_hat, wt = run_gap_tv_scene(truths[si], mask2d, mask3d_shift)
        m = compute_metrics_allbands(truths[si], x_hat)
        gap_tv_recons[sc] = x_hat
        if sc not in all_scene_results:
            all_scene_results[sc] = {}
        all_scene_results[sc]["gap_tv"] = {**m, "wall_time": round(wt, 2)}
        save_array(os.path.join(scene_dir, "x_hat_gap_tv.npy"), x_hat)
        print(f"    PSNR={m['mean_psnr']:.2f} dB, SSIM={m['mean_ssim']:.4f}, time={wt:.1f}s")

    # Deep solvers run in batch on GPU
    print(f"\n{'=' * 70}")
    print("W1: Deep solvers (batch GPU)")
    print("=" * 70)
    deep_recons, deep_times = run_deep_solvers(deep_models, truths, mask2d, mask3d_shift)

    for solver_name in ["hdnet", "mst_s", "mst_l"]:
        recons = deep_recons[solver_name]  # (10, 256, 256, 28)
        wt = deep_times[solver_name]
        per_scene_time = wt / len(scenes)

        for si, sc in enumerate(scenes):
            scene_dir = os.path.join(art_dir, sc)
            os.makedirs(scene_dir, exist_ok=True)
            x_hat = np.clip(recons[si], 0, 1).astype(np.float32)
            m = compute_metrics_allbands(truths[si], x_hat)
            if sc not in all_scene_results:
                all_scene_results[sc] = {}
            all_scene_results[sc][solver_name] = {**m, "wall_time": round(per_scene_time, 2)}
            save_array(os.path.join(scene_dir, f"x_hat_{solver_name}.npy"), x_hat)

        # Summary for this solver
        avg_p = mean([all_scene_results[sc][solver_name]["mean_psnr"] for sc in scenes])
        avg_s = mean([all_scene_results[sc][solver_name]["mean_ssim"] for sc in scenes])
        print(f"  {solver_name}: avg PSNR={avg_p:.2f} dB, avg SSIM={avg_s:.4f}, total time={wt:.1f}s")

    results["w1"] = all_scene_results

    # ── Compute average across all scenes ────────────────────────────────
    solver_names = ["gap_tv", "hdnet", "mst_s", "mst_l"]
    avg_results = {}
    for sname in solver_names:
        psnrs = [all_scene_results[sc][sname]["mean_psnr"] for sc in scenes]
        ssims = [all_scene_results[sc][sname]["mean_ssim"] for sc in scenes]
        times = [all_scene_results[sc][sname]["wall_time"] for sc in scenes]
        avg_results[sname] = {
            "avg_psnr": round(mean(psnrs), 2),
            "avg_ssim": round(mean(ssims), 4),
            "avg_time": round(mean(times), 2),
        }
    results["w1_average"] = avg_results

    # ══════════════════════════════════════════════════════════════════════
    # W2: Realistic mismatch model — 5 scenarios (scene01, GAP-TV)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("W2: Realistic mismatch model (5 scenarios, scene01, GAP-TV)")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    truth01 = truths[0]
    nominal_kw = {"a1": 2.0, "a2": 0.0, "alpha": 0.0}
    noise_kw = {"photon_gain": 1000.0, "read_sigma": 5.0}

    w2_scenarios = {}

    # W2a: Mask translation (dx=2, dy=1)
    print("\n  W2a: Mask translation (dx=2, dy=1)")
    grid_a = _make_grid_2d(
        nominal_kw, "mask_dx", np.arange(-5, 6, dtype=float),
        "mask_dy", np.arange(-3, 4, dtype=float))
    w2_scenarios["w2a"] = run_w2_scenario(
        "W2a", truth01, mask2d, nominal_kw,
        {**nominal_kw, "mask_dx": 2.0, "mask_dy": 1.0, **noise_kw},
        grid_a, rng)

    # W2b: Mask rotation (theta=1.0°)
    print("\n  W2b: Mask rotation (theta=1.0°)")
    grid_b = _make_grid(
        nominal_kw, "mask_theta",
        [np.radians(t) for t in np.arange(-3.0, 3.5, 0.5)])
    w2_scenarios["w2b"] = run_w2_scenario(
        "W2b", truth01, mask2d, nominal_kw,
        {**nominal_kw, "mask_theta": np.radians(1.0), **noise_kw},
        grid_b, rng)

    # W2c: Dispersion slope (a1=2.15 vs nominal 2.0)
    print("\n  W2c: Dispersion slope (a1=2.15)")
    grid_c = _make_grid(
        nominal_kw, "a1",
        np.arange(1.8, 2.325, 0.025).tolist())
    w2_scenarios["w2c"] = run_w2_scenario(
        "W2c", truth01, mask2d, nominal_kw,
        {**nominal_kw, "a1": 2.15, **noise_kw},
        grid_c, rng)

    # W2d: Dispersion axis angle (alpha=2°)
    print("\n  W2d: Dispersion axis angle (alpha=2°)")
    grid_d = _make_grid(
        nominal_kw, "alpha",
        [np.radians(a) for a in np.arange(-5.0, 6.0, 1.0)])
    w2_scenarios["w2d"] = run_w2_scenario(
        "W2d", truth01, mask2d, nominal_kw,
        {**nominal_kw, "alpha": np.radians(2.0), **noise_kw},
        grid_d, rng)

    # W2e: PSF blur (sigma=1.5)
    print("\n  W2e: PSF blur (sigma=1.5)")
    grid_e = _make_grid(
        nominal_kw, "psf_sigma",
        np.arange(0, 3.25, 0.25).tolist())
    w2_scenarios["w2e"] = run_w2_scenario(
        "W2e", truth01, mask2d, nominal_kw,
        {**nominal_kw, "psf_sigma": 1.5, **noise_kw},
        grid_e, rng)

    # Build W2 summary for results JSON
    w2_report = {
        "scenarios": w2_scenarios,
        "a_definition": "callable",
        "a_extraction_method": "provided",
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "mismatch_description": (
            "5 realistic mismatch scenarios: mask translation (dx,dy), "
            "mask rotation (theta), dispersion slope (a1), "
            "dispersion axis angle (alpha), PSF blur (sigma)"),
        "correction_family": "Pre",
        "noise_model": "Poisson(gain=1000) + read(sigma=5.0)",
    }
    results["w2"] = w2_report

    save_operator_meta(rb_dir, {
        "a_definition": "callable",
        "a_extraction_method": "provided",
        "a_sha256": sha256_hex16(mask2d),
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "n_scenarios": 5,
        "scenarios": {k: {
            "injected": v["injected_params"],
            "recovered": v["recovered_params"],
            "nll_decrease_pct": v["nll_decrease_pct"],
            "psnr_delta": v["psnr_delta"],
        } for k, v in w2_scenarios.items()},
        "correction_family": "Pre",
        "noise_model": "Poisson(gain=1000) + read(sigma=5.0)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Print W2 summary table
    print(f"\n{'=' * 70}")
    print("W2 Summary")
    print("=" * 70)
    hdr = (f"{'Scenario':<12} {'NLL decr%':>10} {'PSNR uncorr':>12} "
           f"{'PSNR corr':>10} {'ΔPSNR':>7} {'SAM uncorr':>11} "
           f"{'SAM corr':>9}")
    print(hdr)
    print("-" * len(hdr))
    for key in ["w2a", "w2b", "w2c", "w2d", "w2e"]:
        s = w2_scenarios[key]
        print(f"{s['scenario']:<12} {s['nll_decrease_pct']:>9.1f}% "
              f"{s['psnr_uncorrected']:>11.2f} {s['psnr_corrected']:>10.2f} "
              f"{s['psnr_delta']:>+6.2f} {s['sam_uncorrected']:>10.2f}° "
              f"{s['sam_corrected']:>8.2f}°")

    # ── Trace (from scene01 — 8-stage parametric pipeline) ───────────────
    truth01 = truths[0]
    mask_warped = warp_mask(mask2d, 2.0, 1.0, 0.0)  # example warped mask
    mask3d_local = np.tile(mask2d[:, :, np.newaxis], (1, 1, NC))
    masked = truth01 * mask3d_local
    col_shifts_nom, row_shifts_nom = compute_dispersion_shifts(NC, 2.0, 0.0, 0.0)
    dispersed = shift_np_parametric(masked, col_shifts_nom, row_shifts_nom)
    integrated = np.sum(dispersed, axis=2)
    psf_blurred = apply_psf(integrated, sigma=1.5)
    noisy_y = apply_poisson_read_noise(psf_blurred, photon_gain=1000.0,
                                       read_sigma=5.0,
                                       rng=np.random.RandomState(SEED))

    trace = {}
    trace["00_input_x"] = truth01.copy()
    trace["01_mask_warped"] = mask_warped.copy()
    trace["02_masked"] = masked.astype(np.float32)
    trace["03_dispersed"] = dispersed.astype(np.float32)
    trace["04_integrated"] = integrated.astype(np.float32)
    trace["05_psf_blurred"] = psf_blurred.astype(np.float32)
    trace["06_noisy_y"] = noisy_y.astype(np.float32)
    trace["07_recon_gaptv"] = gap_tv_recons.get(scenes[0], truth01).copy()
    save_trace(rb_dir, trace)

    results["trace"] = []
    for i, key in enumerate(sorted(trace.keys())):
        arr = trace[key]
        results["trace"].append({
            "stage": i, "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape), "dtype": str(arr.dtype),
            "range_min": round(float(arr.min()), 4),
            "range_max": round(float(arr.max()), 4),
            "artifact_path": f"artifacts/trace/{key}.npy",
        })

    # ── Environment ──────────────────────────────────────────────────────
    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()[:12]
    except Exception:
        pass

    results["env"] = {
        "seed": SEED, "pwm_version": git_sha,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
    }
    try:
        import scipy; results["env"]["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    try:
        import torch; results["env"]["torch_version"] = torch.__version__
    except ImportError:
        pass

    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name
    results["scenes"] = scenes

    results_path = os.path.join(rb_dir, "cassi_benchmark_results.json")
    save_json(results_path, results)

    # ══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — All Scenes")
    print("=" * 70)

    header = f"{'Scene':<12}"
    for sn in solver_names:
        header += f" {sn:>14}"
    print(f"\nPSNR (dB):")
    print(header)
    print("-" * len(header))
    for sc in scenes:
        row = f"{sc:<12}"
        for sn in solver_names:
            row += f" {all_scene_results[sc][sn]['mean_psnr']:>14.2f}"
        print(row)
    row = f"{'AVERAGE':<12}"
    for sn in solver_names:
        row += f" {avg_results[sn]['avg_psnr']:>14.2f}"
    print("-" * len(header))
    print(row)

    print(f"\nSSIM:")
    print(header)
    print("-" * len(header))
    for sc in scenes:
        row = f"{sc:<12}"
        for sn in solver_names:
            row += f" {all_scene_results[sc][sn]['mean_ssim']:>14.4f}"
        print(row)
    row = f"{'AVERAGE':<12}"
    for sn in solver_names:
        row += f" {avg_results[sn]['avg_ssim']:>14.4f}"
    print("-" * len(header))
    print(row)

    print(f"\nW2 (5 scenarios on scene01):")
    for key in ["w2a", "w2b", "w2c", "w2d", "w2e"]:
        s = w2_scenarios[key]
        print(f"  {s['scenario']}: NLL decrease {s['nll_decrease_pct']:.1f}%, "
              f"PSNR delta {s['psnr_delta']:+.2f} dB")

    print(f"\nRunBundle: runs/{rb_name}")
    print(f"Results:   {results_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
