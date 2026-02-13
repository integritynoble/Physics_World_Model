#!/usr/bin/env python3
"""CACTI Real-Data Benchmark — 4 Solvers on 6 Grayscale Benchmark Scenes.

Runs GAP-TV, PnP-FFDNet, ELP-Unfolding, and EfficientSCI on the full
grayscale benchmark (kobe32, crash32, aerial32, traffic48, runner40, drop40).

W1: 4-solver comparison on all 6 scenes
W2: Multi-stage realistic mismatch model (spatial + temporal + sensor)
    with Poisson+Read+Quantization noise, evaluated on 3 scenes

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_benchmark.py
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_benchmark.py --scenes kobe32
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
PNPSCI_ROOT = "/home/spiritai/PnP-SCI_python-master"
ELP_ROOT = "/home/spiritai/ELP-Unfolding-master"
ESCI_ROOT = "/home/spiritai/EfficientSCI-main"

# ── Monkey-patch skimage for PnP-SCI compatibility ──────────────────────
# PnP-SCI uses old scikit-image API (n_iter_max, multichannel).
# Current skimage uses (max_num_iter, channel_axis).
import skimage.restoration
import skimage.restoration._denoise as _skd

_orig_tv = _skd.denoise_tv_chambolle
def _compat_tv(image, weight=0.1, eps=0.0002, max_num_iter=200, **kw):
    if "n_iter_max" in kw:
        max_num_iter = kw.pop("n_iter_max")
    kw.pop("multichannel", None)
    return _orig_tv(image, weight=weight, eps=eps, max_num_iter=max_num_iter)

_orig_wavelet = _skd.denoise_wavelet
def _compat_wavelet(image, *a, **kw):
    mc = kw.pop("multichannel", None)
    if mc is not None and "channel_axis" not in kw:
        kw["channel_axis"] = -1 if mc else None
    return _orig_wavelet(image, *a, **kw)

_skd.denoise_tv_chambolle = _compat_tv
_skd.denoise_wavelet = _compat_wavelet
skimage.restoration.denoise_tv_chambolle = _compat_tv
skimage.restoration.denoise_wavelet = _compat_wavelet

from skimage import metrics as _skm
_orig_ssim_sk = _skm.structural_similarity
def _compat_ssim(*a, **kw):
    mc = kw.pop("multichannel", None)
    if mc is not None and "channel_axis" not in kw:
        kw["channel_axis"] = -1 if mc else None
    return _orig_ssim_sk(*a, **kw)
_skm.structural_similarity = _compat_ssim
skimage.metrics.structural_similarity = _compat_ssim

from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)

# ── Constants ────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "cacti"
DATASET_DIR = os.path.join(PNPSCI_ROOT, "dataset", "cacti", "grayscale_benchmark")
ALL_SCENES = ["kobe32", "crash32", "aerial32", "traffic48", "runner40", "drop40"]
MAXB = 255.0
NOISE_SIGMA = 5.0
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# ── Helpers ──────────────────────────────────────────────────────────────

def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def psnr_frame(gt: np.ndarray, rec: np.ndarray, maxval: float = 255.0) -> float:
    mse = np.mean((gt.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(maxval ** 2 / mse))


def ssim_frame(gt: np.ndarray, rec: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        gt.astype(np.float64), rec.astype(np.float64), data_range=255.0
    ))


def compute_metrics_allframes(orig: np.ndarray, recon: np.ndarray) -> dict:
    """Compute per-frame and mean PSNR/SSIM for (H,W,F) arrays in [0,255]."""
    nframes = orig.shape[2]
    psnrs, ssims = [], []
    for f in range(nframes):
        psnrs.append(psnr_frame(orig[:, :, f], recon[:, :, f]))
        ssims.append(ssim_frame(orig[:, :, f], recon[:, :, f]))
    return {
        "mean_psnr": round(mean(psnrs), 2),
        "mean_ssim": round(mean(ssims), 4),
        "per_frame_psnr": [round(p, 2) for p in psnrs],
        "per_frame_ssim": [round(s, 4) for s in ssims],
    }


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def load_scene(scene: str) -> tuple:
    """Load .mat file -> (meas, mask, orig)."""
    path = os.path.join(DATASET_DIR, f"{scene}_cacti.mat")
    data = sio.loadmat(path)
    return np.float32(data["meas"]), np.float32(data["mask"]), np.float32(data["orig"])


# ══════════════════════════════════════════════════════════════════════════
# Parametric CACTI forward model (physically realistic)
# ══════════════════════════════════════════════════════════════════════════

def warp_mask_3d(mask3d, dx=0.0, dy=0.0, theta=0.0, blur_sigma=0.0):
    """Affine-transform 3D mask (H,W,T): translate (dx,dy) + rotate theta (deg)
    + optional Gaussian blur per frame.
    Uses scipy.ndimage for sub-pixel interpolation."""
    from scipy.ndimage import affine_transform, gaussian_filter
    H, W, T = mask3d.shape
    theta_rad = np.radians(theta)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    R_inv = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    cy, cx = H / 2, W / 2
    offset = np.array([cy, cx]) - R_inv @ np.array([cy - dy, cx - dx])

    result = np.zeros_like(mask3d)
    for t in range(T):
        frame = affine_transform(mask3d[:, :, t], R_inv, offset=offset,
                                 order=1, mode='constant', cval=0.0)
        if blur_sigma > 0:
            frame = gaussian_filter(frame, sigma=blur_sigma)
        result[:, :, t] = frame
    return result.astype(np.float32)


def apply_temporal_mixing(x_cube, mask3d, clock_offset=0.0, duty_cycle=1.0,
                          timing_jitter_std=0.0, temporal_tau=0.0, rng=None):
    """Apply temporal coding with realistic timing mismatches.

    Parameters
    ----------
    x_cube : (H,W,T) video cube in [0,255]
    mask3d : (H,W,T) mask array
    clock_offset : float, fractional frame offset (DMD vs camera sync)
    duty_cycle : float in (0,1], effective sub-exposure fraction
    timing_jitter_std : float, per-frame timing jitter (std, in frames)
    temporal_tau : float, temporal blur time constant (frames)
    rng : RandomState for jitter

    Returns
    -------
    y : (H,W) 2D measurement
    """
    H, W, T = x_cube.shape
    if rng is None:
        rng = np.random.RandomState(SEED)

    # Apply clock offset: shift effective mask in time domain
    # This causes frame t to be partially exposed with mask t-1 and t+1
    effective_mask = mask3d.copy()

    if abs(clock_offset) > 1e-6:
        alpha = clock_offset  # fractional shift
        new_mask = np.zeros_like(mask3d)
        for t in range(T):
            t_lo = max(0, t)
            t_hi = min(T - 1, t + 1)
            t_prev = max(0, t - 1)
            if alpha >= 0:
                new_mask[:, :, t] = (1 - alpha) * mask3d[:, :, t] + alpha * mask3d[:, :, t_hi]
            else:
                new_mask[:, :, t] = (1 + alpha) * mask3d[:, :, t] + (-alpha) * mask3d[:, :, t_prev]
        effective_mask = new_mask

    # Apply duty cycle: scale mask transmission
    effective_mask = effective_mask * duty_cycle

    # Apply per-frame timing jitter
    if timing_jitter_std > 0:
        jitter = rng.randn(T) * timing_jitter_std
        for t in range(T):
            j = jitter[t]
            if abs(j) > 1e-6:
                t_lo = max(0, t - 1)
                t_hi = min(T - 1, t + 1)
                leak = min(abs(j), 0.5)
                if j > 0:
                    effective_mask[:, :, t] = (1 - leak) * effective_mask[:, :, t] + \
                                               leak * effective_mask[:, :, t_hi]
                else:
                    effective_mask[:, :, t] = (1 - leak) * effective_mask[:, :, t] + \
                                               leak * effective_mask[:, :, t_lo]

    # Apply temporal response blur (exponential decay across frames)
    if temporal_tau > 0:
        from scipy.ndimage import uniform_filter1d
        kernel_size = max(1, int(2 * temporal_tau + 1))
        if kernel_size > 1:
            for h in range(0, H, 16):
                h_end = min(h + 16, H)
                block = effective_mask[h:h_end, :, :]
                effective_mask[h:h_end, :, :] = uniform_filter1d(
                    block, size=kernel_size, axis=2, mode='nearest')

    # Integrate: y = sum_t(mask_t * x_t)
    y = np.sum(effective_mask * x_cube, axis=2)
    return y.astype(np.float32)


def apply_psf_2d(y, sigma):
    """Apply Gaussian PSF blur to 2D measurement."""
    if sigma <= 0:
        return y
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(y, sigma=sigma).astype(np.float32)


def apply_poisson_read_quant_noise(y_clean, peak_photons=10000.0,
                                    read_sigma=5.0, bit_depth=12,
                                    rng=None):
    """Poisson shot noise + Gaussian read noise + quantization.

    Parameters
    ----------
    y_clean : 2D measurement (in data units, [0, ~sum_of_maxvals])
    peak_photons : photon count at max signal level
    read_sigma : read noise std in photon-equivalent units
    bit_depth : ADC bit depth for quantization (0 = no quantization)
    rng : RandomState

    Returns
    -------
    y_noisy : noisy measurement (same scale as input)
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    y_max = np.max(y_clean) + 1e-10
    # Normalize to [0,1] for photon scaling
    y_norm = np.clip(y_clean / y_max, 0, None)

    # Poisson shot noise
    y_photon = rng.poisson(peak_photons * y_norm) / peak_photons

    # Read noise
    y_noisy = y_photon + rng.randn(*y_clean.shape) * (read_sigma / peak_photons)

    # Quantization
    if bit_depth > 0:
        n_levels = 2 ** bit_depth
        y_noisy = np.round(y_noisy * n_levels) / n_levels

    # Scale back to original range
    y_noisy = y_noisy * y_max
    return y_noisy.astype(np.float32)


def cacti_forward(orig_block, mask3d,
                  mask_dx=0.0, mask_dy=0.0, mask_theta=0.0,
                  mask_blur_sigma=0.0,
                  clock_offset=0.0, duty_cycle=1.0,
                  timing_jitter_std=0.0, temporal_response_tau=0.0,
                  psf_sigma=0.0, gain=1.0, offset=0.0,
                  peak_photons=0, read_sigma=5.0, bit_depth=12,
                  rng=None):
    """Full parametric CACTI forward model.

    Parameters
    ----------
    orig_block : (H,W,T) video cube in [0,255]
    mask3d : (H,W,T) mask array
    [spatial params] mask_dx, mask_dy, mask_theta, mask_blur_sigma
    [temporal params] clock_offset, duty_cycle, timing_jitter_std, temporal_response_tau
    [sensor params] psf_sigma, gain, offset
    [noise params] peak_photons (0=no noise), read_sigma, bit_depth

    Returns
    -------
    y : (H,W) 2D measurement
    mask_used : (H,W,T) effective mask after warping
    """
    # 1. Warp mask (spatial mismatches)
    if abs(mask_dx) > 1e-6 or abs(mask_dy) > 1e-6 or abs(mask_theta) > 1e-6 \
       or mask_blur_sigma > 1e-6:
        mask_used = warp_mask_3d(mask3d, mask_dx, mask_dy, mask_theta,
                                 mask_blur_sigma)
    else:
        mask_used = mask3d.copy()

    # 2. Temporal coding with mismatches
    y = apply_temporal_mixing(orig_block, mask_used,
                              clock_offset=clock_offset,
                              duty_cycle=duty_cycle,
                              timing_jitter_std=timing_jitter_std,
                              temporal_tau=temporal_response_tau,
                              rng=rng)

    # 3. PSF blur (objective lens)
    y = apply_psf_2d(y, psf_sigma)

    # 4. Sensor gain + offset
    y = gain * y + offset * MAXB

    # 5. Noise
    if peak_photons > 0:
        y = apply_poisson_read_quant_noise(y, peak_photons, read_sigma,
                                            bit_depth, rng)

    return y, mask_used


def compute_nll_poisson_read(y_meas, y_pred, peak_photons=10000.0,
                              read_sigma=5.0):
    """NLL for Poisson+Read noise model (Gaussian approximation).
    Uses combined variance: var = y_pred/gain + read_sigma^2."""
    y_max = np.max(y_meas) + 1e-10
    y_norm_pred = np.clip(y_pred / y_max, 1e-10, None)
    sigma2 = y_norm_pred / peak_photons + (read_sigma / peak_photons) ** 2
    sigma2_scaled = sigma2 * y_max ** 2
    residual = (y_meas - y_pred) ** 2
    # Sum of residual^2 / (2*sigma^2)
    nll = float(0.5 * np.sum(residual / (sigma2_scaled + 1e-10)))
    return nll


# ── Grid search helpers ───────────────────────────────────────────────────

def _make_grid_1d(base_kw, param_name, values):
    """Generate search grid varying one param."""
    grid = []
    for v in values:
        trial = dict(base_kw)
        trial[param_name] = float(v)
        grid.append(trial)
    return grid


def _make_grid_2d(base_kw, p1_name, p1_vals, p2_name, p2_vals):
    """Generate 2D search grid."""
    grid = []
    for v1 in p1_vals:
        for v2 in p2_vals:
            trial = dict(base_kw)
            trial[p1_name] = float(v1)
            trial[p2_name] = float(v2)
            grid.append(trial)
    return grid


def _make_grid_3d(base_kw, p1_name, p1_vals, p2_name, p2_vals,
                  p3_name, p3_vals):
    """Generate 3D search grid."""
    grid = []
    for v1 in p1_vals:
        for v2 in p2_vals:
            for v3 in p3_vals:
                trial = dict(base_kw)
                trial[p1_name] = float(v1)
                trial[p2_name] = float(v2)
                trial[p3_name] = float(v3)
                grid.append(trial)
    return grid


# ══════════════════════════════════════════════════════════════════════════
# sys.path management
# ══════════════════════════════════════════════════════════════════════════

def _enter_pnpsci():
    for p in [ELP_ROOT, ESCI_ROOT]:
        if p in sys.path:
            sys.path.remove(p)
    if PNPSCI_ROOT not in sys.path:
        sys.path.insert(0, PNPSCI_ROOT)
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if hasattr(mod, "__file__") and mod.__file__ and "PnP-SCI" not in mod.__file__:
            del sys.modules["utils"]


def _enter_elp():
    for p in [PNPSCI_ROOT, ESCI_ROOT]:
        if p in sys.path:
            sys.path.remove(p)
    if ELP_ROOT not in sys.path:
        sys.path.insert(0, ELP_ROOT)
    for mod_name in ["utils", "train_common"]:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "__file__") and mod.__file__ and "ELP-Unfolding" not in (mod.__file__ or ""):
                del sys.modules[mod_name]


def _enter_esci():
    for p in [PNPSCI_ROOT, ELP_ROOT]:
        if p in sys.path:
            sys.path.remove(p)
    if ESCI_ROOT not in sys.path:
        sys.path.insert(0, ESCI_ROOT)
    for mod_name in ["utils", "train_common"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]


# ══════════════════════════════════════════════════════════════════════════
# Solver 1: GAP-TV (no persistent model)
# ══════════════════════════════════════════════════════════════════════════

def run_gap_tv(meas, mask, orig, nframe):
    _enter_pnpsci()
    from pnp_sci import admmdenoise_cacti
    from utils import A_, At_
    A = lambda x: A_(x, mask)
    At = lambda y: At_(y, mask)
    v, t_elapsed, psnr_list, ssim_list, _ = admmdenoise_cacti(
        meas, mask, A, At,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=nframe, MAXB=MAXB, maskdirection="plain",
        _lambda=1, accelerate=True,
        denoiser="tv", iter_max=40, tv_weight=0.3, tv_iter_max=5,
    )
    # PnP-SCI output is in [0,1] range
    v = np.clip(v * 255.0, 0, 255).astype(np.float32)
    return v, t_elapsed, psnr_list, ssim_list


# ══════════════════════════════════════════════════════════════════════════
# Solver 2: PnP-FFDNet (load model once)
# ══════════════════════════════════════════════════════════════════════════

def load_ffdnet_model():
    _enter_pnpsci()
    import torch
    from packages.ffdnet.models import FFDNet
    ffdnet_path = os.path.join(PNPSCI_ROOT, "packages", "ffdnet", "models", "net_gray.pth")
    net = FFDNet(num_input_channels=1)
    state_dict = torch.load(ffdnet_path, map_location="cuda:0")
    model = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_pnp_ffdnet(meas, mask, orig, nframe, ffdnet_model):
    _enter_pnpsci()
    from pnp_sci import admmdenoise_cacti
    from utils import A_, At_
    A = lambda x: A_(x, mask)
    At = lambda y: At_(y, mask)
    v, t_elapsed, psnr_list, ssim_list, _ = admmdenoise_cacti(
        meas, mask, A, At,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=nframe, MAXB=MAXB, maskdirection="plain",
        _lambda=1, accelerate=True,
        denoiser="ffdnet", model=ffdnet_model,
        iter_max=[10, 10, 10, 10],
        sigma=[50/255, 25/255, 12/255, 6/255],
    )
    v = np.clip(v * 255.0, 0, 255).astype(np.float32)
    return v, t_elapsed, psnr_list, ssim_list


# ══════════════════════════════════════════════════════════════════════════
# Solver 3: ELP-Unfolding (load model once)
# ══════════════════════════════════════════════════════════════════════════

def load_elp_model():
    _enter_elp()
    import torch
    os.environ["WANDB_MODE"] = "disabled"
    elp_args = {
        "GPU": 0, "batch_size": 3, "temporal_length": 8, "patchsize": 256,
        "iter__number": 8, "init_channels": 512, "pres_channels": 512,
        "init_input": 8, "pres_input": 8, "priors": 6,
        "lr": 2e-5, "resume_training": True,
        "code_dir": "/home/spiritai/elpData/traindata/DAVIS-480-train/code",
        "log_dir": "/home/spiritai/ELP-Unfolding-master/trained_dataset",
        "use_first_stage": False,
    }
    from train_common import resume_training
    model, _, _ = resume_training(elp_args)
    model.eval()
    return model


def run_elp_unfolding(orig, mask_dataset, elp_model):
    """Run ELP-Unfolding using a pre-loaded model."""
    _enter_elp()
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_total = orig.shape[2]
    n_coded = n_total // 8
    batch_size = n_coded

    z = np.zeros((256, 256, 8, batch_size), dtype=np.float32)
    for fn in range(batch_size):
        z[:, :, :, fn] = orig[:, :, fn * 8:(fn + 1) * 8]
    z_tensor = torch.from_numpy(z).float().permute(3, 2, 0, 1)
    img_val = z_tensor / 255.0

    mask_tensor = torch.from_numpy(mask_dataset).float().permute(2, 0, 1)
    mask_tensor = mask_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    measurement = torch.sum(img_val * mask_tensor, dim=1, keepdim=True)

    mask_gpu = mask_tensor.to(device)
    meas_gpu = measurement.to(device)
    img_out_ori = torch.ones(batch_size, 8, 256, 256).to(device)

    t0 = time.time()
    with torch.no_grad():
        img_out, _ = elp_model(mask_gpu, meas_gpu, img_out_ori)
    torch.cuda.synchronize()
    t_elapsed = time.time() - t0

    recon = img_out[-1].detach().cpu().numpy()
    recon_255 = np.clip(recon * 255.0, 0, 255)

    recon_hwf = np.zeros((256, 256, n_total), dtype=np.float32)
    for fn in range(batch_size):
        recon_hwf[:, :, fn * 8:(fn + 1) * 8] = recon_255[fn].transpose(1, 2, 0)

    del mask_gpu, meas_gpu, img_out_ori, img_out
    torch.cuda.empty_cache()
    return recon_hwf, t_elapsed


# ══════════════════════════════════════════════════════════════════════════
# Solver 4: EfficientSCI (load model once)
# ══════════════════════════════════════════════════════════════════════════

def load_esci_model():
    _enter_esci()
    import torch
    import einops
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load mask
    esci_mask_path = os.path.join(ESCI_ROOT, "test_datasets", "mask", "efficientsci_mask.mat")
    esci_mask = sio.loadmat(esci_mask_path)["mask"].astype(np.float32)  # (256,256,8)
    esci_mask_t = esci_mask.transpose(2, 0, 1)  # (8,H,W)
    mask_s = np.sum(esci_mask_t, axis=0)
    mask_s[mask_s == 0] = 1

    # Build model
    from cacti.models.efficientsci import EfficientSCI
    model = EfficientSCI(in_ch=64, units=8, group_num=4, color_ch=1).to(device)
    ckpt_path = os.path.join(ESCI_ROOT, "checkpoints", "efficientsci_base.pth")
    resume_dict = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in resume_dict:
        model.load_state_dict(resume_dict["model_state_dict"])
    else:
        model.load_state_dict(resume_dict)
    model.eval()

    Phi = einops.repeat(esci_mask_t, "cr h w -> b cr h w", b=1)
    Phi_s = einops.repeat(mask_s, "h w -> b 1 h w", b=1)
    Phi = torch.from_numpy(Phi).float().to(device)
    Phi_s = torch.from_numpy(Phi_s).float().to(device)

    return model, esci_mask_t, Phi, Phi_s


def run_efficientsci(orig, esci_model, esci_mask_t, Phi, Phi_s):
    """Run EfficientSCI using a pre-loaded model."""
    _enter_esci()
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_total = orig.shape[2]
    n_coded = n_total // 8
    recon_hwf = np.zeros((256, 256, n_total), dtype=np.float32)

    t_total = 0
    for ci in range(n_coded):
        gt_block = orig[:, :, ci * 8:(ci + 1) * 8]
        gt_normed = gt_block / 255.0
        gt_t = gt_normed.transpose(2, 0, 1)
        meas_np = np.sum(gt_t * esci_mask_t, axis=0)
        meas_tensor = torch.from_numpy(meas_np).float().unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = esci_model(meas_tensor, Phi, Phi_s)
        torch.cuda.synchronize()
        t_total += time.time() - t0

        if not isinstance(outputs, list):
            outputs = [outputs]
        output = outputs[-1][0].cpu().numpy().astype(np.float32)
        recon_block = np.clip(output * 255.0, 0, 255)
        recon_hwf[:, :, ci * 8:(ci + 1) * 8] = recon_block.transpose(1, 2, 0)

    return recon_hwf, t_total


# ══════════════════════════════════════════════════════════════════════════
# W2: Multi-stage mismatch correction (realistic CACTI)
# ══════════════════════════════════════════════════════════════════════════

# Injected mismatch parameters (physically realistic set)
W2_INJECTED = {
    "mask_dx": 1.5,
    "mask_dy": 1.0,
    "mask_theta": 0.3,
    "mask_blur_sigma": 0.0,
    "clock_offset": 0.08,
    "duty_cycle": 0.92,
    "timing_jitter_std": 0.0,
    "temporal_response_tau": 0.0,
    "gain": 1.05,
    "offset": 0.005,
}

# Noise parameters (Poisson + Read + Quantization)
W2_NOISE = {
    "peak_photons": 10000.0,
    "read_sigma": 5.0,
    "bit_depth": 12,
}

W2_SCENES = ["kobe32", "crash32", "runner40"]


def run_w2_multistage(orig, mask, scene_name, rng):
    """Multi-stage CACTI operator correction on one scene.

    Stages:
      1. Coarse spatial: grid search mask_dx, mask_dy (integer, ±3)
      2. Refine spatial: grid search mask_dx, mask_dy (±1 around best, 0.25 step)
                         + mask_theta (±0.6°, 0.1° step)
      3. Temporal: grid search clock_offset (±0.3, step 0.05)
                   + duty_cycle (0.8–1.0, step 0.02)
      4. Sensor: grid search gain (0.9–1.1, step 0.02)
                 + offset (-0.02–0.02, step 0.005)

    Returns dict with per-stage NLL and final recon metrics.
    """
    _enter_pnpsci()
    from pnp_sci import admmdenoise_cacti
    from utils import A_, At_

    H, W_im = orig.shape[:2]
    n_total = orig.shape[2]
    nmask = mask.shape[2]
    n_coded = n_total // nmask

    # Nominal parameters (perfect operator)
    nominal_kw = {
        "mask_dx": 0.0, "mask_dy": 0.0, "mask_theta": 0.0,
        "mask_blur_sigma": 0.0,
        "clock_offset": 0.0, "duty_cycle": 1.0,
        "timing_jitter_std": 0.0, "temporal_response_tau": 0.0,
        "gain": 1.0, "offset": 0.0,
    }

    # Generate perturbed measurements (all coded frames)
    perturbed_kw = {**W2_INJECTED, **W2_NOISE}
    y_pert = np.zeros((H, W_im, n_coded), dtype=np.float32)
    for ci in range(n_coded):
        block = orig[:, :, ci * nmask:(ci + 1) * nmask]
        y_ci, _ = cacti_forward(block, mask, **perturbed_kw,
                                rng=np.random.RandomState(SEED + ci))
        y_pert[:, :, ci] = y_ci

    # Generate nominal prediction for NLL baseline
    y_nom = np.zeros((H, W_im, n_coded), dtype=np.float32)
    for ci in range(n_coded):
        block = orig[:, :, ci * nmask:(ci + 1) * nmask]
        y_ci, _ = cacti_forward(block, mask, **nominal_kw)
        y_nom[:, :, ci] = y_ci

    nll_before = compute_nll_poisson_read(y_pert, y_nom,
                                           W2_NOISE["peak_photons"],
                                           W2_NOISE["read_sigma"])

    # Helper: compute NLL for a given parameter set
    def _nll_for_params(kw):
        y_trial = np.zeros((H, W_im, n_coded), dtype=np.float32)
        for ci in range(n_coded):
            block = orig[:, :, ci * nmask:(ci + 1) * nmask]
            y_ci, _ = cacti_forward(block, mask, **kw)
            y_trial[:, :, ci] = y_ci
        return compute_nll_poisson_read(y_pert, y_trial,
                                         W2_NOISE["peak_photons"],
                                         W2_NOISE["read_sigma"])

    stage_results = {}
    best_kw = dict(nominal_kw)

    # ── Stage 1: Coarse spatial (integer dx, dy) ─────────────────────────
    print(f"    [{scene_name}] Stage 1: Coarse spatial (dx,dy integer ±3) ...")
    t0 = time.time()
    grid_1 = _make_grid_2d(best_kw, "mask_dx", np.arange(-3, 4, dtype=float),
                           "mask_dy", np.arange(-3, 4, dtype=float))
    best_nll_1 = np.inf
    for trial_kw in grid_1:
        nll = _nll_for_params(trial_kw)
        if nll < best_nll_1:
            best_nll_1 = nll
            best_kw = dict(trial_kw)
    t1 = time.time() - t0
    stage_results["stage1_coarse_spatial"] = {
        "nll": round(best_nll_1, 1),
        "params": {"mask_dx": best_kw["mask_dx"], "mask_dy": best_kw["mask_dy"]},
        "grid_size": len(grid_1),
        "time": round(t1, 1),
    }
    print(f"      Best: dx={best_kw['mask_dx']:.0f}, dy={best_kw['mask_dy']:.0f}, "
          f"NLL={best_nll_1:.1f} ({t1:.1f}s)")

    # ── Stage 2: Refine spatial (fractional dx,dy + theta) ───────────────
    print(f"    [{scene_name}] Stage 2: Refine spatial + theta ...")
    t0 = time.time()
    dx_center, dy_center = best_kw["mask_dx"], best_kw["mask_dy"]
    grid_2 = _make_grid_3d(
        best_kw,
        "mask_dx", np.arange(dx_center - 1, dx_center + 1.25, 0.25),
        "mask_dy", np.arange(dy_center - 1, dy_center + 1.25, 0.25),
        "mask_theta", np.arange(-0.6, 0.65, 0.1),
    )
    best_nll_2 = np.inf
    for trial_kw in grid_2:
        nll = _nll_for_params(trial_kw)
        if nll < best_nll_2:
            best_nll_2 = nll
            best_kw = dict(trial_kw)
    t2 = time.time() - t0
    stage_results["stage2_refine_spatial"] = {
        "nll": round(best_nll_2, 1),
        "params": {
            "mask_dx": round(best_kw["mask_dx"], 2),
            "mask_dy": round(best_kw["mask_dy"], 2),
            "mask_theta": round(best_kw["mask_theta"], 2),
        },
        "grid_size": len(grid_2),
        "time": round(t2, 1),
    }
    print(f"      Best: dx={best_kw['mask_dx']:.2f}, dy={best_kw['mask_dy']:.2f}, "
          f"theta={best_kw['mask_theta']:.2f}°, NLL={best_nll_2:.1f} ({t2:.1f}s)")

    # ── Stage 3: Temporal (clock_offset + duty_cycle) ────────────────────
    print(f"    [{scene_name}] Stage 3: Temporal (clock_offset, duty_cycle) ...")
    t0 = time.time()
    grid_3 = _make_grid_2d(
        best_kw,
        "clock_offset", np.arange(-0.3, 0.35, 0.05),
        "duty_cycle", np.arange(0.80, 1.02, 0.02),
    )
    best_nll_3 = np.inf
    for trial_kw in grid_3:
        nll = _nll_for_params(trial_kw)
        if nll < best_nll_3:
            best_nll_3 = nll
            best_kw = dict(trial_kw)
    t3 = time.time() - t0
    stage_results["stage3_temporal"] = {
        "nll": round(best_nll_3, 1),
        "params": {
            "clock_offset": round(best_kw["clock_offset"], 3),
            "duty_cycle": round(best_kw["duty_cycle"], 3),
        },
        "grid_size": len(grid_3),
        "time": round(t3, 1),
    }
    print(f"      Best: clock={best_kw['clock_offset']:.3f}, duty={best_kw['duty_cycle']:.3f}, "
          f"NLL={best_nll_3:.1f} ({t3:.1f}s)")

    # ── Stage 4: Sensor (gain + offset) ──────────────────────────────────
    print(f"    [{scene_name}] Stage 4: Sensor (gain, offset) ...")
    t0 = time.time()
    grid_4 = _make_grid_2d(
        best_kw,
        "gain", np.arange(0.90, 1.12, 0.02),
        "offset", np.arange(-0.02, 0.025, 0.005),
    )
    best_nll_4 = np.inf
    for trial_kw in grid_4:
        nll = _nll_for_params(trial_kw)
        if nll < best_nll_4:
            best_nll_4 = nll
            best_kw = dict(trial_kw)
    t4 = time.time() - t0
    stage_results["stage4_sensor"] = {
        "nll": round(best_nll_4, 1),
        "params": {
            "gain": round(best_kw["gain"], 3),
            "offset": round(best_kw["offset"], 4),
        },
        "grid_size": len(grid_4),
        "time": round(t4, 1),
    }
    print(f"      Best: gain={best_kw['gain']:.3f}, offset={best_kw['offset']:.4f}, "
          f"NLL={best_nll_4:.1f} ({t4:.1f}s)")

    nll_after = best_nll_4
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    # ── Reconstruct: uncorrected and corrected ───────────────────────────
    print(f"    [{scene_name}] Reconstructing (uncorrected + corrected) ...")

    # Uncorrected: use nominal mask on perturbed measurement
    A_nom = lambda x: A_(x, mask)
    At_nom = lambda y: At_(y, mask)
    v_uncorr, t_uncorr, psnr_uncorr, ssim_uncorr, _ = admmdenoise_cacti(
        y_pert, mask, A_nom, At_nom,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=n_coded, MAXB=MAXB,
        _lambda=1, accelerate=True,
        denoiser="tv", iter_max=40, tv_weight=0.3, tv_iter_max=5,
    )

    # Corrected: warp mask using best-fit parameters, undo gain/offset
    mask_corrected = warp_mask_3d(mask,
                                   best_kw["mask_dx"], best_kw["mask_dy"],
                                   best_kw["mask_theta"],
                                   best_kw.get("mask_blur_sigma", 0.0))
    # Undo sensor gain/offset from measurement
    y_corr_input = y_pert.copy()
    if abs(best_kw["gain"] - 1.0) > 1e-6 or abs(best_kw["offset"]) > 1e-6:
        y_corr_input = (y_corr_input - best_kw["offset"] * MAXB) / best_kw["gain"]

    A_corr = lambda x: A_(x, mask_corrected)
    At_corr = lambda y: At_(y, mask_corrected)
    v_corr, t_corr, psnr_corr, ssim_corr, _ = admmdenoise_cacti(
        y_corr_input, mask_corrected, A_corr, At_corr,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=n_coded, MAXB=MAXB,
        _lambda=1, accelerate=True,
        denoiser="tv", iter_max=40, tv_weight=0.3, tv_iter_max=5,
    )

    result = {
        "scene": scene_name,
        "nll_before": round(nll_before, 1),
        "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "psnr_uncorrected": round(mean(psnr_uncorr), 2),
        "ssim_uncorrected": round(mean(ssim_uncorr), 4),
        "psnr_corrected": round(mean(psnr_corr), 2),
        "ssim_corrected": round(mean(ssim_corr), 4),
        "psnr_delta": round(mean(psnr_corr) - mean(psnr_uncorr), 2),
        "ssim_delta": round(mean(ssim_corr) - mean(ssim_uncorr), 4),
        "injected_params": {k: round(v, 4) for k, v in W2_INJECTED.items()},
        "recovered_params": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in best_kw.items()},
        "stages": stage_results,
        "noise_model": f"Poisson(peak={W2_NOISE['peak_photons']:.0f}) + "
                        f"Read(sigma={W2_NOISE['read_sigma']:.1f}) + "
                        f"Quant({W2_NOISE['bit_depth']}bit)",
        "mask_nominal_hash": sha256_hex16(mask),
        "mask_corrected_hash": sha256_hex16(mask_corrected),
    }

    print(f"    [{scene_name}] NLL: {nll_before:.1f} → {nll_after:.1f} "
          f"({nll_decrease_pct:.1f}% decrease)")
    print(f"    [{scene_name}] PSNR: {result['psnr_uncorrected']:.2f} → "
          f"{result['psnr_corrected']:.2f} ({result['psnr_delta']:+.2f} dB)")

    return result


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CACTI Real-Data Benchmark")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                        help="Scenes to benchmark (default: all 6)")
    args = parser.parse_args()
    scenes = args.scenes

    results = {}
    print("=" * 70)
    print(f"CACTI Real-Data Benchmark — {len(scenes)} scene(s)")
    print("=" * 70)

    # ── Create RunBundle ─────────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "cacti_benchmark")
    rb_name = os.path.basename(rb_dir)
    art_dir = os.path.join(rb_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    print(f"RunBundle: {rb_name}")

    # ── Load deep models once ────────────────────────────────────────────
    print("\n[0] Loading deep models (one-time) ...")
    t0_load = time.time()

    print("    Loading FFDNet ...")
    ffdnet_model = load_ffdnet_model()

    print("    Loading ELP-Unfolding (6.4 GB checkpoint) ...")
    elp_model = load_elp_model()

    print("    Loading EfficientSCI ...")
    esci_model, esci_mask_t, Phi, Phi_s = load_esci_model()

    print(f"    All models loaded in {time.time() - t0_load:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # W1: 4-Solver Comparison — all scenes
    # ══════════════════════════════════════════════════════════════════════
    all_scene_results = {}

    for si, scene in enumerate(scenes):
        print(f"\n{'=' * 70}")
        print(f"[{si+1}/{len(scenes)}] Scene: {scene}")
        print("=" * 70)

        meas, mask, orig = load_scene(scene)
        n_total = orig.shape[2]
        n_coded = meas.shape[2]
        print(f"  meas: {meas.shape}, mask: {mask.shape}, orig: {orig.shape}")

        scene_dir = os.path.join(art_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)

        solver_results = {}

        # GAP-TV
        print(f"  [1/4] GAP-TV ...")
        v_tv, t_tv, _, _ = run_gap_tv(meas, mask, orig, n_coded)
        m_tv = compute_metrics_allframes(orig, v_tv)
        solver_results["gap_tv"] = {**m_tv, "wall_time": round(t_tv, 2)}
        save_array(os.path.join(scene_dir, "x_hat_gap_tv.npy"), v_tv)
        print(f"    PSNR={m_tv['mean_psnr']:.2f} dB, SSIM={m_tv['mean_ssim']:.4f}, time={t_tv:.1f}s")

        # PnP-FFDNet
        print(f"  [2/4] PnP-FFDNet ...")
        v_ffd, t_ffd, _, _ = run_pnp_ffdnet(meas, mask, orig, n_coded, ffdnet_model)
        m_ffd = compute_metrics_allframes(orig, v_ffd)
        solver_results["pnp_ffdnet"] = {**m_ffd, "wall_time": round(t_ffd, 2)}
        save_array(os.path.join(scene_dir, "x_hat_pnp_ffdnet.npy"), v_ffd)
        print(f"    PSNR={m_ffd['mean_psnr']:.2f} dB, SSIM={m_ffd['mean_ssim']:.4f}, time={t_ffd:.1f}s")

        # ELP-Unfolding
        print(f"  [3/4] ELP-Unfolding ...")
        v_elp, t_elp = run_elp_unfolding(orig, mask, elp_model)
        m_elp = compute_metrics_allframes(orig, v_elp)
        solver_results["elp_unfolding"] = {**m_elp, "wall_time": round(t_elp, 2)}
        save_array(os.path.join(scene_dir, "x_hat_elp_unfolding.npy"), v_elp)
        print(f"    PSNR={m_elp['mean_psnr']:.2f} dB, SSIM={m_elp['mean_ssim']:.4f}, time={t_elp:.1f}s")

        # EfficientSCI
        print(f"  [4/4] EfficientSCI ...")
        v_esci, t_esci = run_efficientsci(orig, esci_model, esci_mask_t, Phi, Phi_s)
        m_esci = compute_metrics_allframes(orig, v_esci)
        solver_results["efficientsci"] = {**m_esci, "wall_time": round(t_esci, 2)}
        save_array(os.path.join(scene_dir, "x_hat_efficientsci.npy"), v_esci)
        print(f"    PSNR={m_esci['mean_psnr']:.2f} dB, SSIM={m_esci['mean_ssim']:.4f}, time={t_esci:.1f}s")

        all_scene_results[scene] = solver_results

        # Per-scene summary
        print(f"\n  {'Solver':<20} {'PSNR':>8} {'SSIM':>8} {'Time':>8}")
        print(f"  {'-'*50}")
        for sname, sr in solver_results.items():
            print(f"  {sname:<20} {sr['mean_psnr']:>8.2f} {sr['mean_ssim']:>8.4f} {sr['wall_time']:>8.1f}")

    results["w1"] = all_scene_results

    # ── Compute average across all scenes ────────────────────────────────
    solver_names = ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficientsci"]
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
    # W2: Multi-stage realistic mismatch correction (3 scenes)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"W2: Multi-stage mismatch correction ({len(W2_SCENES)} scenes, GAP-TV)")
    print(f"  Injected: {W2_INJECTED}")
    print(f"  Noise: Poisson(peak={W2_NOISE['peak_photons']:.0f}) + "
          f"Read(sigma={W2_NOISE['read_sigma']:.1f}) + "
          f"Quant({W2_NOISE['bit_depth']}bit)")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    w2_per_scene = {}

    for w2_scene in W2_SCENES:
        print(f"\n  W2 scene: {w2_scene}")
        meas_w2, mask_w2, orig_w2 = load_scene(w2_scene)
        w2_result = run_w2_multistage(orig_w2, mask_w2, w2_scene, rng)
        w2_per_scene[w2_scene] = w2_result

    # Compute median metrics across scenes
    nll_decreases = [w2_per_scene[s]["nll_decrease_pct"] for s in W2_SCENES]
    psnr_deltas = [w2_per_scene[s]["psnr_delta"] for s in W2_SCENES]
    ssim_deltas = [w2_per_scene[s]["ssim_delta"] for s in W2_SCENES]
    median_nll_dec = round(float(np.median(nll_decreases)), 1)
    median_psnr_delta = round(float(np.median(psnr_deltas)), 2)
    median_ssim_delta = round(float(np.median(ssim_deltas)), 4)

    w2_report = {
        "per_scene": w2_per_scene,
        "median_nll_decrease_pct": median_nll_dec,
        "median_psnr_delta": median_psnr_delta,
        "median_ssim_delta": median_ssim_delta,
        "a_definition": "callable",
        "a_extraction_method": "provided",
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "mismatch_description": (
            "Realistic multi-parameter mismatch: spatial (dx=1.5, dy=1.0, theta=0.3°), "
            "temporal (clock_offset=0.08, duty_cycle=0.92), "
            "sensor (gain=1.05, offset=0.005)"),
        "correction_family": "Pre+PreTemporal+Post",
        "correction_method": "UPWMI_multistage_search",
        "noise_model": f"Poisson(peak={W2_NOISE['peak_photons']:.0f}) + "
                        f"Read(sigma={W2_NOISE['read_sigma']:.1f}) + "
                        f"Quant({W2_NOISE['bit_depth']}bit)",
        "n_stages": 4,
        "stage_names": [
            "coarse_spatial (dx,dy integer)",
            "refine_spatial (dx,dy,theta fractional)",
            "temporal (clock_offset, duty_cycle)",
            "sensor (gain, offset)",
        ],
    }
    results["w2"] = w2_report

    # Save operator metadata (use first scene for representative hash)
    first_scene_result = w2_per_scene[W2_SCENES[0]]
    save_operator_meta(rb_dir, {
        "a_definition": "callable",
        "a_extraction_method": "provided",
        "a_sha256": first_scene_result["mask_nominal_hash"],
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "injected_params": W2_INJECTED,
        "correction_family": "Pre+PreTemporal+Post",
        "correction_method": "UPWMI_multistage_search",
        "noise_model": w2_report["noise_model"],
        "n_scenes": len(W2_SCENES),
        "scenes": W2_SCENES,
        "per_scene_nll_decrease": {s: w2_per_scene[s]["nll_decrease_pct"]
                                    for s in W2_SCENES},
        "per_scene_psnr_delta": {s: w2_per_scene[s]["psnr_delta"]
                                  for s in W2_SCENES},
        "median_nll_decrease_pct": median_nll_dec,
        "median_psnr_delta": median_psnr_delta,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Print W2 summary table
    print(f"\n{'=' * 70}")
    print("W2 Summary — Multi-stage Correction")
    print("=" * 70)
    hdr = f"{'Scene':<12} {'NLL decr%':>10} {'PSNR uncorr':>12} {'PSNR corr':>10} {'ΔPSNR':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in W2_SCENES:
        r = w2_per_scene[s]
        print(f"{s:<12} {r['nll_decrease_pct']:>9.1f}% "
              f"{r['psnr_uncorrected']:>11.2f} {r['psnr_corrected']:>10.2f} "
              f"{r['psnr_delta']:>+6.2f}")
    print("-" * len(hdr))
    print(f"{'MEDIAN':<12} {median_nll_dec:>9.1f}% "
          f"{'':>11} {'':>10} {median_psnr_delta:>+6.2f}")

    # ── Trace (from kobe32 — full pipeline) ─────────────────────────────
    meas_tr, mask_tr, orig_tr = load_scene("kobe32")
    n_coded_tr = meas_tr.shape[2]
    block_tr = orig_tr[:, :, 0:8]
    trace_rng = np.random.RandomState(SEED)

    # Stage-by-stage trace through the physically realistic pipeline
    trace = {}
    trace["00_input_x"] = block_tr.copy()

    # Objective lens (throughput + PSF)
    trace["01_objective"] = apply_psf_2d(
        block_tr[:, :, 0] * 0.95, sigma=0.0).reshape(256, 256, 1).repeat(8, axis=2)

    # Temporal coded aperture (mask modulation)
    mask_warped = warp_mask_3d(mask_tr, dx=1.5, dy=1.0, theta=0.3)
    trace["02_coded_aperture"] = (block_tr * mask_warped).astype(np.float32)

    # Shutter integration (temporal mixing + duty cycle)
    y_integrated = apply_temporal_mixing(block_tr, mask_warped,
                                          clock_offset=0.08, duty_cycle=0.92,
                                          rng=trace_rng)
    trace["03_shutter_integrated"] = y_integrated.copy()

    # Detector (gain + offset)
    y_detector = 1.05 * y_integrated + 0.005 * MAXB
    trace["04_detector"] = y_detector.copy()

    # Noise (Poisson + Read + Quantization)
    y_noisy = apply_poisson_read_quant_noise(y_detector, peak_photons=10000.0,
                                              read_sigma=5.0, bit_depth=12,
                                              rng=np.random.RandomState(SEED))
    trace["05_noisy_y"] = y_noisy.copy()

    # Reconstruction (GAP-TV on nominal)
    v_tv_k, _, _, _ = run_gap_tv(meas_tr, mask_tr, orig_tr, n_coded_tr)
    trace["06_recon_gaptv"] = v_tv_k[:, :, 0:8].copy()
    save_trace(rb_dir, trace)

    results["trace"] = []
    for i, key in enumerate(sorted(trace.keys())):
        arr = trace[key]
        results["trace"].append({
            "stage": i, "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape), "dtype": str(arr.dtype),
            "range_min": round(float(arr.min()), 2),
            "range_max": round(float(arr.max()), 2),
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
    except ImportError: pass
    try:
        import torch; results["env"]["torch_version"] = torch.__version__
    except ImportError: pass

    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name
    results["scenes"] = scenes

    results_path = os.path.join(rb_dir, "cacti_benchmark_results.json")
    save_json(results_path, results)

    # ══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — All Scenes")
    print("=" * 70)

    # Per-scene table for each solver
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
    # Average row
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

    print(f"\nW2 ({len(W2_SCENES)} scenes, multi-stage correction):")
    print(f"  Median NLL decrease: {median_nll_dec:.1f}%")
    print(f"  Median PSNR delta:   {median_psnr_delta:+.2f} dB")
    for s in W2_SCENES:
        r = w2_per_scene[s]
        print(f"  {s}: NLL {r['nll_decrease_pct']:.1f}%, PSNR {r['psnr_delta']:+.2f} dB")
    print(f"\nRunBundle: runs/{rb_name}")
    print(f"Results:   {results_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
