#!/usr/bin/env python3
"""CACTI Competition — Modal GPU Runner.

Runs all 4 InverseNet reconstruction methods on Modal GPU:
  1. GAP-TV           (classical baseline, CPU-friendly)
  2. PnP-FFDNet/DnCNN (plug-and-play deep denoiser)
  3. ELP-Unfolding    (deep unfolded ADMM, ECCV 2022, 565M params)
  4. EfficientSCI     (end-to-end learned, CVPR 2023)

Modes:
  --mode ideal       Scenario I:   clean y from ground truth, nominal mask (validates solver)
  --mode oracle      Scenario III: noisy y + true_spec correction (upper bound with noise)
  --mode competition Full pipeline: grid-search calibration + reconstruction
  --mode baseline    No mismatch correction, use nominal mask

Setup (one-time):
  python modal_runner.py --upload

Run all methods:
  modal run modal_runner.py

Run specific method:
  modal run modal_runner.py --method gap_tv --mode oracle
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import modal

# ── Modal App ────────────────────────────────────────────────────────────────

app = modal.App("cacti-competition")
vol = modal.Volume.from_name("cacti-competition-data", create_if_missing=True)

# Paths on the local machine
_LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Physics_World_Model
_CKPT_DIR = _LOCAL_ROOT / "checkpoint"
_DATA_DIR = _LOCAL_ROOT / "datasets" / "benchmark" / "cacti"
_PWM_CORE = _LOCAL_ROOT / "packages" / "pwm_core"
_ELP_REPO = Path("/home/spiritai/ELP-Unfolding-master")
_ESCI_REPO = Path("/home/spiritai/EfficientSCI-main")
_FFDNET_PKG = Path("/home/spiritai/PnP-SCI_python-master/packages")
_HISVIT_REPO = Path("/home/spiritai/HiSViT")

# ── Container Image ──────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "numpy<2",
        "scipy",
        "h5py",
        "scikit-image",
        "einops",
        "six",
    )
    .add_local_dir(str(_PWM_CORE), "/root/packages/pwm_core", ignore=[".git", "__pycache__", ".pytest_cache"])
    .add_local_dir(str(_ELP_REPO), "/root/repos/ELP-Unfolding-master", ignore=[".git", "__pycache__", "fig"])
    .add_local_dir(str(_ESCI_REPO), "/root/repos/EfficientSCI-main", ignore=[".git", "__pycache__", "test_datasets", "docs"])
    .add_local_dir(str(_FFDNET_PKG), "/root/repos/PnP-SCI/packages", ignore=[".git", "__pycache__"])
    .add_local_dir(str(_HISVIT_REPO), "/root/repos/HiSViT", ignore=[".git", "__pycache__", "checkpoint", "data"])
)

# ── Volume paths (inside container) ─────────────────────────────────────────
VOL_ROOT = "/vol"
VOL_CKPT = f"{VOL_ROOT}/checkpoint"
VOL_DATA = f"{VOL_ROOT}/data"
VOL_RESULTS = f"{VOL_ROOT}/results"

# ── Constants ────────────────────────────────────────────────────────────────
# DL methods require binary {0,1} masks (they were trained on binary masks)
DL_METHODS = {"efficient_sci", "elp_unfolding", "hisvit9", "hisvit13"}


# ============================================================================
# Upload function (local entrypoint)
# ============================================================================

def upload_to_volume():
    """Upload checkpoints and data to the Modal volume (one-time setup)."""
    import subprocess

    vol_name = "cacti-competition-data"

    uploads = [
        # Checkpoints
        (str(_CKPT_DIR / "DnCNN"), "checkpoint/DnCNN"),
        (str(_CKPT_DIR / "PnP-SCI"), "checkpoint/PnP-SCI"),
        (str(_CKPT_DIR / "EfficientSCI"), "checkpoint/EfficientSCI"),
        (str(_CKPT_DIR / "HiSViT"), "checkpoint/HiSViT"),
        # Data
        (str(_DATA_DIR / "public"), "data/public"),
    ]

    # ELP-Unfolding (6.8 GB — upload separately)
    elp_ckpt = str(_CKPT_DIR / "ELP-Unfolding" / "ckptall.pth")
    if Path(elp_ckpt).exists():
        uploads.append((elp_ckpt, "checkpoint/ELP-Unfolding/ckptall.pth"))

    # Dev and Hidden data
    for tier in ["dev", "hidden"]:
        tier_data = _DATA_DIR / tier
        if tier_data.exists():
            uploads.append((str(tier_data), f"data/{tier}"))

    for local, remote in uploads:
        print(f"Uploading {local} -> {vol_name}:/{remote}")
        cmd = ["modal", "volume", "put", vol_name, local, remote]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
        else:
            print(f"  OK")


# ============================================================================
# Core logic (runs inside the Modal container on GPU)
# ============================================================================

import numpy as np


def _setup_paths():
    """Set up Python paths for importing model code inside the container."""
    sys.path.insert(0, "/root/packages/pwm_core")
    sys.path.insert(0, "/root/repos/ELP-Unfolding-master")
    sys.path.insert(0, "/root/repos/EfficientSCI-main")
    sys.path.insert(0, "/root/repos/PnP-SCI/packages")
    sys.path.insert(0, "/root/repos/HiSViT")


def _patch_solver_paths():
    """Patch the hardcoded checkpoint paths in cacti_solvers to use the Volume."""
    import pwm_core.recon.cacti_solvers as cs
    cs._ELP_REPO = "/root/repos/ELP-Unfolding-master"
    cs._ESCI_REPO = "/root/repos/EfficientSCI-main"
    cs._FFDNET_PKG = "/root/repos/PnP-SCI/packages"
    cs._ELP_CKPT = f"{VOL_CKPT}/ELP-Unfolding/ckptall.pth"
    cs._ESCI_CKPT = f"{VOL_CKPT}/EfficientSCI/efficientsci_base.pth"
    cs._FFDNET_WEIGHTS = f"{VOL_CKPT}/PnP-SCI/ffdnet/net_gray.pth"
    cs._DNCNN_WEIGHTS = f"{VOL_CKPT}/DnCNN/dncnn_25.pth"
    cs._HISVIT_REPO = "/root/repos/HiSViT"
    cs._HISVIT9_CKPT = f"{VOL_CKPT}/HiSViT/hisvit9_gray.pth"
    cs._HISVIT13_CKPT = f"{VOL_CKPT}/HiSViT/hisvit13_gray.pth"


# ── Mismatch / Forward Model helpers ────────────────────────────────────────

def apply_mismatch(mask, dx, dy, rot, blur):
    """Apply spatial mismatch to mask — matches build_cacti_package.apply_cacti_mismatch."""
    from scipy.ndimage import shift, rotate, gaussian_filter
    T = mask.shape[2]
    mm = mask.copy()
    for t in range(T):
        f = mm[:, :, t]
        f = shift(f, [dy, dx], order=1, mode="constant")
        if abs(rot) > 1e-6:
            f = rotate(f, rot, reshape=False, order=1, mode="constant")
        if blur > 0:
            f = gaussian_filter(f, sigma=blur)
        mm[:, :, t] = f
    return mm


def forward_model(x, mask, gain=1.0, offset=0.0):
    return gain * np.sum(mask * x, axis=2) + offset


def estimate_gain_offset(y, mask, x_hat):
    """Estimate gain and offset from measurement, mask, and reconstruction."""
    Phi_x = np.sum(mask * x_hat, axis=2).ravel()
    y_flat = y.ravel()
    denom = np.dot(Phi_x, Phi_x)
    if denom < 1e-10:
        return 1.0, 0.0
    gain = np.dot(y_flat, Phi_x) / denom
    offset = float(np.mean(y_flat - gain * Phi_x))
    gain = float(np.dot(y_flat - offset, Phi_x) / denom)
    return gain, offset


def binarize_mask(mask):
    """Binarize mask for DL methods — critical for EfficientSCI/ELP-Unfolding."""
    return (mask > 0.5).astype(np.float32)


# ── GAP-TV fast inner loop ─────────────────────────────────────────────────

def gap_tv_fast(y, mask, n_iter=20, tv_weight=0.15):
    from pwm_core.recon.cacti_solvers import _gap_denoise_core
    return _gap_denoise_core(y, mask, max_iter=n_iter, tv_weight=tv_weight, tv_iter=3)


# ── Grid Search + L-BFGS-B Calibration (InverseNet Scenario IV approach) ───

def grid_search_spec(y, mask_nominal, spec_ranges, inner_iters=20, tv_weight=0.15):
    """Calibrate mismatch parameters using grid search + L-BFGS-B refinement.

    Follows the InverseNet Scenario IV approach:
    1. Coarse grid search over (dx, dy) — most important parameters
    2. L-BFGS-B refinement over all 4 params near best grid point
    3. Analytical gain/offset estimation
    """
    from scipy.optimize import minimize

    range_map = {s["name"]: (s["min"], s["max"]) for s in spec_ranges}
    dx_range = range_map.get("mask_dx", (0.0, 1.0))
    dy_range = range_map.get("mask_dy", (0.0, 1.0))
    rot_range = range_map.get("mask_rotation", (0.0, 0.5))
    blur_range = range_map.get("mask_blur", (0.0, 0.5))

    # Phase 1: Coarse grid search over (dx, dy), fix rot=0, blur=0
    # Step 0.1 gives good coverage of typical [0.2, 0.8] and [0.1, 0.5] ranges
    dx_grid = np.arange(dx_range[0], dx_range[1] + 0.05, 0.1)
    dy_grid = np.arange(dy_range[0], dy_range[1] + 0.05, 0.1)

    print(f"  Grid search: {len(dx_grid)}x{len(dy_grid)} = {len(dx_grid)*len(dy_grid)} points")
    print(f"    dx in [{dx_range[0]:.1f}, {dx_range[1]:.1f}], dy in [{dy_range[0]:.1f}, {dy_range[1]:.1f}]")

    t0 = time.time()
    best_residual = np.inf
    best_dx, best_dy = (dx_range[0] + dx_range[1]) / 2, (dy_range[0] + dy_range[1]) / 2

    for dx_test in dx_grid:
        for dy_test in dy_grid:
            mask_test = apply_mismatch(mask_nominal, dx_test, dy_test, 0, 0)
            x_hat = gap_tv_fast(y, mask_test, n_iter=inner_iters, tv_weight=tv_weight)
            # Estimate gain/offset and compute corrected residual
            gain, offset = estimate_gain_offset(y, mask_test, x_hat)
            y_pred = gain * np.sum(mask_test * x_hat, axis=2) + offset
            residual = float(np.sum((y - y_pred) ** 2))
            if residual < best_residual:
                best_residual = residual
                best_dx, best_dy = dx_test, dy_test

    grid_time = time.time() - t0
    print(f"  Grid best: dx={best_dx:.2f} dy={best_dy:.2f} res={best_residual:.2f} ({grid_time:.1f}s)")

    # Phase 2: L-BFGS-B refinement around best grid point
    # Search all 4 params with tight bounds near grid optimum
    dx_lo = max(dx_range[0], best_dx - 0.15)
    dx_hi = min(dx_range[1], best_dx + 0.15)
    dy_lo = max(dy_range[0], best_dy - 0.15)
    dy_hi = min(dy_range[1], best_dy + 0.15)

    bounds = [(dx_lo, dx_hi), (dy_lo, dy_hi), rot_range, blur_range]
    x0 = np.array([best_dx, best_dy, (rot_range[0] + rot_range[1]) / 2,
                    (blur_range[0] + blur_range[1]) / 2])

    eval_count = [0]

    def objective(params):
        eval_count[0] += 1
        dx, dy, rot, blur = params
        mask_m = apply_mismatch(mask_nominal, dx, dy, rot, blur)
        x_hat = gap_tv_fast(y, mask_m, n_iter=inner_iters, tv_weight=tv_weight)
        gain, offset = estimate_gain_offset(y, mask_m, x_hat)
        y_pred = gain * np.sum(mask_m * x_hat, axis=2) + offset
        res = float(np.sum((y - y_pred) ** 2))
        if eval_count[0] % 5 == 0:
            print(f"    refine {eval_count[0]}: [{dx:.4f}, {dy:.4f}, "
                  f"{rot:.4f}, {blur:.4f}] res={res:.2f}")
        return res

    t1 = time.time()
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 30, "ftol": 1e-8, "gtol": 1e-6})
    refine_time = time.time() - t1

    dx, dy, rot, blur = result.x
    print(f"  Refined: dx={dx:.4f} dy={dy:.4f} rot={rot:.4f} blur={blur:.4f} "
          f"({result.nfev} evals, {refine_time:.1f}s)")

    # Phase 3: Estimate gain/offset from final calibrated mask
    mask_cal = apply_mismatch(mask_nominal, dx, dy, rot, blur)
    x_hat = gap_tv_fast(y, mask_cal, n_iter=inner_iters, tv_weight=tv_weight)
    gain, offset = estimate_gain_offset(y, mask_cal, x_hat)

    total_time = time.time() - t0
    print(f"  Total calibration: {total_time:.1f}s, gain={gain:.4f} offset={offset:.6f}")

    return {
        "mask_dx": round(float(dx), 6),
        "mask_dy": round(float(dy), 6),
        "mask_rotation": round(float(rot), 6),
        "mask_blur": round(float(blur), 6),
        "clock_offset": 0.0,
        "gain_drift": round(float(gain), 6),
        "offset_drift": round(float(offset), 6),
    }


# ── Scoring ──────────────────────────────────────────────────────────────────

def compute_psnr(x_true, x_hat, max_val=1.0):
    mse = float(np.mean((x_true.astype(np.float64) - x_hat.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 60.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x_true, x_hat):
    from scipy.signal import fftconvolve
    def _ssim_2d(a, b, win=7):
        a = np.clip(a, 0, 1).astype(np.float64)
        b = np.clip(b, 0, 1).astype(np.float64)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w = np.ones((win, win), dtype=np.float64) / (win * win)
        mu_a = fftconvolve(a, w, mode="same")
        mu_b = fftconvolve(b, w, mode="same")
        sig_a2 = fftconvolve(a * a, w, mode="same") - mu_a ** 2
        sig_b2 = fftconvolve(b * b, w, mode="same") - mu_b ** 2
        sig_ab = fftconvolve(a * b, w, mode="same") - mu_a * mu_b
        num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
        den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a2 + sig_b2 + C2)
        return float(np.mean(num / den))
    T = x_true.shape[2]
    return float(np.mean([_ssim_2d(x_true[:, :, t], x_hat[:, :, t]) for t in range(T)]))


def compute_consistency(y, x_hat, mask, gain, offset):
    y_pred = forward_model(x_hat, mask, gain, offset)
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-10:
        return 1.0
    return float(1.0 - np.linalg.norm(y - y_pred) / y_norm)


def compute_composite(psnr, ssim, consistency):
    psnr_norm = float(np.clip((psnr - 15) / 30, 0, 1))
    return 0.4 * psnr_norm + 0.4 * ssim + 0.2 * consistency


# ── Reconstruction helper ────────────────────────────────────────────────────

def reconstruct(y, mask, method, device, final_iters=100, binarize_dl=None):
    """Run reconstruction with proper mask handling for each method.

    Args:
        binarize_dl: If True, binarize masks for DL methods. If None (default),
            auto-binarize for DL methods since they expect {0,1} masks.
    """
    from pwm_core.recon.cacti_solvers import gap_tv_cacti, solve_cacti

    y_in = y.astype(np.float32)

    # DL methods (EfficientSCI, HiSViT, ELP) expect binary masks
    should_binarize = binarize_dl if binarize_dl is not None else (method in DL_METHODS)
    if should_binarize:
        mask_in = binarize_mask(mask)
    else:
        mask_in = mask.astype(np.float32)

    if method == "gap_tv":
        return gap_tv_cacti(y_in, mask_in, iterations=final_iters)
    else:
        return solve_cacti(y_in, mask_in, method=method, device=device)


# ============================================================================
# Modal GPU Function — Scenario I: Ideal (clean y, nominal mask)
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={VOL_ROOT: vol},
    timeout=7200,
    memory=32768,
)
def run_ideal(method: str, tier: str, final_iters: int = 100, t_filter: int = 0):
    """Scenario I: Ideal reconstruction.

    Creates clean measurement from ground truth + nominal mask, then reconstructs.
    This validates that the solver achieves reference PSNR (~27-35 dB).
    """
    import h5py

    _setup_paths()
    _patch_solver_paths()

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"SCENARIO I (IDEAL) — {method} on {tier} tier")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if t_filter > 0:
        print(f"T filter: only T={t_filter} samples")

    h5_path = f"{VOL_DATA}/{tier}/cacti_challenge_{tier}.h5"
    print(f"\nLoading: {h5_path}")

    all_scores = []

    with h5py.File(h5_path, "r") as fin:
        sample_keys = sorted(fin.keys())

        for sk in sample_keys:
            grp = fin[sk]
            mask = grp["H_ideal"][:].astype(np.float64)
            T = mask.shape[2]
            if t_filter > 0 and T != t_filter:
                continue
            x_true = grp["x_true"][:] if "x_true" in grp else None
            if x_true is None:
                print(f"  {sk}: no ground truth, skipping")
                continue

            # Create CLEAN measurement from ground truth + nominal mask (no mismatch, no noise)
            y_clean = np.sum(mask * x_true, axis=2)

            print(f"  {sk} (T={T}): y_range=[{y_clean.min():.3f}, {y_clean.max():.3f}]", end="")

            t0 = time.time()
            # In ideal mode, masks are binary — safe to binarize for DL methods
            x_hat = reconstruct(y_clean, mask, method, device, final_iters, binarize_dl=True)
            dt = time.time() - t0

            psnr = compute_psnr(x_true, x_hat, max_val=1.0)
            ssim_val = compute_ssim(x_true, x_hat)
            print(f"  PSNR={psnr:.2f} SSIM={ssim_val:.4f} ({dt:.1f}s)")
            all_scores.append({"sample": sk, "psnr": psnr, "ssim": ssim_val})

    if all_scores:
        avg_psnr = float(np.mean([s["psnr"] for s in all_scores]))
        avg_ssim = float(np.mean([s["ssim"] for s in all_scores]))
        print(f"\nSCENARIO I AVG: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f}")
        print(f"(Expected: GAP-TV ~27 dB, PnP ~30 dB, EfficientSCI ~35 dB)")
        return {"method": f"{method}_ideal", "tier": tier,
                "averages": {"psnr": avg_psnr, "ssim": avg_ssim},
                "scores": all_scores}

    return {"method": f"{method}_ideal", "tier": tier, "scores": []}


# ============================================================================
# Modal GPU Function — Scenario III: Oracle (noisy y + true_spec correction)
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={VOL_ROOT: vol},
    timeout=7200,
    memory=32768,
)
def run_oracle(method: str, tier: str, final_iters: int = 100, t_filter: int = 0):
    """Scenario III: Oracle mismatch correction.

    Uses the true_spec to correct the measurement and mask, then reconstructs.
    This gives the upper bound achievable with perfect calibration.
    """
    import h5py

    _setup_paths()
    _patch_solver_paths()

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"SCENARIO III (ORACLE) — {method} on {tier} tier")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if t_filter > 0:
        print(f"T filter: only T={t_filter} samples")

    h5_path = f"{VOL_DATA}/{tier}/cacti_challenge_{tier}.h5"
    print(f"\nLoading: {h5_path}")

    # Load true_spec from the first sample or from file
    true_spec = None

    all_scores = []

    with h5py.File(h5_path, "r") as fin:
        sample_keys = sorted(fin.keys())

        for sk in sample_keys:
            grp = fin[sk]
            mask = grp["H_ideal"][:].astype(np.float64)
            T = mask.shape[2]
            if t_filter > 0 and T != t_filter:
                continue
            y = grp["y"][:].astype(np.float64)
            x_true = grp["x_true"][:] if "x_true" in grp else None

            # Get true_spec from sample attributes
            if "true_spec" in grp.attrs:
                true_spec = json.loads(grp.attrs["true_spec"])
            if true_spec is None:
                print(f"  {sk}: no true_spec available, skipping")
                continue

            print(f"  {sk} (T={T}):", end="")

            # Apply oracle mismatch correction
            mask_corrected = apply_mismatch(
                mask, true_spec["mask_dx"], true_spec["mask_dy"],
                true_spec["mask_rotation"], true_spec["mask_blur"],
            )
            gain = true_spec["gain_drift"]
            offset = true_spec["offset_drift"]
            y_corrected = (y - offset) / max(abs(gain), 1e-6)

            t0 = time.time()
            x_hat = reconstruct(y_corrected, mask_corrected, method, device, final_iters)
            dt = time.time() - t0

            consistency = compute_consistency(y, x_hat, mask_corrected, gain, offset)

            if x_true is not None:
                psnr = compute_psnr(x_true, x_hat, max_val=1.0)
                ssim_val = compute_ssim(x_true, x_hat)
                composite = compute_composite(psnr, ssim_val, consistency)
                print(f"  PSNR={psnr:.2f} SSIM={ssim_val:.4f} Cons={consistency:.4f} "
                      f"Score={composite:.4f} ({dt:.1f}s)")
                all_scores.append({
                    "sample": sk, "psnr": psnr, "ssim": ssim_val,
                    "consistency": consistency, "composite": composite,
                })
            else:
                print(f"  Cons={consistency:.4f} ({dt:.1f}s)")

    if all_scores:
        avg = {k: float(np.mean([s[k] for s in all_scores]))
               for k in ("psnr", "ssim", "consistency", "composite")}
        print(f"\nSCENARIO III AVG: PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
              f"Cons={avg['consistency']:.4f} Score={avg['composite']:.4f}")
        return {"method": f"{method}_oracle", "tier": tier, "averages": avg, "scores": all_scores}

    return {"method": f"{method}_oracle", "tier": tier, "scores": []}


# ============================================================================
# Modal GPU Function — Competition mode (grid search calibration)
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={VOL_ROOT: vol},
    timeout=7200,
    memory=32768,
)
def run_method(method: str, tier: str, cal_samples: int = 3,
               inner_iters: int = 20, final_iters: int = 100,
               t_filter: int = 0):
    """Full competition pipeline: grid-search calibration + reconstruction.

    Args:
        t_filter: If > 0, only process samples with mask T == t_filter.
    """
    import h5py

    _setup_paths()
    _patch_solver_paths()

    from pwm_core.recon.cacti_solvers import SOLVERS

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"COMPETITION — {method} on {tier} tier")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if t_filter > 0:
        print(f"T filter: only T={t_filter} samples")

    # Check checkpoint existence
    from pwm_core.recon import cacti_solvers as cs
    for name, path in [("DnCNN", cs._DNCNN_WEIGHTS), ("FFDNet", cs._FFDNET_WEIGHTS),
                       ("ELP", cs._ELP_CKPT), ("EfficientSCI", cs._ESCI_CKPT)]:
        print(f"  {name}: {'OK' if os.path.isfile(path) else 'MISSING'} ({path})")

    # Load data
    h5_path = f"{VOL_DATA}/{tier}/cacti_challenge_{tier}.h5"
    print(f"\nLoading: {h5_path}")

    with h5py.File(h5_path, "r") as fin:
        sample_keys = sorted(fin.keys())
        print(f"Total samples in file: {len(sample_keys)}")

        samples = []
        for sk in sample_keys:
            grp = fin[sk]
            mask = grp["H_ideal"][:].astype(np.float64)
            T = mask.shape[2]
            if t_filter > 0 and T != t_filter:
                print(f"  Skipping {sk} (T={T}, filter requires T={t_filter})")
                continue
            samples.append({
                "key": sk,
                "y": grp["y"][:].astype(np.float64),
                "mask": mask,
                "x_true": grp["x_true"][:] if "x_true" in grp else None,
                "spec_ranges": json.loads(grp.attrs["spec_ranges"]),
                "metadata": json.loads(grp.attrs.get("metadata", "{}")),
                "true_spec": json.loads(grp.attrs["true_spec"]) if "true_spec" in grp.attrs else None,
            })

    n_samples = len(samples)
    print(f"Processing {n_samples} samples (after T filter)")

    if n_samples == 0:
        print("No samples match the filter. Exiting.")
        return {"method": method, "tier": tier, "n_samples": 0, "scores": []}

    # Step 1: Calibrate spec using grid search + L-BFGS-B
    print(f"\n--- Step 1: Spec Calibration (Grid Search + L-BFGS-B) ---")
    n_cal = min(cal_samples, n_samples)
    print(f"Calibrating on {n_cal} samples, applying to all {n_samples}")

    specs_found = []
    for i in range(n_cal):
        s = samples[i]
        print(f"\nCalibrating on {s['key']} (T={s['mask'].shape[2]}, "
              f"{s['mask'].shape[0]}x{s['mask'].shape[1]})...")
        spec_i = grid_search_spec(s["y"], s["mask"], s["spec_ranges"],
                                  inner_iters=inner_iters)
        specs_found.append(spec_i)
        print(f"  -> {spec_i}")

    avg_spec = {}
    for k in specs_found[0]:
        avg_spec[k] = round(float(np.mean([sp[k] for sp in specs_found])), 6)
    corrected_spec = avg_spec
    print(f"\nAveraged spec: {corrected_spec}")

    # Show true_spec comparison if available
    if samples[0].get("true_spec"):
        ts = samples[0]["true_spec"]
        print(f"True spec:     {ts}")
        errs = {k: f"{abs(ts.get(k, 0) - corrected_spec.get(k, 0)):.4f}"
                for k in ["mask_dx", "mask_dy", "mask_rotation",
                          "mask_blur", "gain_drift", "offset_drift"]}
        print(f"Spec errors:   {errs}")

    # Step 2: Reconstruct with corrected forward model
    print(f"\n--- Step 2: Reconstruct with {method} ---")
    output_dir = f"{VOL_RESULTS}/{tier}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/submission_{tier}_{method}.h5"

    all_scores = []

    with h5py.File(output_path, "w") as fout:
        fout.attrs.update({"variant": "cacti", "tier": tier, "method": method})

        for i, s in enumerate(samples):
            H, W, T = s["mask"].shape
            print(f"\n[{i+1}/{n_samples}] {s['key']} (T={T}, {H}x{W})")

            spec = corrected_spec
            mask_corrected = apply_mismatch(
                s["mask"], spec["mask_dx"], spec["mask_dy"],
                spec["mask_rotation"], spec["mask_blur"],
            )

            gain = spec["gain_drift"]
            offset = spec["offset_drift"]
            y_corrected = (s["y"] - offset) / max(abs(gain), 1e-6)

            t0 = time.time()
            x_hat = reconstruct(y_corrected, mask_corrected, method, device, final_iters)
            dt = time.time() - t0
            print(f"  Reconstruction: {dt:.1f}s")

            consistency = compute_consistency(s["y"], x_hat, mask_corrected, gain, offset)
            print(f"  Consistency: {consistency:.4f}")

            if s["x_true"] is not None:
                psnr = compute_psnr(s["x_true"], x_hat, max_val=1.0)
                ssim_val = compute_ssim(s["x_true"], x_hat)
                composite = compute_composite(psnr, ssim_val, consistency)
                print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}, Score: {composite:.4f}")
                all_scores.append({
                    "sample": s["key"], "psnr": psnr, "ssim": ssim_val,
                    "consistency": consistency, "composite": composite,
                })

            if s["true_spec"] is not None:
                ts = s["true_spec"]
                errs = {k: f"{abs(ts.get(k, 0) - spec.get(k, 0)):.4f}"
                        for k in ["mask_dx", "mask_dy", "mask_rotation",
                                  "mask_blur", "gain_drift", "offset_drift"]}
                print(f"  Spec errors: {errs}")

            grp = fout.create_group(s["key"])
            grp.create_dataset("x_hat", data=x_hat, compression="gzip", compression_opts=4)
            grp.attrs["corrected_spec"] = json.dumps(spec)
            grp.attrs["consistency"] = consistency

    vol.commit()

    summary = {"method": method, "tier": tier, "n_samples": n_samples,
               "corrected_spec": corrected_spec, "scores": all_scores}

    if all_scores:
        avg = {k: float(np.mean([s[k] for s in all_scores]))
               for k in ("psnr", "ssim", "consistency", "composite")}
        summary["averages"] = avg

        print(f"\n{'='*70}")
        print(f"SUMMARY — {method}")
        print(f"{'='*70}")
        print(f"{'Sample':<12} {'PSNR':>8} {'SSIM':>8} {'Consist':>12} {'Score':>12}")
        print("-" * 70)
        for sc in all_scores:
            print(f"{sc['sample']:<12} {sc['psnr']:>8.2f} {sc['ssim']:>8.4f} "
                  f"{sc['consistency']:>12.4f} {sc['composite']:>12.4f}")
        print("-" * 70)
        print(f"{'AVERAGE':<12} {avg['psnr']:>8.2f} {avg['ssim']:>8.4f} "
              f"{avg['consistency']:>12.4f} {avg['composite']:>12.4f}")

    print(f"\nOutput: {output_path}")
    return summary


# ============================================================================
# Modal GPU Function — Baseline (no mismatch correction)
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={VOL_ROOT: vol},
    timeout=7200,
    memory=32768,
)
def run_baseline(method: str, tier: str, final_iters: int = 100, t_filter: int = 0):
    """Baseline: reconstruct with nominal mask, NO mismatch correction.

    This corresponds to InverseNet Scenario II (corrupted measurement + ideal mask).
    """
    import h5py

    _setup_paths()
    _patch_solver_paths()

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"BASELINE (Scenario II) — {method} on {tier} tier (no correction)")
    print(f"{'='*70}")
    if t_filter > 0:
        print(f"T filter: only T={t_filter} samples")

    h5_path = f"{VOL_DATA}/{tier}/cacti_challenge_{tier}.h5"
    with h5py.File(h5_path, "r") as fin:
        sample_keys = sorted(fin.keys())
        all_scores = []

        filtered_keys = []
        for sk in sample_keys:
            T = fin[sk]["H_ideal"].shape[2]
            if t_filter > 0 and T != t_filter:
                continue
            filtered_keys.append(sk)
        print(f"Processing {len(filtered_keys)}/{len(sample_keys)} samples")

        for i, sk in enumerate(filtered_keys):
            grp = fin[sk]
            y = grp["y"][:].astype(np.float64)
            mask = grp["H_ideal"][:].astype(np.float64)
            x_true = grp["x_true"][:] if "x_true" in grp else None

            print(f"[{i+1}/{len(filtered_keys)}] {sk}", end="")

            t0 = time.time()
            x_hat = reconstruct(y, mask, method, device, final_iters)
            dt = time.time() - t0

            if x_true is not None:
                psnr = compute_psnr(x_true, x_hat, max_val=1.0)
                ssim_val = compute_ssim(x_true, x_hat)
                consistency = compute_consistency(y, x_hat, mask, 1.0, 0.0)
                composite = compute_composite(psnr, ssim_val, consistency)
                print(f"  PSNR={psnr:.2f} SSIM={ssim_val:.4f} Cons={consistency:.4f} "
                      f"Score={composite:.4f} ({dt:.1f}s)")
                all_scores.append({"sample": sk, "psnr": psnr, "ssim": ssim_val,
                                   "consistency": consistency, "composite": composite})
            else:
                print(f"  ({dt:.1f}s)")

    vol.commit()

    if all_scores:
        avg = {k: float(np.mean([s[k] for s in all_scores]))
               for k in ("psnr", "ssim", "consistency", "composite")}
        print(f"\nBASELINE AVG: PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
              f"Cons={avg['consistency']:.4f} Score={avg['composite']:.4f}")
        return {"method": f"{method}_baseline", "tier": tier, "averages": avg, "scores": all_scores}

    return {"method": f"{method}_baseline", "tier": tier, "scores": []}


# ============================================================================
# Local Entrypoint — orchestrate everything
# ============================================================================

@app.local_entrypoint()
def main(
    tier: str = "public",
    method: str = "all",
    mode: str = "competition",
    upload: bool = False,
    cal_samples: int = 3,
    t_filter: int = 0,
):
    """Run CACTI competition entry on Modal GPU.

    Args:
        tier: Which tier to process (public, dev)
        method: Which method to run (gap_tv, pnp_ffdnet, elp_unfolding, efficient_sci, all)
        mode: ideal (Scenario I), oracle (Scenario III), competition, baseline
        upload: Upload data/checkpoints to volume first
        cal_samples: Number of samples for shared spec calibration
        t_filter: Only process samples with T == t_filter (0 = all, 8 = T=8 only)
    """
    if upload:
        print("Uploading data to Modal volume...")
        upload_to_volume()
        print("Upload complete.")
        return

    all_methods = ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficient_sci", "hisvit9", "hisvit13"]
    methods = all_methods if method == "all" else [method]

    # Auto-set t_filter=8 for dev/hidden tiers if not specified
    effective_t = t_filter
    if effective_t == 0 and tier in ("dev", "hidden"):
        effective_t = 8
        print(f"Auto-setting T filter to 8 for {tier} tier (DL models require T=8)")

    print(f"\nMode: {mode}")
    print(f"Running {len(methods)} methods on {tier} tier: {methods}")
    if mode == "competition":
        print(f"Calibration samples: {cal_samples}")
    if effective_t > 0:
        print(f"T filter: only T={effective_t} samples")

    # Select the appropriate function for the mode
    mode_funcs = {
        "ideal": run_ideal,
        "oracle": run_oracle,
        "competition": run_method,
        "baseline": run_baseline,
    }
    func = mode_funcs.get(mode)
    if func is None:
        print(f"Unknown mode: {mode}. Choose from: {list(mode_funcs.keys())}")
        return

    # Launch all methods in parallel
    handles = []
    for m in methods:
        if mode == "competition":
            h = func.spawn(m, tier, cal_samples=cal_samples, t_filter=effective_t)
        elif mode in ("ideal", "oracle", "baseline"):
            h = func.spawn(m, tier, t_filter=effective_t)
        handles.append((m, h))

    # For competition mode, optionally also run baselines for comparison
    if mode == "competition":
        for m in methods:
            h = run_baseline.spawn(m, tier, t_filter=effective_t)
            handles.append((f"{m}_baseline", h))

    # Collect results
    results = {}
    for name, h in handles:
        print(f"\nWaiting for {name}...")
        result = h.get()
        results[name] = result
        if result and "averages" in result:
            avg = result["averages"]
            parts = [f"PSNR={avg.get('psnr', 0):.2f}"]
            if "ssim" in avg:
                parts.append(f"SSIM={avg['ssim']:.4f}")
            if "consistency" in avg:
                parts.append(f"Cons={avg['consistency']:.4f}")
            if "composite" in avg:
                parts.append(f"Score={avg['composite']:.4f}")
            print(f"  {name}: {' '.join(parts)}")

    # Final comparison table
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON — {tier} tier ({mode} mode)")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'PSNR':>8} {'SSIM':>8} {'Consist':>12} {'Score':>12}")
    print("-" * 80)
    for name in sorted(results.keys()):
        r = results[name]
        if r and "averages" in r:
            avg = r["averages"]
            print(f"{name:<30} {avg.get('psnr', 0):>8.2f} {avg.get('ssim', 0):>8.4f} "
                  f"{avg.get('consistency', 0):>12.4f} {avg.get('composite', 0):>12.4f}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload data to Modal volume")
    args = parser.parse_args()
    if args.upload:
        upload_to_volume()
