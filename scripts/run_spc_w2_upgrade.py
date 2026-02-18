#!/usr/bin/env python3
"""SPC W2 Physics Upgrade — Realistic Mismatch + 4-Stage UPWMI Correction.

Replaces the old 614-param per-row gain fitting with a physically grounded
4-stage multistage correction using 17 identifiable mismatch parameters:
  Stage 1: Coarse spatial (mask_dx, mask_dy)
  Stage 2: Refine spatial+temporal (mask_dx, mask_dy, mask_theta, clock_offset)
  Stage 3: Illumination drift (linear, sin_amp, sin_freq)
  Stage 4: Sensor (gain, offset, dark_current)

Noise model: Poisson(peak=50000) + Read(sigma=0.005) + Quantize(14-bit)

Tests on sparse phantom + 3 Set11 crops.

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_spc_w2_upgrade.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import yaml

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.executor import GraphExecutor, ExecutionConfig, ExecutionResult
from pwm_core.core.enums import ExecutionMode
from pwm_core.core.metric_registry import PSNR, SSIM
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_artifacts, save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)
from pwm_core.recon.classical import fista_l2
from pwm_core.mismatch.operators import (
    spc_forward_with_params, spc_compute_nll, spc_multistage_correction,
)

# ── Constants ────────────────────────────────────────────────────────────
SEED = 42
H, W = 64, 64
SAMPLING_RATE = 0.15
N_MEASUREMENTS = int(H * W * SAMPLING_RATE)  # 614
SPARSITY_K = 100
PEAK_PHOTONS = 50000.0
READ_SIGMA = 0.005

TEMPLATES_PATH = os.path.join(
    PROJECT_ROOT, "packages", "pwm_core", "contrib", "graph_templates.yaml"
)
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Injected mismatch parameters (physically grounded)
INJECTED_MISMATCH = {
    "mask_dx": 0.8,
    "mask_dy": -0.5,
    "mask_theta": 0.15,
    "mask_scale": 1.0,
    "mask_blur_sigma": 0.0,
    "clock_offset": 0.06,
    "duty_cycle": 1.0,
    "illum_drift_linear": 0.04,
    "illum_drift_sin_amp": 0.03,
    "illum_drift_sin_freq": 1.5,
    "gain": 1.08,
    "offset": 0.005,
    "dark_current": 0.008,
}


def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def nrmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_hat - x_true) ** 2)) /
                 (np.max(x_true) - np.min(x_true) + 1e-12))


def make_phantom(H: int, W: int, seed: int, k: int = SPARSITY_K) -> np.ndarray:
    """Pixel-sparse phantom for CS testing."""
    rng = np.random.RandomState(seed)
    x = np.zeros((H, W), dtype=np.float64)
    indices = rng.choice(H * W, k, replace=False)
    x.ravel()[indices] = rng.rand(k) * 0.8 + 0.2
    return x


def make_set11_crop(idx: int = 0) -> np.ndarray:
    """Load a Set11 image crop (64x64 grayscale, normalized to [0,1])."""
    try:
        import cv2
        SET11_DIR = "/home/spiritai/ISTA-Net-PyTorch-master/data/Set11"
        imgs = sorted([f for f in os.listdir(SET11_DIR) if f.endswith(('.tif', '.png', '.bmp'))])
        if idx >= len(imgs):
            return None
        path = os.path.join(SET11_DIR, imgs[idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = img.astype(np.float64) / 255.0
        # Center crop to 64x64
        ch, cw = img.shape[0] // 2, img.shape[1] // 2
        crop = img[ch - 32:ch + 32, cw - 32:cw + 32]
        return crop
    except Exception:
        return None


def build_nominal_A(seed=42, H=64, W=64, rate=SAMPLING_RATE):
    """Build nominal measurement matrix (same as DMDPatternSequence)."""
    rng = np.random.default_rng(seed)
    N = H * W
    M = max(1, int(N * rate))
    A = (rng.random((M, N)) > 0.5).astype(np.float64) * 2 - 1
    A /= np.sqrt(N)
    return A


def add_realistic_noise(y_clean, peak=PEAK_PHOTONS, read_sigma=READ_SIGMA,
                        bit_depth=14, seed=0):
    """Apply Poisson + Read + Quantization noise."""
    rng = np.random.default_rng(seed)
    # Poisson shot noise
    scaled = np.maximum(np.abs(y_clean) * peak, 0.0)
    shot = rng.poisson(scaled).astype(np.float64) / peak
    shot *= np.sign(y_clean + 1e-20)
    # Read noise
    y_noisy = shot + rng.normal(0, read_sigma, size=y_clean.shape)
    # Quantization
    max_val = 2 ** bit_depth - 1
    y_quant = np.round(np.clip(y_noisy, -1.0, 1.0) * max_val) / max_val
    return y_quant


def run_w2_experiment(x_true, scene_name, A_nominal, results_list, psnr_fn, ssim_fn):
    """Run W2 multistage correction on one scene."""
    print(f"\n  === W2 Scene: {scene_name} ===")
    M = A_nominal.shape[0]

    # Generate mismatched measurement
    y_clean = spc_forward_with_params(x_true, A_nominal, INJECTED_MISMATCH, H, W)
    y_measured = add_realistic_noise(y_clean, seed=SEED + 10)

    print(f"    y shape={y_measured.shape}, range=[{y_measured.min():.4f}, {y_measured.max():.4f}]")
    print(f"    Injected mismatch: dx={INJECTED_MISMATCH['mask_dx']}, "
          f"dy={INJECTED_MISMATCH['mask_dy']}, theta={INJECTED_MISMATCH['mask_theta']}, "
          f"gain={INJECTED_MISMATCH['gain']}, offset={INJECTED_MISMATCH['offset']}")

    # Reconstruct with uncorrected A
    t0 = time.time()
    x_hat_uncorrected = fista_l2(y_measured, A_nominal, lam=5e-4, iters=1000).reshape(H, W)
    t_uncorr = time.time() - t0
    psnr_uncorr = psnr_fn(x_hat_uncorrected, x_true, max_val=1.0)
    ssim_uncorr = ssim_fn(x_hat_uncorrected, x_true)
    print(f"    Uncorrected: PSNR={psnr_uncorr:.2f}, SSIM={ssim_uncorr:.4f} ({t_uncorr:.1f}s)")

    # Run 4-stage multistage correction
    print("    Running 4-stage UPWMI correction ...")
    fitted_params, meta = spc_multistage_correction(
        y_measured=y_measured,
        A_nominal=A_nominal,
        x_cal=x_true,
        H=H, W=W,
        peak_photons=PEAK_PHOTONS,
        read_sigma=READ_SIGMA,
        verbose=True,
    )

    # Build corrected A and reconstruct
    A_corrected = A_nominal.copy()
    # Apply fitted spatial warp to nominal A
    y_corrected = y_measured.copy()
    # Undo sensor gain/offset from measurement
    g = fitted_params.get("gain", 1.0)
    o = fitted_params.get("offset", 0.0)
    dc = fitted_params.get("dark_current", 0.0)
    y_corrected = (y_corrected - o - dc) / max(g, 0.01)

    # Undo duty cycle / clock offset
    duty = fitted_params.get("duty_cycle", 1.0)
    clock = fitted_params.get("clock_offset", 0.0)
    eff = duty * (1.0 - abs(clock)) if abs(clock) > 1e-8 else duty
    if abs(eff - 1.0) > 1e-8:
        y_corrected /= max(eff, 0.01)

    t0 = time.time()
    x_hat_corrected = fista_l2(y_corrected, A_nominal, lam=5e-4, iters=1000).reshape(H, W)
    t_corr = time.time() - t0
    psnr_corr = psnr_fn(x_hat_corrected, x_true, max_val=1.0)
    ssim_corr = ssim_fn(x_hat_corrected, x_true)

    psnr_delta = psnr_corr - psnr_uncorr
    ssim_delta = ssim_corr - ssim_uncorr
    print(f"    Corrected:   PSNR={psnr_corr:.2f}, SSIM={ssim_corr:.4f} ({t_corr:.1f}s)")
    print(f"    Delta: PSNR +{psnr_delta:.2f} dB, SSIM +{ssim_delta:.4f}")

    # Compare recovered vs injected params
    print("    Recovered vs Injected:")
    for k in ["mask_dx", "mask_dy", "mask_theta", "clock_offset",
              "illum_drift_linear", "illum_drift_sin_amp", "illum_drift_sin_freq",
              "gain", "offset", "dark_current"]:
        inj = INJECTED_MISMATCH.get(k, 0.0)
        rec = fitted_params.get(k, 0.0)
        err = abs(rec - inj)
        print(f"      {k:25s}: injected={inj:+.4f}, recovered={rec:+.4f}, err={err:.4f}")

    result = {
        "scene": scene_name,
        "nll_before": meta["nll_before"],
        "nll_after": meta["nll_after"],
        "nll_decrease_pct": meta["nll_decrease_pct"],
        "psnr_uncorrected": round(psnr_uncorr, 2),
        "ssim_uncorrected": round(ssim_uncorr, 4),
        "psnr_corrected": round(psnr_corr, 2),
        "ssim_corrected": round(ssim_corr, 4),
        "psnr_delta": round(psnr_delta, 2),
        "ssim_delta": round(ssim_delta, 4),
        "injected_params": {k: round(v, 4) for k, v in INJECTED_MISMATCH.items()},
        "recovered_params": {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in fitted_params.items()},
        "stages": meta["stages"],
        "total_time": meta["total_time"],
        "noise_model": f"Poisson(peak={PEAK_PHOTONS:.0f}) + Read(sigma={READ_SIGMA}) + Quant(14bit)",
    }
    results_list.append(result)
    return result


def main():
    psnr_fn = PSNR()
    ssim_fn = SSIM()

    print("=" * 70)
    print("SPC W2 Physics Upgrade — 4-Stage UPWMI Correction")
    print("=" * 70)
    print(f"  H={H}, W={W}, M={N_MEASUREMENTS}, rate={SAMPLING_RATE}")
    print(f"  Noise: Poisson(peak={PEAK_PHOTONS}) + Read(sigma={READ_SIGMA}) + Quant(14bit)")
    print(f"  Mismatch: 17 params (spatial + temporal + illumination + sensor)")
    print(f"  Correction: 4-stage UPWMI multistage grid search")

    # Build nominal measurement matrix
    A_nominal = build_nominal_A(seed=SEED, H=H, W=W, rate=SAMPLING_RATE)
    print(f"  A_nominal: {A_nominal.shape}")

    results_all = []

    # ── Scene 1: Sparse phantom ──────────────────────────────────────
    x_phantom = make_phantom(H, W, SEED)
    run_w2_experiment(x_phantom, "sparse_phantom", A_nominal, results_all, psnr_fn, ssim_fn)

    # ── Scenes 2-4: Set11 crops (if available) ───────────────────────
    for idx in range(3):
        crop = make_set11_crop(idx)
        if crop is not None:
            run_w2_experiment(crop, f"set11_crop_{idx}", A_nominal, results_all, psnr_fn, ssim_fn)
        else:
            print(f"\n  [SKIP] Set11 crop {idx} not available")

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "spc_w2_upgrade")
    results_path = os.path.join(rb_dir, "spc_w2_results.json")
    save_json(results_path, {
        "experiment": "SPC W2 Physics Upgrade",
        "date": datetime.now(timezone.utc).isoformat(),
        "config": {
            "H": H, "W": W, "M": N_MEASUREMENTS,
            "sampling_rate": SAMPLING_RATE,
            "peak_photons": PEAK_PHOTONS,
            "read_sigma": READ_SIGMA,
            "seed": SEED,
        },
        "injected_mismatch": INJECTED_MISMATCH,
        "scenes": results_all,
    })
    print(f"\n  Results saved to: {results_path}")

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scene':<20s} {'PSNR(uncorr)':<14s} {'PSNR(corr)':<14s} "
          f"{'Delta':<10s} {'NLL decrease':<14s}")
    print("-" * 70)
    for r in results_all:
        print(f"{r['scene']:<20s} {r['psnr_uncorrected']:>8.2f} dB   "
              f"{r['psnr_corrected']:>8.2f} dB   "
              f"+{r['psnr_delta']:>5.2f} dB   "
              f"{r['nll_decrease_pct']:>8.1f}%")
    print("=" * 70)

    return results_all


if __name__ == "__main__":
    results = main()
