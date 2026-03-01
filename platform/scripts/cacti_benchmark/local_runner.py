#!/usr/bin/env python3
"""Local CPU runner for CACTI benchmark — runs GAP-TV + blind calibration.

For GPU-dependent methods (EfficientSCI, HiSViT), use modal_runner.py.
"""
from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import shift, rotate, gaussian_filter
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim

# Add pwm_core to path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "packages" / "pwm_core"))
from pwm_core.recon.cacti_solvers import gap_tv_cacti

DATA_DIR = Path(__file__).resolve().parent


def apply_mismatch(mask, dx, dy, rot, blur):
    """Apply mismatch to mask (shift, rotate, blur)."""
    H, W, T = mask.shape
    out = np.empty_like(mask)
    for t in range(T):
        m = mask[:, :, t]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            m = shift(m, [dy, dx], order=1, mode='nearest')
        if abs(rot) > 1e-6:
            m = rotate(m, rot, reshape=False, order=1, mode='nearest')
        if blur > 1e-6:
            m = gaussian_filter(m, sigma=blur)
        out[:, :, t] = m
    return out


def grid_search_spec(y, mask, spec_ranges, inner_iters=20):
    """Grid search + L-BFGS-B for spec calibration."""
    range_map = {s["name"]: (s["min"], s["max"]) for s in spec_ranges}

    dx_min, dx_max = range_map["mask_dx"]
    dy_min, dy_max = range_map["mask_dy"]
    blur_min, blur_max = range_map.get("mask_blur", (0, 0.3))

    # Coarse grid search over dx, dy
    n_dx, n_dy = 11, 7
    best_res, best_dx, best_dy = 1e18, 0.0, 0.0
    dxs = np.linspace(dx_min, dx_max, n_dx)
    dys = np.linspace(dy_min, dy_max, n_dy)

    print(f"  Grid search: {n_dx}x{n_dy} = {n_dx*n_dy} points")
    for dx in dxs:
        for dy in dys:
            mm = apply_mismatch(mask, dx, dy, 0, 0)
            x_hat = gap_tv_cacti(y, mm, iterations=inner_iters)
            res = np.sum((y - np.sum(mm * x_hat, axis=2))**2)
            if res < best_res:
                best_res, best_dx, best_dy = res, dx, dy

    print(f"  Grid best: dx={best_dx:.2f} dy={best_dy:.2f} res={best_res:.2f}")

    # L-BFGS-B refinement over [dx, dy, rot, blur]
    rot_min, rot_max = range_map.get("mask_rotation", (-0.2, 0.2))
    x0 = [best_dx, best_dy, 0.0, (blur_min + blur_max) / 2]
    bounds = [(dx_min, dx_max), (dy_min, dy_max),
              (rot_min, rot_max), (blur_min, blur_max)]

    eval_count = [0]
    def objective(params):
        eval_count[0] += 1
        dx, dy, rot, blur = params
        mm = apply_mismatch(mask, dx, dy, rot, blur)
        x_hat = gap_tv_cacti(y, mm, iterations=inner_iters)
        res = np.sum((y - np.sum(mm * x_hat, axis=2))**2)
        if eval_count[0] % 5 == 0:
            print(f"    refine {eval_count[0]}: [{dx:.4f}, {dy:.4f}, {rot:.4f}, {blur:.4f}] res={res:.2f}")
        return res

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxfun': 80, 'ftol': 1e-4})
    dx, dy, rot, blur = result.x

    # Estimate gain and offset analytically
    mm = apply_mismatch(mask, dx, dy, rot, blur)
    x_hat = gap_tv_cacti(y, mm, iterations=inner_iters)
    Hx = np.sum(mm * x_hat, axis=2)
    gain = np.sum(y * Hx) / max(np.sum(Hx * Hx), 1e-10)
    offset = np.mean(y - gain * Hx)

    print(f"  Refined: dx={dx:.4f} dy={dy:.4f} rot={rot:.4f} blur={blur:.4f} "
          f"({eval_count[0]} evals)")

    return {
        "mask_dx": round(float(dx), 6),
        "mask_dy": round(float(dy), 6),
        "mask_rotation": round(float(rot), 6),
        "mask_blur": round(float(blur), 6),
        "clock_offset": 0.0,
        "gain_drift": round(float(gain), 6),
        "offset_drift": round(float(offset), 6),
    }


def compute_score(x_hat, x_true, y, mask_corrected, gain, offset):
    """Compute composite score: 0.4*PSNR_norm + 0.4*SSIM + 0.2*Consistency."""
    # PSNR
    mse = np.mean((x_hat - x_true)**2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    psnr_norm = np.clip((psnr - 15) / 30, 0, 1)

    # SSIM (per-frame average)
    ssim_vals = []
    T = x_hat.shape[2]
    for t in range(T):
        s = ssim(x_true[:,:,t], x_hat[:,:,t], data_range=1.0)
        ssim_vals.append(s)
    ssim_avg = np.mean(ssim_vals)

    # Consistency
    Hx = gain * np.sum(mask_corrected * x_hat, axis=2) + offset
    cons = 1.0 - np.linalg.norm(y - Hx) / max(np.linalg.norm(y), 1e-10)
    cons = max(cons, 0)

    score = 0.4 * psnr_norm + 0.4 * ssim_avg + 0.2 * cons
    return psnr, ssim_avg, cons, score


def run_tier(tier, method="gap_tv", cal_samples=3, final_iters=100):
    """Run a method on one tier."""
    h5_path = DATA_DIR / tier / f"cacti_challenge_{tier}.h5"
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        return None

    print(f"\n{'='*70}")
    print(f"  {method} on {tier} tier (local CPU)")
    print(f"{'='*70}")

    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted(f.keys())
        samples = []
        for sk in sample_keys:
            grp = f[sk]
            mask = grp["H_ideal"][:].astype(np.float64)
            T = mask.shape[2]
            # Only process T=8 for dev/hidden (consistent with Modal runner)
            if tier in ("dev", "hidden") and T != 8:
                continue
            samples.append({
                "key": sk,
                "y": grp["y"][:].astype(np.float64),
                "mask": mask,
                "x_true": grp["x_true"][:].astype(np.float64) if "x_true" in grp else None,
                "spec_ranges": json.loads(grp.attrs["spec_ranges"]),
                "true_spec": json.loads(grp.attrs["true_spec"]) if "true_spec" in grp.attrs else None,
            })

    n = len(samples)
    print(f"  {n} samples loaded")

    # Step 1: Calibrate
    print(f"\n--- Calibration ({cal_samples} samples) ---")
    specs = []
    for i in range(min(cal_samples, n)):
        s = samples[i]
        print(f"\nCalibrating on {s['key']}...")
        spec = grid_search_spec(s["y"], s["mask"], s["spec_ranges"])
        specs.append(spec)
        print(f"  -> {spec}")

    avg_spec = {k: round(float(np.mean([sp[k] for sp in specs])), 6) for k in specs[0]}
    print(f"\nAveraged spec: {avg_spec}")
    if samples[0].get("true_spec"):
        print(f"True spec:     {samples[0]['true_spec']}")

    # Step 2: Reconstruct
    print(f"\n--- Reconstruction ({n} samples, {final_iters} iters) ---")
    all_scores = []

    for i, s in enumerate(samples):
        H, W, T = s["mask"].shape
        mask_corrected = apply_mismatch(
            s["mask"], avg_spec["mask_dx"], avg_spec["mask_dy"],
            avg_spec["mask_rotation"], avg_spec["mask_blur"],
        )
        gain = avg_spec["gain_drift"]
        offset = avg_spec["offset_drift"]
        y_corrected = (s["y"] - offset) / max(abs(gain), 1e-6)

        t0 = time.time()
        x_hat = gap_tv_cacti(y_corrected, mask_corrected, iterations=final_iters)
        dt = time.time() - t0

        if s["x_true"] is not None:
            psnr, ssim_avg, cons, score = compute_score(
                x_hat, s["x_true"], s["y"], mask_corrected, gain, offset)
            print(f"  [{i+1}/{n}] {s['key']}: PSNR={psnr:.2f} SSIM={ssim_avg:.4f} "
                  f"Cons={cons:.4f} Score={score:.4f} ({dt:.1f}s)")
            all_scores.append({"psnr": psnr, "ssim": ssim_avg,
                             "consistency": cons, "composite": score})

    if all_scores:
        avg = {k: float(np.mean([s[k] for s in all_scores])) for k in all_scores[0]}
        print(f"\nAVERAGE: PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f} "
              f"Cons={avg['consistency']:.4f} Score={avg['composite']:.4f}")
        return avg
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="public", choices=["public", "dev", "hidden"])
    parser.add_argument("--cal-samples", type=int, default=3)
    parser.add_argument("--final-iters", type=int, default=100)
    args = parser.parse_args()

    run_tier(args.tier, cal_samples=args.cal_samples, final_iters=args.final_iters)
