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
from scipy.ndimage import affine_transform, gaussian_filter
from skimage.metrics import structural_similarity as ssim

# Add pwm_core to path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "packages" / "pwm_core"))
from pwm_core.recon.cacti_solvers import gap_tv_cacti

DATA_DIR = Path(__file__).resolve().parent


def apply_mismatch(mask, dx, dy, rot, blur):
    """Apply spatial mismatch using the paper's exact affine_transform.

    Matches validate_cacti_inversenet.py:129-146 exactly: single affine
    combining shift + rotation around image center.
    """
    H, W, T = mask.shape
    out = np.zeros_like(mask)
    cx, cy = W / 2.0, H / 2.0
    th = np.radians(rot)
    cos_t, sin_t = np.cos(th), np.sin(th)
    for t in range(T):
        mat = np.array([
            [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
            [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
        ])
        inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
        frame = affine_transform(mask[:, :, t], inv[:2, :2], offset=inv[:2, 2], cval=0)
        if blur > 0:
            frame = gaussian_filter(frame, sigma=blur)
        out[:, :, t] = frame
    return out.astype(np.float32)


def binarize_mask(mask):
    return (mask > 0.5).astype(np.float64)


def grid_search_spec(samples, spec_ranges, inner_iters=20):
    """Accumulated-residual grid search for spec calibration.

    Takes ALL calibration samples and accumulates residual across them at each
    grid point (following the InverseNet paper). Uses sequential 1D sweeps
    for rotation and blur after finding best (dx, dy).

    Phases:
      1. Coarse 2D grid over (dx, dy) — 13×9 = 117 points
      2. Fine 2D grid around best — 5×5 = 25 points
      3. 1D sweep for rotation — 7 points
      4. 1D sweep for blur — 5 points
      5. Analytical gain/offset estimation

    Args:
        samples: List of dicts with 'y' and 'mask' keys.
        spec_ranges: List of spec range dicts from dataset.
        inner_iters: GAP-TV iterations per evaluation.

    Returns:
        Estimated spec dict with 7 parameters.
    """
    range_map = {s["name"]: (s["min"], s["max"]) for s in spec_ranges}

    dx_min, dx_max = range_map["mask_dx"]
    dy_min, dy_max = range_map["mask_dy"]
    rot_min, rot_max = range_map.get("mask_rotation", (0, 0.3))
    blur_min, blur_max = range_map.get("mask_blur", (0, 0.3))

    N = len(samples)
    ys = [s["y"] for s in samples]
    masks = [s["mask"] for s in samples]

    def accumulated_residual(dx, dy, rot, blur):
        """Reconstruct each sample and sum residual across all.

        Always uses binarized mask to match the forward model used during
        data generation (measurements were produced with binary masks).
        """
        total = 0.0
        for y, mask in zip(ys, masks):
            mm = binarize_mask(apply_mismatch(mask, dx, dy, rot, blur))
            x_hat = gap_tv_cacti(y, mm, iterations=inner_iters, tv_iter=3)
            y_pred = np.sum(mm * x_hat, axis=2)
            total += float(np.sum((y - y_pred) ** 2))
        return total

    # ── Phase 1: Coarse 2D grid over (dx, dy) ──
    n_dx, n_dy = 13, 9
    dxs = np.linspace(dx_min, dx_max, n_dx)
    dys = np.linspace(dy_min, dy_max, n_dy)
    dx_step = dxs[1] - dxs[0] if n_dx > 1 else 0.1
    dy_step = dys[1] - dys[0] if n_dy > 1 else 0.1
    print(f"  Phase 1: Coarse grid {n_dx}x{n_dy} = {n_dx*n_dy} pts x {N} samples "
          f"(dx=[{dx_min:.2f},{dx_max:.2f}], dy=[{dy_min:.2f},{dy_max:.2f}])")

    best_res, best_dx, best_dy = 1e18, 0.0, 0.0
    for i_dx, dx in enumerate(dxs):
        for dy in dys:
            res = accumulated_residual(dx, dy, 0, 0)
            if res < best_res:
                best_res, best_dx, best_dy = res, dx, dy
        print(f"    row {i_dx+1}/{n_dx} done (dx={dx:.3f})")

    print(f"  Phase 1 best: dx={best_dx:.3f} dy={best_dy:.3f} res={best_res:.1f}")

    # ── Phase 2: Fine 2D grid around best ──
    fine_dxs = np.linspace(best_dx - dx_step, best_dx + dx_step, 5)
    fine_dys = np.linspace(best_dy - dy_step, best_dy + dy_step, 5)
    fine_dxs = np.clip(fine_dxs, dx_min, dx_max)
    fine_dys = np.clip(fine_dys, dy_min, dy_max)
    print(f"  Phase 2: Fine grid 5x5 = 25 pts x {N} samples "
          f"(dx=[{fine_dxs[0]:.3f},{fine_dxs[-1]:.3f}], "
          f"dy=[{fine_dys[0]:.3f},{fine_dys[-1]:.3f}])")

    for dx in fine_dxs:
        for dy in fine_dys:
            res = accumulated_residual(dx, dy, 0, 0)
            if res < best_res:
                best_res, best_dx, best_dy = res, dx, dy

    print(f"  Phase 2 best: dx={best_dx:.4f} dy={best_dy:.4f} res={best_res:.1f}")

    # ── Phase 3: 1D sweep for rotation ──
    best_rot = 0.0
    rots = np.linspace(rot_min, rot_max, 7)
    print(f"  Phase 3: Rotation sweep {len(rots)} pts x {N} samples "
          f"(rot=[{rot_min:.3f},{rot_max:.3f}])")

    for rot in rots:
        res = accumulated_residual(best_dx, best_dy, rot, 0)
        if res < best_res:
            best_res, best_rot = res, rot

    if best_rot >= rot_max - 1e-6 or (best_rot <= rot_min + 1e-6 and rot_min > 0):
        print(f"  Phase 3: rot={best_rot:.4f} at boundary → using 0.0")
        best_rot = 0.0
        best_res = accumulated_residual(best_dx, best_dy, 0, 0)
    else:
        print(f"  Phase 3 best: rot={best_rot:.4f} res={best_res:.1f}")

    # ── Phase 4: 1D sweep for blur ──
    best_blur = 0.0
    blurs = np.linspace(blur_min, blur_max, 5)
    print(f"  Phase 4: Blur sweep {len(blurs)} pts x {N} samples "
          f"(blur=[{blur_min:.3f},{blur_max:.3f}])")

    for blur in blurs:
        res = accumulated_residual(best_dx, best_dy, best_rot, blur)
        if res < best_res:
            best_res, best_blur = res, blur

    if best_blur >= blur_max - 1e-6:
        print(f"  Phase 4: blur={best_blur:.4f} at boundary → using 0.0")
        best_blur = 0.0
    else:
        print(f"  Phase 4 best: blur={best_blur:.4f} res={best_res:.1f}")

    # ── Phase 5: Analytical gain/offset ──
    sum_yHx = 0.0
    sum_HxHx = 0.0
    Hx_list = []
    for y, mask in zip(ys, masks):
        mm = binarize_mask(apply_mismatch(mask, best_dx, best_dy, best_rot, best_blur))
        x_hat = gap_tv_cacti(y, mm, iterations=inner_iters, tv_iter=3)
        Hx = np.sum(mm * x_hat, axis=2)
        sum_yHx += float(np.sum(y * Hx))
        sum_HxHx += float(np.sum(Hx * Hx))
        Hx_list.append(Hx)

    gain = sum_yHx / max(sum_HxHx, 1e-10)

    sum_y_minus_gHx = 0.0
    n_pixels = 0
    for y, Hx in zip(ys, Hx_list):
        sum_y_minus_gHx += float(np.sum(y - gain * Hx))
        n_pixels += y.size
    offset = sum_y_minus_gHx / max(n_pixels, 1)

    if not (0.92 <= gain <= 1.08):
        print(f"  Phase 5: gain={gain:.4f} out of range → clamping to 1.0")
        gain = 1.0
        offset = 0.0
    else:
        print(f"  Phase 5: gain={gain:.4f}  offset={offset:.6f}")

    return {
        "mask_dx": round(float(best_dx), 6),
        "mask_dy": round(float(best_dy), 6),
        "mask_rotation": round(float(best_rot), 6),
        "mask_blur": round(float(best_blur), 6),
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

    # Step 1: Calibrate (batch mode — accumulated residual across samples)
    n_cal = min(cal_samples, n)
    print(f"\n--- Calibration ({n_cal} samples, batch mode) ---")
    cal_batch = samples[:n_cal]
    avg_spec = grid_search_spec(cal_batch, samples[0]["spec_ranges"])
    print(f"\nEstimated spec: {avg_spec}")
    if samples[0].get("true_spec"):
        print(f"True spec:     {samples[0]['true_spec']}")

    # Step 2: Reconstruct
    print(f"\n--- Reconstruction ({n} samples, {final_iters} iters) ---")
    all_scores = []

    for i, s in enumerate(samples):
        H, W, T = s["mask"].shape
        mask_corrected = binarize_mask(apply_mismatch(
            s["mask"], avg_spec["mask_dx"], avg_spec["mask_dy"],
            avg_spec["mask_rotation"], avg_spec["mask_blur"],
        ))
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
