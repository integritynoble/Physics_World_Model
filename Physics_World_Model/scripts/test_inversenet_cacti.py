#!/usr/bin/env python3
"""InverseNet CACTI Benchmark — All Solvers × All Scenarios.

Reproduces the InverseNet paper Table results across four scenarios:
  Scenario I   (Ideal):    clean y from ground truth + nominal mask
  Scenario II  (Baseline): corrupted y + nominal mask (no correction)
  Scenario III (Oracle):   corrupted y + true_spec correction
  Scenario IV  (Blind):    corrupted y + estimated spec (accumulated grid search)

Solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI, HiSViT-9, HiSViT-13

Dataset uses peak_photon=10000 noise (~40 dB measurement SNR), matching the
InverseNet paper. Forward model uses binarized warped mask. Standard accelerated
GAP works for all scenarios.

Usage:
    python scripts/test_inversenet_cacti.py
    python scripts/test_inversenet_cacti.py --samples 5       # quick test
    python scripts/test_inversenet_cacti.py --scenario oracle  # single scenario
    python scripts/test_inversenet_cacti.py --scenario blind   # blind calibration
    python scripts/test_inversenet_cacti.py --method gap_tv    # single method
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path

import gc
import numpy as np

# ── Project paths ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# Force-import cacti_solvers from THIS repo (bypass globally installed pwm_core)
import importlib.util as _ilu
_SOLVERS_FILE = str(_ROOT / "packages" / "pwm_core" / "pwm_core" / "recon" / "cacti_solvers.py")
_spec = _ilu.spec_from_file_location("cacti_solvers", _SOLVERS_FILE)
_cs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cs)

SOLVERS = _cs.SOLVERS
solve_cacti = _cs.solve_cacti
_resolve_device = _cs._resolve_device
_cacti_solvers = _cs

# ── Dataset path ─────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "datasets" / "benchmark" / "cacti"

# ── InverseNet paper reference values ────────────────────────────────────────
# Dataset now uses peak_photon=10000 noise (~40 dB SNR), matching the paper.
# Results should be close to these reference values.
PAPER_BASELINES = {
    "scenario_ii": {
        "EfficientSCI":  {"psnr": 27.38, "ssim": 0.927},
        "ELP-Unfolding": {"psnr": 26.50, "ssim": 0.910},
        "PnP-FFDNet":    {"psnr": 20.15, "ssim": 0.650},
        "GAP-TV":        {"psnr": 14.81, "ssim": 0.303},
    },
    "scenario_iii": {
        "EfficientSCI":  {"psnr": 35.39, "ssim": 0.973},
        "ELP-Unfolding": {"psnr": 34.09, "ssim": 0.965},
        "PnP-FFDNet":    {"psnr": 29.28, "ssim": 0.910},
        "GAP-TV":        {"psnr": 26.75, "ssim": 0.870},
    },
}

# Map solver keys to display names
SOLVER_DISPLAY = {
    "gap_tv": "GAP-TV",
    "pnp_ffdnet": "PnP-FFDNet",
    "elp_unfolding": "ELP-Unfolding",
    "efficient_sci": "EfficientSCI",
    "hisvit9": "HiSViT-9",
    "hisvit13": "HiSViT-13",
}

# DL methods that need binary masks
DL_METHODS = {"efficient_sci", "elp_unfolding", "hisvit9", "hisvit13"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def apply_mismatch(mask, dx, dy, rot, blur):
    """Apply spatial mismatch to mask using the paper's exact affine_transform.

    Matches validate_cacti_inversenet.py:129-146 exactly: single affine
    combining shift + rotation around image center.
    """
    from scipy.ndimage import affine_transform, gaussian_filter
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


def compute_psnr(x_true, x_hat, max_val=1.0):
    mse = float(np.mean((x_true.astype(np.float64) - x_hat.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 60.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x_true, x_hat):
    """Per-frame SSIM averaged over temporal dimension."""
    from skimage.metrics import structural_similarity as sk_ssim
    T = x_true.shape[2]
    vals = [sk_ssim(x_true[:, :, t], x_hat[:, :, t], data_range=1.0)
            for t in range(T)]
    return float(np.mean(vals))


def compute_consistency(y, x_hat, mask, gain=1.0, offset=0.0):
    y_pred = gain * np.sum(mask * x_hat, axis=2) + offset
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-10:
        return 1.0
    return max(0.0, float(1.0 - np.linalg.norm(y - y_pred) / y_norm))


def compute_composite(psnr, ssim_val, consistency):
    psnr_norm = float(np.clip((psnr - 15) / 30, 0, 1))
    return 0.4 * psnr_norm + 0.4 * ssim_val + 0.2 * consistency


def binarize_mask(mask):
    return (mask > 0.5).astype(np.float32)


# ── Reconstruction dispatcher ────────────────────────────────────────────────

def reconstruct(y, mask, method, device, iters):
    """Dispatch reconstruction via solve_cacti().

    With peak_photon=10000 noise (~40 dB SNR), the standard accelerated GAP
    works well for all scenarios. No special noisy-data handling needed.
    """
    y_in = y.astype(np.float32)
    mask_in = mask.astype(np.float32)
    if method in DL_METHODS:
        mask_in = binarize_mask(mask).astype(np.float32)
    return solve_cacti(y_in, mask_in, method=method, device=device,
                       iterations=iters)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_tier(tier="public", max_samples=None):
    """Load dataset from HDF5 for a given tier (public, dev, hidden)."""
    import h5py
    h5_path = DATA_DIR / tier / f"cacti_challenge_{tier}.h5"
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found")
        sys.exit(1)

    samples = []
    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted(f.keys())
        if max_samples:
            sample_keys = sample_keys[:max_samples]
        for sk in sample_keys:
            grp = f[sk]
            mask = grp["H_ideal"][:].astype(np.float32)
            T = mask.shape[2]
            # DL methods only support T=8; filter for consistency across tiers
            if tier in ("dev", "hidden") and T != 8:
                continue
            samples.append({
                "key": sk,
                "y": grp["y"][:].astype(np.float32),
                "mask": mask.astype(np.float32),
                "x_true": grp["x_true"][:].astype(np.float32) if "x_true" in grp else None,
                "spec_ranges": json.loads(grp.attrs["spec_ranges"]),
                "true_spec": json.loads(grp.attrs["true_spec"]) if "true_spec" in grp.attrs else None,
                "metadata": json.loads(grp.attrs.get("metadata", "{}")),
            })
    return samples


# ── Scenario runners ─────────────────────────────────────────────────────────

def run_scenario_i(samples, method, device, iters=100, **_kw):
    """Scenario I (Ideal): clean y from ground truth + nominal mask."""
    scores = []
    for i, s in enumerate(samples):
        y_clean = np.sum(s["mask"] * s["x_true"], axis=2)

        t0 = time.time()
        x_hat = reconstruct(y_clean, s["mask"], method, device, iters)
        dt = time.time() - t0

        psnr = compute_psnr(s["x_true"], x_hat)
        ssim_val = compute_ssim(s["x_true"], x_hat)
        cons = compute_consistency(y_clean, x_hat, s["mask"])
        comp = compute_composite(psnr, ssim_val, cons)

        scene = s["metadata"].get("scene", s["key"])
        print(f"    [{i+1:2d}/{len(samples)}] {scene:<14s}  "
              f"PSNR={psnr:6.2f}  SSIM={ssim_val:.4f}  Cons={cons:.4f}  "
              f"Score={comp:.4f}  ({dt:.1f}s)")
        scores.append({"psnr": psnr, "ssim": ssim_val,
                        "consistency": cons, "composite": comp, "time": dt})
        del x_hat, y_clean; gc.collect()
    return scores


def run_scenario_ii(samples, method, device, iters=100, **_kw):
    """Scenario II (Baseline): corrupted y + nominal mask (no correction)."""
    scores = []
    for i, s in enumerate(samples):
        t0 = time.time()
        x_hat = reconstruct(s["y"], s["mask"], method, device, iters)
        dt = time.time() - t0

        psnr = compute_psnr(s["x_true"], x_hat)
        ssim_val = compute_ssim(s["x_true"], x_hat)
        cons = compute_consistency(s["y"], x_hat, s["mask"], 1.0, 0.0)
        comp = compute_composite(psnr, ssim_val, cons)

        scene = s["metadata"].get("scene", s["key"])
        print(f"    [{i+1:2d}/{len(samples)}] {scene:<14s}  "
              f"PSNR={psnr:6.2f}  SSIM={ssim_val:.4f}  Cons={cons:.4f}  "
              f"Score={comp:.4f}  ({dt:.1f}s)")
        scores.append({"psnr": psnr, "ssim": ssim_val,
                        "consistency": cons, "composite": comp, "time": dt})
        del x_hat; gc.collect()
    return scores


def run_scenario_iii(samples, method, device, iters=100, **_kw):
    """Scenario III (Oracle): corrupted y + true_spec correction.

    Uses binarized warped mask to match the forward model used during
    data generation (InverseNet paper binarizes after warping).
    """
    scores = []
    for i, s in enumerate(samples):
        ts = s["true_spec"]
        mask_corrected = apply_mismatch(
            s["mask"], ts["mask_dx"], ts["mask_dy"],
            ts["mask_rotation"], ts["mask_blur"],
        )
        # Binarize to match forward model (data generated with binarized mask)
        mask_corrected = binarize_mask(mask_corrected)
        gain = ts["gain_drift"]
        offset = ts["offset_drift"]
        y_corrected = (s["y"] - offset) / max(abs(gain), 1e-6)

        t0 = time.time()
        x_hat = reconstruct(y_corrected, mask_corrected, method, device, iters)
        dt = time.time() - t0

        psnr = compute_psnr(s["x_true"], x_hat)
        ssim_val = compute_ssim(s["x_true"], x_hat)
        cons = compute_consistency(s["y"], x_hat, mask_corrected, gain, offset)
        comp = compute_composite(psnr, ssim_val, cons)

        scene = s["metadata"].get("scene", s["key"])
        print(f"    [{i+1:2d}/{len(samples)}] {scene:<14s}  "
              f"PSNR={psnr:6.2f}  SSIM={ssim_val:.4f}  Cons={cons:.4f}  "
              f"Score={comp:.4f}  ({dt:.1f}s)")
        scores.append({"psnr": psnr, "ssim": ssim_val,
                        "consistency": cons, "composite": comp, "time": dt})
        del x_hat; gc.collect()
    return scores


# ── Blind calibration ─────────────────────────────────────────────────────────

# Self-contained subprocess script for blind calibration.
# Runs in a separate process to avoid CUDA's ~56 GB virtual memory overhead.
_BLIND_CAL_SUBPROCESS_SCRIPT = r'''
import sys, os, json, gc, tempfile
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

def apply_mismatch(mask, dx, dy, rot, blur):
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
    return (mask > 0.5).astype(np.float32)

def gap_tv(y, Phi, max_iter=20, tv_weight=0.1, tv_iter=3):
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        denoise_tv_chambolle = None
    h, w, nF = Phi.shape
    Phi_sum = np.sum(Phi, axis=2)
    Phi_sum[Phi_sum == 0] = 1
    x = y[:, :, np.newaxis] * Phi / Phi_sum[:, :, np.newaxis]
    y1 = y.copy()
    for _ in range(max_iter):
        yb = np.sum(x * Phi, axis=2)
        y1 = y1 + (y - yb)
        residual = y1 - yb
        x = x + (residual / Phi_sum)[:, :, np.newaxis] * Phi
        if denoise_tv_chambolle is not None:
            for f in range(nF):
                x[:, :, f] = denoise_tv_chambolle(x[:, :, f], weight=tv_weight, max_num_iter=tv_iter)
        else:
            for f in range(nF):
                x[:, :, f] = gaussian_filter(x[:, :, f], sigma=0.5)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)

def main():
    npz_path = sys.argv[1]
    out_path = sys.argv[2]
    inner_iters = int(sys.argv[3])
    data = np.load(npz_path, allow_pickle=True)
    ys = list(data["ys"])
    masks = list(data["masks"])
    ranges = json.loads(str(data["ranges"]))
    N = len(ys)
    range_map = {s["name"]: (s["min"], s["max"]) for s in ranges}
    dx_min, dx_max = range_map["mask_dx"]
    dy_min, dy_max = range_map["mask_dy"]
    rot_min, rot_max = range_map.get("mask_rotation", (0, 0.3))
    blur_min, blur_max = range_map.get("mask_blur", (0, 0.5))

    def accumulated_residual(dx, dy, rot, blur):
        total = 0.0
        for y, mask in zip(ys, masks):
            mm = binarize_mask(apply_mismatch(mask, dx, dy, rot, blur))
            x_hat = gap_tv(y.astype(np.float32), mm.astype(np.float32),
                           max_iter=inner_iters, tv_weight=0.1, tv_iter=3)
            y_pred = np.sum(mm * x_hat, axis=2)
            total += float(np.sum((y - y_pred) ** 2))
            del mm, x_hat, y_pred
        gc.collect()
        return total

    img_h = masks[0].shape[0]
    n_dx, n_dy = (7, 5) if img_h > 256 else (13, 9)
    dxs = np.linspace(dx_min, dx_max, n_dx)
    dys = np.linspace(dy_min, dy_max, n_dy)
    dx_step = dxs[1] - dxs[0] if n_dx > 1 else 0.1
    dy_step = dys[1] - dys[0] if n_dy > 1 else 0.1
    print(f"    Phase 1: Coarse grid {n_dx}x{n_dy} = {n_dx*n_dy} pts x {N} samples "
          f"(dx=[{dx_min:.2f},{dx_max:.2f}], dy=[{dy_min:.2f},{dy_max:.2f}])", flush=True)

    best_res, best_dx, best_dy = 1e18, 0.0, 0.0
    for i_dx, dx in enumerate(dxs):
        for dy in dys:
            res = accumulated_residual(dx, dy, 0, 0)
            if res < best_res:
                best_res, best_dx, best_dy = res, dx, dy
        print(f"      row {i_dx+1}/{n_dx} done (dx={dx:.3f})", flush=True)
    print(f"    Phase 1 best: dx={best_dx:.3f} dy={best_dy:.3f} res={best_res:.1f}", flush=True)

    fine_dxs = np.clip(np.linspace(best_dx - dx_step, best_dx + dx_step, 5), dx_min, dx_max)
    fine_dys = np.clip(np.linspace(best_dy - dy_step, best_dy + dy_step, 5), dy_min, dy_max)
    print(f"    Phase 2: Fine grid 5x5 = 25 pts x {N} samples "
          f"(dx=[{fine_dxs[0]:.3f},{fine_dxs[-1]:.3f}], "
          f"dy=[{fine_dys[0]:.3f},{fine_dys[-1]:.3f}])", flush=True)
    for dx in fine_dxs:
        for dy in fine_dys:
            res = accumulated_residual(dx, dy, 0, 0)
            if res < best_res:
                best_res, best_dx, best_dy = res, dx, dy
    print(f"    Phase 2 best: dx={best_dx:.4f} dy={best_dy:.4f} res={best_res:.1f}", flush=True)

    best_rot = 0.0
    rots = np.linspace(rot_min, rot_max, 7)
    print(f"    Phase 3: Rotation sweep {len(rots)} pts x {N} samples "
          f"(rot=[{rot_min:.3f},{rot_max:.3f}])", flush=True)
    for rot in rots:
        res = accumulated_residual(best_dx, best_dy, rot, 0)
        if res < best_res:
            best_res, best_rot = res, rot
    if best_rot >= rot_max - 1e-6 or (best_rot <= rot_min + 1e-6 and rot_min > 0):
        print(f"    Phase 3: rot={best_rot:.4f} at boundary -> using 0.0", flush=True)
        best_rot = 0.0
        best_res = accumulated_residual(best_dx, best_dy, 0, 0)
    else:
        print(f"    Phase 3 best: rot={best_rot:.4f} res={best_res:.1f}", flush=True)

    best_blur = 0.0
    blurs = np.linspace(blur_min, blur_max, 5)
    print(f"    Phase 4: Blur sweep {len(blurs)} pts x {N} samples "
          f"(blur=[{blur_min:.3f},{blur_max:.3f}])", flush=True)
    for blur in blurs:
        res = accumulated_residual(best_dx, best_dy, best_rot, blur)
        if res < best_res:
            best_res, best_blur = res, blur
    if best_blur >= blur_max - 1e-6:
        print(f"    Phase 4: blur={best_blur:.4f} at boundary -> using 0.0", flush=True)
        best_blur = 0.0
    else:
        print(f"    Phase 4 best: blur={best_blur:.4f} res={best_res:.1f}", flush=True)

    sum_yHx, sum_HxHx = 0.0, 0.0
    Hx_list = []
    for y, mask in zip(ys, masks):
        mm = binarize_mask(apply_mismatch(mask, best_dx, best_dy, best_rot, best_blur))
        x_hat = gap_tv(y.astype(np.float32), mm.astype(np.float32),
                       max_iter=inner_iters, tv_weight=0.1, tv_iter=3)
        Hx = np.sum(mm * x_hat, axis=2)
        sum_yHx += float(np.sum(y * Hx))
        sum_HxHx += float(np.sum(Hx * Hx))
        Hx_list.append(Hx)
        del mm, x_hat
    gc.collect()
    gain = sum_yHx / max(sum_HxHx, 1e-10)
    sum_y_minus_gHx, n_pixels = 0.0, 0
    for y, Hx in zip(ys, Hx_list):
        sum_y_minus_gHx += float(np.sum(y - gain * Hx))
        n_pixels += y.size
    offset = sum_y_minus_gHx / max(n_pixels, 1)
    if not (0.92 <= gain <= 1.08):
        print(f"    Phase 5: gain={gain:.4f} out of range -> clamping to 1.0", flush=True)
        gain, offset = 1.0, 0.0
    else:
        print(f"    Phase 5: gain={gain:.4f}  offset={offset:.6f}", flush=True)

    result = {"mask_dx": best_dx, "mask_dy": best_dy, "mask_rotation": best_rot,
              "mask_blur": best_blur, "gain_drift": gain, "offset_drift": offset}
    with open(out_path, "w") as f:
        json.dump(result, f)
    print("    Calibration subprocess done.", flush=True)

if __name__ == "__main__":
    main()
'''


def blind_calibrate(samples, spec_ranges, inner_iters=20):
    """Estimate mismatch parameters via subprocess (avoids CUDA memory overhead).

    Runs the grid search in a separate Python process that never imports torch,
    avoiding CUDA's ~56 GB virtual memory reservation on Windows.
    """
    import subprocess, tempfile

    N = min(len(samples), 6)
    ys = [s["y"] for s in samples[:N]]
    masks = [s["mask"] for s in samples[:N]]

    # Write inputs to temp file
    tmpdir = tempfile.mkdtemp(prefix="blind_cal_")
    npz_path = os.path.join(tmpdir, "cal_input.npz")
    out_path = os.path.join(tmpdir, "cal_output.json")
    script_path = os.path.join(tmpdir, "blind_cal.py")

    np.savez(npz_path, ys=np.array(ys), masks=np.array(masks),
             ranges=json.dumps(spec_ranges))

    with open(script_path, "w") as f:
        f.write(_BLIND_CAL_SUBPROCESS_SCRIPT)

    # Run in subprocess without torch
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""  # Prevent CUDA init
    proc = subprocess.run(
        [sys.executable, script_path, npz_path, out_path, str(inner_iters)],
        env=env, timeout=7200,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Blind calibration subprocess failed (exit {proc.returncode})")

    with open(out_path) as f:
        est_spec = json.load(f)

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return est_spec

    return {
        "mask_dx": round(float(best_dx), 6),
        "mask_dy": round(float(best_dy), 6),
        "mask_rotation": round(float(best_rot), 6),
        "mask_blur": round(float(best_blur), 6),
        "clock_offset": 0.0,
        "gain_drift": round(float(gain), 6),
        "offset_drift": round(float(offset), 6),
    }


def run_scenario_iv(samples, method, device, iters=100, cal_samples=None,
                    _cached_spec=None, **_kw):
    """Scenario IV (Blind): corrupted y + estimated spec from blind calibration.

    Passes all calibration samples as a batch to blind_calibrate() for
    accumulated-residual grid search (following the InverseNet paper).

    If _cached_spec is provided, skips calibration and reuses the spec.
    """
    if _cached_spec is not None:
        est_spec = _cached_spec
        print(f"\n  --- Using cached calibration spec ---")
    else:
        # Default cal_samples: 3 for large images (512×512), 6 otherwise
        if cal_samples is None:
            cal_samples = 3 if samples[0]["mask"].shape[0] > 256 else 6
        n_cal = min(cal_samples, len(samples))
        print(f"\n  --- Blind Calibration ({n_cal} samples, GAP-TV inner solver) ---")

        t_cal_start = time.time()
        cal_batch = samples[:n_cal]
        est_spec = blind_calibrate(cal_batch, samples[0]["spec_ranges"])
        dt_cal = time.time() - t_cal_start

        # Print calibration summary
        ts = samples[0]["true_spec"]
        print(f"\n  --- Calibration Summary ({dt_cal:.1f}s) ---")
        if ts:
            print(f"  {'Param':<18s}  {'Estimated':>10s}  {'True':>10s}  {'Error':>10s}")
            print(f"  {'-'*52}")
            for k in ["mask_dx", "mask_dy", "mask_rotation", "mask_blur",
                      "gain_drift", "offset_drift"]:
                est = est_spec[k]
                true = ts[k]
                print(f"  {k:<18s}  {est:>10.4f}  {true:>10.4f}  {abs(est-true):>10.4f}")
        else:
            print(f"  Estimated spec: {est_spec}")

    # Reconstruct with estimated spec
    print(f"\n  --- Reconstruction with estimated spec ---")
    scores = []
    for i, s in enumerate(samples):
        mask_corrected = apply_mismatch(
            s["mask"], est_spec["mask_dx"], est_spec["mask_dy"],
            est_spec["mask_rotation"], est_spec["mask_blur"],
        )
        mask_corrected = binarize_mask(mask_corrected)
        gain = est_spec["gain_drift"]
        offset = est_spec["offset_drift"]
        y_corrected = (s["y"] - offset) / max(abs(gain), 1e-6)

        t0 = time.time()
        x_hat = reconstruct(y_corrected, mask_corrected, method, device, iters)
        dt = time.time() - t0

        if s["x_true"] is not None:
            psnr = compute_psnr(s["x_true"], x_hat)
            ssim_val = compute_ssim(s["x_true"], x_hat)
        else:
            psnr, ssim_val = 0.0, 0.0
        cons = compute_consistency(s["y"], x_hat, mask_corrected, gain, offset)
        comp = compute_composite(psnr, ssim_val, cons)

        scene = s["metadata"].get("scene", s["key"])
        print(f"    [{i+1:2d}/{len(samples)}] {scene:<14s}  "
              f"PSNR={psnr:6.2f}  SSIM={ssim_val:.4f}  Cons={cons:.4f}  "
              f"Score={comp:.4f}  ({dt:.1f}s)")
        scores.append({"psnr": psnr, "ssim": ssim_val,
                        "consistency": cons, "composite": comp, "time": dt})
        del x_hat; gc.collect()
    return scores, est_spec


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InverseNet CACTI Benchmark")
    parser.add_argument("--samples", type=int, default=0,
                        help="Max samples to process (0=all)")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "ideal", "baseline", "oracle", "blind"],
                        help="Which InverseNet scenario(s) to run")
    parser.add_argument("--method", default="all",
                        help="Solver name or 'all'")
    parser.add_argument("--iters", type=int, default=100,
                        help="GAP-TV iterations")
    parser.add_argument("--tier", default="public",
                        choices=["public", "dev", "hidden"],
                        help="Dataset tier")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cpu, cuda:0)")
    parser.add_argument("--skip-methods", default="",
                        help="Comma-separated methods to skip (e.g. elp_unfolding,hisvit9)")
    parser.add_argument("--blind-spec", default="",
                        help="Path to pre-computed blind calibration spec JSON "
                             "(from blind_calibrate_cacti.py). Skips calibration phase.")
    args = parser.parse_args()

    if args.device == "auto":
        device = _resolve_device("auto")
    else:
        device = args.device
    print(f"Device: {device}")

    # Check solver availability
    cs = _cacti_solvers
    print("\n--- Solver availability ---")
    available_real = []
    available_fallback = []
    for name in SOLVERS:
        display = SOLVER_DISPLAY.get(name, name)
        if name == "gap_tv":
            print(f"  {display:<16s}  [NATIVE]  Classical TV denoiser")
            available_real.append(name)
        elif name == "pnp_ffdnet":
            if os.path.isfile(cs._DNCNN_WEIGHTS):
                print(f"  {display:<16s}  [DnCNN]   Pretrained denoiser")
                available_real.append(name)
            elif os.path.isfile(cs._FFDNET_WEIGHTS):
                print(f"  {display:<16s}  [FFDNet]  Pretrained denoiser")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> GAP-TV (no weights)")
                available_fallback.append(name)
        elif name == "elp_unfolding":
            if os.path.isfile(cs._ELP_CKPT):
                print(f"  {display:<16s}  [NATIVE]  ECCV 2022 (565M params)")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> 2-pass GAP-TV")
                available_fallback.append(name)
        elif name == "efficient_sci":
            if os.path.isfile(cs._ESCI_CKPT):
                print(f"  {display:<16s}  [NATIVE]  CVPR 2023")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> 3-pass GAP-TV")
                available_fallback.append(name)
        elif name in ("hisvit9", "hisvit13"):
            ckpt = cs._HISVIT9_CKPT if name == "hisvit9" else cs._HISVIT13_CKPT
            if os.path.isfile(ckpt):
                print(f"  {display:<16s}  [NATIVE]  ECCV 2024")
                available_real.append(name)
            else:
                print(f"  {display:<16s}  [FALLBACK] -> 3-pass GAP-TV")
                available_fallback.append(name)

    # Noise model info
    print("\n--- Dataset noise model ---")
    print("  Model: poisson_gaussian (peak_photon=10000, sigma=1.0)")
    print("  This matches the InverseNet paper (~40 dB measurement SNR).")
    print("  Standard accelerated GAP used for all scenarios.")

    # Select methods
    if args.method == "all":
        methods = list(SOLVERS.keys())
    else:
        methods = [args.method]

    # Load data
    max_samples = args.samples if args.samples > 0 else None
    tier = args.tier
    print(f"\n--- Loading {tier} tier data ---")
    samples = load_tier(tier, max_samples)
    print(f"  Loaded {len(samples)} samples ({samples[0]['mask'].shape})")

    # Skip HiSViT for large images (needs >6GB VRAM for 512×512)
    img_h = samples[0]["mask"].shape[0]
    if img_h > 256 and args.method == "all":
        skip = {"hisvit9", "hisvit13"}
        methods = [m for m in methods if m not in skip]
        print(f"  NOTE: Skipping HiSViT methods (VRAM too small for {img_h}x{img_h})")

    # Apply --skip-methods
    if args.skip_methods:
        skip_set = {m.strip() for m in args.skip_methods.split(",")}
        methods = [m for m in methods if m not in skip_set]
        print(f"  NOTE: Skipping methods: {skip_set}")
    if samples[0]["true_spec"]:
        print(f"  True spec: {samples[0]['true_spec']}")

    # Select scenarios
    scenarios = {
        "ideal": ("Scenario I  (Ideal)", run_scenario_i),
        "baseline": ("Scenario II (Baseline)", run_scenario_ii),
        "oracle": ("Scenario III (Oracle)", run_scenario_iii),
        "blind": ("Scenario IV (Blind Cal)", run_scenario_iv),
    }
    if args.scenario == "all":
        run_scenarios = list(scenarios.keys())
    else:
        run_scenarios = [args.scenario]

    # ── Run all combinations ─────────────────────────────────────────────
    all_results = {}

    for sc_key in run_scenarios:
        sc_name, sc_func = scenarios[sc_key]
        all_results[sc_key] = {}

        print(f"\n{'='*78}")
        print(f"  {sc_name}")
        print(f"{'='*78}")

        # For Scenario IV, use pre-computed spec if provided
        cached_spec = None
        if sc_key == "blind" and args.blind_spec:
            with open(args.blind_spec) as f:
                cached_spec = json.load(f)
            print(f"  Using pre-computed blind spec from: {args.blind_spec}")
            print(f"  Spec: {cached_spec}")

        for method in methods:
            display = SOLVER_DISPLAY.get(method, method)
            print(f"\n  --- {display} ---")

            try:
                t_total = time.time()
                if sc_key == "blind":
                    result = sc_func(samples, method, device, iters=args.iters,
                                     _cached_spec=cached_spec)
                    scores, cached_spec = result
                else:
                    scores = sc_func(samples, method, device, iters=args.iters)
                dt_total = time.time() - t_total

                avg = {k: float(np.mean([s[k] for s in scores]))
                       for k in ("psnr", "ssim", "consistency", "composite", "time")}
                avg["total_time"] = dt_total
                all_results[sc_key][method] = avg

                print(f"\n    AVG: PSNR={avg['psnr']:.2f} dB  SSIM={avg['ssim']:.4f}  "
                      f"Cons={avg['consistency']:.4f}  Score={avg['composite']:.4f}  "
                      f"({dt_total:.1f}s total)")
            except (MemoryError, RuntimeError, Exception) as e:
                print(f"\n    FAILED: {type(e).__name__}: {e}")
                print(f"    Skipping {display} for {sc_name}")
                gc.collect()

    # ── Final comparison tables ──────────────────────────────────────────
    print(f"\n\n{'#'*78}")
    print(f"#  INVERSENET CACTI BENCHMARK - FINAL RESULTS")
    shape_str = "x".join(str(d) for d in samples[0]["mask"].shape)
    print(f"#  Dataset: {tier} tier ({len(samples)} samples, {shape_str})")
    print(f"#  Noise: peak_photon=10000 (~40 dB SNR, matching InverseNet paper)")
    print(f"#  Device: {device}")
    print(f"{'#'*78}")

    for sc_key in run_scenarios:
        sc_name = scenarios[sc_key][0]
        results = all_results[sc_key]

        print(f"\n{'='*78}")
        print(f"  {sc_name}")
        print(f"{'='*78}")
        print(f"  {'Method':<20s}  {'PSNR (dB)':>10s}  {'SSIM':>8s}  "
              f"{'Consist':>10s}  {'Score':>8s}  {'Time':>8s}")
        print(f"  {'-'*72}")

        for method in methods:
            if method not in results:
                continue
            avg = results[method]
            display = SOLVER_DISPLAY.get(method, method)
            fallback = " *" if method in available_fallback else ""
            print(f"  {display + fallback:<20s}  {avg['psnr']:>10.2f}  {avg['ssim']:>8.4f}  "
                  f"{avg['consistency']:>10.4f}  {avg['composite']:>8.4f}  "
                  f"{avg['total_time']:>7.1f}s")

        # Compare with paper baselines
        paper_key = f"scenario_{'ii' if sc_key == 'baseline' else 'iii' if sc_key in ('oracle', 'blind') else ''}"
        if paper_key in PAPER_BASELINES:
            print(f"\n  --- InverseNet Paper Reference ---")
            print(f"  {'Method':<20s}  {'Paper PSNR':>10s}  {'Paper SSIM':>10s}")
            print(f"  {'-'*44}")
            for method in methods:
                if method not in results:
                    continue
                display = SOLVER_DISPLAY.get(method, method)
                if display in PAPER_BASELINES[paper_key]:
                    paper = PAPER_BASELINES[paper_key][display]
                    print(f"  {display:<20s}  {paper['psnr']:>10.2f}  {paper['ssim']:>10.3f}")

    if available_fallback:
        print(f"\n  * Methods marked with '*' used GAP-TV fallback (no pretrained weights).")
        print(f"    To get paper-grade results, download checkpoints to:")
        print(f"    {_ROOT / 'checkpoint'}/")

    print(f"\n  NOTE: Dataset uses peak_photon=10000 noise (~40 dB SNR),")
    print(f"  matching the InverseNet paper. Results should be close to paper values.")

    print(f"\n{'='*78}")
    print(f"  DONE")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
