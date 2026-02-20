#!/usr/bin/env python3
"""Scenario IV: Grid-Search Calibration Baseline for InverseNet ECCV 2026.

For each modality, grid-searches over mismatch parameters using the
measurement residual as the objective function.  This provides a practical
calibration baseline between Scenario II (uncorrected) and Scenario III
(oracle).

  CASSI : search dx, dy grid with GAP-TV inner loop
  CACTI : search mask shift + timing with GAP-TV inner loop
  SPC   : search gain_alpha with FISTA-TV inner loop

Usage:
    python run_scenario_iv.py [--device cuda:0] [--modality all|cassi|cacti|spc]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CASSI Scenario IV
# ============================================================================
def cassi_scenario_iv(device: str = "cuda:0") -> Dict:
    """Grid-search calibration for CASSI using GAP-TV.

    Searches dx, dy on a grid [-1.0, 1.0] with 0.1 step.
    Inner loop: GAP-TV with 50 iterations (fast for grid search).
    Outer metric: measurement residual ||y - Phi(x_hat)||^2.
    """
    import scipy.io as sio
    from scipy.ndimage import affine_transform

    DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
    from skimage.restoration import denoise_tv_chambolle

    logger.info("\n" + "=" * 60)
    logger.info("CASSI Scenario IV: Grid-Search Calibration")
    logger.info("=" * 60)

    # Load mask and first 3 scenes (for speed)
    mask_data = sio.loadmat(str(DATASET_SIMU / "mask.mat"))
    mask_ideal = mask_data["mask"].astype(np.float32)
    H, W = mask_ideal.shape
    nC, step = 28, 2
    W_ext = W + (nC - 1) * step

    # True mismatch params
    true_dx, true_dy, true_theta = 0.5, 0.3, 0.1

    def warp_mask(m, dx, dy, theta=0.0):
        cx, cy = W / 2.0, H / 2.0
        th = np.radians(theta)
        cos_t, sin_t = np.cos(th), np.sin(th)
        mat = np.array([
            [cos_t, sin_t, -cx * cos_t - cy * sin_t + cx + dx],
            [-sin_t, cos_t, cx * sin_t - cy * cos_t + cy + dy],
        ])
        inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
        return np.clip(affine_transform(m, inv[:2, :2], offset=inv[:2, 2], cval=0, order=1), 0, 1).astype(np.float32)

    def cassi_fwd(scene, mask):
        y = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            y[:, k * step:k * step + W] += mask * scene[:, :, k]
        return y

    def gap_tv_fast(y, mask, iters=50):
        Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            Phi_sum[:, k * step:k * step + W] += mask ** 2
        Phi_sum = np.maximum(Phi_sum, 1e-10)
        x = np.zeros((H, W, nC), dtype=np.float32)
        for k in range(nC):
            x[:, :, k] = mask * y[:, k * step:k * step + W] / np.maximum(
                Phi_sum[:, k * step:k * step + W], 1e-6)
        y1 = np.zeros((H, W_ext), dtype=np.float32)
        for _ in range(iters):
            y_est = np.zeros((H, W_ext), dtype=np.float32)
            for k in range(nC):
                y_est[:, k * step:k * step + W] += mask * x[:, :, k]
            y1 += (y[:, :W_ext] - y_est)
            norm_r = (y1 - y_est) / Phi_sum
            for k in range(nC):
                x[:, :, k] += mask * norm_r[:, k * step:k * step + W]
            x = denoise_tv_chambolle(np.clip(x, 0, None), weight=0.1,
                                      max_num_iter=3, channel_axis=2).astype(np.float32)
        return np.clip(x, 0, 1).astype(np.float32)

    # Load scenes
    scenes = []
    for si in range(1, 4):  # 3 scenes for speed
        path = DATASET_SIMU / "Truth" / f"scene{si:02d}.mat"
        if not path.exists():
            continue
        d = sio.loadmat(str(path))
        for key in ["img", "Img", "scene", "data"]:
            if key in d:
                scenes.append(d[key].astype(np.float32))
                break

    if not scenes:
        logger.warning("No CASSI scenes found")
        return {}

    logger.info(f"Loaded {len(scenes)} scenes")

    # Generate corrupted measurements with true mismatch
    mask_true = warp_mask(mask_ideal, true_dx, true_dy, true_theta)
    y_corrupts = [cassi_fwd(s, mask_true) for s in scenes]

    # Grid search over dx, dy
    dx_grid = np.arange(-1.0, 1.1, 0.2)
    dy_grid = np.arange(-1.0, 1.1, 0.2)

    logger.info(f"Grid: {len(dx_grid)}x{len(dy_grid)} = {len(dx_grid)*len(dy_grid)} points")

    best_dx, best_dy = 0.0, 0.0
    best_residual = float("inf")
    grid_results = []

    t0 = time.time()
    for dx_test in dx_grid:
        for dy_test in dy_grid:
            mask_test = warp_mask(mask_ideal, dx_test, dy_test)
            total_residual = 0.0
            for si, y_c in enumerate(y_corrupts):
                x_hat = gap_tv_fast(y_c, mask_test, iters=30)
                y_pred = cassi_fwd(x_hat, mask_test)
                ww = min(y_c.shape[1], y_pred.shape[1])
                total_residual += np.sum((y_c[:, :ww] - y_pred[:, :ww]) ** 2)

            grid_results.append({
                "dx": round(float(dx_test), 2),
                "dy": round(float(dy_test), 2),
                "residual": round(float(total_residual), 1),
            })

            if total_residual < best_residual:
                best_residual = total_residual
                best_dx, best_dy = dx_test, dy_test

    calibration_time = time.time() - t0

    logger.info(f"Best: dx={best_dx:.2f}, dy={best_dy:.2f} "
                f"(true: dx={true_dx}, dy={true_dy})")
    logger.info(f"Grid search time: {calibration_time:.1f}s")

    # Evaluate Scenario IV with calibrated mask
    mask_cal = warp_mask(mask_ideal, best_dx, best_dy)
    psnrs_iv = []
    for si, (scene, y_c) in enumerate(zip(scenes, y_corrupts)):
        x_hat = gap_tv_fast(y_c, mask_cal, iters=100)
        mse = float(np.mean((scene - x_hat) ** 2))
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
        psnrs_iv.append(psnr)

    # Compare with Scenario II (ideal mask on corrupted meas)
    psnrs_ii = []
    for si, (scene, y_c) in enumerate(zip(scenes, y_corrupts)):
        x_hat = gap_tv_fast(y_c, mask_ideal, iters=100)
        mse = float(np.mean((scene - x_hat) ** 2))
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
        psnrs_ii.append(psnr)

    # Scenario III (true mask)
    psnrs_iii = []
    for si, (scene, y_c) in enumerate(zip(scenes, y_corrupts)):
        x_hat = gap_tv_fast(y_c, mask_true, iters=100)
        mse = float(np.mean((scene - x_hat) ** 2))
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
        psnrs_iii.append(psnr)

    result = {
        "modality": "cassi",
        "method": "gap_tv",
        "true_params": {"dx": true_dx, "dy": true_dy, "theta": true_theta},
        "estimated_params": {"dx": round(best_dx, 2), "dy": round(best_dy, 2)},
        "scenario_ii_psnr": round(float(np.mean(psnrs_ii)), 2),
        "scenario_iv_psnr": round(float(np.mean(psnrs_iv)), 2),
        "scenario_iii_psnr": round(float(np.mean(psnrs_iii)), 2),
        "calibration_time_s": round(calibration_time, 1),
        "grid_size": len(grid_results),
    }

    logger.info(f"\nCASSI GAP-TV:  II={result['scenario_ii_psnr']:.2f}  "
                f"IV={result['scenario_iv_psnr']:.2f}  "
                f"III={result['scenario_iii_psnr']:.2f}")

    return result


# ============================================================================
# CACTI Scenario IV
# ============================================================================
def cacti_scenario_iv(device: str = "cuda:0") -> Dict:
    """Grid-search calibration for CACTI using GAP-TV.

    Searches timing offset and mask shift on a grid.
    """
    from pwm_core.data.loaders.cacti_bench import CACTIBenchmark
    from scipy.ndimage import shift as ndi_shift
    from skimage.restoration import denoise_tv_chambolle

    logger.info("\n" + "=" * 60)
    logger.info("CACTI Scenario IV: Grid-Search Calibration")
    logger.info("=" * 60)

    # Load benchmark data (first 2 groups for speed)
    try:
        dataset = CACTIBenchmark()
    except Exception as e:
        logger.warning(f"CACTI benchmark not available: {e}")
        return {}

    # True mismatch
    true_dx, true_dy = 0.5, 0.3

    def shift_mask(mask, dx, dy):
        out = np.zeros_like(mask)
        for t in range(mask.shape[2]):
            out[:, :, t] = ndi_shift(mask[:, :, t], [dy, dx], order=1, mode="constant")
        return np.clip(out, 0, 1).astype(np.float32)

    def gap_tv_fast(y, mask, iters=50):
        H, W, nF = mask.shape
        mask_sum = np.sum(mask, axis=2)
        mask_sum[mask_sum == 0] = 1
        x = y[:, :, np.newaxis] * mask / mask_sum[:, :, np.newaxis]
        y1 = y.copy()
        for _ in range(iters):
            yb = np.sum(x * mask, axis=2)
            y1 = y1 + (y - yb)
            x = x + ((y1 - yb) / mask_sum)[:, :, np.newaxis] * mask
            for f in range(nF):
                x[:, :, f] = denoise_tv_chambolle(x[:, :, f], weight=0.1, max_num_iter=3)
            x = np.clip(x, 0, 1)
        return x.astype(np.float32)

    # Process first 2 groups
    groups = []
    for i, (name, gt, mask, meas) in enumerate(dataset):
        if i >= 2:
            break
        groups.append((name, gt, mask, meas))

    if not groups:
        logger.warning("No CACTI groups available")
        return {}

    logger.info(f"Processing {len(groups)} groups")

    # Generate corrupted measurements
    corrupted_data = []
    for name, gt, mask, meas in groups:
        mask_warped = shift_mask(mask, true_dx, true_dy)
        mask_warped_bin = (mask_warped > 0.5).astype(np.float32)
        y_corrupt = np.sum(gt * mask_warped_bin, axis=2)
        corrupted_data.append((name, gt, mask, mask_warped_bin, y_corrupt))

    # Grid search: dx, dy
    dx_grid = np.arange(-1.0, 1.1, 0.25)
    dy_grid = np.arange(-1.0, 1.1, 0.25)
    logger.info(f"Grid: {len(dx_grid)}x{len(dy_grid)}")

    best_dx, best_dy = 0.0, 0.0
    best_residual = float("inf")

    t0 = time.time()
    for dx_test in dx_grid:
        for dy_test in dy_grid:
            total_residual = 0.0
            for name, gt, mask, mask_w, y_c in corrupted_data:
                mask_test = shift_mask(mask, dx_test, dy_test)
                mask_test_bin = (mask_test > 0.5).astype(np.float32)
                x_hat = gap_tv_fast(y_c, mask_test_bin, iters=20)
                y_pred = np.sum(x_hat * mask_test_bin, axis=2)
                total_residual += np.sum((y_c - y_pred) ** 2)

            if total_residual < best_residual:
                best_residual = total_residual
                best_dx, best_dy = dx_test, dy_test

    calibration_time = time.time() - t0

    logger.info(f"Best: dx={best_dx:.2f}, dy={best_dy:.2f}")
    logger.info(f"Calibration time: {calibration_time:.1f}s")

    # Evaluate scenarios
    psnrs_ii, psnrs_iv, psnrs_iii = [], [], []

    for name, gt, mask, mask_w, y_c in corrupted_data:
        # Scenario II
        x_ii = gap_tv_fast(y_c, mask, iters=50)
        psnrs_ii.append(float(10 * np.log10(1.0 / max(np.mean((gt - x_ii)**2), 1e-10))))

        # Scenario IV
        mask_cal = shift_mask(mask, best_dx, best_dy)
        mask_cal_bin = (mask_cal > 0.5).astype(np.float32)
        x_iv = gap_tv_fast(y_c, mask_cal_bin, iters=50)
        psnrs_iv.append(float(10 * np.log10(1.0 / max(np.mean((gt - x_iv)**2), 1e-10))))

        # Scenario III
        x_iii = gap_tv_fast(y_c, mask_w, iters=50)
        psnrs_iii.append(float(10 * np.log10(1.0 / max(np.mean((gt - x_iii)**2), 1e-10))))

    result = {
        "modality": "cacti",
        "method": "gap_tv",
        "true_params": {"dx": true_dx, "dy": true_dy},
        "estimated_params": {"dx": round(best_dx, 2), "dy": round(best_dy, 2)},
        "scenario_ii_psnr": round(float(np.mean(psnrs_ii)), 2),
        "scenario_iv_psnr": round(float(np.mean(psnrs_iv)), 2),
        "scenario_iii_psnr": round(float(np.mean(psnrs_iii)), 2),
        "calibration_time_s": round(calibration_time, 1),
    }

    logger.info(f"\nCACTI GAP-TV:  II={result['scenario_ii_psnr']:.2f}  "
                f"IV={result['scenario_iv_psnr']:.2f}  "
                f"III={result['scenario_iii_psnr']:.2f}")

    return result


# ============================================================================
# SPC Scenario IV
# ============================================================================
def spc_scenario_iv(device: str = "cuda:0") -> Dict:
    """Grid-search calibration for SPC using FISTA-TV.

    Searches gain_alpha over a grid.
    """
    from skimage.restoration import denoise_tv_chambolle

    logger.info("\n" + "=" * 60)
    logger.info("SPC Scenario IV: Grid-Search Calibration")
    logger.info("=" * 60)

    n = 64
    m = int(n * n * 0.25)

    np.random.seed(42)
    x_true = np.random.rand(n, n).astype(np.float32)
    # Make a smoother test image
    from scipy.ndimage import gaussian_filter
    x_true = gaussian_filter(x_true, sigma=2.0)
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min())
    x_flat = x_true.ravel()

    A = np.random.randn(m, n * n).astype(np.float32) / np.sqrt(m)

    # True mismatch: gain drift
    true_alpha = 0.0015
    gain_true = 1.0 + true_alpha * np.exp(np.arange(m, dtype=np.float32) * true_alpha)
    y_corrupt = gain_true * (A @ x_flat) + 0.03 * np.random.randn(m).astype(np.float32)

    def fista_tv(y, A, gain=1.0, iters=200, lam=0.005):
        y_corr = y / np.maximum(gain, 0.01)
        AtA = A.T @ A
        Aty = A.T @ y_corr
        x = np.zeros(n * n, dtype=np.float32)
        L = np.linalg.norm(AtA, ord=2)
        t = 1.0
        x_prev = x.copy()
        for _ in range(iters):
            grad = AtA @ x - Aty
            x_new = x - grad / L
            x_2d = np.clip(x_new.reshape(n, n), 0, 1)
            x_2d = denoise_tv_chambolle(x_2d, weight=lam / L, max_num_iter=3)
            x_new = x_2d.ravel()
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            x = x_new + (t - 1) / t_new * (x_new - x_prev)
            x_prev = x_new
            t = t_new
        return np.clip(x, 0, 1).astype(np.float32)

    # Grid search over alpha
    alpha_grid = np.linspace(0.0, 0.005, 21)
    logger.info(f"Grid: {len(alpha_grid)} alpha values")

    best_alpha = 0.0
    best_residual = float("inf")

    t0 = time.time()
    for alpha_test in alpha_grid:
        gain_test = 1.0 + alpha_test * np.exp(np.arange(m, dtype=np.float32) * alpha_test)
        x_hat = fista_tv(y_corrupt, A, gain=gain_test, iters=100)
        y_pred = gain_test * (A @ x_hat)
        residual = np.sum((y_corrupt - y_pred) ** 2)
        if residual < best_residual:
            best_residual = residual
            best_alpha = alpha_test

    calibration_time = time.time() - t0

    logger.info(f"Best alpha: {best_alpha:.4f} (true: {true_alpha:.4f})")
    logger.info(f"Calibration time: {calibration_time:.1f}s")

    # Evaluate scenarios
    # II: no correction
    x_ii = fista_tv(y_corrupt, A, gain=1.0, iters=200)
    psnr_ii = float(10 * np.log10(1.0 / max(np.mean((x_flat - x_ii)**2), 1e-10)))

    # IV: calibrated
    gain_cal = 1.0 + best_alpha * np.exp(np.arange(m, dtype=np.float32) * best_alpha)
    x_iv = fista_tv(y_corrupt, A, gain=gain_cal, iters=200)
    psnr_iv = float(10 * np.log10(1.0 / max(np.mean((x_flat - x_iv)**2), 1e-10)))

    # III: oracle
    x_iii = fista_tv(y_corrupt, A, gain=gain_true, iters=200)
    psnr_iii = float(10 * np.log10(1.0 / max(np.mean((x_flat - x_iii)**2), 1e-10)))

    result = {
        "modality": "spc",
        "method": "fista_tv",
        "true_params": {"alpha": true_alpha},
        "estimated_params": {"alpha": round(best_alpha, 4)},
        "scenario_ii_psnr": round(psnr_ii, 2),
        "scenario_iv_psnr": round(psnr_iv, 2),
        "scenario_iii_psnr": round(psnr_iii, 2),
        "calibration_time_s": round(calibration_time, 1),
    }

    logger.info(f"\nSPC FISTA-TV:  II={psnr_ii:.2f}  IV={psnr_iv:.2f}  III={psnr_iii:.2f}")

    return result


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Scenario IV: Grid-Search Calibration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modality", default="all", choices=["all", "cassi", "cacti", "spc"])
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("InverseNet Scenario IV: Grid-Search Calibration Baseline")
    logger.info("=" * 60)

    results = {}

    if args.modality in ("all", "cassi"):
        results["cassi"] = cassi_scenario_iv(args.device)

    if args.modality in ("all", "cacti"):
        results["cacti"] = cacti_scenario_iv(args.device)

    if args.modality in ("all", "spc"):
        results["spc"] = spc_scenario_iv(args.device)

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("SCENARIO IV SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Modality':10s} {'Method':12s} {'II':>8s} {'IV':>8s} {'III':>8s} {'IV-II':>8s} {'III-II':>8s}")
    logger.info("-" * 60)
    for mod, r in results.items():
        if r:
            p_ii = r["scenario_ii_psnr"]
            p_iv = r["scenario_iv_psnr"]
            p_iii = r["scenario_iii_psnr"]
            logger.info(f"{mod:10s} {r['method']:12s} {p_ii:8.2f} {p_iv:8.2f} {p_iii:8.2f} "
                        f"{p_iv - p_ii:+8.2f} {p_iii - p_ii:+8.2f}")

    # Save
    out_path = RESULTS_DIR / "scenario_iv_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
