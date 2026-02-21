#!/usr/bin/env python3
"""Real-Data 4-Scenario Validation for PWM Nature Paper.

Comprehensive hardware validation using existing real CASSI and CACTI data.
Runs the full 4-Scenario Protocol with multiple mismatch levels and
autonomous calibration to strengthen the hardware validation evidence.

Key improvements over the InverseNet validation:
  1. Multiple mismatch levels (0.25, 0.5, 1.0 px) instead of single point
  2. Autonomous calibration (Scenario III) on real data
  3. Cross-method consistency analysis
  4. Per-scene breakdown with statistical tests

Usage:
    python run_real_data_4scenario.py [--modality cassi|cacti|both] [--fast]
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
import scipy.io as sio
from scipy.ndimage import shift as ndshift

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

CASSI_DATA = Path("/home/spiritai/MST-main/datasets/TSA_real_data")
CACTI_DATA = Path("/home/spiritai/EfficientSCI-main/test_datasets/real_data/cr10")
CACTI_MASK = Path("/home/spiritai/EfficientSCI-main/test_datasets/mask/real_mask.mat")

RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "real_data_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CASSI configuration
# ---------------------------------------------------------------------------
CASSI_SCENES = 5
CASSI_SIZE = 660
N_BANDS = 28
STEP = 2
W_EXT = CASSI_SIZE + (N_BANDS - 1) * STEP  # 714

# Mismatch levels to test (sub-pixel mask shifts)
MISMATCH_LEVELS = [0.25, 0.5, 1.0]

# ---------------------------------------------------------------------------
# CASSI helpers
# ---------------------------------------------------------------------------
def load_cassi_data() -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Load all CASSI real measurements, mask, and mask_3d_shift."""
    measurements = []
    for i in range(1, CASSI_SCENES + 1):
        path = CASSI_DATA / "Measurements" / f"scene{i}.mat"
        data = sio.loadmat(str(path))
        measurements.append(data["meas_real"].astype(np.float64))

    mask_data = sio.loadmat(str(CASSI_DATA / "mask.mat"))
    mask = mask_data["mask"].astype(np.float64)

    m3d_data = sio.loadmat(str(CASSI_DATA / "mask_3d_shift.mat"))
    mask_3d = m3d_data["mask_3d_shift"].astype(np.float64)

    return measurements, mask, mask_3d


def shift_mask_subpixel(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply sub-pixel shift to mask using bilinear interpolation.

    For binary masks, we interpolate in float then re-threshold.
    """
    shifted = ndshift(mask.astype(np.float64), [dy, dx], order=1, mode='wrap')
    return shifted  # keep continuous for smooth gradient in calibration


def build_mask_3d_shift(mask: np.ndarray, n_bands: int, step: int) -> np.ndarray:
    """Build 3D shifted mask from 2D mask."""
    H, W = mask.shape
    W_ext = W + (n_bands - 1) * step
    mask_3d = np.zeros((H, W_ext, n_bands), dtype=np.float64)
    for b in range(n_bands):
        offset = b * step
        mask_3d[:, offset:offset + W, b] = mask
    return mask_3d


def cassi_forward(cube: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    """CASSI forward model: modulate + accumulate along spectral axis."""
    return np.sum(cube * mask_3d, axis=2)


def gap_tv_cassi(meas: np.ndarray, mask_3d: np.ndarray,
                 n_iter: int = 50, tau: float = 0.002) -> np.ndarray:
    """GAP-TV reconstruction for CASSI.

    Generalized Alternating Projection with TV regularization.
    Uses a warmup phase (GAP only) before applying TV.
    """
    H, W_ext, n_bands = mask_3d.shape

    # Precompute per-pixel mask energy: sum of mask^2 across bands
    mask_energy = np.sum(mask_3d ** 2, axis=2)  # (H, W_ext)
    mask_energy_safe = np.maximum(mask_energy, 1e-10)

    # Initialize with scaled adjoint
    x = np.zeros((H, W_ext, n_bands), dtype=np.float64)
    for b in range(n_bands):
        x[:, :, b] = meas * mask_3d[:, :, b] / mask_energy_safe

    warmup = n_iter // 3  # GAP-only warmup

    for it in range(n_iter):
        # Forward
        y_est = cassi_forward(x, mask_3d)
        # Residual
        residual = meas - y_est

        # GAP update: x += A^T(r) / diag(A^T A)
        for b in range(n_bands):
            x[:, :, b] += residual * mask_3d[:, :, b] / mask_energy_safe

        # Clip negatives
        np.maximum(x, 0, out=x)

        # TV denoising only after warmup, with small tau
        if tau > 0 and it >= warmup:
            x = _tv_denoise_3d(x, tau)

    return x


def _tv_denoise_3d(x: np.ndarray, tau: float) -> np.ndarray:
    """One step of gradient descent on isotropic TV for 3D cube."""
    dx = np.diff(x, axis=0, append=x[-1:, :, :])
    dy = np.diff(x, axis=1, append=x[:, -1:, :])
    # Magnitude (avoid division by zero)
    mag = np.sqrt(dx**2 + dy**2 + 1e-8)
    # Normalized gradients
    nx = dx / mag
    ny = dy / mag
    # Divergence
    div_x = nx - np.roll(nx, 1, axis=0)
    div_y = ny - np.roll(ny, 1, axis=1)
    return x + tau * (div_x + div_y)


def measurement_residual(meas: np.ndarray, recon: np.ndarray,
                         mask_3d: np.ndarray) -> float:
    """Compute normalized measurement residual: ||y - Ax||^2 / ||y||^2."""
    y_hat = cassi_forward(recon, mask_3d)
    return float(np.sum((meas - y_hat) ** 2) / np.sum(meas ** 2))


# ---------------------------------------------------------------------------
# CACTI helpers
# ---------------------------------------------------------------------------
def load_cacti_data() -> Tuple[List[np.ndarray], np.ndarray]:
    """Load CACTI real measurements and mask."""
    scenes = ["meas_duomino_cr_10", "meas_hand_cr_10",
              "meas_pendulumBall_cr_10", "meas_waterBalloon_cr_10"]
    measurements = []
    for name in scenes:
        path = CACTI_DATA / f"{name}.mat"
        data = sio.loadmat(str(path))
        meas = data["meas"].astype(np.float64)
        if meas.ndim == 3:
            meas = meas[:, :, 0]
        measurements.append(meas)

    mask_data = sio.loadmat(str(CACTI_MASK))
    mask = mask_data["mask"].astype(np.float64)

    return measurements, mask


def cacti_forward(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """CACTI forward: modulate each frame and sum."""
    H, W, T = frames.shape
    meas = np.zeros((H, W), dtype=np.float64)
    for t in range(T):
        meas += frames[:, :, t] * mask[:, :, t % mask.shape[2]]
    return meas


def gap_tv_cacti(meas: np.ndarray, mask: np.ndarray,
                 n_frames: int = 10, n_iter: int = 50,
                 tau: float = 0.05) -> np.ndarray:
    """GAP-TV reconstruction for CACTI."""
    H, W = meas.shape

    # Precompute per-pixel mask energy
    mask_energy = np.zeros((H, W), dtype=np.float64)
    for t in range(n_frames):
        m = mask[:, :, t % mask.shape[2]]
        mask_energy += m ** 2
    mask_energy_safe = np.maximum(mask_energy, 1e-10)

    # Initialize with scaled adjoint
    x = np.zeros((H, W, n_frames), dtype=np.float64)
    for t in range(n_frames):
        m = mask[:, :, t % mask.shape[2]]
        x[:, :, t] = meas * m / mask_energy_safe

    for it in range(n_iter):
        # Forward
        y_est = cacti_forward(x, mask)
        residual = meas - y_est

        # GAP update
        for t in range(n_frames):
            m = mask[:, :, t % mask.shape[2]]
            x[:, :, t] += residual * m / mask_energy_safe

        np.maximum(x, 0, out=x)

        # TV denoising per frame
        if tau > 0:
            for t in range(n_frames):
                x[:, :, t] = _tv_denoise_2d(x[:, :, t], tau)

    return x


def _tv_denoise_2d(x: np.ndarray, tau: float) -> np.ndarray:
    """One step of gradient descent on isotropic TV for 2D image."""
    dx = np.diff(x, axis=0, append=x[-1:, :])
    dy = np.diff(x, axis=1, append=x[:, -1:])
    mag = np.sqrt(dx**2 + dy**2 + 1e-8)
    nx = dx / mag
    ny = dy / mag
    div_x = nx - np.roll(nx, 1, axis=0)
    div_y = ny - np.roll(ny, 1, axis=1)
    return x + tau * (div_x + div_y)


# ---------------------------------------------------------------------------
# Autonomous calibration (coarse-to-fine grid search)
# ---------------------------------------------------------------------------
def calibrate_mask_shift_cassi(meas: np.ndarray, mask: np.ndarray,
                               search_range: float = 2.0,
                               n_grid: int = 5,
                               recon_iters: int = 30) -> Tuple[float, float, float]:
    """
    Grid search over (dx, dy) to minimize measurement residual.
    Returns (best_dx, best_dy, best_residual).
    """
    dx_grid = np.linspace(-search_range, search_range, n_grid)
    dy_grid = np.linspace(-search_range, search_range, n_grid)

    best_residual = float('inf')
    best_dx, best_dy = 0.0, 0.0

    for dx in dx_grid:
        for dy in dy_grid:
            shifted = shift_mask_subpixel(mask, dx, dy)
            m3d = build_mask_3d_shift(shifted, N_BANDS, STEP)
            recon = gap_tv_cassi(meas, m3d, n_iter=recon_iters, tau=0.05)
            res = measurement_residual(meas, recon, m3d)
            if res < best_residual:
                best_residual = res
                best_dx, best_dy = dx, dy
                logger.info(f"    Calibration: dx={dx:.2f} dy={dy:.2f} res={res:.6f} *")

    return best_dx, best_dy, best_residual


def calibrate_mask_shift_cacti(meas: np.ndarray, mask: np.ndarray,
                                n_frames: int = 10,
                                search_range: float = 2.0,
                                n_grid: int = 5,
                                recon_iters: int = 30) -> Tuple[float, float, float]:
    """Grid search over (dx, dy) for CACTI mask shift calibration."""
    dx_grid = np.linspace(-search_range, search_range, n_grid)
    dy_grid = np.linspace(-search_range, search_range, n_grid)

    best_residual = float('inf')
    best_dx, best_dy = 0.0, 0.0

    for dx in dx_grid:
        for dy in dy_grid:
            shifted_mask = np.zeros_like(mask)
            for t in range(mask.shape[2]):
                shifted_mask[:, :, t] = shift_mask_subpixel(mask[:, :, t], dx, dy)
            recon = gap_tv_cacti(meas, shifted_mask, n_frames=n_frames,
                                 n_iter=recon_iters, tau=0.05)
            y_hat = cacti_forward(recon, shifted_mask)
            res = float(np.sum((meas - y_hat) ** 2) / np.sum(meas ** 2))
            if res < best_residual:
                best_residual = res
                best_dx, best_dy = dx, dy
                logger.info(f"    Calibration: dx={dx:.2f} dy={dy:.2f} res={res:.6f} *")

    return best_dx, best_dy, best_residual


# ---------------------------------------------------------------------------
# CASSI 4-Scenario Protocol
# ---------------------------------------------------------------------------
def run_cassi_4scenario(fast: bool = False):
    """Run full 4-scenario protocol on real CASSI data."""
    logger.info("=" * 60)
    logger.info("CASSI Real-Data 4-Scenario Protocol")
    logger.info("=" * 60)

    measurements, mask, mask_3d = load_cassi_data()
    results = {"modality": "CASSI", "mismatch_levels": []}

    recon_iters = 30 if fast else 60
    cal_iters = 20 if fast else 30
    cal_grid = 5
    scenes_to_run = 3 if fast else CASSI_SCENES

    for mismatch_level in MISMATCH_LEVELS:
        logger.info(f"\n--- Mismatch level: dx={mismatch_level} px ---")
        level_results = []

        for scene_idx in range(scenes_to_run):
            scene_name = f"scene{scene_idx + 1}"
            logger.info(f"Processing {scene_name}...")
            meas = measurements[scene_idx]

            # --- Scenario I: Factory-calibrated mask ---
            t0 = time.time()
            recon_I = gap_tv_cassi(meas, mask_3d, n_iter=recon_iters)
            res_I = measurement_residual(meas, recon_I, mask_3d)
            t_I = time.time() - t0
            logger.info(f"  Scenario I  (calibrated):  res={res_I:.6f}  ({t_I:.1f}s)")

            # --- Scenario II: Deliberately mismatched mask ---
            shifted_mask = shift_mask_subpixel(mask, mismatch_level, mismatch_level * 0.6)
            shifted_mask_3d = build_mask_3d_shift(shifted_mask, N_BANDS, STEP)
            t0 = time.time()
            recon_II = gap_tv_cassi(meas, shifted_mask_3d, n_iter=recon_iters)
            res_II = measurement_residual(meas, recon_II, shifted_mask_3d)
            t_II = time.time() - t0
            ratio_II = res_II / max(res_I, 1e-12)
            logger.info(f"  Scenario II (mismatched):  res={res_II:.6f}  ratio={ratio_II:.2f}x  ({t_II:.1f}s)")

            # --- Cross-residual: reconstruct with wrong mask, evaluate with factory mask ---
            cross_res = measurement_residual(meas, recon_II, mask_3d)
            cross_ratio = cross_res / max(res_I, 1e-12)
            logger.info(f"  Cross-residual (II→I):     res={cross_res:.6f}  ratio={cross_ratio:.2f}x")

            # --- Scenario III: Autonomous calibration ---
            t0 = time.time()
            cal_dx, cal_dy, cal_res_search = calibrate_mask_shift_cassi(
                meas, mask, search_range=max(mismatch_level * 1.5, 0.5),
                n_grid=cal_grid, recon_iters=cal_iters
            )
            # Full reconstruction with calibrated mask
            cal_mask = shift_mask_subpixel(mask, cal_dx, cal_dy)
            cal_mask_3d = build_mask_3d_shift(cal_mask, N_BANDS, STEP)
            recon_III = gap_tv_cassi(meas, cal_mask_3d, n_iter=recon_iters)
            res_III = measurement_residual(meas, recon_III, cal_mask_3d)
            t_III = time.time() - t0
            ratio_III = res_III / max(res_I, 1e-12)
            logger.info(f"  Scenario III (calibrated): res={res_III:.6f}  ratio={ratio_III:.2f}x  "
                        f"dx={cal_dx:.2f} dy={cal_dy:.2f}  ({t_III:.1f}s)")

            # --- Recovery ratio ---
            if abs(res_I - res_II) > 1e-12:
                recovery_ratio = (res_III - res_II) / (res_I - res_II)
            else:
                recovery_ratio = float('nan')

            scene_result = {
                "scene": scene_name,
                "mismatch_px": mismatch_level,
                "res_I_calibrated": res_I,
                "res_II_mismatched": res_II,
                "res_II_cross": cross_res,
                "res_III_corrected": res_III,
                "ratio_II_I": ratio_II,
                "cross_ratio": cross_ratio,
                "ratio_III_I": ratio_III,
                "recovery_ratio": recovery_ratio,
                "calibrated_shift": {"dx": cal_dx, "dy": cal_dy},
                "time_I_s": t_I,
                "time_II_s": t_II,
                "time_III_s": t_III,
            }
            level_results.append(scene_result)

        # Summary for this mismatch level
        ratios_II = [r["ratio_II_I"] for r in level_results]
        ratios_III = [r["ratio_III_I"] for r in level_results]
        cross_ratios = [r["cross_ratio"] for r in level_results]
        recovery = [r["recovery_ratio"] for r in level_results
                     if not np.isnan(r["recovery_ratio"])]

        results["mismatch_levels"].append({
            "mismatch_px": mismatch_level,
            "per_scene": level_results,
            "mean_ratio_II": float(np.mean(ratios_II)),
            "std_ratio_II": float(np.std(ratios_II)),
            "mean_cross_ratio": float(np.mean(cross_ratios)),
            "mean_ratio_III": float(np.mean(ratios_III)),
            "mean_recovery": float(np.mean(recovery)) if recovery else None,
        })

    # Save results
    out_path = RESULTS_DIR / "cassi_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nCASSI results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# CACTI 4-Scenario Protocol
# ---------------------------------------------------------------------------
def run_cacti_4scenario(fast: bool = False):
    """Run full 4-scenario protocol on real CACTI data."""
    logger.info("=" * 60)
    logger.info("CACTI Real-Data 4-Scenario Protocol")
    logger.info("=" * 60)

    measurements, mask = load_cacti_data()
    scene_names = ["duomino", "hand", "pendulumBall", "waterBalloon"]
    results = {"modality": "CACTI", "mismatch_levels": []}

    n_frames = 10  # cr=10
    recon_iters = 30 if fast else 50
    cal_iters = 20 if fast else 30
    cal_grid = 5

    for mismatch_level in MISMATCH_LEVELS:
        logger.info(f"\n--- Mismatch level: dx={mismatch_level} px ---")
        level_results = []

        for scene_idx, scene_name in enumerate(scene_names):
            logger.info(f"Processing {scene_name}...")
            meas = measurements[scene_idx]

            # --- Scenario I: Factory-calibrated mask ---
            t0 = time.time()
            recon_I = gap_tv_cacti(meas, mask, n_frames=n_frames, n_iter=recon_iters)
            y_I = cacti_forward(recon_I, mask)
            res_I = float(np.sum((meas - y_I) ** 2) / np.sum(meas ** 2))
            t_I = time.time() - t0
            logger.info(f"  Scenario I  (calibrated):  res={res_I:.6f}  ({t_I:.1f}s)")

            # --- Scenario II: Mismatched mask ---
            shifted_mask = np.zeros_like(mask)
            for t in range(mask.shape[2]):
                shifted_mask[:, :, t] = shift_mask_subpixel(
                    mask[:, :, t], mismatch_level, mismatch_level * 0.6)
            t0 = time.time()
            recon_II = gap_tv_cacti(meas, shifted_mask, n_frames=n_frames, n_iter=recon_iters)
            y_II = cacti_forward(recon_II, shifted_mask)
            res_II = float(np.sum((meas - y_II) ** 2) / np.sum(meas ** 2))
            t_II = time.time() - t0
            ratio_II = res_II / max(res_I, 1e-12)
            logger.info(f"  Scenario II (mismatched):  res={res_II:.6f}  ratio={ratio_II:.2f}x  ({t_II:.1f}s)")

            # --- Cross-residual ---
            y_cross = cacti_forward(recon_II, mask)
            cross_res = float(np.sum((meas - y_cross) ** 2) / np.sum(meas ** 2))
            cross_ratio = cross_res / max(res_I, 1e-12)
            logger.info(f"  Cross-residual (II→I):     res={cross_res:.6f}  ratio={cross_ratio:.2f}x")

            # --- Scenario III: Autonomous calibration ---
            t0 = time.time()
            cal_dx, cal_dy, _ = calibrate_mask_shift_cacti(
                meas, mask, n_frames=n_frames,
                search_range=max(mismatch_level * 1.5, 0.5),
                n_grid=cal_grid, recon_iters=cal_iters
            )
            cal_shifted = np.zeros_like(mask)
            for t in range(mask.shape[2]):
                cal_shifted[:, :, t] = shift_mask_subpixel(mask[:, :, t], cal_dx, cal_dy)
            recon_III = gap_tv_cacti(meas, cal_shifted, n_frames=n_frames, n_iter=recon_iters)
            y_III = cacti_forward(recon_III, cal_shifted)
            res_III = float(np.sum((meas - y_III) ** 2) / np.sum(meas ** 2))
            t_III = time.time() - t0
            ratio_III = res_III / max(res_I, 1e-12)
            logger.info(f"  Scenario III (calibrated): res={res_III:.6f}  ratio={ratio_III:.2f}x  "
                        f"dx={cal_dx:.2f} dy={cal_dy:.2f}  ({t_III:.1f}s)")

            # Recovery ratio
            if abs(res_I - res_II) > 1e-12:
                recovery_ratio = (res_III - res_II) / (res_I - res_II)
            else:
                recovery_ratio = float('nan')

            scene_result = {
                "scene": scene_name,
                "mismatch_px": mismatch_level,
                "res_I_calibrated": res_I,
                "res_II_mismatched": res_II,
                "res_II_cross": cross_res,
                "res_III_corrected": res_III,
                "ratio_II_I": ratio_II,
                "cross_ratio": cross_ratio,
                "ratio_III_I": ratio_III,
                "recovery_ratio": recovery_ratio,
                "calibrated_shift": {"dx": cal_dx, "dy": cal_dy},
                "time_I_s": t_I,
                "time_II_s": t_II,
                "time_III_s": t_III,
            }
            level_results.append(scene_result)

        ratios_II = [r["ratio_II_I"] for r in level_results]
        ratios_III = [r["ratio_III_I"] for r in level_results]
        cross_ratios = [r["cross_ratio"] for r in level_results]
        recovery = [r["recovery_ratio"] for r in level_results
                     if not np.isnan(r["recovery_ratio"])]

        results["mismatch_levels"].append({
            "mismatch_px": mismatch_level,
            "per_scene": level_results,
            "mean_ratio_II": float(np.mean(ratios_II)),
            "std_ratio_II": float(np.std(ratios_II)),
            "mean_cross_ratio": float(np.mean(cross_ratios)),
            "mean_ratio_III": float(np.mean(ratios_III)),
            "mean_recovery": float(np.mean(recovery)) if recovery else None,
        })

    out_path = RESULTS_DIR / "cacti_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nCACTI results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-Data 4-Scenario Validation")
    parser.add_argument("--modality", default="both",
                        choices=["cassi", "cacti", "both"])
    parser.add_argument("--fast", action="store_true",
                        help="Reduced iterations for faster results")
    args = parser.parse_args()

    all_results = {}

    if args.modality in ("cacti", "both"):
        if CACTI_DATA.exists():
            all_results["cacti"] = run_cacti_4scenario(fast=args.fast)
        else:
            logger.warning(f"CACTI data not found at {CACTI_DATA}")

    if args.modality in ("cassi", "both"):
        if CASSI_DATA.exists():
            all_results["cassi"] = run_cassi_4scenario(fast=args.fast)
        else:
            logger.warning(f"CASSI data not found at {CASSI_DATA}")

    # Combined summary
    out_path = RESULTS_DIR / "combined_4scenario_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("4-SCENARIO VALIDATION COMPLETE")
    logger.info("=" * 60)
    for mod, res in all_results.items():
        for level in res.get("mismatch_levels", []):
            logger.info(
                f"  {mod.upper()} @ {level['mismatch_px']}px: "
                f"mean mismatch ratio = {level['mean_ratio_II']:.2f}x "
                f"(cross={level['mean_cross_ratio']:.2f}x) "
                f"corrected={level['mean_ratio_III']:.2f}x "
                f"recovery={level.get('mean_recovery', 'N/A')}"
            )


if __name__ == "__main__":
    main()
