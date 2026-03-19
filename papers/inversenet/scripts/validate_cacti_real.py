#!/usr/bin/env python3
"""CACTI Real Data Validation for InverseNet ECCV 2026.

Validates GAP-TV and PnP-FFDNet on 4 real CACTI scenes from the EfficientSCI
real_data dataset (cr=10).  Two conditions:
  - Calibrated:  use the real_mask as-is
  - Mismatched:  shift mask by (dx=0.5, dy=0.3) to simulate operator mismatch

No ground truth — uses measurement residual as quality metric:
  residual = ||y - Phi * x_hat||^2 / ||y||^2

Scenes: duomino, hand, pendulumBall, waterBalloon (cr=10)
Mask: real_mask.mat (512x512x50, first 10 frames used)

Usage:
    python validate_cacti_real.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

CACTI_REAL_DIR = Path("/home/spiritai/EfficientSCI-main/test_datasets/real_data")
CACTI_MASK_DIR = Path("/home/spiritai/EfficientSCI-main/test_datasets/mask")
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR = RESULTS_DIR / "cacti_real_reconstructions"
RECON_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CR = 10  # compression ratio
SCENES = ["duomino", "hand", "pendulumBall", "waterBalloon"]

METHOD_LABELS = {
    "gap_tv": "GAP-TV",
    "pnp_ffdnet": "PnP-FFDNet",
}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_real_mask(cr: int = 10) -> Optional[np.ndarray]:
    """Load real CACTI mask (512x512x50), crop to first `cr` frames."""
    path = CACTI_MASK_DIR / "real_mask.mat"
    if not path.exists():
        logger.error(f"Mask not found: {path}")
        return None
    data = sio.loadmat(str(path))
    mask = data["mask"].astype(np.float32)  # (512, 512, 50)
    return mask[:, :, :cr]  # (512, 512, cr)


def load_real_measurement(scene_name: str, cr: int = 10) -> Optional[np.ndarray]:
    """Load real CACTI measurement from cr10/ directory.

    Each scene has shape (512, 512) or (512, 512, N_snapshots).
    For scenes with multiple snapshots, we use the first one.
    Returns single 2D measurement (512, 512).
    """
    path = CACTI_REAL_DIR / f"cr{cr}" / f"meas_{scene_name}_cr_{cr}.mat"
    if not path.exists():
        logger.warning(f"Measurement not found: {path}")
        return None
    data = sio.loadmat(str(path))
    meas = data["meas"].astype(np.float32)

    if meas.ndim == 3:
        # Multiple snapshots — use first one
        meas = meas[:, :, 0]

    # Normalise to [0, 1]
    meas_max = meas.max()
    if meas_max > 0:
        meas = meas / meas_max

    return meas


def shift_mask_3d(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift each temporal frame of a 3D mask by (dx, dy)."""
    from scipy.ndimage import shift as ndi_shift
    out = np.zeros_like(mask)
    for t in range(mask.shape[2]):
        out[:, :, t] = ndi_shift(mask[:, :, t], [dy, dx], order=1,
                                  mode="constant", cval=0.0)
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# metrics (no-reference)
# ---------------------------------------------------------------------------
def compute_measurement_residual(y: np.ndarray, x_hat: np.ndarray,
                                  mask: np.ndarray) -> float:
    """Relative measurement residual: ||y - Phi*x_hat||^2 / ||y||^2."""
    y_pred = np.sum(x_hat * mask, axis=2)
    residual = np.sum((y - y_pred) ** 2)
    y_norm = np.sum(y ** 2)
    if y_norm < 1e-10:
        return 0.0
    return float(residual / y_norm)


def compute_tv(x: np.ndarray) -> float:
    """Total variation of reconstruction (lower = smoother)."""
    tv = 0.0
    for t in range(x.shape[2]):
        frame = x[:, :, t]
        tv += np.sum(np.abs(np.diff(frame, axis=0)))
        tv += np.sum(np.abs(np.diff(frame, axis=1)))
    return float(tv)


# ---------------------------------------------------------------------------
# reconstruction: GAP-TV
# ---------------------------------------------------------------------------
def gap_tv_cacti_real(y: np.ndarray, mask: np.ndarray,
                      iterations: int = 100, tv_weight: float = 0.15,
                      tv_iter: int = 5) -> np.ndarray:
    """GAP-TV for real CACTI data."""
    from skimage.restoration import denoise_tv_chambolle

    H, W, nF = mask.shape
    mask_sum = np.sum(mask, axis=2)
    mask_sum[mask_sum == 0] = 1

    x = y[:, :, np.newaxis] * mask / mask_sum[:, :, np.newaxis]
    y1 = y.copy()

    for _ in range(iterations):
        yb = np.sum(x * mask, axis=2)
        y1 = y1 + (y - yb)
        residual = (y1 - yb) / mask_sum
        x = x + residual[:, :, np.newaxis] * mask

        for f in range(nF):
            x[:, :, f] = denoise_tv_chambolle(
                x[:, :, f], weight=tv_weight, max_num_iter=tv_iter)

        x = np.clip(x, 0, 1)

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# reconstruction: PnP-FFDNet (from pwm_core)
# ---------------------------------------------------------------------------
def pnp_ffdnet_cacti_real(y: np.ndarray, mask: np.ndarray,
                           device: str = "cuda:0") -> np.ndarray:
    """PnP-FFDNet reconstruction for real CACTI data.

    Uses the GAP + FFDNet deep denoiser from pwm_core.recon.cacti_solvers.
    """
    from pwm_core.recon.cacti_solvers import pnp_ffdnet_cacti
    return pnp_ffdnet_cacti(y, mask, device=device)


# ---------------------------------------------------------------------------
# per-scene validation
# ---------------------------------------------------------------------------
def validate_scene(scene_name: str, meas: np.ndarray, mask: np.ndarray,
                   methods: List[str], device: str,
                   dx_mismatch: float = 0.5,
                   dy_mismatch: float = 0.3) -> Dict:
    """Validate one real CACTI scene."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Scene: {scene_name}  meas={meas.shape}  mask={mask.shape}")
    logger.info(f"{'='*60}")

    result = {"scene": scene_name, "calibrated": {}, "mismatched": {}}

    # ---- Calibrated ----
    logger.info("  Calibrated:")
    for method in methods:
        t0 = time.time()
        try:
            if method == "gap_tv":
                x_hat = gap_tv_cacti_real(meas, mask)
            elif method == "pnp_ffdnet":
                x_hat = pnp_ffdnet_cacti_real(meas, mask, device)
            else:
                raise ValueError(f"Unknown method: {method}")

            residual = compute_measurement_residual(meas, x_hat, mask)
            tv = compute_tv(x_hat)
            result["calibrated"][method] = {
                "residual": round(residual, 6),
                "tv": round(tv, 1),
            }

            # Save reconstruction
            np.save(str(RECON_DIR / f"{scene_name}_calibrated_{method}.npy"), x_hat)
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            result["calibrated"][method] = {"residual": 1.0, "tv": 0.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:14s}: residual={result['calibrated'][method]['residual']:.6f}  "
                     f"TV={result['calibrated'][method]['tv']:.1f}  ({dt:.1f}s)")

    # ---- Mismatched ----
    mask_shifted = shift_mask_3d(mask, dx_mismatch, dy_mismatch)
    logger.info(f"  Mismatched (dx={dx_mismatch}, dy={dy_mismatch}):")
    for method in methods:
        t0 = time.time()
        try:
            if method == "gap_tv":
                x_hat = gap_tv_cacti_real(meas, mask_shifted)
            elif method == "pnp_ffdnet":
                x_hat = pnp_ffdnet_cacti_real(meas, mask_shifted, device)
            else:
                raise ValueError(f"Unknown method: {method}")

            residual = compute_measurement_residual(meas, x_hat, mask_shifted)
            tv = compute_tv(x_hat)
            result["mismatched"][method] = {
                "residual": round(residual, 6),
                "tv": round(tv, 1),
            }

            np.save(str(RECON_DIR / f"{scene_name}_mismatched_{method}.npy"), x_hat)
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            result["mismatched"][method] = {"residual": 1.0, "tv": 0.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:14s}: residual={result['mismatched'][method]['residual']:.6f}  "
                     f"TV={result['mismatched'][method]['tv']:.1f}  ({dt:.1f}s)")

    # Residual ratio
    result["residual_ratio"] = {}
    for method in methods:
        r_cal = result["calibrated"][method]["residual"]
        r_mis = result["mismatched"][method]["residual"]
        if r_cal > 1e-10:
            ratio = r_mis / r_cal
        else:
            ratio = 1.0
        result["residual_ratio"][method] = round(ratio, 3)
        logger.info(f"    {METHOD_LABELS[method]:14s}: residual ratio = {ratio:.3f}x")

    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CACTI Real Data Validation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    methods = ["gap_tv", "pnp_ffdnet"]

    logger.info("=" * 60)
    logger.info("CACTI Real Data Validation for InverseNet ECCV 2026")
    logger.info(f"4 scenes x 2 methods x 2 conditions = 16 reconstructions")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    mask = load_real_mask(CR)
    if mask is None:
        logger.error("Mask not found!")
        return
    logger.info(f"Mask shape: {mask.shape}")

    all_results = []
    t_total = time.time()

    for scene_name in SCENES:
        meas = load_real_measurement(scene_name, CR)
        if meas is None:
            logger.warning(f"{scene_name}: measurement not found, skipping")
            continue

        result = validate_scene(scene_name, meas, mask, methods, args.device)
        all_results.append(result)

    elapsed = time.time() - t_total

    # Summary
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        for condition in ["calibrated", "mismatched"]:
            logger.info(f"\n  {condition.capitalize()}:")
            for method in methods:
                residuals = [r[condition][method]["residual"] for r in all_results]
                logger.info(f"    {METHOD_LABELS[method]:14s}: mean residual = {np.mean(residuals):.6f}")

    # Save
    summary = {
        "experiment": "cacti_real_data",
        "compression_ratio": CR,
        "num_scenes": len(all_results),
        "scenes": SCENES,
        "mismatch": {"dx": 0.5, "dy": 0.3},
        "methods": methods,
        "per_scene": all_results,
        "execution_seconds": round(elapsed, 1),
    }

    out_path = RESULTS_DIR / "cacti_real_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Reconstructions -> {RECON_DIR}")
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
