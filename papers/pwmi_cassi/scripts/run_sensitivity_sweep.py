#!/usr/bin/env python3
"""Sensitivity sweep: vary mismatch magnitude and measure calibration gain.

Runs MST-L and GAP-TV across 7 mismatch scales for 10 KAIST scenes.
Base mismatch: dx=1.5, dy=1.0, theta=0.3
Scales: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

Output: results/sensitivity_results.json

Usage:
    python run_sensitivity_sweep.py [--device cuda:0] [--scenes 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "pwm_core"))

from validate_cassi_pwmi import (
    load_mask, load_scene, warp_affine_2d, cassi_forward,
    add_poisson_gaussian_noise, compute_psnr, compute_ssim, compute_sam,
    calibrate_mismatch, MismatchSpec, RECONSTRUCTION_FUNCTIONS,
    METHOD_LABELS, S_NOM, STEP, N_BANDS,
    DATASET_SIMU, DATASET_REAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCALES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
METHODS = ["gap_tv", "mst_l"]
BASE_DX, BASE_DY, BASE_THETA = 1.5, 1.0, 0.3


def run_sweep_for_scene(scene_idx: int, scene: np.ndarray, mask_ideal: np.ndarray,
                        device: str) -> List[Dict]:
    """Run all scales for one scene."""
    scene_results = []

    for scale in SCALES:
        dx = BASE_DX * scale
        dy = BASE_DY * scale
        theta = BASE_THETA * scale
        mismatch = MismatchSpec(mask_dx=dx, mask_dy=dy, mask_theta=theta)

        logger.info(f"  Scale {scale:.2f}: dx={dx:.2f}, dy={dy:.2f}, theta={theta:.2f}")

        # Generate corrupted measurement
        mask_corrupted = warp_affine_2d(mask_ideal, dx, dy, theta)
        y_corrupt = cassi_forward(scene, mask_corrupted, step=STEP)
        y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=100000, sigma=0.01)

        # Ideal measurement
        y_ideal = cassi_forward(scene, mask_ideal, step=STEP)

        # Oracle mask
        mask_truth = mask_corrupted

        # Calibration
        try:
            est_params, calib_time = calibrate_mismatch(y_corrupt, mask_ideal, device)
            mask_calibrated = warp_affine_2d(
                mask_ideal, est_params["dx"], est_params["dy"], est_params["theta"])
        except Exception as e:
            logger.warning(f"  Calibration failed at scale {scale}: {e}")
            est_params = {"dx": 0.0, "dy": 0.0, "theta": 0.0}
            calib_time = 0.0
            mask_calibrated = mask_ideal

        scale_result = {
            "scene_idx": scene_idx + 1,
            "scale": scale,
            "mismatch": {"dx": dx, "dy": dy, "theta": theta},
            "estimated": est_params,
            "calibration_time": round(calib_time, 1),
        }

        for method in METHODS:
            fn = RECONSTRUCTION_FUNCTIONS[method]

            # Scenario I
            x_i = np.clip(fn(y_ideal, mask_ideal, device), 0, 1)
            # Scenario II
            x_ii = np.clip(fn(y_corrupt, mask_ideal, device), 0, 1)
            # Scenario III
            x_iii = np.clip(fn(y_corrupt, mask_calibrated, device), 0, 1)
            # Scenario IV
            x_iv = np.clip(fn(y_corrupt, mask_truth, device), 0, 1)

            psnr_i = compute_psnr(scene, x_i)
            psnr_ii = compute_psnr(scene, x_ii)
            psnr_iii = compute_psnr(scene, x_iii)
            psnr_iv = compute_psnr(scene, x_iv)

            scale_result[method] = {
                "psnr_i": round(psnr_i, 2),
                "psnr_ii": round(psnr_ii, 2),
                "psnr_iii": round(psnr_iii, 2),
                "psnr_iv": round(psnr_iv, 2),
                "degradation": round(psnr_i - psnr_ii, 2),
                "calibration_gain": round(psnr_iii - psnr_ii, 2),
                "oracle_gain": round(psnr_iv - psnr_ii, 2),
            }

            logger.info(f"    {METHOD_LABELS[method]:8s}: II={psnr_ii:.2f}  "
                        f"III={psnr_iii:.2f}  gain={psnr_iii-psnr_ii:+.2f}")

        scene_results.append(scale_result)

    return scene_results


def main():
    parser = argparse.ArgumentParser(description="PWMI-CASSI Sensitivity Sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", type=int, default=10)
    args = parser.parse_args()

    n_scenes = min(max(args.scenes, 1), 10)

    logger.info("=" * 70)
    logger.info("PWMI-CASSI Sensitivity Sweep")
    logger.info(f"Scales: {SCALES}")
    logger.info(f"Methods: {METHODS}")
    logger.info(f"Scenes: {n_scenes}")
    logger.info("=" * 70)

    mask_ideal = load_mask(DATASET_REAL / "mask.mat")
    if mask_ideal is None:
        mask_ideal = load_mask(DATASET_SIMU / "mask.mat")
    if mask_ideal is None:
        logger.error("No mask found!")
        return

    np.random.seed(42)

    all_results = []
    t_start = time.time()

    for scene_idx in range(n_scenes):
        scene_name = f"scene{scene_idx + 1:02d}"
        scene = load_scene(scene_name)
        if scene is None:
            logger.warning(f"{scene_name} not found, skipping")
            continue

        logger.info(f"\nScene {scene_idx + 1}/{n_scenes}")
        scene_results = run_sweep_for_scene(scene_idx, scene, mask_ideal, args.device)
        all_results.extend(scene_results)

    total_time = time.time() - t_start

    # Aggregate by scale
    summary = {"scales": [], "total_time": round(total_time, 1)}
    for scale in SCALES:
        scale_data = [r for r in all_results if r["scale"] == scale]
        entry = {"scale": scale, "n_scenes": len(scale_data)}
        for method in METHODS:
            psnr_ii = [r[method]["psnr_ii"] for r in scale_data]
            psnr_iii = [r[method]["psnr_iii"] for r in scale_data]
            gains = [r[method]["calibration_gain"] for r in scale_data]
            entry[method] = {
                "psnr_ii_mean": round(float(np.mean(psnr_ii)), 2),
                "psnr_iii_mean": round(float(np.mean(psnr_iii)), 2),
                "gain_mean": round(float(np.mean(gains)), 2),
                "gain_std": round(float(np.std(gains)), 2),
            }
        summary["scales"].append(entry)

    output = {
        "per_scene": all_results,
        "summary": summary,
    }

    out_path = RESULTS_DIR / "sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("Sensitivity sweep complete!")


if __name__ == "__main__":
    main()
