#!/usr/bin/env python3
"""Ablation study: compare Alg1-only vs Alg1+Alg2 calibration.

Runs MST-L across 10 KAIST scenes with 3 calibration levels:
- No correction (Scenario II)
- Alg1-only (grid search stages 0+1 only)
- Alg1+Alg2 (full pipeline, Scenario III)
- Oracle (Scenario IV)

Output: results/ablation_results.json

Usage:
    python run_ablation.py [--device cuda:0] [--scenes 10]
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
    MismatchSpec, S_NOM, STEP, N_BANDS,
    DATASET_SIMU, DATASET_REAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calibrate_alg1_only(y_corrupt: np.ndarray, mask_ideal: np.ndarray,
                        device: str = "cuda:0") -> Dict:
    """Run only grid search stages (0+1) without gradient refinement.

    This simulates "Algorithm 1 only" by running the coarse+fine grid
    search but skipping the gradient stages.
    """
    from pwm_core.calibration.cassi_upwmi_alg12 import (
        MismatchParameters, Algorithm2JointGradientRefinement,
    )
    from pwm_core.calibration.cassi_torch_modules import (
        DifferentiableMaskWarpFixed, DifferentiableCassiForwardSTE,
        DifferentiableGAPTV,
    )
    from pwm_core.recon.gap_tv import gap_tv_cassi
    import torch

    t0 = time.time()
    H, W = mask_ideal.shape
    L = N_BANDS

    x_proxy = gap_tv_cassi(y_corrupt, mask_ideal, n_bands=N_BANDS,
                           iterations=50, lam=0.01, step=STEP)

    dev = torch.device(device)
    y_t = torch.from_numpy(y_corrupt.copy()).unsqueeze(0).float().to(dev)
    mask2d_nom = mask_ideal.astype(np.float32)
    s_nom = S_NOM

    _shared_fwd = DifferentiableCassiForwardSTE(s_nom).to(dev)

    def _gpu_score(dx_v, dy_v, theta_v, n_iter=10):
        gaptv = DifferentiableGAPTV(
            s_nom, H, W, L, n_iter=n_iter, gauss_sigma=0.7,
            use_checkpointing=False,
        ).to(dev)
        gaptv.eval()
        warp = DifferentiableMaskWarpFixed(
            mask2d_nom, dx_init=dx_v, dy_init=dy_v, theta_init=theta_v
        ).to(dev)
        with torch.no_grad():
            mask_w = warp()
            phi_d_t = torch.tensor(0.0, dtype=torch.float32, device=dev)
            x_recon = gaptv(y_t, mask_w, phi_d_t)
            y_pred = _shared_fwd(x_recon, mask_w, phi_d_t)
            hh = min(y_t.shape[1], y_pred.shape[1])
            ww = min(y_t.shape[2], y_pred.shape[2])
            res = y_t[:, :hh, :ww] - y_pred[:, :hh, :ww]
            return torch.sum(res * res).item()

    # Stage 0: Coarse 3D grid (9x9x7=567)
    n_dx, n_dy, n_theta = 9, 9, 7
    dx_grid = np.linspace(-3.0, 3.0, n_dx)
    dy_grid = np.linspace(-3.0, 3.0, n_dy)
    theta_grid = np.linspace(-1.0, 1.0, n_theta)

    best_score = float("inf")
    best = (0.0, 0.0, 0.0)
    top_k = []

    for dx_v in dx_grid:
        for dy_v in dy_grid:
            for th_v in theta_grid:
                sc = _gpu_score(float(dx_v), float(dy_v), float(th_v), n_iter=8)
                if sc < best_score:
                    best_score = sc
                    best = (float(dx_v), float(dy_v), float(th_v))
                top_k.append((sc, float(dx_v), float(dy_v), float(th_v)))
                if len(top_k) > 50:
                    top_k.sort(key=lambda x: x[0])
                    top_k = top_k[:10]

    top_k.sort(key=lambda x: x[0])
    top_k = top_k[:10]

    # Stage 1: Fine grid around top-5
    dx_step = 6.0 / (n_dx - 1)
    dy_step = 6.0 / (n_dy - 1)
    th_step = 2.0 / (n_theta - 1)

    fine_best_score = float("inf")
    fine_best = best

    for _, dx_c, dy_c, th_c in top_k[:5]:
        for ddx in np.linspace(-dx_step, dx_step, 5):
            dxv = np.clip(dx_c + ddx, -3.0, 3.0)
            for ddy in np.linspace(-dy_step, dy_step, 5):
                dyv = np.clip(dy_c + ddy, -3.0, 3.0)
                for dth in np.linspace(-th_step, th_step, 3):
                    thv = np.clip(th_c + dth, -1.0, 1.0)
                    sc = _gpu_score(float(dxv), float(dyv), float(thv), n_iter=12)
                    if sc < fine_best_score:
                        fine_best_score = sc
                        fine_best = (float(dxv), float(dyv), float(thv))

    dt = time.time() - t0
    return {
        "dx": fine_best[0],
        "dy": fine_best[1],
        "theta": fine_best[2],
        "time": round(dt, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="PWMI-CASSI Ablation Study")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", type=int, default=10)
    args = parser.parse_args()

    n_scenes = min(max(args.scenes, 1), 10)

    logger.info("=" * 70)
    logger.info("PWMI-CASSI Ablation: Alg1-only vs Alg1+Alg2")
    logger.info(f"Scenes: {n_scenes}")
    logger.info("=" * 70)

    mask_ideal = load_mask(DATASET_REAL / "mask.mat")
    if mask_ideal is None:
        mask_ideal = load_mask(DATASET_SIMU / "mask.mat")
    if mask_ideal is None:
        logger.error("No mask found!")
        return

    np.random.seed(42)
    mismatch = MismatchSpec()

    from validate_cassi_pwmi import (
        calibrate_mismatch, reconstruct_mst_l,
    )

    all_results = []
    t_start = time.time()

    for scene_idx in range(n_scenes):
        scene_name = f"scene{scene_idx + 1:02d}"
        scene = load_scene(scene_name)
        if scene is None:
            continue

        logger.info(f"\nScene {scene_idx + 1}/{n_scenes}")

        # Generate corrupted measurement
        mask_corrupted = warp_affine_2d(
            mask_ideal, mismatch.mask_dx, mismatch.mask_dy, mismatch.mask_theta)
        y_corrupt = cassi_forward(scene, mask_corrupted, step=STEP)
        y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=100000, sigma=0.01)

        # Ideal measurement
        y_ideal = cassi_forward(scene, mask_ideal, step=STEP)

        # Scenario I: Ideal
        x_i = np.clip(reconstruct_mst_l(y_ideal, mask_ideal, args.device), 0, 1)
        psnr_i = compute_psnr(scene, x_i)

        # Scenario II: No correction
        x_ii = np.clip(reconstruct_mst_l(y_corrupt, mask_ideal, args.device), 0, 1)
        psnr_ii = compute_psnr(scene, x_ii)

        # Alg1-only
        logger.info("  Running Alg1-only (grid search)...")
        alg1_params = calibrate_alg1_only(y_corrupt, mask_ideal, args.device)
        mask_alg1 = warp_affine_2d(
            mask_ideal, alg1_params["dx"], alg1_params["dy"], alg1_params["theta"])
        x_alg1 = np.clip(reconstruct_mst_l(y_corrupt, mask_alg1, args.device), 0, 1)
        psnr_alg1 = compute_psnr(scene, x_alg1)

        # Alg1+Alg2 (full pipeline)
        logger.info("  Running Alg1+Alg2 (full pipeline)...")
        alg2_params, alg2_time = calibrate_mismatch(y_corrupt, mask_ideal, args.device)
        mask_alg2 = warp_affine_2d(
            mask_ideal, alg2_params["dx"], alg2_params["dy"], alg2_params["theta"])
        x_alg2 = np.clip(reconstruct_mst_l(y_corrupt, mask_alg2, args.device), 0, 1)
        psnr_alg2 = compute_psnr(scene, x_alg2)

        # Oracle
        x_iv = np.clip(reconstruct_mst_l(y_corrupt, mask_corrupted, args.device), 0, 1)
        psnr_iv = compute_psnr(scene, x_iv)

        result = {
            "scene_idx": scene_idx + 1,
            "psnr_ideal": round(psnr_i, 2),
            "psnr_no_correction": round(psnr_ii, 2),
            "psnr_alg1_only": round(psnr_alg1, 2),
            "psnr_alg1_alg2": round(psnr_alg2, 2),
            "psnr_oracle": round(psnr_iv, 2),
            "alg1_params": alg1_params,
            "alg2_params": alg2_params,
            "alg2_time": round(alg2_time, 1),
            "gain_alg1": round(psnr_alg1 - psnr_ii, 2),
            "gain_alg1_alg2": round(psnr_alg2 - psnr_ii, 2),
            "gain_oracle": round(psnr_iv - psnr_ii, 2),
        }
        all_results.append(result)

        logger.info(f"  I={psnr_i:.2f}  II={psnr_ii:.2f}  Alg1={psnr_alg1:.2f}  "
                    f"Alg1+2={psnr_alg2:.2f}  Oracle={psnr_iv:.2f}")

    total_time = time.time() - t_start

    # Summary
    if all_results:
        summary = {
            "n_scenes": len(all_results),
            "psnr_ideal_mean": round(float(np.mean([r["psnr_ideal"] for r in all_results])), 2),
            "psnr_no_correction_mean": round(float(np.mean([r["psnr_no_correction"] for r in all_results])), 2),
            "psnr_alg1_only_mean": round(float(np.mean([r["psnr_alg1_only"] for r in all_results])), 2),
            "psnr_alg1_alg2_mean": round(float(np.mean([r["psnr_alg1_alg2"] for r in all_results])), 2),
            "psnr_oracle_mean": round(float(np.mean([r["psnr_oracle"] for r in all_results])), 2),
            "gain_alg1_mean": round(float(np.mean([r["gain_alg1"] for r in all_results])), 2),
            "gain_alg1_alg2_mean": round(float(np.mean([r["gain_alg1_alg2"] for r in all_results])), 2),
            "gain_oracle_mean": round(float(np.mean([r["gain_oracle"] for r in all_results])), 2),
            "total_time": round(total_time, 1),
        }
    else:
        summary = {}

    output = {"per_scene": all_results, "summary": summary}

    out_path = RESULTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
