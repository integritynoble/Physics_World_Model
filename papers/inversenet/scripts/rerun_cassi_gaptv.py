#!/usr/bin/env python3
"""Re-run CASSI GAP-TV only (CPU) and merge results into existing JSON files.

This script re-validates GAP-TV for all 10 KAIST scenes across 3 scenarios
using the corrected parameters (iterations=100, lam=0.1, accelerate=True)
and the dispersion-aware solver for Scenario III.

Usage:
    python rerun_cassi_gaptv.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Reuse helpers from validate_cassi_inversenet
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_cassi_inversenet import (
    PROJECT_ROOT, DATASET_SIMU, RESULTS_DIR,
    MismatchParameters,
    load_mask, load_scene,
    cassi_forward, cassi_forward_with_dispersion,
    warp_affine_2d, add_poisson_gaussian_noise,
    reconstruct_gap_tv, reconstruct_gap_tv_with_dispersion,
    compute_psnr, compute_ssim, compute_sam,
    METHOD_LABELS, RECONSTRUCTION_METHODS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

NUM_SCENES = 10


def run_gaptv_scene(scene_idx: int, scene: np.ndarray, mask_ideal: np.ndarray,
                    mismatch: MismatchParameters) -> dict:
    """Run GAP-TV for one scene across all 3 scenarios."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Scene {scene_idx + 1}/10  (GAP-TV only)")
    logger.info(f"{'='*60}")

    results = {"scene_idx": scene_idx + 1}

    # --- Scenario I: Ideal ---
    logger.info("  Scenario I: Ideal")
    t0 = time.time()
    y_ideal = cassi_forward(scene, mask_ideal, step=2)
    x_hat_i = reconstruct_gap_tv(y_ideal, mask_ideal, device="cpu")
    x_hat_i = np.clip(x_hat_i, 0, 1)
    res_i = {
        "psnr": float(compute_psnr(scene, x_hat_i)),
        "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat_i, axis=2))),
        "sam": float(compute_sam(scene, x_hat_i)),
    }
    logger.info(f"    PSNR={res_i['psnr']:.2f} dB  ({time.time()-t0:.1f}s)")
    results["scenario_i"] = res_i

    # --- Scenario II: Corrupted measurement, ideal operator ---
    logger.info("  Scenario II: Baseline (5-param mismatch, uncorrected)")
    t0 = time.time()
    mask_corrupted = warp_affine_2d(
        mask_ideal, dx=mismatch.mask_dx, dy=mismatch.mask_dy,
        theta=mismatch.mask_theta)
    y_corrupt = cassi_forward_with_dispersion(
        scene, mask_corrupted,
        a1=mismatch.disp_a1, alpha_deg=mismatch.disp_alpha)
    y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=100000, sigma=0.01)

    x_hat_ii = reconstruct_gap_tv(y_corrupt, mask_ideal, device="cpu")
    x_hat_ii = np.clip(x_hat_ii, 0, 1)
    res_ii = {
        "psnr": float(compute_psnr(scene, x_hat_ii)),
        "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat_ii, axis=2))),
        "sam": float(compute_sam(scene, x_hat_ii)),
    }
    logger.info(f"    PSNR={res_ii['psnr']:.2f} dB  ({time.time()-t0:.1f}s)")
    results["scenario_ii"] = res_ii

    # --- Scenario III: Corrupted measurement, oracle operator ---
    logger.info("  Scenario III: Oracle (dispersion-aware GAP-TV)")
    t0 = time.time()
    mask_truth = warp_affine_2d(
        mask_ideal, dx=mismatch.mask_dx, dy=mismatch.mask_dy,
        theta=mismatch.mask_theta)
    x_hat_iii = reconstruct_gap_tv_with_dispersion(
        y_corrupt, mask_truth,
        a1=mismatch.disp_a1, alpha_deg=mismatch.disp_alpha,
        device="cpu")
    x_hat_iii = np.clip(x_hat_iii, 0, 1)
    res_iii = {
        "psnr": float(compute_psnr(scene, x_hat_iii)),
        "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat_iii, axis=2))),
        "sam": float(compute_sam(scene, x_hat_iii)),
    }
    logger.info(f"    PSNR={res_iii['psnr']:.2f} dB  ({time.time()-t0:.1f}s)")
    results["scenario_iii"] = res_iii

    # Gaps
    pi, pii, piii = res_i["psnr"], res_ii["psnr"], res_iii["psnr"]
    results["gaps"] = {
        "gap_i_ii": round(pi - pii, 4),
        "gap_ii_iii": round(piii - pii, 4),
        "gap_iii_i": round(pi - piii, 4),
    }
    logger.info(f"    I={pi:.2f}  II={pii:.2f}  III={piii:.2f}  "
                f"gap_I-II={pi-pii:+.2f}  rec_II-III={piii-pii:+.2f}")

    return results


def merge_results(gaptv_results: list[dict]):
    """Merge GAP-TV results into existing JSON files."""
    detail_path = RESULTS_DIR / "cassi_validation_results.json"
    summary_path = RESULTS_DIR / "cassi_summary.json"

    # --- Update per-scene results ---
    with open(detail_path) as f:
        all_results = json.load(f)

    for gaptv_scene in gaptv_results:
        idx = gaptv_scene["scene_idx"] - 1  # 0-based
        if idx < len(all_results):
            for scen in ["scenario_i", "scenario_ii", "scenario_iii"]:
                all_results[idx][scen]["gap_tv"] = gaptv_scene[scen]
            all_results[idx]["gaps"]["gap_tv"] = gaptv_scene["gaps"]

    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Updated {detail_path}")

    # --- Recompute summary ---
    with open(summary_path) as f:
        summary = json.load(f)

    for scen_key in ["scenario_i", "scenario_ii", "scenario_iii"]:
        psnr_vals = [r[scen_key]["gap_tv"]["psnr"] for r in all_results
                     if r[scen_key]["gap_tv"]["psnr"] > 0]
        ssim_vals = [r[scen_key]["gap_tv"]["ssim"] for r in all_results
                     if r[scen_key]["gap_tv"]["ssim"] > 0]
        sam_vals = [r[scen_key]["gap_tv"]["sam"] for r in all_results
                    if r[scen_key]["gap_tv"]["sam"] < 180]

        summary[scen_key]["gap_tv"] = {
            "psnr_mean": round(float(np.mean(psnr_vals)), 2) if psnr_vals else 0.0,
            "psnr_std": round(float(np.std(psnr_vals)), 2) if psnr_vals else 0.0,
            "ssim_mean": round(float(np.mean(ssim_vals)), 4) if ssim_vals else 0.0,
            "ssim_std": round(float(np.std(ssim_vals)), 4) if ssim_vals else 0.0,
            "sam_mean": round(float(np.mean(sam_vals)), 2) if sam_vals else 0.0,
            "sam_std": round(float(np.std(sam_vals)), 2) if sam_vals else 0.0,
        }

    gap_i_ii = [r["gaps"]["gap_tv"]["gap_i_ii"] for r in all_results]
    gap_ii_iii = [r["gaps"]["gap_tv"]["gap_ii_iii"] for r in all_results]
    summary["gaps"]["gap_tv"] = {
        "gap_i_ii_mean": round(float(np.mean(gap_i_ii)), 2),
        "gap_i_ii_std": round(float(np.std(gap_i_ii)), 2),
        "gap_ii_iii_mean": round(float(np.mean(gap_ii_iii)), 2),
        "gap_ii_iii_std": round(float(np.std(gap_ii_iii)), 2),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Updated {summary_path}")


def main():
    logger.info("=" * 60)
    logger.info("CASSI GAP-TV Re-validation (Chambolle TV, dispersion oracle)")
    logger.info("  iterations=100, tv_weight=0.1, Nesterov acceleration")
    logger.info("  Scenario III: dispersion-aware forward/adjoint")
    logger.info("=" * 60)

    mask_ideal = load_mask(DATASET_SIMU / "mask.mat")
    if mask_ideal is None:
        logger.error("Ideal mask not found at %s", DATASET_SIMU / "mask.mat")
        sys.exit(1)

    mismatch = MismatchParameters()
    np.random.seed(42)

    gaptv_results = []
    t_start = time.time()

    for scene_idx in range(NUM_SCENES):
        scene_name = f"scene{scene_idx + 1:02d}"
        scene = load_scene(scene_name)
        if scene is None:
            logger.warning(f"{scene_name} not found, skipping")
            continue
        result = run_gaptv_scene(scene_idx, scene, mask_ideal, mismatch)
        gaptv_results.append(result)

    total = time.time() - t_start
    logger.info(f"\nTotal time: {total:.1f}s ({total/max(len(gaptv_results),1):.1f}s per scene)")

    if not gaptv_results:
        logger.error("No results collected!")
        sys.exit(1)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("GAP-TV SUMMARY (mean across 10 scenes)")
    logger.info("=" * 60)
    for scen in ["scenario_i", "scenario_ii", "scenario_iii"]:
        psnrs = [r[scen]["psnr"] for r in gaptv_results]
        label = {"scenario_i": "I (Ideal)", "scenario_ii": "II (Baseline)",
                 "scenario_iii": "III (Oracle)"}[scen]
        logger.info(f"  {label:16s}: PSNR = {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB")

    gaps_i_ii = [r["gaps"]["gap_i_ii"] for r in gaptv_results]
    gaps_ii_iii = [r["gaps"]["gap_ii_iii"] for r in gaptv_results]
    logger.info(f"  Gap I-II  = {np.mean(gaps_i_ii):+.2f} dB")
    logger.info(f"  Rec II-III = {np.mean(gaps_ii_iii):+.2f} dB")
    if np.mean(gaps_i_ii) > 0.01:
        rho = np.mean(gaps_ii_iii) / np.mean(gaps_i_ii) * 100
        logger.info(f"  Recovery ratio = {rho:.1f}%")

    # Merge into existing JSON files
    merge_results(gaptv_results)

    # Print LaTeX-ready table rows
    logger.info("\n" + "=" * 60)
    logger.info("LaTeX table rows (for inversenet_paper.tex Table 1)")
    logger.info("=" * 60)

    psnr_i = [r["scenario_i"]["psnr"] for r in gaptv_results]
    psnr_ii = [r["scenario_ii"]["psnr"] for r in gaptv_results]
    psnr_iii = [r["scenario_iii"]["psnr"] for r in gaptv_results]
    ssim_i = [r["scenario_i"]["ssim"] for r in gaptv_results]
    ssim_ii = [r["scenario_ii"]["ssim"] for r in gaptv_results]
    ssim_iii = [r["scenario_iii"]["ssim"] for r in gaptv_results]

    mi, si = np.mean(psnr_i), np.std(psnr_i)
    mii, sii = np.mean(psnr_ii), np.std(psnr_ii)
    miii, siii = np.mean(psnr_iii), np.std(psnr_iii)
    deg = mi - mii
    rec = miii - mii
    rho_val = rec / deg * 100 if deg > 0.5 else None

    ssim_mi = np.mean(ssim_i)
    ssim_mii = np.mean(ssim_ii)
    ssim_miii = np.mean(ssim_iii)

    logger.info("GAP-TV~\\cite{yuan2016gaptv}")
    logger.info(f" & {mi:.2f}{{\\scriptsize$\\pm${si:.2f}}} / .{int(ssim_mi*1000):03d}")
    logger.info(f" & {mii:.2f}{{\\scriptsize$\\pm${sii:.2f}}} / .{int(ssim_mii*1000):03d}")
    logger.info(f" & {miii:.2f}{{\\scriptsize$\\pm${siii:.2f}}} / .{int(ssim_miii*1000):03d}")
    logger.info(f" & {deg:.2f} & {rec:.2f}")
    if rho_val is not None:
        logger.info(f" & {rho_val:.1f}\\% \\\\")
    else:
        logger.info(f" & N/A$^\\dagger$ \\\\")

    # Print per-scene rows for supplementary table
    logger.info("\n" + "=" * 60)
    logger.info("Per-scene rows (for inversenet_supplementary.tex)")
    logger.info("=" * 60)
    for r in gaptv_results:
        idx = r["scene_idx"]
        pi = r["scenario_i"]["psnr"]
        pii = r["scenario_ii"]["psnr"]
        piii = r["scenario_iii"]["psnr"]
        gap = pi - pii
        if gap > 0.5:
            rho_s = f"{(piii - pii) / gap * 100:.1f}\\%"
        else:
            rho_s = "$^\\dagger$"
        logger.info(f"Scene {idx:2d}: {pi:.2f} & {pii:.2f} & {piii:.2f} & {rho_s}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
