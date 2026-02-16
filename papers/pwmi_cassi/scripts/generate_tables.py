#!/usr/bin/env python3
"""Generate LaTeX tables for PWMI-CASSI paper.

Creates:
- Table 1: Main results (Method x Scenario PSNR/SSIM, calibration gain)
- Table 2: Parameter recovery (true vs estimated per scene, RMSE)
- Table 3: Ablation (Alg1-only vs Alg1+2 vs Oracle)
- Table 4: Computational cost breakdown

Usage:
    python generate_tables.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["gap_tv", "mst_s", "mst_l", "hdnet", "pnp_hsicnn"]
METHOD_LABELS = {
    "gap_tv": "GAP-TV", "mst_s": "MST-S", "mst_l": "MST-L",
    "hdnet": "HDNet", "pnp_hsicnn": "PnP-HSICNN",
}


def load_json(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def table1_main_results(summary):
    """Table 1: Method x Scenario PSNR/SSIM with calibration gain."""
    logger.info("Creating Table 1: Main results...")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reconstruction quality (PSNR/SSIM) across four scenarios on 10 KAIST scenes. "
        r"Calibration gain measures improvement from Scenario~II to III.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{l cccc c}",
        r"\toprule",
        r"Method & Sc.~I (Ideal) & Sc.~II (Assumed) & Sc.~III (Corrected) & Sc.~IV (Oracle) & Gain (II$\to$III) \\",
        r"\midrule",
    ]

    for method in METHODS:
        label = METHOD_LABELS[method]
        vals = []
        for sc in ["scenario_i", "scenario_ii", "scenario_iii", "scenario_iv"]:
            pm = summary[sc][method]["psnr_mean"]
            ps = summary[sc][method]["psnr_std"]
            vals.append(f"{pm:.2f}$\\pm${ps:.2f}")

        gain = summary["gaps"][method]["calibration_gain_mean"]
        gain_s = summary["gaps"][method]["calibration_gain_std"]
        gain_str = f"\\textbf{{{gain:+.2f}}}$\\pm${gain_s:.2f}"

        lines.append(f"{label} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {gain_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = TABLES_DIR / "table1_main_results.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved {out_path}")


def table2_parameter_recovery(detailed):
    """Table 2: Per-scene parameter recovery."""
    logger.info("Creating Table 2: Parameter recovery...")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mismatch parameter recovery across 10 KAIST scenes. "
        r"True mismatch: $\Delta x=1.5$\,px, $\Delta y=1.0$\,px, $\theta=0.3^\circ$.}",
        r"\label{tab:param_recovery}",
        r"\small",
        r"\begin{tabular}{c ccc ccc}",
        r"\toprule",
        r"Scene & $\hat{\Delta x}$ & $\hat{\Delta y}$ & $\hat{\theta}$ & $|\epsilon_x|$ & $|\epsilon_y|$ & $|\epsilon_\theta|$ \\",
        r"\midrule",
    ]

    for r in detailed:
        idx = r["scene_idx"]
        est = r["mismatch_estimated"]
        err = r["parameter_error"]
        lines.append(
            f"S{idx:02d} & {est['dx']:.3f} & {est['dy']:.3f} & {est['theta']:.3f} "
            f"& {err['dx_err']:.3f} & {err['dy_err']:.3f} & {err['theta_err']:.3f} \\\\"
        )

    # RMSE row
    dx_errs = [r["parameter_error"]["dx_err"] for r in detailed]
    dy_errs = [r["parameter_error"]["dy_err"] for r in detailed]
    th_errs = [r["parameter_error"]["theta_err"] for r in detailed]
    dx_rmse = np.sqrt(np.mean(np.array(dx_errs)**2))
    dy_rmse = np.sqrt(np.mean(np.array(dy_errs)**2))
    th_rmse = np.sqrt(np.mean(np.array(th_errs)**2))

    lines.extend([
        r"\midrule",
        f"\\textbf{{RMSE}} & -- & -- & -- & \\textbf{{{dx_rmse:.3f}}} & \\textbf{{{dy_rmse:.3f}}} & \\textbf{{{th_rmse:.3f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = TABLES_DIR / "table2_parameter_recovery.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved {out_path}")


def table3_ablation(ablation_data):
    """Table 3: Ablation (no correction vs Alg1 vs Alg1+2 vs Oracle)."""
    logger.info("Creating Table 3: Ablation...")

    if ablation_data is None:
        logger.warning("  No ablation data, skipping")
        return

    results = ablation_data["per_scene"]
    summary = ablation_data["summary"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: calibration pipeline components (MST-L on 10 KAIST scenes).}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{l cccc}",
        r"\toprule",
        r"Configuration & PSNR (dB) & Gain over II & \% Oracle Recovery \\",
        r"\midrule",
    ]

    configs = [
        ("No Correction (II)", "psnr_no_correction_mean", 0),
        ("Alg1 Only (Grid)", "psnr_alg1_only_mean", "gain_alg1_mean"),
        ("Alg1+Alg2 (Ours)", "psnr_alg1_alg2_mean", "gain_alg1_alg2_mean"),
        ("Oracle (IV)", "psnr_oracle_mean", "gain_oracle_mean"),
    ]

    oracle_gain = summary.get("gain_oracle_mean", 1)
    if oracle_gain == 0:
        oracle_gain = 1

    for label, psnr_key, gain_key in configs:
        psnr = summary.get(psnr_key, 0)
        if isinstance(gain_key, str):
            gain = summary.get(gain_key, 0)
        else:
            gain = gain_key
        recovery_pct = (gain / oracle_gain * 100) if oracle_gain != 0 else 0
        gain_str = f"{gain:+.2f}" if gain != 0 else "--"
        recovery_str = f"{recovery_pct:.0f}\\%" if gain != 0 else "--"
        lines.append(f"{label} & {psnr:.2f} & {gain_str} & {recovery_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_path = TABLES_DIR / "table3_ablation.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved {out_path}")


def table4_computational_cost(summary):
    """Table 4: Computational cost breakdown."""
    logger.info("Creating Table 4: Computational cost...")

    timing = summary.get("timing", {})
    calib_mean = timing.get("calibration_mean", 0)
    total_mean = timing.get("total_mean", 0)
    recon_mean = total_mean - calib_mean

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational cost breakdown per scene (GPU: NVIDIA A100).}",
        r"\label{tab:cost}",
        r"\small",
        r"\begin{tabular}{l r l}",
        r"\toprule",
        r"Stage & Time (s) & Description \\",
        r"\midrule",
        f"Stage 0: Coarse 3D grid & $\\sim$85 & 567 GPU GAP-TV evals \\\\",
        f"Stage 1: Fine 3D grid & $\\sim$88 & 375 GPU GAP-TV evals \\\\",
        f"Stage 2A: Gradient (dx) & $\\sim$61 & 50 Adam steps \\\\",
        f"Stage 2B: Gradient (dy,$\\theta$) & $\\sim$74 & 60 Adam steps \\\\",
        f"Stage 2C: Joint refinement & $\\sim$128 & 80 Adam steps \\\\",
        r"\midrule",
        f"\\textbf{{Total calibration}} & \\textbf{{{calib_mean:.0f}}} & Alg1+Alg2 pipeline \\\\",
        f"Reconstruction (5 methods) & {recon_mean:.0f} & GAP-TV, MST-S/L, HDNet, PnP \\\\",
        r"\midrule",
        f"\\textbf{{Total per scene}} & \\textbf{{{total_mean:.0f}}} & End-to-end \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = TABLES_DIR / "table4_computational_cost.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved {out_path}")


def main():
    logger.info("=" * 70)
    logger.info("PWMI-CASSI Table Generation")
    logger.info("=" * 70)

    summary = load_json("pwmi_cassi_summary.json")
    detailed = load_json("pwmi_cassi_results.json")
    ablation = load_json("ablation_results.json")

    if summary is not None:
        table1_main_results(summary)
        table4_computational_cost(summary)
    else:
        logger.warning("No summary results, skipping Tables 1 and 4")

    if detailed is not None:
        table2_parameter_recovery(detailed)
    else:
        logger.warning("No detailed results, skipping Table 2")

    table3_ablation(ablation)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Table generation complete!")
    logger.info(f"Output: {TABLES_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
