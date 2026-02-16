#!/usr/bin/env python3
"""Generate publication-quality figures for PWMI-CASSI paper.

Creates:
1. fig3_scenario_comparison.pdf    -- grouped bar (4 scenarios x 5 methods)
2. fig4_visual_comparison.pdf      -- RGB renderings (3 scenes x 4 scenarios)
3. fig5_parameter_recovery.pdf     -- scatter: estimated vs true
4. fig6_sensitivity_curves.pdf     -- PSNR vs mismatch magnitude
5. fig7_ablation.pdf               -- Alg1-only vs Alg1+Alg2 vs Oracle
6. fig8_per_scene.pdf              -- per-scene PSNR trajectories

Usage:
    python generate_figures.py [--results-dir ../results]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
rcParams["font.size"] = 11
rcParams["font.family"] = "serif"
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 13
rcParams["legend.fontsize"] = 10
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10

METHOD_COLORS = {
    "gap_tv": "#1f77b4",
    "mst_s": "#2ca02c",
    "mst_l": "#d62728",
    "hdnet": "#9467bd",
    "pnp_hsicnn": "#ff7f0e",
}

METHOD_LABELS = {
    "gap_tv": "GAP-TV",
    "mst_s": "MST-S",
    "mst_l": "MST-L",
    "hdnet": "HDNet",
    "pnp_hsicnn": "PnP-HSICNN",
}

SCENARIO_LABELS = {
    "scenario_i": "I (Ideal)",
    "scenario_ii": "II (Assumed)",
    "scenario_iii": "III (Corrected)",
    "scenario_iv": "IV (Oracle)",
}

SCENARIO_COLORS = {
    "scenario_i": "#4CAF50",
    "scenario_ii": "#F44336",
    "scenario_iii": "#2196F3",
    "scenario_iv": "#FF9800",
}

SCENARIOS = ["scenario_i", "scenario_ii", "scenario_iii", "scenario_iv"]
METHODS = ["gap_tv", "mst_s", "mst_l", "hdnet", "pnp_hsicnn"]


def load_main_results():
    """Load main 4-scenario validation results."""
    detail_path = RESULTS_DIR / "pwmi_cassi_results.json"
    summary_path = RESULTS_DIR / "pwmi_cassi_summary.json"
    if not detail_path.exists() or not summary_path.exists():
        logger.warning("Main results not found. Run validate_cassi_pwmi.py first.")
        return None, None
    with open(detail_path) as f:
        detailed = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)
    return detailed, summary


def load_sensitivity_results():
    """Load sensitivity sweep results."""
    path = RESULTS_DIR / "sensitivity_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_ablation_results():
    """Load ablation study results."""
    path = RESULTS_DIR / "ablation_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig3_scenario_comparison(summary):
    """Grouped bar chart: 4 scenarios x 3 methods, PSNR + error bars."""
    logger.info("Creating fig3_scenario_comparison...")

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(SCENARIOS))
    n_methods = len(METHODS)
    width = 0.15

    for i, method in enumerate(METHODS):
        means = [summary[sc][method]["psnr_mean"] for sc in SCENARIOS]
        stds = [summary[sc][method]["psnr_std"] for sc in SCENARIOS]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method], alpha=0.85,
            capsize=3, edgecolor="black", linewidth=0.5,
        )
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Scenario", fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontweight="bold")
    ax.set_title("CASSI Reconstruction Quality Across 4 Scenarios", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim([0, 42])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_scenario_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_scenario_comparison.png", dpi=200, bbox_inches="tight")
    logger.info("  Saved fig3_scenario_comparison.pdf")
    plt.close()


def fig5_parameter_recovery(detailed):
    """Scatter plot: estimated vs true parameters (dx, dy, theta)."""
    logger.info("Creating fig5_parameter_recovery...")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    params = [
        ("dx", "mask_dx", "$\\Delta x$ (pixels)", [-0.5, 3.5]),
        ("dy", "mask_dy", "$\\Delta y$ (pixels)", [-0.5, 2.5]),
        ("theta", "mask_theta", "$\\theta$ (degrees)", [-0.2, 0.8]),
    ]

    for ax, (param_key, true_key, label, lims) in zip(axes, params):
        true_vals = [r["mismatch_injected"][param_key.replace("mask_", "")] for r in detailed]
        est_vals = [r["mismatch_estimated"][param_key.replace("mask_", "")] for r in detailed]

        ax.scatter(true_vals, est_vals, c="#d62728", s=60, edgecolors="black",
                   linewidths=0.5, zorder=3)
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Perfect recovery")
        ax.set_xlabel(f"True {label}", fontweight="bold")
        ax.set_ylabel(f"Estimated {label}", fontweight="bold")
        ax.set_title(f"Parameter: {label}", fontweight="bold")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)

        # Add RMSE annotation
        errs = np.array(est_vals) - np.array(true_vals)
        rmse = np.sqrt(np.mean(errs**2))
        ax.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Mismatch Parameter Recovery (10 KAIST Scenes)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_parameter_recovery.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_parameter_recovery.png", dpi=200, bbox_inches="tight")
    logger.info("  Saved fig5_parameter_recovery.pdf")
    plt.close()


def fig6_sensitivity_curves(sensitivity_data):
    """PSNR vs mismatch magnitude for MST-L and GAP-TV."""
    logger.info("Creating fig6_sensitivity_curves...")

    scales_summary = sensitivity_data["summary"]["scales"]
    scales = [s["scale"] for s in scales_summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    markers = ["o", "s", "D", "^", "v"]
    sweep_methods = [m for m in scales_summary[0].keys()
                     if m not in ("scale", "n_scenes") and isinstance(scales_summary[0][m], dict)]
    for method, marker in zip(sweep_methods, markers):
        psnr_ii = [s[method]["psnr_ii_mean"] for s in scales_summary]
        psnr_iii = [s[method]["psnr_iii_mean"] for s in scales_summary]
        gains = [s[method]["gain_mean"] for s in scales_summary]
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)

        ax1.plot(scales, psnr_ii, f"{marker}--", color=color, markersize=6,
                 label=f"{label} (Sc. II)", alpha=0.7)
        ax1.plot(scales, psnr_iii, f"{marker}-", color=color, markersize=6,
                 label=f"{label} (Sc. III)")

        ax2.plot(scales, gains, f"{marker}-", color=color, markersize=6,
                 label=label)

    ax1.set_xlabel("Mismatch Scale", fontweight="bold")
    ax1.set_ylabel("PSNR (dB)", fontweight="bold")
    ax1.set_title("Reconstruction Quality vs Mismatch Magnitude", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, linestyle="--")

    ax2.set_xlabel("Mismatch Scale", fontweight="bold")
    ax2.set_ylabel("Calibration Gain (dB)", fontweight="bold")
    ax2.set_title("Calibration Benefit vs Mismatch Magnitude", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_sensitivity_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig6_sensitivity_curves.png", dpi=200, bbox_inches="tight")
    logger.info("  Saved fig6_sensitivity_curves.pdf")
    plt.close()


def fig7_ablation(ablation_data):
    """Ablation: No correction vs Alg1-only vs Alg1+Alg2 vs Oracle."""
    logger.info("Creating fig7_ablation...")

    results = ablation_data["per_scene"]
    scenes = [f"S{r['scene_idx']:02d}" for r in results]
    x = np.arange(len(scenes))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.18

    categories = [
        ("psnr_no_correction", "No Correction (II)", "#F44336"),
        ("psnr_alg1_only", "Alg1 Only (Grid)", "#FF9800"),
        ("psnr_alg1_alg2", "Alg1+Alg2 (Ours)", "#2196F3"),
        ("psnr_oracle", "Oracle (IV)", "#4CAF50"),
    ]

    for i, (key, label, color) in enumerate(categories):
        vals = [r[key] for r in results]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85,
               edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Scene", fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontweight="bold")
    ax.set_title("Ablation: Calibration Pipeline Components (MST-L)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add mean annotations
    summary = ablation_data["summary"]
    ax.axhline(y=summary.get("psnr_alg1_alg2_mean", 0), color="#2196F3",
               linestyle=":", alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_ablation.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig7_ablation.png", dpi=200, bbox_inches="tight")
    logger.info("  Saved fig7_ablation.pdf")
    plt.close()


def fig8_per_scene(detailed):
    """Per-scene PSNR trajectories across 4 scenarios for each method."""
    logger.info("Creating fig8_per_scene...")

    n_methods = len(METHODS)
    ncols = min(n_methods, 3)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    scenes = [f"S{r['scene_idx']:02d}" for r in detailed]
    x = np.arange(len(scenes))

    for ax_idx, method in enumerate(METHODS):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        for sc, marker, ls in zip(SCENARIOS, ["o", "s", "D", "^"], ["-", "--", "-.", ":"]):
            vals = [r[sc][method]["psnr"] for r in detailed]
            ax.plot(x, vals, ls, marker=marker, markersize=5, linewidth=1.5,
                    color=SCENARIO_COLORS[sc], label=SCENARIO_LABELS[sc])

        ax.set_xlabel("Scene", fontweight="bold")
        ax.set_ylabel("PSNR (dB)", fontweight="bold")
        ax.set_title(METHOD_LABELS[method], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenes, rotation=45)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")

    # Hide unused subplots
    for ax_idx in range(n_methods, nrows * ncols):
        axes[ax_idx // ncols, ax_idx % ncols].set_visible(False)

    plt.suptitle("Per-Scene PSNR Across 4 Scenarios", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_per_scene.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig8_per_scene.png", dpi=200, bbox_inches="tight")
    logger.info("  Saved fig8_per_scene.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate PWMI-CASSI figures")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    logger.info("=" * 70)
    logger.info("PWMI-CASSI Figure Generation")
    logger.info("=" * 70)

    detailed, summary = load_main_results()
    sensitivity = load_sensitivity_results()
    ablation = load_ablation_results()

    if detailed is not None and summary is not None:
        fig3_scenario_comparison(summary)
        fig5_parameter_recovery(detailed)
        fig8_per_scene(detailed)
    else:
        logger.warning("Skipping main figures (no results)")

    if sensitivity is not None:
        fig6_sensitivity_curves(sensitivity)
    else:
        logger.warning("Skipping sensitivity figure (no results)")

    if ablation is not None:
        fig7_ablation(ablation)
    else:
        logger.warning("Skipping ablation figure (no results)")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Figure generation complete!")
    logger.info(f"Output: {FIGURES_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
