#!/usr/bin/env python3
"""Generate CACTI InverseNet validation figures.

Reads cacti_summary.json + cacti_validation_results.json produced by
validate_cacti_inversenet.py and creates publication-quality plots.

Usage:
    python generate_cacti_figures.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "cacti"
TABLES_DIR  = PROJECT_ROOT / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# -- style ----------------------------------------------------------------
rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.figsize": (12, 6),
})

COLORS = {
    "gap_tv":        "#1f77b4",
    "pnp_ffdnet":    "#2ca02c",
    "elp_unfolding": "#ff7f0e",
    "efficientsci":  "#d62728",
}
LABELS = {
    "gap_tv":        "GAP-TV",
    "pnp_ffdnet":    "PnP-FFDNet",
    "elp_unfolding": "ELP-Unfolding",
    "efficientsci":  "EfficientSCI",
}

SCENARIOS = ["scenario_i", "scenario_ii", "scenario_iii"]
SCEN_LABELS = ["Scenario I\n(Ideal)", "Scenario II\n(Baseline)", "Scenario III\n(Oracle)"]
SCEN_SHORT  = ["Ideal", "Baseline", "Oracle"]


def _col(m):
    return COLORS.get(m, "#999999")

def _lab(m):
    return LABELS.get(m, m.upper())


# =========================================================================
# loaders
# =========================================================================
def load():
    with open(RESULTS_DIR / "cacti_summary.json") as f:
        summary = json.load(f)
    with open(RESULTS_DIR / "cacti_validation_results.json") as f:
        detail = json.load(f)
    return summary, detail


# =========================================================================
# 1. scenario comparison bar chart
# =========================================================================
def plot_scenario_comparison(summary):
    logger.info("Creating scenario comparison plot...")
    methods = list(summary["overall"]["scenario_i"].keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(SCENARIOS))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [summary["overall"][s][m]["psnr_mean"] for s in SCENARIOS]
        off = (i - (len(methods) - 1) / 2) * w
        ax.bar(x + off, vals, w, label=_lab(m), color=_col(m), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(SCEN_LABELS)
    ax.set_ylabel("PSNR (dB)"); ax.set_title("CACTI Reconstruction: Scenario Comparison (SCI Benchmark)")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.3)
    plt.tight_layout()
    out = FIGURES_DIR / "scenario_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


# =========================================================================
# 2. heatmap
# =========================================================================
def plot_heatmap(summary):
    logger.info("Creating method comparison heatmap...")
    methods = list(summary["overall"]["scenario_i"].keys())
    data = np.array([[summary["overall"][s][m]["psnr_mean"] for s in SCENARIOS] for m in methods])
    fig, ax = plt.subplots(figsize=(10, 5))
    vmin, vmax = max(0, data.min() - 2), data.max() + 2
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(3)); ax.set_xticklabels(SCEN_SHORT)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([_lab(m) for m in methods])
    for i in range(len(methods)):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontweight="bold")
    plt.colorbar(im, ax=ax, label="PSNR (dB)")
    ax.set_title("CACTI PSNR Heatmap: Methods x Scenarios")
    plt.tight_layout()
    out = FIGURES_DIR / "method_comparison_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


# =========================================================================
# 3. gap comparison
# =========================================================================
def plot_gaps(summary):
    logger.info("Creating gap comparison plot...")
    methods = list(summary["gaps"].keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(methods))

    g1 = [summary["gaps"][m]["gap_i_ii_mean"] for m in methods]
    ax1.bar(x, g1, color=[_col(m) for m in methods], alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels([_lab(m) for m in methods], rotation=15, ha="right")
    ax1.set_ylabel("PSNR Drop (dB)"); ax1.set_title("Degradation Under Mismatch\n(Scenario I -> II)")
    ax1.grid(axis="y", alpha=0.3)

    g2 = [summary["gaps"][m]["gap_ii_iii_mean"] for m in methods]
    ax2.bar(x, g2, color=[_col(m) for m in methods], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels([_lab(m) for m in methods], rotation=15, ha="right")
    ax2.set_ylabel("PSNR Recovery (dB)"); ax2.set_title("Recovery with Oracle Operator\n(Scenario II -> III)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "gap_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


# =========================================================================
# 4. boxplot PSNR distribution across measurement groups
# =========================================================================
def plot_psnr_boxplot(detail):
    logger.info("Creating PSNR distribution boxplot...")
    methods = list(detail[0]["scenarios"]["scenario_i"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for si, (skey, slab) in enumerate(zip(SCENARIOS, SCEN_SHORT)):
        ax = axes[si]
        data = [[g["scenarios"][skey][m]["psnr"] for g in detail] for m in methods]
        bp = ax.boxplot(data, tick_labels=[_lab(m) for m in methods], patch_artist=True)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(_col(m)); patch.set_alpha(0.7)
        ax.set_ylabel("PSNR (dB)"); ax.set_title(f"Scenario {slab}")
        ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    out = FIGURES_DIR / "psnr_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


# =========================================================================
# 5. SSIM comparison
# =========================================================================
def plot_ssim(summary):
    logger.info("Creating SSIM comparison plot...")
    methods = list(summary["overall"]["scenario_i"].keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(SCENARIOS))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [summary["overall"][s][m]["ssim_mean"] for s in SCENARIOS]
        off = (i - (len(methods) - 1) / 2) * w
        ax.bar(x + off, vals, w, label=_lab(m), color=_col(m), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(SCEN_LABELS)
    ax.set_ylabel("SSIM"); ax.set_title("CACTI Reconstruction: SSIM Comparison")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.0)
    plt.tight_layout()
    out = FIGURES_DIR / "ssim_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


# =========================================================================
# 6. per-video PSNR (Scenario I)
# =========================================================================
def plot_per_video(summary):
    logger.info("Creating per-video PSNR plot...")
    pvid = summary["per_video"]
    methods = list(pvid[0]["scenario_i"].keys())
    vid_names = [v["video"] for v in pvid]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for si, (skey, slab) in enumerate(zip(SCENARIOS, SCEN_SHORT)):
        ax = axes[si]
        for m in methods:
            vals = [v[skey][m]["psnr_mean"] for v in pvid]
            ax.plot(range(len(vals)), vals, "o-", label=_lab(m), color=_col(m), lw=2, ms=7)
        ax.set_xticks(range(len(vid_names)))
        ax.set_xticklabels(vid_names, rotation=30, ha="right")
        ax.set_ylabel("PSNR (dB)"); ax.set_title(f"Scenario {slab}")
        ax.grid(alpha=0.3)
        if si == 0:
            ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "per_video_psnr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {out}")


# =========================================================================
# 7. summary CSV
# =========================================================================
def create_csv(summary):
    logger.info("Creating summary table...")
    methods = list(summary["overall"]["scenario_i"].keys())
    out = TABLES_DIR / "cacti_results_table.csv"
    with open(out, "w") as f:
        f.write("Method,Scenario I,Scenario II,Scenario III,Gap I-II,Recovery II-III\n")
        for m in methods:
            ov = summary["overall"]
            g  = summary["gaps"][m]
            f.write(f"{_lab(m)},"
                    f"{ov['scenario_i'][m]['psnr_mean']:.2f}+/-{ov['scenario_i'][m]['psnr_std']:.2f},"
                    f"{ov['scenario_ii'][m]['psnr_mean']:.2f}+/-{ov['scenario_ii'][m]['psnr_std']:.2f},"
                    f"{ov['scenario_iii'][m]['psnr_mean']:.2f}+/-{ov['scenario_iii'][m]['psnr_std']:.2f},"
                    f"{g['gap_i_ii_mean']:.2f},{g['gap_ii_iii_mean']:.2f}\n")
    logger.info(f"Saved: {out}")


# =========================================================================
# main
# =========================================================================
def main():
    logger.info("=" * 70)
    logger.info("CACTI InverseNet Figure Generation")
    logger.info("=" * 70)
    summary, detail = load()
    logger.info("Results loaded successfully")

    plot_scenario_comparison(summary)
    plot_heatmap(summary)
    plot_gaps(summary)
    plot_psnr_boxplot(detail)
    plot_ssim(summary)
    plot_per_video(summary)
    create_csv(summary)

    logger.info(f"\nAll figures generated! -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
