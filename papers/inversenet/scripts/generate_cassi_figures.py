#!/usr/bin/env python3
"""
Generate visualization figures for CASSI InverseNet validation results.

Creates:
1. Scenario comparison bar chart (3 scenarios x 4 methods)
2. Method comparison heatmap (4 methods x 3 scenarios, PSNR + SSIM)
3. Per-scene PSNR line plot across 10 scenes
4. Gap comparison (degradation vs recovery)
5. SSIM comparison bar chart
6. PSNR distribution boxplot
7. Oracle gain per-scene bar chart

Usage:
    python generate_cassi_figures.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "cassi"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
rcParams['font.size'] = 11
rcParams['font.family'] = 'serif'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

METHOD_COLORS = {
    'gap_tv': '#1f77b4',
    'hdnet': '#ff7f0e',
    'mst_s': '#2ca02c',
    'mst_l': '#d62728'
}

METHOD_LABELS = {
    'gap_tv': 'GAP-TV',
    'hdnet': 'HDNet',
    'mst_s': 'MST-S',
    'mst_l': 'MST-L'
}

SCENARIO_LABELS = {
    'scenario_i': 'Scenario I\n(Ideal)',
    'scenario_ii': 'Scenario II\n(Baseline)',
    'scenario_iii': 'Scenario III\n(Oracle)'
}

SCENARIO_SHORT = {
    'scenario_i': 'Ideal',
    'scenario_ii': 'Baseline',
    'scenario_iii': 'Oracle'
}

SCENARIOS = ['scenario_i', 'scenario_ii', 'scenario_iii']
METHODS = ['gap_tv', 'hdnet', 'mst_s', 'mst_l']


def load_results():
    """Load validation results from JSON files."""
    with open(RESULTS_DIR / "cassi_validation_results.json") as f:
        detailed = json.load(f)
    with open(RESULTS_DIR / "cassi_summary.json") as f:
        summary = json.load(f)
    logger.info(f"Loaded {len(detailed)} scene results")
    return detailed, summary


def get_psnr_mean(summary, scenario, method):
    """Get PSNR mean from summary with flat key structure."""
    return summary[scenario][method]['psnr_mean']


def get_psnr_std(summary, scenario, method):
    """Get PSNR std from summary with flat key structure."""
    return summary[scenario][method]['psnr_std']


def get_ssim_mean(summary, scenario, method):
    """Get SSIM mean from summary with flat key structure."""
    return summary[scenario][method]['ssim_mean']


def get_ssim_std(summary, scenario, method):
    """Get SSIM std from summary with flat key structure."""
    return summary[scenario][method]['ssim_std']


def get_gap_mean(summary, method, gap_key):
    """Get gap mean from summary. gap_key is 'gap_i_ii' or 'gap_ii_iii'."""
    return summary['gaps'][method][f'{gap_key}_mean']


def get_gap_std(summary, method, gap_key):
    """Get gap std from summary. gap_key is 'gap_i_ii' or 'gap_ii_iii'."""
    return summary['gaps'][method][f'{gap_key}_std']


def plot_scenario_comparison(summary):
    """Bar chart comparing PSNR across 3 scenarios for 4 methods with error bars."""
    logger.info("Creating scenario comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(SCENARIOS))
    n_methods = len(METHODS)
    width = 0.18

    for i, method in enumerate(METHODS):
        means = []
        stds = []
        for sc in SCENARIOS:
            means.append(get_psnr_mean(summary, sc, method))
            stds.append(get_psnr_std(summary, sc, method))

        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], alpha=0.85,
                      capsize=3, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('CASSI Reconstruction: Scenario Comparison (4 Methods)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 42])

    ax.axhline(y=30, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scenario_comparison.png", dpi=200, bbox_inches='tight')
    logger.info(f"  Saved scenario_comparison.png")
    plt.close()


def plot_method_comparison_heatmap(summary):
    """Heatmap showing PSNR and SSIM for methods x scenarios."""
    logger.info("Creating method comparison heatmap...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PSNR heatmap
    psnr_data = np.zeros((len(METHODS), len(SCENARIOS)))
    for i, m in enumerate(METHODS):
        for j, s in enumerate(SCENARIOS):
            psnr_data[i, j] = get_psnr_mean(summary, s, m)

    im1 = ax1.imshow(psnr_data, cmap='RdYlGn', aspect='auto', vmin=20, vmax=38)
    ax1.set_xticks(np.arange(len(SCENARIOS)))
    ax1.set_yticks(np.arange(len(METHODS)))
    ax1.set_xticklabels([SCENARIO_SHORT[s] for s in SCENARIOS])
    ax1.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    for i in range(len(METHODS)):
        for j in range(len(SCENARIOS)):
            ax1.text(j, i, f'{psnr_data[i, j]:.1f}',
                     ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    ax1.set_title('PSNR (dB)', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # SSIM heatmap
    ssim_data = np.zeros((len(METHODS), len(SCENARIOS)))
    for i, m in enumerate(METHODS):
        for j, s in enumerate(SCENARIOS):
            ssim_data[i, j] = get_ssim_mean(summary, s, m)

    im2 = ax2.imshow(ssim_data, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=1.0)
    ax2.set_xticks(np.arange(len(SCENARIOS)))
    ax2.set_yticks(np.arange(len(METHODS)))
    ax2.set_xticklabels([SCENARIO_SHORT[s] for s in SCENARIOS])
    ax2.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    for i in range(len(METHODS)):
        for j in range(len(SCENARIOS)):
            ax2.text(j, i, f'{ssim_data[i, j]:.3f}',
                     ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    ax2.set_title('SSIM', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.suptitle('CASSI Method Comparison Across Scenarios', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "method_comparison_heatmap.png", dpi=200, bbox_inches='tight')
    logger.info(f"  Saved method_comparison_heatmap.png")
    plt.close()


def plot_gap_comparison(summary):
    """Bar chart of degradation (I->II) and recovery (II->III) gaps."""
    logger.info("Creating gap comparison plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(METHODS))
    labels = [METHOD_LABELS[m] for m in METHODS]

    # Degradation (Gap I->II)
    gap_i_ii = [get_gap_mean(summary, m, 'gap_i_ii') for m in METHODS]
    gap_i_ii_std = [get_gap_std(summary, m, 'gap_i_ii') for m in METHODS]
    bars1 = ax1.bar(x, gap_i_ii, color=[METHOD_COLORS[m] for m in METHODS],
                    alpha=0.85, yerr=gap_i_ii_std, capsize=5,
                    edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars1, gap_i_ii):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f} dB', ha='center', fontsize=9, fontweight='bold')
    ax1.set_ylabel('PSNR Drop (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('Mismatch Degradation\n(Scenario I -> II)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 18])

    # Recovery (Gap II->III)
    gap_ii_iii = [get_gap_mean(summary, m, 'gap_ii_iii') for m in METHODS]
    gap_ii_iii_std = [get_gap_std(summary, m, 'gap_ii_iii') for m in METHODS]
    bars2 = ax2.bar(x, gap_ii_iii, color=[METHOD_COLORS[m] for m in METHODS],
                    alpha=0.85, yerr=gap_ii_iii_std, capsize=5,
                    edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars2, gap_ii_iii):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f'{val:.2f} dB', ha='center', fontsize=9, fontweight='bold')
    ax2.set_ylabel('PSNR Recovery (dB)', fontsize=11, fontweight='bold')
    ax2.set_title('Oracle Recovery\n(Scenario II -> III)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 4])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gap_comparison.png", dpi=200, bbox_inches='tight')
    logger.info(f"  Saved gap_comparison.png")
    plt.close()


def plot_psnr_distribution(detailed):
    """Boxplot of PSNR distribution across 10 scenes."""
    logger.info("Creating PSNR distribution boxplot...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, sc in enumerate(SCENARIOS):
        ax = axes[idx]
        data = []
        for m in METHODS:
            vals = [r[sc][m]['psnr'] for r in detailed]
            data.append(vals)

        bp = ax.boxplot(data, tick_labels=[METHOD_LABELS[m] for m in METHODS],
                        patch_artist=True, widths=0.5)
        for patch, m in zip(bp['boxes'], METHODS):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.7)

        ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
        ax.set_title(f'{SCENARIO_SHORT[sc]}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('PSNR Distribution Across 10 KAIST Scenes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "psnr_distribution.png", dpi=200, bbox_inches='tight')
    logger.info(f"  Saved psnr_distribution.png")
    plt.close()


def plot_per_scene_psnr(detailed):
    """Line plot showing PSNR for each scene across scenarios."""
    logger.info("Creating per-scene PSNR line plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    scenes = [f"S{r['scene_idx']:02d}" for r in detailed]
    x = np.arange(len(scenes))

    for ax_idx, method in enumerate(METHODS):
        ax = axes[ax_idx]
        for sc, ls, marker in zip(SCENARIOS,
                                   ['-', '--', '-.'],
                                   ['o', 's', '^']):
            vals = [r[sc][method]['psnr'] for r in detailed]
            ax.plot(x, vals, ls, marker=marker, markersize=5, linewidth=1.5,
                    label=SCENARIO_SHORT[sc])

        ax.set_xlabel('Scene', fontsize=11, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
        ax.set_title(f'{METHOD_LABELS[method]}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenes, rotation=45)
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([18, 42])

    plt.suptitle('Per-Scene PSNR Across Scenarios', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_scene_psnr.png", dpi=200, bbox_inches='tight')
    logger.info(f"  Saved per_scene_psnr.png")
    plt.close()


def plot_ssim_comparison(summary):
    """Bar chart comparing SSIM across scenarios."""
    logger.info("Creating SSIM comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(SCENARIOS))
    n_methods = len(METHODS)
    width = 0.18

    for i, method in enumerate(METHODS):
        means = []
        stds = []
        for sc in SCENARIOS:
            means.append(get_ssim_mean(summary, sc, method))
            stds.append(get_ssim_std(summary, sc, method))

        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], alpha=0.85,
                      capsize=3, edgecolor='black', linewidth=0.5)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
    ax.set_title('CASSI Reconstruction: SSIM Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.05])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ssim_comparison.png", dpi=200, bbox_inches='tight')
    logger.info(f"  Saved ssim_comparison.png")
    plt.close()


def plot_oracle_gain_per_scene(detailed):
    """Bar chart showing oracle gain (II->III) for each scene."""
    logger.info("Creating oracle gain per-scene plot...")

    fig, ax = plt.subplots(figsize=(14, 5))

    scenes = [f"S{r['scene_idx']:02d}" for r in detailed]
    x = np.arange(len(scenes))
    n_methods = len(METHODS)
    width = 0.2

    for i, method in enumerate(METHODS):
        gains = [r['gaps'][method]['gap_ii_iii'] for r in detailed]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, gains, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], alpha=0.85,
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Scene', fontsize=11, fontweight='bold')
    ax.set_ylabel('Oracle Gain (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Oracle Recovery (II -> III) Per Scene', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "oracle_gain_per_scene.png", dpi=200, bbox_inches='tight')
    logger.info(f"  Saved oracle_gain_per_scene.png")
    plt.close()


def create_summary_table(summary):
    """Create CSV table with results."""
    logger.info("Creating summary table...")

    output_file = PROJECT_ROOT / "tables" / "cassi_results_table.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Method,Scenario I (PSNR),Scenario II (PSNR),Scenario III (PSNR),"
                "Scenario I (SSIM),Scenario II (SSIM),Scenario III (SSIM),"
                "Gap I->II,Gap II->III\n")
        for method in METHODS:
            pi_m = get_psnr_mean(summary, 'scenario_i', method)
            pi_s = get_psnr_std(summary, 'scenario_i', method)
            pii_m = get_psnr_mean(summary, 'scenario_ii', method)
            pii_s = get_psnr_std(summary, 'scenario_ii', method)
            piv_m = get_psnr_mean(summary, 'scenario_iii', method)
            piv_s = get_psnr_std(summary, 'scenario_iii', method)
            si_m = get_ssim_mean(summary, 'scenario_i', method)
            si_s = get_ssim_std(summary, 'scenario_i', method)
            sii_m = get_ssim_mean(summary, 'scenario_ii', method)
            sii_s = get_ssim_std(summary, 'scenario_ii', method)
            siv_m = get_ssim_mean(summary, 'scenario_iii', method)
            siv_s = get_ssim_std(summary, 'scenario_iii', method)
            g_i_ii = get_gap_mean(summary, method, 'gap_i_ii')
            g_ii_iv = get_gap_mean(summary, method, 'gap_ii_iii')

            f.write(f"{METHOD_LABELS[method]},"
                    f"{pi_m:.2f}+/-{pi_s:.2f},"
                    f"{pii_m:.2f}+/-{pii_s:.2f},"
                    f"{piv_m:.2f}+/-{piv_s:.2f},"
                    f"{si_m:.4f}+/-{si_s:.4f},"
                    f"{sii_m:.4f}+/-{sii_s:.4f},"
                    f"{siv_m:.4f}+/-{siv_s:.4f},"
                    f"{g_i_ii:.2f},{g_ii_iv:.2f}\n")

    logger.info(f"  Saved {output_file}")


def main():
    logger.info("=" * 70)
    logger.info("CASSI InverseNet Figure Generation (Phase 4)")
    logger.info("=" * 70)

    detailed, summary = load_results()

    plot_scenario_comparison(summary)
    plot_method_comparison_heatmap(summary)
    plot_gap_comparison(summary)
    plot_psnr_distribution(detailed)
    plot_per_scene_psnr(detailed)
    plot_ssim_comparison(summary)
    plot_oracle_gain_per_scene(detailed)
    create_summary_table(summary)

    logger.info("")
    logger.info("=" * 70)
    logger.info("All CASSI figures generated!")
    logger.info(f"Output: {FIGURES_DIR}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
