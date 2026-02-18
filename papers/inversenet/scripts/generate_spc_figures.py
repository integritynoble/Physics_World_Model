#!/usr/bin/env python3
"""
Generate visualization figures for SPC (Single-Pixel Camera) InverseNet validation results.

Updated for v4.0 results format (pretrained ISTA-Net + HATNet).

Creates:
1. Scenario comparison bar chart (3 scenarios x 3 methods)
2. Method comparison heatmap (3 methods x 3 scenarios)
3. Per-image PSNR boxplot (distribution across 11 images)
4. Gap comparison (degradation vs recovery)
5. SSIM comparison bar chart
6. Summary CSV table for LaTeX

Usage:
    python generate_spc_figures.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "spc"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style
rcParams['font.size'] = 11
rcParams['figure.figsize'] = (12, 6)
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 10

# Methods and colors (v4.0: actual pretrained models)
METHODS = ['fista_tv', 'ista_net', 'hatnet']
SCENARIOS = ['scenario_i', 'scenario_ii', 'scenario_iii']
SCENARIO_LABELS = ['Scenario I\n(Ideal)', 'Scenario II\n(Baseline)', 'Scenario III\n(Corrected)']
SCENARIO_SHORT = ['Ideal', 'Baseline', 'Corrected']

METHOD_COLORS = {
    'fista_tv': '#1f77b4',     # blue
    'ista_net': '#2ca02c',     # green
    'hatnet': '#ff7f0e',       # orange
}

METHOD_LABELS = {
    'fista_tv': 'FISTA-TV',
    'ista_net': 'ISTA-Net',
    'hatnet': 'HATNet',
}


# ============================================================================
# Data Loading (v4.0 format)
# ============================================================================

def load_results() -> tuple:
    """Load validation results from JSON files."""
    try:
        with open(RESULTS_DIR / "spc_validation_results.json") as f:
            detailed_results = json.load(f)
        with open(RESULTS_DIR / "spc_summary.json") as f:
            summary = json.load(f)
        logger.info("Results loaded successfully")
        return detailed_results, summary
    except FileNotFoundError as e:
        logger.error(f"Results not found: {e}")
        return None, None


def get_psnr(summary: Dict, method: str, scenario: str) -> float:
    """Get mean PSNR from v4.0 summary format."""
    key = f"{method}_{scenario}"
    return summary['methods'][key]['psnr_mean']


def get_psnr_std(summary: Dict, method: str, scenario: str) -> float:
    """Get PSNR std from v4.0 summary format."""
    key = f"{method}_{scenario}"
    return summary['methods'][key]['psnr_std']


def get_ssim(summary: Dict, method: str, scenario: str) -> float:
    """Get mean SSIM from v4.0 summary format."""
    key = f"{method}_{scenario}"
    return summary['methods'][key]['ssim_mean']


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_scenario_comparison(summary: Dict) -> None:
    """Bar chart comparing PSNR across 3 scenarios for 3 methods."""
    logger.info("Creating scenario comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(SCENARIOS))
    width = 0.25

    for i, method in enumerate(METHODS):
        psnr_values = [get_psnr(summary, method, s) for s in SCENARIOS]
        psnr_stds = [get_psnr_std(summary, method, s) for s in SCENARIOS]

        offset = (i - 1) * width
        ax.bar(x + offset, psnr_values, width, yerr=psnr_stds,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method],
               alpha=0.8, capsize=4)

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SPC Reconstruction: Scenario Comparison (Set11, Pretrained Models)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 42])

    plt.tight_layout()
    output_file = FIGURES_DIR / "scenario_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_method_comparison_heatmap(summary: Dict) -> None:
    """Heatmap showing PSNR for 3 methods x 3 scenarios."""
    logger.info("Creating method comparison heatmap...")

    data = np.zeros((len(METHODS), len(SCENARIOS)))
    for i, method in enumerate(METHODS):
        for j, scenario in enumerate(SCENARIOS):
            data[i, j] = get_psnr(summary, method, scenario)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=15, vmax=35)

    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_xticklabels(SCENARIO_SHORT)
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])

    for i in range(len(METHODS)):
        for j in range(len(SCENARIOS)):
            ax.text(j, i, f'{data[i, j]:.1f}',
                    ha="center", va="center", color="black", fontweight='bold',
                    fontsize=14)

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Method', fontsize=12, fontweight='bold')
    ax.set_title('SPC PSNR Heatmap: Methods x Scenarios (Pretrained Models)',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PSNR (dB)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_file = FIGURES_DIR / "method_comparison_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_gap_comparison(summary: Dict) -> None:
    """Dual bar chart: degradation (I->II) and recovery (II->III)."""
    logger.info("Creating gap comparison plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    method_labels = [METHOD_LABELS[m] for m in METHODS]
    x = np.arange(len(METHODS))

    # Degradation (Gap I->II)
    gap_i_ii = [get_psnr(summary, m, 'scenario_i') - get_psnr(summary, m, 'scenario_ii')
                for m in METHODS]
    bars1 = ax1.bar(x, gap_i_ii, color=[METHOD_COLORS[m] for m in METHODS], alpha=0.8)
    ax1.set_ylabel('PSNR Drop (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('Degradation Under Mismatch\n(Scenario I -> II)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(gap_i_ii) * 1.3])
    for bar, val in zip(bars1, gap_i_ii):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}', ha='center', fontweight='bold')

    # Recovery (Gap II->III)
    gap_ii_iii = [get_psnr(summary, m, 'scenario_iii') - get_psnr(summary, m, 'scenario_ii')
                  for m in METHODS]
    bars2 = ax2.bar(x, gap_ii_iii, color=[METHOD_COLORS[m] for m in METHODS], alpha=0.8)
    ax2.set_ylabel('PSNR Recovery (dB)', fontsize=11, fontweight='bold')
    ax2.set_title('Recovery with Gain Correction\n(Scenario II -> III)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(gap_ii_iii) * 1.3])
    for bar, val in zip(bars2, gap_ii_iii):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    output_file = FIGURES_DIR / "gap_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_psnr_distribution(detailed_results: List[Dict]) -> None:
    """Boxplot showing PSNR distribution across 11 images."""
    logger.info("Creating PSNR distribution boxplot...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for scenario_idx, (scenario, label) in enumerate(zip(SCENARIOS, SCENARIO_SHORT)):
        ax = axes[scenario_idx]

        data_by_method = []
        for method in METHODS:
            psnr_values = [r[method][scenario]['psnr'] for r in detailed_results]
            data_by_method.append(psnr_values)

        bp = ax.boxplot(data_by_method,
                        tick_labels=[METHOD_LABELS[m] for m in METHODS],
                        patch_artist=True)

        for patch, method in zip(bp['boxes'], METHODS):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.7)

        ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
        ax.set_title(f'Scenario: {label}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Image PSNR Distribution (11 Set11 Images)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file = FIGURES_DIR / "psnr_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_ssim_comparison(summary: Dict) -> None:
    """Bar chart comparing SSIM across 3 scenarios for 3 methods."""
    logger.info("Creating SSIM comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(SCENARIOS))
    width = 0.25

    for i, method in enumerate(METHODS):
        ssim_values = [get_ssim(summary, method, s) for s in SCENARIOS]
        offset = (i - 1) * width
        ax.bar(x + offset, ssim_values, width,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method], alpha=0.8)

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('SSIM (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('SPC Reconstruction: SSIM Comparison (Pretrained Models)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS)
    ax.legend(loc='upper right', ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    output_file = FIGURES_DIR / "ssim_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_summary_table(summary: Dict) -> None:
    """Create CSV file with results suitable for LaTeX table."""
    logger.info("Creating summary table...")

    output_file = PROJECT_ROOT / "tables" / "spc_results_table.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Method,Scenario I,Scenario II,Scenario III,Gap I-II,Recovery II-III\n")

        for method in METHODS:
            pi = get_psnr(summary, method, 'scenario_i')
            pii = get_psnr(summary, method, 'scenario_ii')
            piii = get_psnr(summary, method, 'scenario_iii')
            pi_s = get_psnr_std(summary, method, 'scenario_i')
            pii_s = get_psnr_std(summary, method, 'scenario_ii')
            piii_s = get_psnr_std(summary, method, 'scenario_iii')

            f.write(f"{METHOD_LABELS[method]},"
                    f"{pi:.2f}+/-{pi_s:.2f},"
                    f"{pii:.2f}+/-{pii_s:.2f},"
                    f"{piii:.2f}+/-{piii_s:.2f},"
                    f"{pi-pii:.2f},{piii-pii:.2f}\n")

    logger.info(f"Saved: {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Generate all figures and tables."""
    logger.info("=" * 70)
    logger.info("SPC InverseNet Figure Generation (v4.0)")
    logger.info("=" * 70)

    detailed_results, summary = load_results()
    if summary is None:
        logger.error("Failed to load results")
        return

    plot_scenario_comparison(summary)
    plot_method_comparison_heatmap(summary)
    plot_gap_comparison(summary)
    plot_psnr_distribution(detailed_results)
    plot_ssim_comparison(summary)
    create_summary_table(summary)

    logger.info(f"\nAll figures generated in {FIGURES_DIR}")


if __name__ == '__main__':
    main()
