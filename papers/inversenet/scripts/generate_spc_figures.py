#!/usr/bin/env python3
"""
Generate visualization figures for SPC (Single-Pixel Camera) InverseNet validation results.

Creates:
1. Scenario comparison bar chart (3 scenarios × 3 methods)
2. Method comparison heatmap (3 methods × 3 scenarios)
3. Per-image PSNR boxplot (distribution across 11 images)
4. Gap comparison (degradation vs recovery)
5. Summary CSV table for LaTeX

Usage:
    python generate_spc_figures.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Colors for methods
METHOD_COLORS = {
    'admm': '#1f77b4',           # blue
    'ista_net_plus': '#ff7f0e',  # orange
    'hatnet': '#2ca02c'          # green
}

METHOD_LABELS = {
    'admm': 'ADMM',
    'ista_net_plus': 'ISTA-Net+',
    'hatnet': 'HATNet'
}


# ============================================================================
# Utility Functions
# ============================================================================

def get_available_methods(summary):
    """Extract available methods from summary data."""
    if 'scenarios' in summary and 'scenario_i' in summary['scenarios']:
        return list(summary['scenarios']['scenario_i'].keys())
    return ['admm', 'ista_net_plus', 'hatnet']


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


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_scenario_comparison(summary: Dict) -> None:
    """
    Create bar chart comparing PSNR across 3 scenarios for available methods.

    X-axis: Scenarios (I, II, IV)
    Y-axis: PSNR (dB)
    Groups: Methods with different colors
    """
    logger.info("Creating scenario comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ['scenario_i', 'scenario_ii', 'scenario_iv']
    scenario_labels = ['Scenario I\n(Ideal)', 'Scenario II\n(Baseline)', 'Scenario IV\n(Oracle)']
    methods = get_available_methods(summary)

    x = np.arange(len(scenarios))
    width = 0.25

    for i, method in enumerate(methods):
        psnr_values = []
        for scenario_key in scenarios:
            psnr_mean = summary['scenarios'][scenario_key][method]['psnr']['mean']
            psnr_values.append(psnr_mean)

        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, psnr_values, width, label=METHOD_LABELS.get(method, method.upper()),
               color=METHOD_COLORS.get(method, '#999999'), alpha=0.8)

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SPC Reconstruction: Scenario Comparison (Set11 Images)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(loc='lower left', ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 40])

    plt.tight_layout()
    output_file = FIGURES_DIR / "scenario_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_method_comparison_heatmap(summary: Dict) -> None:
    """
    Create heatmap showing PSNR for available methods × 3 scenarios.

    Rows: Methods
    Cols: Scenarios
    Values: PSNR (dB)
    """
    logger.info("Creating method comparison heatmap...")

    scenarios = ['scenario_i', 'scenario_ii', 'scenario_iv']
    scenario_labels = ['Ideal', 'Baseline', 'Oracle']
    methods = get_available_methods(summary)
    method_labels = [METHOD_LABELS.get(m, m.upper()) for m in methods]

    # Create data matrix
    data = np.zeros((len(methods), len(scenarios)))
    for i, method in enumerate(methods):
        for j, scenario_key in enumerate(scenarios):
            data[i, j] = summary['scenarios'][scenario_key][method]['psnr']['mean']

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=20, vmax=40)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(scenario_labels)
    ax.set_yticklabels(method_labels)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(scenarios)):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Method', fontsize=12, fontweight='bold')
    ax.set_title('SPC PSNR Heatmap: Methods × Scenarios', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PSNR (dB)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_file = FIGURES_DIR / "method_comparison_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_gap_comparison(summary: Dict) -> None:
    """
    Create comparison of degradation (Gap I→II) and recovery (Gap II→IV).

    Shows how much each method degrades under mismatch and how much
    it recovers when oracle operator is available.
    """
    logger.info("Creating gap comparison plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = get_available_methods(summary)
    method_labels = [METHOD_LABELS.get(m, m.upper()) for m in methods]

    # Degradation (Gap I→II)
    gap_i_ii = [summary['gaps'][m]['gap_i_ii']['mean'] for m in methods]
    x = np.arange(len(methods))
    ax1.bar(x, gap_i_ii, color=[METHOD_COLORS[m] for m in methods], alpha=0.8)
    ax1.set_ylabel('PSNR Drop (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('Degradation Under Mismatch\n(Scenario I → II)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 5])

    # Recovery (Gap II→IV)
    gap_ii_iv = [summary['gaps'][m]['gap_ii_iv']['mean'] for m in methods]
    ax2.bar(x, gap_ii_iv, color=[METHOD_COLORS[m] for m in methods], alpha=0.8)
    ax2.set_ylabel('PSNR Recovery (dB)', fontsize=11, fontweight='bold')
    ax2.set_title('Recovery with Oracle Operator\n(Scenario II → IV)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 3])

    plt.tight_layout()
    output_file = FIGURES_DIR / "gap_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_psnr_distribution(detailed_results: List[Dict]) -> None:
    """
    Create boxplot showing PSNR distribution across 11 images for each method.

    Shows per-method robustness and consistency across different images.
    """
    logger.info("Creating PSNR distribution boxplot...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scenarios = ['scenario_i', 'scenario_ii', 'scenario_iv']
    scenario_labels = ['Ideal', 'Baseline', 'Oracle']

    # Get available methods from first result
    if detailed_results and 'scenario_i' in detailed_results[0]:
        methods = list(detailed_results[0]['scenario_i'].keys())
    else:
        methods = ['admm', 'ista_net_plus', 'hatnet']

    for scenario_idx, (scenario_key, scenario_label) in enumerate(zip(scenarios, scenario_labels)):
        ax = axes[scenario_idx]

        # Collect PSNR values for each method across images
        data_by_method = []
        for method in methods:
            psnr_values = [r[scenario_key][method]['psnr'] for r in detailed_results
                          if r[scenario_key][method]['psnr'] > 0]
            data_by_method.append(psnr_values)

        bp = ax.boxplot(data_by_method, labels=[METHOD_LABELS.get(m, m.upper()) for m in methods],
                       patch_artist=True)

        # Color boxes
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, '#999999'))
            patch.set_alpha(0.7)

        ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
        ax.set_title(f'Scenario {scenario_label}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    output_file = FIGURES_DIR / "psnr_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_ssim_comparison(summary: Dict) -> None:
    """
    Create bar chart comparing SSIM across 3 scenarios for available methods.

    X-axis: Scenarios (I, II, IV)
    Y-axis: SSIM (0-1)
    Groups: Methods with different colors
    """
    logger.info("Creating SSIM comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ['scenario_i', 'scenario_ii', 'scenario_iv']
    scenario_labels = ['Scenario I\n(Ideal)', 'Scenario II\n(Baseline)', 'Scenario IV\n(Oracle)']
    methods = get_available_methods(summary)

    x = np.arange(len(scenarios))
    width = 0.25

    for i, method in enumerate(methods):
        ssim_values = []
        for scenario_key in scenarios:
            ssim_mean = summary['scenarios'][scenario_key][method]['ssim']['mean']
            ssim_values.append(ssim_mean)

        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, ssim_values, width, label=METHOD_LABELS.get(method, method.upper()),
               color=METHOD_COLORS.get(method, '#999999'), alpha=0.8)

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('SSIM (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('SPC Reconstruction: SSIM Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(loc='lower left', ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    output_file = FIGURES_DIR / "ssim_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_summary_table(summary: Dict) -> None:
    """
    Create CSV file with results suitable for LaTeX table.

    Format:
    Method,Scenario I,Scenario II,Scenario IV,Gap I→II,Gap II→IV
    """
    logger.info("Creating summary table...")

    methods = get_available_methods(summary)
    scenarios = ['scenario_i', 'scenario_ii', 'scenario_iv']

    output_file = Path(__file__).parent.parent / "tables" / "spc_results_table.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        # Header
        f.write("Method,Scenario I,Scenario II,Scenario IV,Gap I→II,Gap II→IV\n")

        # Rows
        for method in methods:
            psnr_i = summary['scenarios']['scenario_i'][method]['psnr']['mean']
            psnr_ii = summary['scenarios']['scenario_ii'][method]['psnr']['mean']
            psnr_iv = summary['scenarios']['scenario_iv'][method]['psnr']['mean']
            gap_i_ii = summary['gaps'][method]['gap_i_ii']['mean']
            gap_ii_iv = summary['gaps'][method]['gap_ii_iv']['mean']

            psnr_i_std = summary['scenarios']['scenario_i'][method]['psnr']['std']
            psnr_ii_std = summary['scenarios']['scenario_ii'][method]['psnr']['std']
            psnr_iv_std = summary['scenarios']['scenario_iv'][method]['psnr']['std']

            f.write(f"{METHOD_LABELS.get(method, method.upper())},{psnr_i:.2f}±{psnr_i_std:.2f},"
                   f"{psnr_ii:.2f}±{psnr_ii_std:.2f},{psnr_iv:.2f}±{psnr_iv_std:.2f},"
                   f"{gap_i_ii:.2f},{gap_ii_iv:.2f}\n")

    logger.info(f"Saved: {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Generate all figures and tables."""
    logger.info("="*70)
    logger.info("SPC InverseNet Figure Generation")
    logger.info("="*70)

    # Load results
    detailed_results, summary = load_results()
    if summary is None:
        logger.error("Failed to load results")
        return

    # Generate visualizations
    plot_scenario_comparison(summary)
    plot_method_comparison_heatmap(summary)
    plot_gap_comparison(summary)
    plot_psnr_distribution(detailed_results)
    plot_ssim_comparison(summary)
    create_summary_table(summary)

    logger.info("\n✅ All figures generated!")
    logger.info(f"Output directory: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
