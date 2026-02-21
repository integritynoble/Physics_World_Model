#!/usr/bin/env python3
"""
Generate standalone simulation-to-hardware gap figure for Prof. David Brady.

This figure compares the mismatch degradation observed in simulation (PSNR drop)
versus real hardware (measurement residual ratio) for CASSI and CACTI,
highlighting the systematic gap and its modality-dependent asymmetry.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Style setup ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # TrueType
    'ps.fonttype': 42,
})

COLORS = {
    'cassi_sim': '#2E5FA1',      # Blue
    'cassi_hw': '#7BB3E0',       # Light blue
    'cacti_sim': '#C93C3C',      # Red
    'cacti_hw': '#E8A0A0',       # Light red
    'arrow': '#555555',
    'text': '#333333',
}

# ── Data ─────────────────────────────────────────────────────────────
# CASSI simulation: GAP-TV Sc.I=24.34, Sc.II=20.96 → drop=3.38 dB
# CASSI hardware: residual ratio 1.8x (per-scene: 2.0, 1.6, 1.6, 2.0, 1.8)
# CACTI simulation: EfficientSCI Sc.I=35.33, Sc.II=14.48 → drop=20.85 (use GAP-TV: 28.42-17.60=10.82)
#   For GAP-TV: drop ≈ 10.94 dB (using mean values)
# CACTI hardware: residual ratio 10.4x (per-scene: 10.6, 11.0, 9.4, 10.5)

cassi_sim_drop = 3.38   # dB (GAP-TV)
cacti_sim_drop = 10.94  # dB (GAP-TV)

cassi_hw_ratio = 1.8    # x
cacti_hw_ratio = 10.4   # x

cassi_hw_per_scene = [2.0, 1.6, 1.6, 2.0, 1.8]
cacti_hw_per_scene = [10.625, 11.0, 9.351, 10.5]

# Autonomous calibration results
cassi_recovery = 85     # %
cacti_recovery = 100    # %
cassi_cal_time = 1140   # seconds
cacti_cal_time = 60     # seconds

# ── Create figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))

# Panel (a): Simulation vs Hardware bar comparison
ax1 = fig.add_axes([0.05, 0.15, 0.24, 0.75])

x = np.array([0, 1.2])
width = 0.35

# Simulation bars (PSNR drop in dB)
bars_sim = ax1.bar(x - width/2, [cassi_sim_drop, cacti_sim_drop], width,
                   color=[COLORS['cassi_sim'], COLORS['cacti_sim']],
                   edgecolor='white', linewidth=0.5, label='Simulation (PSNR drop, dB)')

# Add value labels on simulation bars
for bar, val in zip(bars_sim, [cassi_sim_drop, cacti_sim_drop]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}\ndB', ha='center', va='bottom', fontsize=9, fontweight='bold',
             color=COLORS['text'])

# Hardware bars (residual ratio, scaled for comparison)
# We use a secondary y-axis for the ratio
ax1b = ax1.twinx()
bars_hw = ax1b.bar(x + width/2, [cassi_hw_ratio, cacti_hw_ratio], width,
                   color=[COLORS['cassi_hw'], COLORS['cacti_hw']],
                   edgecolor='white', linewidth=0.5, label='Hardware (residual ratio)')

for bar, val in zip(bars_hw, [cassi_hw_ratio, cacti_hw_ratio]):
    ax1b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
              f'{val:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold',
              color=COLORS['text'])

ax1.set_xticks(x)
ax1.set_xticklabels(['CASSI\n(spectral)', 'CACTI\n(temporal)'])
ax1.set_ylabel('Simulation PSNR Drop (dB)', color=COLORS['cassi_sim'])
ax1b.set_ylabel('Hardware Residual Ratio', color=COLORS['cacti_hw'])
ax1.set_ylim(0, 14)
ax1b.set_ylim(0, 14)
ax1.set_title('(a) Simulation vs. Hardware\nMismatch Severity', fontweight='bold')

# Combined legend
sim_patch = mpatches.Patch(facecolor=COLORS['cassi_sim'], label='Simulation (PSNR drop)')
hw_patch = mpatches.Patch(facecolor=COLORS['cassi_hw'], label='Hardware (residual ratio)')
ax1.legend(handles=[sim_patch, hw_patch], loc='upper left', framealpha=0.9, fontsize=8)

# Panel (b): Per-scene hardware residual ratios
ax2 = fig.add_axes([0.34, 0.15, 0.24, 0.75])

# CASSI per-scene
cassi_x = np.arange(len(cassi_hw_per_scene))
ax2.bar(cassi_x - 0.15, cassi_hw_per_scene, 0.3,
        color=COLORS['cassi_sim'], alpha=0.8, label='CASSI (5 scenes)')

# CACTI per-scene
cacti_x = np.arange(len(cacti_hw_per_scene)) + len(cassi_hw_per_scene) + 0.5
ax2.bar(cacti_x + 0.15, cacti_hw_per_scene, 0.3,
        color=COLORS['cacti_sim'], alpha=0.8, label='CACTI (4 scenes)')

# Mean lines
ax2.axhline(y=cassi_hw_ratio, color=COLORS['cassi_sim'], linestyle='--', linewidth=1.5, alpha=0.6)
ax2.text(len(cassi_hw_per_scene) - 0.5, cassi_hw_ratio + 0.3,
         f'mean={cassi_hw_ratio}x', fontsize=8, color=COLORS['cassi_sim'], ha='right')

ax2.axhline(y=cacti_hw_ratio, color=COLORS['cacti_sim'], linestyle='--', linewidth=1.5, alpha=0.6)
ax2.text(cacti_x[-1] + 0.5, cacti_hw_ratio + 0.3,
         f'mean={cacti_hw_ratio}x', fontsize=8, color=COLORS['cacti_sim'], ha='right')

# X labels
all_labels = [f'S{i+1}' for i in range(5)] + [f'{s}' for s in ['duo', 'hand', 'pend', 'water']]
all_x = list(cassi_x - 0.15) + list(cacti_x + 0.15)
ax2.set_xticks(all_x)
ax2.set_xticklabels(all_labels, fontsize=8, rotation=0)
ax2.set_ylabel('Residual Ratio (mismatched / calibrated)')
ax2.set_ylim(0, 13)
ax2.set_title('(b) Per-Scene Hardware\nResidual Ratios', fontweight='bold')
ax2.legend(loc='upper left', framealpha=0.9, fontsize=8)

# Add separator
sep_x = len(cassi_hw_per_scene) + 0.1
ax2.axvline(x=sep_x, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# Panel (c): Gap interpretation — clean table-style summary
ax3 = fig.add_axes([0.63, 0.15, 0.35, 0.75])
ax3.axis('off')
ax3.set_title('(c) Simulation-to-Hardware\nGap Summary', fontweight='bold', pad=10)

# Table data
col_labels = ['', 'Simulation', 'Hardware', 'Gap?']
row_data = [
    ['CASSI\n(spectral)', '3.38 dB\ndrop', '1.8x\nratio', 'LARGE\ngap'],
    ['CACTI\n(temporal)', '10.94 dB\ndrop', '10.4x\nratio', 'No\ngap'],
]

table = ax3.table(
    cellText=[r[1:] for r in row_data],
    rowLabels=[r[0] for r in row_data],
    colLabels=col_labels[1:],
    cellLoc='center',
    rowLoc='center',
    loc='center',
    bbox=[0.0, 0.28, 1.0, 0.6],
)
table.auto_set_font_size(False)
table.set_fontsize(9)

# Style the table
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#cccccc')
    cell.set_linewidth(0.5)
    if row == 0:  # header
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(fontweight='bold', fontsize=8)
    elif col == 2:  # Gap column
        if row == 1:  # CASSI
            cell.set_facecolor('#FFE0E0')
            cell.set_text_props(color=COLORS['cacti_sim'], fontweight='bold')
        elif row == 2:  # CACTI
            cell.set_facecolor('#E0F0E0')
            cell.set_text_props(color=COLORS['cassi_sim'], fontweight='bold')
    cell.set_height(0.15)

# Add explanation text below
ax3.text(0.5, 0.0,
         'CASSI: Real mask has pre-existing\n'
         'manufacturing errors that absorb\n'
         'additional perturbations.\n\n'
         'CACTI: Simpler temporal mask has\n'
         'fewer errors → perturbation propagates\n'
         'at full strength across all frames.',
         ha='center', va='bottom', fontsize=7.5,
         transform=ax3.transAxes,
         style='italic', color='#555555',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F8F8',
                   edgecolor='#CCCCCC', linewidth=0.5))

# ── Save ─────────────────────────────────────────────────────────────
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'summaries')
os.makedirs(outdir, exist_ok=True)

for fmt in ('pdf', 'png'):
    outpath = os.path.join(outdir, f'brady_sim_to_hardware_gap.{fmt}')
    fig.savefig(outpath)
    print(f'Saved: {outpath}')

plt.close(fig)
print('Done.')
