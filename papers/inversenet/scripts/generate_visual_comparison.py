#!/usr/bin/env python3
"""Generate visual reconstruction comparison figure for InverseNet paper (G1).

Produces a 3-row x 6-column figure showing:
  Row 1 (CASSI): GT | Scenario I | Scenario II | Scenario III | Error II | Error III
  Row 2 (CACTI): GT | Scenario I | Scenario II | Scenario III | Error II | Error III
  Row 3 (SPC):   GT | Scenario I | Scenario II | Scenario III | Error II | Error III

Representative selections:
  CASSI: Scene 1 (KAIST), band 14, MST-L
  CACTI: kobe (group 0), frame 4, ELP-Unfolding
  SPC:   cameraman, HATNet
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def psnr(x, y, max_val=1.0):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def load_cassi():
    """Load CASSI Scene 1, band 14, MST-L."""
    path = os.path.join(RESULTS_DIR, "cassi_reconstructions", "scene01.npz")
    d = np.load(path, allow_pickle=True)
    band = 14
    gt = d["gt"][:, :, band]
    sc_i = d["scenario_i_mst_l"][:, :, band]
    sc_ii = d["scenario_ii_mst_l"][:, :, band]
    sc_iii = d["scenario_iii_mst_l"][:, :, band]
    return gt, sc_i, sc_ii, sc_iii, 1.0  # max_val for PSNR


def load_cacti():
    """Load CACTI kobe group 0, frame 4, ELP-Unfolding."""
    path = os.path.join(RESULTS_DIR, "cacti_reconstructions", "kobe.npz")
    d = np.load(path, allow_pickle=True)
    frame = 4
    gt = d["g0_gt"][:, :, frame]
    sc_i = d["g0_scenario_i_elp_unfolding"][:, :, frame]
    sc_ii = d["g0_scenario_ii_elp_unfolding"][:, :, frame]
    sc_iii = d["g0_scenario_iii_elp_unfolding"][:, :, frame]
    return gt, sc_i, sc_ii, sc_iii, 1.0


def load_spc():
    """Load SPC cameraman, HATNet."""
    path = os.path.join(RESULTS_DIR, "spc_reconstructions", "cameraman.npz")
    d = np.load(path, allow_pickle=True)
    gt = d["gt"]
    sc_i = d["scenario_i_hatnet"]
    sc_ii = d["scenario_ii_hatnet"]
    sc_iii = d["scenario_iii_hatnet"]
    return gt, sc_i, sc_ii, sc_iii, 255.0  # SPC data range is [0, 255]


def main():
    print("Loading reconstruction data...")
    rows_data = []
    row_labels = ["CASSI (MST-L, Scene 1 band 14)",
                  "CACTI (ELP-Unfolding, kobe frame 4)",
                  "SPC (HATNet, cameraman)"]

    for loader in [load_cassi, load_cacti, load_spc]:
        gt, sc_i, sc_ii, sc_iii, max_val = loader()
        err_ii = np.abs(gt.astype(np.float64) - sc_ii.astype(np.float64))
        err_iii = np.abs(gt.astype(np.float64) - sc_iii.astype(np.float64))

        psnr_i = psnr(sc_i, gt, max_val)
        psnr_ii = psnr(sc_ii, gt, max_val)
        psnr_iii = psnr(sc_iii, gt, max_val)

        rows_data.append({
            "gt": gt, "sc_i": sc_i, "sc_ii": sc_ii, "sc_iii": sc_iii,
            "err_ii": err_ii, "err_iii": err_iii,
            "psnr_i": psnr_i, "psnr_ii": psnr_ii, "psnr_iii": psnr_iii,
            "max_val": max_val,
        })

    # Create figure
    fig = plt.figure(figsize=(18, 9.5))
    gs = GridSpec(3, 6, figure=fig, wspace=0.05, hspace=0.15)

    col_titles = ["Ground Truth", "Scenario I\n(Ideal)",
                  "Scenario II\n(Baseline)", "Scenario III\n(Oracle)",
                  "Error Map II", "Error Map III"]

    for row_idx, (rd, label) in enumerate(zip(rows_data, row_labels)):
        vmin, vmax = 0, rd["max_val"]
        # Error map scale: use same scale for both error maps per row
        err_max = max(np.percentile(rd["err_ii"], 99),
                      np.percentile(rd["err_iii"], 99))

        images = [rd["gt"], rd["sc_i"], rd["sc_ii"], rd["sc_iii"],
                  rd["err_ii"], rd["err_iii"]]
        psnr_vals = [None, rd["psnr_i"], rd["psnr_ii"], rd["psnr_iii"],
                     None, None]

        for col_idx, (img, pv) in enumerate(zip(images, psnr_vals)):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if col_idx < 4:
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
            else:
                im = ax.imshow(img, cmap="jet", vmin=0, vmax=err_max,
                               interpolation="nearest")

            ax.set_xticks([])
            ax.set_yticks([])

            # Column title on first row only
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10, fontweight="bold")

            # PSNR annotation
            if pv is not None:
                ax.text(0.02, 0.02, f"{pv:.1f} dB", transform=ax.transAxes,
                        fontsize=8, color="yellow", fontweight="bold",
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black",
                                  alpha=0.7))

            # Row label on first column
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")

    out_path = os.path.join(FIGURES_DIR, "visual_comparison.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out_path}")

    out_png = os.path.join(FIGURES_DIR, "visual_comparison.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
