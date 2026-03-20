#!/usr/bin/env python3
"""Generate visual comparison figure (fig4) for PWMI-CASSI paper.

Layout: 4 rows x 4 columns, full page width (~17cm)
  Row 1: MST-L RGB composites  (GT | Sc.II | Sc.III | Sc.IV)
  Row 2: MST-L error maps      (--  | Sc.II | Sc.III | Sc.IV)
  Row 3: HDNet RGB composites   (GT | Sc.II | Sc.III | Sc.IV)
  Row 4: HDNet error maps       (--  | Sc.II | Sc.III | Sc.IV)

PSNR overlay on each reconstruction panel, shared error colorbar.
Scene 1 only (strongest mask-sensitivity contrast).

Usage:
    python generate_visual_figure.py [--npz path/to/scene01.npz]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
rcParams["font.size"] = 9
rcParams["font.family"] = "serif"
rcParams["axes.labelsize"] = 10
rcParams["axes.titlesize"] = 10


def cube_to_rgb(cube: np.ndarray, bands: tuple = (8, 14, 20)) -> np.ndarray:
    """Convert hyperspectral cube to pseudo-RGB using selected bands.

    Bands (8, 14, 20) correspond approximately to (530, 570, 615 nm)
    for the KAIST 28-band dataset (400--700 nm range).

    Uses percentile normalization (2nd--98th) for robust contrast.
    """
    rgb = np.stack([cube[:, :, b] for b in bands], axis=2).astype(np.float64)
    for c in range(3):
        lo = np.percentile(rgb[:, :, c], 2)
        hi = np.percentile(rgb[:, :, c], 98)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo), 0, 1)
    return rgb.astype(np.float32)


def compute_error_map(gt: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Mean absolute error across all 28 spectral bands."""
    return np.mean(np.abs(gt.astype(np.float64) - recon.astype(np.float64)),
                   axis=2).astype(np.float32)


def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    """PSNR in dB (data in [0, 1])."""
    mse = float(np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def create_visual_figure(npz_path: Path) -> Path:
    """Create the 4x4 visual comparison figure.

    Returns path to saved PDF.
    """
    logger.info(f"Loading reconstructions from {npz_path}")
    data = np.load(str(npz_path))

    gt = data["gt"]  # (256, 256, 28)
    logger.info(f"  GT shape: {gt.shape}")

    # Extract reconstructions for MST-L and HDNet across scenarios
    methods = ["mst_l", "hdnet"]
    scenarios = ["scenario_ii", "scenario_iii", "scenario_iv"]
    scenario_labels = ["Sc.II (Mismatch)", "Sc.III (Corrected)", "Sc.IV (Oracle)"]

    recons = {}
    for method in methods:
        for scen in scenarios:
            key = f"{scen}_{method}"
            if key in data:
                recons[key] = data[key]
                logger.info(f"  Loaded {key}: shape={recons[key].shape}")
            else:
                logger.warning(f"  Missing key: {key}")
                recons[key] = np.zeros_like(gt)

    # Compute RGB images and error maps
    gt_rgb = cube_to_rgb(gt)

    # Find global error range for consistent colorbar
    all_errors = []
    for method in methods:
        for scen in scenarios:
            err = compute_error_map(gt, recons[f"{scen}_{method}"])
            all_errors.append(err)
    vmax = np.percentile(np.concatenate([e.ravel() for e in all_errors]), 98)
    vmin = 0.0

    # Create figure: 4 rows x 4 columns
    fig, axes = plt.subplots(4, 4, figsize=(7.0, 7.2),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    method_labels = {"mst_l": "MST-L", "hdnet": "HDNet"}

    for m_idx, method in enumerate(methods):
        row_rgb = m_idx * 2
        row_err = m_idx * 2 + 1

        # Column 0: Ground truth RGB (row_rgb) / blank (row_err)
        ax_gt = axes[row_rgb, 0]
        ax_gt.imshow(gt_rgb)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if m_idx == 0:
            ax_gt.set_title("Ground Truth", fontsize=9, fontweight="bold")
        ax_gt.set_ylabel(f"{method_labels[method]}\nRGB", fontsize=8,
                         fontweight="bold", rotation=0, labelpad=40, va="center")

        ax_blank = axes[row_err, 0]
        ax_blank.axis("off")
        ax_blank.set_ylabel("Error", fontsize=8, fontweight="bold",
                            rotation=0, labelpad=40, va="center")
        # Show a text label in the blank cell
        ax_blank.text(0.5, 0.5, "N/A", ha="center", va="center",
                      fontsize=10, color="gray",
                      transform=ax_blank.transAxes)

        # Columns 1--3: Scenario II, III, IV
        for s_idx, scen in enumerate(scenarios):
            col = s_idx + 1
            key = f"{scen}_{method}"
            recon = recons[key]
            recon_rgb = cube_to_rgb(recon)
            err_map = compute_error_map(gt, recon)
            psnr = compute_psnr(gt, recon)

            # RGB panel
            ax_rgb = axes[row_rgb, col]
            ax_rgb.imshow(recon_rgb)
            ax_rgb.set_xticks([])
            ax_rgb.set_yticks([])
            ax_rgb.text(0.02, 0.98, f"{psnr:.1f} dB",
                        transform=ax_rgb.transAxes,
                        fontsize=7, fontweight="bold", color="white",
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="black", alpha=0.7, edgecolor="none"))
            if m_idx == 0:
                ax_rgb.set_title(scenario_labels[s_idx], fontsize=9,
                                 fontweight="bold")

            # Error map panel
            ax_err = axes[row_err, col]
            im = ax_err.imshow(err_map, cmap="hot", vmin=vmin, vmax=vmax)
            ax_err.set_xticks([])
            ax_err.set_yticks([])

    # Add shared colorbar for error maps
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.33])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MAE", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Save
    out_pdf = FIGURES_DIR / "fig4_visual_comparison.pdf"
    out_png = FIGURES_DIR / "fig4_visual_comparison.png"
    fig.savefig(str(out_pdf), bbox_inches="tight", dpi=300, pad_inches=0.05)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)

    logger.info(f"Saved {out_pdf}")
    logger.info(f"Saved {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Generate fig4 visual comparison for PWMI-CASSI paper")
    parser.add_argument("--npz", type=str,
                        default=str(RESULTS_DIR / "reconstructions" / "scene01.npz"),
                        help="Path to scene NPZ file with reconstruction cubes")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        logger.error(f"NPZ file not found: {npz_path}")
        logger.error("Run: python validate_cassi_pwmi.py --save-recon --scenes 1")
        return

    create_visual_figure(npz_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
