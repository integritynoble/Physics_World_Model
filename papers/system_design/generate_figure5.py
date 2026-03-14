"""Generate Figure 5: Reconstruction comparison for the Nature paper.

Loads CT, MRI, and CASSI data, reconstructs with two experts per modality
(one acting as "Agent" proxy, one as "Expert"), selects the sample closest
to the median PSNR, and produces a 3x4 panel figure.

Usage:
    cd /home/spiritai/pwm/Physics_World_Model
    python papers/system_design/generate_figure5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from papers.system_design.expert_study.data_loader import load_data
from papers.system_design.expert_study.expert_reconstructors import EXPERT_DISPATCH
from papers.system_design.expert_study.evaluate import compute_metrics, _normalize_01

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODALITIES = ["ct", "mri", "sd_cassi"]
ROW_LABELS = {"ct": "CT", "mri": "MRI", "sd_cassi": "CASSI"}
COL_HEADERS = [
    r"Ground truth $\mathbf{x}^*$",
    "Agent recon.",
    "Expert recon.",
    r"$|\Delta|$ (5$\times$)",
]

# Best expert per modality (proxy for "Agent") and a different expert ("Expert")
AGENT_EXPERT = {"ct": "E1", "mri": "E5", "sd_cassi": "E2"}
HUMAN_EXPERT = {"ct": "E3", "mri": "E3", "sd_cassi": "E1"}

CASSI_BAND = 14  # Middle spectral band for display

FIGURES_DIR = ROOT / "papers" / "system_design" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 10  # Must match expert study


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_2d(arr: np.ndarray, modality: str) -> np.ndarray:
    """Extract a 2D displayable image from raw data."""
    if modality == "sd_cassi":
        # 3D cube -> single band
        return arr[:, :, CASSI_BAND].astype(np.float64)
    if np.iscomplexobj(arr):
        return np.abs(arr).astype(np.float64)
    return arr.astype(np.float64)


def _compute_psnr_for_sample(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Compute PSNR for one sample (works for 2D and 3D)."""
    m = compute_metrics(x_hat, x_true)
    return m["psnr"]


def _find_median_sample(expert_func, samples: list[dict]) -> int:
    """Find the sample index whose PSNR is closest to the median."""
    recons = expert_func(samples)
    psnrs = []
    for x_hat, s in zip(recons, samples):
        psnrs.append(_compute_psnr_for_sample(x_hat, s["x_true"]))
    median_psnr = np.median(psnrs)
    idx = int(np.argmin(np.abs(np.array(psnrs) - median_psnr)))
    return idx


def _annotate(ax, psnr: float, ssim: float, fontsize: float = 7):
    """Add PSNR/SSIM annotation in the lower-left corner."""
    txt = f"{psnr:.1f} dB\n{ssim:.3f}"
    ax.text(
        0.03, 0.03, txt,
        transform=ax.transAxes,
        fontsize=fontsize,
        color="white",
        verticalalignment="bottom",
        horizontalalignment="left",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="black",
            alpha=0.6,
            edgecolor="none",
        ),
    )


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------

def generate_figure():
    """Generate the 3x4 reconstruction comparison figure."""
    print("Loading data and running reconstructions...")

    # Collect per-row data: (gt_2d, agent_2d, expert_2d, diff_2d, metrics_agent, metrics_expert)
    rows_data = []

    for modality in MODALITIES:
        print(f"  {ROW_LABELS[modality]}...")
        samples = load_data(modality, n_samples=N_SAMPLES)

        agent_id = AGENT_EXPERT[modality]
        expert_id = HUMAN_EXPERT[modality]

        agent_func = EXPERT_DISPATCH[(agent_id, modality)]
        expert_func = EXPERT_DISPATCH[(expert_id, modality)]

        # Find median-PSNR sample using the agent reconstruction
        median_idx = _find_median_sample(agent_func, samples)
        print(f"    Median sample index: {median_idx}")

        # Reconstruct just the selected sample (wrap in list)
        sel_samples = [samples[median_idx]]
        agent_recon_full = agent_func(sel_samples)[0]
        expert_recon_full = expert_func(sel_samples)[0]
        gt_full = samples[median_idx]["x_true"]

        # Compute metrics on full data (including 3D for CASSI)
        m_agent = compute_metrics(agent_recon_full, gt_full)
        m_expert = compute_metrics(expert_recon_full, gt_full)
        print(f"    Agent ({agent_id}):  PSNR={m_agent['psnr']:.1f} dB, SSIM={m_agent['ssim']:.3f}")
        print(f"    Expert ({expert_id}): PSNR={m_expert['psnr']:.1f} dB, SSIM={m_expert['ssim']:.3f}")

        # Prepare 2D images for display
        gt_2d = _prepare_2d(gt_full, modality)
        agent_2d = _prepare_2d(agent_recon_full, modality)
        expert_2d = _prepare_2d(expert_recon_full, modality)

        # Normalize for display
        gt_disp = _normalize_01(gt_2d)
        agent_disp = _normalize_01(agent_2d)
        expert_disp = _normalize_01(expert_2d)

        # Absolute difference, amplified 5x
        diff = np.abs(agent_disp - expert_disp) * 5.0
        diff = np.clip(diff, 0, 1)

        rows_data.append({
            "modality": modality,
            "gt": gt_disp,
            "agent": agent_disp,
            "expert": expert_disp,
            "diff": diff,
            "m_agent": m_agent,
            "m_expert": m_expert,
        })

    # -----------------------------------------------------------------------
    # Build figure
    # -----------------------------------------------------------------------
    print("  Composing figure...")

    # Nature double-column: 183 mm = 7.2 in
    fig_w = 7.2  # inches
    # Determine aspect ratio from first row's image
    sample_h, sample_w = rows_data[0]["gt"].shape
    panel_aspect = sample_h / sample_w
    # 4 panels across with some space for row labels
    panel_w = (fig_w - 0.6) / 4.0  # subtract space for row labels
    panel_h = panel_w * panel_aspect
    fig_h = panel_h * 3 + 0.7  # 3 rows + space for column headers

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # Use GridSpec for tight control
    gs = GridSpec(
        3, 4,
        figure=fig,
        left=0.07,
        right=0.99,
        bottom=0.01,
        top=0.93,
        wspace=0.04,
        hspace=0.08,
    )

    # Font settings for Nature
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
    })

    for row_idx, rd in enumerate(rows_data):
        modality = rd["modality"]
        is_cassi = (modality == "sd_cassi")
        img_cmap = "viridis" if is_cassi else "gray"

        panels = [
            (rd["gt"], img_cmap, None),
            (rd["agent"], img_cmap, rd["m_agent"]),
            (rd["expert"], img_cmap, rd["m_expert"]),
            (rd["diff"], "inferno", None),
        ]

        for col_idx, (img, cmap, metrics) in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, aspect="equal",
                      interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])

            # Thin border
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("0.5")

            # PSNR/SSIM annotation for agent and expert panels
            if metrics is not None:
                _annotate(ax, metrics["psnr"], metrics["ssim"])

            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(COL_HEADERS[col_idx], fontsize=8, pad=4,
                             fontweight="bold")

            # Row labels (left column only)
            if col_idx == 0:
                ax.set_ylabel(
                    ROW_LABELS[modality],
                    fontsize=9,
                    fontweight="bold",
                    rotation=90,
                    labelpad=8,
                )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    pdf_path = FIGURES_DIR / "figure5_recon_comparison.pdf"
    png_path = FIGURES_DIR / "figure5_recon_comparison.png"

    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"\nFigure saved:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    generate_figure()
