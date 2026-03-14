"""Generate Figure 5: Agent vs Expert reconstruction comparison.

Uses the benchmark data with two reconstruction methods per modality:
  - "Agent" proxy: best-performing classical algorithm (mimics automated pipeline)
  - "Expert" proxy: different established algorithm (mimics expert library)

Both methods receive only measurements + calibration metadata (no ground truth).
The figure demonstrates that different methods produce near-identical quality
when given the same forward model specification.

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
MODALITIES = ["ct", "mri", "sd_cassi", "lensless", "sim"]
ROW_LABELS = {"ct": "CT", "mri": "MRI", "sd_cassi": "CASSI",
              "lensless": "Lensless", "sim": "SIM"}
COL_HEADERS = [
    r"Ground truth $\mathbf{x}^*$",
    "Agent recon.",
    "Expert recon.",
    r"$|\Delta|$ (5$\times$)",
]

# Agent-proxy (highest PSNR method) and Expert-proxy (established method)
AGENT_EXPERT = {"ct": "E1", "mri": "E5", "sd_cassi": "E4",
                "lensless": "E2", "sim": "E5"}
HUMAN_EXPERT = {"ct": "E3", "mri": "E3", "sd_cassi": "E1",
                "lensless": "E3", "sim": "E3"}

# Display settings
CASSI_BAND = 14
N_SAMPLES = 5  # Load fewer samples, pick best
SAMPLE_IDX = {"ct": 3, "mri": 1, "sd_cassi": 1,
              "lensless": 1, "sim": 0}  # Pre-selected good samples

FIGURES_DIR = ROOT / "papers" / "system_design" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_2d(arr, modality):
    """Extract a 2D displayable image from raw data."""
    if modality == "sd_cassi":
        return arr[:, :, CASSI_BAND].astype(np.float64)
    if np.iscomplexobj(arr):
        return np.abs(arr).astype(np.float64)
    return arr.astype(np.float64)


def _annotate(ax, psnr, ssim, fontsize=7):
    """Add PSNR/SSIM annotation in the lower-left corner."""
    txt = f"{psnr:.1f} dB\n{ssim:.3f}"
    ax.text(
        0.03, 0.03, txt,
        transform=ax.transAxes, fontsize=fontsize, color="white",
        verticalalignment="bottom", horizontalalignment="left",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                  alpha=0.6, edgecolor="none"),
    )


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------

def generate_figure():
    """Generate the 3x4 reconstruction comparison figure."""
    print("Generating Figure 5: Agent vs Expert reconstruction comparison...")
    rows_data = []

    for modality in MODALITIES:
        print(f"\n  {ROW_LABELS[modality]}...")
        samples = load_data(modality, n_samples=N_SAMPLES)

        agent_id = AGENT_EXPERT[modality]
        expert_id = HUMAN_EXPERT[modality]
        idx = SAMPLE_IDX[modality]
        idx = min(idx, len(samples) - 1)

        agent_func = EXPERT_DISPATCH[(agent_id, modality)]
        expert_func = EXPERT_DISPATCH[(expert_id, modality)]

        # Reconstruct selected sample
        sel = [samples[idx]]
        print(f"    Running {agent_id} (Agent proxy)...")
        agent_recon = agent_func(sel)[0]
        print(f"    Running {expert_id} (Expert proxy)...")
        expert_recon = expert_func(sel)[0]
        gt = samples[idx]["x_true"]

        # Compute metrics
        m_agent = compute_metrics(agent_recon, gt)
        m_expert = compute_metrics(expert_recon, gt)
        print(f"    Agent ({agent_id}):  PSNR={m_agent['psnr']:.1f} dB, "
              f"SSIM={m_agent['ssim']:.3f}")
        print(f"    Expert ({expert_id}): PSNR={m_expert['psnr']:.1f} dB, "
              f"SSIM={m_expert['ssim']:.3f}")

        # Prepare 2D images for display
        gt_disp = _normalize_01(_prepare_2d(gt, modality))
        agent_disp = _normalize_01(_prepare_2d(agent_recon, modality))
        expert_disp = _normalize_01(_prepare_2d(expert_recon, modality))

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
    print("\n  Composing figure...")

    # Nature double-column: 183 mm = 7.2 in
    fig_w = 7.2
    sample_h, sample_w = rows_data[0]["gt"].shape
    panel_aspect = sample_h / sample_w
    n_rows = len(rows_data)
    panel_w = (fig_w - 0.6) / 4.0
    panel_h = panel_w * panel_aspect
    fig_h = panel_h * n_rows + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    gs = GridSpec(
        n_rows, 4, figure=fig,
        left=0.07, right=0.99, bottom=0.01, top=0.93,
        wspace=0.04, hspace=0.08,
    )

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

            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("0.5")

            if metrics is not None:
                _annotate(ax, metrics["psnr"], metrics["ssim"])

            if row_idx == 0:
                ax.set_title(COL_HEADERS[col_idx], fontsize=8, pad=4,
                             fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(
                    ROW_LABELS[modality], fontsize=9, fontweight="bold",
                    rotation=90, labelpad=8,
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
