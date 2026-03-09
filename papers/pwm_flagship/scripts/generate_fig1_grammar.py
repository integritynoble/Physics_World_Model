#!/usr/bin/env python3
"""
Generate Figure 1: The Universal Grammar of Computational Imaging.

A single unified framework diagram showing the whole grammar:
  Top-left:     Any imaging system
  Top-center:   Compose as DAG from 11 primitives (alphabet)
  Top-right:    Diagnose via 3 gates (rules)
  Bottom-left:  Correct the dominant gate
  Bottom-right: Recovered reconstruction

This is the big-picture overview. Figure 2 details the 11 primitives;
Figure 3 details the 3 gates.

Usage:
    python generate_fig1_grammar.py [--output figures/fig1_overview.pdf]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# ── Colours ─────────────────────────────────────────────────────────────────

# Primitive roles
C_GEN = "#CADCF0";  C_GEN_T = "#2B5A8C"
C_ENC = "#D4C8EC";  C_ENC_T = "#5A3D8C"
C_TRN = "#F2E0B0";  C_TRN_T = "#8C6D2B"
C_DET = "#F2C4C0";  C_DET_T = "#8C3A35"

# Gates
C_G1 = "#3B7DD8";  C_G1_BG = "#D6E5F7"
C_G2 = "#D98C2B";  C_G2_BG = "#F7E8D0"
C_G3 = "#CC4444";  C_G3_BG = "#F7D6D6"

# Framework stage boxes
C_STAGE1 = "#E8EFF8"   # light blue – input
C_STAGE2 = "#EDE8F5"   # light lavender – compose
C_STAGE3 = "#FFF3E0"   # light amber – diagnose
C_STAGE4 = "#FCE4EC"   # light pink – correct
C_STAGE5 = "#E8F5E9"   # light green – recover

C_TEXT   = "#333333"
C_ARROW  = "#777777"
C_EDGE   = "#BBBBBB"
C_CORRECT = "#4CAF78"

PRIM_COLOR_MAP = {
    "P": (C_GEN, C_GEN_T), "C": (C_ENC, C_ENC_T), "M": (C_ENC, C_ENC_T),
    "R": (C_ENC, C_ENC_T), "\u039B": (C_ENC, C_ENC_T), "\u03A0": (C_ENC, C_ENC_T),
    "F": (C_TRN, C_TRN_T), "\u03A3": (C_TRN, C_TRN_T),
    "S": (C_DET, C_DET_T), "W": (C_DET, C_DET_T), "D": (C_DET, C_DET_T),
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def draw_node(ax, x, y, label, fill, width=0.7, height=0.40,
              fontsize=11, edgecolor=C_EDGE, linewidth=0.8,
              text_color=C_TEXT, bold=True, shadow=False, zorder=2):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.06", facecolor=fill,
        edgecolor=edgecolor, linewidth=linewidth, zorder=zorder,
    )
    if shadow:
        box.set_path_effects([
            pe.withSimplePatchShadow(offset=(1.0, -1.0),
                                     shadow_rgbFace="#cccccc", alpha=0.2)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight, zorder=zorder + 1)


def draw_stage_box(ax, x, y, w, h, fill, edgecolor="#CCCCCC", label="",
                   label_fs=13, label_color=C_TEXT):
    """Draw a large rounded stage box with a title at the top."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15", facecolor=fill,
        edgecolor=edgecolor, linewidth=1.2, zorder=0,
    )
    box.set_path_effects([
        pe.withSimplePatchShadow(offset=(2, -2),
                                 shadow_rgbFace="#dddddd", alpha=0.3)])
    ax.add_patch(box)
    if label:
        ax.text(x, y + h / 2 - 0.30, label, ha="center", va="top",
                fontsize=label_fs, fontweight="bold", color=label_color,
                zorder=1)


def draw_big_arrow(ax, x0, y0, x1, y1, color="#999999", lw=2.5):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=20, zorder=5,
    )
    ax.add_patch(arrow)


def draw_small_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.0):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=10, zorder=3,
    )
    ax.add_patch(arrow)


# ── Main figure ─────────────────────────────────────────────────────────────

def main(output_path: Path):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1.5, 9.5)
    ax.axis("off")
    ax.set_facecolor("white")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1: Any Imaging System (left)
    # ════════════════════════════════════════════════════════════════════════
    s1_x, s1_y = 1.5, 6.0
    draw_stage_box(ax, s1_x, s1_y, 2.8, 3.2, C_STAGE1,
                   edgecolor="#B0C4DE", label="Any Imaging\nSystem",
                   label_fs=12, label_color="#3B6FA0")
    # Example modality names
    for i, name in enumerate(["CASSI", "MRI", "CT", "Cryo-EM", "..."]):
        ax.text(s1_x, s1_y - 0.2 - i * 0.45, name,
                fontsize=9, ha="center", va="center", color="#666666",
                style="italic")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2: Compose as DAG from 11 Primitives (centre-top)
    # ════════════════════════════════════════════════════════════════════════
    s2_x, s2_y = 6.5, 6.0
    draw_stage_box(ax, s2_x, s2_y, 4.5, 3.2, C_STAGE2,
                   edgecolor="#C0B0D8",
                   label="Compose: OperatorGraph",
                   label_fs=12, label_color=C_ENC_T)

    # Compact DAG: P → C → Λ → S → D
    dag_prims = ["P", "C", "\u039B", "S", "D"]
    dag_y_top = s2_y + 0.55
    dag_sp = 0.48
    for i, p in enumerate(dag_prims):
        y = dag_y_top - i * dag_sp
        fill, tcol = PRIM_COLOR_MAP[p]
        draw_node(ax, s2_x, y, p, fill, width=0.55, height=0.32,
                  fontsize=10, text_color=tcol, edgecolor=tcol,
                  linewidth=0.6, shadow=True, zorder=2)
        if i > 0:
            draw_small_arrow(ax, s2_x, dag_y_top - (i - 1) * dag_sp - 0.16,
                             s2_x, y + 0.16, color=C_ARROW)

    # 11 primitives ribbon below DAG
    ribbon_y = s2_y - 1.35
    ribbon_prims = ["P", "C", "M", "R", "\u039B",
                    "\u03A0", "F", "\u03A3", "S", "W", "D"]
    rx_start = s2_x - 2.0
    for j, rp in enumerate(ribbon_prims):
        rx = rx_start + j * 0.38
        fill, tcol = PRIM_COLOR_MAP[rp]
        draw_node(ax, rx, ribbon_y, rp, fill, width=0.32, height=0.26,
                  fontsize=7, text_color=tcol, edgecolor=tcol,
                  linewidth=0.4, shadow=False, zorder=2)
    ax.text(s2_x, ribbon_y - 0.28, "11 primitives  (Fig. 2)",
            fontsize=7.5, ha="center", color="#888888", style="italic")

    # Arrow: Stage 1 → Stage 2
    draw_big_arrow(ax, s1_x + 1.5, s1_y, s2_x - 2.4, s2_y, color="#9999BB")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3: Diagnose via 3 Gates (right-top)
    # ════════════════════════════════════════════════════════════════════════
    s3_x, s3_y = 12.5, 6.0
    draw_stage_box(ax, s3_x, s3_y, 5.0, 3.2, C_STAGE3,
                   edgecolor="#E0C8A0",
                   label="Diagnose: Triad Decomposition",
                   label_fs=12, label_color="#B07020")

    # Three gate boxes
    gate_data = [
        ("Gate 1", "Information\ndeficiency", C_G1_BG, C_G1),
        ("Gate 2", "Carrier\nbudget", C_G2_BG, C_G2),
        ("Gate 3", "Operator\nmismatch", C_G3_BG, C_G3),
    ]
    gx_start = s3_x - 1.7
    for k, (gname, gdesc, gbg, gcol) in enumerate(gate_data):
        gx = gx_start + k * 1.7
        gy = s3_y - 0.1
        # Gate box
        gbox = FancyBboxPatch(
            (gx - 0.65, gy - 0.65), 1.30, 1.30,
            boxstyle="round,pad=0.10", facecolor=gbg,
            edgecolor=gcol, linewidth=1.5, zorder=1,
        )
        ax.add_patch(gbox)
        ax.text(gx, gy + 0.25, gname, fontsize=10, ha="center",
                va="center", color=gcol, fontweight="bold", zorder=2)
        ax.text(gx, gy - 0.20, gdesc, fontsize=8, ha="center",
                va="center", color=gcol, linespacing=1.2, zorder=2)

    ax.text(s3_x, s3_y - 1.30, "3 gates  (Fig. 3)",
            fontsize=7.5, ha="center", color="#888888", style="italic")

    # Arrow: Stage 2 → Stage 3
    draw_big_arrow(ax, s2_x + 2.4, s2_y, s3_x - 2.7, s3_y, color="#C0B080")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4: Correct dominant gate (centre-bottom)
    # ════════════════════════════════════════════════════════════════════════
    s4_x, s4_y = 8.0, 1.5
    draw_stage_box(ax, s4_x, s4_y, 5.5, 2.4, C_STAGE4,
                   edgecolor="#E0B0B0",
                   label="Correct: targeted intervention",
                   label_fs=12, label_color="#AA4040")

    # Three correction strategies
    corr_data = [
        ("Gate 1 dominant", "Redesign\nsampling", C_G1_BG, C_G1),
        ("Gate 2 dominant", "Improve\ncarrier budget", C_G2_BG, C_G2),
        ("Gate 3 dominant", "Calibrate\noperator", C_G3_BG, C_G3),
    ]
    cx_start = s4_x - 2.0
    for k, (clabel, cdesc, cbg, ccol) in enumerate(corr_data):
        cx = cx_start + k * 2.0
        cy = s4_y - 0.3
        cbox = FancyBboxPatch(
            (cx - 0.75, cy - 0.55), 1.50, 1.00,
            boxstyle="round,pad=0.08", facecolor=cbg,
            edgecolor=ccol, linewidth=1.2, zorder=1,
        )
        ax.add_patch(cbox)
        ax.text(cx, cy + 0.15, clabel, fontsize=7.5, ha="center",
                va="center", color=ccol, fontweight="bold", zorder=2)
        ax.text(cx, cy - 0.22, cdesc, fontsize=8, ha="center",
                va="center", color="#555555", linespacing=1.2, zorder=2)

    # Arrow: Stage 3 → Stage 4  (diagonal down-left)
    draw_big_arrow(ax, s3_x - 0.5, s3_y - 1.7, s4_x + 1.5, s4_y + 1.35,
                   color="#CC9999")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 5: Corrected Reconstruction (right-bottom)
    # ════════════════════════════════════════════════════════════════════════
    s5_x, s5_y = 16.0, 1.5
    draw_stage_box(ax, s5_x, s5_y, 3.5, 2.4, C_STAGE5,
                   edgecolor="#A0D0A0",
                   label="Recover",
                   label_fs=13, label_color="#2E7D32")

    ax.text(s5_x, s5_y - 0.05, "Corrected\nReconstruction",
            fontsize=13, ha="center", va="center", color="#2E7D32",
            fontweight="bold", linespacing=1.3)
    ax.text(s5_x, s5_y - 0.80, "+0.8 to +13.9 dB",
            fontsize=10, ha="center", color="#555555", fontweight="bold")
    ax.text(s5_x, s5_y - 1.05, "no retraining required",
            fontsize=8, ha="center", color="#888888", style="italic")

    # Arrow: Stage 4 → Stage 5
    draw_big_arrow(ax, s4_x + 2.9, s4_y, s5_x - 1.9, s5_y,
                   color="#88BB88")

    # ════════════════════════════════════════════════════════════════════════
    # Title banner
    # ════════════════════════════════════════════════════════════════════════
    ax.text(10.0, 9.0,
            "The Universal Grammar of Computational Imaging",
            fontsize=16, ha="center", va="center", color=C_TEXT,
            fontweight="bold")
    ax.text(10.0, 8.45,
            "11 primitives (alphabet)  +  3 gates (rules)  =  "
            "design, diagnose, and correct any imaging system",
            fontsize=10, ha="center", va="center", color="#666666")

    # ════════════════════════════════════════════════════════════════════════
    # Panel labels
    # ════════════════════════════════════════════════════════════════════════
    panels = [
        (s1_x - 1.3, s1_y + 1.5, "a"),
        (s2_x - 2.15, s2_y + 1.5, "b"),
        (s3_x - 2.4, s3_y + 1.5, "c"),
        (s4_x - 2.65, s4_y + 1.1, "d"),
        (s5_x - 1.65, s5_y + 1.1, "e"),
    ]
    for px, py, pl in panels:
        ax.text(px, py, pl, fontsize=18, fontweight="bold", color=C_TEXT)

    # ════════════════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════════════════
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {output_path}")

    png_path = output_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 1")
    default_out = (
        Path(__file__).resolve().parent.parent / "figures" / "fig1_overview.pdf"
    )
    parser.add_argument("--output", type=Path, default=default_out)
    args = parser.parse_args()
    main(args.output)
