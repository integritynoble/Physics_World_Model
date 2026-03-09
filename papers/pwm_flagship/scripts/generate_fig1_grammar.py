#!/usr/bin/env python3
"""
Generate Figure 1: The Universal Grammar of Computational Imaging.

Layout (single row, equal spacing):
  a (Any system) → b (Compose: OperatorGraph) → c (Diagnose: 3 gates)
    → d (Correct dominant gate) → e (Recover)

Usage:
    python generate_fig1_grammar.py [--output figures/fig1_overview.pdf]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# ── Colours ─────────────────────────────────────────────────────────────────

C_GEN = "#CADCF0";  C_GEN_T = "#2B5A8C"
C_ENC = "#D4C8EC";  C_ENC_T = "#5A3D8C"
C_TRN = "#F2E0B0";  C_TRN_T = "#8C6D2B"
C_DET = "#F2C4C0";  C_DET_T = "#8C3A35"

C_G1 = "#3B7DD8";  C_G1_BG = "#D6E5F7"
C_G2 = "#D98C2B";  C_G2_BG = "#F7E8D0"
C_G3 = "#CC4444";  C_G3_BG = "#F7D6D6"

C_STAGE1 = "#E8EFF8"
C_STAGE2 = "#EDE8F5"
C_STAGE3 = "#FFF3E0"
C_STAGE4 = "#FCE4EC"
C_STAGE5 = "#E8F5E9"

C_TEXT  = "#333333"
C_ARROW = "#777777"
C_EDGE  = "#BBBBBB"

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
        edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
    if shadow:
        box.set_path_effects([
            pe.withSimplePatchShadow(offset=(1.0, -1.0),
                                     shadow_rgbFace="#cccccc", alpha=0.2)])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight, zorder=zorder + 1)


def draw_stage_box(ax, x, y, w, h, fill, edgecolor="#CCCCCC"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15", facecolor=fill,
        edgecolor=edgecolor, linewidth=1.2, zorder=0)
    box.set_path_effects([
        pe.withSimplePatchShadow(offset=(2, -2),
                                 shadow_rgbFace="#dddddd", alpha=0.3)])
    ax.add_patch(box)


def draw_big_arrow(ax, x0, y0, x1, y1, color="#999999", lw=2.5):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=20, zorder=5))


def draw_small_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=12, zorder=3))


# ── Main figure ─────────────────────────────────────────────────────────────

def main(output_path: Path):
    # Smaller figsize → less downscaling in paper → bigger rendered text
    # At textwidth=183mm≈7.2", scale factor ≈ 7.2/12 = 0.6
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Layout: single row with EQUAL gaps ─────────────────────────────────
    row_y = 3.1         # vertical centre
    row_h = 4.8         # panel height

    # Panel widths
    w_a, w_b, w_c, w_d, w_e = 1.4, 2.0, 3.9, 3.9, 1.8
    total_panels = w_a + w_b + w_c + w_d + w_e  # 12.7
    margin = 0.25
    gap = (15.5 - 2 * margin - total_panels) / 4  # ≈ 0.575

    # Compute centres
    a_left = margin
    a_cx = a_left + w_a / 2
    b_left = a_left + w_a + gap
    b_cx = b_left + w_b / 2
    c_left = b_left + w_b + gap
    c_cx = c_left + w_c / 2
    d_left = c_left + w_c + gap
    d_cx = d_left + w_d / 2
    e_left = d_left + w_d + gap
    e_cx = e_left + w_e / 2

    # ════════════════════════════════════════════════════════════════════════
    # a – Any Imaging System
    # ════════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, a_cx, row_y, w_a, row_h, C_STAGE1, "#B0C4DE")

    ax.text(a_left + 0.10, row_y + row_h / 2 - 0.10, "a",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(a_cx, row_y + row_h / 2 - 0.35, "Any Imaging\nSystem",
            fontsize=13, ha="center", va="top", color="#3B6FA0",
            fontweight="bold", linespacing=1.2, zorder=1)

    modalities = ["CASSI", "MRI", "CT", "Cryo-EM", "OCT", "..."]
    for i, name in enumerate(modalities):
        ax.text(a_cx, row_y + 0.30 - i * 0.42, name,
                fontsize=11, ha="center", va="center", color="#666666",
                style="italic", zorder=1)

    # ════════════════════════════════════════════════════════════════════════
    # b – Compose: OperatorGraph
    # ════════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, b_cx, row_y, w_b, row_h, C_STAGE2, "#C0B0D8")

    ax.text(b_left + 0.08, row_y + row_h / 2 - 0.08, "b",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(b_cx, row_y + row_h / 2 - 0.20,
            "Compose:\nOperatorGraph",
            fontsize=11, ha="center", va="top", color=C_ENC_T,
            fontweight="bold", linespacing=1.15, zorder=1)

    # Subtitle: reference Fig. 2a
    ax.text(b_cx, row_y + row_h / 2 - 0.72,
            "(11 primitives;\nFig. 2a)",
            fontsize=8, ha="center", va="top", color="#888888",
            style="italic", linespacing=1.1, zorder=1)

    # DAG: P → C → W → S → D
    dag_prims = ["P", "C", "W", "S", "D"]
    dag_y_top = row_y + 0.30
    dag_sp = 0.50
    nw, nh = 0.55, 0.32
    dag_cx = b_cx - 0.10
    for i, p in enumerate(dag_prims):
        y = dag_y_top - i * dag_sp
        fill, tcol = PRIM_COLOR_MAP[p]
        draw_node(ax, dag_cx, y, p, fill, width=nw, height=nh,
                  fontsize=12, text_color=tcol, edgecolor=tcol,
                  linewidth=0.7, shadow=True)
        if i > 0:
            draw_small_arrow(ax, dag_cx,
                             dag_y_top - (i - 1) * dag_sp - nh / 2 - 0.02,
                             dag_cx, y + nh / 2 + 0.02, color=C_ARROW)

    ax.text(dag_cx + 0.50, dag_y_top, "e.g.\nCASSI", fontsize=8,
            ha="left", va="center", color="#999999", style="italic",
            linespacing=1.1, zorder=1)

    # Arrow: a → b
    draw_big_arrow(ax, a_left + w_a + 0.06, row_y,
                   b_left - 0.06, row_y, color="#9999BB", lw=2.5)

    # ════════════════════════════════════════════════════════════════════════
    # c – Diagnose: Triad Decomposition
    # ════════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, c_cx, row_y, w_c, row_h, C_STAGE3, "#E0C8A0")

    ax.text(c_left + 0.08, row_y + row_h / 2 - 0.08, "c",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(c_cx, row_y + row_h / 2 - 0.20,
            "Diagnose:\nTriad Decomposition",
            fontsize=12, ha="center", va="top", color="#B07020",
            fontweight="bold", linespacing=1.15, zorder=1)

    gate_data = [
        ("Gate 1", "Information\ndeficiency", C_G1_BG, C_G1),
        ("Gate 2", "Carrier\nbudget", C_G2_BG, C_G2),
        ("Gate 3", "Operator\nmismatch", C_G3_BG, C_G3),
    ]
    gate_sp = 1.20
    gy = row_y - 0.20
    gw, gh = 1.05, 1.50
    for k, (gname, gdesc, gbg, gcol) in enumerate(gate_data):
        gx = c_cx + (k - 1) * gate_sp
        gbox = FancyBboxPatch(
            (gx - gw / 2, gy - gh / 2), gw, gh,
            boxstyle="round,pad=0.08", facecolor=gbg,
            edgecolor=gcol, linewidth=1.3, zorder=1)
        ax.add_patch(gbox)
        ax.text(gx, gy + 0.30, gname, fontsize=12, ha="center",
                va="center", color=gcol, fontweight="bold", zorder=2)
        ax.text(gx, gy - 0.22, gdesc, fontsize=9, ha="center",
                va="center", color=gcol, linespacing=1.15, zorder=2)

    # Arrow: b → c
    draw_big_arrow(ax, b_left + w_b + 0.06, row_y,
                   c_left - 0.06, row_y, color="#C0B080", lw=2.5)

    # ════════════════════════════════════════════════════════════════════════
    # d – Correct: targeted intervention
    # ════════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, d_cx, row_y, w_d, row_h, C_STAGE4, "#E0B0B0")

    ax.text(d_left + 0.08, row_y + row_h / 2 - 0.08, "d",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(d_cx, row_y + row_h / 2 - 0.20,
            "Correct:\ntargeted intervention",
            fontsize=12, ha="center", va="top", color="#AA4040",
            fontweight="bold", linespacing=1.15, zorder=1)

    corr_data = [
        ("Gate 1\ndominant", "Redesign\nsampling", C_G1_BG, C_G1),
        ("Gate 2\ndominant", "Improve\ncarrier", C_G2_BG, C_G2),
        ("Gate 3\ndominant", "Calibrate\noperator", C_G3_BG, C_G3),
    ]
    corr_sp = 1.20
    cy = row_y - 0.20
    cw, ch = 1.05, 1.50
    for k, (clabel, cdesc, cbg, ccol) in enumerate(corr_data):
        cx = d_cx + (k - 1) * corr_sp
        cbox = FancyBboxPatch(
            (cx - cw / 2, cy - ch / 2), cw, ch,
            boxstyle="round,pad=0.08", facecolor=cbg,
            edgecolor=ccol, linewidth=1.0, zorder=1)
        ax.add_patch(cbox)
        ax.text(cx, cy + 0.30, clabel, fontsize=10, ha="center",
                va="center", color=ccol, fontweight="bold",
                linespacing=1.1, zorder=2)
        ax.text(cx, cy - 0.28, cdesc, fontsize=9, ha="center",
                va="center", color="#555555", linespacing=1.1, zorder=2)

    # Arrow: c → d
    draw_big_arrow(ax, c_left + w_c + 0.06, row_y,
                   d_left - 0.06, row_y, color="#CC9999", lw=2.5)

    # ════════════════════════════════════════════════════════════════════════
    # e – Recover
    # ════════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, e_cx, row_y, w_e, row_h, C_STAGE5, "#A0D0A0")

    ax.text(e_left + 0.08, row_y + row_h / 2 - 0.08, "e",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(e_cx, row_y + row_h / 2 - 0.20, "Recover",
            fontsize=14, ha="center", va="top", color="#2E7D32",
            fontweight="bold", zorder=1)

    ax.text(e_cx, row_y + 0.25, "Corrected\nReconstruction",
            fontsize=12, ha="center", va="center", color="#2E7D32",
            fontweight="bold", linespacing=1.3, zorder=1)
    ax.text(e_cx, row_y - 0.55, "+0.8 to\n+13.9 dB",
            fontsize=11, ha="center", color="#555555",
            fontweight="bold", linespacing=1.2, zorder=1)
    ax.text(e_cx, row_y - 1.15, "no retraining",
            fontsize=9, ha="center", color="#888888",
            style="italic", zorder=1)

    # Arrow: d → e
    draw_big_arrow(ax, d_left + w_d + 0.06, row_y,
                   e_left - 0.06, row_y, color="#88BB88", lw=2.5)

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
