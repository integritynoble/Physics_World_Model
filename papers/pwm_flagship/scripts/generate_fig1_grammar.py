#!/usr/bin/env python3
"""
Generate Figure 1: The Universal Grammar of Computational Imaging.

Layout (single row):
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
C_STAGEF = "#F5F5F5"

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


def draw_small_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=10, zorder=3))


# ── Main figure ─────────────────────────────────────────────────────────────

def main(output_path: Path):
    fig, ax = plt.subplots(figsize=(18, 5.0), facecolor="white")
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Layout: single row ─────────────────────────────────────────────────
    # a → b → c → d → e   (pipeline, left to right)
    row1_y = 2.5        # row centre
    row1_h = 3.8        # row height

    # ════════════════════════════════════════════════════════════════════════
    # a – Any Imaging System
    # ════════════════════════════════════════════════════════════════════════
    s1_x, s1_w = 1.3, 2.2
    draw_stage_box(ax, s1_x, row1_y, s1_w, row1_h, C_STAGE1, "#B0C4DE")

    ax.text(s1_x - s1_w / 2 + 0.12, row1_y + row1_h / 2 - 0.12, "a",
            fontsize=14, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s1_x, row1_y + row1_h / 2 - 0.40, "Any Imaging\nSystem",
            fontsize=10, ha="center", va="top", color="#3B6FA0",
            fontweight="bold", linespacing=1.2, zorder=1)

    modalities = ["CASSI", "MRI", "CT", "Cryo-EM", "OCT", "..."]
    for i, name in enumerate(modalities):
        ax.text(s1_x, row1_y + 0.05 - i * 0.35, name,
                fontsize=8.5, ha="center", va="center", color="#666666",
                style="italic", zorder=1)

    # ════════════════════════════════════════════════════════════════════════
    # b – Compose: OperatorGraph
    # ════════════════════════════════════════════════════════════════════════
    s2_x, s2_w = 4.3, 2.4
    draw_stage_box(ax, s2_x, row1_y, s2_w, row1_h, C_STAGE2, "#C0B0D8")

    ax.text(s2_x - s2_w / 2 + 0.12, row1_y + row1_h / 2 - 0.12, "b",
            fontsize=14, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s2_x + 0.10, row1_y + row1_h / 2 - 0.25,
            "Compose:\nOperatorGraph",
            fontsize=9, ha="center", va="top", color=C_ENC_T,
            fontweight="bold", linespacing=1.15, zorder=1)

    # DAG: P → C → W → S → D
    dag_prims = ["P", "C", "W", "S", "D"]
    dag_y_top = row1_y + 0.0
    dag_sp = 0.36
    nw, nh = 0.48, 0.26
    dag_cx = s2_x - 0.15
    for i, p in enumerate(dag_prims):
        y = dag_y_top - i * dag_sp
        fill, tcol = PRIM_COLOR_MAP[p]
        draw_node(ax, dag_cx, y, p, fill, width=nw, height=nh,
                  fontsize=9, text_color=tcol, edgecolor=tcol,
                  linewidth=0.6, shadow=True)
        if i > 0:
            draw_small_arrow(ax, dag_cx,
                             dag_y_top - (i - 1) * dag_sp - nh / 2 - 0.02,
                             dag_cx, y + nh / 2 + 0.02, color=C_ARROW)

    ax.text(dag_cx + 0.45, dag_y_top, "e.g.\nCASSI", fontsize=6,
            ha="left", va="center", color="#999999", style="italic",
            linespacing=1.1, zorder=1)

    # Subtitle: mention 11 primitives between title and DAG
    ax.text(s2_x, row1_y + row1_h / 2 - 0.72,
            "(11 universal primitives)",
            fontsize=7, ha="center", va="top", color="#888888",
            style="italic", zorder=1)

    # Arrow: a → b
    draw_big_arrow(ax, s1_x + s1_w / 2 + 0.08, row1_y,
                   s2_x - s2_w / 2 - 0.08, row1_y, color="#9999BB", lw=2.0)

    # ════════════════════════════════════════════════════════════════════════
    # c – Diagnose: Triad Decomposition
    # ════════════════════════════════════════════════════════════════════════
    s3_x, s3_w = 8.5, 4.8
    draw_stage_box(ax, s3_x, row1_y, s3_w, row1_h, C_STAGE3, "#E0C8A0")

    ax.text(s3_x - s3_w / 2 + 0.12, row1_y + row1_h / 2 - 0.12, "c",
            fontsize=14, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s3_x, row1_y + row1_h / 2 - 0.30,
            "Diagnose: Triad Decomposition",
            fontsize=9.5, ha="center", va="top", color="#B07020",
            fontweight="bold", zorder=1)

    gate_data = [
        ("Gate 1", "Information\ndeficiency", C_G1_BG, C_G1),
        ("Gate 2", "Carrier\nbudget", C_G2_BG, C_G2),
        ("Gate 3", "Operator\nmismatch", C_G3_BG, C_G3),
    ]
    gate_sp = 1.45
    gx_start = s3_x - gate_sp
    gy = row1_y - 0.15
    gw, gh = 1.20, 1.35
    for k, (gname, gdesc, gbg, gcol) in enumerate(gate_data):
        gx = gx_start + k * gate_sp
        gbox = FancyBboxPatch(
            (gx - gw / 2, gy - gh / 2), gw, gh,
            boxstyle="round,pad=0.08", facecolor=gbg,
            edgecolor=gcol, linewidth=1.3, zorder=1)
        ax.add_patch(gbox)
        ax.text(gx, gy + 0.25, gname, fontsize=9, ha="center",
                va="center", color=gcol, fontweight="bold", zorder=2)
        ax.text(gx, gy - 0.20, gdesc, fontsize=7, ha="center",
                va="center", color=gcol, linespacing=1.2, zorder=2)

    # Arrow: b → c
    draw_big_arrow(ax, s2_x + s2_w / 2 + 0.08, row1_y,
                   s3_x - s3_w / 2 - 0.08, row1_y, color="#C0B080", lw=2.0)

    # ════════════════════════════════════════════════════════════════════════
    # d – Correct: targeted intervention
    # ════════════════════════════════════════════════════════════════════════
    s4_x, s4_w = 13.8, 4.8
    draw_stage_box(ax, s4_x, row1_y, s4_w, row1_h, C_STAGE4, "#E0B0B0")

    ax.text(s4_x - s4_w / 2 + 0.12, row1_y + row1_h / 2 - 0.12, "d",
            fontsize=14, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s4_x, row1_y + row1_h / 2 - 0.30,
            "Correct: targeted intervention",
            fontsize=9.5, ha="center", va="top", color="#AA4040",
            fontweight="bold", zorder=1)

    corr_data = [
        ("Gate 1\ndominant", "Redesign\nsampling", C_G1_BG, C_G1),
        ("Gate 2\ndominant", "Improve\ncarrier", C_G2_BG, C_G2),
        ("Gate 3\ndominant", "Calibrate\noperator", C_G3_BG, C_G3),
    ]
    corr_sp = 1.45
    cx_start = s4_x - corr_sp
    cy = row1_y - 0.15
    cw, ch = 1.20, 1.35
    for k, (clabel, cdesc, cbg, ccol) in enumerate(corr_data):
        cx = cx_start + k * corr_sp
        cbox = FancyBboxPatch(
            (cx - cw / 2, cy - ch / 2), cw, ch,
            boxstyle="round,pad=0.08", facecolor=cbg,
            edgecolor=ccol, linewidth=1.0, zorder=1)
        ax.add_patch(cbox)
        ax.text(cx, cy + 0.25, clabel, fontsize=7, ha="center",
                va="center", color=ccol, fontweight="bold",
                linespacing=1.1, zorder=2)
        ax.text(cx, cy - 0.25, cdesc, fontsize=7, ha="center",
                va="center", color="#555555", linespacing=1.1, zorder=2)

    # Arrow: c → d
    draw_big_arrow(ax, s3_x + s3_w / 2 + 0.08, row1_y,
                   s4_x - s4_w / 2 - 0.08, row1_y, color="#CC9999", lw=2.0)

    # ════════════════════════════════════════════════════════════════════════
    # e – Recover
    # ════════════════════════════════════════════════════════════════════════
    s5_x, s5_w = 18.8, 2.8
    draw_stage_box(ax, s5_x, row1_y, s5_w, row1_h, C_STAGE5, "#A0D0A0")

    ax.text(s5_x - s5_w / 2 + 0.12, row1_y + row1_h / 2 - 0.12, "e",
            fontsize=14, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s5_x, row1_y + row1_h / 2 - 0.30, "Recover",
            fontsize=11, ha="center", va="top", color="#2E7D32",
            fontweight="bold", zorder=1)

    ax.text(s5_x, row1_y - 0.05, "Corrected\nReconstruction",
            fontsize=10, ha="center", va="center", color="#2E7D32",
            fontweight="bold", linespacing=1.3, zorder=1)
    ax.text(s5_x, row1_y - 0.75, "+0.8 to +13.9 dB",
            fontsize=8.5, ha="center", color="#555555",
            fontweight="bold", zorder=1)
    ax.text(s5_x, row1_y - 1.05, "no retraining",
            fontsize=7, ha="center", color="#888888",
            style="italic", zorder=1)

    # Arrow: d → e
    draw_big_arrow(ax, s4_x + s4_w / 2 + 0.08, row1_y,
                   s5_x - s5_w / 2 - 0.08, row1_y, color="#88BB88", lw=2.0)

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
