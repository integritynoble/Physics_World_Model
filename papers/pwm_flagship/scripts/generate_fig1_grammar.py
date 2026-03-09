#!/usr/bin/env python3
"""
Generate Figure 1: The Universal Grammar of Computational Imaging.

A single unified framework diagram showing the whole grammar:
  a: Any imaging system  →  b: Compose as OperatorGraph (11 primitives)
  →  c: Diagnose via Triad (3 gates)  →  d: Correct dominant gate
  →  e: Recovered reconstruction

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


def draw_stage_box(ax, x, y, w, h, fill, edgecolor="#CCCCCC"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15", facecolor=fill,
        edgecolor=edgecolor, linewidth=1.2, zorder=0,
    )
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
    fig, ax = plt.subplots(figsize=(16, 8.5), facecolor="white")
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.8, 8.8)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Row geometry ────────────────────────────────────────────────────────
    top_y = 5.8          # centre of top-row boxes
    top_h = 3.8          # height of top-row boxes
    bot_y = 1.0          # centre of bottom-row boxes
    bot_h = 2.2          # height of bottom-row boxes

    # ════════════════════════════════════════════════════════════════════════
    # a – Any Imaging System
    # ════════════════════════════════════════════════════════════════════════
    s1_x = 1.6
    s1_w = 2.6
    draw_stage_box(ax, s1_x, top_y, s1_w, top_h, C_STAGE1, "#B0C4DE")

    # Panel label + title (above box content, inside box)
    ax.text(s1_x - s1_w / 2 + 0.15, top_y + top_h / 2 - 0.15, "a",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s1_x, top_y + top_h / 2 - 0.45, "Any Imaging\nSystem",
            fontsize=12, ha="center", va="top", color="#3B6FA0",
            fontweight="bold", linespacing=1.3, zorder=1)

    modalities = ["CASSI", "MRI", "CT", "Cryo-EM", "OCT", "..."]
    for i, name in enumerate(modalities):
        ax.text(s1_x, top_y + 0.3 - i * 0.42, name,
                fontsize=9.5, ha="center", va="center", color="#666666",
                style="italic", zorder=1)

    # ════════════════════════════════════════════════════════════════════════
    # b – Compose: OperatorGraph
    # ════════════════════════════════════════════════════════════════════════
    s2_x = 6.5
    s2_w = 4.8
    draw_stage_box(ax, s2_x, top_y, s2_w, top_h, C_STAGE2, "#C0B0D8")

    ax.text(s2_x - s2_w / 2 + 0.15, top_y + top_h / 2 - 0.15, "b",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s2_x, top_y + top_h / 2 - 0.45, "Compose: OperatorGraph",
            fontsize=12, ha="center", va="top", color=C_ENC_T,
            fontweight="bold", zorder=1)

    # DAG chain: P → C → Λ → S → D  (vertical, left side of box)
    dag_x = s2_x - 1.2
    dag_prims = ["P", "C", "\u039B", "S", "D"]
    dag_y_top = top_y + 0.85
    dag_sp = 0.50
    nw, nh = 0.55, 0.32
    for i, p in enumerate(dag_prims):
        y = dag_y_top - i * dag_sp
        fill, tcol = PRIM_COLOR_MAP[p]
        draw_node(ax, dag_x, y, p, fill, width=nw, height=nh,
                  fontsize=10, text_color=tcol, edgecolor=tcol,
                  linewidth=0.6, shadow=True)
        if i > 0:
            draw_small_arrow(ax, dag_x, dag_y_top - (i - 1) * dag_sp - nh / 2 - 0.02,
                             dag_x, y + nh / 2 + 0.02, color=C_ARROW)

    # Example label next to DAG
    ax.text(dag_x + 0.50, dag_y_top, "e.g. CASSI", fontsize=7.5,
            ha="left", va="center", color="#999999", style="italic", zorder=1)

    # 11 primitives ribbon (right side of box, vertical column)
    ribbon_x = s2_x + 1.0
    ribbon_prims = ["P", "C", "M", "R", "\u039B",
                    "\u03A0", "F", "\u03A3", "S", "W", "D"]
    n_ribbon = len(ribbon_prims)
    ribbon_sp = 0.28
    ribbon_y_top = top_y + 0.85
    # Draw in a 2-column grid for compactness
    col1 = ribbon_prims[:6]  # P C M R Λ Π
    col2 = ribbon_prims[6:]  # F Σ S W D
    for j, rp in enumerate(col1):
        ry = ribbon_y_top - j * ribbon_sp
        fill, tcol = PRIM_COLOR_MAP[rp]
        draw_node(ax, ribbon_x, ry, rp, fill, width=0.30, height=0.22,
                  fontsize=7, text_color=tcol, edgecolor=tcol,
                  linewidth=0.4, shadow=False, zorder=2)
    for j, rp in enumerate(col2):
        ry = ribbon_y_top - j * ribbon_sp
        fill, tcol = PRIM_COLOR_MAP[rp]
        draw_node(ax, ribbon_x + 0.42, ry, rp, fill, width=0.30, height=0.22,
                  fontsize=7, text_color=tcol, edgecolor=tcol,
                  linewidth=0.4, shadow=False, zorder=2)
    ax.text(ribbon_x + 0.21, ribbon_y_top - max(len(col1), len(col2)) * ribbon_sp - 0.10,
            "11 primitives\n(see Fig. 2)",
            fontsize=7, ha="center", color="#999999", style="italic",
            linespacing=1.2, zorder=1)

    # Arrow: a → b
    draw_big_arrow(ax, s1_x + s1_w / 2 + 0.1, top_y,
                   s2_x - s2_w / 2 - 0.1, top_y, color="#9999BB")

    # ════════════════════════════════════════════════════════════════════════
    # c – Diagnose: Triad Decomposition
    # ════════════════════════════════════════════════════════════════════════
    s3_x = 13.2
    s3_w = 5.4
    draw_stage_box(ax, s3_x, top_y, s3_w, top_h, C_STAGE3, "#E0C8A0")

    ax.text(s3_x - s3_w / 2 + 0.15, top_y + top_h / 2 - 0.15, "c",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s3_x, top_y + top_h / 2 - 0.45,
            "Diagnose: Triad Decomposition",
            fontsize=12, ha="center", va="top", color="#B07020",
            fontweight="bold", zorder=1)

    # Three gate boxes (evenly spaced)
    gate_data = [
        ("Gate 1", "Information\ndeficiency", C_G1_BG, C_G1),
        ("Gate 2", "Carrier\nbudget", C_G2_BG, C_G2),
        ("Gate 3", "Operator\nmismatch", C_G3_BG, C_G3),
    ]
    gate_sp = 1.65
    gx_start = s3_x - gate_sp
    gy = top_y + 0.05
    gw, gh = 1.35, 1.45
    for k, (gname, gdesc, gbg, gcol) in enumerate(gate_data):
        gx = gx_start + k * gate_sp
        gbox = FancyBboxPatch(
            (gx - gw / 2, gy - gh / 2), gw, gh,
            boxstyle="round,pad=0.10", facecolor=gbg,
            edgecolor=gcol, linewidth=1.5, zorder=1,
        )
        ax.add_patch(gbox)
        ax.text(gx, gy + 0.28, gname, fontsize=11, ha="center",
                va="center", color=gcol, fontweight="bold", zorder=2)
        ax.text(gx, gy - 0.20, gdesc, fontsize=8.5, ha="center",
                va="center", color=gcol, linespacing=1.2, zorder=2)

    ax.text(s3_x, top_y - top_h / 2 + 0.35, "3 gates (see Fig. 3)",
            fontsize=7.5, ha="center", color="#999999", style="italic", zorder=1)

    # Arrow: b → c
    draw_big_arrow(ax, s2_x + s2_w / 2 + 0.1, top_y,
                   s3_x - s3_w / 2 - 0.1, top_y, color="#C0B080")

    # ════════════════════════════════════════════════════════════════════════
    # d – Correct: targeted intervention
    # ════════════════════════════════════════════════════════════════════════
    s4_x = 7.5
    s4_w = 6.0
    draw_stage_box(ax, s4_x, bot_y, s4_w, bot_h, C_STAGE4, "#E0B0B0")

    ax.text(s4_x - s4_w / 2 + 0.15, bot_y + bot_h / 2 - 0.10, "d",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s4_x, bot_y + bot_h / 2 - 0.35,
            "Correct: targeted intervention",
            fontsize=12, ha="center", va="top", color="#AA4040",
            fontweight="bold", zorder=1)

    corr_data = [
        ("Gate 1 dominant", "Redesign\nsampling", C_G1_BG, C_G1),
        ("Gate 2 dominant", "Improve\ncarrier budget", C_G2_BG, C_G2),
        ("Gate 3 dominant", "Calibrate\noperator", C_G3_BG, C_G3),
    ]
    corr_sp = 1.85
    cx_start = s4_x - corr_sp
    cy = bot_y - 0.18
    cw, ch = 1.55, 0.95
    for k, (clabel, cdesc, cbg, ccol) in enumerate(corr_data):
        cx = cx_start + k * corr_sp
        cbox = FancyBboxPatch(
            (cx - cw / 2, cy - ch / 2), cw, ch,
            boxstyle="round,pad=0.08", facecolor=cbg,
            edgecolor=ccol, linewidth=1.2, zorder=1,
        )
        ax.add_patch(cbox)
        ax.text(cx, cy + 0.18, clabel, fontsize=8, ha="center",
                va="center", color=ccol, fontweight="bold", zorder=2)
        ax.text(cx, cy - 0.18, cdesc, fontsize=8.5, ha="center",
                va="center", color="#555555", linespacing=1.2, zorder=2)

    # Arrow: c → d  (diagonal down-left)
    draw_big_arrow(ax, s3_x - 1.0, top_y - top_h / 2 - 0.15,
                   s4_x + 1.2, bot_y + bot_h / 2 + 0.15,
                   color="#CC9999")

    # ════════════════════════════════════════════════════════════════════════
    # e – Recover
    # ════════════════════════════════════════════════════════════════════════
    s5_x = 15.5
    s5_w = 3.8
    draw_stage_box(ax, s5_x, bot_y, s5_w, bot_h, C_STAGE5, "#A0D0A0")

    ax.text(s5_x - s5_w / 2 + 0.15, bot_y + bot_h / 2 - 0.10, "e",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(s5_x, bot_y + bot_h / 2 - 0.35, "Recover",
            fontsize=13, ha="center", va="top", color="#2E7D32",
            fontweight="bold", zorder=1)

    ax.text(s5_x, bot_y + 0.05, "Corrected\nReconstruction",
            fontsize=13, ha="center", va="center", color="#2E7D32",
            fontweight="bold", linespacing=1.3, zorder=1)
    ax.text(s5_x, bot_y - 0.60, "+0.8 to +13.9 dB",
            fontsize=10, ha="center", color="#555555",
            fontweight="bold", zorder=1)
    ax.text(s5_x, bot_y - 0.88, "no retraining required",
            fontsize=8, ha="center", color="#888888",
            style="italic", zorder=1)

    # Arrow: d → e
    draw_big_arrow(ax, s4_x + s4_w / 2 + 0.1, bot_y,
                   s5_x - s5_w / 2 - 0.1, bot_y, color="#88BB88")

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
