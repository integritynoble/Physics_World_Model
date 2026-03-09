#!/usr/bin/env python3
"""
Generate Figure 1: The Universal Grammar of Computational Imaging.

Panels:
  a – The 11 physically typed primitives, grouped by role
  b – Three modalities (CASSI, MRI, CT) as typed DAGs over the same alphabet
  c – The 3 gates overlaid on a CASSI DAG
  d – Grammar in action: diagnose dominant gate → correct → recover

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
import numpy as np

# ── Global style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# ── Colour palette (Nature-style muted tones) ──────────────────────────────

# Primitive role colours
C_GEN = "#CADCF0"   # generation  – soft blue
C_ENC = "#D4C8EC"   # encoding    – soft lavender
C_TRN = "#F2E0B0"   # transform   – soft gold
C_DET = "#F2C4C0"   # detection   – soft coral

# Darker accent for text inside nodes by role
C_GEN_T = "#2B5A8C"
C_ENC_T = "#5A3D8C"
C_TRN_T = "#8C6D2B"
C_DET_T = "#8C3A35"

# Gate colours
C_G1 = "#3B7DD8"  # blue   – information deficiency
C_G2 = "#D98C2B"  # amber  – carrier budget
C_G3 = "#CC4444"  # red    – operator mismatch

C_G1_BG = "#D6E5F7"
C_G2_BG = "#F7E8D0"
C_G3_BG = "#F7D6D6"

# Neutral
C_TEXT   = "#333333"
C_ARROW  = "#777777"
C_EDGE   = "#AAAAAA"
C_CORRECT = "#4CAF78"
C_PANEL_BG = "#F7F8FA"

PRIM_COLOR_MAP = {
    "P": (C_GEN, C_GEN_T),
    "C": (C_ENC, C_ENC_T), "M": (C_ENC, C_ENC_T), "R": (C_ENC, C_ENC_T),
    "\u039B": (C_ENC, C_ENC_T), "\u03A0": (C_ENC, C_ENC_T),
    "F": (C_TRN, C_TRN_T), "\u03A3": (C_TRN, C_TRN_T),
    "S": (C_DET, C_DET_T), "W": (C_DET, C_DET_T), "D": (C_DET, C_DET_T),
}


# ── Helper functions ────────────────────────────────────────────────────────

def draw_node(ax, x, y, label, fill, width=0.7, height=0.40,
              fontsize=11, edgecolor=C_EDGE, linewidth=0.8,
              text_color=C_TEXT, bold=True, shadow=False, zorder=2):
    """Draw a rounded-rectangle node with centred label."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.06", facecolor=fill,
        edgecolor=edgecolor, linewidth=linewidth, zorder=zorder,
    )
    if shadow:
        box.set_path_effects([
            pe.withSimplePatchShadow(offset=(1.2, -1.2), shadow_rgbFace="#cccccc",
                                     alpha=0.25),
        ])
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=zorder + 1)
    return box


def draw_arrow(ax, x0, y0, x1, y1, color=C_ARROW, linewidth=1.0,
               style="-|>", mutation_scale=10):
    """Draw a simple arrow between two points."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, color=color,
        linewidth=linewidth, mutation_scale=mutation_scale, zorder=1,
    )
    ax.add_patch(arrow)


def draw_thick_arrow(ax, x0, y0, x1, y1, color=C_ARROW):
    """Draw a thick chevron arrow for major transitions."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", color=color,
        linewidth=2.0, mutation_scale=16, zorder=5,
    )
    ax.add_patch(arrow)


def draw_dag_vertical(ax, x_center, y_top, primitives, spacing=0.58,
                      node_w=0.62, node_h=0.34, fontsize=10):
    """Draw a vertical chain of primitive nodes with arrows. Uses role colours."""
    positions = []
    for i, prim in enumerate(primitives):
        y = y_top - i * spacing
        fill, tcol = PRIM_COLOR_MAP.get(prim, (C_EDGE, C_TEXT))
        draw_node(ax, x_center, y, prim, fill, width=node_w, height=node_h,
                  fontsize=fontsize, text_color=tcol, shadow=True)
        positions.append((x_center, y))
        if i > 0:
            draw_arrow(ax, x_center, positions[i - 1][1] - node_h / 2 - 0.02,
                       x_center, y + node_h / 2 + 0.02, color=C_ARROW)
    return positions


def panel_bg(ax, color=C_PANEL_BG):
    """Add subtle panel background."""
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Panel A: The 11 primitives grouped by role ──────────────────────────────

def panel_a(ax):
    ax.set_xlim(-0.3, 7.0)
    ax.set_ylim(-0.6, 4.5)
    ax.axis("off")
    panel_bg(ax)

    ax.text(-0.1, 4.25, "a", fontsize=18, fontweight="bold", color=C_TEXT)
    ax.text(0.35, 4.25, "11 Universal Primitives", fontsize=11,
            fontweight="bold", color="#555555", va="center")

    groups = [
        ("Generation", [("P", "source \u2192 field")], C_GEN, C_GEN_T),
        ("Encoding",   [("C", "aperture"), ("M", "modulation"),
                        ("R", "rotation"), ("\u039B", "dispersion"),
                        ("\u03A0", "phase")], C_ENC, C_ENC_T),
        ("Transform",  [("F", "Fourier"), ("\u03A3", "integration")], C_TRN, C_TRN_T),
        ("Detection",  [("S", "sampling"), ("W", "weighting"),
                        ("D", "field \u2192 meas.")], C_DET, C_DET_T),
    ]

    y = 3.55
    for group_label, prims, fill, tcol in groups:
        # Group background band
        band_w = 0.85 * len(prims) + 0.6
        band = FancyBboxPatch(
            (1.15, y - 0.30), band_w, 0.60,
            boxstyle="round,pad=0.08", facecolor=fill, alpha=0.25,
            edgecolor="none", zorder=0,
        )
        ax.add_patch(band)

        # Group label
        ax.text(0.05, y, group_label, fontsize=9, fontweight="bold",
                color="#666666", va="center")

        # Primitives
        x = 1.55
        for sym, sig in prims:
            draw_node(ax, x, y, sym, fill, width=0.52, height=0.38,
                      fontsize=12, text_color=tcol, edgecolor=tcol,
                      linewidth=0.6, shadow=True)
            ax.text(x, y - 0.32, sig, fontsize=6.5, ha="center",
                    va="top", color="#888888", style="italic")
            x += 0.85

        y -= 1.05


# ── Panel B: Three modalities as DAGs ───────────────────────────────────────

def panel_b(ax):
    ax.set_xlim(-0.3, 5.8)
    ax.set_ylim(-0.6, 4.5)
    ax.axis("off")
    panel_bg(ax)

    ax.text(-0.1, 4.25, "b", fontsize=18, fontweight="bold", color=C_TEXT)
    ax.text(0.35, 4.25, "Same alphabet, different wiring", fontsize=11,
            fontweight="bold", color="#555555", va="center")

    # Modality title colours
    title_cols = {"CASSI": "#3B6FA0", "MRI": "#7B5030", "CT": "#A04030"}
    chains = {
        "CASSI": ["P", "C", "\u039B", "S", "D"],
        "MRI":   ["P", "W", "F", "S", "D"],
        "CT":    ["P", "R", "\u03A3", "S", "D"],
    }

    x_positions = [0.8, 2.6, 4.4]
    for (name, prims), xc in zip(chains.items(), x_positions):
        ax.text(xc, 3.85, name, fontsize=11, fontweight="bold",
                ha="center", color=title_cols[name])
        draw_dag_vertical(ax, xc, 3.4, prims, spacing=0.56,
                          node_w=0.56, node_h=0.33, fontsize=10)

    # Bracket annotation at bottom
    ax.annotate("", xy=(0.4, -0.30), xytext=(5.0, -0.30),
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.8))
    ax.text(2.7, -0.48, "Any modality = a DAG over the same 11 primitives",
            fontsize=7.5, ha="center", color="#999999", style="italic")


# ── Panel C: 3 gates overlaid on a DAG ─────────────────────────────────────

def panel_c(ax):
    ax.set_xlim(-0.8, 11.5)
    ax.set_ylim(-0.6, 3.0)
    ax.axis("off")
    panel_bg(ax, "#FAFBFD")

    ax.text(-0.5, 2.7, "c", fontsize=18, fontweight="bold", color=C_TEXT)
    ax.text(-0.0, 2.7, "Three gates on the OperatorGraph",
            fontsize=11, fontweight="bold", color="#555555", va="center")

    # Horizontal CASSI DAG — larger, centred
    prims = ["P", "C", "\u039B", "S", "D"]
    node_w, node_h = 0.82, 0.50
    gap = 1.8
    x0 = 1.8
    y_mid = 1.2
    positions = []
    for i, p in enumerate(prims):
        x = x0 + i * gap
        fill, tcol = PRIM_COLOR_MAP[p]
        draw_node(ax, x, y_mid, p, fill, width=node_w, height=node_h,
                  fontsize=14, text_color=tcol, edgecolor=tcol,
                  linewidth=0.8, shadow=True)
        positions.append((x, y_mid))
        if i > 0:
            draw_arrow(ax, positions[i - 1][0] + node_w / 2 + 0.06, y_mid,
                       x - node_w / 2 - 0.06, y_mid,
                       color=C_ARROW, linewidth=1.2, mutation_scale=12)

    # ── Gate 3: Operator mismatch (at C node) ──
    g3x = positions[1][0]
    # Highlight ring around C
    ring3 = mpatches.FancyBboxPatch(
        (g3x - node_w / 2 - 0.10, y_mid - node_h / 2 - 0.10),
        node_w + 0.20, node_h + 0.20,
        boxstyle="round,pad=0.06", facecolor="none",
        edgecolor=C_G3, linewidth=2.5, linestyle="-", zorder=4,
    )
    ax.add_patch(ring3)
    # Label box above
    lbl3 = FancyBboxPatch(
        (g3x - 0.85, y_mid + node_h / 2 + 0.25), 1.70, 0.55,
        boxstyle="round,pad=0.08", facecolor=C_G3_BG,
        edgecolor=C_G3, linewidth=1.2, zorder=3,
    )
    ax.add_patch(lbl3)
    ax.text(g3x, y_mid + node_h / 2 + 0.52,
            "Gate 3\nOperator mismatch", fontsize=8.5, ha="center",
            va="center", color=C_G3, fontweight="bold", linespacing=1.3,
            zorder=4)
    # Connector
    draw_arrow(ax, g3x, y_mid + node_h / 2 + 0.25,
               g3x, y_mid + node_h / 2 + 0.12,
               color=C_G3, linewidth=1.2, style="-|>", mutation_scale=8)

    # ── Gate 1: Information deficiency (at S node) ──
    g1x = positions[3][0]
    ring1 = mpatches.FancyBboxPatch(
        (g1x - node_w / 2 - 0.10, y_mid - node_h / 2 - 0.10),
        node_w + 0.20, node_h + 0.20,
        boxstyle="round,pad=0.06", facecolor="none",
        edgecolor=C_G1, linewidth=2.5, linestyle="-", zorder=4,
    )
    ax.add_patch(ring1)
    lbl1 = FancyBboxPatch(
        (g1x - 0.90, y_mid + node_h / 2 + 0.25), 1.80, 0.55,
        boxstyle="round,pad=0.08", facecolor=C_G1_BG,
        edgecolor=C_G1, linewidth=1.2, zorder=3,
    )
    ax.add_patch(lbl1)
    ax.text(g1x, y_mid + node_h / 2 + 0.52,
            "Gate 1\nInformation deficiency", fontsize=8.5, ha="center",
            va="center", color=C_G1, fontweight="bold", linespacing=1.3,
            zorder=4)
    draw_arrow(ax, g1x, y_mid + node_h / 2 + 0.25,
               g1x, y_mid + node_h / 2 + 0.12,
               color=C_G1, linewidth=1.2, style="-|>", mutation_scale=8)

    # ── Gate 2: Carrier budget (spans full path, below) ──
    g2_left = positions[0][0] - node_w / 2 - 0.18
    g2_right = positions[4][0] + node_w / 2 + 0.18
    g2_band = FancyBboxPatch(
        (g2_left, y_mid - node_h / 2 - 0.55),
        g2_right - g2_left, 0.38,
        boxstyle="round,pad=0.08", facecolor=C_G2_BG,
        edgecolor=C_G2, linewidth=1.2, zorder=0,
    )
    ax.add_patch(g2_band)
    ax.text((g2_left + g2_right) / 2, y_mid - node_h / 2 - 0.36,
            "Gate 2: Carrier budget  (source \u2192 detector photon path)",
            fontsize=9, ha="center", va="center", color=C_G2,
            fontweight="bold", zorder=1)


# ── Panel D: Grammar in action ──────────────────────────────────────────────

def panel_d(ax):
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.3, 2.2)
    ax.axis("off")
    panel_bg(ax, "#FAFBFD")

    ax.text(-0.3, 1.95, "d", fontsize=18, fontweight="bold", color=C_TEXT)
    ax.text(0.15, 1.95, "Grammar in action", fontsize=11,
            fontweight="bold", color="#555555", va="center")

    # Shared parameters
    mini = ["P", "C", "S", "D"]
    nw, nh = 0.50, 0.34
    gap = 0.72

    # ── Step 1: Diagnose ──
    s1_x0 = 0.3
    ax.text(s1_x0 + 1.05, 1.60, "1. Diagnose", fontsize=10,
            fontweight="bold", ha="center", color=C_TEXT)
    for i, p in enumerate(mini):
        x = s1_x0 + i * gap
        fill, tcol = PRIM_COLOR_MAP[p]
        ec = C_G3 if p == "C" else C_EDGE
        lw = 2.8 if p == "C" else 0.8
        draw_node(ax, x, 0.95, p, fill, width=nw, height=nh,
                  fontsize=10, text_color=tcol, edgecolor=ec,
                  linewidth=lw, shadow=True)
        if i > 0:
            draw_arrow(ax, s1_x0 + (i - 1) * gap + nw / 2 + 0.04, 0.95,
                       x - nw / 2 - 0.04, 0.95)
    # Gate 3 callout
    cx = s1_x0 + gap  # C node x
    ax.annotate("Gate 3", xy=(cx, 0.95 + nh / 2 + 0.03),
                xytext=(cx, 1.42),
                fontsize=8, color=C_G3, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C_G3, lw=1.2))
    ax.text(s1_x0 + 1.05, 0.45, "dominant gate identified",
            fontsize=7, ha="center", color="#999999", style="italic")

    # ── Big arrow 1→2 ──
    draw_thick_arrow(ax, s1_x0 + 3 * gap + nw / 2 + 0.25, 0.95,
                     s1_x0 + 3 * gap + nw / 2 + 0.95, 0.95, color="#AAAAAA")

    # ── Step 2: Correct ──
    s2_x0 = 4.2
    ax.text(s2_x0 + 1.05, 1.60, "2. Correct", fontsize=10,
            fontweight="bold", ha="center", color=C_TEXT)
    for i, p in enumerate(mini):
        x = s2_x0 + i * gap
        if p == "C":
            draw_node(ax, x, 0.95, "C*", C_CORRECT, width=nw,
                      height=nh, fontsize=10, edgecolor="#3A8E58",
                      linewidth=2.2, text_color="white", shadow=True)
        else:
            fill, tcol = PRIM_COLOR_MAP[p]
            draw_node(ax, x, 0.95, p, fill, width=nw, height=nh,
                      fontsize=10, text_color=tcol, shadow=True)
        if i > 0:
            draw_arrow(ax, s2_x0 + (i - 1) * gap + nw / 2 + 0.04, 0.95,
                       x - nw / 2 - 0.04, 0.95)
    ax.text(s2_x0 + 1.05, 0.45, "offending primitive corrected",
            fontsize=7, ha="center", color="#999999", style="italic")

    # ── Big arrow 2→3 ──
    draw_thick_arrow(ax, s2_x0 + 3 * gap + nw / 2 + 0.25, 0.95,
                     s2_x0 + 3 * gap + nw / 2 + 0.95, 0.95, color="#AAAAAA")

    # ── Step 3: Recover ──
    rx = 9.5
    ax.text(rx, 1.60, "3. Recover", fontsize=10,
            fontweight="bold", ha="center", color=C_TEXT)
    draw_node(ax, rx, 0.95, "Corrected\nReconstruction",
              "#DBEEDD", width=2.0, height=0.60, fontsize=10,
              edgecolor=C_CORRECT, linewidth=1.8, text_color="#2A6B3F",
              shadow=True)
    ax.text(rx, 0.45, "no retraining required",
            fontsize=7, ha="center", color="#999999", style="italic")


# ── Main composition ────────────────────────────────────────────────────────

def main(output_path: Path):
    fig = plt.figure(figsize=(15, 13), facecolor="white")

    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.75, 0.50],
                          hspace=0.22, wspace=0.18,
                          left=0.03, right=0.97, top=0.97, bottom=0.03)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)

    # Thin separator lines between rows
    for row_frac in [0.62, 0.345]:
        fig.add_artist(plt.Line2D(
            [0.05, 0.95], [row_frac, row_frac],
            transform=fig.transFigure, color="#E0E0E0",
            linewidth=0.6, zorder=0,
        ))

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
