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
import numpy as np


# ── Colour palette ──────────────────────────────────────────────────────────

# Primitive role colours (soft, Nature-style)
C_GEN   = "#B8D4F0"   # generation  – soft blue
C_ENC   = "#C8B8E8"   # encoding    – soft purple
C_TRN   = "#F0D8A0"   # transform   – soft gold
C_DET   = "#F0B8B8"   # detection   – soft red

# Gate colours
C_G1 = "#4A90D9"  # blue   – information deficiency
C_G2 = "#E8963E"  # orange – carrier budget
C_G3 = "#D94A4A"  # red    – operator mismatch

# Neutral
C_BG    = "#FAFAFA"
C_TEXT  = "#2C2C2C"
C_ARROW = "#666666"
C_CORRECT = "#5BAA6A"  # green for correction


# ── Helper functions ────────────────────────────────────────────────────────

def _primitive_color(prim):
    """Return fill colour for a primitive symbol."""
    role_map = {
        "P": C_GEN,
        "C": C_ENC, "M": C_ENC, "R": C_ENC,
        "\u039B": C_ENC, "\u03A0": C_ENC,
        "F": C_TRN, "\u03A3": C_TRN,
        "S": C_DET, "W": C_DET, "D": C_DET,
    }
    return role_map.get(prim, "#DDDDDD")


def draw_node(ax, x, y, label, color, width=0.7, height=0.38,
              fontsize=10, edgecolor="#888888", linewidth=0.8,
              text_color=C_TEXT, bold=False):
    """Draw a rounded-rectangle node with centred label."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=edgecolor, linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=3)
    return box


def draw_arrow(ax, x0, y0, x1, y1, color=C_ARROW, linewidth=1.0,
               style="-|>", mutation_scale=10):
    """Draw a simple arrow between two points."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color,
        linewidth=linewidth, mutation_scale=mutation_scale,
        zorder=1,
    )
    ax.add_patch(arrow)


def draw_dag_chain(ax, x_start, y_top, primitives, colors, spacing=0.55,
                   node_w=0.7, node_h=0.38, fontsize=10, arrow_color=C_ARROW):
    """Draw a vertical chain of primitive nodes with arrows."""
    positions = []
    for i, (prim, col) in enumerate(zip(primitives, colors)):
        y = y_top - i * spacing
        draw_node(ax, x_start, y, prim, col, width=node_w, height=node_h,
                  fontsize=fontsize)
        positions.append((x_start, y))
        if i > 0:
            draw_arrow(ax, x_start, positions[i - 1][1] - node_h / 2 - 0.02,
                       x_start, y + node_h / 2 + 0.02,
                       color=arrow_color)
    return positions


# ── Panel A: The 11 primitives grouped by role ──────────────────────────────

def panel_a(ax):
    """Draw the 11 primitives grouped by physical role."""
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.3, 4.5)
    ax.axis("off")
    ax.text(0.0, 4.2, "a", fontsize=16, fontweight="bold",
            transform=ax.transData)

    # Group definitions: (label, primitives, colour, type_sig)
    groups = [
        ("Generation",  [("P", "source \u2192 field")],
         C_GEN),
        ("Encoding",    [("C", "aperture"), ("M", "modulation"),
                         ("R", "rotation"), ("\u039B", "dispersion"),
                         ("\u03A0", "phase")],
         C_ENC),
        ("Transform",   [("F", "Fourier"), ("\u03A3", "integration")],
         C_TRN),
        ("Detection",   [("S", "sampling"), ("W", "weighting"),
                         ("D", "field \u2192 meas.")],
         C_DET),
    ]

    y_pos = 3.6
    for group_label, prims, color in groups:
        # Group header
        ax.text(0.1, y_pos + 0.05, group_label, fontsize=9, fontweight="bold",
                color="#555555", va="center")
        # Primitive boxes in a row
        x = 1.6
        for sym, sig in prims:
            draw_node(ax, x, y_pos, sym, color, width=0.52, height=0.34,
                      fontsize=11, bold=True)
            ax.text(x, y_pos - 0.28, sig, fontsize=6.5, ha="center",
                    va="top", color="#777777", style="italic")
            x += 0.85
        y_pos -= 1.0


# ── Panel B: Three modalities as DAGs ───────────────────────────────────────

def panel_b(ax):
    """Draw CASSI, MRI, CT as typed DAGs over the 11 primitives."""
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")
    ax.text(-0.3, 4.2, "b", fontsize=16, fontweight="bold")

    # CASSI chain: P → C → Λ → S → D
    cassi_prims = ["P", "C", "\u039B", "S", "D"]
    cassi_cols  = [C_GEN, C_ENC, C_ENC, C_DET, C_DET]
    ax.text(0.7, 4.0, "CASSI", fontsize=10, fontweight="bold",
            ha="center", color="#4A6FA5")
    draw_dag_chain(ax, 0.7, 3.5, cassi_prims, cassi_cols,
                   spacing=0.52, node_w=0.58, node_h=0.32, fontsize=9)

    # MRI chain: P → W → F → S → D
    mri_prims = ["P", "W", "F", "S", "D"]
    mri_cols  = [C_GEN, C_DET, C_TRN, C_DET, C_DET]
    ax.text(2.5, 4.0, "MRI", fontsize=10, fontweight="bold",
            ha="center", color="#8B5E3C")
    draw_dag_chain(ax, 2.5, 3.5, mri_prims, mri_cols,
                   spacing=0.52, node_w=0.58, node_h=0.32, fontsize=9)

    # CT chain: P → R → Σ → S → D
    ct_prims = ["P", "R", "\u03A3", "S", "D"]
    ct_cols  = [C_GEN, C_ENC, C_TRN, C_DET, C_DET]
    ax.text(4.3, 4.0, "CT", fontsize=10, fontweight="bold",
            ha="center", color="#B8463E")
    draw_dag_chain(ax, 4.3, 3.5, ct_prims, ct_cols,
                   spacing=0.52, node_w=0.58, node_h=0.32, fontsize=9)

    # Annotation
    ax.text(2.5, -0.25,
            "Same alphabet, different wiring",
            fontsize=8, ha="center", color="#888888", style="italic")


# ── Panel C: 3 gates overlaid on a DAG ─────────────────────────────────────

def panel_c(ax):
    """Draw a CASSI DAG (horizontal) with the 3 gates highlighted."""
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.8, 2.8)
    ax.axis("off")
    ax.text(-0.3, 2.5, "c", fontsize=16, fontweight="bold")

    # Horizontal CASSI DAG
    prims = ["P", "C", "\u039B", "S", "D"]
    cols  = [C_GEN, C_ENC, C_ENC, C_DET, C_DET]
    node_w, node_h = 0.7, 0.42
    gap = 1.6
    x_start = 1.5
    y_mid = 1.2
    positions = []
    for i, (p, c) in enumerate(zip(prims, cols)):
        x = x_start + i * gap
        draw_node(ax, x, y_mid, p, c, width=node_w, height=node_h,
                  fontsize=12, bold=True)
        positions.append((x, y_mid))
        if i > 0:
            draw_arrow(ax, positions[i - 1][0] + node_w / 2 + 0.04, y_mid,
                       x - node_w / 2 - 0.04, y_mid)

    # Gate 2: Carrier budget – spans entire P→D path (below)
    g2_left = positions[0][0] - node_w / 2 - 0.15
    g2_right = positions[4][0] + node_w / 2 + 0.15
    rect2 = mpatches.FancyBboxPatch(
        (g2_left, y_mid - node_h / 2 - 0.45), g2_right - g2_left, 0.35,
        boxstyle="round,pad=0.06", facecolor=C_G2, alpha=0.15,
        edgecolor=C_G2, linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(rect2)
    ax.text((g2_left + g2_right) / 2, y_mid - node_h / 2 - 0.28,
            "Gate 2: Carrier budget (source \u2192 detector photon path)",
            fontsize=8.5, ha="center", va="center", color=C_G2,
            fontweight="bold")

    # Gate 1: Information deficiency – spans S node (above)
    g1_x = positions[3][0]
    rect1 = mpatches.FancyBboxPatch(
        (g1_x - 0.55, y_mid + node_h / 2 + 0.10), 1.1, 0.45,
        boxstyle="round,pad=0.06", facecolor=C_G1, alpha=0.15,
        edgecolor=C_G1, linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(rect1)
    ax.text(g1_x, y_mid + node_h / 2 + 0.32,
            "Gate 1: Information\ndeficiency",
            fontsize=8, ha="center", va="center", color=C_G1,
            fontweight="bold", linespacing=1.2)
    # Connector line from gate label to node
    draw_arrow(ax, g1_x, y_mid + node_h / 2 + 0.10,
               g1_x, y_mid + node_h / 2 + 0.02,
               color=C_G1, linewidth=1.0, style="-|>", mutation_scale=8)

    # Gate 3: Mismatch – highlights C node specifically (above)
    g3_x = positions[1][0]
    rect3 = mpatches.FancyBboxPatch(
        (g3_x - 0.55, y_mid + node_h / 2 + 0.10), 1.1, 0.45,
        boxstyle="round,pad=0.06", facecolor=C_G3, alpha=0.15,
        edgecolor=C_G3, linewidth=1.5, linestyle="--", zorder=0)
    ax.add_patch(rect3)
    ax.text(g3_x, y_mid + node_h / 2 + 0.32,
            "Gate 3: Operator\nmismatch",
            fontsize=8, ha="center", va="center", color=C_G3,
            fontweight="bold", linespacing=1.2)
    draw_arrow(ax, g3_x, y_mid + node_h / 2 + 0.10,
               g3_x, y_mid + node_h / 2 + 0.02,
               color=C_G3, linewidth=1.0, style="-|>", mutation_scale=8)

    # Label: CASSI example
    ax.text(x_start - 1.0, y_mid, "CASSI\nexample",
            fontsize=8.5, ha="center", va="center", color="#888888",
            style="italic", linespacing=1.3)

    # Caption
    ax.text(5.5, -0.55,
            "Gates diagnose where on the DAG a reconstruction failure originates",
            fontsize=8.5, ha="center", color="#888888", style="italic")


# ── Panel D: Grammar in action ──────────────────────────────────────────────

def panel_d(ax):
    """Show diagnose → correct → recover in a compact strip."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 2.0)
    ax.axis("off")
    ax.text(-0.3, 1.75, "d", fontsize=16, fontweight="bold")

    # Step 1: Diagnose – mini DAG with Gate 3 highlighted
    ax.text(1.2, 1.55, "Diagnose", fontsize=9, fontweight="bold",
            ha="center", color=C_TEXT)
    mini_prims = ["P", "C", "S", "D"]
    mini_cols  = [C_GEN, C_ENC, C_DET, C_DET]
    # Horizontal mini-chain
    x_start = 0.0
    node_w, node_h = 0.45, 0.30
    gap = 0.65
    for i, (p, c) in enumerate(zip(mini_prims, mini_cols)):
        x = x_start + i * gap
        ec = C_G3 if p == "C" else "#888888"
        lw = 2.5 if p == "C" else 0.8
        draw_node(ax, x, 0.9, p, c, width=node_w, height=node_h,
                  fontsize=8, edgecolor=ec, linewidth=lw)
        if i > 0:
            draw_arrow(ax, x_start + (i - 1) * gap + node_w / 2 + 0.03, 0.9,
                       x - node_w / 2 - 0.03, 0.9)
    # Gate 3 label
    ax.annotate("Gate 3", xy=(x_start + gap, 0.9 + node_h / 2 + 0.02),
                xytext=(x_start + gap, 1.35),
                fontsize=7, color=C_G3, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C_G3, lw=1.0))
    ax.text(1.2, 0.35, "dominant gate\nidentified",
            fontsize=6.5, ha="center", color="#888888", style="italic",
            linespacing=1.3)

    # Arrow between steps
    draw_arrow(ax, 2.8, 0.9, 3.5, 0.9, color=C_ARROW, linewidth=1.2,
               mutation_scale=12)

    # Step 2: Correct – C node gets corrected
    ax.text(5.0, 1.55, "Correct", fontsize=9, fontweight="bold",
            ha="center", color=C_TEXT)
    x_start2 = 3.8
    for i, (p, c) in enumerate(zip(mini_prims, mini_cols)):
        x = x_start2 + i * gap
        if p == "C":
            # Corrected node – green border
            draw_node(ax, x, 0.9, "C\u2713", C_CORRECT, width=node_w,
                      height=node_h, fontsize=8, edgecolor="#3D8B4F",
                      linewidth=2.0, text_color="white", bold=True)
        else:
            draw_node(ax, x, 0.9, p, c, width=node_w, height=node_h,
                      fontsize=8)
        if i > 0:
            draw_arrow(ax, x_start2 + (i - 1) * gap + node_w / 2 + 0.03, 0.9,
                       x - node_w / 2 - 0.03, 0.9)
    ax.text(5.0, 0.35, "offending primitive\ncorrected",
            fontsize=6.5, ha="center", color="#888888", style="italic",
            linespacing=1.3)

    # Arrow between steps
    draw_arrow(ax, 6.5, 0.9, 7.2, 0.9, color=C_ARROW, linewidth=1.2,
               mutation_scale=12)

    # Step 3: Recover – result box
    draw_node(ax, 8.5, 0.9, "Corrected\nReconstruction", "#D5E8D4",
              width=2.0, height=0.55, fontsize=8.5,
              edgecolor=C_CORRECT, linewidth=1.5, bold=True)
    ax.text(8.5, 0.35, "no retraining\nrequired",
            fontsize=6.5, ha="center", color="#888888", style="italic",
            linespacing=1.3)


# ── Main composition ────────────────────────────────────────────────────────

def main(output_path: Path):
    fig = plt.figure(figsize=(14, 12), facecolor="white")

    # Layout: 3 rows — top: a + b, middle: c (full width), bottom: d (full width)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.9, 0.55],
                          hspace=0.35, wspace=0.25,
                          left=0.04, right=0.96, top=0.96, bottom=0.04)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])   # full width for gate overlay
    ax_d = fig.add_subplot(gs[2, :])   # full width for correction strip

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {output_path}")

    # Also save PNG for preview
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
