#!/usr/bin/env python3
"""
Generate Figure 1: The Universal Grammar of Computational Imaging.

Layout (two-branch from OperatorGraph):
  a (Existing / Design goal) → b (Compose: OperatorGraph)
      ├─ Top:    c (Diagnose) → d (Correct) → e (Recover)   [existing]
      └─ Bottom: f (Design: gate-guided new modality)        [new]

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
C_STAGE1B = "#E0F2F1"  # Design goal (teal tint)
C_STAGE2 = "#EDE8F5"
C_STAGE3 = "#FFF3E0"
C_STAGE4 = "#FCE4EC"
C_STAGE5 = "#E8F5E9"
C_STAGE6 = "#E0F2F1"   # Design panel (teal-ish)

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
    # figsize (14, 9): scale factor ≈ 7.2/14 = 0.514
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="white")
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 10.0)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Vertical layout ─────────────────────────────────────────────────
    y_top = 7.2       # top-row centre (existing instruments)
    y_bot = 2.6       # bottom-row centre (design new)
    h_top = 4.4       # top-row panel height
    h_bot = 3.8       # bottom-row panel height

    # Left panels span full height
    left_cy = 5.0
    left_h = 8.4      # spans from 0.8 to 9.2

    # ── Horizontal layout (wider gaps for arrows) ─────────────────────
    margin = 0.20

    # Left column: a + b
    w_a = 2.0
    w_b = 2.8
    gap_ab = 0.55     # wider gap for a→b arrow

    a_left = margin
    a_cx = a_left + w_a / 2
    b_left = a_left + w_a + gap_ab
    b_cx = b_left + w_b / 2
    b_right = b_left + w_b

    # Right column starts after b
    gap_br = 0.70     # wider gap for b→c / b→f arrows
    right_start = b_right + gap_br

    # Top row: c, d, e with wider gaps
    w_c = 3.2
    w_d = 3.2
    w_e = 2.0
    gap_top = 0.50    # wider gaps between c,d,e
    c_left = right_start
    c_cx = c_left + w_c / 2
    d_left = c_left + w_c + gap_top
    d_cx = d_left + w_d / 2
    e_left = d_left + w_d + gap_top
    e_cx = e_left + w_e / 2

    # Bottom row: f spans most of right side, with output box at end
    w_out = 1.8       # output box width
    gap_f_out = 0.40
    w_f = e_left + w_e - right_start - w_out - gap_f_out
    f_cx = right_start + w_f / 2
    out_cx = right_start + w_f + gap_f_out + w_out / 2

    # ════════════════════════════════════════════════════════════════════
    # a – Input: Existing System / Design Goal (spans full height)
    # ════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, a_cx, left_cy, w_a, left_h, C_STAGE1, "#B0C4DE")

    ax.text(a_left + 0.10, left_cy + left_h / 2 - 0.18, "a",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)

    # --- Top zone: Existing systems ---
    ax.text(a_cx, left_cy + left_h / 2 - 0.55, "Existing\nSystem",
            fontsize=11, ha="center", va="top", color="#3B6FA0",
            fontweight="bold", linespacing=1.2, zorder=1)

    modalities = ["CASSI", "MRI", "CT", "Cryo-EM", "OCT"]
    mod_y_start = left_cy + 1.20
    for i, name in enumerate(modalities):
        ax.text(a_cx, mod_y_start - i * 0.45, name,
                fontsize=9, ha="center", va="center", color="#666666",
                style="italic", zorder=1)

    # --- Divider line ---
    div_y = left_cy - 1.10
    ax.plot([a_left + 0.25, a_left + w_a - 0.25], [div_y, div_y],
            color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(a_cx, div_y + 0.12, "or", fontsize=8, ha="center",
            va="bottom", color="#999999", style="italic", zorder=1)

    # --- Bottom zone: Design goal ---
    ax.text(a_cx, div_y - 0.30, "Design\nGoal",
            fontsize=11, ha="center", va="top", color="#1A6A6A",
            fontweight="bold", linespacing=1.2, zorder=1)
    ax.text(a_cx, div_y - 1.00, "new modality\nspecification",
            fontsize=8, ha="center", va="top", color="#777777",
            style="italic", linespacing=1.2, zorder=1)

    # ════════════════════════════════════════════════════════════════════
    # b – Compose: OperatorGraph (spans full height)
    # ════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, b_cx, left_cy, w_b, left_h, C_STAGE2, "#C0B0D8")

    ax.text(b_left + 0.08, left_cy + left_h / 2 - 0.18, "b",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(b_cx, left_cy + left_h / 2 - 0.55,
            "Compose:\nOperatorGraph",
            fontsize=11, ha="center", va="top", color=C_ENC_T,
            fontweight="bold", linespacing=1.15, zorder=1)
    ax.text(b_cx, left_cy + left_h / 2 - 1.05,
            "(11 primitives; Fig. 2a)",
            fontsize=8, ha="center", va="top", color="#888888",
            style="italic", zorder=1)

    # --- Top DAG (existing): P → C → W → S → D ---
    nw, nh = 0.58, 0.32
    dag_cx = b_cx - 0.10

    dag_prims = ["P", "C", "W", "S", "D"]
    dag_y_top = left_cy + 0.80
    dag_sp = 0.50
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

    # --- Divider in b ---
    b_div_y = left_cy - 1.20
    ax.plot([b_left + 0.25, b_left + w_b - 0.25], [b_div_y, b_div_y],
            color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=1)

    # --- Bottom DAG (new design): M → F → S → D ---
    new_prims = ["M", "F", "S", "D"]
    new_y_top = b_div_y - 0.40
    new_sp = 0.48
    for i, p in enumerate(new_prims):
        y = new_y_top - i * new_sp
        fill, tcol = PRIM_COLOR_MAP[p]
        draw_node(ax, dag_cx, y, p, fill, width=nw, height=nh,
                  fontsize=12, text_color=tcol, edgecolor=tcol,
                  linewidth=0.7, shadow=True)
        if i > 0:
            draw_small_arrow(ax, dag_cx,
                             new_y_top - (i - 1) * new_sp - nh / 2 - 0.02,
                             dag_cx, y + nh / 2 + 0.02, color=C_ARROW)

    ax.text(dag_cx + 0.50, new_y_top, "e.g. new\nMRI variant", fontsize=8,
            ha="left", va="center", color="#1A6A6A", style="italic",
            linespacing=1.1, zorder=1)

    # Arrow: a → b
    draw_big_arrow(ax, a_left + w_a + 0.08, left_cy,
                   b_left - 0.08, left_cy, color="#9999BB", lw=2.5)

    # ════════════════════════════════════════════════════════════════════
    # Branch labels + arrows from b
    # ════════════════════════════════════════════════════════════════════
    # Top branch arrow: b → c (existing instruments)
    draw_big_arrow(ax, b_right + 0.08, y_top,
                   c_left - 0.08, y_top, color="#C0B080", lw=2.5)
    ax.text((b_right + c_left) / 2, y_top + 0.28,
            "Existing", fontsize=9, ha="center", va="bottom",
            color="#888888", style="italic", zorder=6)

    # Bottom branch arrow: b → f (new instruments)
    draw_big_arrow(ax, b_right + 0.08, y_bot,
                   right_start - 0.08, y_bot, color="#66A0A0", lw=2.5)
    ax.text((b_right + right_start) / 2, y_bot + 0.28,
            "New", fontsize=9, ha="center", va="bottom",
            color="#888888", style="italic", zorder=6)

    # ════════════════════════════════════════════════════════════════════
    # c – Diagnose: Triad Decomposition (top row)
    # ════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, c_cx, y_top, w_c, h_top, C_STAGE3, "#E0C8A0")

    ax.text(c_left + 0.08, y_top + h_top / 2 - 0.18, "c",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(c_cx, y_top + h_top / 2 - 0.50,
            "Diagnose:\nTriad Decomposition",
            fontsize=11, ha="center", va="top", color="#B07020",
            fontweight="bold", linespacing=1.15, zorder=1)

    gate_data = [
        ("Gate 1", "Information\ndeficiency", C_G1_BG, C_G1),
        ("Gate 2", "Carrier\nbudget", C_G2_BG, C_G2),
        ("Gate 3", "Operator\nmismatch", C_G3_BG, C_G3),
    ]
    gate_sp = 1.00
    gy = y_top - 0.30
    gw, gh = 0.90, 1.50
    for k, (gname, gdesc, gbg, gcol) in enumerate(gate_data):
        gx = c_cx + (k - 1) * gate_sp
        gbox = FancyBboxPatch(
            (gx - gw / 2, gy - gh / 2), gw, gh,
            boxstyle="round,pad=0.05", facecolor=gbg,
            edgecolor=gcol, linewidth=1.3, zorder=1)
        ax.add_patch(gbox)
        ax.text(gx, gy + 0.30, gname, fontsize=10, ha="center",
                va="center", color=gcol, fontweight="bold", zorder=2)
        ax.text(gx, gy - 0.20, gdesc, fontsize=8, ha="center",
                va="center", color=gcol, linespacing=1.15, zorder=2)

    # Arrow: c → d
    draw_big_arrow(ax, c_left + w_c + 0.08, y_top,
                   d_left - 0.08, y_top, color="#CC9999", lw=2.5)

    # ════════════════════════════════════════════════════════════════════
    # d – Correct: targeted intervention (top row)
    # ════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, d_cx, y_top, w_d, h_top, C_STAGE4, "#E0B0B0")

    ax.text(d_left + 0.08, y_top + h_top / 2 - 0.18, "d",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(d_cx, y_top + h_top / 2 - 0.50,
            "Correct:\ntargeted intervention",
            fontsize=11, ha="center", va="top", color="#AA4040",
            fontweight="bold", linespacing=1.15, zorder=1)

    corr_data = [
        ("Gate 1\ndominant", "Redesign\nsampling", C_G1_BG, C_G1),
        ("Gate 2\ndominant", "Improve\ncarrier", C_G2_BG, C_G2),
        ("Gate 3\ndominant", "Calibrate\noperator", C_G3_BG, C_G3),
    ]
    corr_sp = 1.00
    cy = y_top - 0.30
    cw, ch = 0.90, 1.50
    for k, (clabel, cdesc, cbg, ccol) in enumerate(corr_data):
        cx = d_cx + (k - 1) * corr_sp
        cbox = FancyBboxPatch(
            (cx - cw / 2, cy - ch / 2), cw, ch,
            boxstyle="round,pad=0.05", facecolor=cbg,
            edgecolor=ccol, linewidth=1.0, zorder=1)
        ax.add_patch(cbox)
        ax.text(cx, cy + 0.30, clabel, fontsize=9, ha="center",
                va="center", color=ccol, fontweight="bold",
                linespacing=1.1, zorder=2)
        ax.text(cx, cy - 0.25, cdesc, fontsize=8, ha="center",
                va="center", color="#555555", linespacing=1.1, zorder=2)

    # Arrow: d → e
    draw_big_arrow(ax, d_left + w_d + 0.08, y_top,
                   e_left - 0.08, y_top, color="#88BB88", lw=2.5)

    # ════════════════════════════════════════════════════════════════════
    # e – Recover (top row)
    # ════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, e_cx, y_top, w_e, h_top, C_STAGE5, "#A0D0A0")

    ax.text(e_left + 0.08, y_top + h_top / 2 - 0.18, "e",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(e_cx, y_top + h_top / 2 - 0.50, "Recover",
            fontsize=12, ha="center", va="top", color="#2E7D32",
            fontweight="bold", zorder=1)

    ax.text(e_cx, y_top + 0.20, "Corrected\nReconstruction",
            fontsize=10, ha="center", va="center", color="#2E7D32",
            fontweight="bold", linespacing=1.3, zorder=1)
    ax.text(e_cx, y_top - 0.55, "+0.8 to\n+13.9 dB",
            fontsize=10, ha="center", color="#555555",
            fontweight="bold", linespacing=1.2, zorder=1)
    ax.text(e_cx, y_top - 1.15, "no retraining",
            fontsize=8, ha="center", color="#888888",
            style="italic", zorder=1)

    # ════════════════════════════════════════════════════════════════════
    # f – Design: gate-guided new modality (bottom row)
    # ════════════════════════════════════════════════════════════════════
    draw_stage_box(ax, f_cx, y_bot, w_f, h_bot, C_STAGE6, "#80B0B0")

    ax.text(right_start + 0.10, y_bot + h_bot / 2 - 0.18, "f",
            fontsize=16, fontweight="bold", color=C_TEXT, va="top", zorder=1)
    ax.text(f_cx, y_bot + h_bot / 2 - 0.45,
            "Design: gate-guided new modality",
            fontsize=11, ha="center", va="top", color="#1A6A6A",
            fontweight="bold", zorder=1)

    # Three design boxes (gate-guided)
    design_data = [
        ("Gate 1", "Sampling\ngeometry",
         "How many\nmeasurements?", C_G1_BG, C_G1),
        ("Gate 2", "Carrier &\nsource",
         "Which carrier?\nSNR budget?", C_G2_BG, C_G2),
        ("Gate 3", "Calibration\nspec",
         "What accuracy\nneeded?", C_G3_BG, C_G3),
    ]
    design_box_w = 2.2
    design_box_h = 2.2
    n_boxes = 3
    total_box_w = n_boxes * design_box_w
    design_gap = (w_f - total_box_w) / (n_boxes + 1)
    dy = y_bot - 0.15

    for k, (dgate, dtitle, dquestion, dbg, dcol) in enumerate(design_data):
        dx = right_start + design_gap + design_box_w / 2 + \
             k * (design_box_w + design_gap)
        dbox = FancyBboxPatch(
            (dx - design_box_w / 2, dy - design_box_h / 2),
            design_box_w, design_box_h,
            boxstyle="round,pad=0.05", facecolor=dbg,
            edgecolor=dcol, linewidth=1.2, zorder=1)
        ax.add_patch(dbox)
        ax.text(dx, dy + 0.60, dgate, fontsize=10, ha="center",
                va="center", color=dcol, fontweight="bold", zorder=2)
        ax.text(dx, dy + 0.10, dtitle, fontsize=9, ha="center",
                va="center", color="#333333", fontweight="bold",
                linespacing=1.15, zorder=2)
        ax.text(dx, dy - 0.50, dquestion, fontsize=8, ha="center",
                va="center", color="#777777", style="italic",
                linespacing=1.15, zorder=2)

    # Arrow: f → output box
    draw_big_arrow(ax, right_start + w_f + 0.08, y_bot,
                   out_cx - w_out / 2 - 0.08, y_bot,
                   color="#55AA99", lw=2.5)

    # Output box: Optimized System
    draw_stage_box(ax, out_cx, y_bot, w_out, h_bot, C_STAGE5, "#80B080")
    ax.text(out_cx, y_bot + 0.30, "Optimized\nNew System",
            fontsize=10, ha="center", va="center", color="#2E7D32",
            fontweight="bold", linespacing=1.3, zorder=1)
    ax.text(out_cx, y_bot - 0.50, "ready to\nbuild",
            fontsize=8, ha="center", va="center", color="#888888",
            style="italic", linespacing=1.2, zorder=1)

    # ════════════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════════════
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
