#!/usr/bin/env python3
"""Generate all paper figures at LNCS print dimensions.

Figures are sized so that matplotlib font sizes match the final PDF exactly —
no shrinking occurs when LaTeX includes the figure at its target width.

    Fig 1: Scenario comparison (3 panels)   — \textwidth   ≈ 7.0"
    Fig 2: Visual comparison (3×6 grid)     — \textwidth   ≈ 7.0"
    Fig 3: Gap comparison (3×2 panels)      — \textwidth   ≈ 7.0"
    Fig 4: Recovery ratio scatter            — 0.75\textwidth ≈ 5.0"
    Fig 5: Residual gap bar chart            — 0.85\textwidth ≈ 5.8"

Usage:
    python papers/inversenet/scripts/generate_paper_figures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(FIGURES_DIR / "cassi").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global matplotlib defaults for print-size figures
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,      # TrueType fonts in PDF
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_json(name: str):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


# ===== CASSI helpers =====
CASSI_METHODS = ["gap_tv", "hdnet", "mst_s", "mst_l"]
CASSI_LABELS = {"gap_tv": "GAP-TV", "hdnet": "HDNet",
                "mst_s": "MST-S", "mst_l": "MST-L"}
CASSI_COLORS = {"gap_tv": "#1f77b4", "hdnet": "#ff7f0e",
                "mst_s": "#2ca02c", "mst_l": "#d62728"}

# ===== CACTI helpers =====
CACTI_METHODS = ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficientsci"]
CACTI_LABELS = {"gap_tv": "GAP-TV", "pnp_ffdnet": "PnP-FFDNet",
                "elp_unfolding": "ELP-Unf.", "efficientsci": "Eff.SCI"}
CACTI_COLORS = {"gap_tv": "#1f77b4", "pnp_ffdnet": "#2ca02c",
                "elp_unfolding": "#ff7f0e", "efficientsci": "#d62728"}

# ===== SPC helpers =====
SPC_METHODS = ["fista_tv", "ista_net", "hatnet"]
SPC_LABELS = {"fista_tv": "FISTA-TV", "ista_net": "ISTA-Net",
              "hatnet": "HATNet"}
SPC_COLORS = {"fista_tv": "#1f77b4", "ista_net": "#2ca02c",
              "hatnet": "#ff7f0e"}

SCENARIOS = ["scenario_i", "scenario_ii", "scenario_iii"]
SCENARIO_TICK = ["I (Ideal)", "II (Mismatch)", "III (Oracle)"]
MOD_COLORS = {"CASSI": "#E91E63", "CACTI": "#2196F3", "SPC": "#FF9800"}


# ###########################################################################
#  FIG 1 — Scenario comparison  (3 subplots, one per modality)
# ###########################################################################
def fig1_scenario_comparison():
    """Single combined figure replacing three separate PNGs."""
    cassi = load_json("cassi_summary.json")
    cacti = load_json("cacti_summary.json")
    spc = load_json("spc_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=False)

    def _bar_panel(ax, methods, labels, colors, get_psnr_fn, title, ylim):
        x = np.arange(len(SCENARIOS))
        n = len(methods)
        w = 0.8 / n
        for i, m in enumerate(methods):
            vals = [get_psnr_fn(m, s) for s in SCENARIOS]
            off = (i - (n - 1) / 2) * w
            ax.bar(x + off, vals, w, color=colors[m], alpha=0.85,
                   edgecolor="black", linewidth=0.3,
                   label=labels[m])
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_TICK, fontsize=6)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4, linestyle="--")
        ax.legend(fontsize=5.5, loc="upper right", handlelength=1,
                  handletextpad=0.3, borderpad=0.3, labelspacing=0.25)

    # CASSI
    _bar_panel(axes[0], CASSI_METHODS, CASSI_LABELS, CASSI_COLORS,
               lambda m, s: cassi[s][m]["psnr_mean"],
               "CASSI", [0, 42])
    axes[0].set_ylabel("PSNR (dB)", fontsize=8, fontweight="bold")

    # CACTI
    _bar_panel(axes[1], CACTI_METHODS, CACTI_LABELS, CACTI_COLORS,
               lambda m, s: cacti["overall"][s][m]["psnr_mean"],
               "CACTI", [0, 42])

    # SPC
    _bar_panel(axes[2], SPC_METHODS, SPC_LABELS, SPC_COLORS,
               lambda m, s: spc["methods"][f"{m}_{s}"]["psnr_mean"],
               "SPC", [0, 42])

    fig.tight_layout(w_pad=0.8)
    out = FIGURES_DIR / "fig1_scenario_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Fig 1 saved: {out}")


# ###########################################################################
#  FIG 2 — Visual reconstruction comparison  (3 rows × 6 cols)
# ###########################################################################
def _psnr(x, y, max_val=1.0):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def fig2_visual_comparison():
    """3×6 grid at print width."""
    recon_dir = RESULTS_DIR

    def _load_cassi():
        d = np.load(recon_dir / "cassi_reconstructions" / "scene01.npz",
                     allow_pickle=True)
        b = 14
        return (d["gt"][:, :, b], d["scenario_i_mst_l"][:, :, b],
                d["scenario_ii_mst_l"][:, :, b],
                d["scenario_iii_mst_l"][:, :, b], 1.0)

    def _load_cacti():
        d = np.load(recon_dir / "cacti_reconstructions" / "kobe.npz",
                     allow_pickle=True)
        f = 4
        return (d["g0_gt"][:, :, f], d["g0_scenario_i_elp_unfolding"][:, :, f],
                d["g0_scenario_ii_elp_unfolding"][:, :, f],
                d["g0_scenario_iii_elp_unfolding"][:, :, f], 1.0)

    def _load_spc():
        d = np.load(recon_dir / "spc_reconstructions" / "cameraman.npz",
                     allow_pickle=True)
        return (d["gt"], d["scenario_i_hatnet"],
                d["scenario_ii_hatnet"], d["scenario_iii_hatnet"], 255.0)

    rows_data = []
    for loader in [_load_cassi, _load_cacti, _load_spc]:
        gt, s1, s2, s3, mv = loader()
        err2 = np.abs(gt.astype(np.float64) - s2.astype(np.float64))
        err3 = np.abs(gt.astype(np.float64) - s3.astype(np.float64))
        rows_data.append(dict(
            gt=gt, s1=s1, s2=s2, s3=s3, err2=err2, err3=err3,
            p1=_psnr(s1, gt, mv), p2=_psnr(s2, gt, mv),
            p3=_psnr(s3, gt, mv), mv=mv))

    fig = plt.figure(figsize=(7.0, 3.8))
    gs = GridSpec(3, 6, figure=fig, wspace=0.04, hspace=0.12)

    col_titles = ["Ground Truth", "Scenario I", "Scenario II",
                  "Scenario III", "Error II", "Error III"]
    row_labels = ["CASSI\n(MST-L)", "CACTI\n(ELP-Unf.)", "SPC\n(HATNet)"]

    for ri, rd in enumerate(rows_data):
        vmin, vmax = 0, rd["mv"]
        err_max = max(np.percentile(rd["err2"], 99),
                      np.percentile(rd["err3"], 99))
        imgs = [rd["gt"], rd["s1"], rd["s2"], rd["s3"],
                rd["err2"], rd["err3"]]
        pvals = [None, rd["p1"], rd["p2"], rd["p3"], None, None]

        for ci, (img, pv) in enumerate(zip(imgs, pvals)):
            ax = fig.add_subplot(gs[ri, ci])
            if ci < 4:
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
            else:
                ax.imshow(img, cmap="jet", vmin=0, vmax=err_max,
                          interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=7.5, fontweight="bold",
                             pad=2)
            if pv is not None:
                ax.text(0.02, 0.02, f"{pv:.1f}", transform=ax.transAxes,
                        fontsize=6.5, color="yellow", fontweight="bold",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black",
                                  alpha=0.7, linewidth=0))
            if ci == 0:
                ax.set_ylabel(row_labels[ri], fontsize=7, fontweight="bold",
                              labelpad=2)

    out = FIGURES_DIR / "visual_comparison.pdf"
    fig.savefig(out, dpi=300, pad_inches=0.03)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Fig 2 saved: {out}")


# ###########################################################################
#  FIG 3 — Combined gap comparison  (3 rows × 2 cols, all modalities)
# ###########################################################################
def fig3_gap_comparison():
    cassi = load_json("cassi_summary.json")
    cacti = load_json("cacti_summary.json")
    spc = load_json("spc_summary.json")

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.0))

    def _gap_row(ax_deg, ax_rec, methods, labels, colors,
                 get_deg, get_rec, title, deg_ylim, rec_ylim):
        x = np.arange(len(methods))
        xlabels = [labels[m] for m in methods]
        w = 0.6

        # Degradation panel
        deg_vals = [get_deg(m) for m in methods]
        bars1 = ax_deg.bar(x, deg_vals,
                           color=[colors[m] for m in methods],
                           alpha=0.85, edgecolor="black", linewidth=0.4,
                           width=w)
        for bar, v in zip(bars1, deg_vals):
            ax_deg.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + deg_ylim[1] * 0.02,
                        f"{v:.1f}", ha="center", fontsize=6, fontweight="bold")
        ax_deg.set_xticks(x)
        ax_deg.set_xticklabels(xlabels, fontsize=6.5)
        ax_deg.set_ylim(deg_ylim)
        ax_deg.grid(axis="y", alpha=0.25, linewidth=0.4, linestyle="--")
        ax_deg.set_title(f"{title} — Degradation (I→II)", fontsize=7.5,
                         fontweight="bold")

        # Recovery panel
        rec_vals = [get_rec(m) for m in methods]
        bars2 = ax_rec.bar(x, rec_vals,
                           color=[colors[m] for m in methods],
                           alpha=0.85, edgecolor="black", linewidth=0.4,
                           width=w)
        for bar, v in zip(bars2, rec_vals):
            ax_rec.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + rec_ylim[1] * 0.02,
                        f"{v:.1f}", ha="center", fontsize=6, fontweight="bold")
        ax_rec.set_xticks(x)
        ax_rec.set_xticklabels(xlabels, fontsize=6.5)
        ax_rec.set_ylim(rec_ylim)
        ax_rec.grid(axis="y", alpha=0.25, linewidth=0.4, linestyle="--")
        ax_rec.set_title(f"{title} — Recovery (II→III)", fontsize=7.5,
                         fontweight="bold")

    # Row 0: CASSI
    _gap_row(axes[0, 0], axes[0, 1],
             CASSI_METHODS, CASSI_LABELS, CASSI_COLORS,
             lambda m: cassi["gaps"][m]["gap_i_ii_mean"],
             lambda m: cassi["gaps"][m]["gap_ii_iii_mean"],
             "CASSI", [0, 18], [0, 10])

    # Row 1: CACTI
    _gap_row(axes[1, 0], axes[1, 1],
             CACTI_METHODS, CACTI_LABELS, CACTI_COLORS,
             lambda m: cacti["gaps"][m]["gap_i_ii_mean"],
             lambda m: cacti["gaps"][m]["gap_ii_iii_mean"],
             "CACTI", [0, 24], [0, 18])

    # Row 2: SPC (compute gaps from summary)
    spc_deg = {}
    spc_rec = {}
    for m in SPC_METHODS:
        pi = spc["methods"][f"{m}_scenario_i"]["psnr_mean"]
        pii = spc["methods"][f"{m}_scenario_ii"]["psnr_mean"]
        piii = spc["methods"][f"{m}_scenario_iii"]["psnr_mean"]
        spc_deg[m] = pi - pii
        spc_rec[m] = piii - pii
    _gap_row(axes[2, 0], axes[2, 1],
             SPC_METHODS, SPC_LABELS, SPC_COLORS,
             lambda m: spc_deg[m],
             lambda m: spc_rec[m],
             "SPC", [0, 16], [0, 14])

    # Shared y-axis labels on left column only
    for row in range(3):
        axes[row, 0].set_ylabel("PSNR Drop (dB)", fontsize=7)
        axes[row, 1].set_ylabel("PSNR Recovery (dB)", fontsize=7)

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    out = FIGURES_DIR / "fig3_gap_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    # Also save to cassi/ for backwards compat
    fig.savefig(FIGURES_DIR / "cassi" / "gap_comparison.pdf")
    fig.savefig(FIGURES_DIR / "cassi" / "gap_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Fig 3 saved: {out}")


# ###########################################################################
#  FIG 4 — Recovery ratio scatter
# ###########################################################################
def fig4_rho_scatter():
    cassi = load_json("cassi_summary.json")
    cacti = load_json("cacti_summary.json")
    spc = load_json("spc_summary.json")

    # Gather data points: (modality, label, psnr_i, rho, method_type)
    points = []

    for m in CASSI_METHODS:
        pi = cassi["scenario_i"][m]["psnr_mean"]
        pii = cassi["scenario_ii"][m]["psnr_mean"]
        piii = cassi["scenario_iii"][m]["psnr_mean"]
        deg = pi - pii
        rho = ((piii - pii) / deg * 100) if deg > 0.5 else None
        mtype = {"gap_tv": "classical", "hdnet": "mask-oblivious",
                 "mst_s": "mask-aware", "mst_l": "mask-aware"}[m]
        points.append(("CASSI", CASSI_LABELS[m], pi, rho, mtype))

    for m in CACTI_METHODS:
        pi = cacti["overall"]["scenario_i"][m]["psnr_mean"]
        pii = cacti["overall"]["scenario_ii"][m]["psnr_mean"]
        piii = cacti["overall"]["scenario_iii"][m]["psnr_mean"]
        deg = pi - pii
        rho = ((piii - pii) / deg * 100) if deg > 0.5 else None
        mtype = {"gap_tv": "classical", "pnp_ffdnet": "classical",
                 "elp_unfolding": "mask-aware",
                 "efficientsci": "mask-aware"}[m]
        points.append(("CACTI", CACTI_LABELS[m], pi, rho, mtype))

    for m in SPC_METHODS:
        pi = spc["methods"][f"{m}_scenario_i"]["psnr_mean"]
        pii = spc["methods"][f"{m}_scenario_ii"]["psnr_mean"]
        piii = spc["methods"][f"{m}_scenario_iii"]["psnr_mean"]
        deg = pi - pii
        rho = ((piii - pii) / deg * 100) if deg > 0.5 else None
        mtype = {"fista_tv": "classical", "ista_net": "mask-aware",
                 "hatnet": "mask-aware"}[m]
        points.append(("SPC", SPC_LABELS[m], pi, rho, mtype))

    type_markers = {"classical": "^", "mask-aware": "o", "mask-oblivious": "X"}
    type_sizes = {"classical": 50, "mask-aware": 50, "mask-oblivious": 65}

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    # Per-label annotation offsets to avoid overlaps
    label_offsets = {
        "FISTA-TV": (-10, 6), "HATNet": (-14, -8),
        "PnP-FFDNet": (-14, -8), "GAP-TV": (6, -6),
        "ISTA-Net": (-12, -8),
    }

    for mod, label, pi, rho, mtype in points:
        if rho is None or label == "HDNet":
            continue  # HDNet handled separately below
        ax.scatter(pi, rho, c=MOD_COLORS[mod], marker=type_markers[mtype],
                   s=type_sizes[mtype], edgecolors="black", linewidths=0.4,
                   zorder=5)
        ofs = label_offsets.get(label, (6, 3))
        ax.annotate(label, (pi, rho), textcoords="offset points",
                    xytext=ofs, fontsize=6.5,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))

    # HDNet special case (rho=0)
    hdnet = [p for p in points if p[1] == "HDNet"][0]
    ax.scatter(hdnet[2], 0, c=MOD_COLORS["CASSI"],
               marker=type_markers["mask-oblivious"],
               s=type_sizes["mask-oblivious"],
               edgecolors="black", linewidths=0.4, zorder=5)
    ax.annotate("HDNet", (hdnet[2], 0), textcoords="offset points",
                xytext=(6, 4), fontsize=6.5,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))

    # Trend line (excl HDNet)
    trend = [(pi, rho) for _, lbl, pi, rho, _ in points
             if rho is not None and lbl != "HDNet"]
    if len(trend) > 2:
        xs, ys = zip(*trend)
        z = np.polyfit(xs, ys, 1)
        xl = np.linspace(min(xs) - 1, max(xs) + 1, 100)
        ax.plot(xl, np.poly1d(z)(xl), "--", color="gray", alpha=0.4,
                linewidth=0.8, label=f"Trend ({z[0]:.1f}%/dB)")

    # Legends
    for mod, c in MOD_COLORS.items():
        ax.scatter([], [], c=c, s=30, label=mod, edgecolors="black",
                   linewidths=0.4)
    for mt, mk in type_markers.items():
        ax.scatter([], [], c="gray", marker=mk, s=30, label=mt,
                   edgecolors="black", linewidths=0.4)
    ax.legend(fontsize=5.5, loc="lower left", ncol=2, framealpha=0.9,
              handletextpad=0.3, columnspacing=0.6, borderpad=0.3)

    ax.set_xlabel("Scenario I PSNR (dB)", fontsize=8)
    ax.set_ylabel("Recovery Ratio ρ (%)", fontsize=8)
    ax.set_title("Recovery Ratio vs. Ideal Performance", fontsize=8,
                 fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_ylim(-5, 100); ax.set_xlim(18, 38)

    # Shaded regions
    ax.axhspan(80, 100, alpha=0.04, color="green")
    ax.axhspan(0, 20, alpha=0.04, color="red")
    ax.text(19, 92, "High recovery", fontsize=5.5, color="green", alpha=0.7)
    ax.text(19, 5, "Low recovery", fontsize=5.5, color="red", alpha=0.7)

    fig.tight_layout()
    out = FIGURES_DIR / "rho_scatter.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Fig 4 saved: {out}")


# ###########################################################################
#  FIG 5 — Residual gap bar chart
# ###########################################################################
def fig5_residual_gap():
    cassi = load_json("cassi_summary.json")
    cacti = load_json("cacti_summary.json")
    spc = load_json("spc_summary.json")

    groups = []

    # CASSI
    cassi_res = []
    for m in CASSI_METHODS:
        pi = cassi["scenario_i"][m]["psnr_mean"]
        piii = cassi["scenario_iii"][m]["psnr_mean"]
        cassi_res.append(pi - piii)
    groups.append(("CASSI", [CASSI_LABELS[m] for m in CASSI_METHODS],
                   cassi_res))

    # CACTI
    cacti_res = []
    for m in CACTI_METHODS:
        pi = cacti["overall"]["scenario_i"][m]["psnr_mean"]
        piii = cacti["overall"]["scenario_iii"][m]["psnr_mean"]
        cacti_res.append(pi - piii)
    groups.append(("CACTI", [CACTI_LABELS[m] for m in CACTI_METHODS],
                   cacti_res))

    # SPC
    spc_res = []
    for m in SPC_METHODS:
        pi = spc["methods"][f"{m}_scenario_i"]["psnr_mean"]
        piii = spc["methods"][f"{m}_scenario_iii"]["psnr_mean"]
        spc_res.append(pi - piii)
    groups.append(("SPC", [SPC_LABELS[m] for m in SPC_METHODS], spc_res))

    fig, ax = plt.subplots(figsize=(5.8, 2.8))

    positions, values, colors, labels_list = [], [], [], []
    pos = 0
    for mod, lbls, gaps in groups:
        for lbl, g in zip(lbls, gaps):
            positions.append(pos)
            values.append(g)
            colors.append(MOD_COLORS[mod])
            labels_list.append(lbl)
            pos += 1
        pos += 0.5

    bars = ax.bar(positions, values, color=colors, edgecolor="black",
                  linewidth=0.4, width=0.65)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_list, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Residual Gap Δ_res (dB)", fontsize=8)
    ax.set_title("Residual Gap: PSNR_I − PSNR_III", fontsize=8,
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)

    # Vertical separators between modality groups
    sep_positions = []
    idx = 0
    for mod, lbls, _ in groups:
        idx += len(lbls)
        sep_positions.append(idx - 0.75)  # halfway into the 0.5 gap
    for sp in sep_positions[:-1]:  # skip last separator
        ax.axvline(x=sp, color="gray", linewidth=0.5, linestyle=":",
                   alpha=0.5)

    # Legend
    for mod, c in MOD_COLORS.items():
        ax.bar([], [], color=c, label=mod, edgecolor="black", linewidth=0.4)
    ax.legend(fontsize=6.5, loc="upper right")

    fig.tight_layout()
    out = FIGURES_DIR / "residual_gap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Fig 5 saved: {out}")


# ###########################################################################
#  Main
# ###########################################################################
def main():
    print("=" * 60)
    print("Generating paper figures at LNCS print dimensions")
    print("=" * 60)

    fig1_scenario_comparison()
    fig3_gap_comparison()
    fig4_rho_scatter()
    fig5_residual_gap()

    # Fig 2 requires NPZ reconstruction files (Git LFS)
    npz_check = RESULTS_DIR / "cassi_reconstructions" / "scene01.npz"
    if npz_check.exists():
        fig2_visual_comparison()
    else:
        print(f"Fig 2 skipped: NPZ files not found at {npz_check}")
        print("  Run 'git lfs pull' to download reconstruction arrays.")

    print("=" * 60)
    print("Done. All figures in:", FIGURES_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
