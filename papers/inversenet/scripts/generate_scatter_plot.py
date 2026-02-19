#!/usr/bin/env python3
"""Generate recovery ratio vs ideal PSNR scatter plot for InverseNet paper (G4).

Scatter plot of rho (y-axis) vs Scenario I PSNR (x-axis) for all 11 methods
across 3 modalities. Color-coded by modality, shape-coded by method type
(classical/operator-aware/mask-oblivious).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_summary(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


def main():
    cassi = load_summary("cassi_summary.json")
    cacti = load_summary("cacti_summary.json")
    spc = load_summary("spc_summary.json")

    # Define all methods with their properties
    # (name, modality, psnr_i, psnr_ii, psnr_iii, method_type)
    methods = []

    # CASSI methods
    for m in ["gap_tv", "hdnet", "mst_s", "mst_l"]:
        pi = cassi["scenario_i"][m]["psnr_mean"]
        pii = cassi["scenario_ii"][m]["psnr_mean"]
        piii = cassi["scenario_iii"][m]["psnr_mean"]
        deg = pi - pii
        rec = piii - pii
        rho = (rec / deg * 100) if deg > 0.5 else None  # skip near-zero deg
        mtype = {"gap_tv": "classical", "hdnet": "mask-oblivious",
                 "mst_s": "operator-aware", "mst_l": "operator-aware"}[m]
        label = {"gap_tv": "GAP-TV", "hdnet": "HDNet",
                 "mst_s": "MST-S", "mst_l": "MST-L"}[m]
        methods.append(("CASSI", label, pi, rho, mtype))

    # CACTI methods
    for m in ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficientsci"]:
        pi = cacti["overall"]["scenario_i"][m]["psnr_mean"]
        pii = cacti["overall"]["scenario_ii"][m]["psnr_mean"]
        piii = cacti["overall"]["scenario_iii"][m]["psnr_mean"]
        deg = pi - pii
        rec = piii - pii
        rho = (rec / deg * 100) if deg > 0.5 else None
        mtype = {"gap_tv": "classical", "pnp_ffdnet": "classical",
                 "elp_unfolding": "operator-aware",
                 "efficientsci": "operator-aware"}[m]
        label = {"gap_tv": "GAP-TV", "pnp_ffdnet": "PnP-FFDNet",
                 "elp_unfolding": "ELP-Unf.",
                 "efficientsci": "Eff.SCI"}[m]
        methods.append(("CACTI", label, pi, rho, mtype))

    # SPC methods
    for m in ["fista_tv", "ista_net", "hatnet"]:
        key_i = f"{m}_scenario_i"
        key_ii = f"{m}_scenario_ii"
        key_iii = f"{m}_scenario_iii"
        pi = spc["methods"][key_i]["psnr_mean"]
        pii = spc["methods"][key_ii]["psnr_mean"]
        piii = spc["methods"][key_iii]["psnr_mean"]
        deg = pi - pii
        rec = piii - pii
        rho = (rec / deg * 100) if deg > 0.5 else None
        mtype = {"fista_tv": "classical", "ista_net": "operator-aware",
                 "hatnet": "operator-aware"}[m]
        label = {"fista_tv": "FISTA-TV", "ista_net": "ISTA-Net",
                 "hatnet": "HATNet"}[m]
        methods.append(("SPC", label, pi, rho, mtype))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by modality
    mod_colors = {"CASSI": "#E91E63", "CACTI": "#2196F3", "SPC": "#FF9800"}
    # Marker by method type
    type_markers = {"classical": "^", "operator-aware": "o", "mask-oblivious": "X"}
    type_sizes = {"classical": 120, "operator-aware": 120, "mask-oblivious": 150}

    # Plot each method
    for mod, label, pi, rho, mtype in methods:
        if rho is None:
            continue
        ax.scatter(pi, rho, c=mod_colors[mod], marker=type_markers[mtype],
                   s=type_sizes[mtype], edgecolors="black", linewidths=0.5,
                   zorder=5)
        # Label each point
        offset_x, offset_y = 0.4, 1.5
        ax.annotate(label, (pi, rho), textcoords="offset points",
                    xytext=(8, 4), fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # HDNet special case: rho=0, plot at bottom
    hdnet_entry = [m for m in methods if m[1] == "HDNet"][0]
    ax.scatter(hdnet_entry[2], 0, c=mod_colors["CASSI"],
               marker=type_markers["mask-oblivious"],
               s=type_sizes["mask-oblivious"],
               edgecolors="black", linewidths=0.5, zorder=5)
    ax.annotate("HDNet", (hdnet_entry[2], 0), textcoords="offset points",
                xytext=(8, 4), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Trend line (excluding HDNet which is mask-oblivious outlier)
    trend_data = [(pi, rho) for mod, label, pi, rho, mtype in methods
                  if rho is not None and label != "HDNet"]
    if len(trend_data) > 2:
        xs, ys = zip(*trend_data)
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xs) - 1, max(xs) + 1, 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.4, linewidth=1,
                label=f"Trend (slope={z[0]:.1f}%/dB)")

    # Legend for modalities
    for mod, color in mod_colors.items():
        ax.scatter([], [], c=color, s=80, label=mod, edgecolors="black",
                   linewidths=0.5)
    # Legend for method types
    for mtype, marker in type_markers.items():
        ax.scatter([], [], c="gray", marker=marker, s=80,
                   label=f"{mtype}", edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Scenario I PSNR (dB) — Ideal Performance", fontsize=11)
    ax.set_ylabel("Recovery Ratio ρ (%)", fontsize=11)
    ax.set_title("Recovery Ratio vs. Ideal Performance\nAcross 11 Methods and 3 Modalities",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 100)
    ax.set_xlim(18, 38)

    # Add shaded regions for interpretation
    ax.axhspan(80, 100, alpha=0.05, color="green")
    ax.axhspan(0, 20, alpha=0.05, color="red")
    ax.text(19, 92, "High recovery", fontsize=7, color="green", alpha=0.7)
    ax.text(19, 5, "Low recovery", fontsize=7, color="red", alpha=0.7)

    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "rho_scatter.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_png = os.path.join(FIGURES_DIR, "rho_scatter.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
