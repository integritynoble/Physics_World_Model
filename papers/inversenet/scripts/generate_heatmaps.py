#!/usr/bin/env python3
"""Generate per-scene heatmaps (G9) and residual gap bar chart (G10) for InverseNet paper.

Produces:
1. Per-scene recovery ratio heatmap for CASSI (10 scenes x 4 methods)
2. Per-video recovery ratio heatmap for CACTI (6 videos x 4 methods)
3. Per-image recovery ratio heatmap for SPC (11 images x 3 methods)
4. Residual gap bar chart (Delta_res per method, grouped by modality)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


def compute_rho(pi, pii, piii, min_deg=0.5):
    """Compute recovery ratio, return NaN if degradation is too small."""
    deg = pi - pii
    if deg < min_deg:
        return np.nan
    return (piii - pii) / deg * 100


def generate_cassi_heatmap():
    """Generate CASSI per-scene recovery ratio heatmap."""
    data = load_json("cassi_validation_results.json")
    methods = ["gap_tv", "hdnet", "mst_s", "mst_l"]
    method_labels = ["GAP-TV", "HDNet", "MST-S", "MST-L"]
    scene_labels = [f"Scene {i+1}" for i in range(10)]

    rho_matrix = np.full((len(methods), len(data)), np.nan)
    for j, scene in enumerate(data):
        for i, m in enumerate(methods):
            pi = scene["scenario_i"][m]["psnr"]
            pii = scene["scenario_ii"][m]["psnr"]
            piii = scene["scenario_iii"][m]["psnr"]
            rho_matrix[i, j] = compute_rho(pi, pii, piii)

    return rho_matrix, method_labels, scene_labels, "CASSI"


def generate_cacti_heatmap():
    """Generate CACTI per-video recovery ratio heatmap."""
    data = load_json("cacti_validation_results.json")
    methods = ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficientsci"]
    method_labels = ["GAP-TV", "PnP-FFDNet", "ELP-Unfolding", "EfficientSCI"]
    video_labels = [v["name"].capitalize() for v in data]

    rho_matrix = np.full((len(methods), len(data)), np.nan)
    for j, video in enumerate(data):
        for i, m in enumerate(methods):
            pi = video["scenarios"]["scenario_i"][m]["psnr"]
            pii = video["scenarios"]["scenario_ii"][m]["psnr"]
            piii = video["scenarios"]["scenario_iii"][m]["psnr"]
            rho_matrix[i, j] = compute_rho(pi, pii, piii)

    return rho_matrix, method_labels, video_labels, "CACTI"


def generate_spc_heatmap():
    """Generate SPC per-image recovery ratio heatmap."""
    data = load_json("spc_validation_results.json")
    methods = ["fista_tv", "ista_net", "hatnet"]
    method_labels = ["FISTA-TV", "ISTA-Net", "HATNet"]
    image_labels = [img["image_name"] for img in data]

    rho_matrix = np.full((len(methods), len(data)), np.nan)
    for j, img in enumerate(data):
        for i, m in enumerate(methods):
            pi = img[m]["scenario_i"]["psnr"]
            pii = img[m]["scenario_ii"]["psnr"]
            piii = img[m]["scenario_iii"]["psnr"]
            rho_matrix[i, j] = compute_rho(pi, pii, piii)

    return rho_matrix, method_labels, image_labels, "SPC"


def plot_heatmap(ax, rho_matrix, method_labels, scene_labels, title):
    """Plot a single heatmap on the given axes."""
    # Custom colormap: red (0%) -> yellow (50%) -> green (100%)
    cmap = LinearSegmentedColormap.from_list(
        "rho", [(0.8, 0.2, 0.2), (1.0, 1.0, 0.4), (0.2, 0.7, 0.2)])

    im = ax.imshow(rho_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(scene_labels)))
    ax.set_xticklabels(scene_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(method_labels)))
    ax.set_yticklabels(method_labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Annotate cells
    for i in range(rho_matrix.shape[0]):
        for j in range(rho_matrix.shape[1]):
            val = rho_matrix[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:.0f}%"
                color = "white" if val < 30 or val > 85 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=6.5, color=color, fontweight="bold")

    return im


def generate_heatmap_figure():
    """Generate combined heatmap figure."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             gridspec_kw={"height_ratios": [4, 4, 3]})

    datasets = [generate_cassi_heatmap, generate_cacti_heatmap,
                generate_spc_heatmap]

    for ax, gen_fn in zip(axes, datasets):
        rho_matrix, method_labels, scene_labels, title = gen_fn()
        im = plot_heatmap(ax, rho_matrix, method_labels, scene_labels, title)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Recovery Ratio ρ (%)", fontsize=9)

    fig.suptitle("Per-Scene Recovery Ratio Heatmaps", fontsize=13,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])

    out_path = os.path.join(FIGURES_DIR, "recovery_heatmaps.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_png = os.path.join(FIGURES_DIR, "recovery_heatmaps.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)


def generate_residual_gap_figure():
    """Generate residual gap bar chart (G10)."""
    cassi = load_json("cassi_summary.json")
    cacti = load_json("cacti_summary.json")
    spc = load_json("spc_summary.json")

    # Compute residual gap = PSNR_I - PSNR_III for each method
    groups = []

    # CASSI
    cassi_methods = ["gap_tv", "hdnet", "mst_s", "mst_l"]
    cassi_labels = ["GAP-TV", "HDNet", "MST-S", "MST-L"]
    cassi_res = []
    for m in cassi_methods:
        pi = cassi["scenario_i"][m]["psnr_mean"]
        piii = cassi["scenario_iii"][m]["psnr_mean"]
        cassi_res.append(pi - piii)
    groups.append(("CASSI", cassi_labels, cassi_res))

    # CACTI
    cacti_methods = ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficientsci"]
    cacti_labels = ["GAP-TV", "PnP-FFDNet", "ELP-Unf.", "Eff.SCI"]
    cacti_res = []
    for m in cacti_methods:
        pi = cacti["overall"]["scenario_i"][m]["psnr_mean"]
        piii = cacti["overall"]["scenario_iii"][m]["psnr_mean"]
        cacti_res.append(pi - piii)
    groups.append(("CACTI", cacti_labels, cacti_res))

    # SPC
    spc_methods = ["fista_tv", "ista_net", "hatnet"]
    spc_labels = ["FISTA-TV", "ISTA-Net", "HATNet"]
    spc_res = []
    for m in spc_methods:
        pi = spc["methods"][f"{m}_scenario_i"]["psnr_mean"]
        piii = spc["methods"][f"{m}_scenario_iii"]["psnr_mean"]
        spc_res.append(pi - piii)
    groups.append(("SPC", spc_labels, spc_res))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    mod_colors = {"CASSI": "#E91E63", "CACTI": "#2196F3", "SPC": "#FF9800"}
    bar_positions = []
    bar_values = []
    bar_colors = []
    bar_labels = []
    pos = 0

    for mod_name, labels, res_gaps in groups:
        for label, gap in zip(labels, res_gaps):
            bar_positions.append(pos)
            bar_values.append(gap)
            bar_colors.append(mod_colors[mod_name])
            bar_labels.append(label)
            pos += 1
        pos += 0.5  # gap between modality groups

    bars = ax.bar(bar_positions, bar_values, color=bar_colors,
                  edgecolor="black", linewidth=0.5, width=0.7)

    # Value labels on bars
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold")

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Residual Gap Δ_res (dB)", fontsize=10)
    ax.set_title("Residual Gap: PSNR_I − PSNR_III\n(Unrecoverable Loss from Measurement Corruption)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Modality group labels
    group_positions = []
    idx = 0
    for mod_name, labels, _ in groups:
        center = np.mean(bar_positions[idx:idx + len(labels)])
        group_positions.append((center, mod_name))
        idx += len(labels)

    # Add modality labels below x-axis
    for center, mod_name in group_positions:
        ax.text(center, -0.12, mod_name, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9, fontweight="bold",
                color=mod_colors[mod_name])

    # Legend
    for mod_name, color in mod_colors.items():
        ax.bar([], [], color=color, label=mod_name, edgecolor="black",
               linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "residual_gap.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_png = os.path.join(FIGURES_DIR, "residual_gap.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating per-scene recovery ratio heatmaps...")
    generate_heatmap_figure()
    print("Generating residual gap bar chart...")
    generate_residual_gap_figure()
    print("Done.")
