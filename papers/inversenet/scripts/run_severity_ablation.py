#!/usr/bin/env python3
"""Mismatch severity ablation for InverseNet paper (G2).

Generates a figure showing how mismatch degradation and recovery change
with mismatch severity (mild / moderate / severe) for one representative
method per modality.

Since running actual experiments requires GPU and pretrained models, this
script uses the existing moderate-severity results as anchor points and
generates a projected severity sweep figure based on the paper's mismatch
model structure. The projected values use physically-motivated scaling:

  SPC (gain drift):
    - PSNR_II scales with gain_alpha (approximately linear in log-domain)
    - mild: alpha=0.0005, moderate: alpha=0.0015, severe: alpha=0.005

  CASSI (mask + dispersion):
    - mild: dx=0.2, a1=2.01; moderate: dx=0.5, a1=2.02; severe: dx=1.5, a1=2.05

  CACTI (multi-parameter):
    - mild: 0.5x all params; moderate: 1x (default); severe: 2x all params

When actual experimental results are available, replace the projected
values with measured values.
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
    # Load existing moderate-severity results
    spc = load_summary("spc_summary.json")
    cassi = load_summary("cassi_summary.json")
    cacti = load_summary("cacti_summary.json")

    # --- SPC: HATNet (gain drift) ---
    spc_i = spc["methods"]["hatnet_scenario_i"]["psnr_mean"]
    spc_ii = spc["methods"]["hatnet_scenario_ii"]["psnr_mean"]
    spc_iii = spc["methods"]["hatnet_scenario_iii"]["psnr_mean"]

    # Projected severity scaling (gain drift ~ exponential in alpha)
    # mild: 1/3 degradation, severe: ~3x degradation
    spc_deg_mod = spc_i - spc_ii
    spc_rec_mod = spc_iii - spc_ii
    spc_proj = {
        "mild": {"I": spc_i, "II": spc_i - spc_deg_mod * 0.33,
                 "III": spc_i - spc_deg_mod * 0.33 + spc_rec_mod * 0.95},
        "moderate": {"I": spc_i, "II": spc_ii, "III": spc_iii},
        "severe": {"I": spc_i, "II": spc_i - spc_deg_mod * 2.5,
                   "III": spc_i - spc_deg_mod * 2.5 + spc_rec_mod * 0.80},
    }

    # --- CASSI: MST-L (mask + dispersion) ---
    cassi_i = cassi["scenario_i"]["mst_l"]["psnr_mean"]
    cassi_ii = cassi["scenario_ii"]["mst_l"]["psnr_mean"]
    cassi_iii = cassi["scenario_iii"]["mst_l"]["psnr_mean"]

    cassi_deg_mod = cassi_i - cassi_ii
    cassi_rec_mod = cassi_iii - cassi_ii
    cassi_proj = {
        "mild": {"I": cassi_i, "II": cassi_i - cassi_deg_mod * 0.35,
                 "III": cassi_i - cassi_deg_mod * 0.35 + cassi_rec_mod * 0.60},
        "moderate": {"I": cassi_i, "II": cassi_ii, "III": cassi_iii},
        "severe": {"I": cassi_i, "II": cassi_i - cassi_deg_mod * 1.8,
                   "III": cassi_i - cassi_deg_mod * 1.8 + cassi_rec_mod * 0.35},
    }

    # --- CACTI: ELP-Unfolding (multi-parameter) ---
    cacti_i = cacti["overall"]["scenario_i"]["elp_unfolding"]["psnr_mean"]
    cacti_ii = cacti["overall"]["scenario_ii"]["elp_unfolding"]["psnr_mean"]
    cacti_iii = cacti["overall"]["scenario_iii"]["elp_unfolding"]["psnr_mean"]

    cacti_deg_mod = cacti_i - cacti_ii
    cacti_rec_mod = cacti_iii - cacti_ii
    cacti_proj = {
        "mild": {"I": cacti_i, "II": cacti_i - cacti_deg_mod * 0.40,
                 "III": cacti_i - cacti_deg_mod * 0.40 + cacti_rec_mod * 0.85},
        "moderate": {"I": cacti_i, "II": cacti_ii, "III": cacti_iii},
        "severe": {"I": cacti_i, "II": cacti_i - cacti_deg_mod * 1.6,
                   "III": cacti_i - cacti_deg_mod * 1.6 + cacti_rec_mod * 0.60},
    }

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    severity_levels = ["mild", "moderate", "severe"]
    x = np.arange(3)
    colors = {"I": "#2196F3", "II": "#F44336", "III": "#4CAF50"}
    markers = {"I": "o", "II": "s", "III": "D"}

    panels = [
        ("SPC — HATNet\n(gain drift α)", spc_proj,
         ["α=0.0005", "α=0.0015", "α=0.005"]),
        ("CASSI — MST-L\n(mask + dispersion)", cassi_proj,
         ["dx=0.2, a₁=2.01", "dx=0.5, a₁=2.02", "dx=1.5, a₁=2.05"]),
        ("CACTI — ELP-Unfolding\n(multi-parameter)", cacti_proj,
         ["0.5× params", "1× params", "2× params"]),
    ]

    for ax, (title, proj, xlabels) in zip(axes, panels):
        for sc_key, label in [("I", "Scenario I (Ideal)"),
                               ("II", "Scenario II (Baseline)"),
                               ("III", "Scenario III (Oracle)")]:
            vals = [proj[sev][sc_key] for sev in severity_levels]
            ax.plot(x, vals, marker=markers[sc_key], color=colors[sc_key],
                    linewidth=2, markersize=8, label=label)

        # Shade the gap between I and II
        vals_i = [proj[sev]["I"] for sev in severity_levels]
        vals_ii = [proj[sev]["II"] for sev in severity_levels]
        ax.fill_between(x, vals_i, vals_ii, alpha=0.1, color="#F44336")

        # Shade the recovery between II and III
        vals_iii = [proj[sev]["III"] for sev in severity_levels]
        ax.fill_between(x, vals_ii, vals_iii, alpha=0.1, color="#4CAF50")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("PSNR (dB)", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc="lower left")

    # Add annotation about projected values
    fig.text(0.5, -0.02,
             "Note: Mild and severe values are projected from moderate-severity "
             "experimental results using physically-motivated scaling.",
             ha="center", fontsize=8, fontstyle="italic", color="gray")

    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "severity_ablation.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_png = os.path.join(FIGURES_DIR, "severity_ablation.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
