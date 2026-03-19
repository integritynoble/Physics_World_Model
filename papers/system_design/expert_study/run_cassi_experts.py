#!/usr/bin/env python3
"""Run all 5 CASSI experts and collect results."""
import sys
import time
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(PROJECT_ROOT / "papers" / "system_design" / "expert_study"))

from data_loader import load_data
from expert_reconstructors import (
    expert_e1_cassi, expert_e2_cassi, expert_e3_cassi,
    expert_e4_cassi, expert_e5_cassi,
)

def psnr(x_true, x_recon):
    mse = np.mean((x_true - x_recon) ** 2)
    if mse < 1e-10:
        return 100.0
    peak = x_true.max()
    return 10 * np.log10(peak ** 2 / mse)

def ssim_3d(x_true, x_recon):
    from skimage.metrics import structural_similarity
    vals = []
    for b in range(x_true.shape[2]):
        dr = x_true[:, :, b].max() - x_true[:, :, b].min()
        if dr < 1e-10:
            dr = 1.0
        vals.append(structural_similarity(
            x_true[:, :, b], x_recon[:, :, b], data_range=dr
        ))
    return np.mean(vals)

def main():
    N_SAMPLES = 5

    print(f"Loading CASSI data ({N_SAMPLES} samples)...", flush=True)
    samples = load_data("sd_cassi", n_samples=N_SAMPLES)
    print(f"Loaded {len(samples)} samples", flush=True)

    experts = [
        ("E1", "POCS/ADMM", expert_e1_cassi),
        ("E2", "FBP/Fourier", expert_e2_cassi),
        ("E3", "FISTA+TV", expert_e3_cassi),
        ("E4", "CG/Iterative", expert_e4_cassi),
        ("E5", "PnP-NLM", expert_e5_cassi),
    ]

    all_results = []
    for eid, philosophy, expert_fn in experts:
        print(f"\n=== {eid} ({philosophy}) ===", flush=True)
        t0 = time.time()
        recons = expert_fn(samples)
        dt = time.time() - t0

        psnrs = []
        ssims = []
        for i, (s, recon) in enumerate(zip(samples, recons)):
            x_true = s["x_true"]
            p = psnr(x_true, recon)
            ss = ssim_3d(x_true, recon)
            psnrs.append(p)
            ssims.append(ss)
            print(f"  sample {i}: PSNR={p:.2f} dB, SSIM={ss:.4f}", flush=True)

        mean_p = np.mean(psnrs)
        std_p = np.std(psnrs)
        mean_s = np.mean(ssims)
        std_s = np.std(ssims)
        print(f"  MEAN: PSNR={mean_p:.2f}±{std_p:.2f} dB, SSIM={mean_s:.4f}±{std_s:.4f}, "
              f"time={dt:.1f}s", flush=True)

        all_results.append({
            "agent_id": eid,
            "agent_name": philosophy,
            "philosophy": philosophy,
            "modality": "sd_cassi",
            "psnr_mean": round(mean_p, 2),
            "psnr_std": round(std_p, 2),
            "ssim_mean": round(mean_s, 4),
            "ssim_std": round(std_s, 4),
            "wall_clock_s": round(dt, 2),
            "n_samples_success": len(psnrs),
            "success": True,
        })

    # Summary
    means = [r["psnr_mean"] for r in all_results]
    print(f"\n=== SUMMARY ===", flush=True)
    for r in all_results:
        print(f"  {r['agent_id']} ({r['philosophy']}): {r['psnr_mean']:.2f}±{r['psnr_std']:.2f} dB, "
              f"SSIM={r['ssim_mean']:.4f}", flush=True)
    print(f"  Overall mean: {np.mean(means):.2f} dB", flush=True)
    print(f"  CoV: {np.std(means)/np.mean(means)*100:.1f}%", flush=True)

    # Save results
    out_path = Path(__file__).parent / "results" / "cassi_improved_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
