#!/usr/bin/env python3
"""Verify ReconFormer and MambaRecon on all 20 BrainWeb scenes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import time
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")

from algorithm_base.mri.solvers import SOLVERS, run_solver, MRIOperator

def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))

def ssim_simple(x_true, x_hat):
    from scipy.ndimage import uniform_filter
    C1, C2 = (0.01)**2, (0.03)**2
    mu_x = uniform_filter(x_true, 7)
    mu_y = uniform_filter(x_hat, 7)
    sigma_x2 = uniform_filter(x_true**2, 7) - mu_x**2
    sigma_y2 = uniform_filter(x_hat**2, 7) - mu_y**2
    sigma_xy = uniform_filter(x_true * x_hat, 7) - mu_x * mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))

SOLVERS_TO_TEST = ["reconformer", "mamba_recon"]

def main():
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])
    print(f"Found {len(h5_files)} scenes. Testing {len(SOLVERS_TO_TEST)} solvers...")

    results = {}
    for key in SOLVERS_TO_TEST:
        psnrs, ssims = [], []
        t0 = time.time()
        for fname in h5_files:
            path = os.path.join(DATA_DIR, fname)
            with h5py.File(path, 'r') as hf:
                x_true = np.array(hf['x_true'], dtype=np.float32)
                y_raw = np.array(hf['y_ideal'], dtype=np.float32)
                mask_data = np.array(hf['sampling_mask'], dtype=np.float32)
            y = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
            op = MRIOperator(mask_data, x_true.shape[0])

            x_hat = run_solver(key, y, op, {})
            if np.iscomplexobj(x_hat):
                x_hat = np.abs(x_hat).astype(np.float32)
            if x_hat.shape != x_true.shape:
                x_hat = x_hat.reshape(x_true.shape) if x_hat.size == x_true.size else x_hat[:x_true.shape[0], :x_true.shape[1]]

            p = psnr(x_true, x_hat)
            s = ssim_simple(x_true, x_hat)
            psnrs.append(p)
            ssims.append(s)

        elapsed = time.time() - t0
        avg_p = np.mean(psnrs)
        avg_s = np.mean(ssims)
        results[key] = {
            "name": SOLVERS[key]["name"],
            "mean_psnr": round(avg_p, 2),
            "mean_ssim": round(avg_s, 4),
            "min_psnr": round(min(psnrs), 2),
            "max_psnr": round(max(psnrs), 2),
            "n_scenes": len(psnrs),
            "total_time": round(elapsed, 1),
        }
        status = "PASS" if avg_p > 10 else "FAIL"
        print(f"  [{status}] {key:20s} {SOLVERS[key]['name']:25s} PSNR={avg_p:6.2f} dB  SSIM={avg_s:.4f}  ({elapsed:.1f}s)")

    out_path = os.path.join(ROOT, "benchmark_results", "mri_reconformer_mamba_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
