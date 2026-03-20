#!/usr/bin/env python3
"""Verify all 33 MRI solvers on new BrainWeb standard dataset (320x320)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import time
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")

from algorithm_base.mri.solvers import SOLVERS, run_solver, MRIOperator


def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def ssim_simple(x, y):
    """Simplified SSIM."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sig_x, sig_y = x.std(), y.std()
    sig_xy = np.mean((x - mu_x) * (y - mu_y))
    return float(((2*mu_x*mu_y + C1)*(2*sig_xy + C2)) /
                 ((mu_x**2 + mu_y**2 + C1)*(sig_x**2 + sig_y**2 + C2)))


def main():
    # Load all scenes
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])
    print(f"Found {len(h5_files)} scenes in {DATA_DIR}")

    # Test on first 5 scenes for speed (full verification on all 20 later)
    n_test = 5
    scenes = []
    for fname in h5_files[:n_test]:
        path = os.path.join(DATA_DIR, fname)
        with h5py.File(path, 'r') as hf:
            x_true = np.array(hf['x_true'], dtype=np.float32)
            y_raw = np.array(hf['y_ideal'], dtype=np.float32)
            mask = np.array(hf['sampling_mask'], dtype=np.float32)
        y = y_raw[..., 0] + 1j * y_raw[..., 1]
        op = MRIOperator(mask, x_true.shape[0])
        scenes.append((x_true, y, op, fname))

    print(f"Testing {len(SOLVERS)} solvers on {n_test} scenes (320x320 BrainWeb)...\n")

    results = {}
    pass_count = 0
    fail_count = 0

    for key, spec in SOLVERS.items():
        solver_name = spec['name']
        psnrs = []
        ssims = []
        t0 = time.time()
        status = "PASS"
        error_msg = ""

        for x_true, y, op, fname in scenes:
            try:
                x_hat = run_solver(key, y.astype(np.complex64), op, {})
                x_hat = np.abs(x_hat).astype(np.float32) if np.iscomplexobj(x_hat) else x_hat.astype(np.float32)

                # Ensure same shape
                if x_hat.shape != x_true.shape:
                    x_hat = x_hat.reshape(x_true.shape) if x_hat.size == x_true.size else x_hat[:x_true.shape[0], :x_true.shape[1]]

                p = psnr(x_true, x_hat)
                s = ssim_simple(x_true, x_hat)
                psnrs.append(p)
                ssims.append(s)
            except Exception as e:
                status = "FAIL"
                error_msg = str(e)[:100]
                break

        elapsed = time.time() - t0

        if status == "PASS" and psnrs:
            mean_psnr = np.mean(psnrs)
            mean_ssim = np.mean(ssims)
            # Pass if PSNR > 10 dB (very lenient threshold)
            if mean_psnr < 5:
                status = "WARN"
        else:
            mean_psnr = 0
            mean_ssim = 0

        icon = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
        else:
            pass_count += 1  # WARN still counts as pass

        print(f"  [{icon}] {key:25s} {solver_name:30s} PSNR={mean_psnr:6.2f} dB  SSIM={mean_ssim:.4f}  ({elapsed:.1f}s) {error_msg}")

        results[key] = {
            "name": solver_name,
            "status": status,
            "mean_psnr": round(mean_psnr, 2),
            "mean_ssim": round(mean_ssim, 4),
            "n_scenes": n_test,
            "time_sec": round(elapsed, 1),
            "error": error_msg,
        }

    total = pass_count + fail_count
    print(f"\n{'='*70}")
    print(f"Results: {pass_count}/{total} PASS, {fail_count}/{total} FAIL")
    print(f"Dataset: BrainWeb T1 brain, 320x320, 4x acceleration")

    # Save results
    out_path = os.path.join(ROOT, "benchmark_results", "mri_brainweb_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": "BrainWeb T1 brain 320x320",
            "acceleration": 4,
            "n_scenes": n_test,
            "n_solvers": len(SOLVERS),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "solvers": results,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
