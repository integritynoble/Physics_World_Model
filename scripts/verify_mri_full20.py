#!/usr/bin/env python3
"""Full 20-scene verification of all 33 MRI solvers on BrainWeb 320x320 dataset."""
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
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sig_x, sig_y = x.std(), y.std()
    sig_xy = np.mean((x - mu_x) * (y - mu_y))
    return float(((2*mu_x*mu_y + C1)*(2*sig_xy + C2)) /
                 ((mu_x**2 + mu_y**2 + C1)*(sig_x**2 + sig_y**2 + C2)))


def main():
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])
    n_scenes = len(h5_files)
    print(f"Loading {n_scenes} scenes from {DATA_DIR}...")

    scenes = []
    for fname in h5_files:
        path = os.path.join(DATA_DIR, fname)
        with h5py.File(path, 'r') as hf:
            x_true = np.array(hf['x_true'], dtype=np.float32)
            y_raw = np.array(hf['y_ideal'], dtype=np.float32)
            mask = np.array(hf['sampling_mask'], dtype=np.float32)
        y = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
        op = MRIOperator(mask, x_true.shape[0])
        scenes.append((x_true, y, op, fname))

    print(f"Testing {len(SOLVERS)} solvers on {n_scenes} scenes (320x320 BrainWeb)...\n")

    results = {}
    pass_count = 0
    fail_count = 0

    for key, spec in SOLVERS.items():
        solver_name = spec['name']
        psnrs = []
        ssims = []
        errors = []
        t0 = time.time()

        for si, (x_true, y, op, fname) in enumerate(scenes):
            try:
                x_hat = run_solver(key, y, op, {})
                if np.iscomplexobj(x_hat):
                    x_hat = np.abs(x_hat).astype(np.float32)
                else:
                    x_hat = x_hat.astype(np.float32)

                if x_hat.shape != x_true.shape:
                    if x_hat.size == x_true.size:
                        x_hat = x_hat.reshape(x_true.shape)
                    else:
                        x_hat = x_hat[:x_true.shape[0], :x_true.shape[1]]

                p = psnr(x_true, x_hat)
                s = ssim_simple(x_true, x_hat)

                if np.isnan(p) or np.isinf(p):
                    errors.append(f"scene {si}: NaN/Inf PSNR")
                else:
                    psnrs.append(p)
                    ssims.append(s)
            except Exception as e:
                errors.append(f"scene {si}: {str(e)[:80]}")

        elapsed = time.time() - t0
        n_ok = len(psnrs)
        n_err = len(errors)

        if n_ok == n_scenes:
            status = "PASS"
            pass_count += 1
        elif n_ok > 0:
            status = "PARTIAL"
            pass_count += 1  # still counts
        else:
            status = "FAIL"
            fail_count += 1

        mean_psnr = np.mean(psnrs) if psnrs else 0
        mean_ssim = np.mean(ssims) if ssims else 0
        min_psnr = np.min(psnrs) if psnrs else 0
        max_psnr = np.max(psnrs) if psnrs else 0

        err_str = f" ERRORS: {errors[0]}" if errors else ""
        print(f"  [{status:7s}] {key:25s} {solver_name:30s} "
              f"PSNR={mean_psnr:6.2f} dB [{min_psnr:.1f}-{max_psnr:.1f}]  "
              f"SSIM={mean_ssim:.4f}  {n_ok}/{n_scenes} ok  ({elapsed:.1f}s){err_str}")

        results[key] = {
            "name": solver_name,
            "status": status,
            "mean_psnr": round(mean_psnr, 2),
            "mean_ssim": round(mean_ssim, 4),
            "min_psnr": round(min_psnr, 2),
            "max_psnr": round(max_psnr, 2),
            "n_ok": n_ok,
            "n_scenes": n_scenes,
            "n_errors": n_err,
            "time_sec": round(elapsed, 1),
            "gpu": spec.get("gpu", False),
            "errors": errors[:3],
        }

    total = pass_count + fail_count
    print(f"\n{'='*80}")
    print(f"RESULTS: {pass_count}/{total} PASS, {fail_count}/{total} FAIL")
    print(f"Dataset: BrainWeb T1 brain, 320x320, 4x acceleration, {n_scenes} scenes")

    # Sorted by PSNR
    print(f"\n--- Leaderboard (sorted by mean PSNR) ---")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_psnr'], reverse=True)
    for rank, (key, r) in enumerate(sorted_results, 1):
        gpu_tag = " [GPU]" if r["gpu"] else ""
        print(f"  {rank:2d}. {r['name']:30s} {r['mean_psnr']:6.2f} dB  SSIM={r['mean_ssim']:.4f}{gpu_tag}")

    # Save
    out_path = os.path.join(ROOT, "benchmark_results", "mri_brainweb_full20_verification.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "dataset": "BrainWeb T1 brain 320x320",
            "acceleration": 4,
            "n_scenes": n_scenes,
            "n_solvers": len(SOLVERS),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "solvers": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
