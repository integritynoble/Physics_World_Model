#!/usr/bin/env python3
"""Test all CT and MRI solvers against rebuilt standard datasets."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Force PWM5 imports (not PWM4)
pwm5_pkg = str(ROOT / "packages" / "pwm_core")
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, pwm5_pkg)
sys.path.insert(0, str(ROOT))
for k in list(sys.modules):
    if 'pwm_core' in k or 'algorithm_base' in k:
        del sys.modules[k]

import numpy as np
import h5py
import time


def psnr(x_hat, x_true):
    mse = np.mean((x_hat.astype(np.float64) - x_true.astype(np.float64))**2)
    if mse < 1e-15:
        return float('inf')
    return float(10 * np.log10(x_true.max()**2 / mse))


def test_ct():
    print("=" * 60)
    print("CT SOLVER TESTS")
    print("=" * 60)

    from algorithm_base.ct.solvers import SOLVERS, run_solver

    ct_dir = ROOT / "datasets" / "benchmark" / "ct" / "standard"
    results = {}

    # Test on 3 samples for speed
    for idx in range(3):
        h5_path = ct_dir / f"standard_ct_{idx:02d}.h5"
        if not h5_path.exists():
            continue

        f = h5py.File(str(h5_path), 'r')
        y = np.array(f['y_ideal'], dtype=np.float32)
        x_true = np.array(f['x_true'], dtype=np.float32)
        f.close()

        print(f"\n  Sample {idx:02d}: y={y.shape}, x_true={x_true.shape}")

        for key, info in SOLVERS.items():
            t0 = time.time()
            try:
                x_hat = run_solver(key, y, None, {'output_size': 256})
                dt = time.time() - t0
                p = psnr(x_hat, x_true)
                if key not in results:
                    results[key] = []
                results[key].append(p)
                print(f"    {key:20s}: PSNR={p:6.2f} dB  ({dt:.1f}s)  "
                      f"shape={x_hat.shape}  [{x_hat.min():.3f},{x_hat.max():.3f}]")
            except Exception as e:
                dt = time.time() - t0
                print(f"    {key:20s}: ERROR ({dt:.1f}s) — {str(e)[:100]}")
                if key not in results:
                    results[key] = []
                results[key].append(-1)

    print("\n  CT Summary:")
    for key, vals in results.items():
        good = [v for v in vals if v > 0]
        if good:
            print(f"    {key:20s}: avg PSNR = {np.mean(good):.2f} dB ({len(good)}/{len(vals)} OK)")
        else:
            print(f"    {key:20s}: ALL FAILED")

    return results


def test_mri():
    print("\n" + "=" * 60)
    print("MRI SOLVER TESTS")
    print("=" * 60)

    from algorithm_base.mri.solvers import SOLVERS, run_solver

    mri_dir = ROOT / "datasets" / "benchmark" / "mri" / "standard"
    results = {}

    for idx in range(3):
        h5_path = mri_dir / f"standard_mri_{idx:02d}.h5"
        if not h5_path.exists():
            continue

        f = h5py.File(str(h5_path), 'r')
        y = np.array(f['y_ideal'], dtype=np.float32)  # (256, 256, 2) real+imag
        x_true = np.array(f['x_true'], dtype=np.float32)
        f.close()

        print(f"\n  Sample {idx:02d}: y={y.shape}, x_true={x_true.shape}")

        for key, info in SOLVERS.items():
            t0 = time.time()
            try:
                x_hat = run_solver(key, y, None, {})
                dt = time.time() - t0
                # Handle shape mismatch
                if x_hat.shape != x_true.shape:
                    if x_hat.ndim > x_true.ndim:
                        x_hat = np.abs(x_hat) if np.iscomplexobj(x_hat) else x_hat
                    if x_hat.shape != x_true.shape:
                        x_hat = x_hat.reshape(x_true.shape) if x_hat.size == x_true.size else x_hat[:x_true.shape[0], :x_true.shape[1]]

                p = psnr(np.abs(x_hat), x_true)
                if key not in results:
                    results[key] = []
                results[key].append(p)
                print(f"    {key:20s}: PSNR={p:6.2f} dB  ({dt:.1f}s)  "
                      f"shape={x_hat.shape}  [{np.abs(x_hat).min():.3f},{np.abs(x_hat).max():.3f}]")
            except Exception as e:
                dt = time.time() - t0
                print(f"    {key:20s}: ERROR ({dt:.1f}s) — {str(e)[:100]}")
                if key not in results:
                    results[key] = []
                results[key].append(-1)

    print("\n  MRI Summary:")
    for key, vals in results.items():
        good = [v for v in vals if v > 0]
        if good:
            print(f"    {key:20s}: avg PSNR = {np.mean(good):.2f} dB ({len(good)}/{len(vals)} OK)")
        else:
            print(f"    {key:20s}: ALL FAILED")

    return results


if __name__ == "__main__":
    ct_results = test_ct()
    mri_results = test_mri()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for modality, results in [("CT", ct_results), ("MRI", mri_results)]:
        print(f"\n  {modality}:")
        for key, vals in results.items():
            good = [v for v in vals if v > 0]
            status = f"avg {np.mean(good):.1f} dB" if good else "FAILED"
            print(f"    {key:20s}: {status}")
