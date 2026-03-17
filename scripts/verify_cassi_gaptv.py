#!/usr/bin/env python3
"""Verify GAP-TV fix on CASSI standard data (KAIST scenes).

Tests that the fixed gap_tv_cassi with skimage Chambolle TV reaches ~24 dB
reference PSNR on real KAIST data.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pwm5_pkg = str(ROOT / "packages" / "pwm_core")
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, pwm5_pkg)
sys.path.insert(0, str(ROOT))
for k in list(sys.modules):
    if 'pwm_core' in k or 'algorithm_base' in k:
        del sys.modules[k]

import numpy as np
import h5py


def psnr(x_hat, x_true):
    x_hat = np.clip(x_hat, 0, 1).astype(np.float64)
    x_true = np.clip(x_true, 0, 1).astype(np.float64)
    mse = np.mean((x_hat - x_true) ** 2)
    if mse < 1e-10: return 100.0
    return float(10 * np.log10(1.0 / mse))


cassi_dir = ROOT / "datasets" / "benchmark" / "cassi" / "standard"
results = []

# Test GAP-TV from gap_tv.py
from pwm_core.recon.gap_tv import gap_tv_cassi

configs = [
    ("GAP-TV 100it lam=0.1", dict(iterations=100, lam=0.1, tv_iter=5)),
    ("GAP-TV 200it lam=0.01", dict(iterations=200, lam=0.01, tv_iter=5)),
]

for label, kwargs in configs:
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"{'='*60}")
    psnrs = []
    for idx in range(10):
        h5_path = cassi_dir / f"standard_cassi_{idx:02d}.h5"
        if not h5_path.exists():
            print(f"  scene{idx+1:02d}: NOT FOUND")
            continue
        f = h5py.File(str(h5_path), 'r')
        y = np.array(f['y_ideal'], dtype=np.float32)
        x_true = np.array(f['x_true'], dtype=np.float32)
        mask = np.array(f['mask'], dtype=np.float32)
        n_bands = len(f['wavelength'])
        step_val = int(f['H_params'].attrs.get('step', 2))
        f.close()

        t0 = time.time()
        x_hat = gap_tv_cassi(y, mask, n_bands, step=step_val, **kwargs)
        dt = time.time() - t0
        p = psnr(x_hat, x_true)
        psnrs.append(p)
        print(f"  scene{idx+1:02d}: {p:.2f} dB  ({dt:.1f}s)")

    if psnrs:
        mean_p = np.mean(psnrs)
        std_p = np.std(psnrs)
        print(f"\n  Mean PSNR: {mean_p:.2f} +/- {std_p:.2f} dB")
        print(f"  Reference: GAP-TV = 24.34 dB (InverseNet)")
        results.append((label, mean_p, std_p))

# Also test via the solver interface
print(f"\n{'='*60}")
print("Testing via algorithm_base solver interface")
print(f"{'='*60}")

from algorithm_base.cassi.solvers import run_solver, CASSIOperator

h5_path = cassi_dir / "standard_cassi_00.h5"
if h5_path.exists():
    f = h5py.File(str(h5_path), 'r')
    y = np.array(f['y_ideal'], dtype=np.float32)
    x_true = np.array(f['x_true'], dtype=np.float32)
    mask = np.array(f['mask'], dtype=np.float32)
    n_bands = len(f['wavelength'])
    step_val = int(f['H_params'].attrs.get('step', 2))
    f.close()

    op = CASSIOperator(mask, n_bands, step_val)

    for key in ["traditional_cpu", "best_quality", "small_gpu"]:
        t0 = time.time()
        x_hat = run_solver(key, y, op)
        dt = time.time() - t0
        p = psnr(x_hat, x_true)
        print(f"  {key:20s}: {p:.2f} dB  ({dt:.1f}s)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for label, mean_p, std_p in results:
    status = "PASS" if mean_p >= 23.0 else "FAIL"
    print(f"  [{status}] {label}: {mean_p:.2f} +/- {std_p:.2f} dB")
print(f"\n  Reference: GAP-TV = 24.34 dB mean (InverseNet ECCV benchmark)")
