#!/usr/bin/env python3
"""GAP-TV tuning round 3 — explore small lam and fewer iters."""
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
    mse = np.mean((x_hat.astype(np.float64) - x_true.astype(np.float64)) ** 2)
    if mse < 1e-15: return float('inf')
    return float(10 * np.log10(float(x_true.max()) ** 2 / mse))

f = h5py.File(str(ROOT / "datasets/benchmark/cassi/standard/standard_cassi_00.h5"), 'r')
y = np.array(f['y_ideal'], dtype=np.float32)
x_true = np.array(f['x_true'], dtype=np.float32)
mask = np.array(f['mask'], dtype=np.float32)
n_bands = len(f['wavelength'])
step = int(f['H_params'].attrs.get('step', 2))
f.close()

from pwm_core.recon.gap_tv import gap_tv_cassi

configs = [
    # Very small lam, various iterations
    (10, 0.001, 1.0, "10it lam=0.001 acc=1.0"),
    (20, 0.001, 1.0, "20it lam=0.001 acc=1.0"),
    (30, 0.001, 1.0, "30it lam=0.001 acc=1.0"),
    (50, 0.001, 1.0, "50it lam=0.001 acc=1.0"),
    (10, 0.005, 1.0, "10it lam=0.005 acc=1.0"),
    (20, 0.005, 1.0, "20it lam=0.005 acc=1.0"),
    (30, 0.005, 1.0, "30it lam=0.005 acc=1.0"),
    (50, 0.005, 1.0, "50it lam=0.005 acc=1.0"),
    (10, 0.01, 1.0, "10it lam=0.01 acc=1.0"),
    (20, 0.01, 1.0, "20it lam=0.01 acc=1.0"),
    (30, 0.01, 1.0, "30it lam=0.01 acc=1.0"),
    # Very few iterations with various lam
    (5, 0.001, 1.0, "5it lam=0.001 acc=1.0"),
    (5, 0.005, 1.0, "5it lam=0.005 acc=1.0"),
    (5, 0.01, 1.0, "5it lam=0.01 acc=1.0"),
    (3, 0.001, 1.0, "3it lam=0.001 acc=1.0"),
    (3, 0.005, 1.0, "3it lam=0.005 acc=1.0"),
    # acc=0.5 sweet spot
    (20, 0.005, 0.5, "20it lam=0.005 acc=0.5"),
    (30, 0.005, 0.5, "30it lam=0.005 acc=0.5"),
    (50, 0.005, 0.5, "50it lam=0.005 acc=0.5"),
    (20, 0.001, 0.5, "20it lam=0.001 acc=0.5"),
    (30, 0.001, 0.5, "30it lam=0.001 acc=0.5"),
    (50, 0.001, 0.5, "50it lam=0.001 acc=0.5"),
    # Try TV-only: no GAP update, just iterate TV on initial back-projection
    # Actually that's just lam with 1 iteration each...
]

print(f"{'Config':45s} {'PSNR':>10s} {'Time':>8s}")
print("-" * 67)

best_psnr = 0
best_cfg = None
for iters, lam, acc, label in configs:
    t0 = time.time()
    x_hat = gap_tv_cassi(y, mask, n_bands, iterations=iters, lam=lam,
                          acc=acc, step=step, accelerate=False, device='cpu')
    dt = time.time() - t0
    p = psnr(x_hat, x_true)
    mark = " **" if p > best_psnr else ""
    if p > best_psnr:
        best_psnr = p
        best_cfg = label
    print(f"{label:45s} {p:8.2f} dB {dt:6.1f}s{mark}")

print(f"\nBest: {best_cfg} -> {best_psnr:.2f} dB")
print(f"Reference: GAP-TV on KAIST = 24.4 dB")
