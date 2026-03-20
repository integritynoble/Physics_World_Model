#!/usr/bin/env python3
"""Parameter sweep for CASSI GAP-TV to match reference PSNR."""
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
    x_h = np.asarray(x_hat, dtype=np.float64)
    x_t = np.asarray(x_true, dtype=np.float64)
    mse = np.mean((x_h - x_t) ** 2)
    if mse < 1e-15:
        return float('inf')
    return float(10 * np.log10(float(x_t.max()) ** 2 / mse))


# Load standard CASSI data
f = h5py.File(str(ROOT / "datasets/benchmark/cassi/standard/standard_cassi_00.h5"), 'r')
y = np.array(f['y_ideal'], dtype=np.float32)
x_true = np.array(f['x_true'], dtype=np.float32)
mask = np.array(f['mask'], dtype=np.float32)
n_bands = len(f['wavelength'])
step = int(f['H_params'].attrs.get('step', 2))
f.close()

print(f"Data: y={y.shape}, x_true={x_true.shape}, n_bands={n_bands}, step={step}")
print(f"x_true range: [{x_true.min():.3f}, {x_true.max():.3f}]")
print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
print(f"mask: unique={np.unique(mask)}, shape={mask.shape}")
print()

from pwm_core.recon.gap_tv import gap_tv_cassi

# Parameter sweep
configs = [
    # (iters, lam, acc, accelerate, label)
    (50, 0.05, 1.0, False, "default (50it, lam=0.05)"),
    (100, 0.05, 1.0, False, "100it, lam=0.05"),
    (100, 0.5, 1.0, False, "100it, lam=0.5"),
    (100, 1.0, 1.0, False, "100it, lam=1.0"),
    (100, 2.0, 1.0, False, "100it, lam=2.0"),
    (100, 5.0, 1.0, False, "100it, lam=5.0"),
    (200, 1.0, 1.0, False, "200it, lam=1.0"),
    (200, 2.0, 1.0, False, "200it, lam=2.0"),
    (50, 1.0, 1.0, False, "50it, lam=1.0"),
    (50, 2.0, 1.0, False, "50it, lam=2.0"),
    (50, 0.5, 1.0, False, "50it, lam=0.5"),
]

print(f"{'Config':40s} {'PSNR':>10s} {'Time':>8s}")
print("-" * 62)

best_psnr = 0
best_cfg = None
for iters, lam, acc, accelerate, label in configs:
    t0 = time.time()
    x_hat = gap_tv_cassi(y, mask, n_bands, iterations=iters, lam=lam,
                          acc=acc, step=step, accelerate=accelerate,
                          device='cpu')
    dt = time.time() - t0
    p = psnr(x_hat, x_true)
    mark = " *" if p > best_psnr else ""
    if p > best_psnr:
        best_psnr = p
        best_cfg = label
    print(f"{label:40s} {p:8.2f} dB {dt:6.1f}s{mark}")

print(f"\nBest: {best_cfg} -> {best_psnr:.2f} dB")
print(f"Reference: GAP-TV on KAIST = 24.4 dB, GAP-TV guided = 26.2 dB")
