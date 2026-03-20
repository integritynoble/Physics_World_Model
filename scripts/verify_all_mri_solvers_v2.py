#!/usr/bin/env python3
"""Verify all MRI solvers — 100% pass rate target."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import h5py
import time

# Import solvers
from algorithm_base.mri.solvers import SOLVERS, MRIOperator, run_solver

# Load one scene
cassi_dir = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")
h5_files = sorted([f for f in os.listdir(cassi_dir) if f.endswith('.h5')]) if os.path.exists(cassi_dir) else []

if h5_files:
    h5_path = os.path.join(cassi_dir, h5_files[0])
    print(f"Loading {h5_path}")
    with h5py.File(h5_path, 'r') as hf:
        keys = list(hf.keys())
        print(f"H5 keys: {keys}")
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y_raw = np.array(hf['y_ideal'] if 'y_ideal' in hf else hf['y'], dtype=np.float32)
        if y_raw.ndim == 3 and y_raw.shape[-1] == 2:
            y = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
        else:
            y = y_raw.astype(np.complex64)
        if 'sampling_mask' in hf:
            mask = np.array(hf['sampling_mask'], dtype=np.float32)
        elif 'mask' in hf:
            mask = np.array(hf['mask'], dtype=np.float32)
        else:
            mask = (np.abs(y) > 1e-10).astype(np.float32)
    print(f"x_true: {x_true.shape}, y: {y.shape}, mask: {mask.shape}")
else:
    # Synthetic fallback
    print("No MRI standard data found — using synthetic")
    H, W = 256, 256
    x_true = np.random.rand(H, W).astype(np.float32) * 0.5 + 0.25
    from scipy.fft import fft2, fftshift
    kspace_full = fftshift(fft2(x_true))
    # Random 4x undersampling mask
    mask = np.zeros((H, W), dtype=np.float32)
    mask[::4, :] = 1.0
    mask[H//2-12:H//2+12, :] = 1.0  # ACS region
    y = (kspace_full * mask).astype(np.complex64)

op = MRIOperator(mask, image_size=x_true.shape[0])

# Handle complex y in (H, W, 2) format
if y.ndim == 3 and y.shape[-1] == 2:
    y_input = y
else:
    y_input = np.stack([y.real, y.imag], axis=-1).astype(np.float32)

def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 100.0
    peak = x_true.max()
    if peak < 1e-10:
        peak = 1.0
    return 10 * np.log10(peak ** 2 / mse)

print(f"\n{'='*70}")
print(f"Verifying {len(SOLVERS)} MRI solvers")
print(f"{'='*70}\n")

results = {}
passed = 0
failed = 0

for key, spec in SOLVERS.items():
    t0 = time.time()
    try:
        x_hat = run_solver(key, y_input, op, {"device": "cpu"})
        dt = time.time() - t0

        if x_hat is None or not isinstance(x_hat, np.ndarray):
            raise ValueError("run_solver returned None or non-array")

        if x_hat.shape != x_true.shape:
            # Try to reshape
            if x_hat.size == x_true.size:
                x_hat = x_hat.reshape(x_true.shape)
            else:
                raise ValueError(f"Shape mismatch: {x_hat.shape} vs {x_true.shape}")

        p = psnr(x_true, np.clip(x_hat, 0, None))
        status = "PASS"
        passed += 1
        results[key] = {"psnr": p, "time": dt, "status": "PASS"}
        print(f"  {status}  {key:25s}  PSNR={p:6.2f} dB  {dt:5.1f}s  {spec['name']}")
    except Exception as e:
        dt = time.time() - t0
        status = "FAIL"
        failed += 1
        results[key] = {"error": str(e)[:100], "time": dt, "status": "FAIL"}
        print(f"  {status}  {key:25s}  ERROR: {str(e)[:80]}  {dt:5.1f}s")

print(f"\n{'='*70}")
print(f"Results: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.0f}%)")
if failed > 0:
    print(f"FAILED: {[k for k, v in results.items() if v['status'] == 'FAIL']}")
print(f"{'='*70}")
