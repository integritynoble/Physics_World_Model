#!/usr/bin/env python3
"""Quick test of ReconFormer and MambaRecon solvers on 1 scene."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")

from algorithm_base.mri.solvers import SOLVERS, run_solver, MRIOperator

def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))

NEW_SOLVERS = ["reconformer", "mamba_recon"]

def main():
    # Load first scene
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])
    fname = h5_files[0]
    path = os.path.join(DATA_DIR, fname)
    with h5py.File(path, 'r') as hf:
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y_raw = np.array(hf['y_ideal'], dtype=np.float32)
        mask = np.array(hf['sampling_mask'], dtype=np.float32)
    y = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
    op = MRIOperator(mask, x_true.shape[0])

    print(f"Testing {len(NEW_SOLVERS)} new solvers on {fname} ({x_true.shape})...")
    print(f"Total registered solvers: {len(SOLVERS)}")
    print()

    for key in NEW_SOLVERS:
        if key not in SOLVERS:
            print(f"  [MISSING] {key} not in SOLVERS registry!")
            continue
        spec = SOLVERS[key]
        t0 = time.time()
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
            elapsed = time.time() - t0
            print(f"  [PASS] {key:20s} {spec['name']:25s} PSNR={p:6.2f} dB  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {key:20s} {spec['name']:25s} Error: {str(e)[:120]}  ({elapsed:.1f}s)")

    print(f"\nDone.")

if __name__ == "__main__":
    main()
