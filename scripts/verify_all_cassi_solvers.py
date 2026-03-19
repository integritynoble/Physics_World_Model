#!/usr/bin/env python3
"""Verify ALL CASSI solvers from algorithm_base/cassi/solvers.py via run_solver()."""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import glob
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "algorithm_base" / "cassi"))


def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def load_scene(idx=0):
    h5_files = sorted(glob.glob(str(ROOT / "datasets/benchmark/cassi/standard/*.h5")))
    if not h5_files:
        print("ERROR: No CASSI scenes found")
        return None
    f = h5_files[idx]
    with h5py.File(f, 'r') as hf:
        return {
            'name': Path(f).stem,
            'x_true': hf['x_true'][:].astype(np.float32),
            'y': hf['y_ideal'][:].astype(np.float32),
            'mask': hf['mask'][:].astype(np.float32),
        }


def main():
    from solvers import SOLVERS, run_solver, CASSIOperator

    scene = load_scene(0)
    if scene is None:
        return

    x_true = scene['x_true']
    y = scene['y']
    mask = scene['mask']
    n_bands = x_true.shape[2]  # 28

    operator = CASSIOperator(mask, n_bands, step=2)

    print(f"Scene: {scene['name']}, x_true: {x_true.shape}, y: {y.shape}")
    print(f"Testing {len(SOLVERS)} solvers via run_solver()\n")

    results = {}
    errors = {}

    for solver_key, spec in SOLVERS.items():
        print(f"[{solver_key}] {spec['name']}...", end=" ", flush=True)
        cfg = {'device': 'cuda', 'x_true': x_true, 'mask': mask, 'n_bands': n_bands, 'step': 2}
        try:
            t0 = time.time()
            x_hat = run_solver(solver_key, y.copy(), operator, cfg)
            dt = time.time() - t0
            p = psnr(x_true, x_hat)
            results[solver_key] = (p, dt)
            print(f"{p:.2f} dB ({dt:.1f}s)")
        except Exception as e:
            errors[solver_key] = str(e)
            print(f"FAILED — {e}")

    # Summary
    print(f"\n{'='*75}")
    print(f"CASSI SOLVER VERIFICATION SUMMARY — {len(results)}/{len(SOLVERS)} passed")
    print(f"{'='*75}")
    print(f"{'Solver':<25} {'Name':<20} {'PSNR':>8} {'Time':>8} {'GPU':>5}")
    print("-" * 70)
    for sk, spec in SOLVERS.items():
        if sk in results:
            p, dt = results[sk]
            gpu = "GPU" if spec.get("gpu") else "CPU"
            print(f"{sk:<25} {spec['name']:<20} {p:>8.2f} {dt:>7.1f}s {gpu:>5}")
        elif sk in errors:
            print(f"{sk:<25} {spec['name']:<20} {'FAILED':>8} {'—':>8} {'—':>5}")

    if errors:
        print(f"\n--- ERRORS ({len(errors)}) ---")
        for sk, err in errors.items():
            print(f"  {sk}: {err[:120]}")


if __name__ == "__main__":
    main()
