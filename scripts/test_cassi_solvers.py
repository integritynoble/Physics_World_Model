#!/usr/bin/env python3
"""Test all CASSI solvers against rebuilt standard dataset."""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Force PWM5 imports
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
    mse = np.mean((x_hat.astype(np.float64) - x_true.astype(np.float64))**2)
    if mse < 1e-15:
        return float('inf')
    return float(10 * np.log10(x_true.max()**2 / mse))


def test_cassi():
    print("=" * 70)
    print("CASSI SOLVER TESTS (28 bands, step=2, correct physics)")
    print("=" * 70)

    cassi_dir = ROOT / "datasets" / "benchmark" / "cassi" / "standard"
    from algorithm_base.cassi.solvers import SOLVERS, run_solver, CASSIOperator

    results = {}

    # Test on 3 samples for speed
    for idx in range(3):
        h5_path = cassi_dir / f"standard_cassi_{idx:02d}.h5"
        if not h5_path.exists():
            print(f"  MISSING: {h5_path}")
            continue

        f = h5py.File(str(h5_path), 'r')
        y = np.array(f['y_ideal'], dtype=np.float32)
        x_true = np.array(f['x_true'], dtype=np.float32)
        mask = np.array(f['mask'], dtype=np.float32)
        wl = np.array(f['wavelength'], dtype=np.float32)
        n_bands = len(wl)
        step = int(f['H_params'].attrs.get('step', 2))
        f.close()

        # Create operator
        operator = CASSIOperator(mask, n_bands, step)

        print(f"\n  Sample {idx:02d}: y={y.shape}, x_true={x_true.shape}, "
              f"n_bands={n_bands}, step={step}")

        for key, info in SOLVERS.items():
            t0 = time.time()
            try:
                # Pass mask and params in cfg for solvers that need them
                cfg = {'mask': mask, 'n_bands': n_bands, 'step': step}
                x_hat = run_solver(key, y, operator, cfg)
                dt = time.time() - t0
                p = psnr(x_hat, x_true)
                if key not in results:
                    results[key] = []
                results[key].append(p)
                print(f"    {key:20s} ({info['name']:20s}): "
                      f"PSNR={p:6.2f} dB  ({dt:.1f}s)  "
                      f"shape={x_hat.shape}")
            except Exception as e:
                dt = time.time() - t0
                print(f"    {key:20s} ({info['name']:20s}): "
                      f"ERROR ({dt:.1f}s) — {str(e)[:100]}")
                if key not in results:
                    results[key] = []
                results[key].append(-1)

    print("\n" + "=" * 70)
    print("CASSI SUMMARY")
    print("=" * 70)

    ref_psnr = {
        "traditional_cpu": 24.4,  # GAP-TV reference
        "best_quality": 26.2,     # GAP-TV guided
        "famous_dl": 26.2,        # GAP-TV fast
        "small_gpu": 26.2,        # GAP-TV small
        "mst_l": 34.9,            # MST-L reference
        "hdnet": 35.0,            # HDNet reference
        "hsi_sdecnn": None,       # No direct reference
    }

    for key, vals in results.items():
        good = [v for v in vals if v > 0]
        ref = ref_psnr.get(key)
        if good:
            avg = np.mean(good)
            ref_str = f" (ref: {ref:.1f} dB)" if ref else ""
            status = "DONE" if ref and avg >= ref - 3 else "gap"
            print(f"  {key:20s}: avg {avg:.1f} dB{ref_str} — {status}")
        else:
            print(f"  {key:20s}: ALL FAILED")

    return results


if __name__ == "__main__":
    test_cassi()
