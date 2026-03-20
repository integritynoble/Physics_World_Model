#!/usr/bin/env python3
"""Comprehensive verification of ALL solvers for CASSI, CT, and MRI."""
from __future__ import annotations
import sys, time, traceback, json
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
    x_h = np.asarray(x_hat, dtype=np.float64)
    x_t = np.asarray(x_true, dtype=np.float64)
    if np.iscomplexobj(x_h):
        x_h = np.abs(x_h)
    mse = np.mean((x_h - x_t) ** 2)
    if mse < 1e-15:
        return float('inf')
    return float(10 * np.log10(float(x_t.max()) ** 2 / mse))


def test_modality(modality, n_samples=3):
    print(f"\n{'=' * 70}")
    print(f"  {modality.upper()} — Comprehensive Solver Verification")
    print(f"{'=' * 70}")

    data_dir = ROOT / "datasets" / "benchmark" / modality / "standard"

    # Import the modality's solver module
    mod = __import__(f"algorithm_base.{modality}.solvers", fromlist=["SOLVERS", "run_solver"])
    SOLVERS = mod.SOLVERS
    run_solver = mod.run_solver

    print(f"  Registered solvers: {list(SOLVERS.keys())}")

    results = {}

    for idx in range(n_samples):
        h5_path = data_dir / f"standard_{modality}_{idx:02d}.h5"
        if not h5_path.exists():
            print(f"  SKIP sample {idx:02d} — file not found")
            continue

        f = h5py.File(str(h5_path), 'r')
        y = np.array(f['y_ideal'], dtype=np.float32)
        x_true = np.array(f['x_true'], dtype=np.float32)

        # Load modality-specific data
        extra = {}
        if modality == "cassi":
            extra['mask'] = np.array(f['mask'], dtype=np.float32)
            extra['n_bands'] = len(f['wavelength'])
            extra['step'] = int(f['H_params'].attrs.get('step', 2))
            # Create operator
            from algorithm_base.cassi.solvers import CASSIOperator
            operator = CASSIOperator(extra['mask'], extra['n_bands'], extra['step'])
        elif modality == "ct":
            operator = None  # Auto-created
            extra['output_size'] = x_true.shape[0]
        elif modality == "mri":
            if 'sampling_mask' in f:
                extra['sampling_mask'] = np.array(f['sampling_mask'], dtype=np.float32)
            operator = None  # Auto-created
        else:
            operator = None
        f.close()

        print(f"\n  Sample {idx:02d}: y={y.shape}, x_true={x_true.shape}")

        for key in SOLVERS:
            info = SOLVERS[key]
            t0 = time.time()
            try:
                cfg = dict(extra)
                if modality == "cassi":
                    x_hat = run_solver(key, y, operator, cfg)
                elif modality == "ct":
                    x_hat = run_solver(key, y, None, cfg)
                elif modality == "mri":
                    x_hat = run_solver(key, y, None, cfg)
                else:
                    x_hat = run_solver(key, y, None, cfg)

                dt = time.time() - t0
                p = psnr(x_hat, x_true)
                status = "OK"
                shape_str = str(x_hat.shape)
            except Exception as e:
                dt = time.time() - t0
                p = -999
                status = f"ERROR: {str(e)[:80]}"
                shape_str = "—"

            if key not in results:
                results[key] = {"name": info["name"], "psnrs": [], "errors": []}
            if p > -900:
                results[key]["psnrs"].append(p)
            else:
                results[key]["errors"].append(status)

            mark = "OK" if p > 0 else ("WARN" if p > -900 else "FAIL")
            psnr_str = f"{p:7.2f} dB" if p > -900 else "  FAIL  "
            print(f"    {mark} {key:20s} ({info['name']:20s}): {psnr_str}  ({dt:.1f}s)  {shape_str if p > -900 else status}")

    return results


def print_summary(all_results):
    print(f"\n{'=' * 70}")
    print(f"  FINAL VERIFICATION SUMMARY")
    print(f"{'=' * 70}")

    total_ok = 0
    total_fail = 0
    total_solvers = 0

    for modality, results in all_results.items():
        print(f"\n  {modality.upper()}:")
        for key, data in results.items():
            total_solvers += 1
            psnrs = data["psnrs"]
            errors = data["errors"]
            if psnrs:
                avg_p = np.mean(psnrs)
                ok_count = len(psnrs)
                total_ok += 1
                mark = "OK" if avg_p > 5 else "WARN"
                print(f"    {mark} {key:20s} ({data['name']:20s}): avg {avg_p:6.1f} dB  ({ok_count} samples OK)")
            else:
                total_fail += 1
                print(f"    FAIL {key:20s} ({data['name']:20s}): ALL FAILED -- {errors[0] if errors else 'unknown'}")

    print(f"\n  Total: {total_ok}/{total_solvers} solvers working, {total_fail} failed")
    return total_ok, total_fail


if __name__ == "__main__":
    all_results = {}
    for modality in ["cassi", "ct", "mri"]:
        all_results[modality] = test_modality(modality, n_samples=2)

    total_ok, total_fail = print_summary(all_results)

    # Save results
    out = {}
    for modality, results in all_results.items():
        out[modality] = {}
        for key, data in results.items():
            out[modality][key] = {
                "name": data["name"],
                "avg_psnr": float(np.mean(data["psnrs"])) if data["psnrs"] else None,
                "n_ok": len(data["psnrs"]),
                "errors": data["errors"],
            }
    with open(ROOT / "benchmark_results" / "solver_verification_final.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to benchmark_results/solver_verification_final.json")
