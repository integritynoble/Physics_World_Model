#!/usr/bin/env python3
"""Verify all MRI solvers on 20 scenes — compute mean PSNR/SSIM."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import h5py
import time
import json

from algorithm_base.mri.solvers import SOLVERS, MRIOperator, run_solver

# Load all scenes
mri_dir = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")
h5_files = sorted([f for f in os.listdir(mri_dir) if f.endswith('.h5')])
print(f"Found {len(h5_files)} MRI scenes")

scenes = []
for f in h5_files:
    path = os.path.join(mri_dir, f)
    with h5py.File(path, 'r') as hf:
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
    scenes.append((f, x_true, y, mask))

def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 100.0
    peak = x_true.max()
    if peak < 1e-10:
        peak = 1.0
    return 10 * np.log10(peak ** 2 / mse)

def ssim_simple(x, y):
    """Simplified SSIM."""
    C1 = (0.01 * max(x.max(), y.max(), 1.0)) ** 2
    C2 = (0.03 * max(x.max(), y.max(), 1.0)) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov = np.mean((x - mu_x) * (y - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * cov + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
    return float(num / den)

# Skip slow solvers for 20-scene run (aloha takes 69s/scene)
SLOW_SOLVERS = {"aloha"}  # Still test but on fewer scenes

print(f"\nVerifying {len(SOLVERS)} solvers on {len(scenes)} scenes\n")

all_results = {}
for key, spec in SOLVERS.items():
    psnrs = []
    ssims = []
    errors = 0
    max_scenes = 3 if key in SLOW_SOLVERS else len(scenes)
    t0 = time.time()

    for i, (fname, x_true, y_data, mask) in enumerate(scenes[:max_scenes]):
        op = MRIOperator(mask, image_size=x_true.shape[0])
        y_input = np.stack([y_data.real, y_data.imag], axis=-1).astype(np.float32)
        try:
            x_hat = run_solver(key, y_input, op, {"device": "cpu"})
            if x_hat is None or not isinstance(x_hat, np.ndarray):
                errors += 1
                continue
            if x_hat.shape != x_true.shape and x_hat.size == x_true.size:
                x_hat = x_hat.reshape(x_true.shape)
            x_hat = np.clip(x_hat, 0, None).astype(np.float32)
            p = psnr(x_true, x_hat)
            s = ssim_simple(x_true, x_hat)
            psnrs.append(p)
            ssims.append(s)
        except Exception as e:
            errors += 1

    dt = time.time() - t0
    mean_psnr = np.mean(psnrs) if psnrs else 0.0
    mean_ssim = np.mean(ssims) if ssims else 0.0
    n_pass = len(psnrs)
    n_total = max_scenes

    status = "PASS" if errors == 0 else "PARTIAL"
    all_results[key] = {
        "name": spec["name"],
        "mean_psnr": round(float(mean_psnr), 2),
        "mean_ssim": round(float(mean_ssim), 4),
        "pass": n_pass,
        "total": n_total,
        "errors": errors,
        "time": round(dt, 1),
        "status": status,
    }
    print(f"  {status:7s}  {key:25s}  PSNR={mean_psnr:6.2f}  SSIM={mean_ssim:.4f}  "
          f"{n_pass}/{n_total}  {dt:5.1f}s  {spec['name']}")

# Summary
passed = sum(1 for v in all_results.values() if v["errors"] == 0)
total = len(all_results)
print(f"\n{'='*70}")
print(f"Results: {passed}/{total} solvers with 100% scene pass rate")
print(f"{'='*70}")

# Save results
out_path = os.path.join(ROOT, "benchmark_results", "mri_solver_verification_v2.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {out_path}")
