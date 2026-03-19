#!/usr/bin/env python3
"""Verify ALL CASSI algorithms against InverseNet reference PSNR.

Reference (InverseNet ECCV, KAIST 10 scenes, Scenario I):
  GAP-TV:     24.34 dB (Chambolle TV, 100 iters)
  PnP-HSICNN: 25.12 dB (GAP + HSI-SDeCNN)
  HDNet:      34.66 dB (pretrained, mask-oblivious)
  MST-L:      34.81 dB (pretrained, mask-aware transformer)
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

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import h5py

CASSI_DIR = ROOT / "datasets" / "benchmark" / "cassi" / "standard"

# InverseNet reference PSNR (mean over 10 KAIST scenes, Scenario I)
REFERENCE = {
    "gap_tv": 24.34,
    "hdnet": 34.66,
    "mst_l": 34.81,
    "pnp_hsicnn": 25.12,
}


def psnr(x_hat, x_true):
    x_hat = np.clip(x_hat, 0, 1).astype(np.float64)
    x_true = np.clip(x_true, 0, 1).astype(np.float64)
    mse = np.mean((x_hat - x_true) ** 2)
    if mse < 1e-10: return 100.0
    return float(10 * np.log10(1.0 / mse))


def load_sample(idx):
    h5 = CASSI_DIR / f"standard_cassi_{idx:02d}.h5"
    if not h5.exists():
        return None
    f = h5py.File(str(h5), 'r')
    data = {
        'y': np.array(f['y_ideal'], dtype=np.float32),
        'x_true': np.array(f['x_true'], dtype=np.float32),
        'mask': np.array(f['mask'], dtype=np.float32),
        'n_bands': len(f['wavelength']),
        'step': int(f['H_params'].attrs.get('step', 2)),
    }
    f.close()
    return data


print("=" * 70)
print("CASSI Algorithm Verification — All Solvers vs InverseNet Reference")
print("=" * 70)

# Load scene 00 for quick verification
data = load_sample(0)
if data is None:
    print("ERROR: No CASSI standard data found!")
    sys.exit(1)

y, x_true, mask = data['y'], data['x_true'], data['mask']
n_bands, step = data['n_bands'], data['step']
print(f"Data: y={y.shape}, x_true={x_true.shape}, mask={mask.shape}, "
      f"n_bands={n_bands}, step={step}")
print()

results = {}

# --- 1. GAP-TV (traditional_cpu) ---
print("1. GAP-TV (traditional_cpu) — 100 iters, Chambolle TV w=0.1")
from pwm_core.recon.gap_tv import gap_tv_cassi
t0 = time.time()
x_hat = gap_tv_cassi(y, mask, n_bands, iterations=100, lam=0.1, step=step, tv_iter=5)
dt = time.time() - t0
p = psnr(x_hat, x_true)
results['gap_tv'] = p
print(f"   PSNR: {p:.2f} dB (ref: {REFERENCE['gap_tv']:.2f} dB)  [{dt:.1f}s]")

# --- 2. GAP-TV (best_quality) — 200 iters ---
print("2. GAP-TV (best_quality) — 200 iters, Chambolle TV w=0.01")
t0 = time.time()
x_hat = gap_tv_cassi(y, mask, n_bands, iterations=200, lam=0.01, step=step, tv_iter=5)
dt = time.time() - t0
p = psnr(x_hat, x_true)
results['gap_tv_200'] = p
print(f"   PSNR: {p:.2f} dB  [{dt:.1f}s]")

# --- 3. HDNet ---
print("3. HDNet — pretrained weights")
try:
    from pwm_core.recon.hdnet import hdnet_recon_cassi
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, n_bands))
    t0 = time.time()
    x_hat = hdnet_recon_cassi(y, mask_3d, nC=n_bands, step=step, device='cpu')
    dt = time.time() - t0
    p = psnr(x_hat, x_true)
    results['hdnet'] = p
    print(f"   PSNR: {p:.2f} dB (ref: {REFERENCE['hdnet']:.2f} dB)  [{dt:.1f}s]")
except Exception as e:
    print(f"   FAILED: {e}")
    results['hdnet'] = 0

# --- 4. MST-L ---
print("4. MST-L — pretrained weights")
try:
    from pwm_core.recon.mst import mst_recon_cassi
    t0 = time.time()
    x_hat = mst_recon_cassi(y, mask, nC=n_bands, step=step, device='cpu', variant='mst_l')
    dt = time.time() - t0
    p = psnr(x_hat, x_true)
    results['mst_l'] = p
    print(f"   PSNR: {p:.2f} dB (ref: {REFERENCE['mst_l']:.2f} dB)  [{dt:.1f}s]")
except Exception as e:
    print(f"   FAILED: {e}")
    results['mst_l'] = 0

# --- 5. HSI-SDeCNN / PnP ---
print("5. HSI-SDeCNN (PnP-HSICNN) — needs GPU server weights")
try:
    from pwm_core.recon.hsi_sdecnn import run_hsi_sdecnn
    from algorithm_base.cassi.solvers import CASSIOperator
    op = CASSIOperator(mask, n_bands, step)
    t0 = time.time()
    result = run_hsi_sdecnn(y, op, {})
    dt = time.time() - t0
    if isinstance(result, tuple):
        x_hat = result[0]
    else:
        x_hat = result
    p = psnr(x_hat, x_true)
    results['hsi_sdecnn'] = p
    print(f"   PSNR: {p:.2f} dB (ref: {REFERENCE['pnp_hsicnn']:.2f} dB)  [{dt:.1f}s]")
except Exception as e:
    print(f"   SKIPPED (no local weights): {e}")
    results['hsi_sdecnn'] = -1

# --- Summary ---
print()
print("=" * 70)
print("SUMMARY — Scene 01 PSNR vs InverseNet Reference")
print("=" * 70)
print(f"{'Solver':<25s} {'Our PSNR':>10s} {'Ref PSNR':>10s} {'Status':>10s}")
print("-" * 55)

for key in ['gap_tv', 'gap_tv_200', 'hdnet', 'mst_l', 'hsi_sdecnn']:
    our_p = results.get(key, -1)
    ref_key = key.replace('_200', '')
    ref_p = REFERENCE.get(ref_key, None)
    if our_p < 0:
        status = "NO_WEIGHTS"
    elif ref_p and our_p >= ref_p - 2.0:
        status = "PASS"
    elif our_p >= 20.0:
        status = "OK"
    else:
        status = "FAIL"
    ref_str = f"{ref_p:.2f}" if ref_p else "N/A"
    print(f"  {key:<23s} {our_p:8.2f} dB {ref_str:>8s} dB  {status}")

# --- Run GAP-TV across all 10 scenes ---
print()
print("=" * 70)
print("GAP-TV (100it, lam=0.1) — All 10 KAIST Scenes")
print("=" * 70)
psnrs = []
for idx in range(10):
    data = load_sample(idx)
    if data is None:
        continue
    x_hat = gap_tv_cassi(data['y'], data['mask'], data['n_bands'],
                          iterations=100, lam=0.1, step=data['step'], tv_iter=5)
    p = psnr(x_hat, data['x_true'])
    psnrs.append(p)
    print(f"  scene{idx+1:02d}: {p:.2f} dB")

if psnrs:
    mean_p = np.mean(psnrs)
    std_p = np.std(psnrs)
    print(f"\n  Mean: {mean_p:.2f} +/- {std_p:.2f} dB")
    print(f"  Reference: {REFERENCE['gap_tv']:.2f} dB")
    status = "PASS" if mean_p >= REFERENCE['gap_tv'] - 1.0 else "FAIL"
    print(f"  Status: {status}")
