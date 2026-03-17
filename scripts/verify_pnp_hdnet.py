#!/usr/bin/env python3
"""Verify PnP-HSICNN and HDNet against InverseNet reference PSNR.

Reference (InverseNet ECCV, KAIST 10 scenes, Scenario I):
  PnP-HSICNN: 25.12 dB (GAP + HSI-SDeCNN, 124 iters)
  HDNet:      34.66 dB (pretrained, mask-oblivious)
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
print("Step 1: Verify HSI-SDeCNN weights load correctly")
print("=" * 70)

import torch
from pwm_core.recon.hsi_sdecnn import HSI_SDeCNN, _find_weights

wp = _find_weights()
print(f"  Weights path: {wp}")

if wp:
    dev = torch.device('cpu')
    model = HSI_SDeCNN(in_nc=7, out_nc=1, nc=128, nb=15).to(dev)
    state = torch.load(str(wp), map_location=dev, weights_only=False)
    model.load_state_dict(state, strict=True)
    print("  Weights loaded with strict=True: OK")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Quick forward pass test
    x_test = torch.randn(1, 7, 256, 256)
    sigma_test = torch.full((1, 1, 1, 1), 10.0 / 255.0)
    with torch.no_grad():
        out_test = model(x_test, sigma_test)
    print(f"  Forward pass: input {x_test.shape} -> output {out_test.shape}")
else:
    print("  ERROR: No weights found!")

print()
print("=" * 70)
print("Step 2: PnP-HSICNN reconstruction on scene 00")
print("=" * 70)

data = load_sample(0)
if data is None:
    print("ERROR: No CASSI standard data found!")
    sys.exit(1)

y, x_true, mask = data['y'], data['x_true'], data['mask']
n_bands, step = data['n_bands'], data['step']
print(f"  Data: y={y.shape}, x_true={x_true.shape}, mask={mask.shape}, "
      f"n_bands={n_bands}, step={step}")

from pwm_core.recon.hsi_sdecnn import gap_sdecnn_cassi

t0 = time.time()
x_hat = gap_sdecnn_cassi(y, mask, n_bands, step=step, device='cpu')
dt = time.time() - t0
p = psnr(x_hat, x_true)
print(f"  PnP-HSICNN PSNR: {p:.2f} dB (ref: 25.12 dB)  [{dt:.1f}s]")
print(f"  Status: {'PASS' if p >= 23.0 else 'FAIL'}")

print()
print("=" * 70)
print("Step 3: HDNet reconstruction on scene 00")
print("=" * 70)

from pwm_core.recon.hdnet import hdnet_recon_cassi, _find_weights as hdnet_find_weights

hdnet_wp = hdnet_find_weights(None)
print(f"  HDNet weights: {hdnet_wp}")

if hdnet_wp:
    # Check checkpoint content
    ckpt = torch.load(hdnet_wp, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    else:
        sd = ckpt
    print(f"  Checkpoint keys: {len(sd)}")
    head_w = sd.get('head.0.weight')
    if head_w is not None:
        print(f"  head.0.weight shape: {head_w.shape} (expect [64, 28, 3, 3])")

    # Check for body structure
    body_keys = sorted([k for k in sd.keys() if k.startswith('body.')])
    print(f"  Body layer count: {len(body_keys)}")

    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, n_bands))
    t0 = time.time()
    x_hat = hdnet_recon_cassi(y, mask_3d, nC=n_bands, step=step, device='cpu')
    dt = time.time() - t0
    p = psnr(x_hat, x_true)
    print(f"  HDNet PSNR: {p:.2f} dB (ref: 34.66 dB)  [{dt:.1f}s]")
    print(f"  Status: {'PASS' if p >= 32.0 else 'NEEDS_INVESTIGATION'}")
else:
    print("  ERROR: No HDNet weights found!")

print()
print("=" * 70)
print("Step 4: PnP-HSICNN on all 10 scenes")
print("=" * 70)

psnrs = []
for idx in range(10):
    data = load_sample(idx)
    if data is None:
        continue
    t0 = time.time()
    x_hat = gap_sdecnn_cassi(data['y'], data['mask'], data['n_bands'],
                              step=data['step'], device='cpu')
    dt = time.time() - t0
    p = psnr(x_hat, data['x_true'])
    psnrs.append(p)
    print(f"  scene{idx+1:02d}: {p:.2f} dB  ({dt:.1f}s)")

if psnrs:
    mean_p = np.mean(psnrs)
    std_p = np.std(psnrs)
    print(f"\n  Mean: {mean_p:.2f} +/- {std_p:.2f} dB")
    print(f"  Reference: 25.12 dB (InverseNet)")
    status = "PASS" if mean_p >= 24.0 else "FAIL"
    print(f"  Status: {status}")
