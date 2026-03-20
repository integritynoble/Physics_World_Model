#!/usr/bin/env python3
"""Test pretrained ReconFormer - check spatial shift in output."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import h5py
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'reference', 'mri', 'reconformer'))
from Recurrent_Transformer import ReconFormer
import transforms as model_transforms

ckpt = os.path.join(ROOT, 'reference', 'mri', 'reconformer_checkpoint.pth')

print("1. Load model (num_iter=2)...", flush=True)
model = ReconFormer(
    in_channels=2, out_channels=2,
    num_ch=(96, 48, 24), num_iter=2,
    down_scales=(2, 1, 1.5), img_size=320,
    num_heads=(6, 6, 6), depths=(2, 1, 1),
    window_sizes=(8, 8, 8), mlp_ratio=2.,
    resi_connection='1conv',
    use_checkpoint=[False]*6,
)
state = torch.load(ckpt, map_location='cpu', weights_only=False)
model.load_state_dict(state)
model.eval()
del state; gc.collect()
print("  OK", flush=True)

print("2. Load data...", flush=True)
with h5py.File(os.path.join(ROOT, 'datasets/benchmark/mri/standard/standard_mri_00.h5'), 'r') as hf:
    x_true = np.array(hf['x_true'], dtype=np.float32)
    y_raw = np.array(hf['y_ideal'], dtype=np.float32)
    mask_data = np.array(hf['sampling_mask'], dtype=np.float32)
print(f"  shapes: x_true={x_true.shape}, y_raw={y_raw.shape}", flush=True)

H, W = x_true.shape
scale = np.sqrt(H * W)

# Prepare with ortho convention using model's own transforms
kspace_backward = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
kspace_ortho = kspace_backward / scale
kspace_2ch = np.stack([kspace_ortho.real.astype(np.float32),
                       kspace_ortho.imag.astype(np.float32)], axis=-1)
k_torch = torch.from_numpy(kspace_2ch).unsqueeze(0)
zf_torch = model_transforms.ifft2(k_torch)

# Check: does the model's ifft2 output match x_true when fftshifted?
zf_mag = model_transforms.complex_abs(zf_torch).squeeze().numpy()
zf_mag_shifted = np.fft.fftshift(zf_mag)
print(f"  zf_mag range: [{zf_mag.min():.4f}, {zf_mag.max():.4f}]", flush=True)
print(f"  x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]", flush=True)

# PSNR of zero-filled (no model) with different shifts
for shift_name, shifted in [("none", zf_mag), ("fftshift", zf_mag_shifted), ("ifftshift", np.fft.ifftshift(zf_mag))]:
    mse = np.mean((x_true - shifted)**2)
    p = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100
    print(f"  ZF PSNR ({shift_name}): {p:.2f} dB", flush=True)

# Normalize
mag = model_transforms.complex_abs(zf_torch)
std_val = float(mag.mean()) + 1e-11

img_t = zf_torch.permute(0, 3, 1, 2).float() / std_val
k0_t = k_torch.permute(0, 3, 1, 2).float() / std_val
mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()

print("3. Forward pass...", flush=True)
sys.stdout.flush()
t0 = time.time()
with torch.no_grad():
    out = model(img_t, k0_t, mask_t)
elapsed = time.time() - t0
print(f"  OK: {out.shape}, {elapsed:.1f}s", flush=True)

out_2ch = out.squeeze().permute(1, 2, 0).numpy() * std_val
result = np.sqrt(out_2ch[..., 0]**2 + out_2ch[..., 1]**2).astype(np.float32)

for shift_name, shifted in [("none", result), ("fftshift", np.fft.fftshift(result)), ("ifftshift", np.fft.ifftshift(result))]:
    mse = np.mean((x_true - shifted)**2)
    p = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100
    print(f"  Model PSNR ({shift_name}): {p:.2f} dB", flush=True)

# Also check if x_true needs to be in shifted domain
x_true_shifted = np.fft.fftshift(x_true)
for shift_name, img in [("none vs shifted_xtrue", result)]:
    mse = np.mean((x_true_shifted - img)**2)
    p = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100
    print(f"  Model PSNR (result vs fftshift(x_true)): {p:.2f} dB", flush=True)

# Pixel-level comparison: are the images clearly shifted?
print(f"\n  Top-left 3x3 of result: {result[:3,:3]}", flush=True)
print(f"  Top-left 3x3 of x_true: {x_true[:3,:3]}", flush=True)
print(f"  Center 3x3 of result: {result[158:161,158:161]}", flush=True)
print(f"  Center 3x3 of x_true: {x_true[158:161,158:161]}", flush=True)
