#!/usr/bin/env python3
"""Test pretrained ReconFormer at different num_iter to find segfault threshold."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import h5py
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'reference', 'mri', 'reconformer'))
from Recurrent_Transformer import ReconFormer

# First test: num_iter=5 with RANDOM data
print("=== Random data, num_iter=5, 320x320 ===", flush=True)
model = ReconFormer(
    in_channels=2, out_channels=2,
    num_ch=(96, 48, 24), num_iter=5,
    down_scales=(2, 1, 1.5), img_size=320,
    num_heads=(6, 6, 6), depths=(2, 1, 1),
    window_sizes=(8, 8, 8), mlp_ratio=2.,
    resi_connection='1conv',
    use_checkpoint=[False]*6,
)
ckpt = os.path.join(ROOT, 'reference', 'mri', 'reconformer_checkpoint.pth')
state = torch.load(ckpt, map_location='cpu', weights_only=False)
model.load_state_dict(state)
model.eval()
del state; gc.collect()

x = torch.randn(1, 2, 320, 320)
k0 = torch.randn(1, 2, 320, 320)
mask = torch.ones(1, 1, 320, 320)
t0 = time.time()
with torch.no_grad():
    out = model(x, k0, mask)
print(f"  OK: {out.shape}, {time.time()-t0:.1f}s", flush=True)
del model, x, k0, mask, out
gc.collect()

# Second test: num_iter=5 with REAL data, proper ortho convention
print("\n=== Real data, num_iter=5, ortho convention ===", flush=True)
model = ReconFormer(
    in_channels=2, out_channels=2,
    num_ch=(96, 48, 24), num_iter=5,
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
print("  Model loaded", flush=True)

with h5py.File(os.path.join(ROOT, 'datasets/benchmark/mri/standard/standard_mri_00.h5'), 'r') as hf:
    x_true = np.array(hf['x_true'], dtype=np.float32)
    y_raw = np.array(hf['y_ideal'], dtype=np.float32)
    mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

H, W = x_true.shape
scale = np.sqrt(H * W)
kspace = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64) / scale

import transforms as model_transforms
kspace_2ch = np.stack([kspace.real.astype(np.float32), kspace.imag.astype(np.float32)], axis=-1)
k_torch = torch.from_numpy(kspace_2ch).unsqueeze(0)
zf_torch = model_transforms.ifft2(k_torch)
mag = model_transforms.complex_abs(zf_torch)
std_val = float(mag.mean()) + 1e-11

img_t = zf_torch.permute(0, 3, 1, 2).float() / std_val
k0_t = k_torch.permute(0, 3, 1, 2).float() / std_val
mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()
print(f"  Data ready: img={img_t.shape}, k0_range=[{k0_t.min():.2f},{k0_t.max():.2f}]", flush=True)

t0 = time.time()
sys.stdout.flush()
with torch.no_grad():
    out = model(img_t, k0_t, mask_t)
elapsed = time.time() - t0
print(f"  OK: {out.shape}, {elapsed:.1f}s", flush=True)

out_2ch = out.squeeze().permute(1, 2, 0).numpy() * std_val
result = np.sqrt(out_2ch[..., 0]**2 + out_2ch[..., 1]**2).astype(np.float32)
mse = np.mean((x_true - result)**2)
psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
print(f"  PSNR = {psnr:.2f} dB", flush=True)
