#!/usr/bin/env python3
"""Step-by-step test of pretrained ReconFormer."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import h5py
import time
from scipy.fft import ifft2, ifftshift

print("Step 1: Import model...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reference', 'mri', 'reconformer'))
from Recurrent_Transformer import ReconFormer
print("  OK", flush=True)

print("Step 2: Create model...", flush=True)
model = ReconFormer(
    in_channels=2, out_channels=2,
    num_ch=(96, 48, 24), num_iter=5,
    down_scales=(2, 1, 1.5), img_size=320,
    num_heads=(6, 6, 6), depths=(2, 1, 1),
    window_sizes=(8, 8, 8), mlp_ratio=2.,
    resi_connection='1conv',
    use_checkpoint=[False]*6,
)
n = sum(p.numel() for p in model.parameters())
print(f"  OK: {n:,} params", flush=True)

print("Step 3: Load checkpoint...", flush=True)
ckpt = os.path.join(os.path.dirname(__file__), '..', 'reference', 'mri', 'reconformer_checkpoint.pth')
state = torch.load(ckpt, map_location='cpu', weights_only=False)
model.load_state_dict(state)
model.eval()
del state; gc.collect()
print("  OK", flush=True)

print("Step 4: Load data...", flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with h5py.File(os.path.join(ROOT, 'datasets/benchmark/mri/standard/standard_mri_00.h5'), 'r') as hf:
    x_true = np.array(hf['x_true'], dtype=np.float32)
    y_raw = np.array(hf['y_ideal'], dtype=np.float32)
    mask_data = np.array(hf['sampling_mask'], dtype=np.float32)
print("  OK", flush=True)

print("Step 5: Prepare inputs...", flush=True)
kspace = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
zf = ifft2(ifftshift(kspace))
zf_2ch = np.stack([zf.real.astype(np.float32), zf.imag.astype(np.float32)], axis=-1)
mag = np.sqrt(zf_2ch[..., 0]**2 + zf_2ch[..., 1]**2)
std_val = float(mag.mean()) + 1e-11

img_t = torch.from_numpy(zf_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_val
k0_2ch = np.stack([kspace.real.astype(np.float32), kspace.imag.astype(np.float32)], axis=-1)
k0_t = torch.from_numpy(k0_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_val
mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()
print(f"  OK: img={img_t.shape}, k0={k0_t.shape}, mask={mask_t.shape}, std={std_val:.6f}", flush=True)

print("Step 6: Forward pass...", flush=True)
t0 = time.time()
try:
    with torch.no_grad():
        out = model(img_t, k0_t, mask_t)
    elapsed = time.time() - t0
    print(f"  OK: {out.shape} ({elapsed:.1f}s)", flush=True)

    out_np = out.squeeze().numpy()
    result = np.sqrt(out_np[0]**2 + out_np[1]**2).astype(np.float32) * std_val
    mse = np.mean((x_true - result)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    print(f"  PSNR = {psnr:.2f} dB", flush=True)
except Exception as e:
    print(f"  FAIL: {e}", flush=True)
    import traceback
    traceback.print_exc()
