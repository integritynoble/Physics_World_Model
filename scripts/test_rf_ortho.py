#!/usr/bin/env python3
"""Test pretrained ReconFormer with proper ortho FFT convention."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import h5py
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'reference', 'mri', 'reconformer'))
from Recurrent_Transformer import ReconFormer

print("1. Create & load model (num_iter=5)...", flush=True)
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
print("  OK", flush=True)

print("2. Load data...", flush=True)
with h5py.File(os.path.join(ROOT, 'datasets/benchmark/mri/standard/standard_mri_00.h5'), 'r') as hf:
    x_true = np.array(hf['x_true'], dtype=np.float32)
    y_raw = np.array(hf['y_ideal'], dtype=np.float32)
    mask_data = np.array(hf['sampling_mask'], dtype=np.float32)
print(f"  x_true: {x_true.shape}, y_raw: {y_raw.shape}, mask: {mask_data.shape}", flush=True)

print("3. Prepare inputs using ortho convention...", flush=True)
# The model's DC layer uses transforms.fft2 with norm="ortho" and fftshift/ifftshift.
# Our kspace was generated with scipy backward convention.
# We need to convert to the convention the model expects.
#
# Model's transforms.fft2(x) does: ifftshift -> torch.fft.fft2(ortho) -> fftshift
# Model's transforms.ifft2(k) does: ifftshift -> torch.fft.ifft2(ortho) -> fftshift
#
# So the model expects k-space in centered, ortho-normalized format.
# Our kspace is in centered, backward-normalized format.
# Scale factor: k_ortho = k_backward / sqrt(H*W)

H, W = x_true.shape
scale = np.sqrt(H * W)  # = 320

kspace_backward = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)  # centered, backward norm

# Convert to ortho convention
kspace_ortho = kspace_backward / scale

# Zero-filled recon using ortho convention
# ortho_ifft2 of centered kspace: ifftshift -> ifft2(ortho) -> fftshift
from numpy.fft import ifft2, ifftshift, fftshift
zf = fftshift(ifft2(ifftshift(kspace_ortho), norm='ortho'))

# Wait, the model's transforms.ifft2 does fftshift(ifft2(ifftshift(k))).
# So if k is centered: ifftshift moves DC to corner, ifft2 ortho, fftshift moves DC to center.
# The result is a spatial-domain image with DC in center.
# This is NOT the standard spatial domain (where DC is at corner).
# The spatial image is "shifted" - this is fine as long as both image and k-space
# use the same convention consistently.

# Actually let me reconsider. For the ReconFormer:
# The training data comes from fastMRI:
# - kspace is provided as centered complex data
# - zero-filled: x_zf = ifft2(kspace) using their transforms.ifft2
# - which does: ifftshift(k) -> ifft2_ortho -> fftshift
# - So x_zf has a specific spatial arrangement

# Let me just use torch to match exactly what transforms does:
kspace_2ch = np.stack([kspace_ortho.real.astype(np.float32),
                        kspace_ortho.imag.astype(np.float32)], axis=-1)
k_torch = torch.from_numpy(kspace_2ch).unsqueeze(0)  # (1, H, W, 2)

# Use the model's own transforms to compute zero-filled image
import transforms as model_transforms
zf_torch = model_transforms.ifft2(k_torch)  # (1, H, W, 2)
print(f"  zf_torch: {zf_torch.shape}, range=[{zf_torch.min():.4f}, {zf_torch.max():.4f}]", flush=True)

# Normalize by mean magnitude
mag = model_transforms.complex_abs(zf_torch)  # (1, H, W)
std_val = float(mag.mean()) + 1e-11
print(f"  std_val={std_val:.6f}", flush=True)

# Prepare model inputs: (B, 2, H, W)
img_t = zf_torch.permute(0, 3, 1, 2).float() / std_val  # (1, 2, H, W)
k0_t = k_torch.permute(0, 3, 1, 2).float() / std_val     # (1, 2, H, W)
mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

print(f"  img_t: {img_t.shape}, range=[{img_t.min():.4f}, {img_t.max():.4f}]", flush=True)
print(f"  k0_t: {k0_t.shape}, range=[{k0_t.min():.4f}, {k0_t.max():.4f}]", flush=True)
print(f"  mask_t: {mask_t.shape}", flush=True)

print("4. Forward pass (CPU, num_iter=5)...", flush=True)
sys.stdout.flush()
t0 = time.time()
try:
    with torch.no_grad():
        out = model(img_t, k0_t, mask_t)
    elapsed = time.time() - t0
    print(f"  OK: {out.shape}, {elapsed:.1f}s", flush=True)

    # Output is (1, 2, H, W) in model's convention, need to un-normalize
    out_2ch = out.squeeze().permute(1, 2, 0).numpy() * std_val  # (H, W, 2)
    # Convert back to spatial domain image magnitude
    result = np.sqrt(out_2ch[..., 0]**2 + out_2ch[..., 1]**2).astype(np.float32)

    # But the model output might be in "shifted" spatial domain
    # Let's try both shifted and unshifted
    mse1 = np.mean((x_true - result)**2)
    psnr1 = 10 * np.log10(1.0 / mse1) if mse1 > 1e-10 else 100.0

    result_shifted = np.fft.fftshift(result)
    mse2 = np.mean((x_true - result_shifted)**2)
    psnr2 = 10 * np.log10(1.0 / mse2) if mse2 > 1e-10 else 100.0

    result_ishifted = np.fft.ifftshift(result)
    mse3 = np.mean((x_true - result_ishifted)**2)
    psnr3 = 10 * np.log10(1.0 / mse3) if mse3 > 1e-10 else 100.0

    print(f"  PSNR (direct)    = {psnr1:.2f} dB", flush=True)
    print(f"  PSNR (fftshift)  = {psnr2:.2f} dB", flush=True)
    print(f"  PSNR (ifftshift) = {psnr3:.2f} dB", flush=True)
    print(f"  result range: [{result.min():.4f}, {result.max():.4f}]", flush=True)
    print(f"  x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]", flush=True)

except Exception as e:
    print(f"  FAIL: {e}", flush=True)
    import traceback
    traceback.print_exc()
