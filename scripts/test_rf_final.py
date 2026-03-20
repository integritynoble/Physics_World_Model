#!/usr/bin/env python3
"""Test pretrained ReconFormer with num_iter=2 and ortho convention."""
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

print("1. Create & load model (num_iter=2)...", flush=True)
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

DATA_DIR = os.path.join(ROOT, 'datasets/benchmark/mri/standard')
h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])

results = []
for scene_idx, fname in enumerate(h5_files[:5]):  # Test 5 scenes
    print(f"\n--- Scene {scene_idx}: {fname} ---", flush=True)
    with h5py.File(os.path.join(DATA_DIR, fname), 'r') as hf:
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y_raw = np.array(hf['y_ideal'], dtype=np.float32)
        mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

    H, W = x_true.shape
    scale = np.sqrt(H * W)

    # ---- Method A: Ortho convention ----
    kspace_backward = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
    kspace_ortho = kspace_backward / scale
    kspace_2ch = np.stack([kspace_ortho.real.astype(np.float32),
                           kspace_ortho.imag.astype(np.float32)], axis=-1)
    k_torch = torch.from_numpy(kspace_2ch).unsqueeze(0)
    zf_torch = model_transforms.ifft2(k_torch)
    mag = model_transforms.complex_abs(zf_torch)
    std_val = float(mag.mean()) + 1e-11

    img_t = zf_torch.permute(0, 3, 1, 2).float() / std_val
    k0_t = k_torch.permute(0, 3, 1, 2).float() / std_val
    mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()

    t0 = time.time()
    with torch.no_grad():
        out = model(img_t, k0_t, mask_t)
    elapsed = time.time() - t0

    out_2ch = out.squeeze().permute(1, 2, 0).numpy() * std_val
    result_a = np.sqrt(out_2ch[..., 0]**2 + out_2ch[..., 1]**2).astype(np.float32)
    mse_a = np.mean((x_true - result_a)**2)
    psnr_a = 10 * np.log10(1.0 / mse_a) if mse_a > 1e-10 else 100.0

    # ---- Method B: Backward convention (original way, just scale k0 differently) ----
    from scipy.fft import ifft2, ifftshift
    kspace_b = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
    zf_b = ifft2(ifftshift(kspace_b))
    zf_2ch = np.stack([zf_b.real.astype(np.float32), zf_b.imag.astype(np.float32)], axis=-1)
    mag_b = np.sqrt(zf_2ch[..., 0]**2 + zf_2ch[..., 1]**2)
    std_b = float(mag_b.mean()) + 1e-11

    img_b = torch.from_numpy(zf_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_b
    # Scale k0 to ortho convention before normalizing
    kspace_b_ortho = kspace_b / scale
    k0_2ch = np.stack([kspace_b_ortho.real.astype(np.float32),
                        kspace_b_ortho.imag.astype(np.float32)], axis=-1)
    k0_b = torch.from_numpy(k0_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_b
    mask_b = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        out_b = model(img_b, k0_b, mask_b)

    out_np = out_b.squeeze().numpy()
    result_b = np.sqrt(out_np[0]**2 + out_np[1]**2).astype(np.float32) * std_b
    mse_b = np.mean((x_true - result_b)**2)
    psnr_b = 10 * np.log10(1.0 / mse_b) if mse_b > 1e-10 else 100.0

    # ---- Method C: Original backward convention (both img and k0 in backward) ----
    k0_2ch_c = np.stack([kspace_b.real.astype(np.float32),
                          kspace_b.imag.astype(np.float32)], axis=-1)
    k0_c = torch.from_numpy(k0_2ch_c).permute(2, 0, 1).unsqueeze(0).float() / std_b

    with torch.no_grad():
        out_c = model(img_b, k0_c, mask_b)

    out_np_c = out_c.squeeze().numpy()
    result_c = np.sqrt(out_np_c[0]**2 + out_np_c[1]**2).astype(np.float32) * std_b
    mse_c = np.mean((x_true - result_c)**2)
    psnr_c = 10 * np.log10(1.0 / mse_c) if mse_c > 1e-10 else 100.0

    print(f"  Method A (full ortho):      PSNR={psnr_a:.2f} dB", flush=True)
    print(f"  Method B (img=backward, k0=ortho): PSNR={psnr_b:.2f} dB", flush=True)
    print(f"  Method C (all backward):    PSNR={psnr_c:.2f} dB", flush=True)
    print(f"  ({elapsed:.1f}s)", flush=True)
    results.append((psnr_a, psnr_b, psnr_c))

    del out, out_b, out_c
    gc.collect()

print("\n=== Summary ===", flush=True)
for i, (a, b, c) in enumerate(results):
    print(f"  Scene {i}: A={a:.2f}, B={b:.2f}, C={c:.2f}", flush=True)
avg_a = np.mean([r[0] for r in results])
avg_b = np.mean([r[1] for r in results])
avg_c = np.mean([r[2] for r in results])
print(f"  Average: A={avg_a:.2f}, B={avg_b:.2f}, C={avg_c:.2f}", flush=True)
