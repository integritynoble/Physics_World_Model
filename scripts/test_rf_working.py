#!/usr/bin/env python3
"""Test pretrained ReconFormer with correct conventions - minimal memory."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'reference', 'mri', 'reconformer'))
from Recurrent_Transformer import ReconFormer
import transforms as model_transforms

ckpt = os.path.join(ROOT, 'reference', 'mri', 'reconformer_checkpoint.pth')
DATA_DIR = os.path.join(ROOT, 'datasets/benchmark/mri/standard')
h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])

# Test multiple scenes with num_iter=1 (reliable on this machine)
for num_iter in [1, 2]:
    print(f"\n{'='*50}", flush=True)
    print(f"Testing num_iter={num_iter}", flush=True)
    print(f"{'='*50}", flush=True)

    model = ReconFormer(
        in_channels=2, out_channels=2,
        num_ch=(96, 48, 24), num_iter=num_iter,
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

    psnrs = []
    for scene_idx in range(min(5, len(h5_files))):
        fname = h5_files[scene_idx]
        with h5py.File(os.path.join(DATA_DIR, fname), 'r') as hf:
            x_true = np.array(hf['x_true'], dtype=np.float32)
            y_raw = np.array(hf['y_ideal'], dtype=np.float32)
            mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

        H, W = x_true.shape
        scale = np.sqrt(H * W)

        # Ortho convention
        kspace = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64) / scale
        kspace_2ch = np.stack([kspace.real.astype(np.float32),
                               kspace.imag.astype(np.float32)], axis=-1)
        k_torch = torch.from_numpy(kspace_2ch).unsqueeze(0)  # (1, H, W, 2)
        zf_torch = model_transforms.ifft2(k_torch)  # (1, H, W, 2)

        mag = model_transforms.complex_abs(zf_torch)
        std_val = float(mag.mean()) + 1e-11

        img_t = zf_torch.permute(0, 3, 1, 2).float() / std_val  # (1, 2, H, W)
        k0_t = k_torch.permute(0, 3, 1, 2).float() / std_val
        mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            out = model(img_t, k0_t, mask_t)

        # Un-normalize and convert to magnitude
        out_2ch = out.squeeze().permute(1, 2, 0).numpy() * std_val  # (H, W, 2)
        result = np.sqrt(out_2ch[..., 0]**2 + out_2ch[..., 1]**2).astype(np.float32)

        # Apply fftshift to convert from model's shifted convention to standard
        result = np.fft.fftshift(result)

        mse = np.mean((x_true - result)**2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
        psnrs.append(psnr)
        print(f"  Scene {scene_idx}: PSNR={psnr:.2f} dB", flush=True)

        del out, img_t, k0_t, mask_t, k_torch, zf_torch
        gc.collect()

    avg = np.mean(psnrs)
    print(f"  Average (5 scenes): {avg:.2f} dB", flush=True)

    del model
    gc.collect()
