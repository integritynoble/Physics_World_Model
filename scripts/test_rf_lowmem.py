#!/usr/bin/env python3
"""Test pretrained ReconFormer with memory-saving strategies."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import h5py
import time
from scipy.fft import ifft2, ifftshift, fft2, fftshift

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model architecture
sys.path.insert(0, os.path.join(ROOT, 'reference', 'mri', 'reconformer'))
from Recurrent_Transformer import ReconFormer

# Strategy 1: float16 + fewer iterations
for num_iter in [1, 2, 3]:
    print(f"\n=== Trying num_iter={num_iter}, float16 on CUDA ===", flush=True)
    try:
        model = ReconFormer(
            in_channels=2, out_channels=2,
            num_ch=(96, 48, 24), num_iter=num_iter,
            down_scales=(2, 1, 1.5), img_size=320,
            num_heads=(6, 6, 6), depths=(2, 1, 1),
            window_sizes=(8, 8, 8), mlp_ratio=2.,
            resi_connection='1conv',
            use_checkpoint=[False]*6,
        )

        # Load checkpoint (only matching keys since num_iter may differ)
        ckpt = os.path.join(ROOT, 'reference', 'mri', 'reconformer_checkpoint.pth')
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(state)
        model.eval()
        del state; gc.collect()

        # Move to CUDA float16
        model = model.half().cuda()
        torch.cuda.empty_cache()

        mem_before = torch.cuda.memory_allocated() / 1e6
        print(f"  Model on CUDA fp16: {mem_before:.1f} MB", flush=True)

        # Load real data
        with h5py.File(os.path.join(ROOT, 'datasets/benchmark/mri/standard/standard_mri_00.h5'), 'r') as hf:
            x_true = np.array(hf['x_true'], dtype=np.float32)
            y_raw = np.array(hf['y_ideal'], dtype=np.float32)
            mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

        kspace = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
        zf = ifft2(ifftshift(kspace))
        zf_2ch = np.stack([zf.real.astype(np.float32), zf.imag.astype(np.float32)], axis=-1)
        mag = np.sqrt(zf_2ch[..., 0]**2 + zf_2ch[..., 1]**2)
        std_val = float(mag.mean()) + 1e-11

        img_t = torch.from_numpy(zf_2ch).permute(2, 0, 1).unsqueeze(0).half().cuda() / std_val
        k0_2ch = np.stack([kspace.real.astype(np.float32), kspace.imag.astype(np.float32)], axis=-1)
        k0_t = torch.from_numpy(k0_2ch).permute(2, 0, 1).unsqueeze(0).half().cuda() / std_val
        mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).half().cuda()

        t0 = time.time()
        with torch.no_grad():
            out = model(img_t, k0_t, mask_t)
        elapsed = time.time() - t0

        out_np = out.squeeze().float().cpu().numpy()
        result = np.sqrt(out_np[0]**2 + out_np[1]**2).astype(np.float32) * std_val
        mse = np.mean((x_true - result)**2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
        print(f"  SUCCESS: PSNR={psnr:.2f} dB, time={elapsed:.1f}s", flush=True)
        print(f"  Peak GPU mem: {torch.cuda.max_memory_allocated()/1e6:.1f} MB", flush=True)

        # Clean up for next iteration
        del model, out, img_t, k0_t, mask_t
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()

# Strategy 2: float32 CPU with num_iter=1
print(f"\n=== Trying num_iter=1, float32 on CPU ===", flush=True)
try:
    model = ReconFormer(
        in_channels=2, out_channels=2,
        num_ch=(96, 48, 24), num_iter=1,
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

    with h5py.File(os.path.join(ROOT, 'datasets/benchmark/mri/standard/standard_mri_00.h5'), 'r') as hf:
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y_raw = np.array(hf['y_ideal'], dtype=np.float32)
        mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

    kspace = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)
    zf = ifft2(ifftshift(kspace))
    zf_2ch = np.stack([zf.real.astype(np.float32), zf.imag.astype(np.float32)], axis=-1)
    mag = np.sqrt(zf_2ch[..., 0]**2 + zf_2ch[..., 1]**2)
    std_val = float(mag.mean()) + 1e-11

    img_t = torch.from_numpy(zf_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_val
    k0_2ch = np.stack([kspace.real.astype(np.float32), kspace.imag.astype(np.float32)], axis=-1)
    k0_t = torch.from_numpy(k0_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_val
    mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()

    t0 = time.time()
    with torch.no_grad():
        out = model(img_t, k0_t, mask_t)
    elapsed = time.time() - t0

    out_np = out.squeeze().numpy()
    result = np.sqrt(out_np[0]**2 + out_np[1]**2).astype(np.float32) * std_val
    mse = np.mean((x_true - result)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    print(f"  SUCCESS: PSNR={psnr:.2f} dB, time={elapsed:.1f}s", flush=True)
except Exception as e:
    print(f"  FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
