#!/usr/bin/env python3
"""Test MiJUN inference on CASSI benchmark data."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core', 'pwm_core', 'recon'))

import numpy as np
import h5py
import time
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 100.0
    peak = np.max(x_true)
    if peak < 1e-10:
        peak = 1.0
    return 10.0 * np.log10(peak ** 2 / mse)

# Load CASSI benchmark data
cassi_dir = ROOT / "datasets" / "benchmark" / "cassi" / "standard"
if not cassi_dir.exists():
    cassi_dir = ROOT / "datasets" / "benchmark" / "sd_cassi" / "standard"

h5_files = sorted(cassi_dir.glob("*.h5")) if cassi_dir.exists() else []
print(f"Found {len(h5_files)} CASSI h5 files in {cassi_dir}")

if not h5_files:
    print("No CASSI data found. Testing with synthetic data.")
    # Create synthetic test data
    H, W, nC = 256, 256, 28
    x_true = np.random.rand(H, W, nC).astype(np.float32)
    mask_2d = (np.random.rand(H, W) > 0.5).astype(np.float32)

    # Create circular-shift measurement
    y = np.zeros((H, W), dtype=np.float32)
    step = 2
    for k in range(nC):
        masked = mask_2d * x_true[:, :, k]
        y += np.roll(masked, shift=step * k, axis=1)

    print(f"x_true: {x_true.shape}, mask: {mask_2d.shape}, y: {y.shape}")

    # Test with architecture directly
    from pwm_core.recon.cassi_arch.MiJUN import MiJUN, shift_3d

    CKPT = str(ROOT / "reference" / "cassi" / "our_mamba_5stg_recovered.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = MiJUN(stage=5, n_bands=28, step=2).to(device)
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    # Prepare input
    mask3d = np.tile(mask_2d[np.newaxis, :, :], (nC, 1, 1))
    mask3d_t = torch.from_numpy(mask3d).unsqueeze(0).float().to(device)
    y_t = torch.from_numpy(y).unsqueeze(0).float().to(device)

    data = {'Y': y_t, 'mask': mask3d_t}

    print("Running MiJUN inference...")
    t0 = time.time()
    with torch.no_grad():
        x_hat_t = model.forward_test(data)
    dt = time.time() - t0
    print(f"Inference time: {dt:.1f}s")

    x_hat = x_hat_t.detach().cpu().numpy()[0]  # (28, H, W)
    x_hat = np.transpose(x_hat, (1, 2, 0))  # (H, W, 28)
    x_hat = np.clip(x_hat, 0, 1).astype(np.float32)

    p = psnr(x_true, x_hat)
    print(f"PSNR (synthetic): {p:.2f} dB")

else:
    # Load real data
    f = h5_files[0]
    print(f"Loading {f.name}")
    with h5py.File(f, 'r') as hf:
        keys = list(hf.keys())
        print(f"H5 keys: {keys}")
        x_true = hf['x_true'][:].astype(np.float32)
        y = hf['y'][:].astype(np.float32) if 'y' in hf else hf['y_ideal'][:].astype(np.float32)
        if 'mask' in hf:
            mask_2d = hf['mask'][:].astype(np.float32)
        elif 'sampling_mask' in hf:
            mask_2d = hf['sampling_mask'][:].astype(np.float32)

    print(f"x_true: {x_true.shape}, y: {y.shape}, mask: {mask_2d.shape}")

    from pwm_core.recon.cassi_models import cassi_model_recon
    device = "cpu"  # CPU for MiJUN — sequential Mamba scan is CPU-friendly
    print(f"Device: {device}")

    t0 = time.time()
    x_hat = cassi_model_recon(
        y=y, mask_2d=mask_2d, model_key="mijun_5stg",
        nC=28, step=2, device=device, x_true=x_true)
    dt = time.time() - t0
    print(f"Inference time: {dt:.1f}s")
    print(f"x_hat: {x_hat.shape}, range: [{x_hat.min():.4f}, {x_hat.max():.4f}]")

    p = psnr(x_true, x_hat)
    print(f"PSNR: {p:.2f} dB")
