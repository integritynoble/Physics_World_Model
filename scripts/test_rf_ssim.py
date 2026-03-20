#!/usr/bin/env python3
"""Get SSIM for pretrained ReconFormer."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
from skimage.metrics import structural_similarity as ssim

from pwm_core.recon.mri_solvers import run_reconformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'datasets/benchmark/mri/standard')
h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])

class MockPhysics:
    def __init__(self, mask):
        self.mask = mask

psnrs, ssims = [], []
for scene_idx in range(20):
    fname = h5_files[scene_idx]
    with h5py.File(os.path.join(DATA_DIR, fname), 'r') as hf:
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y = np.array(hf['y_ideal'], dtype=np.float32)
        mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

    physics = MockPhysics(mask_data)
    result, info = run_reconformer(y, physics, {"device": "cpu"})

    mse = np.mean((x_true - result)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    s = ssim(x_true, result, data_range=x_true.max() - x_true.min())
    psnrs.append(psnr)
    ssims.append(s)
    gc.collect()

print(f"ReconFormer (pretrained): PSNR={np.mean(psnrs):.2f} dB, SSIM={np.mean(ssims):.4f}")
