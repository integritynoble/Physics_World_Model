#!/usr/bin/env python3
"""Test the updated ReconFormer solver through the standard pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'datasets/benchmark/mri/standard')

# Load the solver module
mod = importlib.import_module("pwm_core.recon.mri_solvers")
run_reconformer = mod.run_reconformer

# Also load MRI operator
from pwm_core.physics.mri.mri_operator import MRIOperator

h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])

psnrs = []
for scene_idx in range(min(5, len(h5_files))):
    fname = h5_files[scene_idx]
    with h5py.File(os.path.join(DATA_DIR, fname), 'r') as hf:
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y = np.array(hf['y_ideal'], dtype=np.float32)
        mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

    operator = MRIOperator(x_shape=x_true.shape, sampling_rate=0.25)
    operator.sampling_mask = mask_data

    cfg = {"device": "cpu"}
    result, info = run_reconformer(y, operator, cfg)

    mse = np.mean((x_true - result)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    psnrs.append(psnr)
    print(f"Scene {scene_idx}: PSNR={psnr:.2f} dB  pretrained={info.get('pretrained', False)}", flush=True)

avg = np.mean(psnrs)
print(f"\nAverage PSNR: {avg:.2f} dB ({len(psnrs)} scenes)")
