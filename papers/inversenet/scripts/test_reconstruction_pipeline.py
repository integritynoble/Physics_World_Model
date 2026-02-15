#!/usr/bin/env python3
"""
Quick diagnostic test of CASSI forward model and MST reconstruction.
"""

import numpy as np
import torch
import logging
from pathlib import Path
import scipy.io as sio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test 1: Load scene
logger.info("="*70)
logger.info("TEST 1: Load Scene")
logger.info("="*70)

DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
scene_path = DATASET_SIMU / "Truth" / "scene01.mat"
logger.info(f"Loading scene from: {scene_path}")

data = sio.loadmat(str(scene_path))
for key in ['img', 'Img', 'scene', 'Scene', 'data']:
    if key in data:
        scene = data[key].astype(np.float32)
        logger.info(f"✓ Found scene with key '{key}': shape {scene.shape}, dtype {scene.dtype}")
        logger.info(f"  Range: [{np.min(scene):.4f}, {np.max(scene):.4f}]")
        break

# Test 2: Forward model
logger.info("\n" + "="*70)
logger.info("TEST 2: Forward Model (SimulatedOperatorEnlargedGrid)")
logger.info("="*70)

try:
    from pwm_core.calibration.cassi_upwmi_alg12 import SimulatedOperatorEnlargedGrid

    # Load ideal mask
    mask_path = DATASET_SIMU / "mask.mat"
    mask_data = sio.loadmat(str(mask_path))
    for key in ['mask', 'Mask']:
        if key in mask_data:
            mask = mask_data[key].astype(np.float32)
            logger.info(f"✓ Loaded mask: shape {mask.shape}, range [{np.min(mask):.4f}, {np.max(mask):.4f}]")
            break

    # Create forward model
    op = SimulatedOperatorEnlargedGrid(mask, N=4, K=2, stride=1)
    logger.info("✓ Created SimulatedOperatorEnlargedGrid")

    # Generate measurement
    y_meas = op.forward(scene)
    logger.info(f"✓ Forward model output: shape {y_meas.shape}, dtype {y_meas.dtype}")
    logger.info(f"  Range: [{np.min(y_meas):.4f}, {np.max(y_meas):.4f}]")

except Exception as e:
    logger.error(f"✗ Forward model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: MST Reconstruction
logger.info("\n" + "="*70)
logger.info("TEST 3: MST-L Reconstruction")
logger.info("="*70)

try:
    from pwm_core.recon.mst import create_mst, shift_back_meas_torch

    logger.info("Creating MST-L model...")
    model = create_mst(variant='mst_l', step=2, base_resolution=256)
    logger.info(f"✓ Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move to GPU if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    logger.info(f"✓ Model moved to {device}")

    # Prepare measurement
    logger.info(f"Converting measurement to tensor (shape {y_meas.shape})...")
    y_tensor = torch.from_numpy(y_meas).unsqueeze(0).float().to(device)
    logger.info(f"  Tensor shape: {y_tensor.shape}")

    # Apply shift_back
    logger.info("Applying shift_back_meas_torch...")
    y_shifted = shift_back_meas_torch(y_tensor, step=2, nC=28)
    logger.info(f"✓ Shifted shape: {y_shifted.shape}")
    logger.info(f"  Shifted range: [{y_shifted.min():.4f}, {y_shifted.max():.4f}]")

    # Reconstruct
    logger.info("Running MST-L inference...")
    with torch.no_grad():
        x_recon = model(y_shifted)
    logger.info(f"✓ Reconstruction output shape: {x_recon.shape}")
    logger.info(f"  Reconstruction range: [{x_recon.min():.4f}, {x_recon.max():.4f}]")

    # Convert back to numpy
    x_np = x_recon.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    logger.info(f"✓ Final numpy shape: {x_np.shape}")
    logger.info(f"  Final range: [{np.min(x_np):.4f}, {np.max(x_np):.4f}]")

    # Compute PSNR
    mse = np.mean((scene - x_np) ** 2)
    psnr_val = 10 * np.log10(1.0 / (mse + 1e-10))
    logger.info(f"✓ PSNR: {psnr_val:.2f} dB")

except Exception as e:
    logger.error(f"✗ MST reconstruction failed: {e}")
    import traceback
    traceback.print_exc()

logger.info("\n" + "="*70)
logger.info("DIAGNOSTICS COMPLETE")
logger.info("="*70)
