#!/usr/bin/env python3
"""
Phase 2: Reconstruction Validation for InverseNet CASSI Validation

Tests that MST models produce realistic PSNR values with proper tensor handling.

Usage:
    python phase2_reconstruction_validation.py --device cuda:0
    python phase2_reconstruction_validation.py --device cpu
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio

# Add paths
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "packages"))

from pwm_core.recon.mst import create_mst, shift_back_meas_torch
from pwm_core.analysis.metrics import psnr as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from pwm_core.calibration.cassi_upwmi_alg12 import SimulatedOperatorEnlargedGrid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATASET = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_scene(scene_name: str = "scene01") -> np.ndarray:
    """Load test scene."""
    path = DATASET / "Truth" / f"{scene_name}.mat"
    data = sio.loadmat(str(path))
    for key in ['img', 'Img', 'scene', 'Scene']:
        if key in data:
            scene = data[key].astype(np.float32)
            if scene.ndim == 3 and scene.shape[2] == 28:
                logger.info(f"✓ Loaded scene: {scene.shape}, range [{scene.min():.4f}, {scene.max():.4f}]")
                return scene
    raise ValueError(f"Could not load scene from {path}")


def load_mask() -> np.ndarray:
    """Load and resize mask."""
    path = DATASET / "mask.mat"
    data = sio.loadmat(str(path))
    mask = data['mask'].astype(np.float32)

    # Resize to 256x256 if needed
    if mask.shape != (256, 256):
        from scipy.ndimage import zoom
        scale = (256 / mask.shape[0], 256 / mask.shape[1])
        mask = zoom(mask, scale)

    logger.info(f"✓ Loaded mask: {mask.shape}, range [{mask.min():.4f}, {mask.max():.4f}]")
    return mask


def generate_measurement(scene: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, str]:
    """Generate measurement using SimulatedOperatorEnlargedGrid."""
    logger.info("\n[Forward Model] Generating measurement...")
    
    try:
        op = SimulatedOperatorEnlargedGrid(mask, N=4, K=2)
        logger.info("  Creating operator with enlarged grid (N=4, K=2)...")
        
        # Note: Forward pass is slow due to spectral interpolation
        # For this validation, we'll estimate timing
        logger.info("  Generating measurement (this may take 1-2 minutes)...")
        y = op.forward(scene)
        
        logger.info(f"✓ Measurement generated!")
        logger.info(f"  Shape: {y.shape}")
        logger.info(f"  Range: [{y.min():.6f}, {y.max():.6f}]")
        logger.info(f"  Mean: {y.mean():.6f}, Std: {y.std():.6f}")
        
        return y, "ideal"
    
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        logger.warning("  Falling back to synthetic measurement")
        # Synthetic measurement: average spatial + spectral noise
        y = np.mean(scene, axis=2) + np.random.randn(scene.shape[0], scene.shape[1]) * 0.01
        logger.info(f"✓ Synthetic measurement created: shape {y.shape}")
        return y, "synthetic"


def reconstruct_mst_model(y: np.ndarray, variant: str, device: str) -> Tuple[np.ndarray, bool]:
    """Reconstruct using MST model.
    
    Args:
        y: Measurement (H, W_ext) shape
        variant: 'mst_s' or 'mst_l'
        device: 'cuda:0', 'cpu', etc.
        
    Returns:
        x_recon: (H, W, 28) reconstructed scene
        success: whether reconstruction succeeded
    """
    logger.info(f"\n[{variant.upper()}] Reconstruction...")
    
    try:
        import torch
        
        # Step 1: Load model
        logger.info(f"  Loading {variant} model with pretrained weights...")
        model = create_mst(variant=variant, step=2, base_resolution=256)
        model = model.to(device)
        model.eval()
        logger.info(f"  ✓ Model loaded on {device}")
        
        # Step 2: Prepare tensor
        H, W_ext = y.shape
        logger.info(f"  Input measurement: shape {y.shape}")
        
        y_tensor = torch.from_numpy(y).unsqueeze(0).float().to(device)
        logger.info(f"  After unsqueeze(0): shape {y_tensor.shape}")
        
        # Step 3: Shift back
        logger.info(f"  Applying shift_back transformation (step=2, nC=28)...")
        y_shifted = shift_back_meas_torch(y_tensor, step=2, nC=28)
        logger.info(f"  After shift_back: shape {y_shifted.shape}")
        
        expected_shape = (1, 28, 256, 256)
        if y_shifted.shape != expected_shape:
            logger.warning(f"  ⚠ Unexpected shape {y_shifted.shape}, expected {expected_shape}")
            # Try to continue anyway
        
        # Step 4: Reconstruct
        logger.info(f"  Running {variant} forward pass...")
        with torch.no_grad():
            x_recon_tensor = model(y_shifted)
        logger.info(f"  ✓ Forward pass complete: shape {x_recon_tensor.shape}")
        
        # Step 5: Convert to numpy
        logger.info(f"  Converting to numpy...")
        x_recon = x_recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        x_recon = np.clip(x_recon, 0, 1)
        logger.info(f"  ✓ Reconstruction tensor: shape {x_recon.shape}, range [{x_recon.min():.4f}, {x_recon.max():.4f}]")
        
        return x_recon, True
        
    except ImportError as e:
        logger.error(f"✗ PyTorch not available: {e}")
        return np.zeros((256, 256, 28), dtype=np.float32), False
    except Exception as e:
        logger.error(f"✗ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((256, 256, 28), dtype=np.float32), False


def compute_metrics(scene_true: np.ndarray, x_recon: np.ndarray, variant: str) -> dict:
    """Compute reconstruction metrics."""
    try:
        psnr_val = compute_psnr(scene_true, x_recon, data_range=1.0)

        # SSIM on spatial mean
        spatial_true = np.mean(scene_true, axis=2)
        spatial_recon = np.mean(x_recon, axis=2)
        ssim_val = compute_ssim(spatial_true, spatial_recon, data_range=1.0)

        logger.info(f"  Metrics ({variant}):")
        logger.info(f"    PSNR: {psnr_val:.2f} dB")
        logger.info(f"    SSIM: {ssim_val:.4f}")

        return {
            'variant': variant,
            'psnr': float(psnr_val),
            'ssim': float(ssim_val),
            'success': True
        }
    except Exception as e:
        logger.error(f"  Metric computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'variant': variant,
            'psnr': 0.0,
            'ssim': 0.0,
            'success': False
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Reconstruction Validation')
    parser.add_argument('--device', default='cuda:0', help='Torch device')
    parser.add_argument('--skip-forward', action='store_true', 
                       help='Skip forward model (use synthetic measurement)')
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: RECONSTRUCTION VALIDATION")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading test data...")
    scene = load_scene("scene01")
    mask = load_mask()
    
    # Generate measurement
    if args.skip_forward:
        logger.info("\nUsing synthetic measurement in CASSI format (skipping forward model)...")
        # Create measurement in proper CASSI format: (H, W + (nC-1)*step)
        # For nC=28, step=2: W_ext = 256 + 27*2 = 310
        nC, step = 28, 2
        W_ext = 256 + (nC - 1) * step

        # Create synthetic CASSI measurement: start with spatial average, add noise
        # Real measurements would have spectral shifts, but this is a placeholder
        base = np.mean(scene, axis=2)  # (256, 256)
        y = np.zeros((256, W_ext), dtype=np.float32)
        y[:, :256] = base  # Copy base to first columns
        y[:, 256:] = base[:, :W_ext-256] + np.random.randn(256, W_ext-256) * 0.01  # Extend

        logger.info(f"  Synthetic CASSI measurement: shape {y.shape} (nC={nC}, step={step})")
        meas_type = "synthetic_cassi"
    else:
        y, meas_type = generate_measurement(scene, mask)
    
    logger.info(f"\nMeasurement type: {meas_type}")
    
    # Reconstruct with both models
    results = {
        'scene': 'scene01',
        'measurement_type': meas_type,
        'measurement_shape': y.shape,
        'models': []
    }
    
    for variant in ['mst_s', 'mst_l']:
        logger.info(f"\n{'='*70}")
        x_recon, success = reconstruct_mst_model(y, variant, args.device)
        
        if success:
            metrics = compute_metrics(scene, x_recon, variant)
            results['models'].append(metrics)
            
            # Check if PSNR is in expected range
            psnr_val = metrics['psnr']
            logger.info(f"\n  ✓ {variant.upper()} reconstruction complete!")
            
            if 20 <= psnr_val <= 50:
                logger.info(f"  ✓ PSNR {psnr_val:.2f} dB is in realistic range")
            else:
                logger.warning(f"  ⚠ PSNR {psnr_val:.2f} dB is outside typical range (20-50 dB)")
                if psnr_val < 5:
                    logger.warning(f"    This may indicate model loading or tensor shape issues")
        else:
            logger.error(f"  ✗ {variant.upper()} reconstruction failed")
            results['models'].append({
                'variant': variant,
                'psnr': 0.0,
                'ssim': 0.0,
                'success': False
            })
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PHASE 2 SUMMARY")
    logger.info("="*70)
    
    for result in results['models']:
        if result['success']:
            logger.info(f"  {result['variant'].upper()}: PSNR={result['psnr']:.2f} dB, SSIM={result['ssim']:.4f}")
        else:
            logger.info(f"  {result['variant'].upper()}: FAILED")
    
    # Save results
    import json
    with open(RESULTS_DIR / "phase2_validation.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to {RESULTS_DIR / 'phase2_validation.json'}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ PHASE 2 COMPLETE")
    logger.info("="*70 + "\n")
    
    return all(r.get('success', False) for r in results['models'])


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
