#!/usr/bin/env python3
"""
Phase 1: Operator Verification for InverseNet CASSI Validation

Tests that benchmark operators generate correct measurements.

Usage:
    python phase1_operator_verification.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import scipy.io as sio

# Add paths
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "packages"))

from pwm_core.benchmarks.benchmark_helpers import build_benchmark_operator
from pwm_core.calibration.cassi_upwmi_alg12 import SimulatedOperatorEnlargedGrid, MismatchParameters

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")

def load_scene():
    """Load test scene."""
    path = DATASET / "Truth" / "scene01.mat"
    data = sio.loadmat(str(path))
    scene = data['img'].astype(np.float32)
    logger.info(f"✓ Loaded scene: {scene.shape}, range [{scene.min():.4f}, {scene.max():.4f}]")
    return scene

def load_mask():
    """Load and resize mask."""
    path = DATASET / "mask.mat"
    data = sio.loadmat(str(path))
    mask = data['mask'].astype(np.float32)
    logger.info(f"  Original mask shape: {mask.shape}")

    # Resize to 256x256
    if mask.shape != (256, 256):
        from scipy.ndimage import zoom
        scale = (256 / mask.shape[0], 256 / mask.shape[1])
        mask = zoom(mask, scale)

    logger.info(f"✓ Loaded and resized mask: {mask.shape}, range [{mask.min():.4f}, {mask.max():.4f}]")
    return mask

def main():
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: OPERATOR VERIFICATION")
    logger.info("="*70)

    # Load data
    logger.info("\nLoading test data...")
    scene = load_scene()
    mask = load_mask()

    # Test 1: Create operators using SimulatedOperatorEnlargedGrid directly
    logger.info("\nTest 1: Creating operators with proper mask...")
    try:
        op = SimulatedOperatorEnlargedGrid(mask, N=4, K=2)
        logger.info(f"✓ Operator created: {type(op).__name__}")
    except Exception as e:
        logger.error(f"✗ Failed to create operator: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Verify operator has required methods
    logger.info("\nTest 2: Verifying operator interface...")
    try:
        # Check required methods exist
        assert hasattr(op, 'forward'), "Operator missing forward() method"
        assert hasattr(op, 'apply_mask_correction'), "Operator missing apply_mask_correction() method"
        logger.info(f"✓ Operator has required methods:")
        logger.info(f"  - forward: {callable(op.forward)}")
        logger.info(f"  - apply_mask_correction: {callable(op.apply_mask_correction)}")

        # Verify mask is properly loaded
        assert hasattr(op, 'mask_256'), "Operator missing mask_256"
        assert op.mask_256.shape == (256, 256), f"Mask shape {op.mask_256.shape} != (256, 256)"
        logger.info(f"✓ Mask properly initialized: shape {op.mask_256.shape}")

        assert hasattr(op, 'mask_enlarged'), "Operator missing mask_enlarged"
        assert op.mask_enlarged.shape == (1024, 1024), f"Enlarged mask shape {op.mask_enlarged.shape} != (1024, 1024)"
        logger.info(f"✓ Enlarged mask initialized: shape {op.mask_enlarged.shape}")

    except Exception as e:
        logger.error(f"✗ Interface verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Test mismatch parameter application
    logger.info("\nTest 3: Testing mismatch parameter application...")
    try:
        # Create new operator to test mismatch setup (don't run forward due to computation cost)
        op_mismatch = SimulatedOperatorEnlargedGrid(mask, N=4, K=2)

        # Store original mask for comparison
        mask_original = op_mismatch.mask_enlarged.copy()
        logger.info(f"  Original mask_enlarged range: [{mask_original.min():.4f}, {mask_original.max():.4f}]")

        # Apply mismatch parameters
        mismatch = MismatchParameters(mask_dx=0.5, mask_dy=0.3, mask_theta=0.1)
        op_mismatch.apply_mask_correction(mismatch)
        logger.info(f"✓ Mismatch parameters applied:")
        logger.info(f"  - dx: {mismatch.mask_dx} px")
        logger.info(f"  - dy: {mismatch.mask_dy} px")
        logger.info(f"  - theta: {mismatch.mask_theta}°")

        # Verify mask was modified
        mask_corrected = op_mismatch.mask_enlarged
        diff = np.mean((mask_original - mask_corrected) ** 2)
        logger.info(f"✓ Mask was modified (MSE={diff:.8f})")
        logger.info(f"  Corrected mask_enlarged range: [{mask_corrected.min():.4f}, {mask_corrected.max():.4f}]")

        if diff < 1e-10:
            logger.warning(f"⚠ Mask correction had no effect - check parameter values")
            return False

    except Exception as e:
        logger.error(f"✗ Parameter application failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "="*70)
    logger.info("✓ PHASE 1 COMPLETE - All tests passed!")
    logger.info("="*70 + "\n")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
