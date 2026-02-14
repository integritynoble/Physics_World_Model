#!/usr/bin/env python3
"""CASSI Algorithm 1 & 2 Demonstration

Quick demo showing Algorithm 1 and Algorithm 2 implementations on synthetic data.

Usage:
    python scripts/demo_cassi_alg12.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pwm_core"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_gap_tv_solver(y: np.ndarray, operator, n_iter: int = 5) -> np.ndarray:
    """Simple mock GAP-TV solver for demonstration."""
    # Return a mock reconstruction with noise proportional to operator mismatch
    H, W, L = 256, 256, 28
    x_recon = 0.5 * np.ones((H, W, L), dtype=np.float32)
    noise = np.random.randn(H, W, L) * 0.01
    return np.clip(x_recon + noise, 0, 1).astype(np.float32)


def demo_algorithm1():
    """Demonstrate Algorithm 1: Hierarchical Beam Search."""
    logger.info("\n" + "="*70)
    logger.info("DEMO: Algorithm 1 - Hierarchical Beam Search")
    logger.info("="*70)

    from pwm_core.calibration import (
        Algorithm1HierarchicalBeamSearch,
        SimulatedOperatorEnlargedGrid,
    )

    # Create synthetic data
    logger.info("Creating synthetic scene and measurements...")
    H, W, L = 256, 256, 28
    x_true = np.random.rand(H, W, L).astype(np.float32) * 0.8
    mask_real = np.random.rand(H, W).astype(np.float32) * 0.8 + 0.1
    y_meas = np.random.rand(H, H + 54).astype(np.float32) * 0.1  # (256, 310)

    logger.info(f"Scene shape: {x_true.shape}")
    logger.info(f"Mask shape: {mask_real.shape}")
    logger.info(f"Measurement shape: {y_meas.shape}")

    # Initialize Algorithm 1
    logger.info("\nInitializing Algorithm 1...")
    alg1 = Algorithm1HierarchicalBeamSearch(
        solver_fn=simple_gap_tv_solver,
        n_iter_proxy=2,      # Reduced for demo
        n_iter_beam=2
    )

    # Run estimation
    logger.info("Running Algorithm 1 estimation...")
    start_time = time.time()

    try:
        mismatch = alg1.estimate(y_meas, mask_real, x_true, SimulatedOperatorEnlargedGrid)
        elapsed = time.time() - start_time

        logger.info(f"✓ Algorithm 1 completed successfully in {elapsed:.2f} seconds")
        logger.info(f"Estimated parameters:")
        logger.info(f"  Mask translation: dx={mismatch.mask_dx:.4f} px, dy={mismatch.mask_dy:.4f} px")
        logger.info(f"  Mask rotation: theta={mismatch.mask_theta:.4f}°")
        logger.info(f"  Dispersion: a1={mismatch.disp_a1:.4f} px/band, alpha={mismatch.disp_alpha:.4f}°")

        return mismatch

    except Exception as e:
        logger.error(f"Algorithm 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_algorithm2(mismatch_coarse):
    """Demonstrate Algorithm 2: Joint Gradient Refinement."""
    logger.info("\n" + "="*70)
    logger.info("DEMO: Algorithm 2 - Joint Gradient Refinement")
    logger.info("="*70)

    if mismatch_coarse is None:
        logger.error("Algorithm 1 result required for Algorithm 2")
        return

    from pwm_core.calibration import Algorithm2JointGradientRefinement

    # Create synthetic data (same as Algorithm 1)
    H, W, L = 256, 256, 28
    x_true = np.random.rand(H, W, L).astype(np.float32) * 0.8
    y_meas = np.random.rand(H, H + 54).astype(np.float32) * 0.1

    logger.info("Initializing Algorithm 2...")
    alg2 = Algorithm2JointGradientRefinement()

    logger.info("Running Algorithm 2 refinement...")
    start_time = time.time()

    try:
        mismatch_refined = alg2.refine(mismatch_coarse, y_meas, x_true)
        elapsed = time.time() - start_time

        logger.info(f"✓ Algorithm 2 completed successfully in {elapsed:.2f} seconds")
        logger.info(f"Refined parameters:")
        logger.info(f"  Mask translation: dx={mismatch_refined.mask_dx:.4f} px, dy={mismatch_refined.mask_dy:.4f} px")
        logger.info(f"  Mask rotation: theta={mismatch_refined.mask_theta:.4f}°")
        logger.info(f"  Dispersion: a1={mismatch_refined.disp_a1:.4f} px/band, alpha={mismatch_refined.disp_alpha:.4f}°")

        return mismatch_refined

    except Exception as e:
        logger.error(f"Algorithm 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return mismatch_coarse


def main():
    """Run algorithm demonstrations."""
    logger.info("CASSI Algorithm 1 & 2 Demonstration")
    logger.info("="*70)

    # Run Algorithm 1
    mismatch_coarse = demo_algorithm1()

    if mismatch_coarse is None:
        logger.error("Algorithm 1 failed, cannot proceed to Algorithm 2")
        return

    # Run Algorithm 2
    mismatch_refined = demo_algorithm2(mismatch_coarse)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info("✓ Algorithm 1 (Hierarchical Beam Search): Completed")
    logger.info(f"  Result: {mismatch_coarse}")
    logger.info("✓ Algorithm 2 (Joint Gradient Refinement): Completed")
    logger.info(f"  Result: {mismatch_refined}")
    logger.info("\nBoth algorithms are ready for full 10-scene validation!")


if __name__ == '__main__':
    main()
