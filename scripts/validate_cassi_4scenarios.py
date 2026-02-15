#!/usr/bin/env python3
"""CASSI 4-Scenario Validation Protocol Implementation

Implements comprehensive validation following docs/cassi_plan.md:
- Four scenarios: Ideal, Assumed, Corrected, Truth Forward Model
- Algorithm 1: Hierarchical Beam Search (coarse, ~4.5 hrs/scene)
- Algorithm 2: Joint Gradient Refinement (fine, ~2.5 hrs/scene)
- Metrics: PSNR, SSIM, SAM for all 10 KAIST scenes

Scenario Definitions:
1. Scenario I (Ideal): Clean measurement + ideal mask → oracle reconstruction
2. Scenario II (Assumed): Corrupted measurement + assumed perfect mask → baseline (no correction)
3. Scenario III (Corrected): Corrupted measurement + Alg1+Alg2 estimated correction
4. Scenario IV (Truth FM): Corrupted measurement + true mismatch oracle → upper bound

Usage:
    python scripts/validate_cassi_4scenarios.py [--skip-alg1] [--skip-alg2]
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

# PWM imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pwm_core"))

from pwm_core.calibration import (
    Algorithm1HierarchicalBeamSearch,
    Algorithm2JointGradientRefinement,
    MismatchParameters,
    SimulatedOperatorEnlargedGrid,
    warp_affine_2d,
)
from pwm_core.recon.gap_tv import gap_tv_cassi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
DATASET_REAL = Path("/home/spiritai/MST-main/datasets/TSA_real_data")
REPORTS_DIR = PROJECT_ROOT / "pwm" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# KAIST scene names (tries both formats: kaist1, scene01, etc)
SCENE_NAMES = [
    "scene01", "scene02", "scene03", "scene04", "scene05",
    "scene06", "scene07", "scene08", "scene09", "scene10"
]

# ============================================================================
# Metrics
# ============================================================================

def psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate PSNR in dB."""
    x_true = np.clip(x_true, 0, 1)
    x_recon = np.clip(x_recon, 0, 1)

    mse = np.mean((x_true - x_recon) ** 2)
    if mse < 1e-10:
        return 100.0

    max_val = 1.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def ssim(x_true: np.ndarray, x_recon: np.ndarray, window_size: int = 11) -> float:
    """Calculate SSIM."""
    x_true = np.clip(x_true, 0, 1)
    x_recon = np.clip(x_recon, 0, 1)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window = np.ones((window_size, window_size)) / (window_size ** 2)

    # Calculate means
    mu_true = correlate2d(x_true, window, mode='same', boundary='symm')
    mu_recon = correlate2d(x_recon, window, mode='same', boundary='symm')
    mu_true_sq = mu_true ** 2
    mu_recon_sq = mu_recon ** 2
    mu_cross = mu_true * mu_recon

    # Calculate variances and covariance
    sigma_true_sq = correlate2d(x_true ** 2, window, mode='same', boundary='symm') - mu_true_sq
    sigma_recon_sq = correlate2d(x_recon ** 2, window, mode='same', boundary='symm') - mu_recon_sq
    sigma_cross = correlate2d(x_true * x_recon, window, mode='same', boundary='symm') - mu_cross

    # SSIM
    ssim_map = ((2 * mu_cross + C2) * (2 * sigma_cross + C2)) / \
               ((mu_true_sq + mu_recon_sq + C1) * (sigma_true_sq + sigma_recon_sq + C2))

    return np.mean(ssim_map)


def sam(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate Spectral Angle Mapper (SAM) in degrees."""
    x_true = np.clip(x_true, 1e-6, 1)
    x_recon = np.clip(x_recon, 1e-6, 1)

    # Reshape to (N_pixels, L)
    x_true_flat = x_true.reshape(-1, x_true.shape[2])
    x_recon_flat = x_recon.reshape(-1, x_recon.shape[2])

    # Normalize
    x_true_norm = x_true_flat / (np.linalg.norm(x_true_flat, axis=1, keepdims=True) + 1e-10)
    x_recon_norm = x_recon_flat / (np.linalg.norm(x_recon_flat, axis=1, keepdims=True) + 1e-10)

    # Compute angles
    dots = np.sum(x_true_norm * x_recon_norm, axis=1)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)

    return np.degrees(np.mean(angles))


# ============================================================================
# Data Loading
# ============================================================================

def load_mask(path: Path) -> Optional[np.ndarray]:
    """Load mask from MATLAB .mat file."""
    try:
        data = sio.loadmat(str(path))
        for key in ['mask', 'Mask', 'mask_data']:
            if key in data:
                mask = data[key]
                if isinstance(mask, np.ndarray):
                    return mask.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load mask from {path}: {e}")
    return None


def load_scene(scene_name: str) -> Optional[np.ndarray]:
    """Load scene from MATLAB .mat file."""
    try:
        # Try multiple possible locations
        possible_paths = [
            DATASET_SIMU / f"{scene_name}.mat",
            DATASET_SIMU / "Truth" / f"{scene_name}.mat",
        ]

        for path in possible_paths:
            if path.exists():
                data = sio.loadmat(str(path))
                if 'img' in data:
                    scene = data['img'].astype(np.float32)
                    if scene.ndim == 3 and scene.shape == (256, 256, 28):
                        logger.info(f"Loaded {scene_name} from {path}")
                        return scene
    except Exception as e:
        logger.warning(f"Failed to load scene {scene_name}: {e}")
    return None


def find_mask_files() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find ideal and real mask files."""
    from scipy.ndimage import zoom

    mask_ideal = None
    mask_real = None

    # Search for mask files
    simu_masks = list(DATASET_SIMU.glob("*mask*.mat"))
    real_masks = list(DATASET_REAL.glob("*mask*.mat"))

    if simu_masks:
        mask_ideal = load_mask(simu_masks[0])
        logger.info(f"Loaded ideal mask from {simu_masks[0]}: shape {mask_ideal.shape if mask_ideal is not None else 'None'}")

    if real_masks:
        mask_real = load_mask(real_masks[0])
        logger.info(f"Loaded real mask from {real_masks[0]}: shape {mask_real.shape if mask_real is not None else 'None'}")

        # Resize real mask to match ideal mask size if needed
        if mask_real is not None and mask_ideal is not None:
            if mask_real.shape != mask_ideal.shape:
                scale = mask_ideal.shape[0] / mask_real.shape[0]
                mask_real = zoom(mask_real, scale, order=1).astype(np.float32)
                logger.info(f"Resized real mask to match ideal mask: new shape {mask_real.shape}")

    return mask_ideal, mask_real


# ============================================================================
# Mismatch and Noise
# ============================================================================

def inject_mismatch(
    x_true: np.ndarray,
    mask: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, MismatchParameters]:
    """Inject synthetic mismatch into scene and mask.

    Args:
        x_true: (256, 256, 28) ground truth scene
        mask: (256, 256) ideal mask
        seed: Random seed for reproducibility

    Returns:
        (x_corrupted, mask_corrupted, mismatch_true)
    """
    np.random.seed(seed)

    # Sample mismatch parameters randomly
    mismatch = MismatchParameters(
        mask_dx=np.random.uniform(-3, 3),
        mask_dy=np.random.uniform(-3, 3),
        mask_theta=np.random.uniform(-1, 1),
        disp_a1=np.random.uniform(1.95, 2.05),
        disp_alpha=np.random.uniform(-1, 1)
    )

    # Apply warp to each band
    x_corrupted = np.zeros_like(x_true)
    for k in range(x_true.shape[2]):
        x_corrupted[:, :, k] = warp_affine_2d(
            x_true[:, :, k],
            dx=mismatch.mask_dx,
            dy=mismatch.mask_dy,
            theta=mismatch.mask_theta
        )

    # Apply warp to mask
    mask_corrupted = warp_affine_2d(
        mask,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta
    )

    return x_corrupted, mask_corrupted, mismatch


def add_poisson_gaussian_noise(
    y: np.ndarray,
    peak: float = 10000,
    sigma: float = 1.0
) -> np.ndarray:
    """Add Poisson + Gaussian noise to measurement."""
    # Scale to [0, peak]
    y_max = np.max(y)
    if y_max > 0:
        y_scaled = y / y_max * peak
    else:
        y_scaled = y

    # Poisson noise
    y_poisson = np.random.poisson(y_scaled).astype(np.float32)

    # Gaussian noise
    y_noisy = y_poisson + np.random.normal(0, sigma, y_poisson.shape).astype(np.float32)

    # Scale back
    if peak > 0 and y_max > 0:
        y_noisy = y_noisy / peak * y_max
    elif peak > 0:
        y_noisy = y_noisy / peak

    return np.maximum(y_noisy, 0).astype(np.float32)


# ============================================================================
# Forward Model Wrapper
# ============================================================================

def generate_measurement_with_noise(
    x_scene: np.ndarray,
    mask: np.ndarray,
    peak: float = 10000,
    sigma: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate measurement from scene with noise.

    Args:
        x_scene: (256, 256, 28) scene
        mask: (256, 256) mask
        peak: Poisson peak intensity
        sigma: Gaussian noise std
        seed: Random seed for noise

    Returns:
        y_noisy: (256, 310) noisy measurement
    """
    if seed is not None:
        np.random.seed(seed)

    # Create operator and run forward model
    operator = SimulatedOperatorEnlargedGrid(mask, N=4, K=2, stride=1)
    y_clean = operator.forward(x_scene)

    # Add noise
    y_noisy = add_poisson_gaussian_noise(y_clean, peak=peak, sigma=sigma)

    return y_noisy


def gap_tv_solver_wrapper(
    y_meas: np.ndarray,
    mask_or_operator,
    n_iter: int = 50,
    lam: float = 6.0,
    n_bands: Optional[int] = None
) -> np.ndarray:
    """Wrapper for GAP-TV solver.

    Args:
        y_meas: (H, W) measurement
        mask_or_operator: (H, W) mask OR SimulatedOperatorEnlargedGrid object
        n_iter: Number of iterations
        lam: TV regularization weight
        n_bands: Number of spectral bands (optional, inferred if not provided)

    Returns:
        x_recon: (H, W, n_bands) reconstructed cube
    """
    # Handle both mask arrays and operator objects
    if hasattr(mask_or_operator, 'mask_256'):
        # It's an operator object
        mask = mask_or_operator.mask_256
    else:
        # It's a mask array
        mask = mask_or_operator

    # Infer n_bands if not provided
    if n_bands is None:
        n_bands = y_meas.shape[1] - mask.shape[1] + 1

    return gap_tv_cassi(
        y_meas,
        mask,
        n_bands=n_bands,
        iterations=n_iter,
        lam=lam,
        acc=1.0
    )


# ============================================================================
# Scenario Implementations
# ============================================================================

def scenario_i_ideal(
    x_true: np.ndarray,
    mask_ideal: np.ndarray,
    seed: int
) -> Dict:
    """Scenario I: Ideal reconstruction (oracle).

    Uses ideal mask and clean measurement (no noise).
    """
    logger.info("  Scenario I: Ideal (oracle)")
    start_t = time.time()

    # Generate clean measurement with ideal mask
    np.random.seed(seed)
    operator = SimulatedOperatorEnlargedGrid(mask_ideal, N=4, K=2, stride=1)
    y_clean = operator.forward(x_true)

    # Reconstruct with ideal mask (infer n_bands from measurement size)
    # For CASSI: y.shape = (H, W + n_bands - 1)
    # So: n_bands = y.shape[1] - mask.shape[1] + 1
    n_bands = y_clean.shape[1] - mask_ideal.shape[1] + 1
    x_recon = gap_tv_solver_wrapper(y_clean, mask_ideal, n_iter=50, lam=6.0, n_bands=n_bands)

    # Extract original 28 bands if reconstruction has more
    if x_recon.shape[2] > 28:
        # Downsample spectral dimension to original 28 bands
        from scipy.interpolate import interp1d
        orig_lambda = np.arange(x_recon.shape[2]) / (x_recon.shape[2] - 1)
        target_lambda = np.arange(28) / 27
        x_recon_28 = np.zeros((256, 256, 28), dtype=x_recon.dtype)
        for i in range(256):
            for j in range(256):
                f = interp1d(orig_lambda, x_recon[i, j, :], kind='linear')
                x_recon_28[i, j, :] = f(target_lambda)
        x_recon = x_recon_28

    # Clip to valid range
    x_recon = np.clip(x_recon, 0, 1)

    # Compute metrics
    psnr_val = psnr(x_true, x_recon)
    ssim_val = ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))
    sam_val = sam(x_true, x_recon)

    elapsed = time.time() - start_t
    logger.info(f"    PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, SAM: {sam_val:.2f}°, time: {elapsed:.1f}s")

    return {
        'x_recon': x_recon,
        'y_meas': y_clean,
        'psnr': float(psnr_val),
        'ssim': float(ssim_val),
        'sam': float(sam_val),
        'time': float(elapsed)
    }


def scenario_ii_assumed(
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    mask_ideal: np.ndarray
) -> Dict:
    """Scenario II: Assumed mask (baseline, no correction).

    Uses ideal mask with corrupted measurement (no correction applied).
    This shows the PSNR degradation from mismatch alone.
    """
    logger.info("  Scenario II: Assumed (baseline)")
    start_t = time.time()

    # Infer n_bands from measurement size
    n_bands = y_noisy.shape[1] - mask_ideal.shape[1] + 1

    # Reconstruct with assumed ideal mask (no correction)
    x_recon = gap_tv_solver_wrapper(y_noisy, mask_ideal, n_iter=50, lam=6.0, n_bands=n_bands)

    # Extract original 28 bands if reconstruction has more
    if x_recon.shape[2] > 28:
        from scipy.interpolate import interp1d
        orig_lambda = np.arange(x_recon.shape[2]) / (x_recon.shape[2] - 1)
        target_lambda = np.arange(28) / 27
        x_recon_28 = np.zeros((256, 256, 28), dtype=x_recon.dtype)
        for i in range(256):
            for j in range(256):
                f = interp1d(orig_lambda, x_recon[i, j, :], kind='linear')
                x_recon_28[i, j, :] = f(target_lambda)
        x_recon = x_recon_28

    # Clip to valid range
    x_recon = np.clip(x_recon, 0, 1)

    # Compute metrics
    psnr_val = psnr(x_true, x_recon)
    ssim_val = ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))
    sam_val = sam(x_true, x_recon)

    elapsed = time.time() - start_t
    logger.info(f"    PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, SAM: {sam_val:.2f}°, time: {elapsed:.1f}s")

    return {
        'x_recon': x_recon,
        'y_meas': y_noisy,
        'psnr': float(psnr_val),
        'ssim': float(ssim_val),
        'sam': float(sam_val),
        'time': float(elapsed)
    }


def scenario_iii_corrected(
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    mask_ideal: np.ndarray,
    mask_real: np.ndarray,
    skip_alg1: bool = False,
    skip_alg2: bool = False
) -> Dict:
    """Scenario III: Corrected (Alg1+Alg2 estimated correction).

    Estimates mismatch using Algorithm 1 (coarse) then Algorithm 2 (fine).
    """
    logger.info("  Scenario III: Corrected (Alg1+Alg2)")
    start_t = time.time()

    x_recon = None
    mismatch_alg1 = None
    mismatch_alg2 = None
    time_alg1 = 0
    time_alg2 = 0

    # Algorithm 1: Coarse estimation
    if not skip_alg1:
        logger.info("    Running Algorithm 1 (Hierarchical Beam Search)...")
        alg1_start = time.time()

        alg1 = Algorithm1HierarchicalBeamSearch(
            solver_fn=gap_tv_solver_wrapper,
            n_iter_proxy=5,
            n_iter_beam=10
        )

        mismatch_alg1 = alg1.estimate(
            y_noisy,
            mask_real,
            x_true,
            SimulatedOperatorEnlargedGrid
        )

        time_alg1 = time.time() - alg1_start
        logger.info(f"    Algorithm 1 complete: {mismatch_alg1}, time: {time_alg1:.1f}s")

        # Reconstruct with Algorithm 1 correction
        operator_alg1 = SimulatedOperatorEnlargedGrid(mask_real)
        operator_alg1.apply_mask_correction(mismatch_alg1)
        n_bands = y_noisy.shape[1] - operator_alg1.mask_256.shape[1] + 1
        x_recon = gap_tv_solver_wrapper(y_noisy, operator_alg1.mask_256, n_iter=50, lam=6.0, n_bands=n_bands)

    # Algorithm 2: Fine refinement
    if not skip_alg2 and mismatch_alg1 is not None:
        logger.info("    Running Algorithm 2 (Joint Gradient Refinement)...")
        alg2_start = time.time()

        alg2 = Algorithm2JointGradientRefinement(device="auto", use_checkpointing=True)

        # Dispersion curve for Algorithm 2
        s_nom = np.array([2.0] * 28)

        mismatch_alg2 = alg2.refine(
            mismatch_alg1,
            y_noisy,
            mask_real,
            x_true,
            s_nom,
            SimulatedOperatorEnlargedGrid
        )

        time_alg2 = time.time() - alg2_start
        logger.info(f"    Algorithm 2 complete: {mismatch_alg2}, time: {time_alg2:.1f}s")

        # Reconstruct with Algorithm 2 correction
        operator_alg2 = SimulatedOperatorEnlargedGrid(mask_real)
        operator_alg2.apply_mask_correction(mismatch_alg2)
        n_bands = y_noisy.shape[1] - operator_alg2.mask_256.shape[1] + 1
        x_recon = gap_tv_solver_wrapper(y_noisy, operator_alg2.mask_256, n_iter=50, lam=6.0, n_bands=n_bands)
    elif skip_alg1:
        # Use assumed mask if algorithms skipped
        n_bands = y_noisy.shape[1] - mask_ideal.shape[1] + 1
        x_recon = gap_tv_solver_wrapper(y_noisy, mask_ideal, n_iter=50, lam=6.0, n_bands=n_bands)

    # Clip to valid range
    x_recon = np.clip(x_recon, 0, 1)

    # Compute metrics
    psnr_val = psnr(x_true, x_recon)
    ssim_val = ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))
    sam_val = sam(x_true, x_recon)

    elapsed = time.time() - start_t
    logger.info(f"    PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, SAM: {sam_val:.2f}°, time: {elapsed:.1f}s")

    return {
        'x_recon': x_recon,
        'y_meas': y_noisy,
        'psnr': float(psnr_val),
        'ssim': float(ssim_val),
        'sam': float(sam_val),
        'time': float(elapsed),
        'time_alg1': float(time_alg1),
        'time_alg2': float(time_alg2),
        'mismatch_alg1': mismatch_alg1.__repr__() if mismatch_alg1 else None,
        'mismatch_alg2': mismatch_alg2.__repr__() if mismatch_alg2 else None
    }


def scenario_iv_truth_fm(
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    mask_ideal: np.ndarray,
    mask_real: np.ndarray,
    mismatch_true: MismatchParameters
) -> Dict:
    """Scenario IV: Truth Forward Model (oracle mismatch correction).

    Uses the true mismatch parameters to apply oracle correction.
    Shows the upper bound of what calibration can achieve.
    """
    logger.info("  Scenario IV: Truth Forward Model (oracle)")
    start_t = time.time()

    # Apply oracle mismatch correction using true parameters
    operator_oracle = SimulatedOperatorEnlargedGrid(mask_real)
    operator_oracle.apply_mask_correction(mismatch_true)

    # Reconstruct with oracle correction
    n_bands = y_noisy.shape[1] - operator_oracle.mask_256.shape[1] + 1
    x_recon = gap_tv_solver_wrapper(y_noisy, operator_oracle.mask_256, n_iter=50, lam=6.0, n_bands=n_bands)

    # Clip to valid range
    x_recon = np.clip(x_recon, 0, 1)

    # Compute metrics
    psnr_val = psnr(x_true, x_recon)
    ssim_val = ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))
    sam_val = sam(x_true, x_recon)

    elapsed = time.time() - start_t
    logger.info(f"    PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, SAM: {sam_val:.2f}°, time: {elapsed:.1f}s")

    return {
        'x_recon': x_recon,
        'y_meas': y_noisy,
        'psnr': float(psnr_val),
        'ssim': float(ssim_val),
        'sam': float(sam_val),
        'time': float(elapsed),
        'mismatch_true': mismatch_true.__repr__()
    }


# ============================================================================
# Per-Scene Workflow
# ============================================================================

def validate_scene_4scenarios(
    scene_idx: int,
    scene_name: str,
    x_true: np.ndarray,
    mask_ideal: np.ndarray,
    mask_real: np.ndarray,
    seed: int,
    skip_alg1: bool = False,
    skip_alg2: bool = False
) -> Dict:
    """Run 4-scenario validation for a single scene.

    Args:
        scene_idx: Scene index (0-9)
        scene_name: Scene name (e.g., "kaist1")
        x_true: Ground truth scene (256, 256, 28)
        mask_ideal: Ideal mask (256, 256)
        mask_real: Real mask (256, 256)
        seed: Random seed for reproducibility
        skip_alg1: Skip Algorithm 1
        skip_alg2: Skip Algorithm 2

    Returns:
        Dictionary with results for all 4 scenarios
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Scene {scene_idx + 1}/10: {scene_name}")
    logger.info(f"{'='*70}")

    scene_start = time.time()

    # Step 1: Scenario I (ideal)
    result_i = scenario_i_ideal(x_true, mask_ideal, seed=seed)

    # Step 2: Inject mismatch and generate noisy measurement
    logger.info("Injecting synthetic mismatch and generating noisy measurement...")
    x_corrupted, mask_corrupted, mismatch_true = inject_mismatch(x_true, mask_real, seed=seed)
    y_noisy = generate_measurement_with_noise(x_true, mask_corrupted, peak=10000, sigma=1.0, seed=seed)

    # Step 3: Scenario II (assumed, baseline)
    result_ii = scenario_ii_assumed(x_true, y_noisy, mask_ideal)

    # Step 4: Scenario III (corrected)
    result_iii = scenario_iii_corrected(
        x_true, y_noisy, mask_ideal, mask_real,
        skip_alg1=skip_alg1, skip_alg2=skip_alg2
    )

    # Step 5: Scenario IV (truth FM)
    result_iv = scenario_iv_truth_fm(x_true, y_noisy, mask_ideal, mask_real, mismatch_true)

    # Compute gaps
    gap_i_to_ii = result_i['psnr'] - result_ii['psnr']
    gap_ii_to_iii = result_iii['psnr'] - result_ii['psnr']
    gap_ii_to_iv = result_iv['psnr'] - result_ii['psnr']
    gap_iii_to_iv = result_iv['psnr'] - result_iii['psnr']
    gap_iv_to_i = result_i['psnr'] - result_iv['psnr']

    elapsed = time.time() - scene_start

    logger.info(f"\nScene {scene_idx + 1} Summary:")
    logger.info(f"  Scenario I (Ideal):     PSNR={result_i['psnr']:.2f} dB")
    logger.info(f"  Scenario II (Assumed):  PSNR={result_ii['psnr']:.2f} dB")
    logger.info(f"  Scenario III (Correct): PSNR={result_iii['psnr']:.2f} dB")
    logger.info(f"  Scenario IV (Truth FM): PSNR={result_iv['psnr']:.2f} dB")
    logger.info(f"\n  Gap I→II (degradation):     {gap_i_to_ii:.2f} dB")
    logger.info(f"  Gap II→III (calibration):   {gap_ii_to_iii:.2f} dB")
    logger.info(f"  Gap II→IV (oracle):         {gap_ii_to_iv:.2f} dB")
    logger.info(f"  Gap III→IV (residual):      {gap_iii_to_iv:.2f} dB")
    logger.info(f"  Gap IV→I (solver limit):    {gap_iv_to_i:.2f} dB")
    logger.info(f"  Total time: {elapsed:.1f}s")

    return {
        'scene_idx': scene_idx,
        'scene_name': scene_name,
        'scenario_i': result_i,
        'scenario_ii': result_ii,
        'scenario_iii': result_iii,
        'scenario_iv': result_iv,
        'gaps': {
            'i_to_ii': float(gap_i_to_ii),
            'ii_to_iii': float(gap_ii_to_iii),
            'ii_to_iv': float(gap_ii_to_iv),
            'iii_to_iv': float(gap_iii_to_iv),
            'iv_to_i': float(gap_iv_to_i)
        },
        'mismatch_true': mismatch_true.__repr__(),
        'total_time': float(elapsed)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run 4-scenario validation on all 10 KAIST scenes."""
    import argparse

    parser = argparse.ArgumentParser(description='CASSI 4-scenario validation')
    parser.add_argument('--skip-alg1', action='store_true', help='Skip Algorithm 1')
    parser.add_argument('--skip-alg2', action='store_true', help='Skip Algorithm 2')
    args = parser.parse_args()

    logger.info("CASSI 4-Scenario Validation Protocol")
    logger.info(f"{'='*70}")

    # Load masks
    logger.info("Loading masks...")
    mask_ideal, mask_real = find_mask_files()

    if mask_ideal is None:
        logger.error("Failed to load ideal mask - cannot proceed")
        return 1

    if mask_real is None:
        logger.warning("Failed to load real mask - will use ideal mask")
        mask_real = mask_ideal

    logger.info(f"Ideal mask shape: {mask_ideal.shape}")
    logger.info(f"Real mask shape: {mask_real.shape}")

    # Validate all scenes
    all_results = []
    total_start = time.time()

    for scene_idx, scene_name in enumerate(SCENE_NAMES):
        try:
            # Load scene
            x_true = load_scene(scene_name)
            if x_true is None:
                logger.error(f"Failed to load scene {scene_name}")
                continue

            # Run 4-scenario validation
            result = validate_scene_4scenarios(
                scene_idx,
                scene_name,
                x_true,
                mask_ideal,
                mask_real,
                seed=42 + scene_idx,
                skip_alg1=args.skip_alg1,
                skip_alg2=args.skip_alg2
            )

            all_results.append(result)

        except Exception as e:
            logger.error(f"Error validating scene {scene_name}: {e}", exc_info=True)
            continue

    total_elapsed = time.time() - total_start

    # Compute summary statistics
    if all_results:
        logger.info(f"\n{'='*70}")
        logger.info("Summary Statistics (All Scenes)")
        logger.info(f"{'='*70}")

        psnr_i = [r['scenario_i']['psnr'] for r in all_results]
        psnr_ii = [r['scenario_ii']['psnr'] for r in all_results]
        psnr_iii = [r['scenario_iii']['psnr'] for r in all_results]
        psnr_iv = [r['scenario_iv']['psnr'] for r in all_results]

        gaps_i_ii = [r['gaps']['i_to_ii'] for r in all_results]
        gaps_ii_iii = [r['gaps']['ii_to_iii'] for r in all_results]
        gaps_ii_iv = [r['gaps']['ii_to_iv'] for r in all_results]
        gaps_iii_iv = [r['gaps']['iii_to_iv'] for r in all_results]
        gaps_iv_i = [r['gaps']['iv_to_i'] for r in all_results]

        logger.info(f"Scenario I   (Ideal):     {np.mean(psnr_i):.2f} ± {np.std(psnr_i):.4f} dB")
        logger.info(f"Scenario II  (Assumed):   {np.mean(psnr_ii):.2f} ± {np.std(psnr_ii):.4f} dB")
        logger.info(f"Scenario III (Corrected): {np.mean(psnr_iii):.2f} ± {np.std(psnr_iii):.4f} dB")
        logger.info(f"Scenario IV  (Truth FM):  {np.mean(psnr_iv):.2f} ± {np.std(psnr_iv):.4f} dB")

        logger.info(f"\nGap I→II   (degradation):   {np.mean(gaps_i_ii):.2f} ± {np.std(gaps_i_ii):.4f} dB")
        logger.info(f"Gap II→III (calibration):   {np.mean(gaps_ii_iii):.2f} ± {np.std(gaps_ii_iii):.4f} dB")
        logger.info(f"Gap II→IV  (oracle):        {np.mean(gaps_ii_iv):.2f} ± {np.std(gaps_ii_iv):.4f} dB")
        logger.info(f"Gap III→IV (residual):      {np.mean(gaps_iii_iv):.2f} ± {np.std(gaps_iii_iv):.4f} dB")
        logger.info(f"Gap IV→I   (solver limit):  {np.mean(gaps_iv_i):.2f} ± {np.std(gaps_iv_i):.4f} dB")

        logger.info(f"\nTotal execution time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours)")
        logger.info(f"Average per scene: {total_elapsed/len(all_results):.1f}s")

        # Save results to JSON
        output_file = REPORTS_DIR / "cassi_validation_4scenarios.json"
        summary = {
            'num_scenes': len(all_results),
            'total_time': float(total_elapsed),
            'timestamp': str(Path(__file__).parent.parent),
            'summary': {
                'scenario_i_psnr': float(np.mean(psnr_i)),
                'scenario_ii_psnr': float(np.mean(psnr_ii)),
                'scenario_iii_psnr': float(np.mean(psnr_iii)),
                'scenario_iv_psnr': float(np.mean(psnr_iv)),
                'gap_i_to_ii': float(np.mean(gaps_i_ii)),
                'gap_ii_to_iii': float(np.mean(gaps_ii_iii)),
                'gap_ii_to_iv': float(np.mean(gaps_ii_iv)),
                'gap_iii_to_iv': float(np.mean(gaps_iii_iv)),
                'gap_iv_to_i': float(np.mean(gaps_iv_i))
            },
            'per_scene': all_results
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")
        return 0
    else:
        logger.error("No scenes validated successfully")
        return 1


if __name__ == '__main__':
    exit(main())
