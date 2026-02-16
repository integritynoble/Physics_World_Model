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
    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
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
    peak: float = 100000,
    sigma: float = 0.01,
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

    # Simple CASSI forward model with step=2
    H, W, nC = x_scene.shape
    step = 2
    W_ext = W + (nC - 1) * step
    y_clean = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        y_clean[:, k * step:k * step + W] += mask * x_scene[:, :, k]

    # Add noise
    y_noisy = add_poisson_gaussian_noise(y_clean, peak=peak, sigma=sigma)

    return y_noisy


def reconstruct(method: str, y_meas: np.ndarray, mask: np.ndarray,
                device: str = 'cuda:0') -> np.ndarray:
    """Multi-method reconstruction dispatch.

    Args:
        method: One of 'gap_tv', 'hdnet', 'mst_s', 'mst_l'
        y_meas: (H, W_ext) measurement where W_ext = W + (nC-1)*step
        mask: (H, W) coded aperture mask
        device: torch device for deep learning methods

    Returns:
        x_recon: (H, W, 28) reconstructed cube
    """
    if method == 'gap_tv':
        return gap_tv_cassi(y_meas, mask, n_bands=28, iterations=50, lam=0.05, step=2)
    elif method == 'hdnet':
        from pwm_core.recon.hdnet import hdnet_recon_cassi
        mask_3d = np.repeat(mask[:, :, np.newaxis], 28, axis=2).astype(np.float32)
        return hdnet_recon_cassi(y_meas, mask_3d, nC=28, step=2, device=device, dim=28)
    elif method in ('mst_s', 'mst_l'):
        from pwm_core.recon.mst import mst_recon_cassi
        return mst_recon_cassi(y_meas, mask, nC=28, step=2, device=device, variant=method)
    else:
        raise ValueError(f"Unknown method: {method}")


# Available reconstruction methods
RECONSTRUCTION_METHODS = ['gap_tv', 'hdnet', 'mst_s', 'mst_l']


# ============================================================================
# Scenario Implementations
# ============================================================================

def scenario_i_ideal(
    x_true: np.ndarray,
    mask_ideal: np.ndarray,
    seed: int,
    methods: List[str] = None,
    device: str = 'cuda:0'
) -> Dict:
    """Scenario I: Ideal reconstruction (oracle).

    Uses ideal mask and clean measurement (no noise).
    """
    if methods is None:
        methods = RECONSTRUCTION_METHODS
    logger.info("  Scenario I: Ideal (oracle)")
    start_t = time.time()

    # Generate clean measurement with ideal mask (simple CASSI forward, step=2)
    np.random.seed(seed)
    H, W, nC = x_true.shape
    step = 2
    W_ext = W + (nC - 1) * step
    y_clean = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        y_clean[:, k * step:k * step + W] += mask_ideal * x_true[:, :, k]

    # Reconstruct with each method
    results = {'y_meas': y_clean}
    for method in methods:
        try:
            x_recon = np.clip(reconstruct(method, y_clean, mask_ideal, device=device), 0, 1)
            results[method] = {
                'psnr': float(psnr(x_true, x_recon)),
                'ssim': float(ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))),
                'sam': float(sam(x_true, x_recon)),
            }
            logger.info(f"    {method}: PSNR={results[method]['psnr']:.2f} dB")
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0, 'sam': 180.0}

    results['time'] = float(time.time() - start_t)
    return results


def scenario_ii_assumed(
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    mask_ideal: np.ndarray,
    methods: List[str] = None,
    device: str = 'cuda:0'
) -> Dict:
    """Scenario II: Assumed mask (baseline, no correction).

    Uses ideal mask with corrupted measurement (no correction applied).
    This shows the PSNR degradation from mismatch alone.
    """
    if methods is None:
        methods = RECONSTRUCTION_METHODS
    logger.info("  Scenario II: Assumed (baseline)")
    start_t = time.time()

    results = {'y_meas': y_noisy}
    for method in methods:
        try:
            x_recon = np.clip(reconstruct(method, y_noisy, mask_ideal, device=device), 0, 1)
            results[method] = {
                'psnr': float(psnr(x_true, x_recon)),
                'ssim': float(ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))),
                'sam': float(sam(x_true, x_recon)),
            }
            logger.info(f"    {method}: PSNR={results[method]['psnr']:.2f} dB")
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0, 'sam': 180.0}

    results['time'] = float(time.time() - start_t)
    return results


def scenario_iii_corrected(
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    mask_ideal: np.ndarray,
    mask_real: np.ndarray,
    skip_alg1: bool = False,
    skip_alg2: bool = False,
    methods: List[str] = None,
    device: str = 'cuda:0'
) -> Dict:
    """Scenario III: Corrected (Alg1+Alg2 estimated correction).

    Estimates mismatch using Algorithm 1 (coarse) then Algorithm 2 (fine),
    then reconstructs with the corrected mask using all methods.
    """
    if methods is None:
        methods = RECONSTRUCTION_METHODS
    logger.info("  Scenario III: Corrected (Alg1+Alg2)")
    start_t = time.time()

    mismatch_alg1 = None
    mismatch_alg2 = None
    time_alg1 = 0
    time_alg2 = 0
    corrected_mask = mask_ideal  # fallback

    # Create a solver function for Algorithm 1
    def solver_fn(y_meas, mask_or_op, n_iter=50, lam=0.05):
        mask = mask_or_op.mask_256 if hasattr(mask_or_op, 'mask_256') else mask_or_op
        return gap_tv_cassi(y_meas, mask, n_bands=28, iterations=n_iter, lam=lam, step=2)

    # Algorithm 1: Coarse estimation
    if not skip_alg1:
        logger.info("    Running Algorithm 1 (Hierarchical Beam Search)...")
        alg1_start = time.time()

        alg1 = Algorithm1HierarchicalBeamSearch(
            solver_fn=solver_fn,
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

        # Apply correction to get corrected mask
        corrected_mask = warp_affine_2d(
            mask_ideal,
            dx=mismatch_alg1.mask_dx,
            dy=mismatch_alg1.mask_dy,
            theta=mismatch_alg1.mask_theta
        )

    # Algorithm 2: Fine refinement
    if not skip_alg2 and mismatch_alg1 is not None:
        logger.info("    Running Algorithm 2 (Joint Gradient Refinement)...")
        alg2_start = time.time()

        alg2 = Algorithm2JointGradientRefinement(device="auto", use_checkpointing=True)
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

        corrected_mask = warp_affine_2d(
            mask_ideal,
            dx=mismatch_alg2.mask_dx,
            dy=mismatch_alg2.mask_dy,
            theta=mismatch_alg2.mask_theta
        )

    # Reconstruct with corrected mask using all methods
    results = {'y_meas': y_noisy}
    for method in methods:
        try:
            x_recon = np.clip(reconstruct(method, y_noisy, corrected_mask, device=device), 0, 1)
            results[method] = {
                'psnr': float(psnr(x_true, x_recon)),
                'ssim': float(ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))),
                'sam': float(sam(x_true, x_recon)),
            }
            logger.info(f"    {method}: PSNR={results[method]['psnr']:.2f} dB")
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0, 'sam': 180.0}

    results['time'] = float(time.time() - start_t)
    results['time_alg1'] = float(time_alg1)
    results['time_alg2'] = float(time_alg2)
    results['mismatch_alg1'] = mismatch_alg1.__repr__() if mismatch_alg1 else None
    results['mismatch_alg2'] = mismatch_alg2.__repr__() if mismatch_alg2 else None
    return results


def scenario_iv_truth_fm(
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    mask_ideal: np.ndarray,
    mask_real: np.ndarray,
    mismatch_true: MismatchParameters,
    methods: List[str] = None,
    device: str = 'cuda:0'
) -> Dict:
    """Scenario IV: Truth Forward Model (oracle mismatch correction).

    Uses the true mismatch parameters to apply oracle correction.
    Shows the upper bound of what calibration can achieve.
    """
    if methods is None:
        methods = RECONSTRUCTION_METHODS
    logger.info("  Scenario IV: Truth Forward Model (oracle)")
    start_t = time.time()

    # Apply oracle mismatch correction using true parameters
    oracle_mask = warp_affine_2d(
        mask_ideal,
        dx=mismatch_true.mask_dx,
        dy=mismatch_true.mask_dy,
        theta=mismatch_true.mask_theta
    )

    # Reconstruct with oracle correction using all methods
    results = {'y_meas': y_noisy}
    for method in methods:
        try:
            x_recon = np.clip(reconstruct(method, y_noisy, oracle_mask, device=device), 0, 1)
            results[method] = {
                'psnr': float(psnr(x_true, x_recon)),
                'ssim': float(ssim(np.mean(x_true, axis=2), np.mean(x_recon, axis=2))),
                'sam': float(sam(x_true, x_recon)),
            }
            logger.info(f"    {method}: PSNR={results[method]['psnr']:.2f} dB")
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0, 'sam': 180.0}

    results['time'] = float(time.time() - start_t)
    results['mismatch_true'] = mismatch_true.__repr__()
    return results


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
    skip_alg2: bool = False,
    methods: List[str] = None,
    device: str = 'cuda:0'
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
        methods: List of reconstruction methods
        device: Torch device

    Returns:
        Dictionary with results for all 4 scenarios
    """
    if methods is None:
        methods = RECONSTRUCTION_METHODS
    logger.info(f"\n{'='*70}")
    logger.info(f"Scene {scene_idx + 1}/10: {scene_name}")
    logger.info(f"{'='*70}")

    scene_start = time.time()

    # Step 1: Scenario I (ideal)
    result_i = scenario_i_ideal(x_true, mask_ideal, seed=seed, methods=methods, device=device)

    # Step 2: Inject mismatch and generate noisy measurement
    logger.info("Injecting synthetic mismatch and generating noisy measurement...")
    x_corrupted, mask_corrupted, mismatch_true = inject_mismatch(x_true, mask_real, seed=seed)
    y_noisy = generate_measurement_with_noise(x_true, mask_corrupted, peak=100000, sigma=0.01, seed=seed)

    # Step 3: Scenario II (assumed, baseline)
    result_ii = scenario_ii_assumed(x_true, y_noisy, mask_ideal, methods=methods, device=device)

    # Step 4: Scenario III (corrected)
    result_iii = scenario_iii_corrected(
        x_true, y_noisy, mask_ideal, mask_real,
        skip_alg1=skip_alg1, skip_alg2=skip_alg2,
        methods=methods, device=device
    )

    # Step 5: Scenario IV (truth FM)
    result_iv = scenario_iv_truth_fm(
        x_true, y_noisy, mask_ideal, mask_real, mismatch_true,
        methods=methods, device=device
    )

    # Compute per-method gaps
    gaps = {}
    for method in methods:
        psnr_i = result_i.get(method, {}).get('psnr', 0)
        psnr_ii = result_ii.get(method, {}).get('psnr', 0)
        psnr_iii = result_iii.get(method, {}).get('psnr', 0)
        psnr_iv = result_iv.get(method, {}).get('psnr', 0)
        gaps[method] = {
            'i_to_ii': float(psnr_i - psnr_ii),
            'ii_to_iii': float(psnr_iii - psnr_ii),
            'ii_to_iv': float(psnr_iv - psnr_ii),
            'iii_to_iv': float(psnr_iv - psnr_iii),
            'iv_to_i': float(psnr_i - psnr_iv),
        }

    elapsed = time.time() - scene_start

    logger.info(f"\nScene {scene_idx + 1} Summary:")
    for method in methods:
        pi = result_i.get(method, {}).get('psnr', 0)
        pii = result_ii.get(method, {}).get('psnr', 0)
        piii = result_iii.get(method, {}).get('psnr', 0)
        piv = result_iv.get(method, {}).get('psnr', 0)
        logger.info(f"  {method}: I={pi:.2f} | II={pii:.2f} | III={piii:.2f} | IV={piv:.2f} dB")
    logger.info(f"  Total time: {elapsed:.1f}s")

    return {
        'scene_idx': scene_idx,
        'scene_name': scene_name,
        'scenario_i': {k: v for k, v in result_i.items() if k != 'y_meas'},
        'scenario_ii': {k: v for k, v in result_ii.items() if k != 'y_meas'},
        'scenario_iii': {k: v for k, v in result_iii.items() if k != 'y_meas'},
        'scenario_iv': {k: v for k, v in result_iv.items() if k != 'y_meas'},
        'gaps': gaps,
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
    parser.add_argument('--device', default='cuda:0', help='Torch device')
    parser.add_argument('--methods', nargs='+', default=None,
                        choices=['gap_tv', 'hdnet', 'mst_s', 'mst_l'],
                        help='Reconstruction methods (default: all)')
    args = parser.parse_args()

    methods = args.methods or RECONSTRUCTION_METHODS

    logger.info("CASSI 4-Scenario Validation Protocol")
    logger.info(f"Methods: {methods}")
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
            x_true = load_scene(scene_name)
            if x_true is None:
                logger.error(f"Failed to load scene {scene_name}")
                continue

            result = validate_scene_4scenarios(
                scene_idx,
                scene_name,
                x_true,
                mask_ideal,
                mask_real,
                seed=42 + scene_idx,
                skip_alg1=args.skip_alg1,
                skip_alg2=args.skip_alg2,
                methods=methods,
                device=args.device
            )

            all_results.append(result)

        except Exception as e:
            logger.error(f"Error validating scene {scene_name}: {e}", exc_info=True)
            continue

    total_elapsed = time.time() - total_start

    if not all_results:
        logger.error("No scenes validated successfully")
        return 1

    # Compute per-method summary statistics
    logger.info(f"\n{'='*70}")
    logger.info("Summary Statistics (All Scenes)")
    logger.info(f"{'='*70}")

    summary_methods = {}
    for method in methods:
        for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iii', 'scenario_iv']:
            pvals = [r[scenario_key].get(method, {}).get('psnr', 0) for r in all_results]
            label = scenario_key.replace('_', ' ').upper()
            logger.info(f"  {method} {label}: {np.mean(pvals):.2f} ± {np.std(pvals):.2f} dB")

        # Gaps
        gaps_i_ii = [r['gaps'][method]['i_to_ii'] for r in all_results]
        gaps_ii_iii = [r['gaps'][method]['ii_to_iii'] for r in all_results]
        gaps_ii_iv = [r['gaps'][method]['ii_to_iv'] for r in all_results]
        logger.info(f"  {method} Gap I→II: {np.mean(gaps_i_ii):.2f}, II→III: {np.mean(gaps_ii_iii):.2f}, II→IV: {np.mean(gaps_ii_iv):.2f} dB")

    logger.info(f"\nTotal execution time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours)")
    logger.info(f"Average per scene: {total_elapsed/len(all_results):.1f}s")

    # Save results to JSON
    output_file = REPORTS_DIR / "cassi_validation_4scenarios.json"

    # Build summary
    summary_data = {}
    for method in methods:
        summary_data[method] = {}
        for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iii', 'scenario_iv']:
            pvals = [r[scenario_key].get(method, {}).get('psnr', 0) for r in all_results]
            svals = [r[scenario_key].get(method, {}).get('ssim', 0) for r in all_results]
            summary_data[method][scenario_key] = {
                'psnr_mean': float(np.mean(pvals)),
                'psnr_std': float(np.std(pvals)),
                'ssim_mean': float(np.mean(svals)),
                'ssim_std': float(np.std(svals)),
            }

    output = {
        'num_scenes': len(all_results),
        'methods': methods,
        'total_time': float(total_elapsed),
        'summary': summary_data,
        'per_scene': all_results
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    logger.info(f"\nResults saved to {output_file}")
    return 0


if __name__ == '__main__':
    exit(main())
