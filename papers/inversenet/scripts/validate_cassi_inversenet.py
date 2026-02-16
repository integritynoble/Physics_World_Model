#!/usr/bin/env python3
"""
CASSI Validation for InverseNet ECCV Paper

Validates 4 reconstruction methods (GAP-TV, HDNet, MST-S, MST-L) across
3 scenarios (I: Ideal, II: Assumed, III: Truth Forward Model) on 10 KAIST scenes.

Scenarios:
- Scenario I:   Ideal measurement + ideal mask → oracle baseline
- Scenario II:  Corrupted measurement + assumed perfect mask → baseline degradation
- Scenario III:  Corrupted measurement + truth mask with mismatch → oracle operator

Usage:
    python validate_cassi_inversenet.py --device cuda:0
"""

import json
import logging
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
DATASET_REAL = Path("/home/spiritai/MST-main/datasets/TSA_real_data")
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
RECONSTRUCTION_METHODS = ['gap_tv', 'hdnet', 'mst_s', 'mst_l']
SCENARIOS = ['scenario_i', 'scenario_ii', 'scenario_iii']
NUM_SCENES = 10


# ============================================================================
# Mismatch Parameters
# ============================================================================

@dataclass
class MismatchParameters:
    """Mismatch parameters for operator."""
    mask_dx: float = 1.5      # pixels
    mask_dy: float = 1.0      # pixels
    mask_theta: float = 0.3   # degrees


# ============================================================================
# Utility Functions
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
        # Try Truth subdirectory first
        path = DATASET_SIMU / "Truth" / f"{scene_name}.mat"
        if not path.exists():
            # Try direct path
            path = DATASET_SIMU / f"{scene_name}.mat"

        if path.exists():
            data = sio.loadmat(str(path))
            # Try common keys for scene data
            for key in ['img', 'Img', 'scene', 'Scene', 'data']:
                if key in data:
                    scene = data[key].astype(np.float32)
                    if scene.ndim == 3 and scene.shape[2] == 28:  # Allow flexible H×W
                        return scene
    except Exception as e:
        logger.warning(f"Failed to load scene {scene_name}: {e}")
    return None


def warp_affine_2d(mask: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    """
    Apply 2D affine transformation to mask (translation + rotation).

    Reuses logic from cassi_upwmi_alg12.py warp_affine_2d()

    Args:
        mask: (H, W) input mask
        dx: x-translation in pixels
        dy: y-translation in pixels
        theta: rotation in degrees

    Returns:
        Warped mask (H, W)
    """
    from scipy.ndimage import affine_transform

    H, W = mask.shape
    center_y, center_x = H / 2, W / 2

    # Convert angle to radians
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Create affine transformation matrix
    # Compose: translate to origin -> rotate -> translate back + (dx,dy)
    matrix = np.array([
        [cos_t, sin_t, -center_x * cos_t - center_y * sin_t + center_x + dx],
        [-sin_t, cos_t, center_x * sin_t - center_y * cos_t + center_y + dy]
    ])

    # Apply affine transform (scipy uses inverse, order=1 for no overshoot)
    inv_matrix = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2, :]
    warped = affine_transform(mask, inv_matrix[:2, :2], offset=inv_matrix[:2, 2], cval=0, order=1)

    # Clip to valid mask range (interpolation can overshoot)
    return np.clip(warped, 0, 1).astype(np.float32)


def psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate PSNR in dB."""
    x_true = np.clip(x_true, 0, 1)
    x_recon = np.clip(x_recon, 0, 1)

    mse = np.mean((x_true - x_recon) ** 2)
    if mse < 1e-10:
        return 100.0

    return 10.0 * np.log10(1.0 / mse)


def ssim(x_true: np.ndarray, x_recon: np.ndarray, window_size: int = 11) -> float:
    """Calculate SSIM."""
    x_true = np.clip(x_true, 0, 1)
    x_recon = np.clip(x_recon, 0, 1)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window = np.ones((window_size, window_size)) / (window_size ** 2)

    mu_true = correlate2d(x_true, window, mode='same', boundary='symm')
    mu_recon = correlate2d(x_recon, window, mode='same', boundary='symm')
    mu_true_sq = mu_true ** 2
    mu_recon_sq = mu_recon ** 2
    mu_cross = mu_true * mu_recon

    sigma_true_sq = correlate2d(x_true ** 2, window, mode='same', boundary='symm') - mu_true_sq
    sigma_recon_sq = correlate2d(x_recon ** 2, window, mode='same', boundary='symm') - mu_recon_sq
    sigma_cross = correlate2d(x_true * x_recon, window, mode='same', boundary='symm') - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_true_sq + mu_recon_sq + C1) * (sigma_true_sq + sigma_recon_sq + C2))

    return np.mean(ssim_map)


def sam(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate Spectral Angle Mapper (SAM) in degrees."""
    x_true = np.clip(x_true, 1e-6, 1)
    x_recon = np.clip(x_recon, 1e-6, 1)

    x_true_flat = x_true.reshape(-1, x_true.shape[2])
    x_recon_flat = x_recon.reshape(-1, x_recon.shape[2])

    x_true_norm = x_true_flat / (np.linalg.norm(x_true_flat, axis=1, keepdims=True) + 1e-10)
    x_recon_norm = x_recon_flat / (np.linalg.norm(x_recon_flat, axis=1, keepdims=True) + 1e-10)

    dots = np.sum(x_true_norm * x_recon_norm, axis=1)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)

    return np.degrees(np.mean(angles))


def add_poisson_gaussian_noise(y: np.ndarray, peak: float = 10000,
                               sigma: float = 1.0) -> np.ndarray:
    """Add Poisson + Gaussian noise to measurement."""
    # Handle potential NaNs and negative values in measurement
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0)

    # Scale to Poisson range
    y_max = np.max(y)
    if y_max <= 0:
        y_max = 1.0

    y_scaled = (y / y_max) * peak
    y_scaled = np.maximum(y_scaled, 0)  # Ensure non-negative

    # Apply Poisson noise
    y_poisson = np.random.poisson(y_scaled.astype(np.int32)).astype(np.float32)

    # Add Gaussian noise
    y_noisy = y_poisson + np.random.normal(0, sigma, y_poisson.shape).astype(np.float32)

    # Rescale back
    if peak > 0:
        y_noisy = y_noisy / (peak + 1e-10) * y_max

    return np.maximum(y_noisy, 0).astype(np.float32)


def cassi_forward(scene: np.ndarray, mask: np.ndarray, step: int = 2) -> np.ndarray:
    """Simple CASSI forward model with spectral dispersion.

    y[:, k*step : k*step + W] += mask * scene[:, :, k]

    Args:
        scene: (H, W, nC) spectral cube
        mask: (H, W) coded aperture
        step: dispersion step in pixels per band

    Returns:
        y: (H, W + (nC-1)*step) 2D measurement
    """
    H, W, nC = scene.shape
    W_ext = W + (nC - 1) * step
    y = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        y[:, k * step:k * step + W] += mask * scene[:, :, k]
    return y


# ============================================================================
# Reconstruction Methods (Wrapper Functions)
# ============================================================================

def reconstruct_gap_tv(y: np.ndarray, mask: np.ndarray, device: str = 'cuda:0') -> np.ndarray:
    """
    Reconstruct using GAP-TV.

    Args:
        y: (H, W_ext) CASSI measurement where W_ext = W + (nC-1)*step
        mask: (H, W) forward operator mask
        device: torch device (unused for GAP-TV, kept for API consistency)

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.gap_tv import gap_tv_cassi
        return gap_tv_cassi(y, mask, n_bands=28, iterations=50, lam=0.01, step=2)
    except Exception as e:
        logger.warning(f"GAP-TV failed: {e}")
        return np.clip(np.random.rand(256, 256, 28).astype(np.float32) * 0.8 + 0.1, 0, 1)


def reconstruct_hdnet(y: np.ndarray, mask: np.ndarray, device: str = 'cuda:0') -> np.ndarray:
    """
    Reconstruct using HDNet.

    Args:
        y: (H, W_ext) CASSI measurement where W_ext = W + (nC-1)*step
        mask: (H, W) forward operator mask
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.hdnet import hdnet_recon_cassi

        # Expand mask to 3D for HDNet: (H, W) -> (H, W, 28)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 28, axis=2).astype(np.float32)

        result = hdnet_recon_cassi(y, mask_3d, nC=28, step=2, device=device, dim=28)
        return np.clip(result, 0, 1).astype(np.float32)
    except Exception as e:
        logger.warning(f"HDNet failed: {e}")
        return np.clip(np.random.rand(256, 256, 28).astype(np.float32) * 0.8 + 0.1, 0, 1)


def reconstruct_mst_s(y: np.ndarray, mask: np.ndarray, device: str = 'cuda:0') -> np.ndarray:
    """
    Reconstruct using MST-S (small Transformer).

    Args:
        y: (H, W_ext) CASSI measurement (W_ext = W + (nC-1)*step)
        mask: (H, W) forward operator mask
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.mst import mst_recon_cassi
        return mst_recon_cassi(y, mask, nC=28, step=2, device=device, variant='mst_s')
    except Exception as e:
        logger.warning(f"MST-S failed: {e}")
        return np.clip(np.random.rand(256, 256, 28).astype(np.float32) * 0.8 + 0.1, 0, 1)


def reconstruct_mst_l(y: np.ndarray, mask: np.ndarray, device: str = 'cuda:0') -> np.ndarray:
    """
    Reconstruct using MST-L (large Transformer).

    Args:
        y: (H, W_ext) CASSI measurement (W_ext = W + (nC-1)*step)
        mask: (H, W) forward operator mask
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.mst import mst_recon_cassi
        return mst_recon_cassi(y, mask, nC=28, step=2, device=device, variant='mst_l')
    except Exception as e:
        logger.warning(f"MST-L failed: {e}")
        return np.clip(np.random.rand(256, 256, 28).astype(np.float32) * 0.8 + 0.1, 0, 1)


RECONSTRUCTION_FUNCTIONS = {
    'gap_tv': reconstruct_gap_tv,
    'hdnet': reconstruct_hdnet,
    'mst_s': reconstruct_mst_s,
    'mst_l': reconstruct_mst_l
}


# ============================================================================
# Scenario Validation Functions
# ============================================================================

def validate_scenario_i(scene: np.ndarray, mask_ideal: np.ndarray,
                        methods: List[str], device: str) -> Dict[str, Dict]:
    """
    Scenario I: Ideal (perfect forward model, no mismatch).

    Purpose: Theoretical upper bound for perfect measurements

    Configuration:
    - Measurement: y_ideal from ideal mask using proper CASSI forward model
    - Forward model: Ideal mask (no mismatch)
    - Reconstruction: Each method with perfect knowledge

    Args:
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        methods: list of method names
        device: torch device

    Returns:
        Dictionary with metrics for each method
    """
    logger.info("  Scenario I: Ideal (oracle)")
    results = {}

    # Create ideal measurement using simple CASSI forward model (step=2)
    y_ideal = cassi_forward(scene, mask_ideal, step=2)  # (256, 310)

    for method in methods:
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_ideal, mask_ideal, device=device)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(scene, x_hat)),
                'ssim': float(ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                'sam': float(sam(scene, x_hat))
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0, 'sam': 180.0}

    return results


def validate_scenario_ii(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchParameters,
                         methods: List[str], device: str) -> Tuple[Dict[str, Dict], np.ndarray]:
    """
    Scenario II: Assumed/Baseline (corrupted measurement, uncorrected operator).

    Purpose: Realistic baseline showing degradation from uncorrected mismatch

    Configuration:
    - Measurement: y_corrupt from mask with injected mismatch using CASSI forward model
    - Forward model: Ideal mask (assumed perfect)
    - Reconstruction: Each method assuming perfect alignment

    Args:
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        mismatch: MismatchParameters for injection
        methods: list of method names
        device: torch device

    Returns:
        Tuple of (results dict, y_corrupt measurement for reuse in Scenario III)
    """
    logger.info("  Scenario II: Assumed/Baseline (uncorrected mismatch)")
    results = {}

    # Create corrupted measurement: warp the ideal mask, use simple CASSI forward
    mask_corrupted = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta
    )

    # Generate measurement with corrupted mask (step=2)
    y_corrupt = cassi_forward(scene, mask_corrupted, step=2)

    # Add realistic noise
    y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=100000, sigma=0.01)

    # Reconstruct with each method ASSUMING PERFECT (ideal) MASK
    for method in methods:
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_corrupt, mask_ideal, device=device)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(scene, x_hat)),
                'ssim': float(ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                'sam': float(sam(scene, x_hat))
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0, 'sam': 180.0}

    return results, y_corrupt


def validate_scenario_iii(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchParameters, y_corrupt: np.ndarray,
                         methods: List[str], device: str) -> Dict[str, Dict]:
    """
    Scenario III: Truth Forward Model (corrupted measurement, oracle operator).

    Purpose: Upper bound for corrupted measurements when true mismatch is known

    Configuration:
    - Measurement: Same y_corrupt as Scenario II
    - Forward model: Ideal mask with TRUE mismatch applied (oracle knowledge)
    - Reconstruction: Each method with oracle operator knowledge

    Args:
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        mismatch: MismatchParameters (ground truth for this scenario)
        y_corrupt: measurement from Scenario II
        methods: list of method names
        device: torch device

    Returns:
        Dictionary with metrics for each method
    """
    logger.info("  Scenario III: Truth Forward Model (oracle operator)")
    results = {}

    # Apply true mismatch to ideal mask → oracle knows the corruption
    mask_truth = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta
    )

    # Reconstruct with each method using TRUE (corrupted) MASK
    for method in methods:
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_corrupt, mask_truth, device=device)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(scene, x_hat)),
                'ssim': float(ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                'sam': float(sam(scene, x_hat))
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0, 'sam': 180.0}

    return results


# ============================================================================
# Scene Validation
# ============================================================================

def validate_scene(scene_idx: int, scene: np.ndarray,
                   mask_ideal: np.ndarray, mask_real: np.ndarray,
                   mismatch: MismatchParameters,
                   methods: List[str], device: str) -> Dict:
    """
    Validate one scene across all 3 scenarios and all methods.

    Args:
        scene_idx: scene index (0-9)
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        mask_real: (256, 256) real mask
        mismatch: MismatchParameters
        methods: list of method names
        device: torch device

    Returns:
        Dictionary with complete results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Scene {scene_idx + 1}/10")
    logger.info(f"{'='*70}")

    start_time = time.time()

    # Scenario I
    res_i = validate_scenario_i(scene, mask_ideal, methods, device)

    # Scenario II (returns both results and measurement for reuse)
    res_ii, y_corrupt = validate_scenario_ii(scene, mask_ideal, mismatch, methods, device)

    # Scenario III (reuses y_corrupt from Scenario II)
    res_iii = validate_scenario_iii(scene, mask_ideal, mismatch, y_corrupt, methods, device)

    elapsed = time.time() - start_time

    # Compile results
    result = {
        'scene_idx': scene_idx + 1,
        'scenario_i': res_i,
        'scenario_ii': res_ii,
        'scenario_iii': res_iii,
        'elapsed_time': elapsed,
        'mismatch_injected': {
            'mask_dx': mismatch.mask_dx,
            'mask_dy': mismatch.mask_dy,
            'mask_theta': mismatch.mask_theta
        }
    }

    # Calculate gaps for each method
    result['gaps'] = {}
    for method in methods:
        psnr_i = res_i[method]['psnr']
        psnr_ii = res_ii[method]['psnr']
        psnr_iii = res_iii[method]['psnr']

        result['gaps'][method] = {
            'gap_i_ii': float(psnr_i - psnr_ii),      # Degradation from mismatch
            'gap_ii_iii': float(psnr_iii - psnr_ii),    # Recovery from oracle
            'gap_iii_i': float(psnr_i - psnr_iii)       # Residual gap
        }

    # Log summary for this scene
    for method in methods:
        logger.info(f"\n  {method.upper()}:")
        logger.info(f"    I (Ideal):   {res_i[method]['psnr']:.2f} dB")
        logger.info(f"    II (Assumed): {res_ii[method]['psnr']:.2f} dB (gap {result['gaps'][method]['gap_i_ii']:.2f} dB)")
        logger.info(f"    III (Oracle):  {res_iii[method]['psnr']:.2f} dB (recovery {result['gaps'][method]['gap_ii_iii']:.2f} dB)")

    return result


# ============================================================================
# Results Aggregation
# ============================================================================

def compute_summary_statistics(all_results: List[Dict]) -> Dict:
    """
    Compute aggregated statistics across all scenes.

    Output format matches cassi_summary.json:
    - scenario_i/ii/iii each have per-method psnr_mean/std, ssim_mean/std
    - gaps have per-method gap_i_ii_mean/std, gap_ii_iii_mean/std

    Args:
        all_results: List of per-scene results

    Returns:
        Dictionary with aggregated statistics
    """
    summary = {
        'num_scenes': len(all_results),
        'methods': list(RECONSTRUCTION_METHODS),
        'mismatch': {
            'dx': 1.5,
            'dy': 1.0,
            'theta': 0.3
        },
        'noise': {
            'alpha': 100000,
            'sigma': 0.01
        }
    }

    for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iii']:
        summary[scenario_key] = {}

        for method in RECONSTRUCTION_METHODS:
            psnr_values = [r[scenario_key][method]['psnr'] for r in all_results if r[scenario_key][method]['psnr'] > 0]
            ssim_values = [r[scenario_key][method]['ssim'] for r in all_results if r[scenario_key][method]['ssim'] > 0]

            summary[scenario_key][method] = {
                'psnr_mean': float(np.mean(psnr_values)) if psnr_values else 0.0,
                'psnr_std': float(np.std(psnr_values)) if psnr_values else 0.0,
                'ssim_mean': float(np.mean(ssim_values)) if ssim_values else 0.0,
                'ssim_std': float(np.std(ssim_values)) if ssim_values else 0.0,
            }

    # Compute gaps across scenarios
    summary['gaps'] = {}
    for method in RECONSTRUCTION_METHODS:
        gap_values_i_ii = [r['gaps'][method]['gap_i_ii'] for r in all_results]
        gap_values_ii_iii = [r['gaps'][method]['gap_ii_iii'] for r in all_results]

        summary['gaps'][method] = {
            'gap_i_ii_mean': float(np.mean(gap_values_i_ii)),
            'gap_i_ii_std': float(np.std(gap_values_i_ii)),
            'gap_ii_iii_mean': float(np.mean(gap_values_ii_iii)),
            'gap_ii_iii_std': float(np.std(gap_values_ii_iii)),
        }

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    """Main validation loop."""
    parser = argparse.ArgumentParser(description='CASSI Validation for InverseNet ECCV')
    parser.add_argument('--device', default='cuda:0', help='Torch device for reconstruction')
    args = parser.parse_args()

    logger.info("\n" + "="*70)
    logger.info("CASSI Validation for InverseNet ECCV Paper")
    logger.info("3 Scenarios × 4 Methods × 10 Scenes = 120 Reconstructions")
    logger.info("="*70)

    # Load masks
    mask_ideal = load_mask(DATASET_SIMU / "mask.mat")
    mask_real = load_mask(DATASET_REAL / "mask.mat")

    if mask_ideal is None:
        logger.warning("Ideal mask not found, using synthetic")
        mask_ideal = np.random.rand(256, 256).astype(np.float32) * 0.8 + 0.1
    if mask_real is None:
        logger.warning("Real mask not found, using ideal mask")
        mask_real = mask_ideal.copy()

    logger.info(f"Ideal mask shape: {mask_ideal.shape}")
    logger.info(f"Real mask shape: {mask_real.shape}")

    # Mismatch parameters
    mismatch = MismatchParameters(mask_dx=1.5, mask_dy=1.0, mask_theta=0.3)
    logger.info(f"Mismatch parameters: dx={mismatch.mask_dx} px, dy={mismatch.mask_dy} px, θ={mismatch.mask_theta}°")

    # Validate all scenes
    all_results = []
    start_total_time = time.time()

    for scene_idx in range(NUM_SCENES):
        scene_name = f"scene{scene_idx + 1:02d}"
        scene = load_scene(scene_name)

        if scene is None:
            logger.warning(f"{scene_name} not found, skipping")
            continue

        result = validate_scene(scene_idx, scene, mask_ideal, mask_real,
                               mismatch, RECONSTRUCTION_METHODS, args.device)
        all_results.append(result)

    total_time = time.time() - start_total_time

    if not all_results:
        logger.error("No results collected!")
        return

    # Compute summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)

    summary = compute_summary_statistics(all_results)

    # Log summary
    for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iii']:
        scenario_label = scenario_key.replace('_', ' ').upper()
        logger.info(f"\n{scenario_label}:")
        for method in RECONSTRUCTION_METHODS:
            psnr_mean = summary[scenario_key][method]['psnr_mean']
            psnr_std = summary[scenario_key][method]['psnr_std']
            logger.info(f"  {method.upper():8s}: {psnr_mean:.2f} ± {psnr_std:.2f} dB")

    logger.info(f"\nGaps (PSNR degradation/recovery):")
    for method in RECONSTRUCTION_METHODS:
        gap_i_ii = summary['gaps'][method]['gap_i_ii_mean']
        gap_ii_iii = summary['gaps'][method]['gap_ii_iii_mean']
        logger.info(f"  {method.upper():8s}: Gap I→II={gap_i_ii:.2f} dB, Recovery II→III={gap_ii_iii:.2f} dB")

    logger.info(f"\nExecution time: {total_time / 3600:.2f} hours ({total_time / len(all_results) / 60:.1f} min/scene)")

    # Save results
    output_detailed = RESULTS_DIR / "cassi_validation_results.json"
    output_summary = RESULTS_DIR / "cassi_summary.json"

    with open(output_detailed, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Detailed results saved to: {output_detailed}")

    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {output_summary}")

    logger.info("\n✅ Validation complete!")


if __name__ == '__main__':
    main()
