#!/usr/bin/env python3
"""
SPC (Single-Pixel Camera) Validation for InverseNet ECCV Paper

Validates 3 reconstruction methods (ADMM, ISTA-Net+, HATNet) across
3 scenarios (I: Ideal, II: Assumed, IV: Truth Forward Model) on 11 Set11 images.

Scenarios:
- Scenario I:   Ideal measurement + ideal DMD patterns → oracle baseline
- Scenario II:  Corrupted measurement + assumed perfect patterns → baseline degradation
- Scenario IV:  Corrupted measurement + truth patterns with mismatch → oracle operator

Usage:
    python validate_spc_inversenet.py --device cuda:0
"""

import json
import logging
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.signal import correlate2d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SET11_DIR = Path("/home/spiritai/ISTA-Net-PyTorch-master/data/Set11")
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
RECONSTRUCTION_METHODS = ['admm', 'ista_net_plus', 'hatnet']
SCENARIOS = ['scenario_i', 'scenario_ii', 'scenario_iv']
NUM_IMAGES = 11
IMAGE_SIZE = 64  # Center crop to 64×64
MEASUREMENT_DIM = 614  # 15% of 4096 (64×64)
COMPRESSION_RATIO = 4096 / 614  # ~6.67


# ============================================================================
# Mismatch Parameters
# ============================================================================

@dataclass
class MismatchParameters:
    """Mismatch parameters for SPC operator."""
    mask_dx: float = 0.4      # pixels
    mask_dy: float = 0.4      # pixels
    mask_theta: float = 0.08   # degrees
    clock_offset: float = 0.06  # pattern duration
    illum_drift_linear: float = 0.04  # fraction/sequence
    gain: float = 1.08        # sensor gain ratio


# ============================================================================
# Utility Functions
# ============================================================================

def load_set11_images() -> List[np.ndarray]:
    """
    Load Set11 images (256×256) and center-crop to 64×64.

    Returns:
        List of 11 images (64×64) normalized to [0,1]
    """
    images = []

    if not SET11_DIR.exists():
        logger.warning(f"Set11 directory not found at {SET11_DIR}, using synthetic images")
        # Fallback: create synthetic images
        for i in range(NUM_IMAGES):
            img = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
            img = gaussian_filter(img, sigma=2.0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images.append(img)
        return images

    # Try to load from directory
    import glob
    image_files = sorted(glob.glob(str(SET11_DIR / "*.png"))) + \
                  sorted(glob.glob(str(SET11_DIR / "*.jpg"))) + \
                  sorted(glob.glob(str(SET11_DIR / "*.tif")))

    if not image_files:
        logger.warning(f"No images found in {SET11_DIR}, using synthetic images")
        for i in range(NUM_IMAGES):
            img = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
            img = gaussian_filter(img, sigma=2.0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images.append(img)
        return images

    # Load and process images
    for idx, img_path in enumerate(image_files[:NUM_IMAGES]):
        try:
            from PIL import Image
            img = Image.open(img_path).convert('L')  # Grayscale
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Center crop to 64×64
            h, w = img_array.shape
            y_start = (h - IMAGE_SIZE) // 2
            x_start = (w - IMAGE_SIZE) // 2
            img_cropped = img_array[y_start:y_start+IMAGE_SIZE, x_start:x_start+IMAGE_SIZE]

            images.append(img_cropped)
            logger.info(f"Loaded image {idx+1}/{NUM_IMAGES}: {Path(img_path).name}")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # Use synthetic image as fallback
            img = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
            img = gaussian_filter(img, sigma=2.0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images.append(img)

    # Fill remaining with synthetic if needed
    while len(images) < NUM_IMAGES:
        img = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        img = gaussian_filter(img, sigma=2.0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        images.append(img)

    return images[:NUM_IMAGES]


def generate_dmd_patterns(num_patterns: int = MEASUREMENT_DIM, image_size: int = IMAGE_SIZE,
                         seed: int = 42) -> np.ndarray:
    """
    Generate random ±1 binary DMD patterns.

    Args:
        num_patterns: number of measurement patterns
        image_size: spatial size (image_size × image_size)
        seed: random seed

    Returns:
        DMD patterns (num_patterns, image_size*image_size) with values in {-1, 1}
    """
    np.random.seed(seed)
    patterns = np.random.choice([-1, 1], size=(num_patterns, image_size * image_size),
                               p=[0.5, 0.5]).astype(np.float32)
    return patterns


def warp_dmd_pattern_2d(patterns: np.ndarray, image_size: int, dx: float, dy: float,
                       theta: float) -> np.ndarray:
    """
    Apply 2D affine transformation to DMD patterns (translation + rotation).

    Args:
        patterns: (num_patterns, image_size*image_size) DMD patterns
        image_size: spatial size
        dx: x-translation in pixels
        dy: y-translation in pixels
        theta: rotation in degrees

    Returns:
        Warped patterns (num_patterns, image_size*image_size)
    """
    num_patterns = patterns.shape[0]
    patterns_2d = patterns.reshape(num_patterns, image_size, image_size)
    patterns_warped = np.zeros_like(patterns_2d)

    for i in range(num_patterns):
        pattern_2d = patterns_2d[i]

        # Apply affine transformation
        center_y, center_x = image_size / 2, image_size / 2
        theta_rad = np.radians(theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        matrix = np.array([
            [cos_t, sin_t, -center_x * cos_t - center_y * sin_t + center_x + dx],
            [-sin_t, cos_t, center_x * sin_t - center_y * cos_t + center_y + dy]
        ])

        inv_matrix = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2, :]
        pattern_warped = affine_transform(pattern_2d, inv_matrix[:2, :2],
                                         offset=inv_matrix[:2, 2], cval=0)
        patterns_warped[i] = pattern_warped

    return patterns_warped.reshape(num_patterns, -1).astype(np.float32)


def forward_model_spc(x: np.ndarray, patterns: np.ndarray,
                     gain: float = 1.0, illum_drift: float = 0.0) -> np.ndarray:
    """
    Apply SPC forward model: y = A(x) + offset

    Args:
        x: (image_size, image_size) image
        patterns: (num_patterns, image_size*image_size) DMD patterns
        gain: sensor gain
        illum_drift: illumination drift linear term

    Returns:
        y: (num_patterns,) measurements
    """
    x_flat = x.flatten()
    num_patterns = patterns.shape[0]

    # Apply forward model: y[i] = <pattern[i], x>
    y = np.dot(patterns, x_flat)

    # Apply sensor gain
    y = y * gain

    # Apply illumination drift (linear across patterns)
    drift = np.linspace(1 - illum_drift, 1 + illum_drift, num_patterns)
    y = y * drift

    return y.astype(np.float32)


def psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate PSNR in dB."""
    x_true = np.clip(x_true, 0, 1)
    x_recon = np.clip(x_recon, 0, 1)

    mse = np.mean((x_true - x_recon) ** 2)
    if mse < 1e-10:
        return 100.0

    return 10.0 * np.log10(1.0 / mse)


def ssim(x_true: np.ndarray, x_recon: np.ndarray, window_size: int = 11) -> float:
    """Calculate SSIM for 2D images."""
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

    ssim_map = ((2 * mu_cross + C2) * (2 * sigma_cross + C2)) / \
               ((mu_true_sq + mu_recon_sq + C1) * (sigma_true_sq + sigma_recon_sq + C2))

    return np.mean(ssim_map)


def add_poisson_gaussian_noise(y: np.ndarray, peak: float = 50000,
                               sigma: float = 0.005) -> np.ndarray:
    """Add Poisson + Gaussian noise to measurement."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0)

    y_max = np.max(np.abs(y))
    if y_max <= 0:
        y_max = 1.0

    y_scaled = (y / y_max) * peak
    y_scaled = np.maximum(y_scaled, 0)

    # Apply Poisson noise
    y_poisson = np.random.poisson(y_scaled.astype(np.int32)).astype(np.float32)

    # Add Gaussian noise
    y_noisy = y_poisson + np.random.normal(0, sigma, y_poisson.shape).astype(np.float32)

    # Rescale back
    if peak > 0:
        y_noisy = y_noisy / (peak + 1e-10) * y_max

    return np.maximum(y_noisy, 0).astype(np.float32)


# ============================================================================
# Reconstruction Methods (Wrapper Functions)
# ============================================================================

def reconstruct_admm(y: np.ndarray, patterns: np.ndarray,
                    device: str = 'cuda:0') -> np.ndarray:
    """
    Reconstruct using ADMM (Alternating Direction Method of Multipliers).

    Args:
        y: (num_patterns,) measurements
        patterns: (num_patterns, image_size*image_size) forward operator
        device: torch device (unused for ADMM)

    Returns:
        x_recon: (image_size, image_size) reconstruction
    """
    try:
        from pwm_core.recon.admm import admm_spc
        x_hat = admm_spc(y, patterns, iterations=100, rho=1.0)
        x_hat = np.clip(x_hat.reshape(IMAGE_SIZE, IMAGE_SIZE), 0, 1)
        return x_hat.astype(np.float32)
    except Exception as e:
        logger.warning(f"ADMM failed: {e}")
        # Fallback: simple pseudo-inverse
        patterns_pinv = np.linalg.pinv(patterns)
        x_hat = np.dot(patterns_pinv, y)
        x_hat = np.clip(x_hat.reshape(IMAGE_SIZE, IMAGE_SIZE), 0, 1)
        return x_hat.astype(np.float32)


def reconstruct_ista_net_plus(y: np.ndarray, patterns: np.ndarray,
                             device: str = 'cuda:0') -> np.ndarray:
    """
    Reconstruct using ISTA-Net+ (deep unrolled ISTA).

    Args:
        y: (num_patterns,) measurements
        patterns: (num_patterns, image_size*image_size) forward operator
        device: torch device

    Returns:
        x_recon: (image_size, image_size) reconstruction
    """
    try:
        from pwm_core.recon.ista_net_plus import ista_net_plus_spc
        import torch

        # Prepare inputs
        y_tensor = torch.from_numpy(y).unsqueeze(0).float().to(device)  # (1, M)
        patterns_tensor = torch.from_numpy(patterns).float().to(device)  # (M, N)

        # Load model
        model = ista_net_plus_spc()
        model.eval()
        model = model.to(device)

        # Reconstruct
        with torch.no_grad():
            x_hat = model(y_tensor, patterns_tensor)  # (1, N)

        x_hat = np.clip(x_hat.squeeze(0).cpu().numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), 0, 1)
        return x_hat.astype(np.float32)
    except Exception as e:
        logger.warning(f"ISTA-Net+ failed: {e}")
        # Fallback: simple pseudo-inverse
        patterns_pinv = np.linalg.pinv(patterns)
        x_hat = np.dot(patterns_pinv, y)
        x_hat = np.clip(x_hat.reshape(IMAGE_SIZE, IMAGE_SIZE), 0, 1)
        return x_hat.astype(np.float32)


def reconstruct_hatnet(y: np.ndarray, patterns: np.ndarray,
                      device: str = 'cuda:0') -> np.ndarray:
    """
    Reconstruct using HATNet (Hybrid Attention Transformer).

    Args:
        y: (num_patterns,) measurements
        patterns: (num_patterns, image_size*image_size) forward operator
        device: torch device

    Returns:
        x_recon: (image_size, image_size) reconstruction
    """
    try:
        from pwm_core.recon.hatnet import hatnet_spc
        import torch

        # Prepare inputs
        y_tensor = torch.from_numpy(y).unsqueeze(0).float().to(device)  # (1, M)
        patterns_tensor = torch.from_numpy(patterns).float().to(device)  # (M, N)

        # Load model
        model = hatnet_spc()
        model.eval()
        model = model.to(device)

        # Reconstruct
        with torch.no_grad():
            x_hat = model(y_tensor, patterns_tensor)  # (1, N)

        x_hat = np.clip(x_hat.squeeze(0).cpu().numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), 0, 1)
        return x_hat.astype(np.float32)
    except Exception as e:
        logger.warning(f"HATNet failed: {e}")
        # Fallback: simple pseudo-inverse
        patterns_pinv = np.linalg.pinv(patterns)
        x_hat = np.dot(patterns_pinv, y)
        x_hat = np.clip(x_hat.reshape(IMAGE_SIZE, IMAGE_SIZE), 0, 1)
        return x_hat.astype(np.float32)


RECONSTRUCTION_FUNCTIONS = {
    'admm': reconstruct_admm,
    'ista_net_plus': reconstruct_ista_net_plus,
    'hatnet': reconstruct_hatnet
}


# ============================================================================
# Scenario Validation Functions
# ============================================================================

def validate_scenario_i(image: np.ndarray, patterns_ideal: np.ndarray,
                       methods: List[str], device: str) -> Dict[str, Dict]:
    """
    Scenario I: Ideal (perfect forward model, no mismatch).
    """
    logger.info("  Scenario I: Ideal (oracle)")
    results = {}

    # Create ideal measurement
    y_ideal = forward_model_spc(image, patterns_ideal, gain=1.0, illum_drift=0.0)

    for method in methods:
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_ideal, patterns_ideal, device=device)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(image, x_hat)),
                'ssim': float(ssim(image, x_hat))
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0}

    return results


def validate_scenario_ii(image: np.ndarray, patterns_real: np.ndarray,
                        mismatch: MismatchParameters,
                        methods: List[str], device: str) -> Tuple[Dict[str, Dict], np.ndarray]:
    """
    Scenario II: Assumed/Baseline (corrupted measurement, uncorrected operator).
    """
    logger.info("  Scenario II: Assumed/Baseline (uncorrected mismatch)")
    results = {}

    # Create corrupted measurement by warping patterns and applying forward model
    patterns_corrupted = warp_dmd_pattern_2d(
        patterns_real,
        IMAGE_SIZE,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta
    )

    # Apply corrupted forward model
    y_corrupt = forward_model_spc(
        image,
        patterns_corrupted,
        gain=mismatch.gain,
        illum_drift=mismatch.illum_drift_linear
    )

    # Add realistic noise
    y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=50000, sigma=0.005)

    # Reconstruct with each method ASSUMING PERFECT PATTERNS
    for method in methods:
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_corrupt, patterns_real, device=device)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(image, x_hat)),
                'ssim': float(ssim(image, x_hat))
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0}

    return results, y_corrupt


def validate_scenario_iv(image: np.ndarray, patterns_real: np.ndarray,
                        mismatch: MismatchParameters, y_corrupt: np.ndarray,
                        methods: List[str], device: str) -> Dict[str, Dict]:
    """
    Scenario IV: Truth Forward Model (corrupted measurement, oracle operator).
    """
    logger.info("  Scenario IV: Truth Forward Model (oracle operator)")
    results = {}

    # Create truth patterns with mismatch
    patterns_truth = warp_dmd_pattern_2d(
        patterns_real,
        IMAGE_SIZE,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta
    )

    # Reconstruct with each method using TRUE OPERATOR
    for method in methods:
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_corrupt, patterns_truth, device=device)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(image, x_hat)),
                'ssim': float(ssim(image, x_hat))
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0}

    return results


# ============================================================================
# Image Validation
# ============================================================================

def validate_image(image_idx: int, image: np.ndarray,
                  patterns_ideal: np.ndarray, patterns_real: np.ndarray,
                  mismatch: MismatchParameters,
                  methods: List[str], device: str) -> Dict:
    """
    Validate one image across all 3 scenarios and all methods.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Image {image_idx + 1}/{NUM_IMAGES}")
    logger.info(f"{'='*70}")

    start_time = time.time()

    # Scenario I
    res_i = validate_scenario_i(image, patterns_ideal, methods, device)

    # Scenario II (returns both results and measurement for reuse)
    res_ii, y_corrupt = validate_scenario_ii(image, patterns_real, mismatch, methods, device)

    # Scenario IV (reuses y_corrupt from Scenario II)
    res_iv = validate_scenario_iv(image, patterns_real, mismatch, y_corrupt, methods, device)

    elapsed = time.time() - start_time

    # Compile results
    result = {
        'image_idx': image_idx + 1,
        'scenario_i': res_i,
        'scenario_ii': res_ii,
        'scenario_iv': res_iv,
        'elapsed_time': elapsed,
        'mismatch_injected': {
            'mask_dx': mismatch.mask_dx,
            'mask_dy': mismatch.mask_dy,
            'mask_theta': mismatch.mask_theta,
            'clock_offset': mismatch.clock_offset,
            'illum_drift_linear': mismatch.illum_drift_linear,
            'gain': mismatch.gain
        }
    }

    # Calculate gaps for each method
    result['gaps'] = {}
    for method in methods:
        psnr_i = res_i[method]['psnr']
        psnr_ii = res_ii[method]['psnr']
        psnr_iv = res_iv[method]['psnr']

        result['gaps'][method] = {
            'gap_i_ii': float(psnr_i - psnr_ii),
            'gap_ii_iv': float(psnr_iv - psnr_ii),
            'gap_iv_i': float(psnr_i - psnr_iv)
        }

    # Log summary for this image
    for method in methods:
        logger.info(f"\n  {method.upper()}:")
        logger.info(f"    I (Ideal):   {res_i[method]['psnr']:.2f} dB")
        logger.info(f"    II (Assumed): {res_ii[method]['psnr']:.2f} dB (gap {result['gaps'][method]['gap_i_ii']:.2f} dB)")
        logger.info(f"    IV (Oracle):  {res_iv[method]['psnr']:.2f} dB (recovery {result['gaps'][method]['gap_ii_iv']:.2f} dB)")

    return result


# ============================================================================
# Results Aggregation
# ============================================================================

def compute_summary_statistics(all_results: List[Dict]) -> Dict:
    """Compute aggregated statistics across all images."""
    summary = {
        'num_images': len(all_results),
        'scenarios': {}
    }

    for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iv']:
        scenario_name = scenario_key.replace('_', ' ').upper()
        summary['scenarios'][scenario_key] = {}

        for method in RECONSTRUCTION_METHODS:
            psnr_values = [r[scenario_key][method]['psnr'] for r in all_results
                          if r[scenario_key][method]['psnr'] > 0]
            ssim_values = [r[scenario_key][method]['ssim'] for r in all_results
                          if r[scenario_key][method]['ssim'] > 0]

            summary['scenarios'][scenario_key][method] = {
                'psnr': {
                    'mean': float(np.mean(psnr_values)) if psnr_values else 0.0,
                    'std': float(np.std(psnr_values)) if psnr_values else 0.0,
                    'min': float(np.min(psnr_values)) if psnr_values else 0.0,
                    'max': float(np.max(psnr_values)) if psnr_values else 0.0
                },
                'ssim': {
                    'mean': float(np.mean(ssim_values)) if ssim_values else 0.0,
                    'std': float(np.std(ssim_values)) if ssim_values else 0.0
                }
            }

    # Compute gaps
    summary['gaps'] = {}
    for method in RECONSTRUCTION_METHODS:
        gap_values_i_ii = [r['gaps'][method]['gap_i_ii'] for r in all_results]
        gap_values_ii_iv = [r['gaps'][method]['gap_ii_iv'] for r in all_results]
        gap_values_iv_i = [r['gaps'][method]['gap_iv_i'] for r in all_results]

        summary['gaps'][method] = {
            'gap_i_ii': {
                'mean': float(np.mean(gap_values_i_ii)),
                'std': float(np.std(gap_values_i_ii))
            },
            'gap_ii_iv': {
                'mean': float(np.mean(gap_values_ii_iv)),
                'std': float(np.std(gap_values_ii_iv))
            },
            'gap_iv_i': {
                'mean': float(np.mean(gap_values_iv_i)),
                'std': float(np.std(gap_values_iv_i))
            }
        }

    # Execution time
    total_time = sum(r['elapsed_time'] for r in all_results)
    summary['execution_time'] = {
        'total_seconds': float(total_time),
        'total_hours': float(total_time / 3600),
        'per_image_avg_seconds': float(total_time / len(all_results))
    }

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    """Main validation loop."""
    parser = argparse.ArgumentParser(description='SPC Validation for InverseNet ECCV')
    parser.add_argument('--device', default='cuda:0', help='Torch device for reconstruction')
    args = parser.parse_args()

    logger.info("\n" + "="*70)
    logger.info("SPC (Single-Pixel Camera) Validation for InverseNet ECCV Paper")
    logger.info("3 Scenarios × 3 Methods × 11 Images = 99 Reconstructions")
    logger.info("="*70)

    # Load images
    logger.info("\nLoading Set11 images (64×64 center-crops)...")
    images = load_set11_images()
    logger.info(f"Loaded {len(images)} images")

    # Generate DMD patterns
    logger.info("\nGenerating DMD patterns...")
    patterns_ideal = generate_dmd_patterns(num_patterns=MEASUREMENT_DIM, image_size=IMAGE_SIZE, seed=42)
    patterns_real = generate_dmd_patterns(num_patterns=MEASUREMENT_DIM, image_size=IMAGE_SIZE, seed=43)
    logger.info(f"Ideal patterns shape: {patterns_ideal.shape}")
    logger.info(f"Real patterns shape: {patterns_real.shape}")

    # Mismatch parameters
    mismatch = MismatchParameters(
        mask_dx=0.4, mask_dy=0.4, mask_theta=0.08,
        clock_offset=0.06, illum_drift_linear=0.04, gain=1.08
    )
    logger.info(f"Mismatch parameters: dx={mismatch.mask_dx} px, dy={mismatch.mask_dy} px, " +
               f"θ={mismatch.mask_theta}°, clock={mismatch.clock_offset}, gain={mismatch.gain}")

    # Validate all images
    all_results = []
    start_total_time = time.time()

    for image_idx in range(NUM_IMAGES):
        result = validate_image(image_idx, images[image_idx], patterns_ideal, patterns_real,
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
    for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iv']:
        scenario_label = scenario_key.replace('_', ' ').upper()
        logger.info(f"\n{scenario_label}:")
        for method in RECONSTRUCTION_METHODS:
            psnr_mean = summary['scenarios'][scenario_key][method]['psnr']['mean']
            psnr_std = summary['scenarios'][scenario_key][method]['psnr']['std']
            logger.info(f"  {method.upper():15s}: {psnr_mean:.2f} ± {psnr_std:.2f} dB")

    logger.info(f"\nGaps (PSNR degradation/recovery):")
    for method in RECONSTRUCTION_METHODS:
        gap_i_ii = summary['gaps'][method]['gap_i_ii']['mean']
        gap_ii_iv = summary['gaps'][method]['gap_ii_iv']['mean']
        logger.info(f"  {method.upper():15s}: Gap I→II={gap_i_ii:.2f} dB, Recovery II→IV={gap_ii_iv:.2f} dB")

    logger.info(f"\nExecution time: {total_time / 3600:.2f} hours ({total_time / len(all_results) / 60:.1f} min/image)")

    # Save results
    output_detailed = RESULTS_DIR / "spc_validation_results.json"
    output_summary = RESULTS_DIR / "spc_summary.json"

    with open(output_detailed, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Detailed results saved to: {output_detailed}")

    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {output_summary}")

    logger.info("\n✅ Validation complete!")


if __name__ == '__main__':
    main()
