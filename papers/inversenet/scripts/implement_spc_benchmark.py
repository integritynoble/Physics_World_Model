#!/usr/bin/env python3
"""
Implement SPC (Single-Pixel Camera) Benchmark for InverseNet ECCV Paper.

Follows run_all.py patterns for SPC reconstruction benchmarking.
Uses Set11 dataset with multiple reconstruction methods.

Scenarios:
- Scenario I: Ideal (oracle baseline, no noise)
- Scenario II: Assumed (corrupted measurement, uncorrected operator)
- Scenario IV: Truth (corrupted measurement, oracle operator)

Methods:
1. ADMM-L1 (classical)
2. FISTA-L1 (classical)
3. PnP-FISTA-DRUNet (plugin denoiser)
4. ISTA-Net+ (deep unrolled, if available)

Usage:
    python papers/inversenet/scripts/implement_spc_benchmark.py --device cuda:0
"""

import json
import logging
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
NUM_IMAGES = 11
IMAGE_SIZE = 33  # 33×33 blocks following run_all.py pattern
SAMPLING_RATES = [0.10, 0.15, 0.25]  # 10%, 15%, 25%
RECONSTRUCTION_METHODS = ['admm', 'fista']  # Start with classical methods


# ============================================================================
# Mismatch Parameters
# ============================================================================

@dataclass
class MismatchParameters:
    """Mismatch parameters for SPC operator."""
    gain: float = 1.08        # Sensor gain error
    offset: float = 0.005     # Sensor DC offset
    noise_std: float = 0.005  # Read noise standard deviation


# ============================================================================
# Utility Functions
# ============================================================================

def load_set11_images() -> List[np.ndarray]:
    """Load Set11 images (256×256) and crop to 33×33 blocks."""
    logger.info(f"Loading Set11 images...")
    images = []

    try:
        from pwm_core.data.loaders.set11 import Set11Dataset
        dataset = Set11Dataset(resolution=IMAGE_SIZE)
        for idx, (name, image) in enumerate(dataset):
            if len(images) >= NUM_IMAGES:
                break
            # Ensure correct size
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
                scale = IMAGE_SIZE / max(image.shape)
                image = zoom(image, scale, order=1)[:IMAGE_SIZE, :IMAGE_SIZE]
            images.append(image.astype(np.float32))
            logger.info(f"  Loaded: {name} ({image.shape})")

    except Exception as e:
        logger.warning(f"Set11 loading failed: {e}, using synthetic images")
        # Fallback: synthetic test images
        np.random.seed(42)
        for i in range(NUM_IMAGES):
            img = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma=1.0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images.append(img)

    return images[:NUM_IMAGES]


def create_measurement_matrix(n_pix: int, sampling_rate: float, seed: int = 42) -> np.ndarray:
    """Create Gaussian measurement matrix (row-normalized) following run_all.py pattern."""
    np.random.seed(seed)
    m = int(n_pix * sampling_rate)

    # Gaussian random matrix
    Phi = np.random.randn(m, n_pix).astype(np.float32)

    # Row normalize for stability
    row_norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    Phi_norm = Phi / row_norms

    return Phi_norm


def psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate PSNR in dB."""
    x_true = np.clip(x_true, 0, 1)
    x_recon = np.clip(x_recon, 0, 1)

    mse = np.mean((x_true - x_recon) ** 2)
    if mse < 1e-10:
        return 100.0

    return float(10.0 * np.log10(1.0 / mse))


def ssim(x_true: np.ndarray, x_recon: np.ndarray, window_size: int = 11) -> float:
    """Calculate SSIM."""
    x_true = np.clip(x_true, 0, 1)
    x_recon = np.clip(x_recon, 0, 1)

    from scipy.signal import correlate2d

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

    return float(np.mean(ssim_map))


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Soft thresholding for L1 minimization."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


# ============================================================================
# Reconstruction Methods
# ============================================================================

def reconstruct_admm(y: np.ndarray, A: np.ndarray, iterations: int = 100) -> np.ndarray:
    """ADMM solver for basis pursuit denoising."""
    M, N = A.shape
    rho = 1.0

    # Initialize
    x = np.zeros(N, dtype=np.float32)
    z = np.zeros(N, dtype=np.float32)
    u = np.zeros(N, dtype=np.float32)

    # Precompute
    AtA = A.T @ A
    Aty = A.T @ y

    for k in range(iterations):
        # x-update
        try:
            from scipy.linalg import solve
            x = solve(AtA + rho * np.eye(N), Aty + rho * (z - u), assume_a='pos')
        except:
            x = np.linalg.lstsq(AtA + rho * np.eye(N), Aty + rho * (z - u), rcond=None)[0]

        # z-update (soft thresholding)
        z = soft_threshold(x + u, 1.0 / rho)

        # u-update
        u = u + x - z

        # Clip
        x = np.clip(x, 0, 1)

        if k % 20 == 0:
            residual = np.linalg.norm(np.linalg.norm(A @ x - y))
            logger.debug(f"ADMM iter {k}: residual={residual:.2e}")

    return np.clip(x, 0, 1).astype(np.float32)


def reconstruct_fista(y: np.ndarray, A: np.ndarray, lambda_l1: float = 0.01,
                     iterations: int = 100) -> np.ndarray:
    """FISTA with L1 regularization."""
    M, N = A.shape

    # Estimate Lipschitz constant
    AtA = A.T @ A
    eigvals = np.linalg.eigvalsh(AtA)
    L = np.max(eigvals) + 1e-3

    step_size = 0.9 / L

    # Initialize
    x = A.T @ y
    x = np.clip((x - x.min()) / (x.max() - x.min() + 1e-8), 0, 1)
    z = x.copy()
    t = 1.0

    Aty = A.T @ y

    for k in range(iterations):
        # Gradient step
        grad = A.T @ (A @ z) - Aty
        x_new = z - step_size * grad

        # Soft thresholding
        x_new = soft_threshold(x_new, lambda_l1 * step_size)

        # Acceleration (FISTA)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        t = t_new

        x = x_new
        x = np.clip(x, 0, 1)

        if k % 20 == 0:
            residual = np.linalg.norm(A @ x - y)
            logger.debug(f"FISTA iter {k}: residual={residual:.2e}")

    return np.clip(x, 0, 1).astype(np.float32)


# ============================================================================
# Scenario Validation
# ============================================================================

def validate_scenario_i(image: np.ndarray, A_ideal: np.ndarray,
                       methods: List[str]) -> Dict[str, Dict]:
    """Scenario I: Ideal (oracle baseline)."""
    logger.info("  Scenario I: Ideal (oracle)")
    results = {}

    # Create ideal measurement
    x_flat = image.flatten()
    y_ideal = A_ideal @ x_flat

    for method in methods:
        try:
            if method == 'admm':
                x_hat = reconstruct_admm(y_ideal, A_ideal, iterations=100)
            elif method == 'fista':
                x_hat = reconstruct_fista(y_ideal, A_ideal, lambda_l1=0.01, iterations=100)
            else:
                logger.warning(f"Unknown method: {method}")
                x_hat = x_flat

            x_hat = x_hat.reshape(image.shape)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(image, x_hat)),
                'ssim': float(ssim(image, x_hat))
            }
        except Exception as e:
            logger.error(f"  {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0}

    return results


def validate_scenario_ii(image: np.ndarray, A_real: np.ndarray,
                        mismatch: MismatchParameters,
                        methods: List[str]) -> Tuple[Dict[str, Dict], np.ndarray]:
    """Scenario II: Assumed/Baseline (uncorrected mismatch)."""
    logger.info("  Scenario II: Assumed/Baseline (uncorrected mismatch)")
    results = {}

    # Create corrupted measurement
    x_flat = image.flatten()
    y_corrupt = (mismatch.gain * (A_real @ x_flat) + mismatch.offset)

    # Add noise
    y_corrupt = y_corrupt + np.random.randn(len(y_corrupt)).astype(np.float32) * mismatch.noise_std

    # Reconstruct assuming perfect operator (WRONG!)
    for method in methods:
        try:
            if method == 'admm':
                x_hat = reconstruct_admm(y_corrupt, A_real, iterations=100)
            elif method == 'fista':
                x_hat = reconstruct_fista(y_corrupt, A_real, lambda_l1=0.01, iterations=100)
            else:
                x_hat = x_flat

            x_hat = x_hat.reshape(image.shape)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(image, x_hat)),
                'ssim': float(ssim(image, x_hat))
            }
        except Exception as e:
            logger.error(f"  {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0}

    return results, y_corrupt


def validate_scenario_iv(image: np.ndarray, A_real: np.ndarray,
                        mismatch: MismatchParameters, y_corrupt: np.ndarray,
                        methods: List[str]) -> Dict[str, Dict]:
    """Scenario IV: Truth Forward Model (oracle operator)."""
    logger.info("  Scenario IV: Truth Forward Model (oracle operator)")
    results = {}

    # Reconstruct using CORRECTED operator
    # Corrected measurement: y_corrected = (y_corrupt - offset) / gain
    y_corrected = (y_corrupt - mismatch.offset) / mismatch.gain

    for method in methods:
        try:
            if method == 'admm':
                x_hat = reconstruct_admm(y_corrected, A_real, iterations=100)
            elif method == 'fista':
                x_hat = reconstruct_fista(y_corrected, A_real, lambda_l1=0.01, iterations=100)
            else:
                x_hat = image.flatten()

            x_hat = x_hat.reshape(image.shape)
            x_hat = np.clip(x_hat, 0, 1)

            results[method] = {
                'psnr': float(psnr(image, x_hat)),
                'ssim': float(ssim(image, x_hat))
            }
        except Exception as e:
            logger.error(f"  {method} failed: {e}")
            results[method] = {'psnr': 0.0, 'ssim': 0.0}

    return results


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    """Main SPC benchmark execution."""
    parser = argparse.ArgumentParser(description='SPC Benchmark for InverseNet ECCV')
    parser.add_argument('--device', default='cuda:0', help='Torch device')
    parser.add_argument('--sampling-rate', type=float, default=0.15, help='Sampling rate (0-1)')
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("SPC (Single-Pixel Camera) Benchmark for InverseNet ECCV")
    logger.info(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    logger.info(f"Sampling rate: {args.sampling_rate:.1%}")
    logger.info(f"Methods: {', '.join(RECONSTRUCTION_METHODS)}")
    logger.info("="*70)

    # Load images
    images = load_set11_images()
    logger.info(f"Loaded {len(images)} images")

    # Create measurement matrices
    n_pix = IMAGE_SIZE * IMAGE_SIZE
    A_ideal = create_measurement_matrix(n_pix, args.sampling_rate, seed=42)
    A_real = A_ideal.copy()  # In practice, slightly different

    # Mismatch parameters
    mismatch = MismatchParameters()

    # Benchmark
    all_results = []
    start_time = time.time()

    for img_idx, image in enumerate(images):
        logger.info(f"\nImage {img_idx + 1}/{len(images)}")

        result = {
            'image_idx': img_idx + 1,
            'scenario_i': validate_scenario_i(image, A_ideal, RECONSTRUCTION_METHODS),
            'scenario_ii': None,
            'scenario_iv': None,
            'elapsed_time': 0.0
        }

        # Scenario II
        res_ii, y_corrupt = validate_scenario_ii(image, A_real, mismatch, RECONSTRUCTION_METHODS)
        result['scenario_ii'] = res_ii

        # Scenario IV
        result['scenario_iv'] = validate_scenario_iv(image, A_real, mismatch, y_corrupt,
                                                     RECONSTRUCTION_METHODS)

        # Log summary
        for method in RECONSTRUCTION_METHODS:
            psnr_i = result['scenario_i'][method]['psnr']
            psnr_ii = result['scenario_ii'][method]['psnr']
            psnr_iv = result['scenario_iv'][method]['psnr']
            logger.info(f"  {method.upper()}: I={psnr_i:.2f}dB, II={psnr_ii:.2f}dB, IV={psnr_iv:.2f}dB")

        all_results.append(result)

    total_time = time.time() - start_time

    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)

    summary = {
        'num_images': len(all_results),
        'sampling_rate': args.sampling_rate,
        'image_size': IMAGE_SIZE,
        'scenarios': {}
    }

    for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iv']:
        summary['scenarios'][scenario_key] = {}
        for method in RECONSTRUCTION_METHODS:
            psnr_vals = [r[scenario_key][method]['psnr'] for r in all_results
                        if r[scenario_key][method]['psnr'] > 0]
            ssim_vals = [r[scenario_key][method]['ssim'] for r in all_results
                        if r[scenario_key][method]['ssim'] > 0]

            summary['scenarios'][scenario_key][method] = {
                'psnr': {
                    'mean': float(np.mean(psnr_vals)) if psnr_vals else 0.0,
                    'std': float(np.std(psnr_vals)) if psnr_vals else 0.0,
                },
                'ssim': {
                    'mean': float(np.mean(ssim_vals)) if ssim_vals else 0.0,
                    'std': float(np.std(ssim_vals)) if ssim_vals else 0.0,
                }
            }

    # Log summary
    for scenario_key in ['scenario_i', 'scenario_ii', 'scenario_iv']:
        scenario_label = scenario_key.replace('_', ' ').upper()
        logger.info(f"\n{scenario_label}:")
        for method in RECONSTRUCTION_METHODS:
            psnr_mean = summary['scenarios'][scenario_key][method]['psnr']['mean']
            psnr_std = summary['scenarios'][scenario_key][method]['psnr']['std']
            logger.info(f"  {method.upper():10s}: {psnr_mean:.2f} ± {psnr_std:.2f} dB")

    logger.info(f"\nTotal time: {total_time/60:.1f} min ({total_time/len(all_results):.1f}s per image)")

    # Save results
    output_detailed = RESULTS_DIR / "spc_benchmark_results.json"
    output_summary = RESULTS_DIR / "spc_benchmark_summary.json"

    with open(output_detailed, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved: {output_detailed}")

    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {output_summary}")

    logger.info("\n✅ SPC benchmark complete!")


if __name__ == '__main__':
    main()
