#!/usr/bin/env python3
"""
SPC (Single-Pixel Camera) Validation for InverseNet ECCV Paper — v3.0

Block-based CS on 256×256 Set11 images with 64×64 blocks, 25% compression.
Uses Hadamard measurement matrix + FISTA-TV solver (primary) and
PnP-FISTA with DRUNet denoiser (ISTA-Net+, HATNet proxies).

Key parameters (matching spc_plan_inversenet.md v2.0):
  - Full image size: 256×256 (native Set11 resolution, no cropping)
  - Block size: 64×64 (N=4096 pixels per block, N=2^12 Hadamard-compatible)
  - Blocks per image: 16 (4×4 non-overlapping grid)
  - Compression ratio: 25% (M=1024 measurements per block)
  - Measurement matrix: Hadamard (subsampled rows, orthonormal)
  - Solvers: FISTA-TV (classical), PnP-FISTA+DRUNet (deep learning proxy)

Scenarios:
  - Scenario I:   Ideal measurement + ideal operator
  - Scenario II:  Corrupted measurement + assumed perfect operator (baseline)
  - Scenario III: Corrupted measurement + truth operator (oracle calibration)

Usage:
    python validate_spc_inversenet.py
    python validate_spc_inversenet.py --max-iters 500
"""

import json
import logging
import math
import time
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.linalg import hadamard

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError:
    denoise_tv_chambolle = None

# Deep learning imports (optional)
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))
SET11_DIR = Path("/home/spiritai/ISTA-Net-PyTorch-master/data/Set11")
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Constants
# ============================================================================
FULL_IMAGE_SIZE = 256       # Full Set11 image resolution
BLOCK_SIZE = 64             # Block size for block-based CS
N_PIX = BLOCK_SIZE * BLOCK_SIZE  # 4096 pixels per block
BLOCKS_PER_ROW = FULL_IMAGE_SIZE // BLOCK_SIZE  # 4
BLOCKS_PER_IMAGE = BLOCKS_PER_ROW * BLOCKS_PER_ROW  # 16
DEFAULT_SAMPLING_RATE = 0.25
NUM_IMAGES = 11
NOISE_LEVEL = 0.01          # Gaussian noise std

SCENARIOS = ['scenario_i', 'scenario_ii', 'scenario_iii']


# ============================================================================
# Mismatch Parameters (from spc_plan_inversenet.md)
# ============================================================================
@dataclass
class MismatchParameters:
    """Mismatch parameters for SPC operator."""
    mask_dx: float = 0.4          # pixels
    mask_dy: float = 0.4          # pixels
    mask_theta: float = 0.08      # degrees
    clock_offset: float = 0.06    # pattern duration
    illum_drift_linear: float = 0.04  # fraction/sequence
    gain: float = 1.08            # sensor gain ratio


# ============================================================================
# Image Loading
# ============================================================================
def load_set11_images_256() -> List[Tuple[str, np.ndarray]]:
    """Load Set11 images at native 256×256 resolution."""
    images = []

    if SET11_DIR.exists():
        import glob
        image_files = sorted(
            glob.glob(str(SET11_DIR / "*.tif")) +
            glob.glob(str(SET11_DIR / "*.png")) +
            glob.glob(str(SET11_DIR / "*.jpg"))
        )
        if image_files:
            from PIL import Image
            for img_path in image_files[:NUM_IMAGES]:
                try:
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    h, w = img_array.shape
                    if h != FULL_IMAGE_SIZE or w != FULL_IMAGE_SIZE:
                        from scipy.ndimage import zoom
                        scale_h = FULL_IMAGE_SIZE / h
                        scale_w = FULL_IMAGE_SIZE / w
                        img_array = zoom(img_array, (scale_h, scale_w), order=3)
                        img_array = np.clip(img_array[:FULL_IMAGE_SIZE, :FULL_IMAGE_SIZE], 0, 1)
                    name = Path(img_path).stem
                    images.append((name, img_array.astype(np.float32)))
                    logger.info(f"  Loaded: {name} ({img_array.shape})")
                except Exception as e:
                    logger.warning(f"  Failed to load {img_path}: {e}")

    if not images:
        logger.warning(f"Set11 not found at {SET11_DIR}, generating synthetic images")
        from scipy.ndimage import gaussian_filter
        np.random.seed(123)
        synth_names = [
            'gaussian_blob', 'circles', 'stripes', 'checkerboard', 'phantom',
            'gradient', 'sinusoidal', 'block_letter', 'crosshatch', 'gabor', 'speckle'
        ]
        for i, name in enumerate(synth_names):
            img = np.random.rand(FULL_IMAGE_SIZE, FULL_IMAGE_SIZE).astype(np.float32)
            img = gaussian_filter(img, sigma=8.0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            images.append((name, img))

    return images[:NUM_IMAGES]


# ============================================================================
# Block Partitioning
# ============================================================================
def partition_into_blocks(image: np.ndarray) -> List[np.ndarray]:
    """Partition a 256×256 image into 16 non-overlapping 64×64 blocks."""
    blocks = []
    for bi in range(BLOCKS_PER_ROW):
        for bj in range(BLOCKS_PER_ROW):
            block = image[bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE,
                          bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE].copy()
            blocks.append(block)
    return blocks


def stitch_blocks(blocks: List[np.ndarray]) -> np.ndarray:
    """Stitch 16 blocks of 64×64 back into a 256×256 image."""
    image = np.zeros((FULL_IMAGE_SIZE, FULL_IMAGE_SIZE), dtype=np.float32)
    idx = 0
    for bi in range(BLOCKS_PER_ROW):
        for bj in range(BLOCKS_PER_ROW):
            image[bi*BLOCK_SIZE:(bi+1)*BLOCK_SIZE,
                  bj*BLOCK_SIZE:(bj+1)*BLOCK_SIZE] = blocks[idx]
            idx += 1
    return image


# ============================================================================
# Measurement Matrix
# ============================================================================
def create_measurement_matrix(m: int, n: int, seed: int = 42) -> np.ndarray:
    """Create Hadamard measurement matrix (subsampled rows).

    Hadamard is the standard for SPC: orthonormal rows give better incoherence
    and conditioning than random Gaussian. N=4096=2^12.
    """
    H = hadamard(n).astype(np.float64) / np.sqrt(n)
    np.random.seed(seed)
    rows = np.sort(np.random.choice(n, m, replace=False))
    return H[rows, :].astype(np.float32)


# ============================================================================
# Mismatch Injection
# ============================================================================
def apply_mismatch_to_matrix(Phi: np.ndarray, mismatch: MismatchParameters) -> np.ndarray:
    """Apply mismatch to measurement matrix (creates A_real from A_ideal)."""
    from scipy.ndimage import affine_transform

    m, n = Phi.shape
    Phi_real = np.zeros_like(Phi)

    for i in range(m):
        pattern_2d = Phi[i].reshape(BLOCK_SIZE, BLOCK_SIZE)
        center = BLOCK_SIZE / 2.0
        theta_rad = np.radians(mismatch.mask_theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        offset_x = -center * cos_t - center * sin_t + center + mismatch.mask_dx
        offset_y = center * sin_t - center * cos_t + center + mismatch.mask_dy
        matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        offset = np.array([offset_x, offset_y])
        pattern_warped = affine_transform(pattern_2d, matrix, offset=offset,
                                          order=1, mode='constant', cval=0.0)
        Phi_real[i] = pattern_warped.flatten()

    # Apply gain
    Phi_real *= mismatch.gain

    # Apply illumination drift
    drift = np.linspace(1.0 - mismatch.illum_drift_linear,
                        1.0 + mismatch.illum_drift_linear, m)
    Phi_real = Phi_real * drift[:, np.newaxis]

    return Phi_real.astype(np.float32)


# ============================================================================
# Lipschitz Constant Estimation
# ============================================================================
def estimate_lipschitz(Phi: np.ndarray, n_iters: int = 20) -> float:
    """Estimate Lipschitz constant of Phi^T @ Phi via power iteration."""
    n = Phi.shape[1]
    v = np.random.randn(n).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    for _ in range(n_iters):
        w = Phi.T @ (Phi @ v)
        w_norm = np.linalg.norm(w) + 1e-12
        v = w / w_norm
    w = Phi @ v
    s = np.linalg.norm(w)
    return float(s * s)


# ============================================================================
# Metrics
# ============================================================================
def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """PSNR in dB, data assumed in [0, 1]."""
    x_true = np.clip(x_true.astype(np.float64), 0, 1)
    x_recon = np.clip(x_recon.astype(np.float64), 0, 1)
    mse = np.mean((x_true - x_recon) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """SSIM on 2D images."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            x_true.astype(np.float64), x_recon.astype(np.float64),
            data_range=1.0))
    except ImportError:
        c1, c2 = 0.01**2, 0.03**2
        mu_x, mu_y = x_true.mean(), x_recon.mean()
        var_x, var_y = x_true.var(), x_recon.var()
        cov = np.mean((x_true - mu_x) * (x_recon - mu_y))
        num = (2*mu_x*mu_y + c1) * (2*cov + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        return float(num / den)


# ============================================================================
# FISTA-TV Solver (following run_all.py:_basic_fista pattern)
# ============================================================================
class FISTATVSolver:
    """FISTA with TV proximal for block-based CS reconstruction.

    Following the pattern from run_all.py:1467-1515 and cs_solvers.py:254-305.
    Uses Nesterov-accelerated gradient descent with TV denoising as proximal.
    No Cholesky needed — just matrix-vector products per iteration.
    """

    def __init__(self, Phi: np.ndarray, lam: float = 0.01,
                 max_iter: int = 300, tv_inner_iters: int = 10):
        self.Phi = Phi.astype(np.float32)
        self.m, self.n = Phi.shape
        self.lam = lam
        self.max_iter = max_iter
        self.tv_inner_iters = tv_inner_iters

        # Estimate Lipschitz constant for step size
        self.L = estimate_lipschitz(Phi, n_iters=20)
        self.tau = 0.9 / max(self.L, 1e-8)
        logger.info(f"  FISTA-TV: L={self.L:.4f}, tau={self.tau:.6f}, "
                    f"lam={lam}, iters={max_iter}")

    def solve(self, y: np.ndarray, **kwargs) -> np.ndarray:
        """Reconstruct one 64×64 block using FISTA-TV."""
        max_iters = kwargs.get('max_iters', self.max_iter)

        # Initialize with backprojection, normalized to [0,1]
        x0 = (self.Phi.T @ y.astype(np.float32))
        x0_min, x0_max = x0.min(), x0.max()
        if x0_max - x0_min > 1e-8:
            x0 = (x0 - x0_min) / (x0_max - x0_min)
        x0 = np.clip(x0, 0, 1)

        x = x0.copy()
        z = x0.copy()
        t = 1.0

        for k in range(max_iters):
            # Gradient step on z
            residual = self.Phi @ z - y.astype(np.float32)
            grad = self.Phi.T @ residual
            u = z - self.tau * grad

            # TV proximal step
            u_img = np.clip(u.reshape(BLOCK_SIZE, BLOCK_SIZE), 0, 1)
            if denoise_tv_chambolle is not None:
                z_new_img = denoise_tv_chambolle(
                    u_img.astype(np.float64),
                    weight=self.tau * self.lam,
                    max_num_iter=self.tv_inner_iters)
            else:
                z_new_img = u_img  # fallback: no TV
            z_new = np.clip(z_new_img, 0, 1).flatten().astype(np.float32)

            # FISTA momentum (Nesterov acceleration)
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            x_new = z_new + ((t - 1.0) / t_new) * (z_new - x)
            x_new = np.clip(x_new, 0, 1)

            x = z_new
            z = x_new
            t = t_new

        return np.clip(x.reshape(BLOCK_SIZE, BLOCK_SIZE), 0, 1).astype(np.float32)


# ============================================================================
# PnP-FISTA with DRUNet Denoiser (for ISTA-Net+ / HATNet proxy)
# ============================================================================
class PnPFISTASolver:
    """PnP-FISTA solver with DRUNet denoiser for deep learning reconstruction.

    Following run_all.py:1384-1466 and test_operator_correction.py:774-852.
    Uses FISTA momentum with annealed DRUNet denoising as the proximal operator.
    """

    def __init__(self, Phi: np.ndarray, denoiser, device,
                 max_iter: int = 200, sigma_start: float = 0.08,
                 sigma_end: float = 0.02, pad_mult: int = 8):
        self.Phi = Phi.astype(np.float32)
        self.denoiser = denoiser
        self.device = device
        self.max_iter = max_iter
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.pad_mult = pad_mult

        # Estimate Lipschitz constant for step size
        self.L = estimate_lipschitz(Phi, n_iters=20)
        self.tau = 0.9 / max(self.L, 1e-8)
        logger.info(f"  PnP-FISTA: L={self.L:.4f}, tau={self.tau:.6f}, "
                    f"iters={max_iter}, sigma=[{sigma_start}→{sigma_end}]")

    def solve(self, y: np.ndarray, **kwargs) -> np.ndarray:
        """Reconstruct one 64x64 block using PnP-FISTA + DRUNet."""
        # Initialize with backprojection, normalized to [0,1]
        x0 = self.Phi.T @ y.astype(np.float32)
        x0_min, x0_max = x0.min(), x0.max()
        if x0_max - x0_min > 1e-8:
            x0 = (x0 - x0_min) / (x0_max - x0_min)
        x0 = np.clip(x0, 0, 1)

        Phi_t = torch.from_numpy(self.Phi).float().to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).float().to(self.device)

        x = x0.copy()
        z = x0.copy()
        t = 1.0

        with torch.no_grad():
            for k in range(self.max_iter):
                # Annealing sigma
                a = k / max(self.max_iter - 1, 1)
                sigma_k = (1 - a) * self.sigma_start + a * self.sigma_end

                # Gradient step on z
                z_t = torch.from_numpy(z).float().to(self.device)
                residual = Phi_t @ z_t - y_t
                grad = Phi_t.T @ residual
                u = z_t - self.tau * grad

                # Reshape for denoiser (BCHW)
                u_img = u.reshape(1, 1, BLOCK_SIZE, BLOCK_SIZE)

                # Pad to multiple of pad_mult for U-Net
                H, W = u_img.shape[-2], u_img.shape[-1]
                Hp = int(math.ceil(H / self.pad_mult) * self.pad_mult)
                Wp = int(math.ceil(W / self.pad_mult) * self.pad_mult)
                pad_h = Hp - H
                pad_w = Wp - W
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top

                u_pad = F.pad(u_img, (pad_left, pad_right, pad_top, pad_bottom),
                              mode="reflect")

                # Denoise
                try:
                    z_pad = self.denoiser(u_pad, sigma=sigma_k)
                except TypeError:
                    try:
                        z_pad = self.denoiser(u_pad, noise_level=sigma_k)
                    except TypeError:
                        z_pad = self.denoiser(u_pad)

                # Crop back
                z_new_img = z_pad[:, :, pad_top:pad_top+H,
                                  pad_left:pad_left+W].contiguous()
                z_new = z_new_img.reshape(-1).clamp(0.0, 1.0).cpu().numpy()

                # FISTA momentum
                t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                x_new = z_new + ((t - 1.0) / t_new) * (z_new - x)
                x_new = np.clip(x_new, 0, 1)

                x = z_new
                z = x_new
                t = t_new

        return x.reshape(BLOCK_SIZE, BLOCK_SIZE).astype(np.float32)


def load_drunet_denoiser():
    """Try to load DRUNet denoiser for PnP-FISTA. Returns (denoiser, device) or (None, None)."""
    if not HAS_TORCH:
        logger.info("  PyTorch not available, deep learning solvers disabled")
        return None, None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        from deepinv.models import DRUNet

        for kwargs in [
            {"in_channels": 1, "out_channels": 1, "pretrained": "download"},
            {"in_channels": 1, "out_channels": 1},
            {"pretrained": "download"},
            {},
        ]:
            try:
                denoiser = DRUNet(**kwargs).to(device).eval()
                logger.info(f"  DRUNet denoiser loaded on {device}")
                return denoiser, device
            except Exception:
                continue
    except ImportError:
        pass

    try:
        from deepinv.models import DnCNN
        denoiser = DnCNN(in_channels=1, out_channels=1,
                         pretrained="download").to(device).eval()
        logger.info(f"  DnCNN denoiser loaded on {device} (DRUNet fallback)")
        return denoiser, device
    except (ImportError, Exception) as e:
        logger.info(f"  No deep learning denoiser available: {e}")
        return None, None


# ============================================================================
# Block-Based Reconstruction
# ============================================================================
def reconstruct_image_blockwise(y_blocks: List[np.ndarray],
                                solver,
                                max_iters: int = 300) -> np.ndarray:
    """Reconstruct all 16 blocks and stitch into 256×256 image."""
    recon_blocks = []
    for y_block in y_blocks:
        block_recon = solver.solve(y_block, max_iters=max_iters)
        recon_blocks.append(block_recon)
    return stitch_blocks(recon_blocks)


# ============================================================================
# Scenario Validation
# ============================================================================
def measure_blocks(blocks: List[np.ndarray], Phi: np.ndarray,
                   noise_level: float = 0.0) -> List[np.ndarray]:
    """Measure all 16 blocks: y_b = Phi @ x_b + noise."""
    measurements = []
    for block in blocks:
        x_flat = block.flatten().astype(np.float32)
        y = Phi @ x_flat
        if noise_level > 0:
            y += np.random.randn(len(y)).astype(np.float32) * noise_level
        measurements.append(y)
    return measurements


def validate_image(image_idx: int, name: str, image: np.ndarray,
                   Phi_ideal: np.ndarray, Phi_real: np.ndarray,
                   solvers_ideal: Dict, solvers_real: Dict,
                   methods: List[str],
                   max_iters: int = 300) -> Dict:
    """Validate one 256×256 image across all 3 scenarios with multiple solvers."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Image {image_idx+1}/{NUM_IMAGES}: {name}")
    logger.info(f"{'='*70}")

    start = time.time()
    blocks = partition_into_blocks(image)

    # --- Scenario I: Ideal ---
    logger.info("  Scenario I: Ideal")
    y_ideal = measure_blocks(blocks, Phi_ideal, noise_level=NOISE_LEVEL)

    res_i = {}
    for method in methods:
        solver = solvers_ideal[method]
        recon = reconstruct_image_blockwise(y_ideal, solver, max_iters)
        res_i[method] = {
            'psnr': compute_psnr(image, recon),
            'ssim': compute_ssim(image, recon),
        }

    # --- Scenario II: Corrupted measurement + ideal operator ---
    logger.info("  Scenario II: Baseline (corrupted meas, ideal operator)")
    y_corrupt = measure_blocks(blocks, Phi_real, noise_level=NOISE_LEVEL)

    res_ii = {}
    for method in methods:
        solver = solvers_ideal[method]
        recon = reconstruct_image_blockwise(y_corrupt, solver, max_iters)
        res_ii[method] = {
            'psnr': compute_psnr(image, recon),
            'ssim': compute_ssim(image, recon),
        }

    # --- Scenario III: Corrupted measurement + truth operator ---
    logger.info("  Scenario III: Oracle (corrupted meas, truth operator)")

    res_iii = {}
    for method in methods:
        solver = solvers_real[method]
        recon = reconstruct_image_blockwise(y_corrupt, solver, max_iters)
        res_iii[method] = {
            'psnr': compute_psnr(image, recon),
            'ssim': compute_ssim(image, recon),
        }

    elapsed = time.time() - start

    result = {
        'image_idx': image_idx + 1,
        'image_name': name,
        'scenario_i': res_i,
        'scenario_ii': res_ii,
        'scenario_iii': res_iii,
        'elapsed_time': elapsed,
        'gaps': {},
    }

    for method in methods:
        pi = res_i[method]['psnr']
        pii = res_ii[method]['psnr']
        piii = res_iii[method]['psnr']
        result['gaps'][method] = {
            'gap_i_ii': float(pi - pii),
            'gap_ii_iii': float(piii - pii),
            'gap_iii_i': float(pi - piii),
        }

    # Log per-method results
    for method in methods:
        pi = res_i[method]['psnr']
        pii = res_ii[method]['psnr']
        piii = res_iii[method]['psnr']
        si = res_i[method]['ssim']
        sii = res_ii[method]['ssim']
        siii = res_iii[method]['ssim']
        logger.info(f"  {method:15s}: I={pi:.2f} | II={pii:.2f} | "
                    f"III={piii:.2f} dB | Gap={pi-pii:.2f} | "
                    f"Recovery={piii-pii:.2f}")
        logger.info(f"  {'':15s}  SSIM: I={si:.4f} | II={sii:.4f} | III={siii:.4f}")

    logger.info(f"  Time: {elapsed:.1f}s")

    return result


# ============================================================================
# Summary Statistics
# ============================================================================
def compute_summary(all_results: List[Dict], methods: List[str],
                    sampling_rate: float, m_measurements: int) -> Dict:
    """Aggregate statistics across all images."""
    summary = {
        'num_images': len(all_results),
        'image_size': FULL_IMAGE_SIZE,
        'block_size': BLOCK_SIZE,
        'blocks_per_image': BLOCKS_PER_IMAGE,
        'sampling_rate': sampling_rate,
        'measurements_per_block': m_measurements,
        'scenarios': {},
        'gaps': {},
    }

    for scenario_key in SCENARIOS:
        summary['scenarios'][scenario_key] = {}
        for method in methods:
            psnrs = [r[scenario_key][method]['psnr'] for r in all_results]
            ssims = [r[scenario_key][method]['ssim'] for r in all_results]
            summary['scenarios'][scenario_key][method] = {
                'psnr': {
                    'mean': float(np.mean(psnrs)),
                    'std': float(np.std(psnrs)),
                    'min': float(np.min(psnrs)),
                    'max': float(np.max(psnrs)),
                },
                'ssim': {
                    'mean': float(np.mean(ssims)),
                    'std': float(np.std(ssims)),
                },
            }

    for method in methods:
        gaps_i_ii = [r['gaps'][method]['gap_i_ii'] for r in all_results]
        gaps_ii_iii = [r['gaps'][method]['gap_ii_iii'] for r in all_results]
        gaps_iii_i = [r['gaps'][method]['gap_iii_i'] for r in all_results]
        summary['gaps'][method] = {
            'gap_i_ii': {'mean': float(np.mean(gaps_i_ii)), 'std': float(np.std(gaps_i_ii))},
            'gap_ii_iii': {'mean': float(np.mean(gaps_ii_iii)), 'std': float(np.std(gaps_ii_iii))},
            'gap_iii_i': {'mean': float(np.mean(gaps_iii_i)), 'std': float(np.std(gaps_iii_i))},
        }

    total_time = sum(r['elapsed_time'] for r in all_results)
    summary['execution_time'] = {
        'total_seconds': float(total_time),
        'total_minutes': float(total_time / 60),
        'per_image_seconds': float(total_time / len(all_results)),
    }

    return summary


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='SPC Validation v3.0 for InverseNet ECCV')
    parser.add_argument('--sampling-rate', type=float, default=0.25,
                        help='Compression ratio (default: 0.25)')
    parser.add_argument('--max-iters', type=int, default=300,
                        help='FISTA-TV iterations (default: 300)')
    args = parser.parse_args()

    sampling_rate = args.sampling_rate
    m_measurements = int(N_PIX * sampling_rate)

    logger.info("\n" + "="*70)
    logger.info("SPC Validation v3.0 for InverseNet ECCV Paper")
    logger.info(f"Image: {FULL_IMAGE_SIZE}×{FULL_IMAGE_SIZE} | "
                f"Block: {BLOCK_SIZE}×{BLOCK_SIZE} | "
                f"Blocks/image: {BLOCKS_PER_IMAGE}")
    logger.info(f"Sampling: {sampling_rate*100:.0f}% | "
                f"M={m_measurements} measurements/block | "
                f"N={N_PIX} pixels/block")
    logger.info(f"Measurement matrix: Hadamard (subsampled rows)")
    logger.info(f"FISTA-TV iterations: {args.max_iters}")
    logger.info("="*70)

    # 1. Load images
    logger.info("\nLoading Set11 images (256×256)...")
    images = load_set11_images_256()
    logger.info(f"Loaded {len(images)} images")

    # 2. Create measurement matrices
    logger.info(f"\nCreating Hadamard measurement matrix: Phi in R^{{{m_measurements}x{N_PIX}}}")
    Phi_ideal = create_measurement_matrix(m_measurements, N_PIX, seed=42)
    logger.info(f"Ideal Phi shape: {Phi_ideal.shape}")

    # 3. Create mismatched operator
    mismatch = MismatchParameters()
    logger.info(f"Mismatch: dx={mismatch.mask_dx} px, dy={mismatch.mask_dy} px, "
                f"theta={mismatch.mask_theta} deg, gain={mismatch.gain}")
    Phi_real = apply_mismatch_to_matrix(Phi_ideal, mismatch)
    diff = np.linalg.norm(Phi_real - Phi_ideal) / np.linalg.norm(Phi_ideal)
    logger.info(f"Relative operator difference: {diff:.4f}")

    # 4. Create solvers per method
    logger.info("\nCreating solvers...")

    # FISTA-TV: classical solver (primary)
    fista_ideal = FISTATVSolver(Phi_ideal, lam=0.01, max_iter=args.max_iters,
                                 tv_inner_iters=10)
    fista_real = FISTATVSolver(Phi_real, lam=0.01, max_iter=args.max_iters,
                                tv_inner_iters=10)

    methods = ['fista_tv']
    solvers_ideal = {'fista_tv': fista_ideal}
    solvers_real = {'fista_tv': fista_real}

    # Try to load DRUNet for PnP-FISTA (ISTA-Net+ / HATNet proxies)
    denoiser, device = load_drunet_denoiser()

    if denoiser is not None:
        # ISTA-Net+ proxy: PnP-FISTA with standard params
        pnp_ideal = PnPFISTASolver(Phi_ideal, denoiser, device,
                                    max_iter=200, sigma_start=0.08,
                                    sigma_end=0.02)
        pnp_real = PnPFISTASolver(Phi_real, denoiser, device,
                                   max_iter=200, sigma_start=0.08,
                                   sigma_end=0.02)
        solvers_ideal['ista_net_plus'] = pnp_ideal
        solvers_real['ista_net_plus'] = pnp_real
        methods.append('ista_net_plus')

        # HATNet proxy: PnP-FISTA with more iterations + wider anneal
        hat_ideal = PnPFISTASolver(Phi_ideal, denoiser, device,
                                    max_iter=200, sigma_start=0.10,
                                    sigma_end=0.015)
        hat_real = PnPFISTASolver(Phi_real, denoiser, device,
                                   max_iter=200, sigma_start=0.10,
                                   sigma_end=0.015)
        solvers_ideal['hatnet'] = hat_ideal
        solvers_real['hatnet'] = hat_real
        methods.append('hatnet')
    else:
        # Fallback: use FISTA-TV for all methods
        logger.info("  Deep learning unavailable, all methods use FISTA-TV")
        solvers_ideal['ista_net_plus'] = fista_ideal
        solvers_real['ista_net_plus'] = fista_real
        methods.append('ista_net_plus')
        solvers_ideal['hatnet'] = fista_ideal
        solvers_real['hatnet'] = fista_real
        methods.append('hatnet')

    logger.info(f"Methods: {methods}")
    logger.info(f"Total block reconstructions: {NUM_IMAGES} x {BLOCKS_PER_IMAGE} x "
                f"{len(methods)} methods x 3 scenarios = "
                f"{NUM_IMAGES * BLOCKS_PER_IMAGE * len(methods) * 3}")

    # 5. Validate all images
    all_results = []
    start_total = time.time()

    for idx, (name, image) in enumerate(images):
        result = validate_image(idx, name, image, Phi_ideal, Phi_real,
                                solvers_ideal, solvers_real, methods,
                                args.max_iters)
        all_results.append(result)

    total_time = time.time() - start_total

    # 6. Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)

    summary = compute_summary(all_results, methods, sampling_rate, m_measurements)

    for scenario_key in SCENARIOS:
        label = {'scenario_i': 'SCENARIO I (Ideal)',
                 'scenario_ii': 'SCENARIO II (Baseline)',
                 'scenario_iii': 'SCENARIO III (Oracle)'}[scenario_key]
        logger.info(f"\n{label}:")
        for method in methods:
            s = summary['scenarios'][scenario_key][method]
            logger.info(f"  {method:15s}: {s['psnr']['mean']:.2f} +/- {s['psnr']['std']:.2f} dB, "
                        f"SSIM: {s['ssim']['mean']:.4f}")

    logger.info(f"\nGap Analysis:")
    for method in methods:
        g = summary['gaps'][method]
        logger.info(f"  {method:15s}: Gap I->II={g['gap_i_ii']['mean']:.2f} dB, "
                    f"Recovery II->III={g['gap_ii_iii']['mean']:.2f} dB")

    logger.info(f"\nTotal time: {total_time/60:.1f} min "
                f"({total_time/len(all_results):.1f}s per image)")

    # 7. Save results
    out_detailed = RESULTS_DIR / "spc_validation_results.json"
    out_summary = RESULTS_DIR / "spc_summary.json"

    with open(out_detailed, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nDetailed results: {out_detailed}")

    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {out_summary}")

    logger.info("\nSPC Validation v3.0 complete!")


if __name__ == '__main__':
    main()
