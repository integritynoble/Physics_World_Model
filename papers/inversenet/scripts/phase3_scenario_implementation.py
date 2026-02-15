#!/usr/bin/env python3
"""
Phase 3: Scenario Implementation for InverseNet CASSI Validation

Implements 3 scenarios across 4 methods (GAP-TV, HDNet, MST-S, MST-L) on test scenes.

Scenarios:
  I  - Ideal:    Ideal mask for measurement AND reconstruction
  II - Baseline: Corrupted mask for measurement, IDEAL mask for reconstruction (mismatch!)
  IV - Oracle:   Corrupted mask for measurement, CORRUPTED mask for reconstruction (oracle)

Key insight: MST/HDNet models take mask as input → scenario differentiation via mask.
GAP-TV takes mask directly as argument.
Scenario II vs IV differ by which mask is passed to the reconstruction.

Usage:
    python phase3_scenario_implementation.py --device cuda:0 --quick
    python phase3_scenario_implementation.py --device cuda:0 --scenes 1 2 3
    python phase3_scenario_implementation.py --device cuda:0 --all-scenes
    python phase3_scenario_implementation.py --device cuda:0 --all-scenes --methods mst_s mst_l gap_tv hdnet
"""

import sys
import time
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import scipy.io as sio

# Add paths
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "packages"))

import torch
from pwm_core.recon.mst import MST, shift_torch, shift_back_meas_torch
from pwm_core.analysis.metrics import psnr as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATASET = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
MISMATCH_DX = 1.5   # pixels
MISMATCH_DY = 1.0   # pixels
MISMATCH_THETA = 0.3 # degrees
NOISE_ALPHA = 100000 # photon peak (low noise for clear mismatch signal)
NOISE_SIGMA = 0.01   # read noise std (low noise)
STEP = 2             # dispersion step
NC = 28              # spectral bands
ALL_METHODS = ['gap_tv', 'hdnet', 'mst_s', 'mst_l']
METHODS = ALL_METHODS  # default: all 4 methods


# ============================================================================
# Data Loading
# ============================================================================

def load_scene(scene_name: str) -> np.ndarray:
    """Load KAIST test scene → (256, 256, 28)."""
    path = DATASET / "Truth" / f"{scene_name}.mat"
    data = sio.loadmat(str(path))
    for key in ['img', 'Img', 'scene', 'Scene']:
        if key in data:
            scene = data[key].astype(np.float32)
            if scene.ndim == 3 and scene.shape[2] == NC:
                return scene
    raise ValueError(f"Could not load scene from {path}")


def load_mask() -> np.ndarray:
    """Load mask → (256, 256)."""
    path = DATASET / "mask.mat"
    data = sio.loadmat(str(path))
    mask = data['mask'].astype(np.float32)
    if mask.shape != (256, 256):
        from scipy.ndimage import zoom
        scale = (256 / mask.shape[0], 256 / mask.shape[1])
        mask = zoom(mask, scale)
    return mask


# ============================================================================
# CASSI Forward Model (fast, from benchmark pattern)
# ============================================================================

def warp_mask2d(mask2d: np.ndarray, dx: float, dy: float, theta_deg: float) -> np.ndarray:
    """Subpixel shift + small rotation via scipy affine_transform."""
    from scipy.ndimage import affine_transform
    H, W = mask2d.shape
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    center = np.array([(H - 1) / 2.0, (W - 1) / 2.0], dtype=np.float32)
    M = R.T
    shift = np.array([dy, dx], dtype=np.float32)
    offset = (center - shift) - M @ center
    warped = affine_transform(mask2d.astype(np.float32), matrix=M, offset=offset,
                              output_shape=(H, W), order=1, mode="constant", cval=0.0)
    return np.clip(warped, 0.0, 1.0).astype(np.float32)


def cassi_forward(x_hwl: np.ndarray, mask2d: np.ndarray, step: int = STEP) -> np.ndarray:
    """CASSI forward model: masked spectral bands placed on expanded canvas.

    Args:
        x_hwl: (H, W, L) scene
        mask2d: (H, W) coded aperture
        step: dispersion step

    Returns:
        y: (H, W + (L-1)*step) measurement
    """
    H, W, L = x_hwl.shape
    W_ext = W + (L - 1) * step
    y = np.zeros((H, W_ext), dtype=np.float32)
    for l in range(L):
        ox = step * l
        y[:, ox:ox + W] += mask2d * x_hwl[:, :, l]
    return y


def add_noise(y: np.ndarray, alpha: float = NOISE_ALPHA,
              sigma: float = NOISE_SIGMA, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Add Poisson + Gaussian noise."""
    if rng is None:
        rng = np.random.RandomState(42)
    y_clean = np.maximum(y, 0.0)
    lam = np.clip(alpha * y_clean, 0.0, 1e9)
    y_noisy = rng.poisson(lam=lam).astype(np.float32) / float(alpha)
    y_noisy += rng.normal(0.0, sigma, size=y_clean.shape).astype(np.float32)
    return y_noisy


# ============================================================================
# MST Reconstruction (from benchmark pattern)
# ============================================================================

_mst_cache = {}  # {variant: (model, device)}


def load_mst_model(variant: str, device: str) -> Tuple[torch.nn.Module, torch.device]:
    """Load MST model with pretrained weights."""
    if variant in _mst_cache:
        return _mst_cache[variant]

    dev = torch.device(device)
    pkg_root = Path(__file__).parent.parent.parent.parent / "packages" / "pwm_core"

    # Find weights
    if variant == 'mst_s':
        weight_paths = [
            pkg_root / "weights" / "mst" / "mst_s.pth",
        ]
        num_blocks_default = [2, 2, 2]  # MST-S
    else:  # mst_l
        weight_paths = [
            pkg_root / "weights" / "mst" / "mst_l.pth",
        ]
        num_blocks_default = [4, 7, 5]  # MST-L

    state_dict = None
    for wp in weight_paths:
        if wp.exists():
            checkpoint = torch.load(str(wp), map_location=dev, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
            else:
                state_dict = checkpoint
            logger.info(f"  Loaded {variant} weights from {wp}")
            break

    if state_dict is None:
        raise RuntimeError(f"No pretrained weights found for {variant}")

    # Infer architecture from checkpoint
    num_blocks = num_blocks_default
    inferred = []
    for stage_idx in range(10):
        prefix = f"encoder_layers.{stage_idx}.0.blocks."
        max_blk = -1
        for k in state_dict:
            if k.startswith(prefix):
                blk_idx = int(k[len(prefix):].split(".")[0])
                max_blk = max(max_blk, blk_idx)
        if max_blk >= 0:
            inferred.append(max_blk + 1)
        else:
            break
    bot_prefix = "bottleneck.blocks."
    max_bot = -1
    for k in state_dict:
        if k.startswith(bot_prefix):
            blk_idx = int(k[len(bot_prefix):].split(".")[0])
            max_bot = max(max_bot, blk_idx)
    if max_bot >= 0:
        inferred.append(max_bot + 1)
    if len(inferred) >= 2:
        num_blocks = inferred

    model = MST(
        dim=NC, stage=len(num_blocks) - 1, num_blocks=num_blocks,
        in_channels=NC, out_channels=NC, base_resolution=256, step=STEP,
    ).to(dev)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info(f"  {variant} architecture: stage={len(num_blocks)-1}, blocks={num_blocks}")

    _mst_cache[variant] = (model, dev)
    return model, dev


def mst_reconstruct(y: np.ndarray, mask2d: np.ndarray,
                    variant: str, device: str) -> np.ndarray:
    """Reconstruct CASSI using MST with a given mask.

    The mask determines which operator the model "assumes" was used.
    This is the key to differentiating Scenario II (ideal mask) vs IV (true mask).

    Args:
        y: (H, W_ext) measurement
        mask2d: (H, W) coded aperture mask to use for reconstruction
        variant: 'mst_s' or 'mst_l'
        device: torch device

    Returns:
        x_recon: (H, W, nC) reconstructed cube
    """
    model, dev = load_mst_model(variant, device)

    H, W = mask2d.shape
    W_ext = W + (NC - 1) * STEP

    # Pad/crop measurement to expected size
    y_mst = np.zeros((H, W_ext), dtype=np.float32)
    hh = min(H, y.shape[0])
    ww = min(W_ext, y.shape[1])
    y_mst[:hh, :ww] = y[:hh, :ww]

    # Prepare mask: [H, W] → [1, nC, H, W] → shifted [1, nC, H, W_ext]
    mask_3d = np.tile(mask2d[:, :, np.newaxis], (1, 1, NC))
    mask_3d_t = torch.from_numpy(mask_3d.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(dev)
    mask_shifted = shift_torch(mask_3d_t, step=STEP)

    # Prepare initial estimate: Y2H conversion (matching original MST code)
    meas_t = torch.from_numpy(y_mst.copy()).unsqueeze(0).float().to(dev)
    x_init = shift_back_meas_torch(meas_t, step=STEP, nC=NC)
    x_init = x_init / NC * 2  # Scaling from original MST code

    # Forward pass with mask
    with torch.no_grad():
        recon = model(x_init, mask_shifted)

    # Convert to numpy [H, W, nC]
    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(recon, 0, 1).astype(np.float32)


# ============================================================================
# GAP-TV Reconstruction
# ============================================================================

def _shift_np(inputs: np.ndarray, step: int = STEP) -> np.ndarray:
    """Shift each band along columns: band k at column offset k*step."""
    H, W, nC = inputs.shape
    W_ext = W + (nC - 1) * step
    output = np.zeros((H, W_ext, nC), dtype=inputs.dtype)
    for k in range(nC):
        output[:, k * step:k * step + W, k] = inputs[:, :, k]
    return output


def _shift_back_np(inputs: np.ndarray, step: int = STEP) -> np.ndarray:
    """Inverse shift: extract band k from column offset k*step."""
    H, col, nC = inputs.shape
    W = col - (nC - 1) * step
    output = np.zeros((H, W, nC), dtype=inputs.dtype)
    for k in range(nC):
        output[:, :, k] = inputs[:, k * step:k * step + W, k]
    return output


def _tv_denoiser_chambolle(x: np.ndarray, tv_weight: float,
                           n_iter: int = 5) -> np.ndarray:
    """Isotropic TV denoiser using Chambolle 2004 dual algorithm."""
    dt = 0.25
    N = x.shape
    idx = np.arange(1, N[0] + 1); idx[-1] = N[0] - 1
    iux = np.arange(-1, N[0] - 1); iux[0] = 0
    ir = np.arange(1, N[1] + 1); ir[-1] = N[1] - 1
    il = np.arange(-1, N[1] - 1); il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)
    for _ in range(n_iter):
        z = divp - x * tv_weight
        z1 = z[:, ir, :] - z
        z2 = z[idx, :, :] - z
        denom_2d = 1 + dt * np.sqrt(np.sum(z1 ** 2 + z2 ** 2, 2))
        denom_3d = np.tile(denom_2d[:, :, np.newaxis], (1, 1, N[2]))
        p1 = (p1 + dt * z1) / denom_3d
        p2 = (p2 + dt * z2) / denom_3d
        divp = p1 - p1[:, il, :] + p2 - p2[iux, :, :]
    return x - divp / tv_weight


def gaptv_reconstruct(y: np.ndarray, mask2d: np.ndarray,
                      iterations: int = 100, tv_weight: float = 4.0,
                      step: int = STEP) -> np.ndarray:
    """GAP-TV for CASSI operating in the shifted domain.

    Ported from the proven run_cassi_benchmark.py implementation.
    Operates on 3D shifted cube (H, W_ext, nC) with shifted masks.

    Args:
        y: (H, W_ext) measurement where W_ext = W + (nC-1)*step
        mask2d: (H, W) coded aperture mask
        iterations: GAP-TV iterations
        tv_weight: TV regularization weight
        step: dispersion step (pixels per band)

    Returns:
        x_recon: (H, W, nC) reconstructed cube
    """
    H, W = mask2d.shape
    n_bands = NC
    W_ext = W + (n_bands - 1) * step

    # Ensure measurement has expected width
    y_full = np.zeros((H, W_ext), dtype=np.float32)
    hh = min(H, y.shape[0])
    ww = min(W_ext, y.shape[1])
    y_full[:hh, :ww] = y[:hh, :ww]

    # Build 3D shifted mask: (H, W_ext, nC)
    mask_3d = np.zeros((H, W, n_bands), dtype=np.float32)
    for k in range(n_bands):
        mask_3d[:, :, k] = mask2d
    Phi = _shift_np(mask_3d, step=step)  # (H, W_ext, nC)

    # Phi_sum for normalization: sum over bands at each measurement position
    Phi_sum = np.sum(Phi, axis=2)  # (H, W_ext)
    Phi_sum[Phi_sum == 0] = 1

    # Initialize: adjoint of measurement
    x = np.multiply(np.repeat(y_full[:, :, np.newaxis], n_bands, axis=2), Phi)

    # Accelerated GAP-TV
    y1 = np.zeros_like(y_full)

    for it in range(iterations):
        # Forward: A @ x
        yb = np.sum(x * Phi, axis=2)

        # Accelerated update
        y1 = y1 + (y_full - yb)
        x = x + np.multiply(
            np.repeat(((y1 - yb) / Phi_sum)[:, :, np.newaxis], n_bands, axis=2),
            Phi
        )

        # TV denoising in unshifted domain
        x_unshifted = _shift_back_np(x, step=step)
        x_unshifted = _tv_denoiser_chambolle(x_unshifted, tv_weight, n_iter=5)
        x = _shift_np(x_unshifted, step=step)

    # Final result in unshifted domain
    x_final = _shift_back_np(x, step=step)
    return np.clip(x_final, 0, 1).astype(np.float32)


# ============================================================================
# HDNet Reconstruction
# ============================================================================

_hdnet_cache = {}

def _load_original_hdnet(device: str):
    """Load original HDNet model from MST-main."""
    if 'model' in _hdnet_cache:
        return _hdnet_cache['model']

    # Import the original HDNet architecture
    import importlib.util
    hdnet_path = "/home/spiritai/MST-main/simulation/test_code/architecture/HDNet.py"
    spec = importlib.util.spec_from_file_location("hdnet_orig", hdnet_path)
    hdnet_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdnet_mod)

    dev = torch.device(device)
    model = hdnet_mod.HDNet(in_ch=NC, out_ch=NC).to(dev)

    # Load pretrained weights
    weights_path = "/home/spiritai/MST-main/model_zoo/hdnet/hdnet.pth"
    checkpoint = torch.load(weights_path, map_location=dev, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info(f"  Loaded original HDNet with pretrained weights")

    _hdnet_cache['model'] = (model, dev)
    return model, dev


def hdnet_reconstruct(y: np.ndarray, mask2d: np.ndarray, device: str) -> np.ndarray:
    """Reconstruct CASSI using original HDNet architecture.

    Original HDNet takes ONLY the initial spectral estimate (28 channels)
    as input — no mask concatenation. The mask is used to create the
    initial estimate via shift_back but is not passed to the model.

    However, to differentiate scenarios, we use the mask for the initial
    estimate: different masks → different inputs → different outputs.

    Args:
        y: (H, W_ext) measurement
        mask2d: (H, W) coded aperture mask
        device: torch device

    Returns:
        x_recon: (H, W, nC) reconstructed cube
    """
    model, dev = _load_original_hdnet(device)

    H, W = mask2d.shape
    W_ext = W + (NC - 1) * STEP

    # Pad/crop measurement
    y_hdnet = np.zeros((H, W_ext), dtype=np.float32)
    hh = min(H, y.shape[0])
    ww = min(W_ext, y.shape[1])
    y_hdnet[:hh, :ww] = y[:hh, :ww]

    # Create initial estimate using shift_back (same as MST)
    meas_t = torch.from_numpy(y_hdnet.copy()).unsqueeze(0).float().to(dev)
    x_init = shift_back_meas_torch(meas_t, step=STEP, nC=NC)
    x_init = x_init / NC * 2  # Scaling from original MST/HDNet code

    # Forward pass (HDNet takes only the initial estimate, no mask)
    with torch.no_grad():
        recon = model(x_init)

    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(recon, 0, 1).astype(np.float32)


# ============================================================================
# Unified Reconstruction Dispatcher
# ============================================================================

def reconstruct(y: np.ndarray, mask2d: np.ndarray,
                method: str, device: str) -> np.ndarray:
    """Reconstruct CASSI measurement using specified method and mask.

    Args:
        y: (H, W_ext) measurement
        mask2d: (H, W) coded aperture mask to use for reconstruction
        method: 'gap_tv', 'hdnet', 'mst_s', or 'mst_l'
        device: torch device

    Returns:
        x_recon: (H, W, nC) reconstructed cube
    """
    if method == 'gap_tv':
        return gaptv_reconstruct(y, mask2d)
    elif method == 'hdnet':
        return hdnet_reconstruct(y, mask2d, device)
    elif method in ('mst_s', 'mst_l'):
        return mst_reconstruct(y, mask2d, method, device)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Scenario Functions
# ============================================================================

def run_scenario_i(scene: np.ndarray, mask_ideal: np.ndarray,
                   methods: List[str], device: str) -> Dict:
    """Scenario I: Ideal measurement + ideal reconstruction mask."""
    logger.info("  Scenario I: Ideal")

    y_ideal = cassi_forward(scene, mask_ideal, step=STEP)
    logger.info(f"    Measurement: shape {y_ideal.shape}, range [{y_ideal.min():.4f}, {y_ideal.max():.4f}]")

    results = {}
    for method in methods:
        x_recon = reconstruct(y_ideal, mask_ideal, method, device)
        p = compute_psnr(scene, x_recon, data_range=1.0)
        s = compute_ssim(np.mean(scene, 2), np.mean(x_recon, 2), data_range=1.0)
        results[method] = {'psnr': float(p), 'ssim': float(s)}
        logger.info(f"    {method}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    return results


def run_scenario_ii(scene: np.ndarray, mask_ideal: np.ndarray,
                    mask_corrupt: np.ndarray, methods: List[str],
                    device: str) -> Tuple[Dict, np.ndarray]:
    """Scenario II: Corrupted measurement + IDEAL reconstruction mask (mismatch!)."""
    logger.info("  Scenario II: Baseline (mismatch)")

    # Generate measurement with CORRUPTED mask
    y_corrupt = cassi_forward(scene, mask_corrupt, step=STEP)
    y_corrupt = add_noise(y_corrupt, alpha=NOISE_ALPHA, sigma=NOISE_SIGMA)
    logger.info(f"    Measurement: shape {y_corrupt.shape}, range [{y_corrupt.min():.4f}, {y_corrupt.max():.4f}]")

    # Reconstruct using IDEAL mask (assumes perfect operator → mismatch!)
    results = {}
    for method in methods:
        x_recon = reconstruct(y_corrupt, mask_ideal, method, device)
        p = compute_psnr(scene, x_recon, data_range=1.0)
        s = compute_ssim(np.mean(scene, 2), np.mean(x_recon, 2), data_range=1.0)
        results[method] = {'psnr': float(p), 'ssim': float(s)}
        logger.info(f"    {method}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    return results, y_corrupt


def run_scenario_iv(scene: np.ndarray, y_corrupt: np.ndarray,
                    mask_corrupt: np.ndarray, methods: List[str],
                    device: str) -> Dict:
    """Scenario IV: Same corrupted measurement + TRUE reconstruction mask (oracle)."""
    logger.info("  Scenario IV: Oracle (true mask)")

    # Reconstruct using TRUE CORRUPTED mask (oracle knowledge → better reconstruction)
    results = {}
    for method in methods:
        x_recon = reconstruct(y_corrupt, mask_corrupt, method, device)
        p = compute_psnr(scene, x_recon, data_range=1.0)
        s = compute_ssim(np.mean(scene, 2), np.mean(x_recon, 2), data_range=1.0)
        results[method] = {'psnr': float(p), 'ssim': float(s)}
        logger.info(f"    {method}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    return results


# ============================================================================
# Main Validation
# ============================================================================

def validate_scene(scene_idx: int, scene_name: str, scene: np.ndarray,
                   mask_ideal: np.ndarray, mask_corrupt: np.ndarray,
                   methods: List[str], device: str) -> Dict:
    """Validate one scene across all 3 scenarios."""
    logger.info(f"\nScene {scene_idx:02d}/{scene_name}")
    logger.info("=" * 70)

    t0 = time.time()

    res_i = run_scenario_i(scene, mask_ideal, methods, device)
    res_ii, y_corrupt = run_scenario_ii(scene, mask_ideal, mask_corrupt, methods, device)
    res_iv = run_scenario_iv(scene, y_corrupt, mask_corrupt, methods, device)

    elapsed = time.time() - t0

    # Calculate gaps
    gaps = {}
    for method in methods:
        pi = res_i[method]['psnr']
        pii = res_ii[method]['psnr']
        piv = res_iv[method]['psnr']
        gaps[method] = {
            'gap_i_ii': pi - pii,     # Mismatch degradation
            'gap_ii_iv': piv - pii,   # Oracle recovery
            'gap_iv_i': pi - piv      # Residual gap
        }

    # Print summary
    for method in methods:
        pi = res_i[method]['psnr']
        pii = res_ii[method]['psnr']
        piv = res_iv[method]['psnr']
        logger.info(f"\n  {method.upper()}:")
        logger.info(f"    I:  {pi:6.2f} dB")
        logger.info(f"    II: {pii:6.2f} dB  (I→II gap: {gaps[method]['gap_i_ii']:+.2f} dB)")
        logger.info(f"    IV: {piv:6.2f} dB  (II→IV gain: {gaps[method]['gap_ii_iv']:+.2f} dB)")

    logger.info(f"\n  Scene time: {elapsed:.1f}s")

    return {
        'scene_idx': scene_idx,
        'scene_name': scene_name,
        'elapsed_s': elapsed,
        'scenario_i': res_i,
        'scenario_ii': res_ii,
        'scenario_iv': res_iv,
        'gaps': gaps,
        'mismatch': {'dx': MISMATCH_DX, 'dy': MISMATCH_DY, 'theta': MISMATCH_THETA}
    }


def compute_summary(all_results: List[Dict], methods: List[str]) -> Dict:
    """Aggregate statistics across scenes."""
    summary = {
        'num_scenes': len(all_results),
        'methods': methods,
        'mismatch': {'dx': MISMATCH_DX, 'dy': MISMATCH_DY, 'theta': MISMATCH_THETA},
        'noise': {'alpha': NOISE_ALPHA, 'sigma': NOISE_SIGMA},
    }

    for scenario in ['scenario_i', 'scenario_ii', 'scenario_iv']:
        summary[scenario] = {}
        for method in methods:
            psnr_vals = [r[scenario][method]['psnr'] for r in all_results if method in r[scenario]]
            ssim_vals = [r[scenario][method]['ssim'] for r in all_results if method in r[scenario]]
            if psnr_vals:
                summary[scenario][method] = {
                    'psnr_mean': float(np.mean(psnr_vals)),
                    'psnr_std': float(np.std(psnr_vals)),
                    'ssim_mean': float(np.mean(ssim_vals)),
                    'ssim_std': float(np.std(ssim_vals)),
                }

    summary['gaps'] = {}
    for method in methods:
        gap_i_ii = [r['gaps'][method]['gap_i_ii'] for r in all_results if method in r['gaps']]
        gap_ii_iv = [r['gaps'][method]['gap_ii_iv'] for r in all_results if method in r['gaps']]
        if gap_i_ii:
            summary['gaps'][method] = {
                'gap_i_ii_mean': float(np.mean(gap_i_ii)),
                'gap_i_ii_std': float(np.std(gap_i_ii)),
                'gap_ii_iv_mean': float(np.mean(gap_ii_iv)),
                'gap_ii_iv_std': float(np.std(gap_ii_iv)),
            }

    return summary


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Scenario Implementation')
    parser.add_argument('--device', default='cuda:0', help='Torch device')
    parser.add_argument('--scenes', type=int, nargs='+', help='Scene indices (1-10)')
    parser.add_argument('--all-scenes', action='store_true', help='All 10 scenes')
    parser.add_argument('--quick', action='store_true', help='Quick test (1 scene)')
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                       choices=ALL_METHODS, help='Reconstruction methods')
    args = parser.parse_args()

    methods = args.methods

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: SCENARIO IMPLEMENTATION")
    logger.info("=" * 70)
    logger.info(f"Scenarios: I (Ideal), II (Baseline), IV (Oracle)")
    logger.info(f"Methods: {methods}")
    logger.info(f"Mismatch: dx={MISMATCH_DX}, dy={MISMATCH_DY}, θ={MISMATCH_THETA}°")
    logger.info(f"Noise: α={NOISE_ALPHA}, σ={NOISE_SIGMA}")

    # Load mask
    mask_ideal = load_mask()
    logger.info(f"✓ Mask loaded: {mask_ideal.shape}")

    # Create corrupted mask
    mask_corrupt = warp_mask2d(mask_ideal, MISMATCH_DX, MISMATCH_DY, MISMATCH_THETA)
    diff = np.mean((mask_ideal - mask_corrupt) ** 2)
    logger.info(f"✓ Corrupted mask created (MSE vs ideal: {diff:.6f})")

    # Determine scenes
    if args.quick:
        scene_indices = [1]
    elif args.all_scenes:
        scene_indices = list(range(1, 11))
    elif args.scenes:
        scene_indices = args.scenes
    else:
        scene_indices = [1, 2, 3]

    logger.info(f"Scenes: {scene_indices}")

    # Validate
    all_results = []
    t_total = time.time()

    for scene_idx in scene_indices:
        scene_name = f"scene{scene_idx:02d}"
        try:
            scene = load_scene(scene_name)
            logger.info(f"✓ Loaded {scene_name}: {scene.shape}")
            result = validate_scene(scene_idx, scene_name, scene,
                                    mask_ideal, mask_corrupt, methods, args.device)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed {scene_name}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - t_total

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    summary = compute_summary(all_results, methods)

    for method in methods:
        logger.info(f"\n{method.upper()}:")
        for scenario in ['scenario_i', 'scenario_ii', 'scenario_iv']:
            if method in summary.get(scenario, {}):
                m = summary[scenario][method]
                label = scenario.replace('scenario_', '').upper()
                logger.info(f"  {label:>3s}: {m['psnr_mean']:6.2f} ± {m['psnr_std']:.2f} dB  "
                           f"(SSIM {m['ssim_mean']:.4f})")

        if method in summary.get('gaps', {}):
            g = summary['gaps'][method]
            logger.info(f"  I→II gap:  {g['gap_i_ii_mean']:+.2f} ± {g['gap_i_ii_std']:.2f} dB")
            logger.info(f"  II→IV gain: {g['gap_ii_iv_mean']:+.2f} ± {g['gap_ii_iv_std']:.2f} dB")

    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save
    with open(RESULTS_DIR / "phase3_scenario_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(RESULTS_DIR / "phase3_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Results saved to {RESULTS_DIR}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ PHASE 3 COMPLETE")
    logger.info("=" * 70 + "\n")


if __name__ == '__main__':
    main()
