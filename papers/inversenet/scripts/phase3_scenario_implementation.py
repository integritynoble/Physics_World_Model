#!/usr/bin/env python3
"""
Phase 3: Scenario Implementation for InverseNet CASSI Validation

Implements 3 scenarios across MST-S/MST-L on test scenes using benchmark patterns.

Scenarios:
  I  - Ideal:    Ideal mask for measurement AND reconstruction
  II - Baseline: Corrupted mask for measurement, IDEAL mask for reconstruction (mismatch!)
  IV - Oracle:   Corrupted mask for measurement, CORRUPTED mask for reconstruction (oracle)

Key insight: MST model takes mask as second argument → model(x_init, mask_shifted).
Scenario II vs IV differ by which mask is passed to the reconstruction model.

Usage:
    python phase3_scenario_implementation.py --device cuda:0 --quick
    python phase3_scenario_implementation.py --device cuda:0 --scenes 1 2 3
    python phase3_scenario_implementation.py --device cuda:0 --all-scenes
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
MISMATCH_DX = 0.5   # pixels
MISMATCH_DY = 0.3   # pixels
MISMATCH_THETA = 0.1 # degrees
NOISE_ALPHA = 10000  # photon peak (realistic)
NOISE_SIGMA = 0.05   # read noise std (scaled for measurement range ~0-8)
STEP = 2             # dispersion step
NC = 28              # spectral bands
METHODS = ['mst_s', 'mst_l']


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
# Scenario Functions
# ============================================================================

def run_scenario_i(scene: np.ndarray, mask_ideal: np.ndarray, device: str) -> Dict:
    """Scenario I: Ideal measurement + ideal reconstruction mask."""
    logger.info("  Scenario I: Ideal")

    y_ideal = cassi_forward(scene, mask_ideal, step=STEP)
    logger.info(f"    Measurement: shape {y_ideal.shape}, range [{y_ideal.min():.4f}, {y_ideal.max():.4f}]")

    results = {}
    for method in METHODS:
        x_recon = mst_reconstruct(y_ideal, mask_ideal, method, device)
        p = compute_psnr(scene, x_recon, data_range=1.0)
        s = compute_ssim(np.mean(scene, 2), np.mean(x_recon, 2), data_range=1.0)
        results[method] = {'psnr': float(p), 'ssim': float(s)}
        logger.info(f"    {method}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    return results


def run_scenario_ii(scene: np.ndarray, mask_ideal: np.ndarray,
                    mask_corrupt: np.ndarray, device: str) -> Tuple[Dict, np.ndarray]:
    """Scenario II: Corrupted measurement + IDEAL reconstruction mask (mismatch!)."""
    logger.info("  Scenario II: Baseline (mismatch)")

    # Generate measurement with CORRUPTED mask
    y_corrupt = cassi_forward(scene, mask_corrupt, step=STEP)
    y_corrupt = add_noise(y_corrupt, alpha=NOISE_ALPHA, sigma=NOISE_SIGMA)
    logger.info(f"    Measurement: shape {y_corrupt.shape}, range [{y_corrupt.min():.4f}, {y_corrupt.max():.4f}]")

    # Reconstruct using IDEAL mask (assumes perfect operator → mismatch!)
    results = {}
    for method in METHODS:
        x_recon = mst_reconstruct(y_corrupt, mask_ideal, method, device)
        p = compute_psnr(scene, x_recon, data_range=1.0)
        s = compute_ssim(np.mean(scene, 2), np.mean(x_recon, 2), data_range=1.0)
        results[method] = {'psnr': float(p), 'ssim': float(s)}
        logger.info(f"    {method}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    return results, y_corrupt


def run_scenario_iv(scene: np.ndarray, y_corrupt: np.ndarray,
                    mask_corrupt: np.ndarray, device: str) -> Dict:
    """Scenario IV: Same corrupted measurement + TRUE reconstruction mask (oracle)."""
    logger.info("  Scenario IV: Oracle (true mask)")

    # Reconstruct using TRUE CORRUPTED mask (oracle knowledge → better reconstruction)
    results = {}
    for method in METHODS:
        x_recon = mst_reconstruct(y_corrupt, mask_corrupt, method, device)
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
                   device: str) -> Dict:
    """Validate one scene across all 3 scenarios."""
    logger.info(f"\nScene {scene_idx:02d}/{scene_name}")
    logger.info("=" * 70)

    t0 = time.time()

    res_i = run_scenario_i(scene, mask_ideal, device)
    res_ii, y_corrupt = run_scenario_ii(scene, mask_ideal, mask_corrupt, device)
    res_iv = run_scenario_iv(scene, y_corrupt, mask_corrupt, device)

    elapsed = time.time() - t0

    # Calculate gaps
    gaps = {}
    for method in METHODS:
        pi = res_i[method]['psnr']
        pii = res_ii[method]['psnr']
        piv = res_iv[method]['psnr']
        gaps[method] = {
            'gap_i_ii': pi - pii,     # Mismatch degradation
            'gap_ii_iv': piv - pii,   # Oracle recovery
            'gap_iv_i': pi - piv      # Residual gap
        }

    # Print summary
    for method in METHODS:
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


def compute_summary(all_results: List[Dict]) -> Dict:
    """Aggregate statistics across scenes."""
    summary = {
        'num_scenes': len(all_results),
        'methods': METHODS,
        'mismatch': {'dx': MISMATCH_DX, 'dy': MISMATCH_DY, 'theta': MISMATCH_THETA},
        'noise': {'alpha': NOISE_ALPHA, 'sigma': NOISE_SIGMA},
    }

    for scenario in ['scenario_i', 'scenario_ii', 'scenario_iv']:
        summary[scenario] = {}
        for method in METHODS:
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
    for method in METHODS:
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
    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: SCENARIO IMPLEMENTATION")
    logger.info("=" * 70)
    logger.info(f"Scenarios: I (Ideal), II (Baseline), IV (Oracle)")
    logger.info(f"Methods: {METHODS}")
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
                                    mask_ideal, mask_corrupt, args.device)
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

    summary = compute_summary(all_results)

    for method in METHODS:
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
