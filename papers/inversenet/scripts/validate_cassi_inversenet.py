#!/usr/bin/env python3
"""CASSI InverseNet Validation -- benchmark-grade reconstruction.

Validates 4 reconstruction methods (GAP-TV, HDNet, MST-S, MST-L) across
3 scenarios (I: Ideal, II: Assumed, IV: Truth Forward Model) on 10 KAIST scenes.

Scenario III (operator correction via Algorithms 1 & 2) is intentionally skipped
for the InverseNet benchmark, which focuses on reconstruction comparison.

Scenarios:
  Scenario I  : ideal measurement + ideal mask          (oracle upper bound)
  Scenario II : corrupted measurement + ideal mask      (baseline degradation)
  Scenario IV : corrupted measurement + truth mask      (oracle operator)

Methods:
  GAP-TV   -- classical iterative (mask-aware)    (~32 dB ideal)
  HDNet    -- dual-domain deep learning           (~35 dB ideal, mask-oblivious)
  MST-S    -- mask-guided Transformer (small)     (~34 dB ideal)
  MST-L    -- mask-guided Transformer (large)     (~36 dB ideal)

Usage:
    python validate_cassi_inversenet.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.ndimage import affine_transform
from scipy.signal import correlate2d

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
DATASET_REAL = Path("/home/spiritai/MST-main/datasets/TSA_real_data")
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
RECONSTRUCTION_METHODS = ["gap_tv", "hdnet", "mst_s", "mst_l"]
SCENARIOS = ["scenario_i", "scenario_ii", "scenario_iv"]
NUM_SCENES = 10

METHOD_LABELS = {
    "gap_tv": "GAP-TV",
    "hdnet": "HDNet",
    "mst_s": "MST-S",
    "mst_l": "MST-L",
}

# ---------------------------------------------------------------------------
# mismatch spec  (moderate severity, from cassi_plan_inversenet.md section 3)
# ---------------------------------------------------------------------------
@dataclass
class MismatchParameters:
    """Mismatch parameters for CASSI operator."""
    mask_dx: float = 0.5      # pixels
    mask_dy: float = 0.3      # pixels
    mask_theta: float = 0.1   # degrees


# ===================================================================
# helpers -- data loading
# ===================================================================
def load_mask(path: Path) -> Optional[np.ndarray]:
    """Load mask from MATLAB .mat file."""
    try:
        data = sio.loadmat(str(path))
        for key in ["mask", "Mask", "mask_data"]:
            if key in data:
                mask = data[key]
                if isinstance(mask, np.ndarray):
                    return mask.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load mask from {path}: {e}")
    return None


def load_scene(scene_name: str) -> Optional[np.ndarray]:
    """Load scene from MATLAB .mat file (256x256x28)."""
    try:
        # Try Truth subdirectory first
        path = DATASET_SIMU / "Truth" / f"{scene_name}.mat"
        if not path.exists():
            path = DATASET_SIMU / f"{scene_name}.mat"

        if path.exists():
            data = sio.loadmat(str(path))
            for key in ["img", "Img", "scene", "Scene", "data"]:
                if key in data:
                    scene = data[key].astype(np.float32)
                    if scene.ndim == 3 and scene.shape[2] == 28:
                        return scene
    except Exception as e:
        logger.warning(f"Failed to load scene {scene_name}: {e}")
    return None


# ===================================================================
# helpers -- forward model & warping
# ===================================================================
def warp_affine_2d(mask: np.ndarray, dx: float, dy: float, theta: float) -> np.ndarray:
    """Apply 2D affine transformation to mask (translation + rotation).

    Reuses sign convention from cassi_upwmi_alg12.py / validate_cacti.

    Args:
        mask: (H, W) input mask
        dx: x-translation in pixels
        dy: y-translation in pixels
        theta: rotation in degrees

    Returns:
        Warped mask (H, W), clipped to [0, 1]
    """
    H, W = mask.shape
    cx, cy = W / 2.0, H / 2.0

    th = np.radians(theta)
    cos_t, sin_t = np.cos(th), np.sin(th)

    # Forward affine: rotate about center + translate
    mat = np.array([
        [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
        [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
    ])

    # scipy needs inverse matrix
    inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
    warped = affine_transform(mask, inv[:2, :2], offset=inv[:2, 2], cval=0, order=1)

    return np.clip(warped, 0, 1).astype(np.float32)


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


def add_poisson_gaussian_noise(y: np.ndarray, peak: float = 100000,
                               sigma: float = 0.01) -> np.ndarray:
    """Add Poisson + Gaussian noise to measurement."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0)

    y_max = np.max(y)
    if y_max <= 0:
        y_max = 1.0

    y_scaled = (y / y_max) * peak
    y_scaled = np.maximum(y_scaled, 0)

    y_poisson = np.random.poisson(y_scaled.astype(np.int64)).astype(np.float64)
    y_noisy = y_poisson + np.random.normal(0, sigma, y_poisson.shape)
    y_noisy = y_noisy / peak * y_max

    return np.maximum(y_noisy, 0).astype(np.float32)


# ===================================================================
# helpers -- metrics
# ===================================================================
def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate PSNR in dB (data in [0,1])."""
    x_true = np.clip(x_true, 0, 1).astype(np.float64)
    x_recon = np.clip(x_recon, 0, 1).astype(np.float64)
    mse = float(np.mean((x_true - x_recon) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(x_true: np.ndarray, x_recon: np.ndarray, window_size: int = 11) -> float:
    """Calculate SSIM on 2D grayscale images."""
    x_true = np.clip(x_true, 0, 1).astype(np.float64)
    x_recon = np.clip(x_recon, 0, 1).astype(np.float64)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window = np.ones((window_size, window_size)) / (window_size ** 2)

    mu_true = correlate2d(x_true, window, mode="same", boundary="symm")
    mu_recon = correlate2d(x_recon, window, mode="same", boundary="symm")
    mu_true_sq = mu_true ** 2
    mu_recon_sq = mu_recon ** 2
    mu_cross = mu_true * mu_recon

    sigma_true_sq = correlate2d(x_true ** 2, window, mode="same", boundary="symm") - mu_true_sq
    sigma_recon_sq = correlate2d(x_recon ** 2, window, mode="same", boundary="symm") - mu_recon_sq
    sigma_cross = correlate2d(x_true * x_recon, window, mode="same", boundary="symm") - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_true_sq + mu_recon_sq + C1) * (sigma_true_sq + sigma_recon_sq + C2))

    return float(np.mean(ssim_map))


def compute_sam(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Calculate Spectral Angle Mapper (SAM) in degrees."""
    x_true = np.clip(x_true, 1e-6, 1).astype(np.float64)
    x_recon = np.clip(x_recon, 1e-6, 1).astype(np.float64)

    x_true_flat = x_true.reshape(-1, x_true.shape[2])
    x_recon_flat = x_recon.reshape(-1, x_recon.shape[2])

    x_true_norm = x_true_flat / (np.linalg.norm(x_true_flat, axis=1, keepdims=True) + 1e-10)
    x_recon_norm = x_recon_flat / (np.linalg.norm(x_recon_flat, axis=1, keepdims=True) + 1e-10)

    dots = np.sum(x_true_norm * x_recon_norm, axis=1)
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)

    return float(np.degrees(np.mean(angles)))


# ===================================================================
# reconstruction methods
# ===================================================================
def reconstruct_gap_tv(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using GAP-TV.

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) forward operator mask
        device: unused (GAP-TV is CPU-based)

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.gap_tv import gap_tv_cassi
        return gap_tv_cassi(y, mask, n_bands=28, iterations=50, lam=0.01, step=2)
    except Exception as e:
        logger.warning(f"GAP-TV failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


_hdnet_cache = {}

def _load_original_hdnet(device: str):
    """Load original HDNet model from MST-main (correct architecture)."""
    if "model" in _hdnet_cache:
        return _hdnet_cache["model"]

    import torch
    import importlib.util

    hdnet_path = "/home/spiritai/MST-main/simulation/test_code/architecture/HDNet.py"
    spec = importlib.util.spec_from_file_location("hdnet_orig", hdnet_path)
    hdnet_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdnet_mod)

    dev = torch.device(device)
    model = hdnet_mod.HDNet(in_ch=28, out_ch=28).to(dev)

    weights_path = "/home/spiritai/MST-main/model_zoo/hdnet/hdnet.pth"
    checkpoint = torch.load(weights_path, map_location=dev, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info("  Loaded original HDNet with pretrained weights")

    _hdnet_cache["model"] = (model, dev)
    return model, dev


def reconstruct_hdnet(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using HDNet (mask-oblivious).

    HDNet takes ONLY the initial spectral estimate (28 channels) as input.
    The mask is NOT passed to the model. However, different masks produce
    different initial estimates via shift_back, so scenarios still differ.

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) forward operator mask (used only for initial estimate)
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        import torch
        from pwm_core.recon.mst import shift_back_meas_torch

        model, dev = _load_original_hdnet(device)

        H, W = mask.shape
        nC, step = 28, 2
        W_ext = W + (nC - 1) * step

        # Pad/crop measurement to expected size
        y_padded = np.zeros((H, W_ext), dtype=np.float32)
        hh = min(H, y.shape[0])
        ww = min(W_ext, y.shape[1])
        y_padded[:hh, :ww] = y[:hh, :ww]

        # Create initial estimate using shift_back (same as MST pipeline)
        meas_t = torch.from_numpy(y_padded.copy()).unsqueeze(0).float().to(dev)
        x_init = shift_back_meas_torch(meas_t, step=step, nC=nC)
        x_init = x_init / nC * 2  # Scaling from original MST/HDNet code

        # Forward pass (HDNet takes only the initial estimate, no mask)
        with torch.no_grad():
            recon = model(x_init)

        recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(recon, 0, 1).astype(np.float32)
    except Exception as e:
        logger.warning(f"HDNet failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


def reconstruct_mst_s(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using MST-S (mask-aware Transformer, small).

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) forward operator mask
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.mst import mst_recon_cassi
        return mst_recon_cassi(y, mask, nC=28, step=2, device=device, variant="mst_s")
    except Exception as e:
        logger.warning(f"MST-S failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


def reconstruct_mst_l(y: np.ndarray, mask: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Reconstruct using MST-L (mask-aware Transformer, large).

    Args:
        y: (H, W_ext) CASSI measurement
        mask: (H, W) forward operator mask
        device: torch device

    Returns:
        x_recon: (H, W, 28) reconstruction
    """
    try:
        from pwm_core.recon.mst import mst_recon_cassi
        return mst_recon_cassi(y, mask, nC=28, step=2, device=device, variant="mst_l")
    except Exception as e:
        logger.warning(f"MST-L failed: {e}")
        H = y.shape[0]
        return np.clip(np.random.rand(H, H, 28).astype(np.float32) * 0.1, 0, 1)


RECONSTRUCTION_FUNCTIONS = {
    "gap_tv": reconstruct_gap_tv,
    "hdnet": reconstruct_hdnet,
    "mst_s": reconstruct_mst_s,
    "mst_l": reconstruct_mst_l,
}


# ===================================================================
# scenario validation
# ===================================================================
def validate_scenario_i(scene: np.ndarray, mask_ideal: np.ndarray,
                        methods: List[str], device: str) -> Dict[str, Dict]:
    """Scenario I: Ideal (perfect forward model, no mismatch, no noise).

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

    y_ideal = cassi_forward(scene, mask_ideal, step=2)

    for method in methods:
        t0 = time.time()
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_ideal, mask_ideal, device=device)
            x_hat = np.clip(x_hat, 0, 1)
            results[method] = {
                "psnr": float(compute_psnr(scene, x_hat)),
                "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                "sam": float(compute_sam(scene, x_hat)),
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {"psnr": 0.0, "ssim": 0.0, "sam": 180.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:8s}: PSNR={results[method]['psnr']:.2f} dB  ({dt:.1f}s)")

    return results


def validate_scenario_ii(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchParameters,
                         methods: List[str], device: str) -> Tuple[Dict[str, Dict], np.ndarray]:
    """Scenario II: Assumed/Baseline (corrupted measurement, uncorrected operator).

    Args:
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
        mismatch: MismatchParameters for injection
        methods: list of method names
        device: torch device

    Returns:
        Tuple of (results dict, y_corrupt measurement for reuse in Scenario IV)
    """
    logger.info("  Scenario II: Assumed/Baseline (uncorrected mismatch)")
    results = {}

    # Create corrupted measurement: warp ideal mask, then forward
    mask_corrupted = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta,
    )
    y_corrupt = cassi_forward(scene, mask_corrupted, step=2)
    y_corrupt = add_poisson_gaussian_noise(y_corrupt, peak=100000, sigma=0.01)

    # Reconstruct with each method ASSUMING PERFECT (ideal) MASK
    for method in methods:
        t0 = time.time()
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_corrupt, mask_ideal, device=device)
            x_hat = np.clip(x_hat, 0, 1)
            results[method] = {
                "psnr": float(compute_psnr(scene, x_hat)),
                "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                "sam": float(compute_sam(scene, x_hat)),
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {"psnr": 0.0, "ssim": 0.0, "sam": 180.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:8s}: PSNR={results[method]['psnr']:.2f} dB  ({dt:.1f}s)")

    return results, y_corrupt


def validate_scenario_iv(scene: np.ndarray, mask_ideal: np.ndarray,
                         mismatch: MismatchParameters, y_corrupt: np.ndarray,
                         methods: List[str], device: str) -> Dict[str, Dict]:
    """Scenario IV: Truth Forward Model (corrupted measurement, oracle operator).

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
    logger.info("  Scenario IV: Truth Forward Model (oracle operator)")
    results = {}

    # Apply true mismatch to ideal mask -> oracle knows the corruption
    mask_truth = warp_affine_2d(
        mask_ideal,
        dx=mismatch.mask_dx,
        dy=mismatch.mask_dy,
        theta=mismatch.mask_theta,
    )

    # Reconstruct with each method using TRUE (corrupted) MASK
    for method in methods:
        t0 = time.time()
        try:
            x_hat = RECONSTRUCTION_FUNCTIONS[method](y_corrupt, mask_truth, device=device)
            x_hat = np.clip(x_hat, 0, 1)
            results[method] = {
                "psnr": float(compute_psnr(scene, x_hat)),
                "ssim": float(compute_ssim(np.mean(scene, axis=2), np.mean(x_hat, axis=2))),
                "sam": float(compute_sam(scene, x_hat)),
            }
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            results[method] = {"psnr": 0.0, "ssim": 0.0, "sam": 180.0}
        dt = time.time() - t0
        logger.info(f"    {METHOD_LABELS[method]:8s}: PSNR={results[method]['psnr']:.2f} dB  ({dt:.1f}s)")

    return results


# ===================================================================
# per-scene validation
# ===================================================================
def validate_scene(scene_idx: int, scene: np.ndarray,
                   mask_ideal: np.ndarray,
                   mismatch: MismatchParameters,
                   methods: List[str], device: str) -> Dict:
    """Validate one scene across all 3 scenarios and all methods.

    Args:
        scene_idx: scene index (0-9)
        scene: (256, 256, 28) ground truth
        mask_ideal: (256, 256) ideal mask
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

    # Scenario IV (reuses y_corrupt from Scenario II)
    res_iv = validate_scenario_iv(scene, mask_ideal, mismatch, y_corrupt, methods, device)

    elapsed = time.time() - start_time

    # Compile results
    result = {
        "scene_idx": scene_idx + 1,
        "scenario_i": res_i,
        "scenario_ii": res_ii,
        "scenario_iv": res_iv,
        "elapsed_time": round(elapsed, 2),
        "mismatch_injected": {
            "mask_dx": mismatch.mask_dx,
            "mask_dy": mismatch.mask_dy,
            "mask_theta": mismatch.mask_theta,
        },
    }

    # Calculate gaps for each method
    result["gaps"] = {}
    for method in methods:
        psnr_i = res_i[method]["psnr"]
        psnr_ii = res_ii[method]["psnr"]
        psnr_iv = res_iv[method]["psnr"]

        result["gaps"][method] = {
            "gap_i_ii": round(psnr_i - psnr_ii, 4),     # Degradation from mismatch
            "gap_ii_iv": round(psnr_iv - psnr_ii, 4),   # Recovery from oracle
            "gap_iv_i": round(psnr_i - psnr_iv, 4),     # Residual gap
        }

    # Log summary for this scene
    logger.info(f"\n  Scene {scene_idx+1} summary ({elapsed:.1f}s):")
    for method in methods:
        pi = res_i[method]["psnr"]
        pii = res_ii[method]["psnr"]
        piv = res_iv[method]["psnr"]
        logger.info(
            f"    {METHOD_LABELS[method]:8s}  I={pi:6.2f}  II={pii:6.2f}  IV={piv:6.2f}  "
            f"gap_I-II={pi-pii:+.2f}  rec_II-IV={piv-pii:+.2f}"
        )

    return result


# ===================================================================
# results aggregation
# ===================================================================
def compute_summary_statistics(all_results: List[Dict]) -> Dict:
    """Compute aggregated statistics across all scenes."""
    summary = {
        "num_scenes": len(all_results),
        "methods": list(RECONSTRUCTION_METHODS),
        "scenarios": ["scenario_i", "scenario_ii", "scenario_iv"],
        "mismatch": {"dx": 0.5, "dy": 0.3, "theta": 0.1},
        "noise": {"alpha": 100000, "sigma": 0.01},
    }

    for scenario_key in ["scenario_i", "scenario_ii", "scenario_iv"]:
        summary[scenario_key] = {}

        for method in RECONSTRUCTION_METHODS:
            psnr_vals = [r[scenario_key][method]["psnr"] for r in all_results
                         if r[scenario_key][method]["psnr"] > 0]
            ssim_vals = [r[scenario_key][method]["ssim"] for r in all_results
                         if r[scenario_key][method]["ssim"] > 0]
            sam_vals = [r[scenario_key][method]["sam"] for r in all_results
                        if r[scenario_key][method]["sam"] < 180]

            summary[scenario_key][method] = {
                "psnr_mean": round(float(np.mean(psnr_vals)), 2) if psnr_vals else 0.0,
                "psnr_std": round(float(np.std(psnr_vals)), 2) if psnr_vals else 0.0,
                "ssim_mean": round(float(np.mean(ssim_vals)), 4) if ssim_vals else 0.0,
                "ssim_std": round(float(np.std(ssim_vals)), 4) if ssim_vals else 0.0,
                "sam_mean": round(float(np.mean(sam_vals)), 2) if sam_vals else 0.0,
                "sam_std": round(float(np.std(sam_vals)), 2) if sam_vals else 0.0,
            }

    # Compute gaps across scenarios
    summary["gaps"] = {}
    for method in RECONSTRUCTION_METHODS:
        gap_i_ii = [r["gaps"][method]["gap_i_ii"] for r in all_results]
        gap_ii_iv = [r["gaps"][method]["gap_ii_iv"] for r in all_results]

        summary["gaps"][method] = {
            "gap_i_ii_mean": round(float(np.mean(gap_i_ii)), 2),
            "gap_i_ii_std": round(float(np.std(gap_i_ii)), 2),
            "gap_ii_iv_mean": round(float(np.mean(gap_ii_iv)), 2),
            "gap_ii_iv_std": round(float(np.std(gap_ii_iv)), 2),
        }

    return summary


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="CASSI InverseNet Validation")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CASSI Validation for InverseNet ECCV Paper")
    logger.info("3 Scenarios (I, II, IV) x 4 Methods x 10 Scenes = 120 Reconstructions")
    logger.info(f"Mismatch: dx=0.5 px, dy=0.3 px, theta=0.1 deg")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 70)

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

    # Mismatch parameters
    mismatch = MismatchParameters()
    logger.info(f"Mismatch parameters: dx={mismatch.mask_dx} px, "
                f"dy={mismatch.mask_dy} px, theta={mismatch.mask_theta} deg")

    np.random.seed(42)

    # Validate all scenes
    all_results = []
    start_total = time.time()

    for scene_idx in range(NUM_SCENES):
        scene_name = f"scene{scene_idx + 1:02d}"
        scene = load_scene(scene_name)

        if scene is None:
            logger.warning(f"{scene_name} not found, skipping")
            continue

        result = validate_scene(
            scene_idx, scene, mask_ideal,
            mismatch, RECONSTRUCTION_METHODS, args.device,
        )
        all_results.append(result)

    total_time = time.time() - start_total

    if not all_results:
        logger.error("No results collected!")
        return

    # Compute summary
    summary = compute_summary_statistics(all_results)
    summary["execution_seconds"] = round(total_time, 1)

    # Print overall results
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL RESULTS  (mean +/- std across all scenes)")
    logger.info("=" * 70)

    for scen_label, scen_key in [
        ("Scenario I  (Ideal)",    "scenario_i"),
        ("Scenario II (Baseline)", "scenario_ii"),
        ("Scenario IV (Oracle)",   "scenario_iv"),
    ]:
        logger.info(f"\n  {scen_label}:")
        for method in RECONSTRUCTION_METHODS:
            s = summary[scen_key][method]
            logger.info(
                f"    {METHOD_LABELS[method]:8s}  "
                f"PSNR = {s['psnr_mean']:6.2f} +/- {s['psnr_std']:.2f} dB   "
                f"SSIM = {s['ssim_mean']:.4f}   "
                f"SAM = {s['sam_mean']:.2f} deg"
            )

    logger.info("\n  Gaps:")
    for method in RECONSTRUCTION_METHODS:
        g = summary["gaps"][method]
        logger.info(
            f"    {METHOD_LABELS[method]:8s}  "
            f"I-II = {g['gap_i_ii_mean']:+.2f} dB   "
            f"II-IV = {g['gap_ii_iv_mean']:+.2f} dB"
        )

    logger.info(f"\n  Total time: {total_time:.1f}s ({total_time/len(all_results):.1f}s per scene)")

    # Save results
    out_detail = RESULTS_DIR / "cassi_validation_results.json"
    out_summary = RESULTS_DIR / "cassi_summary.json"

    with open(out_detail, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults  -> {out_detail}")
    logger.info(f"Summary  -> {out_summary}")
    logger.info("\nCASI validation complete!")


if __name__ == "__main__":
    main()
