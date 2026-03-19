#!/usr/bin/env python3
"""CASSI Real Data Validation for InverseNet ECCV 2026.

Validates classical/PnP reconstruction methods on 5 real CASSI scenes from
the TSA real dataset.  Two conditions:
  - Calibrated:  use the hardware-calibrated mask
  - Mismatched:  shift mask by (dx=0.5, dy=0.3) to simulate operator mismatch

No ground truth — uses measurement residual as quality metric.
Reference reconstructions used for supplementary PSNR only.

Methods:
  GAP-TV     -- classical iterative (mask-aware)
  PnP-HSICNN -- GAP + HSI-SDeCNN deep denoiser (mask-aware)

Usage:
    python validate_cassi_real.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

DATASET_REAL = Path("/home/spiritai/MST-main/datasets/TSA_real_data")
RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RECON_DIR = RESULTS_DIR / "cassi_real_reconstructions"
RECON_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
NUM_SCENES = 5
SCENE_SIZE = 660
N_BANDS = 28
STEP = 2
W_EXT = SCENE_SIZE + (N_BANDS - 1) * STEP  # 714

METHOD_LABELS = {
    "gap_tv": "GAP-TV",
    "pnp_hsicnn": "PnP-HSICNN",
}

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_real_measurement(scene_idx: int) -> Optional[np.ndarray]:
    """Load real measurement (660x714)."""
    path = DATASET_REAL / "Measurements" / f"scene{scene_idx}.mat"
    if not path.exists():
        return None
    data = sio.loadmat(str(path))
    return data["meas_real"].astype(np.float32)


def load_reference_reconstruction(scene_idx: int) -> Optional[np.ndarray]:
    """Load reference reconstruction (660x660x28) as pseudo-ground-truth."""
    path = DATASET_REAL / "TSA_reconstruction" / f"Recon_scene{scene_idx}.mat"
    if not path.exists():
        return None
    data = sio.loadmat(str(path))
    return data["recon"].astype(np.float32)


def load_mask() -> Optional[np.ndarray]:
    """Load real mask (660x660)."""
    path = DATASET_REAL / "mask.mat"
    if not path.exists():
        return None
    data = sio.loadmat(str(path))
    return data["mask"].astype(np.float32)


def shift_mask_2d(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel shift of 2D mask via bilinear interpolation."""
    from scipy.ndimage import shift as ndi_shift
    shifted = ndi_shift(mask, [dy, dx], order=1, mode="constant", cval=0.0)
    return np.clip(shifted, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """PSNR (dB), data range [0, max(x_true)]."""
    x_true = x_true.astype(np.float64)
    x_recon = x_recon.astype(np.float64)
    mse = float(np.mean((x_true - x_recon) ** 2))
    if mse < 1e-10:
        return 100.0
    max_val = float(np.max(x_true))
    if max_val < 1e-10:
        max_val = 1.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim_2d(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """SSIM on 2D arrays."""
    from scipy.signal import fftconvolve
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    w = np.ones((win, win), dtype=np.float64) / (win * win)
    mu_a = fftconvolve(a, w, mode="same")
    mu_b = fftconvolve(b, w, mode="same")
    sig_a2 = fftconvolve(a * a, w, mode="same") - mu_a ** 2
    sig_b2 = fftconvolve(b * b, w, mode="same") - mu_b ** 2
    sig_ab = fftconvolve(a * b, w, mode="same") - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


def compute_ssim(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """Band-averaged SSIM for 3D cubes."""
    if x_true.ndim == 3:
        vals = [compute_ssim_2d(x_true[:, :, k], x_recon[:, :, k])
                for k in range(x_true.shape[2])]
        return float(np.mean(vals))
    return compute_ssim_2d(x_true, x_recon)


# ---------------------------------------------------------------------------
# CASSI forward model (for measurement residual computation)
# ---------------------------------------------------------------------------
def cassi_forward(x: np.ndarray, mask: np.ndarray,
                  step: int = STEP, nC: int = N_BANDS) -> np.ndarray:
    """CASSI forward model."""
    H, W = mask.shape[:2]
    W_ext = W + (nC - 1) * step
    y = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        y[:, k * step:k * step + W] += mask * x[:, :, k]
    return y


def compute_measurement_residual(meas: np.ndarray, x_hat: np.ndarray,
                                  mask: np.ndarray) -> float:
    """Normalised measurement residual: ||y - Phi(x_hat)||^2 / ||y||^2."""
    y_pred = cassi_forward(x_hat, mask)
    hh = min(meas.shape[0], y_pred.shape[0])
    ww = min(meas.shape[1], y_pred.shape[1])
    residual = float(np.sum((meas[:hh, :ww] - y_pred[:hh, :ww]) ** 2))
    norm_y = float(np.sum(meas[:hh, :ww] ** 2))
    if norm_y < 1e-10:
        return 0.0
    return residual / norm_y


# ---------------------------------------------------------------------------
# reconstruction: GAP-TV for real CASSI data (660x714)
# ---------------------------------------------------------------------------
def gap_tv_real(meas: np.ndarray, mask: np.ndarray,
                iterations: int = 100, tv_weight: float = 0.1,
                tv_iter: int = 5) -> np.ndarray:
    """GAP-TV for real CASSI data using 2D mask + integer step."""
    from skimage.restoration import denoise_tv_chambolle

    H = mask.shape[0]
    nC = N_BANDS
    step = STEP
    W_ext = H + (nC - 1) * step

    mask = np.clip(mask, 0, 1).astype(np.float32)
    Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        Phi_sum[:, k * step:k * step + H] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    x = np.zeros((H, H, nC), dtype=np.float32)
    for k in range(nC):
        x[:, :, k] = mask * meas[:, k * step:k * step + H] / np.maximum(
            Phi_sum[:, k * step:k * step + H], 1e-6)

    y1 = np.zeros((H, W_ext), dtype=np.float32)
    y = np.zeros((H, W_ext), dtype=np.float32)
    hh = min(H, meas.shape[0])
    ww = min(W_ext, meas.shape[1])
    y[:hh, :ww] = meas[:hh, :ww]

    for _ in range(iterations):
        y_est = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            y_est[:, k * step:k * step + H] += mask * x[:, :, k]

        y1 += (y - y_est)
        norm_r = (y1 - y_est) / Phi_sum
        for k in range(nC):
            x[:, :, k] += mask * norm_r[:, k * step:k * step + H]

        x = denoise_tv_chambolle(
            np.clip(x, 0, None), weight=tv_weight,
            max_num_iter=tv_iter, channel_axis=2,
        ).astype(np.float32)

    return np.clip(x, 0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# reconstruction: PnP-HSICNN for real CASSI data (660x714)
# ---------------------------------------------------------------------------
_pnp_hsicnn_cache = {}


def _load_pnp_hsicnn(device: str):
    """Load HSI-SDeCNN denoiser from PnP-CASSI."""
    if "model" in _pnp_hsicnn_cache:
        return _pnp_hsicnn_cache["model"]

    import torch
    import importlib.util

    hsi_path = "/home/spiritai/PnP-CASSI-main/hsi.py"
    spec = importlib.util.spec_from_file_location("hsi_sdecnn", hsi_path)
    hsi_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hsi_mod)

    dev = torch.device(device)
    model = hsi_mod.HSI_SDeCNN(in_nc=7, out_nc=1, nc=128, nb=15).to(dev)

    weights_path = "/home/spiritai/PnP-CASSI-main/check_points/deep_denoiser.pth"
    state_dict = torch.load(weights_path, map_location=dev, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info("  Loaded HSI-SDeCNN denoiser with pretrained weights")

    _pnp_hsicnn_cache["model"] = (model, dev)
    return model, dev


def _apply_hsicnn_denoiser(x: np.ndarray, model, dev,
                           nC: int = 28, sigma_val: float = 10.0) -> np.ndarray:
    """Apply HSI-SDeCNN band-by-band with 7-channel context window."""
    import torch

    result = np.zeros_like(x)
    sigma_t = torch.full((1, 1, 1, 1), sigma_val / 255.0, device=dev)

    for i in range(nC):
        if i < 3:
            if i == 0:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 0],
                                    x[:, :, 0:4]))
            elif i == 1:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 0],
                                    x[:, :, 1:5]))
            else:
                net_in = np.dstack((x[:, :, 0], x[:, :, 0], x[:, :, 1],
                                    x[:, :, 2:6]))
        elif i > nC - 4:
            if i == nC - 3:
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1],
                                    x[:, :, i + 2], x[:, :, i + 2]))
            elif i == nC - 2:
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i + 1],
                                    x[:, :, i + 1], x[:, :, i + 1]))
            else:
                net_in = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i],
                                    x[:, :, i], x[:, :, i]))
        else:
            net_in = x[:, :, i - 3:i + 4]

        net_t = (torch.from_numpy(np.ascontiguousarray(net_in))
                 .permute(2, 0, 1).float().unsqueeze(0).to(dev))
        with torch.no_grad():
            out = model(net_t, sigma_t)
        result[:, :, i] = out.squeeze().cpu().numpy()

    return result


def _is_hsicnn_iter(k: int) -> bool:
    """True for iterations 83+ with 3/4 HSICNN schedule."""
    if k < 83:
        return False
    return (k - 83) % 4 < 3


def pnp_hsicnn_real(meas: np.ndarray, mask: np.ndarray,
                     device: str = "cuda:0") -> np.ndarray:
    """PnP-HSICNN for real CASSI data (handles full 660x660 natively).

    124-iter GAP with hybrid TV (iters 0-82) + HSICNN (iters 83-123).
    HSI-SDeCNN uses PixelUnshuffle -> 330x330 internal, fits in GPU memory.
    """
    from skimage.restoration import denoise_tv_chambolle

    model, dev = _load_pnp_hsicnn(device)

    H = mask.shape[0]
    nC = N_BANDS
    step = STEP
    W_ext = H + (nC - 1) * step

    mask_c = np.clip(mask, 0, 1).astype(np.float32)
    Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        Phi_sum[:, k * step:k * step + H] += mask_c ** 2
    Phi_sum = np.maximum(Phi_sum, 1.0)

    # Initialize with adjoint
    x = np.zeros((H, H, nC), dtype=np.float32)
    for k in range(nC):
        x[:, :, k] = mask_c * meas[:, k * step:k * step + H]

    y_nom = np.zeros((H, W_ext), dtype=np.float32)
    hh = min(H, meas.shape[0])
    ww = min(W_ext, meas.shape[1])
    y_nom[:hh, :ww] = meas[:hh, :ww]

    y1 = np.zeros_like(y_nom)
    nsig_tv = 12.75
    n_total = 124

    for k_iter in range(n_total):
        y_est = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            y_est[:, k * step:k * step + H] += mask_c * x[:, :, k]

        y1 += (y_nom - y_est)
        norm_r = (y1 - y_est) / Phi_sum

        for k in range(nC):
            x[:, :, k] += mask_c * norm_r[:, k * step:k * step + H]

        if _is_hsicnn_iter(k_iter):
            x = _apply_hsicnn_denoiser(
                np.clip(x, 0, None), model, dev, nC=nC, sigma_val=10.0)
        else:
            x = denoise_tv_chambolle(
                np.clip(x, 0, None), weight=nsig_tv / 255.0,
                max_num_iter=5, channel_axis=2).astype(np.float32)

    return np.clip(x, 0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# main validation
# ---------------------------------------------------------------------------
def validate_scene(scene_idx: int, meas: np.ndarray,
                   mask: np.ndarray, reference: Optional[np.ndarray],
                   methods: List[str], device: str,
                   dx_mismatch: float = 0.5,
                   dy_mismatch: float = 0.3) -> Dict:
    """Validate one real scene with calibrated and mismatched masks.

    Computes measurement residual (ground-truth-free metric) for all methods.
    Also computes PSNR against reference reconstruction if available.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Real Scene {scene_idx}")
    logger.info(f"{'='*60}")

    reference_norm = None
    if reference is not None:
        ref_max = reference.max()
        reference_norm = reference / ref_max if ref_max > 0 else reference

    result = {"scene_idx": scene_idx, "calibrated": {}, "mismatched": {}}
    mask_shifted = shift_mask_2d(mask, dx_mismatch, dy_mismatch)

    for condition, mask_used in [("calibrated", mask), ("mismatched", mask_shifted)]:
        logger.info(f"  {condition.capitalize()}:")
        for method in methods:
            t0 = time.time()
            try:
                if method == "gap_tv":
                    x_hat = gap_tv_real(meas, mask_used)
                elif method == "pnp_hsicnn":
                    x_hat = pnp_hsicnn_real(meas, mask_used, device)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Cross-residual: always use CALIBRATED mask for residual
                residual = compute_measurement_residual(meas, x_hat, mask)

                # PSNR against reference (supplementary only)
                psnr_ref = 0.0
                ssim_ref = 0.0
                if reference_norm is not None:
                    x_max = x_hat.max()
                    x_hat_norm = x_hat / x_max if x_max > 0 else x_hat
                    psnr_ref = compute_psnr(reference_norm, x_hat_norm)
                    ssim_ref = compute_ssim(reference_norm, x_hat_norm)

                result[condition][method] = {
                    "residual": round(residual, 8),
                    "psnr_vs_ref": round(psnr_ref, 2),
                    "ssim_vs_ref": round(ssim_ref, 4),
                }

                np.save(str(RECON_DIR / f"scene{scene_idx}_{condition}_{method}.npy"), x_hat)
            except Exception as e:
                logger.error(f"    {method} failed: {e}")
                result[condition][method] = {"residual": -1.0, "psnr_vs_ref": 0.0, "ssim_vs_ref": 0.0}

            dt = time.time() - t0
            r = result[condition][method]
            logger.info(f"    {METHOD_LABELS[method]:12s}: residual={r['residual']:.6f}  "
                        f"PSNR_ref={r['psnr_vs_ref']:.2f}  ({dt:.1f}s)")

    # Compute residual ratio (mismatched / calibrated)
    result["residual_ratio"] = {}
    for method in methods:
        r_cal = result["calibrated"][method]["residual"]
        r_mis = result["mismatched"][method]["residual"]
        if r_cal > 0 and r_mis > 0:
            ratio = r_mis / r_cal
        else:
            ratio = 0.0
        result["residual_ratio"][method] = round(ratio, 1)
        logger.info(f"    {METHOD_LABELS[method]:12s}: residual ratio = {ratio:.1f}x")

    return result


def main():
    parser = argparse.ArgumentParser(description="CASSI Real Data Validation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    methods = ["gap_tv", "pnp_hsicnn"]

    logger.info("=" * 60)
    logger.info("CASSI Real Data Validation for InverseNet ECCV 2026")
    logger.info(f"5 scenes x 2 methods x 2 conditions = 20 reconstructions")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    mask = load_mask()
    if mask is None:
        logger.error("Mask not found!")
        return

    logger.info(f"Mask shape: {mask.shape}")

    all_results = []
    t_total = time.time()

    for si in range(1, NUM_SCENES + 1):
        meas = load_real_measurement(si)
        ref = load_reference_reconstruction(si)
        if meas is None:
            logger.warning(f"Scene {si}: measurement missing, skipping")
            continue

        result = validate_scene(si, meas, mask, ref, methods, args.device)
        all_results.append(result)

    elapsed = time.time() - t_total

    # Summary
    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY  (mean across scenes)")
        logger.info(f"{'='*60}")
        for condition in ["calibrated", "mismatched"]:
            logger.info(f"\n  {condition.capitalize()}:")
            for method in methods:
                residuals = [r[condition][method]["residual"] for r in all_results
                             if r[condition][method]["residual"] >= 0]
                if residuals:
                    logger.info(f"    {METHOD_LABELS[method]:12s}: residual={np.mean(residuals):.6f}")

        logger.info(f"\n  Residual ratio (mismatched / calibrated):")
        for method in methods:
            ratios = [r["residual_ratio"][method] for r in all_results
                      if r["residual_ratio"][method] > 0]
            if ratios:
                logger.info(f"    {METHOD_LABELS[method]:12s}: {np.mean(ratios):.1f}x")

    # Save results
    summary = {
        "experiment": "cassi_real_data",
        "num_scenes": len(all_results),
        "mismatch": {"dx": 0.5, "dy": 0.3},
        "methods": methods,
        "per_scene": all_results,
        "execution_seconds": round(elapsed, 1),
    }

    summary["mean"] = {}
    for condition in ["calibrated", "mismatched"]:
        summary["mean"][condition] = {}
        for method in methods:
            residuals = [r[condition][method]["residual"] for r in all_results
                         if r[condition][method]["residual"] >= 0]
            psnrs = [r[condition][method]["psnr_vs_ref"] for r in all_results
                     if r[condition][method]["psnr_vs_ref"] > 0]
            summary["mean"][condition][method] = {
                "residual_mean": round(float(np.mean(residuals)), 8) if residuals else 0.0,
                "psnr_ref_mean": round(float(np.mean(psnrs)), 2) if psnrs else 0.0,
            }
    summary["mean"]["residual_ratio"] = {}
    for method in methods:
        ratios = [r["residual_ratio"][method] for r in all_results
                  if r["residual_ratio"][method] > 0]
        summary["mean"]["residual_ratio"][method] = round(float(np.mean(ratios)), 1) if ratios else 0.0

    out_path = RESULTS_DIR / "cassi_real_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Reconstructions -> {RECON_DIR}")
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
