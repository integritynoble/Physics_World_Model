#!/usr/bin/env python3
"""CASSI Real Data Validation for InverseNet ECCV 2026.

Validates reconstruction methods on 5 real CASSI scenes from the TSA real
dataset.  Two conditions:
  - Calibrated:  use the hardware-calibrated mask_3d_shift
  - Mismatched:  shift mask by (dx=0.5, dy=0.3) to simulate operator mismatch

Reference reconstructions (Recon_scene*.mat) serve as pseudo-ground-truth
for PSNR/SSIM computation.

Methods:
  GAP-TV   -- classical iterative (mask-aware)
  HDNet    -- dual-domain deep learning (mask-oblivious)
  MST-S    -- mask-guided Transformer (small)
  MST-L    -- mask-guided Transformer (large)

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
    "hdnet": "HDNet",
    "mst_s": "MST-S",
    "mst_l": "MST-L",
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


def load_mask_3d_shift() -> Optional[np.ndarray]:
    """Load pre-computed mask_3d_shift (660x714x28)."""
    path = DATASET_REAL / "mask_3d_shift.mat"
    if not path.exists():
        return None
    data = sio.loadmat(str(path))
    return data["mask_3d_shift"].astype(np.float32)


def build_mask_3d_shift(mask: np.ndarray) -> np.ndarray:
    """Build mask_3d_shift from 2D mask (replicate MST pipeline)."""
    H, W = mask.shape
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, N_BANDS))
    mask_3d_shift = np.zeros((H, H + (N_BANDS - 1) * STEP, N_BANDS), dtype=np.float32)
    mask_3d_shift[:, 0:H, :] = mask_3d
    for t in range(N_BANDS):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], STEP * t, axis=1)
    return mask_3d_shift


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
    """CASSI forward model: y(i,j) = sum_lambda M(i, j-d(lambda)) * x(i,j,lambda).

    Args:
        x: reconstructed cube (H, W, nC)
        mask: 2D binary mask (H, W)
        step: dispersion step (pixels per band)
        nC: number of spectral bands

    Returns:
        y: simulated measurement (H, W_ext) where W_ext = W + (nC-1)*step
    """
    H, W = mask.shape[:2]
    W_ext = W + (nC - 1) * step
    y = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        y[:, k * step:k * step + W] += mask * x[:, :, k]
    return y


def compute_measurement_residual(meas: np.ndarray, x_hat: np.ndarray,
                                  mask: np.ndarray) -> float:
    """Normalised measurement residual: ||y - Phi(x_hat)||^2 / ||y||^2.

    This metric requires no ground truth — only the measurement and the
    assumed forward operator. A small residual means the reconstruction
    is consistent with the measurement under the assumed mask.
    """
    y_pred = cassi_forward(x_hat, mask)
    # Match sizes (meas may be slightly different due to rounding)
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

    # Normalised back-projection init
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
# reconstruction: deep learning methods for real CASSI
# ---------------------------------------------------------------------------
import torch

_model_cache = {}


def _build_mask_3d_shift_torch(mask: np.ndarray, device: str):
    """Build mask_3d_shift tensor for MST pipeline."""
    mask_3d_shift = build_mask_3d_shift(mask)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    m3d = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    m3ds = torch.FloatTensor(mask_3d_shift_s.copy()).unsqueeze(0).to(device)
    return m3d, m3ds


def _load_model(method: str, device: str):
    """Load and cache a pretrained model."""
    cache_key = f"{method}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    import importlib.util

    dev = torch.device(device)

    if method == "hdnet":
        hdnet_path = "/home/spiritai/MST-main/simulation/test_code/architecture/HDNet.py"
        spec = importlib.util.spec_from_file_location("hdnet_orig", hdnet_path)
        hdnet_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hdnet_mod)
        model = hdnet_mod.HDNet(in_ch=28, out_ch=28).to(dev)
        weights_path = "/home/spiritai/MST-main/model_zoo/hdnet/hdnet.pth"
        checkpoint = torch.load(weights_path, map_location=dev, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
    elif method in ("mst_s", "mst_l"):
        sys.path.insert(0, "/home/spiritai/MST-main/simulation/test_code")
        from architecture import MST
        dim = 28
        if method == "mst_s":
            stage = 2
            model = MST(dim=dim, stage=stage, num_blocks=[2, 2, 2]).to(dev)
            weights_path = "/home/spiritai/MST-main/model_zoo/mst/mst_s.pth"
        else:
            stage = 2
            model = MST(dim=dim, stage=stage, num_blocks=[4, 7, 5]).to(dev)
            weights_path = "/home/spiritai/MST-main/model_zoo/mst/mst_l.pth"
        checkpoint = torch.load(weights_path, map_location=dev, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.eval()
    _model_cache[cache_key] = model
    logger.info(f"  Loaded {method} model")
    return model


def _crop_for_mst(meas: np.ndarray, mask: np.ndarray,
                   crop_h: int = 256, step: int = STEP, nC: int = N_BANDS):
    """Center-crop measurement and mask to fit MST's hardcoded 256 spatial dim.

    MST has shift_back with hardcoded `256//row` which breaks for non-256 sizes.
    We crop to (crop_h, crop_w_ext) measurement and (crop_h, crop_h) mask.

    Returns:
        meas_crop: (crop_h, crop_w_ext)
        mask_crop: (crop_h, crop_h)
        (r0, c0): crop origin for later reference extraction
    """
    H_full = mask.shape[0]
    W_ext = meas.shape[1]
    crop_w_ext = crop_h + (nC - 1) * step  # 256 + 54 = 310

    r0 = (H_full - crop_h) // 2
    c0 = (H_full - crop_h) // 2  # mask is HxH

    meas_crop = meas[r0:r0 + crop_h, c0:c0 + crop_w_ext]
    mask_crop = mask[r0:r0 + crop_h, c0:c0 + crop_h]
    return meas_crop, mask_crop, (r0, c0)


def reconstruct_dl_real(meas: np.ndarray, mask: np.ndarray,
                        method: str, device: str) -> np.ndarray:
    """Reconstruct real CASSI data using deep learning method.

    HDNet: fully convolutional, handles full 660x714 input.
    MST-S/L: internal shift_back hardcoded for 256 spatial dim,
             so we center-crop to 256x310 and return 256x256x28.
    """
    from pwm_core.recon.mst import shift_back_meas_torch

    dev = torch.device(device)
    model = _load_model(method, device)

    is_mst = method in ("mst_s", "mst_l")

    if is_mst:
        # Crop for MST compatibility
        meas_proc, mask_proc, crop_origin = _crop_for_mst(meas, mask)
    else:
        meas_proc, mask_proc = meas, mask
        crop_origin = None

    # Normalise measurement (from MST test.py)
    meas_norm = meas_proc.copy()
    meas_max = meas_norm.max()
    if meas_max > 0:
        meas_norm = meas_norm / meas_max * 0.8

    # Build mask tensors
    m3d, m3ds = _build_mask_3d_shift_torch(mask_proc, device)
    meas_t = torch.FloatTensor(meas_norm).unsqueeze(0).to(dev)

    if method == "hdnet":
        # HDNet takes only initial spectral estimate (no mask input)
        x_init = shift_back_meas_torch(meas_t, step=STEP, nC=N_BANDS)
        x_init = x_init / N_BANDS * 2
        with torch.no_grad():
            out = model(x_init)
    else:
        # MST-S/L: forward(x, mask) where x=[B,28,H,W] is shifted-back meas
        x_init = shift_back_meas_torch(meas_t, step=STEP, nC=N_BANDS)
        x_init = x_init / N_BANDS * 2
        with torch.no_grad():
            out = model(x_init, m3d)

    recon = out.squeeze(0).clamp(min=0.0, max=1.0)
    recon = recon.permute(1, 2, 0).cpu().numpy()  # (H, W, 28)
    return recon.astype(np.float32)


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
    Also computes PSNR against reference reconstruction if available (for
    supplementary analysis only — not used as the primary metric).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Real Scene {scene_idx}")
    logger.info(f"{'='*60}")

    # Normalise reference to [0, 1] if available (for supplementary PSNR only)
    reference_norm = None
    ref_crop = None
    if reference is not None:
        ref_max = reference.max()
        reference_norm = reference / ref_max if ref_max > 0 else reference
        H_full = reference_norm.shape[0]
        crop_h = 256
        r0 = (H_full - crop_h) // 2
        ref_crop = reference_norm[r0:r0 + crop_h, r0:r0 + crop_h, :]

    result = {"scene_idx": scene_idx, "calibrated": {}, "mismatched": {}}
    mask_shifted = shift_mask_2d(mask, dx_mismatch, dy_mismatch)

    for condition, mask_used in [("calibrated", mask), ("mismatched", mask_shifted)]:
        logger.info(f"  {condition.capitalize()}:")
        for method in methods:
            t0 = time.time()
            try:
                if method == "gap_tv":
                    x_hat = gap_tv_real(meas, mask_used)
                else:
                    x_hat = reconstruct_dl_real(meas, mask_used, method, device)

                # --- Measurement residual (primary metric, no ground truth needed) ---
                is_mst = method in ("mst_s", "mst_l")
                if is_mst:
                    # MST reconstructs a 256x256 crop; compute residual on that crop
                    _, mask_crop, (r0c, c0c) = _crop_for_mst(meas, mask_used)
                    meas_crop = meas[r0c:r0c + 256, c0c:c0c + 256 + (N_BANDS - 1) * STEP]
                    residual = compute_measurement_residual(meas_crop, x_hat, mask_crop)
                else:
                    residual = compute_measurement_residual(meas, x_hat, mask_used)

                # --- PSNR against reference (supplementary only) ---
                psnr_ref = 0.0
                ssim_ref = 0.0
                if reference_norm is not None:
                    x_max = x_hat.max()
                    x_hat_norm = x_hat / x_max if x_max > 0 else x_hat
                    ref_for_psnr = ref_crop if is_mst else reference_norm
                    psnr_ref = compute_psnr(ref_for_psnr, x_hat_norm)
                    ssim_ref = compute_ssim(ref_for_psnr, x_hat_norm)

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
            logger.info(f"    {METHOD_LABELS[method]:8s}: residual={r['residual']:.6f}  "
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
        logger.info(f"    {METHOD_LABELS[method]:8s}: residual ratio = {ratio:.1f}×")

    return result


def main():
    parser = argparse.ArgumentParser(description="CASSI Real Data Validation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    methods = ["gap_tv", "hdnet", "mst_s", "mst_l"]

    logger.info("=" * 60)
    logger.info("CASSI Real Data Validation for InverseNet ECCV 2026")
    logger.info(f"5 scenes x 4 methods x 2 conditions = 40 reconstructions")
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
        if meas is None or ref is None:
            logger.warning(f"Scene {si}: data missing, skipping")
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
                    logger.info(f"    {METHOD_LABELS[method]:8s}: residual={np.mean(residuals):.6f}")

        logger.info(f"\n  Residual ratio (mismatched / calibrated):")
        for method in methods:
            ratios = [r["residual_ratio"][method] for r in all_results
                      if r["residual_ratio"][method] > 0]
            if ratios:
                logger.info(f"    {METHOD_LABELS[method]:8s}: {np.mean(ratios):.1f}×")

    # Save results
    summary = {
        "experiment": "cassi_real_data",
        "num_scenes": len(all_results),
        "mismatch": {"dx": 0.5, "dy": 0.3},
        "methods": methods,
        "per_scene": all_results,
        "execution_seconds": round(elapsed, 1),
    }

    # Compute means for summary
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
