#!/usr/bin/env python3
"""Gate 1 & Gate 2 validation experiments across all 7 modalities.

Gate 1 (Information Deficiency): Extreme compression sweeps
Gate 2 (Carrier Budget): Noise/photon-level sweeps

Modalities: CACTI, CASSI, SPC, MRI, CT, Lensless, Ptychography

Usage:
    python papers/inversenet/scripts/validate_gate1_gate2.py --device cuda:0
    python papers/inversenet/scripts/validate_gate1_gate2.py --quick --device cpu
    python papers/inversenet/scripts/validate_gate1_gate2.py --modalities mri ct lensless
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DAVIS_DIR = Path("/home/spiritai/elpData/traindata/DAVIS-480-train/JPEGImages/480p")
KAIST_DIR = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")
SET11_DIR = Path("/home/spiritai/ISTA-Net-PyTorch-master/data/Set11")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("gate12")


# ===================================================================
# Shared Metrics
# ===================================================================
def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray,
                 data_range: float = 1.0) -> float:
    """PSNR between two images in [0, data_range]."""
    mse = float(np.mean((x_true.astype(np.float64) - x_recon.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    return 10.0 * math.log10(data_range ** 2 / mse)


def compute_ssim(x_true: np.ndarray, x_recon: np.ndarray,
                 data_range: float = 1.0) -> float:
    """SSIM between two images."""
    try:
        from skimage.metrics import structural_similarity
        # Handle multi-channel
        if x_true.ndim == 3:
            return float(structural_similarity(
                x_true.astype(np.float64), x_recon.astype(np.float64),
                data_range=data_range, channel_axis=2))
        return float(structural_similarity(
            x_true.astype(np.float64), x_recon.astype(np.float64),
            data_range=data_range))
    except Exception:
        return 0.0


# ===================================================================
# CASSI helpers (inline to avoid fragile cross-script imports)
# ===================================================================
def cassi_forward(scene: np.ndarray, mask: np.ndarray,
                  step: int = 2) -> np.ndarray:
    """CASSI forward model: y[:, k*step:k*step+W] += mask * scene[:,:,k]."""
    H, W, nC = scene.shape
    W_ext = W + (nC - 1) * step
    y = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        y[:, k * step:k * step + W] += mask * scene[:, :, k]
    return y


def gap_tv_cassi(y: np.ndarray, mask: np.ndarray, n_bands: int = 28,
                 step: int = 2, iterations: int = 100,
                 tv_weight: float = 0.1, tv_iter: int = 5) -> np.ndarray:
    """GAP-TV for CASSI (Chambolle TV, Nesterov acceleration)."""
    from skimage.restoration import denoise_tv_chambolle

    H, W = mask.shape
    nC = n_bands
    W_ext = W + (nC - 1) * step
    mask = np.clip(mask, 0, 1).astype(np.float32)

    Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
    for k in range(nC):
        Phi_sum[:, k * step:k * step + W] += mask ** 2
    Phi_sum = np.maximum(Phi_sum, 1e-10)

    x = np.zeros((H, W, nC), dtype=np.float32)
    for k in range(nC):
        x[:, :, k] = mask * y[:, k * step:k * step + W] / np.maximum(
            Phi_sum[:, k * step:k * step + W], 1e-6)

    y1 = np.zeros((H, W_ext), dtype=np.float32)
    for _ in range(iterations):
        y_est = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            y_est[:, k * step:k * step + W] += mask * x[:, :, k]
        y1 += (y[:, :W_ext] - y_est)
        norm_r = (y1 - y_est) / Phi_sum
        for k in range(nC):
            x[:, :, k] += mask * norm_r[:, k * step:k * step + W]
        x = denoise_tv_chambolle(
            np.clip(x, 0, None), weight=tv_weight,
            max_num_iter=tv_iter, channel_axis=2,
        ).astype(np.float32)

    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# SPC helpers (inline block-based CS)
# ===================================================================
BLOCK_SIZE = 33
N_PIX = BLOCK_SIZE ** 2  # 1089


def imread_CS_py(Iorg: np.ndarray):
    """Pad image to multiple of 33."""
    row, col = Iorg.shape
    row_pad = (BLOCK_SIZE - row % BLOCK_SIZE) % BLOCK_SIZE
    col_pad = (BLOCK_SIZE - col % BLOCK_SIZE) % BLOCK_SIZE
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad], dtype=Iorg.dtype)), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad], dtype=Iorg.dtype)), axis=0)
    row_new, col_new = Ipad.shape
    return Iorg, row, col, Ipad, row_new, col_new


def img2col_py(Ipad: np.ndarray) -> np.ndarray:
    """Extract 33x33 column blocks."""
    row, col = Ipad.shape
    row_block = row // BLOCK_SIZE
    col_block = col // BLOCK_SIZE
    block_num = int(row_block * col_block)
    img_col = np.zeros([N_PIX, block_num], dtype=np.float32)
    count = 0
    for x in range(0, row - BLOCK_SIZE + 1, BLOCK_SIZE):
        for yy in range(0, col - BLOCK_SIZE + 1, BLOCK_SIZE):
            img_col[:, count] = Ipad[x:x + BLOCK_SIZE, yy:yy + BLOCK_SIZE].reshape(-1)
            count += 1
    return img_col


def col2im_CS_py(X_col: np.ndarray, row: int, col: int,
                 row_new: int, col_new: int) -> np.ndarray:
    """Reconstruct image from column blocks."""
    X0_rec = np.zeros([row_new, col_new], dtype=np.float32)
    count = 0
    for x in range(0, row_new - BLOCK_SIZE + 1, BLOCK_SIZE):
        for yy in range(0, col_new - BLOCK_SIZE + 1, BLOCK_SIZE):
            X0_rec[x:x + BLOCK_SIZE, yy:yy + BLOCK_SIZE] = \
                X_col[:, count].reshape([BLOCK_SIZE, BLOCK_SIZE])
            count += 1
    return X0_rec[:row, :col]


class FISTATVSolver:
    """FISTA-TV solver for block-based CS (33x33 blocks)."""

    def __init__(self, Phi: np.ndarray, lam: float = 0.005,
                 max_iter: int = 300, tv_inner_iters: int = 10):
        self.Phi = Phi.astype(np.float32)
        self.lam = lam
        self.max_iter = max_iter
        self.tv_inner_iters = tv_inner_iters
        self.L = self._estimate_L(Phi, n_iters=20)
        self.tau = 0.9 / max(self.L, 1e-8)

    @staticmethod
    def _estimate_L(Phi: np.ndarray, n_iters: int = 20) -> float:
        n = Phi.shape[1]
        v = np.random.randn(n).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(n_iters):
            w = Phi.T @ (Phi @ v)
            wn = np.linalg.norm(w) + 1e-12
            v = w / wn
        w = Phi @ v
        s = np.linalg.norm(w)
        return float(s * s)

    def solve_batch(self, y_Bm: np.ndarray) -> np.ndarray:
        """y_Bm: [B, m]. Returns [B, n]."""
        from skimage.restoration import denoise_tv_chambolle

        B = y_Bm.shape[0]
        y = y_Bm.astype(np.float32)

        x0 = y @ self.Phi  # [B, n]
        for b in range(B):
            mn, mx = x0[b].min(), x0[b].max()
            if mx - mn > 1e-8:
                x0[b] = (x0[b] - mn) / (mx - mn)
        x0 = np.clip(x0, 0, 1)

        x = x0.copy()
        z = x0.copy()
        t = 1.0

        for _ in range(self.max_iter):
            residual = z @ self.Phi.T - y
            grad = residual @ self.Phi
            u = z - self.tau * grad

            z_new = np.zeros_like(u)
            for b in range(B):
                u_img = np.clip(u[b].reshape(BLOCK_SIZE, BLOCK_SIZE), 0, 1)
                z_img = denoise_tv_chambolle(
                    u_img.astype(np.float64),
                    weight=self.tau * self.lam,
                    max_num_iter=self.tv_inner_iters)
                z_new[b] = np.clip(z_img, 0, 1).flatten().astype(np.float32)

            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            x_new = z_new + ((t - 1.0) / t_new) * (z_new - x)
            x_new = np.clip(x_new, 0, 1)

            x = z_new
            z = x_new
            t = t_new

        return np.clip(x, 0, 1)


# ===================================================================
# Noise helper
# ===================================================================
def add_poisson_gaussian_noise(y: np.ndarray, peak: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Add Poisson + Gaussian noise to measurement."""
    y = np.maximum(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), 0)
    y_max = max(float(np.max(y)), 1e-10)
    y_scaled = np.clip(y / y_max * peak, 0, None)
    y_poisson = rng.poisson(y_scaled.astype(np.float64)).astype(np.float64)
    sigma_read = max(0.01 * math.sqrt(peak), 1.0)
    y_noisy = y_poisson + rng.normal(0, sigma_read, y_poisson.shape)
    return (y_noisy / peak * y_max).astype(np.float32)


def add_gaussian_noise(y: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise scaled to signal range."""
    y_range = max(float(np.max(np.abs(y))), 1e-10)
    return (y + rng.normal(0, sigma * y_range, y.shape)).astype(y.dtype)


# ===================================================================
# CACTI Gate 1: Sweep compression ratio
# ===================================================================
def run_cacti_gate1(cr_values: List[int], n_sequences: int = 3,
                    device: str = "cpu") -> Dict:
    """CACTI Gate 1: information deficiency via extreme temporal compression."""
    from pwm_core.recon.cacti_solvers import gap_tv_cacti

    logger.info("=== CACTI Gate 1: CR sweep ===")

    # Find DAVIS sequences with enough frames
    if not DAVIS_DIR.exists():
        logger.warning(f"DAVIS directory not found: {DAVIS_DIR}")
        return {"error": "DAVIS not found"}

    seq_dirs = sorted([d for d in DAVIS_DIR.iterdir() if d.is_dir()])
    # Pick sequences with many frames
    seq_info = []
    for d in seq_dirs:
        frames = sorted(d.glob("*.jpg"))
        if len(frames) >= max(cr_values):
            seq_info.append((d.name, len(frames), frames))
    seq_info.sort(key=lambda x: -x[1])  # longest first

    if not seq_info:
        logger.warning("No DAVIS sequences with enough frames")
        return {"error": "insufficient frames"}

    selected = seq_info[:n_sequences]
    logger.info(f"  Selected sequences: {[s[0] for s in selected]}")

    rng = np.random.default_rng(42)
    results = {}

    for seq_name, n_frames, frame_paths in selected:
        logger.info(f"  Sequence: {seq_name} ({n_frames} frames)")

        # Load all frames as grayscale 256x256
        from PIL import Image
        all_frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("L")
            img = np.array(img, dtype=np.float32) / 255.0
            # Center crop to 256x256
            h, w = img.shape
            ch, cw = h // 2, w // 2
            img = img[ch - 128:ch + 128, cw - 128:cw + 128]
            all_frames.append(img)
        all_frames = np.array(all_frames)  # (N, 256, 256)

        seq_results = {}
        for cr in cr_values:
            if len(all_frames) < cr:
                logger.warning(f"    CR={cr}: not enough frames ({len(all_frames)})")
                continue

            # Take first CR frames as ground truth
            gt = all_frames[:cr]  # (CR, 256, 256)
            gt_hwt = np.transpose(gt, (1, 2, 0))  # (256, 256, CR)

            # Random binary mask
            mask = (rng.random((256, 256, cr)) > 0.5).astype(np.float32)

            # Forward: y = sum(frames * mask, axis=2)
            y = np.sum(gt_hwt * mask, axis=2).astype(np.float32)

            # Reconstruct with GAP-TV
            t0 = time.time()
            try:
                x_hat = gap_tv_cacti(y, mask, iterations=80, tv_weight=0.1)
                x_hat = np.clip(x_hat, 0, 1).astype(np.float32)
            except Exception as e:
                logger.warning(f"    CR={cr}: GAP-TV failed: {e}")
                x_hat = np.zeros_like(gt_hwt)
            dt = time.time() - t0

            psnr = compute_psnr(gt_hwt, x_hat)
            ssim = compute_ssim(gt_hwt, x_hat)
            logger.info(f"    CR={cr}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f} ({dt:.1f}s)")
            seq_results[f"cr_{cr}"] = {"psnr": round(psnr, 2), "ssim": round(ssim, 4)}

        results[seq_name] = seq_results

    return results


# ===================================================================
# CACTI Gate 2: Noise sweep
# ===================================================================
def run_cacti_gate2(photon_levels: List[int], n_sequences: int = 2,
                    device: str = "cpu") -> Dict:
    """CACTI Gate 2: carrier budget via photon-level sweep at CR=8."""
    from pwm_core.recon.cacti_solvers import gap_tv_cacti, pnp_ffdnet_cacti

    logger.info("=== CACTI Gate 2: Noise sweep (CR=8) ===")

    if not DAVIS_DIR.exists():
        return {"error": "DAVIS not found"}

    seq_dirs = sorted([d for d in DAVIS_DIR.iterdir() if d.is_dir()])
    selected = []
    for d in seq_dirs:
        frames = sorted(d.glob("*.jpg"))
        if len(frames) >= 8:
            selected.append((d.name, frames))
    selected = selected[:n_sequences]

    rng = np.random.default_rng(42)
    results = {}
    cr = 8

    for seq_name, frame_paths in selected:
        from PIL import Image
        frames = []
        for fp in frame_paths[:cr]:
            img = Image.open(fp).convert("L")
            img = np.array(img, dtype=np.float32) / 255.0
            h, w = img.shape
            ch, cw = h // 2, w // 2
            img = img[ch - 128:ch + 128, cw - 128:cw + 128]
            frames.append(img)
        gt = np.stack(frames, axis=-1)  # (256, 256, 8)

        mask = (rng.random((256, 256, cr)) > 0.5).astype(np.float32)
        y_clean = np.sum(gt * mask, axis=2).astype(np.float32)

        seq_results = {}
        for photon in photon_levels:
            y_noisy = add_poisson_gaussian_noise(y_clean, peak=photon, rng=rng)

            methods_psnr = {}
            for mname, solver in [("gap_tv", gap_tv_cacti)]:
                try:
                    x_hat = solver(y_noisy, mask, iterations=80)
                    x_hat = np.clip(x_hat, 0, 1).astype(np.float32)
                    p = compute_psnr(gt, x_hat)
                    s = compute_ssim(gt, x_hat)
                    methods_psnr[mname] = {"psnr": round(p, 2), "ssim": round(s, 4)}
                except Exception as e:
                    logger.warning(f"    {mname} failed at photon={photon}: {e}")
                    methods_psnr[mname] = {"psnr": 0.0, "ssim": 0.0}

            logger.info(f"  {seq_name} photon={photon}: " +
                        ", ".join(f"{k}={v['psnr']:.2f}dB" for k, v in methods_psnr.items()))
            seq_results[f"photon_{photon}"] = methods_psnr

        results[seq_name] = seq_results

    return results


# ===================================================================
# CASSI Gate 1: Mask transmittance sweep
# ===================================================================
def run_cassi_gate1(transmittances: List[float], n_scenes: int = 3,
                    device: str = "cpu") -> Dict:
    """CASSI Gate 1: sweep mask transmittance (fewer open pixels = less info)."""
    import scipy.io as sio

    logger.info("=== CASSI Gate 1: Transmittance sweep ===")

    # Load mask
    mask_path = KAIST_DIR / "mask.mat"
    if not mask_path.exists():
        return {"error": "CASSI mask not found"}

    data = sio.loadmat(str(mask_path))
    for key in ["mask", "Mask", "mask_data"]:
        if key in data:
            full_mask = data[key].astype(np.float32)
            break
    else:
        return {"error": "mask key not found"}

    # Use 2D mask (take first slice if 3D)
    if full_mask.ndim == 3:
        full_mask = full_mask[:, :, 0]
    full_mask = np.clip(full_mask, 0, 1)

    scene_names = [f"scene{i:02d}" for i in [1, 3, 5, 7, 9]][:n_scenes]
    rng = np.random.default_rng(42)
    results = {}

    for scene_name in scene_names:
        scene_path = KAIST_DIR / "Truth" / f"{scene_name}.mat"
        if not scene_path.exists():
            logger.warning(f"  Scene {scene_name} not found")
            continue

        data = sio.loadmat(str(scene_path))
        scene = None
        for key in ["img", "Img", "scene", "Scene", "data"]:
            if key in data and data[key].ndim == 3 and data[key].shape[2] == 28:
                scene = data[key].astype(np.float32)
                break
        if scene is None:
            continue

        H, W = scene.shape[:2]
        mask_2d = full_mask[:H, :W]
        original_transmittance = float(np.mean(mask_2d > 0.5))

        scene_results = {}
        for target_t in transmittances:
            # Reduce transmittance by randomly zeroing open pixels
            open_pixels = mask_2d > 0.5
            n_open = int(np.sum(open_pixels))
            n_target = int(target_t * H * W)
            n_target = min(n_target, n_open)

            if n_target < n_open:
                # Randomly zero out some open pixels
                open_indices = np.argwhere(open_pixels)
                n_to_close = n_open - n_target
                close_idx = rng.choice(len(open_indices), size=n_to_close, replace=False)
                reduced_mask = mask_2d.copy()
                for idx in close_idx:
                    r, c = open_indices[idx]
                    reduced_mask[r, c] = 0.0
            else:
                reduced_mask = mask_2d.copy()

            actual_t = float(np.mean(reduced_mask > 0.5))

            # Forward model
            y = cassi_forward(scene, reduced_mask, step=2)

            # Reconstruct
            t0 = time.time()
            x_hat = gap_tv_cassi(y, reduced_mask, n_bands=28, step=2)
            dt = time.time() - t0

            psnr = compute_psnr(scene, x_hat)
            ssim = compute_ssim(scene, x_hat)
            logger.info(f"  {scene_name} T={target_t:.0%} (actual={actual_t:.1%}): "
                        f"PSNR={psnr:.2f} dB ({dt:.1f}s)")
            scene_results[f"t_{int(target_t*100)}"] = {
                "psnr": round(psnr, 2), "ssim": round(ssim, 4),
                "actual_transmittance": round(actual_t, 4),
            }

        results[scene_name] = scene_results

    return results


# ===================================================================
# CASSI Gate 2: Noise sweep
# ===================================================================
def run_cassi_gate2(photon_levels: List[int], n_scenes: int = 3,
                    device: str = "cpu") -> Dict:
    """CASSI Gate 2: noise sweep at standard transmittance."""
    import scipy.io as sio

    logger.info("=== CASSI Gate 2: Noise sweep ===")

    mask_path = KAIST_DIR / "mask.mat"
    if not mask_path.exists():
        return {"error": "CASSI mask not found"}

    data = sio.loadmat(str(mask_path))
    for key in ["mask", "Mask", "mask_data"]:
        if key in data:
            full_mask = data[key].astype(np.float32)
            break
    else:
        return {"error": "mask key not found"}

    if full_mask.ndim == 3:
        full_mask = full_mask[:, :, 0]
    full_mask = np.clip(full_mask, 0, 1)

    scene_names = [f"scene{i:02d}" for i in [1, 3, 5]][:n_scenes]
    rng = np.random.default_rng(42)
    results = {}

    for scene_name in scene_names:
        scene_path = KAIST_DIR / "Truth" / f"{scene_name}.mat"
        if not scene_path.exists():
            continue

        data = sio.loadmat(str(scene_path))
        scene = None
        for key in ["img", "Img", "scene", "Scene", "data"]:
            if key in data and data[key].ndim == 3 and data[key].shape[2] == 28:
                scene = data[key].astype(np.float32)
                break
        if scene is None:
            continue

        H, W = scene.shape[:2]
        mask_2d = full_mask[:H, :W]

        y_clean = cassi_forward(scene, mask_2d, step=2)

        scene_results = {}
        for photon in photon_levels:
            y_noisy = add_poisson_gaussian_noise(y_clean, peak=photon, rng=rng)

            x_hat = gap_tv_cassi(y_noisy, mask_2d, n_bands=28, step=2)
            psnr = compute_psnr(scene, x_hat)
            ssim = compute_ssim(scene, x_hat)

            logger.info(f"  {scene_name} photon={photon}: PSNR={psnr:.2f} dB")
            scene_results[f"photon_{photon}"] = {
                "psnr": round(psnr, 2), "ssim": round(ssim, 4)
            }

        results[scene_name] = scene_results

    return results


# ===================================================================
# SPC Gate 1: CS ratio sweep
# ===================================================================
def run_spc_gate1(ratios: List[float], n_images: int = 3,
                  device: str = "cpu") -> Dict:
    """SPC Gate 1: sweep CS ratio (fewer measurements = less info)."""
    logger.info("=== SPC Gate 1: CS ratio sweep ===")

    if not SET11_DIR.exists():
        return {"error": "Set11 not found"}

    image_names = ["cameraman.tif", "lena256.tif", "boats.tif",
                   "Monarch.tif", "barbara.tif"][:n_images]
    rng_phi = np.random.default_rng(2026)
    results = {}

    for img_name in image_names:
        img_path = SET11_DIR / img_name
        if not img_path.exists():
            continue

        from PIL import Image
        img = Image.open(img_path).convert("L")
        Iorg = np.array(img, dtype=np.float32)
        _, row, col, Ipad, row_new, col_new = imread_CS_py(Iorg)
        Icol = img2col_py(Ipad).transpose() / 255.0  # [B, 1089]

        img_results = {}
        for ratio in ratios:
            m = max(int(ratio * N_PIX), 1)

            # Generate random Gaussian Phi
            Phi = rng_phi.standard_normal((m, N_PIX)).astype(np.float32) / math.sqrt(m)

            # Measurement
            y = Icol @ Phi.T  # [B, m]

            # Solve with FISTA-TV
            t0 = time.time()
            solver = FISTATVSolver(Phi, lam=0.005, max_iter=300)
            x_hat = solver.solve_batch(y)  # [B, n]
            dt = time.time() - t0

            rec = col2im_CS_py(x_hat.transpose(), row, col, row_new, col_new)
            rec_255 = np.clip(rec * 255, 0, 255)

            psnr = compute_psnr(Iorg, rec_255, data_range=255.0)
            ssim = compute_ssim(Iorg / 255.0, rec / 1.0)

            logger.info(f"  {img_name} ratio={ratio:.0%}: PSNR={psnr:.2f} dB ({dt:.1f}s)")
            img_results[f"ratio_{int(ratio*100)}"] = {
                "psnr": round(psnr, 2), "ssim": round(ssim, 4)
            }

        results[img_name] = img_results

    return results


# ===================================================================
# SPC Gate 2: Noise sweep
# ===================================================================
def run_spc_gate2(sigmas: List[float], n_images: int = 3,
                  device: str = "cpu") -> Dict:
    """SPC Gate 2: noise sweep at 25% CS ratio."""
    import scipy.io as sio

    logger.info("=== SPC Gate 2: Noise sweep (25% ratio) ===")

    if not SET11_DIR.exists():
        return {"error": "Set11 not found"}

    # Load standard 25% Phi
    phi_path = Path("/home/spiritai/ISTA-Net-PyTorch-master/sampling_matrix/phi_0_25_1089.mat")
    if phi_path.exists():
        phi_data = sio.loadmat(str(phi_path))
        Phi = phi_data.get("phi", phi_data.get("Phi"))
        if Phi is not None:
            Phi = Phi.astype(np.float32)
        else:
            Phi = np.random.default_rng(42).standard_normal((272, N_PIX)).astype(np.float32) / math.sqrt(272)
    else:
        Phi = np.random.default_rng(42).standard_normal((272, N_PIX)).astype(np.float32) / math.sqrt(272)

    solver = FISTATVSolver(Phi, lam=0.005, max_iter=300)

    image_names = ["cameraman.tif", "lena256.tif", "boats.tif"][:n_images]
    rng = np.random.default_rng(42)
    results = {}

    for img_name in image_names:
        img_path = SET11_DIR / img_name
        if not img_path.exists():
            continue

        from PIL import Image
        img = Image.open(img_path).convert("L")
        Iorg = np.array(img, dtype=np.float32)
        _, row, col, Ipad, row_new, col_new = imread_CS_py(Iorg)
        Icol = img2col_py(Ipad).transpose() / 255.0  # [B, 1089]

        y_clean = Icol @ Phi.T  # [B, m]

        img_results = {}
        for sigma in sigmas:
            if sigma == 0:
                y = y_clean.copy()
            else:
                noise = rng.standard_normal(y_clean.shape).astype(np.float32)
                y = y_clean + sigma * noise

            x_hat = solver.solve_batch(y)
            rec = col2im_CS_py(x_hat.transpose(), row, col, row_new, col_new)
            rec_255 = np.clip(rec * 255, 0, 255)

            psnr = compute_psnr(Iorg, rec_255, data_range=255.0)
            ssim = compute_ssim(Iorg / 255.0, rec)

            logger.info(f"  {img_name} sigma={sigma}: PSNR={psnr:.2f} dB")
            img_results[f"sigma_{sigma}"] = {
                "psnr": round(psnr, 2), "ssim": round(ssim, 4)
            }

        results[img_name] = img_results

    return results


# ===================================================================
# MRI Gate 1: Sampling rate sweep
# ===================================================================
def run_mri_gate1(sampling_rates: List[float],
                  device: str = "cpu") -> Dict:
    """MRI Gate 1: sweep k-space sampling rate."""
    from pwm_core.physics.mri.mri_operator import MRIOperator
    from pwm_core.recon.mri_solvers import zero_filled_reconstruction, cs_mri_wavelet

    logger.info("=== MRI Gate 1: Sampling rate sweep ===")

    # Shepp-Logan phantom
    try:
        from skimage.data import shepp_logan_phantom
        phantom = shepp_logan_phantom()
    except ImportError:
        phantom = _make_simple_phantom(128)

    from skimage.transform import resize
    phantom = resize(phantom, (128, 128), anti_aliasing=True).astype(np.float32)
    phantom = phantom / max(phantom.max(), 1e-10)

    results = {}
    for rate in sampling_rates:
        op = MRIOperator(x_shape=(128, 128), sampling_rate=rate, seed=42)
        kspace = op.forward(phantom)

        methods = {}
        # Zero-filled
        x_zf = zero_filled_reconstruction(kspace, mask=op.mask)
        x_zf = np.abs(x_zf).astype(np.float32)
        x_zf = x_zf / max(x_zf.max(), 1e-10)
        methods["zero_filled"] = {
            "psnr": round(compute_psnr(phantom, x_zf), 2),
            "ssim": round(compute_ssim(phantom, x_zf), 4),
        }

        # CS-MRI wavelet
        x_cs = cs_mri_wavelet(kspace, mask=op.mask, lam=0.01, iterations=50)
        x_cs = np.abs(x_cs).astype(np.float32)
        x_cs = x_cs / max(x_cs.max(), 1e-10)
        methods["cs_wavelet"] = {
            "psnr": round(compute_psnr(phantom, x_cs), 2),
            "ssim": round(compute_ssim(phantom, x_cs), 4),
        }

        logger.info(f"  rate={rate:.0%}: ZF={methods['zero_filled']['psnr']:.2f}dB, "
                    f"CS={methods['cs_wavelet']['psnr']:.2f}dB")
        results[f"rate_{int(rate*100)}"] = methods

    return results


# ===================================================================
# MRI Gate 2: Noise sweep
# ===================================================================
def run_mri_gate2(noise_levels: List[float],
                  device: str = "cpu") -> Dict:
    """MRI Gate 2: Gaussian noise sweep at 25% sampling."""
    from pwm_core.physics.mri.mri_operator import MRIOperator
    from pwm_core.recon.mri_solvers import zero_filled_reconstruction, cs_mri_wavelet

    logger.info("=== MRI Gate 2: Noise sweep (25% sampling) ===")

    try:
        from skimage.data import shepp_logan_phantom
        phantom = shepp_logan_phantom()
    except ImportError:
        phantom = _make_simple_phantom(128)

    from skimage.transform import resize
    phantom = resize(phantom, (128, 128), anti_aliasing=True).astype(np.float32)
    phantom = phantom / max(phantom.max(), 1e-10)

    op = MRIOperator(x_shape=(128, 128), sampling_rate=0.25, seed=42)
    kspace_clean = op.forward(phantom)

    rng = np.random.default_rng(42)
    results = {}

    for sigma in noise_levels:
        if sigma == 0:
            kspace = kspace_clean.copy()
        else:
            # Gaussian noise on complex k-space
            k_range = max(float(np.max(np.abs(kspace_clean))), 1e-10)
            noise_re = rng.normal(0, sigma * k_range, kspace_clean.shape)
            noise_im = rng.normal(0, sigma * k_range, kspace_clean.shape)
            kspace = kspace_clean + (noise_re + 1j * noise_im).astype(kspace_clean.dtype)

        methods = {}
        x_zf = zero_filled_reconstruction(kspace, mask=op.mask)
        x_zf = np.abs(x_zf).astype(np.float32)
        x_zf = x_zf / max(x_zf.max(), 1e-10)
        methods["zero_filled"] = {
            "psnr": round(compute_psnr(phantom, x_zf), 2),
            "ssim": round(compute_ssim(phantom, x_zf), 4),
        }

        x_cs = cs_mri_wavelet(kspace, mask=op.mask, lam=0.01, iterations=50)
        x_cs = np.abs(x_cs).astype(np.float32)
        x_cs = x_cs / max(x_cs.max(), 1e-10)
        methods["cs_wavelet"] = {
            "psnr": round(compute_psnr(phantom, x_cs), 2),
            "ssim": round(compute_ssim(phantom, x_cs), 4),
        }

        logger.info(f"  sigma={sigma}: ZF={methods['zero_filled']['psnr']:.2f}dB, "
                    f"CS={methods['cs_wavelet']['psnr']:.2f}dB")
        results[f"sigma_{sigma}"] = methods

    return results


# ===================================================================
# CT Gate 1: Angle sweep
# ===================================================================
def run_ct_gate1(n_angles_list: List[int],
                 device: str = "cpu") -> Dict:
    """CT Gate 1: sweep number of projection angles."""
    from pwm_core.physics.tomography.ct_operator import CTOperator
    from pwm_core.recon.ct_solvers import fbp_2d, sart_2d

    logger.info("=== CT Gate 1: Angle sweep ===")

    try:
        from skimage.data import shepp_logan_phantom
        phantom = shepp_logan_phantom()
    except ImportError:
        phantom = _make_simple_phantom(128)

    from skimage.transform import resize
    phantom = resize(phantom, (128, 128), anti_aliasing=True).astype(np.float32)
    phantom = phantom / max(phantom.max(), 1e-10)

    results = {}
    for n_angles in n_angles_list:
        op = CTOperator(x_shape=(128, 128), n_angles=n_angles)
        sinogram = op.forward(phantom)
        angles_rad = np.radians(op.angles)

        methods = {}
        # FBP
        x_fbp = fbp_2d(sinogram, angles_rad, filter_type='ramlak', output_size=128)
        x_fbp = np.clip(x_fbp, 0, None).astype(np.float32)
        x_fbp = x_fbp / max(x_fbp.max(), 1e-10)
        methods["fbp"] = {
            "psnr": round(compute_psnr(phantom, x_fbp), 2),
            "ssim": round(compute_ssim(phantom, x_fbp), 4),
        }

        # SART
        x_sart = sart_2d(sinogram, angles_rad, output_size=128,
                         iterations=20, relaxation=0.5)
        x_sart = np.clip(x_sart, 0, None).astype(np.float32)
        x_sart = x_sart / max(x_sart.max(), 1e-10)
        methods["sart"] = {
            "psnr": round(compute_psnr(phantom, x_sart), 2),
            "ssim": round(compute_ssim(phantom, x_sart), 4),
        }

        logger.info(f"  angles={n_angles}: FBP={methods['fbp']['psnr']:.2f}dB, "
                    f"SART={methods['sart']['psnr']:.2f}dB")
        results[f"angles_{n_angles}"] = methods

    return results


# ===================================================================
# CT Gate 2: Noise sweep
# ===================================================================
def run_ct_gate2(photon_counts: List[int],
                 device: str = "cpu") -> Dict:
    """CT Gate 2: Poisson noise sweep at 180 angles."""
    from pwm_core.physics.tomography.ct_operator import CTOperator
    from pwm_core.recon.ct_solvers import fbp_2d, sart_2d

    logger.info("=== CT Gate 2: Noise sweep (180 angles) ===")

    try:
        from skimage.data import shepp_logan_phantom
        phantom = shepp_logan_phantom()
    except ImportError:
        phantom = _make_simple_phantom(128)

    from skimage.transform import resize
    phantom = resize(phantom, (128, 128), anti_aliasing=True).astype(np.float32)
    phantom = phantom / max(phantom.max(), 1e-10)

    op = CTOperator(x_shape=(128, 128), n_angles=180)
    sinogram_clean = op.forward(phantom)
    angles_rad = np.radians(op.angles)

    rng = np.random.default_rng(42)
    results = {}

    for photon in photon_counts:
        # Poisson noise on sinogram
        sino_max = max(float(np.max(sinogram_clean)), 1e-10)
        sino_scaled = np.clip(sinogram_clean / sino_max * photon, 0, None)
        sino_noisy = rng.poisson(sino_scaled).astype(np.float64)
        sino_noisy = (sino_noisy / photon * sino_max).astype(np.float32)

        methods = {}
        x_fbp = fbp_2d(sino_noisy, angles_rad, filter_type='ramlak', output_size=128)
        x_fbp = np.clip(x_fbp, 0, None).astype(np.float32)
        x_fbp = x_fbp / max(x_fbp.max(), 1e-10)
        methods["fbp"] = {
            "psnr": round(compute_psnr(phantom, x_fbp), 2),
            "ssim": round(compute_ssim(phantom, x_fbp), 4),
        }

        x_sart = sart_2d(sino_noisy, angles_rad, output_size=128,
                         iterations=20, relaxation=0.5)
        x_sart = np.clip(x_sart, 0, None).astype(np.float32)
        x_sart = x_sart / max(x_sart.max(), 1e-10)
        methods["sart"] = {
            "psnr": round(compute_psnr(phantom, x_sart), 2),
            "ssim": round(compute_ssim(phantom, x_sart), 4),
        }

        logger.info(f"  photon={photon}: FBP={methods['fbp']['psnr']:.2f}dB, "
                    f"SART={methods['sart']['psnr']:.2f}dB")
        results[f"photon_{photon}"] = methods

    return results


# ===================================================================
# Lensless Gate 1: PSF sigma sweep (larger = more blur = less info)
# ===================================================================
def run_lensless_gate1(blur_sigmas: List[float],
                       device: str = "cpu") -> Dict:
    """Lensless Gate 1: sweep Gaussian blur sigma (larger = destroys more info).

    Uses a Gaussian PSF that genuinely attenuates high-frequency content,
    unlike the random-phase diffuser PSF which preserves frequency information.
    Fourier magnitudes of a Gaussian PSF decay as exp(-sigma^2 * f^2 / 2),
    so larger sigma = more high-frequency information is irrecoverably lost.
    """
    from pwm_core.recon.lensless_solver import tikhonov_lensless, admm_tv_lensless

    logger.info("=== Lensless Gate 1: Gaussian blur sigma sweep ===")

    try:
        from skimage.data import camera
        img = camera().astype(np.float32) / 255.0
    except ImportError:
        img = _make_simple_phantom(128)

    from skimage.transform import resize
    from scipy.ndimage import gaussian_filter
    img = resize(img, (128, 128), anti_aliasing=True).astype(np.float32)

    results = {}
    for sigma in blur_sigmas:
        # Create Gaussian PSF (genuinely loses high-frequency info)
        H, W = 128, 128
        psf = np.zeros((H, W), dtype=np.float64)
        psf[H // 2, W // 2] = 1.0
        psf = gaussian_filter(psf, sigma=sigma)
        psf = (psf / psf.sum()).astype(np.float32)

        # Forward: convolve with Gaussian PSF
        from scipy.fft import fft2, ifft2
        meas = np.real(ifft2(fft2(img) * fft2(psf))).astype(np.float32)

        methods = {}
        # Tikhonov
        x_tik = tikhonov_lensless(meas, psf)
        x_tik = np.clip(x_tik, 0, 1).astype(np.float32)
        methods["tikhonov"] = {
            "psnr": round(compute_psnr(img, x_tik), 2),
            "ssim": round(compute_ssim(img, x_tik), 4),
        }

        # ADMM-TV
        x_admm = admm_tv_lensless(meas, psf, iters=100)
        x_admm = np.clip(x_admm, 0, 1).astype(np.float32)
        methods["admm_tv"] = {
            "psnr": round(compute_psnr(img, x_admm), 2),
            "ssim": round(compute_ssim(img, x_admm), 4),
        }

        logger.info(f"  blur_sigma={sigma}: Tik={methods['tikhonov']['psnr']:.2f}dB, "
                    f"ADMM-TV={methods['admm_tv']['psnr']:.2f}dB")
        results[f"blur_sigma_{sigma}"] = methods

    return results


# ===================================================================
# Lensless Gate 2: Noise sweep
# ===================================================================
def run_lensless_gate2(photon_levels: List[int],
                       device: str = "cpu") -> Dict:
    """Lensless Gate 2: photon-level sweep at psf_sigma=10."""
    from pwm_core.physics.lensless.lensless_operator import LenslessOperator
    from pwm_core.recon.lensless_solver import tikhonov_lensless, admm_tv_lensless

    logger.info("=== Lensless Gate 2: Noise sweep (sigma=10) ===")

    try:
        from skimage.data import camera
        img = camera().astype(np.float32) / 255.0
    except ImportError:
        img = _make_simple_phantom(128)

    from skimage.transform import resize
    img = resize(img, (128, 128), anti_aliasing=True).astype(np.float32)

    op = LenslessOperator(x_shape=(128, 128), psf_sigma=10.0, seed=42)
    meas_clean = op.forward(img)

    rng = np.random.default_rng(42)
    results = {}

    for photon in photon_levels:
        meas_noisy = add_poisson_gaussian_noise(meas_clean, peak=photon, rng=rng)

        methods = {}
        x_tik = tikhonov_lensless(meas_noisy, op.psf)
        x_tik = np.clip(x_tik, 0, 1).astype(np.float32)
        methods["tikhonov"] = {
            "psnr": round(compute_psnr(img, x_tik), 2),
            "ssim": round(compute_ssim(img, x_tik), 4),
        }

        x_admm = admm_tv_lensless(meas_noisy, op.psf, iters=100)
        x_admm = np.clip(x_admm, 0, 1).astype(np.float32)
        methods["admm_tv"] = {
            "psnr": round(compute_psnr(img, x_admm), 2),
            "ssim": round(compute_ssim(img, x_admm), 4),
        }

        logger.info(f"  photon={photon}: Tik={methods['tikhonov']['psnr']:.2f}dB, "
                    f"ADMM-TV={methods['admm_tv']['psnr']:.2f}dB")
        results[f"photon_{photon}"] = methods

    return results


# ===================================================================
# Ptychography Gate 1: Position sweep
# ===================================================================
def run_ptycho_gate1(n_positions_list: List[int],
                     device: str = "cpu") -> Dict:
    """Ptychography Gate 1: sweep number of scan positions."""
    from pwm_core.physics.microscopy.ptychography_operator import PtychographyOperator
    from pwm_core.recon.ptychography_solver import epie

    logger.info("=== Ptychography Gate 1: Position sweep ===")

    try:
        from skimage.data import camera
        img = camera().astype(np.float32) / 255.0
    except ImportError:
        img = _make_simple_phantom(64)

    from skimage.transform import resize
    img = resize(img, (64, 64), anti_aliasing=True).astype(np.float32)

    results = {}
    for n_pos in n_positions_list:
        op = PtychographyOperator(
            x_shape=(64, 64), n_positions=n_pos, probe_size=32, seed=42)
        patterns = op.forward(img)

        # Reconstruct with ePIE
        positions_arr = np.array(op.positions)
        obj, probe = epie(
            patterns, positions_arr,
            object_shape=(64, 64),
            probe_init=op.probe.astype(np.complex64),
            iterations=200, alpha=1.0, beta=0.5)

        x_hat = np.abs(obj).astype(np.float32)
        x_hat = x_hat / max(x_hat.max(), 1e-10)

        psnr = compute_psnr(img, x_hat)
        ssim = compute_ssim(img, x_hat)

        logger.info(f"  n_positions={n_pos}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        results[f"pos_{n_pos}"] = {
            "psnr": round(psnr, 2), "ssim": round(ssim, 4),
        }

    return results


# ===================================================================
# Ptychography Gate 2: Noise sweep
# ===================================================================
def run_ptycho_gate2(photon_counts: List[int],
                     device: str = "cpu") -> Dict:
    """Ptychography Gate 2: Poisson noise sweep at 16 positions."""
    from pwm_core.physics.microscopy.ptychography_operator import PtychographyOperator
    from pwm_core.recon.ptychography_solver import epie

    logger.info("=== Ptychography Gate 2: Noise sweep (16 positions) ===")

    try:
        from skimage.data import camera
        img = camera().astype(np.float32) / 255.0
    except ImportError:
        img = _make_simple_phantom(64)

    from skimage.transform import resize
    img = resize(img, (64, 64), anti_aliasing=True).astype(np.float32)

    op = PtychographyOperator(
        x_shape=(64, 64), n_positions=16, probe_size=32, seed=42)
    patterns_clean = op.forward(img)

    rng = np.random.default_rng(42)
    results = {}

    for photon in photon_counts:
        # Poisson noise on diffraction patterns
        pat_max = max(float(np.max(patterns_clean)), 1e-10)
        pat_scaled = np.clip(patterns_clean / pat_max * photon, 0, None)
        pat_noisy = rng.poisson(pat_scaled).astype(np.float64)
        pat_noisy = (pat_noisy / photon * pat_max).astype(np.float32)

        positions_arr = np.array(op.positions)
        obj, probe = epie(
            pat_noisy, positions_arr,
            object_shape=(64, 64),
            probe_init=op.probe.astype(np.complex64),
            iterations=200, alpha=1.0, beta=0.5)

        x_hat = np.abs(obj).astype(np.float32)
        x_hat = x_hat / max(x_hat.max(), 1e-10)

        psnr = compute_psnr(img, x_hat)
        ssim = compute_ssim(img, x_hat)

        logger.info(f"  photon={photon}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        results[f"photon_{photon}"] = {
            "psnr": round(psnr, 2), "ssim": round(ssim, 4),
        }

    return results


# ===================================================================
# Utility
# ===================================================================
def _make_simple_phantom(size: int) -> np.ndarray:
    """Fallback phantom if skimage not available."""
    x = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    for r in range(size):
        for c in range(size):
            if (r - cy) ** 2 + (c - cx) ** 2 < (size // 3) ** 2:
                x[r, c] = 1.0
            elif (r - cy) ** 2 + (c - cx) ** 2 < (size // 4) ** 2:
                x[r, c] = 0.5
    return x


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Gate 1 & Gate 2 validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer images/scenes per modality")
    parser.add_argument("--modalities", nargs="+",
                        default=["cacti", "cassi", "spc", "mri", "ct",
                                 "lensless", "ptychography"],
                        help="Modalities to run")
    args = parser.parse_args()

    quick = args.quick
    device = args.device
    n_img = 1 if quick else 3

    all_results = {"metadata": {
        "quick_mode": quick, "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    # ---- Gate 1 experiments ----
    gate1 = {}

    if "cacti" in args.modalities:
        cr_values = [8, 16, 32, 64] if not quick else [8, 32]
        gate1["cacti"] = run_cacti_gate1(cr_values, n_sequences=n_img, device=device)

    if "cassi" in args.modalities:
        transmittances = [0.50, 0.25, 0.10, 0.05, 0.02] if not quick else [0.50, 0.05]
        gate1["cassi"] = run_cassi_gate1(transmittances, n_scenes=n_img, device=device)

    if "spc" in args.modalities:
        ratios = [0.25, 0.10, 0.05, 0.02, 0.01] if not quick else [0.25, 0.05]
        gate1["spc"] = run_spc_gate1(ratios, n_images=n_img, device=device)

    if "mri" in args.modalities:
        rates = [0.25, 0.10, 0.05, 0.02] if not quick else [0.25, 0.05]
        gate1["mri"] = run_mri_gate1(rates, device=device)

    if "ct" in args.modalities:
        angles = [180, 90, 30, 10, 5] if not quick else [180, 10]
        gate1["ct"] = run_ct_gate1(angles, device=device)

    if "lensless" in args.modalities:
        blur_sigmas = [1, 3, 5, 10, 20] if not quick else [1, 10]
        gate1["lensless"] = run_lensless_gate1(blur_sigmas, device=device)

    if "ptychography" in args.modalities:
        positions = [16, 9, 4, 1] if not quick else [16, 4]
        gate1["ptychography"] = run_ptycho_gate1(positions, device=device)

    all_results["gate1"] = gate1

    # ---- Gate 2 experiments ----
    gate2 = {}

    if "cacti" in args.modalities:
        photons = [10000, 1000, 100, 10] if not quick else [10000, 100]
        gate2["cacti"] = run_cacti_gate2(photons, n_sequences=n_img, device=device)

    if "cassi" in args.modalities:
        photons = [10000, 1000, 100, 10] if not quick else [10000, 100]
        gate2["cassi"] = run_cassi_gate2(photons, n_scenes=n_img, device=device)

    if "spc" in args.modalities:
        sigmas = [0, 0.01, 0.05, 0.1, 0.3] if not quick else [0, 0.1]
        gate2["spc"] = run_spc_gate2(sigmas, n_images=n_img, device=device)

    if "mri" in args.modalities:
        noise = [0, 0.01, 0.05, 0.1, 0.3] if not quick else [0, 0.1]
        gate2["mri"] = run_mri_gate2(noise, device=device)

    if "ct" in args.modalities:
        photons = [100000, 10000, 1000, 100] if not quick else [100000, 100]
        gate2["ct"] = run_ct_gate2(photons, device=device)

    if "lensless" in args.modalities:
        photons = [10000, 1000, 100, 10] if not quick else [10000, 100]
        gate2["lensless"] = run_lensless_gate2(photons, device=device)

    if "ptychography" in args.modalities:
        photons = [10000, 1000, 100, 10] if not quick else [10000, 100]
        gate2["ptychography"] = run_ptycho_gate2(photons, device=device)

    all_results["gate2"] = gate2

    # Save results
    out_path = RESULTS_DIR / "gate1_gate2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("GATE 1 & GATE 2 VALIDATION SUMMARY")
    print("=" * 70)

    for gate_name, gate_data in [("Gate 1 (Information Deficiency)", gate1),
                                  ("Gate 2 (Carrier Budget)", gate2)]:
        print(f"\n--- {gate_name} ---")
        for modality, mod_data in gate_data.items():
            if isinstance(mod_data, dict) and "error" not in mod_data:
                print(f"\n  {modality.upper()}:")
                # Flatten and print
                for key, val in mod_data.items():
                    if isinstance(val, dict):
                        # Could be per-scene or per-setting
                        for subkey, subval in val.items():
                            if isinstance(subval, dict) and "psnr" in subval:
                                print(f"    {key}/{subkey}: {subval['psnr']:.2f} dB")
                            elif isinstance(subval, dict):
                                for method, mval in subval.items():
                                    if isinstance(mval, dict) and "psnr" in mval:
                                        print(f"    {key}/{subkey}/{method}: {mval['psnr']:.2f} dB")


if __name__ == "__main__":
    main()
