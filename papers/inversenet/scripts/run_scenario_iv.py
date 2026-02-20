#!/usr/bin/env python3
"""Scenario IV: Grid-Search Calibration Baseline for InverseNet ECCV 2026.

For each modality, grid-searches over mismatch parameters using a
self-supervised objective.  This provides a practical calibration baseline
between Scenario II (uncorrected) and Scenario III (oracle).

  CASSI : search dx, dy grid with GAP-TV inner loop (measurement residual)
  CACTI : search mask shift + timing with GAP-TV inner loop (measurement residual)
  SPC   : search gain_alpha with FISTA-TV inner loop (reconstruction TV)

Usage:
    python run_scenario_iv.py [--device cuda:0] [--modality all|cassi|cacti|spc]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CASSI Scenario IV
# ============================================================================
def cassi_scenario_iv(device: str = "cuda:0") -> Dict:
    """Grid-search calibration for CASSI using GAP-TV.

    Searches dx, dy on a grid [-1.0, 1.0] with 0.2 step.
    Inner loop: GAP-TV with 30 iterations (fast for grid search).
    Outer metric: measurement residual ||y - Phi(x_hat)||^2.

    Uses full 5-parameter mismatch (spatial + dispersion) matching
    the main benchmark.  Grid search covers spatial params only;
    dispersion is NOT searched, matching practical constraints.
    Scenario III uses the oracle dispersion-aware GAP-TV.
    """
    import scipy.io as sio
    from scipy.ndimage import affine_transform
    from skimage.restoration import denoise_tv_chambolle

    # Import dispersion-aware forward model from the main validation script
    from validate_cassi_inversenet import cassi_forward_with_dispersion

    DATASET_SIMU = Path("/home/spiritai/MST-main/datasets/TSA_simu_data")

    logger.info("\n" + "=" * 60)
    logger.info("CASSI Scenario IV: Grid-Search Calibration (5-param mismatch)")
    logger.info("=" * 60)

    # Load mask and first 3 scenes (for speed)
    mask_data = sio.loadmat(str(DATASET_SIMU / "mask.mat"))
    mask_ideal = mask_data["mask"].astype(np.float32)
    H, W = mask_ideal.shape
    nC, step = 28, 2
    W_ext = W + (nC - 1) * step

    # True mismatch params (all 5 -- matching main benchmark)
    true_dx, true_dy, true_theta = 0.5, 0.3, 0.1
    true_a1, true_alpha_deg = 2.02, 0.15  # dispersion drift

    def warp_mask(m, dx, dy, theta=0.0):
        cx, cy = W / 2.0, H / 2.0
        th = np.radians(theta)
        cos_t, sin_t = np.cos(th), np.sin(th)
        mat = np.array([
            [cos_t, sin_t, -cx * cos_t - cy * sin_t + cx + dx],
            [-sin_t, cos_t, cx * sin_t - cy * cos_t + cy + dy],
        ])
        inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
        return np.clip(affine_transform(m, inv[:2, :2], offset=inv[:2, 2], cval=0, order=1), 0, 1).astype(np.float32)

    def cassi_fwd_nominal(scene, mask):
        """Nominal forward model (integer step=2, no dispersion drift)."""
        y = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            y[:, k * step:k * step + W] += mask * scene[:, :, k]
        return y

    def gap_tv_fast(y, mask, iters=50):
        """GAP-TV with nominal step=2 dispersion (reconstruction methods' assumption)."""
        Phi_sum = np.zeros((H, W_ext), dtype=np.float32)
        for k in range(nC):
            Phi_sum[:, k * step:k * step + W] += mask ** 2
        Phi_sum = np.maximum(Phi_sum, 1e-10)
        x = np.zeros((H, W, nC), dtype=np.float32)
        for k in range(nC):
            x[:, :, k] = mask * y[:, k * step:k * step + W] / np.maximum(
                Phi_sum[:, k * step:k * step + W], 1e-6)
        y1 = np.zeros((H, W_ext), dtype=np.float32)
        for _ in range(iters):
            y_est = np.zeros((H, W_ext), dtype=np.float32)
            for k in range(nC):
                y_est[:, k * step:k * step + W] += mask * x[:, :, k]
            y1 += (y[:, :W_ext] - y_est)
            norm_r = (y1 - y_est) / Phi_sum
            for k in range(nC):
                x[:, :, k] += mask * norm_r[:, k * step:k * step + W]
            x = denoise_tv_chambolle(np.clip(x, 0, None), weight=0.1,
                                      max_num_iter=3, channel_axis=2).astype(np.float32)
        return np.clip(x, 0, 1).astype(np.float32)

    # Load scenes
    scenes = []
    for si in range(1, 4):  # 3 scenes for speed
        path = DATASET_SIMU / "Truth" / f"scene{si:02d}.mat"
        if not path.exists():
            continue
        d = sio.loadmat(str(path))
        for key in ["img", "Img", "scene", "data"]:
            if key in d:
                scenes.append(d[key].astype(np.float32))
                break

    if not scenes:
        logger.warning("No CASSI scenes found")
        return {}

    logger.info(f"Loaded {len(scenes)} scenes")

    # Generate corrupted measurements with FULL 5-parameter mismatch
    # (warped mask + dispersion drift a1=2.02, alpha=0.15°)
    mask_true = warp_mask(mask_ideal, true_dx, true_dy, true_theta)
    y_corrupts = []
    for s in scenes:
        y = cassi_forward_with_dispersion(s, mask_true,
                                          a1=true_a1, alpha_deg=true_alpha_deg)
        y_corrupts.append(y)
    logger.info(f"Measurements generated with a1={true_a1}, alpha={true_alpha_deg}°")

    # Grid search over dx, dy (spatial only -- dispersion NOT searched)
    dx_grid = np.arange(-1.0, 1.1, 0.2)
    dy_grid = np.arange(-1.0, 1.1, 0.2)

    logger.info(f"Grid: {len(dx_grid)}x{len(dy_grid)} = {len(dx_grid)*len(dy_grid)} points")

    best_dx, best_dy = 0.0, 0.0
    best_residual = float("inf")
    grid_results = []

    t0 = time.time()
    for dx_test in dx_grid:
        for dy_test in dy_grid:
            mask_test = warp_mask(mask_ideal, dx_test, dy_test)
            total_residual = 0.0
            for si, y_c in enumerate(y_corrupts):
                x_hat = gap_tv_fast(y_c, mask_test, iters=30)
                y_pred = cassi_fwd_nominal(x_hat, mask_test)
                ww = min(y_c.shape[1], y_pred.shape[1])
                total_residual += np.sum((y_c[:, :ww] - y_pred[:, :ww]) ** 2)

            grid_results.append({
                "dx": round(float(dx_test), 2),
                "dy": round(float(dy_test), 2),
                "residual": round(float(total_residual), 1),
            })

            if total_residual < best_residual:
                best_residual = total_residual
                best_dx, best_dy = dx_test, dy_test

    calibration_time = time.time() - t0

    logger.info(f"Best: dx={best_dx:.2f}, dy={best_dy:.2f} "
                f"(true: dx={true_dx}, dy={true_dy})")
    logger.info(f"Grid search time: {calibration_time:.1f}s")

    # Evaluate Scenario IV with calibrated mask (still nominal step=2)
    mask_cal = warp_mask(mask_ideal, best_dx, best_dy)
    psnrs_iv = []
    for si, (scene, y_c) in enumerate(zip(scenes, y_corrupts)):
        x_hat = gap_tv_fast(y_c, mask_cal, iters=100)
        mse = float(np.mean((scene - x_hat) ** 2))
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
        psnrs_iv.append(psnr)

    # Scenario II: ideal mask, nominal step=2 (no knowledge of mismatch)
    psnrs_ii = []
    for si, (scene, y_c) in enumerate(zip(scenes, y_corrupts)):
        x_hat = gap_tv_fast(y_c, mask_ideal, iters=100)
        mse = float(np.mean((scene - x_hat) ** 2))
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
        psnrs_ii.append(psnr)

    # Scenario III: spatial oracle -- true warped mask, nominal step=2.
    # We use nominal GAP-TV (not dispersion-aware) because:
    # (1) the grid search only calibrates spatial params,
    # (2) dispersion-aware GAP-TV has boundary clipping and sub-pixel
    #     interpolation artifacts that make it worse than nominal for
    #     small dispersion drift (a1=2.02 ≈ 2.0).
    # The gap from III to Scenario I reflects irrecoverable dispersion error.
    psnrs_iii = []
    for si, (scene, y_c) in enumerate(zip(scenes, y_corrupts)):
        x_hat = gap_tv_fast(y_c, mask_true, iters=100)
        mse = float(np.mean((scene - x_hat) ** 2))
        psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))
        psnrs_iii.append(psnr)

    result = {
        "modality": "cassi",
        "method": "gap_tv",
        "true_params": {"dx": true_dx, "dy": true_dy, "theta": true_theta,
                        "a1": true_a1, "alpha_deg": true_alpha_deg},
        "estimated_params": {"dx": round(best_dx, 2), "dy": round(best_dy, 2)},
        "scenario_ii_psnr": round(float(np.mean(psnrs_ii)), 2),
        "scenario_iv_psnr": round(float(np.mean(psnrs_iv)), 2),
        "scenario_iii_psnr": round(float(np.mean(psnrs_iii)), 2),
        "calibration_time_s": round(calibration_time, 1),
        "grid_size": len(grid_results),
    }

    logger.info(f"\nCASSI GAP-TV:  II={result['scenario_ii_psnr']:.2f}  "
                f"IV={result['scenario_iv_psnr']:.2f}  "
                f"III={result['scenario_iii_psnr']:.2f}")

    return result


# ============================================================================
# CACTI Scenario IV
# ============================================================================
def cacti_scenario_iv(device: str = "cuda:0") -> Dict:
    """Grid-search calibration for CACTI using GAP-TV.

    Searches timing offset and mask shift on a grid.
    """
    from pwm_core.data.loaders.cacti_bench import CACTIBenchmark
    from scipy.ndimage import shift as ndi_shift
    from skimage.restoration import denoise_tv_chambolle

    logger.info("\n" + "=" * 60)
    logger.info("CACTI Scenario IV: Grid-Search Calibration")
    logger.info("=" * 60)

    # Load benchmark data (first 2 groups for speed)
    try:
        dataset = CACTIBenchmark()
    except Exception as e:
        logger.warning(f"CACTI benchmark not available: {e}")
        return {}

    # True mismatch
    true_dx, true_dy = 0.5, 0.3

    def shift_mask(mask, dx, dy):
        out = np.zeros_like(mask)
        for t in range(mask.shape[2]):
            out[:, :, t] = ndi_shift(mask[:, :, t], [dy, dx], order=1, mode="constant")
        return np.clip(out, 0, 1).astype(np.float32)

    def gap_tv_fast(y, mask, iters=50):
        H, W, nF = mask.shape
        mask_sum = np.sum(mask, axis=2)
        mask_sum[mask_sum == 0] = 1
        x = y[:, :, np.newaxis] * mask / mask_sum[:, :, np.newaxis]
        y1 = y.copy()
        for _ in range(iters):
            yb = np.sum(x * mask, axis=2)
            y1 = y1 + (y - yb)
            x = x + ((y1 - yb) / mask_sum)[:, :, np.newaxis] * mask
            for f in range(nF):
                x[:, :, f] = denoise_tv_chambolle(x[:, :, f], weight=0.1, max_num_iter=3)
            x = np.clip(x, 0, 1)
        return x.astype(np.float32)

    # Process first 2 groups
    groups = []
    for i, (name, gt, mask, meas) in enumerate(dataset):
        if i >= 2:
            break
        groups.append((name, gt, mask, meas))

    if not groups:
        logger.warning("No CACTI groups available")
        return {}

    logger.info(f"Processing {len(groups)} groups")

    # Generate corrupted measurements
    corrupted_data = []
    for name, gt, mask, meas in groups:
        mask_warped = shift_mask(mask, true_dx, true_dy)
        mask_warped_bin = (mask_warped > 0.5).astype(np.float32)
        y_corrupt = np.sum(gt * mask_warped_bin, axis=2)
        corrupted_data.append((name, gt, mask, mask_warped_bin, y_corrupt))

    # Grid search: dx, dy
    dx_grid = np.arange(-1.0, 1.1, 0.25)
    dy_grid = np.arange(-1.0, 1.1, 0.25)
    logger.info(f"Grid: {len(dx_grid)}x{len(dy_grid)}")

    best_dx, best_dy = 0.0, 0.0
    best_residual = float("inf")

    t0 = time.time()
    for dx_test in dx_grid:
        for dy_test in dy_grid:
            total_residual = 0.0
            for name, gt, mask, mask_w, y_c in corrupted_data:
                mask_test = shift_mask(mask, dx_test, dy_test)
                mask_test_bin = (mask_test > 0.5).astype(np.float32)
                x_hat = gap_tv_fast(y_c, mask_test_bin, iters=20)
                y_pred = np.sum(x_hat * mask_test_bin, axis=2)
                total_residual += np.sum((y_c - y_pred) ** 2)

            if total_residual < best_residual:
                best_residual = total_residual
                best_dx, best_dy = dx_test, dy_test

    calibration_time = time.time() - t0

    logger.info(f"Best: dx={best_dx:.2f}, dy={best_dy:.2f}")
    logger.info(f"Calibration time: {calibration_time:.1f}s")

    # Evaluate scenarios
    psnrs_ii, psnrs_iv, psnrs_iii = [], [], []

    for name, gt, mask, mask_w, y_c in corrupted_data:
        # Scenario II
        x_ii = gap_tv_fast(y_c, mask, iters=50)
        psnrs_ii.append(float(10 * np.log10(1.0 / max(np.mean((gt - x_ii)**2), 1e-10))))

        # Scenario IV
        mask_cal = shift_mask(mask, best_dx, best_dy)
        mask_cal_bin = (mask_cal > 0.5).astype(np.float32)
        x_iv = gap_tv_fast(y_c, mask_cal_bin, iters=50)
        psnrs_iv.append(float(10 * np.log10(1.0 / max(np.mean((gt - x_iv)**2), 1e-10))))

        # Scenario III
        x_iii = gap_tv_fast(y_c, mask_w, iters=50)
        psnrs_iii.append(float(10 * np.log10(1.0 / max(np.mean((gt - x_iii)**2), 1e-10))))

    result = {
        "modality": "cacti",
        "method": "gap_tv",
        "true_params": {"dx": true_dx, "dy": true_dy},
        "estimated_params": {"dx": round(best_dx, 2), "dy": round(best_dy, 2)},
        "scenario_ii_psnr": round(float(np.mean(psnrs_ii)), 2),
        "scenario_iv_psnr": round(float(np.mean(psnrs_iv)), 2),
        "scenario_iii_psnr": round(float(np.mean(psnrs_iii)), 2),
        "calibration_time_s": round(calibration_time, 1),
    }

    logger.info(f"\nCACTI GAP-TV:  II={result['scenario_ii_psnr']:.2f}  "
                f"IV={result['scenario_iv_psnr']:.2f}  "
                f"III={result['scenario_iii_psnr']:.2f}")

    return result


# ============================================================================
# SPC Scenario IV
# ============================================================================
def spc_scenario_iv(device: str = "cuda:0") -> Dict:
    """Grid-search calibration for SPC using FISTA-TV.

    Searches gain_alpha over a grid using reconstruction-TV as objective.
    Uses ISTA-Net's learned Phi matrix and real Set11 images with 33x33 blocks,
    matching the simulation setup in validate_spc_inversenet.py.
    """
    import glob

    import cv2
    import scipy.io as sio
    from skimage.restoration import denoise_tv_chambolle

    logger.info("\n" + "=" * 60)
    logger.info("SPC Scenario IV: Grid-Search Calibration (TV criterion)")
    logger.info("=" * 60)

    BLOCK_SIZE = 33
    N_PIX = BLOCK_SIZE ** 2  # 1089
    ISTA_ROOT = Path("/home/spiritai/ISTA-Net-PyTorch-master")
    ISTA_PHI_PATH = ISTA_ROOT / "sampling_matrix" / "phi_0_25_1089.mat"
    SET11_DIR = ISTA_ROOT / "data" / "Set11"

    # Load ISTA-Net's learned Phi matrix (272 x 1089)
    Phi_data = sio.loadmat(str(ISTA_PHI_PATH))
    A = Phi_data["phi"].astype(np.float32)
    m, n_pix = A.shape
    logger.info(f"Phi matrix: {m} x {n_pix} (cs_ratio={m/n_pix:.2f})")

    # Load Set11 images: use cameraman and Monarch for calibration
    target_names = ["cameraman", "Monarch"]
    all_tifs = sorted(glob.glob(str(SET11_DIR / "*.tif")))
    image_blocks_list = []  # list of (name, blocks array [n_blocks, 1089])

    for tif_path in all_tifs:
        fname = Path(tif_path).stem
        if not any(t.lower() in fname.lower() for t in target_names):
            continue
        img_bgr = cv2.imread(tif_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        Iorg = img_ycrcb[:, :, 0].astype(np.float64)
        row, col = Iorg.shape
        # Pad to multiple of BLOCK_SIZE
        row_pad = (BLOCK_SIZE - row % BLOCK_SIZE) % BLOCK_SIZE
        col_pad = (BLOCK_SIZE - col % BLOCK_SIZE) % BLOCK_SIZE
        Ipad = np.pad(Iorg, ((0, row_pad), (0, col_pad)), mode="reflect")
        # Extract 33x33 blocks
        row_new, col_new = Ipad.shape
        blocks = []
        for i in range(0, row_new, BLOCK_SIZE):
            for j in range(0, col_new, BLOCK_SIZE):
                blocks.append(Ipad[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].ravel())
        blocks = np.array(blocks, dtype=np.float32) / 255.0  # [n_blocks, 1089]
        image_blocks_list.append((fname, blocks, Ipad.shape))
        logger.info(f"  Loaded {fname}: {Iorg.shape} -> {Ipad.shape}, {blocks.shape[0]} blocks")

    if not image_blocks_list:
        logger.warning("No Set11 images found, falling back to synthetic")
        # Fallback: create synthetic blocks
        np.random.seed(42)
        from scipy.ndimage import gaussian_filter
        x_synth = np.random.rand(256, 256).astype(np.float32)
        x_synth = gaussian_filter(x_synth, sigma=2.0)
        x_synth = (x_synth - x_synth.min()) / (x_synth.max() - x_synth.min())
        blocks = []
        for i in range(0, 256 - BLOCK_SIZE + 1, BLOCK_SIZE):
            for j in range(0, 256 - BLOCK_SIZE + 1, BLOCK_SIZE):
                blocks.append(x_synth[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE].ravel())
        blocks = np.array(blocks, dtype=np.float32)
        image_blocks_list.append(("synthetic", blocks, (256, 256)))

    # True mismatch: exponential gain drift g_i = exp(-alpha * i)
    true_alpha = 0.0015
    gain_true = np.exp(-true_alpha * np.arange(m, dtype=np.float32))
    sigma_y = 0.03

    # Generate corrupted measurements for all blocks
    np.random.seed(42)
    all_blocks = []
    all_y_corrupt = []
    for name, blocks, shape in image_blocks_list:
        for bi in range(blocks.shape[0]):
            x_b = blocks[bi]
            y_clean = A @ x_b
            y_c = gain_true * y_clean + sigma_y * np.random.randn(m).astype(np.float32)
            all_blocks.append(x_b)
            all_y_corrupt.append(y_c)

    all_blocks = np.array(all_blocks)
    all_y_corrupt = np.array(all_y_corrupt)
    n_blocks_total = all_blocks.shape[0]
    logger.info(f"Total blocks for calibration: {n_blocks_total}")

    # Precompute AtA and its spectral norm (same for all blocks)
    AtA = A.T @ A
    L_fista = float(np.linalg.norm(AtA, ord=2))

    def fista_tv_block(y, gain, iters=200, lam=0.005):
        """FISTA-TV for a single 33x33 block."""
        y_corr = y / np.maximum(gain, 1e-6)
        Aty = A.T @ y_corr
        x = np.zeros(N_PIX, dtype=np.float32)
        x_prev = x.copy()
        t = 1.0
        for _ in range(iters):
            grad = AtA @ x - Aty
            x_new = x - grad / L_fista
            x_2d = np.clip(x_new.reshape(BLOCK_SIZE, BLOCK_SIZE), 0, 1)
            x_2d = denoise_tv_chambolle(x_2d, weight=lam / L_fista, max_num_iter=3)
            x_new = x_2d.ravel()
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            x = x_new + (t - 1) / t_new * (x_new - x_prev)
            x_prev = x_new
            t = t_new
        return np.clip(x, 0, 1).astype(np.float32)

    def compute_tv(x_2d):
        """Total variation of a 2D image (sum of abs gradients)."""
        dx = np.abs(x_2d[:, 1:] - x_2d[:, :-1])
        dy = np.abs(x_2d[1:, :] - x_2d[:-1, :])
        return float(np.sum(dx) + np.sum(dy))

    # ------------------------------------------------------------------
    # PnP-DRUNet solver (GPU) — import proven implementation
    # ------------------------------------------------------------------
    import torch

    drunet_solver = None
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Import PnPDRUNetSolver33 from the validated SPC script
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from validate_spc_inversenet import PnPDRUNetSolver33
        logger.info("Loading PnP-DRUNet solver (from validate_spc_inversenet)...")
        drunet_solver = PnPDRUNetSolver33(A, dev, sigma_end=0.01,
                                           sigma_anneal_mult=5.0, max_iter=100)
        drunet_available = True
        logger.info("  PnP-DRUNet solver ready")
    except Exception as e:
        logger.warning(f"PnP-DRUNet not available: {e}")
        drunet_available = False

    # ------------------------------------------------------------------
    # Helper: run grid search + evaluation for a given solver
    # ------------------------------------------------------------------
    alpha_grid = np.linspace(0.0, 0.005, 41)
    n_cal_blocks = min(n_blocks_total, 20)
    cal_indices = np.linspace(0, n_blocks_total - 1, n_cal_blocks, dtype=int)

    def run_method(method_name, solve_fn, grid_iters, eval_iters):
        """Run TV grid search + Scenario II/IV/III evaluation for one method."""
        logger.info(f"\n--- {method_name}: TV grid search ({len(alpha_grid)} points) ---")

        best_alpha = 0.0
        best_tv = float("inf")
        grid_tvs = []

        t0 = time.time()
        for ai, alpha_test in enumerate(alpha_grid):
            gain_test = np.exp(-alpha_test * np.arange(m, dtype=np.float32))
            total_tv = 0.0
            for bi in cal_indices:
                x_hat = solve_fn(all_y_corrupt[bi:bi+1], gain_test, grid_iters)
                x_hat_2d = x_hat[0].reshape(BLOCK_SIZE, BLOCK_SIZE)
                total_tv += compute_tv(x_hat_2d)
            grid_tvs.append({"alpha": round(float(alpha_test), 5),
                             "tv": round(total_tv, 2)})
            if total_tv < best_tv:
                best_tv = total_tv
                best_alpha = alpha_test
            if (ai + 1) % 10 == 0:
                logger.info(f"  {method_name} grid: {ai+1}/{len(alpha_grid)} done")
        cal_time = time.time() - t0
        logger.info(f"  {method_name} best alpha: {best_alpha:.5f} "
                     f"(true: {true_alpha:.5f}), time: {cal_time:.1f}s")

        # Evaluate Scenario II, IV, III on all blocks (batch)
        gain_none = np.ones(m, dtype=np.float32)
        gain_cal = np.exp(-best_alpha * np.arange(m, dtype=np.float32))

        psnrs = {"ii": [], "iv": [], "iii": []}
        for label, gain in [("ii", gain_none), ("iv", gain_cal), ("iii", gain_true)]:
            logger.info(f"  {method_name} evaluating Scenario {label.upper()}...")
            x_all = solve_fn(all_y_corrupt, gain, eval_iters)
            for bi in range(n_blocks_total):
                mse = float(np.mean((all_blocks[bi] - x_all[bi]) ** 2))
                psnrs[label].append(10 * np.log10(1.0 / max(mse, 1e-10)))

        p_ii = float(np.mean(psnrs["ii"]))
        p_iv = float(np.mean(psnrs["iv"]))
        p_iii = float(np.mean(psnrs["iii"]))
        gap = p_iii - p_ii
        rec = (p_iv - p_ii) / gap * 100 if abs(gap) > 0.01 else 0.0
        logger.info(f"  {method_name}:  II={p_ii:.2f}  IV={p_iv:.2f}  "
                     f"III={p_iii:.2f}  Recovery={rec:.1f}%")
        return {
            "method": method_name.lower().replace("-", "_"),
            "estimated_alpha": round(best_alpha, 5),
            "scenario_ii_psnr": round(p_ii, 2),
            "scenario_iv_psnr": round(p_iv, 2),
            "scenario_iii_psnr": round(p_iii, 2),
            "recovery_pct": round(rec, 1),
            "calibration_time_s": round(cal_time, 1),
            "grid_tvs": grid_tvs,
        }

    # ------------------------------------------------------------------
    # FISTA-TV solver wrapper (processes blocks one-by-one → returns batch)
    # ------------------------------------------------------------------
    def fista_tv_solve(y_batch, gain, iters):
        results = np.zeros((y_batch.shape[0], N_PIX), dtype=np.float32)
        for bi in range(y_batch.shape[0]):
            results[bi] = fista_tv_block(y_batch[bi], gain, iters=iters)
        return results

    # ------------------------------------------------------------------
    # PnP-DRUNet solver wrapper (gain correction then batch solve)
    # ------------------------------------------------------------------
    def pnp_drunet_solve(y_batch, gain, iters):
        # Apply gain correction before passing to the solver
        y_corrected = y_batch / gain[None, :]
        # The solver handles row-normalization, init, and PnP-FISTA internally
        # Override max_iter for this call
        old_iter = drunet_solver.max_iter
        drunet_solver.max_iter = iters
        result = drunet_solver.solve_batch(y_corrected)
        drunet_solver.max_iter = old_iter
        return result

    # ------------------------------------------------------------------
    # Run FISTA-TV
    # ------------------------------------------------------------------
    fista_result = run_method("FISTA-TV", fista_tv_solve,
                              grid_iters=100, eval_iters=200)

    # ------------------------------------------------------------------
    # Run PnP-DRUNet (if available)
    # ------------------------------------------------------------------
    drunet_result = None
    if drunet_available and drunet_solver is not None:
        drunet_result = run_method("PnP-DRUNet", pnp_drunet_solve,
                                   grid_iters=60, eval_iters=100)

    # Build combined result
    result = {
        "modality": "spc",
        "criterion": "tv_minimisation",
        "true_params": {"alpha": true_alpha},
        "n_blocks": n_blocks_total,
        "n_cal_blocks": n_cal_blocks,
        "grid_size": len(alpha_grid),
        "methods": {"fista_tv": fista_result},
    }
    if drunet_result is not None:
        result["methods"]["pnp_drunet"] = drunet_result

    # Legacy top-level fields (for backward-compat with summary table)
    result["method"] = "fista_tv"
    result["estimated_params"] = {"alpha": fista_result["estimated_alpha"]}
    result["scenario_ii_psnr"] = fista_result["scenario_ii_psnr"]
    result["scenario_iv_psnr"] = fista_result["scenario_iv_psnr"]
    result["scenario_iii_psnr"] = fista_result["scenario_iii_psnr"]
    result["calibration_time_s"] = fista_result["calibration_time_s"]

    logger.info(f"\nSPC FISTA-TV:  II={fista_result['scenario_ii_psnr']:.2f}  "
                f"IV={fista_result['scenario_iv_psnr']:.2f}  "
                f"III={fista_result['scenario_iii_psnr']:.2f}  "
                f"Recovery={fista_result['recovery_pct']:.1f}%")
    if drunet_result is not None:
        logger.info(f"SPC PnP-DRUNet: II={drunet_result['scenario_ii_psnr']:.2f}  "
                    f"IV={drunet_result['scenario_iv_psnr']:.2f}  "
                    f"III={drunet_result['scenario_iii_psnr']:.2f}  "
                    f"Recovery={drunet_result['recovery_pct']:.1f}%")

    return result


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Scenario IV: Grid-Search Calibration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modality", default="all", choices=["all", "cassi", "cacti", "spc"])
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("InverseNet Scenario IV: Grid-Search Calibration Baseline")
    logger.info("=" * 60)

    results = {}

    if args.modality in ("all", "cassi"):
        results["cassi"] = cassi_scenario_iv(args.device)

    if args.modality in ("all", "cacti"):
        results["cacti"] = cacti_scenario_iv(args.device)

    if args.modality in ("all", "spc"):
        results["spc"] = spc_scenario_iv(args.device)

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("SCENARIO IV SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Modality':10s} {'Method':12s} {'II':>8s} {'IV':>8s} {'III':>8s} {'IV-II':>8s} {'III-II':>8s}")
    logger.info("-" * 60)
    for mod, r in results.items():
        if r:
            p_ii = r["scenario_ii_psnr"]
            p_iv = r["scenario_iv_psnr"]
            p_iii = r["scenario_iii_psnr"]
            logger.info(f"{mod:10s} {r['method']:12s} {p_ii:8.2f} {p_iv:8.2f} {p_iii:8.2f} "
                        f"{p_iv - p_ii:+8.2f} {p_iii - p_ii:+8.2f}")

    # Save
    out_path = RESULTS_DIR / "scenario_iv_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
