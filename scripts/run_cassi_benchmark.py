#!/usr/bin/env python3
"""CASSI Real-Data Benchmark — 4 Solvers on 10 Hyperspectral Scenes.

Runs GAP-TV, HDNet, MST-S, and MST-L on the TSA simulation benchmark
(scene01–scene10, 256×256×28, step=2 dispersion).

W1: 4-solver comparison on all 10 scenes
W2: Mask-shift mismatch + correction on scene01 (GAP-TV)

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cassi_benchmark.py
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cassi_benchmark.py --scenes scene01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import scipy.io as sio

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "packages", "pwm_core"))

# External codebases (paths only — added to sys.path on demand to avoid conflicts)
PNPCASSI_ROOT = "/home/spiritai/PnP-CASSI-main"
MST_TEST_ROOT = "/home/spiritai/MST-main/simulation/test_code"

from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)

# ── Constants ────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "cassi"
DATASET_DIR = "/home/spiritai/MST-main/datasets/TSA_simu_data"
MASK_PATH = DATASET_DIR
TRUTH_PATH = os.path.join(DATASET_DIR, "Truth")
ALL_SCENES = [f"scene{i:02d}" for i in range(1, 11)]
PIXEL_MAX = 1.0       # truth is in [0, ~0.91]
NC = 28               # spectral bands
STEP = 2              # dispersion step
NOISE_SIGMA = 0.01    # for NLL computation
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Model weight paths
HDNET_WEIGHTS = "/home/spiritai/MST-main/model_zoo/hdnet/hdnet.pth"
MST_S_WEIGHTS = "/home/spiritai/MST-main/model_zoo/mst/mst_s.pth"
MST_L_WEIGHTS = "/home/spiritai/MST-main/model_zoo/mst/mst_l.pth"


# ── Helpers ──────────────────────────────────────────────────────────────

def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def psnr_band(gt: np.ndarray, rec: np.ndarray, maxval: float = 1.0) -> float:
    mse = np.mean((gt.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(maxval ** 2 / mse))


def ssim_band(gt: np.ndarray, rec: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        gt.astype(np.float64), rec.astype(np.float64), data_range=1.0
    ))


def compute_metrics_allbands(orig: np.ndarray, recon: np.ndarray) -> dict:
    """Compute per-band and mean PSNR/SSIM for (H,W,L) arrays in [0,1]."""
    nbands = orig.shape[2]
    psnrs, ssims = [], []
    for b in range(nbands):
        psnrs.append(psnr_band(orig[:, :, b], recon[:, :, b]))
        ssims.append(ssim_band(orig[:, :, b], recon[:, :, b]))
    return {
        "mean_psnr": round(mean(psnrs), 2),
        "mean_ssim": round(mean(ssims), 4),
        "per_band_psnr": [round(p, 2) for p in psnrs],
        "per_band_ssim": [round(s, 4) for s in ssims],
    }


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def load_mask():
    """Load 2D mask (256,256) and pre-shifted 3D mask (256,310,28)."""
    mask2d = sio.loadmat(os.path.join(MASK_PATH, "mask.mat"))["mask"].astype(np.float32)
    mask3d_shift = sio.loadmat(
        os.path.join(MASK_PATH, "mask_3d_shift.mat")
    )["mask_3d_shift"].astype(np.float32)
    return mask2d, mask3d_shift


def load_truth_scene(scene: str) -> np.ndarray:
    """Load single scene truth (256,256,28) float32 in [0,~0.91]."""
    path = os.path.join(TRUTH_PATH, f"{scene}.mat")
    return sio.loadmat(path)["img"].astype(np.float32)


def make_mask3d_from_2d(mask2d: np.ndarray, step: int = STEP) -> np.ndarray:
    """Construct shifted 3D mask (H, W+(NC-1)*step, NC) from 2D mask."""
    H, W = mask2d.shape
    mask3d = np.tile(mask2d[:, :, np.newaxis], (1, 1, NC))
    return shift_np(mask3d, step=step)


def shift_np(inputs: np.ndarray, step: int = STEP) -> np.ndarray:
    """Shift each band along columns: band i is shifted by i*step pixels."""
    row, col, nC = inputs.shape
    output = np.zeros((row, col + (nC - 1) * step, nC), dtype=inputs.dtype)
    for i in range(nC):
        output[:, i * step:i * step + col, i] = inputs[:, :, i]
    return output


def shift_back_np(inputs: np.ndarray, step: int = STEP) -> np.ndarray:
    """Inverse shift: extract band i from column offset i*step."""
    row, col, nC = inputs.shape
    W = col - (nC - 1) * step
    output = np.zeros((row, W, nC), dtype=inputs.dtype)
    for i in range(nC):
        output[:, :, i] = inputs[:, i * step:i * step + W, i]
    return output


# ══════════════════════════════════════════════════════════════════════════
# sys.path management
# ══════════════════════════════════════════════════════════════════════════

def _enter_pnpcassi():
    """Set sys.path for PnP-CASSI utils (A, At, shift, TV_denoiser)."""
    if MST_TEST_ROOT in sys.path:
        sys.path.remove(MST_TEST_ROOT)
    if PNPCASSI_ROOT not in sys.path:
        sys.path.insert(0, PNPCASSI_ROOT)
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if hasattr(mod, "__file__") and mod.__file__ and "PnP-CASSI" not in (mod.__file__ or ""):
            del sys.modules["utils"]


def _enter_mst():
    """Set sys.path for MST test_code (model_generator, utils)."""
    if PNPCASSI_ROOT in sys.path:
        sys.path.remove(PNPCASSI_ROOT)
    if MST_TEST_ROOT not in sys.path:
        sys.path.insert(0, MST_TEST_ROOT)
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if hasattr(mod, "__file__") and mod.__file__ and "MST-main" not in (mod.__file__ or ""):
            del sys.modules["utils"]


# ══════════════════════════════════════════════════════════════════════════
# Solver 1: GAP-TV (self-contained, no PnP-CASSI import needed)
# ══════════════════════════════════════════════════════════════════════════

def _A_cassi(x, Phi):
    """Forward model: collapse shifted 3D cube via element-wise mask."""
    return np.sum(x * Phi, axis=2)


def _At_cassi(y, Phi):
    """Transpose: expand 2D measurement back to 3D."""
    return np.multiply(np.repeat(y[:, :, np.newaxis], Phi.shape[2], axis=2), Phi)


def _tv_denoiser(x, _lambda, n_iter_max):
    """Isotropic TV denoiser using dual variable (Chambolle 2004)."""
    dt = 0.25
    N = x.shape
    idx = np.arange(1, N[0] + 1); idx[-1] = N[0] - 1
    iux = np.arange(-1, N[0] - 1); iux[0] = 0
    ir = np.arange(1, N[1] + 1); ir[-1] = N[1] - 1
    il = np.arange(-1, N[1] - 1); il[0] = 0
    p1 = np.zeros_like(x)
    p2 = np.zeros_like(x)
    divp = np.zeros_like(x)
    for _ in range(n_iter_max):
        z = divp - x * _lambda
        z1 = z[:, ir, :] - z
        z2 = z[idx, :, :] - z
        denom_2d = 1 + dt * np.sqrt(np.sum(z1 ** 2 + z2 ** 2, 2))
        denom_3d = np.tile(denom_2d[:, :, np.newaxis], (1, 1, N[2]))
        p1 = (p1 + dt * z1) / denom_3d
        p2 = (p2 + dt * z2) / denom_3d
        divp = p1 - p1[:, il, :] + p2 - p2[iux, :, :]
    return x - divp / _lambda


def gap_tv_cassi(y, Phi, step=STEP, iter_max=50, tv_weight=0.1,
                 tv_iter_max=5, _lambda=1, accelerate=True,
                 X_orig=None, show_iqa=True):
    """GAP-TV reconstruction for CASSI (step=2, numpy, CPU).

    Operates in the shifted domain.  Phi is the 3D shifted mask.
    Returns (x_unshifted, psnr_list, ssim_list).
    """
    x = _At_cassi(y, Phi)
    y1 = np.zeros_like(y)
    Phi_sum = np.sum(Phi, 2)
    Phi_sum[Phi_sum == 0] = 1
    psnr_list, ssim_list = [], []

    for it in range(iter_max):
        yb = _A_cassi(x, Phi)
        if accelerate:
            y1 = y1 + (y - yb)
            x = x + _lambda * _At_cassi((y1 - yb) / Phi_sum, Phi)
        else:
            x = x + _lambda * _At_cassi((y - yb) / Phi_sum, Phi)

        # Shift back for TV denoising
        x_unshifted = shift_back_np(x, step=step)
        x_unshifted = _tv_denoiser(x_unshifted, tv_weight, n_iter_max=tv_iter_max)
        # Shift forward again
        x = shift_np(x_unshifted, step=step)

        if show_iqa and X_orig is not None:
            from skimage.metrics import structural_similarity
            p = psnr_band(X_orig, x_unshifted, maxval=1.0)
            s = float(np.mean([
                structural_similarity(
                    X_orig[:, :, b].astype(np.float64),
                    x_unshifted[:, :, b].astype(np.float64),
                    data_range=1.0
                ) for b in range(X_orig.shape[2])
            ]))
            psnr_list.append(p)
            ssim_list.append(s)
            if (it + 1) % 10 == 0:
                print(f"      GAP-TV iter {it+1:3d}: PSNR {p:.2f} dB, SSIM {s:.4f}")

    x_final = shift_back_np(x, step=step)
    return x_final, psnr_list, ssim_list


def run_gap_tv_scene(truth, mask2d, mask3d_shift):
    """Run GAP-TV on one scene.  Returns (recon, wall_time)."""
    # Generate measurement in shifted domain
    truth_shift = shift_np(truth, step=STEP)  # (256, 310, 28)
    y = _A_cassi(truth_shift, mask3d_shift)   # (256, 310)

    t0 = time.time()
    x_hat, psnr_list, ssim_list = gap_tv_cassi(
        y, mask3d_shift, step=STEP, iter_max=50,
        tv_weight=0.1, tv_iter_max=5, _lambda=1, accelerate=True,
        X_orig=truth, show_iqa=True,
    )
    wall = time.time() - t0
    return np.clip(x_hat, 0, 1).astype(np.float32), wall


# ══════════════════════════════════════════════════════════════════════════
# Deep solver loading (MST codebase)
# ══════════════════════════════════════════════════════════════════════════

def load_deep_models():
    """Load HDNet, MST-S, MST-L from the MST codebase. Returns dict of models."""
    _enter_mst()
    import torch
    from architecture import model_generator

    models = {}

    # HDNet — model_generator returns (model, fdl_loss) for hdnet
    print("    Loading HDNet ...")
    hdnet_model, _ = model_generator("hdnet", HDNET_WEIGHTS)
    hdnet_model.eval()
    models["hdnet"] = hdnet_model

    # MST-S
    print("    Loading MST-S ...")
    mst_s = model_generator("mst_s", MST_S_WEIGHTS)
    mst_s.eval()
    models["mst_s"] = mst_s

    # MST-L
    print("    Loading MST-L ...")
    mst_l = model_generator("mst_l", MST_L_WEIGHTS)
    mst_l.eval()
    models["mst_l"] = mst_l

    return models


def run_deep_solvers(models, truths, mask2d, mask3d_shift):
    """Run all 3 deep solvers on 10 scenes (batch GPU inference).

    Args:
        models: dict with 'hdnet', 'mst_s', 'mst_l'
        truths: list of 10 truth arrays (256,256,28)
        mask2d: (256,256) float32
        mask3d_shift: (256,310,28) float32

    Returns:
        dict solver_name -> list of 10 recon arrays (256,256,28)
        dict solver_name -> wall_time
    """
    _enter_mst()
    import torch
    from utils import init_mask, init_meas, shift as mst_shift

    n_scenes = len(truths)

    # Prepare mask tensors
    mask3d_batch = torch.from_numpy(
        np.tile(mask2d[np.newaxis, :, :], (NC, 1, 1))  # (28, 256, 256)
    ).unsqueeze(0).expand(n_scenes, NC, 256, 256).cuda().float()

    # Shifted mask for MST models: (nscenes, 28, 256, 310)
    Phi_batch = torch.from_numpy(
        mask3d_shift.transpose(2, 0, 1)  # (28, 256, 310)
    ).unsqueeze(0).expand(n_scenes, NC, 256, 310).cuda().float()

    # Prepare truth tensor: (N, 28, 256, 256) — CHW format
    test_gt = np.stack(truths, axis=0)  # (10, 256, 256, 28)
    test_gt = torch.from_numpy(
        test_gt.transpose(0, 3, 1, 2)  # (10, 28, 256, 256)
    ).cuda().float()

    # Generate measurement H (unfolded): init_meas with Y2H=True
    # H = shift_back(sum(shift(mask3d * x, 2), dim=1) / nC * 2)
    # Using MST's gen_meas_torch via init_meas
    temp = mst_shift(mask3d_batch * test_gt, STEP)           # (10, 28, 256, 310)
    meas = torch.sum(temp, 1)                                 # (10, 256, 310)
    H = meas / NC * 2
    # shift_back: [bs, 256, 310] -> [bs, 28, 256, 256]
    from utils import shift_back as mst_shift_back
    H = mst_shift_back(H, step=STEP)                          # (10, 28, 256, 256)

    results = {}
    times = {}

    # HDNet: input = H only (no mask)
    print("  [2/4] HDNet (batch GPU) ...")
    hdnet = models["hdnet"]
    hdnet.eval()
    t0 = time.time()
    with torch.no_grad():
        out_hdnet = hdnet(H)
    torch.cuda.synchronize()
    times["hdnet"] = time.time() - t0
    results["hdnet"] = out_hdnet.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    print(f"    HDNet done in {times['hdnet']:.1f}s")

    # MST-S: input = H + Phi (shifted mask)
    print("  [3/4] MST-S (batch GPU) ...")
    mst_s = models["mst_s"]
    mst_s.eval()
    t0 = time.time()
    with torch.no_grad():
        out_mst_s = mst_s(H, Phi_batch)
    torch.cuda.synchronize()
    times["mst_s"] = time.time() - t0
    results["mst_s"] = out_mst_s.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    print(f"    MST-S done in {times['mst_s']:.1f}s")

    # MST-L: input = H + Phi (shifted mask)
    print("  [4/4] MST-L (batch GPU) ...")
    mst_l = models["mst_l"]
    mst_l.eval()
    t0 = time.time()
    with torch.no_grad():
        out_mst_l = mst_l(H, Phi_batch)
    torch.cuda.synchronize()
    times["mst_l"] = time.time() - t0
    results["mst_l"] = out_mst_l.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    print(f"    MST-L done in {times['mst_l']:.1f}s")

    return results, times


# ══════════════════════════════════════════════════════════════════════════
# W2: Mask-shift mismatch + correction (scene01, GAP-TV)
# ══════════════════════════════════════════════════════════════════════════

def run_w2_mask_shift(truth, mask2d, mask3d_shift):
    """W2: Inject 2px horizontal mask shift, grid search to recover, compare."""
    shift_inject = 2

    # Perturbed mask: shift 2D mask horizontally by 2 pixels
    mask2d_pert = np.roll(mask2d, shift_inject, axis=1)
    mask3d_pert = make_mask3d_from_2d(mask2d_pert, step=STEP)

    # Generate measurement with perturbed mask
    truth_shift = shift_np(truth, step=STEP)
    y_pert = _A_cassi(truth_shift, mask3d_pert)  # (256, 310)

    # NLL with nominal (un-shifted) mask
    y_pred_nominal = _A_cassi(truth_shift, mask3d_shift)
    nll_before = compute_nll_gaussian(y_pert, y_pred_nominal, sigma=NOISE_SIGMA)

    # Uncorrected reconstruction: use nominal mask on perturbed measurement
    print("    Uncorrected reconstruction ...")
    t0 = time.time()
    x_uncorr, _, _ = gap_tv_cassi(
        y_pert, mask3d_shift, step=STEP, iter_max=50,
        tv_weight=0.1, tv_iter_max=5, _lambda=1, accelerate=True,
        X_orig=truth, show_iqa=False,
    )
    t_uncorr = time.time() - t0
    x_uncorr = np.clip(x_uncorr, 0, 1).astype(np.float32)
    m_uncorr = compute_metrics_allbands(truth, x_uncorr)

    # Grid search over horizontal shifts [-5, +5]
    print("    Grid search over horizontal shifts [-5, +5] ...")
    best_nll, best_shift = np.inf, 0
    for trial_shift in range(-5, 6):
        mask2d_trial = np.roll(mask2d, trial_shift, axis=1)
        mask3d_trial = make_mask3d_from_2d(mask2d_trial, step=STEP)
        y_pred_trial = _A_cassi(truth_shift, mask3d_trial)
        nll_trial = compute_nll_gaussian(y_pert, y_pred_trial, sigma=NOISE_SIGMA)
        if nll_trial < best_nll:
            best_nll = nll_trial
            best_shift = trial_shift

    print(f"    Best shift: {best_shift} pixels (NLL={best_nll:.1f})")
    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    # Corrected reconstruction: use best-shift mask
    mask2d_corr = np.roll(mask2d, best_shift, axis=1)
    mask3d_corr = make_mask3d_from_2d(mask2d_corr, step=STEP)

    print("    Corrected reconstruction ...")
    t0 = time.time()
    x_corr, _, _ = gap_tv_cassi(
        y_pert, mask3d_corr, step=STEP, iter_max=50,
        tv_weight=0.1, tv_iter_max=5, _lambda=1, accelerate=True,
        X_orig=truth, show_iqa=False,
    )
    t_corr = time.time() - t0
    x_corr = np.clip(x_corr, 0, 1).astype(np.float32)
    m_corr = compute_metrics_allbands(truth, x_corr)

    return {
        "shift_injected": shift_inject,
        "shift_found": best_shift,
        "nll_before": round(nll_before, 1),
        "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "psnr_uncorrected": m_uncorr["mean_psnr"],
        "ssim_uncorrected": m_uncorr["mean_ssim"],
        "psnr_corrected": m_corr["mean_psnr"],
        "ssim_corrected": m_corr["mean_ssim"],
        "psnr_delta": round(m_corr["mean_psnr"] - m_uncorr["mean_psnr"], 2),
        "ssim_delta": round(m_corr["mean_ssim"] - m_uncorr["mean_ssim"], 4),
        "mask_nominal_hash": sha256_hex16(mask2d),
        "mask_corrected_hash": sha256_hex16(mask2d_corr),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CASSI Real-Data Benchmark")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                        help="Scenes to benchmark (default: all 10)")
    args = parser.parse_args()
    scenes = args.scenes

    results = {}
    print("=" * 70)
    print(f"CASSI Real-Data Benchmark — {len(scenes)} scene(s)")
    print("=" * 70)

    # ── Create RunBundle ─────────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "cassi_benchmark")
    rb_name = os.path.basename(rb_dir)
    art_dir = os.path.join(rb_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    print(f"RunBundle: {rb_name}")

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[0] Loading dataset and models ...")
    mask2d, mask3d_shift = load_mask()
    print(f"  Mask: {mask2d.shape}, Shift mask: {mask3d_shift.shape}")

    truths = []
    for sc in scenes:
        t = load_truth_scene(sc)
        truths.append(t)
        print(f"  {sc}: {t.shape}, range [{t.min():.3f}, {t.max():.3f}]")

    # ── Load deep models once ────────────────────────────────────────────
    t0_load = time.time()
    deep_models = load_deep_models()
    print(f"    All models loaded in {time.time() - t0_load:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # W1: 4-Solver Comparison — all scenes
    # ══════════════════════════════════════════════════════════════════════
    all_scene_results = {}

    # GAP-TV runs per-scene on CPU
    print(f"\n{'=' * 70}")
    print("W1: GAP-TV (per-scene, CPU)")
    print("=" * 70)
    gap_tv_recons = {}
    for si, sc in enumerate(scenes):
        print(f"\n  [{si+1}/{len(scenes)}] {sc} ...")
        scene_dir = os.path.join(art_dir, sc)
        os.makedirs(scene_dir, exist_ok=True)

        x_hat, wt = run_gap_tv_scene(truths[si], mask2d, mask3d_shift)
        m = compute_metrics_allbands(truths[si], x_hat)
        gap_tv_recons[sc] = x_hat
        if sc not in all_scene_results:
            all_scene_results[sc] = {}
        all_scene_results[sc]["gap_tv"] = {**m, "wall_time": round(wt, 2)}
        save_array(os.path.join(scene_dir, "x_hat_gap_tv.npy"), x_hat)
        print(f"    PSNR={m['mean_psnr']:.2f} dB, SSIM={m['mean_ssim']:.4f}, time={wt:.1f}s")

    # Deep solvers run in batch on GPU
    print(f"\n{'=' * 70}")
    print("W1: Deep solvers (batch GPU)")
    print("=" * 70)
    deep_recons, deep_times = run_deep_solvers(deep_models, truths, mask2d, mask3d_shift)

    for solver_name in ["hdnet", "mst_s", "mst_l"]:
        recons = deep_recons[solver_name]  # (10, 256, 256, 28)
        wt = deep_times[solver_name]
        per_scene_time = wt / len(scenes)

        for si, sc in enumerate(scenes):
            scene_dir = os.path.join(art_dir, sc)
            os.makedirs(scene_dir, exist_ok=True)
            x_hat = np.clip(recons[si], 0, 1).astype(np.float32)
            m = compute_metrics_allbands(truths[si], x_hat)
            if sc not in all_scene_results:
                all_scene_results[sc] = {}
            all_scene_results[sc][solver_name] = {**m, "wall_time": round(per_scene_time, 2)}
            save_array(os.path.join(scene_dir, f"x_hat_{solver_name}.npy"), x_hat)

        # Summary for this solver
        avg_p = mean([all_scene_results[sc][solver_name]["mean_psnr"] for sc in scenes])
        avg_s = mean([all_scene_results[sc][solver_name]["mean_ssim"] for sc in scenes])
        print(f"  {solver_name}: avg PSNR={avg_p:.2f} dB, avg SSIM={avg_s:.4f}, total time={wt:.1f}s")

    results["w1"] = all_scene_results

    # ── Compute average across all scenes ────────────────────────────────
    solver_names = ["gap_tv", "hdnet", "mst_s", "mst_l"]
    avg_results = {}
    for sname in solver_names:
        psnrs = [all_scene_results[sc][sname]["mean_psnr"] for sc in scenes]
        ssims = [all_scene_results[sc][sname]["mean_ssim"] for sc in scenes]
        times = [all_scene_results[sc][sname]["wall_time"] for sc in scenes]
        avg_results[sname] = {
            "avg_psnr": round(mean(psnrs), 2),
            "avg_ssim": round(mean(ssims), 4),
            "avg_time": round(mean(times), 2),
        }
    results["w1_average"] = avg_results

    # ══════════════════════════════════════════════════════════════════════
    # W2: Mask-shift mismatch + correction (scene01, GAP-TV)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("W2: Mask-shift mismatch + correction (scene01, GAP-TV)")
    print("=" * 70)

    w2 = run_w2_mask_shift(truths[0], mask2d, mask3d_shift)

    w2_report = {k: v for k, v in w2.items()}
    w2_report["a_definition"] = "callable"
    w2_report["a_extraction_method"] = "provided"
    w2_report["linearity"] = "linear"
    w2_report["mismatch_type"] = "synthetic_injected"
    w2_report["mismatch_description"] = f"Mask-detector horizontal shift: {w2['shift_injected']}px"
    w2_report["correction_family"] = "Pre"
    results["w2"] = w2_report

    save_operator_meta(rb_dir, {
        "a_definition": "callable",
        "a_extraction_method": "provided",
        "a_sha256": w2["mask_nominal_hash"],
        "linearity": "linear",
        "mismatch_type": "synthetic_injected",
        "mismatch_params": {"shift_injected": w2["shift_injected"]},
        "correction_family": "Pre",
        "fitted_params": {"shift": w2["shift_found"]},
        "nll_before": w2["nll_before"],
        "nll_after": w2["nll_after"],
        "nll_decrease_pct": w2["nll_decrease_pct"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    print(f"\n    Injected shift:    {w2['shift_injected']}px")
    print(f"    Recovered shift:   {w2['shift_found']}px")
    print(f"    NLL decrease:      {w2['nll_decrease_pct']:.1f}%")
    print(f"    PSNR delta:        {w2['psnr_delta']:+.2f} dB")

    # ── Trace (from scene01) ─────────────────────────────────────────────
    truth01 = truths[0]
    truth01_shift = shift_np(truth01, step=STEP)
    y01 = _A_cassi(truth01_shift, mask3d_shift)

    trace = {}
    trace["00_input_x"] = truth01.copy()
    trace["01_masked"] = (truth01 * np.tile(mask2d[:, :, np.newaxis], (1, 1, NC))).astype(np.float32)
    trace["02_shifted"] = truth01_shift.copy()
    trace["03_measurement"] = y01.copy()
    trace["04_recon_gaptv"] = gap_tv_recons.get(scenes[0], truth01).copy()
    save_trace(rb_dir, trace)

    results["trace"] = []
    for i, key in enumerate(sorted(trace.keys())):
        arr = trace[key]
        results["trace"].append({
            "stage": i, "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape), "dtype": str(arr.dtype),
            "range_min": round(float(arr.min()), 4),
            "range_max": round(float(arr.max()), 4),
            "artifact_path": f"artifacts/trace/{key}.npy",
        })

    # ── Environment ──────────────────────────────────────────────────────
    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()[:12]
    except Exception:
        pass

    results["env"] = {
        "seed": SEED, "pwm_version": git_sha,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
    }
    try:
        import scipy; results["env"]["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    try:
        import torch; results["env"]["torch_version"] = torch.__version__
    except ImportError:
        pass

    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name
    results["scenes"] = scenes

    results_path = os.path.join(rb_dir, "cassi_benchmark_results.json")
    save_json(results_path, results)

    # ══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — All Scenes")
    print("=" * 70)

    header = f"{'Scene':<12}"
    for sn in solver_names:
        header += f" {sn:>14}"
    print(f"\nPSNR (dB):")
    print(header)
    print("-" * len(header))
    for sc in scenes:
        row = f"{sc:<12}"
        for sn in solver_names:
            row += f" {all_scene_results[sc][sn]['mean_psnr']:>14.2f}"
        print(row)
    row = f"{'AVERAGE':<12}"
    for sn in solver_names:
        row += f" {avg_results[sn]['avg_psnr']:>14.2f}"
    print("-" * len(header))
    print(row)

    print(f"\nSSIM:")
    print(header)
    print("-" * len(header))
    for sc in scenes:
        row = f"{sc:<12}"
        for sn in solver_names:
            row += f" {all_scene_results[sc][sn]['mean_ssim']:>14.4f}"
        print(row)
    row = f"{'AVERAGE':<12}"
    for sn in solver_names:
        row += f" {avg_results[sn]['avg_ssim']:>14.4f}"
    print("-" * len(header))
    print(row)

    print(f"\nW2 (scene01): NLL decrease {w2['nll_decrease_pct']:.1f}%, "
          f"PSNR delta {w2['psnr_delta']:+.2f} dB")
    print(f"\nRunBundle: runs/{rb_name}")
    print(f"Results:   {results_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
