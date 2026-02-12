#!/usr/bin/env python3
"""CACTI Real-Data Benchmark — 4 Solvers on 6 Grayscale Benchmark Scenes.

Runs GAP-TV, PnP-FFDNet, ELP-Unfolding, and EfficientSCI on the full
grayscale benchmark (kobe32, crash32, aerial32, traffic48, runner40, drop40).

W1: 4-solver comparison on all 6 scenes
W2: Mask-shift mismatch + correction on kobe32 (GAP-TV)

Usage:
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_benchmark.py
    PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_benchmark.py --scenes kobe32
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
PNPSCI_ROOT = "/home/spiritai/PnP-SCI_python-master"
ELP_ROOT = "/home/spiritai/ELP-Unfolding-master"
ESCI_ROOT = "/home/spiritai/EfficientSCI-main"

# ── Monkey-patch skimage for PnP-SCI compatibility ──────────────────────
# PnP-SCI uses old scikit-image API (n_iter_max, multichannel).
# Current skimage uses (max_num_iter, channel_axis).
import skimage.restoration
import skimage.restoration._denoise as _skd

_orig_tv = _skd.denoise_tv_chambolle
def _compat_tv(image, weight=0.1, eps=0.0002, max_num_iter=200, **kw):
    if "n_iter_max" in kw:
        max_num_iter = kw.pop("n_iter_max")
    kw.pop("multichannel", None)
    return _orig_tv(image, weight=weight, eps=eps, max_num_iter=max_num_iter)

_orig_wavelet = _skd.denoise_wavelet
def _compat_wavelet(image, *a, **kw):
    mc = kw.pop("multichannel", None)
    if mc is not None and "channel_axis" not in kw:
        kw["channel_axis"] = -1 if mc else None
    return _orig_wavelet(image, *a, **kw)

_skd.denoise_tv_chambolle = _compat_tv
_skd.denoise_wavelet = _compat_wavelet
skimage.restoration.denoise_tv_chambolle = _compat_tv
skimage.restoration.denoise_wavelet = _compat_wavelet

from skimage import metrics as _skm
_orig_ssim_sk = _skm.structural_similarity
def _compat_ssim(*a, **kw):
    mc = kw.pop("multichannel", None)
    if mc is not None and "channel_axis" not in kw:
        kw["channel_axis"] = -1 if mc else None
    return _orig_ssim_sk(*a, **kw)
_skm.structural_similarity = _compat_ssim
skimage.metrics.structural_similarity = _compat_ssim

from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.artifacts import (
    save_trace, save_operator_meta, compute_operator_hash,
    save_json, save_array,
)

# ── Constants ────────────────────────────────────────────────────────────
SEED = 42
MODALITY = "cacti"
DATASET_DIR = os.path.join(PNPSCI_ROOT, "dataset", "cacti", "grayscale_benchmark")
ALL_SCENES = ["kobe32", "crash32", "aerial32", "traffic48", "runner40", "drop40"]
MAXB = 255.0
NOISE_SIGMA = 5.0
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# ── Helpers ──────────────────────────────────────────────────────────────

def sha256_hex16(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def psnr_frame(gt: np.ndarray, rec: np.ndarray, maxval: float = 255.0) -> float:
    mse = np.mean((gt.astype(np.float64) - rec.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(maxval ** 2 / mse))


def ssim_frame(gt: np.ndarray, rec: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        gt.astype(np.float64), rec.astype(np.float64), data_range=255.0
    ))


def compute_metrics_allframes(orig: np.ndarray, recon: np.ndarray) -> dict:
    """Compute per-frame and mean PSNR/SSIM for (H,W,F) arrays in [0,255]."""
    nframes = orig.shape[2]
    psnrs, ssims = [], []
    for f in range(nframes):
        psnrs.append(psnr_frame(orig[:, :, f], recon[:, :, f]))
        ssims.append(ssim_frame(orig[:, :, f], recon[:, :, f]))
    return {
        "mean_psnr": round(mean(psnrs), 2),
        "mean_ssim": round(mean(ssims), 4),
        "per_frame_psnr": [round(p, 2) for p in psnrs],
        "per_frame_ssim": [round(s, 4) for s in ssims],
    }


def compute_nll_gaussian(y: np.ndarray, y_hat: np.ndarray, sigma: float) -> float:
    return float(0.5 * np.sum((y - y_hat) ** 2) / sigma ** 2)


def load_scene(scene: str) -> tuple:
    """Load .mat file -> (meas, mask, orig)."""
    path = os.path.join(DATASET_DIR, f"{scene}_cacti.mat")
    data = sio.loadmat(path)
    return np.float32(data["meas"]), np.float32(data["mask"]), np.float32(data["orig"])


# ══════════════════════════════════════════════════════════════════════════
# sys.path management
# ══════════════════════════════════════════════════════════════════════════

def _enter_pnpsci():
    for p in [ELP_ROOT, ESCI_ROOT]:
        if p in sys.path:
            sys.path.remove(p)
    if PNPSCI_ROOT not in sys.path:
        sys.path.insert(0, PNPSCI_ROOT)
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if hasattr(mod, "__file__") and mod.__file__ and "PnP-SCI" not in mod.__file__:
            del sys.modules["utils"]


def _enter_elp():
    for p in [PNPSCI_ROOT, ESCI_ROOT]:
        if p in sys.path:
            sys.path.remove(p)
    if ELP_ROOT not in sys.path:
        sys.path.insert(0, ELP_ROOT)
    for mod_name in ["utils", "train_common"]:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, "__file__") and mod.__file__ and "ELP-Unfolding" not in (mod.__file__ or ""):
                del sys.modules[mod_name]


def _enter_esci():
    for p in [PNPSCI_ROOT, ELP_ROOT]:
        if p in sys.path:
            sys.path.remove(p)
    if ESCI_ROOT not in sys.path:
        sys.path.insert(0, ESCI_ROOT)
    for mod_name in ["utils", "train_common"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]


# ══════════════════════════════════════════════════════════════════════════
# Solver 1: GAP-TV (no persistent model)
# ══════════════════════════════════════════════════════════════════════════

def run_gap_tv(meas, mask, orig, nframe):
    _enter_pnpsci()
    from pnp_sci import admmdenoise_cacti
    from utils import A_, At_
    A = lambda x: A_(x, mask)
    At = lambda y: At_(y, mask)
    v, t_elapsed, psnr_list, ssim_list, _ = admmdenoise_cacti(
        meas, mask, A, At,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=nframe, MAXB=MAXB, maskdirection="plain",
        _lambda=1, accelerate=True,
        denoiser="tv", iter_max=40, tv_weight=0.3, tv_iter_max=5,
    )
    # PnP-SCI output is in [0,1] range
    v = np.clip(v * 255.0, 0, 255).astype(np.float32)
    return v, t_elapsed, psnr_list, ssim_list


# ══════════════════════════════════════════════════════════════════════════
# Solver 2: PnP-FFDNet (load model once)
# ══════════════════════════════════════════════════════════════════════════

def load_ffdnet_model():
    _enter_pnpsci()
    import torch
    from packages.ffdnet.models import FFDNet
    ffdnet_path = os.path.join(PNPSCI_ROOT, "packages", "ffdnet", "models", "net_gray.pth")
    net = FFDNet(num_input_channels=1)
    state_dict = torch.load(ffdnet_path, map_location="cuda:0")
    model = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_pnp_ffdnet(meas, mask, orig, nframe, ffdnet_model):
    _enter_pnpsci()
    from pnp_sci import admmdenoise_cacti
    from utils import A_, At_
    A = lambda x: A_(x, mask)
    At = lambda y: At_(y, mask)
    v, t_elapsed, psnr_list, ssim_list, _ = admmdenoise_cacti(
        meas, mask, A, At,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=nframe, MAXB=MAXB, maskdirection="plain",
        _lambda=1, accelerate=True,
        denoiser="ffdnet", model=ffdnet_model,
        iter_max=[10, 10, 10, 10],
        sigma=[50/255, 25/255, 12/255, 6/255],
    )
    v = np.clip(v * 255.0, 0, 255).astype(np.float32)
    return v, t_elapsed, psnr_list, ssim_list


# ══════════════════════════════════════════════════════════════════════════
# Solver 3: ELP-Unfolding (load model once)
# ══════════════════════════════════════════════════════════════════════════

def load_elp_model():
    _enter_elp()
    import torch
    os.environ["WANDB_MODE"] = "disabled"
    elp_args = {
        "GPU": 0, "batch_size": 3, "temporal_length": 8, "patchsize": 256,
        "iter__number": 8, "init_channels": 512, "pres_channels": 512,
        "init_input": 8, "pres_input": 8, "priors": 6,
        "lr": 2e-5, "resume_training": True,
        "code_dir": "/home/spiritai/elpData/traindata/DAVIS-480-train/code",
        "log_dir": "/home/spiritai/ELP-Unfolding-master/trained_dataset",
        "use_first_stage": False,
    }
    from train_common import resume_training
    model, _, _ = resume_training(elp_args)
    model.eval()
    return model


def run_elp_unfolding(orig, mask_dataset, elp_model):
    """Run ELP-Unfolding using a pre-loaded model."""
    _enter_elp()
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_total = orig.shape[2]
    n_coded = n_total // 8
    batch_size = n_coded

    z = np.zeros((256, 256, 8, batch_size), dtype=np.float32)
    for fn in range(batch_size):
        z[:, :, :, fn] = orig[:, :, fn * 8:(fn + 1) * 8]
    z_tensor = torch.from_numpy(z).float().permute(3, 2, 0, 1)
    img_val = z_tensor / 255.0

    mask_tensor = torch.from_numpy(mask_dataset).float().permute(2, 0, 1)
    mask_tensor = mask_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    measurement = torch.sum(img_val * mask_tensor, dim=1, keepdim=True)

    mask_gpu = mask_tensor.to(device)
    meas_gpu = measurement.to(device)
    img_out_ori = torch.ones(batch_size, 8, 256, 256).to(device)

    t0 = time.time()
    with torch.no_grad():
        img_out, _ = elp_model(mask_gpu, meas_gpu, img_out_ori)
    torch.cuda.synchronize()
    t_elapsed = time.time() - t0

    recon = img_out[-1].detach().cpu().numpy()
    recon_255 = np.clip(recon * 255.0, 0, 255)

    recon_hwf = np.zeros((256, 256, n_total), dtype=np.float32)
    for fn in range(batch_size):
        recon_hwf[:, :, fn * 8:(fn + 1) * 8] = recon_255[fn].transpose(1, 2, 0)

    del mask_gpu, meas_gpu, img_out_ori, img_out
    torch.cuda.empty_cache()
    return recon_hwf, t_elapsed


# ══════════════════════════════════════════════════════════════════════════
# Solver 4: EfficientSCI (load model once)
# ══════════════════════════════════════════════════════════════════════════

def load_esci_model():
    _enter_esci()
    import torch
    import einops
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load mask
    esci_mask_path = os.path.join(ESCI_ROOT, "test_datasets", "mask", "efficientsci_mask.mat")
    esci_mask = sio.loadmat(esci_mask_path)["mask"].astype(np.float32)  # (256,256,8)
    esci_mask_t = esci_mask.transpose(2, 0, 1)  # (8,H,W)
    mask_s = np.sum(esci_mask_t, axis=0)
    mask_s[mask_s == 0] = 1

    # Build model
    from cacti.models.efficientsci import EfficientSCI
    model = EfficientSCI(in_ch=64, units=8, group_num=4, color_ch=1).to(device)
    ckpt_path = os.path.join(ESCI_ROOT, "checkpoints", "efficientsci_base.pth")
    resume_dict = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in resume_dict:
        model.load_state_dict(resume_dict["model_state_dict"])
    else:
        model.load_state_dict(resume_dict)
    model.eval()

    Phi = einops.repeat(esci_mask_t, "cr h w -> b cr h w", b=1)
    Phi_s = einops.repeat(mask_s, "h w -> b 1 h w", b=1)
    Phi = torch.from_numpy(Phi).float().to(device)
    Phi_s = torch.from_numpy(Phi_s).float().to(device)

    return model, esci_mask_t, Phi, Phi_s


def run_efficientsci(orig, esci_model, esci_mask_t, Phi, Phi_s):
    """Run EfficientSCI using a pre-loaded model."""
    _enter_esci()
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_total = orig.shape[2]
    n_coded = n_total // 8
    recon_hwf = np.zeros((256, 256, n_total), dtype=np.float32)

    t_total = 0
    for ci in range(n_coded):
        gt_block = orig[:, :, ci * 8:(ci + 1) * 8]
        gt_normed = gt_block / 255.0
        gt_t = gt_normed.transpose(2, 0, 1)
        meas_np = np.sum(gt_t * esci_mask_t, axis=0)
        meas_tensor = torch.from_numpy(meas_np).float().unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = esci_model(meas_tensor, Phi, Phi_s)
        torch.cuda.synchronize()
        t_total += time.time() - t0

        if not isinstance(outputs, list):
            outputs = [outputs]
        output = outputs[-1][0].cpu().numpy().astype(np.float32)
        recon_block = np.clip(output * 255.0, 0, 255)
        recon_hwf[:, :, ci * 8:(ci + 1) * 8] = recon_block.transpose(1, 2, 0)

    return recon_hwf, t_total


# ══════════════════════════════════════════════════════════════════════════
# W2: Mismatch + Correction (mask shift) — kobe32 only
# ══════════════════════════════════════════════════════════════════════════

def run_w2_mask_shift(meas, mask, orig, nframe):
    _enter_pnpsci()
    from pnp_sci import admmdenoise_cacti
    from utils import A_, At_

    shift_inject = 2
    mask_perturbed = np.roll(mask, shift_inject, axis=0)
    n_total = orig.shape[2]
    n_coded = n_total // 8
    nmask = mask.shape[2]

    y_pert = np.zeros((256, 256, n_coded), dtype=np.float32)
    for ci in range(n_coded):
        block = orig[:, :, ci * nmask:(ci + 1) * nmask]
        y_pert[:, :, ci] = np.sum(block * mask_perturbed, axis=2)

    y_pred_nominal = np.zeros_like(y_pert)
    for ci in range(n_coded):
        block = orig[:, :, ci * nmask:(ci + 1) * nmask]
        y_pred_nominal[:, :, ci] = np.sum(block * mask, axis=2)
    nll_before = compute_nll_gaussian(y_pert, y_pred_nominal, sigma=NOISE_SIGMA)

    A_nom = lambda x: A_(x, mask)
    At_nom = lambda y: At_(y, mask)
    v_uncorr, t_uncorr, psnr_uncorr, ssim_uncorr, _ = admmdenoise_cacti(
        y_pert, mask, A_nom, At_nom,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=nframe, MAXB=MAXB,
        _lambda=1, accelerate=True,
        denoiser="tv", iter_max=40, tv_weight=0.3, tv_iter_max=5,
    )

    print("    Grid search over vertical shifts [-5, +5] ...")
    best_nll, best_shift = np.inf, 0
    for trial_shift in range(-5, 6):
        mask_trial = np.roll(mask, trial_shift, axis=0)
        y_pred_trial = np.zeros_like(y_pert)
        for ci in range(n_coded):
            block = orig[:, :, ci * nmask:(ci + 1) * nmask]
            y_pred_trial[:, :, ci] = np.sum(block * mask_trial, axis=2)
        nll_trial = compute_nll_gaussian(y_pert, y_pred_trial, sigma=NOISE_SIGMA)
        if nll_trial < best_nll:
            best_nll = nll_trial
            best_shift = trial_shift

    print(f"    Best shift: {best_shift} pixels (NLL={best_nll:.1f})")
    mask_corrected = np.roll(mask, best_shift, axis=0)
    nll_after = best_nll
    nll_decrease_pct = (nll_before - nll_after) / (nll_before + 1e-12) * 100

    A_corr = lambda x: A_(x, mask_corrected)
    At_corr = lambda y: At_(y, mask_corrected)
    v_corr, t_corr, psnr_corr, ssim_corr, _ = admmdenoise_cacti(
        y_pert, mask_corrected, A_corr, At_corr,
        projmeth="gap", v0=None, orig=orig,
        iframe=0, nframe=nframe, MAXB=MAXB,
        _lambda=1, accelerate=True,
        denoiser="tv", iter_max=40, tv_weight=0.3, tv_iter_max=5,
    )

    return {
        "shift_injected": shift_inject,
        "shift_found": best_shift,
        "nll_before": round(nll_before, 1),
        "nll_after": round(nll_after, 1),
        "nll_decrease_pct": round(nll_decrease_pct, 1),
        "psnr_uncorrected": round(mean(psnr_uncorr), 2),
        "ssim_uncorrected": round(mean(ssim_uncorr), 4),
        "psnr_corrected": round(mean(psnr_corr), 2),
        "ssim_corrected": round(mean(ssim_corr), 4),
        "psnr_delta": round(mean(psnr_corr) - mean(psnr_uncorr), 2),
        "ssim_delta": round(mean(ssim_corr) - mean(ssim_uncorr), 4),
        "mask_nominal_hash": sha256_hex16(mask),
        "mask_corrected_hash": sha256_hex16(mask_corrected),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CACTI Real-Data Benchmark")
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                        help="Scenes to benchmark (default: all 6)")
    args = parser.parse_args()
    scenes = args.scenes

    results = {}
    print("=" * 70)
    print(f"CACTI Real-Data Benchmark — {len(scenes)} scene(s)")
    print("=" * 70)

    # ── Create RunBundle ─────────────────────────────────────────────────
    os.makedirs(RUNS_DIR, exist_ok=True)
    rb_dir = write_runbundle_skeleton(RUNS_DIR, "cacti_benchmark")
    rb_name = os.path.basename(rb_dir)
    art_dir = os.path.join(rb_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    print(f"RunBundle: {rb_name}")

    # ── Load deep models once ────────────────────────────────────────────
    print("\n[0] Loading deep models (one-time) ...")
    t0_load = time.time()

    print("    Loading FFDNet ...")
    ffdnet_model = load_ffdnet_model()

    print("    Loading ELP-Unfolding (6.4 GB checkpoint) ...")
    elp_model = load_elp_model()

    print("    Loading EfficientSCI ...")
    esci_model, esci_mask_t, Phi, Phi_s = load_esci_model()

    print(f"    All models loaded in {time.time() - t0_load:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # W1: 4-Solver Comparison — all scenes
    # ══════════════════════════════════════════════════════════════════════
    all_scene_results = {}

    for si, scene in enumerate(scenes):
        print(f"\n{'=' * 70}")
        print(f"[{si+1}/{len(scenes)}] Scene: {scene}")
        print("=" * 70)

        meas, mask, orig = load_scene(scene)
        n_total = orig.shape[2]
        n_coded = meas.shape[2]
        print(f"  meas: {meas.shape}, mask: {mask.shape}, orig: {orig.shape}")

        scene_dir = os.path.join(art_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)

        solver_results = {}

        # GAP-TV
        print(f"  [1/4] GAP-TV ...")
        v_tv, t_tv, _, _ = run_gap_tv(meas, mask, orig, n_coded)
        m_tv = compute_metrics_allframes(orig, v_tv)
        solver_results["gap_tv"] = {**m_tv, "wall_time": round(t_tv, 2)}
        save_array(os.path.join(scene_dir, "x_hat_gap_tv.npy"), v_tv)
        print(f"    PSNR={m_tv['mean_psnr']:.2f} dB, SSIM={m_tv['mean_ssim']:.4f}, time={t_tv:.1f}s")

        # PnP-FFDNet
        print(f"  [2/4] PnP-FFDNet ...")
        v_ffd, t_ffd, _, _ = run_pnp_ffdnet(meas, mask, orig, n_coded, ffdnet_model)
        m_ffd = compute_metrics_allframes(orig, v_ffd)
        solver_results["pnp_ffdnet"] = {**m_ffd, "wall_time": round(t_ffd, 2)}
        save_array(os.path.join(scene_dir, "x_hat_pnp_ffdnet.npy"), v_ffd)
        print(f"    PSNR={m_ffd['mean_psnr']:.2f} dB, SSIM={m_ffd['mean_ssim']:.4f}, time={t_ffd:.1f}s")

        # ELP-Unfolding
        print(f"  [3/4] ELP-Unfolding ...")
        v_elp, t_elp = run_elp_unfolding(orig, mask, elp_model)
        m_elp = compute_metrics_allframes(orig, v_elp)
        solver_results["elp_unfolding"] = {**m_elp, "wall_time": round(t_elp, 2)}
        save_array(os.path.join(scene_dir, "x_hat_elp_unfolding.npy"), v_elp)
        print(f"    PSNR={m_elp['mean_psnr']:.2f} dB, SSIM={m_elp['mean_ssim']:.4f}, time={t_elp:.1f}s")

        # EfficientSCI
        print(f"  [4/4] EfficientSCI ...")
        v_esci, t_esci = run_efficientsci(orig, esci_model, esci_mask_t, Phi, Phi_s)
        m_esci = compute_metrics_allframes(orig, v_esci)
        solver_results["efficientsci"] = {**m_esci, "wall_time": round(t_esci, 2)}
        save_array(os.path.join(scene_dir, "x_hat_efficientsci.npy"), v_esci)
        print(f"    PSNR={m_esci['mean_psnr']:.2f} dB, SSIM={m_esci['mean_ssim']:.4f}, time={t_esci:.1f}s")

        all_scene_results[scene] = solver_results

        # Per-scene summary
        print(f"\n  {'Solver':<20} {'PSNR':>8} {'SSIM':>8} {'Time':>8}")
        print(f"  {'-'*50}")
        for sname, sr in solver_results.items():
            print(f"  {sname:<20} {sr['mean_psnr']:>8.2f} {sr['mean_ssim']:>8.4f} {sr['wall_time']:>8.1f}")

    results["w1"] = all_scene_results

    # ── Compute average across all scenes ────────────────────────────────
    solver_names = ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficientsci"]
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
    # W2: Mask-shift mismatch + correction (kobe32 only)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("W2: Mask-shift mismatch + correction (kobe32, GAP-TV)")
    print("=" * 70)

    meas_k, mask_k, orig_k = load_scene("kobe32")
    n_coded_k = meas_k.shape[2]
    w2 = run_w2_mask_shift(meas_k, mask_k, orig_k, n_coded_k)

    w2_report = {k: v for k, v in w2.items()}
    w2_report["a_definition"] = "callable"
    w2_report["a_extraction_method"] = "provided"
    w2_report["linearity"] = "linear"
    w2_report["mismatch_type"] = "synthetic_injected"
    w2_report["mismatch_description"] = f"Mask-detector vertical shift: {w2['shift_injected']}px"
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

    # ── Trace (from kobe32) ──────────────────────────────────────────────
    trace = {}
    trace["00_input_x"] = orig_k[:, :, 0:8].copy()
    trace["01_masked"] = (orig_k[:, :, 0:8] * mask_k).astype(np.float32)
    trace["02_measurement"] = meas_k[:, :, 0:1].copy()
    v_tv_k, _, _, _ = run_gap_tv(meas_k, mask_k, orig_k, n_coded_k)
    trace["03_recon_gaptv"] = v_tv_k[:, :, 0:8].copy()
    save_trace(rb_dir, trace)

    results["trace"] = []
    for i, key in enumerate(sorted(trace.keys())):
        arr = trace[key]
        results["trace"].append({
            "stage": i, "node_id": key.split("_", 1)[1] if "_" in key else key,
            "output_shape": str(arr.shape), "dtype": str(arr.dtype),
            "range_min": round(float(arr.min()), 2),
            "range_max": round(float(arr.max()), 2),
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
    except ImportError: pass
    try:
        import torch; results["env"]["torch_version"] = torch.__version__
    except ImportError: pass

    results["rb_dir"] = rb_dir
    results["rb_name"] = rb_name
    results["scenes"] = scenes

    results_path = os.path.join(rb_dir, "cacti_benchmark_results.json")
    save_json(results_path, results)

    # ══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY — All Scenes")
    print("=" * 70)

    # Per-scene table for each solver
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
    # Average row
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

    print(f"\nW2 (kobe32): NLL decrease {w2['nll_decrease_pct']:.1f}%, PSNR delta {w2['psnr_delta']:+.2f} dB")
    print(f"\nRunBundle: runs/{rb_name}")
    print(f"Results:   {results_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
