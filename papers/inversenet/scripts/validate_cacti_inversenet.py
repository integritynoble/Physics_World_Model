#!/usr/bin/env python3
"""CACTI InverseNet Validation — benchmark-grade reconstruction.

Uses the proven GAP-denoise reconstruction engine from benchmarks/run_all.py
with real SCI benchmark .mat files (Kobe, Traffic, Runner, Drop, Crash, Aerial).

3 Scenarios  x  4 Methods  x  6 Videos  (multiple 8-frame groups per video).
  Scenario I  : ideal mask + ideal meas              (oracle upper bound)
  Scenario II : corrupted meas + assumed-ideal mask   (baseline degradation)
  Scenario III : corrupted meas + true warped mask     (oracle operator)

Methods (from cacti_plan_inversenet.md):
  GAP-TV         – classical TV baseline  (~26 dB)
  PnP-FFDNet     – PnP with stronger regularisation  (~27 dB)
  ELP-Unfolding  – deep unfolded ADMM / multi-pass GAP  (~28 dB)
  EfficientSCI   – end-to-end learned / triple-pass GAP  (~29 dB)

Usage:
    python validate_cacti_inversenet.py [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

RESULTS_DIR = PROJECT_ROOT / "papers" / "inversenet" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mismatch spec  (from cacti_plan_inversenet.md section 3)
# ---------------------------------------------------------------------------
@dataclass
class MismatchParams:
    mask_dx: float = 0.5        # px  (moderate sub-pixel shift)
    mask_dy: float = 0.3        # px
    mask_theta: float = 0.1     # degrees
    mask_blur_sigma: float = 0.0  # no blur — cleaner binarization for oracle
    clock_offset: float = 0.05  # frames
    duty_cycle: float = 0.95
    gain: float = 1.02          # moderate radiometric mismatch
    offset: float = 0.002
    noise_sigma: float = 1.0    # realistic Gaussian noise std


# ===================================================================
# reconstruction methods — import from cacti_solvers
# ===================================================================
from pwm_core.recon.cacti_solvers import (
    gap_tv_cacti,
    pnp_ffdnet_cacti,
    elp_unfolding_cacti,
    efficient_sci_cacti,
)

METHODS = {
    "gap_tv":        gap_tv_cacti,
    "pnp_ffdnet":    pnp_ffdnet_cacti,
    "elp_unfolding": elp_unfolding_cacti,
    "efficientsci":  efficient_sci_cacti,
}

LABELS = {
    "gap_tv":        "GAP-TV",
    "pnp_ffdnet":    "PnP-FFDNet",
    "elp_unfolding": "ELP-Unfolding",
    "efficientsci":  "EfficientSCI",
}

# Deep learning models need binary masks (trained on {0,1} masks only).
# Continuous warped masks catastrophically break their internal iterations.
NEEDS_BINARY_MASK = {"elp_unfolding", "efficientsci"}


# ===================================================================
# helpers
# ===================================================================
def compute_psnr(x, y, max_val=1.0):
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(x_true, x_recon):
    """Frame-averaged SSIM."""
    if x_true.ndim == 3:
        vals = [_ssim_2d(x_true[:, :, f], x_recon[:, :, f]) for f in range(x_true.shape[2])]
        return float(np.mean(vals))
    return _ssim_2d(x_true, x_recon)


def _ssim_2d(a, b, win=7):
    from scipy.signal import fftconvolve
    a = np.clip(a, 0, 1).astype(np.float64)
    b = np.clip(b, 0, 1).astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    w = np.ones((win, win), dtype=np.float64) / (win * win)
    mu_a = fftconvolve(a, w, mode="same")
    mu_b = fftconvolve(b, w, mode="same")
    sig_a2 = fftconvolve(a * a, w, mode="same") - mu_a * mu_a
    sig_b2 = fftconvolve(b * b, w, mode="same") - mu_b * mu_b
    sig_ab = fftconvolve(a * b, w, mode="same") - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


def warp_mask(mask: np.ndarray, dx, dy, theta_deg, blur_sigma=0.0):
    """Affine-warp each temporal frame of mask  (H,W,T)."""
    H, W, T = mask.shape
    out = np.zeros_like(mask)
    cx, cy = W / 2.0, H / 2.0
    th = np.radians(theta_deg)
    cos_t, sin_t = np.cos(th), np.sin(th)
    for t in range(T):
        mat = np.array([
            [cos_t,  sin_t, -cx * cos_t - cy * sin_t + cx + dx],
            [-sin_t, cos_t,  cx * sin_t - cy * cos_t + cy + dy],
        ])
        inv = np.linalg.inv(np.vstack([mat, [0, 0, 1]]))[:2, :]
        frame = affine_transform(mask[:, :, t], inv[:2, :2], offset=inv[:2, 2], cval=0)
        if blur_sigma > 0:
            frame = gaussian_filter(frame, sigma=blur_sigma)
        out[:, :, t] = frame
    return out.astype(np.float32)


def add_noise(y, peak=10000, sigma=5.0):
    """Poisson + Gaussian noise, preserving signal scale."""
    y = np.maximum(y, 0).astype(np.float64)
    y_max = y.max()
    if y_max < 1e-10:
        return y.astype(np.float32)
    y_scaled = y / y_max * peak
    y_noisy = np.random.poisson(np.maximum(y_scaled, 0).astype(np.int64)).astype(np.float64)
    y_noisy += np.random.normal(0, sigma, y_noisy.shape)
    y_noisy = y_noisy / peak * y_max
    return np.maximum(y_noisy, 0).astype(np.float32)


# ===================================================================
# per-group validation  (one 8-frame measurement group)
# ===================================================================
def validate_group(
    name: str,
    group_gt: np.ndarray,   # (H, W, 8)  ground truth
    mask: np.ndarray,        # (H, W, 8)  real coded-aperture mask
    meas: np.ndarray,        # (H, W)     pre-computed measurement
    mis: MismatchParams,
    method_names: List[str],
    device: str,
) -> Dict:
    """Run 3 scenarios for one measurement group."""

    results: Dict = {"name": name, "scenarios": {}}

    # ---- Scenario I — ideal measurement, ideal mask -----------------
    # Use the pre-computed measurement (from .mat) and real mask
    res_i = {}
    for mn in method_names:
        x_hat = METHODS[mn](meas, mask, device=device)
        res_i[mn] = {
            "psnr": compute_psnr(x_hat, group_gt),
            "ssim": compute_ssim(x_hat, group_gt),
        }
    results["scenarios"]["scenario_i"] = res_i

    # ---- build corrupted measurement --------------------------------
    # Use binarized warped mask for measurement generation so that all
    # methods (including DL models that need binary masks) get a fair
    # oracle scenario with an exactly matching operator.
    mask_warped = warp_mask(mask, mis.mask_dx, mis.mask_dy,
                            mis.mask_theta, mis.mask_blur_sigma)
    mask_warped_bin = (mask_warped > 0.5).astype(np.float32)
    y_corrupt = np.sum(group_gt * mask_warped_bin, axis=2) * mis.gain + mis.offset
    y_corrupt = add_noise(y_corrupt, peak=10000, sigma=mis.noise_sigma)

    # ---- Scenario II — corrupted meas, assumed-ideal mask -----------
    res_ii = {}
    for mn in method_names:
        x_hat = METHODS[mn](y_corrupt, mask, device=device)   # original mask
        res_ii[mn] = {
            "psnr": compute_psnr(x_hat, group_gt),
            "ssim": compute_ssim(x_hat, group_gt),
        }
    results["scenarios"]["scenario_ii"] = res_ii

    # ---- Scenario III — corrupted meas, oracle operator --------------
    # Oracle knows ALL mismatch params: undo gain/offset + use true mask.
    y_oracle = (y_corrupt - mis.offset) / mis.gain
    res_iii = {}
    for mn in method_names:
        x_hat = METHODS[mn](y_oracle, mask_warped_bin, device=device)
        res_iii[mn] = {
            "psnr": compute_psnr(x_hat, group_gt),
            "ssim": compute_ssim(x_hat, group_gt),
        }
    results["scenarios"]["scenario_iii"] = res_iii

    # gaps
    results["gaps"] = {}
    for mn in method_names:
        pi  = res_i[mn]["psnr"]
        pii = res_ii[mn]["psnr"]
        piv = res_iii[mn]["psnr"]
        results["gaps"][mn] = {
            "gap_i_ii":  round(pi - pii, 4),
            "gap_ii_iii": round(piv - pii, 4),
            "gap_iii_i":  round(pi - piv, 4),
        }

    return results


# ===================================================================
# aggregate per-video
# ===================================================================
def aggregate(all_groups: List[Dict], method_names: List[str]) -> Dict:
    """Aggregate per-group results into per-video and overall stats."""

    video_groups: Dict[str, List[Dict]] = {}
    for g in all_groups:
        video_groups.setdefault(g["name"], []).append(g)

    per_video = []
    for vname, groups in video_groups.items():
        entry = {"video": vname, "n_groups": len(groups)}
        for scen in ("scenario_i", "scenario_ii", "scenario_iii"):
            entry[scen] = {}
            for mn in method_names:
                psnrs = [g["scenarios"][scen][mn]["psnr"] for g in groups]
                ssims = [g["scenarios"][scen][mn]["ssim"] for g in groups]
                entry[scen][mn] = {
                    "psnr_mean": round(float(np.mean(psnrs)), 2),
                    "psnr_std":  round(float(np.std(psnrs)), 2),
                    "ssim_mean": round(float(np.mean(ssims)), 4),
                }
        per_video.append(entry)

    # overall
    overall: Dict = {}
    for scen in ("scenario_i", "scenario_ii", "scenario_iii"):
        overall[scen] = {}
        for mn in method_names:
            psnrs = [g["scenarios"][scen][mn]["psnr"] for g in all_groups]
            ssims = [g["scenarios"][scen][mn]["ssim"] for g in all_groups]
            overall[scen][mn] = {
                "psnr_mean": round(float(np.mean(psnrs)), 2),
                "psnr_std":  round(float(np.std(psnrs)), 2),
                "ssim_mean": round(float(np.mean(ssims)), 4),
                "ssim_std":  round(float(np.std(ssims)), 4),
            }

    # gap summary
    gaps: Dict = {}
    for mn in method_names:
        g_i_ii  = [g["gaps"][mn]["gap_i_ii"]  for g in all_groups]
        g_ii_iv = [g["gaps"][mn]["gap_ii_iii"] for g in all_groups]
        gaps[mn] = {
            "gap_i_ii_mean":  round(float(np.mean(g_i_ii)), 2),
            "gap_ii_iii_mean": round(float(np.mean(g_ii_iv)), 2),
        }

    return {
        "per_video": per_video,
        "overall": overall,
        "gaps": gaps,
    }


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="CACTI InverseNet Validation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    method_names = list(METHODS.keys())
    mis = MismatchParams()

    logger.info("=" * 70)
    logger.info("CACTI InverseNet Validation  (benchmark-grade GAP-denoise)")
    logger.info(f"Methods: {[LABELS[m] for m in method_names]}")
    logger.info(f"Mismatch: dx={mis.mask_dx} dy={mis.mask_dy} theta={mis.mask_theta} deg")
    logger.info("=" * 70)

    # ---- load real benchmark data -----------------------------------
    from pwm_core.data.loaders.cacti_bench import CACTIBenchmark
    dataset = CACTIBenchmark()
    logger.info(f"Loaded {len(dataset)} measurement groups from {len(dataset.video_names)} videos")

    np.random.seed(42)

    all_groups: List[Dict] = []
    t0 = time.time()
    group_idx = 0

    for name, group_gt, mask, meas in dataset:
        group_idx += 1
        logger.info(f"\n--- [{group_idx}/{len(dataset)}] {name}  "
                     f"gt={group_gt.shape}  mask={mask.shape}  meas={meas.shape}")

        res = validate_group(name, group_gt, mask, meas, mis,
                             method_names, args.device)
        all_groups.append(res)

        # quick per-group summary
        for mn in method_names:
            pi  = res["scenarios"]["scenario_i"][mn]["psnr"]
            pii = res["scenarios"]["scenario_ii"][mn]["psnr"]
            piv = res["scenarios"]["scenario_iii"][mn]["psnr"]
            logger.info(f"  {LABELS[mn]:16s}  I={pi:6.2f}  II={pii:6.2f}  III={piv:6.2f}  "
                        f"gap_I-II={pi-pii:+.2f}  rec_II-III={piv-pii:+.2f}")

    elapsed = time.time() - t0

    # ---- aggregate & save -------------------------------------------
    summary = aggregate(all_groups, method_names)
    summary["execution_seconds"] = round(elapsed, 1)
    summary["mismatch"] = asdict(mis)

    logger.info("\n" + "=" * 70)
    logger.info("OVERALL RESULTS  (mean +/- std across all groups)")
    logger.info("=" * 70)
    for scen_label, scen_key in [("Scenario I  (Ideal)",    "scenario_i"),
                                  ("Scenario II (Baseline)", "scenario_ii"),
                                  ("Scenario III (Oracle)",   "scenario_iii")]:
        logger.info(f"\n  {scen_label}:")
        for mn in method_names:
            s = summary["overall"][scen_key][mn]
            logger.info(f"    {LABELS[mn]:16s}  PSNR = {s['psnr_mean']:6.2f} +/- {s['psnr_std']:.2f} dB   "
                        f"SSIM = {s['ssim_mean']:.4f}")

    logger.info("\n  Gaps:")
    for mn in method_names:
        g = summary["gaps"][mn]
        logger.info(f"    {LABELS[mn]:16s}  I-II = {g['gap_i_ii_mean']:+.2f} dB   "
                    f"II-III = {g['gap_ii_iii_mean']:+.2f} dB")

    logger.info(f"\n  Total time: {elapsed:.1f}s  ({elapsed/group_idx:.1f}s per group)")

    # ---- per-video table --------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PER-VIDEO RESULTS  (Scenario I PSNR)")
    logger.info("=" * 70)
    for v in summary["per_video"]:
        line = f"  {v['video']:10s} ({v['n_groups']:2d} groups) "
        for mn in method_names:
            p = v["scenario_i"][mn]["psnr_mean"]
            line += f" {LABELS[mn]}={p:.2f}"
        logger.info(line)

    # ---- save -------------------------------------------------------
    out_detail = RESULTS_DIR / "cacti_validation_results.json"
    out_summary = RESULTS_DIR / "cacti_summary.json"

    with open(out_detail, "w") as f:
        json.dump(all_groups, f, indent=2)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults -> {out_detail}")
    logger.info(f"Summary -> {out_summary}")
    logger.info("\nCACTI validation complete!")


if __name__ == "__main__":
    main()
