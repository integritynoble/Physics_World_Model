"""PWM Flagship -- CACTI full-pipeline depth experiment.

Demonstrates the complete PWM pipeline on CACTI (Coded Aperture Compressive
Temporal Imaging) data:

1. **Design**      -- Video frame rate vs compression tradeoffs.
2. **Pre-flight**  -- Temporal compression feasibility prediction.
3. **Calibration** -- mask_shift mismatch family correction with uncertainty.
4. **Reconstruction** -- Oracle vs wrong vs calibrated operators.

All results output as RunBundles v0.3.0 with full provenance.

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.cacti_loop --out_dir results/flagship_cacti
    PYTHONPATH=. python -m experiments.pwm_flagship.cacti_loop --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from experiments.inversenet.manifest_schema import Severity
from experiments.inversenet.mismatch_sweep import apply_mismatch, get_delta_theta

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

SPATIAL_SIZE = (64, 64)
PWM_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"
BASE_SEED = 8000

FRAME_COUNTS = [4, 8, 16]
PHOTON_LEVELS = [1e3, 1e4, 1e5]
MISMATCH_FAMILY = "mask_shift"
SEVERITIES = [Severity.mild, Severity.moderate, Severity.severe]
N_CAL_FRAMES = 3
BOOTSTRAP_K = 20
GAP_TV_ITERS = 30
GAP_TV_ITERS_SMOKE = 10


# -- Helpers -----------------------------------------------------------------

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(arr: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(arr.tobytes()).hexdigest()


def _make_run_bundle(
    spec_id: str,
    metrics: Dict[str, float],
    artifacts: Dict[str, str],
    hashes: Dict[str, str],
    seeds: List[int],
) -> Dict[str, Any]:
    return {
        "version": BUNDLE_VERSION,
        "spec_id": spec_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "git_hash": _git_hash(),
            "seeds": seeds,
            "platform": platform.platform(),
            "pwm_version": PWM_VERSION,
        },
        "metrics": metrics,
        "artifacts": artifacts,
        "hashes": hashes,
    }


# -- Synthetic video GT ------------------------------------------------------

def _make_video_gt(
    H: int, W: int, T: int, rng: np.random.Generator
) -> np.ndarray:
    video = np.zeros((H, W, T), dtype=np.float32)
    base = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=5.0)
    base -= base.min()
    base /= base.max() + 1e-8
    for t in range(T):
        phase = 2.0 * np.pi * t / T
        shift_y = int(2 * np.sin(phase))
        shift_x = int(2 * np.cos(phase))
        frame = np.roll(np.roll(base, shift_y, axis=0), shift_x, axis=1)
        frame = frame * (0.8 + 0.2 * np.cos(phase))
        video[:, :, t] = frame
    return np.clip(video, 0, 1).astype(np.float32)


# -- CACTI forward model -----------------------------------------------------

def _generate_masks(H: int, W: int, T: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_mask = (rng.random((H, W)) > 0.5).astype(np.float32)
    masks = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        masks[:, :, t] = np.roll(base_mask, t, axis=0)
    return masks


def _cacti_forward(video: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.sum(video * masks, axis=2).astype(np.float32)


def _apply_photon_noise(
    y: np.ndarray, photon_level: float, rng: np.random.Generator
) -> np.ndarray:
    from pwm_core.noise.apply import apply_photon_noise
    return apply_photon_noise(y, photon_level, rng)


def _generate_calibration_captures(
    masks: np.ndarray, n_cal: int, rng: np.random.Generator
) -> np.ndarray:
    H, W, T = masks.shape
    cal = []
    for i in range(n_cal):
        level = (i + 1) / n_cal
        flat = np.full((H, W, T), level, dtype=np.float32)
        y_cal = _cacti_forward(flat, masks)
        cal.append(y_cal)
    return np.array(cal, dtype=np.float32)


# -- GAP-TV reconstruction for CACTI ----------------------------------------

def _cacti_gap_tv(
    y: np.ndarray,
    masks: np.ndarray,
    n_iter: int = 30,
) -> np.ndarray:
    H, W, T = masks.shape
    x = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        x[:, :, t] = y * masks[:, :, t] / max(T, 1)

    for _ in range(n_iter):
        yb = _cacti_forward(x, masks)
        r = y - yb
        for t in range(T):
            x[:, :, t] += r * masks[:, :, t] / max(T, 1)
        for t in range(T):
            x[:, :, t] = gaussian_filter(x[:, :, t], sigma=0.3)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


# -- Metrics -----------------------------------------------------------------

def _compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    max_val = max(float(x.max()), float(y.max()), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


def _compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x64, y64 = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
    mu_x, mu_y = x64.mean(), y64.mean()
    sig_x, sig_y = x64.std(), y64.std()
    sig_xy = float(np.mean((x64 - mu_x) * (y64 - mu_y)))
    return float(
        (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    )


# ============================================================================
# Stage 1: Design
# ============================================================================

def design_stage(smoke: bool = False) -> List[Dict[str, Any]]:
    """Video frame rate vs compression tradeoffs."""
    if smoke:
        return [{"n_frames": 4, "photon_level": 1e4}]
    variants = []
    for nf in FRAME_COUNTS:
        for photon in PHOTON_LEVELS:
            variants.append({"n_frames": nf, "photon_level": photon})
    logger.info("Design stage: %d CACTI variants proposed", len(variants))
    return variants


# ============================================================================
# Stage 2: Pre-flight
# ============================================================================

def preflight_stage(
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Temporal compression feasibility prediction."""
    results = []
    for v in variants:
        nf = v["n_frames"]
        photon = v["photon_level"]

        snr_db = 10 * np.log10(photon)
        cr = 1.0 / nf  # temporal compression ratio

        if cr >= 0.25:
            recoverability = "excellent"
            expected_psnr = 28.0
        elif cr >= 0.125:
            recoverability = "sufficient"
            expected_psnr = 24.0
        else:
            recoverability = "marginal"
            expected_psnr = 20.0

        if photon >= 1e4:
            quality_tier = "acceptable"
        elif photon >= 1e3:
            quality_tier = "marginal"
        else:
            quality_tier = "insufficient"

        result = dict(v)
        result.update({
            "temporal_cr": cr,
            "snr_db": float(snr_db),
            "quality_tier": quality_tier,
            "recoverability": recoverability,
            "expected_psnr_db": expected_psnr,
            "predicted_psnr_band": [expected_psnr - 3.0, expected_psnr + 3.0],
            "mismatch_identifiable": True,
            "proceed_recommended": quality_tier != "insufficient",
        })
        results.append(result)

    logger.info("Pre-flight: %d CACTI variants analyzed", len(results))
    return results


# ============================================================================
# Stage 3: Calibration -- mask_shift with bootstrap
# ============================================================================

def calibration_stage(
    y: np.ndarray,
    masks_true: np.ndarray,
    masks_mm: np.ndarray,
    y_cal: np.ndarray,
    severity: Severity,
    seed: int,
    bootstrap_k: int = 20,
    gap_tv_iters: int = 30,
    psnr_without: Optional[float] = None,
) -> Dict[str, Any]:
    """Calibrate mask_shift via search + bootstrap CI."""
    from pwm_core.mismatch.uncertainty import bootstrap_correction
    from pwm_core.mismatch.capture_advisor import suggest_next_capture

    delta = get_delta_theta("cacti", MISMATCH_FAMILY, severity)
    true_shift = delta.get("shift_px", 0)

    def correction_fn(data_subset: np.ndarray) -> Dict[str, float]:
        """Estimate mask shift from calibration captures."""
        n = data_subset.shape[0]
        H, W, T = masks_true.shape

        best_shift = 0
        best_score = float("inf")
        for dx in range(-6, 7):
            shifted = np.roll(masks_true, dx, axis=0)
            score = 0.0
            for i in range(n):
                level = (i + 1) / max(n, 1)
                flat = np.full((H, W, T), level, dtype=np.float32)
                y_pred = np.sum(flat * shifted, axis=2)
                y_obs = data_subset[i] if data_subset.ndim > 1 else data_subset
                score += float(np.sum((y_pred - y_obs) ** 2))
            if score < best_score:
                best_score = score
                best_shift = dx

        shift_err = abs(best_shift - true_shift)
        psnr_proxy = float(10 * np.log10(1.0 / max(shift_err + 0.1, 1e-6)))

        return {
            "shift_px": float(best_shift),
            "psnr": psnr_proxy,
        }

    # Run bootstrap
    cal_result = bootstrap_correction(
        correction_fn, y_cal, K=bootstrap_k, seed=seed,
        psnr_without_correction=psnr_without,
    )

    advice = suggest_next_capture(cal_result)

    est_shift = int(round(cal_result.theta_corrected.get("shift_px", 0.0)))
    masks_cal = np.roll(masks_true, -est_shift, axis=0)

    return {
        "correction_result": cal_result,
        "capture_advice": advice,
        "true_shift": true_shift,
        "est_shift": est_shift,
        "shift_ci": cal_result.theta_uncertainty.get("shift_px", [0.0, 0.0]),
        "improvement_db": cal_result.improvement_db,
        "masks_cal": masks_cal,
    }


# ============================================================================
# Stage 4: Reconstruction
# ============================================================================

def reconstruction_stage(
    x_gt: np.ndarray,
    y: np.ndarray,
    masks_true: np.ndarray,
    masks_wrong: np.ndarray,
    masks_cal: np.ndarray,
    gap_tv_iters: int = 30,
) -> Dict[str, Any]:
    """Reconstruct with oracle, wrong, and calibrated masks."""
    t0 = time.time()

    x_oracle = _cacti_gap_tv(y, masks_true, n_iter=gap_tv_iters)
    x_wrong = _cacti_gap_tv(y, masks_wrong, n_iter=gap_tv_iters)
    x_cal = _cacti_gap_tv(y, masks_cal, n_iter=gap_tv_iters)

    runtime = time.time() - t0

    # Compute metrics per-frame averaged
    psnr_o = _compute_psnr(x_gt, x_oracle)
    psnr_w = _compute_psnr(x_gt, x_wrong)
    psnr_c = _compute_psnr(x_gt, x_cal)

    return {
        "x_oracle": x_oracle,
        "x_wrong": x_wrong,
        "x_cal": x_cal,
        "psnr_oracle": psnr_o,
        "psnr_wrong": psnr_w,
        "psnr_cal": psnr_c,
        "ssim_oracle": _compute_ssim(x_gt.mean(axis=2), x_oracle.mean(axis=2)),
        "ssim_wrong": _compute_ssim(x_gt.mean(axis=2), x_wrong.mean(axis=2)),
        "ssim_cal": _compute_ssim(x_gt.mean(axis=2), x_cal.mean(axis=2)),
        "runtime_s": runtime,
    }


# ============================================================================
# Full CACTI loop
# ============================================================================

def run_cacti_loop(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Execute full CACTI pipeline: design -> preflight -> cal -> recon."""
    os.makedirs(out_dir, exist_ok=True)
    H, W = SPATIAL_SIZE

    variants = design_stage(smoke=smoke)
    preflight = preflight_stage(variants)
    active = [v for v in preflight if v["proceed_recommended"]]
    if not active:
        active = preflight[:1]

    severities = [Severity.mild] if smoke else SEVERITIES
    bootstrap_k = 5 if smoke else BOOTSTRAP_K
    gap_tv_iters = GAP_TV_ITERS_SMOKE if smoke else GAP_TV_ITERS

    all_results: List[Dict[str, Any]] = []

    for var_idx, variant in enumerate(active):
        nf = variant["n_frames"]
        photon = variant["photon_level"]

        for sev in severities:
            seed = BASE_SEED + var_idx * 100 + hash(sev.value) % 100
            rng = np.random.default_rng(seed)

            sid = (
                f"cacti_flagship_f{nf:02d}_p{photon:.0e}_"
                f"{MISMATCH_FAMILY}_{sev.value}_s{seed}"
            ).replace("+", "")

            logger.info("CACTI loop: %s", sid)

            x_gt = _make_video_gt(H, W, nf, rng)
            masks_true = _generate_masks(H, W, nf, seed)
            y_clean = _cacti_forward(x_gt, masks_true)
            y_noisy = _apply_photon_noise(y_clean, photon, rng)

            mm = apply_mismatch(
                "cacti", MISMATCH_FAMILY, sev, masks=masks_true, rng=rng,
            )
            masks_mm = mm.get("masks", masks_true)

            # Re-measure with mismatched masks
            y_mm = _cacti_forward(x_gt, masks_mm)
            y_mm = _apply_photon_noise(y_mm, photon, rng)

            y_cal = _generate_calibration_captures(masks_true, N_CAL_FRAMES, rng)

            # Quick pre-cal PSNR
            x_wrong_quick = _cacti_gap_tv(y_mm, masks_true, n_iter=gap_tv_iters // 2)
            psnr_wrong_quick = _compute_psnr(x_gt, x_wrong_quick)

            # Calibration
            cal = calibration_stage(
                y_mm, masks_true, masks_mm, y_cal, sev, seed,
                bootstrap_k=bootstrap_k,
                gap_tv_iters=gap_tv_iters,
                psnr_without=psnr_wrong_quick,
            )

            # Reconstruction
            recon = reconstruction_stage(
                x_gt, y_mm, masks_true, masks_mm, cal["masks_cal"],
                gap_tv_iters=gap_tv_iters,
            )

            # Save
            sample_dir = os.path.join(out_dir, sid)
            os.makedirs(sample_dir, exist_ok=True)

            np.save(os.path.join(sample_dir, "x_gt.npy"), x_gt)
            np.save(os.path.join(sample_dir, "x_cal.npy"), recon["x_cal"])

            metrics = {
                "psnr_oracle_db": recon["psnr_oracle"],
                "psnr_wrong_db": recon["psnr_wrong"],
                "psnr_cal_db": recon["psnr_cal"],
                "psnr_gain_db": recon["psnr_cal"] - recon["psnr_wrong"],
                "ssim_oracle": recon["ssim_oracle"],
                "ssim_wrong": recon["ssim_wrong"],
                "ssim_cal": recon["ssim_cal"],
                "runtime_s": recon["runtime_s"],
                "cal_improvement_db": cal["improvement_db"],
                "est_shift": cal["est_shift"],
                "true_shift": cal["true_shift"],
            }

            artifacts = {"x_gt": "x_gt.npy", "x_cal": "x_cal.npy"}
            hashes = {
                k: _sha256(np.load(os.path.join(sample_dir, v)))
                for k, v in artifacts.items()
            }

            bundle = _make_run_bundle(sid, metrics, artifacts, hashes, [seed])
            with open(os.path.join(sample_dir, "runbundle_manifest.json"), "w") as f:
                json.dump(bundle, f, indent=2)

            result = {
                "sample_id": sid,
                "variant": {k: v for k, v in variant.items() if k != "resolution"},
                "severity": sev.value,
                "calibration": {
                    "est_shift": cal["est_shift"],
                    "true_shift": cal["true_shift"],
                    "shift_ci": [float(v) for v in cal["shift_ci"]],
                    "improvement_db": cal["improvement_db"],
                    "capture_advice_summary": cal["capture_advice"].summary,
                },
                "metrics": metrics,
                "bundle": bundle,
            }
            all_results.append(result)

    summary_path = os.path.join(out_dir, "cacti_flagship_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("CACTI flagship: %d results -> %s", len(all_results), summary_path)
    return all_results


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: CACTI full-pipeline experiment"
    )
    parser.add_argument("--out_dir", default="results/flagship_cacti")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_cacti_loop(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
