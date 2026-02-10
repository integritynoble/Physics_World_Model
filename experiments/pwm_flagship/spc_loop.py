"""PWM Flagship -- SPC full-pipeline depth experiment.

Demonstrates the complete PWM pipeline on Single-Pixel Camera data:

1. **Design**      -- Propose system variants under constraints.
2. **Pre-flight**  -- Photon x Recoverability x Mismatch x Solver Fit
                      predicts success / failure bands.
3. **Calibration** -- Correct gain mismatch family with bootstrap CI
                      + next-capture suggestions.
4. **Reconstruction** -- Oracle vs wrong vs calibrated operators.
                      Record PSNR / SSIM / runtime.

All results output as RunBundles v0.3.0 with full provenance.

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.spc_loop --out_dir results/flagship_spc
    PYTHONPATH=. python -m experiments.pwm_flagship.spc_loop --smoke
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from experiments.inversenet.manifest_schema import Severity
from experiments.inversenet.mismatch_sweep import apply_mismatch, get_delta_theta

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

IMAGE_SIZE = (64, 64)
PWM_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"
BASE_SEED = 7000

# Design sweep axes
COMPRESSION_RATIOS = [0.10, 0.25, 0.50]
PHOTON_LEVELS = [1e3, 1e4, 1e5]
MISMATCH_FAMILY = "gain"
SEVERITIES = [Severity.mild, Severity.moderate, Severity.severe]
N_CAL_PATTERNS = 5
BOOTSTRAP_K = 20


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


# -- Synthetic ground truth --------------------------------------------------

def _make_ground_truth(rng: np.random.Generator) -> np.ndarray:
    raw = rng.random(IMAGE_SIZE).astype(np.float32)
    smooth = gaussian_filter(raw, sigma=5.0)
    smooth -= smooth.min()
    smooth /= smooth.max() + 1e-8
    return smooth.astype(np.float32)


# -- SPC forward model -------------------------------------------------------

def _build_measurement_matrix(cr: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    M = max(1, int(N * cr))
    A = (rng.random((M, N)) > 0.5).astype(np.float32) * 2 - 1
    A /= np.sqrt(N)
    return A


def _apply_photon_noise(
    y: np.ndarray, photon_level: float, rng: np.random.Generator
) -> np.ndarray:
    scale = photon_level / (np.abs(y).max() + 1e-10)
    y_scaled = np.maximum(y * scale, 0)
    y_noisy = rng.poisson(y_scaled).astype(np.float32)
    read_sigma = np.sqrt(photon_level) * 0.01
    y_noisy += rng.normal(0, read_sigma, size=y.shape).astype(np.float32)
    y_noisy /= scale
    return y_noisy


def _generate_calibration_captures(
    A: np.ndarray, n_patterns: int, rng: np.random.Generator
) -> np.ndarray:
    N = A.shape[1]
    cal = []
    for i in range(n_patterns):
        level = (i + 1) / n_patterns
        x_cal = np.full(N, level, dtype=np.float32)
        y_cal = A @ x_cal
        cal.append(y_cal)
    return np.array(cal, dtype=np.float32)


# -- Simple ISTA reconstruction ----------------------------------------------

def _ista_recon(
    y: np.ndarray, A: np.ndarray, n_iter: int = 50, lam: float = 0.01
) -> np.ndarray:
    """Simple ISTA reconstruction for SPC."""
    N = A.shape[1]
    H, W = IMAGE_SIZE
    x = np.zeros(N, dtype=np.float64)
    L = float(np.linalg.norm(A.T @ A, ord=2))
    if L < 1e-10:
        L = 1.0
    step = 1.0 / L
    for _ in range(n_iter):
        grad = A.T @ (A @ x - y.astype(np.float64))
        x = x - step * grad
        # Soft threshold
        x = np.sign(x) * np.maximum(np.abs(x) - lam * step, 0)
    return np.clip(x.reshape(H, W), 0, 1).astype(np.float32)


# -- Metric helpers -----------------------------------------------------------

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
# Stage 1: Design -- propose system variants
# ============================================================================

def design_stage(
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Propose system variants under photon/compression/resolution constraints."""
    if smoke:
        variants = [{"cr": 0.25, "photon_level": 1e4, "resolution": IMAGE_SIZE}]
    else:
        variants = []
        for cr in COMPRESSION_RATIOS:
            for photon in PHOTON_LEVELS:
                variants.append({
                    "cr": cr,
                    "photon_level": photon,
                    "resolution": IMAGE_SIZE,
                })
    logger.info("Design stage: %d system variants proposed", len(variants))
    return variants


# ============================================================================
# Stage 2: Pre-flight -- predict success/failure bands
# ============================================================================

def preflight_stage(
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Photon x Recoverability x Mismatch analysis for each variant.

    Returns each variant annotated with pre-flight predictions.
    """
    results = []
    for v in variants:
        cr = v["cr"]
        photon = v["photon_level"]

        # Photon budget -> SNR regime
        snr_db = 10 * np.log10(photon)
        if photon >= 1e5:
            quality_tier = "excellent"
            noise_regime = "shot_limited"
        elif photon >= 1e4:
            quality_tier = "acceptable"
            noise_regime = "shot_limited"
        elif photon >= 1e3:
            quality_tier = "marginal"
            noise_regime = "photon_starved"
        else:
            quality_tier = "insufficient"
            noise_regime = "photon_starved"

        # Recoverability from CR
        if cr >= 0.50:
            recoverability = "excellent"
            expected_psnr = 28.0
        elif cr >= 0.25:
            recoverability = "sufficient"
            expected_psnr = 24.0
        else:
            recoverability = "marginal"
            expected_psnr = 20.0

        # Mismatch identifiability prediction
        # gain family is identifiable from flat-field calibration
        mismatch_identifiable = True
        mismatch_correction_method = "flat_field_gain_estimation"

        # Predicted success band
        predicted_psnr_low = expected_psnr - 3.0
        predicted_psnr_high = expected_psnr + 3.0

        # Solver fitness
        solver_fit = "ISTA" if cr <= 0.25 else "ISTA_warm"

        result = dict(v)
        result.update({
            "snr_db": float(snr_db),
            "quality_tier": quality_tier,
            "noise_regime": noise_regime,
            "recoverability": recoverability,
            "expected_psnr_db": expected_psnr,
            "predicted_psnr_band": [predicted_psnr_low, predicted_psnr_high],
            "mismatch_identifiable": mismatch_identifiable,
            "mismatch_correction_method": mismatch_correction_method,
            "solver_fit": solver_fit,
            "proceed_recommended": quality_tier != "insufficient",
        })
        results.append(result)

    logger.info("Pre-flight: %d variants analyzed", len(results))
    return results


# ============================================================================
# Stage 3: Calibration -- correct gain mismatch with bootstrap CI
# ============================================================================

def calibration_stage(
    y: np.ndarray,
    A: np.ndarray,
    A_mm: np.ndarray,
    y_cal: np.ndarray,
    severity: Severity,
    seed: int,
    bootstrap_k: int = 20,
    psnr_without: Optional[float] = None,
) -> Dict[str, Any]:
    """Calibrate gain mismatch using flat-field calibration captures.

    Uses bootstrap_correction() for uncertainty bands and
    suggest_next_capture() for capture advisor suggestions.
    """
    from pwm_core.mismatch.uncertainty import bootstrap_correction
    from pwm_core.mismatch.capture_advisor import suggest_next_capture

    delta = get_delta_theta("spc", MISMATCH_FAMILY, severity)
    true_gain = delta.get("gain_factor", 1.0)
    true_bias = delta.get("bias", 0.0)

    # Define correction function: estimate gain from calibration data
    def correction_fn(data_subset: np.ndarray) -> Dict[str, float]:
        """Estimate gain and bias from calibration captures."""
        # data_subset: (n_cal, M) -- each row is a calibration measurement
        n_cal = data_subset.shape[0]
        M = data_subset.shape[1] if data_subset.ndim > 1 else data_subset.shape[0]

        # Expected calibration response (from true operator, known patterns)
        N = A.shape[1]
        expected = []
        for i in range(n_cal):
            level = (i + 1) / max(n_cal, 1)
            x_cal = np.full(N, level, dtype=np.float32)
            expected.append(A @ x_cal)
        expected = np.array(expected, dtype=np.float64)

        # Estimate gain and bias via least-squares: y_obs = gain * y_expected + bias
        if expected.size > 0 and data_subset.size > 0:
            e_flat = expected.ravel()
            d_flat = data_subset.ravel()[:e_flat.size]
            if len(e_flat) > 1:
                A_ls = np.column_stack([e_flat, np.ones_like(e_flat)])
                result_ls, _, _, _ = np.linalg.lstsq(A_ls, d_flat, rcond=None)
                est_gain = float(result_ls[0])
                est_bias = float(result_ls[1])
            else:
                est_gain = 1.0
                est_bias = 0.0
        else:
            est_gain = 1.0
            est_bias = 0.0

        # Compute PSNR proxy (inverse of estimation error)
        gain_err = abs(est_gain - true_gain)
        bias_err = abs(est_bias - true_bias)
        # Proxy: lower error => higher PSNR
        err = max(gain_err + bias_err, 1e-6)
        psnr_proxy = float(10 * np.log10(1.0 / err))

        return {
            "gain": est_gain,
            "bias": est_bias,
            "psnr": psnr_proxy,
        }

    # Run bootstrap correction
    cal_result = bootstrap_correction(
        correction_fn,
        y_cal,
        K=bootstrap_k,
        seed=seed,
        psnr_without_correction=psnr_without,
    )

    # Capture advisor
    advice = suggest_next_capture(cal_result)

    return {
        "correction_result": cal_result,
        "capture_advice": advice,
        "true_gain": true_gain,
        "true_bias": true_bias,
        "est_gain": cal_result.theta_corrected.get("gain", 1.0),
        "est_bias": cal_result.theta_corrected.get("bias", 0.0),
        "gain_ci": cal_result.theta_uncertainty.get("gain", [1.0, 1.0]),
        "bias_ci": cal_result.theta_uncertainty.get("bias", [0.0, 0.0]),
        "improvement_db": cal_result.improvement_db,
    }


# ============================================================================
# Stage 4: Reconstruction -- oracle vs wrong vs calibrated
# ============================================================================

def reconstruction_stage(
    x_gt: np.ndarray,
    y: np.ndarray,
    A_true: np.ndarray,
    A_wrong: np.ndarray,
    cal_result: Dict[str, Any],
    n_iter: int = 50,
) -> Dict[str, Any]:
    """Reconstruct using oracle, wrong (nominal), and calibrated operators."""
    t0 = time.time()

    # Calibrated operator: undo gain/bias from measurement
    est_gain = cal_result.get("est_gain", 1.0)
    est_bias = cal_result.get("est_bias", 0.0)
    y_corrected = (y - est_bias) / max(est_gain, 1e-6)

    # Oracle reconstruction (true operator, clean-ish measurement)
    x_oracle = _ista_recon(y, A_true, n_iter=n_iter)

    # Wrong reconstruction (nominal operator, mismatched measurement)
    x_wrong = _ista_recon(y, A_wrong, n_iter=n_iter)

    # Calibrated reconstruction (nominal operator, corrected measurement)
    x_cal = _ista_recon(y_corrected, A_wrong, n_iter=n_iter)

    runtime = time.time() - t0

    return {
        "x_oracle": x_oracle,
        "x_wrong": x_wrong,
        "x_cal": x_cal,
        "psnr_oracle": _compute_psnr(x_gt, x_oracle),
        "psnr_wrong": _compute_psnr(x_gt, x_wrong),
        "psnr_cal": _compute_psnr(x_gt, x_cal),
        "ssim_oracle": _compute_ssim(x_gt, x_oracle),
        "ssim_wrong": _compute_ssim(x_gt, x_wrong),
        "ssim_cal": _compute_ssim(x_gt, x_cal),
        "runtime_s": runtime,
    }


# ============================================================================
# Full SPC loop
# ============================================================================

def run_spc_loop(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the full SPC pipeline: design -> preflight -> cal -> recon."""
    os.makedirs(out_dir, exist_ok=True)

    # Stage 1: Design
    variants = design_stage(smoke=smoke)

    # Stage 2: Pre-flight
    preflight = preflight_stage(variants)

    # Filter to recommended variants
    active = [v for v in preflight if v["proceed_recommended"]]
    if not active:
        active = preflight[:1]

    severities = [Severity.mild] if smoke else SEVERITIES
    bootstrap_k = 5 if smoke else BOOTSTRAP_K
    recon_iters = 20 if smoke else 50

    all_results: List[Dict[str, Any]] = []

    for var_idx, variant in enumerate(active):
        cr = variant["cr"]
        photon = variant["photon_level"]

        for sev in severities:
            seed = BASE_SEED + var_idx * 100 + hash(sev.value) % 100
            rng = np.random.default_rng(seed)

            sid = (
                f"spc_flagship_cr{int(cr * 100):03d}_p{photon:.0e}_"
                f"{MISMATCH_FAMILY}_{sev.value}_s{seed}"
            ).replace("+", "")

            logger.info("SPC loop: %s", sid)

            # Generate data
            x_gt = _make_ground_truth(rng)
            A_true = _build_measurement_matrix(cr, seed)
            y_clean = A_true @ x_gt.flatten()
            y_noisy = _apply_photon_noise(y_clean, photon, rng)

            # Apply gain mismatch
            mm = apply_mismatch(
                "spc", MISMATCH_FAMILY, sev, y=y_noisy, mask=A_true, rng=rng,
            )
            y_mm = mm.get("y", y_noisy)
            A_mm = mm.get("mask", A_true)

            # Calibration captures (from true operator)
            y_cal = _generate_calibration_captures(A_true, N_CAL_PATTERNS, rng)

            # Also produce mismatched calibration captures for bootstrap
            delta = get_delta_theta("spc", MISMATCH_FAMILY, sev)
            gain_factor = delta.get("gain_factor", 1.0)
            bias_val = delta.get("bias", 0.0)
            y_cal_mm = y_cal * gain_factor + bias_val

            # Stage 3: Pre-reconstruction PSNR (wrong operator)
            x_wrong_quick = _ista_recon(y_mm, A_true, n_iter=recon_iters // 2)
            psnr_wrong_quick = _compute_psnr(x_gt, x_wrong_quick)

            # Stage 3: Calibration
            cal = calibration_stage(
                y_mm, A_true, A_mm, y_cal_mm, sev, seed,
                bootstrap_k=bootstrap_k,
                psnr_without=psnr_wrong_quick,
            )

            # Stage 4: Reconstruction
            recon = reconstruction_stage(
                x_gt, y_mm, A_true, A_true, cal, n_iter=recon_iters,
            )

            # Save results
            sample_dir = os.path.join(out_dir, sid)
            os.makedirs(sample_dir, exist_ok=True)

            np.save(os.path.join(sample_dir, "x_gt.npy"), x_gt)
            np.save(os.path.join(sample_dir, "x_oracle.npy"), recon["x_oracle"])
            np.save(os.path.join(sample_dir, "x_wrong.npy"), recon["x_wrong"])
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
                "est_gain": cal["est_gain"],
                "est_bias": cal["est_bias"],
            }

            artifacts = {
                "x_gt": "x_gt.npy",
                "x_oracle": "x_oracle.npy",
                "x_wrong": "x_wrong.npy",
                "x_cal": "x_cal.npy",
            }
            hashes = {
                k: _sha256(np.load(os.path.join(sample_dir, v)))
                for k, v in artifacts.items()
            }

            bundle = _make_run_bundle(
                spec_id=sid,
                metrics=metrics,
                artifacts=artifacts,
                hashes=hashes,
                seeds=[seed],
            )

            with open(os.path.join(sample_dir, "runbundle_manifest.json"), "w") as f:
                json.dump(bundle, f, indent=2)

            result = {
                "sample_id": sid,
                "variant": variant,
                "severity": sev.value,
                "preflight": {
                    k: v for k, v in variant.items()
                    if k not in ("resolution",)
                },
                "calibration": {
                    "est_gain": cal["est_gain"],
                    "est_bias": cal["est_bias"],
                    "gain_ci": [float(v) for v in cal["gain_ci"]],
                    "bias_ci": [float(v) for v in cal["bias_ci"]],
                    "improvement_db": cal["improvement_db"],
                    "capture_advice_summary": cal["capture_advice"].summary,
                },
                "metrics": metrics,
                "bundle": bundle,
            }
            all_results.append(result)

    # Save combined summary
    summary_path = os.path.join(out_dir, "spc_flagship_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("SPC flagship: %d results -> %s", len(all_results), summary_path)
    return all_results


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: SPC full-pipeline experiment"
    )
    parser.add_argument("--out_dir", default="results/flagship_spc")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick validation run (1 variant, 1 severity)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_spc_loop(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
