"""Sweep calibration budget (number of captures) per mismatch family.

For each budget level x mismatch family:
- Measure theta-error vs calibration budget
- Measure uncertainty band width vs calibration budget
- Show "next capture" advisor value: uncertainty reduction after each capture

Usage::

    python -m experiments.pwmi_cassi.cal_budget --out_dir results/pwmi_cassi_budget
    python -m experiments.pwmi_cassi.cal_budget --smoke

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

from experiments.inversenet.manifest_schema import Severity
from experiments.inversenet.mismatch_sweep import apply_mismatch, get_delta_theta
from experiments.inversenet.gen_cassi import (
    _cassi_forward,
    _default_cassi_theta,
    _generate_coded_aperture,
    _apply_photon_noise,
    _make_hsi_gt,
    SPATIAL_SIZE,
)
from experiments.pwmi_cassi.run_families import (
    _cassi_gap_tv,
    _upwmi_calibrate_cassi,
    _theta_rmse_vec,
    _compute_psnr,
    _git_hash,
    _make_run_bundle,
    MISMATCH_FAMILIES,
    N_BANDS,
    PHOTON_LEVEL,
    GAP_TV_ITERS,
    GAP_TV_ITERS_SMOKE,
    PWM_VERSION,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

BUDGET_LEVELS = [1, 3, 5, 10]
BUDGET_LEVELS_SMOKE = [1, 3]
BASE_SEED = 7000
N_TRIALS = 3
BOOTSTRAP_K = 15


# ── Calibration capture generator ────────────────────────────────────────

def _generate_cal_captures_variable(
    mask: np.ndarray,
    theta: Dict[str, Any],
    n_bands: int,
    n_captures: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n_captures calibration captures (narrowband flat-field).

    Each capture isolates a different spectral band, spread uniformly
    across the spectral range.
    """
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift

    H, W = mask.shape
    cal = []
    selected_bands = np.linspace(0, n_bands - 1, min(n_captures, n_bands), dtype=int)
    # If n_captures > n_bands, repeat with random noise variation
    while len(cal) < n_captures:
        for b in selected_bands:
            if len(cal) >= n_captures:
                break
            cube_cal = np.zeros((H, W, n_bands), dtype=np.float32)
            cube_cal[:, :, b] = 1.0
            y_cal = _cassi_forward(cube_cal, mask, theta)
            # Add small noise for diversity
            y_cal += rng.normal(0, 0.01, size=y_cal.shape).astype(np.float32)
            cal.append(y_cal)
    return np.array(cal[:n_captures], dtype=np.float32)


# ── Single budget trial ─────────────────────────────────────────────────

def run_budget_trial(
    family: str,
    severity: Severity,
    n_captures: int,
    seed: int,
    gap_tv_iters: int,
) -> Dict[str, Any]:
    """Run one calibration trial with a given capture budget."""
    rng = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    x_gt = _make_hsi_gt(H, W, N_BANDS, rng)
    mask = _generate_coded_aperture(H, W, seed)
    theta_true = _default_cassi_theta(N_BANDS)

    # Apply mismatch
    mm = apply_mismatch(
        "cassi", family, severity,
        y=None, mask=mask, theta=theta_true, rng=rng,
    )
    theta_mm = mm.get("theta", theta_true)
    mask_mm = mm.get("mask", mask)

    y = _cassi_forward(x_gt, mask_mm, theta_mm)
    y = _apply_photon_noise(y, PHOTON_LEVEL, rng)

    # Generate calibration captures with the given budget
    y_cal = _generate_cal_captures_variable(
        mask, theta_true, N_BANDS, n_captures, rng,
    )

    # UPWMI calibration (family-aware)
    theta_est, runtime = _upwmi_calibrate_cassi(
        y, mask, theta_true, N_BANDS, y_cal,
        iters=gap_tv_iters, rng=rng, family=family,
    )

    theta_error = _theta_rmse_vec(theta_mm, theta_est)

    # Reconstruct
    x_cal = _cassi_gap_tv(y, mask, theta_est, N_BANDS, iters=gap_tv_iters)
    psnr_cal = _compute_psnr(x_gt, x_cal)

    return {
        "family": family,
        "severity": severity.value,
        "n_captures": n_captures,
        "seed": seed,
        "theta_error_rmse": theta_error,
        "psnr_cal_db": psnr_cal,
        "runtime_s": runtime,
    }


# ── Budget sweep for one family ─────────────────────────────────────────

def sweep_budget_family(
    family: str,
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Sweep calibration budget for one mismatch family.

    Returns list of result dicts, one per (budget_level, severity).
    """
    from pwm_core.mismatch.uncertainty import bootstrap_correction
    from pwm_core.mismatch.capture_advisor import suggest_next_capture

    budget_levels = BUDGET_LEVELS_SMOKE if smoke else BUDGET_LEVELS
    severities = [Severity.moderate] if smoke else [Severity.mild, Severity.moderate, Severity.severe]
    n_trials = 1 if smoke else N_TRIALS
    gap_tv_iters = GAP_TV_ITERS_SMOKE if smoke else GAP_TV_ITERS
    bootstrap_k = 3 if smoke else BOOTSTRAP_K

    results: List[Dict[str, Any]] = []

    for sev in severities:
        budget_curve: List[Dict[str, Any]] = []

        for n_cap in budget_levels:
            logger.info(
                f"  Budget sweep: family={family}, sev={sev.value}, "
                f"n_captures={n_cap}"
            )

            trial_results = []
            for trial_idx in range(n_trials):
                seed = BASE_SEED + hash((family, sev.value, n_cap, trial_idx)) % 10000
                trial = run_budget_trial(
                    family, sev, n_cap, seed, gap_tv_iters,
                )
                trial_results.append(trial)

            theta_errors = [t["theta_error_rmse"] for t in trial_results]
            psnr_vals = [t["psnr_cal_db"] for t in trial_results]
            runtimes = [t["runtime_s"] for t in trial_results]

            # Bootstrap CI on theta error
            err_data = np.array(theta_errors, dtype=np.float64).reshape(-1, 1)
            if err_data.shape[0] >= 2:
                def correction_fn(data_subset: np.ndarray) -> Dict[str, float]:
                    vals = data_subset.ravel()
                    return {
                        "theta_error": float(np.mean(vals)),
                        "psnr": float(np.mean(psnr_vals)),
                    }

                ci_result = bootstrap_correction(
                    correction_fn, err_data, K=bootstrap_k, seed=42,
                )
                ci_width = (
                    ci_result.theta_uncertainty["theta_error"][1]
                    - ci_result.theta_uncertainty["theta_error"][0]
                )
                ci_band = ci_result.theta_uncertainty["theta_error"]
            else:
                ci_width = 0.0
                ci_band = [theta_errors[0], theta_errors[0]]

            # Capture advisor: simulate advice based on current CI
            advisor_result = suggest_next_capture(
                {
                    "theta_corrected": {"disp_step": float(np.mean(theta_errors))},
                    "theta_uncertainty": {"disp_step": ci_band},
                },
            )

            budget_point = {
                "n_captures": n_cap,
                "theta_error_mean": float(np.mean(theta_errors)),
                "theta_error_std": float(np.std(theta_errors)),
                "theta_error_ci": [float(ci_band[0]), float(ci_band[1])],
                "ci_width": float(ci_width),
                "psnr_mean": float(np.mean(psnr_vals)),
                "runtime_mean": float(np.mean(runtimes)),
                "advisor_n_underdetermined": advisor_result.n_underdetermined,
                "advisor_all_constrained": advisor_result.all_parameters_constrained,
                "advisor_summary": advisor_result.summary,
            }
            budget_curve.append(budget_point)

        # Compute uncertainty reduction between budget levels
        for i in range(1, len(budget_curve)):
            prev_width = budget_curve[i - 1]["ci_width"]
            curr_width = budget_curve[i]["ci_width"]
            if prev_width > 0:
                reduction_pct = (prev_width - curr_width) / prev_width * 100
            else:
                reduction_pct = 0.0
            budget_curve[i]["uncertainty_reduction_pct"] = float(reduction_pct)
        if budget_curve:
            budget_curve[0]["uncertainty_reduction_pct"] = 0.0

        spec_id = f"pwmi_cassi_budget_{family}_{sev.value}"
        bundle = _make_run_bundle(
            spec_id=spec_id,
            metrics={
                "budget_levels": budget_levels,
                "theta_error_at_max_budget": budget_curve[-1]["theta_error_mean"],
                "ci_width_at_max_budget": budget_curve[-1]["ci_width"],
            },
            artifacts={"budget_curve": f"{spec_id}_curve.json"},
            hashes={
                "budget_curve": "sha256:" + hashlib.sha256(
                    json.dumps(budget_curve, sort_keys=True).encode()
                ).hexdigest()
            },
            seeds=[BASE_SEED],
        )

        sev_dir = os.path.join(out_dir, spec_id)
        os.makedirs(sev_dir, exist_ok=True)

        curve_path = os.path.join(sev_dir, f"{spec_id}_curve.json")
        with open(curve_path, "w") as f:
            json.dump(budget_curve, f, indent=2)

        bundle_path = os.path.join(sev_dir, "runbundle_manifest.json")
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)

        results.append({
            "family": family,
            "severity": sev.value,
            "budget_curve": budget_curve,
            "bundle": bundle,
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def run_all_budget_sweeps(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Run budget sweeps for all families."""
    os.makedirs(out_dir, exist_ok=True)
    families = [MISMATCH_FAMILIES[0]] if smoke else MISMATCH_FAMILIES

    all_results: List[Dict[str, Any]] = []
    for fam in families:
        logger.info(f"Budget sweep: family={fam}")
        fam_results = sweep_budget_family(fam, out_dir, smoke=smoke)
        all_results.extend(fam_results)

    summary_path = os.path.join(out_dir, "budget_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Budget sweep results -> {summary_path}")
    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PWMI-CASSI: Sweep calibration budget per family"
    )
    parser.add_argument("--out_dir", default="results/pwmi_cassi_budget")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick validation (fewer budget levels, 1 trial)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_all_budget_sweeps(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
