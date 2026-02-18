"""Run 5 baselines on identical CASSI data for comparison.

Baselines:
1. No calibration (wrong/nominal operator)
2. Grid search (brute-force parameter sweep)
3. Gradient descent (differentiable approach)
4. UPWMI (derivative-free beam search -- our method)
5. UPWMI + gradient refinement (full pipeline -- our method)

For each baseline:
- Record: theta-error, PSNR, SSIM, runtime
- Output comparison tables as JSON in RunBundle format

Usage::

    python -m experiments.pwmi_cassi.comparisons --out_dir results/pwmi_cassi_compare
    python -m experiments.pwmi_cassi.comparisons --smoke

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
from experiments.inversenet.mismatch_sweep import apply_mismatch
from experiments.inversenet.gen_cassi import (
    _cassi_forward,
    _default_cassi_theta,
    _generate_coded_aperture,
    _generate_calibration_captures,
    _apply_photon_noise,
    _make_hsi_gt,
    SPATIAL_SIZE,
)
from experiments.pwmi_cassi.run_families import (
    _cassi_gap_tv,
    _compute_psnr,
    _compute_ssim,
    _theta_rmse_vec,
    _upwmi_calibrate_cassi,
    _git_hash,
    _make_run_bundle,
    MISMATCH_FAMILIES,
    N_BANDS,
    PHOTON_LEVEL,
    GAP_TV_ITERS,
    GAP_TV_ITERS_SMOKE,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

BASE_SEED = 9000
N_TRIALS = 5
BASELINE_NAMES = [
    "no_calibration",
    "grid_search",
    "gradient_descent",
    "upwmi",
    "upwmi_gradient",
]


# ── Baseline implementations ─────────────────────────────────────────────

def _baseline_no_calibration(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    y_cal: np.ndarray,
    gap_tv_iters: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], float]:
    """Baseline 1: no calibration, use nominal theta."""
    t0 = time.time()
    # Just return nominal theta without any correction
    return dict(theta_nominal), time.time() - t0


def _baseline_grid_search(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    y_cal: np.ndarray,
    gap_tv_iters: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], float]:
    """Baseline 2: brute-force grid search over dispersion polynomial."""
    t0 = time.time()

    best_theta = dict(theta_nominal)
    best_residual = float("inf")

    # Coarse grid over disp_poly_x[1] only (simplest calibration)
    for a1 in np.linspace(0.2, 3.0, 15):
        test_theta = dict(theta_nominal)
        test_theta["disp_poly_x"] = [0.0, float(a1), 0.0]
        x_test = _cassi_gap_tv(y, mask, test_theta, n_bands, iters=max(gap_tv_iters // 3, 5))
        yb = _cassi_forward(x_test, mask, test_theta)
        res = float(np.sum((y - yb) ** 2))
        if res < best_residual:
            best_residual = res
            best_theta = dict(test_theta)

    runtime = time.time() - t0
    return best_theta, runtime


def _baseline_gradient_descent(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    y_cal: np.ndarray,
    gap_tv_iters: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], float]:
    """Baseline 3: finite-difference gradient descent on dispersion params.

    Approximates gradient via central differences on the forward residual
    and performs Adam-like updates.
    """
    t0 = time.time()

    # Optimize disp_poly_x as a 3-vector
    poly = np.array(
        theta_nominal.get("disp_poly_x", [0.0, 1.0, 0.0]), dtype=np.float64
    )

    lr = 0.1
    eps_fd = 0.05  # finite-difference step
    beta1, beta2 = 0.9, 0.999
    m = np.zeros_like(poly)
    v = np.zeros_like(poly)
    eps_adam = 1e-8

    def _residual(poly_test: np.ndarray) -> float:
        test_theta = dict(theta_nominal)
        test_theta["disp_poly_x"] = poly_test.tolist()
        x_test = _cassi_gap_tv(
            y, mask, test_theta, n_bands,
            iters=max(gap_tv_iters // 4, 3),
        )
        yb = _cassi_forward(x_test, mask, test_theta)
        return float(np.sum((y - yb) ** 2))

    n_gd_iters = 15
    for step in range(n_gd_iters):
        # Central-difference gradient
        grad = np.zeros_like(poly)
        f_center = _residual(poly)
        for dim in range(len(poly)):
            poly_plus = poly.copy()
            poly_plus[dim] += eps_fd
            poly_minus = poly.copy()
            poly_minus[dim] -= eps_fd
            grad[dim] = (_residual(poly_plus) - _residual(poly_minus)) / (2 * eps_fd)

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        poly -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Clamp to reasonable range
        poly[0] = np.clip(poly[0], -5.0, 5.0)
        poly[1] = np.clip(poly[1], 0.1, 5.0)
        poly[2] = np.clip(poly[2], -1.0, 1.0)

    best_theta = dict(theta_nominal)
    best_theta["disp_poly_x"] = poly.tolist()
    runtime = time.time() - t0
    return best_theta, runtime


def _baseline_upwmi(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    y_cal: np.ndarray,
    gap_tv_iters: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], float]:
    """Baseline 4: UPWMI (derivative-free beam search)."""
    return _upwmi_calibrate_cassi(
        y, mask, theta_nominal, n_bands, y_cal,
        iters=gap_tv_iters, rng=rng,
    )


def _baseline_upwmi_gradient(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    y_cal: np.ndarray,
    gap_tv_iters: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], float]:
    """Baseline 5: UPWMI + gradient refinement (full pipeline).

    First runs UPWMI coarse search, then refines with gradient descent.
    """
    t0 = time.time()

    # Phase 1: UPWMI coarse
    theta_coarse, _ = _upwmi_calibrate_cassi(
        y, mask, theta_nominal, n_bands, y_cal,
        iters=gap_tv_iters, rng=rng,
    )

    # Phase 2: Gradient refinement from UPWMI solution
    poly = np.array(
        theta_coarse.get("disp_poly_x", [0.0, 1.0, 0.0]), dtype=np.float64
    )
    lr = 0.02
    eps_fd = 0.02

    def _residual(poly_test: np.ndarray) -> float:
        test_theta = dict(theta_nominal)
        test_theta["disp_poly_x"] = poly_test.tolist()
        x_test = _cassi_gap_tv(
            y, mask, test_theta, n_bands,
            iters=max(gap_tv_iters // 3, 5),
        )
        yb = _cassi_forward(x_test, mask, test_theta)
        return float(np.sum((y - yb) ** 2))

    # 5 gradient steps for local refinement
    for _ in range(5):
        grad = np.zeros_like(poly)
        for dim in range(len(poly)):
            poly_plus = poly.copy()
            poly_plus[dim] += eps_fd
            poly_minus = poly.copy()
            poly_minus[dim] -= eps_fd
            grad[dim] = (_residual(poly_plus) - _residual(poly_minus)) / (2 * eps_fd)

        poly -= lr * grad
        poly[0] = np.clip(poly[0], -5.0, 5.0)
        poly[1] = np.clip(poly[1], 0.1, 5.0)
        poly[2] = np.clip(poly[2], -1.0, 1.0)

    best_theta = dict(theta_coarse)
    best_theta["disp_poly_x"] = poly.tolist()
    runtime = time.time() - t0
    return best_theta, runtime


_BASELINE_FNS = {
    "no_calibration": _baseline_no_calibration,
    "grid_search": _baseline_grid_search,
    "gradient_descent": _baseline_gradient_descent,
    "upwmi": _baseline_upwmi,
    "upwmi_gradient": _baseline_upwmi_gradient,
}


# ── Per-trial comparison runner ──────────────────────────────────────────

def run_comparison_trial(
    family: str,
    severity: Severity,
    seed: int,
    gap_tv_iters: int,
) -> Dict[str, Any]:
    """Run all 5 baselines on identical data for one trial."""
    rng_base = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    # Generate identical data for all baselines
    x_gt = _make_hsi_gt(H, W, N_BANDS, rng_base)
    mask = _generate_coded_aperture(H, W, seed)
    theta_true = _default_cassi_theta(N_BANDS)

    mm = apply_mismatch(
        "cassi", family, severity,
        y=None, mask=mask, theta=theta_true, rng=rng_base,
    )
    theta_mm = mm.get("theta", theta_true)
    mask_mm = mm.get("mask", mask)

    y = _cassi_forward(x_gt, mask_mm, theta_mm)
    y = _apply_photon_noise(y, PHOTON_LEVEL, rng_base)

    y_cal = _generate_calibration_captures(mask, theta_true, N_BANDS, 4, rng_base)

    baseline_results: Dict[str, Dict[str, float]] = {}

    for bname in BASELINE_NAMES:
        logger.info(f"    baseline={bname}")
        rng_bl = np.random.default_rng(seed + hash(bname) % 10000)
        fn = _BASELINE_FNS[bname]

        theta_est, runtime = fn(
            y, mask, theta_true, N_BANDS, y_cal, gap_tv_iters, rng_bl,
        )

        # Reconstruct
        x_hat = _cassi_gap_tv(y, mask, theta_est, N_BANDS, iters=gap_tv_iters)
        psnr = _compute_psnr(x_gt, x_hat)
        ssim = _compute_ssim(
            x_gt.mean(axis=2) if x_gt.ndim == 3 else x_gt,
            x_hat.mean(axis=2) if x_hat.ndim == 3 else x_hat,
        )
        theta_error = _theta_rmse_vec(theta_mm, theta_est)

        baseline_results[bname] = {
            "theta_error_rmse": theta_error,
            "psnr_db": psnr,
            "ssim": ssim,
            "runtime_s": runtime,
        }

    return {
        "family": family,
        "severity": severity.value,
        "seed": seed,
        "baselines": baseline_results,
    }


# ── Full comparison for one family ───────────────────────────────────────

def run_family_comparison(
    family: str,
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Run 5 baselines for all severity levels of one family."""
    severities = [Severity.moderate] if smoke else [Severity.mild, Severity.moderate, Severity.severe]
    n_trials = 1 if smoke else N_TRIALS
    gap_tv_iters = GAP_TV_ITERS_SMOKE if smoke else GAP_TV_ITERS

    results: List[Dict[str, Any]] = []

    for sev in severities:
        logger.info(f"  Comparison: family={family}, sev={sev.value}")

        trial_results = []
        for trial_idx in range(n_trials):
            seed = BASE_SEED + hash((family, sev.value, trial_idx)) % 10000
            trial = run_comparison_trial(family, sev, seed, gap_tv_iters)
            trial_results.append(trial)

        # Aggregate per baseline
        comparison_table: Dict[str, Dict[str, Any]] = {}
        for bname in BASELINE_NAMES:
            metric_keys = ["theta_error_rmse", "psnr_db", "ssim", "runtime_s"]
            agg: Dict[str, Any] = {"baseline": bname}
            for mk in metric_keys:
                vals = [t["baselines"][bname][mk] for t in trial_results]
                agg[f"{mk}_mean"] = float(np.mean(vals))
                agg[f"{mk}_std"] = float(np.std(vals))
            comparison_table[bname] = agg

        spec_id = f"pwmi_cassi_compare_{family}_{sev.value}"
        metrics_flat: Dict[str, Any] = {}
        for bname, agg in comparison_table.items():
            for k, v in agg.items():
                if k != "baseline":
                    metrics_flat[f"{bname}_{k}"] = v

        bundle = _make_run_bundle(
            spec_id=spec_id,
            metrics=metrics_flat,
            artifacts={
                "comparison_table": f"{spec_id}_table.json",
                "trial_results": f"{spec_id}_trials.json",
            },
            hashes={
                "comparison_table": "sha256:" + hashlib.sha256(
                    json.dumps(comparison_table, sort_keys=True).encode()
                ).hexdigest(),
            },
            seeds=[BASE_SEED],
        )

        sev_dir = os.path.join(out_dir, spec_id)
        os.makedirs(sev_dir, exist_ok=True)

        table_path = os.path.join(sev_dir, f"{spec_id}_table.json")
        with open(table_path, "w") as f:
            json.dump(comparison_table, f, indent=2)

        trials_path = os.path.join(sev_dir, f"{spec_id}_trials.json")
        with open(trials_path, "w") as f:
            json.dump(trial_results, f, indent=2)

        bundle_path = os.path.join(sev_dir, "runbundle_manifest.json")
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)

        results.append({
            "family": family,
            "severity": sev.value,
            "comparison_table": comparison_table,
            "bundle": bundle,
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def run_all_comparisons(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Run baseline comparisons for all mismatch families."""
    os.makedirs(out_dir, exist_ok=True)
    families = [MISMATCH_FAMILIES[0]] if smoke else MISMATCH_FAMILIES

    all_results: List[Dict[str, Any]] = []
    for fam in families:
        logger.info(f"Comparison: family={fam}")
        fam_results = run_family_comparison(fam, out_dir, smoke=smoke)
        all_results.extend(fam_results)

    summary_path = os.path.join(out_dir, "comparisons_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Comparison results -> {summary_path}")
    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PWMI-CASSI: Run 5 baselines on identical CASSI data"
    )
    parser.add_argument("--out_dir", default="results/pwmi_cassi_compare")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick validation (1 family, 1 severity, 1 trial)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_all_comparisons(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
