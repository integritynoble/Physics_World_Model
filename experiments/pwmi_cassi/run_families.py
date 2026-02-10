"""Run CASSI calibration across all mismatch families on InverseNet splits.

For each mismatch family at multiple severity levels:
- Record theta-error (RMSE per parameter) before and after correction
- Record PSNR/SSIM improvement from reconstruction with corrected operator
- Record runtime
- Use bootstrap_correction() for CI on all results

Output: results as RunBundle manifests with full provenance.

Usage::

    python -m experiments.pwmi_cassi.run_families --out_dir results/pwmi_cassi_families
    python -m experiments.pwmi_cassi.run_families --smoke

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
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.inversenet.manifest_schema import Severity
from experiments.inversenet.mismatch_sweep import (
    CASSI_DISP_STEP_TABLE,
    CASSI_MASK_SHIFT_TABLE,
    CASSI_PSF_BLUR_TABLE,
    get_delta_theta,
    apply_mismatch,
)
from experiments.inversenet.gen_cassi import (
    _cassi_forward,
    _default_cassi_theta,
    _generate_coded_aperture,
    _generate_calibration_captures,
    _apply_photon_noise,
    _make_hsi_gt,
    SPATIAL_SIZE,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

MISMATCH_FAMILIES = ["disp_step", "mask_shift", "PSF_blur"]
SEVERITIES = [Severity.mild, Severity.moderate, Severity.severe]
N_BANDS = 8
PHOTON_LEVEL = 1e4
N_TRIALS = 5
BASE_SEED = 5000
BOOTSTRAP_K = 20
GAP_TV_ITERS = 30
GAP_TV_ITERS_SMOKE = 10

PWM_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"


# ── Helpers ──────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__),
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
    """Create a v0.3.0 RunBundle manifest dict."""
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


# ── Reconstruction ───────────────────────────────────────────────────────

def _cassi_gap_tv(
    y: np.ndarray,
    mask: np.ndarray,
    theta: Dict[str, Any],
    n_bands: int,
    iters: int = 30,
) -> np.ndarray:
    """GAP-TV for CASSI (self-contained, no external dependencies beyond scipy)."""
    from scipy.ndimage import gaussian_filter
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift

    H, W = mask.shape
    x = np.zeros((H, W, n_bands), dtype=np.float32)
    for l_idx in range(n_bands):
        dx, dy = dispersion_shift(theta, band=l_idx)
        x[:, :, l_idx] = np.roll(
            np.roll(y * mask, -int(round(dy)), axis=0),
            -int(round(dx)), axis=1,
        ) / max(n_bands, 1)

    for _ in range(iters):
        yb = np.zeros((H, W), dtype=np.float32)
        for l_idx in range(n_bands):
            dx, dy = dispersion_shift(theta, band=l_idx)
            band = np.roll(
                np.roll(x[:, :, l_idx], int(round(dy)), axis=0),
                int(round(dx)), axis=1,
            )
            yb += band * mask
        r = y - yb
        for l_idx in range(n_bands):
            dx, dy = dispersion_shift(theta, band=l_idx)
            upd = np.roll(
                np.roll(r * mask, -int(round(dy)), axis=0),
                -int(round(dx)), axis=1,
            ) / max(n_bands, 1)
            x[:, :, l_idx] += upd
        for l_idx in range(n_bands):
            x[:, :, l_idx] = gaussian_filter(x[:, :, l_idx], sigma=0.3)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


# ── Metric helpers ───────────────────────────────────────────────────────

def _compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    max_val = max(float(x.max()), float(y.max()), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


def _compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x64 = x.astype(np.float64).ravel()
    y64 = y.astype(np.float64).ravel()
    mu_x, mu_y = x64.mean(), y64.mean()
    sig_x, sig_y = x64.std(), y64.std()
    sig_xy = float(np.mean((x64 - mu_x) * (y64 - mu_y)))
    return float(
        (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    )


def _theta_rmse_vec(
    theta_true: Dict[str, Any], theta_est: Dict[str, Any]
) -> float:
    """RMSE between two theta dicts over numeric and list values."""
    diffs: List[float] = []
    for k in theta_true:
        if k not in theta_est:
            continue
        tv, ev = theta_true[k], theta_est[k]
        if isinstance(tv, (list, tuple)) and isinstance(ev, (list, tuple)):
            for a, b in zip(tv, ev):
                try:
                    diffs.append((float(a) - float(b)) ** 2)
                except (TypeError, ValueError):
                    continue
        else:
            try:
                diffs.append((float(tv) - float(ev)) ** 2)
            except (TypeError, ValueError):
                continue
    if not diffs:
        return 0.0
    return float(np.sqrt(np.mean(diffs)))


# ── UPWMI calibration engine ────────────────────────────────────────────

def _upwmi_calibrate_disp(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    iters: int = 30,
) -> Dict[str, Any]:
    """UPWMI dispersion polynomial search (for disp_step family).

    Coarse grid over disp_poly_x[0..2] then fine refinement.
    """
    nominal_poly = theta_nominal.get("disp_poly_x", [0.0, 1.0, 0.0])

    # Ranges centered around nominal, always include nominal value
    poly_x_0_range = np.array(sorted(set(
        list(np.linspace(nominal_poly[0] - 3.0, nominal_poly[0] + 3.0, 7))
        + [nominal_poly[0]]
    )))
    poly_x_1_range = np.array(sorted(set(
        list(np.linspace(max(0.1, nominal_poly[1] - 2.0), nominal_poly[1] + 2.0, 9))
        + [nominal_poly[1]]
    )))
    poly_x_2_range = np.array(sorted(set(
        list(np.linspace(nominal_poly[2] - 0.5, nominal_poly[2] + 0.5, 5))
        + [nominal_poly[2]]
    )))

    best_theta = dict(theta_nominal)
    # Score nominal as baseline
    x_nom = _cassi_gap_tv(y, mask, theta_nominal, n_bands, iters=max(iters // 3, 5))
    yb_nom = _cassi_forward(x_nom, mask, theta_nominal)
    best_residual = float(np.sum((y - yb_nom) ** 2))

    coarse_iters = max(iters // 3, 5)
    for a0 in poly_x_0_range:
        for a1 in poly_x_1_range:
            for a2 in poly_x_2_range:
                test_theta = dict(theta_nominal)
                test_theta["disp_poly_x"] = [float(a0), float(a1), float(a2)]
                x_test = _cassi_gap_tv(y, mask, test_theta, n_bands, iters=coarse_iters)
                yb = _cassi_forward(x_test, mask, test_theta)
                res = float(np.sum((y - yb) ** 2))
                if res < best_residual:
                    best_residual = res
                    best_theta = dict(test_theta)

    # Fine refinement (3 rounds, decreasing step)
    best_poly = list(best_theta.get("disp_poly_x", list(nominal_poly)))
    refine_iters = max(iters // 2, 5)
    for scale in [1.0, 0.5, 0.25]:
        for dim in range(3):
            center = best_poly[dim]
            step = [0.5, 0.3, 0.1][dim] * scale
            for delta in np.linspace(-step, step, 5):
                test_poly = list(best_poly)
                test_poly[dim] = center + delta
                test_theta = dict(best_theta)
                test_theta["disp_poly_x"] = [float(v) for v in test_poly]
                x_test = _cassi_gap_tv(y, mask, test_theta, n_bands, iters=refine_iters)
                yb = _cassi_forward(x_test, mask, test_theta)
                res = float(np.sum((y - yb) ** 2))
                if res < best_residual:
                    best_residual = res
                    best_theta = dict(test_theta)
                    best_poly = list(test_theta["disp_poly_x"])

    return best_theta


def _upwmi_calibrate_mask_shift(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    iters: int = 30,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """UPWMI mask shift search (for mask_shift family).

    Searches over integer (dx, dy) shifts of the coded aperture mask.
    Returns (theta, corrected_mask).
    """
    best_mask = mask.copy()
    best_theta = dict(theta_nominal)
    # Score nominal
    x_nom = _cassi_gap_tv(y, mask, theta_nominal, n_bands, iters=max(iters // 3, 5))
    yb_nom = _cassi_forward(x_nom, mask, theta_nominal)
    best_residual = float(np.sum((y - yb_nom) ** 2))

    coarse_iters = max(iters // 3, 5)
    # Search over mask shifts
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            test_mask = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            x_test = _cassi_gap_tv(y, test_mask, theta_nominal, n_bands, iters=coarse_iters)
            yb = _cassi_forward(x_test, test_mask, theta_nominal)
            res = float(np.sum((y - yb) ** 2))
            if res < best_residual:
                best_residual = res
                best_mask = test_mask.copy()

    return best_theta, best_mask


def _upwmi_calibrate_psf(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    iters: int = 30,
) -> Tuple[Dict[str, Any], float]:
    """UPWMI PSF blur calibration (for PSF_blur family).

    Deconvolves the measurement with candidate PSF widths and picks the
    one yielding the lowest forward residual.
    Returns (theta, best_psf_sigma).
    """
    from scipy.ndimage import gaussian_filter

    best_sigma = 0.0
    best_theta = dict(theta_nominal)
    # Score with no deconvolution
    x_nom = _cassi_gap_tv(y, mask, theta_nominal, n_bands, iters=max(iters // 3, 5))
    yb_nom = _cassi_forward(x_nom, mask, theta_nominal)
    best_residual = float(np.sum((y - yb_nom) ** 2))

    coarse_iters = max(iters // 3, 5)
    for sigma in np.linspace(0.0, 4.0, 9):
        if sigma > 0:
            # Approximate deconvolution: apply inverse filter by sharpening
            y_deconv = y.copy()
            y_blurred = gaussian_filter(y, sigma=sigma)
            # Wiener-like: y_sharp = y + alpha*(y - y_blurred)
            alpha = min(sigma, 2.0)
            y_deconv = np.clip(y + alpha * (y - y_blurred), 0, None).astype(np.float32)
        else:
            y_deconv = y

        x_test = _cassi_gap_tv(y_deconv, mask, theta_nominal, n_bands, iters=coarse_iters)
        yb = _cassi_forward(x_test, mask, theta_nominal)
        # Compare against original (blurred) measurement
        if sigma > 0:
            yb_blurred = gaussian_filter(yb, sigma=sigma)
        else:
            yb_blurred = yb
        res = float(np.sum((y - yb_blurred) ** 2))
        if res < best_residual:
            best_residual = res
            best_sigma = sigma

    return best_theta, best_sigma


def _upwmi_calibrate_cassi(
    y: np.ndarray,
    mask: np.ndarray,
    theta_nominal: Dict[str, Any],
    n_bands: int,
    y_cal: np.ndarray,
    iters: int = 30,
    rng: Optional[np.random.Generator] = None,
    family: str = "disp_step",
) -> Tuple[Dict[str, Any], float]:
    """UPWMI-style derivative-free beam search for CASSI theta.

    Dispatches to family-specific calibration engines.
    Returns (theta_best, runtime_s).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t0 = time.time()

    if family == "disp_step":
        best_theta = _upwmi_calibrate_disp(
            y, mask, theta_nominal, n_bands, iters,
        )
    elif family == "mask_shift":
        best_theta, _mask = _upwmi_calibrate_mask_shift(
            y, mask, theta_nominal, n_bands, iters,
        )
    elif family == "PSF_blur":
        best_theta, _sigma = _upwmi_calibrate_psf(
            y, mask, theta_nominal, n_bands, iters,
        )
    else:
        # Fallback: dispersion search
        best_theta = _upwmi_calibrate_disp(
            y, mask, theta_nominal, n_bands, iters,
        )

    runtime = time.time() - t0
    return best_theta, runtime


# ── Per-family experiment runner ─────────────────────────────────────────

def run_one_trial(
    family: str,
    severity: Severity,
    seed: int,
    n_bands: int,
    photon_level: float,
    gap_tv_iters: int,
) -> Dict[str, Any]:
    """Run a single trial: generate data, calibrate, reconstruct, measure."""
    rng = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    # Generate ground truth
    x_gt = _make_hsi_gt(H, W, n_bands, rng)
    mask = _generate_coded_aperture(H, W, seed)
    theta_true = _default_cassi_theta(n_bands)

    # Apply mismatch
    mm = apply_mismatch(
        "cassi", family, severity,
        y=None, mask=mask, theta=theta_true, rng=rng,
    )
    delta_theta = mm["delta_theta"]
    theta_mm = mm.get("theta", theta_true)
    mask_mm = mm.get("mask", mask)

    # Generate measurement with mismatched operator
    y = _cassi_forward(x_gt, mask_mm, theta_mm)
    y = _apply_photon_noise(y, photon_level, rng)

    # For PSF_blur, apply the blur to the measurement directly
    if family == "PSF_blur" and "y" in mm:
        y = mm["y"]

    # Calibration captures
    y_cal = _generate_calibration_captures(mask, theta_true, n_bands, 4, rng)

    # Reconstruct with WRONG (nominal) operator and mask
    x_wrong = _cassi_gap_tv(y, mask, theta_true, n_bands, iters=gap_tv_iters)
    psnr_wrong = _compute_psnr(x_gt, x_wrong)
    ssim_wrong = _compute_ssim(
        x_gt.mean(axis=2) if x_gt.ndim == 3 else x_gt,
        x_wrong.mean(axis=2) if x_wrong.ndim == 3 else x_wrong,
    )

    # UPWMI calibration (family-aware)
    theta_est, cal_runtime = _upwmi_calibrate_cassi(
        y, mask, theta_true, n_bands, y_cal,
        iters=gap_tv_iters, rng=rng, family=family,
    )

    # For mask_shift family, also get the corrected mask
    if family == "mask_shift":
        _, mask_cal = _upwmi_calibrate_mask_shift(
            y, mask, theta_true, n_bands, iters=gap_tv_iters,
        )
    else:
        mask_cal = mask

    # Reconstruct with calibrated operator
    x_cal = _cassi_gap_tv(y, mask_cal, theta_est, n_bands, iters=gap_tv_iters)
    psnr_cal = _compute_psnr(x_gt, x_cal)
    ssim_cal = _compute_ssim(
        x_gt.mean(axis=2) if x_gt.ndim == 3 else x_gt,
        x_cal.mean(axis=2) if x_cal.ndim == 3 else x_cal,
    )

    # For PSF_blur: deconvolve then reconstruct
    if family == "PSF_blur":
        from scipy.ndimage import gaussian_filter
        _, best_sigma = _upwmi_calibrate_psf(
            y, mask, theta_true, n_bands, iters=gap_tv_iters,
        )
        if best_sigma > 0:
            y_deconv = y + min(best_sigma, 2.0) * (
                y - gaussian_filter(y, sigma=best_sigma)
            )
            y_deconv = np.clip(y_deconv, 0, None).astype(np.float32)
        else:
            y_deconv = y
        x_cal = _cassi_gap_tv(y_deconv, mask, theta_est, n_bands, iters=gap_tv_iters)
        psnr_cal = _compute_psnr(x_gt, x_cal)
        ssim_cal = _compute_ssim(
            x_gt.mean(axis=2) if x_gt.ndim == 3 else x_gt,
            x_cal.mean(axis=2) if x_cal.ndim == 3 else x_cal,
        )

    # Theta errors (for disp_step, compare dispersion poly;
    # for mask_shift/PSF_blur, compare applicable params)
    theta_error_before = _theta_rmse_vec(theta_mm, theta_true)
    theta_error_after = _theta_rmse_vec(theta_mm, theta_est)

    return {
        "family": family,
        "severity": severity.value,
        "seed": seed,
        "n_bands": n_bands,
        "photon_level": photon_level,
        "theta_error_before": theta_error_before,
        "theta_error_after": theta_error_after,
        "psnr_wrong_db": psnr_wrong,
        "psnr_cal_db": psnr_cal,
        "psnr_gain_db": psnr_cal - psnr_wrong,
        "ssim_wrong": ssim_wrong,
        "ssim_cal": ssim_cal,
        "ssim_gain": ssim_cal - ssim_wrong,
        "runtime_s": cal_runtime,
        "theta_true": theta_true,
        "theta_mm": theta_mm,
        "theta_est": theta_est,
        "delta_theta": delta_theta,
    }


def run_family_experiment(
    family: str,
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Run all severity levels for one mismatch family with bootstrap CI."""
    from pwm_core.mismatch.uncertainty import bootstrap_correction

    severities = [Severity.mild] if smoke else SEVERITIES
    n_trials = 1 if smoke else N_TRIALS
    gap_tv_iters = GAP_TV_ITERS_SMOKE if smoke else GAP_TV_ITERS
    bootstrap_k = 3 if smoke else BOOTSTRAP_K

    results: List[Dict[str, Any]] = []

    for sev in severities:
        logger.info(f"  Family={family}, Severity={sev.value}")

        trial_results: List[Dict[str, Any]] = []
        for trial_idx in range(n_trials):
            seed = BASE_SEED + hash((family, sev.value, trial_idx)) % 10000
            trial = run_one_trial(
                family, sev, seed, N_BANDS, PHOTON_LEVEL, gap_tv_iters,
            )
            trial_results.append(trial)

        # Aggregate metrics
        psnr_gains = [t["psnr_gain_db"] for t in trial_results]
        theta_errors_before = [t["theta_error_before"] for t in trial_results]
        theta_errors_after = [t["theta_error_after"] for t in trial_results]
        runtimes = [t["runtime_s"] for t in trial_results]
        psnr_cals = [t["psnr_cal_db"] for t in trial_results]
        ssim_cals = [t["ssim_cal"] for t in trial_results]

        # Bootstrap CI via correction_fn interface
        cal_data = np.array(psnr_gains, dtype=np.float64).reshape(-1, 1)
        if cal_data.shape[0] >= 2:
            def correction_fn(data_subset: np.ndarray) -> Dict[str, float]:
                vals = data_subset.ravel()
                return {
                    "psnr_gain": float(np.mean(vals)),
                    "psnr": float(np.mean(vals)),
                }

            ci_result = bootstrap_correction(
                correction_fn, cal_data, K=bootstrap_k, seed=42,
            )
            ci_psnr_gain = ci_result.theta_uncertainty.get(
                "psnr_gain", [float(np.min(psnr_gains)), float(np.max(psnr_gains))]
            )
        else:
            ci_psnr_gain = [float(psnr_gains[0]), float(psnr_gains[0])]

        metrics = {
            "psnr_gain_db_mean": float(np.mean(psnr_gains)),
            "psnr_gain_db_std": float(np.std(psnr_gains)),
            "psnr_gain_db_ci_low": float(ci_psnr_gain[0]),
            "psnr_gain_db_ci_high": float(ci_psnr_gain[1]),
            "psnr_cal_db_mean": float(np.mean(psnr_cals)),
            "ssim_cal_mean": float(np.mean(ssim_cals)),
            "theta_error_before_mean": float(np.mean(theta_errors_before)),
            "theta_error_after_mean": float(np.mean(theta_errors_after)),
            "theta_error_reduction": float(
                np.mean(theta_errors_before) - np.mean(theta_errors_after)
            ),
            "runtime_s_mean": float(np.mean(runtimes)),
            "n_trials": n_trials,
        }

        spec_id = f"pwmi_cassi_families_{family}_{sev.value}"
        seeds_used = [t["seed"] for t in trial_results]

        bundle = _make_run_bundle(
            spec_id=spec_id,
            metrics=metrics,
            artifacts={"trial_results": f"{spec_id}_trials.json"},
            hashes={
                "trial_results": "sha256:" + hashlib.sha256(
                    json.dumps(
                        [{k: v for k, v in t.items() if k not in ("theta_true", "theta_mm", "theta_est")}
                         for t in trial_results],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
            },
            seeds=seeds_used,
        )

        # Save per-severity results
        sev_dir = os.path.join(out_dir, spec_id)
        os.makedirs(sev_dir, exist_ok=True)

        trials_path = os.path.join(sev_dir, f"{spec_id}_trials.json")
        serializable_trials = []
        for t in trial_results:
            st = {}
            for k, v in t.items():
                if isinstance(v, dict):
                    st[k] = {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                             for kk, vv in v.items()}
                else:
                    st[k] = v
            serializable_trials.append(st)
        with open(trials_path, "w") as f:
            json.dump(serializable_trials, f, indent=2)

        bundle_path = os.path.join(sev_dir, "runbundle_manifest.json")
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)

        results.append({
            "family": family,
            "severity": sev.value,
            "metrics": metrics,
            "bundle": bundle,
        })

    return results


# ── Main entry point ─────────────────────────────────────────────────────

def run_all_families(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Run calibration experiments across all mismatch families."""
    os.makedirs(out_dir, exist_ok=True)
    families = [MISMATCH_FAMILIES[0]] if smoke else MISMATCH_FAMILIES

    all_results: List[Dict[str, Any]] = []
    for fam in families:
        logger.info(f"Running family: {fam}")
        fam_results = run_family_experiment(fam, out_dir, smoke=smoke)
        all_results.extend(fam_results)

    # Save combined summary
    summary_path = os.path.join(out_dir, "families_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All family results -> {summary_path}")
    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PWMI-CASSI: Run calibration across mismatch families"
    )
    parser.add_argument("--out_dir", default="results/pwmi_cassi_families")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick validation run (1 family, 1 severity, 1 trial)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_all_families(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
