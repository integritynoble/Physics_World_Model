"""PWM Flagship -- CASSI full-pipeline depth experiment.

References PWMI-CASSI (Paper 3) results from ``experiments/pwmi_cassi/``
and adds design -> preflight -> calibration -> reconstruction loop for
the spectral imaging modality.

1. **Design**      -- Spectral band count vs compression.
2. **Pre-flight**  -- Spectral imaging feasibility.
3. **Calibration** -- Reference experiments/pwmi_cassi/ disp_step results.
4. **Reconstruction** -- Calibrated vs uncalibrated improvement.

Usage::

    PYTHONPATH=. python -m experiments.pwm_flagship.cassi_loop --out_dir results/flagship_cassi
    PYTHONPATH=. python -m experiments.pwm_flagship.cassi_loop --smoke
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
from experiments.inversenet.gen_cassi import (
    _cassi_forward,
    _default_cassi_theta,
    _generate_coded_aperture,
    _generate_calibration_captures,
    _apply_photon_noise,
    SPATIAL_SIZE,
)
from experiments.pwmi_cassi.run_families import (
    _cassi_gap_tv,
    _compute_psnr,
    _compute_ssim,
    _upwmi_calibrate_cassi,
)

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

PWM_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"
BASE_SEED = 9000

BAND_COUNTS = [8, 16, 28]
PHOTON_LEVELS = [1e3, 1e4, 1e5]
MISMATCH_FAMILY = "disp_step"
SEVERITIES = [Severity.mild, Severity.moderate, Severity.severe]
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
    spec_id: str, metrics: Dict[str, float],
    artifacts: Dict[str, str], hashes: Dict[str, str],
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


# -- Synthetic HSI GT --------------------------------------------------------

def _make_hsi_gt(
    H: int, W: int, L: int, rng: np.random.Generator
) -> np.ndarray:
    cube = np.zeros((H, W, L), dtype=np.float32)
    n_basis = min(3, L)
    bases = []
    for _ in range(n_basis):
        b = gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0)
        b -= b.min()
        b /= b.max() + 1e-8
        bases.append(b)
    wavelengths = np.linspace(0, 1, L)
    for l_idx in range(L):
        w = wavelengths[l_idx]
        cube[:, :, l_idx] = (
            bases[0 % n_basis] * np.exp(-10 * (w - 0.3) ** 2)
            + bases[1 % n_basis] * np.exp(-10 * (w - 0.6) ** 2)
            + bases[2 % n_basis] * 0.2
        )
    cube -= cube.min()
    cube /= cube.max() + 1e-8
    return cube.astype(np.float32)


# ============================================================================
# Stages
# ============================================================================

def design_stage(smoke: bool = False) -> List[Dict[str, Any]]:
    """Spectral band count vs compression."""
    if smoke:
        return [{"n_bands": 8, "photon_level": 1e4}]
    variants = []
    for nb in BAND_COUNTS:
        for photon in PHOTON_LEVELS:
            variants.append({"n_bands": nb, "photon_level": photon})
    logger.info("Design stage: %d CASSI variants proposed", len(variants))
    return variants


def preflight_stage(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Spectral imaging feasibility."""
    results = []
    for v in variants:
        nb = v["n_bands"]
        photon = v["photon_level"]
        snr_db = 10 * np.log10(photon)
        cr = 1.0 / nb

        if nb <= 8:
            recoverability = "excellent"
            expected_psnr = 28.0
        elif nb <= 16:
            recoverability = "sufficient"
            expected_psnr = 24.0
        else:
            recoverability = "marginal"
            expected_psnr = 20.0

        quality_tier = "acceptable" if photon >= 1e4 else "marginal"

        result = dict(v)
        result.update({
            "spectral_cr": cr,
            "snr_db": float(snr_db),
            "quality_tier": quality_tier,
            "recoverability": recoverability,
            "expected_psnr_db": expected_psnr,
            "predicted_psnr_band": [expected_psnr - 3.0, expected_psnr + 3.0],
            "proceed_recommended": quality_tier != "insufficient",
        })
        results.append(result)
    return results


# ============================================================================
# Full CASSI loop
# ============================================================================

def run_cassi_loop(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Execute full CASSI pipeline referencing PWMI-CASSI results."""
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
        nb = variant["n_bands"]
        photon = variant["photon_level"]

        for sev in severities:
            seed = BASE_SEED + var_idx * 100 + hash(sev.value) % 100
            rng = np.random.default_rng(seed)

            sid = (
                f"cassi_flagship_b{nb:02d}_p{photon:.0e}_"
                f"{MISMATCH_FAMILY}_{sev.value}_s{seed}"
            ).replace("+", "")

            logger.info("CASSI loop: %s", sid)

            x_gt = _make_hsi_gt(H, W, nb, rng)
            mask = _generate_coded_aperture(H, W, seed)
            theta_true = _default_cassi_theta(nb)

            # Apply disp_step mismatch
            mm = apply_mismatch(
                "cassi", MISMATCH_FAMILY, sev,
                y=None, mask=mask, theta=theta_true, rng=rng,
            )
            theta_mm = mm.get("theta", theta_true)

            # Generate measurement with mismatched operator
            y = _cassi_forward(x_gt, mask, theta_mm)
            y = _apply_photon_noise(y, photon, rng)

            # Calibration captures
            y_cal = _generate_calibration_captures(mask, theta_true, nb, 4, rng)

            # Reconstruct with wrong (nominal) operator
            x_wrong = _cassi_gap_tv(y, mask, theta_true, nb, iters=gap_tv_iters)
            psnr_wrong = _compute_psnr(x_gt, x_wrong)
            ssim_wrong = _compute_ssim(
                x_gt.mean(axis=2), x_wrong.mean(axis=2),
            )

            # UPWMI calibration (reuse PWMI-CASSI engine)
            theta_est, cal_runtime = _upwmi_calibrate_cassi(
                y, mask, theta_true, nb, y_cal,
                iters=gap_tv_iters, rng=rng, family=MISMATCH_FAMILY,
            )

            # Reconstruct with calibrated operator
            x_cal = _cassi_gap_tv(y, mask, theta_est, nb, iters=gap_tv_iters)
            psnr_cal = _compute_psnr(x_gt, x_cal)
            ssim_cal = _compute_ssim(
                x_gt.mean(axis=2), x_cal.mean(axis=2),
            )

            # Oracle reconstruction (true operator)
            x_oracle = _cassi_gap_tv(y, mask, theta_mm, nb, iters=gap_tv_iters)
            psnr_oracle = _compute_psnr(x_gt, x_oracle)

            # Save
            sample_dir = os.path.join(out_dir, sid)
            os.makedirs(sample_dir, exist_ok=True)

            np.save(os.path.join(sample_dir, "x_gt.npy"), x_gt)
            np.save(os.path.join(sample_dir, "x_cal.npy"), x_cal)

            metrics = {
                "psnr_oracle_db": psnr_oracle,
                "psnr_wrong_db": psnr_wrong,
                "psnr_cal_db": psnr_cal,
                "psnr_gain_db": psnr_cal - psnr_wrong,
                "ssim_wrong": ssim_wrong,
                "ssim_cal": ssim_cal,
                "runtime_s": cal_runtime,
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
                "variant": {k: v for k, v in variant.items()},
                "severity": sev.value,
                "metrics": metrics,
                "bundle": bundle,
            }
            all_results.append(result)

    summary_path = os.path.join(out_dir, "cassi_flagship_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("CASSI flagship: %d results -> %s", len(all_results), summary_path)
    return all_results


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWM Flagship: CASSI full-pipeline experiment"
    )
    parser.add_argument("--out_dir", default="results/flagship_cassi")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_cassi_loop(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
