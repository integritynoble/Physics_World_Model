"""PWMI-CASSI -- Agent ablation study (F.6).

Measures CASSI reconstruction degradation when pipeline components are removed:

1. **Remove PhotonAgent**        -- photon-starved, unadapted solver settings.
2. **Remove Recoverability**     -- sparse coded aperture (10% open).
3. **Remove mismatch priors**    -- wrong operator, no calibration.
4. **Remove RunBundle discipline** -- stale mask (15% flipped), wrong iterations.

Each ablation compares against a full-pipeline baseline on the same data.
Results include per-ablation PSNR degradation with bootstrap error bars.

Usage::

    PYTHONPATH=. python -m experiments.pwmi_cassi.ablations --out_dir results/pwmi_cassi_ablations
    PYTHONPATH=. python -m experiments.pwmi_cassi.ablations --smoke
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
from typing import Any, Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter

from experiments.inversenet.gen_cassi import (
    SPATIAL_SIZE,
    _default_cassi_theta,
)

logger = logging.getLogger(__name__)

PWM_VERSION = "0.3.0"
BUNDLE_VERSION = "0.3.0"
BASE_SEED = 7000
N_BOOTSTRAP_ABLATION = 5
N_BANDS = 8
RECON_ITERS = 30
RECON_ITERS_SMOKE = 10


# -- Helpers -----------------------------------------------------------------

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


def _compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    max_val = max(float(x.max()), float(y.max()), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


def _make_run_bundle(
    spec_id: str, metrics: Dict[str, float], seeds: List[int],
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
        "artifacts": {},
        "hashes": {},
    }


# -- CASSI helpers -----------------------------------------------------------

def _generate_cassi_data(seed: int, nb: int = N_BANDS):
    """Generate a synthetic CASSI scene (cube + mask + theta + measurement)."""
    rng = np.random.default_rng(seed)
    H, W = SPATIAL_SIZE

    # Spectral cube
    bases = [gaussian_filter(rng.random((H, W)).astype(np.float32), sigma=6.0) for _ in range(3)]
    for b in bases:
        b -= b.min()
        b /= b.max() + 1e-8
    cube = np.zeros((H, W, nb), dtype=np.float32)
    wl = np.linspace(0, 1, nb)
    for l in range(nb):
        cube[:, :, l] = (
            bases[0] * np.exp(-10 * (wl[l] - 0.3) ** 2)
            + bases[1] * np.exp(-10 * (wl[l] - 0.6) ** 2)
            + bases[2] * 0.2
        )
    cube -= cube.min()
    cube /= cube.max() + 1e-8

    mask = (rng.random((H, W)) > 0.5).astype(np.float32)
    theta = _default_cassi_theta(nb)

    return cube, mask, theta, rng


def _cassi_forward_simple(cube, mask, theta, nb):
    """Compute CASSI measurement y = sum_l shift(cube_l) * mask."""
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift
    H, W = mask.shape
    y = np.zeros((H, W), dtype=np.float32)
    for l in range(nb):
        dx, dy = dispersion_shift(theta, band=l)
        band = np.roll(np.roll(cube[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1)
        y += band * mask
    return y


def _cassi_recon(y, mask, theta, nb, n_iter=30):
    """GAP-TV reconstruction for CASSI."""
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift
    H, W = mask.shape
    x = np.zeros((H, W, nb), dtype=np.float32)
    for l in range(nb):
        dx, dy = dispersion_shift(theta, band=l)
        x[:, :, l] = np.roll(
            np.roll(y * mask, -int(round(dy)), axis=0),
            -int(round(dx)), axis=1,
        ) / max(nb, 1)

    for _ in range(n_iter):
        yb = np.zeros((H, W), dtype=np.float32)
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            band = np.roll(np.roll(x[:, :, l], int(round(dy)), axis=0), int(round(dx)), axis=1)
            yb += band * mask
        r = y - yb
        for l in range(nb):
            dx, dy = dispersion_shift(theta, band=l)
            upd = np.roll(
                np.roll(r * mask, -int(round(dy)), axis=0),
                -int(round(dx)), axis=1,
            ) / max(nb, 1)
            x[:, :, l] += upd
        for l in range(nb):
            x[:, :, l] = gaussian_filter(x[:, :, l], sigma=0.3)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


# -- Ablation functions ------------------------------------------------------

def _run_full_pipeline(seed: int, recon_iters: int) -> Dict[str, float]:
    """Baseline: full pipeline with correct operator, good photon level."""
    cube, mask, theta, rng = _generate_cassi_data(seed)
    y = _cassi_forward_simple(cube, mask, theta, N_BANDS)
    y = y + rng.normal(0, 0.01, size=y.shape).astype(np.float32)
    x_hat = _cassi_recon(y, mask, theta, N_BANDS, n_iter=recon_iters)
    return {"psnr": _compute_psnr(cube, x_hat)}


def _ablation_no_photon(seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 1: Remove PhotonAgent -- photon-starved, unadapted solver.

    Without PhotonAgent, system uses very low photon count and default
    solver settings (too few iterations, no noise-adapted regularization).
    """
    cube, mask, theta, rng = _generate_cassi_data(seed)
    y = _cassi_forward_simple(cube, mask, theta, N_BANDS)
    # Very low photon count (PhotonAgent would flag this)
    photon_level = 10.0
    scale = photon_level / (np.abs(y).max() + 1e-10)
    y = rng.poisson(np.maximum(y * scale, 0)).astype(np.float32) / scale
    # Fewer iterations (no adaptation)
    x_hat = _cassi_recon(y, mask, theta, N_BANDS, n_iter=max(recon_iters // 3, 3))
    return {"psnr": _compute_psnr(cube, x_hat)}


def _ablation_no_recoverability(seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 2: Remove Recoverability -- sparse mask (10% open).

    Without RecoverabilityAgent, system uses a poorly designed coded aperture
    with only 10% transmittance, causing severe information loss.
    """
    cube, mask, theta, rng = _generate_cassi_data(seed)
    # Sparse mask: only 10% open (RecoverabilityAgent would flag this)
    sparse_mask = (rng.random(mask.shape) > 0.9).astype(np.float32)
    y = _cassi_forward_simple(cube, sparse_mask, theta, N_BANDS)
    y = y + rng.normal(0, 0.01, size=y.shape).astype(np.float32)
    x_hat = _cassi_recon(y, sparse_mask, theta, N_BANDS, n_iter=recon_iters)
    return {"psnr": _compute_psnr(cube, x_hat)}


def _ablation_no_mismatch(seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 3: Remove mismatch priors -- wrong operator, no calibration.

    Uses a completely different coded aperture for reconstruction than what
    was used for measurement (simulating unknown mask without calibration).
    """
    cube, mask, theta, rng = _generate_cassi_data(seed)
    y = _cassi_forward_simple(cube, mask, theta, N_BANDS)
    y = y + rng.normal(0, 0.01, size=y.shape).astype(np.float32)
    # Wrong mask: different random pattern (no calibration)
    rng_wrong = np.random.default_rng(seed + 5555)
    mask_wrong = (rng_wrong.random(mask.shape) > 0.5).astype(np.float32)
    x_hat = _cassi_recon(y, mask_wrong, theta, N_BANDS, n_iter=recon_iters)
    return {"psnr": _compute_psnr(cube, x_hat)}


def _ablation_no_runbundle(seed: int, recon_iters: int) -> Dict[str, float]:
    """Ablation 4: Remove RunBundle discipline -- stale mask, wrong config.

    Without RunBundle hash verification, the mask used for reconstruction
    has 15% of entries flipped (stale/corrupted calibration data) and the
    solver uses suboptimal settings.
    """
    cube, mask, theta, rng = _generate_cassi_data(seed)
    y = _cassi_forward_simple(cube, mask, theta, N_BANDS)
    y = y + rng.normal(0, 0.01, size=y.shape).astype(np.float32)
    # Stale mask: 15% of entries flipped (no hash verification)
    rng_stale = np.random.default_rng(seed + 99999)
    mask_stale = mask.copy()
    flip = rng_stale.random(mask.shape) < 0.15
    mask_stale[flip] = 1.0 - mask_stale[flip]
    x_hat = _cassi_recon(y, mask_stale, theta, N_BANDS, n_iter=max(recon_iters // 3, 3))
    return {"psnr": _compute_psnr(cube, x_hat)}


# -- Main runner -------------------------------------------------------------

ABLATION_NAMES = [
    "no_photon",
    "no_recoverability",
    "no_mismatch",
    "no_runbundle",
]

ABLATION_FNS = {
    "no_photon": _ablation_no_photon,
    "no_recoverability": _ablation_no_recoverability,
    "no_mismatch": _ablation_no_mismatch,
    "no_runbundle": _ablation_no_runbundle,
}


def run_ablations(
    out_dir: str,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Run all 4 CASSI-specific ablations (F.6).

    Returns list of dicts with baseline_psnr, ablation_psnr, degradation_db,
    and bootstrap confidence intervals.
    """
    os.makedirs(out_dir, exist_ok=True)

    n_trials = 2 if smoke else N_BOOTSTRAP_ABLATION
    recon_iters = RECON_ITERS_SMOKE if smoke else RECON_ITERS
    ablation_names = ABLATION_NAMES[:2] if smoke else ABLATION_NAMES

    # Baseline: full pipeline
    baseline_psnrs = []
    for trial in range(n_trials):
        seed = BASE_SEED + trial * 7
        r = _run_full_pipeline(seed, recon_iters)
        baseline_psnrs.append(r["psnr"])
    baseline_mean = float(np.mean(baseline_psnrs))
    baseline_std = float(np.std(baseline_psnrs))

    logger.info("CASSI baseline: PSNR=%.2f +/- %.2f dB", baseline_mean, baseline_std)

    all_results: List[Dict[str, Any]] = []

    for ablation_name in ablation_names:
        ablation_fn = ABLATION_FNS[ablation_name]
        ablation_psnrs = []

        for trial in range(n_trials):
            seed = BASE_SEED + trial * 7
            r = ablation_fn(seed, recon_iters)
            ablation_psnrs.append(r["psnr"])

        ablation_mean = float(np.mean(ablation_psnrs))
        ablation_std = float(np.std(ablation_psnrs))
        degradation_db = baseline_mean - ablation_mean

        # Bootstrap error bars
        if n_trials >= 3:
            boot_degradations = []
            boot_rng = np.random.default_rng(42)
            for _ in range(50):
                idx = boot_rng.integers(0, n_trials, size=n_trials)
                boot_base = float(np.mean([baseline_psnrs[i] for i in idx]))
                boot_abl = float(np.mean([ablation_psnrs[i] for i in idx]))
                boot_degradations.append(boot_base - boot_abl)
            degradation_ci = [
                float(np.percentile(boot_degradations, 2.5)),
                float(np.percentile(boot_degradations, 97.5)),
            ]
        else:
            degradation_ci = [degradation_db, degradation_db]

        result = {
            "modality": "cassi",
            "ablation": ablation_name,
            "baseline_psnr_db": baseline_mean,
            "baseline_std_db": baseline_std,
            "ablation_psnr_db": ablation_mean,
            "ablation_std_db": ablation_std,
            "degradation_db": degradation_db,
            "degradation_ci": degradation_ci,
            "n_trials": n_trials,
        }

        logger.info(
            "  %s: degradation=%.2f dB [%.2f, %.2f]",
            ablation_name, degradation_db,
            degradation_ci[0], degradation_ci[1],
        )

        all_results.append(result)

    # Summary table
    logger.info("=" * 60)
    logger.info("%-22s %10s %15s", "Ablation", "Degrad(dB)", "CI")
    logger.info("-" * 60)
    for r in all_results:
        logger.info(
            "%-22s %10.2f [%6.2f, %6.2f]",
            r["ablation"], r["degradation_db"],
            r["degradation_ci"][0], r["degradation_ci"][1],
        )

    # Save
    bundle = _make_run_bundle(
        "pwmi_cassi_ablations",
        {"n_ablations": len(all_results)},
        [BASE_SEED],
    )

    output = {
        "results": all_results,
        "bundle": bundle,
    }

    with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("CASSI ablations: %d results -> %s", len(all_results), out_dir)
    return all_results


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PWMI-CASSI: Agent ablation study (F.6)"
    )
    parser.add_argument("--out_dir", default="results/pwmi_cassi_ablations")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ablations(args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
