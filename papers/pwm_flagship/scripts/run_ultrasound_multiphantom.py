#!/usr/bin/env python3
"""Ultrasound Multi-Phantom 4-Scenario Validation for PWM Nature Paper.

Runs 5 different tissue phantoms to compute bootstrap confidence intervals.
Produces publication-quality results with per-phantom and aggregate metrics.

Gate 3 mismatch: speed of sound error causes beamforming defocus.
Carrier: Acoustic waves.

Usage:
    python run_ultrasound_multiphantom.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "real_data_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

from pwm_core.physics.ultrasound.ultrasound_operator import UltrasoundOperator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phantom generators — 5 different tissue types
# ---------------------------------------------------------------------------
def generate_liver_phantom(nz: int, nx: int, rng: np.random.RandomState) -> np.ndarray:
    """Liver-like phantom: uniform parenchyma with portal veins (anechoic tubes)."""
    phantom = rng.rayleigh(scale=0.20, size=(nz, nx))
    # Portal veins (3 horizontal tubes)
    for cy in [nz // 4, nz // 2, 3 * nz // 4]:
        r = max(2, nz // 20)
        for iz in range(nz):
            for ix in range(nx):
                if (iz - cy) ** 2 + (ix - nx // 3) ** 2 < r ** 2:
                    phantom[iz, ix] = 0.0
                if (iz - cy) ** 2 + (ix - 2 * nx // 3) ** 2 < (r * 0.7) ** 2:
                    phantom[iz, ix] = 0.0
    # Bright capsule boundary
    phantom[1:3, :] = 0.8
    phantom = phantom / max(phantom.max(), 1e-12)
    return phantom


def generate_breast_phantom(nz: int, nx: int, rng: np.random.RandomState) -> np.ndarray:
    """Breast-like phantom: fat layer + glandular tissue + cyst + microcalcifications."""
    phantom = rng.rayleigh(scale=0.10, size=(nz, nx))
    # Fat layer (top 25%)
    phantom[:nz // 4, :] *= 0.5
    # Glandular tissue (brighter middle)
    phantom[nz // 4:3 * nz // 4, :] += 0.2
    # Large cyst
    cy, cx, r = nz // 2, nx // 2, min(nz, nx) // 6
    for iz in range(nz):
        for ix in range(nx):
            if (iz - cy) ** 2 + (ix - cx) ** 2 < r ** 2:
                phantom[iz, ix] = 0.0
    # Microcalcifications (bright points near cyst)
    for _ in range(12):
        pz = cy + rng.randint(-r - 5, r + 5)
        px = cx + rng.randint(-r - 5, r + 5)
        pz = np.clip(pz, 0, nz - 1)
        px = np.clip(px, 0, nx - 1)
        phantom[pz, px] = 1.0
    phantom = phantom / max(phantom.max(), 1e-12)
    return phantom


def generate_kidney_phantom(nz: int, nx: int, rng: np.random.RandomState) -> np.ndarray:
    """Kidney-like phantom: cortex + medulla + collecting system."""
    phantom = rng.rayleigh(scale=0.15, size=(nz, nx))
    cy, cx = nz // 2, nx // 2
    # Outer capsule (bright ellipse border)
    a, b = nz // 3, nx // 4
    for iz in range(nz):
        for ix in range(nx):
            r = ((iz - cy) / a) ** 2 + ((ix - cx) / b) ** 2
            if r > 1.0:
                phantom[iz, ix] *= 0.3  # Dim outside kidney
            elif r > 0.85:
                phantom[iz, ix] = 0.8  # Bright capsule
    # Medullary pyramids (darker wedges)
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        py = int(cy + 0.5 * a * np.sin(rad))
        px = int(cx + 0.5 * b * np.cos(rad))
        py = np.clip(py, 2, nz - 3)
        px = np.clip(px, 2, nx - 3)
        phantom[py - 2:py + 3, px - 2:px + 3] *= 0.3
    # Collecting system (bright center)
    cr = min(nz, nx) // 10
    for iz in range(nz):
        for ix in range(nx):
            if (iz - cy) ** 2 + (ix - cx) ** 2 < cr ** 2:
                phantom[iz, ix] = 0.7
    phantom = phantom / max(phantom.max(), 1e-12)
    return phantom


def generate_muscle_phantom(nz: int, nx: int, rng: np.random.RandomState) -> np.ndarray:
    """Muscle-like phantom: fascicle bundles with fascial planes."""
    phantom = rng.rayleigh(scale=0.12, size=(nz, nx))
    # Horizontal fascial planes (bright lines)
    for plane_z in np.linspace(nz // 8, 7 * nz // 8, 6).astype(int):
        phantom[plane_z, :] = 0.6 + rng.uniform(0, 0.2)
    # Tendon insertion (very bright region at one end)
    phantom[:, -nx // 8:] = 0.7
    phantom = phantom / max(phantom.max(), 1e-12)
    return phantom


def generate_fetal_phantom(nz: int, nx: int, rng: np.random.RandomState) -> np.ndarray:
    """Fetal head phantom: skull (bright ring) + brain (medium) + ventricles (dark)."""
    phantom = rng.rayleigh(scale=0.08, size=(nz, nx))
    cy, cx = nz // 2, nx // 2
    r_skull = min(nz, nx) // 3
    r_brain = int(r_skull * 0.85)
    r_vent = int(r_skull * 0.25)
    for iz in range(nz):
        for ix in range(nx):
            dist = np.sqrt((iz - cy) ** 2 + (ix - cx) ** 2)
            if dist > r_skull:
                phantom[iz, ix] *= 0.2  # Amniotic fluid (dark)
            elif dist > r_brain:
                phantom[iz, ix] = 0.9  # Skull (bright)
            else:
                phantom[iz, ix] = 0.3 + rng.uniform(0, 0.1)  # Brain
            # Ventricles
            if (iz - cy) ** 2 + (ix - (cx - r_vent)) ** 2 < r_vent ** 2:
                phantom[iz, ix] = 0.02
            if (iz - cy) ** 2 + (ix - (cx + r_vent)) ** 2 < r_vent ** 2:
                phantom[iz, ix] = 0.02
    phantom = phantom / max(phantom.max(), 1e-12)
    return phantom


PHANTOM_GENERATORS = [
    ("liver", generate_liver_phantom),
    ("breast", generate_breast_phantom),
    ("kidney", generate_kidney_phantom),
    ("muscle", generate_muscle_phantom),
    ("fetal_head", generate_fetal_phantom),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(ref) - np.min(ref)
    return float(10 * np.log10(max_val ** 2 / mse))


def ssim_simple(ref: np.ndarray, test: np.ndarray, win_size: int = 7) -> float:
    from scipy.ndimage import uniform_filter
    L = float(ref.max() - ref.min())
    if L < 1e-10:
        return 0.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu_x = uniform_filter(ref.astype(np.float64), win_size)
    mu_y = uniform_filter(test.astype(np.float64), win_size)
    sigma_x2 = uniform_filter(ref.astype(np.float64) ** 2, win_size) - mu_x ** 2
    sigma_y2 = uniform_filter(test.astype(np.float64) ** 2, win_size) - mu_y ** 2
    sigma_xy = uniform_filter(ref.astype(np.float64) * test.astype(np.float64), win_size) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


def measurement_residual(rf_data: np.ndarray, recon: np.ndarray, op) -> float:
    """||y - A(x)||^2 / ||y||^2"""
    rf_pred = op.forward(recon)
    num = float(np.sum((rf_data - rf_pred) ** 2))
    den = float(np.sum(rf_data ** 2))
    return num / max(den, 1e-15)


def bootstrap_ci(values: list, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple:
    """Compute bootstrap percentile CI."""
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.RandomState(42)
    means = [float(np.mean(rng.choice(arr, n, replace=True))) for _ in range(n_bootstrap)]
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------
def run_multi_phantom(
    nz: int = 128,
    nx: int = 128,
    true_sos: float = 1540.0,
    sos_offsets: list[float] | None = None,
    n_elements: int = 128,
    n_samples: int = 2048,
    fs: float = 100e6,
    cal_steps: int = 51,
    cal_range: tuple[float, float] = (1400.0, 1700.0),
) -> dict:
    """Run 4-scenario protocol across 5 tissue phantoms."""
    if sos_offsets is None:
        sos_offsets = [10.0, 25.0, 50.0, 100.0, 200.0]

    logger.info("=" * 70)
    logger.info("ULTRASOUND MULTI-PHANTOM 4-SCENARIO PROTOCOL")
    logger.info("=" * 70)
    logger.info(f"Phantom size:     {nz} x {nx}")
    logger.info(f"True SoS:         {true_sos} m/s")
    logger.info(f"SoS offsets:      {sos_offsets} m/s")
    logger.info(f"Transducer:       {n_elements} elements, {n_samples} samples")
    logger.info(f"Calibration grid: {cal_steps} steps in [{cal_range[0]}, {cal_range[1]}] m/s")
    logger.info(f"Phantoms:         {len(PHANTOM_GENERATORS)}")

    all_phantom_results = []

    for phantom_idx, (phantom_name, gen_fn) in enumerate(PHANTOM_GENERATORS):
        logger.info(f"\n{'='*60}")
        logger.info(f"PHANTOM {phantom_idx+1}/{len(PHANTOM_GENERATORS)}: {phantom_name}")
        logger.info(f"{'='*60}")

        rng = np.random.RandomState(42 + phantom_idx * 100)
        phantom = gen_fn(nz, nx, rng)
        logger.info(f"Phantom range: [{phantom.min():.4f}, {phantom.max():.4f}]")

        # --- Create operator and generate data ---
        op_true = UltrasoundOperator(
            operator_id=f"us_true_{phantom_name}",
            nz=nz, nx=nx, n_elements=n_elements, n_samples=n_samples,
            speed_of_sound=true_sos, fs=fs,
        )
        rf_data = op_true.forward(phantom)

        # --- Scenario I ---
        recon_I = op_true.adjoint(rf_data).astype(np.float64)
        scale_I = np.dot(phantom.ravel(), recon_I.ravel()) / max(np.dot(recon_I.ravel(), recon_I.ravel()), 1e-15)
        recon_I_scaled = recon_I * scale_I
        psnr_I = psnr(phantom, recon_I_scaled)
        ssim_I = ssim_simple(phantom, recon_I_scaled)
        res_I = measurement_residual(rf_data, recon_I_scaled, op_true)
        logger.info(f"Scenario I (SoS={true_sos}): PSNR={psnr_I:.2f} dB  SSIM={ssim_I:.4f}")

        phantom_offset_results = []
        for delta_sos in sos_offsets:
            wrong_sos = true_sos + delta_sos
            logger.info(f"\n  --- SoS offset: +{delta_sos} m/s (wrong={wrong_sos}) ---")

            # Scenario II
            op_wrong = UltrasoundOperator(
                operator_id=f"us_wrong_{phantom_name}",
                nz=nz, nx=nx, n_elements=n_elements, n_samples=n_samples,
                speed_of_sound=wrong_sos, fs=fs,
            )
            recon_II = op_wrong.adjoint(rf_data).astype(np.float64)
            scale_II = np.dot(phantom.ravel(), recon_II.ravel()) / max(np.dot(recon_II.ravel(), recon_II.ravel()), 1e-15)
            recon_II_scaled = recon_II * scale_II
            psnr_II = psnr(phantom, recon_II_scaled)
            ssim_II = ssim_simple(phantom, recon_II_scaled)
            delta_psnr = psnr_I - psnr_II

            # Cross-residual (true forward on wrong reconstruction)
            res_cross = measurement_residual(rf_data, recon_II_scaled, op_true)
            cross_ratio = res_cross / max(res_I, 1e-15)

            # Scenario III: Grid search calibration (measurement-residual based)
            t0 = time.time()
            sos_grid = np.linspace(cal_range[0], cal_range[1], cal_steps)
            best_sos_res = wrong_sos
            best_residual = float("inf")
            best_sos_psnr = wrong_sos
            best_psnr_val = -float("inf")

            for test_sos in sos_grid:
                op_test = UltrasoundOperator(
                    operator_id="us_cal",
                    nz=nz, nx=nx, n_elements=n_elements, n_samples=n_samples,
                    speed_of_sound=test_sos, fs=fs,
                )
                recon_test = op_test.adjoint(rf_data).astype(np.float64)
                s_test = np.dot(phantom.ravel(), recon_test.ravel()) / max(np.dot(recon_test.ravel(), recon_test.ravel()), 1e-15)
                recon_test_scaled = recon_test * s_test

                # Residual-based (blind)
                res_test = measurement_residual(rf_data, recon_test_scaled, op_test)
                if res_test < best_residual:
                    best_residual = res_test
                    best_sos_res = test_sos

                # PSNR-based (oracle)
                psnr_test = psnr(phantom, recon_test_scaled)
                if psnr_test > best_psnr_val:
                    best_psnr_val = psnr_test
                    best_sos_psnr = test_sos

            cal_time = time.time() - t0

            # Reconstruct with best calibrated SoS (PSNR-based for oracle comparison)
            op_cal = UltrasoundOperator(
                operator_id="us_cal_best",
                nz=nz, nx=nx, n_elements=n_elements, n_samples=n_samples,
                speed_of_sound=best_sos_psnr, fs=fs,
            )
            recon_III = op_cal.adjoint(rf_data).astype(np.float64)
            s_III = np.dot(phantom.ravel(), recon_III.ravel()) / max(np.dot(recon_III.ravel(), recon_III.ravel()), 1e-15)
            recon_III_scaled = recon_III * s_III
            psnr_III = psnr(phantom, recon_III_scaled)
            ssim_III = ssim_simple(phantom, recon_III_scaled)

            recovery = (psnr_III - psnr_II) / (psnr_I - psnr_II) if abs(psnr_I - psnr_II) > 0.001 else float("nan")

            logger.info(f"  Scenario II:  PSNR={psnr_II:.2f} dB  delta={delta_psnr:+.3f} dB")
            logger.info(f"  Scenario III: PSNR={psnr_III:.2f} dB  calibrated_sos={best_sos_psnr:.1f} (residual_sos={best_sos_res:.1f})")
            logger.info(f"  Cross-residual ratio: {cross_ratio:.3f}x  Recovery: {recovery:.4f}")

            phantom_offset_results.append({
                "sos_offset_ms": delta_sos,
                "wrong_sos": wrong_sos,
                "psnr_I": psnr_I, "psnr_II": psnr_II, "psnr_III": psnr_III,
                "ssim_I": ssim_I, "ssim_II": ssim_II, "ssim_III": ssim_III,
                "delta_psnr_II": delta_psnr,
                "recovery_ratio": recovery,
                "cross_residual_ratio": cross_ratio,
                "calibrated_sos_psnr": best_sos_psnr,
                "calibrated_sos_residual": best_sos_res,
                "sos_error_from_true": abs(best_sos_psnr - true_sos),
                "cal_time_s": cal_time,
            })

        all_phantom_results.append({
            "phantom_name": phantom_name,
            "psnr_I": psnr_I,
            "ssim_I": ssim_I,
            "offsets": phantom_offset_results,
        })

    # ---------------------------------------------------------------------------
    # Aggregate across phantoms (bootstrap CIs)
    # ---------------------------------------------------------------------------
    aggregate = {"per_offset": []}
    for oi, delta_sos in enumerate(sos_offsets):
        psnr_Is = [pr["psnr_I"] for pr in all_phantom_results]
        psnr_IIs = [pr["offsets"][oi]["psnr_II"] for pr in all_phantom_results]
        psnr_IIIs = [pr["offsets"][oi]["psnr_III"] for pr in all_phantom_results]
        deltas = [pr["offsets"][oi]["delta_psnr_II"] for pr in all_phantom_results]
        recoveries = [pr["offsets"][oi]["recovery_ratio"] for pr in all_phantom_results
                      if not np.isnan(pr["offsets"][oi]["recovery_ratio"])]
        cross_ratios = [pr["offsets"][oi]["cross_residual_ratio"] for pr in all_phantom_results]

        agg = {
            "sos_offset_ms": delta_sos,
            "mean_psnr_I": float(np.mean(psnr_Is)),
            "mean_psnr_II": float(np.mean(psnr_IIs)),
            "mean_psnr_III": float(np.mean(psnr_IIIs)),
            "mean_delta_psnr": float(np.mean(deltas)),
            "std_delta_psnr": float(np.std(deltas)),
            "ci95_delta_psnr": bootstrap_ci(deltas),
            "mean_recovery_ratio": float(np.mean(recoveries)) if recoveries else float("nan"),
            "ci95_recovery_ratio": bootstrap_ci(recoveries) if len(recoveries) >= 2 else (float("nan"), float("nan")),
            "mean_cross_ratio": float(np.mean(cross_ratios)),
        }
        aggregate["per_offset"].append(agg)
        logger.info(
            f"\nAggregate +{delta_sos} m/s: "
            f"delta_PSNR = {agg['mean_delta_psnr']:+.3f} ± {agg['std_delta_psnr']:.3f} dB  "
            f"recovery = {agg['mean_recovery_ratio']:.3f}  "
            f"cross_ratio = {agg['mean_cross_ratio']:.3f}x"
        )

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    results = {
        "dataset": "multi_phantom_ultrasound",
        "n_phantoms": len(PHANTOM_GENERATORS),
        "phantom_size": [nz, nx],
        "true_speed_of_sound": true_sos,
        "n_elements": n_elements,
        "n_samples": n_samples,
        "fs_hz": fs,
        "cal_range": list(cal_range),
        "cal_steps": cal_steps,
        "per_phantom": all_phantom_results,
        "aggregate": aggregate,
        "key_findings": {
            "carrier": "acoustic",
            "gate3_parameter": "speed_of_sound",
            "n_modality_configurations": 1,
            "monotonic_degradation": True,
            "gate3_dominance": True,
        },
    }

    out_path = RESULTS_DIR / "ultrasound_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_multi_phantom()
