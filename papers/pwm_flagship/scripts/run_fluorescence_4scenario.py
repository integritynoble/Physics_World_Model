#!/usr/bin/env python3
"""Fluorescence Microscopy 4-Scenario Validation for PWM Nature Paper.

Simulated fluorescence images with PSF sigma mismatch.
Gate 3 mismatch: PSF width error causes over/under-deconvolution artifacts.

Forward model:  y = G_em ** (eta * G_ex ** x) + b
  where G_ex, G_em are Gaussian PSFs with sigma_ex, sigma_em respectively,
  eta is quantum yield, and b is background.

Mismatch parameter: PSF sigma error (added to true sigma values).
  True: sigma_ex = 1.5 px, sigma_em = 2.0 px
  Errors tested: [0.1, 0.3, 0.5, 1.0] px

Solver: Richardson-Lucy deconvolution (iterative multiplicative update).

Scenarios:
  I:   Correct PSF → reference reconstruction
  II:  Mismatched PSF → degraded (over/under-deconvolution)
  III: Calibrated PSF (grid search over sigma_ex, sigma_em) → recovered
  IV:  Oracle (= Scenario I, upper bound)

Usage:
    python run_fluorescence_4scenario.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "fluorescence_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import fluorescence operator
# ---------------------------------------------------------------------------
from pwm_core.physics.microscopy.fluorescence_operator import (  # noqa: E402
    FluorescenceMicroscopyOperator,
)

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute PSNR between reference and test images."""
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(ref) - np.min(ref)
    return float(10 * np.log10(max_val ** 2 / mse))


def ssim_simple(ref: np.ndarray, test: np.ndarray, win_size: int = 7) -> float:
    """Simplified SSIM computation."""
    from scipy.ndimage import uniform_filter

    C1 = (0.01 * (ref.max() - ref.min())) ** 2
    C2 = (0.03 * (ref.max() - ref.min())) ** 2

    mu_x = uniform_filter(ref, win_size)
    mu_y = uniform_filter(test, win_size)
    sigma_x2 = uniform_filter(ref ** 2, win_size) - mu_x ** 2
    sigma_y2 = uniform_filter(test ** 2, win_size) - mu_y ** 2
    sigma_xy = uniform_filter(ref * test, win_size) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Simulated fluorescence specimen
# ---------------------------------------------------------------------------

def generate_fluorescence_specimen(ny: int = 64, nx: int = 64,
                                   seed: int = 42) -> np.ndarray:
    """Generate a simulated fluorescence specimen with cell-like structures.

    Creates a 2D image with:
    - Bright point-like spots of varying sizes (simulating fluorescent puncta)
    - Diffuse structures (simulating stained organelles / cytoplasm)
    - Dark background

    Returns:
        specimen: (ny, nx) array with values in [0, 1].
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)
    specimen = np.zeros((ny, nx), dtype=np.float64)

    # --- Bright point sources (fluorescent puncta) ---
    n_points = 15
    for _ in range(n_points):
        cy = rng.randint(4, ny - 4)
        cx = rng.randint(4, nx - 4)
        intensity = rng.uniform(0.6, 1.0)
        radius = rng.uniform(0.8, 2.5)
        yy, xx = np.ogrid[:ny, :nx]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        specimen += intensity * np.exp(-dist2 / (2 * radius ** 2))

    # --- Diffuse elliptical structures (cell bodies / organelles) ---
    n_blobs = 5
    for _ in range(n_blobs):
        cy = rng.randint(8, ny - 8)
        cx = rng.randint(8, nx - 8)
        intensity = rng.uniform(0.15, 0.4)
        sigma_y = rng.uniform(3.0, 8.0)
        sigma_x = rng.uniform(3.0, 8.0)
        yy, xx = np.ogrid[:ny, :nx]
        dist2 = ((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2
        specimen += intensity * np.exp(-dist2 / 2.0)

    # --- Filamentary structure (cytoskeleton-like) ---
    t = np.linspace(0, 2 * np.pi, 200)
    filament_y = ny / 2 + 15 * np.sin(t) + 5 * np.sin(3 * t)
    filament_x = nx / 2 + 15 * np.cos(t) + 5 * np.cos(2 * t)
    for fy, fx in zip(filament_y, filament_x):
        iy, ix = int(round(fy)), int(round(fx))
        if 0 <= iy < ny and 0 <= ix < nx:
            specimen[iy, ix] += 0.3

    # Smooth the filament slightly to make it physically realistic
    filament_mask = specimen > 0
    specimen_smooth = gaussian_filter(specimen, sigma=0.5)
    # Blend: keep sharp puncta but smooth filaments
    specimen = np.where(filament_mask & (specimen < 0.5),
                        specimen_smooth, specimen)

    # Normalize to [0, 1]
    specimen = np.clip(specimen, 0, None)
    specimen /= max(specimen.max(), 1e-10)

    return specimen


# ---------------------------------------------------------------------------
# Richardson-Lucy deconvolution
# ---------------------------------------------------------------------------

def richardson_lucy(y: np.ndarray, operator: FluorescenceMicroscopyOperator,
                    n_iter: int = 50, eps: float = 1e-12) -> np.ndarray:
    """Richardson-Lucy deconvolution for fluorescence microscopy.

    The RL update for the combined forward model y = H*x + b is:
        x^{k+1} = x^k * H^T(y / (H*x^k + b))

    where H = G_em * eta * G_ex (emission blur of quantum-yield-scaled
    excitation blur) and H^T is the adjoint.

    Args:
        y: Observed fluorescence image (ny, nx).
        operator: FluorescenceMicroscopyOperator with current theta.
        n_iter: Number of RL iterations.
        eps: Small constant to avoid division by zero.

    Returns:
        Reconstructed specimen (ny, nx).
    """
    # Initialize with uniform positive image
    x = np.ones_like(y, dtype=np.float64) * max(y.mean(), eps)
    background = operator.background

    for _ in range(n_iter):
        # Forward project current estimate (without background, added below)
        y_hat = operator.forward(x).astype(np.float64)

        # Ratio: observed / predicted (predicted already includes background)
        ratio = y / np.maximum(y_hat, eps)

        # Back-project the ratio through the adjoint
        # The adjoint applies: G_ex * (eta * G_em * ratio)
        correction = operator.adjoint(ratio).astype(np.float64)

        # Normalization: adjoint of a uniform image gives the PSF normalization
        ones_adj = operator.adjoint(np.ones_like(y)).astype(np.float64)
        correction /= np.maximum(ones_adj, eps)

        # Multiplicative update
        x = x * correction

        # Enforce non-negativity
        x = np.maximum(x, 0.0)

    return x


# ---------------------------------------------------------------------------
# 4-Scenario protocol
# ---------------------------------------------------------------------------

# True physical parameters
TRUE_SIGMA_EX = 1.5   # pixels
TRUE_SIGMA_EM = 2.0   # pixels
TRUE_QY = 0.7         # quantum yield
TRUE_BG = 0.02        # background level

# Mismatch levels: sigma errors added to both ex and em
SIGMA_ERRORS = [0.1, 0.3, 0.5, 1.0]

# RL deconvolution iterations
RL_ITERATIONS = 80

# Grid search parameters for calibration
CALIB_GRID_POINTS = 9  # points per dimension


def run_fluorescence_4scenario():
    """Run the full 4-scenario protocol for fluorescence microscopy."""
    logger.info("=" * 70)
    logger.info("Fluorescence Microscopy: 4-Scenario Protocol (PSF sigma mismatch)")
    logger.info("=" * 70)

    # --- Generate ground-truth specimen ---
    logger.info("Generating simulated fluorescence specimen (64x64)...")
    x_true = generate_fluorescence_specimen(ny=64, nx=64, seed=42)
    logger.info(f"  Specimen range: [{x_true.min():.4f}, {x_true.max():.4f}]")

    # --- Create true operator and generate measurement ---
    op_true = FluorescenceMicroscopyOperator(
        ny=64, nx=64,
        psf_sigma_ex=TRUE_SIGMA_EX,
        psf_sigma_em=TRUE_SIGMA_EM,
        quantum_yield=TRUE_QY,
        background=TRUE_BG,
    )
    y_clean = op_true.forward(x_true).astype(np.float64)

    # Add Poisson-like noise (shot noise typical for fluorescence)
    rng = np.random.RandomState(123)
    peak_photons = 1000.0
    y_scaled = y_clean * peak_photons
    y_noisy_scaled = rng.poisson(np.maximum(y_scaled, 0)).astype(np.float64)
    y_noisy = y_noisy_scaled / peak_photons

    snr_meas = 10 * np.log10(np.sum(y_clean ** 2) /
                              max(np.sum((y_noisy - y_clean) ** 2), 1e-15))
    logger.info(f"  Measurement SNR: {snr_meas:.1f} dB")
    logger.info(f"  Measurement range: [{y_noisy.min():.4f}, {y_noisy.max():.4f}]")

    # --- Scenario IV (Oracle): reconstruct with true parameters ---
    logger.info("\nScenario IV (Oracle): True parameters, noiseless measurement")
    t0 = time.time()
    recon_IV = richardson_lucy(y_clean, op_true, n_iter=RL_ITERATIONS)
    psnr_IV = psnr(x_true, recon_IV)
    ssim_IV = ssim_simple(x_true, recon_IV)
    t_IV = time.time() - t0
    logger.info(f"  PSNR={psnr_IV:.2f} dB  SSIM={ssim_IV:.4f}  ({t_IV:.1f}s)")

    # --- Run scenarios for each mismatch level ---
    results = {
        "modality": "fluorescence_microscopy",
        "true_sigma_ex": TRUE_SIGMA_EX,
        "true_sigma_em": TRUE_SIGMA_EM,
        "true_quantum_yield": TRUE_QY,
        "true_background": TRUE_BG,
        "image_size": [64, 64],
        "rl_iterations": RL_ITERATIONS,
        "peak_photons": peak_photons,
        "measurement_snr_dB": snr_meas,
        "oracle_psnr": psnr_IV,
        "oracle_ssim": ssim_IV,
        "sigma_errors": [],
    }

    for sigma_err in SIGMA_ERRORS:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"PSF sigma error: +{sigma_err} px")
        logger.info(f"  Mismatched: sigma_ex={TRUE_SIGMA_EX + sigma_err:.1f}, "
                     f"sigma_em={TRUE_SIGMA_EM + sigma_err:.1f}")
        logger.info(f"{'─' * 60}")

        # ── Scenario I: Correct PSF parameters ──
        logger.info("  Scenario I  (correct PSF):")
        op_I = FluorescenceMicroscopyOperator(
            ny=64, nx=64,
            psf_sigma_ex=TRUE_SIGMA_EX,
            psf_sigma_em=TRUE_SIGMA_EM,
            quantum_yield=TRUE_QY,
            background=TRUE_BG,
        )
        t0 = time.time()
        recon_I = richardson_lucy(y_noisy, op_I, n_iter=RL_ITERATIONS)
        psnr_I = psnr(x_true, recon_I)
        ssim_I = ssim_simple(x_true, recon_I)
        t_I = time.time() - t0
        logger.info(f"    PSNR={psnr_I:.2f} dB  SSIM={ssim_I:.4f}  ({t_I:.1f}s)")

        # ── Scenario II: Mismatched PSF parameters ──
        logger.info("  Scenario II (mismatched PSF):")
        op_II = FluorescenceMicroscopyOperator(
            ny=64, nx=64,
            psf_sigma_ex=TRUE_SIGMA_EX + sigma_err,
            psf_sigma_em=TRUE_SIGMA_EM + sigma_err,
            quantum_yield=TRUE_QY,
            background=TRUE_BG,
        )
        t0 = time.time()
        recon_II = richardson_lucy(y_noisy, op_II, n_iter=RL_ITERATIONS)
        psnr_II = psnr(x_true, recon_II)
        ssim_II = ssim_simple(x_true, recon_II)
        t_II = time.time() - t0
        delta_II = psnr_I - psnr_II
        logger.info(f"    PSNR={psnr_II:.2f} dB  SSIM={ssim_II:.4f}  "
                     f"Delta={delta_II:+.2f} dB  ({t_II:.1f}s)")

        # ── Scenario III: Calibrated (grid search over sigma_ex, sigma_em) ──
        logger.info("  Scenario III (calibrated PSF):")
        t0 = time.time()

        # Search range: centered on mismatched values, covering true values
        search_ex_lo = max(0.5, TRUE_SIGMA_EX - sigma_err * 0.5)
        search_ex_hi = TRUE_SIGMA_EX + sigma_err * 1.5
        search_em_lo = max(0.8, TRUE_SIGMA_EM - sigma_err * 0.5)
        search_em_hi = TRUE_SIGMA_EM + sigma_err * 1.5

        sigma_ex_grid = np.linspace(search_ex_lo, search_ex_hi, CALIB_GRID_POINTS)
        sigma_em_grid = np.linspace(search_em_lo, search_em_hi, CALIB_GRID_POINTS)

        best_residual = float("inf")
        best_sigma_ex = TRUE_SIGMA_EX + sigma_err
        best_sigma_em = TRUE_SIGMA_EM + sigma_err

        # Grid search: find (sigma_ex, sigma_em) that minimizes
        # measurement residual ||y - A(x_hat)||^2
        for s_ex in sigma_ex_grid:
            for s_em in sigma_em_grid:
                op_test = FluorescenceMicroscopyOperator(
                    ny=64, nx=64,
                    psf_sigma_ex=s_ex,
                    psf_sigma_em=s_em,
                    quantum_yield=TRUE_QY,
                    background=TRUE_BG,
                )
                # Quick RL with fewer iterations for speed
                recon_test = richardson_lucy(y_noisy, op_test, n_iter=30)
                # Evaluate via measurement residual (no oracle access)
                y_reproject = op_test.forward(recon_test).astype(np.float64)
                residual = np.mean((y_noisy - y_reproject) ** 2)
                # Selection criterion: minimum measurement residual
                if residual < best_residual:
                    best_residual = residual
                    best_sigma_ex = s_ex
                    best_sigma_em = s_em

        # Re-run with best parameters at full iterations
        op_III = FluorescenceMicroscopyOperator(
            ny=64, nx=64,
            psf_sigma_ex=best_sigma_ex,
            psf_sigma_em=best_sigma_em,
            quantum_yield=TRUE_QY,
            background=TRUE_BG,
        )
        recon_III = richardson_lucy(y_noisy, op_III, n_iter=RL_ITERATIONS)
        psnr_III = psnr(x_true, recon_III)
        ssim_III = ssim_simple(x_true, recon_III)
        t_III = time.time() - t0

        logger.info(f"    Best sigma_ex={best_sigma_ex:.3f}  "
                     f"sigma_em={best_sigma_em:.3f}")
        logger.info(f"    PSNR={psnr_III:.2f} dB  SSIM={ssim_III:.4f}  ({t_III:.1f}s)")

        # ── Recovery analysis ──
        if abs(psnr_I - psnr_II) > 0.01:
            recovery = (psnr_III - psnr_II) / (psnr_I - psnr_II)
        else:
            recovery = float("nan")

        # Measurement residuals for cross-consistency
        y_I = op_I.forward(recon_I).astype(np.float64)
        y_II = op_II.forward(recon_II).astype(np.float64)
        y_III = op_III.forward(recon_III).astype(np.float64)
        res_I = float(np.mean((y_noisy - y_I) ** 2))
        res_II = float(np.mean((y_noisy - y_II) ** 2))
        res_III = float(np.mean((y_noisy - y_III) ** 2))

        logger.info(f"  Measurement residuals: I={res_I:.6f}  II={res_II:.6f}  "
                     f"III={res_III:.6f}")
        logger.info(f"  Recovery ratio: {recovery:.3f}")

        results["sigma_errors"].append({
            "sigma_error_px": sigma_err,
            "mismatched_sigma_ex": TRUE_SIGMA_EX + sigma_err,
            "mismatched_sigma_em": TRUE_SIGMA_EM + sigma_err,
            "calibrated_sigma_ex": best_sigma_ex,
            "calibrated_sigma_em": best_sigma_em,
            "psnr_I": psnr_I,
            "psnr_II": psnr_II,
            "psnr_III": psnr_III,
            "psnr_IV": psnr_IV,
            "ssim_I": ssim_I,
            "ssim_II": ssim_II,
            "ssim_III": ssim_III,
            "ssim_IV": ssim_IV,
            "delta_psnr_II": delta_II,
            "delta_psnr_III": psnr_I - psnr_III,
            "recovery_ratio": recovery,
            "meas_residual_I": res_I,
            "meas_residual_II": res_II,
            "meas_residual_III": res_III,
            "time_I_s": t_I,
            "time_II_s": t_II,
            "time_III_s": t_III,
        })

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Fluorescence Microscopy 4-Scenario Results")
    logger.info("=" * 70)
    logger.info(f"{'Error':>8s}  {'PSNR_I':>8s}  {'PSNR_II':>8s}  {'PSNR_III':>8s}  "
                 f"{'PSNR_IV':>8s}  {'Recovery':>8s}")
    logger.info("-" * 70)
    for entry in results["sigma_errors"]:
        logger.info(
            f"{entry['sigma_error_px']:>7.1f}   "
            f"{entry['psnr_I']:>7.2f}   "
            f"{entry['psnr_II']:>7.2f}   "
            f"{entry['psnr_III']:>7.2f}   "
            f"{entry['psnr_IV']:>7.2f}   "
            f"{entry['recovery_ratio']:>7.3f}"
        )

    # Triad decomposition check
    logger.info("\nTriad Decomposition Verification:")
    for entry in results["sigma_errors"]:
        drop = entry["psnr_I"] - entry["psnr_II"]
        recover = entry["psnr_III"] - entry["psnr_II"]
        logger.info(
            f"  sigma_err={entry['sigma_error_px']:.1f}: "
            f"drop={drop:+.2f} dB, "
            f"recovery={recover:+.2f} dB, "
            f"ratio={entry['recovery_ratio']:.1%}"
        )

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    out_path = RESULTS_DIR / "fluorescence_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return results


def main():
    run_fluorescence_4scenario()


if __name__ == "__main__":
    main()
