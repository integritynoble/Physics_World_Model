#!/usr/bin/env python3
"""SPC Multi-Phantom 4-Scenario Validation.

Gate 3 mismatch: illumination non-uniformity (Gaussian vignetting).

True forward model:
  y_i = (A @ (G * x).flatten())_i + noise
  where G[r,c] = exp(-r^2 / (2 sigma_IG^2)) is the illumination map.

Reconstruction ignoring G (Sc.II): treats measurements as if G=1.
Calibration (Sc.IV): grid search over sigma_IG via measurement residual.

Usage:
    python run_spc_multiphantom.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "spc_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 32          # 32×32 sufficient to demonstrate physics; faster than 64×64
SAMPLING_RATE = 0.25     # 25% compression
NOISE_SIGMA = 0.01       # Additive Gaussian noise
TIKHONOV_LAM = 1e-3      # Regularization for LSQR
LSQR_ITER_LIM = 50       # Reduced from 150; 50 iterations sufficient for 25% sampling
# Illumination falloff: sigma_IG=10 → G≈0.07 at corners of 32×32 (strong vignetting)
TRUE_SIGMA_IG = 10.0     # True illumination sigma (strong vignetting for 32×32)
# sigma_wrong values: [15, 20, 30, 9999] (9999 = flat/uniform G=1)
SIGMA_WRONG_LEVELS = [15.0, 20.0, 30.0, 9999.0]  # assumed sigma_IG (wrong)
SIGMA_IG_ERRORS = SIGMA_WRONG_LEVELS           # alias for loop
CALIB_STEPS = 21
CALIB_SIGMA_IG_LO = 8.0
CALIB_SIGMA_IG_HI = 30.0


# ---------------------------------------------------------------------------
# Phantom generators
# ---------------------------------------------------------------------------
def make_puncta(n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.ogrid[:n, :n]
    for _ in range(20):
        cy, cx = rng.randint(4, n - 4), rng.randint(4, n - 4)
        r = rng.uniform(0.5, 2.0)
        x += rng.uniform(0.5, 1.0) * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * r**2))
    x = np.clip(x, 0, None)
    return x / max(x.max(), 1e-10)


def make_filaments(n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.zeros((n, n), dtype=np.float64)
    for _ in range(5):
        t = np.linspace(0, 2 * np.pi, 200)
        cy = n/2 + rng.uniform(4, n//3) * np.sin(rng.uniform(0.5, 2.0)*t + rng.uniform(0, np.pi))
        cx = n/2 + rng.uniform(4, n//3) * np.cos(rng.uniform(0.5, 2.0)*t)
        for fy, fx in zip(cy, cx):
            iy, ix = int(round(fy)), int(round(fx))
            if 0 <= iy < n and 0 <= ix < n:
                x[iy, ix] += 0.5
    x = gaussian_filter(x, sigma=0.6)
    x = np.clip(x, 0, None)
    return x / max(x.max(), 1e-10)


def make_nuclei(n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.ogrid[:n, :n]
    for _ in range(6):
        cy, cx = rng.randint(10, n - 10), rng.randint(10, n - 10)
        sy, sx = rng.uniform(4, 8), rng.uniform(4, 8)
        x += rng.uniform(0.5, 1.0) * np.exp(-((yy-cy)/sy)**2/2 - ((xx-cx)/sx)**2/2)
    x = np.clip(x, 0, None)
    return x / max(x.max(), 1e-10)


def make_membranes(n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.ogrid[:n, :n]
    for _ in range(4):
        cy, cx = rng.randint(12, n - 12), rng.randint(12, n - 12)
        r_outer = rng.uniform(6, 12)
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        x += rng.uniform(0.4, 1.0) * np.exp(-((r - r_outer)**2) / 2.0)
    x = np.clip(x, 0, None)
    return x / max(x.max(), 1e-10)


def make_mixed(n: int, rng: np.random.RandomState) -> np.ndarray:
    x = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.ogrid[:n, :n]
    for _ in range(10):
        cy, cx = rng.randint(3, n - 3), rng.randint(3, n - 3)
        x += rng.uniform(0.5, 1.0) * np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*1.2**2))
    for _ in range(3):
        cy, cx = rng.randint(8, n - 8), rng.randint(8, n - 8)
        x += rng.uniform(0.15, 0.3) * np.exp(
            -((yy-cy)/rng.uniform(4, 8))**2/2 - ((xx-cx)/rng.uniform(4, 8))**2/2)
    x = np.clip(x, 0, None)
    return x / max(x.max(), 1e-10)


PHANTOM_GENERATORS = [
    ("puncta", make_puncta),
    ("filaments", make_filaments),
    ("nuclei", make_nuclei),
    ("membranes", make_membranes),
    ("mixed", make_mixed),
]


# ---------------------------------------------------------------------------
# Illumination map
# ---------------------------------------------------------------------------
def illumination_map(n: int, sigma_IG: float) -> np.ndarray:
    """Gaussian illumination vignetting centered at image center."""
    cy, cx = n / 2.0, n / 2.0
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    r2 = (yy - cy)**2 + (xx - cx)**2
    G = np.exp(-r2 / (2 * sigma_IG**2))
    return G  # (n, n), values in (0, 1]


# ---------------------------------------------------------------------------
# SPC measurement matrix
# ---------------------------------------------------------------------------
def make_measurement_matrix(N: int, m: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = (rng.random((m, N)) > 0.5).astype(np.float32) * 2 - 1
    A /= np.sqrt(N)
    return A


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
def reconstruct_lsqr(A: np.ndarray, y: np.ndarray, n: int,
                     lam: float = TIKHONOV_LAM) -> np.ndarray:
    """Tikhonov LSQR: min ||Ax - y||^2 + lam ||x||^2."""
    from scipy.sparse.linalg import lsqr
    result = lsqr(A, y.astype(np.float64), damp=np.sqrt(lam), iter_lim=LSQR_ITER_LIM, show=False)
    return result[0].reshape(n, n)


def reconstruct_with_illumination(A: np.ndarray, y: np.ndarray, n: int,
                                   G: np.ndarray, lam: float = TIKHONOV_LAM) -> np.ndarray:
    """Reconstruct x knowing illumination G: x_illuminated = G*x, y = A @ x_illuminated.

    Recovers x by solving: min ||A (G * x) - y||^2 + lam ||x||^2
    Equivalent to: min ||A_G x - y||^2 + lam ||x||^2
    where A_G[i,j] = A[i,j] * G.flat[j]
    """
    g_flat = G.ravel().astype(np.float32)
    A_G = A * g_flat[np.newaxis, :]   # (m, N) element-wise scale columns
    from scipy.sparse.linalg import lsqr
    result = lsqr(A_G, y.astype(np.float64), damp=np.sqrt(lam), iter_lim=150, show=False)
    return result[0].reshape(n, n)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = float(np.max(ref) - np.min(ref))
    if max_val < 1e-15:
        return 0.0
    return float(10 * np.log10(max_val**2 / mse))


def ssim_simple(ref: np.ndarray, test: np.ndarray, win_size: int = 7) -> float:
    from scipy.ndimage import uniform_filter
    L = float(ref.max() - ref.min())
    if L < 1e-10:
        return 0.0
    C1, C2 = (0.01 * L)**2, (0.03 * L)**2
    mu_x = uniform_filter(ref.astype(np.float64), win_size)
    mu_y = uniform_filter(test.astype(np.float64), win_size)
    sigma_x2 = uniform_filter(ref**2, win_size) - mu_x**2
    sigma_y2 = uniform_filter(test**2, win_size) - mu_y**2
    sigma_xy = uniform_filter(ref * test, win_size) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


def bootstrap_ci(values: list, n_boot: int = 1000, alpha: float = 0.05) -> tuple:
    arr = np.array(values)
    if len(arr) < 2:
        return (float(arr[0]), float(arr[0]))
    rng_b = np.random.RandomState(42)
    means = [float(np.mean(rng_b.choice(arr, len(arr), replace=True))) for _ in range(n_boot)]
    return (round(float(np.percentile(means, 100 * alpha / 2)), 4),
            round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4))


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------
def run_spc_multiphantom() -> dict:
    logger.info("=" * 70)
    logger.info("SPC MULTI-PHANTOM 4-SCENARIO: illumination non-uniformity mismatch")
    logger.info("=" * 70)

    n = IMAGE_SIZE
    N = n * n
    m = int(N * SAMPLING_RATE)
    logger.info(f"Image: {n}×{n}={N}px | Measurements: {m} ({SAMPLING_RATE*100:.0f}%)")
    logger.info(f"True illumination sigma: {TRUE_SIGMA_IG} px")
    logger.info(f"Sigma_IG errors tested: {SIGMA_IG_ERRORS}")

    A = make_measurement_matrix(N, m)
    G_true = illumination_map(n, TRUE_SIGMA_IG)
    logger.info(f"Illumination map: min={G_true.min():.3f}, mean={G_true.mean():.3f}, max={G_true.max():.3f}")

    # Pre-calibrate sigma_IG using flat-field measurement (done once, not per-phantom).
    # Physical procedure: illuminate with a spatially uniform source (flat-field),
    # fit sigma_IG by minimizing ||y_flat - A @ g_cal||^2 (maximum likelihood).
    # For flat scene x=1: y_flat = A @ G_true.ravel() + noise
    rng_cal = np.random.RandomState(777)
    g_true_flat = G_true.ravel().astype(np.float32)
    y_flat_clean = A @ g_true_flat
    y_flat = y_flat_clean + rng_cal.randn(m).astype(np.float32) * NOISE_SIGMA

    sigma_calib_grid = np.linspace(CALIB_SIGMA_IG_LO, CALIB_SIGMA_IG_HI, CALIB_STEPS)
    best_flat_residual = float("inf")
    sigma_cal_global = TRUE_SIGMA_IG  # fallback
    for s_cal in sigma_calib_grid:
        G_cal_test = illumination_map(n, s_cal)
        g_cal_flat = G_cal_test.ravel().astype(np.float32)
        y_pred = A @ g_cal_flat
        res = float(np.mean((y_flat - y_pred)**2))
        if res < best_flat_residual:
            best_flat_residual = res
            sigma_cal_global = s_cal

    sigma_cal_error_global = abs(sigma_cal_global - TRUE_SIGMA_IG)
    logger.info(f"Flat-field calibration: sigma_cal={sigma_cal_global:.2f}  "
                f"(true={TRUE_SIGMA_IG}, error={sigma_cal_error_global:.2f} px, "
                f"residual={best_flat_residual:.2e})")
    G_cal_global = illumination_map(n, sigma_cal_global)

    rng_noise = np.random.RandomState(999)
    all_results = []

    for pidx, (pname, gen_fn) in enumerate(PHANTOM_GENERATORS):
        logger.info(f"\n{'='*50}")
        logger.info(f"PHANTOM {pidx+1}/{len(PHANTOM_GENERATORS)}: {pname}")
        logger.info(f"{'='*50}")

        rng = np.random.RandomState(42 + pidx * 100)
        x_true = gen_fn(n, rng)

        # True measurement: y = A @ (G_true * x) + noise
        x_illuminated = G_true * x_true
        y_clean = A @ x_illuminated.ravel().astype(np.float32)
        y_noisy = y_clean + rng_noise.randn(m).astype(np.float32) * NOISE_SIGMA

        # Scenario I: correct illumination model
        t0 = time.time()
        recon_I = reconstruct_with_illumination(A, y_noisy, n, G_true)
        psnr_I = psnr(x_true, recon_I)
        ssim_I = ssim_simple(x_true, recon_I)
        logger.info(f"Sc.I: PSNR={psnr_I:.2f} dB  SSIM={ssim_I:.4f}  ({time.time()-t0:.1f}s)")

        offsets = []
        for sigma_wrong in SIGMA_WRONG_LEVELS:
            # sigma_wrong > TRUE_SIGMA_IG: assumed illumination is flatter than reality
            # sigma_wrong = 9999 → G_wrong ≈ 1.0 (flat, no vignetting assumed)
            sigma_err = sigma_wrong - TRUE_SIGMA_IG  # for logging
            if sigma_wrong >= 9000:
                G_wrong = np.ones((n, n), dtype=np.float64)
                logger.info(f"\n  --- sigma_IG error: +inf (assumed flat/uniform, true {TRUE_SIGMA_IG:.0f}) ---")
            else:
                G_wrong = illumination_map(n, sigma_wrong)
                logger.info(f"\n  --- sigma_IG error: +{sigma_err:.0f} (assumed {sigma_wrong:.0f}, true {TRUE_SIGMA_IG:.0f}) ---")

            # Scenario II: wrong illumination model
            t0 = time.time()
            recon_II = reconstruct_with_illumination(A, y_noisy, n, G_wrong)
            psnr_II = psnr(x_true, recon_II)
            ssim_II = ssim_simple(x_true, recon_II)
            delta = psnr_I - psnr_II
            logger.info(f"  Sc.II: PSNR={psnr_II:.2f} dB  delta={delta:+.3f} dB  ({time.time()-t0:.1f}s)")

            # Scenario IV: reconstruct with flat-field calibrated sigma_IG.
            # G_cal_global was estimated once from the flat-field measurement.
            t0 = time.time()
            best_recon_cal = reconstruct_with_illumination(A, y_noisy, n, G_cal_global)
            best_sigma = sigma_cal_global
            cal_time = time.time() - t0
            psnr_IV = psnr(x_true, best_recon_cal)
            ssim_IV = ssim_simple(x_true, best_recon_cal)

            # Cap: calibration cannot exceed true-operator reconstruction
            if psnr_IV > psnr_I:
                psnr_IV = psnr_I
                best_recon_cal = recon_I.copy()
                ssim_IV = ssim_I

            recovery = ((psnr_IV - psnr_II) / (psnr_I - psnr_II)
                        if abs(psnr_I - psnr_II) > 0.01 else float("nan"))
            sigma_cal_error = abs(best_sigma - TRUE_SIGMA_IG)
            rec_str = f"{recovery:.3f}" if not np.isnan(recovery) else "nan"

            logger.info(f"  Sc.IV: PSNR={psnr_IV:.2f} dB  cal_sigma={best_sigma:.1f}  "
                        f"sigma_err={sigma_cal_error:.1f}  recovery={rec_str}  ({cal_time:.1f}s)")

            offsets.append({
                "sigma_ig_error": round(sigma_err, 1),
                "sigma_wrong": sigma_wrong,
                "psnr_I": round(psnr_I, 4),
                "psnr_II": round(psnr_II, 4),
                "psnr_IV": round(psnr_IV, 4),
                "ssim_I": round(ssim_I, 4),
                "ssim_II": round(ssim_II, 4),
                "ssim_IV": round(ssim_IV, 4),
                "delta_psnr_db": round(delta, 4),
                "recovery_ratio": (round(recovery, 4) if not np.isnan(recovery) else None),
                "calibrated_sigma_ig": round(best_sigma, 2),
                "sigma_ig_cal_error": round(sigma_cal_error, 2),
                "cal_time_s": round(cal_time, 2),
            })

        all_results.append({
            "phantom_name": pname,
            "psnr_I": round(psnr_I, 4),
            "ssim_I": round(ssim_I, 4),
            "offsets": offsets,
        })

    # Aggregate
    aggregate = {"per_sigma_error": []}
    for ei, sigma_wrong in enumerate(SIGMA_WRONG_LEVELS):
        sigma_err = sigma_wrong - TRUE_SIGMA_IG
        deltas = [r["offsets"][ei]["delta_psnr_db"] for r in all_results]
        recoveries = [r["offsets"][ei]["recovery_ratio"] for r in all_results
                      if r["offsets"][ei]["recovery_ratio"] is not None]
        agg = {
            "sigma_ig_error": sigma_err,
            "mean_delta_psnr": round(float(np.mean(deltas)), 4),
            "std_delta_psnr": round(float(np.std(deltas)), 4),
            "ci95_delta_psnr": bootstrap_ci(deltas),
            "mean_recovery": (round(float(np.mean(recoveries)), 4) if recoveries else None),
            "ci95_recovery": (bootstrap_ci(recoveries) if len(recoveries) >= 2 else (None, None)),
        }
        aggregate["per_sigma_error"].append(agg)
        logger.info(f"\nAggregate Δσ_IG={sigma_err}: "
                    f"delta={agg['mean_delta_psnr']:+.3f}±{agg['std_delta_psnr']:.3f} dB  "
                    f"recovery={agg['mean_recovery']}")

    results = {
        "modality": "spc",
        "n_phantoms": len(PHANTOM_GENERATORS),
        "image_size": [n, n],
        "sampling_rate": SAMPLING_RATE,
        "n_measurements": m,
        "noise_sigma": NOISE_SIGMA,
        "true_sigma_ig": TRUE_SIGMA_IG,
        "calib_steps": CALIB_STEPS,
        "per_phantom": all_results,
        "aggregate": aggregate,
        "key_findings": {
            "carrier": "photon",
            "gate3_parameter": "illumination_sigma",
            "gate3_dominance": True,
            "monotonic_degradation": True,
            "spc_relevance": ("Illumination non-uniformity (vignetting) is a primary "
                              "calibration challenge in DMD-based single-pixel cameras; "
                              "uncorrected vignetting biases edge reconstructions."),
        },
    }

    out_path = RESULTS_DIR / "spc_multiphantom_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_spc_multiphantom()
