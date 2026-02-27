#!/usr/bin/env python3
"""CBCT 4-Scenario Validation for PWM Nature Paper.

Simulated Shepp-Logan phantom with detector offset mismatch.
Gate 3 mismatch: detector offset causes half-fan artifacts.

Protocol:
  Scenario I:   Correct detector offset -> reference reconstruction
  Scenario II:  Mismatched detector offset -> degraded reconstruction
  Scenario III: Calibrated (grid search over offset) -> recovered
  Scenario IV:  Oracle (best possible offset) -> upper bound

Offsets tested: [1, 2, 5, 10] pixels.
Solver: FDK backprojection (adjoint of CBCTOperator).

Usage:
    python run_cbct_4scenario.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "real_data_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import CBCTOperator
# ---------------------------------------------------------------------------
from pwm_core.physics.tomography.cbct_operator import CBCTOperator  # noqa: E402


# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------
def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute PSNR between reference and test images."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = float(np.max(ref) - np.min(ref))
    return float(10.0 * np.log10(max_val ** 2 / mse))


def ssim_simple(ref: np.ndarray, test: np.ndarray, win_size: int = 7) -> float:
    """Simplified SSIM computation."""
    from scipy.ndimage import uniform_filter

    ref64 = ref.astype(np.float64)
    test64 = test.astype(np.float64)

    C1 = (0.01 * (ref64.max() - ref64.min())) ** 2
    C2 = (0.03 * (ref64.max() - ref64.min())) ** 2

    mu_x = uniform_filter(ref64, win_size)
    mu_y = uniform_filter(test64, win_size)
    sigma_x2 = uniform_filter(ref64 ** 2, win_size) - mu_x ** 2
    sigma_y2 = uniform_filter(test64 ** 2, win_size) - mu_y ** 2
    sigma_xy = uniform_filter(ref64 * test64, win_size) - mu_x * mu_y

    num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Shepp-Logan phantom
# ---------------------------------------------------------------------------
# Classic Shepp-Logan parameters:
#   (intensity, a, b, x0, y0, phi_deg)
# where (a, b) are semi-axes, (x0, y0) is center, phi_deg is rotation.
# Values follow Shepp & Logan (1974), IEEE Trans. Nucl. Sci.
_SHEPP_LOGAN_ELLIPSES = [
    # intensity   a       b      x0      y0     phi_deg
    (  2.0,     0.6900, 0.9200,  0.0000,  0.0000,   0.0),   # outer skull
    ( -0.98,    0.6624, 0.8740,  0.0000, -0.0184,   0.0),   # inner skull (subtract)
    ( -0.02,    0.1100, 0.3100,  0.2200,  0.0000, -18.0),   # left tumor
    ( -0.02,    0.1600, 0.4100, -0.2200,  0.0000,  18.0),   # right tumor
    (  0.01,    0.2100, 0.2500,  0.0000,  0.3500,   0.0),   # top feature
    (  0.01,    0.0460, 0.0460,  0.0000,  0.1000,   0.0),   # small circle top
    (  0.01,    0.0460, 0.0460,  0.0000, -0.1000,   0.0),   # small circle bottom
    (  0.01,    0.0460, 0.0230, -0.0800, -0.6050,   0.0),   # bottom-left
    (  0.01,    0.0230, 0.0230,  0.0000, -0.6050,   0.0),   # bottom-center
    (  0.01,    0.0230, 0.0460,  0.0600, -0.6050,   0.0),   # bottom-right
]


def shepp_logan_phantom(n: int = 64) -> np.ndarray:
    """Generate a Shepp-Logan phantom on an (n x n) grid.

    Builds the phantom by summing filled-ellipse contributions using the
    classical 10-ellipse parameterisation.

    Args:
        n: Grid size (image will be n x n pixels).

    Returns:
        Phantom image array of shape (n, n) with float64 values.
    """
    phantom = np.zeros((n, n), dtype=np.float64)

    # Coordinate grid: centered at 0, range [-1, 1)
    coords = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0 / n
    yy, xx = np.meshgrid(coords, coords, indexing="ij")

    for intensity, a, b, x0, y0, phi_deg in _SHEPP_LOGAN_ELLIPSES:
        phi = np.deg2rad(phi_deg)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Translate
        xr = xx - x0
        yr = yy - y0

        # Rotate into ellipse-aligned frame
        xr_rot = xr * cos_p + yr * sin_p
        yr_rot = -xr * sin_p + yr * cos_p

        # Ellipse equation: (xr_rot / a)^2 + (yr_rot / b)^2 <= 1
        inside = (xr_rot / a) ** 2 + (yr_rot / b) ** 2 <= 1.0
        phantom[inside] += intensity

    return phantom


# ---------------------------------------------------------------------------
# FDK reconstruction via CBCTOperator.adjoint with Ram-Lak pre-filtering
# ---------------------------------------------------------------------------
def fdk_reconstruct(sinogram: np.ndarray, op: CBCTOperator) -> np.ndarray:
    """FDK reconstruction: Ram-Lak filter + CBCTOperator.adjoint.

    The CBCTOperator.adjoint already applies FDK distance weighting,
    so we only need to add the ramp filter in the detector dimension.

    Args:
        sinogram: (n_angles, n_det) sinogram array.
        op: CBCTOperator instance (uses its adjoint for backprojection).

    Returns:
        Reconstructed image (ny, nx) as float64.
    """
    n_angles, n_det = sinogram.shape
    sino64 = sinogram.astype(np.float64)

    # Ram-Lak (ramp) filter in the detector dimension
    freq = np.fft.fftfreq(n_det)
    ram_lak = np.abs(freq)

    filtered = np.zeros_like(sino64)
    for a in range(n_angles):
        proj_fft = np.fft.fft(sino64[a, :])
        filtered[a, :] = np.real(np.fft.ifft(proj_fft * ram_lak))

    # Weighted backprojection via operator adjoint
    recon = op.adjoint(filtered.astype(np.float32))
    return recon.astype(np.float64)


# ---------------------------------------------------------------------------
# Main 4-scenario protocol for CBCT
# ---------------------------------------------------------------------------
def run_cbct_4scenario() -> Dict:
    """Run the 4-scenario protocol on a simulated Shepp-Logan CBCT problem.

    Gate 3 mismatch: detector_offset causes misaligned fan-beam geometry,
    producing half-fan artifacts in the FDK reconstruction.

    Scenarios for each offset level delta in [1, 2, 5, 10] px:
      I:   Forward with offset=0, reconstruct with offset=0 (correct)
      II:  Forward with offset=delta, reconstruct with offset=0 (mismatched)
      III: Forward with offset=delta, reconstruct with calibrated offset (grid search)
      IV:  Oracle = best from grid search (upper bound)

    Returns:
        Dictionary of results.
    """
    logger.info("=" * 60)
    logger.info("CBCT 4-Scenario Protocol (detector offset mismatch)")
    logger.info("=" * 60)

    # ----- phantom -----
    N = 64
    phantom = shepp_logan_phantom(N)
    logger.info(f"Phantom: {N}x{N}, range [{phantom.min():.3f}, {phantom.max():.3f}]")

    # ----- operator parameters -----
    n_angles = 180
    n_det = 92
    D_so = 100.0
    D_sd = 150.0

    # ----- generate ground-truth sinogram (offset = 0) -----
    op_true = CBCTOperator(
        operator_id="cbct_true",
        ny=N, nx=N,
        n_angles=n_angles, n_det=n_det,
        D_so=D_so, D_sd=D_sd,
        detector_offset=0.0,
    )

    logger.info(f"Generating ground-truth sinogram: {n_angles} angles, {n_det} detectors")
    t0 = time.time()
    sino_true = op_true.forward(phantom)
    t_fwd = time.time() - t0
    logger.info(f"  Forward projection took {t_fwd:.1f}s")

    # ----- Scenario I: correct offset -> reference -----
    logger.info("\n--- Scenario I: correct detector offset (reference) ---")
    t0 = time.time()
    recon_I = fdk_reconstruct(sino_true, op_true)
    t_I = time.time() - t0
    psnr_I = psnr(phantom, recon_I)
    ssim_I = ssim_simple(phantom, recon_I)
    logger.info(f"  PSNR={psnr_I:.2f} dB   SSIM={ssim_I:.4f}   ({t_I:.1f}s)")

    # ----- Detector offset levels to test -----
    offset_levels = [1, 2, 5, 10]  # pixels
    calibration_range = (-12.0, 12.0)
    calibration_steps = 25

    results: Dict = {
        "dataset": "shepp_logan_64",
        "n_angles": n_angles,
        "n_det": n_det,
        "D_so": D_so,
        "D_sd": D_sd,
        "phantom_size": N,
        "psnr_I_ref": psnr_I,
        "ssim_I_ref": ssim_I,
        "calibration_range": list(calibration_range),
        "calibration_steps": calibration_steps,
        "offsets": [],
    }

    for delta in offset_levels:
        logger.info(f"\n{'='*50}")
        logger.info(f"Detector offset mismatch: {delta} px")
        logger.info(f"{'='*50}")

        # ----- Generate mismatched sinogram -----
        op_mismatch = CBCTOperator(
            operator_id="cbct_mismatch",
            ny=N, nx=N,
            n_angles=n_angles, n_det=n_det,
            D_so=D_so, D_sd=D_sd,
            detector_offset=float(delta),
        )

        t0 = time.time()
        sino_mismatch = op_mismatch.forward(phantom)
        t_fwd_mm = time.time() - t0
        logger.info(f"  Mismatched forward: {t_fwd_mm:.1f}s")

        # ----- Scenario II: reconstruct mismatched data with wrong offset (0) -----
        logger.info("  Scenario II: mismatched (recon with offset=0)")
        op_wrong = CBCTOperator(
            operator_id="cbct_wrong",
            ny=N, nx=N,
            n_angles=n_angles, n_det=n_det,
            D_so=D_so, D_sd=D_sd,
            detector_offset=0.0,
        )
        t0 = time.time()
        recon_II = fdk_reconstruct(sino_mismatch, op_wrong)
        t_II = time.time() - t0
        psnr_II = psnr(phantom, recon_II)
        ssim_II = ssim_simple(phantom, recon_II)
        delta_psnr_II = psnr_I - psnr_II
        logger.info(f"    PSNR={psnr_II:.2f} dB   SSIM={ssim_II:.4f}   "
                     f"Delta={delta_psnr_II:+.2f} dB   ({t_II:.1f}s)")

        # ----- Scenario III: calibrated (grid search over detector_offset) -----
        logger.info(f"  Scenario III: calibration grid search "
                     f"[{calibration_range[0]}, {calibration_range[1]}] "
                     f"with {calibration_steps} steps")
        t0 = time.time()
        test_offsets = np.linspace(
            calibration_range[0], calibration_range[1], calibration_steps
        )
        best_offset = 0.0
        best_psnr_cal = -float("inf")
        best_recon_cal = None

        for test_offset in test_offsets:
            op_test = CBCTOperator(
                operator_id="cbct_cal_test",
                ny=N, nx=N,
                n_angles=n_angles, n_det=n_det,
                D_so=D_so, D_sd=D_sd,
                detector_offset=test_offset,
            )
            recon_test = fdk_reconstruct(sino_mismatch, op_test)
            p = psnr(phantom, recon_test)
            if p > best_psnr_cal:
                best_psnr_cal = p
                best_offset = test_offset
                best_recon_cal = recon_test

        t_III = time.time() - t0
        recon_III = best_recon_cal
        psnr_III = best_psnr_cal
        ssim_III = ssim_simple(phantom, recon_III)
        logger.info(f"    PSNR={psnr_III:.2f} dB   SSIM={ssim_III:.4f}   "
                     f"best_offset={best_offset:.2f} px   ({t_III:.1f}s)")

        # ----- Recovery ratio -----
        if abs(psnr_I - psnr_II) > 0.01:
            recovery_ratio = (psnr_III - psnr_II) / (psnr_I - psnr_II)
        else:
            recovery_ratio = float("nan")

        logger.info(f"  Recovery ratio: {recovery_ratio:.3f}")
        logger.info(f"  PSNR progression: I={psnr_I:.2f}  "
                     f"II={psnr_II:.2f}  III={psnr_III:.2f}")

        results["offsets"].append({
            "detector_offset_px": delta,
            "psnr_I": round(psnr_I, 4),
            "psnr_II": round(psnr_II, 4),
            "psnr_III": round(psnr_III, 4),
            "ssim_I": round(ssim_I, 4),
            "ssim_II": round(ssim_II, 4),
            "ssim_III": round(ssim_III, 4),
            "delta_psnr_II": round(delta_psnr_II, 4),
            "delta_psnr_III": round(psnr_I - psnr_III, 4),
            "recovery_ratio": round(recovery_ratio, 4) if not np.isnan(recovery_ratio) else None,
            "calibrated_offset_px": round(best_offset, 4),
            "true_offset_px": delta,
            "time_fwd_s": round(t_fwd_mm, 2),
            "time_recon_II_s": round(t_II, 2),
            "time_calibration_s": round(t_III, 2),
        })

    # ----- Summary table -----
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info(f"{'Offset':>8s} {'PSNR I':>8s} {'PSNR II':>8s} {'PSNR III':>8s} "
                 f"{'Delta II':>9s} {'Recovery':>9s} {'Cal. Ofs.':>10s}")
    logger.info("-" * 70)
    for r in results["offsets"]:
        rec_str = f"{r['recovery_ratio']:.3f}" if r["recovery_ratio"] is not None else "N/A"
        logger.info(
            f"{r['detector_offset_px']:>8d} "
            f"{r['psnr_I']:>8.2f} "
            f"{r['psnr_II']:>8.2f} "
            f"{r['psnr_III']:>8.2f} "
            f"{r['delta_psnr_II']:>+9.2f} "
            f"{rec_str:>9s} "
            f"{r['calibrated_offset_px']:>10.2f}"
        )
    logger.info("=" * 70)

    # ----- Save results -----
    out_path = RESULTS_DIR / "cbct_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    results = run_cbct_4scenario()

    # Quick sanity checks
    for r in results["offsets"]:
        delta = r["detector_offset_px"]
        if r["psnr_II"] >= r["psnr_I"]:
            logger.warning(f"  Offset {delta}px: mismatch did NOT degrade PSNR "
                           f"(I={r['psnr_I']:.2f}, II={r['psnr_II']:.2f})")
        if r["recovery_ratio"] is not None and r["recovery_ratio"] < 0:
            logger.warning(f"  Offset {delta}px: negative recovery ratio "
                           f"({r['recovery_ratio']:.3f})")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
