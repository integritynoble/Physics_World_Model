#!/usr/bin/env python3
"""CBCT 4-Scenario Validation for PWM Nature Paper.

Simulated Shepp-Logan phantom with detector offset mismatch.
Gate 3 mismatch: detector offset in the sinogram causes misaligned
backprojection artifacts.

Protocol:
  Generate clean sinogram via Radon transform (CTOperator).
  For each offset delta, shift the sinogram by delta pixels to simulate
  hardware detector misalignment.
  Scenario I:   FBP knowing the true shift (correct compensation)
  Scenario II:  FBP assuming shift=0 (wrong) -> degraded
  Scenario III: Calibrate shift via PSNR grid search -> recovered

Offsets tested: [2, 5, 10, 20] pixels.
Solver: Filtered back-projection (ramp filter + adjoint).

Usage:
    python run_cbct_4scenario.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.ndimage import shift as ndimage_shift

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
# Import operators
# ---------------------------------------------------------------------------
from pwm_core.physics.tomography.cbct_operator import CBCTOperator  # noqa: E402
from pwm_core.physics.tomography.ct_operator import CTOperator  # noqa: E402


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
_SHEPP_LOGAN_ELLIPSES = [
    (  2.0,     0.6900, 0.9200,  0.0000,  0.0000,   0.0),
    ( -0.98,    0.6624, 0.8740,  0.0000, -0.0184,   0.0),
    ( -0.02,    0.1100, 0.3100,  0.2200,  0.0000, -18.0),
    ( -0.02,    0.1600, 0.4100, -0.2200,  0.0000,  18.0),
    (  0.01,    0.2100, 0.2500,  0.0000,  0.3500,   0.0),
    (  0.01,    0.0460, 0.0460,  0.0000,  0.1000,   0.0),
    (  0.01,    0.0460, 0.0460,  0.0000, -0.1000,   0.0),
    (  0.01,    0.0460, 0.0230, -0.0800, -0.6050,   0.0),
    (  0.01,    0.0230, 0.0230,  0.0000, -0.6050,   0.0),
    (  0.01,    0.0230, 0.0460,  0.0600, -0.6050,   0.0),
]


def shepp_logan_phantom(n: int = 64) -> np.ndarray:
    """Generate a Shepp-Logan phantom on an (n x n) grid."""
    phantom = np.zeros((n, n), dtype=np.float64)
    coords = np.linspace(-1.0, 1.0, n, endpoint=False) + 1.0 / n
    yy, xx = np.meshgrid(coords, coords, indexing="ij")

    for intensity, a, b, x0, y0, phi_deg in _SHEPP_LOGAN_ELLIPSES:
        phi = np.deg2rad(phi_deg)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        xr = xx - x0
        yr = yy - y0
        xr_rot = xr * cos_p + yr * sin_p
        yr_rot = -xr * sin_p + yr * cos_p
        inside = (xr_rot / a) ** 2 + (yr_rot / b) ** 2 <= 1.0
        phantom[inside] += intensity

    return phantom


# ---------------------------------------------------------------------------
# Sinogram shift (detector offset simulation)
# ---------------------------------------------------------------------------
def shift_sinogram(sinogram: np.ndarray, delta: float) -> np.ndarray:
    """Shift sinogram along detector axis by delta pixels (sub-pixel via interpolation).

    Simulates detector offset: each projection row is shifted by the same amount.
    """
    shifted = ndimage_shift(sinogram.astype(np.float64), [0, delta], order=3, mode='constant')
    return shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# FBP reconstruction (ramp filter + backprojection)
# ---------------------------------------------------------------------------
def fbp_reconstruct(
    sinogram: np.ndarray,
    ct_op: CTOperator,
    det_offset: float = 0.0,
) -> np.ndarray:
    """Filtered back-projection with optional detector offset compensation.

    Steps:
    1. Shift sinogram by -det_offset to compensate for known detector offset
    2. Apply Shepp-Logan (smoothed ramp) filter
    3. Backproject via CT operator adjoint

    Args:
        sinogram: (n_angles, n_det) sinogram array.
        ct_op: CTOperator for backprojection.
        det_offset: Known detector offset to compensate (pixels).

    Returns:
        Reconstructed image (ny, nx) as float64.
    """
    sino64 = sinogram.astype(np.float64)

    # Compensate detector offset by shifting sinogram back
    if abs(det_offset) > 1e-6:
        sino64 = ndimage_shift(sino64, [0, -det_offset], order=3, mode='constant')

    # Shepp-Logan (smoothed ramp) filter — less noise amplification than pure ramp
    n_det = sino64.shape[1]
    freq = np.fft.fftfreq(n_det)
    abs_freq = np.abs(freq)
    # Shepp-Logan window: sinc(f/f_max) applied to ramp
    shepp_logan = abs_freq * np.where(
        abs_freq > 0,
        np.sin(np.pi * freq) / (np.pi * freq + 1e-15),
        1.0,
    )

    n_angles = sino64.shape[0]
    filtered = np.zeros_like(sino64)
    for a in range(n_angles):
        proj_fft = np.fft.fft(sino64[a, :])
        filtered[a, :] = np.real(np.fft.ifft(proj_fft * shepp_logan))

    recon = ct_op.adjoint(filtered.astype(np.float32))
    return recon.astype(np.float64)


# ---------------------------------------------------------------------------
# Main 4-scenario protocol
# ---------------------------------------------------------------------------
def run_cbct_4scenario() -> Dict:
    """Run the 4-scenario protocol on a simulated Shepp-Logan CT problem.

    Gate 3 mismatch: detector_offset causes shifted projections,
    producing misaligned artifacts in the FBP reconstruction.

    Returns:
        Dictionary of results.
    """
    logger.info("=" * 60)
    logger.info("CBCT 4-Scenario Protocol (detector offset mismatch)")
    logger.info("=" * 60)

    # ----- phantom -----
    N = 128
    phantom = shepp_logan_phantom(N)
    logger.info(f"Phantom: {N}x{N}, range [{phantom.min():.3f}, {phantom.max():.3f}]")

    # ----- operator -----
    n_angles = 360
    ct_op = CTOperator(x_shape=(N, N), n_angles=n_angles)
    n_det = N  # parallel-beam detector width = image width

    # ----- Generate clean sinogram (no offset) -----
    t0 = time.time()
    sinogram_clean = ct_op.forward(phantom)
    t_fwd = time.time() - t0
    logger.info(f"Forward (Radon): {sinogram_clean.shape}, range "
                f"[{sinogram_clean.min():.2f}, {sinogram_clean.max():.2f}], time={t_fwd:.1f}s")

    # ----- Detector offset levels to test -----
    offset_levels = [2, 5, 10, 20]  # pixels
    calibration_range = (-25.0, 25.0)
    calibration_steps = 21

    logger.info(f"Geometry: {n_angles} angles, {n_det} detectors (parallel beam)")
    logger.info(f"Offsets to test: {offset_levels} px")
    logger.info(f"Calibration: [{calibration_range[0]}, {calibration_range[1]}] "
                f"with {calibration_steps} steps")

    results: Dict = {
        "dataset": f"shepp_logan_{N}",
        "n_angles": n_angles,
        "n_det": n_det,
        "geometry": "parallel_beam",
        "phantom_size": N,
        "calibration_range": list(calibration_range),
        "calibration_steps": calibration_steps,
        "offsets": [],
    }

    for delta in offset_levels:
        logger.info(f"\n{'='*50}")
        logger.info(f"Detector offset mismatch: {delta} px")
        logger.info(f"{'='*50}")

        # ----- Create sinogram with detector offset (shift) -----
        sinogram = shift_sinogram(sinogram_clean, float(delta))
        logger.info(f"  Sinogram shifted by {delta} px")

        # ----- Scenario I: FBP with correct offset compensation -----
        logger.info("  Scenario I: FBP with correct offset (reference)")
        t0 = time.time()
        recon_I = fbp_reconstruct(sinogram, ct_op, det_offset=float(delta))
        s_I = np.dot(phantom.ravel(), recon_I.ravel()) / max(
            np.dot(recon_I.ravel(), recon_I.ravel()), 1e-15)
        recon_I *= s_I
        t_I = time.time() - t0
        psnr_I = psnr(phantom, recon_I)
        ssim_I = ssim_simple(phantom, recon_I)
        logger.info(f"    PSNR={psnr_I:.2f} dB   SSIM={ssim_I:.4f}   ({t_I:.1f}s)")

        # ----- Scenario II: FBP with offset=0 (wrong) -----
        logger.info("  Scenario II: mismatched (FBP with offset=0)")
        t0 = time.time()
        recon_II = fbp_reconstruct(sinogram, ct_op, det_offset=0.0)
        s_II = np.dot(phantom.ravel(), recon_II.ravel()) / max(
            np.dot(recon_II.ravel(), recon_II.ravel()), 1e-15)
        recon_II *= s_II
        t_II = time.time() - t0
        psnr_II = psnr(phantom, recon_II)
        ssim_II = ssim_simple(phantom, recon_II)
        delta_psnr_II = psnr_I - psnr_II
        logger.info(f"    PSNR={psnr_II:.2f} dB   SSIM={ssim_II:.4f}   "
                     f"Delta={delta_psnr_II:+.2f} dB   ({t_II:.1f}s)")

        # ----- Scenario III: calibrate offset via PSNR grid search -----
        logger.info(f"  Scenario III: PSNR-based calibration "
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
            recon_test = fbp_reconstruct(sinogram, ct_op, det_offset=test_offset)
            s_test = np.dot(phantom.ravel(), recon_test.ravel()) / max(
                np.dot(recon_test.ravel(), recon_test.ravel()), 1e-15)
            recon_test *= s_test
            psnr_test = psnr(phantom, recon_test)
            if psnr_test > best_psnr_cal:
                best_psnr_cal = psnr_test
                best_offset = test_offset
                best_recon_cal = recon_test.copy()

        t_III = time.time() - t0
        recon_III = best_recon_cal
        psnr_III = psnr(phantom, recon_III)
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
            "calibrated_psnr": round(best_psnr_cal, 4),
            "time_recon_I_s": round(t_I, 2),
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
