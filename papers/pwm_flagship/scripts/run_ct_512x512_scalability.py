#!/usr/bin/env python3
"""CT 512x512 Scalability Experiment for PWM Nature Paper.

Demonstrates that Gate 3 dominance persists at clinical resolution (512x512)
with the same 4-scenario protocol used for 128x128 validation.

Usage:
    python run_ct_512x512_scalability.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as ndimage_shift

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pwm_core"))

RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "real_data_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

from pwm_core.physics.tomography.ct_operator import CTOperator  # noqa: E402


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = float(np.max(ref) - np.min(ref))
    return float(10.0 * np.log10(max_val ** 2 / mse))


def ssim_simple(ref: np.ndarray, test: np.ndarray, win_size: int = 7) -> float:
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


def shepp_logan_phantom(n: int = 512) -> np.ndarray:
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


def shift_sinogram(sinogram: np.ndarray, delta: float) -> np.ndarray:
    shifted = ndimage_shift(sinogram.astype(np.float64), [0, delta], order=3, mode='constant')
    return shifted.astype(np.float32)


def fbp_reconstruct(sinogram: np.ndarray, ct_op: CTOperator, det_offset: float = 0.0) -> np.ndarray:
    sino64 = sinogram.astype(np.float64)
    if abs(det_offset) > 1e-6:
        sino64 = ndimage_shift(sino64, [0, -det_offset], order=3, mode='constant')
    n_det = sino64.shape[1]
    freq = np.fft.fftfreq(n_det)
    abs_freq = np.abs(freq)
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


def main():
    logger.info("=" * 60)
    logger.info("CT 512x512 Scalability Experiment")
    logger.info("=" * 60)

    N = 512
    n_angles = 720  # clinical-grade angular sampling
    phantom = shepp_logan_phantom(N)
    logger.info(f"Phantom: {N}x{N}, range [{phantom.min():.3f}, {phantom.max():.3f}]")

    ct_op = CTOperator(x_shape=(N, N), n_angles=n_angles)
    n_det = N

    t0 = time.time()
    sinogram_clean = ct_op.forward(phantom)
    t_fwd = time.time() - t0
    logger.info(f"Forward: {sinogram_clean.shape}, time={t_fwd:.1f}s")

    offset_levels = [2, 5, 10, 20]
    calibration_range = (-25.0, 25.0)
    calibration_steps = 21

    results = {
        "experiment": "ct_512x512_scalability",
        "dataset": f"shepp_logan_{N}",
        "phantom_size": N,
        "n_angles": n_angles,
        "n_det": n_det,
        "geometry": "parallel_beam",
        "calibration_range": list(calibration_range),
        "calibration_steps": calibration_steps,
        "offsets": [],
    }

    for delta in offset_levels:
        logger.info(f"\n{'='*50}")
        logger.info(f"Detector offset: {delta} px")
        logger.info(f"{'='*50}")

        sinogram = shift_sinogram(sinogram_clean, float(delta))

        # Scenario I: correct offset
        t0 = time.time()
        recon_I = fbp_reconstruct(sinogram, ct_op, det_offset=float(delta))
        s_I = np.dot(phantom.ravel(), recon_I.ravel()) / max(
            np.dot(recon_I.ravel(), recon_I.ravel()), 1e-15)
        recon_I *= s_I
        t_I = time.time() - t0
        psnr_I = psnr(phantom, recon_I)
        ssim_I = ssim_simple(phantom, recon_I)
        logger.info(f"  I:   PSNR={psnr_I:.2f}  SSIM={ssim_I:.4f}  ({t_I:.1f}s)")

        # Scenario II: wrong offset (0)
        t0 = time.time()
        recon_II = fbp_reconstruct(sinogram, ct_op, det_offset=0.0)
        s_II = np.dot(phantom.ravel(), recon_II.ravel()) / max(
            np.dot(recon_II.ravel(), recon_II.ravel()), 1e-15)
        recon_II *= s_II
        t_II = time.time() - t0
        psnr_II = psnr(phantom, recon_II)
        ssim_II = ssim_simple(phantom, recon_II)
        delta_psnr = psnr_I - psnr_II
        logger.info(f"  II:  PSNR={psnr_II:.2f}  SSIM={ssim_II:.4f}  Delta={delta_psnr:+.2f}dB  ({t_II:.1f}s)")

        # Scenario III: grid-search calibration
        t0 = time.time()
        test_offsets = np.linspace(calibration_range[0], calibration_range[1], calibration_steps)
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
        psnr_III = psnr(phantom, best_recon_cal)
        ssim_III = ssim_simple(phantom, best_recon_cal)
        logger.info(f"  III: PSNR={psnr_III:.2f}  SSIM={ssim_III:.4f}  cal_offset={best_offset:.2f}px  ({t_III:.1f}s)")

        recovery = (psnr_III - psnr_II) / (psnr_I - psnr_II) if abs(psnr_I - psnr_II) > 0.01 else float("nan")
        logger.info(f"  Recovery ratio: {recovery:.3f}")

        results["offsets"].append({
            "detector_offset_px": delta,
            "psnr_I": round(psnr_I, 4),
            "psnr_II": round(psnr_II, 4),
            "psnr_III": round(psnr_III, 4),
            "ssim_I": round(ssim_I, 4),
            "ssim_II": round(ssim_II, 4),
            "ssim_III": round(ssim_III, 4),
            "delta_psnr": round(delta_psnr, 4),
            "recovery_ratio": round(recovery, 4) if not np.isnan(recovery) else None,
            "calibrated_offset_px": round(best_offset, 4),
            "time_I_s": round(t_I, 2),
            "time_II_s": round(t_II, 2),
            "time_III_s": round(t_III, 2),
        })

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — 512x512 CT Scalability")
    logger.info(f"{'Offset':>8s} {'PSNR I':>8s} {'PSNR II':>8s} {'PSNR III':>8s} "
                f"{'Delta':>8s} {'Recovery':>9s} {'Cal.Ofs.':>9s}")
    logger.info("-" * 70)
    for r in results["offsets"]:
        rec_str = f"{r['recovery_ratio']:.3f}" if r["recovery_ratio"] is not None else "N/A"
        logger.info(
            f"{r['detector_offset_px']:>8d} "
            f"{r['psnr_I']:>8.2f} "
            f"{r['psnr_II']:>8.2f} "
            f"{r['psnr_III']:>8.2f} "
            f"{r['delta_psnr']:>+8.2f} "
            f"{rec_str:>9s} "
            f"{r['calibrated_offset_px']:>9.2f}"
        )
    logger.info("=" * 70)

    out_path = RESULTS_DIR / "ct_512x512_scalability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
