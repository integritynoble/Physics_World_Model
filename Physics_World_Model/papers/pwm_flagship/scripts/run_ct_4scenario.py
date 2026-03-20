#!/usr/bin/env python3
"""CT Real-Data 4-Scenario Validation for PWM Nature Paper.

Uses the FIPS walnut micro-CT dataset (parallel-beam sinograms) and
the Helsinki Tomography Challenge 2022 dataset (fan-beam sinograms)
to validate the Triad Decomposition on CT with center-of-rotation (CoR) mismatch.

Gate 3 mismatch model: the center of rotation is offset by Δ pixels,
causing the reconstructed image to have double-image artifacts.

Usage:
    python run_ct_4scenario.py [--dataset walnut|htc|both]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "papers" / "pwm_flagship" / "results" / "real_data_4scenario"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FBP implementation (parallel beam)
# ---------------------------------------------------------------------------
def fbp_parallel(sinogram: np.ndarray, angles: np.ndarray,
                 cor_offset: float = 0.0, n_recon: int = None) -> np.ndarray:
    """Filtered back-projection for parallel-beam CT.

    Args:
        sinogram: (n_det, n_angles) array of projections
        angles: (n_angles,) array of projection angles in radians
        cor_offset: center-of-rotation offset in pixels (shifts sinogram)
        n_recon: reconstruction grid size (default = n_det)

    Returns:
        (n_recon, n_recon) reconstructed image
    """
    n_det, n_angles = sinogram.shape
    if n_recon is None:
        n_recon = n_det

    # Apply CoR offset by shifting sinogram
    if abs(cor_offset) > 1e-6:
        from scipy.ndimage import shift as ndshift
        sino_shifted = np.zeros_like(sinogram)
        for a in range(n_angles):
            sino_shifted[:, a] = ndshift(sinogram[:, a], cor_offset, order=1, mode='constant')
        sinogram = sino_shifted

    # Ram-Lak filter
    freq = np.fft.fftfreq(n_det)
    ram_lak = np.abs(freq)
    # Apply filter to each projection
    filtered = np.zeros_like(sinogram)
    for a in range(n_angles):
        proj_fft = np.fft.fft(sinogram[:, a])
        filtered[:, a] = np.real(np.fft.ifft(proj_fft * ram_lak))

    # Back-projection
    recon = np.zeros((n_recon, n_recon), dtype=np.float64)
    center = n_recon / 2.0
    det_center = n_det / 2.0
    x = np.arange(n_recon) - center
    y = np.arange(n_recon) - center
    xx, yy = np.meshgrid(x, y)

    for a in range(n_angles):
        theta = angles[a]
        # Project pixel coordinates onto detector
        t = xx * np.cos(theta) + yy * np.sin(theta)
        # Convert to detector pixel indices
        det_idx = t + det_center
        # Bilinear interpolation
        det_idx_floor = np.floor(det_idx).astype(int)
        frac = det_idx - det_idx_floor
        valid = (det_idx_floor >= 0) & (det_idx_floor < n_det - 1)
        contrib = np.zeros((n_recon, n_recon))
        contrib[valid] = (filtered[det_idx_floor[valid], a] * (1 - frac[valid]) +
                          filtered[det_idx_floor[valid] + 1, a] * frac[valid])
        recon += contrib

    recon *= np.pi / n_angles
    return recon


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute PSNR between reference and test images."""
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    max_val = np.max(ref) - np.min(ref)
    return float(10 * np.log10(max_val**2 / mse))


def ssim_simple(ref: np.ndarray, test: np.ndarray, win_size: int = 7) -> float:
    """Simplified SSIM computation."""
    from scipy.ndimage import uniform_filter
    C1 = (0.01 * (ref.max() - ref.min()))**2
    C2 = (0.03 * (ref.max() - ref.min()))**2

    mu_x = uniform_filter(ref, win_size)
    mu_y = uniform_filter(test, win_size)
    sigma_x2 = uniform_filter(ref**2, win_size) - mu_x**2
    sigma_y2 = uniform_filter(test**2, win_size) - mu_y**2
    sigma_xy = uniform_filter(ref * test, win_size) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


def sinogram_residual(sinogram: np.ndarray, recon: np.ndarray,
                      angles: np.ndarray, cor_offset: float = 0.0) -> float:
    """Compute forward projection residual: ||y - Ax||^2 / ||y||^2.

    Forward projects the reconstruction and compares to measured sinogram.
    """
    n_det = sinogram.shape[0]
    n_recon = recon.shape[0]
    center = n_recon / 2.0
    det_center = n_det / 2.0
    x = np.arange(n_recon) - center
    y = np.arange(n_recon) - center
    xx, yy = np.meshgrid(x, y)

    residual_sum = 0.0
    sino_sum = 0.0

    for a in range(len(angles)):
        theta = angles[a]
        t = xx * np.cos(theta) + yy * np.sin(theta) + cor_offset
        det_idx = t + det_center
        # Forward project: sum recon values onto detector
        proj = np.zeros(n_det)
        det_idx_floor = np.floor(det_idx).astype(int)
        frac = det_idx - det_idx_floor
        valid = (det_idx_floor >= 0) & (det_idx_floor < n_det - 1)
        np.add.at(proj, det_idx_floor[valid], recon[valid] * (1 - frac[valid]))
        np.add.at(proj, det_idx_floor[valid] + 1, recon[valid] * frac[valid])

        residual_sum += np.sum((sinogram[:, a] - proj)**2)
        sino_sum += np.sum(sinogram[:, a]**2)

    return float(residual_sum / max(sino_sum, 1e-15))


# ---------------------------------------------------------------------------
# FIPS Walnut dataset
# ---------------------------------------------------------------------------
def run_walnut_4scenario():
    """4-scenario protocol on FIPS walnut micro-CT data."""
    logger.info("=" * 60)
    logger.info("FIPS Walnut CT: 4-Scenario Protocol (CoR mismatch)")
    logger.info("=" * 60)

    data_path = Path("/home/spiritai/real_datasets/ct")
    d = sio.loadmat(str(data_path / "FullSizeSinograms.mat"))
    sinogram_full = d["sinogram1200"].astype(np.float64)  # (2296, 1200)

    # Ground truth is in a separate file
    gt_data = sio.loadmat(str(data_path / "GroundTruthReconstruction.mat"))
    gt_fbp = gt_data["FBP1200"].astype(np.float64)  # (2296, 2296)

    n_det, n_angles = sinogram_full.shape
    logger.info(f"Sinogram: {n_det} detectors x {n_angles} angles")
    logger.info(f"Ground truth FBP: {gt_fbp.shape}")

    # Use the 120-angle sinogram for speed
    sinogram = d["sinogram120"].astype(np.float64)  # (2296, 120)
    n_det_ds, n_angles_ds = sinogram.shape
    angles = np.linspace(0, np.pi, n_angles_ds, endpoint=False)

    # Crop to manageable size: center 512 detectors
    crop = 512
    start = (n_det_ds - crop) // 2
    sinogram_crop = sinogram[start:start + crop, :]
    gt_crop = gt_fbp[start:start + crop, start:start + crop]

    logger.info(f"Using 120-angle sinogram, cropped to {crop} detectors")

    # CoR offsets to test
    cor_offsets = [1.0, 3.0, 5.0]  # pixels
    results = {"dataset": "FIPS_walnut", "cor_offsets": []}

    for cor_offset in cor_offsets:
        logger.info(f"\n--- CoR offset: {cor_offset} px ---")

        # Scenario I: Correct CoR
        t0 = time.time()
        recon_I = fbp_parallel(sinogram_crop, angles, cor_offset=0.0, n_recon=crop)
        psnr_I = psnr(gt_crop, recon_I)
        t_I = time.time() - t0
        logger.info(f"  Scenario I  (correct CoR):   PSNR={psnr_I:.2f} dB  ({t_I:.1f}s)")

        # Scenario II: Mismatched CoR
        t0 = time.time()
        recon_II = fbp_parallel(sinogram_crop, angles, cor_offset=cor_offset, n_recon=crop)
        psnr_II = psnr(gt_crop, recon_II)
        t_II = time.time() - t0
        delta_II = psnr_I - psnr_II
        logger.info(f"  Scenario II (CoR+{cor_offset}px):     PSNR={psnr_II:.2f} dB  "
                    f"Δ={delta_II:+.2f} dB  ({t_II:.1f}s)")

        # Scenario III: Calibrated (search over CoR)
        t0 = time.time()
        best_cor = 0.0
        best_psnr = -float('inf')
        for test_cor in np.linspace(-cor_offset * 1.5, cor_offset * 1.5, 15):
            recon_test = fbp_parallel(sinogram_crop, angles, cor_offset=test_cor, n_recon=crop)
            p = psnr(gt_crop, recon_test)
            if p > best_psnr:
                best_psnr = p
                best_cor = test_cor
        recon_III = fbp_parallel(sinogram_crop, angles, cor_offset=best_cor, n_recon=crop)
        psnr_III = psnr(gt_crop, recon_III)
        t_III = time.time() - t0
        logger.info(f"  Scenario III (calibrated):   PSNR={psnr_III:.2f} dB  "
                    f"best_cor={best_cor:.2f}  ({t_III:.1f}s)")

        # Measurement residual analysis
        res_I = sinogram_residual(sinogram_crop, recon_I, angles, cor_offset=0.0)
        res_II = sinogram_residual(sinogram_crop, recon_II, angles, cor_offset=0.0)
        cross_ratio = res_II / max(res_I, 1e-15)
        logger.info(f"  Sino residual I={res_I:.6f}  II={res_II:.6f}  ratio={cross_ratio:.2f}x")

        # Recovery ratio
        if abs(psnr_I - psnr_II) > 0.01:
            recovery = (psnr_III - psnr_II) / (psnr_I - psnr_II)
        else:
            recovery = float('nan')

        results["cor_offsets"].append({
            "cor_offset_px": cor_offset,
            "psnr_I": psnr_I,
            "psnr_II": psnr_II,
            "psnr_III": psnr_III,
            "delta_psnr_II": delta_II,
            "delta_psnr_III": psnr_I - psnr_III,
            "recovery_ratio": recovery,
            "sino_res_I": res_I,
            "sino_res_II": res_II,
            "sino_cross_ratio": cross_ratio,
            "best_cor": best_cor,
        })

    out_path = RESULTS_DIR / "ct_walnut_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nWalnut CT results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# HTC 2022 dataset (fan-beam → treat as parallel-beam approximation)
# ---------------------------------------------------------------------------
def run_htc_4scenario():
    """4-scenario protocol on HTC 2022 CT data."""
    logger.info("=" * 60)
    logger.info("HTC 2022 CT: 4-Scenario Protocol (CoR mismatch)")
    logger.info("=" * 60)

    data_path = Path("/home/spiritai/real_datasets/ct")
    if not (data_path / "htc2022_ta_full.mat").exists():
        logger.warning("HTC data not found, skipping")
        return None

    d = sio.loadmat(str(data_path / "htc2022_ta_full.mat"))
    ct = d["CtDataFull"][0, 0]
    sinogram = ct["sinogram"].astype(np.float64)  # (721, 560)
    params = ct["parameters"][0, 0]
    angles_deg = params["angles"].flatten()  # 721 angles
    angles = np.deg2rad(angles_deg) if angles_deg.max() > 10 else angles_deg

    gt_data = sio.loadmat(str(data_path / "htc2022_ta_full_recon_fbp.mat"))
    gt_fbp = gt_data["reconFullFbp"].astype(np.float64)  # (512, 512)

    n_det, n_angles = sinogram.shape
    logger.info(f"Sinogram: {n_det} detectors x {n_angles} angles")
    logger.info(f"Angles range: {np.rad2deg(angles[0]):.1f} to {np.rad2deg(angles[-1]):.1f} deg")
    logger.info(f"Ground truth FBP: {gt_fbp.shape}")

    # Note: HTC is fan-beam, but we treat as parallel-beam approximation
    # (geometric magnification ~1.35, so parallel-beam is a reasonable approx)
    n_recon = gt_fbp.shape[0]

    cor_offsets = [1.0, 3.0, 5.0]
    results = {"dataset": "HTC2022_ta", "cor_offsets": []}

    for cor_offset in cor_offsets:
        logger.info(f"\n--- CoR offset: {cor_offset} px ---")

        # Scenario I: Correct CoR
        t0 = time.time()
        recon_I = fbp_parallel(sinogram, angles, cor_offset=0.0, n_recon=n_recon)
        psnr_I = psnr(gt_fbp, recon_I)
        ssim_I = ssim_simple(gt_fbp, recon_I)
        t_I = time.time() - t0
        logger.info(f"  Scenario I  (correct):  PSNR={psnr_I:.2f} dB  SSIM={ssim_I:.4f}  ({t_I:.1f}s)")

        # Scenario II: Mismatched CoR
        t0 = time.time()
        recon_II = fbp_parallel(sinogram, angles, cor_offset=cor_offset, n_recon=n_recon)
        psnr_II = psnr(gt_fbp, recon_II)
        ssim_II = ssim_simple(gt_fbp, recon_II)
        t_II = time.time() - t0
        delta = psnr_I - psnr_II
        logger.info(f"  Scenario II (CoR+{cor_offset}px): PSNR={psnr_II:.2f} dB  SSIM={ssim_II:.4f}  "
                    f"Δ={delta:+.2f} dB  ({t_II:.1f}s)")

        # Scenario III: Oracle search
        t0 = time.time()
        best_cor = 0.0
        best_psnr = -float('inf')
        for test_cor in np.linspace(-cor_offset * 1.5, cor_offset * 1.5, 15):
            recon_test = fbp_parallel(sinogram, angles, cor_offset=test_cor, n_recon=n_recon)
            p = psnr(gt_fbp, recon_test)
            if p > best_psnr:
                best_psnr = p
                best_cor = test_cor
        psnr_III = best_psnr
        t_III = time.time() - t0
        logger.info(f"  Scenario III (oracle):  PSNR={psnr_III:.2f} dB  best_cor={best_cor:.2f}  ({t_III:.1f}s)")

        if abs(psnr_I - psnr_II) > 0.01:
            recovery = (psnr_III - psnr_II) / (psnr_I - psnr_II)
        else:
            recovery = float('nan')

        results["cor_offsets"].append({
            "cor_offset_px": cor_offset,
            "psnr_I": psnr_I,
            "psnr_II": psnr_II,
            "psnr_III": psnr_III,
            "ssim_I": ssim_I,
            "ssim_II": ssim_II,
            "delta_psnr": delta,
            "recovery_ratio": recovery,
            "best_cor": best_cor,
        })

    out_path = RESULTS_DIR / "ct_htc_4scenario_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nHTC CT results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both", choices=["walnut", "htc", "both"])
    args = parser.parse_args()

    all_results = {}
    if args.dataset in ("walnut", "both"):
        all_results["walnut"] = run_walnut_4scenario()
    if args.dataset in ("htc", "both"):
        r = run_htc_4scenario()
        if r:
            all_results["htc"] = r

    out_path = RESULTS_DIR / "ct_combined_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nCombined CT results saved to {out_path}")


if __name__ == "__main__":
    main()
