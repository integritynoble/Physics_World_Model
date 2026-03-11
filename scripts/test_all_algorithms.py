#!/usr/bin/env python3
"""Comprehensive algorithm testing for all complete benchmark modalities.

GPU Server responsibility #3: Test all algorithms for each modality.
Tests low-level solver functions directly (no physics operator needed).

Modalities tested: 13 fully complete (public/dev/hidden with spec+true_spec)
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import yaml
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

BENCHMARK_DIR = ROOT / "datasets" / "benchmark"
CONFIG_DIR = ROOT / "benchmarks" / "configs"
RESULTS_DIR = ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> Optional[float]:
    if np.iscomplexobj(gt):
        gt = np.abs(gt)
    if np.iscomplexobj(recon):
        recon = np.abs(recon)
    if gt.shape != recon.shape:
        return None
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = gt.max() - gt.min()
    if data_range == 0:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> Optional[float]:
    if np.iscomplexobj(gt):
        gt = np.abs(gt)
    if np.iscomplexobj(recon):
        recon = np.abs(recon)
    if gt.shape != recon.shape:
        return None
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    if data_range == 0:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = gt.mean(), recon.mean()
    var_x, var_y = gt.var(), recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


def run_and_measure(solver_fn, gt, *args, **kwargs) -> Dict:
    """Run solver, compute metrics, return result dict."""
    try:
        start = time.time()
        recon = solver_fn(*args, **kwargs)
        exec_time = time.time() - start

        if np.iscomplexobj(recon):
            recon = np.abs(recon)
        gt_real = np.abs(gt) if np.iscomplexobj(gt) else gt

        # Match shapes if possible
        if recon.shape != gt_real.shape:
            return {
                "status": "shape_mismatch",
                "recon_shape": list(recon.shape),
                "gt_shape": list(gt_real.shape),
                "exec_time_sec": exec_time
            }

        psnr = compute_psnr(gt_real, recon)
        ssim = compute_ssim(gt_real, recon)
        return {
            "status": "completed",
            "psnr_db": psnr,
            "ssim": ssim,
            "exec_time_sec": exec_time,
            "result_shape": list(recon.shape)
        }
    except Exception as e:
        return {"status": f"error: {str(e)[:80]}"}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_sample(modality_id: str, tier: str = "public", idx: int = 0) -> Optional[Dict]:
    tier_dir = BENCHMARK_DIR / modality_id / tier

    # Try flat layout (h5 in mod_dir, not in tier_dir)
    if not tier_dir.exists():
        mod_dir = BENCHMARK_DIR / modality_id
        flat_h5 = list(mod_dir.glob(f"*_{tier}*.h5")) or list(mod_dir.glob(f"*{tier}*.h5"))
        if flat_h5:
            try:
                with h5py.File(flat_h5[0], "r") as f:
                    keys = sorted([k for k in f.keys() if k.startswith("sample")])
                    if keys and idx < len(keys):
                        data = {k: f[keys[idx]][k][:] for k in f[keys[idx]].keys()}
                        return {"sample": keys[idx], "data": data}
            except Exception as e:
                print(f"    H5 flat error: {e}")
        return None

    # HDF5 format in tier subdirectory
    h5_files = list(tier_dir.glob("*_challenge_*.h5"))
    if h5_files:
        try:
            with h5py.File(h5_files[0], "r") as f:
                keys = sorted([k for k in f.keys() if k.startswith("sample")])
                if keys and idx < len(keys):
                    data = {k: f[keys[idx]][k][:] for k in f[keys[idx]].keys()}
                    return {"sample": keys[idx], "data": data}
        except Exception as e:
            print(f"    H5 error: {e}")

    # Directory format (CT)
    dirs = sorted([d for d in tier_dir.iterdir()
                   if d.is_dir() and d.name.startswith("sample_")])
    if dirs and idx < len(dirs):
        try:
            data = {}
            for npy in dirs[idx].glob("*.npy"):
                data[npy.stem] = np.load(npy)
            if data:
                return {"sample": dirs[idx].name, "data": data}
        except Exception as e:
            print(f"    Dir error: {e}")

    return None


# ── Per-Modality Solver Tests ────────────────────────────────────────────────

def test_ct(data: Dict) -> Dict:
    """CT: FBP and SART on sinogram data."""
    from pwm_core.recon.ct_solvers import fbp_2d, sart_2d

    meas = data["measurement"]
    angles = data["angles"]
    gt = data["groundtruth"]
    out_size = gt.shape[0]

    # Angles to radians
    angles_rad = np.deg2rad(angles) if angles.max() > 2 * np.pi else angles

    results = {}
    results["fbp_ramlak"] = run_and_measure(
        lambda: fbp_2d(meas, angles_rad, "ramlak", out_size), gt)
    results["fbp_shepp_logan"] = run_and_measure(
        lambda: fbp_2d(meas, angles_rad, "shepp_logan", out_size), gt)
    results["sart_10iter"] = run_and_measure(
        lambda: sart_2d(meas, angles_rad, out_size, iterations=10), gt)
    return results


def test_mri(data: Dict) -> Dict:
    """MRI: Zero-filled, CS-MRI, SENSE."""
    from pwm_core.recon.mri_solvers import (
        zero_filled_reconstruction, cs_mri_wavelet, sense_reconstruction,
        estimate_sensitivity_maps
    )

    kspace = data["kspace_undersampled"]
    mask = data["mask"]
    gt = data["x_true"]
    coil_maps = data.get("coil_maps")

    results = {}
    results["zero_filled"] = run_and_measure(
        lambda: zero_filled_reconstruction(kspace, mask), gt)
    results["cs_mri_wavelet"] = run_and_measure(
        lambda: np.abs(cs_mri_wavelet(kspace, mask, lam=0.01, iterations=30)), gt)

    if coil_maps is not None:
        results["sense"] = run_and_measure(
            lambda: np.abs(sense_reconstruction(
                kspace, coil_maps.astype(np.complex64), mask, 0.001, 30)), gt)

    return results


def test_pet(data: Dict) -> Dict:
    """PET: FBP on sinogram."""
    from pwm_core.recon.ct_solvers import fbp_2d

    sino = data["sinogram_measured"]
    angles = data["angles_deg"]
    gt = data["x_true"]
    out_size = gt.shape[0]

    angles_rad = np.deg2rad(angles)

    results = {}
    results["fbp_ramlak"] = run_and_measure(
        lambda: fbp_2d(sino, angles_rad, "ramlak", out_size), gt)
    results["fbp_shepp_logan"] = run_and_measure(
        lambda: fbp_2d(sino, angles_rad, "shepp_logan", out_size), gt)

    # Compare with precomputed FBP
    if "reconstruction_fbp" in data:
        recon_fbp = data["reconstruction_fbp"]
        psnr = compute_psnr(gt, recon_fbp)
        ssim = compute_ssim(gt, recon_fbp)
        results["precomputed_fbp"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(recon_fbp.shape)
        }
    return results


def test_spect(data: Dict) -> Dict:
    """SPECT: FBP on sinogram."""
    from pwm_core.recon.ct_solvers import fbp_2d

    sino = data.get("sinogram_measured")
    if sino is None:
        sino = data.get("y")
    angles = data["angles_deg"]
    gt = data["x_true"]
    out_size = gt.shape[0]

    angles_rad = np.deg2rad(angles.astype(np.float64))

    results = {}
    results["fbp_ramlak"] = run_and_measure(
        lambda: fbp_2d(sino, angles_rad, "ramlak", out_size), gt)

    if "reconstruction_fbp" in data:
        recon_fbp = data["reconstruction_fbp"]
        psnr = compute_psnr(gt, recon_fbp)
        ssim = compute_ssim(gt, recon_fbp)
        results["precomputed_fbp"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(recon_fbp.shape)
        }
    return results


def test_ultrasound(data: Dict) -> Dict:
    """Ultrasound: Richardson-Lucy deconvolution with PSF."""
    from pwm_core.recon.richardson_lucy import richardson_lucy_2d

    measurement = data["bmode_measured"]
    psf = data["psf"]
    gt = data["x_true"]

    results = {}
    results["rl_20iter"] = run_and_measure(
        lambda: richardson_lucy_2d(measurement, psf, iterations=20), gt)
    results["rl_50iter"] = run_and_measure(
        lambda: richardson_lucy_2d(measurement, psf, iterations=50), gt)
    return results


def test_oct(data: Dict) -> Dict:
    """OCT: Compare B-scan baseline (no spectral data available)."""
    gt = data["x_true"]
    results = {}

    # Baseline: bscan_measured vs x_true
    if "bscan_measured" in data:
        bm = data["bscan_measured"]
        psnr = compute_psnr(gt, bm)
        ssim = compute_ssim(gt, bm)
        results["bscan_baseline"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(bm.shape)
        }
    if "bscan_ideal" in data:
        bi = data["bscan_ideal"]
        psnr = compute_psnr(gt, bi)
        ssim = compute_ssim(gt, bi)
        results["bscan_ideal_baseline"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(bi.shape)
        }
    return results


def test_mammography(data: Dict) -> Dict:
    """Mammography: Compare precomputed reconstruction."""
    gt = data["x_true"]
    results = {}

    if "reconstruction" in data:
        recon = data["reconstruction"]
        psnr = compute_psnr(gt, recon)
        ssim = compute_ssim(gt, recon)
        results["precomputed_recon"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(recon.shape)
        }
    return results


def test_fundus(data: Dict) -> Dict:
    """Fundus: Richardson-Lucy deconvolution with PSF."""
    from pwm_core.recon.richardson_lucy import richardson_lucy_2d

    measurement = data["y"]
    psf = data["psf"]
    gt = data["x_true"]

    results = {}
    results["rl_20iter"] = run_and_measure(
        lambda: richardson_lucy_2d(measurement, psf, iterations=20), gt)
    results["rl_50iter"] = run_and_measure(
        lambda: richardson_lucy_2d(measurement, psf, iterations=50), gt)

    if "reconstruction_wiener" in data:
        rw = data["reconstruction_wiener"]
        psnr = compute_psnr(gt, rw)
        ssim = compute_ssim(gt, rw)
        results["precomputed_wiener"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(rw.shape)
        }
    return results


def test_endoscopy(data: Dict) -> Dict:
    """Endoscopy: Richardson-Lucy deconvolution with H_ideal PSF."""
    from pwm_core.recon.richardson_lucy import richardson_lucy_2d

    measurement = data["y"]
    psf = data["H_ideal"]
    gt = data["x_true"]

    results = {}
    results["rl_20iter"] = run_and_measure(
        lambda: richardson_lucy_2d(measurement, psf, iterations=20), gt)
    results["rl_50iter"] = run_and_measure(
        lambda: richardson_lucy_2d(measurement, psf, iterations=50), gt)

    if "reconstruction" in data:
        recon = data["reconstruction"]
        psnr = compute_psnr(gt, recon)
        ssim = compute_ssim(gt, recon)
        results["precomputed_recon"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(recon.shape)
        }
    return results


def test_fmri(data: Dict) -> Dict:
    """fMRI: Zero-filled IFFT reconstruction from k-space."""
    from pwm_core.recon.mri_solvers import zero_filled_reconstruction

    kspace = data["y"]  # complex64
    gt = data["x_true"]

    results = {}
    results["zero_filled"] = run_and_measure(
        lambda: zero_filled_reconstruction(kspace), gt)
    return results


def test_diffusion_mri(data: Dict) -> Dict:
    """Diffusion MRI: Zero-filled IFFT reconstruction from k-space."""
    from pwm_core.recon.mri_solvers import zero_filled_reconstruction

    kspace = data["y"]  # complex64
    gt = data["x_true"]

    results = {}
    results["zero_filled"] = run_and_measure(
        lambda: zero_filled_reconstruction(kspace), gt)
    return results


def test_cryo_em(data: Dict) -> Dict:
    """Cryo-EM: Compare precomputed Wiener CTF + RL deconv."""
    from pwm_core.recon.richardson_lucy import richardson_lucy_2d

    measurement = data["y"]
    gt = data["x_true"]

    results = {}

    if "reconstruction_wiener" in data:
        rw = data["reconstruction_wiener"]
        psnr = compute_psnr(gt, rw)
        ssim = compute_ssim(gt, rw)
        results["precomputed_wiener"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(rw.shape)
        }

    # Attempt RL with CTF as PSF (H_ideal)
    if "H_ideal" in data:
        ctf = data["H_ideal"]
        # CTF is oscillating (-1 to 1), take abs for PSF-like kernel
        psf = np.abs(ctf)
        if psf.sum() > 0:
            psf = psf / psf.sum()
        results["rl_ctf_20iter"] = run_and_measure(
            lambda: richardson_lucy_2d(measurement, psf, iterations=20), gt)

    return results


def test_cacti(data: Dict) -> Dict:
    """CACTI: Temporal coded aperture imaging."""
    gt = data["x_true"]  # (256, 256, 8)
    y = data["y"]        # (256, 256) compressed
    H = data["H_ideal"]  # (256, 256, 8) masks

    results = {}

    # Simple baseline: scale y by mask and replicate
    try:
        start = time.time()
        recon = np.zeros_like(gt)
        for t in range(gt.shape[2]):
            mask_t = H[:, :, t]
            safe_mask = np.where(mask_t > 0.01, mask_t, 1.0)
            recon[:, :, t] = y / safe_mask
        recon = np.clip(recon, 0, gt.max())
        exec_time = time.time() - start

        psnr = compute_psnr(gt, recon)
        ssim = compute_ssim(gt, recon)
        results["mask_division_baseline"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": exec_time, "result_shape": list(recon.shape)
        }
    except Exception as e:
        results["mask_division_baseline"] = {"status": f"error: {str(e)[:80]}"}

    # Try GAP-TV if available
    try:
        from pwm_core.recon.gap_tv import gap_tv_cacti
        start = time.time()
        recon = gap_tv_cacti(y, H, lam=0.1, iterations=20)
        exec_time = time.time() - start
        psnr = compute_psnr(gt, recon)
        ssim = compute_ssim(gt, recon)
        results["gap_tv"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": exec_time, "result_shape": list(recon.shape)
        }
    except Exception as e:
        results["gap_tv"] = {"status": f"error: {str(e)[:80]}"}

    return results


def test_cbct(data: Dict) -> Dict:
    """CBCT: FBP on cone-beam sinogram (angles in degrees)."""
    from pwm_core.recon.ct_solvers import fbp_2d

    sino = data.get("y")
    angles = data.get("H_ideal")  # angles array
    gt = data.get("x_true")

    if sino is None or angles is None or gt is None:
        return {"missing_data": {"status": "missing_data"}}

    angles_rad = np.deg2rad(angles.astype(np.float64))
    out_size = gt.shape[0]

    results = {}
    results["fbp_ramlak"] = run_and_measure(
        lambda: fbp_2d(sino, angles_rad, "ramlak", out_size), gt)
    results["fbp_shepp_logan"] = run_and_measure(
        lambda: fbp_2d(sino, angles_rad, "shepp_logan", out_size), gt)
    return results


def _microscopy_rl_test(data: Dict, modality_name: str) -> Dict:
    """Generic microscopy test: RL deconv with normalized PSF + precomputed baseline."""
    from pwm_core.recon.richardson_lucy import richardson_lucy_2d

    measurement = data["y"]
    gt = data["x_true"]
    results = {}

    # Precomputed baseline
    if "reconstruction_baseline" in data:
        rb = data["reconstruction_baseline"]
        psnr = compute_psnr(gt, rb)
        ssim = compute_ssim(gt, rb)
        results["precomputed_baseline"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(rb.shape)
        }

    # RL with PSF from H_ideal (normalize to sum=1)
    if "H_ideal" in data:
        h = data["H_ideal"].copy()
        # Use center crop as PSF if H_ideal is full-size (same as image)
        if h.shape == measurement.shape:
            cy, cx = h.shape[0] // 2, h.shape[1] // 2
            ks = min(21, h.shape[0])
            half = ks // 2
            psf = h[cy - half:cy + half + 1, cx - half:cx + half + 1]
        else:
            psf = h
        psf = np.abs(psf)
        if psf.sum() > 0:
            psf = psf / psf.sum()
        results["rl_20iter"] = run_and_measure(
            lambda: richardson_lucy_2d(measurement, psf, iterations=20), gt)

    return results


def test_confocal_3d(data: Dict) -> Dict:
    return _microscopy_rl_test(data, "confocal_3d")


def test_lightsheet(data: Dict) -> Dict:
    from pwm_core.recon.lightsheet_solver import fourier_notch_destripe

    results = _microscopy_rl_test(data, "lightsheet")

    # Also try Fourier notch destripe
    measurement = data["y"]
    gt = data["x_true"]
    results["fourier_notch"] = run_and_measure(
        lambda: fourier_notch_destripe(measurement), gt)

    return results


def test_palm_storm(data: Dict) -> Dict:
    return _microscopy_rl_test(data, "palm_storm")


def test_sim(data: Dict) -> Dict:
    """SIM: Wiener-SIM from raw_frames + precomputed baseline."""
    gt = data["x_true"]
    results = {}

    # Precomputed baseline
    if "reconstruction_baseline" in data:
        rb = data["reconstruction_baseline"]
        psnr = compute_psnr(gt, rb)
        ssim = compute_ssim(gt, rb)
        results["precomputed_baseline"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(rb.shape)
        }

    # Wiener SIM reconstruction from raw_frames
    if "raw_frames" in data:
        try:
            from pwm_core.recon.sim_solver import wiener_sim_2d
            raw = data["raw_frames"]  # (9, 256, 256)
            start = time.time()
            sim_recon = wiener_sim_2d(raw, n_angles=3, n_phases=3)
            exec_time = time.time() - start
            # SIM output is 2x resolution, downsample to compare
            if sim_recon.shape != gt.shape:
                from scipy.ndimage import zoom
                scale = [gt.shape[i] / sim_recon.shape[i] for i in range(2)]
                sim_recon = zoom(sim_recon, scale, order=1)
            psnr = compute_psnr(gt, sim_recon)
            ssim = compute_ssim(gt, sim_recon)
            results["wiener_sim"] = {
                "status": "completed", "psnr_db": psnr, "ssim": ssim,
                "exec_time_sec": exec_time, "result_shape": list(sim_recon.shape)
            }
        except Exception as e:
            results["wiener_sim"] = {"status": f"error: {str(e)[:80]}"}

    return results


def test_sted(data: Dict) -> Dict:
    return _microscopy_rl_test(data, "sted")


def test_two_photon(data: Dict) -> Dict:
    return _microscopy_rl_test(data, "two_photon")


def _precomputed_baseline_test(data: Dict) -> Dict:
    """Generic test: compare reconstruction_baseline vs x_true."""
    gt = data.get("x_true")
    rb = data.get("reconstruction_baseline")
    results = {}
    if gt is not None and rb is not None:
        if np.iscomplexobj(gt):
            gt = np.abs(gt)
        if np.iscomplexobj(rb):
            rb = np.abs(rb)
        psnr = compute_psnr(gt, rb)
        ssim = compute_ssim(gt, rb)
        results["precomputed_baseline"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": 0.0, "result_shape": list(rb.shape)
        }
    else:
        results["precomputed_baseline"] = {"status": "missing_data"}
    return results


def test_sem(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_tem(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_widefield(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_photoacoustic(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_holography(data: Dict) -> Dict:
    """Holography: phase retrieval from intensity pattern."""
    results = {}
    y = data.get("y")
    amp = data.get("x_true_amplitude")
    phase = data.get("x_true_phase")

    if amp is not None and y is not None:
        # Baseline: take sqrt of intensity as amplitude
        start = time.time()
        amp_recon = np.sqrt(np.maximum(y.astype(np.float32), 0))
        exec_time = time.time() - start
        psnr = compute_psnr(amp, amp_recon)
        ssim = compute_ssim(amp, amp_recon)
        results["sqrt_intensity_amplitude"] = {
            "status": "completed", "psnr_db": psnr, "ssim": ssim,
            "exec_time_sec": exec_time, "result_shape": list(amp_recon.shape)
        }
    return results


def test_ptychography(data: Dict) -> Dict:
    """Ptychography: compare precomputed baseline."""
    results = _precomputed_baseline_test(data)
    # Also check phase recovery baseline
    gt_phase = data.get("x_true_phase")
    rb = data.get("reconstruction_baseline")
    if gt_phase is not None and rb is not None:
        psnr_p = compute_psnr(gt_phase, rb)
        ssim_p = compute_ssim(gt_phase, rb)
        results["precomputed_phase_baseline"] = {
            "status": "completed", "psnr_db": psnr_p, "ssim": ssim_p,
            "exec_time_sec": 0.0, "result_shape": list(rb.shape)
        }
    return results


def test_lensless(data: Dict) -> Dict:
    """Lensless: Wiener deconvolution baseline."""
    gt = data.get("x_true")
    y = data.get("y")
    H = data.get("H_ideal")
    results = {}
    if gt is None or y is None or H is None:
        return {"missing_data": {"status": "missing_data"}}
    # Simple Wiener in frequency domain
    start = time.time()
    Y = np.fft.fft2(y.astype(np.float32))
    H_f = np.fft.fft2(H.astype(np.float32), s=y.shape)
    lam = 0.01
    recon = np.real(np.fft.ifft2(Y * np.conj(H_f) / (np.abs(H_f)**2 + lam)))
    recon = np.clip(recon, 0, None).astype(np.float32)
    exec_time = time.time() - start
    psnr = compute_psnr(gt, recon)
    ssim = compute_ssim(gt, recon)
    results["wiener_deconv"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": exec_time, "result_shape": list(recon.shape)
    }
    return results


def test_gaussian_splatting(data: Dict) -> Dict:
    """Gaussian splatting: alpha-blending baseline."""
    gt = data.get("x_true")  # (H, W, 3) or (H, W)
    y = data.get("y")         # may be (N_views, H, W, 3) multi-view
    results = {}
    if gt is None or y is None:
        return {"missing_data": {"status": "missing_data"}}
    # Use first view if multi-view
    y_eval = y[0] if y.ndim == 4 else y
    if y_eval.shape != gt.shape:
        # Try matching shapes
        if y_eval.ndim == 3 and gt.ndim == 3 and y_eval.shape[2] == gt.shape[2]:
            pass  # same shape
        elif y_eval.ndim == 2 and gt.ndim == 3:
            y_eval = y_eval[..., np.newaxis]
    psnr = compute_psnr(gt, y_eval)
    ssim = compute_ssim(gt.mean(-1) if gt.ndim == 3 else gt,
                        y_eval.mean(-1) if y_eval.ndim == 3 else y_eval)
    results["direct_render_baseline"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": 0.0, "result_shape": list(y_eval.shape)
    }
    return results


def test_sar(data: Dict) -> Dict:
    """SAR: Compare precomputed Lee filter baseline."""
    return _precomputed_baseline_test(data)


def test_insar(data: Dict) -> Dict:
    """InSAR: Phase unwrapping baseline."""
    gt = data.get("x_true")
    y = data.get("y")  # wrapped interferogram
    results = {}
    if gt is None or y is None:
        return {"missing_data": {"status": "missing_data"}}
    # Baseline: no unwrapping (wrapped output)
    psnr = compute_psnr(gt, y.astype(np.float32))
    ssim = compute_ssim(gt, y.astype(np.float32))
    results["wrapped_phase_baseline"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": 0.0, "result_shape": list(y.shape)
    }
    return results


def test_multispectral_sat(data: Dict) -> Dict:
    """Multispectral satellite: pansharpening baseline (upsample ms_lr)."""
    gt = data.get("x_true")
    ms_lr = data.get("ms_lr")
    pan_hr = data.get("pan_hr")
    results = {}
    if gt is None or ms_lr is None:
        return {"missing_data": {"status": "missing_data"}}
    # Baseline: upsample ms_lr using bicubic interpolation
    from scipy.ndimage import zoom
    start = time.time()
    scale = gt.shape[1] / ms_lr.shape[1]
    recon = np.stack([
        zoom(ms_lr[b], scale, order=3).astype(np.float32) for b in range(ms_lr.shape[0])
    ])
    exec_time = time.time() - start
    psnr = compute_psnr(gt, recon)
    ssim = compute_ssim(gt.mean(0), recon.mean(0))  # compare mean band
    results["bicubic_upsample"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": exec_time, "result_shape": list(recon.shape)
    }
    return results


def test_asl_mri(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_mrs(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_mr_fingerprinting(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_mr_elastography(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_mra(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_swi(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_fpm(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_odt(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_phase_retrieval(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def _multichannel_precomputed_test(data: Dict) -> Dict:
    """For modalities where x_true has multiple channels but baseline is single-channel."""
    gt = data.get("x_true")
    rb = data.get("reconstruction_baseline")
    results = {}
    if gt is None or rb is None:
        results["precomputed_baseline"] = {"status": "missing_data"}
        return results
    # Use first channel of gt if multi-channel
    gt_eval = gt[0] if gt.ndim == 3 else gt
    if np.iscomplexobj(gt_eval):
        gt_eval = np.abs(gt_eval)
    if np.iscomplexobj(rb):
        rb = np.abs(rb)
    psnr = compute_psnr(gt_eval, rb)
    ssim = compute_ssim(gt_eval, rb)
    results["precomputed_baseline"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": 0.0, "result_shape": list(rb.shape)
    }
    return results


def test_pet_ct(data: Dict) -> Dict:
    return _multichannel_precomputed_test(data)


def test_pet_mr(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_spect_ct(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_spectral_ct(data: Dict) -> Dict:
    return _multichannel_precomputed_test(data)


def test_eels(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_industrial_ct(data: Dict) -> Dict:
    return _precomputed_baseline_test(data)


def test_sd_cassi(data: Dict) -> Dict:
    """SD-CASSI: coded aperture spectral imaging baseline."""
    gt = data.get("x_true")   # (H, W, n_bands) spectral cube
    y = data.get("y")          # (H, W+bands-1) compressed measurement
    H = data.get("H_ideal")    # (H, W) coded aperture mask
    results = {}
    if gt is None or y is None:
        return {"missing_data": {"status": "missing_data"}}
    # Baseline: replicate y across spectral bands (unmixing-free estimate)
    start = time.time()
    n_bands = gt.shape[2] if gt.ndim == 3 else 1
    H_size = gt.shape[1]
    # Use center slice of y for each band
    recon = np.zeros_like(gt, dtype=np.float32)
    for b in range(n_bands):
        col_start = b
        col_end = col_start + H_size
        if col_end <= y.shape[1]:
            recon[:, :, b] = y[:, col_start:col_end].astype(np.float32)
        else:
            recon[:, :, b] = y[:, :H_size].astype(np.float32)
    exec_time = time.time() - start
    psnr = compute_psnr(gt.astype(np.float32), recon)
    ssim = compute_ssim(gt[:, :, 0].astype(np.float32), recon[:, :, 0])
    results["spectral_shift_baseline"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": exec_time, "result_shape": list(recon.shape)
    }
    return results


def test_spc_kronecker(data: Dict) -> Dict:
    """SPC Kronecker: single-pixel camera Kronecker structured baseline."""
    gt = data.get("x_true")   # (N, N)
    y = data.get("y")          # (m_row, m_col) Kronecker measurements
    H = data.get("H_ideal")    # (m_col, N^2) or similar
    results = {}
    if gt is None or y is None:
        return {"missing_data": {"status": "missing_data"}}
    # Baseline: column-sum backprojection
    start = time.time()
    # y is (m_row, m_col), H is (m_col, N_col^2)
    # Simple baseline: compute row sum / col sum and outer product
    row_sum = y.sum(axis=1).astype(np.float64)  # (m_row,)
    col_sum = y.sum(axis=0).astype(np.float64)  # (m_col,)
    N = gt.shape[0]
    # Reshape col_sum to image estimate
    n_col = int(np.sqrt(len(col_sum)))
    n_row = int(np.sqrt(len(row_sum)))
    col_img = col_sum[:n_col*n_col].reshape(n_col, n_col) if n_col*n_col <= len(col_sum) else np.outer(row_sum[:N], col_sum[:N])
    # Resize to gt shape
    from scipy.ndimage import zoom
    if col_img.shape != gt.shape:
        scale = [gt.shape[i] / col_img.shape[i] for i in range(2)]
        col_img = zoom(col_img, scale, order=1)
    recon = col_img.astype(np.float32)
    exec_time = time.time() - start
    psnr = compute_psnr(gt.astype(np.float32), recon)
    ssim = compute_ssim(gt.astype(np.float32), recon)
    results["backprojection_baseline"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": exec_time, "result_shape": list(recon.shape)
    }
    return results


# ── Main Driver ──────────────────────────────────────────────────────────────

MODALITY_TESTS = {
    # Core medical imaging
    "ct": ("X-ray CT (Radon FBP/SART)", test_ct),
    "mri": ("MRI (k-space ZF/CS/SENSE)", test_mri),
    "pet": ("PET (Radon FBP)", test_pet),
    "spect": ("SPECT (Radon FBP)", test_spect),
    "ultrasound": ("Ultrasound (RL deconv)", test_ultrasound),
    "oct": ("OCT (B-scan baseline)", test_oct),
    "mammography": ("Mammography (precomputed)", test_mammography),
    "fundus": ("Fundus (RL deconv)", test_fundus),
    "endoscopy": ("Endoscopy (RL deconv)", test_endoscopy),
    "fmri": ("fMRI (ZF-IFFT)", test_fmri),
    "diffusion_mri": ("Diffusion MRI (ZF-IFFT)", test_diffusion_mri),
    "cbct": ("CBCT (cone-beam CT FBP)", test_cbct),
    # MRI variants
    "asl_mri": ("ASL-MRI (precomputed)", test_asl_mri),
    "mrs": ("MRS (precomputed)", test_mrs),
    "mr_fingerprinting": ("MR Fingerprinting (precomputed)", test_mr_fingerprinting),
    "mr_elastography": ("MR Elastography (precomputed)", test_mr_elastography),
    "mra": ("MRA (precomputed)", test_mra),
    "swi": ("SWI (precomputed)", test_swi),
    # Nuclear medicine
    "pet_ct": ("PET-CT (precomputed)", test_pet_ct),
    "pet_mr": ("PET-MR (precomputed)", test_pet_mr),
    "spect_ct": ("SPECT-CT (precomputed)", test_spect_ct),
    "spectral_ct": ("Spectral CT (precomputed)", test_spectral_ct),
    "industrial_ct": ("Industrial CT (precomputed)", test_industrial_ct),
    # Microscopy
    "cryo_em": ("Cryo-EM (Wiener CTF)", test_cryo_em),
    "sem": ("SEM (precomputed)", test_sem),
    "tem": ("TEM (precomputed)", test_tem),
    "widefield": ("Widefield (precomputed)", test_widefield),
    "confocal_3d": ("Confocal 3D (RL deconv)", test_confocal_3d),
    "lightsheet": ("Lightsheet (RL + destripe)", test_lightsheet),
    "palm_storm": ("PALM/STORM (RL deconv)", test_palm_storm),
    "sim": ("SIM (Wiener-SIM)", test_sim),
    "sted": ("STED (RL deconv)", test_sted),
    "two_photon": ("Two-photon (RL deconv)", test_two_photon),
    "eels": ("EELS (precomputed)", test_eels),
    # Computational imaging
    "cacti": ("CACTI (coded aperture)", test_cacti),
    "holography": ("Holography (phase retrieval)", test_holography),
    "ptychography": ("Ptychography (precomputed)", test_ptychography),
    "lensless": ("Lensless (Wiener deconv)", test_lensless),
    "gaussian_splatting": ("Gaussian Splatting (alpha-blend)", test_gaussian_splatting),
    "fpm": ("FPM (precomputed)", test_fpm),
    "odt": ("ODT (precomputed)", test_odt),
    "phase_retrieval": ("Phase Retrieval (precomputed)", test_phase_retrieval),
    "sd_cassi": ("SD-CASSI (coded aperture)", test_sd_cassi),
    "spc_kronecker": ("SPC Kronecker (backprojection)", test_spc_kronecker),
    # Remote sensing
    "sar": ("SAR (Lee filter baseline)", test_sar),
    "insar": ("InSAR (phase unwrapping)", test_insar),
    "multispectral_sat": ("Multispectral Sat (pansharpening)", test_multispectral_sat),
    # Photoacoustic
    "photoacoustic": ("Photoacoustic (precomputed)", test_photoacoustic),
}

# ── Auto-register all remaining modalities using precomputed baseline ──────────
def test_light_field(data: Dict) -> Dict:
    """Light field: shift-add reconstruction."""
    gt = data.get("x_true")  # (H, W)
    rb = data.get("reconstruction_baseline")  # may be smaller
    results = {}
    if gt is None or rb is None:
        return {"missing_data": {"status": "missing_data"}}
    # Resize rb to match gt if needed
    if rb.shape != gt.shape:
        from scipy.ndimage import zoom
        scale = [gt.shape[i] / rb.shape[i] for i in range(2)]
        rb = zoom(rb, scale, order=1).astype(np.float32)
    psnr = compute_psnr(gt, rb)
    ssim = compute_ssim(gt, rb)
    results["precomputed_baseline"] = {
        "status": "completed", "psnr_db": psnr, "ssim": ssim,
        "exec_time_sec": 0.0, "result_shape": list(rb.shape)
    }
    return results


_ALL_PRECOMPUTED = [
    "acoustic_emission", "acoustic_microscopy", "active_thermography", "adaptive_optics",
    "afm", "angiography", "atom_probe", "bioluminescence_tomo", "brachytherapy_img",
    "brillouin", "cars", "cathodoluminescence", "cest_mri", "ceus", "clem",
    "coded_exposure", "confocal_endomicroscopy", "confocal_livecell", "coronagraphy",
    "cryo_et", "ct_fluorescence", "cup", "dark_field", "desi", "dexa", "dic",
    "digital_breast_tomo", "dna_paint", "ebsd", "eddy_current", "edx_mapping",
    "eht_imaging", "elastography", "electron_diffraction", "electron_holography",
    "electron_tomography", "entangled_photon", "event_camera", "expansion", "fib_sem",
    "flash_lidar", "flim", "fluoroscopy", "ftir_imaging", "fwi", "ghost_imaging",
    "gpr", "gravitational_wave", "hdr_imaging", "hyperspectral_remote",
    "impedance_tomo", "integral", "ism", "ivus", "lattice_lightsheet", "libs",
    "lidar", "lucky_imaging", "machine_vision", "magnetic_particle",
    "maldi_msi", "matrix", "mfm", "minflux", "muon_tomo", "nerf",
    "neutron_diffraction", "neutron_tomo", "nirs_brain", "nsom", "ocean_acoustic_tomo",
    "ocean_color", "octa", "panorama", "particle_calorimetry", "passive_microwave",
    "phase_contrast", "photometric_stereo", "polarization", "polsar", "portal_imaging",
    "proton_radiography", "proton_therapy_img", "pump_probe", "quantum_illumination",
    "radio_astronomy", "radio_interferometry", "raman_imaging", "saxs", "seismic_tomo",
    "shearography", "shg", "sims", "solar_imaging", "sonar", "spinning_disk", "srs",
    "stem", "stm", "streak_camera", "structured_light", "talbot_lau", "terahertz",
    "three_photon", "tirf", "tof_camera", "ultrasonic_phased_array", "us_mri",
    "waxs", "weather_radar", "widefield_lowdose", "xfel_sfx", "xray_crystallography",
    "xray_ndt", "xray_radiography", "xrf_imaging", "xrf_tomo",
    "doppler_ultrasound", "dot",
]
MODALITY_TESTS["light_field"] = ("Light Field (shift-add)", test_light_field)
for _mod in _ALL_PRECOMPUTED:
    if _mod not in MODALITY_TESTS:
        MODALITY_TESTS[_mod] = (f"{_mod.replace('_', ' ').title()} (precomputed)", _precomputed_baseline_test)


def main():
    print("\n" + "=" * 80)
    print("PWM5 GPU SERVER - COMPREHENSIVE ALGORITHM TESTING")
    print(f"Testing {len(MODALITY_TESTS)} modalities with all available solvers")
    print("=" * 80 + "\n")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tier": "public",
        "gpu_server": os.environ.get("HOSTNAME", "local-gpu-server"),
        "modalities": {},
        "summary": {
            "total_modalities": len(MODALITY_TESTS),
            "completed": 0,
            "partial": 0,
            "failed": 0,
            "total_solvers": 0,
            "solvers_passed": 0,
        }
    }

    for mod_id, (desc, test_fn) in MODALITY_TESTS.items():
        print(f"{mod_id:20} {desc}")

        sample = load_sample(mod_id, "public", 0)
        if not sample:
            print(f"  >> NO DATA\n")
            all_results["summary"]["failed"] += 1
            continue

        try:
            solver_results = test_fn(sample["data"])
        except Exception as e:
            print(f"  >> ERROR: {str(e)[:60]}\n")
            all_results["summary"]["failed"] += 1
            continue

        n_total = len(solver_results)
        n_pass = 0
        for sname, sres in solver_results.items():
            status = sres.get("status", "unknown")
            if status == "completed" and sres.get("psnr_db") is not None:
                psnr = sres["psnr_db"]
                ssim = sres["ssim"]
                t = sres.get("exec_time_sec", 0)
                print(f"  {sname:30} PSNR={psnr:7.2f} dB  SSIM={ssim:.4f}  t={t:.2f}s")
                n_pass += 1
            else:
                print(f"  {sname:30} {status}")

        all_results["modalities"][mod_id] = {
            "description": desc,
            "sample": sample["sample"],
            "solvers": solver_results,
        }
        all_results["summary"]["total_solvers"] += n_total
        all_results["summary"]["solvers_passed"] += n_pass

        if n_pass == n_total:
            all_results["summary"]["completed"] += 1
        elif n_pass > 0:
            all_results["summary"]["partial"] += 1
        else:
            all_results["summary"]["failed"] += 1
        print()

    # Save
    out_path = RESULTS_DIR / "comprehensive_algorithm_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    s = all_results["summary"]
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Modalities completed:   {s['completed']}")
    print(f"Partial:                {s['partial']}")
    print(f"Failed:                 {s['failed']}")
    print(f"Total solver tests:     {s['total_solvers']}")
    print(f"Passed:                 {s['solvers_passed']}")
    if s["total_solvers"] > 0:
        rate = 100 * s["solvers_passed"] / s["total_solvers"]
        print(f"Pass rate:              {rate:.1f}%")
    print("=" * 80)
    print(f"Results: {out_path}\n")


if __name__ == "__main__":
    main()
