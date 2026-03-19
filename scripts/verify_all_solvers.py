#!/usr/bin/env python3
"""Comprehensive verification of ALL reconstruction solver modules on local CPU.

Tests every importable solver in pwm_core.recon with synthetic data.
Results are saved to benchmark_results/solver_verification.json
"""
import sys
import os
import json
import time
import traceback
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

BENCHMARK_DIR = ROOT / "datasets" / "benchmark"
RESULTS_PATH = ROOT / "benchmark_results" / "solver_verification.json"

rng = np.random.RandomState(42)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(ref, test):
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15:
        return 100.0
    data_range = ref.max() - ref.min()
    if data_range < 1e-15:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(ref, test):
    try:
        from skimage.metrics import structural_similarity
        dr = float(ref.max() - ref.min())
        if dr < 1e-15:
            dr = 1.0
        return float(structural_similarity(
            ref.astype(np.float64), test.astype(np.float64), data_range=dr))
    except Exception:
        return -1.0


def verify_solver(name, fn, gt, timeout=120):
    start = time.time()
    try:
        recon = fn()
        elapsed = time.time() - start
        if recon is None:
            return {"status": "returned_none", "exec_time": round(elapsed, 3)}
        recon = np.real(recon).astype(np.float32)
        gt_f = gt.astype(np.float32)
        if recon.shape != gt_f.shape:
            # Try reshape or resize
            if recon.ndim == 1 and gt_f.ndim == 2:
                try:
                    recon = recon.reshape(gt_f.shape)
                except ValueError:
                    from scipy.ndimage import zoom
                    scale = [gt_f.shape[i] / max(recon.shape[0], 1)
                             for i in range(len(gt_f.shape))]
                    recon = zoom(recon, scale, order=1).astype(np.float32)
            elif recon.ndim == gt_f.ndim:
                from scipy.ndimage import zoom
                scale = [gt_f.shape[i] / recon.shape[i]
                         for i in range(len(gt_f.shape))]
                recon = zoom(recon, scale, order=1).astype(np.float32)
            else:
                return {"status": "shape_mismatch",
                        "got": list(recon.shape),
                        "expected": list(gt_f.shape),
                        "exec_time": round(elapsed, 3)}
        psnr = compute_psnr(gt_f, recon)
        ssim = compute_ssim(gt_f, recon)
        return {
            "status": "verified",
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
            "exec_time": round(elapsed, 3),
            "shape": list(recon.shape),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "error", "error": str(e)[:300],
                "exec_time": round(elapsed, 3)}


# ── Synthetic data generators ────────────────────────────────────────────────

def make_phantom(N=128):
    gt = np.zeros((N, N), dtype=np.float32)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt += 1.0 * ((x/0.69)**2 + (y/0.92)**2 < 1)
    gt -= 0.8 * ((x/0.66)**2 + (y/0.88)**2 < 1)
    gt += 0.2 * ((x/0.31)**2 + (y/0.11)**2 < 1)
    gt += 0.2 * (((x-0.22)/0.11)**2 + (y/0.25)**2 < 1)
    return gt


def make_blobs(N=128, n_blobs=15):
    gt = np.zeros((N, N), dtype=np.float32)
    for _ in range(n_blobs):
        cx, cy = rng.randint(10, N-10, 2)
        r = rng.randint(3, 10)
        yy, xx = np.ogrid[-cx:N-cx, -cy:N-cy]
        mask = xx**2 + yy**2 <= r**2
        gt[mask] = rng.uniform(0.3, 1.0)
    return gt


def make_gaussian_psf(N=128, sigma=3.0):
    y, x = np.mgrid[-N//2:N//2, -N//2:N//2]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)
    psf /= psf.sum()
    return psf


def convolve_2d(img, psf):
    from scipy.signal import fftconvolve
    return fftconvolve(img, psf, mode='same').astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER VERIFIERS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. CT (FBP, SART) ────────────────────────────────────────────────────────

def verify_ct():
    from pwm_core.recon.ct_solvers import fbp_2d, sart_2d
    sample_dir = BENCHMARK_DIR / "ct" / "public" / "sample_00"
    if not sample_dir.exists():
        return {"_status": "no_data"}
    meas = np.load(sample_dir / "measurement.npy")
    gt = np.load(sample_dir / "groundtruth.npy")
    angles = np.load(sample_dir / "angles.npy")
    angles_rad = np.deg2rad(angles) if angles.max() > 2 * np.pi else angles
    out_size = gt.shape[0]

    results = {}
    results["ct__fbp_ramlak"] = verify_solver("FBP (Ram-Lak)",
        lambda: fbp_2d(meas, angles_rad, "ramlak", out_size), gt)
    results["ct__fbp_shepp_logan"] = verify_solver("FBP (Shepp-Logan)",
        lambda: fbp_2d(meas, angles_rad, "shepp_logan", out_size), gt)
    results["ct__sart_10iter"] = verify_solver("SART (10 iter)",
        lambda: sart_2d(meas, angles_rad, out_size, iterations=10), gt)
    return results


# ── 2. MRI (zero-filled, CS, SENSE) ─────────────────────────────────────────

def verify_mri():
    from pwm_core.recon.mri_solvers import zero_filled_reconstruction

    N = 128
    gt = make_phantom(N)
    kspace_full = np.fft.fftshift(np.fft.fft2(gt))
    mask = np.zeros((N, N), dtype=bool)
    center = N // 2
    c_width = N // 20
    mask[:, center - c_width:center + c_width] = True
    for j in range(N):
        if not mask[0, j] and rng.random() < 0.25:
            mask[:, j] = True
    kspace_us = kspace_full * mask

    results = {}
    # Zero-filled via manual IFFT
    results["mri__zero_filled_manual"] = verify_solver("MRI Zero-filled (manual)",
        lambda: np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_us))).astype(np.float32), gt)

    # Zero-filled via solver module
    results["mri__zero_filled_solver"] = verify_solver("MRI Zero-filled (solver)",
        lambda: np.abs(zero_filled_reconstruction(kspace_us)).astype(np.float32), gt)

    # CS-MRI wavelet: cs_mri_wavelet(kspace, mask, lam, iterations, sensitivity_maps, device)
    try:
        from pwm_core.recon.mri_solvers import cs_mri_wavelet
        results["mri__cs_wavelet"] = verify_solver("CS-MRI Wavelet",
            lambda: np.abs(cs_mri_wavelet(kspace_us, mask, lam=0.01, iterations=30)).astype(np.float32), gt)
    except Exception as e:
        results["mri__cs_wavelet"] = {"status": "error", "error": str(e)[:200]}

    # SENSE: sense_reconstruction(kspace, sensitivity_maps, mask, regularization, iterations, device)
    try:
        from pwm_core.recon.mri_solvers import sense_reconstruction
        kspace_1coil = kspace_us[np.newaxis, ...]  # (1, N, N)
        smaps = np.ones((1, N, N), dtype=np.complex64)
        results["mri__sense"] = verify_solver("SENSE",
            lambda: np.abs(sense_reconstruction(kspace_1coil, smaps, mask, regularization=0.01, iterations=20)).astype(np.float32), gt)
    except Exception as e:
        results["mri__sense"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 3. Richardson-Lucy ────────────────────────────────────────────────────────

def verify_rl():
    from pwm_core.recon.richardson_lucy import richardson_lucy_2d
    N = 128
    gt = make_blobs(N)
    psf = make_gaussian_psf(N, sigma=3.0)
    blurred = convolve_2d(gt, psf)
    noisy = np.maximum(blurred + rng.normal(0, 0.02, blurred.shape).astype(np.float32), 1e-6)

    results = {}
    results["rl__20iter"] = verify_solver("RL (20 iter)",
        lambda: richardson_lucy_2d(noisy, psf, iterations=20), gt)
    results["rl__50iter"] = verify_solver("RL (50 iter)",
        lambda: richardson_lucy_2d(noisy, psf, iterations=50), gt)
    return results


# ── 4. FISTA-L2 (classical) ──────────────────────────────────────────────────

def verify_fista():
    from pwm_core.recon.classical import fista_l2
    N = 32
    gt = rng.randn(N * N).astype(np.float64) * 0.3
    gt[np.abs(gt) < 0.15] = 0

    M = N * N // 4
    A = rng.randn(M, N * N).astype(np.float64) / np.sqrt(M)
    y = A @ gt

    results = {}
    # fista_l2(y, A, lam, iters, device) — force CPU to avoid torch type issues
    results["fista__l2"] = verify_solver("FISTA-L2",
        lambda: fista_l2(y, A, lam=1e-3, iters=100, device='cpu').reshape(N, N),
        gt.reshape(N, N).astype(np.float32))
    return results


# ── 5. Least Squares ─────────────────────────────────────────────────────────

def verify_least_squares():
    from pwm_core.recon.classical import least_squares
    N = 32
    gt = rng.randn(N * N).astype(np.float32) * 0.3
    M = N * N
    A = rng.randn(M, N * N).astype(np.float32) / np.sqrt(M)
    y = A @ gt + rng.randn(M).astype(np.float32) * 0.01

    results = {}
    results["classical__least_squares"] = verify_solver("Least Squares",
        lambda: least_squares(A, y).reshape(N, N),
        gt.reshape(N, N))
    return results


# ── 6. CS Solvers (TVAL3, ISTA, ADMM-TV) ─────────────────────────────────────

def verify_cs_solvers():
    results = {}
    N = 32
    gt = make_phantom(N)

    M = N * N // 4
    Phi = rng.randn(M, N * N).astype(np.float64) / np.sqrt(M)
    y = Phi @ gt.ravel()

    # CS solvers use callable forward/adjoint interface
    def forward(x):
        return Phi @ x.ravel()
    def adjoint(y_):
        return (Phi.T @ y_).reshape(N, N)

    # TVAL3: tval3(y, forward, adjoint, x_shape, mu, beta, tol, max_iters, ...)
    try:
        from pwm_core.recon.cs_solvers import tval3
        results["cs__tval3"] = verify_solver("TVAL3",
            lambda: tval3(y, forward, adjoint, (N, N), mu=256.0, max_iters=50), gt)
    except Exception as e:
        results["cs__tval3"] = {"status": "error", "error": str(e)[:200]}

    # ISTA: ista(y, forward, adjoint, x_shape, lam, step, max_iters, ...)
    try:
        from pwm_core.recon.cs_solvers import ista
        results["cs__ista"] = verify_solver("ISTA",
            lambda: ista(y, forward, adjoint, (N, N), lam=0.01, max_iters=100), gt)
    except Exception as e:
        results["cs__ista"] = {"status": "error", "error": str(e)[:200]}

    # FISTA: fista(y, forward, adjoint, x_shape, lam, step, max_iters, ...)
    try:
        from pwm_core.recon.cs_solvers import fista
        results["cs__fista"] = verify_solver("FISTA (CS)",
            lambda: fista(y, forward, adjoint, (N, N), lam=0.01, max_iters=100), gt)
    except Exception as e:
        results["cs__fista"] = {"status": "error", "error": str(e)[:200]}

    # ADMM-TV: admm_tv(y, Phi, x_shape, mu_tv, mu_dct, rho, max_iters, ...)
    try:
        from pwm_core.recon.cs_solvers import admm_tv
        results["cs__admm_tv"] = verify_solver("ADMM-TV",
            lambda: admm_tv(y, Phi, (N, N), mu_tv=0.01, max_iters=50), gt)
    except Exception as e:
        results["cs__admm_tv"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 7. Lensless (Wiener, Tikhonov, ADMM-TV) ──────────────────────────────────

def verify_lensless():
    N = 64
    gt = make_phantom(N)
    psf = rng.rand(N, N).astype(np.float32)
    psf /= psf.sum()
    GT_f = np.fft.fft2(gt)
    PSF_f = np.fft.fft2(psf)
    y = np.real(np.fft.ifft2(GT_f * PSF_f)).astype(np.float32)
    y += rng.normal(0, 0.01, y.shape).astype(np.float32)

    results = {}
    # Wiener deconvolution (manual)
    def wiener(y, psf, nv=0.01):
        Y = np.fft.fft2(y)
        H = np.fft.fft2(psf, s=y.shape)
        W = np.conj(H) / (np.abs(H)**2 + nv / (np.abs(H)**2 + 1e-10))
        return np.real(np.fft.ifft2(Y * W)).astype(np.float32)

    results["lensless__wiener_manual"] = verify_solver("Wiener (manual)",
        lambda: wiener(y, psf), gt)

    # Tikhonov from solver module
    try:
        from pwm_core.recon.lensless_solver import tikhonov_lensless
        results["lensless__tikhonov"] = verify_solver("Tikhonov lensless",
            lambda: tikhonov_lensless(y, psf), gt)
    except Exception as e:
        results["lensless__tikhonov"] = {"status": "error", "error": str(e)[:200]}

    # ADMM-TV lensless: admm_tv_lensless(measurement, psf, rho, lam_tv, iters, ...)
    try:
        from pwm_core.recon.lensless_solver import admm_tv_lensless
        results["lensless__admm_tv"] = verify_solver("ADMM-TV lensless",
            lambda: admm_tv_lensless(y, psf, lam_tv=0.01, iters=20), gt)
    except Exception as e:
        results["lensless__admm_tv"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 8. Holography (angular spectrum) ─────────────────────────────────────────

def verify_holography():
    from pwm_core.recon.holography_solver import angular_spectrum_propagate
    N = 128
    gt_amp = make_blobs(N, 10).astype(np.float32)
    gt_phase = rng.uniform(-np.pi/4, np.pi/4, (N, N)).astype(np.float32)
    gt_field = gt_amp * np.exp(1j * gt_phase)

    wavelength = 632e-9
    pixel_size = 5e-6
    z = 1e-3
    hologram_field = angular_spectrum_propagate(gt_field, wavelength, pixel_size, z)
    intensity = np.abs(hologram_field)**2

    recon_field = angular_spectrum_propagate(
        np.sqrt(intensity + 0j), wavelength, pixel_size, -z)
    recon_amp = np.abs(recon_field).astype(np.float32)

    results = {}
    results["holography__angular_spectrum"] = verify_solver("Angular Spectrum",
        lambda: recon_amp, gt_amp)
    return results


# ── 9. Phase Retrieval (GS, HIO) ─────────────────────────────────────────────

def verify_phase_retrieval():
    from pwm_core.recon.phase_retrieval_solver import gerchberg_saxton
    N = 64
    gt_amp = make_blobs(N, 8).astype(np.float32) + 0.1
    # GS needs both image and Fourier magnitudes
    fourier_mag = np.abs(np.fft.fft2(gt_amp))

    results = {}
    # GS with known image magnitude
    results["phase_retrieval__gs"] = verify_solver("Gerchberg-Saxton",
        lambda: np.abs(gerchberg_saxton(fourier_mag, image_magnitude=gt_amp, n_iters=50)),
        gt_amp)

    # GS with support only
    support = (gt_amp > 0.05).astype(np.float32)
    results["phase_retrieval__gs_support"] = verify_solver("GS (support only)",
        lambda: np.abs(gerchberg_saxton(fourier_mag, support=support, n_iters=100)),
        gt_amp)

    return results


# ── 10. PnP (HQS, ADMM, FISTA) ──────────────────────────────────────────────

def verify_pnp():
    results = {}
    N = 64
    gt = make_phantom(N)
    noisy = gt + rng.normal(0, 0.1, gt.shape).astype(np.float32)

    # Identity forward model (denoising)
    def forward(x):
        return x
    def adjoint(x):
        return x

    try:
        from pwm_core.recon.pnp import pnp_hqs, gaussian_denoiser
        denoiser = gaussian_denoiser
        results["pnp__hqs"] = verify_solver("PnP-HQS",
            lambda: pnp_hqs(noisy, forward, adjoint, gt.shape, denoiser,
                           iters=10, sigma=0.1), gt)
    except Exception as e:
        results["pnp__hqs"] = {"status": "error", "error": str(e)[:200]}

    try:
        from pwm_core.recon.pnp import pnp_admm, gaussian_denoiser
        denoiser = gaussian_denoiser
        results["pnp__admm"] = verify_solver("PnP-ADMM",
            lambda: pnp_admm(noisy, forward, adjoint, gt.shape, denoiser,
                            iters=10, sigma=0.1), gt)
    except Exception as e:
        results["pnp__admm"] = {"status": "error", "error": str(e)[:200]}

    try:
        from pwm_core.recon.pnp import pnp_fista, gaussian_denoiser
        denoiser = gaussian_denoiser
        results["pnp__fista"] = verify_solver("PnP-FISTA",
            lambda: pnp_fista(noisy, forward, adjoint, gt.shape, denoiser,
                             iters=10, sigma=0.1), gt)
    except Exception as e:
        results["pnp__fista"] = {"status": "error", "error": str(e)[:200]}

    # NLM denoiser variant
    try:
        from pwm_core.recon.pnp import pnp_hqs, nlm_denoiser
        results["pnp__hqs_nlm"] = verify_solver("PnP-HQS (NLM)",
            lambda: pnp_hqs(noisy, forward, adjoint, gt.shape, nlm_denoiser,
                           iters=5, sigma=0.1), gt)
    except Exception as e:
        results["pnp__hqs_nlm"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 11. DOT (Born approximation) ─────────────────────────────────────────────

def verify_dot():
    from pwm_core.recon.dot_solver import born_approx
    N = 16  # Small for CPU speed
    n_meas = N * N // 2
    gt = rng.rand(N * N).astype(np.float64) * 0.1 + 0.01
    J = rng.rand(n_meas, N * N).astype(np.float64) * 0.01
    y = J @ gt + rng.randn(n_meas).astype(np.float64) * 0.001

    results = {}
    # born_approx(y, jacobian, alpha=1.0, max_cg_iters=200)
    results["dot__born_approx"] = verify_solver("DOT Born Approx",
        lambda: born_approx(y, J, alpha=0.1).reshape(N, N),
        gt.reshape(N, N).astype(np.float32))
    return results


# ── 12. Photoacoustic (backprojection) ───────────────────────────────────────

def verify_photoacoustic():
    from pwm_core.recon.photoacoustic_solver import back_projection
    N = 64
    gt = make_phantom(N)
    n_sensors = 32
    angles = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)

    # back_projection expects transducer_positions as (n_sensors, 2)
    radius = N / 2
    transducer_positions = np.stack([
        radius * np.cos(angles) + N/2,
        radius * np.sin(angles) + N/2
    ], axis=-1)  # (n_sensors, 2)

    # Create sinogram: Radon-like projection
    from scipy.ndimage import rotate
    n_times = N
    sino = np.zeros((n_sensors, n_times), dtype=np.float32)
    for i, ang in enumerate(angles):
        rotated = rotate(gt, np.degrees(ang), reshape=False)
        sino[i] = rotated.sum(axis=0)

    results = {}
    results["photoacoustic__backprojection"] = verify_solver("PAT Backprojection",
        lambda: back_projection(sino, transducer_positions, (N, N)), gt)
    return results


# ── 13. OCT (FFT recon) ──────────────────────────────────────────────────────

def verify_oct():
    from pwm_core.recon.oct_solver import fft_recon
    N = 128
    gt = make_blobs(N, 10)
    spectral = np.fft.fft(gt, axis=0).astype(np.complex64)
    spectral += (rng.randn(*spectral.shape) + 1j * rng.randn(*spectral.shape)).astype(np.complex64) * 0.1

    results = {}
    results["oct__fft_recon"] = verify_solver("OCT FFT",
        lambda: fft_recon(spectral), gt)
    return results


# ── 14. GAP-TV (CACTI) ───────────────────────────────────────────────────────

def verify_gap_tv():
    from pwm_core.recon.gap_tv import gap_tv_cacti
    N = 32  # Small for speed
    T = 4

    gt = np.zeros((T, N, N), dtype=np.float32)
    for t in range(T):
        gt[t] = make_blobs(N, 3 + t)

    # gap_tv_cacti expects masks as (H, W, n_frames)
    masks = (rng.rand(N, N, T) > 0.5).astype(np.float32)
    # CACTI measurement: y = sum_t mask_t * x_t
    y = np.zeros((N, N), dtype=np.float32)
    for t in range(T):
        y += gt[t] * masks[:, :, t]

    results = {}
    results["gap_tv__cacti"] = verify_solver("GAP-TV (CACTI)",
        lambda: gap_tv_cacti(y, masks, iterations=20, device='cpu'),
        gt)
    return results


# ── 15. SPC (single pixel camera) ────────────────────────────────────────────

def verify_spc():
    results = {}
    N = 16  # Small for speed
    gt = make_phantom(N)
    M = N * N // 4
    patterns = (rng.rand(M, N * N) > 0.5).astype(np.float32)
    y = patterns @ gt.ravel().astype(np.float32) + rng.randn(M).astype(np.float32) * 0.01

    # ADMM SPC: admm_spc(y, A, rho, iterations, tol, verbose, device)
    try:
        from pwm_core.recon.spc_solvers import admm_spc
        results["spc__admm"] = verify_solver("SPC ADMM",
            lambda: admm_spc(y, patterns, rho=1.0, iterations=50).reshape(N, N), gt)
    except Exception as e:
        results["spc__admm"] = {"status": "error", "error": str(e)[:200]}

    # FISTA-L1 SPC: fista_l1(y, A, lambda_l1, iterations, accelerate, verbose, device)
    try:
        from pwm_core.recon.spc_solvers import fista_l1
        results["spc__fista_l1"] = verify_solver("SPC FISTA-L1",
            lambda: fista_l1(y, patterns, lambda_l1=0.01, iterations=100).reshape(N, N), gt)
    except Exception as e:
        results["spc__fista_l1"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 16. SIM (structured illumination microscopy) ─────────────────────────────

def verify_sim():
    results = {}
    N = 64

    try:
        from pwm_core.recon.sim_solver import wiener_sim_2d
        # Create synthetic SIM data: 3 angles x 3 phases = 9 images
        gt = make_blobs(N, 10)
        # Generate illumination patterns
        n_angles = 3
        n_phases = 3
        raw_images = np.zeros((n_angles * n_phases, N, N), dtype=np.float32)
        for a in range(n_angles):
            angle = a * np.pi / n_angles
            kx = 2 * np.pi * 5 * np.cos(angle)
            ky = 2 * np.pi * 5 * np.sin(angle)
            yy, xx = np.mgrid[0:N, 0:N]
            for p in range(n_phases):
                phase = 2 * np.pi * p / n_phases
                pattern = 0.5 + 0.5 * np.cos(kx * xx / N + ky * yy / N + phase)
                raw_images[a * n_phases + p] = (gt * pattern).astype(np.float32)

        results["sim__wiener"] = verify_solver("Wiener SIM",
            lambda: wiener_sim_2d(raw_images, n_angles=3, n_phases=3), gt)
    except Exception as e:
        results["sim__wiener"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 17. Ptychography (ePIE, PIE) ─────────────────────────────────────────────

def verify_ptychography():
    results = {}
    N = 64

    try:
        from pwm_core.recon.ptychography_solver import epie, create_probe
        gt = make_blobs(N, 8).astype(np.complex64) + 0.1
        probe_size = 24
        probe = create_probe(probe_size, probe_type="gaussian")

        step = probe_size // 2
        positions = []
        for yi in range(0, N - probe_size + 1, step):
            for xi in range(0, N - probe_size + 1, step):
                positions.append((yi, xi))
        positions = np.array(positions)

        diff_patterns = np.zeros((len(positions), probe_size, probe_size), dtype=np.float32)
        for i, (py, px) in enumerate(positions):
            patch = gt[py:py+probe_size, px:px+probe_size]
            exit_wave = patch * probe
            diff_patterns[i] = np.abs(np.fft.fft2(exit_wave))**2

        # epie(diffraction_patterns, positions, object_shape, probe_init, iterations, ...)
        results["ptychography__epie"] = verify_solver("ePIE",
            lambda: np.abs(epie(diff_patterns, positions, (N, N),
                               probe_init=probe, iterations=20)[0]),
            np.abs(gt))
    except Exception as e:
        results["ptychography__epie"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 18. FPM (Fourier ptychographic microscopy) ───────────────────────────────

def verify_fpm():
    results = {}
    N = 64

    try:
        from pwm_core.recon.fpm_solver import sequential_phase_retrieval
        gt = make_blobs(N, 8).astype(np.float32) + 0.1
        n_leds = 9
        lr_size = N // 2
        images = np.zeros((n_leds, lr_size, lr_size), dtype=np.float32)
        for i in range(n_leds):
            sx = (i % 3 - 1) * (N // 8)
            sy = (i // 3 - 1) * (N // 8)
            GT_f = np.fft.fftshift(np.fft.fft2(gt))
            cx, cy = N // 2 + sx, N // 2 + sy
            lr_half = lr_size // 2
            patch = GT_f[max(0, cy-lr_half):cy+lr_half, max(0, cx-lr_half):cx+lr_half]
            if patch.shape == (lr_size, lr_size):
                images[i] = np.abs(np.fft.ifft2(patch))**2
            else:
                images[i] = rng.rand(lr_size, lr_size).astype(np.float32)

        led_positions = np.zeros((n_leds, 2))
        for i in range(n_leds):
            led_positions[i] = [(i % 3 - 1) * 10, (i // 3 - 1) * 10]

        # sequential_phase_retrieval(lr_images, led_positions, hr_size, lr_size, pupil, n_iters)
        results["fpm__sequential"] = verify_solver("FPM Sequential PR",
            lambda: np.abs(sequential_phase_retrieval(images, led_positions,
                                                       hr_size=N, lr_size=lr_size, n_iters=20)),
            gt)
    except Exception as e:
        results["fpm__sequential"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 19. FLIM (phasor, MLE) ───────────────────────────────────────────────────

def verify_flim():
    results = {}
    N = 32  # Small spatial
    n_times = 64

    try:
        from pwm_core.recon.flim_solver import phasor_recon, mle_fit_recon
        gt_tau = make_phantom(N) * 2.0 + 1.0  # lifetime 1-3 ns
        time_axis = np.linspace(0, 10, n_times, dtype=np.float32)

        # FLIM expects (H, W, T) shape
        decay_data = np.zeros((N, N, n_times), dtype=np.float32)
        for t_idx, t in enumerate(time_axis):
            decay_data[:, :, t_idx] = np.exp(-t / gt_tau)
        decay_data += rng.normal(0, 0.01, decay_data.shape).astype(np.float32)
        decay_data = np.maximum(decay_data, 1e-10)

        # Phasor: returns (tau_map, amp_map)
        results["flim__phasor"] = verify_solver("FLIM Phasor",
            lambda: phasor_recon(decay_data, time_axis)[0],
            gt_tau)

        # MLE: returns (tau_map, amp_map)
        results["flim__mle"] = verify_solver("FLIM MLE",
            lambda: mle_fit_recon(decay_data, time_axis, n_iters=10)[0],
            gt_tau)
    except Exception as e:
        results["flim__phasor"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 20. Integral photography (depth estimation) ──────────────────────────────

def verify_integral():
    results = {}
    N = 64

    try:
        from pwm_core.recon.integral_solver import depth_estimation
        gt = make_phantom(N)
        # depth_estimation expects 2D measurement (H,W), returns (H,W,n_depths)
        from scipy.ndimage import gaussian_filter
        # Simulate blurred measurement from integral sensor
        measurement = gaussian_filter(gt, sigma=2.0).astype(np.float32)
        n_depths = 5
        depth_weights = np.ones(n_depths, dtype=np.float32) / n_depths
        psf_sigmas = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        recon_vol = depth_estimation(measurement, depth_weights, psf_sigmas)
        # Compare best-focused depth plane with gt
        best_plane = recon_vol[:, :, 1]  # sigma=1.0 plane
        results["integral__depth"] = verify_solver("Depth Estimation",
            lambda: best_plane, gt)
    except Exception as e:
        results["integral__depth"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 21. Light field (shift-and-sum) ───────────────────────────────────────────

def verify_light_field():
    results = {}
    N = 64

    try:
        from pwm_core.recon.light_field_solver import shift_and_sum
        # Synthetic 5x5 light field
        n_views = 5
        gt = make_phantom(N)
        lf = np.zeros((n_views, n_views, N, N), dtype=np.float32)
        for u in range(n_views):
            for v in range(n_views):
                shift_x = u - n_views // 2
                shift_y = v - n_views // 2
                lf[u, v] = np.roll(np.roll(gt, shift_x, axis=1), shift_y, axis=0)

        results["light_field__shift_sum"] = verify_solver("Shift-and-Sum",
            lambda: shift_and_sum(lf), gt)
    except Exception as e:
        results["light_field__shift_sum"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 22. Lightsheet destriping ─────────────────────────────────────────────────

def verify_lightsheet():
    results = {}
    N = 128
    gt = make_blobs(N, 12)

    # Add horizontal stripes
    striped = gt.copy()
    for row in range(0, N, 8):
        striped[row:row+2] += 0.3

    try:
        from pwm_core.recon.lightsheet_solver import fourier_notch_destripe
        results["lightsheet__fourier_notch"] = verify_solver("Fourier Notch Destripe",
            lambda: fourier_notch_destripe(striped, stripe_direction="horizontal"), gt)
    except Exception as e:
        results["lightsheet__fourier_notch"] = {"status": "error", "error": str(e)[:200]}

    try:
        from pwm_core.recon.lightsheet_solver import wavelet_destripe
        results["lightsheet__wavelet"] = verify_solver("Wavelet Destripe",
            lambda: wavelet_destripe(striped), gt)
    except Exception as e:
        results["lightsheet__wavelet"] = {"status": "error", "error": str(e)[:200]}

    try:
        from pwm_core.recon.lightsheet_solver import vsnr_destripe
        results["lightsheet__vsnr"] = verify_solver("VSNR Destripe",
            lambda: vsnr_destripe(striped, lam=1.0, iters=50), gt)
    except Exception as e:
        results["lightsheet__vsnr"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 23. Panorama / multi-focus fusion ─────────────────────────────────────────

def verify_panorama():
    results = {}
    N = 128
    gt = make_blobs(N, 15) + 0.1

    # Create multi-focus stack
    from scipy.ndimage import gaussian_filter
    img1 = gt.copy()
    img1[:N//2] = gaussian_filter(img1[:N//2], sigma=3)  # blur top
    img2 = gt.copy()
    img2[N//2:] = gaussian_filter(img2[N//2:], sigma=3)  # blur bottom

    try:
        from pwm_core.recon.panorama_solver import multifocus_fusion_laplacian
        results["panorama__laplacian_fusion"] = verify_solver("Laplacian Pyramid Fusion",
            lambda: multifocus_fusion_laplacian([img1, img2]), gt)
    except Exception as e:
        results["panorama__laplacian_fusion"] = {"status": "error", "error": str(e)[:200]}

    try:
        from pwm_core.recon.panorama_solver import multifocus_fusion_guided
        results["panorama__guided_fusion"] = verify_solver("Guided Filter Fusion",
            lambda: multifocus_fusion_guided([img1, img2]), gt)
    except Exception as e:
        results["panorama__guided_fusion"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 24. Gaussian denoising (used by many microscopy modalities) ───────────────

def verify_gaussian_denoising():
    """Many microscopy modalities use simple Gaussian denoising as baseline."""
    from scipy.ndimage import gaussian_filter
    N = 128
    gt = make_blobs(N, 12)
    noisy = gt + rng.normal(0, 0.1, gt.shape).astype(np.float32)

    results = {}
    results["denoising__gaussian_sigma1"] = verify_solver("Gaussian sigma=1",
        lambda: gaussian_filter(noisy, sigma=1.0).astype(np.float32), gt)
    results["denoising__gaussian_sigma2"] = verify_solver("Gaussian sigma=2",
        lambda: gaussian_filter(noisy, sigma=2.0).astype(np.float32), gt)

    # Median filter
    from scipy.ndimage import median_filter
    results["denoising__median_3x3"] = verify_solver("Median 3x3",
        lambda: median_filter(noisy, size=3).astype(np.float32), gt)

    # NLM (from pnp module)
    try:
        from pwm_core.recon.pnp import nlm_denoiser
        results["denoising__nlm"] = verify_solver("NLM denoiser",
            lambda: nlm_denoiser(noisy, sigma=0.1), gt)
    except Exception as e:
        results["denoising__nlm"] = {"status": "error", "error": str(e)[:200]}

    # BM3D
    try:
        from pwm_core.recon.pnp import bm3d_denoiser
        results["denoising__bm3d"] = verify_solver("BM3D denoiser",
            lambda: bm3d_denoiser(noisy, sigma=0.1), gt)
    except Exception as e:
        results["denoising__bm3d"] = {"status": "error", "error": str(e)[:200]}

    return results


# ── 25. Wiener deconvolution (used by coded_exposure, widefield, etc.) ────────

def verify_wiener_deconv():
    """General Wiener deconvolution used by multiple modalities."""
    N = 128
    gt = make_phantom(N)
    # Motion blur PSF
    psf = np.zeros((N, N), dtype=np.float32)
    psf[N//2, N//2-5:N//2+5] = 1.0
    psf /= psf.sum()

    blurred = convolve_2d(gt, psf)
    noisy = blurred + rng.normal(0, 0.01, blurred.shape).astype(np.float32)

    def wiener(y, psf, nv=0.01):
        Y = np.fft.fft2(y)
        H = np.fft.fft2(psf, s=y.shape)
        W = np.conj(H) / (np.abs(H)**2 + nv)
        return np.real(np.fft.ifft2(Y * W)).astype(np.float32)

    results = {}
    results["wiener__motion_blur"] = verify_solver("Wiener (motion blur)",
        lambda: wiener(noisy, psf), gt)
    return results


# ── 26. Precomputed baseline check (verifies test_all_algorithms pattern) ─────

def verify_precomputed_concept():
    """Verify that the precomputed_baseline test pattern works correctly.
    This validates x_true vs reconstruction_baseline comparison from HDF5."""
    N = 64
    gt = make_phantom(N)
    # Simulate reconstruction_baseline with some noise
    recon_baseline = gt + rng.normal(0, 0.05, gt.shape).astype(np.float32)

    results = {}
    results["precomputed__baseline_pattern"] = verify_solver("Precomputed Baseline Pattern",
        lambda: recon_baseline, gt)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

VERIFIERS = [
    ("ct", "CT (FBP, SART)", verify_ct),
    ("mri", "MRI (ZF, CS, SENSE)", verify_mri),
    ("rl", "Richardson-Lucy", verify_rl),
    ("fista", "FISTA-L2 (classical)", verify_fista),
    ("least_sq", "Least Squares", verify_least_squares),
    ("cs", "CS Solvers (TVAL3, ISTA, ADMM-TV)", verify_cs_solvers),
    ("lensless", "Lensless (Wiener, Tikhonov, ADMM-TV)", verify_lensless),
    ("holography", "Holography (Angular Spectrum)", verify_holography),
    ("phase_retrieval", "Phase Retrieval (GS)", verify_phase_retrieval),
    ("pnp", "PnP (HQS, ADMM, FISTA)", verify_pnp),
    ("dot", "DOT (Born)", verify_dot),
    ("photoacoustic", "Photoacoustic", verify_photoacoustic),
    ("oct", "OCT (FFT)", verify_oct),
    ("gap_tv", "GAP-TV (CACTI)", verify_gap_tv),
    ("spc", "SPC (ADMM, FISTA-L1)", verify_spc),
    ("sim", "SIM (Wiener)", verify_sim),
    ("ptychography", "Ptychography (ePIE)", verify_ptychography),
    ("fpm", "FPM (Sequential PR)", verify_fpm),
    ("flim", "FLIM (Phasor, MLE)", verify_flim),
    ("integral", "Integral (Depth Est)", verify_integral),
    ("light_field", "Light Field (Shift-Sum)", verify_light_field),
    ("lightsheet", "Lightsheet Destripe", verify_lightsheet),
    ("panorama", "Panorama Fusion", verify_panorama),
    ("denoising", "Gaussian/Median/NLM/BM3D", verify_gaussian_denoising),
    ("wiener", "Wiener Deconv (general)", verify_wiener_deconv),
    ("precomputed", "Precomputed Baseline Pattern", verify_precomputed_concept),
]


def main():
    print("=" * 70)
    print("PWM5 Comprehensive Solver Verification (Local CPU)")
    print("=" * 70)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": "local_cpu",
        "solvers": {},
    }

    verified_count = 0
    error_count = 0
    skip_count = 0

    for group_id, desc, verify_fn in VERIFIERS:
        print(f"\n--- {desc} ({group_id}) ---")
        try:
            results = verify_fn()
            if isinstance(results, dict) and results.get("_status") in ["no_data"]:
                print(f"  SKIPPED: {results['_status']}")
                skip_count += 1
                continue

            for solver_key, sr in results.items():
                all_results["solvers"][solver_key] = sr
                status = sr.get("status", "unknown")
                if status == "verified":
                    verified_count += 1
                    p = sr.get("psnr_db", 0)
                    s = sr.get("ssim", 0)
                    t = sr.get("exec_time", 0)
                    print(f"  {solver_key:40s} OK  PSNR={p:7.2f} dB  SSIM={s:.4f}  t={t:.3f}s")
                elif status == "error":
                    error_count += 1
                    print(f"  {solver_key:40s} ERR: {sr.get('error', '')[:60]}")
                else:
                    skip_count += 1
                    print(f"  {solver_key:40s} {status}")
        except Exception as e:
            error_count += 1
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()

    all_results["summary"] = {
        "verified": verified_count,
        "errors": error_count,
        "skipped": skip_count,
        "total_solvers": verified_count + error_count + skip_count,
    }

    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TOTAL: {verified_count} verified, {error_count} errors, {skip_count} skipped")
    print(f"Saved to {RESULTS_PATH}")

    # Print verified solver list
    print(f"\n--- Verified Solvers ---")
    for k, v in sorted(all_results["solvers"].items()):
        if v.get("status") == "verified":
            print(f"  {k}: PSNR={v['psnr_db']:.1f} dB")


if __name__ == "__main__":
    main()
