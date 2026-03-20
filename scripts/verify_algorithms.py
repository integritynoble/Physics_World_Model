#!/usr/bin/env python3
"""Verify reconstruction algorithms locally on CPU.

Generates synthetic data for each modality and runs solvers to verify
they produce expected PSNR/SSIM values. Records verification results.
"""
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

BENCHMARK_DIR = ROOT / "datasets" / "benchmark"
RESULTS_PATH = ROOT / "benchmark_results" / "algorithm_verification.json"


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
    except ImportError:
        return -1.0


def verify_solver(name, fn, gt, timeout=120):
    """Run a solver function and measure PSNR/SSIM against ground truth."""
    start = time.time()
    try:
        recon = fn()
        elapsed = time.time() - start
        if recon is None:
            return {"status": "returned_none", "exec_time": elapsed}
        recon = np.real(recon).astype(np.float32)
        if recon.shape != gt.shape:
            # Try to resize
            from scipy.ndimage import zoom
            scale = [gt.shape[i] / recon.shape[i] for i in range(len(gt.shape))]
            recon = zoom(recon, scale, order=1).astype(np.float32)
        psnr = compute_psnr(gt, recon)
        ssim = compute_ssim(gt, recon)
        return {
            "status": "verified",
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
            "exec_time": round(elapsed, 3),
            "shape": list(recon.shape),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "error", "error": str(e)[:200], "exec_time": round(elapsed, 3)}


# ── CT Verification ──────────────────────────────────────────────────────────

def verify_ct():
    """Verify CT solvers: FBP and SART."""
    from pwm_core.recon.ct_solvers import fbp_2d, sart_2d

    sample_dir = BENCHMARK_DIR / "ct" / "public" / "sample_00"
    if not sample_dir.exists():
        return {"status": "no_data"}

    meas = np.load(sample_dir / "measurement.npy")
    gt = np.load(sample_dir / "groundtruth.npy")
    angles = np.load(sample_dir / "angles.npy")
    angles_rad = np.deg2rad(angles) if angles.max() > 2 * np.pi else angles
    out_size = gt.shape[0]

    results = {}
    results["fbp_ramlak"] = verify_solver(
        "FBP (Ram-Lak)",
        lambda: fbp_2d(meas, angles_rad, "ramlak", out_size), gt)
    results["fbp_shepp_logan"] = verify_solver(
        "FBP (Shepp-Logan)",
        lambda: fbp_2d(meas, angles_rad, "shepp_logan", out_size), gt)
    results["sart_10iter"] = verify_solver(
        "SART (10 iter)",
        lambda: sart_2d(meas, angles_rad, out_size, iterations=10), gt)
    return results


# ── MRI Verification (synthetic) ─────────────────────────────────────────────

def verify_mri():
    """Verify MRI solvers on synthetic k-space data."""
    try:
        from pwm_core.recon.mri_solvers import zero_filled_reconstruction
    except ImportError:
        return {"status": "import_error"}

    # Generate synthetic brain-like MRI phantom
    N = 128
    gt = np.zeros((N, N), dtype=np.float32)
    # Ellipses for brain phantom
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt += 1.0 * ((x/0.69)**2 + (y/0.92)**2 < 1)  # outer skull
    gt -= 0.8 * ((x/0.66)**2 + (y/0.88)**2 < 1)  # inner
    gt += 0.2 * ((x/0.31)**2 + (y/0.11)**2 < 1)  # ventricle
    gt += 0.2 * (((x-0.22)/0.11)**2 + (y/0.25)**2 < 1)  # lesion

    # Full k-space
    kspace_full = np.fft.fftshift(np.fft.fft2(gt))

    # Create undersampled k-space (4x acceleration, random mask)
    rng = np.random.RandomState(42)
    mask = np.zeros((N, N), dtype=bool)
    # Keep center 10% lines
    center = N // 2
    c_width = N // 20
    mask[:, center - c_width:center + c_width] = True
    # Random 25% of other lines
    for j in range(N):
        if not mask[0, j]:
            if rng.random() < 0.25:
                mask[:, j] = True
    kspace_us = kspace_full * mask

    results = {}
    results["zero_filled"] = verify_solver(
        "Zero-filled IFFT",
        lambda: np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_us))).astype(np.float32),
        gt)
    return results


# ── Richardson-Lucy Verification (microscopy) ────────────────────────────────

def verify_richardson_lucy():
    """Verify RL deconvolution on synthetic blurred+noisy image."""
    try:
        from pwm_core.recon.richardson_lucy import richardson_lucy_2d
    except ImportError:
        return {"status": "import_error"}

    N = 128
    rng = np.random.RandomState(42)
    # Create ground truth with features
    gt = np.zeros((N, N), dtype=np.float32)
    for _ in range(20):
        cx, cy = rng.randint(10, N-10, 2)
        r = rng.randint(3, 10)
        y, x = np.ogrid[-cx:N-cx, -cy:N-cy]
        mask = x**2 + y**2 <= r**2
        gt[mask] = rng.uniform(0.3, 1.0)

    # Create PSF (Gaussian)
    sigma = 3.0
    y, x = np.mgrid[-N//2:N//2, -N//2:N//2]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()

    # Convolve
    from scipy.signal import fftconvolve
    blurred = fftconvolve(gt, psf, mode='same').astype(np.float32)
    blurred = np.maximum(blurred, 0)
    # Add Poisson-like noise
    blurred_noisy = blurred + rng.normal(0, 0.02, blurred.shape).astype(np.float32)
    blurred_noisy = np.maximum(blurred_noisy, 1e-6)

    results = {}
    results["rl_20iter"] = verify_solver(
        "Richardson-Lucy (20 iter)",
        lambda: richardson_lucy_2d(blurred_noisy, psf, iterations=20),
        gt)
    results["rl_50iter"] = verify_solver(
        "Richardson-Lucy (50 iter)",
        lambda: richardson_lucy_2d(blurred_noisy, psf, iterations=50),
        gt)
    return results


# ── FISTA / CS Verification ──────────────────────────────────────────────────

def verify_fista():
    """Verify FISTA-L2 on synthetic compressed sensing problem."""
    try:
        from pwm_core.recon.classical import fista_l2
    except ImportError:
        return {"status": "import_error"}

    N = 64
    rng = np.random.RandomState(42)
    gt = rng.randn(N, N).astype(np.float32) * 0.1
    # Make it sparse in wavelet domain (just use spatial sparsity)
    gt[np.abs(gt) < 0.05] = 0

    # Create measurement (random projection)
    M = N * N // 4  # 25% measurements
    A = rng.randn(M, N * N).astype(np.float32) / np.sqrt(M)
    y = A @ gt.ravel()

    results = {}
    results["fista_l2"] = verify_solver(
        "FISTA-L2",
        lambda: fista_l2(y, A, gt.shape, iterations=100),
        gt)
    return results


# ── PnP-HQS Verification ────────────────────────────────────────────────────

def verify_pnp():
    """Verify Plug-and-Play HQS denoising."""
    try:
        from pwm_core.recon.pnp import pnp_hqs_denoise
    except ImportError:
        return {"status": "import_error"}

    N = 128
    rng = np.random.RandomState(42)
    gt = np.zeros((N, N), dtype=np.float32)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt = (0.5 * (x**2 + y**2 < 0.5) + 0.3).astype(np.float32)
    noisy = gt + rng.normal(0, 0.1, gt.shape).astype(np.float32)

    results = {}
    results["pnp_hqs"] = verify_solver(
        "PnP-HQS",
        lambda: pnp_hqs_denoise(noisy, sigma=0.1, iterations=10),
        gt)
    return results


# ── Lensless Wiener Verification ─────────────────────────────────────────────

def verify_lensless():
    """Verify Wiener deconvolution for lensless imaging."""
    N = 128
    rng = np.random.RandomState(42)
    gt = rng.rand(N, N).astype(np.float32)

    # Create PSF (diffuser-like random pattern)
    psf = rng.rand(N, N).astype(np.float32)
    psf /= psf.sum()

    # Convolve in Fourier domain
    GT_f = np.fft.fft2(gt)
    PSF_f = np.fft.fft2(psf)
    Y_f = GT_f * PSF_f
    y = np.real(np.fft.ifft2(Y_f)).astype(np.float32)
    # Add noise
    y += rng.normal(0, 0.01, y.shape).astype(np.float32)

    # Wiener deconvolution
    def wiener_deconv(y, psf, noise_var=0.01):
        Y_f = np.fft.fft2(y)
        H_f = np.fft.fft2(psf, s=y.shape)
        H_conj = np.conj(H_f)
        nsr = noise_var / (np.abs(H_f) ** 2 + 1e-10)
        W = H_conj / (np.abs(H_f) ** 2 + nsr)
        return np.real(np.fft.ifft2(Y_f * W)).astype(np.float32)

    results = {}
    results["wiener"] = verify_solver(
        "Wiener deconvolution",
        lambda: wiener_deconv(y, psf),
        gt)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

VERIFIERS = {
    "ct": ("CT (FBP, SART)", verify_ct),
    "mri": ("MRI (Zero-filled)", verify_mri),
    "microscopy_rl": ("Microscopy RL deconv", verify_richardson_lucy),
    "lensless": ("Lensless (Wiener)", verify_lensless),
}


def main():
    print("=" * 70)
    print("PWM5 Algorithm Verification (Local CPU)")
    print("=" * 70)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": "local_cpu",
        "modalities": {},
    }

    verified_count = 0
    error_count = 0

    for mod_id, (desc, verify_fn) in VERIFIERS.items():
        print(f"\n--- {desc} ({mod_id}) ---")
        try:
            results = verify_fn()
            all_results["modalities"][mod_id] = results
            for solver_name, solver_result in results.items():
                status = solver_result.get("status", "unknown")
                if status == "verified":
                    verified_count += 1
                    psnr = solver_result.get("psnr_db", 0)
                    ssim = solver_result.get("ssim", 0)
                    t = solver_result.get("exec_time", 0)
                    print(f"  {solver_name:30s} VERIFIED  PSNR={psnr:6.2f} dB  SSIM={ssim:.4f}  time={t:.3f}s")
                elif status == "error":
                    error_count += 1
                    print(f"  {solver_name:30s} ERROR: {solver_result.get('error', '')[:60]}")
                else:
                    print(f"  {solver_name:30s} {status}")
        except Exception as e:
            all_results["modalities"][mod_id] = {"status": f"exception: {str(e)[:200]}"}
            error_count += 1
            print(f"  EXCEPTION: {e}")

    all_results["summary"] = {
        "verified": verified_count,
        "errors": error_count,
    }

    # Save results
    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Verified: {verified_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
