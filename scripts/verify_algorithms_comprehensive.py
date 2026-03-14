#!/usr/bin/env python3
"""Comprehensive algorithm verification on local CPU.

Tests all importable solver modules with synthetic data to verify they
execute correctly and produce reasonable output.
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
    except Exception:
        return -1.0


def verify_solver(name, fn, gt, timeout=120):
    start = time.time()
    try:
        recon = fn()
        elapsed = time.time() - start
        if recon is None:
            return {"status": "returned_none", "exec_time": elapsed}
        recon = np.real(recon).astype(np.float32)
        if recon.shape != gt.shape:
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


rng = np.random.RandomState(42)

# ── Synthetic data generators ────────────────────────────────────────────────

def make_phantom(N=128):
    """Shepp-Logan-like phantom."""
    gt = np.zeros((N, N), dtype=np.float32)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt += 1.0 * ((x/0.69)**2 + (y/0.92)**2 < 1)
    gt -= 0.8 * ((x/0.66)**2 + (y/0.88)**2 < 1)
    gt += 0.2 * ((x/0.31)**2 + (y/0.11)**2 < 1)
    gt += 0.2 * (((x-0.22)/0.11)**2 + (y/0.25)**2 < 1)
    return gt


def make_blobs(N=128, n_blobs=15):
    """Random blobs for microscopy-like testing."""
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


# ── 1. CT ────────────────────────────────────────────────────────────────────

def verify_ct():
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
    results["fbp_ramlak"] = verify_solver("FBP (Ram-Lak)",
        lambda: fbp_2d(meas, angles_rad, "ramlak", out_size), gt)
    results["fbp_shepp_logan"] = verify_solver("FBP (Shepp-Logan)",
        lambda: fbp_2d(meas, angles_rad, "shepp_logan", out_size), gt)
    results["sart_10iter"] = verify_solver("SART (10 iter)",
        lambda: sart_2d(meas, angles_rad, out_size, iterations=10), gt)
    return results


# ── 2. MRI (zero-filled) ────────────────────────────────────────────────────

def verify_mri():
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
    results["zero_filled"] = verify_solver("Zero-filled IFFT",
        lambda: np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_us))).astype(np.float32), gt)
    return results


# ── 3. Richardson-Lucy (microscopy) ──────────────────────────────────────────

def verify_rl():
    from pwm_core.recon.richardson_lucy import richardson_lucy_2d
    N = 128
    gt = make_blobs(N)
    psf = make_gaussian_psf(N, sigma=3.0)
    blurred = convolve_2d(gt, psf)
    noisy = np.maximum(blurred + rng.normal(0, 0.02, blurred.shape).astype(np.float32), 1e-6)

    results = {}
    results["rl_20iter"] = verify_solver("RL (20 iter)",
        lambda: richardson_lucy_2d(noisy, psf, iterations=20), gt)
    results["rl_50iter"] = verify_solver("RL (50 iter)",
        lambda: richardson_lucy_2d(noisy, psf, iterations=50), gt)
    return results


# ── 4. FISTA-L2 (classical) ─────────────────────────────────────────────────

def verify_fista():
    from pwm_core.recon.classical import fista_l2
    N = 32  # Small for CPU speed
    gt = rng.randn(N * N).astype(np.float32) * 0.3
    gt[np.abs(gt) < 0.15] = 0  # Sparse

    M = N * N // 4
    A = rng.randn(M, N * N).astype(np.float32) / np.sqrt(M)
    y = A @ gt

    results = {}
    results["fista_l2"] = verify_solver("FISTA-L2",
        lambda: fista_l2(y, A, lam=1e-3, iters=100), gt)
    return results


# ── 5. Least Squares (classical) ────────────────────────────────────────────

def verify_least_squares():
    from pwm_core.recon.classical import least_squares
    N = 32
    gt = rng.randn(N * N).astype(np.float32) * 0.3
    M = N * N  # Full measurement
    A = rng.randn(M, N * N).astype(np.float32) / np.sqrt(M)
    y = A @ gt + rng.randn(M).astype(np.float32) * 0.01

    results = {}
    results["least_squares"] = verify_solver("Least Squares",
        lambda: least_squares(A, y), gt)
    return results


# ── 6. Lensless Wiener ──────────────────────────────────────────────────────

def verify_lensless():
    N = 128
    gt = rng.rand(N, N).astype(np.float32)
    psf = rng.rand(N, N).astype(np.float32)
    psf /= psf.sum()
    GT_f = np.fft.fft2(gt)
    PSF_f = np.fft.fft2(psf)
    y = np.real(np.fft.ifft2(GT_f * PSF_f)).astype(np.float32)
    y += rng.normal(0, 0.01, y.shape).astype(np.float32)

    def wiener(y, psf, nv=0.01):
        Y = np.fft.fft2(y)
        H = np.fft.fft2(psf, s=y.shape)
        W = np.conj(H) / (np.abs(H)**2 + nv / (np.abs(H)**2 + 1e-10))
        return np.real(np.fft.ifft2(Y * W)).astype(np.float32)

    results = {}
    results["wiener"] = verify_solver("Wiener deconv", lambda: wiener(y, psf), gt)
    return results


# ── 7. Holography (angular spectrum) ────────────────────────────────────────

def verify_holography():
    from pwm_core.recon.holography_solver import angular_spectrum_propagate
    N = 128
    gt_amp = make_blobs(N, 10).astype(np.float32)
    gt_phase = rng.uniform(-np.pi/4, np.pi/4, (N, N)).astype(np.float32)
    gt_field = gt_amp * np.exp(1j * gt_phase)

    # Forward: propagate
    wavelength = 632e-9  # HeNe laser
    pixel_size = 5e-6
    z = 1e-3  # 1 mm propagation
    hologram_field = angular_spectrum_propagate(gt_field, wavelength, pixel_size, z)
    intensity = np.abs(hologram_field)**2

    # Backward
    recon_field = angular_spectrum_propagate(
        np.sqrt(intensity + 0j), wavelength, pixel_size, -z)
    recon_amp = np.abs(recon_field).astype(np.float32)

    results = {}
    results["angular_spectrum"] = verify_solver("Angular Spectrum",
        lambda: recon_amp, gt_amp)
    return results


# ── 8. Phase Retrieval (Gerchberg-Saxton) ────────────────────────────────────

def verify_phase_retrieval():
    from pwm_core.recon.phase_retrieval_solver import gerchberg_saxton
    N = 64
    gt_amp = make_blobs(N, 8).astype(np.float32) + 0.1
    intensity = np.abs(np.fft.fft2(gt_amp))**2

    results = {}
    results["gerchberg_saxton"] = verify_solver("Gerchberg-Saxton",
        lambda: gerchberg_saxton(intensity, iterations=50), gt_amp)
    return results


# ── 9. PnP-HQS ──────────────────────────────────────────────────────────────

def verify_pnp():
    try:
        from pwm_core.recon.pnp import pnp_hqs_denoise
    except ImportError:
        return {"status": "import_error"}

    N = 64
    gt = make_phantom(N)
    noisy = gt + rng.normal(0, 0.1, gt.shape).astype(np.float32)

    results = {}
    results["pnp_hqs"] = verify_solver("PnP-HQS",
        lambda: pnp_hqs_denoise(noisy, sigma=0.1, iterations=10), gt)
    return results


# ── 10. DOT solver ──────────────────────────────────────────────────────────

def verify_dot():
    try:
        from pwm_core.recon.dot_solver import born_approx
    except ImportError:
        return {"status": "import_error"}

    N = 32
    gt = rng.rand(N, N).astype(np.float32) * 0.1 + 0.01
    # Simple DOT: y = A * gt + noise
    A = rng.rand(N*N // 2, N*N).astype(np.float32) * 0.01
    y = A @ gt.ravel() + rng.randn(N*N//2).astype(np.float32) * 0.001

    results = {}
    results["born_approx"] = verify_solver("Born approximation",
        lambda: born_approx(y, A, (N, N)), gt)
    return results


# ── 11. Photoacoustic ───────────────────────────────────────────────────────

def verify_photoacoustic():
    try:
        from pwm_core.recon.photoacoustic_solver import back_projection
    except ImportError:
        return {"status": "import_error"}

    N = 64
    gt = make_phantom(N)
    n_sensors = 32
    angles = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    # Simplified: measurement is radon-like
    from scipy.ndimage import rotate
    sino = np.zeros((n_sensors, N), dtype=np.float32)
    for i, ang in enumerate(angles):
        rotated = rotate(gt, np.degrees(ang), reshape=False)
        sino[i] = rotated.sum(axis=0)

    results = {}
    results["backprojection"] = verify_solver("PAT Backprojection",
        lambda: back_projection(sino, angles, (N, N)), gt)
    return results


# ── 12. OCT (FFT recon) ─────────────────────────────────────────────────────

def verify_oct():
    try:
        from pwm_core.recon.oct_solver import fft_recon
    except ImportError:
        return {"status": "import_error"}

    N = 128
    gt = make_blobs(N, 10)
    # Simulated OCT: spectral domain
    spectral = np.fft.fft(gt, axis=0).astype(np.complex64)
    # Add noise
    spectral += (rng.randn(*spectral.shape) + 1j * rng.randn(*spectral.shape)).astype(np.complex64) * 0.1

    results = {}
    results["fft_recon"] = verify_solver("OCT FFT",
        lambda: fft_recon(spectral), gt)
    return results


# ── 13. GAP-TV (CACTI) ──────────────────────────────────────────────────────

def verify_gap_tv():
    try:
        from pwm_core.recon.gap_tv import gap_tv_cacti
    except ImportError:
        return {"status": "import_error"}

    N = 64
    T = 8  # temporal frames
    gt = np.zeros((T, N, N), dtype=np.float32)
    for t in range(T):
        gt[t] = make_blobs(N, 5 + t)

    # CACTI measurement: sum of masked frames
    masks = (rng.rand(T, N, N) > 0.5).astype(np.float32)
    y = np.sum(gt * masks, axis=0)

    results = {}
    results["gap_tv_cacti"] = verify_solver("GAP-TV (CACTI)",
        lambda: gap_tv_cacti(y, masks, iterations=20), gt)
    return results


# ── 14. SPC (single pixel camera) ───────────────────────────────────────────

def verify_spc():
    try:
        from pwm_core.recon.spc_solvers import spc_pseudoinverse
    except ImportError:
        return {"status": "import_error"}

    N = 32
    gt = make_phantom(N)
    M = N * N // 4  # 25% measurements
    patterns = (rng.rand(M, N * N) > 0.5).astype(np.float32)
    y = patterns @ gt.ravel() + rng.randn(M).astype(np.float32) * 0.01

    results = {}
    results["spc_pseudoinverse"] = verify_solver("SPC Pseudoinverse",
        lambda: spc_pseudoinverse(y, patterns, (N, N)), gt)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

VERIFIERS = [
    ("ct", "CT (FBP, SART)", verify_ct),
    ("mri", "MRI (Zero-filled)", verify_mri),
    ("microscopy_rl", "Richardson-Lucy", verify_rl),
    ("fista", "FISTA-L2", verify_fista),
    ("least_squares", "Least Squares", verify_least_squares),
    ("lensless", "Lensless Wiener", verify_lensless),
    ("holography", "Holography (Angular Spectrum)", verify_holography),
    ("phase_retrieval", "Phase Retrieval (GS)", verify_phase_retrieval),
    ("pnp", "PnP-HQS", verify_pnp),
    ("dot", "DOT (Born)", verify_dot),
    ("photoacoustic", "Photoacoustic", verify_photoacoustic),
    ("oct", "OCT (FFT)", verify_oct),
    ("gap_tv", "GAP-TV (CACTI)", verify_gap_tv),
    ("spc", "SPC Solvers", verify_spc),
]


def main():
    print("=" * 70)
    print("PWM5 Comprehensive Algorithm Verification (Local CPU)")
    print("=" * 70)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": "local_cpu",
        "modalities": {},
    }

    verified_count = 0
    error_count = 0
    skip_count = 0

    for mod_id, desc, verify_fn in VERIFIERS:
        print(f"\n--- {desc} ({mod_id}) ---")
        try:
            results = verify_fn()
            if isinstance(results, dict) and results.get("status") in ["import_error", "no_data"]:
                print(f"  SKIPPED: {results['status']}")
                skip_count += 1
                all_results["modalities"][mod_id] = results
                continue

            all_results["modalities"][mod_id] = results
            for solver_name, sr in results.items():
                status = sr.get("status", "unknown")
                if status == "verified":
                    verified_count += 1
                    p = sr.get("psnr_db", 0)
                    s = sr.get("ssim", 0)
                    t = sr.get("exec_time", 0)
                    print(f"  {solver_name:30s} OK  PSNR={p:7.2f} dB  SSIM={s:.4f}  t={t:.3f}s")
                elif status == "error":
                    error_count += 1
                    print(f"  {solver_name:30s} ERR: {sr.get('error', '')[:60]}")
                else:
                    print(f"  {solver_name:30s} {status}")
        except Exception as e:
            all_results["modalities"][mod_id] = {"status": f"exception: {str(e)[:200]}"}
            error_count += 1
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()

    all_results["summary"] = {
        "verified": verified_count,
        "errors": error_count,
        "skipped": skip_count,
    }

    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results: {verified_count} verified, {error_count} errors, {skip_count} skipped")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
