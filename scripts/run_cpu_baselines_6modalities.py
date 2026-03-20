#!/usr/bin/env python3
"""CPU baseline algorithms for 6 benchmark modalities.

Each modality's challenge HDF5 uses a standardized format:
  - PSF runners (phase_retrieval, fpm, odt, raman_imaging, ftir_imaging):
      H_ideal: (13, 13) PSF kernel
      x_true:  (256, 256) ground truth
      y:       (256, 256) blurred + noisy measurement
  - Mask runner (ghost_imaging):
      H_ideal: (128, 128) measurement mask
      x_true:  (128, 128) ground truth
      y:       (128, 128) masked measurement

Algorithms applied:
  - phase_retrieval: Gerchberg-Saxton alternating projection (Fourier modulus constraint)
  - fpm: Wiener deconvolution (PSF-based)
  - odt: Wiener deconvolution with TV-like regularization
  - ghost_imaging: Wiener deconvolution (mask-based in Fourier domain)
  - raman_imaging: Wiener deconvolution (PSF-based)
  - ftir_imaging: Wiener deconvolution (PSF-based)

Since the challenge data uses standardized PSF/mask forward models (y = H*x + noise),
the appropriate classical CPU baseline is Wiener deconvolution for PSF types
and pseudo-inverse/Wiener for mask types. For phase_retrieval we also apply
a GS-inspired iterative refinement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve

# ============================================================================
# Paths
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "benchmark" / "challenge-data" / "v1.0"

MODALITIES = [
    "phase_retrieval",
    "fpm",
    "odt",
    "ghost_imaging",
    "raman_imaging",
    "ftir_imaging",
]


# ============================================================================
# Metrics
# ============================================================================

def compute_psnr(ref: np.ndarray, est: np.ndarray) -> float:
    """PSNR between two images (uses data range from ref)."""
    ref64 = ref.astype(np.float64)
    est64 = est.astype(np.float64)
    mse = np.mean((ref64 - est64) ** 2)
    if mse < 1e-15:
        return 100.0
    data_range = ref64.max() - ref64.min()
    if data_range < 1e-12:
        return 0.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(ref: np.ndarray, est: np.ndarray) -> float:
    """Simplified global SSIM."""
    ref64 = ref.astype(np.float64)
    est64 = est.astype(np.float64)
    data_range = ref64.max() - ref64.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = ref64.mean()
    mu_y = est64.mean()
    var_x = ref64.var()
    var_y = est64.var()
    cov_xy = np.mean((ref64 - mu_x) * (est64 - mu_y))
    return float(
        (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    )


def align_recon(gt: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Remove global scale/offset ambiguity: aligned = a*recon + b (min MSE)."""
    r = recon.ravel().astype(np.float64)
    g = gt.ravel().astype(np.float64)
    n = len(r)
    sr, sg = r.sum(), g.sum()
    srr = (r * r).sum()
    srg = (r * g).sum()
    denom = n * srr - sr * sr
    if abs(denom) < 1e-12:
        return recon
    a = (n * srg - sr * sg) / denom
    b = (sg - a * sr) / n
    aligned = a * recon + b
    return np.clip(aligned, gt.min(), gt.max()).astype(np.float64)


# ============================================================================
# Wiener Deconvolution (for PSF-based forward models)
# ============================================================================

def wiener_deconv_psf(y: np.ndarray, psf: np.ndarray, noise_var: float = 0.001) -> np.ndarray:
    """Wiener deconvolution for y = conv(x, psf) + noise.

    Parameters
    ----------
    y : (H, W) measurement
    psf : (kh, kw) point spread function
    noise_var : noise variance estimate (regularization)

    Returns
    -------
    x_recon : (H, W) reconstructed image
    """
    H, W = y.shape
    # Zero-pad PSF to image size
    psf_padded = np.zeros((H, W), dtype=np.float64)
    kh, kw = psf.shape
    # Place PSF centered in the padded array
    ph = kh // 2
    pw = kw // 2
    psf_padded[:kh, :kw] = psf
    # Shift so center of PSF is at (0,0) for correct convolution
    psf_padded = np.roll(np.roll(psf_padded, -ph, axis=0), -pw, axis=1)

    # Wiener filter in Fourier domain
    Y = np.fft.fft2(y.astype(np.float64))
    H_fft = np.fft.fft2(psf_padded)
    H_conj = np.conj(H_fft)
    H_abs2 = np.abs(H_fft) ** 2

    # Estimate noise power from high-frequency content
    signal_power = np.mean(np.abs(Y) ** 2)
    # Use noise_var as fraction of signal power for regularization
    nsr = noise_var * signal_power

    # Wiener filter: X = H* Y / (|H|^2 + NSR)
    X_recon = H_conj * Y / (H_abs2 + nsr + 1e-12)
    x_recon = np.fft.ifft2(X_recon).real

    return x_recon


def wiener_deconv_psf_tv(y: np.ndarray, psf: np.ndarray,
                          noise_var: float = 0.001, tv_weight: float = 0.01,
                          n_iter: int = 30) -> np.ndarray:
    """Iterative Wiener deconvolution with TV-like edge preservation.

    Uses half-quadratic splitting: alternates between Wiener step and
    TV-proximal denoising step.

    Parameters
    ----------
    y : (H, W) measurement
    psf : (kh, kw) PSF
    noise_var : regularization parameter
    tv_weight : TV regularization strength
    n_iter : number of iterations

    Returns
    -------
    x_recon : (H, W) reconstructed image
    """
    H, W = y.shape

    # Initial Wiener estimate
    x_est = wiener_deconv_psf(y, psf, noise_var)

    # Precompute FFT of PSF
    psf_padded = np.zeros((H, W), dtype=np.float64)
    kh, kw = psf.shape
    ph, pw = kh // 2, kw // 2
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(np.roll(psf_padded, -ph, axis=0), -pw, axis=1)
    H_fft = np.fft.fft2(psf_padded)
    H_conj = np.conj(H_fft)
    H_abs2 = np.abs(H_fft) ** 2
    Y = np.fft.fft2(y.astype(np.float64))

    signal_power = np.mean(np.abs(Y) ** 2)
    nsr = noise_var * signal_power

    for it in range(n_iter):
        # Data fidelity step (Wiener)
        X_est = np.fft.fft2(x_est)
        residual_fft = Y - H_fft * X_est
        gradient_fft = H_conj * residual_fft
        step = 0.5 / (np.max(H_abs2) + 1e-8)
        x_est_new = x_est + step * np.fft.ifft2(gradient_fft).real

        # Simple TV proximal step (gradient-based smoothing)
        # Compute gradient magnitudes
        dy = np.diff(x_est_new, axis=0, append=x_est_new[-1:, :])
        dx = np.diff(x_est_new, axis=1, append=x_est_new[:, -1:])
        grad_mag = np.sqrt(dy ** 2 + dx ** 2 + 1e-8)

        # Soft threshold on gradients (anisotropic TV proximal)
        dy_shrink = dy * np.maximum(1.0 - tv_weight / grad_mag, 0.0)
        dx_shrink = dx * np.maximum(1.0 - tv_weight / grad_mag, 0.0)

        # Reconstruct from shrunk gradients (adjoint of diff)
        div_y = np.diff(dy_shrink, axis=0, prepend=dy_shrink[:1, :])
        div_x = np.diff(dx_shrink, axis=1, prepend=dx_shrink[:, :1])
        x_est = x_est_new - tv_weight * (div_y + div_x)

    return x_est


# ============================================================================
# Gerchberg-Saxton for Phase Retrieval
# ============================================================================

def gerchberg_saxton_psf(y: np.ndarray, psf: np.ndarray,
                          n_iterations: int = 100) -> np.ndarray:
    """GS-inspired alternating projection for PSF-based phase retrieval.

    Since the challenge data stores y as a real-valued blurred image
    (not intensity patterns), we use an iterative approach:
    1. Forward: x_est -> conv(x_est, psf) -> predicted y
    2. Constraint: replace predicted amplitude with measured y
    3. Backward: deconvolve to get updated x_est

    This is essentially Richardson-Lucy / MLEM style iteration.

    Parameters
    ----------
    y : (H, W) measurement
    psf : (kh, kw) PSF
    n_iterations : number of iterations

    Returns
    -------
    x_recon : (H, W) reconstructed image
    """
    H, W = y.shape

    # Start with Wiener as initial estimate
    x_est = wiener_deconv_psf(y, psf, noise_var=0.005)
    x_est = np.clip(x_est, 0, None)

    # Pad PSF for fftconvolve
    psf_norm = psf / (psf.sum() + 1e-12)
    psf_flipped = psf_norm[::-1, ::-1]

    for it in range(n_iterations):
        # Forward model
        y_pred = fftconvolve(x_est, psf_norm, mode='same')
        y_pred = np.clip(y_pred, 1e-12, None)

        # Richardson-Lucy update ratio
        ratio = y / (y_pred + 1e-12)
        correction = fftconvolve(ratio, psf_flipped, mode='same')

        # Update
        x_est = x_est * correction
        x_est = np.clip(x_est, 0, None)

    return x_est


# ============================================================================
# Mask-based Reconstruction (for ghost_imaging)
# ============================================================================

def reconstruct_masked(y: np.ndarray, mask: np.ndarray,
                       n_iter: int = 50) -> np.ndarray:
    """Reconstruct from masked measurement y = mask * x + noise.

    For ghost imaging: mask is binary (0 or 1). Where mask=1,
    we observe x directly. Where mask=0, we must interpolate.

    Uses iterative Fourier-domain inpainting:
    1. Start with x = y (observed pixels) + 0 (missing pixels)
    2. Low-pass filter to smooth out discontinuities
    3. Replace observed pixels with measured values
    4. Repeat

    Parameters
    ----------
    y : (H, W) measurement (y = mask * x + noise)
    mask : (H, W) binary measurement mask
    n_iter : number of inpainting iterations

    Returns
    -------
    x_recon : (H, W) reconstructed image
    """
    from scipy.ndimage import gaussian_filter

    mask64 = mask.astype(np.float64)
    y64 = y.astype(np.float64)

    # Where mask=1, y gives us x directly (up to noise)
    # Where mask=0, we have no information -> fill by inpainting

    # Initialize: observed pixels = y, missing = local average
    x_est = y64.copy()

    # Iterative Fourier-domain inpainting with decreasing bandwidth
    for it in range(n_iter):
        # Smooth the current estimate (low-pass)
        sigma = max(0.5, 3.0 * (1.0 - it / n_iter))
        x_smooth = gaussian_filter(x_est, sigma=sigma)

        # Data fidelity: keep observed pixels, fill missing with smooth estimate
        x_est = mask64 * y64 + (1.0 - mask64) * x_smooth

    return x_est


# ============================================================================
# Algorithm Dispatcher
# ============================================================================

def run_algorithm(modality: str, y: np.ndarray, H_ideal: np.ndarray,
                  x_true: np.ndarray) -> tuple[np.ndarray, str]:
    """Run the appropriate baseline algorithm for each modality.

    Parameters
    ----------
    modality : modality name
    y : measurement data
    H_ideal : ideal forward model (PSF or mask)
    x_true : ground truth (for reference only)

    Returns
    -------
    x_recon : reconstructed image
    algo_name : name of algorithm used
    """
    if modality == "phase_retrieval":
        # Gerchberg-Saxton / Richardson-Lucy iterative
        x_recon = gerchberg_saxton_psf(y, H_ideal, n_iterations=100)
        return x_recon, "Gerchberg-Saxton / Richardson-Lucy (100 iter)"

    elif modality == "fpm":
        # FPM has heavy mismatch (Pearson corr ~0.17 between y and ideal)
        # plus Poisson+Gaussian noise. Wiener is the best classical approach.
        x_recon = wiener_deconv_psf(y, H_ideal, noise_var=0.005)
        return x_recon, "Wiener Deconvolution"

    elif modality == "odt":
        # Wiener + TV regularization
        x_recon = wiener_deconv_psf_tv(y, H_ideal, noise_var=0.005,
                                        tv_weight=0.005, n_iter=30)
        return x_recon, "Wiener + TV Regularization (30 iter)"

    elif modality == "ghost_imaging":
        # Mask-based inpainting reconstruction
        x_recon = reconstruct_masked(y, H_ideal, n_iter=50)
        return x_recon, "Iterative Fourier Inpainting (50 iter)"

    elif modality == "raman_imaging":
        # Wiener + TV for spectral data
        x_recon = wiener_deconv_psf_tv(y, H_ideal, noise_var=0.003,
                                        tv_weight=0.003, n_iter=30)
        return x_recon, "Wiener + TV Regularization (30 iter)"

    elif modality == "ftir_imaging":
        # Wiener + TV for spectral data
        x_recon = wiener_deconv_psf_tv(y, H_ideal, noise_var=0.003,
                                        tv_weight=0.003, n_iter=30)
        return x_recon, "Wiener + TV Regularization (30 iter)"

    else:
        raise ValueError(f"Unknown modality: {modality}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("CPU Baseline Algorithms — 6 Benchmark Modalities")
    print("=" * 72)
    print(f"Data directory: {DATA_DIR}")
    print()

    results = {}
    all_psnr = []
    all_ssim = []

    for modality in MODALITIES:
        h5_path = DATA_DIR / f"{modality}_challenge_public.h5"
        if not h5_path.exists():
            print(f"  [SKIP] {modality}: HDF5 not found at {h5_path}")
            continue

        print(f"--- {modality.upper().replace('_', ' ')} ---")

        with h5py.File(h5_path, "r") as f:
            grp = f["sample_00"]
            x_true = grp["x_true"][:]
            y = grp["y"][:]
            H_ideal = grp["H_ideal"][:]

        print(f"  x_true: shape={x_true.shape}, range=[{x_true.min():.4f}, {x_true.max():.4f}]")
        print(f"  y:      shape={y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
        print(f"  H_ideal: shape={H_ideal.shape}, range=[{H_ideal.min():.4f}, {H_ideal.max():.4f}]")

        t0 = time.time()
        x_recon, algo_name = run_algorithm(modality, y, H_ideal, x_true)
        elapsed = time.time() - t0

        # Align reconstruction to ground truth (remove scale/offset ambiguity)
        x_recon_aligned = align_recon(x_true, x_recon)

        psnr = compute_psnr(x_true, x_recon_aligned)
        ssim = compute_ssim(x_true, x_recon_aligned)

        all_psnr.append(psnr)
        all_ssim.append(ssim)

        results[modality] = {
            "algorithm": algo_name,
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
            "time_s": round(elapsed, 2),
        }

        print(f"  Algorithm: {algo_name}")
        print(f"  PSNR:  {psnr:.2f} dB")
        print(f"  SSIM:  {ssim:.4f}")
        print(f"  Time:  {elapsed:.2f}s")
        print()

    # Summary table
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Modality':<22s} {'Algorithm':<45s} {'PSNR':>8s} {'SSIM':>8s} {'Time':>7s}")
    print("-" * 92)
    for modality, res in results.items():
        print(f"{modality:<22s} {res['algorithm']:<45s} {res['psnr_db']:>7.2f}  {res['ssim']:>7.4f}  {res['time_s']:>6.2f}s")
    print("-" * 92)
    if all_psnr:
        print(f"{'AVERAGE':<22s} {'':45s} {np.mean(all_psnr):>7.2f}  {np.mean(all_ssim):>7.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
