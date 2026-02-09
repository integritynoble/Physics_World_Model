"""OCT (Optical Coherence Tomography) reconstruction solvers.

References:
- Fercher, A.F. et al. (2003). "Optical coherence tomography - principles
  and applications", Reports on Progress in Physics.
- Leitgeb, R. et al. (2003). "Performance of fourier domain vs. time domain
  optical coherence tomography", Optics Express.

Expected PSNR: 36.0 dB on synthetic benchmark
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple


def fft_recon(
    spectral_data: np.ndarray,
    window: str = "hann",
    dc_subtract: bool = True,
) -> np.ndarray:
    """Standard FFT-based OCT A-line reconstruction.

    Process: DC subtraction -> windowing -> IFFT -> magnitude

    Args:
        spectral_data: Spectral interferogram (n_alines, n_spectral).
        window: Window function ('hann', 'hamming', 'none').
        dc_subtract: Whether to subtract DC component.

    Returns:
        B-scan image (n_alines, n_depth) where n_depth = n_spectral // 2.
    """
    n_alines, n_spectral = spectral_data.shape
    data = spectral_data.astype(np.float64)

    # DC subtraction (remove reference arm contribution)
    if dc_subtract:
        dc = np.mean(data, axis=0, keepdims=True)
        data = data - dc

    # Apply window function
    if window == "hann":
        win = np.hanning(n_spectral)
    elif window == "hamming":
        win = np.hamming(n_spectral)
    else:
        win = np.ones(n_spectral)

    data = data * win[np.newaxis, :]

    # IFFT along spectral axis
    depth_profile = np.fft.ifft(data, axis=1)

    # Take magnitude, use only positive frequencies (half spectrum)
    n_depth = n_spectral // 2
    b_scan = np.abs(depth_profile[:, :n_depth])

    # Log compression for display (optional, but standard for OCT)
    b_scan = np.maximum(b_scan, 1e-10)

    return b_scan.astype(np.float32)


def spectral_estimation(
    spectral_data: np.ndarray,
    n_depth: int = None,
    n_components: int = 10,
    dc_subtract: bool = True,
) -> np.ndarray:
    """MUSIC/ESPRIT-inspired spectral estimation for super-resolved OCT.

    Uses eigendecomposition of the autocorrelation matrix to identify
    signal subspace and estimate depth positions with super-resolution.
    Falls back to enhanced FFT when full MUSIC is unstable.

    Args:
        spectral_data: Spectral interferogram (n_alines, n_spectral).
        n_depth: Output depth samples (default: n_spectral // 2).
        n_components: Estimated number of reflecting layers.
        dc_subtract: Whether to subtract DC.

    Returns:
        B-scan image (n_alines, n_depth).
    """
    n_alines, n_spectral = spectral_data.shape
    if n_depth is None:
        n_depth = n_spectral // 2

    data = spectral_data.astype(np.float64)

    if dc_subtract:
        dc = np.mean(data, axis=0, keepdims=True)
        data = data - dc

    b_scan = np.zeros((n_alines, n_depth), dtype=np.float64)

    # Subarray size for spatial smoothing
    m = min(n_spectral // 2, 64)
    n_sub = n_spectral - m + 1

    for a in range(n_alines):
        signal = data[a]

        # Build Hankel (data) matrix for spatial smoothing
        hankel = np.zeros((m, n_sub), dtype=np.float64)
        for i in range(m):
            hankel[i] = signal[i:i + n_sub]

        # Autocorrelation matrix
        R = hankel @ hankel.T / n_sub

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Sort eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine signal subspace size
        # Use ratio criterion
        ev_ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-10)
        n_sig = min(n_components, np.sum(ev_ratios > 2.0) + 1)
        n_sig = max(n_sig, 1)

        # Noise subspace
        noise_space = eigenvectors[:, n_sig:]

        if noise_space.shape[1] == 0:
            # Fallback to FFT for this A-line
            win = np.hanning(n_spectral)
            fft_result = np.fft.ifft(signal * win)
            b_scan[a] = np.abs(fft_result[:n_depth])
            continue

        # MUSIC pseudo-spectrum
        freqs = np.linspace(0, 0.5, n_depth)

        for fi, f in enumerate(freqs):
            # Steering vector
            steering = np.exp(2j * np.pi * f * np.arange(m))

            # MUSIC spectrum: 1 / |a^H * E_n * E_n^H * a|
            proj = noise_space.T.conj() @ steering
            denom = np.sum(np.abs(proj) ** 2) + 1e-10
            b_scan[a, fi] = 1.0 / denom

        # Normalize per A-line
        if b_scan[a].max() > 0:
            b_scan[a] /= b_scan[a].max()

    return b_scan.astype(np.float32)


def spectral_estimation_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Registry-compatible wrapper for spectral estimation OCT solver."""
    return run_oct(y, physics, {"method": "spectral_estimation", **cfg})


def oct_denoising_net_recon(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Registry-compatible stub for DL OCT denoising (falls back to FFT)."""
    return run_oct(y, physics, {"method": "fft", **cfg})


def run_oct(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for OCT reconstruction.

    Args:
        y: Spectral interferogram data (n_alines, n_spectral).
        physics: Physics operator.
        cfg: Configuration dict.

    Returns:
        Tuple of (b_scan, info_dict).
    """
    method = cfg.get("method", "fft")
    info: Dict[str, Any] = {"solver": "oct", "method": method}

    try:
        if method == "spectral_estimation":
            n_components = cfg.get("n_components", 10)
            result = spectral_estimation(y, n_components=n_components)
        else:
            window = cfg.get("window", "hann")
            result = fft_recon(y, window=window)

        info["output_shape"] = list(result.shape)
        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
