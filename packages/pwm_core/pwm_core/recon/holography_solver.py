"""Digital Holography Reconstruction: Angular Spectrum Method.

Phase retrieval from off-axis holograms.

References:
- Goodman, J.W. (2005). "Introduction to Fourier Optics"
- Kim, M.K. (2011). "Principles and techniques of digital holographic microscopy"

Benchmark: Simulated holograms from DIV2K
Expected PSNR: 30-40 dB (amplitude), phase recovery ~0.1 rad RMS error
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def angular_spectrum_propagate(
    field: np.ndarray,
    wavelength: float,
    pixel_size: float,
    distance: float,
) -> np.ndarray:
    """Propagate field using Angular Spectrum Method.

    Args:
        field: Complex field (H, W)
        wavelength: Light wavelength (same units as pixel_size)
        pixel_size: Pixel size in object plane
        distance: Propagation distance (same units)

    Returns:
        Propagated complex field
    """
    h, w = field.shape
    k = 2 * np.pi / wavelength

    # Spatial frequencies
    fy = np.fft.fftfreq(h, d=pixel_size)
    fx = np.fft.fftfreq(w, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    # Transfer function
    # H = exp(i * k * z * sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2))
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2

    # Evanescent wave cutoff
    propagating = arg >= 0
    sqrt_arg = np.sqrt(np.abs(arg))

    transfer_function = np.zeros_like(arg, dtype=np.complex64)
    transfer_function[propagating] = np.exp(1j * k * distance * sqrt_arg[propagating])

    # Propagate
    F_field = fft2(field)
    F_propagated = F_field * transfer_function
    propagated = ifft2(F_propagated)

    return propagated.astype(np.complex64)


def extract_plus_one_order(
    hologram: np.ndarray,
    carrier_freq: Optional[Tuple[float, float]] = None,
    window_size: Optional[float] = None,
) -> np.ndarray:
    """Extract +1 diffraction order from off-axis hologram.

    For off-axis holography, the hologram contains:
    - DC term (|R|^2 + |O|^2)
    - +1 order (R*.O) shifted by carrier frequency
    - -1 order (R.O*) shifted by -carrier frequency

    Args:
        hologram: Off-axis hologram intensity
        carrier_freq: Carrier frequency (ky, kx) in normalized units
                     If None, auto-detected from peak
        window_size: Size of extraction window (default: 1/3 of spectrum)

    Returns:
        Complex field of +1 order (contains object information)
    """
    h, w = hologram.shape
    hologram = hologram.astype(np.float32)

    # FFT of hologram
    F_holo = fftshift(fft2(hologram))

    if carrier_freq is None:
        # Auto-detect carrier frequency from peak
        magnitude = np.abs(F_holo)

        # Mask DC region
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dc_mask = ((y - center_y)**2 + (x - center_x)**2) > (min(h, w) * 0.1)**2

        # Also only look in one half (upper or right) to get +1 order
        half_mask = y < center_y  # Upper half for +1

        masked_mag = magnitude * dc_mask * half_mask
        peak_idx = np.unravel_index(np.argmax(masked_mag), magnitude.shape)

        carrier_freq = (
            (peak_idx[0] - center_y) / h,
            (peak_idx[1] - center_x) / w
        )

    if window_size is None:
        window_size = min(h, w) / 3

    # Create extraction window centered on carrier frequency
    ky, kx = carrier_freq
    center_y = int(h / 2 + ky * h)
    center_x = int(w / 2 + kx * w)

    y, x = np.ogrid[:h, :w]
    window = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * (window_size / 2)**2))

    # Extract +1 order
    plus_one = F_holo * window

    # Shift to center
    plus_one_shifted = np.roll(np.roll(plus_one, -int(ky * h), axis=0), -int(kx * w), axis=1)

    # Inverse FFT
    field = ifft2(ifftshift(plus_one_shifted))

    return field.astype(np.complex64)


def phase_unwrap_2d(
    wrapped_phase: np.ndarray,
) -> np.ndarray:
    """Simple 2D phase unwrapping using quality-guided algorithm.

    Args:
        wrapped_phase: Wrapped phase (-pi, pi)

    Returns:
        Unwrapped phase
    """
    try:
        from skimage.restoration import unwrap_phase
        return unwrap_phase(wrapped_phase)
    except ImportError:
        pass

    # Simple flood-fill unwrapping as fallback
    h, w = wrapped_phase.shape
    unwrapped = wrapped_phase.copy()

    # Row-wise unwrapping
    for i in range(h):
        for j in range(1, w):
            diff = wrapped_phase[i, j] - wrapped_phase[i, j-1]
            if diff > np.pi:
                unwrapped[i, j:] -= 2 * np.pi
            elif diff < -np.pi:
                unwrapped[i, j:] += 2 * np.pi

    # Column-wise correction
    for j in range(w):
        for i in range(1, h):
            diff = unwrapped[i, j] - unwrapped[i-1, j]
            if diff > np.pi:
                unwrapped[i:, j] -= 2 * np.pi
            elif diff < -np.pi:
                unwrapped[i:, j] += 2 * np.pi

    return unwrapped.astype(np.float32)


def reconstruct_offaxis_hologram(
    hologram: np.ndarray,
    wavelength: float = 633e-9,
    pixel_size: float = 5e-6,
    prop_distance: float = 0.0,
    carrier_freq: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full off-axis hologram reconstruction.

    Args:
        hologram: Off-axis hologram intensity
        wavelength: Light wavelength in meters
        pixel_size: Detector pixel size in meters
        prop_distance: Propagation distance (0 = focused)
        carrier_freq: Carrier frequency (auto-detect if None)

    Returns:
        Tuple of (amplitude, phase)
    """
    # Extract +1 order
    field = extract_plus_one_order(hologram, carrier_freq)

    # Propagate if needed
    if abs(prop_distance) > 1e-10:
        field = angular_spectrum_propagate(
            field, wavelength, pixel_size, prop_distance
        )

    # Extract amplitude and phase
    amplitude = np.abs(field).astype(np.float32)
    phase_wrapped = np.angle(field).astype(np.float32)

    # Unwrap phase
    phase = phase_unwrap_2d(phase_wrapped)

    return amplitude, phase


def tv_holography_admm(
    hologram: np.ndarray,
    wavelength: float = 633e-9,
    pixel_size: float = 5e-6,
    prop_distance: float = 0.0,
    lam_tv: float = 0.01,
    rho: float = 1.0,
    iters: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """TV-regularized holographic phase recovery using ADMM.

    Solves: min_x ||I - |H(x)|^2||^2 + lam_tv * TV(angle(x))

    where H is the propagation operator. This is particularly useful
    for in-line holography or when the carrier frequency is low.

    Args:
        hologram: Hologram intensity (H, W)
        wavelength: Light wavelength
        pixel_size: Detector pixel size
        prop_distance: Propagation distance
        lam_tv: TV regularization weight for phase
        rho: ADMM penalty parameter
        iters: Number of ADMM iterations

    Returns:
        Tuple of (amplitude, phase)
    """
    h, w = hologram.shape
    hologram = hologram.astype(np.float32)
    sqrt_I = np.sqrt(np.maximum(hologram, 0))

    # Build propagation transfer function
    k = 2 * np.pi / wavelength
    fy = np.fft.fftfreq(h, d=pixel_size)
    fx = np.fft.fftfreq(w, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    propagating = arg >= 0
    sqrt_arg = np.sqrt(np.abs(arg))
    H_tf = np.zeros_like(arg, dtype=np.complex64)
    H_tf[propagating] = np.exp(1j * k * prop_distance * sqrt_arg[propagating])
    H_conj = np.conj(H_tf)

    def propagate(field):
        return ifft2(fft2(field) * H_tf)

    def back_propagate(field):
        return ifft2(fft2(field) * H_conj)

    # Initialize with back-propagation of sqrt(hologram)
    x = back_propagate(sqrt_I.astype(np.complex64))

    # ADMM variables for TV on phase
    z = np.zeros((h, w, 2), dtype=np.float32)  # TV auxiliary (gradient of phase)
    u = np.zeros((h, w, 2), dtype=np.float32)  # dual variable

    for it in range(iters):
        # x-update: data fidelity + quadratic penalty
        # Gradient of data fidelity: H^*(|Hx| - sqrt(I)) * exp(j*angle(Hx))
        Hx = propagate(x)
        mag_Hx = np.abs(Hx) + 1e-10
        phase_Hx = np.angle(Hx)
        Hx_corrected = sqrt_I * np.exp(1j * phase_Hx)
        grad_data = back_propagate(Hx - Hx_corrected)

        # Phase TV gradient contribution
        phase_x = np.angle(x)
        dx_phase = np.diff(phase_x, axis=1, append=phase_x[:, -1:])
        dy_phase = np.diff(phase_x, axis=0, append=phase_x[-1:, :])
        grad_phase = np.stack([dx_phase, dy_phase], axis=-1)

        # Proximal gradient on phase via z,u
        v = grad_phase - z + u
        div_v = np.zeros_like(phase_x)
        div_v[:, :-1] += v[:, :-1, 0]
        div_v[:, 1:] -= v[:, :-1, 0]
        div_v[:-1, :] += v[:-1, :, 1]
        div_v[1:, :] -= v[:-1, :, 1]

        # Update x (gradient step)
        step = 0.5 / (1 + rho)
        x = x - step * (grad_data + rho * np.abs(x) * np.exp(1j * phase_x) * 1j * div_v)

        # z-update: proximal of TV norm (soft thresholding on gradient)
        phase_x = np.angle(x)
        dx_phase = np.diff(phase_x, axis=1, append=phase_x[:, -1:])
        dy_phase = np.diff(phase_x, axis=0, append=phase_x[-1:, :])
        grad_phase = np.stack([dx_phase, dy_phase], axis=-1)

        v = grad_phase + u
        norm_v = np.sqrt(v[..., 0]**2 + v[..., 1]**2 + 1e-10)
        shrink = np.maximum(norm_v - lam_tv / rho, 0) / norm_v
        z = v * shrink[..., np.newaxis]

        # u-update
        u = u + grad_phase - z

    amplitude = np.abs(x).astype(np.float32)
    phase = phase_unwrap_2d(np.angle(x).astype(np.float32))

    return amplitude, phase


def run_holography_reconstruction(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run holography reconstruction.

    Args:
        y: Hologram intensity (H, W)
        physics: Holography physics operator
        cfg: Configuration with:
            - wavelength: Light wavelength (default: 633e-9)
            - pixel_size: Detector pixel size (default: 5e-6)
            - prop_distance: Propagation distance (default: 0)
            - output: 'amplitude', 'phase', or 'complex' (default: 'amplitude')

    Returns:
        Tuple of (reconstructed, info_dict)
    """
    wavelength = cfg.get("wavelength", 633e-9)
    pixel_size = cfg.get("pixel_size", 5e-6)
    prop_distance = cfg.get("prop_distance", 0.0)
    output_type = cfg.get("output", "amplitude")

    info = {
        "solver": "angular_spectrum",
        "wavelength": wavelength,
        "pixel_size": pixel_size,
    }

    try:
        # Get parameters from physics if available
        if hasattr(physics, 'wavelength'):
            wavelength = physics.wavelength
        if hasattr(physics, 'pixel_size'):
            pixel_size = physics.pixel_size
        if hasattr(physics, 'prop_distance'):
            prop_distance = physics.prop_distance

        if hasattr(physics, 'info'):
            op_info = physics.info()
            wavelength = op_info.get('wavelength', wavelength)
            pixel_size = op_info.get('pixel_size', pixel_size)
            prop_distance = op_info.get('prop_distance', prop_distance)

        # Handle different input shapes
        if y.ndim == 2:
            hologram = y
        elif y.ndim == 3:
            # Multiple holograms - use first or average
            hologram = y[0] if y.shape[0] < y.shape[2] else y.mean(axis=2)
        else:
            info["error"] = "unexpected_input_shape"
            return y.astype(np.float32), info

        amplitude, phase = reconstruct_offaxis_hologram(
            hologram, wavelength, pixel_size, prop_distance
        )

        info["phase_range"] = (float(phase.min()), float(phase.max()))

        if output_type == "phase":
            return phase, info
        elif output_type == "complex":
            # Stack amplitude and phase
            result = np.stack([amplitude, phase], axis=-1)
            return result, info
        else:
            return amplitude, info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
