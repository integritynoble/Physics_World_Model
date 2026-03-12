"""Mismatch correction utilities applied before reconstruction."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from ..schemas import MismatchSpec


def apply_mismatch_corrections(
    measurements: np.ndarray,
    corrections: "list[MismatchSpec]",
    modality: str = "",
) -> np.ndarray:
    """Apply each mismatch correction in order.

    Each MismatchSpec.type routes to a specific correction function.
    Unknown types are silently skipped (warning printed).
    """
    y = measurements.copy().astype(np.float64)

    for spec in corrections:
        correction_fn = _CORRECTION_REGISTRY.get(spec.type)
        if correction_fn is not None:
            y = correction_fn(y, spec, modality)
        else:
            print(f"[mismatch] No correction registered for '{spec.type}' — skipping")

    return y


# ── Correction functions ───────────────────────────────────────────────────────

def _correct_beam_hardening(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Polynomial linearization of beam-hardening bias in sinograms.

    Uses a second-order polynomial: y_corrected = a0 + a1*y + a2*y^2
    with coefficients calibrated from water phantom convention.
    """
    a0, a1, a2 = 0.0, 1.0, -0.05   # approximate BH correction coefficients
    return (a0 + a1 * y + a2 * y ** 2).astype(np.float64)


def _correct_scatter(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Subtract a low-frequency scatter estimate (Gaussian blur of sinogram)."""
    sigma = 20.0   # scatter kernel width in pixels
    scatter_estimate = gaussian_filter(y, sigma=sigma)
    return np.maximum(y - 0.1 * scatter_estimate, 0.0)


def _correct_center_of_rotation(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Correct CoR offset by aligning the 0° and 180° projections."""
    if y.ndim < 2:
        return y
    # Estimate shift from cross-correlation of first and last row
    n_angles = y.shape[0]
    if n_angles < 2:
        return y
    proj_0   = y[0]
    proj_180 = y[n_angles // 2][::-1]   # flipped for parallel beam
    corr     = np.correlate(proj_0 - proj_0.mean(),
                             proj_180 - proj_180.mean(), mode="full")
    shift    = int(np.argmax(corr) - len(proj_0) + 1)
    return np.roll(y, -shift // 2, axis=1)


def _correct_b0_inhomogeneity(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Remove low-frequency phase ramp from k-space (B0 off-resonance)."""
    if not np.iscomplexobj(y):
        return y
    # Remove linear phase in k-space centre
    h, w = y.shape
    ramp_h = np.exp(-2j * np.pi * np.arange(h)[:, None] * 0.001)
    ramp_w = np.exp(-2j * np.pi * np.arange(w)[None, :] * 0.001)
    return (y * ramp_h * ramp_w).astype(np.complex64)


def _correct_motion(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Simple navigator-based motion correction (soft Gaussian windowing)."""
    # Conservative: apply mild Gaussian smoothing to suppress motion ghosts
    if np.iscomplexobj(y):
        mag = gaussian_filter(np.abs(y), sigma=0.5)
        return mag * np.exp(1j * np.angle(y))
    return gaussian_filter(y, sigma=0.5)


def _correct_illumination_nonuniformity(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Flat-field correction using estimated low-frequency background."""
    if y.ndim < 2:
        return y
    flat_field = gaussian_filter(y.astype(np.float64), sigma=max(y.shape) * 0.1)
    flat_field = np.maximum(flat_field, 1e-8)
    return (y / flat_field * flat_field.mean()).astype(np.float64)


def _correct_photobleaching(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Exponential bleaching correction along time/stack axis (last axis)."""
    if y.ndim < 3:
        return y
    n = y.shape[-1]
    t = np.arange(n, dtype=np.float64)
    rate = 0.01   # approximate bleaching rate per frame
    correction = np.exp(rate * t)
    return (y * correction).astype(y.dtype)


def _correct_spherical_aberration(
    y: np.ndarray, spec, modality: str
) -> np.ndarray:
    """Deconvolve mild spherical aberration via Wiener filter."""
    if y.ndim < 2:
        return y
    Y      = np.fft.fft2(y)
    snr    = 50.0
    sigma  = 0.5   # aberration kernel width
    # Gaussian PSF model for mild SA
    ny, nx = y.shape
    fy     = np.fft.fftfreq(ny)[:, None]
    fx     = np.fft.fftfreq(nx)[None, :]
    H      = np.exp(-2 * np.pi**2 * sigma**2 * (fy**2 + fx**2))
    W      = np.conj(H) / (np.abs(H)**2 + 1.0 / snr)
    return np.real(np.fft.ifft2(Y * W)).astype(np.float64)


# ── Registry ───────────────────────────────────────────────────────────────────

_CORRECTION_REGISTRY: dict[str, callable] = {
    "beam_hardening":              _correct_beam_hardening,
    "scatter":                     _correct_scatter,
    "center_of_rotation_offset":   _correct_center_of_rotation,
    "center_of_rotation":          _correct_center_of_rotation,
    "b0_inhomogeneity":            _correct_b0_inhomogeneity,
    "motion":                      _correct_motion,
    "illumination_nonuniformity":  _correct_illumination_nonuniformity,
    "photobleaching":              _correct_photobleaching,
    "spherical_aberration":        _correct_spherical_aberration,
}
