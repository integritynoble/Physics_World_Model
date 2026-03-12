"""Physics-based noise models for forward simulation."""
from __future__ import annotations
import numpy as np


def apply_poisson_noise(
    signal: np.ndarray,
    I0: float = 1e5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply Poisson (shot) noise to a transmittance signal.

    The measurement model is:
        y_counts ~ Poisson(I0 * T)  where T = signal / signal.max()
    Result is rescaled back to the original signal range.
    """
    if rng is None:
        rng = np.random.default_rng()
    s = np.asarray(signal, dtype=np.float64)
    max_val = s.max()
    if max_val <= 0:
        return s.astype(np.float32)
    T = s / max_val
    T = np.clip(T, 0.0, 1.0)
    counts = rng.poisson(I0 * T).astype(np.float64)
    return (counts / I0 * max_val).astype(np.float32)


def apply_gaussian_noise(
    signal: np.ndarray,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian (readout / thermal) noise.

    Args:
        signal: Input signal (any units).
        sigma:  Standard deviation in the same units as signal (electrons, ADU, etc.)
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


def apply_dark_current(
    signal: np.ndarray,
    rate: float = 0.1,
    exposure_s: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Poisson-distributed dark current.

    Args:
        rate:       Dark current in electrons/pixel/second.
        exposure_s: Exposure time in seconds.
    """
    if rng is None:
        rng = np.random.default_rng()
    mean_dark = rate * exposure_s
    dark = rng.poisson(mean_dark, size=signal.shape).astype(np.float32)
    return (signal + dark).astype(np.float32)


def add_mixed_poisson_gaussian(
    signal: np.ndarray,
    I0: float = 1e5,
    sigma_readout: float = 5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Combined Poisson + Gaussian model (common in CCD/sCMOS detectors)."""
    s = apply_poisson_noise(signal, I0=I0, rng=rng)
    return apply_gaussian_noise(s, sigma=sigma_readout, rng=rng)
