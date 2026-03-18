"""X-ray radiography helper functions.

Pure functions for planar X-ray radiography forward modelling.
These are NOT standalone operators (per D4 rule); they are called
from graph primitives or test harnesses.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def planar_beer_lambert(
    x: np.ndarray,
    I_0: float = 10000.0,
    mu: float = 1.0,
) -> np.ndarray:
    """Compute transmitted intensity via Beer-Lambert law.

    Parameters
    ----------
    x : np.ndarray
        Tissue thickness / attenuation map (non-negative).
    I_0 : float
        Incident photon flux (counts or intensity).
    mu : float
        Linear attenuation coefficient.

    Returns
    -------
    np.ndarray
        Transmitted intensity: I_0 * exp(-mu * x).
    """
    x = np.asarray(x, dtype=np.float64)
    return I_0 * np.exp(-mu * x)


def scatter_estimate(
    y: np.ndarray,
    fraction: float = 0.1,
    sigma: float = 5.0,
) -> np.ndarray:
    """Estimate smooth scatter component from a radiograph.

    Models scatter as a low-pass-filtered fraction of the primary signal.

    Parameters
    ----------
    y : np.ndarray
        Measured radiograph (transmitted intensity image).
    fraction : float
        Scatter-to-primary ratio (typically 0.05 -- 0.3).
    sigma : float
        Gaussian smoothing sigma for the scatter kernel.

    Returns
    -------
    np.ndarray
        Estimated scatter component (same shape as y).
    """
    y = np.asarray(y, dtype=np.float64)
    return fraction * ndimage.gaussian_filter(y, sigma=sigma)
