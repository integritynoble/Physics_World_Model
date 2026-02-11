"""SPECT (Single Photon Emission Computed Tomography) helper functions.

Pure functions for SPECT forward modelling with collimator response.
These are NOT standalone operators (per D4 rule).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def collimator_projection(
    x: np.ndarray,
    n_angles: int = 32,
    n_detectors: int = 64,
    collimator_response: float = 2.0,
) -> np.ndarray:
    """Forward project an emission image through a collimator model.

    Applies depth-dependent Gaussian blurring (collimator response)
    before Radon-style projection.

    Parameters
    ----------
    x : np.ndarray
        2D emission activity map (H, W), non-negative.
    n_angles : int
        Number of angular projections.
    n_detectors : int
        Number of detector bins per angle.
    collimator_response : float
        Gaussian sigma for collimator PSF (pixels).

    Returns
    -------
    np.ndarray
        Sinogram, shape (n_angles, n_detectors).
    """
    x = np.asarray(x, dtype=np.float64)

    # Apply collimator PSF (depth-dependent blur approximated as uniform)
    blurred = ndimage.gaussian_filter(x, sigma=collimator_response)

    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = np.zeros((n_angles, n_detectors), dtype=np.float64)

    for i, angle in enumerate(angles):
        rotated = ndimage.rotate(blurred, angle, reshape=False, order=1, mode="constant")
        projection = np.sum(rotated, axis=0)
        if len(projection) != n_detectors:
            indices = np.linspace(0, len(projection) - 1, n_detectors)
            sinogram[i] = np.interp(indices, np.arange(len(projection)), projection)
        else:
            sinogram[i] = projection

    return sinogram


def attenuation_correction(
    sinogram: np.ndarray,
    mu_map: np.ndarray,
) -> np.ndarray:
    """Apply attenuation correction to a SPECT sinogram.

    Parameters
    ----------
    sinogram : np.ndarray
        Measured sinogram (n_angles, n_detectors).
    mu_map : np.ndarray
        Attenuation sinogram (same shape).

    Returns
    -------
    np.ndarray
        Attenuation-corrected sinogram.
    """
    sinogram = np.asarray(sinogram, dtype=np.float64)
    mu_map = np.asarray(mu_map, dtype=np.float64)
    acf = np.exp(mu_map)
    return sinogram * acf
