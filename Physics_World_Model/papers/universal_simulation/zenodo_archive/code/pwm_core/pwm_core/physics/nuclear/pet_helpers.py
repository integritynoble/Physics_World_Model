"""PET (Positron Emission Tomography) helper functions.

Pure functions for PET forward modelling and MLEM reconstruction.
These are NOT standalone operators (per D4 rule).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def system_matrix_projection(
    x: np.ndarray,
    n_angles: int = 32,
    n_detectors: int = 64,
) -> np.ndarray:
    """Forward project an emission image to a sinogram.

    Uses rotation-based Radon projection as a simplified PET system matrix.

    Parameters
    ----------
    x : np.ndarray
        2D emission activity map (H, W), non-negative.
    n_angles : int
        Number of angular projections.
    n_detectors : int
        Number of detector bins per angle (defaults to image width).

    Returns
    -------
    np.ndarray
        Sinogram, shape (n_angles, n_detectors).
    """
    x = np.asarray(x, dtype=np.float64)
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = np.zeros((n_angles, n_detectors), dtype=np.float64)

    for i, angle in enumerate(angles):
        rotated = ndimage.rotate(x, angle, reshape=False, order=1, mode="constant")
        projection = np.sum(rotated, axis=0)
        # Resample to n_detectors if sizes differ
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
    """Apply attenuation correction to a PET sinogram.

    Divides by the attenuation correction factor (ACF) computed from
    the attenuation map line integrals.

    Parameters
    ----------
    sinogram : np.ndarray
        Measured sinogram (n_angles, n_detectors).
    mu_map : np.ndarray
        Attenuation sinogram (same shape), representing line integrals of mu.

    Returns
    -------
    np.ndarray
        Attenuation-corrected sinogram.
    """
    sinogram = np.asarray(sinogram, dtype=np.float64)
    mu_map = np.asarray(mu_map, dtype=np.float64)
    acf = np.exp(mu_map)
    return sinogram * acf


def mlem_update(
    x: np.ndarray,
    y: np.ndarray,
    A_forward,
    A_adjoint,
    n_iter: int = 10,
) -> np.ndarray:
    """Run MLEM (Maximum-Likelihood Expectation-Maximization) reconstruction.

    Parameters
    ----------
    x : np.ndarray
        Initial estimate (non-negative, same shape as object).
    y : np.ndarray
        Measured data (sinogram, non-negative).
    A_forward : callable
        Forward projection function: x -> sinogram.
    A_adjoint : callable
        Back-projection function: sinogram -> image.
    n_iter : int
        Number of MLEM iterations.

    Returns
    -------
    np.ndarray
        Reconstructed image after n_iter iterations.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64)
    eps = 1e-10

    # Sensitivity image: A^T(1)
    ones_sino = np.ones_like(y)
    sensitivity = A_adjoint(ones_sino)
    sensitivity = np.maximum(sensitivity, eps)

    for _ in range(n_iter):
        y_est = A_forward(x) + eps
        ratio = y / y_est
        correction = A_adjoint(ratio)
        x = x * correction / sensitivity

    return x
