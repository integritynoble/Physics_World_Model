"""Electron Tomography helper functions (pure, stateless, D4-compliant).

Functions
---------
tilt_project     Project a 3D volume at a given tilt angle.
alignment_shift  Shift a 2D projection (sub-pixel alignment).
sirt_recon       Basic SIRT reconstruction from tilt-series projections.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def tilt_project(
    volume: np.ndarray,
    angle_deg: float = 0.0,
    axis: str = "y",
) -> np.ndarray:
    """Project a 3D volume along a tilt angle (Radon-like).

    Rotates the volume about the specified axis and sums along it.

    Parameters
    ----------
    volume : ndarray
        3D array of shape (D, H, W).
    angle_deg : float
        Tilt angle in degrees.
    axis : str
        Rotation axis: 'y' (default) or 'x'.

    Returns
    -------
    ndarray
        2D projection image.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.ndim}D")

    D, H, W = vol.shape

    if axis == "y":
        # Rotate in the (D, W) plane (axes 0, 2) then sum along axis 0
        rotated = ndimage.rotate(
            vol, angle_deg, axes=(0, 2), reshape=False, order=1, mode="constant"
        )
        return rotated.sum(axis=0)
    elif axis == "x":
        # Rotate in the (D, H) plane (axes 0, 1) then sum along axis 0
        rotated = ndimage.rotate(
            vol, angle_deg, axes=(0, 1), reshape=False, order=1, mode="constant"
        )
        return rotated.sum(axis=0)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")


def alignment_shift(
    proj: np.ndarray,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    """Shift a 2D projection by sub-pixel amounts.

    Parameters
    ----------
    proj : ndarray
        2D projection image.
    dx, dy : float
        Shift amounts in pixels.

    Returns
    -------
    ndarray
        Shifted projection.
    """
    p = np.asarray(proj, dtype=np.float64)
    return ndimage.shift(p, [dy, dx], order=1, mode="constant")


def sirt_recon(
    projections: np.ndarray,
    angles: np.ndarray,
    n_iter: int = 10,
) -> np.ndarray:
    """Basic SIRT (Simultaneous Iterative Reconstruction Technique).

    Reconstructs a 3D volume from a set of 2D projections at given tilt angles.

    Parameters
    ----------
    projections : ndarray
        3D array of shape (n_angles, H, W) -- stack of 2D projections.
    angles : ndarray
        1D array of tilt angles in degrees, length n_angles.
    n_iter : int
        Number of SIRT iterations.

    Returns
    -------
    ndarray
        Reconstructed 3D volume of shape (D, H, W) where D = H.
    """
    projs = np.asarray(projections, dtype=np.float64)
    angs = np.asarray(angles, dtype=np.float64)

    n_angles, H, W = projs.shape
    D = H  # Assume isotropic reconstruction grid

    # Initialize volume to zero
    vol = np.zeros((D, H, W), dtype=np.float64)

    # Relaxation parameter
    lam = 1.0 / max(n_angles, 1)

    for _ in range(n_iter):
        for i in range(n_angles):
            # Forward project current estimate
            proj_est = tilt_project(vol, angs[i], axis="y")

            # Compute residual
            residual = projs[i] - proj_est

            # Back-project residual: distribute evenly across depth slices
            # (simple uniform smearing -- SIRT approximation)
            update = np.zeros_like(vol)
            rotated_resid = np.stack([residual] * D, axis=0)  # (D, H, W)
            # Rotate back
            update = ndimage.rotate(
                rotated_resid, -angs[i], axes=(0, 2),
                reshape=False, order=1, mode="constant",
            )

            vol += lam * update

    return vol
