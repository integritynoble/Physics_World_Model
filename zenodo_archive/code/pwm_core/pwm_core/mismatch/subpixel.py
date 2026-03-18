"""Sub-pixel shift and warp utilities for mismatch injection.

Replaces ``np.roll`` (wrapping, integer-only) with physically correct
``scipy.ndimage.shift`` / ``affine_transform`` (zero-padded boundaries,
sub-pixel interpolation).

Key differences from ``np.roll``:
- ``mode="constant"`` fills boundaries with zeros instead of wrapping.
- ``order=1`` enables sub-pixel (bilinear) interpolation.
- 0.5 px shifts produce a measurable change (``np.roll`` rounds to 0).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage


def subpixel_shift_2d(
    arr: np.ndarray,
    dx: float,
    dy: float,
    *,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Shift a 2-D array by ``(dx, dy)`` pixels with sub-pixel accuracy.

    Parameters
    ----------
    arr : ndarray, shape (H, W)
        Input image.
    dx : float
        Horizontal shift (positive = rightward).
    dy : float
        Vertical shift (positive = downward).
    order : int
        Spline interpolation order (0 = nearest, 1 = bilinear).
    mode : str
        Boundary handling (``"constant"`` fills with *cval*).
    cval : float
        Fill value for ``mode="constant"``.

    Returns
    -------
    ndarray, shape (H, W)
        Shifted array (same dtype as input cast to float64).
    """
    if dx == 0.0 and dy == 0.0:
        return arr.copy()
    # scipy.ndimage.shift uses (row, col) = (dy, dx)
    return ndimage.shift(
        arr.astype(np.float64),
        shift=[dy, dx],
        order=order,
        mode=mode,
        cval=cval,
    )


def subpixel_shift_3d_spatial(
    arr: np.ndarray,
    dx: float,
    dy: float,
    **kwargs,
) -> np.ndarray:
    """Apply :func:`subpixel_shift_2d` to each frame of a (H, W, T) array.

    Parameters
    ----------
    arr : ndarray, shape (H, W, T)
        Stack of 2-D frames.
    dx, dy : float
        Spatial shift applied identically to every frame.
    **kwargs
        Forwarded to :func:`subpixel_shift_2d`.

    Returns
    -------
    ndarray, shape (H, W, T)
        Shifted stack.
    """
    out = np.empty_like(arr, dtype=np.float64)
    for t in range(arr.shape[2]):
        out[:, :, t] = subpixel_shift_2d(arr[:, :, t], dx, dy, **kwargs)
    return out


def subpixel_warp_2d(
    arr: np.ndarray,
    dx: float,
    dy: float,
    theta_deg: float = 0.0,
    *,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Shift + rotate a 2-D array via affine transform.

    Matches the ``_warp_mask2d`` convention used in
    ``benchmarks/_cassi_upwmi.py`` (lines 91-104).

    Parameters
    ----------
    arr : ndarray, shape (H, W)
        Input image.
    dx : float
        Horizontal shift (pixels, positive = rightward).
    dy : float
        Vertical shift (pixels, positive = downward).
    theta_deg : float
        Counter-clockwise rotation in degrees.
    order : int
        Spline interpolation order.
    mode : str
        Boundary handling.
    cval : float
        Fill value for ``mode="constant"``.

    Returns
    -------
    ndarray, shape (H, W)
        Warped array.
    """
    if dx == 0.0 and dy == 0.0 and theta_deg == 0.0:
        return arr.copy()

    H, W = arr.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    theta_rad = np.deg2rad(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    # Rotation matrix (maps output coords to input coords)
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    # Offset: rotate around centre, then shift
    offset = np.array([cy, cx]) - R @ np.array([cy - dy, cx - dx])

    return ndimage.affine_transform(
        arr.astype(np.float64),
        matrix=R,
        offset=offset,
        order=order,
        mode=mode,
        cval=cval,
        output_shape=(H, W),
    )
