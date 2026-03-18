"""SEM helper functions (pure, stateless, D4-compliant).

Functions
---------
se_yield      Secondary electron yield map from material density map.
bse_yield     Backscattered electron yield map.
apply_scan_drift  Sub-pixel shift to simulate SEM scan drift.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def se_yield(
    material_map: np.ndarray,
    voltage_kv: float = 15.0,
) -> np.ndarray:
    """Compute secondary electron yield map.

    Simple linear model: SE yield scales with material density and inversely
    with beam voltage (higher voltage => deeper penetration => fewer SE escape).

    Parameters
    ----------
    material_map : ndarray
        2D array of material density values (arbitrary units, non-negative).
    voltage_kv : float
        Accelerating voltage in kilovolts.

    Returns
    -------
    ndarray
        SE yield map, same shape as material_map.
    """
    mat = np.asarray(material_map, dtype=np.float64)
    # SE yield ~ density / sqrt(voltage), clamped to [0, 1]
    scale = 1.0 / max(np.sqrt(voltage_kv), 1e-12)
    y = mat * scale
    return np.clip(y, 0.0, 1.0)


def bse_yield(
    material_map: np.ndarray,
    voltage_kv: float = 15.0,
    angle: float = 0.0,
) -> np.ndarray:
    """Compute backscattered electron yield map.

    BSE yield approximately scales with atomic number Z (here proxied by
    material density) squared, with weak voltage and angular dependence.

    Parameters
    ----------
    material_map : ndarray
        2D material density/atomic-number proxy.
    voltage_kv : float
        Accelerating voltage in kV.
    angle : float
        Incidence angle in radians (0 = normal).

    Returns
    -------
    ndarray
        BSE yield map, same shape as material_map.
    """
    mat = np.asarray(material_map, dtype=np.float64)
    # BSE yield ~ mat^2 * (1 + cos(angle)) / 2, weak voltage dependence
    angular_factor = (1.0 + np.cos(angle)) / 2.0
    voltage_factor = 1.0 / (1.0 + 0.01 * voltage_kv)
    y = mat ** 2 * angular_factor * voltage_factor
    return np.clip(y, 0.0, 1.0)


def apply_scan_drift(
    image: np.ndarray,
    drift_x: float = 0.0,
    drift_y: float = 0.0,
) -> np.ndarray:
    """Apply sub-pixel scan drift to an SEM image.

    Parameters
    ----------
    image : ndarray
        2D image array.
    drift_x, drift_y : float
        Sub-pixel drift in x and y directions (pixels).

    Returns
    -------
    ndarray
        Shifted image, same shape.
    """
    img = np.asarray(image, dtype=np.float64)
    return ndimage.shift(img, [drift_y, drift_x], order=1, mode="constant")
