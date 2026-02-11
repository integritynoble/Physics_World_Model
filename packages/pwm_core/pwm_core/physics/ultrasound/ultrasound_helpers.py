"""Ultrasound helper functions.

Pure functions for ultrasound forward modelling (RF data generation,
delay-and-sum beamforming, impulse response filtering).
These are NOT standalone operators (per D4 rule).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def propagate_rf(
    x: np.ndarray,
    speed: float = 1540.0,
    n_elements: int = 32,
    element_pitch: float = 0.3e-3,
    n_samples: int = 512,
    fs: float = 40e6,
) -> np.ndarray:
    """Generate synthetic RF channel data from a 2D tissue reflectivity map.

    Uses a simplified pulse-echo model: for each transducer element, sum
    contributions from all pixels weighted by round-trip time delay.

    Parameters
    ----------
    x : np.ndarray
        2D tissue reflectivity map (Nz, Nx).
    speed : float
        Speed of sound in m/s.
    n_elements : int
        Number of transducer elements.
    element_pitch : float
        Spacing between elements in metres.
    n_samples : int
        Number of time samples per channel.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        RF channel data, shape (n_elements, n_samples).
    """
    x = np.asarray(x, dtype=np.float64)
    Nz, Nx = x.shape[-2], x.shape[-1]

    # Pixel grid (assume square pixels spanning the element aperture)
    pixel_size_x = (n_elements * element_pitch) / Nx
    pixel_size_z = (n_samples / fs * speed / 2.0) / Nz

    rf = np.zeros((n_elements, n_samples), dtype=np.float64)

    for elem in range(n_elements):
        elem_x = (elem - n_elements / 2.0) * element_pitch
        for iz in range(Nz):
            z_pos = (iz + 0.5) * pixel_size_z
            for ix in range(Nx):
                x_pos = (ix - Nx / 2.0) * pixel_size_x
                dist = np.sqrt((elem_x - x_pos) ** 2 + z_pos ** 2)
                t_round = 2.0 * dist / speed
                sample_idx = int(t_round * fs)
                if 0 <= sample_idx < n_samples:
                    rf[elem, sample_idx] += x[iz, ix]

    return rf


def delay_and_sum(
    rf: np.ndarray,
    speed: float = 1540.0,
    focus_depth: float = 0.02,
    element_pitch: float = 0.3e-3,
    pixel_pitch: float = 0.3e-3,
    grid_shape: tuple = (64, 64),
) -> np.ndarray:
    """Delay-and-sum beamforming from RF channel data.

    Parameters
    ----------
    rf : np.ndarray
        RF channel data, shape (n_elements, n_samples).
    speed : float
        Speed of sound in m/s.
    focus_depth : float
        Focal depth in metres (used to set image depth range).
    element_pitch : float
        Element spacing in metres.
    pixel_pitch : float
        Output pixel spacing in metres.
    grid_shape : tuple
        Output image shape (Nz, Nx).

    Returns
    -------
    np.ndarray
        Beamformed image, shape grid_shape.
    """
    rf = np.asarray(rf, dtype=np.float64)
    n_elements, n_samples = rf.shape
    Nz, Nx = grid_shape
    fs_approx = n_samples * speed / (2.0 * focus_depth) if focus_depth > 0 else 40e6

    image = np.zeros((Nz, Nx), dtype=np.float64)
    for iz in range(Nz):
        z_pos = (iz + 0.5) * focus_depth / Nz
        for ix in range(Nx):
            x_pos = (ix - Nx / 2.0) * pixel_pitch
            total = 0.0
            for elem in range(n_elements):
                elem_x = (elem - n_elements / 2.0) * element_pitch
                dist = np.sqrt((elem_x - x_pos) ** 2 + z_pos ** 2)
                t_round = 2.0 * dist / speed
                sample_idx = int(t_round * fs_approx)
                if 0 <= sample_idx < n_samples:
                    total += rf[elem, sample_idx]
            image[iz, ix] = total

    return image


def apply_impulse_response(
    rf: np.ndarray,
    ir: np.ndarray,
) -> np.ndarray:
    """Convolve RF channel data with a transducer impulse response.

    Parameters
    ----------
    rf : np.ndarray
        RF channel data, shape (n_elements, n_samples).
    ir : np.ndarray
        1D impulse response kernel.

    Returns
    -------
    np.ndarray
        Filtered RF data (same shape as rf, truncated to original length).
    """
    rf = np.asarray(rf, dtype=np.float64)
    ir = np.asarray(ir, dtype=np.float64)
    filtered = np.zeros_like(rf)
    for elem in range(rf.shape[0]):
        conv = np.convolve(rf[elem], ir, mode="same")
        filtered[elem] = conv
    return filtered
