"""TEM helper functions (pure, stateless, D4-compliant).

Functions
---------
compute_ctf               Contrast Transfer Function array.
phase_object_transmission Complex transmission through thin phase object.
ctf_zeros                 Zero-crossing spatial frequencies of the CTF.
"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_ctf(
    freqs_x: np.ndarray,
    freqs_y: np.ndarray,
    defocus_nm: float = -50.0,
    Cs_mm: float = 1.0,
    wavelength_pm: float = 2.51,
) -> np.ndarray:
    """Compute the TEM Contrast Transfer Function.

    CTF(q) = sin( pi * lambda * defocus * q^2
                  - 0.5 * pi * Cs * lambda^3 * q^4 )

    Parameters
    ----------
    freqs_x, freqs_y : ndarray
        Spatial frequency arrays (1/pixel units, from np.fft.fftfreq).
    defocus_nm : float
        Defocus in nanometers (negative = underfocus).
    Cs_mm : float
        Spherical aberration coefficient in millimeters.
    wavelength_pm : float
        Electron wavelength in picometers (2.51 pm for 200 kV).

    Returns
    -------
    ndarray
        2D CTF array (real-valued, range [-1, 1]).
    """
    FY, FX = np.meshgrid(
        np.asarray(freqs_y, dtype=np.float64),
        np.asarray(freqs_x, dtype=np.float64),
        indexing="ij",
    )
    q2 = FX ** 2 + FY ** 2

    wl_nm = wavelength_pm * 1e-3  # pm -> nm
    Cs_nm = Cs_mm * 1e6           # mm -> nm

    chi = (
        np.pi * wl_nm * defocus_nm * q2
        - 0.5 * np.pi * Cs_nm * wl_nm ** 3 * q2 ** 2
    )
    return np.sin(chi)


def phase_object_transmission(
    x: np.ndarray,
    sigma: float = 0.00729,
    V: float = 1.0,
) -> np.ndarray:
    """Compute complex transmission for a thin phase object.

    T(r) = exp(i * sigma * V * x(r))

    Parameters
    ----------
    x : ndarray
        2D projected potential (real-valued).
    sigma : float
        Interaction parameter (rad / (V * nm)).
    V : float
        Mean inner potential scaling (volts).

    Returns
    -------
    ndarray
        Complex transmission array.
    """
    phase = sigma * V * np.asarray(x, dtype=np.float64)
    return np.exp(1j * phase)


def ctf_zeros(
    defocus_nm: float = -50.0,
    Cs_mm: float = 1.0,
    wavelength_pm: float = 2.51,
    n_zeros: int = 5,
) -> List[float]:
    """Compute the first n zero-crossing frequencies of the CTF.

    Zero crossings occur where chi(q) = n * pi, i.e.,
    pi * lambda * defocus * q^2 - 0.5 * pi * Cs * lambda^3 * q^4 = n * pi

    For the first passband (neglecting Cs for small q):
    q_n = sqrt(n / (lambda * |defocus|))

    Parameters
    ----------
    defocus_nm : float
        Defocus in nm (negative = underfocus).
    Cs_mm : float
        Spherical aberration coefficient in mm.
    wavelength_pm : float
        Electron wavelength in pm.
    n_zeros : int
        Number of zero crossings to return.

    Returns
    -------
    list[float]
        Spatial frequencies (1/nm) of the first n zero crossings.
    """
    wl_nm = wavelength_pm * 1e-3
    df = abs(defocus_nm)
    Cs_nm = Cs_mm * 1e6

    zeros: List[float] = []

    # Numerical root finding via dense sampling
    q_max = 2.0 / wl_nm  # reasonable upper bound
    q_arr = np.linspace(1e-6, q_max, 100000)
    q2 = q_arr ** 2
    chi = np.pi * wl_nm * defocus_nm * q2 - 0.5 * np.pi * Cs_nm * wl_nm ** 3 * q2 ** 2
    sin_chi = np.sin(chi)

    # Find sign changes
    sign_changes = np.where(np.diff(np.sign(sin_chi)))[0]
    for idx in sign_changes:
        if len(zeros) >= n_zeros:
            break
        # Linear interpolation for zero crossing
        q0 = q_arr[idx]
        q1 = q_arr[idx + 1]
        s0 = sin_chi[idx]
        s1 = sin_chi[idx + 1]
        if abs(s1 - s0) > 1e-30:
            q_zero = q0 - s0 * (q1 - q0) / (s1 - s0)
        else:
            q_zero = (q0 + q1) / 2.0
        zeros.append(float(q_zero))

    return zeros[:n_zeros]
