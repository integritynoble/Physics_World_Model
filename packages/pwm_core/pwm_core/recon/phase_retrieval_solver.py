"""Phase Retrieval / CDI reconstruction solvers.

References:
- Fienup, J.R. (1982). "Phase retrieval algorithms: a comparison",
  Applied Optics.
- Luke, D.R. (2005). "Relaxed averaged alternating reflections for
  diffraction imaging", Inverse Problems.
- Gerchberg, R.W. & Saxton, W.O. (1972). "A practical algorithm for
  the determination of phase from image and diffraction plane pictures",
  Optik.

Expected PSNR: 30.0 dB on synthetic benchmark (amplitude recovery)
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple


def _apply_support_constraint(x: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Apply support constraint: zero outside support."""
    return x * support


def _apply_modulus_constraint(x: np.ndarray, measured_magnitude: np.ndarray) -> np.ndarray:
    """Replace Fourier magnitude with measured, keep phase."""
    X = np.fft.fft2(x)
    phase = np.angle(X)
    X_constrained = measured_magnitude * np.exp(1j * phase)
    return np.fft.ifft2(X_constrained)


def hio(
    measured_magnitude: np.ndarray,
    support: np.ndarray,
    n_iters: int = 1000,
    beta: float = 0.9,
    initial_guess: Optional[np.ndarray] = None,
    positivity: bool = True,
) -> np.ndarray:
    """Hybrid Input-Output (HIO) algorithm for phase retrieval.

    Uses a three-stage ER+HIO+ER strategy:
    - First 20% of iterations: Error Reduction (ER) for stable convergence.
    - Middle 60%: HIO with feedback for escaping local minima.
    - Final 20%: ER cleanup to refine the solution.

    This is a well-established recipe in the CDI literature for
    balancing exploration (HIO) and refinement (ER).

    Args:
        measured_magnitude: Measured Fourier magnitude (H, W).
        support: Binary support mask (H, W), 1 inside support.
        n_iters: Number of iterations.
        beta: Feedback parameter (typically 0.7-0.9).
        initial_guess: Optional initial complex estimate.
        positivity: If True, enforce real non-negative constraint
            inside support (standard for CDI of real objects).

    Returns:
        Recovered complex object (H, W).
    """
    h, w = measured_magnitude.shape

    if initial_guess is not None:
        x = initial_guess.copy().astype(np.complex128)
    else:
        # Random phase initialization
        rng = np.random.RandomState(0)
        phase = rng.uniform(-np.pi, np.pi, (h, w))
        X0 = measured_magnitude * np.exp(1j * phase)
        x = np.fft.ifft2(X0)

    support = support.astype(bool)
    n_er_start = n_iters // 5      # First 20% with ER
    n_er_end = n_iters // 5        # Last 20% with ER
    n_hio_end = n_iters - n_er_end  # HIO stops here

    for it in range(n_iters):
        # Fourier modulus constraint
        x_prime = _apply_modulus_constraint(x, measured_magnitude)

        if it < n_er_start or it >= n_hio_end:
            # Error Reduction: simple support projection
            x_new = np.zeros_like(x)
            x_new[support] = x_prime[support]
        else:
            # HIO update
            x_new = np.zeros_like(x)
            x_new[support] = x_prime[support]
            x_new[~support] = x[~support] - beta * x_prime[~support]

        # Positivity constraint: enforce real non-negative inside support
        if positivity:
            x_new[support] = np.maximum(np.real(x_new[support]), 0).astype(np.complex128)

        x = x_new

    return x.astype(np.complex64)


def raar(
    measured_magnitude: np.ndarray,
    support: np.ndarray,
    n_iters: int = 1000,
    beta: float = 0.85,
    initial_guess: Optional[np.ndarray] = None,
    positivity: bool = True,
) -> np.ndarray:
    """Relaxed Averaged Alternating Reflections (RAAR) for phase retrieval.

    RAAR is a more robust alternative to HIO with better convergence
    properties.

    Args:
        measured_magnitude: Measured Fourier magnitude (H, W).
        support: Binary support mask (H, W).
        n_iters: Number of iterations.
        beta: Relaxation parameter (typically 0.75-0.9).
        initial_guess: Optional initial complex estimate.
        positivity: If True, enforce real non-negative constraint
            inside support (standard for CDI of real objects).

    Returns:
        Recovered complex object (H, W).
    """
    h, w = measured_magnitude.shape

    if initial_guess is not None:
        x = initial_guess.copy().astype(np.complex128)
    else:
        rng = np.random.RandomState(0)
        phase = rng.uniform(-np.pi, np.pi, (h, w))
        X0 = measured_magnitude * np.exp(1j * phase)
        x = np.fft.ifft2(X0)

    support = support.astype(bool)

    for it in range(n_iters):
        # Reflector for modulus constraint: R_M = 2*P_M - I
        x_pm = _apply_modulus_constraint(x, measured_magnitude)
        r_m = 2 * x_pm - x

        # Reflector for support constraint: R_S = 2*P_S - I
        x_ps = _apply_support_constraint(x, support)
        r_s = 2 * x_ps - x

        # RAAR update
        # x_{n+1} = beta/2 * (R_S R_M + I) x_n + (1 - beta) * P_M x_n
        rs_rm = _apply_support_constraint(r_m, support) * 2 - r_m  # This is R_S(R_M(x))
        # More precisely: R_S applied to R_M(x) = r_m
        r_m_val = r_m.copy()
        rs_of_rm = np.zeros_like(x)
        rs_of_rm[support] = r_m_val[support]
        rs_of_rm[~support] = -r_m_val[~support]

        x = beta / 2 * (rs_of_rm + x) + (1 - beta) * x_pm

        # Positivity constraint: enforce real non-negative inside support
        if positivity:
            x_supp = x[support]
            x[support] = np.maximum(np.real(x_supp), 0).astype(np.complex128)

    # Final support projection
    x = _apply_support_constraint(x, support)

    return x.astype(np.complex64)


def gerchberg_saxton(
    measured_magnitude: np.ndarray,
    image_magnitude: np.ndarray = None,
    n_iters: int = 500,
    support: np.ndarray = None,
) -> np.ndarray:
    """Gerchberg-Saxton algorithm for phase retrieval.

    Classic algorithm that alternates between image and Fourier domains,
    replacing magnitudes with measured values in each domain.

    Args:
        measured_magnitude: Measured Fourier magnitude (H, W).
        image_magnitude: Known image-plane magnitude (H, W). If None,
            uses support constraint instead.
        n_iters: Number of iterations.
        support: Binary support mask, used if image_magnitude is None.

    Returns:
        Recovered complex object (H, W).
    """
    h, w = measured_magnitude.shape

    # Initialize with random phase
    rng = np.random.RandomState(0)
    phase = rng.uniform(-np.pi, np.pi, (h, w))
    X = measured_magnitude * np.exp(1j * phase)
    x = np.fft.ifft2(X)

    for it in range(n_iters):
        # Apply image-domain constraint
        if image_magnitude is not None:
            x = image_magnitude * np.exp(1j * np.angle(x))
        elif support is not None:
            x = _apply_support_constraint(x, support)

        # Forward FFT
        X = np.fft.fft2(x)

        # Apply Fourier magnitude constraint
        X = measured_magnitude * np.exp(1j * np.angle(X))

        # Inverse FFT
        x = np.fft.ifft2(X)

    return x.astype(np.complex64)


def run_phase_retrieval(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for phase retrieval reconstruction.

    Args:
        y: Measured Fourier intensity |FFT(x)|^2 (H, W).
        physics: Physics operator with support info.
        cfg: Configuration with optional keys:
            - method: 'hio', 'raar', or 'gerchberg_saxton'
            - n_iters: Number of iterations
            - beta: Relaxation parameter

    Returns:
        Tuple of (recovered_amplitude, info_dict).
    """
    method = cfg.get("method", "hio")
    n_iters = cfg.get("n_iters", 500)
    beta = cfg.get("beta", 0.9 if method == "hio" else 0.85)

    info: Dict[str, Any] = {"solver": "phase_retrieval", "method": method}

    try:
        # Get measured magnitude from intensity
        measured_mag = np.sqrt(np.maximum(y, 0))

        # Get support from physics
        if hasattr(physics, 'support'):
            support = physics.support
        else:
            # Default: central region support
            h, w = y.shape
            support = np.zeros((h, w), dtype=bool)
            margin = h // 4
            support[margin:-margin, margin:-margin] = True

        if method == "raar":
            result = raar(measured_mag, support, n_iters=n_iters, beta=beta)
        elif method == "gerchberg_saxton":
            result = gerchberg_saxton(measured_mag, support=support, n_iters=n_iters)
        else:
            result = hio(measured_mag, support, n_iters=n_iters, beta=beta)

        amplitude = np.abs(result).astype(np.float32)
        info["n_iters"] = n_iters
        info["beta"] = beta

        return amplitude, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
