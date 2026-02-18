"""Tier-1 Adapter: Wave Approximations
=======================================

Wraps wave-optics approximation primitives: Fourier optics, paraxial
propagation, Born approximation, angular spectrum method.

Tier-1 primitives use approximate wave physics that is cheap to compute
and has well-defined adjoints (typically via conjugation in Fourier domain).

Examples:
- Fresnel propagation
- Angular spectrum propagation
- Paraxial beam propagation
- Convolution with optical transfer function (OTF)
- Born/Rytov scattering approximation

Usage:
    from pwm_core.graph.adapters import Tier1Adapter

    adapter = Tier1Adapter(
        forward_kernel=my_fresnel_propagation,
        adjoint_kernel=my_fresnel_adjoint,
        params={"wavelength_m": 532e-9, "distance_m": 0.01},
        input_shape=(256, 256),
        output_shape=(256, 256),
        name="fresnel_prop",
    )
    report = adapter.validate()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pwm_core.graph.adapters.tier_adapter import TierAdapter


class Tier1Adapter(TierAdapter):
    """Adapter for Tier-1 wave approximation primitives.

    Tier-1 primitives are typically linear and operate in the
    Fourier domain. Their adjoint is the conjugate transfer function.
    """

    _physics_tier = "tier1_approx"
    _physics_subrole = "propagation"

    def __init__(
        self,
        forward_kernel: Callable,
        adjoint_kernel: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        input_shape: Tuple[int, ...] = (64, 64),
        output_shape: Tuple[int, ...] = (64, 64),
        name: str = "tier1_approx",
        subrole: str = "propagation",
    ):
        super().__init__(
            forward_kernel=forward_kernel,
            adjoint_kernel=adjoint_kernel,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
            name=name,
            is_linear=True,
        )
        self._physics_subrole = subrole

    def check_parseval(
        self,
        n_trials: int = 5,
        tol: float = 0.01,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Check energy conservation (Parseval's theorem).

        For lossless wave propagation, ||A(x)||^2 should equal ||x||^2.
        """
        rng = np.random.default_rng(seed)
        ratios = []

        for _ in range(n_trials):
            x = rng.standard_normal(self._input_shape)
            Ax = self._forward_kernel(x, self._params)
            ratio = float(np.linalg.norm(Ax) / max(np.linalg.norm(x), 1e-30))
            ratios.append(ratio)

        max_deviation = max(abs(r - 1.0) for r in ratios)
        return {
            "passed": max_deviation < tol,
            "max_deviation_from_unity": max_deviation,
            "mean_ratio": float(np.mean(ratios)),
            "note": "For lossless propagation, energy ratio should be ~1.0",
        }


# ---------------------------------------------------------------------------
# Built-in Tier-1 kernels
# ---------------------------------------------------------------------------


def fresnel_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Fresnel propagation via angular spectrum method.

    Propagates a 2D complex field by distance z using the Fresnel
    approximation in the spatial frequency domain.
    """
    wavelength = params.get("wavelength_m", 532e-9)
    z = params.get("distance_m", 0.01)
    pixel_size = params.get("pixel_size_m", 1e-6)
    ny, nx = x.shape[-2], x.shape[-1]

    # Frequency coordinates
    fy = np.fft.fftfreq(ny, d=pixel_size)
    fx = np.fft.fftfreq(nx, d=pixel_size)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")

    # Fresnel transfer function
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    # Propagate
    X_fft = np.fft.fft2(x)
    Y_fft = X_fft * H
    return np.real(np.fft.ifft2(Y_fft))


def fresnel_adjoint(y: np.ndarray, params: dict) -> np.ndarray:
    """Adjoint of Fresnel propagation (conjugate transfer function)."""
    wavelength = params.get("wavelength_m", 532e-9)
    z = params.get("distance_m", 0.01)
    pixel_size = params.get("pixel_size_m", 1e-6)
    ny, nx = y.shape[-2], y.shape[-1]

    fy = np.fft.fftfreq(ny, d=pixel_size)
    fx = np.fft.fftfreq(nx, d=pixel_size)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")

    # Conjugate transfer function
    H_conj = np.exp(1j * np.pi * wavelength * z * (FX**2 + FY**2))

    Y_fft = np.fft.fft2(y)
    X_fft = Y_fft * H_conj
    return np.real(np.fft.ifft2(X_fft))


def otf_convolution_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Convolution with an Optical Transfer Function (OTF)."""
    sigma = params.get("sigma", 2.0)
    ny, nx = x.shape[-2], x.shape[-1]
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    OTF = np.exp(-2 * np.pi**2 * sigma**2 * (FX**2 + FY**2))

    X_fft = np.fft.fft2(x)
    return np.real(np.fft.ifft2(X_fft * OTF))


def otf_convolution_adjoint(y: np.ndarray, params: dict) -> np.ndarray:
    """Adjoint of OTF convolution (real-symmetric OTF is self-adjoint)."""
    return otf_convolution_forward(y, params)
