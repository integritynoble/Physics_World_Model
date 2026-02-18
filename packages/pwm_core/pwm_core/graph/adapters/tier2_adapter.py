"""Tier-2 Adapter: Full Transport
==================================

Wraps full-physics simulation kernels: Maxwell solvers, FDTD, BPM,
Monte Carlo photon transport, Mie scattering, radiative transfer.

Tier-2 primitives are the most physically accurate non-learned models.
They are typically expensive to evaluate and may not have analytical
adjoints (numerical adjoint via AD or finite differences may be needed).

Examples:
- FDTD electromagnetic simulation
- Beam Propagation Method (BPM)
- Monte Carlo photon transport
- Mie scattering (exact solution)
- Born series (beyond first order)

Usage:
    from pwm_core.graph.adapters import Tier2Adapter

    adapter = Tier2Adapter(
        forward_kernel=my_fdtd_simulation,
        adjoint_kernel=my_fdtd_adjoint,
        params={"wavelength_m": 532e-9, "n_medium": 1.33},
        input_shape=(128, 128),
        output_shape=(128, 128),
        name="fdtd_scatter",
    )
    report = adapter.validate()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pwm_core.graph.adapters.tier_adapter import TierAdapter


class Tier2Adapter(TierAdapter):
    """Adapter for Tier-2 full transport primitives.

    Tier-2 primitives may be linear or nonlinear. The adapter
    provides extra validation for physically meaningful outputs:
    - Positivity check (for intensity-domain outputs)
    - Reciprocity check (for linear optics)
    - Compute cost estimation
    """

    _physics_tier = "tier2_full"
    _physics_subrole = "interaction"

    def __init__(
        self,
        forward_kernel: Callable,
        adjoint_kernel: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        input_shape: Tuple[int, ...] = (64, 64),
        output_shape: Tuple[int, ...] = (64, 64),
        name: str = "tier2_full",
        is_linear: bool = True,
        subrole: str = "interaction",
    ):
        super().__init__(
            forward_kernel=forward_kernel,
            adjoint_kernel=adjoint_kernel,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
            name=name,
            is_linear=is_linear,
        )
        self._physics_subrole = subrole

    def check_positivity(
        self,
        n_trials: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Check if output is non-negative for non-negative input.

        Relevant for intensity-domain operators (optical power, photon counts).
        """
        rng = np.random.default_rng(seed)
        violations = 0

        for _ in range(n_trials):
            x = np.abs(rng.standard_normal(self._input_shape))
            Ax = self._forward_kernel(x, self._params)
            if np.any(np.asarray(Ax) < -1e-10):
                violations += 1

        return {
            "passed": violations == 0,
            "n_violations": violations,
            "n_trials": n_trials,
            "note": "Non-negative input should produce non-negative output "
                    "for intensity-domain operators",
        }

    def check_reciprocity(
        self,
        n_trials: int = 5,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Check Lorentz reciprocity: H(r1, r2) == H(r2, r1).

        For linear optical systems, the impulse response should be
        symmetric under source/detector exchange. This is tested
        via the adjoint: <Ax, y> == <x, A^T y>.
        """
        # This is equivalent to the adjoint check for linear operators
        if not self._is_linear:
            return {"passed": True, "note": "skipped (nonlinear)"}
        return self.check_adjoint(n_trials=n_trials, tol=tol, seed=seed)

    def estimate_compute_cost(
        self,
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """Estimate wall-clock cost of a single forward evaluation."""
        import time
        rng = np.random.default_rng(0)
        x = rng.standard_normal(self._input_shape)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self._forward_kernel(x, self._params)
            times.append(time.perf_counter() - t0)

        return {
            "mean_seconds": float(np.mean(times)),
            "std_seconds": float(np.std(times)),
            "n_runs": n_runs,
            "input_shape": list(self._input_shape),
        }


# ---------------------------------------------------------------------------
# Built-in Tier-2 kernels
# ---------------------------------------------------------------------------


def mie_scatter_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Simplified Mie scattering (placeholder for full implementation).

    In a real implementation, this would compute the exact Mie solution
    for a distribution of spheres.
    """
    n_sphere = params.get("n_sphere", 1.5)
    n_medium = params.get("n_medium", 1.0)
    wavelength = params.get("wavelength_m", 532e-9)

    # Simplified: phase shift proportional to refractive index contrast
    contrast = (n_sphere - n_medium) / n_medium
    phase = 2 * np.pi * contrast / wavelength * 1e-6
    # Apply as a multiplicative modulation
    return x * np.cos(phase * np.abs(x))


def mie_scatter_adjoint(y: np.ndarray, params: dict) -> np.ndarray:
    """Adjoint of simplified Mie scattering."""
    n_sphere = params.get("n_sphere", 1.5)
    n_medium = params.get("n_medium", 1.0)
    wavelength = params.get("wavelength_m", 532e-9)

    contrast = (n_sphere - n_medium) / n_medium
    phase = 2 * np.pi * contrast / wavelength * 1e-6
    return y * np.cos(phase * np.abs(y))
