"""Tier-0 Adapter: Geometry
===========================

Wraps geometric optics primitives: coordinate transforms, projection,
ray tracing, magnification, rotation.

Tier-0 primitives are the simplest and cheapest. They involve no wave
physics, just coordinate/spatial transformations.

Examples:
- Image rotation/flip
- Magnification/demagnification
- Coordinate transform (Cartesian -> polar)
- Geometric projection (parallel beam, fan beam)
- Ray transfer matrix operations

Usage:
    from pwm_core.graph.adapters import Tier0Adapter

    adapter = Tier0Adapter(
        forward_kernel=my_rotation_forward,
        adjoint_kernel=my_rotation_adjoint,
        params={"angle_deg": 45.0},
        input_shape=(256, 256),
        output_shape=(256, 256),
        name="rotation_45",
    )
    report = adapter.validate()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pwm_core.graph.adapters.tier_adapter import TierAdapter


class Tier0Adapter(TierAdapter):
    """Adapter for Tier-0 geometry primitives.

    Tier-0 primitives are always linear and deterministic.
    The adjoint of a geometry transform is typically its inverse
    (e.g., rotate by -angle, demagnify by 1/M).
    """

    _physics_tier = "tier0_geometry"
    _physics_subrole = "transport"

    def __init__(
        self,
        forward_kernel: Callable,
        adjoint_kernel: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        input_shape: Tuple[int, ...] = (64, 64),
        output_shape: Tuple[int, ...] = (64, 64),
        name: str = "tier0_geometry",
    ):
        super().__init__(
            forward_kernel=forward_kernel,
            adjoint_kernel=adjoint_kernel,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
            name=name,
            is_linear=True,  # Geometry ops are always linear
        )

    def check_invertibility(
        self,
        n_trials: int = 5,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Check if adjoint approximately inverts forward (geometry check).

        For pure geometry (rotation, flip), A^T A ~ I.
        """
        if self._adjoint_kernel is None:
            return {"passed": False, "error": "no adjoint provided"}

        rng = np.random.default_rng(seed)
        errors = []

        for _ in range(n_trials):
            x = rng.standard_normal(self._input_shape)
            Ax = self._forward_kernel(x, self._params)
            ATAx = self._adjoint_kernel(np.asarray(Ax), self._params)
            ATAx = np.asarray(ATAx)

            denom = max(np.linalg.norm(x), 1e-30)
            rel_err = float(np.linalg.norm(ATAx - x) / denom)
            errors.append(rel_err)

        max_err = max(errors)
        return {
            "passed": max_err < tol,
            "max_reconstruction_error": max_err,
            "note": "A^T A ~ I check (geometry should be near-invertible)",
        }


# ---------------------------------------------------------------------------
# Built-in geometry kernels
# ---------------------------------------------------------------------------


def flip_horizontal_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Flip image horizontally."""
    return np.fliplr(x)


def flip_horizontal_adjoint(y: np.ndarray, params: dict) -> np.ndarray:
    """Adjoint of horizontal flip = horizontal flip (self-adjoint)."""
    return np.fliplr(y)


def magnify_forward(x: np.ndarray, params: dict) -> np.ndarray:
    """Uniform magnification by factor M using interpolation."""
    from scipy.ndimage import zoom
    M = params.get("magnification", 2.0)
    return zoom(x, M, order=1)


def magnify_adjoint(y: np.ndarray, params: dict) -> np.ndarray:
    """Adjoint of magnification (demagnify)."""
    from scipy.ndimage import zoom
    M = params.get("magnification", 2.0)
    return zoom(y, 1.0 / M, order=1)
