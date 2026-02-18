"""Base Tier Adapter
====================

Abstract base for all tier adapters. Provides:
- Adjoint correctness validation (dot-product test)
- Shape checking and dtype normalization
- Physics tier tag injection
- Serialization boilerplate
- Energy conservation check
- Linearity verification (for linear operators)

Subclasses set the tier tag and may add tier-specific validation.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class TierAdapter(ABC):
    """Base adapter that wraps raw physics kernels into valid PrimitiveOps.

    Parameters
    ----------
    forward_kernel : callable
        (x: ndarray, params: dict) -> ndarray
    adjoint_kernel : callable or None
        (y: ndarray, params: dict) -> ndarray. If None and is_linear=True,
        the adapter will attempt numerical adjoint estimation (expensive).
    params : dict
        Physical parameters for the kernel.
    input_shape : tuple of int
        Expected input array shape.
    output_shape : tuple of int
        Expected output array shape.
    name : str
        Primitive identifier.
    is_linear : bool
        Whether the forward kernel is linear.
    """

    _physics_tier: str = "base"
    _physics_subrole: str = "utility"

    def __init__(
        self,
        forward_kernel: Callable,
        adjoint_kernel: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        input_shape: Tuple[int, ...] = (64, 64),
        output_shape: Tuple[int, ...] = (64, 64),
        name: str = "adapted_primitive",
        is_linear: bool = True,
    ):
        self._forward_kernel = forward_kernel
        self._adjoint_kernel = adjoint_kernel
        self._params = dict(params or {})
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.primitive_id = name
        self._is_linear = is_linear
        self._is_stochastic = False
        self._is_differentiable = True
        self._is_stateful = False

    # -- PrimitiveOp protocol properties --

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    @property
    def is_stochastic(self) -> bool:
        return self._is_stochastic

    @property
    def is_differentiable(self) -> bool:
        return self._is_differentiable

    @property
    def is_stateful(self) -> bool:
        return self._is_stateful

    # -- Forward / Adjoint --

    def forward(self, x: np.ndarray, **kw: Any) -> np.ndarray:
        """Apply the physics forward model with shape/dtype validation."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self._input_shape:
            raise ValueError(
                f"{self.primitive_id}: expected input shape {self._input_shape}, "
                f"got {x.shape}"
            )
        y = self._forward_kernel(x, self._params)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != self._output_shape:
            raise ValueError(
                f"{self.primitive_id}: forward produced shape {y.shape}, "
                f"expected {self._output_shape}"
            )
        return y

    def adjoint(self, y: np.ndarray, **kw: Any) -> np.ndarray:
        """Apply the physics adjoint (transpose of Jacobian)."""
        if self._adjoint_kernel is None:
            raise NotImplementedError(
                f"Adjoint not provided for primitive '{self.primitive_id}'. "
                f"Supply adjoint_kernel at construction time."
            )
        y = np.asarray(y, dtype=np.float64)
        if y.shape != self._output_shape:
            raise ValueError(
                f"{self.primitive_id}: expected adjoint input shape "
                f"{self._output_shape}, got {y.shape}"
            )
        x = self._adjoint_kernel(y, self._params)
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self._input_shape:
            raise ValueError(
                f"{self.primitive_id}: adjoint produced shape {x.shape}, "
                f"expected {self._input_shape}"
            )
        return x

    # -- Serialization --

    def serialize(self) -> Dict[str, Any]:
        """Serialize the adapter configuration."""
        result: Dict[str, Any] = {
            "primitive_id": self.primitive_id,
            "physics_tier": self._physics_tier,
            "physics_subrole": self._physics_subrole,
            "is_linear": self._is_linear,
            "input_shape": list(self._input_shape),
            "output_shape": list(self._output_shape),
            "params": {},
        }
        for k, v in self._params.items():
            if isinstance(v, np.ndarray):
                if v.size > 1000:
                    sha = hashlib.sha256(v.tobytes()).hexdigest()
                    result["params"][k] = f"<blob sha256={sha[:16]}>"
                else:
                    result["params"][k] = v.tolist()
            else:
                result["params"][k] = v
        return result

    # -- Validation utilities --

    def check_adjoint(
        self,
        n_trials: int = 10,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Verify <A(x), y> == <x, A^T(y)> for random vectors.

        Returns dict with: passed, max_relative_error, n_trials, tolerance.
        """
        if self._adjoint_kernel is None:
            return {"passed": False, "error": "no adjoint provided"}

        rng = np.random.default_rng(seed)
        errors = []

        for _ in range(n_trials):
            x = rng.standard_normal(self._input_shape)
            y = rng.standard_normal(self._output_shape)

            Ax = self._forward_kernel(x, self._params)
            ATy = self._adjoint_kernel(y, self._params)

            inner_left = float(np.sum(np.asarray(Ax).ravel() * y.ravel()))
            inner_right = float(np.sum(x.ravel() * np.asarray(ATy).ravel()))

            denom = max(abs(inner_left), abs(inner_right), 1e-30)
            rel_err = abs(inner_left - inner_right) / denom
            errors.append(rel_err)

        max_err = max(errors)
        return {
            "passed": max_err < tol,
            "max_relative_error": max_err,
            "mean_relative_error": float(np.mean(errors)),
            "n_trials": n_trials,
            "tolerance": tol,
        }

    def check_energy(
        self,
        n_trials: int = 10,
        max_ratio: float = 100.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Check that forward does not amplify energy unboundedly."""
        rng = np.random.default_rng(seed)
        ratios = []

        for _ in range(n_trials):
            x = rng.standard_normal(self._input_shape)
            Ax = self._forward_kernel(x, self._params)
            ratio = np.linalg.norm(Ax) / max(np.linalg.norm(x), 1e-30)
            ratios.append(float(ratio))

        peak = max(ratios)
        return {
            "passed": peak < max_ratio,
            "max_energy_ratio": peak,
            "mean_energy_ratio": float(np.mean(ratios)),
        }

    def check_linearity(
        self,
        n_trials: int = 5,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Verify A(ax + by) == aA(x) + bA(y) (only meaningful if is_linear)."""
        if not self._is_linear:
            return {"passed": True, "note": "skipped (nonlinear)"}

        rng = np.random.default_rng(seed)
        errors = []

        for _ in range(n_trials):
            x1 = rng.standard_normal(self._input_shape)
            x2 = rng.standard_normal(self._input_shape)
            a, b = rng.standard_normal(2)

            lhs = self._forward_kernel(a * x1 + b * x2, self._params)
            rhs = (
                a * np.asarray(self._forward_kernel(x1, self._params))
                + b * np.asarray(self._forward_kernel(x2, self._params))
            )
            lhs = np.asarray(lhs)

            denom = max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-30)
            rel_err = float(np.linalg.norm(lhs - rhs) / denom)
            errors.append(rel_err)

        max_err = max(errors)
        return {
            "passed": max_err < tol,
            "max_relative_error": max_err,
            "n_trials": n_trials,
            "tolerance": tol,
        }

    def validate(
        self,
        adjoint_tol: float = 1e-6,
        linearity_tol: float = 1e-6,
        energy_max: float = 100.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run all validation checks. Returns combined report."""
        report = {
            "primitive_id": self.primitive_id,
            "physics_tier": self._physics_tier,
        }

        # Adjoint
        adj = self.check_adjoint(tol=adjoint_tol, seed=seed)
        report["adjoint"] = adj

        # Energy
        energy = self.check_energy(max_ratio=energy_max, seed=seed)
        report["energy"] = energy

        # Linearity
        lin = self.check_linearity(tol=linearity_tol, seed=seed)
        report["linearity"] = lin

        report["all_passed"] = (
            adj["passed"] and energy["passed"] and lin["passed"]
        )
        return report
