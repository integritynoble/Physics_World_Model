"""Validators for compiled forward models.

Produces a ForwardModelReport: dimension check, adjoint dot-product test
(linear only), linearity classification (superposition probe), and a
conditioning probe (power-iteration spectral norm + energy ratio).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.forward_compiler.ir import ForwardModel
from pwm_core.forward_compiler.compiler import CompiledOperator, compile_model
from pwm_core.physics.base import AdjointCheckReport


@dataclass
class ForwardModelReport:
    name: str
    x_shape: Tuple[int, ...]
    y_shape: Tuple[int, ...]
    is_linear: bool
    adjoint: Optional[AdjointCheckReport]
    linearity_probe: Dict[str, Any]
    conditioning: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    ok: bool = True

    def summary(self) -> str:
        adj = self.adjoint.summary() if self.adjoint else "adjoint=N/A (non-linear)"
        sn = self.conditioning.get("spectral_norm")
        sn_str = f"{sn:.3g}" if isinstance(sn, (int, float)) else "N/A"
        return (f"ForwardModelReport[{self.name}] ok={self.ok} linear={self.is_linear} "
                f"x{self.x_shape}->y{self.y_shape} | {adj} | ||A||~{sn_str}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "x_shape": list(self.x_shape),
            "y_shape": list(self.y_shape),
            "is_linear": self.is_linear,
            "adjoint": (None if self.adjoint is None else {
                "passed": self.adjoint.passed,
                "max_relative_error": self.adjoint.max_relative_error,
                "tolerance": self.adjoint.tolerance,
            }),
            "linearity_probe": self.linearity_probe,
            "conditioning": self.conditioning,
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


def validate_dimensions(model: ForwardModel) -> Tuple[bool, str, Optional[Tuple[int, ...]]]:
    """Compile (which propagates shapes) and report success/failure."""
    try:
        op = compile_model(model)
        return True, "ok", op.y_shape
    except Exception as exc:  # noqa: BLE001 - report any shape/primitive error
        return False, f"{type(exc).__name__}: {exc}", None


def classify_linearity(op: CompiledOperator, seed: int = 0,
                       n_trials: int = 3, tol: float = 1e-6) -> Dict[str, Any]:
    """Probe A(a*x + b*z) ?= a*A(x) + b*A(z)."""
    rng = np.random.default_rng(seed)
    max_res = 0.0
    for _ in range(n_trials):
        x = rng.standard_normal(op.x_shape)
        z = rng.standard_normal(op.x_shape)
        a, b = float(rng.standard_normal()), float(rng.standard_normal())
        lhs = op.forward(a * x + b * z)
        rhs = a * op.forward(x) + b * op.forward(z)
        denom = max(float(np.linalg.norm(rhs.ravel())), 1e-30)
        res = float(np.linalg.norm((lhs - rhs).ravel())) / denom
        max_res = max(max_res, res)
    return {"is_linear": bool(max_res < tol), "max_residual": max_res}


def probe_conditioning(op: CompiledOperator, n_iter: int = 30,
                       seed: int = 0) -> Dict[str, Any]:
    """Power-iteration estimate of the spectral norm ||A|| and an energy ratio.

    energy_ratio = ||A x|| / ||x|| for a random x (a coarse well-posedness
    signal). Requires a linear operator (uses adjoint)."""
    if not op.is_linear:
        return {"spectral_norm": None, "energy_ratio": None,
                "note": "conditioning probe requires a linear operator"}
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(op.x_shape)
    x = x / max(float(np.linalg.norm(x.ravel())), 1e-30)
    sigma = 0.0
    for _ in range(n_iter):
        Ax = op.forward(x)
        ATAx = op.adjoint(Ax)
        nrm = float(np.linalg.norm(ATAx.ravel()))
        if nrm < 1e-30:
            break
        x = ATAx / nrm
        sigma = np.sqrt(nrm)
    x0 = rng.standard_normal(op.x_shape)
    energy_ratio = (float(np.linalg.norm(op.forward(x0).ravel())) /
                    max(float(np.linalg.norm(x0.ravel())), 1e-30))
    return {"spectral_norm": float(sigma), "energy_ratio": float(energy_ratio)}


def validate_forward_model(model: ForwardModel) -> ForwardModelReport:
    warnings: List[str] = []
    ok, msg, y_shape = validate_dimensions(model)
    if not ok:
        return ForwardModelReport(
            name=model.name, x_shape=tuple(model.x_shape), y_shape=(),
            is_linear=False, adjoint=None, linearity_probe={},
            conditioning={}, warnings=[f"dimension error: {msg}"], ok=False)

    op = compile_model(model)
    lin = classify_linearity(op)
    cond = probe_conditioning(op)

    adjoint_report = None
    if op.is_linear:
        adjoint_report = op.check_adjoint(n_trials=3, tol=1e-4)
        if not adjoint_report.passed:
            warnings.append(f"adjoint check failed: {adjoint_report.summary()}")
    else:
        warnings.append("operator is non-linear; adjoint test skipped")

    overall_ok = ok and (adjoint_report is None or adjoint_report.passed)
    return ForwardModelReport(
        name=model.name, x_shape=tuple(op.x_shape), y_shape=tuple(op.y_shape),
        is_linear=op.is_linear, adjoint=adjoint_report, linearity_probe=lin,
        conditioning=cond, warnings=warnings, ok=overall_ok)
