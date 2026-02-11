"""CorrectedOperator -- wraps a base operator with learnable correction.

Implements D3: operator-correction wraps A, not signals.

Two correction families:
  PrePostCorrection:  A'(x) = P(alpha) . A( Q(beta)(x) )
  LowRankCorrection:  A'(x) = A(x) + U . diag(alpha) . V^T . x
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.graph.ir_types import ParameterSpec


class OperatorCorrection(ABC):
    """Base class for operator-level corrections."""

    @abstractmethod
    def apply_forward(self, base_op: Any, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def apply_adjoint(self, base_op: Any, y: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def learnable_params(self) -> Dict[str, ParameterSpec]:
        ...

    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def set_params(self, params: Dict[str, float]) -> None:
        ...


class PrePostCorrection(OperatorCorrection):
    """A'(x) = post_scale * A(pre_scale * x + pre_shift) + post_shift

    Simplified version of P(alpha) . A( Q(beta)(x) ).
    pre_scale, pre_shift = pre-correction on x before A
    post_scale, post_shift = post-correction on A(x) after A
    """

    def __init__(
        self,
        pre_scale: float = 1.0,
        pre_shift: float = 0.0,
        post_scale: float = 1.0,
        post_shift: float = 0.0,
    ):
        self._pre_scale = pre_scale
        self._pre_shift = pre_shift
        self._post_scale = post_scale
        self._post_shift = post_shift

    def apply_forward(self, base_op: Any, x: np.ndarray) -> np.ndarray:
        x_corrected = self._pre_scale * x + self._pre_shift
        Ax = base_op.forward(x_corrected)
        return self._post_scale * Ax + self._post_shift

    def apply_adjoint(self, base_op: Any, y: np.ndarray) -> np.ndarray:
        y_corrected = self._post_scale * y
        ATy = base_op.adjoint(y_corrected)
        return self._pre_scale * ATy

    def learnable_params(self) -> Dict[str, ParameterSpec]:
        return {
            "pre_scale": ParameterSpec(name="pre_scale", lower=0.5, upper=2.0),
            "pre_shift": ParameterSpec(name="pre_shift", lower=-1.0, upper=1.0),
            "post_scale": ParameterSpec(name="post_scale", lower=0.5, upper=2.0),
            "post_shift": ParameterSpec(name="post_shift", lower=-1.0, upper=1.0),
        }

    def get_params(self) -> Dict[str, float]:
        return {
            "pre_scale": self._pre_scale,
            "pre_shift": self._pre_shift,
            "post_scale": self._post_scale,
            "post_shift": self._post_shift,
        }

    def set_params(self, params: Dict[str, float]) -> None:
        if "pre_scale" in params:
            self._pre_scale = params["pre_scale"]
        if "pre_shift" in params:
            self._pre_shift = params["pre_shift"]
        if "post_scale" in params:
            self._post_scale = params["post_scale"]
        if "post_shift" in params:
            self._post_shift = params["post_shift"]


class LowRankCorrection(OperatorCorrection):
    """A'(x) = A(x) + U @ diag(alpha) @ V^T @ x

    U: (M, rank), V: (N, rank), alphas: (rank,)
    """

    def __init__(
        self,
        U: np.ndarray,
        V: np.ndarray,
        alphas: Optional[np.ndarray] = None,
    ):
        self._U = np.asarray(U, dtype=np.float64)
        self._V = np.asarray(V, dtype=np.float64)
        rank = self._U.shape[1]
        self._alphas = np.ones(rank) if alphas is None else np.asarray(alphas, dtype=np.float64)
        assert self._U.shape[1] == self._V.shape[1] == len(self._alphas)

    def apply_forward(self, base_op: Any, x: np.ndarray) -> np.ndarray:
        x_flat = x.ravel().astype(np.float64)
        Ax = base_op.forward(x)
        Ax_flat = Ax.ravel().astype(np.float64)
        Vx = self._V.T @ x_flat
        delta = self._U @ (self._alphas * Vx)
        return (Ax_flat + delta).reshape(Ax.shape)

    def apply_adjoint(self, base_op: Any, y: np.ndarray) -> np.ndarray:
        y_flat = y.ravel().astype(np.float64)
        ATy = base_op.adjoint(y)
        ATy_flat = ATy.ravel().astype(np.float64)
        Uy = self._U.T @ y_flat
        delta = self._V @ (self._alphas * Uy)
        return (ATy_flat + delta).reshape(ATy.shape)

    def learnable_params(self) -> Dict[str, ParameterSpec]:
        params = {}
        for i in range(len(self._alphas)):
            params[f"alpha_{i}"] = ParameterSpec(
                name=f"alpha_{i}", lower=-10.0, upper=10.0
            )
        return params

    def get_params(self) -> Dict[str, float]:
        return {f"alpha_{i}": float(a) for i, a in enumerate(self._alphas)}

    def set_params(self, params: Dict[str, float]) -> None:
        for i in range(len(self._alphas)):
            key = f"alpha_{i}"
            if key in params:
                self._alphas[i] = params[key]


class CorrectedOperator:
    """Wraps base_op with correction. Satisfies PhysicsOperator protocol.

    Usage:
        A_corrected = CorrectedOperator(A_base, PrePostCorrection(pre_scale=1.1))
        y = A_corrected.forward(x)
    """

    def __init__(self, base_op: Any, correction: OperatorCorrection):
        self.base_op = base_op
        self.correction = correction

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.correction.apply_forward(self.base_op, x)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self.correction.apply_adjoint(self.base_op, y)

    @property
    def x_shape(self):
        return getattr(self.base_op, 'x_shape', None)

    @property
    def y_shape(self):
        return getattr(self.base_op, 'y_shape', None)
