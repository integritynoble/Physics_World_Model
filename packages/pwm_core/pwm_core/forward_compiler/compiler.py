"""Compile a ForwardModel into an executable CompiledOperator (a BaseOperator).

forward = primitives applied left->right; adjoint = primitive adjoints applied
right->left (linear models only). The operator inherits BaseOperator.check_adjoint
(inner-product test) and serialize/metadata for free.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from pwm_core.forward_compiler.ir import ForwardModel, Stage
from pwm_core.forward_compiler.primitives import Primitive, get_primitive
from pwm_core.physics.base import BaseOperator


class CompiledOperator(BaseOperator):
    """A BaseOperator produced by compiling a ForwardModel."""

    def __init__(self, model: ForwardModel,
                 x_shape: Tuple[int, ...], y_shape: Tuple[int, ...],
                 stages_resolved: List[Tuple[Primitive, Dict[str, Any]]]) -> None:
        # BaseOperator is a dataclass; we set its attributes directly rather
        # than invoking its generated __init__.
        self.operator_id = model.name
        self.theta = {}
        self.model = model
        self._x_shape = tuple(x_shape)
        self._y_shape = tuple(y_shape)
        self._stages = stages_resolved
        self._is_linear = all(p.is_linear for p, _ in stages_resolved)
        self._supports_autodiff = self._is_linear

    def forward(self, x: np.ndarray) -> np.ndarray:
        cur = np.asarray(x)
        for prim, params in self._stages:
            cur = prim.forward(cur, **params)
        return cur

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        if not self._is_linear:
            raise ValueError(
                f"operator {self.operator_id!r} is non-linear; adjoint is undefined")
        cur = np.asarray(y)
        for prim, params in reversed(self._stages):
            if prim.adjoint is None:
                raise ValueError(f"primitive {prim.name!r} has no adjoint")
            cur = prim.adjoint(cur, **params)
        return cur


def compile_model(model: ForwardModel) -> CompiledOperator:
    """Resolve primitives, propagate shapes, inject derived params, return op."""
    shape: Tuple[int, ...] = tuple(model.x_shape)
    resolved: List[Tuple[Primitive, Dict[str, Any]]] = []
    for stage in model.stages:
        prim = get_primitive(stage.op)
        params = dict(stage.params)
        # band_sum's adjoint needs the band count from the *input* shape.
        if stage.op == "band_sum" and "n_bands" not in params:
            if len(shape) < 3:
                raise ValueError(
                    f"band_sum stage requires a (...,L) input, got shape {shape}")
            params["n_bands"] = int(shape[-1])
        out_shape = prim.out_shape(shape, **params)
        resolved.append((prim, params))
        shape = tuple(int(d) for d in out_shape)
    return CompiledOperator(model, x_shape=tuple(model.x_shape),
                            y_shape=shape, stages_resolved=resolved)


def as_torch(op: CompiledOperator):
    """Wrap a *linear* CompiledOperator as a differentiable torch callable.

    Returns a function f(x: Tensor) -> Tensor whose backward applies the
    operator's adjoint (the exact vjp for a linear map). Raises if the operator
    is non-linear.
    """
    if not op.is_linear:
        raise ValueError(
            f"as_torch requires a linear operator; {op.operator_id!r} is non-linear")
    import torch

    class _OpFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            dtype, device = x.dtype, x.device
            ctx._meta = (dtype, device)
            y = op.forward(x.detach().cpu().numpy())
            return torch.as_tensor(np.asarray(y), dtype=dtype, device=device)

        @staticmethod
        def backward(ctx, grad_out):
            dtype, device = ctx._meta
            gx = op.adjoint(grad_out.detach().cpu().numpy())
            return torch.as_tensor(np.asarray(gx), dtype=dtype, device=device)

    return _OpFn.apply
