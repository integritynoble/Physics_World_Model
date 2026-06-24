"""Compile a ForwardModel into an executable CompiledOperator (a BaseOperator).

forward = primitives applied left->right; adjoint = primitive adjoints applied
right->left (linear models only). The operator inherits BaseOperator.check_adjoint
(inner-product test) and serialize/metadata for free.

Special stage op "native_operator": delegates forward/adjoint to any
pwm_core.physics BaseOperator subclass. Params:
  class   — importable dotted path, e.g. "pwm_core.physics.mri.mri_operator.MRIOperator"
  kwargs  — passed to the constructor (list values are converted to tuples)
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.forward_compiler.ir import ForwardModel, Stage
from pwm_core.forward_compiler.primitives import Primitive, get_primitive
from pwm_core.physics.base import BaseOperator

_NATIVE_OP = "native_operator"


class CompiledOperator(BaseOperator):
    """A BaseOperator produced by compiling a ForwardModel."""

    def __init__(self, model: ForwardModel,
                 x_shape: Tuple[int, ...], y_shape: Tuple[int, ...],
                 stages_resolved: List[Tuple[Primitive, Dict[str, Any]]]) -> None:
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


class NativeCompiledOperator(BaseOperator):
    """Wraps a pwm_core.physics BaseOperator as a CompiledOperator.

    Produced when a ForwardModel has a single 'native_operator' stage.
    Exposes the same interface as CompiledOperator so the harness tools
    (fm_simulate, fm_validate) work unchanged.
    """

    def __init__(self, model: ForwardModel, native_op: BaseOperator) -> None:
        self.operator_id = model.name
        self.theta = {}
        self.model = model
        self._native = native_op
        self._x_shape = tuple(native_op.x_shape)
        self._y_shape = tuple(native_op.y_shape)
        self._is_linear = True
        self._supports_autodiff = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._native.forward(np.asarray(x))

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self._native.adjoint(np.asarray(y))


def _instantiate_native(stage: Stage) -> BaseOperator:
    cls_path = stage.params.get("class", "")
    if not cls_path:
        raise ValueError("native_operator stage requires a 'class' param")
    module_path, cls_name = cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    kwargs = {}
    for k, v in stage.params.get("kwargs", {}).items():
        kwargs[k] = tuple(v) if isinstance(v, list) else v
    return cls(**kwargs)


def compile_model(model: ForwardModel) -> "CompiledOperator | NativeCompiledOperator":
    """Resolve primitives, propagate shapes, inject derived params, return op.

    If the model contains a single native_operator stage, returns a
    NativeCompiledOperator wrapping the physics operator directly.
    """
    # Native-operator shortcut: single stage that wraps a BaseOperator subclass.
    if len(model.stages) == 1 and model.stages[0].op == _NATIVE_OP:
        native = _instantiate_native(model.stages[0])
        return NativeCompiledOperator(model, native)

    shape: Tuple[int, ...] = tuple(model.x_shape)
    resolved: List[Tuple[Primitive, Dict[str, Any]]] = []
    for stage in model.stages:
        if stage.op == _NATIVE_OP:
            raise ValueError(
                "native_operator may only appear as the sole stage in a ForwardModel")
        prim = get_primitive(stage.op)
        params = dict(stage.params)
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
