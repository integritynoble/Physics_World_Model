# packages/pwm_core/tests/test_fc_torch.py
"""as_torch: a linear CompiledOperator is differentiable (backward = adjoint)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pwm_core.forward_compiler import ForwardModel, Stage, compile_model
from pwm_core.forward_compiler.compiler import as_torch


def _scale_mask_model(H=4, W=4, L=2, c=3.0):
    mask = np.random.default_rng(2).random((H, W))
    return ForwardModel(
        name="lin_demo",
        x_shape=(H, W, L),
        stages=[
            Stage(op="scale", params={"c": c}),
            Stage(op="mask_multiply", params={"mask": mask}),
            Stage(op="band_sum", params={}),
        ],
    )


def test_as_torch_forward_matches_numpy():
    op = compile_model(_scale_mask_model())
    fn = as_torch(op)
    x = np.random.default_rng(0).standard_normal((4, 4, 2))
    y_np = op.forward(x)
    y_t = fn(torch.tensor(x, dtype=torch.float64, requires_grad=True))
    assert np.allclose(y_t.detach().numpy(), y_np, atol=1e-6)


def test_as_torch_grad_equals_adjoint():
    op = compile_model(_scale_mask_model())
    fn = as_torch(op)
    x = torch.tensor(np.random.default_rng(0).standard_normal((4, 4, 2)),
                     dtype=torch.float64, requires_grad=True)
    y = fn(x)
    y.sum().backward()              # d/dx sum(A x) = A^T(ones)
    expected = op.adjoint(np.ones(op.y_shape))
    assert np.allclose(x.grad.numpy(), expected, atol=1e-6)


def test_as_torch_rejects_nonlinear():
    m = ForwardModel(name="nl", x_shape=(4,),
                     stages=[Stage(op="square_magnitude", params={})])
    op = compile_model(m)
    with pytest.raises(ValueError, match="linear"):
        as_torch(op)
