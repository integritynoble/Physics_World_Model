# packages/pwm_core/tests/test_fc_compiler.py
"""compile_model: composition, shape inference, adjoint, linearity flags."""
from __future__ import annotations

import numpy as np
import pytest

from pwm_core.forward_compiler import ForwardModel, Stage, compile_model
from pwm_core.physics.base import BaseOperator


def _cassi_model(H=8, W=8, L=4):
    mask = np.random.default_rng(1).integers(0, 2, size=(H, W)).astype(np.float64)
    disp = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0], "disp_poly_y": [0.0, 0.0]}
    return ForwardModel(
        name="cassi_demo",
        x_shape=(H, W, L),
        stages=[
            Stage(op="band_shift", params={"dispersion": disp}),
            Stage(op="mask_multiply", params={"mask": mask}),
            Stage(op="band_sum", params={}),
        ],
        metadata={"modality": "cassi"},
    )


def test_compiled_operator_is_physics_operator():
    op = compile_model(_cassi_model())
    assert isinstance(op, BaseOperator)
    assert op.x_shape == (8, 8, 4)
    assert op.y_shape == (8, 8)
    assert op.is_linear is True
    assert op.supports_autodiff is True


def test_compiled_forward_shapes():
    op = compile_model(_cassi_model())
    x = np.random.default_rng(0).standard_normal((8, 8, 4))
    y = op.forward(x)
    assert y.shape == (8, 8)


def test_compiled_operator_passes_builtin_adjoint_check():
    op = compile_model(_cassi_model())
    report = op.check_adjoint(n_trials=3, tol=1e-4)
    assert report.passed, report.summary()


def test_nonlinear_model_blocks_adjoint():
    m = ForwardModel(
        name="intensity",
        x_shape=(4, 4),
        stages=[Stage(op="square_magnitude", params={})],
    )
    op = compile_model(m)
    assert op.is_linear is False
    assert op.supports_autodiff is False
    with pytest.raises(ValueError, match="non-linear"):
        op.adjoint(np.ones((4, 4)))


def test_band_sum_n_bands_injected():
    # band_sum adjoint needs n_bands; compiler must inject it from inferred shape.
    op = compile_model(_cassi_model(L=5))
    back = op.adjoint(np.ones((8, 8)))
    assert back.shape == (8, 8, 5)
