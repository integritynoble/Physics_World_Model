# packages/pwm_core/tests/test_fc_compiler.py
"""compile_model: composition, shape inference, adjoint, linearity flags."""
from __future__ import annotations

import numpy as np
import pytest

from pwm_core.forward_compiler import (
    ForwardModel, Stage, compile_model, from_modality, NativeCompiledOperator,
)
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


# ── native_operator stage: wrap any pwm_core.physics operator ──────────────────

_NATIVE_MODALITIES = [
    ("mri",          dict(H=16, W=16)),
    ("ct",           dict(H=16, W=16, n_angles=18)),
    ("lensless",     dict(H=16, W=16)),
    ("holography",   dict(H=16, W=16)),
    ("ptychography", dict(H=16, W=16, probe_size=4, n_positions=3)),
    ("fluorescence", dict(H=16, W=16)),
    ("lightsheet",   dict(H=8,  W=8,  D=4)),
    ("ultrasound",   dict(H=16, W=16)),
    ("photoacoustic",dict(H=16, W=16)),
]


@pytest.mark.parametrize("modality,kw", _NATIVE_MODALITIES)
def test_native_modality_forward_runs(modality, kw):
    op = compile_model(from_modality(modality, **kw))
    assert isinstance(op, NativeCompiledOperator)
    x = np.random.default_rng(0).standard_normal(op.x_shape).astype(np.float64)
    y = op.forward(x)
    assert np.all(np.isfinite(y))
    # y_shape is derived from the real forward output, not the operator's
    # (possibly stale) declared y_shape — must match the actual array.
    assert tuple(op.y_shape) == tuple(np.asarray(y).shape)


@pytest.mark.parametrize("modality,kw", _NATIVE_MODALITIES)
def test_native_modality_validate_ok(modality, kw):
    # Native operators ship reconstruction-style adjoints (not exact transposes);
    # validate must treat the adjoint check as ADVISORY → ok stays True with at
    # most a warning, never a crash from a wrong-shaped probe.
    from pwm_core.forward_compiler import validate_forward_model
    report = validate_forward_model(from_modality(modality, **kw))
    assert report.ok is True, report.summary()


def test_native_operator_only_as_sole_stage():
    m = ForwardModel(
        name="bad_native",
        x_shape=(8, 8),
        stages=[
            Stage(op="scale", params={"c": 2.0}),
            Stage(op="native_operator",
                  params={"class": "pwm_core.physics.mri.mri_operator.MRIOperator"}),
        ],
    )
    with pytest.raises(ValueError, match="sole stage"):
        compile_model(m)


def test_from_modality_unknown_raises():
    with pytest.raises(ValueError, match="unknown modality"):
        from_modality("xray_banana", H=8, W=8)
