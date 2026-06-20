# packages/pwm_core/tests/test_fc_validate.py
"""Validators: dimensions, linearity classification, conditioning, report."""
from __future__ import annotations

import numpy as np
import pytest

from pwm_core.forward_compiler import ForwardModel, Stage, compile_model
from pwm_core.forward_compiler.validate import (
    validate_dimensions, classify_linearity, probe_conditioning,
    validate_forward_model, ForwardModelReport,
)


def _cassi_model(H=8, W=8, L=4):
    mask = np.random.default_rng(1).integers(0, 2, size=(H, W)).astype(np.float64)
    disp = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0], "disp_poly_y": [0.0, 0.0]}
    return ForwardModel(
        name="cassi_demo", x_shape=(H, W, L),
        stages=[Stage(op="band_shift", params={"dispersion": disp}),
                Stage(op="mask_multiply", params={"mask": mask}),
                Stage(op="band_sum", params={})],
        metadata={"modality": "cassi"})


def test_validate_dimensions_ok():
    ok, msg, y_shape = validate_dimensions(_cassi_model())
    assert ok, msg
    assert y_shape == (8, 8)


def test_validate_dimensions_catches_bad_pipeline():
    # band_sum on a 2-D input is invalid (needs a band axis)
    bad = ForwardModel(name="bad", x_shape=(8, 8),
                       stages=[Stage(op="band_sum", params={})])
    ok, msg, _ = validate_dimensions(bad)
    assert not ok
    assert "band_sum" in msg


def test_classify_linearity_linear_op():
    op = compile_model(_cassi_model())
    res = classify_linearity(op)
    assert res["is_linear"] is True
    assert res["max_residual"] < 1e-6


def test_classify_linearity_nonlinear_op():
    m = ForwardModel(name="nl", x_shape=(4, 4),
                     stages=[Stage(op="square_magnitude", params={})])
    op = compile_model(m)
    res = classify_linearity(op)
    assert res["is_linear"] is False
    assert res["max_residual"] > 1e-3


def test_probe_conditioning_returns_spectral_norm():
    op = compile_model(_cassi_model())
    res = probe_conditioning(op, n_iter=30)
    assert res["spectral_norm"] > 0.0
    assert 0.0 <= res["energy_ratio"] <= 2.0


def test_validate_forward_model_report():
    rep = validate_forward_model(_cassi_model())
    assert isinstance(rep, ForwardModelReport)
    assert rep.ok is True
    assert rep.is_linear is True
    assert rep.adjoint is not None and rep.adjoint.passed
    assert rep.y_shape == (8, 8)
    assert "spectral_norm" in rep.conditioning
    s = rep.summary()
    assert "cassi_demo" in s


def test_validate_forward_model_nonlinear_skips_adjoint():
    m = ForwardModel(name="nl", x_shape=(4, 4),
                     stages=[Stage(op="square_magnitude", params={})])
    rep = validate_forward_model(m)
    assert rep.is_linear is False
    assert rep.adjoint is None
    assert any("non-linear" in w for w in rep.warnings)
