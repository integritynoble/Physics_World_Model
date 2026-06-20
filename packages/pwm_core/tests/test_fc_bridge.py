# packages/pwm_core/tests/test_fc_bridge.py
"""bridge: modality template -> ForwardModel, and spec-fields -> ForwardModel."""
from __future__ import annotations

import numpy as np
import pytest

from pwm_core.forward_compiler import compile_model
from pwm_core.forward_compiler.bridge import from_modality, from_spec_fields


def test_from_modality_cassi_builds_three_stages():
    mask = np.random.default_rng(0).integers(0, 2, (8, 8)).astype(np.float64)
    disp = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0], "disp_poly_y": [0.0, 0.0]}
    m = from_modality("cassi", H=8, W=8, N_bands=4, mask=mask, dispersion=disp)
    assert [s.op for s in m.stages] == ["band_shift", "mask_multiply", "band_sum"]
    assert m.x_shape == (8, 8, 4)
    op = compile_model(m)
    assert op.y_shape == (8, 8)
    assert op.check_adjoint(n_trials=2, tol=1e-4).passed


def test_from_modality_unknown_raises():
    with pytest.raises(ValueError, match="unknown modality"):
        from_modality("not_a_modality", H=4, W=4)


def test_from_spec_fields_cassi():
    mask = np.ones((8, 8), dtype=np.float64)
    fields = {
        "spec_type": "cassi",
        "six_tuple": {"omega": {"H": 8, "W": 8, "N_bands": 4}},
        "protocol_fields": {"disp_a1_nominal": 1.0},
    }
    m = from_spec_fields(fields, mask=mask)
    assert m.metadata.get("modality") == "cassi"
    assert m.x_shape == (8, 8, 4)
    op = compile_model(m)
    assert op.y_shape == (8, 8)
