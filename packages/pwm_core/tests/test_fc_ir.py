# packages/pwm_core/tests/test_fc_ir.py
"""IR round-trip + validation for the forward-model compiler."""
from __future__ import annotations

import numpy as np
import pytest

from pwm_core.forward_compiler.ir import ForwardModel, Stage


def test_stage_roundtrip_plain_params():
    s = Stage(op="scale", params={"c": 2.0})
    assert Stage.from_dict(s.to_dict()) == s


def test_forward_model_roundtrip():
    m = ForwardModel(
        name="demo",
        x_shape=(4, 4, 3),
        stages=[Stage(op="scale", params={"c": 2.0}),
                Stage(op="band_sum", params={})],
        dtype="float32",
        metadata={"modality": "demo"},
    )
    m2 = ForwardModel.from_dict(m.to_dict())
    assert m2 == m
    assert m2.x_shape == (4, 4, 3)
    assert [st.op for st in m2.stages] == ["scale", "band_sum"]


def test_forward_model_requires_name_and_stages():
    with pytest.raises(ValueError):
        ForwardModel(name="", x_shape=(2,), stages=[Stage(op="scale", params={})])
    with pytest.raises(ValueError):
        ForwardModel(name="x", x_shape=(2,), stages=[])


def test_array_param_survives_in_memory():
    mask = np.ones((4, 4), dtype=np.float32)
    s = Stage(op="mask_multiply", params={"mask": mask})
    # to_dict keeps the array object (persistence is a tool-layer concern)
    assert s.to_dict()["params"]["mask"] is mask
