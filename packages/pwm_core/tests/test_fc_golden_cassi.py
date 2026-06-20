# packages/pwm_core/tests/test_fc_golden_cassi.py
"""Golden test: compiled CASSI forward matches hand-written CASSIOperator."""
from __future__ import annotations

import numpy as np

from pwm_core.forward_compiler import compile_model
from pwm_core.forward_compiler.bridge import from_modality
from pwm_core.physics.spectral.cassi_operator import CASSIOperator
from pwm_core.core.registry import get_registry


def test_compiled_cassi_matches_handwritten():
    H, W, L = 12, 12, 6
    rng = np.random.default_rng(7)
    mask = rng.integers(0, 2, (H, W)).astype(np.float32)
    theta = {"L": L, "dispersion_model": "poly",
             "disp_poly_x": [0.0, 1.0], "disp_poly_y": [0.0, 0.0]}

    hand = CASSIOperator(operator_id="cassi", theta=theta, mask=mask)
    cube = rng.standard_normal((H, W, L)).astype(np.float32)
    y_hand = hand.forward(cube)

    model = from_modality("cassi", H=H, W=W, N_bands=L,
                          mask=mask.astype(np.float64),
                          dispersion={"dispersion_model": "poly",
                                      "disp_poly_x": [0.0, 1.0],
                                      "disp_poly_y": [0.0, 0.0]})
    op = compile_model(model)
    y_comp = op.forward(cube.astype(np.float64))

    assert y_comp.shape == y_hand.shape
    assert np.allclose(y_comp, y_hand, atol=1e-4), \
        f"max abs diff {np.max(np.abs(y_comp - y_hand))}"


def test_compiler_factory_registered():
    reg = get_registry()
    assert "forward_compiler" in reg.operators
    factory = reg.operators["forward_compiler"]
    op = factory(from_modality("cassi", H=8, W=8, N_bands=4))
    assert op.y_shape == (8, 8)
