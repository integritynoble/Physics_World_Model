# packages/pwm_core/tests/test_fc_primitives.py
"""Built-in linear primitives: forward shapes + adjoint correctness."""
from __future__ import annotations

import numpy as np
import pytest

from pwm_core.forward_compiler.primitives import get_primitive, PRIMITIVES


def _adjoint_inner_product_ok(prim, in_shape, params, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(in_shape).astype(np.float64)
    out_shape = prim.out_shape(in_shape, **params)
    y = rng.standard_normal(out_shape).astype(np.float64)
    Ax = prim.forward(x, **params).astype(np.float64)
    ATy = prim.adjoint(y, **params).astype(np.float64)
    lhs = float(np.sum(Ax.ravel() * y.ravel()))
    rhs = float(np.sum(x.ravel() * ATy.ravel()))
    denom = max(abs(lhs), abs(rhs), 1e-30)
    return abs(lhs - rhs) / denom < tol


def test_scale_forward_and_adjoint():
    p = get_primitive("scale")
    x = np.ones((3, 3), dtype=np.float32)
    assert np.allclose(p.forward(x, c=2.0), 2.0)
    assert p.out_shape((3, 3), c=2.0) == (3, 3)
    assert p.is_linear
    assert _adjoint_inner_product_ok(p, (3, 3), {"c": 2.0})


def test_mask_multiply_broadcasts_over_bands():
    p = get_primitive("mask_multiply")
    mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    x = np.ones((2, 2, 3), dtype=np.float32)
    y = p.forward(x, mask=mask)
    assert y.shape == (2, 2, 3)
    assert y[0, 1, 0] == 0.0 and y[0, 0, 0] == 1.0
    assert p.out_shape((2, 2, 3), mask=mask) == (2, 2, 3)
    assert _adjoint_inner_product_ok(p, (2, 2, 3), {"mask": mask.astype(np.float64)})


def test_band_sum_collapses_last_axis():
    p = get_primitive("band_sum")
    x = np.ones((2, 2, 4), dtype=np.float32)
    y = p.forward(x)
    assert y.shape == (2, 2)
    assert np.allclose(y, 4.0)
    assert p.out_shape((2, 2, 4)) == (2, 2)
    assert _adjoint_inner_product_ok(p, (2, 2, 4), {"n_bands": 4})


def test_band_shift_adjoint():
    p = get_primitive("band_shift")
    disp = {"dispersion_model": "poly", "disp_poly_x": [0.0, 1.0], "disp_poly_y": [0.0, 0.0]}
    in_shape = (8, 8, 4)
    assert p.out_shape(in_shape, dispersion=disp) == (8, 8, 4)
    assert _adjoint_inner_product_ok(p, in_shape, {"dispersion": disp}, tol=1e-4)


def test_unknown_primitive_raises():
    with pytest.raises(KeyError):
        get_primitive("does_not_exist")


def test_registry_contains_linear_builtins():
    for name in ("scale", "mask_multiply", "band_shift", "band_sum"):
        assert name in PRIMITIVES


def test_band_shift_adjoint_subpixel_exact():
    # Fractional dispersion must still pass the adjoint inner-product test
    # (the old reverse-shift adjoint failed here at ~6e-2).
    p = get_primitive("band_shift")
    disp = {"dispersion_model": "poly", "disp_poly_x": [0.3, 0.7], "disp_poly_y": [0.0, 0.5]}
    in_shape = (8, 8, 4)
    assert _adjoint_inner_product_ok(p, in_shape, {"dispersion": disp}, tol=1e-9)


def test_square_magnitude_nonlinear_no_adjoint():
    p = get_primitive("square_magnitude")
    x = np.array([-2.0, 3.0], dtype=np.float64)
    assert np.allclose(p.forward(x), [4.0, 9.0])
    assert p.is_linear is False
    assert p.adjoint is None
    assert p.out_shape((2,)) == (2,)


def test_gaussian_noise_nonlinear_and_seeded():
    p = get_primitive("gaussian_noise")
    x = np.zeros((4, 4), dtype=np.float64)
    y1 = p.forward(x, sigma=0.5, seed=7)
    y2 = p.forward(x, sigma=0.5, seed=7)
    assert p.is_linear is False
    assert p.adjoint is None
    assert np.allclose(y1, y2)          # seeded => reproducible
    assert y1.std() > 0.0               # noise actually added
    assert p.out_shape((4, 4), sigma=0.5) == (4, 4)
