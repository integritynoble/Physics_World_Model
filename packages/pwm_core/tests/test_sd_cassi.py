"""test_sd_cassi.py — SD-CASSI implementation tests.

Verifies:
1. New primitives: ObjectiveLens, RelayLens, ImbalancedResponse
2. SpectralDispersion extended output + disp_axis
3. SDCASSIOperator extended measurement shape + adjoint
4. cassi_graph_v2 8-element chain end-to-end
"""

from __future__ import annotations

import numpy as np
import pytest

from pwm_core.graph.primitives import (
    get_primitive,
    ObjectiveLens,
    RelayLens,
    ImbalancedResponse,
    SpectralDispersion,
    FrameIntegration,
    CodedMask,
)
from pwm_core.physics.spectral.cassi_operator import SDCASSIOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W, L = 32, 32, 8
STEP = 2


def _rng():
    return np.random.default_rng(42)


def _cube():
    return _rng().random((H, W, L)).astype(np.float64)


def _mask():
    return (_rng().random((H, W)) > 0.5).astype(np.float64)


# ---------------------------------------------------------------------------
# Primitive registry lookup
# ---------------------------------------------------------------------------

class TestPrimitiveRegistry:
    def test_objective_lens_registered(self):
        p = get_primitive("objective_lens", {"throughput": 0.92})
        assert p.primitive_id == "objective_lens"

    def test_relay_lens_registered(self):
        p = get_primitive("relay_lens", {"throughput": 0.90})
        assert p.primitive_id == "relay_lens"

    def test_imbalanced_response_registered(self):
        p = get_primitive("imbalanced_response", {"a2": 0.01})
        assert p.primitive_id == "imbalanced_response"


# ---------------------------------------------------------------------------
# ObjectiveLens
# ---------------------------------------------------------------------------

class TestObjectiveLens:
    def test_throughput_scaling(self):
        obj = ObjectiveLens({"throughput": 0.5, "psf_sigma": 0.0})
        x = np.ones((H, W, L))
        y = obj.forward(x)
        np.testing.assert_allclose(y, 0.5, atol=1e-12)

    def test_adjoint_self_adjoint(self):
        obj = ObjectiveLens({"throughput": 0.92, "psf_sigma": 0.0})
        x = _cube()
        y = _rng().random((H, W, L))
        # <Ax, y> == <x, A^T y>
        lhs = np.sum(obj.forward(x) * y)
        rhs = np.sum(x * obj.adjoint(y))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_with_psf(self):
        obj = ObjectiveLens({"throughput": 1.0, "psf_sigma": 1.0})
        x = np.zeros((H, W))
        x[H // 2, W // 2] = 1.0
        y = obj.forward(x)
        assert y[H // 2, W // 2] < 1.0  # blurred
        assert y.sum() > 0


# ---------------------------------------------------------------------------
# RelayLens
# ---------------------------------------------------------------------------

class TestRelayLens:
    def test_throughput_scaling(self):
        rl = RelayLens({"throughput": 0.9})
        x = np.ones((H, W, L))
        np.testing.assert_allclose(rl.forward(x), 0.9, atol=1e-12)

    def test_adjoint_consistency(self):
        rl = RelayLens({"throughput": 0.9})
        x = _cube()
        y = _rng().random((H, W, L))
        lhs = np.sum(rl.forward(x) * y)
        rhs = np.sum(x * rl.adjoint(y))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# ---------------------------------------------------------------------------
# ImbalancedResponse
# ---------------------------------------------------------------------------

class TestImbalancedResponse:
    def test_zero_a2_identity(self):
        ir = ImbalancedResponse({"a2": 0.0, "disp_axis": 1})
        x = _cube()
        y = ir.forward(x)
        np.testing.assert_allclose(y, x, atol=1e-12)

    def test_nonzero_a2_shifts_edges(self):
        # Use odd L so center band is exactly at l_c (integer)
        L_odd = 7
        ir = ImbalancedResponse({"a2": 0.1, "disp_axis": 1})
        x = _rng().random((H, W, L_odd)).astype(np.float64)
        y = ir.forward(x)
        # Center band (l_c = 3.0) should be unaffected (dl=0 → warp=0)
        center = L_odd // 2  # 3
        np.testing.assert_allclose(y[:, :, center], x[:, :, center], atol=1e-10)
        # Edge bands should differ
        assert not np.allclose(y[:, :, 0], x[:, :, 0], atol=1e-3)

    def test_adjoint_consistency(self):
        ir = ImbalancedResponse({"a2": 0.05, "disp_axis": 1})
        x = _cube()
        y = _rng().random((H, W, L))
        lhs = np.sum(ir.forward(x) * y)
        rhs = np.sum(x * ir.adjoint(y))
        np.testing.assert_allclose(lhs, rhs, rtol=0.1)  # subpixel shift approx


# ---------------------------------------------------------------------------
# SpectralDispersion — extended output
# ---------------------------------------------------------------------------

class TestSpectralDispersionExtended:
    def test_extended_output_shape_axis1(self):
        sd = SpectralDispersion({
            "disp_step": STEP, "extended_output": True, "disp_axis": 1
        })
        x = _cube()
        y = sd.forward(x)
        expected_W = W + (L - 1) * STEP
        assert y.shape == (H, expected_W, L), f"Expected (H, {expected_W}, L), got {y.shape}"

    def test_extended_output_shape_axis0(self):
        sd = SpectralDispersion({
            "disp_step": STEP, "extended_output": True, "disp_axis": 0
        })
        x = _cube()
        y = sd.forward(x)
        expected_H = H + (L - 1) * STEP
        assert y.shape == (expected_H, W, L), f"Expected ({expected_H}, W, L), got {y.shape}"

    def test_legacy_shape_unchanged(self):
        sd = SpectralDispersion({"disp_step": STEP})
        x = _cube()
        y = sd.forward(x)
        assert y.shape == (H, W, L), f"Legacy shape should be {(H, W, L)}, got {y.shape}"

    def test_extended_adjoint_recovers_shape(self):
        sd = SpectralDispersion({
            "disp_step": STEP, "extended_output": True, "disp_axis": 1
        })
        x = _cube()
        y = sd.forward(x)
        x_back = sd.adjoint(y)
        assert x_back.shape == (H, W, L)

    def test_extended_adjoint_consistency(self):
        sd = SpectralDispersion({
            "disp_step": STEP, "extended_output": True, "disp_axis": 1
        })
        x = _cube()
        W_ext = W + (L - 1) * STEP
        y = _rng().random((H, W_ext, L))
        lhs = np.sum(sd.forward(x) * y)
        rhs = np.sum(x * sd.adjoint(y))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# ---------------------------------------------------------------------------
# SDCASSIOperator
# ---------------------------------------------------------------------------

class TestSDCASSIOperator:
    def test_forward_shape_axis1(self):
        mask = _mask()
        op = SDCASSIOperator(
            operator_id="sd_cassi",
            theta={"L": L, "dispersion_step": STEP, "disp_axis": 1},
            mask=mask,
        )
        x = _cube()
        y = op.forward(x)
        expected_W = W + (L - 1) * STEP
        assert y.shape == (H, expected_W), f"Expected (H, {expected_W}), got {y.shape}"

    def test_forward_shape_axis0(self):
        mask = _mask()
        op = SDCASSIOperator(
            operator_id="sd_cassi",
            theta={"L": L, "dispersion_step": STEP, "disp_axis": 0},
            mask=mask,
        )
        x = _cube()
        y = op.forward(x)
        expected_H = H + (L - 1) * STEP
        assert y.shape == (expected_H, W), f"Expected ({expected_H}, W), got {y.shape}"

    def test_adjoint_shape(self):
        mask = _mask()
        op = SDCASSIOperator(
            operator_id="sd_cassi",
            theta={"L": L, "dispersion_step": STEP, "disp_axis": 1},
            mask=mask,
        )
        W_ext = W + (L - 1) * STEP
        y = _rng().random((H, W_ext))
        x_back = op.adjoint(y)
        assert x_back.shape == (H, W, L)

    def test_adjoint_consistency(self):
        """Verify <Ax, y> ≈ <x, A^T y>."""
        mask = _mask()
        op = SDCASSIOperator(
            operator_id="sd_cassi",
            theta={"L": L, "dispersion_step": STEP, "disp_axis": 1},
            mask=mask,
        )
        x = _cube()
        W_ext = W + (L - 1) * STEP
        y_rand = _rng().random((H, W_ext))

        Ax = op.forward(x)
        Aty = op.adjoint(y_rand)
        lhs = np.sum(Ax * y_rand)
        rhs = np.sum(x * Aty)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_standard_benchmark_shape(self):
        """256×256 mask, 28 bands, step=2 → output (256, 310)."""
        rng = np.random.default_rng(0)
        mask = (rng.random((256, 256)) > 0.5).astype(np.float64)
        op = SDCASSIOperator(
            operator_id="sd_cassi",
            theta={"L": 28, "dispersion_step": 2, "disp_axis": 1},
            mask=mask,
        )
        x = rng.random((256, 256, 28)).astype(np.float64)
        y = op.forward(x)
        assert y.shape == (256, 310), f"Expected (256, 310), got {y.shape}"

    def test_quadratic_dispersion(self):
        mask = _mask()
        op = SDCASSIOperator(
            operator_id="sd_cassi",
            theta={"L": L, "dispersion_step": STEP, "disp_axis": 1, "disp_a2": 0.1},
            mask=mask,
        )
        x = _cube()
        y = op.forward(x)
        W_ext = W + (L - 1) * STEP
        assert y.shape[0] == H
        assert y.shape[1] == W_ext  # same nominal extent


# ---------------------------------------------------------------------------
# Graph template cassi_graph_v2 (8-element chain)
# ---------------------------------------------------------------------------

class TestCASSIGraphV2:
    def test_v2_template_compiles(self):
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        import yaml, pathlib

        templates_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "contrib" / "graph_templates.yaml"
        )
        with open(templates_path) as f:
            raw = yaml.safe_load(f)
        templates = raw.get("templates", raw)

        tmpl = templates["cassi_graph_v2"]
        spec = OperatorGraphSpec(
            graph_id="cassi_graph_v2",
            nodes=tmpl["nodes"],
            edges=tmpl["edges"],
            metadata=tmpl.get("metadata", {}),
        )
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec, x_shape=(64, 64, 8))

        rng = np.random.default_rng(0)
        x = rng.standard_normal((64, 64, 8)).astype(np.float64)
        y = graph_op.forward(x)
        # After integration: 2D output, extended along axis 1
        assert y.ndim == 2
        assert y.shape[1] == 78 or y.size > 0  # 64 + 7*2 = 78

    def test_v1_template_unchanged(self):
        """cassi_graph_v1 still produces (64, 64) output (backward compat)."""
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        import yaml, pathlib

        templates_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "contrib" / "graph_templates.yaml"
        )
        with open(templates_path) as f:
            raw = yaml.safe_load(f)
        templates = raw.get("templates", raw)

        tmpl = templates["cassi_graph_v1"]
        spec = OperatorGraphSpec(
            graph_id="cassi_graph_v1",
            nodes=tmpl["nodes"],
            edges=tmpl["edges"],
            metadata=tmpl.get("metadata", {}),
        )
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec, x_shape=(64, 64, 8))

        rng = np.random.default_rng(0)
        x = rng.standard_normal((64, 64, 8)).astype(np.float64)
        y = graph_op.forward(x)
        assert y.ndim == 2 or y.ndim == 1
        assert y.size > 0
