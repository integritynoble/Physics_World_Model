"""Tests for R4 OperatorCorrectionNode primitives and graph integration."""

import warnings

import numpy as np
import pytest

from pwm_core.core.enums import ExecutionMode
from pwm_core.graph.canonical import validate_canonical_chain
from pwm_core.graph.compiler import GraphCompilationError, GraphCompiler
from pwm_core.graph.executor import ExecutionConfig, GraphExecutor
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.ir_types import CorrectionKind, NodeRole
from pwm_core.graph.primitives import (
    AffineCorrectionNode,
    FieldMapCorrectionNode,
    PRIMITIVE_REGISTRY,
    ResidualCorrectionNode,
    get_primitive,
)


# ---------------------------------------------------------------------------
# Primitives registered
# ---------------------------------------------------------------------------


class TestCorrectionPrimitivesRegistered:
    def test_affine_in_registry(self):
        assert "affine_correction" in PRIMITIVE_REGISTRY

    def test_residual_in_registry(self):
        assert "residual_correction" in PRIMITIVE_REGISTRY

    def test_field_map_in_registry(self):
        assert "field_map_correction" in PRIMITIVE_REGISTRY


# ---------------------------------------------------------------------------
# IR type enums
# ---------------------------------------------------------------------------


class TestCorrectionEnums:
    def test_correction_in_node_role(self):
        assert NodeRole.correction == "correction"

    def test_correction_kind_values(self):
        assert CorrectionKind.affine == "affine"
        assert CorrectionKind.residual == "residual"
        assert CorrectionKind.lut == "lut"
        assert CorrectionKind.field_map == "field_map"


# ---------------------------------------------------------------------------
# AffineCorrectionNode
# ---------------------------------------------------------------------------


class TestAffineCorrectionNode:
    def test_forward_gain_offset_scalar(self):
        prim = AffineCorrectionNode(params={"gain": 2.0, "offset": 0.5})
        x = np.ones((8, 8), dtype=np.float64)
        y = prim.forward(x)
        np.testing.assert_allclose(y, 2.0 * 1.0 + 0.5)

    def test_adjoint_gain_scalar(self):
        prim = AffineCorrectionNode(params={"gain": 2.0, "offset": 0.5})
        y = np.ones((8, 8), dtype=np.float64)
        x_adj = prim.adjoint(y)
        # Adjoint of affine: gain * y (offset is dropped)
        np.testing.assert_allclose(x_adj, 2.0)

    def test_forward_array_gain(self):
        gain = np.random.rand(8, 8) + 0.5
        offset = np.random.rand(8, 8) * 0.1
        prim = AffineCorrectionNode(params={"gain": gain, "offset": offset})
        x = np.random.rand(8, 8)
        y = prim.forward(x)
        np.testing.assert_allclose(y, x * gain + offset, rtol=1e-12)

    def test_forward_default_params(self):
        """Default gain=1.0, offset=0.0 should act as identity."""
        prim = AffineCorrectionNode(params={})
        x = np.random.rand(8, 8)
        y = prim.forward(x)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    def test_is_linear(self):
        prim = AffineCorrectionNode(params={"gain": 1.5})
        assert prim.is_linear is True

    def test_node_role(self):
        assert AffineCorrectionNode._node_role == "correction"


# ---------------------------------------------------------------------------
# ResidualCorrectionNode
# ---------------------------------------------------------------------------


class TestResidualCorrectionNode:
    def test_forward_scalar(self):
        prim = ResidualCorrectionNode(params={"residual": 0.3})
        x = np.ones((8, 8), dtype=np.float64)
        y = prim.forward(x)
        np.testing.assert_allclose(y, 1.3)

    def test_adjoint_is_identity(self):
        prim = ResidualCorrectionNode(params={"residual": 0.3})
        y = np.random.rand(8, 8)
        x_adj = prim.adjoint(y)
        np.testing.assert_allclose(x_adj, y, rtol=1e-12)

    def test_forward_array_residual(self):
        residual = np.random.rand(8, 8)
        prim = ResidualCorrectionNode(params={"residual": residual})
        x = np.random.rand(8, 8)
        y = prim.forward(x)
        np.testing.assert_allclose(y, x + residual, rtol=1e-12)


# ---------------------------------------------------------------------------
# FieldMapCorrectionNode
# ---------------------------------------------------------------------------


class TestFieldMapCorrectionNode:
    def test_forward_scalar(self):
        prim = FieldMapCorrectionNode(params={"field_map": 1.5})
        x = np.ones((8, 8), dtype=np.float64)
        y = prim.forward(x)
        np.testing.assert_allclose(y, 1.5)

    def test_forward_array(self):
        fmap = np.random.rand(8, 8) + 0.5
        prim = FieldMapCorrectionNode(params={"field_map": fmap})
        x = np.random.rand(8, 8)
        y = prim.forward(x)
        np.testing.assert_allclose(y, x * fmap, rtol=1e-12)

    def test_adjoint_is_self_adjoint(self):
        """For real field map, adjoint == forward (self-adjoint)."""
        fmap = np.random.rand(8, 8) + 0.5
        prim = FieldMapCorrectionNode(params={"field_map": fmap})
        y = np.random.rand(8, 8)
        x_adj = prim.adjoint(y)
        np.testing.assert_allclose(x_adj, y * fmap, rtol=1e-12)


# ---------------------------------------------------------------------------
# Canonical chain integration
# ---------------------------------------------------------------------------


def _make_graph_with_correction(correction_id="affine_correction",
                                 correction_params=None):
    """Source -> element -> correction -> sensor -> noise."""
    if correction_params is None:
        correction_params = {"gain": 1.1, "offset": 0.01}
    return OperatorGraphSpec.model_validate({
        "graph_id": "test_correction_chain",
        "metadata": {
            "canonical_chain": True,
            "x_shape": [16, 16],
            "y_shape": [16, 16],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "blur", "primitive_id": "conv2d",
             "role": "transport", "params": {"sigma": 1.0, "mode": "reflect"}},
            {"node_id": "correct", "primitive_id": correction_id,
             "role": "correction", "params": correction_params},
            {"node_id": "sensor", "primitive_id": "photon_sensor",
             "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
             "role": "noise",
             "params": {"peak_photons": 10000.0, "read_sigma": 0.01, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "blur"},
            {"source": "blur", "target": "correct"},
            {"source": "correct", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })


class TestCanonicalChainCorrection:
    def test_graph_with_correction_compiles(self):
        """A graph with 1 correction node between element and sensor compiles."""
        spec = _make_graph_with_correction()
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec)
        assert graph_op.graph_id == "test_correction_chain"

    def test_graph_with_correction_validates(self):
        """Canonical chain validator accepts 1 correction node."""
        spec = _make_graph_with_correction()
        validate_canonical_chain(spec)  # Should not raise

    def test_multiple_corrections_rejected(self):
        """More than 1 correction node should be rejected."""
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_multi_correction",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 1.0, "mode": "reflect"}},
                {"node_id": "correct1", "primitive_id": "affine_correction",
                 "role": "correction", "params": {"gain": 1.1, "offset": 0.0}},
                {"node_id": "correct2", "primitive_id": "residual_correction",
                 "role": "correction", "params": {"residual": 0.01}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {"quantum_efficiency": 0.9}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise",
                 "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "correct1"},
                {"source": "correct1", "target": "correct2"},
                {"source": "correct2", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="correction"):
            validate_canonical_chain(spec)

    def test_correction_does_not_count_as_element(self):
        """Graph with only correction + source + sensor + noise -> rejected for missing element."""
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_no_element_only_correction",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "correct", "primitive_id": "affine_correction",
                 "role": "correction", "params": {"gain": 1.1, "offset": 0.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {"quantum_efficiency": 0.9}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise",
                 "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "correct"},
                {"source": "correct", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="element"):
            validate_canonical_chain(spec)

    def test_forward_with_correction_node(self):
        """Compile a correction graph and run forward to verify it works end-to-end."""
        spec = _make_graph_with_correction(
            correction_params={"gain": 2.0, "offset": 0.0}
        )
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec)
        executor = GraphExecutor(graph_op)
        x = np.random.rand(16, 16)
        result = executor.execute(
            x=x,
            config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False),
        )
        assert result.y is not None
        assert result.y.shape == (16, 16)


# ---------------------------------------------------------------------------
# Mode I functional test with correction
# ---------------------------------------------------------------------------


class TestModeIWithCorrection:
    def test_affine_correction_changes_recon(self):
        """Mode I with affine correction produces different recon than without."""
        # Graph without correction
        spec_plain = OperatorGraphSpec.model_validate({
            "graph_id": "test_plain",
            "metadata": {
                "canonical_chain": True,
                "x_shape": [16, 16],
                "y_shape": [16, 16],
            },
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 1.0, "mode": "reflect"}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor",
                 "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise",
                 "params": {"peak_photons": 10000.0, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })

        # Graph with correction
        spec_corrected = _make_graph_with_correction(
            correction_params={"gain": 2.0, "offset": 0.1}
        )

        compiler = GraphCompiler()

        graph_plain = compiler.compile(spec_plain)
        graph_corrected = compiler.compile(spec_corrected)

        executor_plain = GraphExecutor(graph_plain)
        executor_corrected = GraphExecutor(graph_corrected)

        x_true = np.random.rand(16, 16) * 0.5 + 0.1

        # Simulate with each
        y_plain = executor_plain.execute(
            x=x_true,
            config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False),
        ).y

        y_corrected = executor_corrected.execute(
            x=x_true,
            config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False),
        ).y

        # The correction should make outputs different
        assert not np.allclose(y_plain, y_corrected)

        # Reconstruct from plain measurement using both operators
        recon_plain = executor_plain.execute(
            y=y_plain,
            config=ExecutionConfig(
                mode=ExecutionMode.invert, solver_ids=["lsq"],
            ),
        ).x_recon

        recon_corrected = executor_corrected.execute(
            y=y_corrected,
            config=ExecutionConfig(
                mode=ExecutionMode.invert, solver_ids=["lsq"],
            ),
        ).x_recon

        # Both should produce valid reconstructions
        assert recon_plain is not None
        assert recon_corrected is not None
        # They should differ because of the correction
        assert not np.allclose(recon_plain, recon_corrected)


# ---------------------------------------------------------------------------
# Deprecation warning for provided_operator
# ---------------------------------------------------------------------------


class TestProvidedOperatorDeprecation:
    def test_provided_operator_emits_deprecation_warning(self):
        """provided_operator still works but emits DeprecationWarning."""
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_depr",
            "metadata": {
                "canonical_chain": True,
                "x_shape": [16, 16],
                "y_shape": [16, 16],
            },
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport",
                 "params": {"sigma": 1.0, "mode": "reflect"}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor",
                 "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise",
                 "params": {"peak_photons": 10000.0, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        executor = GraphExecutor(graph)

        x_true = np.random.rand(16, 16)
        sim = executor.execute(
            x=x_true,
            config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False),
        )

        class SimpleOp:
            def forward(self, x):
                from scipy import ndimage
                return ndimage.gaussian_filter(x.astype(np.float64), sigma=1.0)

            def adjoint(self, y):
                from scipy import ndimage
                return ndimage.gaussian_filter(y.astype(np.float64), sigma=1.0)

            def info(self):
                return {"operator_id": "simple_op", "is_linear": True}

            @property
            def _x_shape(self):
                return (16, 16)

            @property
            def _y_shape(self):
                return (16, 16)

            @property
            def _is_linear(self):
                return True

        config = ExecutionConfig(
            mode=ExecutionMode.invert,
            solver_ids=["lsq"],
            provided_operator=SimpleOp(),
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = executor.execute(y=sim.y, config=config)
            # Filter for DeprecationWarning about provided_operator
            depr_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "provided_operator" in str(x.message)
            ]
            assert len(depr_warnings) >= 1, (
                f"Expected DeprecationWarning about provided_operator, "
                f"got warnings: {[str(x.message) for x in w]}"
            )
        assert result.x_recon is not None
