"""Tests for canonical chain validation and all 26 v2 template compilation."""

import os

import numpy as np
import pytest
import yaml

from pwm_core.graph.canonical import validate_canonical_chain
from pwm_core.graph.compiler import GraphCompilationError, GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "packages", "pwm_core", "contrib", "graph_templates.yaml"
)


def _load_templates():
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("templates", {})


def _v2_templates():
    """Return all v2 template keys and data."""
    templates = _load_templates()
    return {k: v for k, v in templates.items() if k.endswith("_v2")}


# ---------------------------------------------------------------------------
# Valid canonical chain tests
# ---------------------------------------------------------------------------


class TestValidCanonicalChain:
    def test_minimal_valid_chain(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_valid",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {"quantum_efficiency": 0.9}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        # Should not raise
        validate_canonical_chain(spec)

    def test_multiple_elements(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_multi_elem",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "mask", "primitive_id": "coded_mask",
                 "role": "transport", "params": {"seed": 42, "H": 64, "W": 64}},
                {"node_id": "disperse", "primitive_id": "spectral_dispersion",
                 "role": "transport", "params": {"disp_step": 1.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {"quantum_efficiency": 0.9}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "mask"},
                {"source": "mask", "target": "disperse"},
                {"source": "disperse", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        validate_canonical_chain(spec)


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------


class TestCanonicalRejections:
    def test_missing_source_raises(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_no_source",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="source"):
            validate_canonical_chain(spec)

    def test_missing_sensor_raises(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_no_sensor",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "noise"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="sensor"):
            validate_canonical_chain(spec)

    def test_missing_noise_raises(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_no_noise",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="noise"):
            validate_canonical_chain(spec)

    def test_missing_elements_raises(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_no_elem",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="element"):
            validate_canonical_chain(spec)

    def test_noise_not_sink_raises(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_noise_not_sink",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
                {"node_id": "extra", "primitive_id": "identity",
                 "role": "transport", "params": {}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
                {"source": "noise", "target": "extra"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="sink"):
            validate_canonical_chain(spec)

    def test_no_path_source_to_sensor_raises(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_disconnected",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                # Missing: blur->sensor, sensor->noise
                {"source": "sensor", "target": "noise"},
            ],
        })
        with pytest.raises(GraphCompilationError, match="path"):
            validate_canonical_chain(spec)


# ---------------------------------------------------------------------------
# Role inference tests
# ---------------------------------------------------------------------------


class TestRoleInference:
    def test_infer_from_primitive_id(self):
        """Role inferred from primitive _node_role class attribute."""
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_infer",
            "metadata": {"canonical_chain": True},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "params": {}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        # Should not raise — roles inferred from primitive class
        validate_canonical_chain(spec)


# ---------------------------------------------------------------------------
# Compiler integration
# ---------------------------------------------------------------------------


class TestCompilerCanonicalIntegration:
    def test_compile_with_canonical_flag(self):
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_compile_canon",
            "metadata": {"canonical_chain": True, "x_shape": [64, 64], "y_shape": [64, 64]},
            "nodes": [
                {"node_id": "source", "primitive_id": "photon_source",
                 "role": "source", "params": {"strength": 1.0}},
                {"node_id": "blur", "primitive_id": "conv2d",
                 "role": "transport", "params": {"sigma": 2.0, "mode": "reflect"}},
                {"node_id": "sensor", "primitive_id": "photon_sensor",
                 "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
                 "role": "noise", "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [
                {"source": "source", "target": "blur"},
                {"source": "blur", "target": "sensor"},
                {"source": "sensor", "target": "noise"},
            ],
        })
        graph_op = compiler.compile(spec)
        assert graph_op.graph_id == "test_compile_canon"

    def test_compile_without_flag_skips_validation(self):
        """v1 templates without canonical_chain flag should still compile."""
        compiler = GraphCompiler()
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_v1_compat",
            "metadata": {"modality": "widefield"},
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "params": {"sigma": 2.0, "mode": "reflect"}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian",
                 "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [{"source": "blur", "target": "noise"}],
        })
        graph_op = compiler.compile(spec)
        assert graph_op is not None


# ---------------------------------------------------------------------------
# All 26 v2 templates compile + validate
# ---------------------------------------------------------------------------


class TestAll26V2Templates:
    @pytest.fixture(scope="class")
    def v2_templates(self):
        return _v2_templates()

    def test_have_26_v2_templates(self, v2_templates):
        assert len(v2_templates) >= 26, (
            f"Expected >= 26 v2 templates, found {len(v2_templates)}: "
            f"{sorted(v2_templates.keys())}"
        )

    def test_all_v2_compile(self, v2_templates):
        compiler = GraphCompiler()
        failures = []
        for key, tdata in sorted(v2_templates.items()):
            tdata_clean = {k: v for k, v in tdata.items() if k != "description"}
            tdata_clean["graph_id"] = key
            try:
                spec = OperatorGraphSpec.model_validate(tdata_clean)
                compiler.compile(spec)
            except Exception as e:
                failures.append(f"{key}: {e}")
        if failures:
            pytest.fail(
                f"{len(failures)} v2 templates failed to compile:\n"
                + "\n".join(failures)
            )

    def test_all_v2_have_canonical_chain_metadata(self, v2_templates):
        for key, tdata in v2_templates.items():
            meta = tdata.get("metadata", {})
            assert meta.get("canonical_chain") is True, (
                f"Template {key} missing canonical_chain: true in metadata"
            )

    def test_all_v2_have_modality_metadata(self, v2_templates):
        for key, tdata in v2_templates.items():
            meta = tdata.get("metadata", {})
            assert "modality" in meta, f"Template {key} missing modality in metadata"

    def test_widefield_v2_forward(self, v2_templates):
        """Smoke test: compile and run forward on widefield_graph_v2."""
        compiler = GraphCompiler()
        tdata = v2_templates.get("widefield_graph_v2")
        if tdata is None:
            pytest.skip("widefield_graph_v2 not found")
        tdata_clean = {k: v for k, v in tdata.items() if k != "description"}
        tdata_clean["graph_id"] = "widefield_graph_v2"
        spec = OperatorGraphSpec.model_validate(tdata_clean)
        graph_op = compiler.compile(spec)
        x = np.random.rand(64, 64)
        y = graph_op.forward(x)
        assert y.shape == (64, 64)
