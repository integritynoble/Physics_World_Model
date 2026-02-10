"""test_mode1_e2e.py — Track B Mode 1 end-to-end integration tests.

Verifies that ``build_operator()`` goes through GraphCompiler → GraphOperator
→ GraphOperatorAdapter for all modalities that have graph templates, and that
the full run_pipeline produces a valid RunBundle.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from pwm_core.api.types import (
    ExperimentInput,
    ExperimentSpec,
    ExperimentStates,
    InputMode,
    PhysicsState,
    TaskKind,
    TaskState,
)
from pwm_core.core.physics_factory import (
    _STOCHASTIC_PRIMITIVE_IDS,
    _strip_stochastic_nodes,
    _try_build_graph_operator,
    build_operator,
)
from pwm_core.graph.adapter import GraphOperatorAdapter


# ---------------------------------------------------------------------------
# Helper: build a minimal ExperimentSpec
# ---------------------------------------------------------------------------


def _make_spec(
    modality: str,
    dims: Dict[str, Any] | None = None,
) -> ExperimentSpec:
    """Build a minimal simulate-mode ExperimentSpec for a given modality."""
    return ExperimentSpec(
        id=f"test_{modality}",
        input=ExperimentInput(mode=InputMode.simulate),
        states=ExperimentStates(
            physics=PhysicsState(modality=modality, dims=dims),
            task=TaskState(kind=TaskKind.simulate_recon_analyze),
        ),
    )


# ---------------------------------------------------------------------------
# Load templates for parametrized tests
# ---------------------------------------------------------------------------

TEMPLATES_PATH = (
    Path(__file__).resolve().parent.parent / "contrib" / "graph_templates.yaml"
)


def _load_templates() -> Dict[str, Any]:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("templates", {})


TEMPLATES = _load_templates()

# Modalities that have graph templates (extract unique modality values)
_GRAPH_MODALITIES = sorted(
    {
        (tmpl.get("metadata") or {}).get("modality", "").lower()
        for tmpl in TEMPLATES.values()
        if (tmpl.get("metadata") or {}).get("modality")
    }
)


# ---------------------------------------------------------------------------
# Tests: graph-first operator construction
# ---------------------------------------------------------------------------


class TestGraphOperatorUsed:
    """Verify build_operator returns GraphOperatorAdapter for templated modalities."""

    @pytest.mark.parametrize("modality", ["cassi", "widefield", "ct", "spc"])
    def test_build_operator_returns_adapter(self, modality: str) -> None:
        spec = _make_spec(modality)
        op = build_operator(spec)
        assert isinstance(op, GraphOperatorAdapter), (
            f"Expected GraphOperatorAdapter for '{modality}', got {type(op).__name__}"
        )

    @pytest.mark.parametrize("modality", ["cassi", "widefield", "ct", "spc"])
    def test_adapter_info_backend(self, modality: str) -> None:
        spec = _make_spec(modality)
        op = build_operator(spec)
        info = op.info()
        assert info["backend"] == "graph_operator"
        assert info["modality"] == modality

    @pytest.mark.parametrize("modality", _GRAPH_MODALITIES)
    def test_all_templated_modalities_build(self, modality: str) -> None:
        """Every modality with a graph template should build via graph-first."""
        op = _try_build_graph_operator(modality, (64, 64))
        assert op is not None, f"Graph-first build failed for '{modality}'"


# ---------------------------------------------------------------------------
# Tests: stochastic node stripping
# ---------------------------------------------------------------------------


class TestStripStochastic:
    """Verify stochastic nodes are removed before compile."""

    def test_strip_removes_noise_nodes(self) -> None:
        template = {
            "graph_id": "test_strip",
            "metadata": {"modality": "test"},
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d", "params": {"sigma": 2.0}},
                {"node_id": "noise", "primitive_id": "poisson_gaussian", "params": {}},
            ],
            "edges": [{"source": "blur", "target": "noise"}],
        }
        stripped = _strip_stochastic_nodes(template)
        node_ids = [n["node_id"] for n in stripped["nodes"]]
        assert "noise" not in node_ids
        assert "blur" in node_ids
        assert len(stripped["edges"]) == 0

    def test_strip_preserves_deterministic_nodes(self) -> None:
        template = {
            "graph_id": "test_keep",
            "metadata": {"modality": "test"},
            "nodes": [
                {"node_id": "mask", "primitive_id": "coded_mask", "params": {}},
                {"node_id": "disp", "primitive_id": "spectral_dispersion", "params": {}},
                {"node_id": "integ", "primitive_id": "frame_integration", "params": {}},
                {"node_id": "noise", "primitive_id": "poisson", "params": {}},
            ],
            "edges": [
                {"source": "mask", "target": "disp"},
                {"source": "disp", "target": "integ"},
                {"source": "integ", "target": "noise"},
            ],
        }
        stripped = _strip_stochastic_nodes(template)
        node_ids = [n["node_id"] for n in stripped["nodes"]]
        assert node_ids == ["mask", "disp", "integ"]
        # Edge to noise removed, others kept
        assert len(stripped["edges"]) == 2
        assert {"source": "mask", "target": "disp"} in stripped["edges"]
        assert {"source": "disp", "target": "integ"} in stripped["edges"]

    def test_all_four_stochastic_types_stripped(self) -> None:
        for prim_id in _STOCHASTIC_PRIMITIVE_IDS:
            template = {
                "graph_id": f"test_{prim_id}",
                "metadata": {"modality": "test"},
                "nodes": [
                    {"node_id": "op", "primitive_id": "identity", "params": {}},
                    {"node_id": "noise", "primitive_id": prim_id, "params": {}},
                ],
                "edges": [{"source": "op", "target": "noise"}],
            }
            stripped = _strip_stochastic_nodes(template)
            node_ids = [n["node_id"] for n in stripped["nodes"]]
            assert "noise" not in node_ids, f"Failed to strip {prim_id}"

    @pytest.mark.parametrize("modality", ["cassi", "widefield", "ct"])
    def test_compiled_graph_has_no_stochastic(self, modality: str) -> None:
        """After graph-first build, no stochastic nodes should remain."""
        op = _try_build_graph_operator(modality, (64, 64))
        assert op is not None
        graph_op = op.graph_op
        for node_id, prim in graph_op.forward_plan:
            assert not prim.is_stochastic, (
                f"Stochastic node '{node_id}' ({prim.primitive_id}) "
                f"still present in compiled graph for '{modality}'"
            )


# ---------------------------------------------------------------------------
# Tests: forward/adjoint correctness
# ---------------------------------------------------------------------------


class TestGraphOperatorForwardAdjoint:
    """Verify forward/adjoint work on graph-based operators."""

    def test_cassi_forward_adjoint(self) -> None:
        spec = _make_spec("cassi", dims={"H": 64, "W": 64, "L": 8})
        op = build_operator(spec)
        assert isinstance(op, GraphOperatorAdapter)

        x = np.random.default_rng(0).random((64, 64, 8)).astype(np.float64)
        y = op.forward(x)
        assert y.shape == (64, 64), f"Expected (64, 64), got {y.shape}"

        x_back = op.adjoint(y)
        assert x_back.shape == (64, 64, 8), f"Expected (64, 64, 8), got {x_back.shape}"

    def test_widefield_forward_adjoint(self) -> None:
        spec = _make_spec("widefield")
        op = build_operator(spec)
        assert isinstance(op, GraphOperatorAdapter)

        x = np.random.default_rng(0).random((64, 64)).astype(np.float64)
        y = op.forward(x)
        assert y.shape == (64, 64)

        x_back = op.adjoint(y)
        assert x_back.shape == (64, 64)

    def test_ct_forward_adjoint(self) -> None:
        spec = _make_spec("ct")
        op = build_operator(spec)
        assert isinstance(op, GraphOperatorAdapter)

        x = np.random.default_rng(0).random((64, 64)).astype(np.float64)
        y = op.forward(x)
        x_back = op.adjoint(y)
        assert x_back.shape == (64, 64)


# ---------------------------------------------------------------------------
# Tests: serialization + provenance
# ---------------------------------------------------------------------------


class TestSerializationProvenance:
    """Verify GraphOperatorAdapter serialize produces valid output."""

    def test_serialize_has_required_fields(self) -> None:
        spec = _make_spec("cassi", dims={"H": 64, "W": 64, "L": 8})
        op = build_operator(spec)
        assert isinstance(op, GraphOperatorAdapter)

        data = op.serialize()
        assert "graph_id" in data
        assert "graph_sha256" in data
        assert "nodes" in data
        assert "edges" in data
        assert "x_shape" in data
        assert "y_shape" in data
        assert data["adapter"] == "GraphOperatorAdapter"
        assert data["modality"] == "cassi"

    def test_serialize_sha256_present(self) -> None:
        spec = _make_spec("widefield")
        op = build_operator(spec)
        data = op.serialize()
        assert isinstance(data["graph_sha256"], str)
        assert len(data["graph_sha256"]) == 64  # SHA256 hex length

    def test_metadata_method(self) -> None:
        spec = _make_spec("ct")
        op = build_operator(spec)
        assert isinstance(op, GraphOperatorAdapter)
        meta = op.metadata()
        assert meta.modality == "ct"
        assert meta.operator_id.startswith("ct")

    def test_get_set_theta(self) -> None:
        spec = _make_spec("widefield")
        op = build_operator(spec)
        assert isinstance(op, GraphOperatorAdapter)
        theta = op.get_theta()
        assert isinstance(theta, dict)
        assert len(theta) > 0
        # set_theta should not raise
        op.set_theta(theta)


# ---------------------------------------------------------------------------
# Tests: full pipeline (Mode 1 e2e)
# ---------------------------------------------------------------------------


class TestMode1Pipeline:
    """Full run_pipeline e2e tests."""

    def test_run_pipeline_cassi(self) -> None:
        """run_pipeline with CASSI spec produces a RunBundle with diagnosis."""
        from pwm_core.core.runner import run_pipeline

        spec = _make_spec("cassi", dims={"H": 64, "W": 64, "L": 8})
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(spec, out_dir=tmpdir)
            assert result.spec_id == "test_cassi"
            assert result.diagnosis is not None
            assert result.diagnosis.verdict is not None
            assert result.runbundle_path is not None
            assert os.path.isdir(result.runbundle_path)

    def test_run_pipeline_widefield(self) -> None:
        from pwm_core.core.runner import run_pipeline

        spec = _make_spec("widefield")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(spec, out_dir=tmpdir)
            assert result.spec_id == "test_widefield"
            assert result.diagnosis is not None
            assert result.runbundle_path is not None

    def test_runbundle_has_provenance(self) -> None:
        """RunBundle must contain provenance.json with SHA256 hashes."""
        from pwm_core.core.runner import run_pipeline

        spec = _make_spec("cassi", dims={"H": 64, "W": 64, "L": 8})
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(spec, out_dir=tmpdir, seeds=[42])
            rb_dir = result.runbundle_path
            assert rb_dir is not None

            import json

            prov_path = os.path.join(rb_dir, "provenance.json")
            assert os.path.exists(prov_path), "provenance.json missing from RunBundle"
            with open(prov_path) as f:
                prov = json.load(f)
            assert "array_hashes" in prov
            assert "y" in prov["array_hashes"]
            assert "x_hat" in prov["array_hashes"]

    @pytest.mark.parametrize("modality", ["cassi", "widefield", "ct"])
    def test_multiple_modalities_e2e(self, modality: str) -> None:
        """Mode 1 works for multiple modalities end-to-end."""
        from pwm_core.core.runner import run_pipeline

        dims = {"H": 64, "W": 64, "L": 8} if modality == "cassi" else None
        spec = _make_spec(modality, dims=dims)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(spec, out_dir=tmpdir)
            assert result.diagnosis is not None
            assert result.recon is not None
            assert len(result.recon) > 0
