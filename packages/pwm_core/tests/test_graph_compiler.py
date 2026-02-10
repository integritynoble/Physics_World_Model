"""test_graph_compiler.py

Compile all 26 templates and verify serialization round-trip.

Tests
-----
* All 26 templates must compile without error.
* Serialized GraphOperator must contain required fields.
* DAG validation catches cycles.
* Unknown primitive_id is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from pwm_core.graph.compiler import GraphCompilationError, GraphCompiler
from pwm_core.graph.graph_spec import GraphEdge, GraphNode, OperatorGraphSpec

# ---------------------------------------------------------------------------
# Load templates
# ---------------------------------------------------------------------------

TEMPLATES_PATH = (
    Path(__file__).resolve().parent.parent / "contrib" / "graph_templates.yaml"
)


def _load_templates() -> Dict[str, Any]:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("templates", {})


TEMPLATES = _load_templates()
TEMPLATE_IDS = sorted(TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGraphCompiler:
    """Test compilation of all graph templates."""

    @pytest.fixture
    def compiler(self) -> GraphCompiler:
        return GraphCompiler()

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_compile_template(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Each of the 26 templates must compile without error."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec)
        assert graph_op.graph_id == template_id
        assert len(graph_op.forward_plan) == len(template["nodes"])

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_serialization_roundtrip(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Serialized output must contain required fields and be JSON-safe."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec)
        serialized = graph_op.serialize()

        # Required fields
        assert "graph_id" in serialized
        assert "graph_sha256" in serialized
        assert "all_linear" in serialized
        assert "nodes" in serialized
        assert "edges" in serialized
        assert "blobs" in serialized

        assert serialized["graph_id"] == template_id
        assert isinstance(serialized["graph_sha256"], str)
        assert len(serialized["graph_sha256"]) == 64  # SHA256 hex

        # Must be JSON-serializable
        json_str = json.dumps(serialized, default=str)
        assert len(json_str) > 0

        # Round-trip: parse back (shallow check)
        parsed = json.loads(json_str)
        assert parsed["graph_id"] == template_id
        assert parsed["n_nodes"] == len(template["nodes"])

    def test_all_26_modalities_have_templates(self) -> None:
        """Verify that all 26 modality keys have a graph template."""
        expected_modalities = [
            "widefield", "widefield_lowdose", "confocal_livecell",
            "confocal_3d", "sim", "lightsheet", "cassi", "spc", "cacti",
            "matrix", "ct", "mri", "ptychography", "holography",
            "nerf", "gaussian_splatting", "lensless", "panorama",
            "light_field", "dot", "photoacoustic", "oct", "flim",
            "fpm", "phase_retrieval", "integral",
        ]
        for modality in expected_modalities:
            template_id = f"{modality}_graph_v1"
            assert template_id in TEMPLATES, (
                f"Missing template for modality '{modality}': "
                f"expected '{template_id}' in graph_templates.yaml"
            )

    def test_cycle_detection(self, compiler: GraphCompiler) -> None:
        """Cyclic graph must be rejected."""
        spec = OperatorGraphSpec(
            graph_id="cycle_test",
            nodes=[
                GraphNode(node_id="a", primitive_id="identity"),
                GraphNode(node_id="b", primitive_id="identity"),
            ],
            edges=[
                GraphEdge(source="a", target="b"),
                GraphEdge(source="b", target="a"),
            ],
        )
        with pytest.raises(GraphCompilationError, match="cycle"):
            compiler.compile(spec)

    def test_unknown_primitive_rejected(self, compiler: GraphCompiler) -> None:
        """Node referencing a non-existent primitive must be rejected."""
        spec = OperatorGraphSpec(
            graph_id="bad_primitive_test",
            nodes=[
                GraphNode(node_id="x", primitive_id="nonexistent_op_v99"),
            ],
            edges=[],
        )
        with pytest.raises(GraphCompilationError, match="unknown"):
            compiler.compile(spec)

    def test_duplicate_node_ids_rejected(self) -> None:
        """OperatorGraphSpec must reject duplicate node_ids."""
        with pytest.raises(Exception, match="[Dd]uplicate"):
            OperatorGraphSpec(
                graph_id="dup_test",
                nodes=[
                    GraphNode(node_id="a", primitive_id="identity"),
                    GraphNode(node_id="a", primitive_id="identity"),
                ],
                edges=[],
            )

    def test_edge_references_nonexistent_node(self) -> None:
        """Edge referencing a non-existent node must be rejected."""
        with pytest.raises(Exception, match="not found"):
            OperatorGraphSpec(
                graph_id="bad_edge_test",
                nodes=[
                    GraphNode(node_id="a", primitive_id="identity"),
                ],
                edges=[
                    GraphEdge(source="a", target="nonexistent"),
                ],
            )

    def test_from_dict_utility(self) -> None:
        """GraphCompiler.from_dict should parse a dict into OperatorGraphSpec."""
        data = {
            "graph_id": "test_dict",
            "nodes": [
                {"node_id": "pass", "primitive_id": "identity"},
            ],
            "edges": [],
        }
        spec = GraphCompiler.from_dict(data)
        assert isinstance(spec, OperatorGraphSpec)
        assert spec.graph_id == "test_dict"
