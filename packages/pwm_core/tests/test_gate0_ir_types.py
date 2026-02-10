"""test_gate0_ir_types.py

Gate 0 (G0.3) — 26-Template Validation for IR formal types.

Tests
-----
* Every compiled template has NodeTags on ALL nodes.
* Tags match the primitive's actual properties.
* Serialized output includes per-node tags and top-level node_tags dict.
* NodeTags, TensorSpec, ParameterSpec Pydantic models validate correctly.
* GraphOperator satisfies LinearLikeOperator protocol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.ir_types import NodeTags, ParameterSpec, TensorSpec
from pwm_core.graph.primitives import PRIMITIVE_REGISTRY, get_primitive
from pwm_core.recon.protocols import LinearLikeOperator

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
# IR type unit tests
# ---------------------------------------------------------------------------


class TestIRTypes:
    """Unit tests for NodeTags, TensorSpec, ParameterSpec."""

    def test_node_tags_defaults(self) -> None:
        tags = NodeTags()
        assert tags.is_linear is True
        assert tags.is_stochastic is False
        assert tags.is_differentiable is True
        assert tags.is_stateful is False

    def test_node_tags_custom(self) -> None:
        tags = NodeTags(
            is_linear=False,
            is_stochastic=True,
            is_differentiable=False,
            is_stateful=True,
        )
        assert tags.is_linear is False
        assert tags.is_stochastic is True
        assert tags.is_differentiable is False
        assert tags.is_stateful is True

    def test_node_tags_extra_field_rejected(self) -> None:
        with pytest.raises(Exception):
            NodeTags(is_linear=True, bogus_field=True)

    def test_node_tags_serialization(self) -> None:
        tags = NodeTags(is_linear=False, is_stochastic=True)
        d = tags.model_dump()
        assert d["is_linear"] is False
        assert d["is_stochastic"] is True
        # Round-trip
        tags2 = NodeTags.model_validate(d)
        assert tags2 == tags

    def test_tensor_spec_defaults(self) -> None:
        ts = TensorSpec()
        assert ts.dtype == "float64"
        assert ts.unit == "arbitrary"
        assert ts.domain == "real"

    def test_tensor_spec_custom(self) -> None:
        ts = TensorSpec(
            shape=[64, 64, 28],
            dtype="complex128",
            unit="photons",
            domain="complex",
        )
        assert ts.shape == [64, 64, 28]
        assert ts.dtype == "complex128"

    def test_parameter_spec(self) -> None:
        ps = ParameterSpec(
            name="sigma",
            lower=0.1,
            upper=10.0,
            prior="log_uniform",
            parameterization="log",
            identifiability_hint="identifiable",
        )
        assert ps.name == "sigma"
        assert ps.lower == 0.1
        assert ps.upper == 10.0
        assert ps.prior == "log_uniform"

    def test_parameter_spec_defaults(self) -> None:
        ps = ParameterSpec(name="dx")
        assert ps.lower == 0.0
        assert ps.upper == 1.0
        assert ps.prior == "uniform"
        assert ps.parameterization == "identity"
        assert ps.identifiability_hint == "unknown"


# ---------------------------------------------------------------------------
# Primitive tag consistency
# ---------------------------------------------------------------------------


class TestPrimitiveTags:
    """Verify all primitives expose the 4 tag properties."""

    @pytest.mark.parametrize(
        "primitive_id", sorted(PRIMITIVE_REGISTRY.keys())
    )
    def test_primitive_has_tag_properties(self, primitive_id: str) -> None:
        prim = get_primitive(primitive_id)
        # All 4 properties must be booleans
        assert isinstance(prim.is_linear, bool)
        assert isinstance(prim.is_stochastic, bool)
        assert isinstance(prim.is_differentiable, bool)
        assert isinstance(prim.is_stateful, bool)

    def test_noise_primitives_are_stochastic(self) -> None:
        """Noise primitives must be tagged stochastic."""
        noise_ids = ["poisson", "gaussian", "poisson_gaussian", "fpn"]
        for pid in noise_ids:
            prim = get_primitive(pid)
            assert prim.is_stochastic, (
                f"Primitive '{pid}' should be stochastic"
            )

    def test_linear_primitives_are_differentiable(self) -> None:
        """All linear primitives must be differentiable."""
        for pid, cls in PRIMITIVE_REGISTRY.items():
            prim = cls()
            if prim.is_linear:
                assert prim.is_differentiable, (
                    f"Linear primitive '{pid}' should be differentiable"
                )


# ---------------------------------------------------------------------------
# 26-template NodeTags validation
# ---------------------------------------------------------------------------


class TestTemplateNodeTags:
    """Every compiled template must have NodeTags on ALL nodes."""

    @pytest.fixture
    def compiler(self) -> GraphCompiler:
        return GraphCompiler()

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_all_nodes_have_tags(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Compiled graph must have NodeTags for every node."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec)

        # Every node must have tags
        for node_id, _ in graph_op.forward_plan:
            assert node_id in graph_op.node_tags, (
                f"Node '{node_id}' in graph '{template_id}' has no NodeTags"
            )
            tags = graph_op.node_tags[node_id]
            assert isinstance(tags, NodeTags)

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_tags_match_primitive_properties(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """NodeTags must match the actual primitive properties."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec)

        for node_id, prim in graph_op.forward_plan:
            tags = graph_op.node_tags[node_id]
            assert tags.is_linear == prim.is_linear, (
                f"is_linear mismatch for node '{node_id}' in '{template_id}'"
            )
            assert tags.is_stochastic == prim.is_stochastic, (
                f"is_stochastic mismatch for node '{node_id}' in '{template_id}'"
            )
            assert tags.is_differentiable == prim.is_differentiable, (
                f"is_differentiable mismatch for '{node_id}' in '{template_id}'"
            )
            assert tags.is_stateful == prim.is_stateful, (
                f"is_stateful mismatch for node '{node_id}' in '{template_id}'"
            )

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_serialize_includes_tags(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Serialized output must include per-node tags and top-level node_tags."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec)
        serialized = graph_op.serialize()

        # Top-level node_tags dict
        assert "node_tags" in serialized
        assert isinstance(serialized["node_tags"], dict)
        assert len(serialized["node_tags"]) == len(graph_op.forward_plan)

        # Each node in serialized["nodes"] must have tags
        for node_data in serialized["nodes"]:
            assert "tags" in node_data, (
                f"Node '{node_data['node_id']}' in '{template_id}' "
                "missing 'tags' in serialized output"
            )
            tags_data = node_data["tags"]
            assert "is_linear" in tags_data
            assert "is_stochastic" in tags_data
            assert "is_differentiable" in tags_data
            assert "is_stateful" in tags_data

        # JSON round-trip
        json_str = json.dumps(serialized, default=str)
        parsed = json.loads(json_str)
        assert parsed["node_tags"] == serialized["node_tags"]


# ---------------------------------------------------------------------------
# LinearLikeOperator protocol
# ---------------------------------------------------------------------------


class TestLinearLikeOperator:
    """GraphOperator must satisfy LinearLikeOperator protocol."""

    def test_graph_operator_satisfies_protocol(self) -> None:
        """A compiled GraphOperator must be a LinearLikeOperator."""
        compiler = GraphCompiler()
        # Use a simple linear template
        spec = OperatorGraphSpec(
            graph_id="protocol_test",
            nodes=[
                {"node_id": "pass", "primitive_id": "identity"},
            ],
            edges=[],
            metadata={"x_shape": [8, 8], "y_shape": [8, 8]},
        )
        graph_op = compiler.compile(spec)
        assert isinstance(graph_op, LinearLikeOperator), (
            "GraphOperator must satisfy LinearLikeOperator protocol"
        )
        assert hasattr(graph_op, "forward")
        assert hasattr(graph_op, "adjoint")
        assert hasattr(graph_op, "x_shape")
        assert hasattr(graph_op, "y_shape")
        assert hasattr(graph_op, "all_linear")
