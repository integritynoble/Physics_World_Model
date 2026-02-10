"""test_graph_adjoint.py

For each graph template that declares only linear nodes,
compile and run check_adjoint().

Tests
-----
* CT and Widefield breadth-anchor templates must pass.
* Any template with all-linear nodes must pass adjoint check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import PRIMITIVE_REGISTRY

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


def _is_all_linear(template: Dict[str, Any]) -> bool:
    """Check if a template contains only linear primitives."""
    for node in template.get("nodes", []):
        pid = node.get("primitive_id", "")
        if pid in PRIMITIVE_REGISTRY:
            cls = PRIMITIVE_REGISTRY[pid]
            if hasattr(cls, "_is_linear") and not cls._is_linear:
                return False
        else:
            return False
    return True


# Collect linear-only templates
LINEAR_TEMPLATES = {
    tid: t for tid, t in TEMPLATES.items() if _is_all_linear(t)
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGraphAdjoint:
    """Test adjoint consistency for all-linear graph templates."""

    @pytest.fixture
    def compiler(self) -> GraphCompiler:
        return GraphCompiler()

    @pytest.mark.parametrize(
        "template_id",
        sorted(LINEAR_TEMPLATES.keys()),
        ids=sorted(LINEAR_TEMPLATES.keys()),
    )
    def test_linear_template_adjoint(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Compile a linear template and verify adjoint consistency."""
        template = LINEAR_TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec)
        assert graph_op.all_linear, (
            f"Expected all-linear graph for {template_id}"
        )

        report = graph_op.check_adjoint(n_trials=3, rtol=1e-4, seed=42)
        assert report.passed, (
            f"Adjoint check FAILED for {template_id}: "
            f"max_rel_err={report.max_relative_error:.2e}, "
            f"tol={report.tolerance:.2e}"
        )

    def test_widefield_adjoint(self, compiler: GraphCompiler) -> None:
        """Breadth anchor: widefield_graph_v1 must compile and pass adjoint."""
        assert "widefield_graph_v1" in TEMPLATES
        template = TEMPLATES["widefield_graph_v1"]
        # The widefield template has noise (non-linear) at the end.
        # For adjoint testing, build a linear-only version.
        linear_nodes = [
            n
            for n in template["nodes"]
            if _node_is_linear(n)
        ]
        linear_edges = [
            e
            for e in template["edges"]
            if e["source"] in {n["node_id"] for n in linear_nodes}
            and e["target"] in {n["node_id"] for n in linear_nodes}
        ]

        # If the whole template is linear, test directly
        if _is_all_linear(template):
            spec = OperatorGraphSpec(
                graph_id="widefield_graph_v1",
                nodes=template["nodes"],
                edges=template["edges"],
                metadata=template.get("metadata", {}),
            )
            graph_op = compiler.compile(spec)
            report = graph_op.check_adjoint(n_trials=3, rtol=1e-4, seed=42)
            assert report.passed
        else:
            # Test with only the linear (blur) node
            if linear_nodes:
                spec = OperatorGraphSpec(
                    graph_id="widefield_linear_only",
                    nodes=linear_nodes,
                    edges=linear_edges,
                    metadata=template.get("metadata", {}),
                )
                graph_op = compiler.compile(spec)
                assert graph_op.all_linear
                report = graph_op.check_adjoint(n_trials=3, rtol=1e-4, seed=42)
                assert report.passed

    def test_ct_adjoint(self, compiler: GraphCompiler) -> None:
        """Breadth anchor: CT radon (small, matrix-based) must pass adjoint.

        We use a small image (16x16) with few angles so the matrix-based
        path triggers, giving exact adjoint consistency.
        """
        from pwm_core.graph.graph_spec import GraphEdge, GraphNode

        spec = OperatorGraphSpec(
            graph_id="ct_small_linear",
            nodes=[
                GraphNode(
                    node_id="radon",
                    primitive_id="ct_radon",
                    params={"n_angles": 18, "H": 16, "W": 16},
                ),
            ],
            edges=[],
            metadata={"x_shape": [16, 16], "y_shape": [18, 16]},
        )
        graph_op = compiler.compile(
            spec, x_shape=(16, 16), y_shape=(18, 16)
        )
        assert graph_op.all_linear
        report = graph_op.check_adjoint(
            n_trials=3,
            rtol=1e-4,
            seed=42,
        )
        assert report.passed, (
            f"CT adjoint check FAILED: "
            f"max_rel_err={report.max_relative_error:.2e}"
        )


def _node_is_linear(node: Dict[str, Any]) -> bool:
    """Check if a single node references a linear primitive."""
    pid = node.get("primitive_id", "")
    if pid in PRIMITIVE_REGISTRY:
        cls = PRIMITIVE_REGISTRY[pid]
        return getattr(cls, "_is_linear", False)
    return False
