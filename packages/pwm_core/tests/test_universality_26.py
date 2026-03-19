"""test_universality_26.py

Verify that all 26 graph templates compile, validate, serialize, and
(for linear subgraphs) pass the adjoint check.

Tests
-----
* All 26 templates compile to OperatorGraph.
* All validate against OperatorGraphSpec schema.
* All serialize to valid JSON.
* Fully-linear graphs pass adjoint check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec

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

EXPECTED_MODALITIES = [
    "widefield", "widefield_lowdose", "confocal_livecell",
    "confocal_3d", "sim", "lightsheet", "cassi", "spc", "cacti",
    "matrix", "ct", "mri", "ptychography", "holography",
    "nerf", "gaussian_splatting", "lensless", "panorama",
    "light_field", "dot", "photoacoustic", "oct", "flim",
    "fpm", "phase_retrieval", "integral",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUniversality26:
    """Verify all 26 templates pass the universality checklist."""

    @pytest.fixture
    def compiler(self) -> GraphCompiler:
        return GraphCompiler()

    def test_26_templates_exist(self) -> None:
        """Exactly 26 modalities must have graph templates."""
        for modality in EXPECTED_MODALITIES:
            template_id = f"{modality}_graph_v1"
            assert template_id in TEMPLATES, (
                f"Missing template for modality '{modality}': "
                f"expected '{template_id}'"
            )
        assert len(TEMPLATES) >= 26, (
            f"Expected >= 26 templates, got {len(TEMPLATES)}"
        )

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_schema_valid(self, template_id: str) -> None:
        """Each template must validate against OperatorGraphSpec schema."""
        template = TEMPLATES[template_id]
        template_clean = {
            k: v for k, v in template.items() if k != "description"
        }
        template_clean["graph_id"] = template_id
        spec = OperatorGraphSpec.model_validate(template_clean)
        assert spec.graph_id == template_id
        assert len(spec.nodes) > 0

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_compile(self, compiler: GraphCompiler, template_id: str) -> None:
        """Each template must compile to an OperatorGraph."""
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
    def test_serialize_to_json(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Serialized output must be valid JSON with required fields."""
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

        # Must be JSON-serializable
        json_str = json.dumps(serialized, default=str)
        assert len(json_str) > 0

        # Round-trip parse
        parsed = json.loads(json_str)
        assert parsed["graph_id"] == template_id

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_adjoint_for_linear(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Linear subgraphs must pass the adjoint check.

        Non-linear graphs (containing magnitude_sq, noise, saturation, etc.)
        get a non-applicable pass since adjoint is only defined for linear ops.
        """
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec)

        if graph_op.all_linear:
            report = graph_op.check_adjoint(n_trials=3, rtol=1e-3, seed=42)
            assert report.passed, (
                f"Adjoint check failed for linear graph {template_id}: "
                f"max_rel_err={report.max_relative_error:.2e}"
            )
        else:
            # Non-linear: check_adjoint should report not-passed gracefully
            report = graph_op.check_adjoint(n_trials=1, rtol=1e-3, seed=42)
            assert not report.passed, (
                f"Non-linear graph {template_id} should not pass adjoint check"
            )
