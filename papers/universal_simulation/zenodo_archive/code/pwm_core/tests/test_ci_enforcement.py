"""test_ci_enforcement.py

Gate 0 (G0.4) — CI enforcement tests.

Tests
-----
* No-bypass lint: no modality-specific forward code outside GraphOperator.
* Serialize roundtrip determinism: same graph produces identical SHA256 hashes.
* Adjoint gate: all linear templates pass adjoint check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

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

# Identify linear templates
COMPILER = GraphCompiler()


def _is_linear_template(tid: str) -> bool:
    template = TEMPLATES[tid]
    spec = OperatorGraphSpec(
        graph_id=tid,
        nodes=template["nodes"],
        edges=template["edges"],
        metadata=template.get("metadata", {}),
    )
    graph_op = COMPILER.compile(spec)
    return graph_op.all_linear


LINEAR_TEMPLATE_IDS = [tid for tid in TEMPLATE_IDS if _is_linear_template(tid)]


# ---------------------------------------------------------------------------
# G0.4a: No-bypass lint
# ---------------------------------------------------------------------------

# Source directories that should NOT contain modality-specific forward code
RECON_SRC = Path(__file__).resolve().parent.parent / "pwm_core" / "recon"

# Patterns that indicate modality-specific forward bypass
# (functions that replicate what GraphOperator.forward does)
BYPASS_PATTERNS = [
    "def forward_cassi(",
    "def forward_cacti(",
    "def forward_spc(",
    "def cassi_forward(",
    "def cacti_forward(",
    "def spc_forward(",
]


class TestNoBypassLint:
    """No modality-specific forward code should exist outside GraphOperator."""

    def test_no_modality_forward_in_recon(self) -> None:
        """Recon solvers must not contain modality-specific forward functions."""
        violations: List[str] = []
        if not RECON_SRC.is_dir():
            pytest.skip("recon source directory not found")

        for py_file in RECON_SRC.glob("*.py"):
            content = py_file.read_text()
            for pattern in BYPASS_PATTERNS:
                if pattern in content:
                    violations.append(
                        f"{py_file.name} contains '{pattern}'"
                    )

        assert not violations, (
            "Modality-specific forward code found outside GraphOperator:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# G0.4b: Serialize roundtrip determinism
# ---------------------------------------------------------------------------


class TestSerializeRoundtripDeterminism:
    """Compiling the same spec twice must produce identical serialization."""

    @pytest.mark.parametrize("template_id", TEMPLATE_IDS, ids=TEMPLATE_IDS)
    def test_deterministic_hash(self, template_id: str) -> None:
        """graph_sha256 must be identical across two compile+serialize calls."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )

        compiler1 = GraphCompiler()
        compiler2 = GraphCompiler()

        graph_op1 = compiler1.compile(spec)
        graph_op2 = compiler2.compile(spec)

        ser1 = graph_op1.serialize()
        ser2 = graph_op2.serialize()

        assert ser1["graph_sha256"] == ser2["graph_sha256"], (
            f"Non-deterministic hash for '{template_id}': "
            f"{ser1['graph_sha256']} != {ser2['graph_sha256']}"
        )

        # Full JSON round-trip equality
        json1 = json.dumps(ser1, sort_keys=True, default=str)
        json2 = json.dumps(ser2, sort_keys=True, default=str)
        assert json1 == json2, (
            f"Non-deterministic serialization for '{template_id}'"
        )


# ---------------------------------------------------------------------------
# G0.4c: Adjoint gate for linear templates
# ---------------------------------------------------------------------------


class TestAdjointGate:
    """All fully-linear templates must pass the adjoint consistency check."""

    @pytest.mark.parametrize(
        "template_id", LINEAR_TEMPLATE_IDS, ids=LINEAR_TEMPLATE_IDS
    )
    def test_adjoint_passes(self, template_id: str) -> None:
        """<Ax, y> == <x, A^T y> for all linear templates."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec)

        assert graph_op.all_linear, (
            f"Template '{template_id}' expected to be linear"
        )

        report = graph_op.check_adjoint(n_trials=3, rtol=1e-3, seed=42)
        assert report.passed, (
            f"Adjoint gate FAILED for '{template_id}': "
            f"max_rel_err={report.max_relative_error:.2e}, "
            f"tol={report.tolerance:.2e}"
        )


# ---------------------------------------------------------------------------
# E.6: Template stability test
# ---------------------------------------------------------------------------


class TestTemplateStability:
    """Verify graph_templates.yaml structure doesn't regress."""

    def test_minimum_template_count(self) -> None:
        """Must have at least 26 templates."""
        assert len(TEMPLATES) >= 26, (
            f"Expected >= 26 templates, got {len(TEMPLATES)}"
        )

    def test_all_templates_have_metadata(self) -> None:
        """Every template must have a metadata section."""
        missing = [tid for tid in TEMPLATE_IDS if "metadata" not in TEMPLATES[tid]]
        assert not missing, (
            f"Templates missing 'metadata': {missing}"
        )

    def test_all_templates_have_modality(self) -> None:
        """Every template metadata must include a modality field."""
        missing = []
        for tid in TEMPLATE_IDS:
            meta = TEMPLATES[tid].get("metadata", {})
            if not meta.get("modality"):
                missing.append(tid)
        assert not missing, (
            f"Templates missing 'metadata.modality': {missing}"
        )

    def test_known_modalities_have_templates(self) -> None:
        """Key modalities must have at least one template."""
        modalities = set()
        for tid, tmpl in TEMPLATES.items():
            meta = tmpl.get("metadata", {})
            if meta.get("modality"):
                modalities.add(meta["modality"])

        required = ["cassi", "widefield", "ct", "spc"]
        missing = [m for m in required if m not in modalities]
        assert not missing, (
            f"Required modalities without templates: {missing}. "
            f"Found modalities: {sorted(modalities)}"
        )

    def test_all_templates_have_nodes(self) -> None:
        """Every template must have at least one node."""
        empty = [tid for tid in TEMPLATE_IDS if not TEMPLATES[tid].get("nodes")]
        assert not empty, f"Templates with no nodes: {empty}"

    def test_all_templates_have_edges_or_single_node(self) -> None:
        """Templates with >1 node must have edges."""
        bad = []
        for tid in TEMPLATE_IDS:
            tmpl = TEMPLATES[tid]
            nodes = tmpl.get("nodes", [])
            edges = tmpl.get("edges", [])
            if len(nodes) > 1 and not edges:
                bad.append(tid)
        assert not bad, f"Multi-node templates without edges: {bad}"
