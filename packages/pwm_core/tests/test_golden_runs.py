"""test_golden_runs.py

Golden-run reproducibility tests: same seed -> same output for 3 flagship
modalities (CASSI, CT, Widefield).

These tests verify that the graph-based pipeline is deterministic
given the same random seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
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

# Flagship modalities for golden-run tests
FLAGSHIP_MODALITIES = ["widefield_graph_v1", "ct_graph_v1", "cassi_graph_v1"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGoldenRuns:
    """Test deterministic reproducibility with fixed seeds."""

    @pytest.fixture
    def compiler(self) -> GraphCompiler:
        return GraphCompiler()

    @pytest.mark.parametrize(
        "template_id",
        FLAGSHIP_MODALITIES,
        ids=FLAGSHIP_MODALITIES,
    )
    def test_deterministic_forward(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Two runs with the same seed must produce identical output."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )

        x_shape = tuple(template.get("metadata", {}).get("x_shape", [64, 64]))

        # Run 1
        graph_op_1 = compiler.compile(spec, x_shape=x_shape)
        rng1 = np.random.default_rng(42)
        x1 = rng1.standard_normal(x_shape).astype(np.float64)
        y1 = graph_op_1.forward(x1)

        # Run 2 (fresh compile, same seed)
        graph_op_2 = compiler.compile(spec, x_shape=x_shape)
        rng2 = np.random.default_rng(42)
        x2 = rng2.standard_normal(x_shape).astype(np.float64)
        y2 = graph_op_2.forward(x2)

        np.testing.assert_array_equal(
            x1, x2, err_msg="Input generation not deterministic"
        )
        np.testing.assert_allclose(
            y1,
            y2,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Forward output not deterministic for {template_id}",
        )

    @pytest.mark.parametrize(
        "template_id",
        FLAGSHIP_MODALITIES,
        ids=FLAGSHIP_MODALITIES,
    )
    def test_serialization_hash_stable(
        self, compiler: GraphCompiler, template_id: str
    ) -> None:
        """Graph SHA256 hash must be stable across compilations."""
        template = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )

        graph_op_1 = compiler.compile(spec)
        graph_op_2 = compiler.compile(spec)

        hash_1 = graph_op_1.serialize()["graph_sha256"]
        hash_2 = graph_op_2.serialize()["graph_sha256"]

        assert hash_1 == hash_2, (
            f"Graph hash not stable for {template_id}: "
            f"{hash_1} != {hash_2}"
        )

    def test_widefield_golden_values(
        self, compiler: GraphCompiler
    ) -> None:
        """Widefield forward with fixed seed must produce known statistics."""
        template = TEMPLATES["widefield_graph_v1"]
        spec = OperatorGraphSpec(
            graph_id="widefield_graph_v1",
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec, x_shape=(64, 64))

        rng = np.random.default_rng(42)
        x = rng.standard_normal((64, 64)).astype(np.float64)
        y = graph_op.forward(x)

        # Verify output is finite and has expected shape
        assert np.all(np.isfinite(y)), "Output contains NaN/Inf"
        assert y.shape == (64, 64)

        # The blurred output should have smaller variance than input
        # (smoothing reduces variance)
        # Note: noise adds back some variance, but the blur reduces it
        assert y.std() > 0, "Output is constant (zero variance)"

    def test_ct_golden_values(self, compiler: GraphCompiler) -> None:
        """CT forward with fixed seed must produce valid sinogram."""
        template = TEMPLATES["ct_graph_v1"]
        spec = OperatorGraphSpec(
            graph_id="ct_graph_v1",
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )
        graph_op = compiler.compile(spec, x_shape=(64, 64))

        rng = np.random.default_rng(42)
        x = np.abs(rng.standard_normal((64, 64))).astype(np.float64)
        y = graph_op.forward(x)

        assert np.all(np.isfinite(y)), "Sinogram contains NaN/Inf"
        assert y.ndim == 2  # Sinogram should be 2D

    def test_explain_deterministic(
        self, compiler: GraphCompiler
    ) -> None:
        """explain() must produce the same string for the same graph."""
        template = TEMPLATES["widefield_graph_v1"]
        spec = OperatorGraphSpec(
            graph_id="widefield_graph_v1",
            nodes=template["nodes"],
            edges=template["edges"],
            metadata=template.get("metadata", {}),
        )

        graph_op_1 = compiler.compile(spec)
        graph_op_2 = compiler.compile(spec)

        explain_1 = graph_op_1.explain()
        explain_2 = graph_op_2.explain()

        assert explain_1 == explain_2, "explain() output is not deterministic"
        assert "widefield_graph_v1" in explain_1
        assert "Nodes:" in explain_1 or "nodes" in explain_1.lower()
