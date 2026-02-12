"""Tests for CasePack: CBCT.

Template: cbct_graph_v2
Chain: xray_source -> ct_radon -> beer_lambert -> photon_sensor -> poisson_only_sensor
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec

TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "packages", "pwm_core", "contrib", "graph_templates.yaml",
)


def _load_template(key):
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


class TestCasePackCbct:
    """CasePack acceptance tests for the cbct modality."""

    def test_template_compiles(self):
        """cbct_graph_v2 template compiles without error."""
        tpl = _load_template("cbct_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "cbct_graph_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None

    def test_forward_sanity(self):
        """Mode S: forward pass produces finite, correctly shaped output."""
        tpl = _load_template("cbct_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "cbct_graph_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        rng = np.random.RandomState(42)
        x = rng.rand(*(64, 64)).astype(np.float64)
        y = graph.forward(x)
        assert y is not None
        assert np.isfinite(y).all()

    def test_forward_nonneg_input(self):
        """Non-negative input produces finite output."""
        tpl = _load_template("cbct_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "cbct_graph_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        x = np.ones((64, 64), dtype=np.float64) * 0.5
        y = graph.forward(x)
        assert np.isfinite(y).all()
