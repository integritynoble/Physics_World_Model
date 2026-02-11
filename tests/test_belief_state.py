"""Tests for pwm_core.mismatch.belief_state."""

import numpy as np
import pytest

from pwm_core.graph.ir_types import ParameterSpec
from pwm_core.mismatch.belief_state import BeliefState, build_belief_from_graph
from pwm_core.mismatch.parameterizations import ThetaSpace


class TestBeliefState:
    def test_creation_empty(self):
        bs = BeliefState()
        assert bs.theta == {}
        assert bs.params == {}
        assert bs.history == []
        assert bs.uncertainty is None

    def test_creation_with_params(self):
        ps = ParameterSpec(name="sigma", lower=0.0, upper=10.0)
        bs = BeliefState(
            params={"sigma": ps},
            theta={"sigma": 2.0},
        )
        assert bs.theta["sigma"] == 2.0
        assert bs.params["sigma"].lower == 0.0

    def test_update_pushes_history(self):
        bs = BeliefState(theta={"a": 1.0, "b": 2.0})
        bs.update({"a": 3.0, "b": 4.0})
        assert len(bs.history) == 1
        assert bs.history[0] == {"a": 1.0, "b": 2.0}
        assert bs.theta == {"a": 3.0, "b": 4.0}

    def test_update_with_uncertainty(self):
        bs = BeliefState(theta={"x": 1.0})
        bs.update({"x": 2.0}, uncertainty={"x": 0.1})
        assert bs.uncertainty == {"x": 0.1}
        assert bs.theta == {"x": 2.0}

    def test_multiple_updates(self):
        bs = BeliefState(theta={"a": 0.0})
        bs.update({"a": 1.0})
        bs.update({"a": 2.0})
        bs.update({"a": 3.0})
        assert len(bs.history) == 3
        assert bs.history[-1] == {"a": 2.0}

    def test_get_bounds(self):
        ps = ParameterSpec(name="sigma", lower=0.5, upper=5.0)
        bs = BeliefState(params={"sigma": ps})
        low, high = bs.get_bounds("sigma")
        assert low == 0.5
        assert high == 5.0

    def test_get_bounds_missing_raises(self):
        bs = BeliefState()
        with pytest.raises(KeyError):
            bs.get_bounds("nonexistent")

    def test_to_theta_space(self):
        ps_a = ParameterSpec(name="a", lower=0.0, upper=10.0, units="px")
        ps_b = ParameterSpec(name="b", lower=-1.0, upper=1.0, units="rad")
        bs = BeliefState(params={"a": ps_a, "b": ps_b}, theta={"a": 5.0, "b": 0.0})
        ts = bs.to_theta_space()
        assert isinstance(ts, ThetaSpace)
        assert "a" in ts.params
        assert "b" in ts.params
        assert ts.params["a"]["low"] == 0.0
        assert ts.params["a"]["high"] == 10.0
        assert ts.params["b"]["unit"] == "rad"


class TestBuildBeliefFromGraph:
    def test_from_compiled_graph(self):
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec

        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_belief",
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "params": {"sigma": 2.0, "mode": "reflect"},
                 "learnable": ["sigma"]},
                {"node_id": "noise", "primitive_id": "poisson_gaussian",
                 "params": {"peak_photons": 1e4, "read_sigma": 0.01, "seed": 0}},
            ],
            "edges": [{"source": "blur", "target": "noise"}],
        })
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec)
        bs = build_belief_from_graph(graph_op)

        assert "blur.sigma" in bs.theta
        assert bs.theta["blur.sigma"] == 2.0
        assert "blur.sigma" in bs.params

    def test_no_learnable_params(self):
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec

        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_no_learn",
            "nodes": [
                {"node_id": "id", "primitive_id": "identity", "params": {}},
            ],
            "edges": [],
        })
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec)
        bs = build_belief_from_graph(graph_op)
        assert len(bs.theta) == 0
