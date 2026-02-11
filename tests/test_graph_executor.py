"""Tests for pwm_core.graph.executor: Mode S/I/C + operator correction."""

import numpy as np
import pytest

from pwm_core.core.enums import ExecutionMode
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.executor import ExecutionConfig, ExecutionResult, GraphExecutor
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.objectives.base import ObjectiveSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_widefield_graph():
    """Build a simple canonical widefield graph for testing."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "test_widefield_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "widefield",
            "x_shape": [64, 64],
            "y_shape": [64, 64],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "blur", "primitive_id": "conv2d",
             "role": "transport", "params": {"sigma": 2.0, "mode": "reflect"},
             "learnable": ["sigma"]},
            {"node_id": "sensor", "primitive_id": "photon_sensor",
             "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
             "role": "noise", "params": {"peak_photons": 10000.0, "read_sigma": 0.01, "seed": 42}},
        ],
        "edges": [
            {"source": "source", "target": "blur"},
            {"source": "blur", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    compiler = GraphCompiler()
    return compiler.compile(spec)


def _build_linear_graph():
    """Build a simple linear canonical graph (no nonlinear noise) for adjoint tests."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "test_linear_v2",
        "metadata": {
            "canonical_chain": True,
            "x_shape": [64, 64],
            "y_shape": [64, 64],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "blur", "primitive_id": "conv2d",
             "role": "transport", "params": {"sigma": 2.0, "mode": "reflect"}},
            {"node_id": "sensor", "primitive_id": "photon_sensor",
             "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
             "role": "noise", "params": {"peak_photons": 10000.0, "read_sigma": 0.01, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "blur"},
            {"source": "blur", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    compiler = GraphCompiler()
    return compiler.compile(spec)


# ---------------------------------------------------------------------------
# Mode S: Simulate
# ---------------------------------------------------------------------------


class TestModeSimulate:
    def test_simulate_produces_output(self):
        graph = _build_widefield_graph()
        executor = GraphExecutor(graph)
        x = np.random.rand(64, 64)
        config = ExecutionConfig(mode=ExecutionMode.simulate, seed=42)
        result = executor.execute(x=x, config=config)
        assert result.mode == ExecutionMode.simulate
        assert result.y is not None
        assert result.y.shape == (64, 64)

    def test_simulate_adds_noise(self):
        graph = _build_widefield_graph()
        executor = GraphExecutor(graph)
        x = np.ones((64, 64)) * 0.5

        result_noisy = executor.execute(
            x=x, config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=True)
        )
        result_clean = executor.execute(
            x=x, config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False)
        )
        # Clean and noisy should differ
        assert not np.allclose(result_noisy.y, result_clean.y)

    def test_simulate_requires_x(self):
        graph = _build_widefield_graph()
        executor = GraphExecutor(graph)
        with pytest.raises(ValueError, match="requires input x"):
            executor.execute(config=ExecutionConfig(mode=ExecutionMode.simulate))

    def test_simulate_stores_y_clean(self):
        graph = _build_widefield_graph()
        executor = GraphExecutor(graph)
        x = np.random.rand(64, 64)
        result = executor.execute(
            x=x, config=ExecutionConfig(mode=ExecutionMode.simulate, seed=0)
        )
        assert "y_clean" in result.diagnostics


# ---------------------------------------------------------------------------
# Mode I: Invert
# ---------------------------------------------------------------------------


class TestModeInvert:
    def test_invert_produces_reconstruction(self):
        graph = _build_linear_graph()
        executor = GraphExecutor(graph)

        # Generate measurement
        x_true = np.random.rand(64, 64)
        sim_result = executor.execute(
            x=x_true,
            config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False),
        )

        # Reconstruct
        config = ExecutionConfig(
            mode=ExecutionMode.invert,
            solver_ids=["lsq"],
        )
        result = executor.execute(y=sim_result.y, config=config)
        assert result.mode == ExecutionMode.invert
        assert result.x_recon is not None
        assert result.x_recon.shape == (64, 64)

    def test_invert_requires_y(self):
        graph = _build_linear_graph()
        executor = GraphExecutor(graph)
        with pytest.raises(ValueError, match="requires measurement y"):
            executor.execute(config=ExecutionConfig(mode=ExecutionMode.invert))


# ---------------------------------------------------------------------------
# Objective inference
# ---------------------------------------------------------------------------


class TestObjectiveInference:
    def test_poisson_gaussian_sensor_infers_mixed(self):
        graph = _build_widefield_graph()
        executor = GraphExecutor(graph)
        obj = executor._infer_objective_from_noise()
        assert obj.kind == "mixed_poisson_gaussian"

    def test_no_noise_node_defaults_gaussian(self):
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_no_noise",
            "metadata": {"x_shape": [64, 64], "y_shape": [64, 64]},
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "params": {"sigma": 2.0, "mode": "reflect"}},
            ],
            "edges": [],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        executor = GraphExecutor(graph)
        obj = executor._infer_objective_from_noise()
        assert obj.kind == "gaussian"


# ---------------------------------------------------------------------------
# Operator correction mode
# ---------------------------------------------------------------------------


class TestOperatorCorrection:
    def test_correction_with_provided_operator(self):
        """Use a provided_operator instead of graph-derived A."""
        graph = _build_linear_graph()
        executor = GraphExecutor(graph)

        # Generate measurement from graph
        x_true = np.random.rand(64, 64)
        sim = executor.execute(
            x=x_true,
            config=ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False),
        )

        # Build a simple provided operator
        class SimpleOp:
            def forward(self, x):
                from scipy import ndimage
                return ndimage.gaussian_filter(x.astype(np.float64), sigma=2.0)

            def adjoint(self, y):
                from scipy import ndimage
                return ndimage.gaussian_filter(y.astype(np.float64), sigma=2.0)

            def info(self):
                return {"operator_id": "simple_op", "is_linear": True}

            @property
            def _x_shape(self):
                return (64, 64)

            @property
            def _y_shape(self):
                return (64, 64)

            @property
            def _is_linear(self):
                return True

        config = ExecutionConfig(
            mode=ExecutionMode.invert,
            solver_ids=["lsq"],
            provided_operator=SimpleOp(),
        )
        result = executor.execute(y=sim.y, config=config)
        assert result.x_recon is not None


# ---------------------------------------------------------------------------
# BeliefState integration
# ---------------------------------------------------------------------------


class TestExecutorBeliefState:
    def test_belief_state_exists(self):
        graph = _build_widefield_graph()
        executor = GraphExecutor(graph)
        bs = executor.belief_state
        assert bs is not None
        # Widefield has learnable sigma
        assert "blur.sigma" in bs.theta
