"""Tests for multi-input graph nodes and DAG executor."""
import numpy as np
import pytest
from pwm_core.graph.compiler import GraphCompiler, GraphCompilationError
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import Interference, BasePrimitive


class TestMultiInput:
    def test_interference_forward_multi(self):
        prim = Interference()
        signal = np.ones((8, 8)) * 2.0
        ref = np.ones((8, 8)) * 1.0
        result = prim.forward_multi({"signal": signal, "reference": ref})
        expected = np.abs(signal + ref) ** 2
        np.testing.assert_allclose(result, expected)

    def test_interference_n_inputs(self):
        prim = Interference()
        assert prim._n_inputs == 2

    def test_base_primitive_default_n_inputs(self):
        """All existing primitives should have _n_inputs = 1."""
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        for pid, cls in PRIMITIVE_REGISTRY.items():
            if pid == "interference":
                continue
            inst = cls()
            assert getattr(inst, '_n_inputs', 1) == 1, f"{pid} has unexpected _n_inputs"

    def test_forward_multi_default_delegates(self):
        """BasePrimitive.forward_multi should delegate to forward()."""
        from pwm_core.graph.primitives import Identity
        prim = Identity()
        x = np.random.randn(8, 8)
        result = prim.forward_multi({"input": x})
        np.testing.assert_array_equal(result, x)

    def test_existing_linear_graph_unchanged(self):
        """Ensure existing single-input graphs produce identical results."""
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_compat",
            "metadata": {"x_shape": [32, 32], "y_shape": [32, 32]},
            "nodes": [
                {"node_id": "blur", "primitive_id": "conv2d",
                 "params": {"sigma": 2.0, "mode": "reflect"}},
            ],
            "edges": [],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        x = np.random.RandomState(42).randn(32, 32)
        y = graph.forward(x)
        assert y.shape == (32, 32)
        # Should be a blurred version
        assert not np.allclose(y, x)

    def test_edge_map_populated(self):
        """Compiler should populate edge_map on GraphOperator."""
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_edge_map",
            "nodes": [
                {"node_id": "a", "primitive_id": "identity", "params": {}},
                {"node_id": "b", "primitive_id": "identity", "params": {}},
            ],
            "edges": [{"source": "a", "target": "b"}],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert "b" in graph.edge_map
        assert graph.edge_map["b"] == ["a"]

    def test_multi_input_validation_fails(self):
        """Compiler should reject graphs with wrong number of inputs for multi-input nodes."""
        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_bad_multi",
            "nodes": [
                {"node_id": "a", "primitive_id": "identity", "params": {}},
                {"node_id": "b", "primitive_id": "interference", "params": {}},
            ],
            "edges": [{"source": "a", "target": "b"}],  # Only 1 edge, but interference needs 2
        })
        compiler = GraphCompiler()
        with pytest.raises(GraphCompilationError, match="expects 2 inputs"):
            compiler.compile(spec)

    def test_multi_input_dag_execution(self):
        """End-to-end test: multi-input node in a DAG with GraphExecutor."""
        from pwm_core.graph.executor import GraphExecutor, ExecutionConfig, ExecutionMode

        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_interference_dag",
            "metadata": {"x_shape": [8, 8], "y_shape": [8, 8]},
            "nodes": [
                {"node_id": "signal", "primitive_id": "identity", "params": {}},
                {"node_id": "reference", "primitive_id": "identity", "params": {}},
                {"node_id": "interfere", "primitive_id": "interference", "params": {}},
            ],
            "edges": [
                {"source": "signal", "target": "interfere"},
                {"source": "reference", "target": "interfere"},
            ],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        executor = GraphExecutor(graph)

        # Test DAG execution via executor
        x = np.ones((8, 8)) * 2.0
        config = ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False)
        result = executor.execute(x=x, config=config)

        # Both signal and reference get x, so result should be |2+2|^2 = 16
        expected = np.ones((8, 8)) * 16.0
        np.testing.assert_allclose(result.y, expected)

    def test_multi_input_with_executor(self):
        """Test multi-input node with GraphExecutor."""
        from pwm_core.graph.executor import GraphExecutor, ExecutionConfig, ExecutionMode

        spec = OperatorGraphSpec.model_validate({
            "graph_id": "test_executor_multi",
            "metadata": {"x_shape": [8, 8], "y_shape": [8, 8]},
            "nodes": [
                {"node_id": "signal", "primitive_id": "identity", "params": {}},
                {"node_id": "reference", "primitive_id": "identity", "params": {}},
                {"node_id": "interfere", "primitive_id": "interference", "params": {}},
            ],
            "edges": [
                {"source": "signal", "target": "interfere"},
                {"source": "reference", "target": "interfere"},
            ],
        })
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        executor = GraphExecutor(graph)

        x = np.ones((8, 8)) * 3.0
        config = ExecutionConfig(mode=ExecutionMode.simulate, add_noise=False)
        result = executor.execute(x=x, config=config)

        # Both inputs get x=3, so |3+3|^2 = 36
        expected = np.ones((8, 8)) * 36.0
        np.testing.assert_allclose(result.y, expected)
