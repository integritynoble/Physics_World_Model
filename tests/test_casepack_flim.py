"""Tests for FLIM: template, forward, RL recon, W2 correction."""
import numpy as np
import pytest
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_graph():
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "flim_graph_v2",
        "metadata": {"canonical_chain": True, "modality": "flim",
                      "x_shape": [32, 32], "y_shape": [32, 32]},
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source", "role": "source", "params": {"strength": 1.0}},
            {"node_id": "blur", "primitive_id": "conv2d", "role": "transport", "params": {"sigma": 2.0, "mode": "reflect"}},
            {"node_id": "gate", "primitive_id": "temporal_mask", "role": "transport", "params": {"H": 32, "W": 32, "T": 1, "seed": 42}},
            {"node_id": "sensor", "primitive_id": "photon_sensor", "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "role": "noise", "params": {"peak_photons": 3000.0, "read_sigma": 0.03, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "blur"},
            {"source": "blur", "target": "gate"},
            {"source": "gate", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    return GraphCompiler().compile(spec)


class TestFLIM:
    def test_template_compiles(self):
        graph = _build_graph()
        assert graph.graph_id == "flim_graph_v2"

    def test_forward_sanity(self):
        graph = _build_graph()
        x = np.random.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert y.shape == (32, 32)
        assert np.all(np.isfinite(y))

    def test_temporal_mask_modulation(self):
        gate = get_primitive("temporal_mask", {"H": 32, "W": 32, "T": 1, "seed": 42})
        x = np.ones((32, 32), dtype=np.float64)
        y = gate.forward(x)
        # Mask should have binary values (0 or 1)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_w2_nll_decreases(self):
        src = get_primitive("photon_source", {"strength": 1.0})
        blur = get_primitive("conv2d", {"sigma": 2.0, "mode": "reflect"})
        gate = get_primitive("temporal_mask", {"H": 32, "W": 32, "T": 1, "seed": 42})
        def make_fwd(gain):
            s = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": gain})
            def fwd(x): return s.forward(gate.forward(blur.forward(src.forward(x))))
            return fwd
        rng = np.random.RandomState(42)
        x_true = rng.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        sigma_noise = 0.03
        y_measured = make_fwd(1.4)(x_true) + rng.randn(32, 32) * sigma_noise
        nll_before = float(np.sum((y_measured - make_fwd(1.0)(x_true))**2 / sigma_noise**2))
        best_nll = np.inf
        for tg in np.arange(0.5, 2.0, 0.1):
            nll_t = float(np.sum((y_measured - make_fwd(tg)(x_true))**2 / sigma_noise**2))
            if nll_t < best_nll: best_nll = nll_t
        assert best_nll < nll_before
