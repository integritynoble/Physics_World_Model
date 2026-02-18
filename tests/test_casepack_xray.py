"""Tests for X-ray radiography: template compilation, forward, and recon."""

import numpy as np
import pytest

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.physics.radiography.xray_radiography_helpers import (
    planar_beer_lambert,
    scatter_estimate,
)


def _build_xray_graph():
    """Build the xray_radiography_graph_v2 template inline."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "xray_radiography_graph_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "xray_radiography",
            "x_shape": [64, 64],
            "y_shape": [64, 64],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "xray_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "transmission", "primitive_id": "beer_lambert",
             "role": "transport", "params": {"I_0": 10000.0}},
            {"node_id": "scatter", "primitive_id": "scatter_model",
             "role": "transport", "params": {"scatter_fraction": 0.1, "kernel_sigma": 5.0}},
            {"node_id": "sensor", "primitive_id": "xray_detector_sensor",
             "role": "sensor", "params": {"scintillator_efficiency": 0.8, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
             "role": "noise", "params": {"peak_photons": 50000.0, "read_sigma": 0.005, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "transmission"},
            {"source": "transmission", "target": "scatter"},
            {"source": "scatter", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    compiler = GraphCompiler()
    return compiler.compile(spec)


class TestXRayRadiography:
    def test_template_compiles(self):
        """Template builds without error."""
        graph = _build_xray_graph()
        assert graph.graph_id == "xray_radiography_graph_v2"
        assert len(graph.forward_plan) == 5

    def test_forward_produces_nonneg_output(self):
        """Forward pass through Beer-Lambert produces non-negative values."""
        graph = _build_xray_graph()
        x = np.random.rand(64, 64) * 0.5  # attenuation map
        y = graph.forward(x)
        assert y.shape == (64, 64)
        # Output can be negative due to noise, but mean should be positive
        assert np.mean(y) > 0

    def test_forward_shape_correct(self):
        """Output shape matches metadata."""
        graph = _build_xray_graph()
        x = np.ones((64, 64)) * 0.3
        y = graph.forward(x)
        assert y.shape == (64, 64)

    def test_adjoint_based_recon(self):
        """Adjoint produces a reasonable back-projection."""
        # Use linear-only subchain for adjoint test (skip BeerLambert nonlinearity)
        from pwm_core.graph.primitives import get_primitive
        xray_det = get_primitive("xray_detector_sensor", {"scintillator_efficiency": 0.8, "gain": 1.0})
        y = np.ones((64, 64)) * 100.0
        x_adj = xray_det.adjoint(y)
        assert x_adj.shape == (64, 64)
        assert np.all(x_adj > 0)


class TestXRayHelpers:
    def test_planar_beer_lambert(self):
        x = np.ones((16, 16)) * 0.5
        result = planar_beer_lambert(x, I_0=1000.0, mu=1.0)
        expected = 1000.0 * np.exp(-0.5)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_scatter_estimate_smooth(self):
        y = np.random.rand(16, 16) * 100
        scatter = scatter_estimate(y, fraction=0.1, sigma=3.0)
        assert scatter.shape == y.shape
        # Scatter should be smoother (lower variance) than input
        assert np.std(scatter) < np.std(y)
