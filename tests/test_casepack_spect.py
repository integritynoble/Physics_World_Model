"""Tests for SPECT: template compilation, forward sanity."""

import numpy as np
import pytest

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive
from pwm_core.physics.nuclear.spect_helpers import (
    attenuation_correction,
    collimator_projection,
)


def _build_spect_v2_graph():
    """Build the spect_graph_v2 canonical template."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "spect_graph_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "spect",
            "x_shape": [32, 32],
            "y_shape": [16, 32],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "generic_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "projection", "primitive_id": "emission_projection",
             "role": "transport", "params": {"n_angles": 16, "x_shape": [32, 32]}},
            {"node_id": "sensor", "primitive_id": "photon_sensor",
             "role": "sensor", "params": {"quantum_efficiency": 0.85, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_only_sensor",
             "role": "noise", "params": {"peak_photons": 100000.0, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "projection"},
            {"source": "projection", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    compiler = GraphCompiler()
    return compiler.compile(spec)


class TestSPECT:
    def test_template_compiles(self):
        """SPECT v2 template compiles without error."""
        graph = _build_spect_v2_graph()
        assert graph.graph_id == "spect_graph_v2"
        assert len(graph.forward_plan) == 4

    def test_forward_sanity(self):
        """Forward pass produces finite non-negative output."""
        graph = _build_spect_v2_graph()
        x = np.random.rand(32, 32) * 0.5
        y = graph.forward(x)
        assert np.all(np.isfinite(y))
        assert np.all(y >= 0)

    def test_forward_shape(self):
        """Output shape matches (n_angles, n_detectors)."""
        graph = _build_spect_v2_graph()
        x = np.ones((32, 32)) * 0.3
        y = graph.forward(x)
        assert y.shape == (16, 32)

    def test_emission_projection_primitive(self):
        """EmissionProjection primitive used in SPECT works correctly."""
        proj = get_primitive("emission_projection", {"n_angles": 16, "x_shape": [32, 32]})
        x = np.zeros((32, 32))
        x[12:20, 12:20] = 1.0
        y = proj.forward(x)
        assert y.shape == (16, 32)
        assert np.sum(y) > 0


class TestSPECTHelpers:
    def test_collimator_projection_shape(self):
        x = np.random.rand(16, 16)
        sino = collimator_projection(x, n_angles=8, n_detectors=16, collimator_response=1.5)
        assert sino.shape == (8, 16)

    def test_collimator_smoothing(self):
        """Collimator blur should smooth the projection compared to raw."""
        x = np.zeros((16, 16))
        x[8, 8] = 1.0  # point source
        # No collimator blur
        sino_sharp = collimator_projection(x, n_angles=4, n_detectors=16, collimator_response=0.01)
        # With collimator blur
        sino_blurred = collimator_projection(x, n_angles=4, n_detectors=16, collimator_response=3.0)
        # Blurred should have lower peak value
        assert np.max(sino_blurred) < np.max(sino_sharp)

    def test_attenuation_correction_identity(self):
        """Zero attenuation map should return original sinogram."""
        sino = np.ones((8, 16)) * 50.0
        mu_map = np.zeros((8, 16))  # no attenuation
        corrected = attenuation_correction(sino, mu_map)
        np.testing.assert_allclose(corrected, sino, rtol=1e-10)
