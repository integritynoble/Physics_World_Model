"""Tests for ultrasound: template compilation, state chain, forward sanity."""

import numpy as np
import pytest

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive
from pwm_core.physics.ultrasound.ultrasound_helpers import (
    apply_impulse_response,
    delay_and_sum,
    propagate_rf,
)


def _build_ultrasound_v2_graph():
    """Build the ultrasound_graph_v2 canonical template."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "ultrasound_graph_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "ultrasound",
            "x_shape": [64, 64],
            "y_shape": [64],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "acoustic_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "propagate", "primitive_id": "acoustic_propagation",
             "role": "transport", "params": {
                 "speed_of_sound": 1540.0, "n_sensors": 32, "x_shape": [64, 64]}},
            {"node_id": "beamform", "primitive_id": "beamform_delay",
             "role": "transport", "params": {"n_elements": 32}},
            {"node_id": "sensor", "primitive_id": "acoustic_receive_sensor",
             "role": "sensor", "params": {"sensitivity": 1.0}},
            {"node_id": "noise", "primitive_id": "gaussian_sensor_noise",
             "role": "noise", "params": {"sigma": 0.01, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "propagate"},
            {"source": "propagate", "target": "beamform"},
            {"source": "beamform", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    compiler = GraphCompiler()
    return compiler.compile(spec)


class TestUltrasound:
    def test_template_compiles(self):
        """Ultrasound v2 template compiles without error."""
        graph = _build_ultrasound_v2_graph()
        assert graph.graph_id == "ultrasound_graph_v2"
        assert len(graph.forward_plan) == 5

    def test_forward_sanity(self):
        """Forward pass produces finite output."""
        graph = _build_ultrasound_v2_graph()
        x = np.random.rand(64, 64) * 0.5
        y = graph.forward(x)
        assert np.all(np.isfinite(y))
        assert y.size > 0

    def test_state_chain_shapes(self):
        """Verify shapes at each stage of the ultrasound chain."""
        x = np.random.rand(64, 64) * 0.5

        # Stage 1: AcousticSource (passthrough)
        source = get_primitive("acoustic_source", {"strength": 1.0})
        s1 = source.forward(x)
        assert s1.shape == (64, 64)

        # Stage 2: AcousticPropagation (64x64 -> 32x64)
        prop = get_primitive("acoustic_propagation", {
            "speed_of_sound": 1540.0, "n_sensors": 32, "x_shape": [64, 64]})
        s2 = prop.forward(s1)
        assert s2.shape[0] == 32  # n_sensors angles
        assert s2.ndim == 2

        # Stage 3: BeamformDelay (mean across axis 0)
        bf = get_primitive("beamform_delay", {"n_elements": 32})
        s3 = bf.forward(s2)
        assert s3.ndim == 1

        # Stage 4: AcousticReceiveSensor (passthrough with sensitivity)
        recv = get_primitive("acoustic_receive_sensor", {"sensitivity": 1.0})
        s4 = recv.forward(s3)
        assert s4.shape == s3.shape

    def test_beamformed_output_shape(self):
        """Beamformed output is 1D (column-mean of propagation result)."""
        graph = _build_ultrasound_v2_graph()
        x = np.random.rand(64, 64) * 0.5
        y = graph.forward(x)
        assert y.ndim == 1


class TestUltrasoundHelpers:
    def test_propagate_rf_shape(self):
        """propagate_rf returns (n_elements, n_samples) shape."""
        x = np.zeros((8, 8))
        x[4, 4] = 1.0  # point scatterer
        rf = propagate_rf(x, speed=1540.0, n_elements=4, element_pitch=0.3e-3,
                          n_samples=64, fs=40e6)
        assert rf.shape == (4, 64)

    def test_delay_and_sum_shape(self):
        """delay_and_sum returns expected grid shape."""
        rf = np.random.rand(4, 64)
        img = delay_and_sum(rf, speed=1540.0, focus_depth=0.01,
                            element_pitch=0.3e-3, pixel_pitch=0.3e-3,
                            grid_shape=(8, 8))
        assert img.shape == (8, 8)

    def test_apply_impulse_response_preserves_shape(self):
        """apply_impulse_response does not change RF shape."""
        rf = np.random.rand(4, 64)
        ir = np.array([0.25, 0.5, 0.25])
        filtered = apply_impulse_response(rf, ir)
        assert filtered.shape == rf.shape

    def test_point_scatterer_nonzero_rf(self):
        """A point scatterer should produce non-zero RF data."""
        x = np.zeros((8, 8))
        x[4, 4] = 1.0
        rf = propagate_rf(x, speed=1540.0, n_elements=4, element_pitch=0.3e-3,
                          n_samples=64, fs=40e6)
        assert np.any(rf != 0)
