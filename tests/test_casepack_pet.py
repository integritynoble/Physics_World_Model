"""Tests for PET: template compilation, emission projection, MLEM recon."""

import numpy as np
import pytest

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive
from pwm_core.physics.nuclear.pet_helpers import (
    attenuation_correction,
    mlem_update,
    system_matrix_projection,
)


def _build_pet_v2_graph():
    """Build the pet_graph_v2 canonical template."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "pet_graph_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "pet",
            "x_shape": [32, 32],
            "y_shape": [16, 32],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "generic_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "projection", "primitive_id": "emission_projection",
             "role": "transport", "params": {"n_angles": 16, "x_shape": [32, 32]}},
            {"node_id": "scatter", "primitive_id": "scatter_model",
             "role": "transport", "params": {"scatter_fraction": 0.15, "kernel_sigma": 3.0}},
            {"node_id": "sensor", "primitive_id": "photon_sensor",
             "role": "sensor", "params": {"quantum_efficiency": 0.85, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_only_sensor",
             "role": "noise", "params": {"peak_photons": 100000.0, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "projection"},
            {"source": "projection", "target": "scatter"},
            {"source": "scatter", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    compiler = GraphCompiler()
    return compiler.compile(spec)


class TestPET:
    def test_template_compiles(self):
        """PET v2 template compiles without error."""
        graph = _build_pet_v2_graph()
        assert graph.graph_id == "pet_graph_v2"
        assert len(graph.forward_plan) == 5

    def test_emission_projection_shape(self):
        """EmissionProjection produces sinogram of correct shape."""
        proj = get_primitive("emission_projection", {"n_angles": 16, "x_shape": [32, 32]})
        x = np.random.rand(32, 32)
        y = proj.forward(x)
        assert y.shape == (16, 32)

    def test_forward_output_nonneg(self):
        """PET forward output is non-negative (Poisson noise preserves this)."""
        graph = _build_pet_v2_graph()
        x = np.random.rand(32, 32) * 0.5
        y = graph.forward(x)
        assert np.all(y >= 0)

    def test_emission_projection_adjoint(self):
        """EmissionProjection adjoint produces image of correct shape."""
        proj = get_primitive("emission_projection", {"n_angles": 16, "x_shape": [32, 32]})
        y = np.ones((16, 32))
        x_adj = proj.adjoint(y)
        assert x_adj.shape == (32, 32)


class TestPETHelpers:
    def test_system_matrix_projection_shape(self):
        x = np.random.rand(16, 16)
        sino = system_matrix_projection(x, n_angles=8, n_detectors=16)
        assert sino.shape == (8, 16)

    def test_attenuation_correction(self):
        sino = np.ones((8, 16)) * 100.0
        mu_map = np.ones((8, 16)) * 0.5
        corrected = attenuation_correction(sino, mu_map)
        # exp(0.5) * 100 ~ 164.87
        expected = 100.0 * np.exp(0.5)
        np.testing.assert_allclose(corrected, expected, rtol=1e-10)

    def test_mlem_recon_convergence(self):
        """MLEM should reduce residual over iterations."""
        from pwm_core.graph.primitives import get_primitive
        proj = get_primitive("emission_projection", {"n_angles": 8, "x_shape": [16, 16]})

        x_gt = np.zeros((16, 16))
        x_gt[5:11, 5:11] = 1.0
        y = proj.forward(x_gt) + 0.01  # small background to avoid zeros

        x0 = np.ones((16, 16)) * np.mean(y) / 16  # initial uniform estimate
        x_recon = mlem_update(
            x0, y,
            A_forward=proj.forward,
            A_adjoint=proj.adjoint,
            n_iter=5,
        )
        assert x_recon.shape == (16, 16)
        assert np.all(x_recon >= 0)
        # Reconstructed center should be brighter than edges
        center = np.mean(x_recon[5:11, 5:11])
        edge = np.mean(x_recon[:3, :3])
        assert center > edge
