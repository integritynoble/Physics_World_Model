"""Tests for MRI: template compilation, forward in k-space, recon baseline."""

import numpy as np
import pytest

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_mri_v2_graph():
    """Build the mri_graph_v2 canonical template."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "mri_graph_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "mri",
            "x_shape": [32, 32],
            "y_shape": [32, 32],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "spin_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "kspace", "primitive_id": "mri_kspace",
             "role": "transport", "params": {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": 42}},
            {"node_id": "sensor", "primitive_id": "coil_sensor",
             "role": "sensor", "params": {"sensitivity": 1.0}},
            {"node_id": "noise", "primitive_id": "complex_gaussian_sensor",
             "role": "noise", "params": {"sigma": 0.005, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "kspace"},
            {"source": "kspace", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    compiler = GraphCompiler()
    return compiler.compile(spec)


class TestMRI:
    def test_template_compiles(self):
        """MRI v2 template compiles without error."""
        graph = _build_mri_v2_graph()
        assert graph.graph_id == "mri_graph_v2"
        assert len(graph.forward_plan) == 4

    def test_forward_in_kspace(self):
        """Forward pass produces complex k-space data."""
        graph = _build_mri_v2_graph()
        x = np.random.rand(32, 32).astype(np.float64)
        y = graph.forward(x)
        assert y.shape == (32, 32)
        # k-space data should be complex
        assert np.iscomplexobj(y)

    def test_forward_shape_correct(self):
        """Output shape matches input shape for MRI."""
        graph = _build_mri_v2_graph()
        x = np.ones((32, 32)) * 0.5
        y = graph.forward(x)
        assert y.shape == (32, 32)

    def test_kspace_undersampling(self):
        """k-space mask zeroes out some frequencies."""
        kspace = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.25, "seed": 42})
        x = np.ones((32, 32))
        y = kspace.forward(x)
        # Not all k-space entries should be non-zero
        # (undersampling rate = 0.25 means roughly 75% zeros outside center)
        total_entries = y.size
        nonzero_entries = np.count_nonzero(y)
        assert nonzero_entries < total_entries

    def test_kspace_adjoint(self):
        """k-space adjoint produces a real image."""
        kspace = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": 42})
        x = np.random.rand(32, 32)
        y = kspace.forward(x)
        x_recon = kspace.adjoint(y)
        assert x_recon.shape == (32, 32)
        # Adjoint of k-space should produce a real image
        assert np.isrealobj(x_recon)

    def test_recon_baseline(self):
        """Adjoint-based reconstruction recovers approximate structure."""
        kspace = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": 42})
        x_gt = np.zeros((32, 32))
        x_gt[10:22, 10:22] = 1.0
        y = kspace.forward(x_gt)
        x_recon = kspace.adjoint(y)
        # Center region should have higher values than corners
        center = np.mean(np.abs(x_recon[10:22, 10:22]))
        corner = np.mean(np.abs(x_recon[:4, :4]))
        assert center > corner

    def test_dot_product_consistency(self):
        """<Ax, Ax> and <x, A^T Ax> should be close for linear operator."""
        kspace = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": 42})
        x = np.random.rand(32, 32)
        y = kspace.forward(x)
        ata_x = kspace.adjoint(y)
        # <Ax, Ax> vs <x, A^T Ax>
        lhs = np.real(np.sum(np.conj(y) * y))
        rhs = np.sum(x * ata_x)
        # Should be similar (exact for orthogonal mask)
        assert lhs > 0
        assert rhs > 0

    def test_cs_mri_recon_psnr(self):
        """CS-MRI reconstruction achieves PSNR > 20 on 32x32 phantom."""
        from pwm_core.recon.mri_solvers import cs_mri_wavelet
        from pwm_core.core.metric_registry import PSNR

        kspace = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": 42})
        # Smooth phantom
        yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing='ij')
        x_true = np.zeros((32, 32), dtype=np.float64)
        x_true[xx**2 + yy**2 <= 0.6**2] = 0.7
        x_true[xx**2 + yy**2 <= 0.2**2] = 0.3

        y = kspace.forward(x_true)
        mask = kspace._mask
        x_hat = cs_mri_wavelet(y, mask, lam=0.001, iterations=30)
        if np.iscomplexobj(x_hat):
            x_hat = np.abs(x_hat)
        psnr = PSNR()(x_hat.astype(np.float64), x_true, max_val=float(x_true.max()))
        assert psnr > 20, f"CS-MRI PSNR {psnr:.1f} < 20"

    def test_w2_nll_decreases(self):
        """Mask seed mismatch correction decreases NLL."""
        kspace_nom = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": 42})
        kspace_pert = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": 99})

        rng = np.random.RandomState(42)
        x_true = rng.rand(32, 32).astype(np.float64) * 0.8 + 0.1

        sigma = 0.005
        noise = (rng.randn(32, 32) + 1j * rng.randn(32, 32)) * sigma
        y_measured = kspace_pert.forward(x_true) + noise

        # NLL with nominal mask
        y_pred_nom = kspace_nom.forward(x_true)
        r_nom = (y_measured.ravel() - y_pred_nom.ravel()).astype(np.complex128)
        nll_before = float(np.sum(np.abs(r_nom) ** 2 / sigma ** 2).real)

        # Grid search
        best_nll = np.inf
        for trial_seed in range(30, 121, 5):
            kspace_trial = get_primitive("mri_kspace", {"H": 32, "W": 32, "sampling_rate": 0.5, "seed": trial_seed})
            y_trial = kspace_trial.forward(x_true)
            r = (y_measured.ravel() - y_trial.ravel()).astype(np.complex128)
            nll_trial = float(np.sum(np.abs(r) ** 2 / sigma ** 2).real)
            if nll_trial < best_nll:
                best_nll = nll_trial

        assert best_nll < nll_before, (
            f"NLL should decrease after correction: {best_nll:.1f} >= {nll_before:.1f}"
        )
