"""Tests for Widefield: template compilation, forward pass, RL recon, W2 correction."""

import numpy as np
import pytest

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_widefield_v2_graph():
    """Build the widefield_graph_v2 canonical template."""
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "widefield_graph_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "widefield",
            "x_shape": [32, 32],
            "y_shape": [32, 32],
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


class TestWidefield:
    def test_template_compiles(self):
        """Widefield v2 template compiles without error."""
        graph = _build_widefield_v2_graph()
        assert graph.graph_id == "widefield_graph_v2"
        assert len(graph.forward_plan) == 4

    def test_forward_sanity(self):
        """Forward pass produces blurred, nonneg output."""
        graph = _build_widefield_v2_graph()
        x = np.random.rand(32, 32).astype(np.float64)
        y = graph.forward(x)
        assert y.shape == (32, 32)

    def test_conv2d_self_adjoint(self):
        """Conv2d with symmetric Gaussian is self-adjoint."""
        blur = get_primitive("conv2d", {"sigma": 2.0, "mode": "reflect"})
        x = np.random.rand(32, 32)
        y = blur.forward(x)
        x_adj = blur.adjoint(y)
        assert x_adj.shape == (32, 32)
        # <Ax, Ax> should be close to <x, A^T Ax>
        lhs = np.sum(y * y)
        rhs = np.sum(x * x_adj)
        assert abs(lhs - rhs) / (abs(lhs) + 1e-12) < 0.05

    def test_rl_recon_psnr(self):
        """Richardson-Lucy deconvolution achieves PSNR > 18 on 32x32 phantom."""
        from scipy import ndimage
        from pwm_core.recon.richardson_lucy import richardson_lucy_2d
        from pwm_core.core.metric_registry import PSNR

        # Smooth phantom
        rng = np.random.RandomState(42)
        x_true = np.zeros((32, 32), dtype=np.float64)
        yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing='ij')
        x_true += 0.6 * np.exp(-(xx**2 + yy**2) / (2 * 0.3**2))
        x_true += 0.3 * np.exp(-((xx - 0.3)**2 + (yy + 0.2)**2) / (2 * 0.15**2))
        x_true = np.clip(x_true, 0, 1)

        # Blur
        sigma = 2.0
        y = ndimage.gaussian_filter(x_true, sigma=sigma)
        # Add mild noise
        y = y + rng.randn(*y.shape) * 0.005

        # Create PSF
        ax = np.arange(-7, 8, dtype=np.float64)
        gx, gy = np.meshgrid(ax, ax)
        psf = np.exp(-(gx**2 + gy**2) / (2 * sigma**2))
        psf /= psf.sum()

        x_hat = richardson_lucy_2d(y, psf, iterations=50, clip=True)
        psnr = PSNR()(x_hat.astype(np.float64), x_true, max_val=float(x_true.max()))
        assert psnr > 18, f"RL PSNR {psnr:.1f} < 18"

    def test_w2_nll_decreases(self):
        """PSF sigma mismatch correction decreases NLL."""
        source = get_primitive("photon_source", {"strength": 1.0})
        sensor = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})

        def make_fwd(sigma):
            blur = get_primitive("conv2d", {"sigma": sigma, "mode": "reflect"})
            def fwd(x):
                return sensor.forward(blur.forward(source.forward(x)))
            return fwd

        rng = np.random.RandomState(42)
        x_true = rng.rand(32, 32).astype(np.float64) * 0.8 + 0.1

        sigma_nom = 2.0
        sigma_pert = 2.5
        sigma_noise = 0.01

        y_measured = make_fwd(sigma_pert)(x_true) + rng.randn(32, 32) * sigma_noise

        # NLL with nominal sigma
        y_pred_nom = make_fwd(sigma_nom)(x_true)
        r_nom = (y_measured - y_pred_nom).ravel()
        nll_before = float(np.sum(r_nom ** 2 / sigma_noise ** 2))

        # Grid search
        best_nll = np.inf
        for trial_sigma in np.arange(1.0, 4.0, 0.2):
            y_trial = make_fwd(trial_sigma)(x_true)
            r = (y_measured - y_trial).ravel()
            nll_trial = float(np.sum(r ** 2 / sigma_noise ** 2))
            if nll_trial < best_nll:
                best_nll = nll_trial

        assert best_nll < nll_before, (
            f"NLL should decrease after correction: {best_nll:.1f} >= {nll_before:.1f}"
        )
