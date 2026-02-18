"""Tests for Lensless: template, forward, RL recon, W2 correction."""
import numpy as np
import pytest
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_graph():
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "lensless_graph_v2",
        "metadata": {"canonical_chain": True, "modality": "lensless",
                      "x_shape": [32, 32], "y_shape": [32, 32]},
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source", "role": "source", "params": {"strength": 1.0}},
            {"node_id": "psf_conv", "primitive_id": "conv2d", "role": "transport", "params": {"sigma": 5.0, "mode": "constant"}},
            {"node_id": "sensor", "primitive_id": "photon_sensor", "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "role": "noise", "params": {"peak_photons": 20000.0, "read_sigma": 0.01, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "psf_conv"},
            {"source": "psf_conv", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    return GraphCompiler().compile(spec)


class TestLensless:
    def test_template_compiles(self):
        graph = _build_graph()
        assert graph.graph_id == "lensless_graph_v2"

    def test_forward_sanity(self):
        graph = _build_graph()
        x = np.random.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert y.shape == (32, 32)
        assert np.all(np.isfinite(y))

    def test_rl_recon_psnr(self):
        from pwm_core.recon.richardson_lucy import richardson_lucy_2d
        from pwm_core.core.metric_registry import PSNR
        from scipy import ndimage
        yy, xx = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing='ij')
        x_true = 0.05 + 0.5 * np.exp(-(xx**2 + yy**2) / (2 * 0.3**2))
        x_true = np.clip(x_true, 0, 1).astype(np.float64)
        y = ndimage.gaussian_filter(x_true, sigma=5.0, mode='constant') + np.random.RandomState(42).randn(64, 64) * 0.01
        ax = np.arange(-15, 16, dtype=np.float64)
        xg, yg = np.meshgrid(ax, ax)
        psf = np.exp(-(xg**2 + yg**2) / (2 * 5.0**2))
        psf /= psf.sum()
        x_hat = richardson_lucy_2d(y, psf, iterations=80, clip=True)
        psnr = PSNR()(x_hat.astype(np.float64), x_true, max_val=float(x_true.max()))
        assert psnr > 15, f"RL PSNR {psnr:.1f} < 15"

    def test_w2_nll_decreases(self):
        src = get_primitive("photon_source", {"strength": 1.0})
        sens = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})
        def make_fwd(gain):
            blur = get_primitive("conv2d", {"sigma": 5.0, "mode": "constant"})
            s = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": gain})
            def fwd(x): return s.forward(blur.forward(src.forward(x)))
            return fwd
        rng = np.random.RandomState(42)
        x_true = rng.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        sigma_noise = 0.01
        y_measured = make_fwd(1.3)(x_true) + rng.randn(32, 32) * sigma_noise
        nll_before = float(np.sum((y_measured - make_fwd(1.0)(x_true))**2 / sigma_noise**2))
        best_nll = np.inf
        for tg in np.arange(0.5, 2.0, 0.1):
            nll_t = float(np.sum((y_measured - make_fwd(tg)(x_true))**2 / sigma_noise**2))
            if nll_t < best_nll: best_nll = nll_t
        assert best_nll < nll_before
