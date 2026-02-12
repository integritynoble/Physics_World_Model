"""Tests for Confocal 3D: template, 3D RL recon, W2 correction."""
import numpy as np
import pytest
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_graph():
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "confocal_3d_graph_v2",
        "metadata": {"canonical_chain": True, "modality": "confocal_3d",
                      "x_shape": [8, 16, 16], "y_shape": [8, 16, 16]},
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source", "role": "source", "params": {"strength": 1.0}},
            {"node_id": "blur_3d", "primitive_id": "conv3d", "role": "transport", "params": {"sigma": [3.0, 1.5, 1.5], "mode": "reflect"}},
            {"node_id": "sensor", "primitive_id": "photon_sensor", "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "role": "noise", "params": {"peak_photons": 8000.0, "read_sigma": 0.02, "seed": 0}},
        ],
        "edges": [{"source": "source", "target": "blur_3d"}, {"source": "blur_3d", "target": "sensor"}, {"source": "sensor", "target": "noise"}],
    })
    return GraphCompiler().compile(spec)


class TestConfocal3d:
    def test_template_compiles(self):
        graph = _build_graph()
        assert graph.graph_id == "confocal_3d_graph_v2"

    def test_forward_3d(self):
        graph = _build_graph()
        y = graph.forward(np.random.rand(8, 16, 16).astype(np.float64))
        assert y.shape == (8, 16, 16)

    def test_rl3d_recon_psnr(self):
        from pwm_core.recon.richardson_lucy import richardson_lucy_3d
        from pwm_core.core.metric_registry import PSNR
        rng = np.random.RandomState(42)
        x_true = np.zeros((8,16,16), dtype=np.float64)
        zz,yy,xx = np.meshgrid(np.linspace(-1,1,8), np.linspace(-1,1,16), np.linspace(-1,1,16), indexing='ij')
        x_true += 0.6 * np.exp(-(xx**2+yy**2+zz**2)/(2*0.4**2))
        x_true = np.clip(x_true, 0, 1)
        from scipy import ndimage
        y = ndimage.gaussian_filter(x_true, sigma=[3.0,1.5,1.5]) + rng.randn(8,16,16)*0.01
        sigma = [3.0,1.5,1.5]; sz=5
        ax = np.arange(-sz//2+1, sz//2+1, dtype=np.float64)
        zg,yg,xg = np.meshgrid(ax,ax,ax,indexing='ij')
        psf = np.exp(-(zg**2/(2*sigma[0]**2)+yg**2/(2*sigma[1]**2)+xg**2/(2*sigma[2]**2)))
        psf /= psf.sum()
        x_hat = richardson_lucy_3d(y, psf, iterations=20, clip=True)
        psnr = PSNR()(x_hat.astype(np.float64), x_true, max_val=float(x_true.max()))
        assert psnr > 15, f"RL3D PSNR {psnr:.1f} < 15"

    def test_w2_nll_decreases(self):
        src = get_primitive("photon_source", {"strength": 1.0})
        sens = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})
        def make_fwd(lat_sigma):
            blur = get_primitive("conv3d", {"sigma": [3.0, lat_sigma, lat_sigma], "mode": "reflect"})
            def fwd(x): return sens.forward(blur.forward(src.forward(x)))
            return fwd
        rng = np.random.RandomState(42)
        x_true = rng.rand(8,16,16).astype(np.float64)*0.8+0.1
        sigma_noise = 0.02
        y_measured = make_fwd(2.5)(x_true) + rng.randn(8,16,16)*sigma_noise
        nll_before = float(np.sum((y_measured - make_fwd(1.5)(x_true))**2/sigma_noise**2))
        best_nll = np.inf
        for ts in np.arange(1.0, 3.0, 0.3):
            nll_t = float(np.sum((y_measured - make_fwd(ts)(x_true))**2/sigma_noise**2))
            if nll_t < best_nll: best_nll = nll_t
        assert best_nll < nll_before
