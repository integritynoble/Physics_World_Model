"""Tests for OCT: template, forward, adjoint recon, W2 correction."""
import numpy as np
import pytest
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_graph():
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "oct_graph_v2",
        "metadata": {"canonical_chain": True, "modality": "oct",
                      "x_shape": [32, 32], "y_shape": [32, 32]},
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source", "role": "source", "params": {"strength": 1.0}},
            {"node_id": "prop", "primitive_id": "angular_spectrum", "role": "transport", "params": {"wavelength": 1.3e-6, "distance": 1.0e-3, "pixel_size": 5.0e-6}},
            {"node_id": "sensor", "primitive_id": "photon_sensor", "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "role": "noise", "params": {"peak_photons": 50000.0, "read_sigma": 0.005, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "prop"},
            {"source": "prop", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    return GraphCompiler().compile(spec)


class TestOCT:
    def test_template_compiles(self):
        graph = _build_graph()
        assert graph.graph_id == "oct_graph_v2"

    def test_forward_sanity(self):
        graph = _build_graph()
        x = np.random.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert y.shape == (32, 32)
        assert np.all(np.isfinite(y))

    def test_adjoint_recon_psnr(self):
        from pwm_core.core.metric_registry import PSNR
        yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing='ij')
        x_true = 0.05 + 0.5 * np.exp(-(xx**2 + yy**2) / (2 * 0.3**2))
        x_true = np.clip(x_true, 0, 1).astype(np.float64)
        src = get_primitive("photon_source", {"strength": 1.0})
        prop = get_primitive("angular_spectrum", {"wavelength": 1.3e-6, "distance": 1.0e-3, "pixel_size": 5.0e-6})
        sens = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})
        y = sens.forward(prop.forward(src.forward(x_true)))
        y = np.real(y) if np.iscomplexobj(y) else y
        y += np.random.RandomState(42).randn(*y.shape) * 0.005
        x_hat = src.adjoint(prop.adjoint(sens.adjoint(y)))
        x_hat = np.real(x_hat) if np.iscomplexobj(x_hat) else x_hat
        x_hat = np.clip(x_hat, 0, None)
        if x_hat.max() > 0:
            x_hat = x_hat / x_hat.max() * x_true.max()
        psnr = PSNR()(x_hat.astype(np.float64), x_true, max_val=float(x_true.max()))
        assert psnr > 8, f"Adjoint PSNR {psnr:.1f} < 8"

    def test_w2_nll_decreases(self):
        src = get_primitive("photon_source", {"strength": 1.0})
        def make_fwd(dist):
            p = get_primitive("angular_spectrum", {"wavelength": 1.3e-6, "distance": dist, "pixel_size": 5.0e-6})
            s = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": 1.0})
            def fwd(x):
                out = s.forward(p.forward(src.forward(x)))
                return np.real(out) if np.iscomplexobj(out) else out
            return fwd
        rng = np.random.RandomState(42)
        x_true = rng.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        sigma_noise = 0.005
        y_measured = make_fwd(1.5e-3)(x_true) + rng.randn(32, 32) * sigma_noise
        nll_before = float(np.sum((y_measured - make_fwd(1.0e-3)(x_true))**2 / sigma_noise**2))
        best_nll = np.inf
        for td in np.linspace(0.5e-3, 2.5e-3, 21):
            nll_t = float(np.sum((y_measured - make_fwd(td)(x_true))**2 / sigma_noise**2))
            if nll_t < best_nll: best_nll = nll_t
        assert best_nll < nll_before
