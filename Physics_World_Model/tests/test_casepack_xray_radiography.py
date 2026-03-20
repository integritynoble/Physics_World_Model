"""Tests for X-ray Radiography: template, forward, recon, W2 correction."""
import numpy as np
import pytest
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_graph():
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "xray_radiography_graph_v2",
        "metadata": {"canonical_chain": True, "modality": "xray_radiography",
                      "x_shape": [32, 32], "y_shape": [32, 32]},
        "nodes": [
            {"node_id": "source", "primitive_id": "xray_source", "role": "source", "params": {"strength": 1.0}},
            {"node_id": "transmission", "primitive_id": "beer_lambert", "role": "transport", "params": {"I_0": 10000.0}},
            {"node_id": "scatter", "primitive_id": "scatter_model", "role": "transport", "params": {"scatter_fraction": 0.1, "kernel_sigma": 5.0}},
            {"node_id": "sensor", "primitive_id": "xray_detector_sensor", "role": "sensor", "params": {"scintillator_efficiency": 0.8, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor", "role": "noise", "params": {"peak_photons": 50000.0, "read_sigma": 0.005, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "transmission"},
            {"source": "transmission", "target": "scatter"},
            {"source": "scatter", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    return GraphCompiler().compile(spec)


class TestXrayRadiography:
    def test_template_compiles(self):
        graph = _build_graph()
        assert graph.graph_id == "xray_radiography_graph_v2"

    def test_forward_sanity(self):
        graph = _build_graph()
        x = np.random.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert y.shape == (32, 32)
        assert np.all(np.isfinite(y))

    def test_beer_lambert_inversion(self):
        from pwm_core.core.metric_registry import PSNR
        src = get_primitive("xray_source", {"strength": 1.0})
        bl = get_primitive("beer_lambert", {"I_0": 10000.0})
        sens = get_primitive("xray_detector_sensor", {"scintillator_efficiency": 0.8, "gain": 1.0})
        yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing='ij')
        x_true = 0.05 + 0.5 * np.exp(-(xx**2 + yy**2) / (2 * 0.3**2))
        x_true = np.clip(x_true, 0, 1).astype(np.float64)
        y = sens.forward(bl.forward(src.forward(x_true)))
        x_hat = -np.log(np.clip(y / (0.8 * 10000.0), 1e-8, None))
        if x_hat.max() > 0:
            x_hat = x_hat / x_hat.max() * x_true.max()
        psnr = PSNR()(x_hat.astype(np.float64), x_true, max_val=float(x_true.max()))
        assert psnr > 15, f"BL inversion PSNR {psnr:.1f} < 15"

    def test_w2_nll_decreases(self):
        src = get_primitive("xray_source", {"strength": 1.0})
        bl = get_primitive("beer_lambert", {"I_0": 10000.0})
        scat = get_primitive("scatter_model", {"scatter_fraction": 0.1, "kernel_sigma": 5.0})
        def make_fwd(gain):
            s = get_primitive("xray_detector_sensor", {"scintillator_efficiency": 0.8, "gain": gain})
            def fwd(x): return s.forward(scat.forward(bl.forward(src.forward(x))))
            return fwd
        rng = np.random.RandomState(42)
        x_true = rng.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        sigma_noise = 0.005
        y_measured = make_fwd(1.3)(x_true) + rng.randn(32, 32) * sigma_noise
        nll_before = float(np.sum((y_measured - make_fwd(1.0)(x_true))**2 / sigma_noise**2))
        best_nll = np.inf
        for tg in np.arange(0.5, 2.0, 0.1):
            nll_t = float(np.sum((y_measured - make_fwd(tg)(x_true))**2 / sigma_noise**2))
            if nll_t < best_nll: best_nll = nll_t
        assert best_nll < nll_before
