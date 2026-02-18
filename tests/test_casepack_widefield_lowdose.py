"""Tests for Widefield Low-Dose: template compilation, RL recon, W2 correction."""

import numpy as np
import pytest

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import get_primitive


def _build_widefield_lowdose_v2_graph():
    spec = OperatorGraphSpec.model_validate({
        "graph_id": "widefield_lowdose_graph_v2",
        "metadata": {
            "canonical_chain": True,
            "modality": "widefield_lowdose",
            "x_shape": [32, 32],
            "y_shape": [32, 32],
        },
        "nodes": [
            {"node_id": "source", "primitive_id": "photon_source",
             "role": "source", "params": {"strength": 1.0}},
            {"node_id": "blur", "primitive_id": "conv2d",
             "role": "transport", "params": {"sigma": 3.0, "mode": "reflect"}},
            {"node_id": "sensor", "primitive_id": "photon_sensor",
             "role": "sensor", "params": {"quantum_efficiency": 0.9, "gain": 1.0}},
            {"node_id": "noise", "primitive_id": "poisson_gaussian_sensor",
             "role": "noise", "params": {"peak_photons": 1000.0, "read_sigma": 0.05, "seed": 0}},
        ],
        "edges": [
            {"source": "source", "target": "blur"},
            {"source": "blur", "target": "sensor"},
            {"source": "sensor", "target": "noise"},
        ],
    })
    return GraphCompiler().compile(spec)


class TestWidefieldLowdose:
    def test_template_compiles(self):
        graph = _build_widefield_lowdose_v2_graph()
        assert graph.graph_id == "widefield_lowdose_graph_v2"
        assert len(graph.forward_plan) == 4

    def test_forward_produces_noisy_output(self):
        graph = _build_widefield_lowdose_v2_graph()
        x = np.random.rand(32, 32).astype(np.float64) * 0.5 + 0.1
        y = graph.forward(x)
        assert y.shape == (32, 32)

    def test_rl_recon_psnr(self):
        """Richardson-Lucy on low-dose data achieves PSNR > 15."""
        from scipy import ndimage
        from pwm_core.recon.richardson_lucy import richardson_lucy_2d
        from pwm_core.core.metric_registry import PSNR

        rng = np.random.RandomState(42)
        x_true = np.zeros((32, 32), dtype=np.float64)
        yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing='ij')
        x_true += 0.6 * np.exp(-(xx**2 + yy**2) / (2 * 0.3**2))
        x_true = np.clip(x_true, 0, 1)

        sigma = 3.0
        y = ndimage.gaussian_filter(x_true, sigma=sigma)
        y = y + rng.randn(*y.shape) * 0.02

        ax = np.arange(-7, 8, dtype=np.float64)
        gx, gy = np.meshgrid(ax, ax)
        psf = np.exp(-(gx**2 + gy**2) / (2 * sigma**2))
        psf /= psf.sum()

        x_hat = richardson_lucy_2d(y, psf, iterations=30, clip=True)
        psnr = PSNR()(x_hat.astype(np.float64), x_true, max_val=float(x_true.max()))
        assert psnr > 15, f"RL PSNR {psnr:.1f} < 15"

    def test_w2_nll_decreases(self):
        """Gain mismatch correction decreases NLL."""
        source = get_primitive("photon_source", {"strength": 1.0})
        blur = get_primitive("conv2d", {"sigma": 3.0, "mode": "reflect"})

        def make_fwd(gain):
            sensor = get_primitive("photon_sensor", {"quantum_efficiency": 0.9, "gain": gain})
            def fwd(x): return sensor.forward(blur.forward(source.forward(x)))
            return fwd

        rng = np.random.RandomState(42)
        x_true = rng.rand(32, 32).astype(np.float64) * 0.8 + 0.1
        sigma_noise = 0.02

        y_measured = make_fwd(1.3)(x_true) + rng.randn(32, 32) * sigma_noise

        y_pred_nom = make_fwd(1.0)(x_true)
        nll_before = float(np.sum((y_measured - y_pred_nom)**2 / sigma_noise**2))

        best_nll = np.inf
        for trial_gain in np.arange(0.5, 1.55, 0.05):
            y_trial = make_fwd(trial_gain)(x_true)
            nll_trial = float(np.sum((y_measured - y_trial)**2 / sigma_noise**2))
            if nll_trial < best_nll:
                best_nll = nll_trial

        assert best_nll < nll_before, (
            f"NLL should decrease: {best_nll:.1f} >= {nll_before:.1f}"
        )
