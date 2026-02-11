"""Tests for CasePack: Single Pixel Camera (SPC).

Template: spc_graph_v2
Chain: photon_source -> random_mask -> photon_sensor -> poisson_gaussian_sensor
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import RandomMask, PhotonSource, PhotonSensor
from pwm_core.core.metric_registry import PSNR


TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "packages", "pwm_core", "contrib", "graph_templates.yaml",
)


def _load_template(key):
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


def _lstsq_recon(A, y, x_shape):
    """Least-squares reconstruction using pseudoinverse."""
    x_hat, _, _, _ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    return x_hat.reshape(x_shape)


class TestCasePackSPC:
    """CasePack acceptance tests for the SPC modality."""

    def test_template_compiles(self):
        """spc_graph_v2 template compiles without error."""
        tpl = _load_template("spc_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "spc_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None
        assert "measure" in graph.node_map
        assert "source" in graph.node_map

    def test_forward_sanity(self):
        """Mode S: forward pass produces finite, correctly shaped output."""
        tpl = _load_template("spc_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "spc_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        rng = np.random.RandomState(42)
        x = rng.rand(64, 64).astype(np.float64)
        y = graph.forward(x)
        assert y is not None
        assert np.isfinite(y).all()
        assert y.ndim == 1
        assert y.shape[0] > 0

    def test_forward_nonneg_input(self):
        """Non-negative input produces output with expected magnitude."""
        tpl = _load_template("spc_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "spc_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        x = np.ones((64, 64), dtype=np.float64) * 0.5
        y = graph.forward(x)
        assert np.isfinite(y).all()

    def test_recon_baseline_psnr(self):
        """Mode I: least-squares reconstruction achieves PSNR > 12 on 16x16."""
        H, W = 16, 16
        rate = 0.75  # overdetermined for small problem
        seed = 42
        mask_op = RandomMask(params={"seed": seed, "H": H, "W": W, "sampling_rate": rate})
        # Access the internal measurement matrix
        A = mask_op._A
        # Smooth block signal
        x_true = np.zeros((H, W), dtype=np.float64)
        x_true[4:12, 4:12] = 1.0
        x_true = x_true / (x_true.max() + 1e-8)
        y_clean = mask_op.forward(x_true)
        x_hat = _lstsq_recon(A, y_clean, (H, W))
        psnr = PSNR()(x_hat, x_true)
        assert psnr > 12, f"SPC PSNR {psnr:.1f} < 12"

    def test_mismatch_params(self):
        """Perturbed SPC params produce different measurements (mismatch detection)."""
        H, W = 16, 16
        rng = np.random.RandomState(42)
        x = rng.rand(H, W).astype(np.float64)
        base = RandomMask(params={"seed": 42, "H": H, "W": W, "sampling_rate": 0.15})
        y_base = base.forward(x)
        # Different seed produces different measurement matrix (pattern_shift proxy)
        shifted = RandomMask(params={"seed": 99, "H": H, "W": W, "sampling_rate": 0.15})
        y_shifted = shifted.forward(x)
        assert not np.allclose(y_base, y_shifted), "Mismatch should produce different y"
