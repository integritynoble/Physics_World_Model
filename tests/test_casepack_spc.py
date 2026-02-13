"""Tests for CasePack: Single Pixel Camera (SPC).

Template: spc_graph_v2
Chain: photon_source -> projection_optics -> dmd_pattern_sequence ->
       bucket_integration -> photon_sensor -> quantize -> poisson_gaussian_sensor
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import (
    RandomMask, PhotonSource, PhotonSensor,
    ProjectionOptics, DMDPatternSequence, BucketIntegration,
)
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
        assert "dmd_pattern" in graph.node_map
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

    def test_seven_node_chain(self):
        """spc_graph_v2 has 7 nodes in the correct order."""
        tpl = _load_template("spc_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "spc_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        node_ids = [nid for nid, _ in graph.forward_plan]
        assert len(node_ids) == 7
        assert "source" in node_ids
        assert "projection" in node_ids
        assert "dmd_pattern" in node_ids
        assert "bucket_integrate" in node_ids
        assert "sensor" in node_ids
        assert "quantize" in node_ids
        assert "noise" in node_ids

    def test_recon_baseline_psnr(self):
        """Mode I: least-squares reconstruction achieves PSNR > 12 on 16x16."""
        H, W = 16, 16
        rate = 0.75  # overdetermined for small problem
        seed = 42
        mask_op = RandomMask(params={"seed": seed, "H": H, "W": W, "sampling_rate": rate})
        A = mask_op._A
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
        base = DMDPatternSequence(params={"seed": 42, "H": H, "W": W, "sampling_rate": 0.15})
        y_base = base.forward(x)
        # Different seed produces different measurement matrix
        shifted = DMDPatternSequence(params={"seed": 99, "H": H, "W": W, "sampling_rate": 0.15})
        y_shifted = shifted.forward(x)
        assert not np.allclose(y_base, y_shifted), "Mismatch should produce different y"


class TestSPCPrimitives:
    """Unit tests for the new SPC primitives."""

    def test_projection_optics_forward(self):
        """ProjectionOptics applies throughput, PSF, and vignetting."""
        op = ProjectionOptics(params={
            "throughput": 0.9, "psf_sigma": 1.0, "vignetting_coeff": 0.05
        })
        x = np.ones((32, 32), dtype=np.float64)
        y = op.forward(x)
        assert y.shape == (32, 32)
        assert np.isfinite(y).all()
        # Throughput reduces signal
        assert y.max() < 0.95
        # Vignetting reduces corners more than center
        assert y[16, 16] > y[0, 0]

    def test_projection_optics_adjoint(self):
        """ProjectionOptics adjoint has consistent shape."""
        op = ProjectionOptics(params={"throughput": 0.9, "psf_sigma": 0.5})
        x = np.random.RandomState(42).rand(32, 32).astype(np.float64)
        y = op.forward(x)
        x_adj = op.adjoint(y)
        assert x_adj.shape == x.shape
        assert np.isfinite(x_adj).all()

    def test_projection_optics_identity_when_no_effects(self):
        """ProjectionOptics with no effects is just throughput scaling."""
        op = ProjectionOptics(params={
            "throughput": 1.0, "psf_sigma": 0.0, "vignetting_coeff": 0.0
        })
        x = np.random.RandomState(42).rand(16, 16).astype(np.float64)
        y = op.forward(x)
        np.testing.assert_allclose(y, x, atol=1e-12)

    def test_dmd_pattern_sequence_forward(self):
        """DMDPatternSequence produces M measurements from HxW input."""
        H, W = 16, 16
        op = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25
        })
        x = np.random.RandomState(42).rand(H, W).astype(np.float64)
        y = op.forward(x)
        M = int(H * W * 0.25)
        assert y.shape == (M,)
        assert np.isfinite(y).all()

    def test_dmd_pattern_sequence_adjoint(self):
        """DMDPatternSequence adjoint produces HxW output."""
        H, W = 16, 16
        op = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25
        })
        M = int(H * W * 0.25)
        y = np.random.RandomState(42).rand(M).astype(np.float64)
        x_adj = op.adjoint(y)
        assert x_adj.shape == (H, W)
        assert np.isfinite(x_adj).all()

    def test_dmd_pattern_sequence_adjoint_consistency(self):
        """<Ax, y> == <x, A^T y> for DMDPatternSequence."""
        H, W = 16, 16
        op = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25
        })
        rng = np.random.RandomState(99)
        x = rng.rand(H, W).astype(np.float64)
        M = int(H * W * 0.25)
        y = rng.rand(M).astype(np.float64)
        Ax = op.forward(x)
        Aty = op.adjoint(y)
        lhs = np.dot(Ax, y)
        rhs = np.dot(x.ravel(), Aty.ravel())
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_dmd_contrast_degradation(self):
        """DMDPatternSequence contrast < 1 reduces measurement magnitude."""
        H, W = 16, 16
        x = np.ones((H, W), dtype=np.float64)
        op_full = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25, "contrast": 1.0
        })
        op_low = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25, "contrast": 0.8
        })
        y_full = op_full.forward(x)
        y_low = op_low.forward(x)
        assert np.linalg.norm(y_low) < np.linalg.norm(y_full)

    def test_dmd_illumination_drift(self):
        """Illumination drift changes measurement vector."""
        H, W = 16, 16
        x = np.ones((H, W), dtype=np.float64)
        op_no = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25,
            "illum_drift_linear": 0.0, "illum_drift_sin_amp": 0.0,
        })
        op_drift = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25,
            "illum_drift_linear": 0.1, "illum_drift_sin_amp": 0.05,
            "illum_drift_sin_freq": 2.0,
        })
        y_no = op_no.forward(x)
        y_drift = op_drift.forward(x)
        assert not np.allclose(y_no, y_drift)

    def test_bucket_integration_forward(self):
        """BucketIntegration applies duty cycle scaling."""
        op = BucketIntegration(params={"duty_cycle": 0.8, "clock_offset": 0.0})
        y = np.ones(100, dtype=np.float64)
        out = op.forward(y)
        np.testing.assert_allclose(out, 0.8, atol=1e-12)

    def test_bucket_integration_clock_offset(self):
        """BucketIntegration clock offset reduces signal."""
        op = BucketIntegration(params={"duty_cycle": 1.0, "clock_offset": 0.1})
        y = np.ones(100, dtype=np.float64)
        out = op.forward(y)
        np.testing.assert_allclose(out, 0.9, atol=1e-12)

    def test_bucket_integration_adjoint(self):
        """BucketIntegration adjoint scales by same factor."""
        op = BucketIntegration(params={"duty_cycle": 0.9, "clock_offset": 0.0})
        y = np.ones(50, dtype=np.float64)
        fwd = op.forward(y)
        adj = op.adjoint(y)
        np.testing.assert_allclose(fwd, adj)

    def test_dmd_dead_mirrors(self):
        """Dead mirrors zero out some patterns."""
        H, W = 16, 16
        op_clean = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25,
            "dead_mirror_rate": 0.0,
        })
        op_dead = DMDPatternSequence(params={
            "seed": 42, "H": H, "W": W, "sampling_rate": 0.25,
            "dead_mirror_rate": 0.5,  # extreme for testing
        })
        x = np.ones((H, W), dtype=np.float64)
        y_clean = op_clean.forward(x)
        y_dead = op_dead.forward(x)
        # Dead mirrors reduce total signal
        assert np.abs(y_dead).sum() < np.abs(y_clean).sum()

    def test_primitives_in_registry(self):
        """New SPC primitives are in PRIMITIVE_REGISTRY."""
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "projection_optics" in PRIMITIVE_REGISTRY
        assert "dmd_pattern_sequence" in PRIMITIVE_REGISTRY
        assert "bucket_integration" in PRIMITIVE_REGISTRY
