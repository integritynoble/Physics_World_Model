"""Tests for CasePack: Coded Aperture Compressive Temporal Imaging (CACTI).

Template: cacti_graph_v2
Chain: photon_source -> temporal_mask -> photon_sensor -> poisson_gaussian_sensor
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import TemporalMask
from pwm_core.core.metric_registry import PSNR


TEMPLATES_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "packages", "pwm_core", "contrib", "graph_templates.yaml",
)


def _load_template(key):
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data["templates"][key]


def _build_system_matrix(fwd, x_shape, y_shape):
    """Build the explicit (M, N) system matrix by probing basis vectors."""
    N = int(np.prod(x_shape))
    M = int(np.prod(y_shape))
    A = np.zeros((M, N), dtype=np.float64)
    for j in range(N):
        e = np.zeros(N, dtype=np.float64)
        e[j] = 1.0
        A[:, j] = fwd(e.reshape(x_shape)).ravel()
    return A


class TestCasePackCACTI:
    """CasePack acceptance tests for the CACTI modality."""

    def test_template_compiles(self):
        """cacti_graph_v2 template compiles without error."""
        tpl = _load_template("cacti_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "cacti_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None
        assert "mask" in graph.node_map
        assert "source" in graph.node_map

    def test_forward_sanity(self):
        """Mode S: forward pass produces finite 2D output from 3D video cube."""
        tpl = _load_template("cacti_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "cacti_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        rng = np.random.RandomState(42)
        x = rng.rand(64, 64, 8).astype(np.float64)
        y = graph.forward(x)
        assert y is not None
        assert np.isfinite(y).all()
        # CACTI collapses temporal axis: output should be 2D
        assert y.ndim == 2

    def test_forward_nonneg(self):
        """Non-negative video cube produces non-negative-mean compressed frame."""
        H, W, T = 16, 16, 8
        mask_op = TemporalMask(params={"H": H, "W": W, "T": T, "seed": 42})
        rng = np.random.RandomState(42)
        x = rng.rand(H, W, T).astype(np.float64)
        y = mask_op.forward(x)
        assert np.mean(y) >= 0, "CACTI forward on non-neg input should have non-neg mean"

    def test_recon_baseline_psnr(self):
        """Mode I: least-squares reconstruction achieves PSNR > 12 on 8x8x2.

        Generates x_true in the row space of A (via A^T @ c) so the
        min-norm lstsq solution exactly recovers it.
        """
        H, W, T = 8, 8, 2
        seed = 42
        mask_op = TemporalMask(params={"H": H, "W": W, "T": T, "seed": seed})

        # Build explicit system matrix
        y0 = mask_op.forward(np.zeros((H, W, T)))
        A = _build_system_matrix(mask_op.forward, (H, W, T), y0.shape)

        # Generate x_true in range(A^T) so lstsq recovers it exactly
        rng = np.random.RandomState(7)
        c = rng.randn(A.shape[0])
        x_range = A.T @ c
        x_true = x_range.reshape(H, W, T)
        x_true = x_true / (np.abs(x_true).max() + 1e-8)
        y_clean = mask_op.forward(x_true)

        x_hat, _, _, _ = np.linalg.lstsq(A, y_clean.ravel(), rcond=None)
        x_hat = x_hat.reshape(H, W, T)
        psnr = PSNR()(x_hat, x_true)
        assert psnr > 12, f"CACTI PSNR {psnr:.1f} < 12"

    def test_mismatch_timing_jitter(self):
        """Different mask seeds simulate timing jitter mismatch."""
        H, W, T = 16, 16, 8
        rng = np.random.RandomState(42)
        x = rng.rand(H, W, T).astype(np.float64)
        mask_base = TemporalMask(params={"H": H, "W": W, "T": T, "seed": 42})
        mask_jitter = TemporalMask(params={"H": H, "W": W, "T": T, "seed": 99})
        y_base = mask_base.forward(x)
        y_jitter = mask_jitter.forward(x)
        assert not np.allclose(y_base, y_jitter), "Timing jitter mismatch should change y"

    def test_mismatch_mask_motion(self):
        """Rolled mask simulates mask_motion_dx mismatch."""
        H, W, T = 16, 16, 8
        rng = np.random.RandomState(42)
        x = rng.rand(H, W, T).astype(np.float64)
        mask_op = TemporalMask(params={"H": H, "W": W, "T": T, "seed": 42})
        y_base = mask_op.forward(x)
        # Shift x spatially to simulate mask_motion_dx
        x_shifted = np.roll(x, shift=2, axis=1)
        y_shifted = mask_op.forward(x_shifted)
        assert not np.allclose(y_base, y_shifted), "Mask motion mismatch should change y"

    def test_gap_tv_recon_psnr(self):
        """GAP-TV on 16x16x4 row-space phantom achieves PSNR > 15."""
        from pwm_core.recon.gap_tv import gap_tv_cacti

        H, W, T = 16, 16, 4
        mask_op = TemporalMask(params={"H": H, "W": W, "T": T, "seed": 42})
        masks = mask_op._masks.copy()

        # Build explicit system matrix for row-space phantom
        y0 = mask_op.forward(np.zeros((H, W, T)))
        A = _build_system_matrix(mask_op.forward, (H, W, T), y0.shape)

        # Generate x_true in range(A^T) so GAP-TV can recover it
        rng = np.random.RandomState(7)
        c = rng.randn(A.shape[0])
        x_true = (A.T @ c).reshape(H, W, T)
        x_true = x_true / (np.abs(x_true).max() + 1e-8) * 0.8

        y_clean = mask_op.forward(x_true)
        y_noisy = y_clean + rng.randn(*y_clean.shape) * 0.001

        x_hat = gap_tv_cacti(
            y_noisy, masks[:H, :W, :T],
            iterations=50, lam=0.0001,
        )
        psnr = PSNR()(x_hat, x_true, max_val=1.0)
        assert psnr > 15, f"GAP-TV CACTI PSNR {psnr:.1f} < 15"

    def test_w2_nll_decreases(self):
        """Timing jitter mismatch correction decreases NLL."""
        H, W, T = 16, 16, 4
        seed = 42
        mask_nom = TemporalMask(params={"H": H, "W": W, "T": T, "seed": seed})
        mask_pert = TemporalMask(params={"H": H, "W": W, "T": T, "seed": 99})

        rng = np.random.RandomState(seed)
        x_true = rng.rand(H, W, T).astype(np.float64) * 0.8 + 0.1

        # "Measured" y from perturbed operator + noise
        sigma = 0.01
        y_measured = mask_pert.forward(x_true) + rng.randn(H, W) * sigma

        # NLL with nominal (uncorrected) operator
        y_pred_nom = mask_nom.forward(x_true)
        nll_before = 0.5 * np.sum((y_measured - y_pred_nom) ** 2) / sigma ** 2

        # Grid search for best seed
        best_nll = np.inf
        for trial_seed in range(30, 121, 3):
            mask_trial = TemporalMask(params={"H": H, "W": W, "T": T, "seed": trial_seed})
            y_trial = mask_trial.forward(x_true)
            nll_trial = 0.5 * np.sum((y_measured - y_trial) ** 2) / sigma ** 2
            if nll_trial < best_nll:
                best_nll = nll_trial

        nll_after = best_nll
        assert nll_after < nll_before, (
            f"NLL should decrease after correction: {nll_after:.1f} >= {nll_before:.1f}"
        )
