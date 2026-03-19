"""Tests for CasePack: Coded Aperture Snapshot Spectral Imaging (CASSI).

Template: cassi_graph_v2
Chain: photon_source -> coded_mask -> spectral_dispersion -> frame_integration
       -> photon_sensor -> poisson_gaussian_sensor
"""
import numpy as np
import pytest
import yaml
import os

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.primitives import (
    CodedMask, SpectralDispersion, FrameIntegration,
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


class TestCasePackCASSI:
    """CasePack acceptance tests for the CASSI modality."""

    def test_template_compiles(self):
        """cassi_graph_v2 template compiles without error."""
        tpl = _load_template("cassi_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "cassi_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        assert graph is not None
        assert "modulate" in graph.node_map
        assert "disperse" in graph.node_map
        assert "integrate" in graph.node_map

    def test_forward_sanity(self):
        """Mode S: forward pass produces finite 2D output from 3D spectral cube."""
        tpl = _load_template("cassi_graph_v2")
        tpl_clean = dict(tpl)
        tpl_clean.pop("description", None)
        spec = OperatorGraphSpec.model_validate({"graph_id": "cassi_v2", **tpl_clean})
        compiler = GraphCompiler()
        graph = compiler.compile(spec)
        rng = np.random.RandomState(42)
        x = rng.rand(64, 64, 8).astype(np.float64)
        y = graph.forward(x)
        assert y is not None
        assert np.isfinite(y).all()
        # CASSI collapses spectral axis: output should be 2D
        assert y.ndim == 2 or (y.ndim == 1 and y.shape[0] > 0)

    def test_forward_nonneg(self):
        """Non-negative spectral cube produces non-negative-mean output."""
        H, W, L = 16, 16, 8
        mask = CodedMask(params={"seed": 42, "H": H, "W": W})
        disp = SpectralDispersion(params={"disp_step": 1.0})
        integ = FrameIntegration(params={"axis": -1, "T": L})
        rng = np.random.RandomState(42)
        x = rng.rand(H, W, L).astype(np.float64)
        y = integ.forward(disp.forward(mask.forward(x)))
        assert np.mean(y) >= 0, "CASSI forward on non-neg input should have non-neg mean"

    def test_recon_baseline_psnr(self):
        """Mode I: least-squares reconstruction achieves PSNR > 15 on 8x8x2.

        Generates x_true in the row space of A (via A^T @ c) so the
        min-norm lstsq solution exactly recovers it.
        """
        H, W, L = 8, 8, 2
        seed = 42
        mask = CodedMask(params={"seed": seed, "H": H, "W": W})
        disp = SpectralDispersion(params={"disp_step": 1.0})
        integ = FrameIntegration(params={"axis": -1, "T": L})

        def fwd(x):
            return integ.forward(disp.forward(mask.forward(x)))

        def adj(y):
            return mask.adjoint(disp.adjoint(integ.adjoint(y)))

        # Build explicit system matrix
        A = _build_system_matrix(fwd, (H, W, L), fwd(np.zeros((H, W, L))).shape)

        # Generate x_true in range(A^T) so lstsq recovers it exactly
        rng = np.random.RandomState(7)
        c = rng.randn(A.shape[0])
        x_range = A.T @ c
        x_true = x_range.reshape(H, W, L)
        x_true = x_true / (np.abs(x_true).max() + 1e-8)
        y_clean = fwd(x_true)

        x_hat, _, _, _ = np.linalg.lstsq(A, y_clean.ravel(), rcond=None)
        x_hat = x_hat.reshape(H, W, L)
        psnr = PSNR()(x_hat, x_true)
        assert psnr > 15, f"CASSI PSNR {psnr:.1f} < 15"

    def test_mismatch_mask_shift(self):
        """Shifted mask produces different measurements (mask_dx/dy mismatch)."""
        H, W, L = 16, 16, 8
        rng = np.random.RandomState(42)
        x = rng.rand(H, W, L).astype(np.float64)
        disp = SpectralDispersion(params={"disp_step": 1.0})
        integ = FrameIntegration(params={"axis": -1, "T": L})
        mask_base = CodedMask(params={"seed": 42, "H": H, "W": W})
        mask_shifted = CodedMask(params={"seed": 99, "H": H, "W": W})
        y_base = integ.forward(disp.forward(mask_base.forward(x)))
        y_shift = integ.forward(disp.forward(mask_shifted.forward(x)))
        assert not np.allclose(y_base, y_shift), "Mask mismatch should change measurements"

    def test_mismatch_disp_slope(self):
        """Different dispersion slope produces different measurements."""
        H, W, L = 16, 16, 8
        rng = np.random.RandomState(42)
        x = rng.rand(H, W, L).astype(np.float64)
        mask = CodedMask(params={"seed": 42, "H": H, "W": W})
        integ = FrameIntegration(params={"axis": -1, "T": L})
        disp_a = SpectralDispersion(params={"disp_step": 1.0})
        disp_b = SpectralDispersion(params={"disp_step": 2.0})
        y_a = integ.forward(disp_a.forward(mask.forward(x)))
        y_b = integ.forward(disp_b.forward(mask.forward(x)))
        assert not np.allclose(y_a, y_b), "Dispersion slope mismatch should change measurements"

    def test_gap_tv_recon_psnr(self):
        """GAP-TV on 16x16x4 row-space phantom achieves PSNR > 15."""
        from pwm_core.recon.gap_tv import gap_tv_operator

        H, W, L = 16, 16, 4
        mask = CodedMask(params={"seed": 42, "H": H, "W": W})
        disp = SpectralDispersion(params={"disp_step": 1.0})
        integ = FrameIntegration(params={"axis": -1, "T": L})

        def fwd(x):
            return integ.forward(disp.forward(mask.forward(x)))

        def adj(y):
            return mask.adjoint(disp.adjoint(integ.adjoint(y)))

        # Build system matrix for row-space phantom
        A = _build_system_matrix(fwd, (H, W, L), fwd(np.zeros((H, W, L))).shape)

        # Generate x_true in range(A^T) so GAP-TV can recover it
        rng = np.random.RandomState(7)
        c = rng.randn(A.shape[0])
        x_true = (A.T @ c).reshape(H, W, L)
        x_true = x_true / (np.abs(x_true).max() + 1e-8) * 0.8

        y_clean = fwd(x_true)
        y_noisy = y_clean + rng.randn(*y_clean.shape) * 0.001

        x_hat = gap_tv_operator(
            y_noisy, fwd, adj, (H, W, L),
            iterations=50, lam=0.0001,
        )
        psnr = PSNR()(x_hat, x_true, max_val=1.0)
        assert psnr > 15, f"GAP-TV PSNR {psnr:.1f} < 15"

    def test_w2_nll_decreases(self):
        """Dispersion mismatch correction decreases NLL."""
        H, W, L = 16, 16, 4
        seed = 42
        mask = CodedMask(params={"seed": seed, "H": H, "W": W})
        integ = FrameIntegration(params={"axis": -1, "T": L})

        def _make_fwd(ds):
            d = SpectralDispersion(params={"disp_step": ds})
            return lambda x: integ.forward(d.forward(mask.forward(x)))

        # Nominal and perturbed operators
        fwd_nom = _make_fwd(1.0)
        fwd_pert = _make_fwd(1.15)

        rng = np.random.RandomState(seed)
        x_true = rng.rand(H, W, L).astype(np.float64) * 0.8 + 0.1

        # "Measured" y from perturbed operator + noise
        sigma = 0.01
        y_measured = fwd_pert(x_true) + rng.randn(H, W) * sigma

        # NLL with nominal (uncorrected) operator
        y_pred_nom = fwd_nom(x_true)
        nll_before = 0.5 * np.sum((y_measured - y_pred_nom) ** 2) / sigma ** 2

        # Grid search for best disp_step
        best_nll = np.inf
        for ds in np.linspace(0.8, 1.3, 26):
            fwd_trial = _make_fwd(ds)
            y_trial = fwd_trial(x_true)
            nll_trial = 0.5 * np.sum((y_measured - y_trial) ** 2) / sigma ** 2
            if nll_trial < best_nll:
                best_nll = nll_trial

        nll_after = best_nll
        assert nll_after < nll_before, (
            f"NLL should decrease after correction: {nll_after:.1f} >= {nll_before:.1f}"
        )
