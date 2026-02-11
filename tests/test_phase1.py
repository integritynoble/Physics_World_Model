"""Tests for Phase 1: Extended node families, CorrectedOperator, new primitives."""
import numpy as np
import pytest


# -----------------------------------------------------------------------
# CorrectedOperator tests (P1.1)
# -----------------------------------------------------------------------

class TestCorrectedOperator:
    def _make_matrix_op(self, M=16, N=16):
        """Create a simple matrix operator for testing."""
        rng = np.random.RandomState(42)
        A = rng.randn(M, N)
        class MatOp:
            def forward(self, x):
                return A @ x.ravel()
            def adjoint(self, y):
                return A.T @ y.ravel()
        return MatOp(), A

    def test_prepost_identity(self):
        from pwm_core.graph.corrected_operator import CorrectedOperator, PrePostCorrection
        op, A = self._make_matrix_op()
        corr = PrePostCorrection(pre_scale=1.0, pre_shift=0.0, post_scale=1.0, post_shift=0.0)
        cop = CorrectedOperator(op, corr)
        x = np.random.randn(16)
        np.testing.assert_allclose(cop.forward(x), op.forward(x), atol=1e-10)

    def test_prepost_scale(self):
        from pwm_core.graph.corrected_operator import CorrectedOperator, PrePostCorrection
        op, A = self._make_matrix_op()
        corr = PrePostCorrection(pre_scale=2.0, post_scale=0.5)
        cop = CorrectedOperator(op, corr)
        x = np.random.randn(16)
        expected = 0.5 * (A @ (2.0 * x))
        np.testing.assert_allclose(cop.forward(x), expected, atol=1e-10)

    def test_prepost_nll_improvement(self):
        """Mode C success criterion: correction improves NLL."""
        from pwm_core.graph.corrected_operator import CorrectedOperator, PrePostCorrection
        op, A = self._make_matrix_op()
        x = np.random.randn(16)
        # True operator has scale 1.1
        y = 1.1 * (A @ x) + 0.01 * np.random.randn(16)
        # Uncorrected NLL
        nll_uncorrected = 0.5 * np.sum((y - A @ x) ** 2)
        # Corrected
        corr = PrePostCorrection(post_scale=1.1)
        cop = CorrectedOperator(op, corr)
        nll_corrected = 0.5 * np.sum((y - cop.forward(x)) ** 2)
        assert nll_corrected < nll_uncorrected

    def test_lowrank_identity(self):
        from pwm_core.graph.corrected_operator import CorrectedOperator, LowRankCorrection
        op, A = self._make_matrix_op()
        U = np.zeros((16, 2))
        V = np.zeros((16, 2))
        corr = LowRankCorrection(U, V)
        cop = CorrectedOperator(op, corr)
        x = np.random.randn(16)
        np.testing.assert_allclose(cop.forward(x), op.forward(x), atol=1e-10)

    def test_lowrank_nll_improvement(self):
        from pwm_core.graph.corrected_operator import CorrectedOperator, LowRankCorrection
        rng = np.random.RandomState(42)
        M, N, rank = 16, 16, 2
        A = rng.randn(M, N)
        # True perturbation
        U_true = rng.randn(M, rank) * 0.1
        V_true = rng.randn(N, rank) * 0.1
        alphas_true = np.array([0.5, 0.3])

        class MatOp:
            def forward(self, x): return A @ x.ravel()
            def adjoint(self, y): return A.T @ y.ravel()

        op = MatOp()
        x = rng.randn(N)
        y_true = A @ x + U_true @ (alphas_true * (V_true.T @ x)) + 0.01 * rng.randn(M)

        nll_base = 0.5 * np.sum((y_true - A @ x) ** 2)
        corr = LowRankCorrection(U_true, V_true, alphas_true)
        cop = CorrectedOperator(op, corr)
        nll_corr = 0.5 * np.sum((y_true - cop.forward(x)) ** 2)
        assert nll_corr < nll_base

    def test_learnable_params(self):
        from pwm_core.graph.corrected_operator import PrePostCorrection
        corr = PrePostCorrection()
        params = corr.learnable_params()
        assert "pre_scale" in params
        assert "post_scale" in params

    def test_get_set_params(self):
        from pwm_core.graph.corrected_operator import PrePostCorrection
        corr = PrePostCorrection(pre_scale=1.5)
        assert corr.get_params()["pre_scale"] == 1.5
        corr.set_params({"pre_scale": 2.0})
        assert corr.get_params()["pre_scale"] == 2.0


# -----------------------------------------------------------------------
# ExplicitLinearOperator tests (P1.2)
# -----------------------------------------------------------------------

class TestExplicitLinearOperator:
    def test_dense_forward(self):
        from pwm_core.graph.primitives import ExplicitLinearOperator
        A = np.random.randn(8, 16)
        prim = ExplicitLinearOperator(params={"matrix": A, "y_shape": (8,)})
        x = np.random.randn(16)
        np.testing.assert_allclose(prim.forward(x), A @ x, atol=1e-10)

    def test_dense_adjoint(self):
        from pwm_core.graph.primitives import ExplicitLinearOperator
        A = np.random.randn(8, 16)
        prim = ExplicitLinearOperator(params={"matrix": A, "x_shape": (16,)})
        y = np.random.randn(8)
        np.testing.assert_allclose(prim.adjoint(y), A.T @ y, atol=1e-10)

    def test_callback_mode(self):
        from pwm_core.graph.primitives import ExplicitLinearOperator
        A = np.random.randn(8, 16)
        prim = ExplicitLinearOperator(params={
            "forward_fn": lambda x: A @ x.ravel(),
            "adjoint_fn": lambda y: A.T @ y.ravel(),
        })
        x = np.random.randn(16)
        np.testing.assert_allclose(prim.forward(x), A @ x, atol=1e-10)

    def test_hash_stability(self):
        from pwm_core.graph.primitives import ExplicitLinearOperator
        A = np.eye(8)
        p1 = ExplicitLinearOperator(params={"matrix": A})
        p2 = ExplicitLinearOperator(params={"matrix": A})
        assert p1.compute_hash() == p2.compute_hash()

    def test_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "explicit_linear_operator" in PRIMITIVE_REGISTRY


# -----------------------------------------------------------------------
# Electron primitives tests (P1.3)
# -----------------------------------------------------------------------

class TestElectronPrimitives:
    def test_beam_source_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "electron_beam_source" in PRIMITIVE_REGISTRY

    def test_beam_source_forward(self):
        from pwm_core.graph.primitives import ElectronBeamSource
        src = ElectronBeamSource(params={"beam_current_na": 2.0})
        x = np.ones((16, 16))
        y = src.forward(x)
        np.testing.assert_allclose(y, 2.0)

    def test_detector_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "electron_detector_sensor" in PRIMITIVE_REGISTRY

    def test_detector_linear(self):
        from pwm_core.graph.primitives import ElectronDetectorSensor
        det = ElectronDetectorSensor(params={"collection_efficiency": 0.5, "gain": 2.0})
        x = np.ones((16, 16))
        np.testing.assert_allclose(det.forward(x), 1.0)

    def test_yield_model_multi_input(self):
        from pwm_core.graph.primitives import YieldModel
        ym = YieldModel(params={"yield_coeff": 0.3})
        incident = np.ones((8, 8)) * 2.0
        x_sample = np.ones((8, 8)) * 0.5
        result = ym.forward_multi({"incident": incident, "x": x_sample})
        np.testing.assert_allclose(result, 0.3 * 2.0 * 0.5)

    def test_thin_object_multi_input(self):
        from pwm_core.graph.primitives import ThinObjectPhase
        top = ThinObjectPhase(params={"sigma": 0.01})
        incident = np.ones((8, 8))
        x_sample = np.zeros((8, 8))  # zero potential = full transmission
        result = top.forward_multi({"incident": incident, "x": x_sample})
        np.testing.assert_allclose(result, 1.0)


# -----------------------------------------------------------------------
# Specialized sensors tests (P1.4)
# -----------------------------------------------------------------------

class TestSpecializedSensors:
    def test_single_pixel_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "single_pixel_sensor" in PRIMITIVE_REGISTRY

    def test_single_pixel_forward_shape(self):
        from pwm_core.graph.primitives import SinglePixelSensor
        sp = SinglePixelSensor(params={"n_patterns": 32, "seed": 42})
        x = np.random.randn(16, 16)
        y = sp.forward(x)
        assert y.shape == (32,)

    def test_xray_detector_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "xray_detector_sensor" in PRIMITIVE_REGISTRY

    def test_xray_detector_forward(self):
        from pwm_core.graph.primitives import XRayDetectorSensor
        det = XRayDetectorSensor(params={"scintillator_efficiency": 0.8, "gain": 2.0, "offset": 10.0})
        x = np.ones((16, 16))
        expected = 2.0 * 0.8 * 1.0 + 10.0
        np.testing.assert_allclose(det.forward(x), expected)

    def test_acoustic_receive_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "acoustic_receive_sensor" in PRIMITIVE_REGISTRY


# -----------------------------------------------------------------------
# Correlated noise tests (P1.5)
# -----------------------------------------------------------------------

class TestCorrelatedNoise:
    def test_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "correlated_noise_sensor" in PRIMITIVE_REGISTRY

    def test_forward_adds_noise(self):
        from pwm_core.graph.primitives import CorrelatedNoiseSensor
        cn = CorrelatedNoiseSensor(params={"base_sigma": 0.1, "correlation_type": "none", "seed": 42})
        x = np.ones((16, 16))
        y = cn.forward(x)
        assert not np.allclose(y, x)
        assert y.shape == x.shape

    def test_spatial_correlation(self):
        from pwm_core.graph.primitives import CorrelatedNoiseSensor
        cn = CorrelatedNoiseSensor(params={"base_sigma": 0.5, "correlation_type": "spatial", "correlation_length": 3.0, "seed": 42})
        x = np.zeros((32, 32))
        y = cn.forward(x)
        assert y.shape == (32, 32)

    def test_likelihood_raises(self):
        from pwm_core.graph.primitives import CorrelatedNoiseSensor
        cn = CorrelatedNoiseSensor(params={})
        with pytest.raises(NotImplementedError, match="whitening"):
            cn.likelihood(np.zeros(10), np.zeros(10))

    def test_simulation_only_flag(self):
        from pwm_core.graph.primitives import CorrelatedNoiseSensor
        assert CorrelatedNoiseSensor._simulation_only is True


# -----------------------------------------------------------------------
# New element primitives tests (P1.6)
# -----------------------------------------------------------------------

class TestNewElementPrimitives:
    def test_ctf_transfer_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "ctf_transfer" in PRIMITIVE_REGISTRY

    def test_ctf_forward_shape(self):
        from pwm_core.graph.primitives import CTFTransfer
        ctf = CTFTransfer(params={"defocus_nm": 100.0})
        x = np.random.randn(32, 32)
        y = ctf.forward(x)
        assert y.shape == (32, 32)

    def test_beamform_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "beamform_delay" in PRIMITIVE_REGISTRY

    def test_beamform_forward(self):
        from pwm_core.graph.primitives import BeamformDelay
        bf = BeamformDelay(params={"n_elements": 8})
        x = np.random.randn(8, 32)
        y = bf.forward(x)
        assert y.shape == (32,)

    def test_emission_projection_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "emission_projection" in PRIMITIVE_REGISTRY

    def test_emission_projection_forward(self):
        from pwm_core.graph.primitives import EmissionProjection
        ep = EmissionProjection(params={"n_angles": 16})
        x = np.random.randn(32, 32)
        y = ep.forward(x)
        assert y.shape == (16, 32)

    def test_scatter_model_registered(self):
        from pwm_core.graph.primitives import PRIMITIVE_REGISTRY
        assert "scatter_model" in PRIMITIVE_REGISTRY

    def test_scatter_model_forward(self):
        from pwm_core.graph.primitives import ScatterModel
        sm = ScatterModel(params={"scatter_fraction": 0.1, "kernel_sigma": 2.0})
        x = np.random.randn(32, 32)
        y = sm.forward(x)
        assert y.shape == (32, 32)
        # Should add scatter, so output != input
        assert not np.allclose(y, x)
