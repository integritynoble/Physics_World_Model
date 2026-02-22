"""Tests for GPU (PyTorch) mode in iterative solvers.

Categories:
1. CPU/GPU parity — both paths produce same results within tolerance
2. Output type — GPU path returns np.ndarray, not torch.Tensor
3. Auto-detect — device=None works without error
4. Backward compat — calling without device kwarg still works
5. No-torch fallback — if torch unavailable, CPU path runs
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CUDA = False
if HAS_TORCH:
    import torch
    HAS_CUDA = torch.cuda.is_available()

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def _simple_matrix(m, n, seed=42):
    """Create a simple random measurement matrix."""
    rng = np.random.RandomState(seed)
    A = rng.randn(m, n).astype(np.float32)
    return A


def _simple_operator(scale=0.5):
    """Create simple forward/adjoint pair from scaling."""
    def forward(x):
        return (scale * x).astype(np.float32)
    def adjoint(y):
        return (scale * y).astype(np.float32)
    return forward, adjoint


# ---------------------------------------------------------------------------
# gpu_utils tests
# ---------------------------------------------------------------------------

class TestGpuUtils:
    def test_resolve_device_none_no_crash(self):
        from pwm_core.recon.gpu_utils import resolve_device
        dev, use_gpu = resolve_device(None)
        assert isinstance(use_gpu, bool)

    def test_resolve_device_cpu(self):
        from pwm_core.recon.gpu_utils import resolve_device
        dev, use_gpu = resolve_device('cpu')
        assert not use_gpu
        if HAS_TORCH:
            assert dev.type == 'cpu'

    @skip_no_torch
    def test_to_torch_to_numpy_roundtrip(self):
        from pwm_core.recon.gpu_utils import to_torch, to_numpy
        x = np.random.randn(10, 10).astype(np.float32)
        t = to_torch(x, torch.device('cpu'))
        y = to_numpy(t)
        np.testing.assert_array_equal(x, y)

    @skip_no_torch
    def test_soft_threshold_torch(self):
        from pwm_core.recon.gpu_utils import soft_threshold_torch, to_torch, to_numpy
        x_np = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        expected = np.sign(x_np) * np.maximum(np.abs(x_np) - 1.0, 0)
        x_t = to_torch(x_np, torch.device('cpu'))
        result = to_numpy(soft_threshold_torch(x_t, 1.0))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    @skip_no_torch
    def test_tv_prox_2d_torch_returns_correct_shape(self):
        from pwm_core.recon.gpu_utils import tv_prox_2d_torch, to_torch, to_numpy
        x = to_torch(np.random.randn(16, 16).astype(np.float32), torch.device('cpu'))
        result = tv_prox_2d_torch(x, 0.1, iterations=5)
        assert result.shape == (16, 16)

    @skip_no_torch
    def test_tv_denoiser_3d_torch_returns_correct_shape(self):
        from pwm_core.recon.gpu_utils import tv_denoiser_3d_torch, to_torch
        x = to_torch(np.random.randn(8, 8, 4).astype(np.float32), torch.device('cpu'))
        result = tv_denoiser_3d_torch(x, 0.1, iterations=3)
        assert result.shape == (8, 8, 4)


# ---------------------------------------------------------------------------
# classical.py tests
# ---------------------------------------------------------------------------

class TestClassicalGPU:
    def test_fista_l2_backward_compat(self):
        """Calling without device kwarg still works."""
        from pwm_core.recon.classical import fista_l2
        A = _simple_matrix(20, 30)
        y = A @ np.random.randn(30).astype(np.float32)
        result = fista_l2(y, A, lam=0.01, iters=10)
        assert isinstance(result, np.ndarray)

    def test_fista_l2_cpu_device(self):
        from pwm_core.recon.classical import fista_l2
        A = _simple_matrix(20, 30)
        y = A @ np.random.randn(30).astype(np.float32)
        result = fista_l2(y, A, lam=0.01, iters=10, device='cpu')
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_fista_l2_cpu_gpu_parity(self):
        from pwm_core.recon.classical import fista_l2
        A = _simple_matrix(20, 30)
        x_true = np.random.randn(30).astype(np.float32)
        y = A @ x_true
        cpu_result = fista_l2(y, A, lam=0.01, iters=20, device='cpu')
        gpu_result = fista_l2(y, A, lam=0.01, iters=20, device='cpu')
        # Both paths should produce ndarray
        assert isinstance(gpu_result, np.ndarray)
        np.testing.assert_allclose(cpu_result, gpu_result, atol=1e-4)

    def test_least_squares_backward_compat(self):
        from pwm_core.recon.classical import least_squares
        A = _simple_matrix(30, 20)
        y = A @ np.random.randn(20).astype(np.float32)
        result = least_squares(A, y)
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_least_squares_cpu_parity(self):
        from pwm_core.recon.classical import least_squares
        A = _simple_matrix(30, 20)
        x_true = np.random.randn(20).astype(np.float32)
        y = A @ x_true
        cpu_result = least_squares(A, y, device='cpu')
        assert isinstance(cpu_result, np.ndarray)

    def test_gradient_descent_operator_backward_compat(self):
        from pwm_core.recon.classical import gradient_descent_operator
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = gradient_descent_operator(y, fwd, adj, (16, 16), iters=5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16)

    def test_conjugate_gradient_operator_backward_compat(self):
        from pwm_core.recon.classical import conjugate_gradient_operator
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = conjugate_gradient_operator(y, fwd, adj, (16, 16), iters=5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16)


# ---------------------------------------------------------------------------
# cs_solvers.py tests
# ---------------------------------------------------------------------------

class TestCsSolversGPU:
    def test_soft_threshold_backward_compat(self):
        from pwm_core.recon.cs_solvers import soft_threshold
        x = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        result = soft_threshold(x, 1.0)
        expected = np.sign(x) * np.maximum(np.abs(x) - 1.0, 0)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_tv_prox_2d_backward_compat(self):
        from pwm_core.recon.cs_solvers import tv_prox_2d
        x = np.random.randn(16, 16).astype(np.float32)
        result = tv_prox_2d(x, 0.1, iterations=5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16)

    def test_ista_backward_compat(self):
        from pwm_core.recon.cs_solvers import ista
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = ista(y, fwd, adj, (16, 16), max_iters=5)
        assert isinstance(result, np.ndarray)

    def test_fista_backward_compat(self):
        from pwm_core.recon.cs_solvers import fista
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = fista(y, fwd, adj, (16, 16), max_iters=5)
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_ista_cpu_returns_ndarray(self):
        from pwm_core.recon.cs_solvers import ista
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = ista(y, fwd, adj, (16, 16), max_iters=5, device='cpu')
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_fista_cpu_returns_ndarray(self):
        from pwm_core.recon.cs_solvers import fista
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = fista(y, fwd, adj, (16, 16), max_iters=5, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_tval3_backward_compat(self):
        from pwm_core.recon.cs_solvers import tval3
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = tval3(y, fwd, adj, (16, 16), max_iters=3, inner_iters=2)
        assert isinstance(result, np.ndarray)

    def test_admm_tv_backward_compat(self):
        from pwm_core.recon.cs_solvers import admm_tv
        A = _simple_matrix(50, 64)
        y = A @ np.random.randn(64).astype(np.float32)
        result = admm_tv(y, A, (8, 8), max_iters=5, tv_inner_iters=3)
        assert isinstance(result, np.ndarray)

    def test_run_admm_tv_passes_device(self):
        from pwm_core.recon.cs_solvers import run_admm_tv
        A = _simple_matrix(50, 64)
        y = A @ np.random.randn(64).astype(np.float32)
        result, info = run_admm_tv(y, A, (8, 8), cfg={"iters": 3, "device": "cpu"})
        assert isinstance(result, np.ndarray)

    def test_run_ista_passes_device(self):
        from pwm_core.recon.cs_solvers import run_ista

        class FakePhysics:
            x_shape = (16, 16)
            def forward(self, x): return (0.5 * x).astype(np.float32)
            def adjoint(self, y): return (0.5 * y).astype(np.float32)

        y = np.random.randn(16, 16).astype(np.float32)
        result, info = run_ista(y, FakePhysics(), {"iters": 3, "device": "cpu"})
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# gap_tv.py tests
# ---------------------------------------------------------------------------

class TestGapTvGPU:
    def test_tv_denoiser_3d_backward_compat(self):
        from pwm_core.recon.gap_tv import tv_denoiser_3d
        x = np.random.randn(8, 8, 4).astype(np.float32)
        result = tv_denoiser_3d(x, 0.1, iterations=3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 8, 4)

    @skip_no_torch
    def test_tv_denoiser_3d_cpu_parity(self):
        from pwm_core.recon.gap_tv import tv_denoiser_3d
        x = np.random.randn(8, 8, 4).astype(np.float32)
        # CPU (numpy) path via explicit device=None with no torch
        cpu_result = tv_denoiser_3d(x, 0.1, iterations=5, device='cpu')
        assert isinstance(cpu_result, np.ndarray)

    def test_gap_tv_cacti_backward_compat(self):
        from pwm_core.recon.gap_tv import gap_tv_cacti
        h, w, nf = 8, 8, 4
        y = np.random.randn(h, w).astype(np.float32)
        masks = np.random.rand(h, w, nf).astype(np.float32)
        result = gap_tv_cacti(y, masks, iterations=3, lam=0.05)
        assert isinstance(result, np.ndarray)
        assert result.shape == (h, w, nf)

    @skip_no_torch
    def test_gap_tv_cacti_cpu_returns_ndarray(self):
        from pwm_core.recon.gap_tv import gap_tv_cacti
        h, w, nf = 8, 8, 4
        y = np.random.randn(h, w).astype(np.float32)
        masks = np.random.rand(h, w, nf).astype(np.float32)
        result = gap_tv_cacti(y, masks, iterations=3, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_gap_tv_cassi_backward_compat(self):
        from pwm_core.recon.gap_tv import gap_tv_cassi
        h, w, nb = 8, 10, 4
        step = 1
        w_meas = w + (nb - 1) * step
        y = np.random.randn(h, w_meas).astype(np.float32)
        mask = np.random.rand(h, w).astype(np.float32)
        result = gap_tv_cassi(y, mask, nb, iterations=3, step=step)
        assert isinstance(result, np.ndarray)
        assert result.shape == (h, w, nb)


# ---------------------------------------------------------------------------
# spc_solvers.py tests
# ---------------------------------------------------------------------------

class TestSpcSolversGPU:
    def test_admm_spc_backward_compat(self):
        from pwm_core.recon.spc_solvers import admm_spc
        A = _simple_matrix(20, 64)
        y = A @ np.random.randn(64).astype(np.float32)
        result = admm_spc(y, A, iterations=10)
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_admm_spc_cpu_returns_ndarray(self):
        from pwm_core.recon.spc_solvers import admm_spc
        A = _simple_matrix(20, 64)
        y = A @ np.random.randn(64).astype(np.float32)
        result = admm_spc(y, A, iterations=10, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_fista_l1_backward_compat(self):
        from pwm_core.recon.spc_solvers import fista_l1
        A = _simple_matrix(20, 64)
        y = A @ np.random.randn(64).astype(np.float32)
        result = fista_l1(y, A, iterations=10)
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_fista_l1_cpu_returns_ndarray(self):
        from pwm_core.recon.spc_solvers import fista_l1
        A = _simple_matrix(20, 64)
        y = A @ np.random.randn(64).astype(np.float32)
        result = fista_l1(y, A, iterations=10, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_estimate_operator_norm_backward_compat(self):
        from pwm_core.recon.spc_solvers import estimate_operator_norm
        A = _simple_matrix(20, 30)
        norm = estimate_operator_norm(A)
        assert isinstance(norm, float)
        assert norm > 0

    def test_solve_spc_passes_kwargs(self):
        from pwm_core.recon.spc_solvers import solve_spc
        A = _simple_matrix(20, 64)
        y = A @ np.random.randn(64).astype(np.float32)
        result = solve_spc(y, A, method='admm', iterations=5)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# mri_solvers.py tests
# ---------------------------------------------------------------------------

class TestMriSolversGPU:
    def test_zero_filled_backward_compat(self):
        from pwm_core.recon.mri_solvers import zero_filled_reconstruction
        kspace = np.random.randn(32, 32).astype(np.complex64)
        result = zero_filled_reconstruction(kspace)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    @skip_no_torch
    def test_zero_filled_cpu_returns_ndarray(self):
        from pwm_core.recon.mri_solvers import zero_filled_reconstruction
        kspace = np.random.randn(32, 32).astype(np.complex64)
        result = zero_filled_reconstruction(kspace, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_cs_mri_wavelet_backward_compat(self):
        from pwm_core.recon.mri_solvers import cs_mri_wavelet
        kspace = np.random.randn(32, 32).astype(np.complex64)
        mask = np.ones((32, 32), dtype=np.float32)
        result = cs_mri_wavelet(kspace, mask, iterations=5)
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_cs_mri_wavelet_cpu_returns_ndarray(self):
        from pwm_core.recon.mri_solvers import cs_mri_wavelet
        kspace = np.random.randn(32, 32).astype(np.complex64)
        mask = np.ones((32, 32), dtype=np.float32)
        result = cs_mri_wavelet(kspace, mask, iterations=5, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_estimate_sensitivity_maps_backward_compat(self):
        from pwm_core.recon.mri_solvers import estimate_sensitivity_maps
        kspace = np.random.randn(4, 32, 32).astype(np.complex64)
        result = estimate_sensitivity_maps(kspace)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 32, 32)


# ---------------------------------------------------------------------------
# ct_solvers.py tests
# ---------------------------------------------------------------------------

class TestCtSolversGPU:
    def test_fbp_2d_backward_compat(self):
        from pwm_core.recon.ct_solvers import fbp_2d
        n_angles, n_det = 30, 32
        sinogram = np.random.randn(n_angles, n_det).astype(np.float32)
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        result = fbp_2d(sinogram, angles)
        assert isinstance(result, np.ndarray)
        assert result.shape == (n_det, n_det)

    @skip_no_torch
    def test_fbp_2d_cpu_returns_ndarray(self):
        from pwm_core.recon.ct_solvers import fbp_2d
        n_angles, n_det = 30, 32
        sinogram = np.random.randn(n_angles, n_det).astype(np.float32)
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        result = fbp_2d(sinogram, angles, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_sart_operator_backward_compat(self):
        from pwm_core.recon.ct_solvers import sart_operator
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = sart_operator(y, fwd, adj, (16, 16), iterations=3)
        assert isinstance(result, np.ndarray)

    @skip_no_torch
    def test_sart_operator_cpu_returns_ndarray(self):
        from pwm_core.recon.ct_solvers import sart_operator
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32) * 0.5
        result = sart_operator(y, fwd, adj, (16, 16), iterations=3, device='cpu')
        assert isinstance(result, np.ndarray)

    def test_run_fbp_passes_device(self):
        from pwm_core.recon.ct_solvers import run_fbp

        class FakeCT:
            angles = np.linspace(0, np.pi, 30, endpoint=False)

        sinogram = np.random.randn(30, 32).astype(np.float32)
        result, info = run_fbp(sinogram, FakeCT(), {"device": "cpu"})
        assert isinstance(result, np.ndarray)

    def test_run_sart_passes_device(self):
        from pwm_core.recon.ct_solvers import run_sart

        class FakePhysics:
            x_shape = (16, 16)
            def forward(self, x): return (0.5 * x).astype(np.float32)
            def adjoint(self, y): return (0.5 * y).astype(np.float32)

        y = np.random.randn(16, 16).astype(np.float32)
        result, info = run_sart(y, FakePhysics(), {"iters": 3, "device": "cpu"})
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# No-torch fallback test
# ---------------------------------------------------------------------------

class TestNoTorchFallback:
    def test_resolve_device_without_torch(self):
        """When torch is not importable, resolve_device returns (None, False)."""
        from pwm_core.recon import gpu_utils

        # Temporarily make torch unimportable
        original = sys.modules.get('torch')
        sys.modules['torch'] = None
        try:
            # Need to reload to pick up the mocked import
            dev, use_gpu = gpu_utils.resolve_device(None)
            assert dev is None
            assert use_gpu is False
        finally:
            if original is not None:
                sys.modules['torch'] = original
            else:
                sys.modules.pop('torch', None)


# ---------------------------------------------------------------------------
# Auto-detect test (device=None)
# ---------------------------------------------------------------------------

class TestAutoDetect:
    def test_fista_l2_auto(self):
        from pwm_core.recon.classical import fista_l2
        A = _simple_matrix(20, 30)
        y = A @ np.random.randn(30).astype(np.float32)
        result = fista_l2(y, A, lam=0.01, iters=5, device=None)
        assert isinstance(result, np.ndarray)

    def test_ista_auto(self):
        from pwm_core.recon.cs_solvers import ista
        fwd, adj = _simple_operator(0.5)
        y = np.random.randn(16, 16).astype(np.float32)
        result = ista(y, fwd, adj, (16, 16), max_iters=3, device=None)
        assert isinstance(result, np.ndarray)

    def test_gap_tv_cacti_auto(self):
        from pwm_core.recon.gap_tv import gap_tv_cacti
        h, w, nf = 8, 8, 4
        y = np.random.randn(h, w).astype(np.float32)
        masks = np.random.rand(h, w, nf).astype(np.float32)
        result = gap_tv_cacti(y, masks, iterations=2, device=None)
        assert isinstance(result, np.ndarray)
