"""Tests for CASSI Algorithm 1 & 2 Implementations

Test comprehensive mismatch correction algorithms for SD-CASSI.
"""

import logging
import pytest
import numpy as np

from pwm_core.calibration import (
    Algorithm1HierarchicalBeamSearch,
    Algorithm2JointGradientRefinement,
    MismatchParameters,
    SimulatedOperatorEnlargedGrid,
    warp_affine_2d,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def synthetic_scene():
    """Create a synthetic hyperspectral scene."""
    np.random.seed(42)
    return np.random.rand(256, 256, 28).astype(np.float32) * 0.8


@pytest.fixture
def synthetic_mask():
    """Create a synthetic coded aperture mask."""
    np.random.seed(43)
    return (np.random.rand(256, 256).astype(np.float32) * 0.8 + 0.1)


@pytest.fixture
def synthetic_measurement(synthetic_scene):
    """Create a synthetic measurement."""
    return np.random.rand(256, 310).astype(np.float32) * 0.1


def mock_solver(y, operator, n_iter):
    """Mock solver for testing."""
    return np.ones((256, 256, 28), dtype=np.float32) * 0.5


# ============================================================================
# MismatchParameters Tests
# ============================================================================

class TestMismatchParameters:
    """Test MismatchParameters data class."""

    def test_initialization(self):
        """Test parameter initialization."""
        params = MismatchParameters(
            mask_dx=1.5, mask_dy=0.5, mask_theta=0.2,
            disp_a1=2.01, disp_alpha=0.1
        )
        assert params.mask_dx == 1.5
        assert params.mask_dy == 0.5
        assert params.mask_theta == 0.2
        assert params.disp_a1 == 2.01
        assert params.disp_alpha == 0.1

    def test_as_tuple(self):
        """Test as_tuple conversion."""
        params = MismatchParameters(
            mask_dx=1.0, mask_dy=2.0, mask_theta=3.0,
            disp_a1=2.01, disp_alpha=0.5
        )
        assert params.as_tuple() == (1.0, 2.0, 3.0, 2.01, 0.5)

    def test_copy(self):
        """Test parameter copying."""
        params = MismatchParameters(mask_dx=1.5, mask_dy=0.5)
        params_copy = params.copy()
        assert params_copy.mask_dx == params.mask_dx
        assert params_copy.mask_dy == params.mask_dy
        # Verify independent copy
        params_copy.mask_dx = 99.0
        assert params.mask_dx != params_copy.mask_dx


# ============================================================================
# Warp Affine Tests
# ============================================================================

class TestWarpAffine:
    """Test 2D affine warping."""

    def test_identity_warp(self, synthetic_mask):
        """Test identity warp (no transformation)."""
        warped = warp_affine_2d(synthetic_mask, dx=0, dy=0, theta=0)
        assert warped.shape == synthetic_mask.shape
        # Small difference due to interpolation
        assert np.allclose(warped, synthetic_mask, atol=0.05)

    def test_translation_warp(self, synthetic_mask):
        """Test translation warp."""
        warped = warp_affine_2d(synthetic_mask, dx=2, dy=1, theta=0)
        assert warped.shape == synthetic_mask.shape
        assert warped.dtype == synthetic_mask.dtype

    def test_rotation_warp(self, synthetic_mask):
        """Test rotation warp."""
        warped = warp_affine_2d(synthetic_mask, dx=0, dy=0, theta=5)
        assert warped.shape == synthetic_mask.shape
        assert warped.dtype == synthetic_mask.dtype


# ============================================================================
# SimulatedOperatorEnlargedGrid Tests
# ============================================================================

class TestSimulatedOperatorEnlargedGrid:
    """Test enlarged grid forward operator."""

    def test_operator_initialization(self, synthetic_mask):
        """Test operator initialization."""
        operator = SimulatedOperatorEnlargedGrid(synthetic_mask)
        assert operator.mask_256.shape == (256, 256)
        assert operator.mask_enlarged.shape == (1024, 1024)
        assert operator.N == 4
        assert operator.K == 2
        assert operator.stride == 1

    def test_operator_forward(self, synthetic_mask, synthetic_scene):
        """Test forward operator."""
        operator = SimulatedOperatorEnlargedGrid(synthetic_mask)
        y = operator.forward(synthetic_scene)
        assert y.shape == (256, 310)
        assert y.dtype == np.float32

    def test_mask_correction(self, synthetic_mask):
        """Test mask correction."""
        operator = SimulatedOperatorEnlargedGrid(synthetic_mask)
        mismatch = MismatchParameters(mask_dx=1.0, mask_dy=0.5, mask_theta=0.1)
        operator.apply_mask_correction(mismatch)
        # Verify mask is updated
        assert operator.mask_enlarged.shape == (1024, 1024)


# ============================================================================
# Algorithm 1 Tests
# ============================================================================

class TestAlgorithm1:
    """Test Algorithm 1: Hierarchical Beam Search."""

    def test_algorithm1_initialization(self):
        """Test Algorithm 1 initialization."""
        alg1 = Algorithm1HierarchicalBeamSearch(
            solver_fn=mock_solver,
            n_iter_proxy=2,
            n_iter_beam=2
        )
        assert alg1.solver_fn == mock_solver
        assert alg1.n_iter_proxy == 2
        assert alg1.n_iter_beam == 2

    def test_algorithm1_1d_sweep(self, synthetic_mask, synthetic_scene, synthetic_measurement):
        """Test 1D parameter sweep."""
        alg1 = Algorithm1HierarchicalBeamSearch(mock_solver)
        best_val = alg1.search_1d_parameter(
            'dx',
            np.linspace(-1, 1, 3),
            synthetic_measurement,
            synthetic_mask,
            synthetic_scene,
            SimulatedOperatorEnlargedGrid
        )
        assert isinstance(best_val, (int, float))
        assert -1 <= best_val <= 1

    def test_algorithm1_beam_search(self, synthetic_mask, synthetic_scene, synthetic_measurement):
        """Test 3D beam search."""
        alg1 = Algorithm1HierarchicalBeamSearch(mock_solver)
        candidates = alg1.beam_search_affine(
            dx_init=0.5, dy_init=0.5, theta_init=0.1,
            y_meas=synthetic_measurement,
            mask_real=synthetic_mask,
            x_true=synthetic_scene,
            operator_class=SimulatedOperatorEnlargedGrid,
            beam_width=3
        )
        assert len(candidates) == 3
        assert all(len(c) == 3 for c in candidates)  # (dx, dy, theta)

    def test_algorithm1_estimate(self, synthetic_mask, synthetic_scene, synthetic_measurement):
        """Test full Algorithm 1 estimation."""
        alg1 = Algorithm1HierarchicalBeamSearch(
            mock_solver,
            n_iter_proxy=1,
            n_iter_beam=1
        )
        mismatch = alg1.estimate(
            synthetic_measurement,
            synthetic_mask,
            synthetic_scene,
            SimulatedOperatorEnlargedGrid
        )
        assert isinstance(mismatch, MismatchParameters)
        assert abs(mismatch.mask_dx) <= 3
        assert abs(mismatch.mask_dy) <= 3
        assert abs(mismatch.mask_theta) <= 1
        assert 1.95 <= mismatch.disp_a1 <= 2.05
        assert abs(mismatch.disp_alpha) <= 1


# ============================================================================
# Algorithm 2 Tests
# ============================================================================

class TestAlgorithm2:
    """Test Algorithm 2: Joint Gradient Refinement."""

    def test_algorithm2_initialization(self):
        """Test Algorithm 2 initialization."""
        alg2 = Algorithm2JointGradientRefinement()
        assert alg2 is not None

    def test_algorithm2_refine(self, synthetic_scene, synthetic_measurement):
        """Test Algorithm 2 refinement."""
        alg2 = Algorithm2JointGradientRefinement()
        coarse = MismatchParameters(
            mask_dx=1.0, mask_dy=0.5, mask_theta=0.1,
            disp_a1=2.0, disp_alpha=0.0
        )
        refined = alg2.refine(coarse, synthetic_measurement, synthetic_scene)
        assert isinstance(refined, MismatchParameters)
        # In current implementation, returns coarse as fallback
        assert refined.mask_dx == coarse.mask_dx


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for both algorithms."""

    def test_algorithm1_then_algorithm2(self, synthetic_mask, synthetic_scene, synthetic_measurement):
        """Test full Algorithm 1 → Algorithm 2 pipeline."""
        # Algorithm 1
        alg1 = Algorithm1HierarchicalBeamSearch(
            mock_solver,
            n_iter_proxy=1,
            n_iter_beam=1
        )
        mismatch_coarse = alg1.estimate(
            synthetic_measurement,
            synthetic_mask,
            synthetic_scene,
            SimulatedOperatorEnlargedGrid
        )
        assert isinstance(mismatch_coarse, MismatchParameters)

        # Algorithm 2
        alg2 = Algorithm2JointGradientRefinement()
        mismatch_fine = alg2.refine(mismatch_coarse, synthetic_measurement, synthetic_scene)
        assert isinstance(mismatch_fine, MismatchParameters)

        # Both should have valid parameters
        assert abs(mismatch_fine.mask_dx) <= 3
        assert abs(mismatch_fine.mask_dy) <= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
