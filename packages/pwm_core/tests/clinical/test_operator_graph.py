"""Comprehensive tests for CTOperatorGraph and related CT models.

Tests cover geometry classification, forward/adjoint projection shapes,
reprojection error behaviour, HU calibration estimation, operator
correction logic, and MismatchEstimate validation constraints.

All tests use small images (32x32 or 64x64) for speed.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

scipy = pytest.importorskip("scipy")

from pwm_core.clinical.ct.operator_graph import (
    CTGeometry,
    CTOperatorGraph,
    CTOperatorParams,
    MismatchEstimate,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_parallel_geometry(
    num_angles: int = 30,
    num_detectors: int = 32,
    detector_spacing_mm: float = 1.0,
    **kwargs,
) -> CTGeometry:
    """Create a parallel-beam CTGeometry (source_to_center_mm=0)."""
    return CTGeometry(
        num_angles=num_angles,
        num_detectors=num_detectors,
        detector_spacing_mm=detector_spacing_mm,
        source_to_center_mm=0.0,
        source_to_detector_mm=0.0,
        **kwargs,
    )


def _make_fan_geometry(
    num_angles: int = 30,
    num_detectors: int = 32,
    detector_spacing_mm: float = 1.0,
    source_to_center_mm: float = 500.0,
    source_to_detector_mm: float = 1000.0,
    **kwargs,
) -> CTGeometry:
    """Create a fan-beam CTGeometry (source_to_center_mm > 0)."""
    return CTGeometry(
        num_angles=num_angles,
        num_detectors=num_detectors,
        detector_spacing_mm=detector_spacing_mm,
        source_to_center_mm=source_to_center_mm,
        source_to_detector_mm=source_to_detector_mm,
        **kwargs,
    )


def _make_operator(geometry: CTGeometry, **kwargs) -> CTOperatorGraph:
    """Create a CTOperatorGraph from a geometry with optional param overrides."""
    params = CTOperatorParams(geometry=geometry, **kwargs)
    return CTOperatorGraph(params)


def _disk_phantom(n: int = 32, radius: float | None = None) -> np.ndarray:
    """Create a centred disk phantom of shape (n, n)."""
    if radius is None:
        radius = n / 4.0
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2.0
    image = np.zeros((n, n), dtype=np.float64)
    image[xx ** 2 + yy ** 2 <= radius ** 2] = 1.0
    return image


# ---------------------------------------------------------------------------
# 1 & 2. Geometry beam-type properties
# ---------------------------------------------------------------------------

class TestGeometryBeamType:
    """Tests for CTGeometry.is_parallel_beam and .is_fan_beam."""

    def test_geometry_parallel_beam(self):
        geom = _make_parallel_geometry()
        assert geom.is_parallel_beam is True
        assert geom.is_fan_beam is False

    def test_geometry_fan_beam(self):
        geom = _make_fan_geometry()
        assert geom.is_fan_beam is True
        assert geom.is_parallel_beam is False

    def test_geometry_zero_source_is_parallel(self):
        """Explicit source_to_center_mm=0.0 gives parallel beam."""
        geom = CTGeometry(
            num_angles=10,
            num_detectors=16,
            detector_spacing_mm=1.0,
            source_to_center_mm=0.0,
        )
        assert geom.is_parallel_beam is True
        assert geom.is_fan_beam is False


# ---------------------------------------------------------------------------
# 3 & 4. Forward projection
# ---------------------------------------------------------------------------

class TestForwardProjection:
    """Tests for CTOperatorGraph.forward()."""

    def test_forward_output_shape(self):
        """Parallel-beam forward should return (num_angles, num_detectors)."""
        num_angles, num_detectors = 20, 32
        geom = _make_parallel_geometry(
            num_angles=num_angles, num_detectors=num_detectors,
        )
        op = _make_operator(geom)
        image = _disk_phantom(32)

        sinogram = op.forward(image)

        assert sinogram.shape == (num_angles, num_detectors)
        assert sinogram.dtype == np.float64

    def test_forward_zero_image_gives_zero_sinogram(self):
        """A zero image must produce an all-zero sinogram (no noise)."""
        geom = _make_parallel_geometry(num_angles=15, num_detectors=32)
        op = _make_operator(geom, noise_std=0.0)
        image = np.zeros((32, 32), dtype=np.float64)

        sinogram = op.forward(image)

        np.testing.assert_allclose(sinogram, 0.0, atol=1e-14)

    def test_forward_nonzero_image_gives_nonzero_sinogram(self):
        """A non-trivial phantom must produce a sinogram with non-zero values."""
        geom = _make_parallel_geometry(num_angles=20, num_detectors=32)
        op = _make_operator(geom)
        image = _disk_phantom(32)

        sinogram = op.forward(image)

        assert np.any(sinogram > 0), "Sinogram of a disk phantom should not be all-zero"

    def test_forward_fan_beam_output_shape(self):
        """Fan-beam forward should also return (num_angles, num_detectors)."""
        num_angles, num_detectors = 15, 32
        geom = _make_fan_geometry(
            num_angles=num_angles, num_detectors=num_detectors,
        )
        op = _make_operator(geom)
        image = _disk_phantom(32)

        sinogram = op.forward(image)

        assert sinogram.shape == (num_angles, num_detectors)


# ---------------------------------------------------------------------------
# 5. Adjoint (backprojection)
# ---------------------------------------------------------------------------

class TestAdjoint:
    """Tests for CTOperatorGraph.adjoint()."""

    def test_adjoint_output_shape(self):
        """Adjoint should return a square image with side = num_detectors."""
        num_angles, num_detectors = 20, 32
        geom = _make_parallel_geometry(
            num_angles=num_angles, num_detectors=num_detectors,
        )
        op = _make_operator(geom)
        sinogram = np.random.default_rng(42).standard_normal(
            (num_angles, num_detectors)
        )

        backprojected = op.adjoint(sinogram)

        assert backprojected.shape == (num_detectors, num_detectors)
        assert backprojected.dtype == np.float64

    def test_adjoint_zero_sinogram_gives_zero_image(self):
        """Zero sinogram must produce a zero backprojected image."""
        num_angles, num_detectors = 20, 32
        geom = _make_parallel_geometry(
            num_angles=num_angles, num_detectors=num_detectors,
        )
        op = _make_operator(geom)
        sinogram = np.zeros((num_angles, num_detectors), dtype=np.float64)

        backprojected = op.adjoint(sinogram)

        np.testing.assert_allclose(backprojected, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 6 & 7. Reprojection error
# ---------------------------------------------------------------------------

class TestReprojectionError:
    """Tests for CTOperatorGraph.reprojection_error()."""

    def test_reprojection_error_perfect_match_near_zero(self):
        """When measured sinogram == forward(image), error should be ~0."""
        geom = _make_parallel_geometry(num_angles=20, num_detectors=32)
        op = _make_operator(geom, noise_std=0.0)
        image = _disk_phantom(32)

        sinogram = op.forward(image)
        error = op.reprojection_error(image, sinogram)

        assert error == pytest.approx(0.0, abs=1e-10)

    def test_reprojection_error_mismatched_is_positive(self):
        """A random measured sinogram should give a positive error."""
        geom = _make_parallel_geometry(num_angles=20, num_detectors=32)
        op = _make_operator(geom, noise_std=0.0)
        image = _disk_phantom(32)

        rng = np.random.default_rng(99)
        fake_sinogram = rng.standard_normal((20, 32))

        error = op.reprojection_error(image, fake_sinogram)

        assert error > 0.0, "Mismatched sinogram should give positive error"

    def test_reprojection_error_zero_measured_returns_inf(self):
        """If the measured sinogram is all zeros, error should be inf."""
        geom = _make_parallel_geometry(num_angles=20, num_detectors=32)
        op = _make_operator(geom, noise_std=0.0)
        image = _disk_phantom(32)
        zero_sinogram = np.zeros((20, 32), dtype=np.float64)

        error = op.reprojection_error(image, zero_sinogram)

        assert error == float("inf")


# ---------------------------------------------------------------------------
# 8 & 9. HU calibration estimation
# ---------------------------------------------------------------------------

class TestHUCalibrationEstimation:
    """Tests for CTOperatorGraph.estimate_hu_calibration()."""

    def test_estimate_hu_calibration_perfect_calibration(self):
        """When measured == expected, slope should be ~1 and intercept ~0."""
        geom = _make_parallel_geometry()
        op = _make_operator(geom)

        phantom_rois = {
            "water": 0.0,
            "air": -1000.0,
            "bone": 1000.0,
            "acrylic": 120.0,
        }
        expected_hu = {
            "water": 0.0,
            "air": -1000.0,
            "bone": 1000.0,
            "acrylic": 120.0,
        }

        slope_est, intercept_est = op.estimate_hu_calibration(
            phantom_rois, expected_hu,
        )

        assert slope_est.parameter_name == "hu_calibration_slope"
        assert slope_est.estimated_value == pytest.approx(1.0, abs=1e-6)
        assert slope_est.method == "linear_regression"

        assert intercept_est.parameter_name == "hu_calibration_intercept"
        assert intercept_est.estimated_value == pytest.approx(0.0, abs=1e-4)
        assert intercept_est.method == "linear_regression"

        # Confidence should be very high (R^2 = 1.0 for perfect fit)
        assert slope_est.confidence == pytest.approx(1.0, abs=1e-4)

    def test_estimate_hu_calibration_with_drift(self):
        """A known linear drift (slope=1.05, intercept=10) should be recovered."""
        geom = _make_parallel_geometry()
        op = _make_operator(geom)

        drift_slope = 1.05
        drift_intercept = 10.0

        expected_hu = {
            "water": 0.0,
            "air": -1000.0,
            "bone": 1000.0,
            "polyethylene": -100.0,
            "acrylic": 120.0,
        }
        # Simulate drifted measurements
        phantom_rois = {
            k: drift_slope * v + drift_intercept
            for k, v in expected_hu.items()
        }

        slope_est, intercept_est = op.estimate_hu_calibration(
            phantom_rois, expected_hu,
        )

        assert slope_est.estimated_value == pytest.approx(drift_slope, abs=1e-5)
        assert intercept_est.estimated_value == pytest.approx(drift_intercept, abs=1e-3)

    def test_estimate_hu_calibration_needs_at_least_2_inserts(self):
        """Fewer than 2 matching inserts must raise ValueError."""
        geom = _make_parallel_geometry()
        op = _make_operator(geom)

        # Only 1 overlapping key
        phantom_rois = {"water": 0.0}
        expected_hu = {"water": 0.0, "air": -1000.0}

        with pytest.raises(ValueError, match="at least 2"):
            op.estimate_hu_calibration(phantom_rois, expected_hu)

    def test_estimate_hu_calibration_no_overlap_raises(self):
        """Zero overlapping keys must raise ValueError."""
        geom = _make_parallel_geometry()
        op = _make_operator(geom)

        phantom_rois = {"insert_A": 50.0}
        expected_hu = {"insert_B": 50.0}

        with pytest.raises(ValueError, match="at least 2"):
            op.estimate_hu_calibration(phantom_rois, expected_hu)


# ---------------------------------------------------------------------------
# 10 & 11. Operator correction
# ---------------------------------------------------------------------------

class TestCorrectOperator:
    """Tests for CTOperatorGraph.correct_operator()."""

    def test_correct_operator_applies_estimates(self):
        """High-confidence estimates should be applied to the corrected params."""
        geom = _make_parallel_geometry()
        op = _make_operator(geom)

        estimates = [
            MismatchEstimate(
                parameter_name="center_of_rotation_offset_mm",
                estimated_value=2.5,
                confidence=0.9,
                method="sinogram_symmetry",
            ),
            MismatchEstimate(
                parameter_name="hu_calibration_slope",
                estimated_value=1.03,
                confidence=0.85,
                method="linear_regression",
            ),
            MismatchEstimate(
                parameter_name="hu_calibration_intercept",
                estimated_value=5.0,
                confidence=0.85,
                method="linear_regression",
            ),
        ]

        corrected = op.correct_operator(estimates)

        assert isinstance(corrected, CTOperatorParams)
        assert corrected.geometry.center_of_rotation_offset_mm == 2.5
        assert corrected.hu_calibration_slope == 1.03
        assert corrected.hu_calibration_intercept == 5.0

    def test_correct_operator_skips_low_confidence(self):
        """Estimates with confidence < 0.1 should not be applied."""
        original_cor = 0.0
        geom = _make_parallel_geometry(center_of_rotation_offset_mm=original_cor)
        op = _make_operator(geom, hu_calibration_slope=1.0)

        estimates = [
            MismatchEstimate(
                parameter_name="center_of_rotation_offset_mm",
                estimated_value=5.0,
                confidence=0.05,  # Below the 0.1 threshold
                method="sinogram_symmetry",
            ),
            MismatchEstimate(
                parameter_name="hu_calibration_slope",
                estimated_value=1.2,
                confidence=0.02,  # Below the 0.1 threshold
                method="linear_regression",
            ),
        ]

        corrected = op.correct_operator(estimates)

        # Both should remain at their original values
        assert corrected.geometry.center_of_rotation_offset_mm == original_cor
        assert corrected.hu_calibration_slope == 1.0

    def test_correct_operator_does_not_mutate_original(self):
        """Calling correct_operator should leave the original params unchanged."""
        geom = _make_parallel_geometry(center_of_rotation_offset_mm=0.0)
        op = _make_operator(geom, hu_calibration_slope=1.0)

        estimates = [
            MismatchEstimate(
                parameter_name="center_of_rotation_offset_mm",
                estimated_value=3.0,
                confidence=0.9,
                method="sinogram_symmetry",
            ),
        ]

        corrected = op.correct_operator(estimates)

        # Original should be untouched
        assert op.params.geometry.center_of_rotation_offset_mm == 0.0
        # Corrected should have the new value
        assert corrected.geometry.center_of_rotation_offset_mm == 3.0

    def test_correct_operator_empty_estimates(self):
        """An empty estimate list should return params identical to the original."""
        geom = _make_parallel_geometry(center_of_rotation_offset_mm=1.5)
        op = _make_operator(geom, hu_calibration_slope=0.98)

        corrected = op.correct_operator([])

        assert corrected.geometry.center_of_rotation_offset_mm == 1.5
        assert corrected.hu_calibration_slope == 0.98

    def test_correct_operator_applies_detector_offset(self):
        """detector_offset_mm should be applied when confidence is sufficient."""
        geom = _make_parallel_geometry()
        op = _make_operator(geom)

        estimates = [
            MismatchEstimate(
                parameter_name="detector_offset_mm",
                estimated_value=0.75,
                confidence=0.5,
                method="manual",
            ),
        ]

        corrected = op.correct_operator(estimates)

        assert corrected.geometry.detector_offset_mm == 0.75


# ---------------------------------------------------------------------------
# 12. MismatchEstimate confidence bounds
# ---------------------------------------------------------------------------

class TestMismatchEstimate:
    """Tests for MismatchEstimate Pydantic validation."""

    def test_mismatch_estimate_confidence_bounds(self):
        """Confidence must be in [0.0, 1.0]."""
        # Valid at boundaries
        est_0 = MismatchEstimate(
            parameter_name="test",
            estimated_value=1.0,
            confidence=0.0,
            method="test_method",
        )
        assert est_0.confidence == 0.0

        est_1 = MismatchEstimate(
            parameter_name="test",
            estimated_value=1.0,
            confidence=1.0,
            method="test_method",
        )
        assert est_1.confidence == 1.0

    def test_mismatch_estimate_confidence_above_one_rejected(self):
        """Confidence > 1.0 must be rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            MismatchEstimate(
                parameter_name="test",
                estimated_value=1.0,
                confidence=1.5,
                method="test_method",
            )

    def test_mismatch_estimate_confidence_below_zero_rejected(self):
        """Confidence < 0.0 must be rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            MismatchEstimate(
                parameter_name="test",
                estimated_value=1.0,
                confidence=-0.1,
                method="test_method",
            )

    def test_mismatch_estimate_fields_stored_correctly(self):
        """All fields should round-trip correctly."""
        est = MismatchEstimate(
            parameter_name="center_of_rotation_offset_mm",
            estimated_value=2.345,
            confidence=0.87,
            method="sinogram_symmetry",
        )
        assert est.parameter_name == "center_of_rotation_offset_mm"
        assert est.estimated_value == 2.345
        assert est.confidence == 0.87
        assert est.method == "sinogram_symmetry"


# ---------------------------------------------------------------------------
# CoR offset estimation (smoke test)
# ---------------------------------------------------------------------------

class TestEstimateCorOffset:
    """Smoke tests for CTOperatorGraph.estimate_cor_offset()."""

    def test_estimate_cor_offset_returns_mismatch_estimate(self):
        """estimate_cor_offset should return a MismatchEstimate."""
        geom = _make_parallel_geometry(num_angles=30, num_detectors=32)
        op = _make_operator(geom, noise_std=0.0)
        image = _disk_phantom(32)
        sinogram = op.forward(image)

        result = op.estimate_cor_offset(sinogram)

        assert isinstance(result, MismatchEstimate)
        assert result.parameter_name == "center_of_rotation_offset_mm"
        assert result.method == "sinogram_symmetry"
        assert 0.0 <= result.confidence <= 1.0

    def test_estimate_cor_offset_too_few_angles(self):
        """With very few angles (< 4), confidence should be low."""
        geom = _make_parallel_geometry(num_angles=2, num_detectors=32)
        op = _make_operator(geom, noise_std=0.0)
        sinogram = np.ones((2, 32), dtype=np.float64)

        result = op.estimate_cor_offset(sinogram)

        # Only 1 conjugate pair -> low confidence
        assert result.confidence <= 0.5
