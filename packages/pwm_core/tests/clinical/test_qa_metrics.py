"""Unit tests for CT QA metric computation.

Each test verifies a metric function against known ground truth
computed from the synthetic ACR phantom.  The phantom is generated
with deterministic geometry and noise (seed=42), so expected values
are tightly constrained.

All metric functions in ``qa_metrics.py`` accept 3-D volumes and return
:class:`MetricResult` objects with a ``.value`` attribute.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwm_core.clinical.ct.qa_metrics import (
    QAMetricsReport,
    compute_all_metrics,
    compute_ct_number_insert,
    compute_ct_number_water,
    compute_geometric_accuracy,
    compute_noise_std,
    compute_uniformity,
    _find_phantom_center,
)

from tests.clinical.conftest import (
    CENTER_X,
    CENTER_Y,
    INSERT_DEFS,
    INSERT_OFFSET_PX,
    PHANTOM_DIAMETER_MM,
    PHANTOM_MATRIX,
    PIXEL_SPACING_MM,
)

# Convert insert offset from pixels to mm for the API
_INSERT_OFFSET_MM = 50.0
_INSERT_ROI_RADIUS_MM = 4.0
_WATER_ROI_RADIUS_MM = 20.0


# ===================================================================
# CT Number tests
# ===================================================================


class TestCTNumberWater:
    """Water ROI CT number should be close to 0 HU."""

    def test_ct_number_water(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Measure CT number in a central water ROI.

        The water region is the centre of the phantom body (no inserts).
        With 5 HU Gaussian noise, the mean should be within ~3 HU of 0.
        """
        result = compute_ct_number_water(
            volume=synthetic_acr_phantom,
            center=(CENTER_Y, CENTER_X),
            radius_mm=_WATER_ROI_RADIUS_MM,
            pixel_spacing=pixel_spacing,
        )
        assert result.status == "computed"
        assert abs(result.value) < 3.0, (
            f"Water CT number {result.value:.2f} HU is too far from 0 HU"
        )


class TestCTNumberBone:
    """Bone insert CT number should be close to 955 HU."""

    def test_ct_number_bone(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Measure CT number at the bone insert location (90 degrees, 955 HU nominal)."""
        angle_deg, expected_hu = INSERT_DEFS["bone"]
        result = compute_ct_number_insert(
            volume=synthetic_acr_phantom,
            insert_name="bone",
            center=(CENTER_Y, CENTER_X),
            angle_deg=angle_deg,
            radius_mm=_INSERT_ROI_RADIUS_MM,
            offset_mm=_INSERT_OFFSET_MM,
            pixel_spacing=pixel_spacing,
            expected_hu=expected_hu,
        )
        assert result.status == "computed"
        assert abs(result.value - expected_hu) < 30.0, (
            f"Bone CT number {result.value:.2f} HU too far from {expected_hu} HU"
        )


class TestCTNumberAir:
    """Air insert CT number should be close to -1000 HU."""

    def test_ct_number_air(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Measure CT number at the air insert location (180 degrees, -1000 HU nominal)."""
        angle_deg, expected_hu = INSERT_DEFS["air"]
        result = compute_ct_number_insert(
            volume=synthetic_acr_phantom,
            insert_name="air",
            center=(CENTER_Y, CENTER_X),
            angle_deg=angle_deg,
            radius_mm=_INSERT_ROI_RADIUS_MM,
            offset_mm=_INSERT_OFFSET_MM,
            pixel_spacing=pixel_spacing,
            expected_hu=expected_hu,
        )
        assert result.status == "computed"
        assert abs(result.value - expected_hu) < 30.0, (
            f"Air CT number {result.value:.2f} HU too far from {expected_hu} HU"
        )


class TestCTNumberAcrylic:
    """Acrylic insert CT number should be close to 120 HU."""

    def test_ct_number_acrylic(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Measure CT number at the acrylic insert location (270 degrees, 120 HU nominal)."""
        angle_deg, expected_hu = INSERT_DEFS["acrylic"]
        result = compute_ct_number_insert(
            volume=synthetic_acr_phantom,
            insert_name="acrylic",
            center=(CENTER_Y, CENTER_X),
            angle_deg=angle_deg,
            radius_mm=_INSERT_ROI_RADIUS_MM,
            offset_mm=_INSERT_OFFSET_MM,
            pixel_spacing=pixel_spacing,
            expected_hu=expected_hu,
        )
        assert result.status == "computed"
        assert abs(result.value - expected_hu) < 30.0, (
            f"Acrylic CT number {result.value:.2f} HU too far from {expected_hu} HU"
        )


class TestCTNumberPolyethylene:
    """Polyethylene insert CT number should be close to -95 HU."""

    def test_ct_number_polyethylene(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Measure CT number at the polyethylene insert (0 degrees, -95 HU nominal)."""
        angle_deg, expected_hu = INSERT_DEFS["polyethylene"]
        result = compute_ct_number_insert(
            volume=synthetic_acr_phantom,
            insert_name="polyethylene",
            center=(CENTER_Y, CENTER_X),
            angle_deg=angle_deg,
            radius_mm=_INSERT_ROI_RADIUS_MM,
            offset_mm=_INSERT_OFFSET_MM,
            pixel_spacing=pixel_spacing,
            expected_hu=expected_hu,
        )
        assert result.status == "computed"
        assert abs(result.value - expected_hu) < 30.0, (
            f"Polyethylene CT number {result.value:.2f} HU too far from {expected_hu} HU"
        )


# ===================================================================
# Uniformity
# ===================================================================


class TestUniformity:
    """Uniformity (max difference between centre and peripheral ROIs)."""

    def test_uniformity(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """A uniform water phantom should have uniformity < 5 HU.

        Uniformity is defined as the maximum absolute difference between the
        central ROI mean and any of the four peripheral ROI means.

        We use peripheral_offset_mm=25.0 to keep the ROIs away from the
        material inserts that are placed at 50 mm from centre.
        """
        result = compute_uniformity(
            volume=synthetic_acr_phantom,
            center=(CENTER_Y, CENTER_X),
            radius_mm=10.0,
            peripheral_offset_mm=25.0,
            pixel_spacing=pixel_spacing,
        )
        assert result.status == "computed"
        assert result.value < 5.0, (
            f"Uniformity {result.value:.2f} HU exceeds 5 HU on a clean phantom"
        )


# ===================================================================
# Noise
# ===================================================================


class TestNoiseStd:
    """Noise standard deviation in the uniform water region."""

    def test_noise_std(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Measured noise should be close to the injected 5 HU (within +/- 2)."""
        result = compute_noise_std(
            volume=synthetic_acr_phantom,
            center=(CENTER_Y, CENTER_X),
            radius_mm=_WATER_ROI_RADIUS_MM,
            pixel_spacing=pixel_spacing,
        )
        assert result.status == "computed"
        assert abs(result.value - 5.0) < 2.0, (
            f"Noise std {result.value:.2f} HU is not close to injected 5 HU"
        )


# ===================================================================
# Geometric accuracy
# ===================================================================


class TestGeometricAccuracy:
    """Measured phantom diameter error should be small."""

    def test_geometric_accuracy(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """The maximum diameter error should be within 2 mm of the true 200 mm.

        compute_geometric_accuracy returns the max absolute error in mm.
        """
        result = compute_geometric_accuracy(
            volume=synthetic_acr_phantom,
            pixel_spacing=pixel_spacing,
            expected_diameter_mm=PHANTOM_DIAMETER_MM,
        )
        assert result.status == "computed"
        assert result.value < 2.0, (
            f"Geometric accuracy error {result.value:.2f} mm exceeds 2 mm"
        )


# ===================================================================
# Phantom centre detection
# ===================================================================


class TestPhantomCenterDetection:
    """Auto-detected phantom centre should be within 2 pixels of truth."""

    def test_phantom_center_detection(
        self,
        synthetic_acr_phantom: np.ndarray,
    ) -> None:
        """Detect the phantom centre from the central slice and compare
        to the known (128, 128).
        """
        mid_slice = synthetic_acr_phantom[synthetic_acr_phantom.shape[0] // 2]
        cy, cx = _find_phantom_center(mid_slice)
        assert abs(cy - CENTER_Y) < 2.0, (
            f"Detected centre Y={cy} is >2 px from true {CENTER_Y}"
        )
        assert abs(cx - CENTER_X) < 2.0, (
            f"Detected centre X={cx} is >2 px from true {CENTER_X}"
        )


# ===================================================================
# Compute-all integration
# ===================================================================


class TestComputeAllMetrics:
    """compute_all_metrics should return a complete QAMetricsReport."""

    def test_compute_all_metrics(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
        sample_casepack_config: dict,
    ) -> None:
        """Run the full metric battery on the synthetic phantom.

        compute_all_metrics expects a CTScanBundle-like object and a
        casepack config dict.
        """
        from pwm_core.clinical.ct.dicom_ingester import CTScanBundle

        # Build a minimal scan bundle from the synthetic phantom
        scan_bundle = CTScanBundle(
            scanner_id="SYN-CT-001",
            study_date="20260101",
            manufacturer="SYNTHETIC",
            model="TestScanner",
            slices=synthetic_acr_phantom,
            pixel_spacing_mm=pixel_spacing,
            slice_thickness_mm=5.0,
        )

        report = compute_all_metrics(
            scan_bundle=scan_bundle,
            casepack=sample_casepack_config,
        )

        assert isinstance(report, QAMetricsReport)

        expected_keys = set(
            sample_casepack_config["casepack"]["metric_set"]
        )
        actual_keys = set(report.metrics.keys())
        missing = expected_keys - actual_keys
        assert not missing, (
            f"QAMetricsReport is missing metrics: {missing}"
        )
