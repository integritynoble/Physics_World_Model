"""Regression tests: known-good phantom -> expected metrics within tolerance.

These tests ensure that metric computation is deterministic and
reproducible, and that the metrics degrade predictably with increasing
noise levels.

All metric functions accept 3-D volumes and return MetricResult objects.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwm_core.clinical.ct.qa_metrics import (
    compute_ct_number_insert,
    compute_ct_number_water,
    compute_geometric_accuracy,
    compute_noise_std,
    compute_uniformity,
)

from tests.clinical.conftest import (
    CENTER_X,
    CENTER_Y,
    INSERT_DEFS,
    PHANTOM_DIAMETER_MM,
    PHANTOM_MATRIX,
    PHANTOM_RADIUS_PX,
    PHANTOM_SLICES,
    PIXEL_SPACING_MM,
    _circle_mask,
)

_WATER_ROI_RADIUS_MM = 20.0
_INSERT_OFFSET_MM = 50.0
_INSERT_ROI_RADIUS_MM = 4.0


# ===================================================================
# Known-phantom regression
# ===================================================================


class TestKnownPhantomMetrics:
    """Verify all metrics on the synthetic phantom match expected values."""

    def test_known_phantom_metrics(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Compute each metric and compare against bounded expectations.

        Because the phantom uses a fixed seed (42), results should be
        reproducible across runs.
        """
        # Water CT number: expect ~0 HU (within noise)
        water = compute_ct_number_water(
            volume=synthetic_acr_phantom,
            center=(CENTER_Y, CENTER_X),
            radius_mm=_WATER_ROI_RADIUS_MM,
            pixel_spacing=pixel_spacing,
        )
        assert water.status == "computed"
        assert abs(water.value) < 3.0, (
            f"Water CT number regression: {water.value:.4f} HU"
        )

        # Noise std: expect ~5 HU
        noise = compute_noise_std(
            volume=synthetic_acr_phantom,
            center=(CENTER_Y, CENTER_X),
            radius_mm=_WATER_ROI_RADIUS_MM,
            pixel_spacing=pixel_spacing,
        )
        assert noise.status == "computed"
        assert abs(noise.value - 5.0) < 2.0, (
            f"Noise std regression: {noise.value:.4f} HU"
        )

        # Uniformity: expect < 5 HU
        # Use peripheral_offset_mm=25.0 to keep ROIs in pure water,
        # well away from material inserts at 50 mm from centre.
        uniformity = compute_uniformity(
            volume=synthetic_acr_phantom,
            center=(CENTER_Y, CENTER_X),
            radius_mm=10.0,
            peripheral_offset_mm=25.0,
            pixel_spacing=pixel_spacing,
        )
        assert uniformity.status == "computed"
        assert uniformity.value < 5.0, (
            f"Uniformity regression: {uniformity.value:.4f} HU"
        )

        # Geometric accuracy: expect error < 2 mm
        geo = compute_geometric_accuracy(
            volume=synthetic_acr_phantom,
            pixel_spacing=pixel_spacing,
            expected_diameter_mm=PHANTOM_DIAMETER_MM,
        )
        assert geo.status == "computed"
        assert geo.value < 2.0, (
            f"Geometric accuracy regression: {geo.value:.4f} mm"
        )


# ===================================================================
# Reproducibility
# ===================================================================


class TestReproducibility:
    """Running metrics twice on the same input should produce identical results."""

    def test_reproducibility(
        self,
        synthetic_acr_phantom: np.ndarray,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Compute metrics twice on the identical volume and verify
        that every value matches exactly.
        """
        results_a: dict[str, float] = {}
        results_b: dict[str, float] = {}

        for label, results in [("a", results_a), ("b", results_b)]:
            results["ct_number_water"] = compute_ct_number_water(
                volume=synthetic_acr_phantom,
                center=(CENTER_Y, CENTER_X),
                radius_mm=_WATER_ROI_RADIUS_MM,
                pixel_spacing=pixel_spacing,
            ).value

            results["noise_std"] = compute_noise_std(
                volume=synthetic_acr_phantom,
                center=(CENTER_Y, CENTER_X),
                radius_mm=_WATER_ROI_RADIUS_MM,
                pixel_spacing=pixel_spacing,
            ).value

            results["uniformity"] = compute_uniformity(
                volume=synthetic_acr_phantom,
                center=(CENTER_Y, CENTER_X),
                radius_mm=10.0,
                peripheral_offset_mm=25.0,
                pixel_spacing=pixel_spacing,
            ).value

            results["geometric_accuracy"] = compute_geometric_accuracy(
                volume=synthetic_acr_phantom,
                pixel_spacing=pixel_spacing,
                expected_diameter_mm=PHANTOM_DIAMETER_MM,
            ).value

        for key in results_a:
            assert results_a[key] == results_b[key], (
                f"Reproducibility failure for '{key}': "
                f"{results_a[key]} != {results_b[key]}"
            )


# ===================================================================
# Noise sensitivity
# ===================================================================


class TestNoiseSensitivity:
    """Metrics should degrade predictably with increasing noise."""

    def test_noise_sensitivity(
        self,
        pixel_spacing: tuple[float, float],
    ) -> None:
        """Generate phantoms with noise levels 2, 5, 10, and 20 HU.

        Verify that:
        1. measured noise_std increases monotonically
        2. water CT number remains unbiased (mean ~0)
        """
        noise_levels = [2.0, 5.0, 10.0, 20.0]
        measured_noise: list[float] = []
        measured_water: list[float] = []

        for noise_std in noise_levels:
            rng = np.random.default_rng(seed=int(noise_std * 100))
            vol = np.full(
                (PHANTOM_SLICES, PHANTOM_MATRIX, PHANTOM_MATRIX),
                -1000.0,
                dtype=np.float64,
            )
            body_mask = _circle_mask(
                (PHANTOM_MATRIX, PHANTOM_MATRIX),
                CENTER_Y, CENTER_X, PHANTOM_RADIUS_PX,
            )
            for z in range(PHANTOM_SLICES):
                vol[z][body_mask] = 0.0

            vol += rng.normal(0.0, noise_std, size=vol.shape)

            noise_result = compute_noise_std(
                volume=vol,
                center=(CENTER_Y, CENTER_X),
                radius_mm=_WATER_ROI_RADIUS_MM,
                pixel_spacing=pixel_spacing,
            )
            water_result = compute_ct_number_water(
                volume=vol,
                center=(CENTER_Y, CENTER_X),
                radius_mm=_WATER_ROI_RADIUS_MM,
                pixel_spacing=pixel_spacing,
            )

            measured_noise.append(noise_result.value)
            measured_water.append(water_result.value)

        # 1. Noise should increase monotonically
        for i in range(len(measured_noise) - 1):
            assert measured_noise[i] < measured_noise[i + 1], (
                f"Noise did not increase: {measured_noise}"
            )

        # 2. Water CT number should remain unbiased
        for i, w in enumerate(measured_water):
            assert abs(w) < 10.0, (
                f"Water CT number biased at noise level "
                f"{noise_levels[i]}: {w:.2f} HU"
            )
