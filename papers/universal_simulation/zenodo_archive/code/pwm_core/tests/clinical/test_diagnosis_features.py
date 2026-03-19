"""Unit tests for artifact feature extraction.

Each test verifies that a specific artifact feature index responds
correctly to the presence or absence of the corresponding artifact
in the synthetic ACR phantom volume.

All functions in ``diagnosis_features.py`` operate on 2-D slices and
return scalar float values.
"""

from __future__ import annotations

import numpy as np
import pytest

from pwm_core.clinical.ct.diagnosis_features import (
    compute_cupping_index,
    compute_hu_drift_index,
    compute_noise_ratio,
    compute_ring_index,
    compute_streak_index,
)

from tests.clinical.conftest import (
    CENTER_X,
    CENTER_Y,
    PHANTOM_MATRIX,
    PHANTOM_RADIUS_PX,
    PIXEL_SPACING_MM,
)

# Pixel-domain radii for ring analysis.
# The ring artifact is injected in the band 30-50 px from centre.
# Use an analysis annulus that covers this band.
_INNER_RADIUS_PX = 25
_OUTER_RADIUS_PX = 55
_RADIUS_PX = int(PHANTOM_RADIUS_PX)


# ===================================================================
# Ring index
# ===================================================================


class TestRingIndex:
    """Ring artifact feature index tests."""

    def test_ring_index_clean(
        self,
        synthetic_acr_phantom_with_artifacts: dict[str, np.ndarray],
    ) -> None:
        """Clean phantom should have ring_index < 1.5 (near 1.0 for uniform).

        On a clean phantom the azimuthal variance at any fixed radius
        should be dominated by noise alone, giving a ratio near 1.0.
        """
        clean = synthetic_acr_phantom_with_artifacts["clean"]
        mid_slice = clean[clean.shape[0] // 2]

        ring_idx = compute_ring_index(
            volume_slice=mid_slice,
            center=(CENTER_Y, CENTER_X),
            inner_radius_px=_INNER_RADIUS_PX,
            outer_radius_px=_OUTER_RADIUS_PX,
        )
        # On a clean phantom with noise, the ring index is the mean
        # azimuthal variance / overall variance, which should be close
        # to 1.0 (no systematic azimuthal structure).
        assert ring_idx < 1.5, (
            f"Ring index {ring_idx:.3f} on clean phantom is unexpectedly high"
        )

    def test_ring_index_artifact(
        self,
        synthetic_acr_phantom_with_artifacts: dict[str, np.ndarray],
    ) -> None:
        """Phantom with injected ring artifact should have higher ring_index.

        The ring artifact adds a 30 HU sinusoidal azimuthal modulation
        in a narrow radial band, increasing azimuthal variance.
        """
        ring = synthetic_acr_phantom_with_artifacts["ring"]
        mid_slice = ring[ring.shape[0] // 2]

        clean = synthetic_acr_phantom_with_artifacts["clean"]
        clean_slice = clean[clean.shape[0] // 2]

        ring_idx_artifact = compute_ring_index(
            volume_slice=mid_slice,
            center=(CENTER_Y, CENTER_X),
            inner_radius_px=_INNER_RADIUS_PX,
            outer_radius_px=_OUTER_RADIUS_PX,
        )
        ring_idx_clean = compute_ring_index(
            volume_slice=clean_slice,
            center=(CENTER_Y, CENTER_X),
            inner_radius_px=_INNER_RADIUS_PX,
            outer_radius_px=_OUTER_RADIUS_PX,
        )
        assert ring_idx_artifact > ring_idx_clean, (
            f"Ring index with artifact ({ring_idx_artifact:.3f}) should exceed "
            f"clean ({ring_idx_clean:.3f})"
        )


# ===================================================================
# Cupping index
# ===================================================================


class TestCuppingIndex:
    """Cupping artifact feature index tests."""

    def test_cupping_index_clean(
        self,
        synthetic_acr_phantom_with_artifacts: dict[str, np.ndarray],
    ) -> None:
        """Clean phantom should have cupping_index close to 0.

        Cupping index = centre mean - periphery mean.  On a flat water
        phantom, this should be small (noise-dominated).
        """
        clean = synthetic_acr_phantom_with_artifacts["clean"]
        mid_slice = clean[clean.shape[0] // 2]

        cupping_idx = compute_cupping_index(
            volume_slice=mid_slice,
            center=(CENTER_Y, CENTER_X),
            radius_px=_RADIUS_PX,
        )
        assert abs(cupping_idx) < 3.0, (
            f"Cupping index {cupping_idx:.2f} HU on clean phantom is too large"
        )

    def test_cupping_index_artifact(
        self,
        synthetic_acr_phantom_with_artifacts: dict[str, np.ndarray],
    ) -> None:
        """Phantom with cupping artifact should have cupping_index > 5 HU.

        The injected cupping adds a -20 HU parabolic depression at the
        centre, making centre darker than periphery.  The cupping_index
        (centre - periphery) should be negative and large in magnitude.
        """
        cupping = synthetic_acr_phantom_with_artifacts["cupping"]
        mid_slice = cupping[cupping.shape[0] // 2]

        cupping_idx = compute_cupping_index(
            volume_slice=mid_slice,
            center=(CENTER_Y, CENTER_X),
            radius_px=_RADIUS_PX,
        )
        # Centre is depressed by ~20 HU, so cupping_idx (centre - periph)
        # should be significantly negative.
        assert abs(cupping_idx) > 5.0, (
            f"Cupping index magnitude {abs(cupping_idx):.2f} HU on cupping phantom is below 5"
        )


# ===================================================================
# Streak index
# ===================================================================


class TestStreakIndex:
    """Streak artifact feature index tests."""

    def test_streak_index_clean(
        self,
        synthetic_acr_phantom_with_artifacts: dict[str, np.ndarray],
    ) -> None:
        """Clean phantom should have streak_index near 1.0 (isotropic).

        Without directional artifacts, the gradient energy should be
        roughly equal in all directions, yielding a ratio near 1.0.
        """
        clean = synthetic_acr_phantom_with_artifacts["clean"]
        mid_slice = clean[clean.shape[0] // 2]

        streak_idx = compute_streak_index(
            volume_slice=mid_slice,
            center=(CENTER_Y, CENTER_X),
            radius_px=_RADIUS_PX,
        )
        # Isotropic noise -> ratio near 1.0
        assert streak_idx < 1.5, (
            f"Streak index {streak_idx:.3f} on clean phantom is too high"
        )

    def test_streak_index_artifact(
        self,
        synthetic_acr_phantom_with_artifacts: dict[str, np.ndarray],
    ) -> None:
        """Phantom with streak artifact should have elevated streak_index.

        The injected streak adds +200 HU horizontal lines through the
        centre, creating strong horizontal gradient energy.
        """
        streak = synthetic_acr_phantom_with_artifacts["streak"]
        mid_slice = streak[streak.shape[0] // 2]

        clean = synthetic_acr_phantom_with_artifacts["clean"]
        clean_slice = clean[clean.shape[0] // 2]

        streak_idx_artifact = compute_streak_index(
            volume_slice=mid_slice,
            center=(CENTER_Y, CENTER_X),
            radius_px=_RADIUS_PX,
        )
        streak_idx_clean = compute_streak_index(
            volume_slice=clean_slice,
            center=(CENTER_Y, CENTER_X),
            radius_px=_RADIUS_PX,
        )
        assert streak_idx_artifact > streak_idx_clean, (
            f"Streak index with artifact ({streak_idx_artifact:.3f}) should "
            f"exceed clean ({streak_idx_clean:.3f})"
        )


# ===================================================================
# Noise ratio
# ===================================================================


class TestNoiseRatio:
    """Noise ratio feature index tests.

    compute_noise_ratio(current_noise_std, baseline_noise_std) -> float
    """

    def test_noise_ratio_stable(self) -> None:
        """Comparing same noise level should yield ratio ~1.0."""
        ratio = compute_noise_ratio(
            current_noise_std=5.0,
            baseline_noise_std=5.0,
        )
        assert abs(ratio - 1.0) < 0.01, (
            f"Noise ratio {ratio:.4f} should be 1.0 for equal noise levels"
        )

    def test_noise_ratio_degraded(self) -> None:
        """Doubled noise should yield a ratio of 2.0."""
        ratio = compute_noise_ratio(
            current_noise_std=10.0,
            baseline_noise_std=5.0,
        )
        assert abs(ratio - 2.0) < 0.01, (
            f"Noise ratio {ratio:.4f} should be 2.0 for doubled noise"
        )


# ===================================================================
# HU drift index
# ===================================================================


class TestHUDriftIndex:
    """HU drift feature index tests.

    compute_hu_drift_index(current_inserts, baseline_inserts) -> float
    The index is RMS deviation / expected range across inserts.
    """

    def test_hu_drift_index_stable(self) -> None:
        """Same insert measurements should yield drift_index == 0."""
        current_inserts = {
            "bone": 955.0,
            "air": -1000.0,
            "acrylic": 120.0,
            "polyethylene": -95.0,
        }
        baseline_inserts = dict(current_inserts)

        drift = compute_hu_drift_index(
            current_inserts=current_inserts,
            baseline_inserts=baseline_inserts,
        )
        assert drift < 0.001, (
            f"HU drift index {drift:.6f} should be ~0 for identical inserts"
        )

    def test_hu_drift_index_drifted(self) -> None:
        """Shifted insert values should yield a positive drift_index.

        A systematic +30 HU shift on all inserts.  The expected range
        for ACR inserts is 955 - (-1000) = 1955 HU.  RMS of a constant
        30 HU shift = 30 HU.  Index = 30 / 1955 ~ 0.0153.
        """
        baseline_inserts = {
            "bone": 955.0,
            "air": -1000.0,
            "acrylic": 120.0,
            "polyethylene": -95.0,
        }
        current_inserts = {k: v + 30.0 for k, v in baseline_inserts.items()}

        drift = compute_hu_drift_index(
            current_inserts=current_inserts,
            baseline_inserts=baseline_inserts,
        )
        # 30 HU / 1955 HU range ~ 0.0153
        assert drift > 0.01, (
            f"HU drift index {drift:.6f} should be > 0.01 for +30 HU shift"
        )
        assert drift < 0.05, (
            f"HU drift index {drift:.6f} should be < 0.05 for moderate shift"
        )
