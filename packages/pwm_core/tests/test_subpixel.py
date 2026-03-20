"""Tests for sub-pixel shift utilities and mismatch agent physics classification."""

import os
import sys

import numpy as np
import pytest

# Ensure package is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pwm_core.mismatch.subpixel import (
    subpixel_shift_2d,
    subpixel_shift_3d_spatial,
    subpixel_warp_2d,
)


# ── Sub-pixel shift tests ────────────────────────────────────────────────


class TestSubpixelShift2D:
    """Tests for subpixel_shift_2d."""

    def test_half_pixel_shift_nonzero(self):
        """0.5 px shift must produce a measurable change (the original bug)."""
        rng = np.random.default_rng(42)
        arr = rng.random((64, 64)).astype(np.float32)
        shifted = subpixel_shift_2d(arr, 0.5, 0.0)
        diff = np.abs(shifted - arr).max()
        assert diff > 0.01, f"0.5 px shift produced negligible diff={diff}"

    def test_zero_shift_identity(self):
        """Zero shift must be exact identity (copy)."""
        arr = np.arange(25, dtype=np.float64).reshape(5, 5)
        out = subpixel_shift_2d(arr, 0.0, 0.0)
        np.testing.assert_array_equal(out, arr)
        assert out is not arr, "Should be a copy, not the same object"

    def test_integer_shift_matches_interior(self):
        """Integer-pixel shift must match np.roll in the interior region."""
        rng = np.random.default_rng(123)
        arr = rng.random((32, 32)).astype(np.float64)
        dx, dy = 3, -2
        shifted = subpixel_shift_2d(arr, float(dx), float(dy))
        rolled = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
        # Interior: avoid the boundary (5 px margin)
        m = 5
        interior_shifted = shifted[m:-m, m:-m]
        interior_rolled = rolled[m:-m, m:-m]
        np.testing.assert_allclose(
            interior_shifted, interior_rolled, atol=1e-10,
            err_msg="Integer shift should match np.roll in interior"
        )

    def test_energy_nonincreasing(self):
        """mode='constant' can only lose (or preserve) boundary energy."""
        rng = np.random.default_rng(7)
        arr = rng.random((32, 32)).astype(np.float64)
        original_energy = np.sum(arr ** 2)
        shifted = subpixel_shift_2d(arr, 2.5, -1.3)
        shifted_energy = np.sum(shifted ** 2)
        assert shifted_energy <= original_energy * 1.01, (
            f"Energy increased: {shifted_energy:.4f} > {original_energy:.4f}"
        )


class TestSubpixelShift3D:
    """Tests for subpixel_shift_3d_spatial."""

    def test_3d_shift_all_frames(self):
        """Each frame of (H,W,T) should be shifted independently."""
        rng = np.random.default_rng(99)
        arr = rng.random((16, 16, 4)).astype(np.float32)
        shifted = subpixel_shift_3d_spatial(arr, 1.5, 0.5)
        for t in range(4):
            expected = subpixel_shift_2d(arr[:, :, t], 1.5, 0.5)
            np.testing.assert_allclose(shifted[:, :, t], expected, atol=1e-12)


class TestSubpixelWarp2D:
    """Tests for subpixel_warp_2d."""

    def test_pure_shift_changes_array(self):
        """Warp with theta=0 should produce a measurable shift."""
        rng = np.random.default_rng(55)
        arr = rng.random((32, 32)).astype(np.float64)
        warped = subpixel_warp_2d(arr, 1.5, -0.7, theta_deg=0.0)
        diff = np.abs(warped - arr).max()
        assert diff > 0.01, f"Warp should change the array, diff={diff}"

    def test_rotation_changes_array(self):
        """Non-zero rotation must change the array."""
        rng = np.random.default_rng(22)
        arr = rng.random((32, 32)).astype(np.float64)
        warped = subpixel_warp_2d(arr, 0.0, 0.0, theta_deg=5.0)
        diff = np.abs(warped - arr).max()
        assert diff > 0.01, f"5-degree rotation produced negligible diff={diff}"

    def test_zero_warp_identity(self):
        """Zero shift + zero rotation = identity."""
        arr = np.eye(8, dtype=np.float64)
        out = subpixel_warp_2d(arr, 0.0, 0.0, theta_deg=0.0)
        np.testing.assert_array_equal(out, arr)


# ── CASSI mask shift regression ──────────────────────────────────────────


class TestCASSIMaskShiftMild:
    """Regression: mild severity (0.5 px) must actually change the mask."""

    def test_cassi_mask_shift_mild_nonzero(self):
        """apply_cassi_mask_shift with dx=0.5 must differ from identity."""
        from experiments.inversenet.mismatch_sweep import apply_cassi_mask_shift
        rng = np.random.default_rng(42)
        mask = (rng.random((64, 64)) > 0.5).astype(np.float32)
        delta = {"mask_dx": 0.5, "mask_dy": 0.5}
        shifted = apply_cassi_mask_shift(mask, delta, rng)
        diff = np.abs(shifted - mask).sum()
        assert diff > 0, "Mild severity mask_shift should change the mask"


# ── MismatchAgent classification tests ───────────────────────────────────


class TestClassifyParamPhysics:
    """Tests for MismatchAgent.classify_param_physics."""

    def test_explicit_param_type_returned(self):
        """If param_spec has param_type set, return it directly."""
        from pwm_core.agents.mismatch_agent import MismatchAgent

        class FakeSpec:
            param_type = "spatial_shift"
            unit = "pixels"

        result = MismatchAgent.classify_param_physics("mask_dx", FakeSpec())
        assert result == "spatial_shift"

    def test_infer_from_name_dx(self):
        """Infer spatial_shift from name containing _dx and unit=pixels."""
        from pwm_core.agents.mismatch_agent import MismatchAgent

        class FakeSpec:
            param_type = None
            unit = "pixels"

        result = MismatchAgent.classify_param_physics("mask_dx", FakeSpec())
        assert result == "spatial_shift"

    def test_infer_rotation(self):
        """Infer rotation from name containing 'rotation' and unit=degrees."""
        from pwm_core.agents.mismatch_agent import MismatchAgent

        class FakeSpec:
            param_type = None
            unit = "degrees"

        result = MismatchAgent.classify_param_physics("mask_rotation", FakeSpec())
        assert result == "rotation"

    def test_infer_scale(self):
        """Infer scale from name containing 'gain'."""
        from pwm_core.agents.mismatch_agent import MismatchAgent

        class FakeSpec:
            param_type = None
            unit = "dimensionless"

        result = MismatchAgent.classify_param_physics("gain_error", FakeSpec())
        assert result == "scale"

    def test_fallback_to_scale(self):
        """Unknown names default to 'scale'."""
        from pwm_core.agents.mismatch_agent import MismatchAgent

        class FakeSpec:
            param_type = None
            unit = "unknown"

        result = MismatchAgent.classify_param_physics("mystery_param", FakeSpec())
        assert result == "scale"


class TestValidateMismatchEffect:
    """Tests for MismatchAgent.validate_mismatch_effect."""

    def test_warns_subthreshold_spatial(self):
        """Sub-threshold spatial shift (< 0.5 px) should warn."""
        from pwm_core.agents.mismatch_agent import MismatchAgent
        result = MismatchAgent.validate_mismatch_effect(
            "mask_dx", "spatial_shift", 0.3
        )
        assert not result["effective"]
        assert result["warning"] is not None
        assert "np.roll" in result["warning"]

    def test_adequate_spatial_shift(self):
        """Above-threshold spatial shift should be effective."""
        from pwm_core.agents.mismatch_agent import MismatchAgent
        result = MismatchAgent.validate_mismatch_effect(
            "mask_dx", "spatial_shift", 1.0
        )
        assert result["effective"]
        assert result["warning"] is None

    def test_warns_subthreshold_rotation(self):
        """Sub-threshold rotation (< 0.01 deg) should warn."""
        from pwm_core.agents.mismatch_agent import MismatchAgent
        result = MismatchAgent.validate_mismatch_effect(
            "tilt", "rotation", 0.005
        )
        assert not result["effective"]

    def test_scale_always_effective(self):
        """Scale parameters are always considered effective."""
        from pwm_core.agents.mismatch_agent import MismatchAgent
        result = MismatchAgent.validate_mismatch_effect(
            "gain", "scale", 0.001
        )
        assert result["effective"]


class TestMismatchReportParamType:
    """MismatchReport should include param_types field."""

    def test_report_includes_param_type(self):
        from pwm_core.agents.contracts import MismatchReport
        report = MismatchReport(
            modality_key="cassi",
            mismatch_family="correction_v1",
            parameters={"mask_dx": {"typical_error": 0.5, "range": [-3, 3], "unit": "pixels"}},
            severity_score=0.3,
            correction_method="correction_v1",
            expected_improvement_db=3.0,
            explanation="Test",
            param_types={"mask_dx": "spatial_shift"},
            subpixel_warnings=["mask_dx warning"],
        )
        assert report.param_types == {"mask_dx": "spatial_shift"}
        assert len(report.subpixel_warnings) == 1

    def test_report_optional_fields_none(self):
        """param_types and subpixel_warnings can be None."""
        from pwm_core.agents.contracts import MismatchReport
        report = MismatchReport(
            modality_key="spc",
            mismatch_family="correction_v1",
            parameters={"gain": {"typical_error": 0.05, "range": [0.8, 1.2], "unit": "ratio"}},
            severity_score=0.1,
            correction_method="correction_v1",
            expected_improvement_db=1.0,
            explanation="Test",
            param_types=None,
            subpixel_warnings=None,
        )
        assert report.param_types is None
        assert report.subpixel_warnings is None
