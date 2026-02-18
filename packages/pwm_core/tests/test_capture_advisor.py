"""test_capture_advisor.py

Verify that the capture advisor returns non-empty, actionable suggestions
when the calibration uncertainty bands are wide, and stays quiet when
parameters are well-constrained.

Run:
    pytest -q packages/pwm_core/tests/test_capture_advisor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure pwm_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pwm_core.mismatch.capture_advisor import suggest_next_capture, CaptureAdvice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_wide_correction_result():
    """A correction result with wide uncertainty bands."""
    return {
        "theta_corrected": {
            "gain": 1.05,
            "bias": -0.02,
        },
        "theta_uncertainty": {
            "gain": [0.6, 1.5],   # width 0.9, threshold 0.3 => wide
            "bias": [-0.2, 0.15], # width 0.35, threshold 0.1 => wide
        },
    }


def _make_narrow_correction_result():
    """A correction result with narrow (well-constrained) uncertainty bands."""
    return {
        "theta_corrected": {
            "gain": 1.02,
            "bias": -0.01,
        },
        "theta_uncertainty": {
            "gain": [1.00, 1.04],  # width 0.04, threshold 0.3 => OK
            "bias": [-0.02, 0.00], # width 0.02, threshold 0.1 => OK
        },
    }


def _make_mixed_correction_result():
    """One parameter constrained, one underdetermined."""
    return {
        "theta_corrected": {
            "gain": 1.0,
            "bias": 0.0,
        },
        "theta_uncertainty": {
            "gain": [0.5, 2.0],    # width 1.5 => very wide
            "bias": [-0.01, 0.01], # width 0.02 => well-constrained
        },
    }


def _make_cassi_correction_result():
    """CASSI-style correction result with dispersion parameters."""
    return {
        "theta_corrected": {
            "dx0": 0.5,
            "dy0": -0.3,
            "disp_poly_x_0": 2.1,
            "disp_poly_x_1": 1.0,
        },
        "theta_uncertainty": {
            "dx0": [-2.0, 3.0],         # width 5.0 => wide
            "dy0": [-0.5, 0.0],         # width 0.5 => borderline
            "disp_poly_x_0": [-1.0, 5.0], # width 6.0 => wide
            "disp_poly_x_1": [0.8, 1.2],  # width 0.4 => OK
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCaptureAdvisor:
    """Verify capture advisor behaviour."""

    def test_wide_bands_produce_suggestions(self):
        """When uncertainty bands are wide, advice must be non-empty."""
        cr = _make_wide_correction_result()
        advice = suggest_next_capture(cr)

        assert isinstance(advice, CaptureAdvice)
        assert not advice.all_parameters_constrained
        assert advice.n_underdetermined >= 1
        assert len(advice.suggestions) >= 1

    def test_narrow_bands_no_suggestions(self):
        """When bands are narrow, advisor says all constrained."""
        cr = _make_narrow_correction_result()
        advice = suggest_next_capture(cr)

        assert isinstance(advice, CaptureAdvice)
        assert advice.all_parameters_constrained
        assert advice.n_underdetermined == 0
        assert len(advice.suggestions) == 0

    def test_mixed_identifies_underdetermined_only(self):
        """Only the wide parameter should appear in suggestions."""
        cr = _make_mixed_correction_result()
        advice = suggest_next_capture(cr)

        assert advice.n_underdetermined == 1
        params = [s["parameter"] for s in advice.suggestions]
        assert "gain" in params
        assert "bias" not in params

    def test_suggestions_are_actionable(self):
        """Each suggestion must have concrete fields."""
        cr = _make_wide_correction_result()
        advice = suggest_next_capture(cr)

        for s in advice.suggestions:
            # Required fields
            assert "parameter" in s
            assert "action" in s
            assert "geometry" in s
            assert "expected_reduction_pct" in s
            assert "ci_width" in s
            assert "current_ci" in s

            # Action must be a non-trivial string (> 10 chars)
            assert len(s["action"]) > 10, (
                f"Action for {s['parameter']} is too short: {s['action']!r}"
            )

            # Geometry must be a recognized string
            assert isinstance(s["geometry"], str)
            assert len(s["geometry"]) > 0

            # Expected reduction must be positive percentage
            assert 0 < s["expected_reduction_pct"] <= 100

    def test_summary_is_informative(self):
        """Summary must mention the underdetermined parameters."""
        cr = _make_wide_correction_result()
        advice = suggest_next_capture(cr)

        assert len(advice.summary) > 20
        # Should mention at least one parameter name
        assert "gain" in advice.summary or "bias" in advice.summary

    def test_cassi_dispersion_advice(self):
        """CASSI-style parameters get domain-specific capture advice."""
        cr = _make_cassi_correction_result()
        advice = suggest_next_capture(cr)

        assert advice.n_underdetermined >= 2
        params = [s["parameter"] for s in advice.suggestions]
        assert "dx0" in params
        assert "disp_poly_x_0" in params

        # Check that domain-specific geometry is suggested
        for s in advice.suggestions:
            if s["parameter"] == "dx0":
                assert "point_source" in s["geometry"]
            if s["parameter"] == "disp_poly_x_0":
                assert "monochromatic" in s["geometry"] or "wavelength" in s["geometry"]

    def test_suggestions_sorted_by_ci_width(self):
        """Suggestions must be sorted widest-first."""
        cr = _make_cassi_correction_result()
        advice = suggest_next_capture(cr)

        widths = [s["ci_width"] for s in advice.suggestions]
        assert widths == sorted(widths, reverse=True), (
            f"Suggestions not sorted by CI width: {widths}"
        )

    def test_pydantic_correction_result_input(self):
        """Advisor must accept a pydantic CorrectionResult object too."""
        from pwm_core.agents.contracts import CorrectionResult

        cr = CorrectionResult(
            theta_corrected={"gain": 1.05, "bias": -0.02},
            theta_uncertainty={
                "gain": [0.6, 1.5],
                "bias": [-0.2, 0.15],
            },
            improvement_db=3.5,
            n_evaluations=20,
            convergence_curve=[25.0, 26.0, 27.0],
            bootstrap_seeds=[42, 43, 44],
            resampling_indices=[[0, 1], [1, 2], [0, 2]],
        )
        advice = suggest_next_capture(cr)

        assert not advice.all_parameters_constrained
        assert advice.n_underdetermined >= 1

    def test_custom_thresholds_via_system_spec(self):
        """Custom width thresholds from system_spec should override
        defaults."""
        cr = _make_wide_correction_result()
        # Set a very high threshold for gain so it is considered OK
        system_spec = {
            "width_thresholds": {"gain": 5.0},
        }
        advice = suggest_next_capture(cr, system_spec=system_spec)

        # Only bias should be flagged now
        params = [s["parameter"] for s in advice.suggestions]
        assert "gain" not in params
        assert "bias" in params

    def test_empty_uncertainty_returns_message(self):
        """No uncertainty data should produce an informative message."""
        cr = {"theta_corrected": {"gain": 1.0}, "theta_uncertainty": {}}
        advice = suggest_next_capture(cr)

        assert advice.all_parameters_constrained
        assert "bootstrap_correction" in advice.summary.lower() or (
            "no uncertainty" in advice.summary.lower()
        )

    def test_all_constrained_summary(self):
        """When all params constrained, summary should say so."""
        cr = _make_narrow_correction_result()
        advice = suggest_next_capture(cr)

        assert "well-constrained" in advice.summary.lower() or (
            "no additional" in advice.summary.lower()
        )
