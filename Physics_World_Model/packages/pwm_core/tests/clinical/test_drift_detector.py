"""Unit tests for DriftDetector statistical process control.

Tests cover measurement persistence, control chart construction,
all five Western Electric rules, overall status aggregation, and
edge cases such as insufficient data and multiple independent metrics.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pwm_core.clinical.ct.drift_detector import (
    ControlChart,
    DriftAlert,
    DriftDetector,
    DriftReport,
)


# ===================================================================
# Helpers
# ===================================================================

def _make_detector(tmp_path: Path) -> DriftDetector:
    """Create a DriftDetector rooted at a temporary directory."""
    return DriftDetector(history_path=tmp_path / "drift_history")


def _add_stable_measurements(
    detector: DriftDetector,
    scanner_id: str = "CT-001",
    metric: str = "noise_std",
    values: list[float] | None = None,
    *,
    start_day: int = 1,
) -> None:
    """Add a sequence of measurements for a single metric.

    By default adds 10 tightly clustered values that should not trigger
    any Western Electric rules.
    """
    if values is None:
        values = [5.0, 5.1, 4.9, 5.05, 4.95, 5.02, 4.98, 5.03, 4.97, 5.01]

    for i, v in enumerate(values):
        date = f"2026-01-{start_day + i:02d}"
        detector.add_measurement(scanner_id, date, {metric: v})


def _bessel_mean(values: list[float]) -> float:
    """Reference arithmetic mean."""
    return sum(values) / len(values)


def _bessel_std(values: list[float]) -> float:
    """Reference sample standard deviation with Bessel's correction."""
    n = len(values)
    if n < 2:
        return 0.0
    m = _bessel_mean(values)
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


# ===================================================================
# Tests
# ===================================================================


class TestEmptyHistory:
    """No measurements at all should yield a STABLE report with no alerts."""

    def test_empty_history_returns_stable(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        report = detector.detect_drift("CT-UNKNOWN")

        assert isinstance(report, DriftReport)
        assert report.scanner_id == "CT-UNKNOWN"
        assert report.alerts == []
        assert report.control_charts == {}
        assert report.overall_status == "STABLE"


class TestAddAndRetrieveMeasurement:
    """Measurements should round-trip through JSON persistence."""

    def test_add_and_retrieve_measurement(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector.add_measurement("CT-001", "2026-01-15", {"noise_std": 5.2, "uniformity": 2.1})

        history = detector.get_history("CT-001")
        assert len(history) == 1
        assert history[0]["date"] == "2026-01-15"
        assert history[0]["metrics"]["noise_std"] == 5.2
        assert history[0]["metrics"]["uniformity"] == 2.1

    def test_multiple_measurements_accumulate(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        for i in range(5):
            detector.add_measurement("CT-001", f"2026-01-{i+1:02d}", {"noise_std": 5.0 + i * 0.1})

        history = detector.get_history("CT-001")
        assert len(history) == 5


class TestInsufficientDataNoRulesApplied:
    """With fewer than 5 measurements, control charts can be built but
    Western Electric rules must NOT fire."""

    def test_insufficient_data_no_rules_applied(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # Add only 4 measurements -- enough for a control chart (>= 2)
        # but not enough for WE rules (need >= 5).
        # Use values that would trigger Rule 1 if rules were applied:
        # three normal points + one extreme outlier.
        _add_stable_measurements(
            detector, values=[5.0, 5.0, 5.0, 100.0],
        )

        report = detector.detect_drift("CT-001")
        assert report.overall_status == "STABLE"
        assert report.alerts == []
        # Control chart should still exist
        assert "noise_std" in report.control_charts

    def test_exactly_5_identical_values_no_rules_fire(self, tmp_path: Path) -> None:
        """At exactly 5 identical measurements, std = 0, so sigma-based
        rules return early and no alerts fire."""
        detector = _make_detector(tmp_path)
        _add_stable_measurements(
            detector, values=[5.0, 5.0, 5.0, 5.0, 5.0],
        )

        report = detector.detect_drift("CT-001")
        # std is 0, so _apply_western_electric returns early
        assert report.overall_status == "STABLE"
        assert report.alerts == []


class TestControlChartComputedCorrectly:
    """Verify mean, std, UCL, LCL match manual calculations."""

    def test_control_chart_computed_correctly(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        values = [10.0, 12.0, 11.0, 13.0, 9.0]
        _add_stable_measurements(detector, values=values)

        chart = detector.get_control_chart("CT-001", "noise_std")

        expected_mean = _bessel_mean(values)
        expected_std = _bessel_std(values)

        assert isinstance(chart, ControlChart)
        assert chart.metric_name == "noise_std"
        assert len(chart.values) == 5
        assert len(chart.dates) == 5

        assert abs(chart.mean - expected_mean) < 1e-9
        assert abs(chart.std - expected_std) < 1e-9
        assert abs(chart.ucl - (expected_mean + 3.0 * expected_std)) < 1e-9
        assert abs(chart.lcl - (expected_mean - 3.0 * expected_std)) < 1e-9
        assert abs(chart.ucl_2sigma - (expected_mean + 2.0 * expected_std)) < 1e-9
        assert abs(chart.lcl_2sigma - (expected_mean - 2.0 * expected_std)) < 1e-9


class TestRule1ThreeSigmaFail:
    """Rule 1: A single point outside 3-sigma should produce a FAIL alert."""

    def test_rule1_3sigma_fail(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # Use 20 stable values to establish a tight mean/std, then add
        # one extreme outlier.  With 20 points near 10.0 and one at 50.0,
        # the outlier still exceeds the UCL of the full series.
        base_values = [10.0, 10.1, 9.9, 10.05, 9.95] * 4  # 20 points
        outlier = 50.0  # far enough to exceed UCL even in full series

        all_values = base_values + [outlier]
        _add_stable_measurements(detector, values=all_values)

        report = detector.detect_drift("CT-001")

        rule1_alerts = [a for a in report.alerts if a.rule_triggered == "WE_Rule_1"]
        assert len(rule1_alerts) >= 1
        assert rule1_alerts[0].alert_level == "FAIL"
        assert report.overall_status == "ACTION_REQUIRED"

    def test_rule1_below_lcl(self, tmp_path: Path) -> None:
        """Rule 1 should also fire when the point is below LCL."""
        detector = _make_detector(tmp_path)

        base_values = [10.0, 10.1, 9.9, 10.05, 9.95] * 4  # 20 points
        outlier = -30.0  # far below LCL

        all_values = base_values + [outlier]
        _add_stable_measurements(detector, values=all_values)

        report = detector.detect_drift("CT-001")
        rule1_alerts = [a for a in report.alerts if a.rule_triggered == "WE_Rule_1"]
        assert len(rule1_alerts) >= 1
        assert rule1_alerts[0].alert_level == "FAIL"


class TestRule2TwoOfThreeActionRequired:
    """Rule 2: 2 of 3 consecutive points outside 2-sigma (same side)
    should produce an ACTION_REQUIRED alert."""

    def test_rule2_2of3_action_required(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # Use 20 stable values to establish tight stats, then add 3 values
        # where 2 are above 2-sigma of the full series.
        base_values = [10.0, 10.1, 9.9, 10.05, 9.95] * 4  # 20 points
        # With 20 points near 10.0, the full-series mean stays near 10
        # even after adding a few high values.  We use 30.0 for the highs
        # which will be well above 2-sigma.
        tail = [30.0, 10.0, 30.0]  # 2 of 3 above 2-sigma
        all_values = base_values + tail
        _add_stable_measurements(detector, values=all_values)

        report = detector.detect_drift("CT-001")
        rule2_alerts = [a for a in report.alerts if a.rule_triggered == "WE_Rule_2"]
        assert len(rule2_alerts) >= 1
        assert rule2_alerts[0].alert_level == "ACTION_REQUIRED"


class TestRule3FourOfFiveWarning:
    """Rule 3: 4 of 5 consecutive points outside 1-sigma (same side)
    should produce a WARNING alert."""

    def test_rule3_4of5_warning(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # Use 20 stable values to establish tight stats, then add 5 values
        # where 4 are above 1-sigma of the full series.
        base_values = [10.0, 10.1, 9.9, 10.05, 9.95] * 4  # 20 points
        # 4 moderately high values + 1 at center.  With tight baseline,
        # 15.0 will be well above 1-sigma.
        tail = [15.0, 15.0, 10.0, 15.0, 15.0]
        all_values = base_values + tail
        _add_stable_measurements(detector, values=all_values)

        report = detector.detect_drift("CT-001")
        rule3_alerts = [a for a in report.alerts if a.rule_triggered == "WE_Rule_3"]
        assert len(rule3_alerts) >= 1
        assert rule3_alerts[0].alert_level == "WARNING"


class TestRule4RunOf7Warning:
    """Rule 4: 7 consecutive points on the same side of the mean
    should produce a WARNING alert."""

    def test_rule4_run_of_7_warning(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # We need 7 points all above the mean at the end of the series.
        # Strategy: start with points below and at mean to pull the mean
        # down, then add 7 points slightly above it.
        #
        # We use a large enough series so that the last 7 values are all
        # strictly above the overall mean.
        low_part = [8.0, 8.0, 8.0]
        high_part = [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]
        all_values = low_part + high_part  # 10 total
        # mean = (3*8 + 7*12) / 10 = (24 + 84)/10 = 10.8
        # All of high_part (12.0) are above 10.8 => Rule 4 fires

        _add_stable_measurements(detector, values=all_values)

        report = detector.detect_drift("CT-001")
        rule4_alerts = [a for a in report.alerts if a.rule_triggered == "WE_Rule_4"]
        assert len(rule4_alerts) >= 1
        assert rule4_alerts[0].alert_level == "WARNING"


class TestRule5TrendOf7Warning:
    """Rule 5: 7 consecutive points all increasing or all decreasing
    should produce a WARNING alert."""

    def test_rule5_trend_of_7_increasing(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # 7 strictly increasing values at the end
        values = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
        _add_stable_measurements(detector, values=values)

        report = detector.detect_drift("CT-001")
        rule5_alerts = [a for a in report.alerts if a.rule_triggered == "WE_Rule_5"]
        assert len(rule5_alerts) >= 1
        assert rule5_alerts[0].alert_level == "WARNING"

    def test_rule5_trend_of_7_decreasing(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        values = [10.6, 10.5, 10.4, 10.3, 10.2, 10.1, 10.0]
        _add_stable_measurements(detector, values=values)

        report = detector.detect_drift("CT-001")
        rule5_alerts = [a for a in report.alerts if a.rule_triggered == "WE_Rule_5"]
        assert len(rule5_alerts) >= 1
        assert rule5_alerts[0].alert_level == "WARNING"


class TestOverallStatusPrecedence:
    """ACTION_REQUIRED (from FAIL or ACTION_REQUIRED alerts) should
    override WATCH (from WARNING alerts)."""

    def test_overall_status_action_required_overrides_watch(
        self, tmp_path: Path,
    ) -> None:
        detector = _make_detector(tmp_path)

        # Use 20 stable baseline values, then add 7 strictly increasing
        # values ending with a massive outlier.  This triggers BOTH:
        #   Rule 5 (trend of 7 increasing) -> WARNING
        #   Rule 1 (last point outside 3-sigma) -> FAIL
        base_values = [10.0, 10.1, 9.9, 10.05, 9.95] * 4  # 20 points
        # 7 increasing values ending at 50 (well beyond 3-sigma)
        trend = [11.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
        all_values = base_values + trend
        _add_stable_measurements(detector, values=all_values)

        report = detector.detect_drift("CT-001")
        alert_levels = {a.alert_level for a in report.alerts}

        # Should have both WARNING-level and FAIL/ACTION_REQUIRED-level alerts
        assert "FAIL" in alert_levels or "ACTION_REQUIRED" in alert_levels
        assert report.overall_status == "ACTION_REQUIRED"


class TestStableProcessNoAlerts:
    """A well-behaved process should produce zero alerts and STABLE status."""

    def test_stable_process_no_alerts(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # 10 measurements randomly scattered near the mean -- no patterns
        values = [10.0, 9.8, 10.2, 9.9, 10.1, 9.85, 10.15, 9.95, 10.05, 10.0]
        _add_stable_measurements(detector, values=values)

        report = detector.detect_drift("CT-001")

        # Filter out INFO-level alerts (those do not affect STABLE status)
        non_info_alerts = [a for a in report.alerts if a.alert_level != "INFO"]
        assert non_info_alerts == []
        assert report.overall_status == "STABLE"


class TestGetControlChartInsufficientData:
    """get_control_chart should raise ValueError when fewer than 2 data
    points are available."""

    def test_get_control_chart_insufficient_data_raises(
        self, tmp_path: Path,
    ) -> None:
        detector = _make_detector(tmp_path)

        # No measurements at all
        with pytest.raises(ValueError, match="Insufficient data"):
            detector.get_control_chart("CT-001", "noise_std")

    def test_get_control_chart_one_point_raises(
        self, tmp_path: Path,
    ) -> None:
        detector = _make_detector(tmp_path)
        detector.add_measurement("CT-001", "2026-01-01", {"noise_std": 5.0})

        with pytest.raises(ValueError, match="Insufficient data"):
            detector.get_control_chart("CT-001", "noise_std")

    def test_get_control_chart_two_points_succeeds(
        self, tmp_path: Path,
    ) -> None:
        detector = _make_detector(tmp_path)
        detector.add_measurement("CT-001", "2026-01-01", {"noise_std": 5.0})
        detector.add_measurement("CT-001", "2026-01-02", {"noise_std": 6.0})

        chart = detector.get_control_chart("CT-001", "noise_std")
        assert isinstance(chart, ControlChart)
        assert len(chart.values) == 2


class TestMultipleMetricsIndependentlyTracked:
    """Different metrics should be tracked independently; an alert on one
    metric should not affect the control chart of another."""

    def test_multiple_metrics_independently_tracked(
        self, tmp_path: Path,
    ) -> None:
        detector = _make_detector(tmp_path)

        # Add measurements with two metrics: noise_std stays stable,
        # uniformity drifts badly.
        for i in range(7):
            date = f"2026-01-{i+1:02d}"
            detector.add_measurement("CT-001", date, {
                "noise_std": 5.0 + (i % 2) * 0.01,  # rock-stable, alternating
                "uniformity": 2.0 + i * 0.5,          # monotonically increasing trend
            })

        report = detector.detect_drift("CT-001")

        # uniformity should trigger Rule 5 (7 increasing)
        uniformity_alerts = [
            a for a in report.alerts if a.metric_name == "uniformity"
        ]
        assert len(uniformity_alerts) >= 1
        rule5_uniformity = [
            a for a in uniformity_alerts if a.rule_triggered == "WE_Rule_5"
        ]
        assert len(rule5_uniformity) >= 1

        # noise_std should NOT have a trend alert
        noise_rule5 = [
            a for a in report.alerts
            if a.metric_name == "noise_std" and a.rule_triggered == "WE_Rule_5"
        ]
        assert noise_rule5 == []

        # Both metrics should have independent control charts
        assert "noise_std" in report.control_charts
        assert "uniformity" in report.control_charts
        assert (
            report.control_charts["noise_std"].mean
            != report.control_charts["uniformity"].mean
        )


class TestDriftReportFields:
    """DriftReport should carry correct scanner_id, a valid date string,
    and properly typed fields."""

    def test_drift_report_fields(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        _add_stable_measurements(detector, values=[5.0, 5.1, 4.9, 5.05, 4.95])

        report = detector.detect_drift("CT-001")

        assert report.scanner_id == "CT-001"
        assert isinstance(report.date, str)
        assert len(report.date) > 0  # ISO timestamp
        assert isinstance(report.alerts, list)
        assert isinstance(report.control_charts, dict)
        assert report.overall_status in {"STABLE", "WATCH", "ACTION_REQUIRED"}


class TestDriftAlertFields:
    """When an alert fires, its fields should be properly populated."""

    def test_drift_alert_fields(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # Trigger a Rule-1 FAIL using 20 stable baseline + extreme outlier
        base_values = [10.0, 10.1, 9.9, 10.05, 9.95] * 4  # 20 points
        outlier = 50.0  # well beyond 3-sigma
        _add_stable_measurements(detector, values=base_values + [outlier])

        report = detector.detect_drift("CT-001")
        assert len(report.alerts) >= 1

        alert = report.alerts[0]
        assert isinstance(alert, DriftAlert)
        assert alert.metric_name == "noise_std"
        assert alert.alert_level in {"INFO", "WARNING", "ACTION_REQUIRED", "FAIL"}
        assert isinstance(alert.rule_triggered, str)
        assert len(alert.description) > 0
        assert isinstance(alert.current_value, float)
        assert "mean" in alert.control_limits
        assert "std" in alert.control_limits
        assert "ucl" in alert.control_limits
        assert "lcl" in alert.control_limits


class TestPersistenceAcrossInstances:
    """Data written by one DriftDetector instance should be readable by
    a new instance pointed at the same directory."""

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        history_dir = tmp_path / "shared_history"

        detector1 = DriftDetector(history_path=history_dir)
        detector1.add_measurement("CT-001", "2026-01-01", {"noise_std": 5.0})
        detector1.add_measurement("CT-001", "2026-01-02", {"noise_std": 5.1})

        # Create a fresh instance pointing at the same directory
        detector2 = DriftDetector(history_path=history_dir)
        history = detector2.get_history("CT-001")

        assert len(history) == 2
        assert history[0]["metrics"]["noise_std"] == 5.0
        assert history[1]["metrics"]["noise_std"] == 5.1


class TestZeroVarianceNoAlerts:
    """When all values are identical (std = 0), sigma-based rules cannot
    fire and should produce no alerts."""

    def test_zero_variance_no_alerts(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        # 10 identical measurements
        _add_stable_measurements(detector, values=[5.0] * 10)

        report = detector.detect_drift("CT-001")
        assert report.alerts == []
        assert report.overall_status == "STABLE"
