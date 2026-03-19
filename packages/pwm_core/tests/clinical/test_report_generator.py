"""Unit tests for the ReportGenerator and its JSON / evidence outputs.

Covers JSON report structure, overall PASS/FAIL decision logic,
SHA-256 integrity hash, evidence directory scaffolding, metric logs,
ROI overlay manifests, trend plot manifests, and optional fields
(diagnosis, baseline_ref, drift_alerts).

PDF generation is intentionally skipped because it requires fpdf2
and is best-effort.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pwm_core.clinical.ct.report_generator import ReportGenerator, ReportConfig


# ---------------------------------------------------------------------------
# Helpers: minimal input data factories
# ---------------------------------------------------------------------------

def _make_config(output_dir: Path) -> ReportConfig:
    """Return a minimal ReportConfig pointing at *output_dir*."""
    return ReportConfig(
        output_dir=output_dir,
        scanner_id="CT-TEST-001",
        date="2026-02-19",
        casepack_id="acr_monthly_2026-02",
        physicist_name="Dr. Test",
        technologist_name="Tech. A",
    )


def _make_metrics_report() -> dict:
    """Return a simple metrics_report dict with two metrics."""
    return {
        "ct_number_water": {
            "name": "CT Number Accuracy (Water)",
            "value": 0.5,
            "unit": "HU",
        },
        "uniformity": {
            "name": "Uniformity",
            "value": 2.1,
            "unit": "HU",
        },
    }


def _make_threshold_results_all_pass() -> dict:
    """Return threshold results where every metric passes."""
    return {
        "ct_number_water": {
            "status": "PASS",
            "standard_threshold": {"low": -5.0, "high": 5.0, "unit": "HU"},
            "applied_threshold": {"low": -5.0, "high": 5.0, "unit": "HU"},
            "threshold_layer": "standard",
        },
        "uniformity": {
            "status": "PASS",
            "standard_threshold": {"low": 0.0, "high": 5.0, "unit": "HU"},
            "applied_threshold": {"low": 0.0, "high": 5.0, "unit": "HU"},
            "threshold_layer": "standard",
        },
    }


def _make_threshold_results_one_fail() -> dict:
    """Return threshold results where one metric fails."""
    results = _make_threshold_results_all_pass()
    results["uniformity"]["status"] = "FAIL"
    return results


def _make_threshold_results_warning_only() -> dict:
    """Return threshold results with one WARNING and one PASS."""
    results = _make_threshold_results_all_pass()
    results["uniformity"]["status"] = "WARNING"
    return results


def _load_json_report(output_dir: Path) -> dict:
    """Load and return the physicist_report.json from *output_dir*."""
    json_path = output_dir / "physicist_report.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJSONReportWritten:
    """1. The generator must produce a valid JSON file at the expected path."""

    def test_json_report_written(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_all_pass())

        json_path = tmp_path / "physicist_report.json"
        assert json_path.exists(), "physicist_report.json was not created"

        # Must be parseable JSON
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

        # Check top-level required keys
        for key in (
            "version", "scanner_id", "date", "casepack",
            "overall_decision", "metrics", "sha256",
        ):
            assert key in data, f"Missing top-level key: {key}"


class TestOverallPassAllMetricsPass:
    """2. overall_decision must be PASS when every metric passes."""

    def test_overall_pass_all_metrics_pass(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_all_pass())

        data = _load_json_report(tmp_path)
        assert data["overall_decision"] == "PASS"


class TestOverallFailAnyMetricFails:
    """3. overall_decision must be FAIL when any single metric fails."""

    def test_overall_fail_any_metric_fails(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_one_fail())

        data = _load_json_report(tmp_path)
        assert data["overall_decision"] == "FAIL"


class TestWarningTreatedAsPass:
    """4. WARNING status must not block an overall PASS decision."""

    def test_warning_treated_as_pass(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_warning_only())

        data = _load_json_report(tmp_path)
        assert data["overall_decision"] == "PASS", (
            "WARNING should be treated as PASS for the aggregate decision"
        )


class TestSHA256Computed:
    """5. The sha256 field must be present, non-empty, and hex-encoded."""

    def test_sha256_computed_and_nonempty(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_all_pass())

        data = _load_json_report(tmp_path)
        sha = data["sha256"]
        assert isinstance(sha, str)
        assert len(sha) == 64, "SHA-256 hex digest should be 64 characters"
        # Verify it is valid hexadecimal
        int(sha, 16)


class TestEvidenceDirectoryCreated:
    """6. The evidence/ folder and its three sub-dirs must be created."""

    def test_evidence_directory_created(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_all_pass())

        evidence_dir = tmp_path / "evidence"
        assert evidence_dir.is_dir(), "evidence/ directory was not created"

        for subdir in ("roi_overlays", "trend_plots", "metric_logs"):
            assert (evidence_dir / subdir).is_dir(), (
                f"evidence/{subdir}/ directory was not created"
            )


class TestMetricLogsWrittenPerMetric:
    """7. A JSON log file must exist in metric_logs/ for each metric."""

    def test_metric_logs_written_per_metric(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        metrics_report = _make_metrics_report()
        gen = ReportGenerator()
        gen.generate(config, metrics_report, _make_threshold_results_all_pass())

        log_dir = tmp_path / "evidence" / "metric_logs"
        assert log_dir.is_dir()

        log_files = list(log_dir.glob("*.json"))
        # There should be one log per metric key in the metrics_report
        assert len(log_files) == len(metrics_report), (
            f"Expected {len(metrics_report)} metric log files, "
            f"found {len(log_files)}"
        )

        # Each log file must be valid JSON with a metric_key field
        for log_file in log_files:
            entry = json.loads(log_file.read_text(encoding="utf-8"))
            assert "metric_key" in entry
            assert "scanner_id" in entry
            assert entry["scanner_id"] == "CT-TEST-001"


class TestROIOverlayManifestCreated:
    """8. roi_overlays/manifest.json must exist with expected structure."""

    def test_roi_overlay_manifest_created(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_all_pass())

        manifest_path = tmp_path / "evidence" / "roi_overlays" / "manifest.json"
        assert manifest_path.exists(), "roi_overlays/manifest.json was not created"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["scanner_id"] == "CT-TEST-001"
        assert manifest["date"] == "2026-02-19"
        assert "expected_overlays" in manifest
        assert isinstance(manifest["expected_overlays"], list)


class TestTrendPlotManifestCreated:
    """9. trend_plots/manifest.json must exist with expected structure."""

    def test_trend_plot_manifest_created(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()
        gen.generate(config, _make_metrics_report(), _make_threshold_results_all_pass())

        manifest_path = tmp_path / "evidence" / "trend_plots" / "manifest.json"
        assert manifest_path.exists(), "trend_plots/manifest.json was not created"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["scanner_id"] == "CT-TEST-001"
        assert manifest["date"] == "2026-02-19"
        assert "generated_plots" in manifest
        assert isinstance(manifest["generated_plots"], list)


class TestDiagnosisIncludedInJSON:
    """10. When a diagnosis_report is provided it must appear in the JSON."""

    def test_diagnosis_included_in_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()

        diagnosis_report = {
            "root_cause": "Tube current drift",
            "suggested_actions": [
                {"description": "Recalibrate tube current"},
                {"description": "Repeat QC in 24h"},
            ],
        }

        gen.generate(
            config,
            _make_metrics_report(),
            _make_threshold_results_one_fail(),
            diagnosis_report=diagnosis_report,
        )

        data = _load_json_report(tmp_path)
        assert data["diagnosis"] is not None
        assert data["diagnosis"]["root_cause"] == "Tube current drift"
        assert len(data["diagnosis"]["suggested_actions"]) == 2


class TestBaselineRefIncludedInJSON:
    """11. When baseline_comparison is provided, baseline_ref must appear."""

    def test_baseline_ref_included_in_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()

        baseline_comparison = {
            "baseline_id": "baseline-2026-01-15-CT-TEST-001",
            "ct_number_water": {"baseline_value": 0.3, "value": 0.5},
            "uniformity": {"baseline_value": 2.0, "value": 2.1},
        }

        gen.generate(
            config,
            _make_metrics_report(),
            _make_threshold_results_all_pass(),
            baseline_comparison=baseline_comparison,
        )

        data = _load_json_report(tmp_path)
        assert data["baseline_ref"] == "baseline-2026-01-15-CT-TEST-001"


class TestDriftAlertsIncludedInJSON:
    """12. When drift_report has alerts, they must appear in the JSON."""

    def test_drift_alerts_included_in_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        gen = ReportGenerator()

        drift_report = {
            "alerts": [
                {
                    "metric": "ct_number_water",
                    "type": "trend_shift",
                    "message": "3-sigma shift detected over last 6 months",
                },
            ],
            "trends": {},
        }

        gen.generate(
            config,
            _make_metrics_report(),
            _make_threshold_results_all_pass(),
            drift_report=drift_report,
        )

        data = _load_json_report(tmp_path)
        assert isinstance(data["drift_alerts"], list)
        assert len(data["drift_alerts"]) == 1
        assert data["drift_alerts"][0]["metric"] == "ct_number_water"
        assert data["drift_alerts"][0]["type"] == "trend_shift"
