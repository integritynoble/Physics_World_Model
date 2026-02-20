"""Report generator for clinical CT QC.

Produces three outputs per QC run:
1. physicist_report.json -- structured, versioned, machine-readable
2. physicist_report.pdf  -- human-readable with sign-off block
3. evidence/ folder      -- ROI overlays, trend plots, metric derivation logs

Primary PDF renderer: fpdf2 (pure Python, no system deps).
Fallback: weasyprint (richer CSS but requires Cairo/Pango).
If neither is available, only JSON is produced and a warning is logged.

Usage
-----
>>> from pwm_core.clinical.ct.report_generator import ReportGenerator, ReportConfig
>>> config = ReportConfig(
...     output_dir=Path("./qc_output"),
...     scanner_id="CT-001",
...     date="2026-02-19",
...     casepack_id="acr_monthly_2026-02",
... )
>>> gen = ReportGenerator()
>>> output_dir = gen.generate(config, metrics_report, threshold_results)
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default template directory (sibling to this package)
_DEFAULT_TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent / "common" / "report_templates"
)

_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ReportMetricEntry(BaseModel):
    """A single metric result with threshold context and trend data.

    Attributes
    ----------
    name : str
        Human-readable metric name (e.g., ``"CT Number Accuracy (Water)"``).
    value : float
        Measured value from the QC run.
    unit : str
        Unit of measurement (e.g., ``"HU"``, ``"mm"``, ``"lp/cm"``).
    status : Literal["PASS", "WARNING", "FAIL"]
        Result of threshold evaluation.
    standard_threshold : dict
        Threshold from the compliance standard (ACR, AAPM TG-233).
    applied_threshold : dict
        Actual threshold applied after site/scanner overrides.
    threshold_layer : str
        Which threshold layer was used (``"standard"``, ``"site"``,
        ``"scanner"``).
    baseline_value : float | None
        Previous baseline value for trend comparison.
    delta : float | None
        Signed difference from baseline (value - baseline_value).
    """

    name: str
    value: float
    unit: str
    status: Literal["PASS", "WARNING", "FAIL"]
    standard_threshold: dict = Field(default_factory=dict)
    applied_threshold: dict = Field(default_factory=dict)
    threshold_layer: str = "standard"
    baseline_value: float | None = None
    delta: float | None = None


class PhysicistReportJSON(BaseModel):
    """Structured, versioned JSON output for machine consumption.

    This is the canonical record of a QC run. Other formats (PDF, HTML) are
    derived from this data. The ``sha256`` field is computed over the
    serialized content (excluding itself) for tamper detection.

    Attributes
    ----------
    version : str
        Schema version string (semver).
    scanner_id : str
        Unique scanner identifier (e.g., ``"CT-001"``).
    date : str
        ISO-8601 date of the QC run.
    casepack : str
        Identifier of the casepack/protocol that was executed.
    overall_decision : Literal["PASS", "FAIL"]
        Aggregate decision: PASS if every metric passes, FAIL otherwise.
    metrics : dict[str, ReportMetricEntry]
        Metric key -> detailed result entry.
    diagnosis : dict | None
        Diagnosis report from the troubleshoot pipeline, if any.
    baseline_ref : str | None
        Reference to the baseline snapshot used for comparison.
    drift_alerts : list[dict]
        Active drift alerts that exceed control-chart limits.
    sha256 : str
        SHA-256 hash of the JSON payload (computed at generation time).
    """

    version: str = _VERSION
    scanner_id: str
    date: str
    casepack: str
    overall_decision: Literal["PASS", "FAIL"]
    metrics: dict[str, ReportMetricEntry] = Field(default_factory=dict)
    diagnosis: dict | None = None
    baseline_ref: str | None = None
    drift_alerts: list[dict] = Field(default_factory=list)
    sha256: str = ""


class ReportConfig(BaseModel):
    """Configuration for a single report generation run.

    Attributes
    ----------
    output_dir : Path
        Directory where all report artifacts will be written. Created if
        it does not exist.
    scanner_id : str
        Scanner identifier (must match scanner registry).
    date : str
        ISO-8601 date string for this QC run.
    casepack_id : str
        Which casepack was executed.
    physicist_name : str
        Name for the sign-off block (blank if unsigned).
    technologist_name : str
        Name of the technologist who ran the scan.
    compliance_standards : list[str]
        Standards referenced in the report footer.
    """

    output_dir: Path
    scanner_id: str
    date: str
    casepack_id: str
    physicist_name: str = ""
    technologist_name: str = ""
    compliance_standards: list[str] = Field(
        default_factory=lambda: ["ACR", "AAPM_TG233"]
    )


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Triple-output report generator for CT QC runs.

    Produces a JSON report, a PDF report, and an evidence artifacts folder
    for each QC run. The JSON report is the canonical machine-readable
    record; the PDF is for human review and regulatory sign-off.

    Parameters
    ----------
    template_dir : Path | None
        Directory containing HTML/CSS templates. Defaults to the built-in
        ``report_templates`` directory.
    """

    def __init__(self, template_dir: Path | None = None) -> None:
        self.template_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        config: ReportConfig,
        metrics_report: Any,
        threshold_results: dict,
        baseline_comparison: dict | None = None,
        drift_report: Any | None = None,
        diagnosis_report: Any | None = None,
    ) -> Path:
        """Generate all report artifacts for a QC run.

        Creates ``config.output_dir`` if it does not exist, then produces:

        - ``physicist_report.json`` -- structured data
        - ``physicist_report.pdf``  -- human-readable (if fpdf2 available)
        - ``evidence/``             -- ROI overlays, trend plots, logs

        Parameters
        ----------
        config : ReportConfig
            Output and scanner configuration.
        metrics_report : Any
            Metrics result object (dict or Pydantic model with metric data).
        threshold_results : dict
            Mapping of metric_key -> threshold evaluation result dict.
        baseline_comparison : dict | None
            Baseline comparison data (metric_key -> baseline delta info).
        drift_report : Any | None
            Drift detection results (control-chart alerts, trend data).
        diagnosis_report : Any | None
            Diagnosis pipeline output for failed metrics.

        Returns
        -------
        Path
            The output directory containing all generated artifacts.
        """
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. JSON (always produced)
        json_path = self._generate_json(
            config, metrics_report, threshold_results,
            baseline_comparison, drift_report, diagnosis_report,
        )
        logger.info("JSON report written: %s", json_path)

        # 2. PDF (best-effort)
        try:
            pdf_path = self._generate_pdf(
                config, metrics_report, threshold_results,
                baseline_comparison, drift_report, diagnosis_report,
            )
            logger.info("PDF report written: %s", pdf_path)
        except Exception as exc:
            warnings.warn(
                f"PDF generation failed ({exc}). Only JSON report available.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning("PDF generation failed: %s", exc)

        # 3. Evidence folder (best-effort)
        try:
            evidence_dir = self._generate_evidence(
                config, metrics_report, baseline_comparison, drift_report,
            )
            logger.info("Evidence artifacts written: %s", evidence_dir)
        except Exception as exc:
            warnings.warn(
                f"Evidence generation failed ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning("Evidence generation failed: %s", exc)

        return output_dir

    # ------------------------------------------------------------------
    # JSON generation
    # ------------------------------------------------------------------

    def _generate_json(
        self,
        config: ReportConfig,
        metrics_report: Any,
        threshold_results: dict,
        baseline_comparison: dict | None,
        drift_report: Any | None,
        diagnosis_report: Any | None,
    ) -> Path:
        """Build and write the structured JSON report.

        Parameters
        ----------
        config : ReportConfig
            Report configuration.
        metrics_report : Any
            Raw metrics data.
        threshold_results : dict
            Threshold evaluation results per metric.
        baseline_comparison : dict | None
            Baseline deltas.
        drift_report : Any | None
            Drift detection report.
        diagnosis_report : Any | None
            Diagnosis report for failures.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        metric_entries = self._create_metric_entries(
            metrics_report, threshold_results, baseline_comparison,
        )

        overall = self._compute_overall_decision(threshold_results)

        # Extract drift alerts
        drift_alerts: list[dict] = []
        if drift_report is not None:
            if isinstance(drift_report, dict):
                drift_alerts = drift_report.get("alerts", [])
            elif hasattr(drift_report, "alerts"):
                drift_alerts = [
                    a if isinstance(a, dict) else a.model_dump()
                    for a in getattr(drift_report, "alerts", [])
                ]

        # Extract diagnosis dict
        diagnosis_dict: dict | None = None
        if diagnosis_report is not None:
            if isinstance(diagnosis_report, dict):
                diagnosis_dict = diagnosis_report
            elif hasattr(diagnosis_report, "model_dump"):
                diagnosis_dict = diagnosis_report.model_dump()
            elif hasattr(diagnosis_report, "dict"):
                diagnosis_dict = diagnosis_report.model_dump()

        # Build baseline reference string
        baseline_ref: str | None = None
        if baseline_comparison is not None:
            baseline_ref = baseline_comparison.get(
                "baseline_id",
                baseline_comparison.get("reference", None),
            )

        report = PhysicistReportJSON(
            version=_VERSION,
            scanner_id=config.scanner_id,
            date=config.date,
            casepack=config.casepack_id,
            overall_decision=overall,
            metrics=metric_entries,
            diagnosis=diagnosis_dict,
            baseline_ref=baseline_ref,
            drift_alerts=drift_alerts,
            sha256="",  # placeholder — computed below
        )

        # Compute SHA-256 over the content (excluding the hash field itself)
        payload = report.model_dump()
        payload["sha256"] = ""
        payload_bytes = json.dumps(
            payload, sort_keys=True, default=str
        ).encode("utf-8")
        report.sha256 = hashlib.sha256(payload_bytes).hexdigest()

        output_path = Path(config.output_dir) / "physicist_report.json"
        output_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return output_path

    # ------------------------------------------------------------------
    # PDF generation
    # ------------------------------------------------------------------

    def _generate_pdf(
        self,
        config: ReportConfig,
        metrics_report: Any,
        threshold_results: dict,
        baseline_comparison: dict | None,
        drift_report: Any | None,
        diagnosis_report: Any | None,
    ) -> Path:
        """Generate the human-readable PDF report.

        Uses fpdf2 as the primary renderer. Falls back to JSON-only if
        fpdf2 is not available.

        Returns
        -------
        Path
            Path to the generated PDF file.

        Raises
        ------
        ImportError
            If fpdf2 is not installed.
        """
        overall = self._compute_overall_decision(threshold_results)
        table_rows = self._create_metric_table_rows(
            metrics_report, threshold_results, baseline_comparison,
        )

        # Extract diagnosis info
        diagnosis_dict: dict | None = None
        if diagnosis_report is not None:
            if isinstance(diagnosis_report, dict):
                diagnosis_dict = diagnosis_report
            elif hasattr(diagnosis_report, "model_dump"):
                diagnosis_dict = diagnosis_report.model_dump()

        # Extract drift data
        drift_alerts: list[dict] = []
        if drift_report is not None:
            if isinstance(drift_report, dict):
                drift_alerts = drift_report.get("alerts", [])
            elif hasattr(drift_report, "alerts"):
                drift_alerts = [
                    a if isinstance(a, dict) else a.model_dump()
                    for a in getattr(drift_report, "alerts", [])
                ]

        data = {
            "config": config,
            "overall_decision": overall,
            "table_rows": table_rows,
            "diagnosis": diagnosis_dict,
            "drift_alerts": drift_alerts,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

        output_path = Path(config.output_dir) / "physicist_report.pdf"
        self._render_pdf_fpdf2(data, output_path)
        return output_path

    def _render_pdf_fpdf2(self, data: dict, output_path: Path) -> None:
        """Render the QC report PDF using fpdf2.

        Layout:
        - Header with scanner ID, date, protocol
        - Summary box with PASS/FAIL (green/red background)
        - Metrics table with threshold columns
        - Drift alerts section (if any)
        - Diagnosis section (if any failures)
        - Signature block
        - Footer with standards and version info

        Parameters
        ----------
        data : dict
            Pre-assembled report data with keys: ``config``, ``overall_decision``,
            ``table_rows``, ``diagnosis``, ``drift_alerts``, ``timestamp``.
        output_path : Path
            Destination path for the PDF file.

        Raises
        ------
        ImportError
            If fpdf2 is not installed.
        """
        try:
            from fpdf import FPDF
        except ImportError as exc:
            raise ImportError(
                "fpdf2 is required for PDF report generation. "
                "Install it with: pip install fpdf2"
            ) from exc

        config: ReportConfig = data["config"]
        overall: str = data["overall_decision"]
        table_rows: list[dict] = data["table_rows"]
        diagnosis: dict | None = data["diagnosis"]
        drift_alerts: list[dict] = data["drift_alerts"]
        timestamp: str = data["timestamp"]

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # ---- Header ----
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "CT Quality Control Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Generated: {timestamp}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # Scanner info block
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Scanner Information", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        info_lines = [
            f"Scanner ID: {config.scanner_id}",
            f"Date: {config.date}",
            f"Casepack: {config.casepack_id}",
        ]
        if config.technologist_name:
            info_lines.append(f"Technologist: {config.technologist_name}")
        for line in info_lines:
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ---- Summary box ----
        if overall == "PASS":
            r, g, b = 34, 139, 34  # forest green
        else:
            r, g, b = 200, 30, 30  # red

        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 14, f"  OVERALL: {overall}  ", new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

        # ---- Metrics table ----
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Metrics Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Column definitions: (header, width)
        columns = [
            ("Metric", 38),
            ("Value", 18),
            ("Unit", 14),
            ("Baseline", 20),
            ("Std Thresh", 26),
            ("Applied Thresh", 26),
            ("Layer", 20),
            ("Status", 18),
        ]
        total_width = sum(w for _, w in columns)

        # Table header
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(60, 60, 80)
        pdf.set_text_color(255, 255, 255)
        for header, width in columns:
            pdf.cell(width, 6, header, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        # Table rows
        pdf.set_font("Helvetica", "", 7)
        for i, row in enumerate(table_rows):
            # Alternating row background
            if i % 2 == 0:
                pdf.set_fill_color(245, 245, 250)
            else:
                pdf.set_fill_color(255, 255, 255)

            status = row.get("status", "")

            # Row cells
            cells = [
                row.get("name", ""),
                f"{row.get('value', ''):.2f}" if isinstance(row.get("value"), (int, float)) else str(row.get("value", "")),
                row.get("unit", ""),
                f"{row.get('baseline_value', ''):.2f}" if isinstance(row.get("baseline_value"), (int, float)) else str(row.get("baseline_value", "-")),
                _format_threshold(row.get("standard_threshold", {})),
                _format_threshold(row.get("applied_threshold", {})),
                row.get("threshold_layer", ""),
                status,
            ]

            for idx, ((_, width), cell_text) in enumerate(zip(columns, cells)):
                # Color the status cell
                if idx == len(columns) - 1:
                    if status == "PASS":
                        pdf.set_text_color(34, 139, 34)
                    elif status == "FAIL":
                        pdf.set_text_color(200, 30, 30)
                    elif status == "WARNING":
                        pdf.set_text_color(200, 150, 0)
                    pdf.set_font("Helvetica", "B", 7)

                pdf.cell(width, 5, str(cell_text)[:30], border=1, fill=True, align="C")

                # Reset after status cell
                if idx == len(columns) - 1:
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Helvetica", "", 7)

            pdf.ln()

        pdf.ln(4)

        # ---- Drift alerts section ----
        if drift_alerts:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Drift Alerts", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for alert in drift_alerts:
                metric_name = alert.get("metric", alert.get("metric_key", "unknown"))
                alert_type = alert.get("type", alert.get("alert_type", "drift"))
                message = alert.get("message", alert.get("description", ""))
                pdf.set_text_color(200, 100, 0)
                pdf.cell(0, 5, f"  [{alert_type}] {metric_name}: {message}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)
        else:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, "No drift alerts.", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)

        # ---- Diagnosis section ----
        if diagnosis and overall == "FAIL":
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Diagnosis", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)

            root_cause = diagnosis.get("root_cause", diagnosis.get("summary", ""))
            if root_cause:
                pdf.multi_cell(0, 5, f"Root cause: {root_cause}")

            actions = diagnosis.get("suggested_actions", diagnosis.get("actions", []))
            if actions:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, "Recommended Actions:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                for action in actions:
                    if isinstance(action, dict):
                        desc = action.get("description", action.get("action", str(action)))
                    else:
                        desc = str(action)
                    pdf.cell(0, 5, f"  - {desc}", new_x="LMARGIN", new_y="NEXT")

            pdf.ln(4)

        # ---- Signature block ----
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Sign-Off", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)

        physicist_display = config.physicist_name if config.physicist_name else "____________"
        pdf.cell(90, 6, f"Reviewed by: {physicist_display}")
        pdf.cell(0, 6, "Date: _____________", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.cell(90, 6, "Signature: __________________________")
        pdf.cell(0, 6, "Title: ______________", new_x="LMARGIN", new_y="NEXT")

        # ---- Footer ----
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        standards_str = ", ".join(config.compliance_standards)
        pdf.cell(
            0, 4,
            f"Generated by PWM Clinical QC Copilot v{_VERSION} | "
            f"Standards: {standards_str}",
            new_x="LMARGIN", new_y="NEXT",
            align="C",
        )
        pdf.cell(
            0, 4,
            "This report is for quality assurance purposes. "
            "It does not constitute a clinical diagnosis.",
            new_x="LMARGIN", new_y="NEXT",
            align="C",
        )
        pdf.set_text_color(0, 0, 0)

        # Write PDF
        pdf.output(str(output_path))

    # ------------------------------------------------------------------
    # Evidence artifacts
    # ------------------------------------------------------------------

    def _generate_evidence(
        self,
        config: ReportConfig,
        metrics_report: Any,
        baseline_comparison: dict | None,
        drift_report: Any | None,
    ) -> Path:
        """Create the evidence/ folder with supporting artifacts.

        Generates:
        - ``roi_overlays/`` -- PNG images showing ROI placement on phantom slices
        - ``trend_plots/``  -- Time-series plots for trended metrics
        - ``metric_logs/``  -- Per-metric derivation logs (JSON)

        Parameters
        ----------
        config : ReportConfig
            Report configuration.
        metrics_report : Any
            Metrics data containing ROI and measurement details.
        baseline_comparison : dict | None
            Baseline comparison data.
        drift_report : Any | None
            Drift detection report with historical data.

        Returns
        -------
        Path
            Path to the evidence directory.
        """
        evidence_dir = Path(config.output_dir) / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        roi_dir = evidence_dir / "roi_overlays"
        roi_dir.mkdir(exist_ok=True)

        trend_dir = evidence_dir / "trend_plots"
        trend_dir.mkdir(exist_ok=True)

        log_dir = evidence_dir / "metric_logs"
        log_dir.mkdir(exist_ok=True)

        # ---- Metric derivation logs ----
        metrics_data = _extract_metrics_dict(metrics_report)
        for metric_key, metric_data in metrics_data.items():
            log_entry = {
                "metric_key": metric_key,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "scanner_id": config.scanner_id,
                "casepack_id": config.casepack_id,
                "measurement": metric_data if isinstance(metric_data, dict) else {"value": metric_data},
            }
            if baseline_comparison and metric_key in baseline_comparison:
                log_entry["baseline_delta"] = baseline_comparison[metric_key]

            log_path = log_dir / f"{_safe_filename(metric_key)}.json"
            log_path.write_text(
                json.dumps(log_entry, indent=2, default=str),
                encoding="utf-8",
            )

        # ---- ROI overlay generation ----
        # ROI overlays require image data from metrics_report.
        # Write a manifest indicating expected overlays even if image
        # data is not present (e.g., for headless/CI runs).
        roi_manifest = {
            "scanner_id": config.scanner_id,
            "date": config.date,
            "expected_overlays": [],
        }

        if hasattr(metrics_report, "roi_images"):
            roi_images = metrics_report.roi_images
        elif isinstance(metrics_report, dict):
            roi_images = metrics_report.get("roi_images", {})
        else:
            roi_images = {}

        for roi_name, image_data in roi_images.items():
            overlay_path = roi_dir / f"{_safe_filename(roi_name)}.png"
            roi_manifest["expected_overlays"].append(roi_name)
            try:
                _save_roi_overlay(image_data, overlay_path)
            except Exception as exc:
                logger.warning(
                    "Failed to save ROI overlay for %s: %s", roi_name, exc
                )

        (roi_dir / "manifest.json").write_text(
            json.dumps(roi_manifest, indent=2, default=str),
            encoding="utf-8",
        )

        # ---- Trend plot generation ----
        trend_manifest = {
            "scanner_id": config.scanner_id,
            "date": config.date,
            "generated_plots": [],
        }

        if drift_report is not None:
            trend_data = _extract_trend_data(drift_report)
            for metric_key, series in trend_data.items():
                plot_path = trend_dir / f"{_safe_filename(metric_key)}_trend.png"
                try:
                    _save_trend_plot(metric_key, series, plot_path)
                    trend_manifest["generated_plots"].append(metric_key)
                except Exception as exc:
                    logger.warning(
                        "Failed to save trend plot for %s: %s", metric_key, exc
                    )

        (trend_dir / "manifest.json").write_text(
            json.dumps(trend_manifest, indent=2, default=str),
            encoding="utf-8",
        )

        return evidence_dir

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _compute_overall_decision(self, threshold_results: dict) -> str:
        """Compute aggregate PASS/FAIL from individual metric results.

        The overall decision is PASS if and only if every metric threshold
        evaluation passed. A single FAIL in any metric produces an overall
        FAIL. WARNING status is treated as PASS for the aggregate decision
        (warnings are advisory, not blocking).

        Parameters
        ----------
        threshold_results : dict
            Mapping of metric_key -> evaluation dict. Each dict should
            contain a ``"status"`` key with value ``"PASS"``,
            ``"WARNING"``, or ``"FAIL"``.

        Returns
        -------
        str
            ``"PASS"`` or ``"FAIL"``.
        """
        if not threshold_results:
            return "PASS"

        for metric_key, result in threshold_results.items():
            status = _get_status(result)
            if status == "FAIL":
                return "FAIL"

        return "PASS"

    # ------------------------------------------------------------------
    # Table row construction
    # ------------------------------------------------------------------

    def _create_metric_table_rows(
        self,
        metrics_report: Any,
        threshold_results: dict,
        baseline_comparison: dict | None = None,
    ) -> list[dict]:
        """Build flat table rows for the PDF metrics table.

        Each row is a dict with keys matching the PDF column headers.

        Parameters
        ----------
        metrics_report : Any
            Metrics data (dict or object with metric measurements).
        threshold_results : dict
            Threshold evaluation results per metric.
        baseline_comparison : dict | None
            Baseline deltas for each metric.

        Returns
        -------
        list[dict]
            Ordered list of row dicts for the metrics table.
        """
        rows: list[dict] = []
        metrics_data = _extract_metrics_dict(metrics_report)

        for metric_key in sorted(threshold_results.keys()):
            result = threshold_results[metric_key]
            metric_info = metrics_data.get(metric_key, {})

            # Extract value
            if isinstance(metric_info, dict):
                value = metric_info.get("value", metric_info.get("measured", 0.0))
                unit = metric_info.get("unit", "")
                name = metric_info.get("name", metric_key)
            elif isinstance(metric_info, (int, float)):
                value = float(metric_info)
                unit = ""
                name = metric_key
            else:
                value = 0.0
                unit = ""
                name = metric_key

            # Extract threshold info
            if isinstance(result, dict):
                status = result.get("status", "PASS")
                standard_threshold = result.get("standard_threshold", {})
                applied_threshold = result.get("applied_threshold", result.get("threshold", {}))
                threshold_layer = result.get("threshold_layer", result.get("layer", "standard"))
            else:
                status = _get_status(result)
                standard_threshold = {}
                applied_threshold = {}
                threshold_layer = "standard"

            # Baseline
            baseline_value = None
            if baseline_comparison and metric_key in baseline_comparison:
                bl = baseline_comparison[metric_key]
                if isinstance(bl, dict):
                    baseline_value = bl.get("baseline_value", bl.get("value"))
                elif isinstance(bl, (int, float)):
                    baseline_value = float(bl)

            rows.append({
                "name": name,
                "value": value,
                "unit": unit,
                "baseline_value": baseline_value,
                "standard_threshold": standard_threshold,
                "applied_threshold": applied_threshold,
                "threshold_layer": threshold_layer,
                "status": status,
            })

        return rows

    # ------------------------------------------------------------------
    # Metric entry construction (for JSON)
    # ------------------------------------------------------------------

    def _create_metric_entries(
        self,
        metrics_report: Any,
        threshold_results: dict,
        baseline_comparison: dict | None,
    ) -> dict[str, ReportMetricEntry]:
        """Build ReportMetricEntry objects for the JSON report.

        Parameters
        ----------
        metrics_report : Any
            Metrics data.
        threshold_results : dict
            Threshold evaluation results.
        baseline_comparison : dict | None
            Baseline comparison data.

        Returns
        -------
        dict[str, ReportMetricEntry]
            Mapping of metric_key -> structured metric entry.
        """
        entries: dict[str, ReportMetricEntry] = {}
        table_rows = self._create_metric_table_rows(
            metrics_report, threshold_results, baseline_comparison,
        )

        for row in table_rows:
            name = row["name"]
            value = row["value"]
            baseline_value = row.get("baseline_value")
            delta = None
            if baseline_value is not None and isinstance(value, (int, float)):
                delta = float(value) - float(baseline_value)

            entry = ReportMetricEntry(
                name=name,
                value=float(value) if isinstance(value, (int, float)) else 0.0,
                unit=row.get("unit", ""),
                status=row.get("status", "PASS"),
                standard_threshold=row.get("standard_threshold", {}),
                applied_threshold=row.get("applied_threshold", {}),
                threshold_layer=row.get("threshold_layer", "standard"),
                baseline_value=baseline_value,
                delta=delta,
            )
            entries[name] = entry

        return entries


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_metrics_dict(metrics_report: Any) -> dict:
    """Extract a flat dict of metric data from various input formats.

    Supports dicts, Pydantic models with ``model_dump()``, and objects
    with a ``metrics`` attribute.
    """
    if isinstance(metrics_report, dict):
        # If it has a 'metrics' sub-key, use that
        if "metrics" in metrics_report:
            return metrics_report["metrics"]
        return metrics_report
    elif hasattr(metrics_report, "model_dump"):
        data = metrics_report.model_dump()
        return data.get("metrics", data)
    elif hasattr(metrics_report, "metrics"):
        m = metrics_report.metrics
        return m if isinstance(m, dict) else {}
    return {}


def _get_status(result: Any) -> str:
    """Extract status string from a threshold result."""
    if isinstance(result, dict):
        return result.get("status", "PASS")
    elif isinstance(result, str):
        return result if result in ("PASS", "WARNING", "FAIL") else "PASS"
    elif hasattr(result, "status"):
        return str(result.status)
    return "PASS"


def _format_threshold(threshold: dict) -> str:
    """Format a threshold dict into a compact display string.

    Examples: ``"0-5 HU"``, ``">2.0 mm"``, ``"+-0.5 HU"``.
    """
    if not threshold:
        return "-"

    parts = []
    low = threshold.get("low", threshold.get("min"))
    high = threshold.get("high", threshold.get("max"))
    unit = threshold.get("unit", "")
    tolerance = threshold.get("tolerance")

    if tolerance is not None:
        parts.append(f"+/-{tolerance}")
    elif low is not None and high is not None:
        parts.append(f"{low}-{high}")
    elif low is not None:
        parts.append(f">{low}")
    elif high is not None:
        parts.append(f"<{high}")

    if unit:
        parts.append(unit)

    return " ".join(parts) if parts else "-"


def _safe_filename(name: str) -> str:
    """Convert a metric name to a safe filename component."""
    import re
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:80]


def _save_roi_overlay(image_data: Any, output_path: Path) -> None:
    """Save an ROI overlay image as PNG.

    Accepts numpy arrays or PIL Images. Uses matplotlib if available,
    otherwise falls back to raw numpy-to-PNG via PIL.
    """
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not available; skipping ROI overlay save")
        return

    if isinstance(image_data, np.ndarray):
        try:
            from PIL import Image
            # Normalize to 0-255 uint8 for saving
            arr = image_data.astype(np.float64)
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
            arr = arr.astype(np.uint8)
            img = Image.fromarray(arr)
            img.save(str(output_path))
        except ImportError:
            # Fallback: save as raw .npy and log
            npy_path = output_path.with_suffix(".npy")
            np.save(str(npy_path), image_data)
            logger.info(
                "PIL not available; saved ROI overlay as .npy: %s", npy_path
            )
    elif hasattr(image_data, "save"):
        # PIL Image or similar
        image_data.save(str(output_path))


def _extract_trend_data(drift_report: Any) -> dict[str, list]:
    """Extract time-series data from a drift report for plotting.

    Returns a dict of metric_key -> list of (date, value) tuples.
    """
    if isinstance(drift_report, dict):
        return drift_report.get("trends", drift_report.get("series", {}))
    elif hasattr(drift_report, "trends"):
        trends = drift_report.trends
        return trends if isinstance(trends, dict) else {}
    return {}


def _save_trend_plot(
    metric_key: str, series: list, output_path: Path
) -> None:
    """Generate and save a trend line plot for a single metric.

    Uses matplotlib if available. Falls back to saving the series as JSON.

    Parameters
    ----------
    metric_key : str
        Name of the metric being plotted.
    series : list
        List of (date_str, value) tuples or dicts with ``date``/``value`` keys.
    output_path : Path
        Destination path for the PNG plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: save raw data as JSON
        json_path = output_path.with_suffix(".json")
        json_path.write_text(
            json.dumps({"metric": metric_key, "series": series}, default=str),
            encoding="utf-8",
        )
        logger.info(
            "matplotlib not available; saved trend data as JSON: %s", json_path
        )
        return

    dates = []
    values = []
    for point in series:
        if isinstance(point, dict):
            dates.append(str(point.get("date", "")))
            values.append(float(point.get("value", 0.0)))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            dates.append(str(point[0]))
            values.append(float(point[1]))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(values)), values, marker="o", linewidth=1.5, markersize=4)
    ax.set_title(f"Trend: {metric_key}", fontsize=10)
    ax.set_ylabel("Value")
    ax.set_xlabel("Measurement")

    if dates:
        # Show a subset of date labels to avoid crowding
        step = max(1, len(dates) // 8)
        tick_positions = list(range(0, len(dates), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [dates[i] for i in tick_positions],
            rotation=45, ha="right", fontsize=7,
        )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
