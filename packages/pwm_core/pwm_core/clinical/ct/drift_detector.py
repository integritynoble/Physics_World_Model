"""Drift detection for clinical QC time series.

Implements statistical process control (SPC) with control charts and
Western Electric rules for detecting scanner performance drift.

The :class:`DriftDetector` maintains a per-scanner JSON history of QC
measurements and, on each call to :meth:`detect_drift`, builds Shewhart
control charts and applies the five Western Electric rules:

1. **Single point outside 3-sigma** -- immediate FAIL.
2. **2 of 3 consecutive points outside 2-sigma** (same side) --
   ACTION_REQUIRED.
3. **4 of 5 consecutive points outside 1-sigma** (same side) -- WARNING.
4. **Run of 7 consecutive points on the same side of the mean** -- WARNING.
5. **Trend of 7 consecutive points all increasing or all decreasing** --
   WARNING.

Usage::

    detector = DriftDetector(Path("qc_history/"))
    detector.add_measurement("CT-001", "2026-01-15", {"noise_std": 5.2, ...})
    report = detector.detect_drift("CT-001")
    for alert in report.alerts:
        print(f"[{alert.alert_level}] {alert.metric_name}: {alert.description}")
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DriftAlert(BaseModel):
    """A single drift alert for one metric.

    Attributes
    ----------
    metric_name : str
        Name of the QC metric that triggered the alert.
    alert_level : Literal["INFO", "WARNING", "ACTION_REQUIRED", "FAIL"]
        Severity level.
    rule_triggered : str
        Identifier of the Western Electric rule that fired.
    description : str
        Human-readable explanation of the alert.
    current_value : float
        Most recent measured value.
    control_limits : dict
        Control chart limits (``mean``, ``std``, ``ucl``, ``lcl``, etc.).
    """

    metric_name: str
    alert_level: Literal["INFO", "WARNING", "ACTION_REQUIRED", "FAIL"]
    rule_triggered: str
    description: str
    current_value: float
    control_limits: dict[str, float]


class ControlChart(BaseModel):
    """Shewhart control chart data for a single metric.

    Attributes
    ----------
    metric_name : str
        Name of the QC metric.
    values : list[float]
        Historical measurement values in chronological order.
    dates : list[str]
        Corresponding ISO-8601 date strings.
    mean : float
        Process mean (centre line).
    std : float
        Process standard deviation.
    ucl : float
        Upper control limit (mean + 3 * std).
    lcl : float
        Lower control limit (mean - 3 * std).
    ucl_2sigma : float
        Upper 2-sigma warning limit (mean + 2 * std).
    lcl_2sigma : float
        Lower 2-sigma warning limit (mean - 2 * std).
    """

    metric_name: str
    values: list[float]
    dates: list[str]
    mean: float
    std: float
    ucl: float
    lcl: float
    ucl_2sigma: float
    lcl_2sigma: float


class DriftReport(BaseModel):
    """Complete drift detection output for a scanner.

    Attributes
    ----------
    scanner_id : str
        Scanner identifier.
    date : str
        ISO-8601 timestamp of the analysis.
    alerts : list[DriftAlert]
        All triggered alerts across all metrics.
    control_charts : dict[str, ControlChart]
        Per-metric control chart data.
    overall_status : Literal["STABLE", "WATCH", "ACTION_REQUIRED"]
        Aggregate scanner status:

        * ``STABLE`` -- no alerts above INFO.
        * ``WATCH`` -- WARNING-level alerts present.
        * ``ACTION_REQUIRED`` -- ACTION_REQUIRED or FAIL alerts present.
    """

    scanner_id: str
    date: str
    alerts: list[DriftAlert]
    control_charts: dict[str, ControlChart]
    overall_status: Literal["STABLE", "WATCH", "ACTION_REQUIRED"]


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Statistical process control for CT QC time series.

    Maintains per-scanner measurement history and applies Western Electric
    rules to detect performance drift.

    Parameters
    ----------
    history_path : Path
        Directory where per-scanner JSON history files are stored.
        Each scanner gets one file: ``<scanner_id>.json``.

    Storage format::

        {
            "scanner_id": "CT-001",
            "measurements": [
                {"date": "2026-01-01", "metrics": {"noise_std": 5.0, ...}},
                {"date": "2026-01-15", "metrics": {"noise_std": 5.1, ...}},
                ...
            ]
        }
    """

    # Minimum number of measurements before control chart statistics
    # are considered meaningful.
    _MIN_MEASUREMENTS: int = 5

    def __init__(self, history_path: Path) -> None:
        self.history_path = Path(history_path)
        self.history_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_measurement(
        self,
        scanner_id: str,
        date: str,
        metrics: dict[str, float],
    ) -> None:
        """Append a QC measurement to the scanner's history.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.
        date : str
            ISO-8601 date string (e.g. ``"2026-01-15"``).
        metrics : dict[str, float]
            Measured metric values.
        """
        history = self._load_history(scanner_id)
        history.append({"date": date, "metrics": metrics})
        self._save_history(scanner_id, history)
        logger.debug(
            "Added measurement for scanner '%s' on %s (%d metrics).",
            scanner_id, date, len(metrics),
        )

    def detect_drift(
        self,
        scanner_id: str,
        baseline: Any = None,
    ) -> DriftReport:
        """Analyse measurement history and detect drift.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.
        baseline : Any, optional
            A :class:`~pwm_core.clinical.ct.baseline.CommissioningBundle`
            to use as the process target.  If provided, the baseline
            metric values are used as the centre line instead of the
            historical mean.

        Returns
        -------
        DriftReport
            Complete drift analysis with alerts and control charts.
        """
        history = self._load_history(scanner_id)

        if not history:
            logger.warning(
                "No measurement history for scanner '%s'.", scanner_id,
            )
            return DriftReport(
                scanner_id=scanner_id,
                date=datetime.now(timezone.utc).isoformat(),
                alerts=[],
                control_charts={},
                overall_status="STABLE",
            )

        # Collect all metric names across history
        all_metric_names: set[str] = set()
        for entry in history:
            all_metric_names.update(entry.get("metrics", {}).keys())

        all_alerts: list[DriftAlert] = []
        control_charts: dict[str, ControlChart] = {}

        for metric_name in sorted(all_metric_names):
            # Extract time series for this metric
            values: list[float] = []
            dates: list[str] = []
            for entry in history:
                m = entry.get("metrics", {})
                if metric_name in m:
                    values.append(float(m[metric_name]))
                    dates.append(entry.get("date", ""))

            if len(values) < 2:
                continue

            # Determine centre line and standard deviation
            if baseline is not None:
                baseline_metrics = getattr(baseline, "metrics", {})
                if metric_name in baseline_metrics:
                    baseline_entry = baseline_metrics[metric_name]
                    center = float(
                        baseline_entry.get("value", baseline_entry)
                        if isinstance(baseline_entry, dict)
                        else baseline_entry
                    )
                    # Use std from early stable period when baseline is set
                    ref_window = values[:self._MIN_MEASUREMENTS]
                    std = _std(ref_window) if len(ref_window) >= 2 else _std(values)
                else:
                    center = _mean(values)
                    std = _std(values)
            else:
                center = _mean(values)
                std = _std(values)

            # Build control chart
            chart = ControlChart(
                metric_name=metric_name,
                values=values,
                dates=dates,
                mean=center,
                std=std,
                ucl=center + 3.0 * std,
                lcl=center - 3.0 * std,
                ucl_2sigma=center + 2.0 * std,
                lcl_2sigma=center - 2.0 * std,
            )
            control_charts[metric_name] = chart

            # Apply Western Electric rules (only if enough data)
            if len(values) >= self._MIN_MEASUREMENTS:
                metric_alerts = self._apply_western_electric(
                    values, center, std, metric_name,
                )
                all_alerts.extend(metric_alerts)

        # Determine overall status
        overall_status = self._compute_overall_status(all_alerts)

        return DriftReport(
            scanner_id=scanner_id,
            date=datetime.now(timezone.utc).isoformat(),
            alerts=all_alerts,
            control_charts=control_charts,
            overall_status=overall_status,
        )

    def get_history(self, scanner_id: str) -> list[dict[str, Any]]:
        """Retrieve the full measurement history for a scanner.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.

        Returns
        -------
        list[dict]
            List of measurement entries, each with ``"date"`` and
            ``"metrics"`` keys.
        """
        return self._load_history(scanner_id)

    def get_control_chart(
        self,
        scanner_id: str,
        metric_name: str,
    ) -> ControlChart:
        """Build and return the control chart for a single metric.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.
        metric_name : str
            Name of the QC metric.

        Returns
        -------
        ControlChart
            Populated control chart.

        Raises
        ------
        ValueError
            If there are insufficient data points for the metric.
        """
        history = self._load_history(scanner_id)

        values: list[float] = []
        dates: list[str] = []
        for entry in history:
            m = entry.get("metrics", {})
            if metric_name in m:
                values.append(float(m[metric_name]))
                dates.append(entry.get("date", ""))

        if len(values) < 2:
            raise ValueError(
                f"Insufficient data for metric '{metric_name}' "
                f"on scanner '{scanner_id}': need >= 2 points, "
                f"have {len(values)}."
            )

        center = _mean(values)
        std = _std(values)

        return ControlChart(
            metric_name=metric_name,
            values=values,
            dates=dates,
            mean=center,
            std=std,
            ucl=center + 3.0 * std,
            lcl=center - 3.0 * std,
            ucl_2sigma=center + 2.0 * std,
            lcl_2sigma=center - 2.0 * std,
        )

    # ------------------------------------------------------------------
    # Western Electric rules
    # ------------------------------------------------------------------

    def _apply_western_electric(
        self,
        values: list[float],
        mean: float,
        std: float,
        metric_name: str = "",
    ) -> list[DriftAlert]:
        """Apply Western Electric rules to a measurement time series.

        Parameters
        ----------
        values : list[float]
            Chronological measurement values.
        mean : float
            Process centre line.
        std : float
            Process standard deviation.
        metric_name : str
            Metric name (for alert messages).

        Returns
        -------
        list[DriftAlert]
            Alerts triggered by the rules (may be empty).
        """
        alerts: list[DriftAlert] = []

        if std < 1e-12:
            # Zero variance -- cannot apply sigma-based rules
            return alerts

        control_limits = {
            "mean": mean,
            "std": std,
            "ucl": mean + 3.0 * std,
            "lcl": mean - 3.0 * std,
            "ucl_2sigma": mean + 2.0 * std,
            "lcl_2sigma": mean - 2.0 * std,
        }

        latest = values[-1]

        # --- Rule 1: Single point outside 3-sigma -------------------------
        if latest > mean + 3.0 * std or latest < mean - 3.0 * std:
            side = "above UCL" if latest > mean else "below LCL"
            alerts.append(DriftAlert(
                metric_name=metric_name,
                alert_level="FAIL",
                rule_triggered="WE_Rule_1",
                description=(
                    f"Single point {side} (3-sigma): "
                    f"{metric_name} = {latest:.4f}, "
                    f"limits = [{mean - 3*std:.4f}, {mean + 3*std:.4f}]"
                ),
                current_value=latest,
                control_limits=control_limits,
            ))

        # --- Rule 2: 2 of 3 consecutive outside 2-sigma (same side) -------
        if len(values) >= 3:
            last3 = values[-3:]
            above_2sigma = sum(1 for v in last3 if v > mean + 2.0 * std)
            below_2sigma = sum(1 for v in last3 if v < mean - 2.0 * std)

            if above_2sigma >= 2:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="ACTION_REQUIRED",
                    rule_triggered="WE_Rule_2",
                    description=(
                        f"2 of 3 consecutive points above 2-sigma for "
                        f"{metric_name}: values = {last3}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))
            elif below_2sigma >= 2:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="ACTION_REQUIRED",
                    rule_triggered="WE_Rule_2",
                    description=(
                        f"2 of 3 consecutive points below 2-sigma for "
                        f"{metric_name}: values = {last3}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))

        # --- Rule 3: 4 of 5 consecutive outside 1-sigma (same side) -------
        if len(values) >= 5:
            last5 = values[-5:]
            above_1sigma = sum(1 for v in last5 if v > mean + 1.0 * std)
            below_1sigma = sum(1 for v in last5 if v < mean - 1.0 * std)

            if above_1sigma >= 4:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="WARNING",
                    rule_triggered="WE_Rule_3",
                    description=(
                        f"4 of 5 consecutive points above 1-sigma for "
                        f"{metric_name}: values = {last5}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))
            elif below_1sigma >= 4:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="WARNING",
                    rule_triggered="WE_Rule_3",
                    description=(
                        f"4 of 5 consecutive points below 1-sigma for "
                        f"{metric_name}: values = {last5}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))

        # --- Rule 4: Run of 7 on same side of mean ------------------------
        if len(values) >= 7:
            last7 = values[-7:]
            all_above = all(v > mean for v in last7)
            all_below = all(v < mean for v in last7)

            if all_above:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="WARNING",
                    rule_triggered="WE_Rule_4",
                    description=(
                        f"Run of 7 consecutive points above mean for "
                        f"{metric_name}: values = {last7}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))
            elif all_below:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="WARNING",
                    rule_triggered="WE_Rule_4",
                    description=(
                        f"Run of 7 consecutive points below mean for "
                        f"{metric_name}: values = {last7}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))

        # --- Rule 5: Trend of 7 all increasing or all decreasing ----------
        if len(values) >= 7:
            last7 = values[-7:]
            all_increasing = all(
                last7[i + 1] > last7[i] for i in range(6)
            )
            all_decreasing = all(
                last7[i + 1] < last7[i] for i in range(6)
            )

            if all_increasing:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="WARNING",
                    rule_triggered="WE_Rule_5",
                    description=(
                        f"Trend of 7 consecutive increasing points for "
                        f"{metric_name}: values = {last7}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))
            elif all_decreasing:
                alerts.append(DriftAlert(
                    metric_name=metric_name,
                    alert_level="WARNING",
                    rule_triggered="WE_Rule_5",
                    description=(
                        f"Trend of 7 consecutive decreasing points for "
                        f"{metric_name}: values = {last7}"
                    ),
                    current_value=latest,
                    control_limits=control_limits,
                ))

        return alerts

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def _load_history(self, scanner_id: str) -> list[dict[str, Any]]:
        """Load measurement history from the JSON file.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.

        Returns
        -------
        list[dict]
            List of measurement entries.
        """
        path = self.history_path / f"{scanner_id}.json"
        if not path.exists():
            return []

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw.get("measurements", [])
        except Exception:
            logger.warning(
                "Failed to load history for scanner '%s'; returning empty.",
                scanner_id, exc_info=True,
            )
            return []

    def _save_history(
        self,
        scanner_id: str,
        measurements: list[dict[str, Any]],
    ) -> None:
        """Persist measurement history to the JSON file.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.
        measurements : list[dict]
            Full list of measurement entries.
        """
        path = self.history_path / f"{scanner_id}.json"
        data = {
            "scanner_id": scanner_id,
            "measurements": measurements,
        }
        path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Status aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall_status(
        alerts: list[DriftAlert],
    ) -> Literal["STABLE", "WATCH", "ACTION_REQUIRED"]:
        """Determine overall scanner status from triggered alerts.

        Parameters
        ----------
        alerts : list[DriftAlert]
            All alerts across all metrics.

        Returns
        -------
        Literal["STABLE", "WATCH", "ACTION_REQUIRED"]
            Aggregate status.
        """
        if not alerts:
            return "STABLE"

        levels = {a.alert_level for a in alerts}

        if "FAIL" in levels or "ACTION_REQUIRED" in levels:
            return "ACTION_REQUIRED"
        elif "WARNING" in levels:
            return "WATCH"
        else:
            return "STABLE"


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers (no numpy dependency required)
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of floats.

    Parameters
    ----------
    values : list[float]
        Input values (must be non-empty).

    Returns
    -------
    float
        Arithmetic mean.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Compute the sample standard deviation of a list of floats.

    Uses Bessel's correction (N-1 denominator) when N >= 2.

    Parameters
    ----------
    values : list[float]
        Input values (must be non-empty).

    Returns
    -------
    float
        Sample standard deviation.
    """
    n = len(values)
    if n < 2:
        return 0.0

    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)
