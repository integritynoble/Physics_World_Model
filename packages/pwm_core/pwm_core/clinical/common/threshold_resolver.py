"""Layered threshold resolution for clinical QA metrics.

Resolves thresholds through 4 override layers:

1. **standard_default** -- ACR/AAPM published thresholds
2. **scanner_model** -- vendor/model-specific adjustments
3. **protocol** -- kernel, slice thickness, dose level specifics
4. **site_override** -- local physicist customization

Later layers override earlier ones. The standard_default is always preserved
in the result for audit comparison.

Handles both absolute (``pass_range``) and relative-to-baseline
(``tolerance_from_baseline_sigma``, ``tolerance_from_nominal``) threshold
forms.

Usage
-----
>>> from pwm_core.clinical.common.threshold_resolver import ThresholdResolver
>>> resolver = ThresholdResolver(Path("thresholds.yaml"))
>>> result = resolver.evaluate("noise_std", value=8.2, scanner_model="SOMATOM Force")
>>> result.status
'PASS'

YAML Schema
-----------
The threshold YAML file should follow this structure::

    metrics:
      noise_std:
        standard_default:
          threshold_type: ABSOLUTE_RANGE
          pass_range: [0.0, 10.0]
          unit: HU
        scanner_model:
          SOMATOM Force:
            pass_range: [0.0, 9.0]
        protocol:
          thin_slice:
            pass_range: [0.0, 12.0]
        site_override:
          pass_range: [0.0, 11.0]
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ThresholdType(Enum):
    """Classification of threshold comparison methods.

    Attributes
    ----------
    ABSOLUTE_RANGE
        Value must fall within an absolute [min, max] range.
    RELATIVE_TO_BASELINE_SIGMA
        Value must be within N standard deviations of a baseline value.
    RELATIVE_TO_NOMINAL
        Value must be within a percentage or absolute tolerance of a
        nominal (expected) value.
    """

    ABSOLUTE_RANGE = "ABSOLUTE_RANGE"
    RELATIVE_TO_BASELINE_SIGMA = "RELATIVE_TO_BASELINE_SIGMA"
    RELATIVE_TO_NOMINAL = "RELATIVE_TO_NOMINAL"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResolvedThreshold(BaseModel):
    """A fully resolved threshold for a single metric.

    Attributes
    ----------
    metric_name : str
        Name of the metric this threshold applies to.
    threshold_type : ThresholdType
        How the threshold is evaluated.
    pass_range : tuple[float, float] | None
        Absolute [min, max] range for ABSOLUTE_RANGE thresholds.
    tolerance_sigma : float | None
        Number of standard deviations for RELATIVE_TO_BASELINE_SIGMA.
    tolerance_from_nominal : float | None
        Tolerance value for RELATIVE_TO_NOMINAL (units depend on metric).
    unit : str
        Unit of measurement (e.g., ``"HU"``, ``"mm"``, ``"%"``).
    source : str
        Description of where this threshold came from (e.g., file path,
        standard name).
    resolved_layer : str
        Which layer provided the final threshold value:
        ``"standard_default"``, ``"scanner_model"``, ``"protocol"``,
        or ``"site_override"``.
    standard_threshold : dict[str, Any]
        The original standard_default threshold specification, preserved
        for audit comparison in reports.
    """

    metric_name: str
    threshold_type: ThresholdType
    pass_range: tuple[float, float] | None = None
    tolerance_sigma: float | None = None
    tolerance_from_nominal: float | None = None
    unit: str = ""
    source: str = ""
    resolved_layer: str = "standard_default"
    standard_threshold: dict[str, Any] = Field(default_factory=dict)


class ThresholdResult(BaseModel):
    """Result of evaluating a metric value against a resolved threshold.

    Attributes
    ----------
    metric_name : str
        Name of the evaluated metric.
    value : float
        The measured value.
    status : Literal["PASS", "WARNING", "FAIL"]
        Evaluation result. ``WARNING`` is triggered when the value is
        within 10% of the fail boundary.
    resolved_threshold : ResolvedThreshold
        The threshold that was applied for evaluation.
    standard_threshold : ResolvedThreshold
        The standard_default threshold, always included for audit trail.
    """

    metric_name: str
    value: float
    status: Literal["PASS", "WARNING", "FAIL"]
    resolved_threshold: ResolvedThreshold
    standard_threshold: ResolvedThreshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LAYER_ORDER: list[str] = [
    "standard_default",
    "scanner_model",
    "protocol",
    "site_override",
]

_WARNING_MARGIN_FRACTION: float = 0.10  # 10% of range for WARNING zone


def _parse_threshold_type(raw: str) -> ThresholdType:
    """Convert a string to ThresholdType, case-insensitively.

    Parameters
    ----------
    raw : str
        The threshold type string from YAML.

    Returns
    -------
    ThresholdType
        The parsed enum value.

    Raises
    ------
    ValueError
        If the string does not match any known threshold type.
    """
    normalized = raw.strip().upper()
    try:
        return ThresholdType(normalized)
    except ValueError:
        valid = [t.value for t in ThresholdType]
        raise ValueError(
            f"Unknown threshold type '{raw}'. Valid types: {valid}"
        ) from None


def _layer_to_resolved(
    layer_data: dict[str, Any],
    metric_name: str,
    layer_name: str,
    base: ResolvedThreshold | None = None,
) -> ResolvedThreshold:
    """Merge a layer's data into a ResolvedThreshold.

    Parameters
    ----------
    layer_data : dict[str, Any]
        Raw YAML data for this layer.
    metric_name : str
        The metric name.
    layer_name : str
        Which layer this data came from.
    base : ResolvedThreshold | None
        Previous resolved threshold to use as fallback for unspecified fields.

    Returns
    -------
    ResolvedThreshold
        Updated threshold with this layer's overrides applied.
    """
    # Start from base or create new
    if base is not None:
        result_data = base.model_dump()
    else:
        result_data = {
            "metric_name": metric_name,
            "threshold_type": ThresholdType.ABSOLUTE_RANGE,
            "pass_range": None,
            "tolerance_sigma": None,
            "tolerance_from_nominal": None,
            "unit": "",
            "source": "",
            "resolved_layer": layer_name,
            "standard_threshold": {},
        }

    # Apply overrides from this layer
    if "threshold_type" in layer_data:
        result_data["threshold_type"] = _parse_threshold_type(
            layer_data["threshold_type"]
        )

    if "pass_range" in layer_data:
        pr = layer_data["pass_range"]
        if isinstance(pr, (list, tuple)) and len(pr) == 2:
            result_data["pass_range"] = (float(pr[0]), float(pr[1]))

    if "tolerance_sigma" in layer_data:
        result_data["tolerance_sigma"] = float(layer_data["tolerance_sigma"])
    # Also support the longer YAML key name
    if "tolerance_from_baseline_sigma" in layer_data:
        result_data["tolerance_sigma"] = float(
            layer_data["tolerance_from_baseline_sigma"]
        )
        # Infer threshold_type if not explicitly set
        if result_data["threshold_type"] == ThresholdType.ABSOLUTE_RANGE and result_data["pass_range"] is None:
            result_data["threshold_type"] = ThresholdType.RELATIVE_TO_BASELINE_SIGMA

    if "tolerance_from_nominal" in layer_data:
        result_data["tolerance_from_nominal"] = float(
            layer_data["tolerance_from_nominal"]
        )
        # Infer threshold_type if not explicitly set
        if result_data["threshold_type"] == ThresholdType.ABSOLUTE_RANGE and result_data["pass_range"] is None:
            result_data["threshold_type"] = ThresholdType.RELATIVE_TO_NOMINAL

    # Handle non-standard threshold keys by converting to pass_range.
    # min_visible_targets: N  ->  pass_range: [N, inf]  (more is better)
    if "min_visible_targets" in layer_data:
        n = float(layer_data["min_visible_targets"])
        result_data["pass_range"] = (n, math.inf)
        result_data["threshold_type"] = ThresholdType.ABSOLUTE_RANGE

    # max_artifact_score: N  ->  pass_range: [-inf, N]  (less is better)
    if "max_artifact_score" in layer_data:
        n = float(layer_data["max_artifact_score"])
        result_data["pass_range"] = (-math.inf, n)
        result_data["threshold_type"] = ThresholdType.ABSOLUTE_RANGE

    # min_resolvable_lp_per_cm: N  ->  pass_range: [N, inf]  (more is better)
    if "min_resolvable_lp_per_cm" in layer_data:
        n = float(layer_data["min_resolvable_lp_per_cm"])
        result_data["pass_range"] = (n, math.inf)
        result_data["threshold_type"] = ThresholdType.ABSOLUTE_RANGE

    if "unit" in layer_data:
        result_data["unit"] = str(layer_data["unit"])

    if "source" in layer_data:
        result_data["source"] = str(layer_data["source"])

    result_data["resolved_layer"] = layer_name

    return ResolvedThreshold(**result_data)


# ---------------------------------------------------------------------------
# Main resolver class
# ---------------------------------------------------------------------------

class ThresholdResolver:
    """Resolve and evaluate thresholds from a 4-layer YAML configuration.

    Parameters
    ----------
    threshold_yaml_path : Path
        Path to the threshold YAML file. See module docstring for schema.

    Examples
    --------
    >>> resolver = ThresholdResolver(Path("thresholds.yaml"))
    >>> threshold = resolver.resolve("noise_std", scanner_model="SOMATOM Force")
    >>> threshold.pass_range
    (0.0, 9.0)

    >>> result = resolver.evaluate("noise_std", value=8.2)
    >>> result.status
    'PASS'
    """

    def __init__(self, threshold_yaml_path: Path) -> None:
        """Load the threshold YAML file.

        Parameters
        ----------
        threshold_yaml_path : Path
            Path to the threshold YAML file.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        FileNotFoundError
            If the YAML file does not exist.
        """
        self._yaml_path = Path(threshold_yaml_path)
        raw_data = self._load_yaml()
        self._data: dict[str, Any] = self._normalize_schema(raw_data)

    def _load_yaml(self) -> dict[str, Any]:
        """Load and parse the threshold YAML file.

        Returns
        -------
        dict[str, Any]
            Parsed YAML data.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for threshold resolution. "
                "Install it with: pip install pyyaml"
            ) from exc

        if not self._yaml_path.exists():
            raise FileNotFoundError(
                f"Threshold YAML not found: {self._yaml_path}"
            )

        with open(self._yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Threshold YAML must be a mapping, got {type(data).__name__}"
            )

        return data

    @staticmethod
    def _normalize_schema(data: dict[str, Any]) -> dict[str, Any]:
        """Detect YAML schema format and normalise to metrics-first layout.

        The resolver's internal logic expects the **metrics-first** schema::

            metrics:
              metric_name:
                standard_default: { ... }
                scanner_model:
                  ScannerX: { ... }

        The production YAML uses a **layer-first** schema::

            standard_default:
              metric_name: { ... }
            scanner_model:
              ScannerX:
                metric_name: { ... }

        If the data already contains a top-level ``metrics`` key it is
        returned unchanged.  Otherwise the layer-first data is transposed
        into the metrics-first format.

        Parameters
        ----------
        data : dict[str, Any]
            Raw parsed YAML data.

        Returns
        -------
        dict[str, Any]
            Data guaranteed to have a ``metrics`` top-level key.
        """
        if "metrics" in data:
            return data

        # Detect layer-first format by checking for known layer keys
        layer_keys = set(_LAYER_ORDER)
        if not layer_keys & set(data.keys()):
            # Neither format detected; return as-is and let downstream
            # validation report the problem.
            return data

        metrics: dict[str, dict[str, Any]] = {}

        # Layer 1 — standard_default: {metric: config}
        for metric_name, config in data.get("standard_default", {}).items():
            if isinstance(config, dict):
                metrics.setdefault(metric_name, {})["standard_default"] = config

        # Layer 2 — scanner_model: {scanner: {metric: config}}
        for scanner_name, scanner_metrics in data.get("scanner_model", {}).items():
            if not isinstance(scanner_metrics, dict):
                continue
            for metric_name, config in scanner_metrics.items():
                if isinstance(config, dict):
                    metrics.setdefault(metric_name, {}).setdefault(
                        "scanner_model", {}
                    )[scanner_name] = config

        # Layer 3 — protocol: {protocol: {metric: config}}
        for protocol_name, protocol_metrics in data.get("protocol", {}).items():
            if not isinstance(protocol_metrics, dict):
                continue
            for metric_name, config in protocol_metrics.items():
                if isinstance(config, dict):
                    metrics.setdefault(metric_name, {}).setdefault(
                        "protocol", {}
                    )[protocol_name] = config

        # Layer 4 — site_override: {site_or_metric: config}
        # In the layer-first format, site_override can be either
        # {site_name: {metric: config}} or directly {metric: config}.
        site_data = data.get("site_override", {})
        if isinstance(site_data, dict):
            for key, value in site_data.items():
                if isinstance(value, dict):
                    # Could be a metric config directly or a site grouping.
                    # If the key is a known metric, treat as direct override.
                    if key in metrics:
                        metrics[key]["site_override"] = value
                    else:
                        # Assume it's a site-name grouping: {metric: config}
                        for metric_name, config in value.items():
                            if isinstance(config, dict):
                                metrics.setdefault(metric_name, {})[
                                    "site_override"
                                ] = config

        logger.debug(
            "Transposed layer-first YAML to metrics-first format "
            "(%d metrics detected).",
            len(metrics),
        )

        # Preserve non-layer metadata (version, threshold_set_id, etc.)
        result = {k: v for k, v in data.items() if k not in layer_keys}
        result["metrics"] = metrics
        return result

    def _get_metric_config(self, metric_name: str) -> dict[str, Any]:
        """Retrieve the raw configuration for a metric.

        Parameters
        ----------
        metric_name : str
            The metric to look up.

        Returns
        -------
        dict[str, Any]
            The metric's layer configuration from the YAML.

        Raises
        ------
        KeyError
            If the metric is not defined in the YAML.
        """
        metrics = self._data.get("metrics", {})
        if metric_name not in metrics:
            raise KeyError(
                f"Metric '{metric_name}' not found in threshold YAML. "
                f"Available metrics: {sorted(metrics.keys())}"
            )
        return metrics[metric_name]

    def resolve(
        self,
        metric_name: str,
        scanner_model: str | None = None,
        protocol: str | None = None,
    ) -> ResolvedThreshold:
        """Resolve the effective threshold for a metric by walking all layers.

        Later layers override earlier ones. The ``standard_default`` layer
        is always preserved in the result's ``standard_threshold`` field.

        Parameters
        ----------
        metric_name : str
            Name of the QA metric (e.g., ``"noise_std"``).
        scanner_model : str | None
            Scanner model name for layer 2 lookup. If ``None``, this layer
            is skipped.
        protocol : str | None
            Protocol name for layer 3 lookup. If ``None``, this layer
            is skipped.

        Returns
        -------
        ResolvedThreshold
            The fully resolved threshold with source attribution.

        Raises
        ------
        KeyError
            If *metric_name* is not defined in the threshold YAML.
        ValueError
            If the standard_default layer is missing for this metric.
        """
        metric_config = self._get_metric_config(metric_name)

        # Layer 1: standard_default (required)
        standard_data = metric_config.get("standard_default")
        if not standard_data or not isinstance(standard_data, dict):
            raise ValueError(
                f"Metric '{metric_name}' is missing a 'standard_default' layer."
            )

        resolved = _layer_to_resolved(
            standard_data, metric_name, "standard_default"
        )
        resolved.source = str(self._yaml_path)

        # Preserve the standard_default for audit
        standard_snapshot = resolved.model_dump()
        resolved.standard_threshold = standard_snapshot

        # Layer 2: scanner_model
        scanner_layer = metric_config.get("scanner_model", {})
        if scanner_model and isinstance(scanner_layer, dict):
            model_data = scanner_layer.get(scanner_model)
            if model_data and isinstance(model_data, dict):
                resolved = _layer_to_resolved(
                    model_data, metric_name, "scanner_model", base=resolved
                )
                resolved.standard_threshold = standard_snapshot

        # Layer 3: protocol
        protocol_layer = metric_config.get("protocol", {})
        if protocol and isinstance(protocol_layer, dict):
            protocol_data = protocol_layer.get(protocol)
            if protocol_data and isinstance(protocol_data, dict):
                resolved = _layer_to_resolved(
                    protocol_data, metric_name, "protocol", base=resolved
                )
                resolved.standard_threshold = standard_snapshot

        # Layer 4: site_override
        site_layer = metric_config.get("site_override")
        if site_layer and isinstance(site_layer, dict):
            resolved = _layer_to_resolved(
                site_layer, metric_name, "site_override", base=resolved
            )
            resolved.standard_threshold = standard_snapshot

        return resolved

    def resolve_all(
        self,
        metric_names: list[str],
        scanner_model: str | None = None,
        protocol: str | None = None,
    ) -> dict[str, ResolvedThreshold]:
        """Resolve thresholds for multiple metrics at once.

        Parameters
        ----------
        metric_names : list[str]
            List of metric names to resolve.
        scanner_model : str | None
            Scanner model for layer 2 lookup.
        protocol : str | None
            Protocol for layer 3 lookup.

        Returns
        -------
        dict[str, ResolvedThreshold]
            Mapping from metric name to resolved threshold.
        """
        return {
            name: self.resolve(name, scanner_model=scanner_model, protocol=protocol)
            for name in metric_names
        }

    def evaluate(
        self,
        metric_name: str,
        value: float,
        baseline_value: float | None = None,
        nominal_value: float | None = None,
        scanner_model: str | None = None,
        protocol: str | None = None,
    ) -> ThresholdResult:
        """Evaluate a measured value against the resolved threshold.

        Parameters
        ----------
        metric_name : str
            Name of the QA metric.
        value : float
            The measured value to evaluate.
        baseline_value : float | None
            Historical baseline value, required for
            ``RELATIVE_TO_BASELINE_SIGMA`` thresholds.
        nominal_value : float | None
            Expected nominal value, required for ``RELATIVE_TO_NOMINAL``
            thresholds.
        scanner_model : str | None
            Scanner model for threshold resolution.
        protocol : str | None
            Protocol for threshold resolution.

        Returns
        -------
        ThresholdResult
            Evaluation result with pass/warning/fail status and full
            threshold provenance.

        Raises
        ------
        ValueError
            If required reference values (baseline or nominal) are missing
            for the threshold type.
        """
        resolved = self.resolve(
            metric_name, scanner_model=scanner_model, protocol=protocol
        )

        # Also resolve the standard_default-only version for audit
        standard_resolved = self.resolve(metric_name)
        # Force standard_resolved to only reflect standard_default
        metric_config = self._get_metric_config(metric_name)
        standard_data = metric_config.get("standard_default", {})
        standard_threshold = _layer_to_resolved(
            standard_data, metric_name, "standard_default"
        )
        standard_threshold.source = str(self._yaml_path)
        standard_threshold.standard_threshold = standard_data

        status = self._evaluate_status(
            value=value,
            threshold=resolved,
            baseline_value=baseline_value,
            nominal_value=nominal_value,
        )

        return ThresholdResult(
            metric_name=metric_name,
            value=value,
            status=status,
            resolved_threshold=resolved,
            standard_threshold=standard_threshold,
        )

    def _evaluate_status(
        self,
        value: float,
        threshold: ResolvedThreshold,
        baseline_value: float | None = None,
        nominal_value: float | None = None,
    ) -> Literal["PASS", "WARNING", "FAIL"]:
        """Determine PASS/WARNING/FAIL status for a value.

        WARNING is triggered when the value is within 10% of the fail
        boundary (measured as 10% of the total pass range width for
        ABSOLUTE_RANGE, or 10% of the tolerance for relative types).

        Parameters
        ----------
        value : float
            Measured value.
        threshold : ResolvedThreshold
            The resolved threshold to evaluate against.
        baseline_value : float | None
            Baseline for sigma-based thresholds.
        nominal_value : float | None
            Nominal for relative-to-nominal thresholds.

        Returns
        -------
        Literal["PASS", "WARNING", "FAIL"]
            The evaluation status.
        """
        if threshold.threshold_type == ThresholdType.ABSOLUTE_RANGE:
            return self._evaluate_absolute_range(value, threshold)

        elif threshold.threshold_type == ThresholdType.RELATIVE_TO_BASELINE_SIGMA:
            return self._evaluate_relative_sigma(
                value, threshold, baseline_value
            )

        elif threshold.threshold_type == ThresholdType.RELATIVE_TO_NOMINAL:
            return self._evaluate_relative_nominal(
                value, threshold, nominal_value
            )

        else:
            logger.warning(
                "Unknown threshold type '%s' for metric '%s', defaulting to FAIL.",
                threshold.threshold_type,
                threshold.metric_name,
            )
            return "FAIL"

    def _evaluate_absolute_range(
        self,
        value: float,
        threshold: ResolvedThreshold,
    ) -> Literal["PASS", "WARNING", "FAIL"]:
        """Evaluate against an absolute [min, max] range.

        Parameters
        ----------
        value : float
            Measured value.
        threshold : ResolvedThreshold
            Must have ``pass_range`` set.

        Returns
        -------
        Literal["PASS", "WARNING", "FAIL"]
        """
        if threshold.pass_range is None:
            logger.warning(
                "ABSOLUTE_RANGE threshold for '%s' has no pass_range, "
                "defaulting to FAIL.",
                threshold.metric_name,
            )
            return "FAIL"

        low, high = threshold.pass_range
        range_width = high - low

        if range_width <= 0:
            # Degenerate range: exact match required
            return "PASS" if value == low else "FAIL"

        # Check FAIL first
        if value < low or value > high:
            return "FAIL"

        # For infinite or semi-infinite ranges, the WARNING margin
        # only applies on finite boundaries.  Use 10% of the distance
        # from the value to the finite boundary, measured relative to
        # the boundary magnitude (with a floor of 10% of 1 unit).
        if math.isinf(range_width):
            if not math.isinf(low):
                finite_margin = max(abs(low), 1.0) * _WARNING_MARGIN_FRACTION
                if (value - low) < finite_margin:
                    return "WARNING"
            if not math.isinf(high):
                finite_margin = max(abs(high), 1.0) * _WARNING_MARGIN_FRACTION
                if (high - value) < finite_margin:
                    return "WARNING"
            return "PASS"

        warning_margin = range_width * _WARNING_MARGIN_FRACTION

        # Check WARNING: within 10% of boundary
        if (value - low) < warning_margin or (high - value) < warning_margin:
            return "WARNING"

        return "PASS"

    def _evaluate_relative_sigma(
        self,
        value: float,
        threshold: ResolvedThreshold,
        baseline_value: float | None,
    ) -> Literal["PASS", "WARNING", "FAIL"]:
        """Evaluate against a baseline +/- N sigma tolerance.

        Note: ``tolerance_sigma`` is treated as an absolute tolerance in the
        metric's natural units (e.g. HU for CT numbers), not as a multiplier
        on the process standard deviation.  This provides consistent,
        deterministic thresholds without requiring historical variance data.

        Parameters
        ----------
        value : float
            Measured value.
        threshold : ResolvedThreshold
            Must have ``tolerance_sigma`` set.
        baseline_value : float | None
            The baseline value. Required.

        Returns
        -------
        Literal["PASS", "WARNING", "FAIL"]
        """
        if baseline_value is None:
            raise ValueError(
                f"Metric '{threshold.metric_name}' uses "
                f"RELATIVE_TO_BASELINE_SIGMA but no baseline_value was provided."
            )

        if threshold.tolerance_sigma is None:
            logger.warning(
                "RELATIVE_TO_BASELINE_SIGMA threshold for '%s' has no "
                "tolerance_sigma, defaulting to FAIL.",
                threshold.metric_name,
            )
            return "FAIL"

        sigma = threshold.tolerance_sigma
        deviation = abs(value - baseline_value)

        if deviation > sigma:
            return "FAIL"

        # WARNING within 10% of the boundary
        warning_boundary = sigma * (1.0 - _WARNING_MARGIN_FRACTION)
        if deviation > warning_boundary:
            return "WARNING"

        return "PASS"

    def _evaluate_relative_nominal(
        self,
        value: float,
        threshold: ResolvedThreshold,
        nominal_value: float | None,
    ) -> Literal["PASS", "WARNING", "FAIL"]:
        """Evaluate against a nominal value +/- tolerance.

        Parameters
        ----------
        value : float
            Measured value.
        threshold : ResolvedThreshold
            Must have ``tolerance_from_nominal`` set.
        nominal_value : float | None
            The expected nominal value. Required.

        Returns
        -------
        Literal["PASS", "WARNING", "FAIL"]
        """
        if nominal_value is None:
            raise ValueError(
                f"Metric '{threshold.metric_name}' uses "
                f"RELATIVE_TO_NOMINAL but no nominal_value was provided."
            )

        if threshold.tolerance_from_nominal is None:
            logger.warning(
                "RELATIVE_TO_NOMINAL threshold for '%s' has no "
                "tolerance_from_nominal, defaulting to FAIL.",
                threshold.metric_name,
            )
            return "FAIL"

        tolerance = threshold.tolerance_from_nominal
        deviation = abs(value - nominal_value)

        if deviation > tolerance:
            return "FAIL"

        # WARNING within 10% of the boundary
        warning_boundary = tolerance * (1.0 - _WARNING_MARGIN_FRACTION)
        if deviation > warning_boundary:
            return "WARNING"

        return "PASS"
