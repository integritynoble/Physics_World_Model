"""Immutable CommissioningBundle for clinical QC baselines.

Baselines are versioned, signed, and chained. New baseline = new version
(never overwrite). Every baseline records the service_event that triggered it.

A :class:`CommissioningBundle` captures the complete scanner state at
commissioning time --- all ACR phantom metrics, the operator-graph
parameters, and cryptographic integrity hashes.  The :class:`BaselineManager`
handles persistence, retrieval, and delta-comparison against live QC
measurements.

Typical workflow::

    mgr = BaselineManager(Path("baselines/"))
    bundle = mgr.create_baseline(
        scanner_id="CT-001",
        scanner_model="GE Revolution Apex",
        metrics={"ct_number_water": {"value": 0.5, "unit": "HU"}, ...},
        operator_graph_state={"kVp": 120, "mAs": 200},
        approved_by="J. Smith, MS, DABR",
        service_event="initial_installation",
        casepack_id="acr_ct_v1.0",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CommissioningBundle(BaseModel, frozen=True):
    """Immutable snapshot of a scanner's commissioning state.

    Every field is frozen after construction.  Changing any parameter
    requires creating a new version that chains to the previous one
    via ``previous_version``.

    Attributes
    ----------
    version : str
        Semantic version string (e.g. ``"1.0.0"``).
    scanner_id : str
        Site-unique scanner identifier (e.g. ``"CT-001"``).
    scanner_model : str
        Manufacturer model name (e.g. ``"GE Revolution Apex"``).
    date : str
        ISO-8601 timestamp of baseline creation.
    approved_by : str
        Name and credentials of the approving physicist.
    service_event : str
        Event that triggered commissioning (``"initial_installation"``,
        ``"tube_change"``, ``"software_upgrade"``, ``"annual_qc"``, etc.).
    metrics : dict[str, dict]
        Mapping of metric name to ``{"value": ..., "unit": ...}``.
    operator_graph_state : dict
        Scanner acquisition parameters at commissioning time.
    sha256_inputs : str
        SHA-256 hex digest of the input data (metrics + operator state).
    sha256_outputs : str
        SHA-256 hex digest of the serialised bundle (excluding this field).
    provenance : dict
        Build provenance (``pwm_version``, ``git_hash``, ``casepack``).
    previous_version : str | None
        Version string of the preceding baseline, or ``None`` for the first.
    """

    version: str
    scanner_id: str
    scanner_model: str
    date: str
    approved_by: str
    service_event: str
    metrics: dict[str, dict[str, Any]]
    operator_graph_state: dict[str, Any] = Field(default_factory=dict)
    sha256_inputs: str
    sha256_outputs: str
    provenance: dict[str, Any]
    previous_version: str | None = None


class BaselineComparison(BaseModel):
    """Per-metric delta between a live measurement and the baseline.

    Attributes
    ----------
    metric_name : str
        Name of the QC metric.
    baseline_value : float
        Value stored in the commissioning bundle.
    current_value : float
        Today's measured value.
    delta : float
        ``current_value - baseline_value``.
    delta_percent : float
        Percentage change relative to the baseline.
    status : Literal["STABLE", "DRIFTED", "ALERT"]
        Qualitative assessment:

        * ``STABLE`` -- delta < 5 %
        * ``DRIFTED`` -- 5 % <= delta < 15 %
        * ``ALERT``  -- delta >= 15 %
    """

    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_percent: float
    status: Literal["STABLE", "DRIFTED", "ALERT"]


# ---------------------------------------------------------------------------
# Input sanitisation helpers
# ---------------------------------------------------------------------------

_SAFE_SCANNER_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _sanitize_scanner_id(scanner_id: str) -> str:
    """Validate that *scanner_id* is safe for use in file paths.

    Only alphanumeric characters, underscores, and hyphens are allowed.

    Raises
    ------
    ValueError
        If the scanner_id contains unsafe characters.
    """
    if not scanner_id or not _SAFE_SCANNER_ID_RE.match(scanner_id):
        raise ValueError(
            f"Invalid scanner_id '{scanner_id}': must be non-empty and "
            "contain only alphanumeric characters, underscores, or hyphens."
        )
    return scanner_id


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class BaselineManager:
    """Create, store, retrieve, and compare commissioning baselines.

    Storage layout::

        storage_path/
            CT-001/
                v1.0.0.json
                v1.1.0.json
            CT-002/
                v1.0.0.json

    Parameters
    ----------
    storage_path : Path
        Root directory for baseline JSON files.
    """

    # Percentage thresholds for comparison status classification.
    _DRIFT_THRESHOLD_PERCENT: float = 5.0
    _ALERT_THRESHOLD_PERCENT: float = 15.0

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_baseline(
        self,
        scanner_id: str,
        scanner_model: str,
        metrics: dict[str, dict[str, Any]],
        operator_graph_state: dict[str, Any],
        approved_by: str,
        service_event: str,
        casepack_id: str,
        previous_version: str | None = None,
    ) -> CommissioningBundle:
        """Create a new commissioning baseline, sign it, and persist it.

        Parameters
        ----------
        scanner_id : str
            Site-unique scanner identifier.
        scanner_model : str
            Manufacturer model name.
        metrics : dict[str, dict]
            QC metrics mapping (metric_name -> {value, unit}).
        operator_graph_state : dict
            Scanner acquisition parameters at commissioning.
        approved_by : str
            Approving physicist name / credentials.
        service_event : str
            Triggering event (e.g. ``"tube_change"``).
        casepack_id : str
            CasePack used for commissioning measurements.
        previous_version : str | None
            Predecessor baseline version, if any.

        Returns
        -------
        CommissioningBundle
            The frozen, signed commissioning bundle.
        """
        # Determine version number
        existing = self.list_baselines(scanner_id)
        if existing:
            latest = existing[-1]
            version = self._increment_version(latest.version)
            if previous_version is None:
                previous_version = latest.version
        else:
            version = "1.0.0"

        now = datetime.now(timezone.utc).isoformat()

        # Compute input hash (metrics + operator state)
        input_data = {
            "metrics": metrics,
            "operator_graph_state": operator_graph_state,
        }
        sha256_inputs = self._compute_sha256(input_data)

        # Build provenance
        provenance = self._build_provenance(casepack_id)

        # Compute output hash (everything except sha256_outputs itself)
        output_data = {
            "version": version,
            "scanner_id": scanner_id,
            "scanner_model": scanner_model,
            "date": now,
            "approved_by": approved_by,
            "service_event": service_event,
            "metrics": metrics,
            "operator_graph_state": operator_graph_state,
            "sha256_inputs": sha256_inputs,
            "provenance": provenance,
            "previous_version": previous_version,
        }
        sha256_outputs = self._compute_sha256(output_data)

        bundle = CommissioningBundle(
            version=version,
            scanner_id=scanner_id,
            scanner_model=scanner_model,
            date=now,
            approved_by=approved_by,
            service_event=service_event,
            metrics=metrics,
            operator_graph_state=operator_graph_state,
            sha256_inputs=sha256_inputs,
            sha256_outputs=sha256_outputs,
            provenance=provenance,
            previous_version=previous_version,
        )

        self._save(bundle)
        logger.info(
            "Created baseline v%s for scanner '%s' (event: %s).",
            version, scanner_id, service_event,
        )
        return bundle

    def get_active_baseline(self, scanner_id: str) -> CommissioningBundle | None:
        """Return the latest (most recent version) baseline for a scanner.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.

        Returns
        -------
        CommissioningBundle | None
            The latest baseline, or ``None`` if none exist.
        """
        baselines = self.list_baselines(scanner_id)
        return baselines[-1] if baselines else None

    def get_baseline(
        self, scanner_id: str, version: str,
    ) -> CommissioningBundle | None:
        """Retrieve a specific baseline version for a scanner.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.
        version : str
            Version string (e.g. ``"1.0.0"``).

        Returns
        -------
        CommissioningBundle | None
            The matching baseline, or ``None`` if not found.
        """
        _sanitize_scanner_id(scanner_id)
        scanner_dir = self.storage_path / scanner_id
        path = scanner_dir / f"v{version}.json"
        # Ensure resolved path stays within the expected directory
        if not path.resolve().is_relative_to(self.storage_path.resolve()):
            raise ValueError(
                f"Resolved path '{path.resolve()}' escapes storage directory."
            )
        if path.exists():
            return self._load(path)
        return None

    def list_baselines(self, scanner_id: str) -> list[CommissioningBundle]:
        """Return the full audit chain for a scanner, sorted by version.

        Parameters
        ----------
        scanner_id : str
            Scanner identifier.

        Returns
        -------
        list[CommissioningBundle]
            All baselines in version order (earliest first).
        """
        _sanitize_scanner_id(scanner_id)
        scanner_dir = self.storage_path / scanner_id
        if not scanner_dir.exists():
            return []

        bundles: list[CommissioningBundle] = []
        for path in sorted(scanner_dir.glob("v*.json")):
            try:
                bundles.append(self._load(path))
            except Exception:
                logger.warning(
                    "Failed to load baseline file '%s'; skipping.",
                    path, exc_info=True,
                )
        return bundles

    def compare_to_baseline(
        self,
        current_metrics: dict[str, float],
        baseline: CommissioningBundle,
    ) -> dict[str, BaselineComparison]:
        """Compare current QC measurements against a commissioning baseline.

        Parameters
        ----------
        current_metrics : dict[str, float]
            Today's measured values keyed by metric name.
        baseline : CommissioningBundle
            The reference commissioning bundle.

        Returns
        -------
        dict[str, BaselineComparison]
            Per-metric comparison results.
        """
        comparisons: dict[str, BaselineComparison] = {}

        # Unwrap MetricResult objects to floats if needed
        unwrapped: dict[str, float] = {}
        for k, v in current_metrics.items():
            if isinstance(v, (int, float)):
                unwrapped[k] = float(v)
            elif hasattr(v, "value"):
                unwrapped[k] = float(v.value)
            else:
                unwrapped[k] = float(v)
        current_metrics = unwrapped

        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline.metrics:
                logger.warning(
                    "Metric '%s' not found in baseline v%s; skipping.",
                    metric_name, baseline.version,
                )
                continue

            baseline_entry = baseline.metrics[metric_name]
            baseline_value = float(baseline_entry.get("value", 0.0))

            delta = current_value - baseline_value

            # Use a minimum denominator to prevent inflated percentages
            # for near-zero metrics (e.g., water CT number ~0 HU)
            denom = max(abs(baseline_value), 10.0)
            delta_percent = abs(delta / denom) * 100.0

            status: Literal["STABLE", "DRIFTED", "ALERT"]
            if delta_percent >= self._ALERT_THRESHOLD_PERCENT:
                status = "ALERT"
            elif delta_percent >= self._DRIFT_THRESHOLD_PERCENT:
                status = "DRIFTED"
            else:
                status = "STABLE"

            comparisons[metric_name] = BaselineComparison(
                metric_name=metric_name,
                baseline_value=baseline_value,
                current_value=current_value,
                delta=delta,
                delta_percent=delta_percent,
                status=status,
            )

        return comparisons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sha256(data: dict[str, Any]) -> str:
        """Compute SHA-256 hex digest of a deterministically serialised dict.

        Parameters
        ----------
        data : dict
            Arbitrary JSON-serialisable dictionary.

        Returns
        -------
        str
            64-character hex digest.
        """
        serialised = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def _save(self, bundle: CommissioningBundle) -> None:
        """Persist a commissioning bundle as a JSON file.

        Uses atomic write (temp file + ``os.replace``) to prevent
        data loss from concurrent writes or interrupted I/O.

        Parameters
        ----------
        bundle : CommissioningBundle
            The bundle to persist.
        """
        _sanitize_scanner_id(bundle.scanner_id)
        scanner_dir = self.storage_path / bundle.scanner_id
        scanner_dir.mkdir(parents=True, exist_ok=True)
        path = scanner_dir / f"v{bundle.version}.json"
        # Ensure resolved path stays within the expected directory
        if not path.resolve().is_relative_to(self.storage_path.resolve()):
            raise ValueError(
                f"Resolved path '{path.resolve()}' escapes storage directory."
            )

        content = json.dumps(bundle.model_dump(), indent=2, default=str)
        # Atomic write: write to temp file, then replace target
        fd, tmp_path = tempfile.mkstemp(
            dir=str(scanner_dir), suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(content)
            os.replace(tmp_path, str(path))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.debug("Saved baseline to '%s'.", path)

    @staticmethod
    def _load(path: Path) -> CommissioningBundle:
        """Load a commissioning bundle from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file.

        Returns
        -------
        CommissioningBundle
            Deserialised and validated bundle.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        return CommissioningBundle(**raw)

    @staticmethod
    def _increment_version(version: str) -> str:
        """Increment the patch component of a semantic version string.

        Parameters
        ----------
        version : str
            Current version (e.g. ``"1.2.3"``).

        Returns
        -------
        str
            Next version (e.g. ``"1.2.4"``).
        """
        parts = version.split(".")
        if len(parts) != 3:
            return f"{version}.1"
        major, minor, patch = parts
        return f"{major}.{minor}.{int(patch) + 1}"

    @staticmethod
    def _build_provenance(casepack_id: str) -> dict[str, Any]:
        """Build provenance metadata for a baseline bundle.

        Parameters
        ----------
        casepack_id : str
            CasePack identifier used for commissioning.

        Returns
        -------
        dict
            Provenance with ``pwm_version``, ``git_hash``, ``casepack``.
        """
        provenance: dict[str, Any] = {
            "casepack": casepack_id,
            "pwm_version": "unknown",
            "git_hash": "unknown",
        }

        try:
            import importlib.metadata
            provenance["pwm_version"] = importlib.metadata.version("pwm-core")
        except Exception:
            pass

        try:
            import subprocess
            if shutil.which("git") is not None:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    provenance["git_hash"] = result.stdout.strip()
            else:
                provenance["git_hash"] = "unavailable"
        except Exception:
            pass

        return provenance
