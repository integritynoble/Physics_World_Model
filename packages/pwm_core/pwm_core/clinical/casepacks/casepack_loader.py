"""CasePack loader and validator.

CasePacks are versioned, immutable QC workflow packages that define ROIs,
metrics, thresholds, and report templates for each phantom/test combination.

Each CasePack is stored as a YAML file with a well-defined schema. The loader
discovers CasePacks from configured search paths, validates their structure,
and returns strongly-typed ``CasePackConfig`` objects.

Usage
-----
>>> from pwm_core.clinical.casepacks.casepack_loader import CasePackLoader
>>> loader = CasePackLoader()
>>> config = loader.load("acr_ct_annual")
>>> config.phantom_type
'ACR CT'
>>> loader.list_available()
['acr_ct_annual', ...]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# Default search paths (relative to this module's location)
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_SEARCH_PATHS: list[Path] = [
    _MODULE_DIR,                          # casepacks/ directory itself
    _MODULE_DIR.parent / "contrib",       # clinical/contrib/
    _MODULE_DIR.parents[2] / "contrib",   # packages/pwm_core/contrib/ (project-level)
]

# Known metric names for validation
_KNOWN_METRICS: set[str] = {
    "noise_std",
    "uniformity",
    "ct_number_accuracy",
    "ct_number_water",
    "ct_number_air",
    "ct_number_acrylic",
    "ct_number_bone",
    "ct_number_polyethylene",
    "high_contrast_resolution",
    "low_contrast_detectability",
    "slice_thickness",
    "cnr",
    "snr",
    "mtf_50",
    "mtf_10",
    "artifact_index",
    "artifact_evaluation",
    "geometric_accuracy",
    "patient_positioning_accuracy",
    "dose_ctdi_vol",
    "dose_dlp",
    "uniformity_integral",
    "suv_accuracy",
    "spatial_resolution",
    "sensitivity",
    "scatter_fraction",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SeriesSelectionRule(BaseModel):
    """Rule for selecting a DICOM series from a study.

    Attributes
    ----------
    name : str
        Human-readable name for this selection rule (e.g.,
        ``"axial_standard"``).
    match : dict
        Dictionary of DICOM tag names to expected values or patterns.
        Example: ``{"SeriesDescription": "*AXIAL*", "SliceThickness": 5.0}``
    fallback : str
        Name of a fallback rule to try if this rule matches no series.
        Empty string if no fallback.
    """

    name: str
    match: dict[str, Any] = Field(default_factory=dict)
    fallback: str = ""


class ROIDefinition(BaseModel):
    """Definition of a region of interest for QC measurement.

    Flexible schema supporting various ROI geometries: circular ROIs
    at fixed positions, auto-centered ROIs, peripheral ROIs at angular
    offsets, etc.

    Attributes
    ----------
    shape : str
        ROI shape (e.g., ``"circle"``, ``"rectangle"``, ``"annulus"``).
    center_method : str | None
        How to find the ROI center: ``"phantom_center"``, ``"centroid"``,
        ``"fixed"``, or ``None`` for position-list ROIs.
    radius_mm : float | None
        Radius in mm for circular ROIs.
    offset_angle_deg : float | None
        Angular offset from 12-o'clock in degrees for peripheral ROIs.
    offset_radius_mm : float | None
        Radial offset from center in mm for peripheral ROIs.
    expected_hu : float | None
        Expected CT number in HU for this ROI material.
    slice_selection : str | None
        How to select the measurement slice: ``"center"``, ``"thickest"``,
        ``"module_N"`` (ACR phantom module number), etc.
    positions : list[str] | None
        Named positions for multi-ROI measurements
        (e.g., ``["center", "12", "3", "6", "9"]`` for ACR uniformity).
    count : int | None
        Number of ROIs to place (for uniformity or sampling patterns).
    offset_from_center_mm : float | None
        Distance from phantom center in mm (alternative to angle+radius).
    """

    shape: str
    center_method: str | None = None
    radius_mm: float | None = None
    offset_angle_deg: float | None = None
    offset_radius_mm: float | None = None
    expected_hu: float | None = None
    slice_selection: str | None = None
    positions: list[str] | None = None
    count: int | None = None
    offset_from_center_mm: float | None = None


class _SeriesSelectionConfig(BaseModel):
    """Internal model for the series_selection block in a CasePack."""

    rules: list[SeriesSelectionRule] = Field(default_factory=list)
    log_selection: bool = True


class CasePackConfig(BaseModel):
    """Complete configuration for a clinical QA CasePack.

    A CasePack defines everything needed to execute a QC workflow:
    which series to select, where to place ROIs, which metrics to compute,
    what thresholds to apply, and how to format the report.

    Attributes
    ----------
    id : str
        Unique CasePack identifier (e.g., ``"acr_ct_annual"``).
    name : str
        Human-readable name (e.g., ``"ACR CT Annual QC"``).
    version : str
        Semantic version of this CasePack.
    min_pwm_version : str
        Minimum PWM version required to run this CasePack.
    author : str
        Author or organization that created this CasePack.
    phantom_type : str
        Type of phantom this CasePack is designed for
        (e.g., ``"ACR CT"``, ``"Catphan 604"``).
    series_selection : dict[str, Any]
        Series selection configuration with ``rules`` list and
        ``log_selection`` flag.
    roi_definitions : dict[str, Any]
        Named ROI definitions. Keys are measurement names, values are
        ROI configurations (can be parsed into :class:`ROIDefinition`).
    metric_set : list[str]
        Ordered list of metric names to compute.
    threshold_set : str
        Name or path of the threshold YAML file to use.
    report_template : str
        Name or path of the report template.
    evidence_artifacts : list[str]
        List of evidence artifact types to generate (e.g.,
        ``["roi_overlay_png", "histogram_csv", "trend_json"]``).
    """

    id: str
    name: str
    version: str
    min_pwm_version: str = "0.4.0"
    author: str
    phantom_type: str
    series_selection: dict[str, Any] = Field(default_factory=dict)
    roi_definitions: dict[str, Any] = Field(default_factory=dict)
    metric_set: list[str] = Field(default_factory=list)
    threshold_set: str = ""
    report_template: str = ""
    evidence_artifacts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CasePack loader
# ---------------------------------------------------------------------------

class CasePackLoader:
    """Discovers, loads, and validates CasePack YAML definitions.

    Parameters
    ----------
    search_paths : list[Path] | None
        Directories to search for CasePack YAML files. If ``None``,
        the default paths (``casepacks/`` and ``contrib/``) are used.

    Examples
    --------
    >>> loader = CasePackLoader()
    >>> ids = loader.list_available()
    >>> if "acr_ct_annual" in ids:
    ...     config = loader.load("acr_ct_annual")
    """

    def __init__(self, search_paths: list[Path] | None = None) -> None:
        """Initialize the loader with search paths for CasePack discovery."""
        if search_paths is not None:
            self._search_paths = [Path(p) for p in search_paths]
        else:
            self._search_paths = list(_DEFAULT_SEARCH_PATHS)

        # Cache: casepack_id -> file path
        self._index: dict[str, Path] = {}
        self._build_index()

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load a YAML file and return its contents as a dict.

        Parameters
        ----------
        path : Path
            Path to the YAML file.

        Returns
        -------
        dict[str, Any]
            Parsed YAML content.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        FileNotFoundError
            If the YAML file does not exist.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for CasePack loading. "
                "Install it with: pip install pyyaml"
            ) from exc

        if not path.exists():
            raise FileNotFoundError(f"CasePack YAML not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"CasePack YAML must be a mapping, got {type(data).__name__}: {path}"
            )
        return data

    def _build_index(self) -> None:
        """Scan search paths and index available CasePacks by ID."""
        self._index.clear()

        for search_dir in self._search_paths:
            if not search_dir.is_dir():
                logger.debug("CasePack search path does not exist: %s", search_dir)
                continue

            for yaml_path in sorted(search_dir.glob("*.yaml")):
                try:
                    data = self._load_yaml(yaml_path)
                except Exception:  # noqa: BLE001 — index build is best-effort
                    logger.debug(
                        "Skipping non-parseable YAML in CasePack search: %s",
                        yaml_path,
                        exc_info=True,
                    )
                    continue

                casepack_id = data.get("id") or data.get("casepack", {}).get("id")
                if casepack_id and isinstance(casepack_id, str):
                    if casepack_id in self._index:
                        logger.warning(
                            "Duplicate CasePack ID '%s': %s shadows %s",
                            casepack_id,
                            yaml_path,
                            self._index[casepack_id],
                        )
                    self._index[casepack_id] = yaml_path

            # Also search one level of subdirectories
            for subdir in sorted(search_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for yaml_path in sorted(subdir.glob("*.yaml")):
                    try:
                        data = self._load_yaml(yaml_path)
                    except Exception:  # noqa: BLE001
                        continue
                    casepack_id = data.get("id") or data.get("casepack", {}).get("id")
                    if casepack_id and isinstance(casepack_id, str):
                        if casepack_id not in self._index:
                            self._index[casepack_id] = yaml_path

    def load(self, casepack_id: str) -> CasePackConfig:
        """Load and validate a CasePack by its unique ID.

        Parameters
        ----------
        casepack_id : str
            The unique identifier of the CasePack to load.

        Returns
        -------
        CasePackConfig
            The validated CasePack configuration.

        Raises
        ------
        KeyError
            If no CasePack with the given ID is found in any search path.
        ValueError
            If the YAML file is invalid or fails validation.
        """
        if casepack_id not in self._index:
            # Re-index in case new files were added at runtime
            self._build_index()

        if casepack_id not in self._index:
            available = ", ".join(sorted(self._index.keys())) or "(none)"
            raise KeyError(
                f"CasePack '{casepack_id}' not found. "
                f"Available CasePacks: {available}"
            )

        yaml_path = self._index[casepack_id]
        data = self._load_yaml(yaml_path)
        data = data.get("casepack", data)

        try:
            config = CasePackConfig(**data)
        except Exception as exc:
            raise ValueError(
                f"CasePack '{casepack_id}' failed validation ({yaml_path}): {exc}"
            ) from exc

        # Run semantic validation and log warnings
        warnings = self.validate(config)
        for warning in warnings:
            logger.warning("CasePack '%s': %s", casepack_id, warning)

        return config

    def validate(self, config: CasePackConfig) -> list[str]:
        """Validate a CasePack configuration and return warnings.

        Checks for common issues such as unknown metric names, missing
        ROI definitions, empty series selection rules, etc. This does NOT
        raise exceptions -- it returns a list of human-readable warnings.

        Parameters
        ----------
        config : CasePackConfig
            The CasePack configuration to validate.

        Returns
        -------
        list[str]
            List of validation warning messages. Empty if no issues found.
        """
        warnings: list[str] = []

        # Check for empty required fields
        if not config.id:
            warnings.append("CasePack 'id' is empty.")
        if not config.name:
            warnings.append("CasePack 'name' is empty.")
        if not config.version:
            warnings.append("CasePack 'version' is empty.")
        if not config.author:
            warnings.append("CasePack 'author' is empty.")
        if not config.phantom_type:
            warnings.append("CasePack 'phantom_type' is empty.")

        # Check metric_set for unknown metrics
        unknown_metrics = set(config.metric_set) - _KNOWN_METRICS
        if unknown_metrics:
            warnings.append(
                f"Unknown metric(s): {sorted(unknown_metrics)}. "
                f"These may be custom metrics -- ensure they have implementations."
            )

        # Check that metric_set is not empty
        if not config.metric_set:
            warnings.append("'metric_set' is empty -- no metrics will be computed.")

        # Check series_selection has rules
        series_sel = config.series_selection
        if not series_sel:
            warnings.append("'series_selection' is empty.")
        elif "rules" in series_sel:
            rules = series_sel["rules"]
            if isinstance(rules, list) and len(rules) == 0:
                warnings.append("'series_selection.rules' is empty.")

        # Check roi_definitions is not empty
        if not config.roi_definitions:
            warnings.append(
                "'roi_definitions' is empty -- no ROIs will be placed."
            )

        # Check threshold_set and report_template are specified
        if not config.threshold_set:
            warnings.append("'threshold_set' is not specified.")
        if not config.report_template:
            warnings.append("'report_template' is not specified.")

        # Check evidence_artifacts
        if not config.evidence_artifacts:
            warnings.append(
                "'evidence_artifacts' is empty -- no evidence will be saved."
            )

        return warnings

    def list_available(self) -> list[str]:
        """Return all available CasePack IDs from the search paths.

        Returns
        -------
        list[str]
            Sorted list of CasePack IDs discovered in the search paths.
        """
        # Re-index to pick up any newly added files
        self._build_index()
        return sorted(self._index.keys())

    def refresh(self) -> None:
        """Re-scan search paths and rebuild the CasePack index.

        Call this after adding or removing YAML files from the search paths.
        """
        self._build_index()
