"""DICOM ingester for clinical CT QC.

PHI-safe validation, deterministic series selection based on CasePack rules,
canonical resampling for vendor-neutral ROI placement, and full selection logging.
Supports vendor-specific private tag extraction when specified in CasePack.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PHI-sensitive DICOM tag names that must not leak into outputs
# ---------------------------------------------------------------------------
_PHI_TAGS: set[str] = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "ReferringPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
    "OperatorsName",
    "PhysiciansOfRecord",
    "PerformingPhysicianName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "MedicalRecordLocator",
    "EthnicGroup",
    "Occupation",
    "AdditionalPatientHistory",
    "PatientComments",
}

# Patterns that indicate a phantom (not a real patient) for PHI gating
_PHANTOM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"phantom", re.IGNORECASE),
    re.compile(r"acr", re.IGNORECASE),
    re.compile(r"catphan", re.IGNORECASE),
    re.compile(r"qa[\s_-]?test", re.IGNORECASE),
    re.compile(r"qc[\s_-]?test", re.IGNORECASE),
    re.compile(r"daily[\s_-]?qa", re.IGNORECASE),
    re.compile(r"test[\s_-]?pattern", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SeriesInfo(BaseModel):
    """Summarises one DICOM series discovered in a directory scan."""

    model_config = ConfigDict(extra="forbid")

    series_uid: str
    series_description: str = ""
    num_images: int = 0
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[Tuple[float, float]] = None
    image_type: Optional[str] = None
    modality: str = "CT"
    rows: Optional[int] = None
    columns: Optional[int] = None


class SeriesSelectionLog(BaseModel):
    """Audit record produced each time a series is selected (or rejected)."""

    model_config = ConfigDict(extra="forbid")

    selected_series: str
    rule_name: str
    reason: str
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CTScanBundle(BaseModel):
    """Vendor-neutral representation of a single CT acquisition.

    The ``slices`` field holds a numpy ``ndarray`` (Z, Y, X) in Hounsfield
    Units.  ``arbitrary_types_allowed`` is enabled so numpy arrays can be
    stored without conversion.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    scanner_id: str
    study_date: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    protocol: Dict[str, Any] = Field(default_factory=dict)
    slices: Any  # np.ndarray placeholder -- shape (Z, Y, X), dtype float64
    pixel_spacing_mm: Tuple[float, float]
    slice_thickness_mm: float
    slice_locations: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    series_selection_log: List[SeriesSelectionLog] = Field(default_factory=list)
    phi_validation: Dict[str, Any] = Field(default_factory=dict)
    private_tags: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main ingester
# ---------------------------------------------------------------------------

class DICOMIngester:
    """PHI-safe DICOM ingester with CasePack-driven series selection.

    Parameters
    ----------
    phi_strict : bool
        When ``True`` (the default), the ingester will reject any study whose
        DICOM metadata does not match known phantom patterns.  Set to
        ``False`` only for development/testing with non-phantom data.
    """

    def __init__(self, phi_strict: bool = True) -> None:
        self.phi_strict = phi_strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        dicom_path: Path,
        casepack_config: Any = None,
    ) -> CTScanBundle:
        """Ingest a directory of DICOM files and return a :class:`CTScanBundle`.

        Parameters
        ----------
        dicom_path : Path
            Directory containing DICOM ``.dcm`` files (or extensionless DICOM).
        casepack_config : Any, optional
            Parsed CasePack configuration dict.  When provided, the
            ``series_selection.rules`` key drives deterministic series
            selection.  If *None*, the first axial CT series is chosen.

        Returns
        -------
        CTScanBundle
            Fully assembled volume with metadata, PHI validation record, and
            selection audit log.

        Raises
        ------
        FileNotFoundError
            If *dicom_path* does not exist or contains no DICOM files.
        ValueError
            If PHI validation fails under ``phi_strict`` mode.
        ImportError
            If ``pydicom`` or ``numpy`` are not installed.
        """
        try:
            import pydicom  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "pydicom is required for DICOM ingestion. "
                "Install it with: pip install pydicom"
            ) from exc

        try:
            import numpy as np  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "numpy is required for DICOM ingestion. "
                "Install it with: pip install numpy"
            ) from exc

        dicom_path = Path(dicom_path)
        if not dicom_path.is_dir():
            raise FileNotFoundError(
                f"DICOM path does not exist or is not a directory: {dicom_path}"
            )

        # Step 1 -- discover all series in the directory
        series_list = self._discover_series(dicom_path)
        if not series_list:
            raise FileNotFoundError(
                f"No DICOM files found in {dicom_path}"
            )
        logger.info("Discovered %d series in %s", len(series_list), dicom_path)

        # Step 2 -- select the target series using CasePack rules
        rules: list[dict] = []
        private_tag_rules: dict | None = None
        target_spacing: tuple[float, float, float] | None = None

        if casepack_config is not None:
            cp = (
                casepack_config.get("casepack", casepack_config)
                if isinstance(casepack_config, dict)
                else casepack_config
            )
            if isinstance(cp, dict):
                ss = cp.get("series_selection", {})
                rules = ss.get("rules", [])
                private_tag_rules = cp.get("private_tags", None)
                resample_cfg = cp.get("resampling", None)
                if isinstance(resample_cfg, dict):
                    target_spacing = tuple(resample_cfg.get("target_spacing", []))  # type: ignore[arg-type]

        datasets, selection_log = self._select_series(
            series_list, rules, dicom_path
        )
        logger.info(
            "Selected series %s via rule '%s'",
            selection_log.selected_series,
            selection_log.rule_name,
        )

        # Step 3 -- PHI validation
        phi_result = self._validate_phi(datasets)
        logger.info("PHI validation: %s", phi_result.get("status", "unknown"))

        # Step 4 -- extract volume in Hounsfield Units
        volume, slice_locations = self._extract_volume(datasets)
        logger.info("Extracted volume shape %s", volume.shape)

        # Step 5 -- extract private tags if configured
        private_tags = self._extract_private_tags(datasets[0], private_tag_rules)

        # Step 6 -- read common metadata from the first dataset
        ds0 = datasets[0]
        pixel_spacing = (
            float(ds0.PixelSpacing[0]),
            float(ds0.PixelSpacing[1]),
        ) if hasattr(ds0, "PixelSpacing") and ds0.PixelSpacing else (1.0, 1.0)
        slice_thickness = (
            float(ds0.SliceThickness)
            if hasattr(ds0, "SliceThickness") and ds0.SliceThickness
            else 1.0
        )

        # Step 7 -- canonical resampling (if target spacing configured)
        if target_spacing is not None and len(target_spacing) >= 2:
            volume = self._resample_canonical(
                volume,
                current_spacing=(pixel_spacing[0], pixel_spacing[1], slice_thickness),
                target_spacing=target_spacing,
            )
            # After resampling, update spacing to the target
            pixel_spacing = (target_spacing[0], target_spacing[1])
            if len(target_spacing) >= 3:
                slice_thickness = target_spacing[2]

        # Step 8 -- assemble protocol dict
        protocol: Dict[str, Any] = {}
        for tag_name in (
            "KVP", "XRayTubeCurrent", "ConvolutionKernel",
            "ReconstructionDiameter", "FilterType", "RotationDirection",
            "ExposureTime", "Exposure", "CTDIvol",
        ):
            if hasattr(ds0, tag_name):
                val = getattr(ds0, tag_name, None)
                if val is not None:
                    protocol[tag_name] = (
                        str(val) if not isinstance(val, (int, float)) else val
                    )

        # Step 9 -- build metadata dict (PHI-safe)
        metadata: Dict[str, Any] = {
            "num_slices": volume.shape[0],
            "rows": volume.shape[1],
            "columns": volume.shape[2],
            "bits_allocated": getattr(ds0, "BitsAllocated", None),
            "bits_stored": getattr(ds0, "BitsStored", None),
            "photometric_interpretation": getattr(
                ds0, "PhotometricInterpretation", None
            ),
        }

        scanner_id = getattr(ds0, "StationName", "unknown")
        study_date = getattr(ds0, "StudyDate", "unknown")
        manufacturer = getattr(ds0, "Manufacturer", None)
        model_name = getattr(ds0, "ManufacturerModelName", None)

        return CTScanBundle(
            scanner_id=str(scanner_id),
            study_date=str(study_date),
            manufacturer=str(manufacturer) if manufacturer else None,
            model=str(model_name) if model_name else None,
            protocol=protocol,
            slices=volume,
            pixel_spacing_mm=pixel_spacing,
            slice_thickness_mm=slice_thickness,
            slice_locations=slice_locations,
            metadata=metadata,
            series_selection_log=[selection_log],
            phi_validation=phi_result,
            private_tags=private_tags,
        )

    # ------------------------------------------------------------------
    # Series discovery
    # ------------------------------------------------------------------

    def _discover_series(self, dicom_path: Path) -> list[SeriesInfo]:
        """Scan *dicom_path* for DICOM files and group them by SeriesInstanceUID.

        Parameters
        ----------
        dicom_path : Path
            Root directory to scan (non-recursive by default; scans one level).

        Returns
        -------
        list[SeriesInfo]
            One entry per unique series, sorted by series description.
        """
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError(
                "pydicom is required. Install with: pip install pydicom"
            ) from exc

        series_map: Dict[str, Dict[str, Any]] = {}  # uid -> aggregated info

        # Scan both top-level and one level of subdirectories
        candidate_files: list[Path] = []
        for child in dicom_path.iterdir():
            if child.is_file():
                candidate_files.append(child)
            elif child.is_dir():
                for grandchild in child.iterdir():
                    if grandchild.is_file():
                        candidate_files.append(grandchild)

        for fpath in candidate_files:
            try:
                ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)
            except Exception:
                continue  # not a valid DICOM file

            uid = getattr(ds, "SeriesInstanceUID", None)
            if uid is None:
                continue
            uid = str(uid)

            if uid not in series_map:
                pixel_spacing = None
                if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
                    try:
                        pixel_spacing = (
                            float(ds.PixelSpacing[0]),
                            float(ds.PixelSpacing[1]),
                        )
                    except (IndexError, TypeError, ValueError):
                        pass

                image_type_raw = getattr(ds, "ImageType", None)
                image_type = (
                    "\\".join(image_type_raw)
                    if isinstance(image_type_raw, (list, tuple))
                    else str(image_type_raw) if image_type_raw else None
                )

                series_map[uid] = {
                    "series_uid": uid,
                    "series_description": str(
                        getattr(ds, "SeriesDescription", "")
                    ),
                    "num_images": 0,
                    "slice_thickness": (
                        float(ds.SliceThickness)
                        if hasattr(ds, "SliceThickness") and ds.SliceThickness
                        else None
                    ),
                    "pixel_spacing": pixel_spacing,
                    "image_type": image_type,
                    "modality": str(getattr(ds, "Modality", "CT")),
                    "rows": int(ds.Rows) if hasattr(ds, "Rows") else None,
                    "columns": int(ds.Columns) if hasattr(ds, "Columns") else None,
                }

            series_map[uid]["num_images"] += 1

        result = [SeriesInfo(**info) for info in series_map.values()]
        result.sort(key=lambda s: s.series_description)
        return result

    # ------------------------------------------------------------------
    # Series selection
    # ------------------------------------------------------------------

    def _select_series(
        self,
        series_list: list[SeriesInfo],
        rules: list[dict],
        dicom_path: Path,
    ) -> tuple[list[Any], SeriesSelectionLog]:
        """Apply CasePack series-selection rules to pick the target series.

        Selection is deterministic: rules are evaluated in order and the first
        matching series wins.  If no rule matches, the series with the most
        images is chosen as a fallback.

        Parameters
        ----------
        series_list : list[SeriesInfo]
            Discovered series from :meth:`_discover_series`.
        rules : list[dict]
            CasePack ``series_selection.rules`` entries.
        dicom_path : Path
            Root DICOM directory (needed to re-read full datasets).

        Returns
        -------
        tuple[list[pydicom.Dataset], SeriesSelectionLog]
            The DICOM datasets for the selected series, plus the audit log.
        """
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError(
                "pydicom is required. Install with: pip install pydicom"
            ) from exc

        selected_uid: str | None = None
        matched_rule_name = "default_fallback"
        matched_reason = "No CasePack rules provided or none matched; selected series with most images"

        alternatives: list[dict] = []

        for rule in rules:
            rule_name = rule.get("name", "unnamed_rule")
            match_criteria = rule.get("match", {})

            for si in series_list:
                if self._series_matches_rule(si, match_criteria):
                    selected_uid = si.series_uid
                    matched_rule_name = rule_name
                    matched_reason = (
                        f"Matched rule '{rule_name}': "
                        f"description='{si.series_description}', "
                        f"thickness={si.slice_thickness}, "
                        f"images={si.num_images}"
                    )
                    break
            if selected_uid is not None:
                break

        # Build alternatives list
        for si in series_list:
            if si.series_uid != selected_uid:
                alternatives.append({
                    "series_uid": si.series_uid,
                    "series_description": si.series_description,
                    "num_images": si.num_images,
                    "slice_thickness": si.slice_thickness,
                })

        # Fallback: pick series with most images
        if selected_uid is None:
            best = max(series_list, key=lambda s: s.num_images)
            selected_uid = best.series_uid
            matched_reason = (
                f"Fallback: selected series '{best.series_description}' "
                f"with {best.num_images} images (most images)"
            )

        # Load full datasets for the selected series
        datasets = self._load_series_datasets(dicom_path, selected_uid)

        log_entry = SeriesSelectionLog(
            selected_series=selected_uid,
            rule_name=matched_rule_name,
            reason=matched_reason,
            alternatives_considered=alternatives,
        )

        return datasets, log_entry

    def _series_matches_rule(
        self,
        series_info: SeriesInfo,
        match_criteria: dict,
    ) -> bool:
        """Evaluate whether *series_info* satisfies all *match_criteria*.

        Supported criteria keys (from CasePack YAML):
        - ``series_description_contains``: list of substrings (any must match)
        - ``slice_thickness_range``: [min, max] in mm
        - ``image_type_contains``: substring that must appear in ImageType
        - ``modality``: exact modality string (e.g. "CT")
        - ``min_images``: minimum number of images in the series

        Returns
        -------
        bool
            ``True`` if all specified criteria are satisfied.
        """
        # series_description_contains -- any substring matches
        desc_patterns: list[str] = match_criteria.get(
            "series_description_contains", []
        )
        if desc_patterns:
            desc_lower = series_info.series_description.lower()
            if not any(p.lower() in desc_lower for p in desc_patterns):
                return False

        # slice_thickness_range
        thickness_range: list[float] | None = match_criteria.get(
            "slice_thickness_range", None
        )
        if thickness_range is not None and len(thickness_range) == 2:
            if series_info.slice_thickness is None:
                return False
            if not (thickness_range[0] <= series_info.slice_thickness <= thickness_range[1]):
                return False

        # image_type_contains
        image_type_pattern: str | None = match_criteria.get(
            "image_type_contains", None
        )
        if image_type_pattern is not None:
            if series_info.image_type is None:
                return False
            if image_type_pattern.upper() not in series_info.image_type.upper():
                return False

        # modality
        modality_filter: str | None = match_criteria.get("modality", None)
        if modality_filter is not None:
            if series_info.modality.upper() != modality_filter.upper():
                return False

        # min_images
        min_images: int | None = match_criteria.get("min_images", None)
        if min_images is not None:
            if series_info.num_images < min_images:
                return False

        return True

    def _load_series_datasets(
        self,
        dicom_path: Path,
        series_uid: str,
    ) -> list[Any]:
        """Load full DICOM datasets (with pixel data) for a given series UID.

        Parameters
        ----------
        dicom_path : Path
            Root DICOM directory.
        series_uid : str
            Target SeriesInstanceUID.

        Returns
        -------
        list[pydicom.Dataset]
            All datasets belonging to the series, unsorted.
        """
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError(
                "pydicom is required. Install with: pip install pydicom"
            ) from exc

        datasets: list[Any] = []

        candidate_files: list[Path] = []
        for child in dicom_path.iterdir():
            if child.is_file():
                candidate_files.append(child)
            elif child.is_dir():
                for grandchild in child.iterdir():
                    if grandchild.is_file():
                        candidate_files.append(grandchild)

        for fpath in candidate_files:
            try:
                ds = pydicom.dcmread(str(fpath))
            except Exception:
                continue
            if str(getattr(ds, "SeriesInstanceUID", "")) == series_uid:
                datasets.append(ds)

        return datasets

    # ------------------------------------------------------------------
    # PHI validation
    # ------------------------------------------------------------------

    def _validate_phi(self, datasets: list[Any]) -> dict:
        """Run PHI-safety validation on the loaded DICOM datasets.

        The validation checks whether any PHI-sensitive tags contain
        values that appear to reference a real patient (vs. a phantom).
        When ``phi_strict`` is ``True``, non-phantom studies are rejected.

        Parameters
        ----------
        datasets : list[pydicom.Dataset]
            Datasets to validate.

        Returns
        -------
        dict
            Validation result with keys ``status``, ``is_phantom``,
            ``phi_tags_present``, and ``details``.

        Raises
        ------
        ValueError
            If ``phi_strict`` is ``True`` and the study does not appear
            to be a phantom scan.
        """
        if not datasets:
            return {"status": "no_data", "is_phantom": False, "phi_tags_present": [], "details": ""}

        ds0 = datasets[0]
        phi_tags_present: list[str] = []
        phi_values: Dict[str, str] = {}
        is_phantom = False

        # Collect PHI-bearing tag values
        for tag_name in _PHI_TAGS:
            val = getattr(ds0, tag_name, None)
            if val is not None and str(val).strip():
                phi_tags_present.append(tag_name)
                phi_values[tag_name] = str(val)

        # Determine if this looks like a phantom scan
        phantom_indicators: list[str] = []
        fields_to_check: list[str] = [
            str(getattr(ds0, "PatientName", "")),
            str(getattr(ds0, "PatientID", "")),
            str(getattr(ds0, "SeriesDescription", "")),
            str(getattr(ds0, "StudyDescription", "")),
            str(getattr(ds0, "ProtocolName", "")),
            str(getattr(ds0, "InstitutionalDepartmentName", "")),
        ]

        for field_val in fields_to_check:
            for pattern in _PHANTOM_PATTERNS:
                if pattern.search(field_val):
                    phantom_indicators.append(
                        f"'{field_val}' matched pattern '{pattern.pattern}'"
                    )

        is_phantom = len(phantom_indicators) > 0

        result = {
            "status": "pass" if is_phantom else "fail",
            "is_phantom": is_phantom,
            "phi_tags_present": phi_tags_present,
            "phantom_indicators": phantom_indicators,
            "details": (
                f"Found {len(phantom_indicators)} phantom indicator(s). "
                f"{len(phi_tags_present)} PHI tag(s) present in header."
            ),
        }

        if self.phi_strict and not is_phantom:
            raise ValueError(
                "PHI validation failed: study does not appear to be a phantom scan. "
                f"PHI tags present: {phi_tags_present}. "
                "Set phi_strict=False to override (development only)."
            )

        return result

    # ------------------------------------------------------------------
    # Volume extraction
    # ------------------------------------------------------------------

    def _extract_volume(
        self,
        datasets: list[Any],
    ) -> tuple[Any, list[float]]:
        """Sort DICOM slices and stack into a 3-D Hounsfield-Unit volume.

        Sorting priority:
        1. ``SliceLocation`` (if present and unique per slice)
        2. ``InstanceNumber`` (fallback)
        3. Original file order (last resort)

        Rescale is applied via ``RescaleSlope`` and ``RescaleIntercept``
        to convert stored pixel values to Hounsfield Units::

            HU = pixel_value * RescaleSlope + RescaleIntercept

        Parameters
        ----------
        datasets : list[pydicom.Dataset]
            DICOM datasets with pixel data.

        Returns
        -------
        tuple[np.ndarray, list[float]]
            - 3-D float64 array of shape (Z, Y, X) in Hounsfield Units.
            - Sorted list of slice locations.

        Raises
        ------
        ValueError
            If no pixel data can be extracted.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "numpy is required. Install with: pip install numpy"
            ) from exc

        if not datasets:
            raise ValueError("No DICOM datasets provided for volume extraction.")

        # Sort datasets
        datasets_with_loc: list[tuple[float, int, Any]] = []
        for idx, ds in enumerate(datasets):
            loc = getattr(ds, "SliceLocation", None)
            inst = getattr(ds, "InstanceNumber", idx)
            try:
                loc_f = float(loc) if loc is not None else None
            except (TypeError, ValueError):
                loc_f = None
            try:
                inst_i = int(inst) if inst is not None else idx
            except (TypeError, ValueError):
                inst_i = idx
            datasets_with_loc.append((loc_f if loc_f is not None else 0.0, inst_i, ds))

        # Prefer SliceLocation if values are diverse (i.e. not all the same)
        locations = [t[0] for t in datasets_with_loc]
        if len(set(locations)) > 1:
            datasets_with_loc.sort(key=lambda t: t[0])
        else:
            datasets_with_loc.sort(key=lambda t: t[1])

        sorted_datasets = [t[2] for t in datasets_with_loc]
        slice_locations_out = [t[0] for t in datasets_with_loc]

        # Stack pixel arrays and apply rescale
        slices: list[Any] = []
        for ds in sorted_datasets:
            try:
                pixel_array = ds.pixel_array.astype(np.float64)
            except Exception as exc:
                logger.warning(
                    "Could not extract pixel_array from dataset: %s", exc
                )
                continue

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu_slice = pixel_array * slope + intercept
            slices.append(hu_slice)

        if not slices:
            raise ValueError(
                "No pixel data could be extracted from the provided datasets."
            )

        volume = np.stack(slices, axis=0)
        return volume, slice_locations_out

    # ------------------------------------------------------------------
    # Private tag extraction
    # ------------------------------------------------------------------

    def _extract_private_tags(
        self,
        dataset: Any,
        private_tag_rules: dict | None,
    ) -> dict:
        """Extract vendor-specific private DICOM tags as specified by CasePack.

        Private tags are identified by their ``(group, element)`` pairs and
        an associated label.  For example, a CasePack might request::

            private_tags:
              siemens_ctdi:
                group: 0x7005
                element: 0x0010
                label: "Siemens CTDIw"

        Parameters
        ----------
        dataset : pydicom.Dataset
            A single representative DICOM dataset (typically the first slice).
        private_tag_rules : dict | None
            Mapping of ``{tag_key: {group, element, label}}`` from CasePack.
            If *None* or empty, returns an empty dict.

        Returns
        -------
        dict
            Mapping ``{label: value}`` for each successfully extracted tag.
        """
        if not private_tag_rules:
            return {}

        result: dict = {}
        for tag_key, tag_spec in private_tag_rules.items():
            group = tag_spec.get("group")
            element = tag_spec.get("element")
            label = tag_spec.get("label", tag_key)

            if group is None or element is None:
                logger.warning(
                    "Private tag '%s' missing group or element; skipping.",
                    tag_key,
                )
                continue

            try:
                # Support both int and hex-string notation
                if isinstance(group, str):
                    group = int(group, 16) if group.startswith("0x") else int(group)
                if isinstance(element, str):
                    element = int(element, 16) if element.startswith("0x") else int(element)

                tag_value = dataset[group, element].value
                result[label] = (
                    str(tag_value)
                    if not isinstance(tag_value, (int, float))
                    else tag_value
                )
            except (KeyError, IndexError, TypeError) as exc:
                logger.debug(
                    "Private tag '%s' (0x%04X,0x%04X) not found: %s",
                    tag_key,
                    group,
                    element,
                    exc,
                )

        return result

    # ------------------------------------------------------------------
    # Canonical resampling
    # ------------------------------------------------------------------

    def _resample_canonical(
        self,
        volume: Any,
        current_spacing: tuple[float, ...],
        target_spacing: tuple[float, ...] | None,
    ) -> Any:
        """Resample a volume to canonical isotropic (or specified) spacing.

        Uses trilinear interpolation via ``scipy.ndimage.zoom``.  If
        ``target_spacing`` is *None* or matches ``current_spacing``, the
        volume is returned unchanged.

        Parameters
        ----------
        volume : np.ndarray
            3-D array of shape (Z, Y, X).
        current_spacing : tuple[float, ...]
            Current voxel spacing in mm as (row_spacing, col_spacing, slice_spacing).
        target_spacing : tuple[float, ...] | None
            Desired voxel spacing in mm.  Must have at least 2 elements
            (row, col); a third element sets Z spacing.

        Returns
        -------
        np.ndarray
            Resampled volume.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "numpy is required. Install with: pip install numpy"
            ) from exc

        if target_spacing is None:
            return volume

        if len(target_spacing) < 2:
            logger.warning("target_spacing must have >= 2 elements; skipping resample.")
            return volume

        # Compute zoom factors: current / target  (bigger spacing -> shrink)
        zoom_row = current_spacing[0] / target_spacing[0] if target_spacing[0] > 0 else 1.0
        zoom_col = current_spacing[1] / target_spacing[1] if target_spacing[1] > 0 else 1.0
        zoom_z = 1.0
        if len(current_spacing) >= 3 and len(target_spacing) >= 3:
            zoom_z = (
                current_spacing[2] / target_spacing[2]
                if target_spacing[2] > 0
                else 1.0
            )

        zoom_factors = (zoom_z, zoom_row, zoom_col)

        # Check if resampling is actually needed
        if all(abs(z - 1.0) < 1e-6 for z in zoom_factors):
            logger.debug("Current spacing matches target; skipping resample.")
            return volume

        try:
            from scipy.ndimage import zoom as ndimage_zoom
        except ImportError:
            logger.warning(
                "scipy is not installed; canonical resampling skipped. "
                "Install with: pip install scipy"
            )
            return volume

        logger.info(
            "Resampling volume with zoom factors (Z, Y, X) = %s",
            zoom_factors,
        )
        resampled = ndimage_zoom(volume, zoom_factors, order=1)  # trilinear
        return resampled
