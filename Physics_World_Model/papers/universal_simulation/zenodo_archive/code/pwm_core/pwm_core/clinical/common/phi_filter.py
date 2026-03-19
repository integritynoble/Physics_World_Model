"""PHI safety filter for clinical DICOM ingestion.

Enforces phantom-only studies by default. Provides opt-in de-identification
hooks for future real-world clinical use.

Usage
-----
>>> from pwm_core.clinical.common.phi_filter import validate_phantom_study, is_phantom_safe
>>> result = validate_phantom_study(ds)
>>> if not is_phantom_safe(ds):
...     raise RuntimeError("Non-phantom study rejected by PHI filter")
"""

from __future__ import annotations

import copy
import re
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# DICOM tag tuples to remove during de-identification (PS3.15 Annex E subset)
# Format: (group, element) pairs
# ---------------------------------------------------------------------------
PHI_TAGS_TO_REMOVE: list[tuple[int, int]] = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0032),  # PatientBirthTime
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1000),  # OtherPatientIDs
    (0x0010, 0x1001),  # OtherPatientNames
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1020),  # PatientSize
    (0x0010, 0x1030),  # PatientWeight
    (0x0010, 0x1040),  # PatientAddress
    (0x0010, 0x2154),  # PatientTelephoneNumbers
    (0x0010, 0x2160),  # EthnicGroup
    (0x0010, 0x21F0),  # PatientReligiousPreference
    (0x0010, 0x4000),  # PatientComments
    (0x0008, 0x0050),  # AccessionNumber
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x0092),  # ReferringPhysicianAddress
    (0x0008, 0x0094),  # ReferringPhysicianTelephoneNumbers
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0008, 0x1049),  # PhysiciansOfRecordIdentificationSequence
    (0x0008, 0x1050),  # PerformingPhysicianName
    (0x0008, 0x1060),  # NameOfPhysiciansReadingStudy
    (0x0008, 0x1070),  # OperatorsName
    (0x0020, 0x0010),  # StudyID
    (0x0040, 0xA123),  # PersonName
    (0x0032, 0x1032),  # RequestingPhysician
    (0x0032, 0x1033),  # RequestingService
]

# ---------------------------------------------------------------------------
# Clinically relevant tags to RETAIN during de-identification
# These are essential for QC analysis and safe to keep
# ---------------------------------------------------------------------------
CLINICAL_TAGS_TO_RETAIN: list[tuple[int, int]] = [
    (0x0018, 0x0060),  # KVP
    (0x0018, 0x1151),  # XRayTubeCurrent (mA)
    (0x0018, 0x1150),  # ExposureTime
    (0x0018, 0x0050),  # SliceThickness
    (0x0018, 0x0088),  # SpacingBetweenSlices
    (0x0018, 0x1100),  # ReconstructionDiameter
    (0x0018, 0x1160),  # FilterType
    (0x0018, 0x1210),  # ConvolutionKernel
    (0x0018, 0x5100),  # PatientPosition
    (0x0018, 0x0015),  # BodyPartExamined
    (0x0018, 0x0022),  # ScanOptions
    (0x0018, 0x0090),  # DataCollectionDiameter
    (0x0018, 0x1120),  # GantryDetectorTilt
    (0x0018, 0x1130),  # TableHeight
    (0x0018, 0x1140),  # RotationDirection
    (0x0018, 0x9306),  # SingleCollimationWidth
    (0x0018, 0x9307),  # TotalCollimationWidth
    (0x0018, 0x9311),  # SpiralPitchFactor
    (0x0028, 0x0010),  # Rows
    (0x0028, 0x0011),  # Columns
    (0x0028, 0x0030),  # PixelSpacing
    (0x0028, 0x1050),  # WindowCenter
    (0x0028, 0x1051),  # WindowWidth
    (0x0028, 0x1052),  # RescaleIntercept
    (0x0028, 0x1053),  # RescaleSlope
    (0x0020, 0x0032),  # ImagePositionPatient
    (0x0020, 0x0037),  # ImageOrientationPatient
    (0x0020, 0x1041),  # SliceLocation
    (0x0008, 0x0008),  # ImageType
    (0x0008, 0x0060),  # Modality
    (0x0008, 0x0070),  # Manufacturer
    (0x0008, 0x1090),  # ManufacturerModelName
    (0x0008, 0x1010),  # StationName
    (0x0018, 0x1000),  # DeviceSerialNumber
    (0x0018, 0x1020),  # SoftwareVersions
    (0x0018, 0x0010),  # ContrastBolusAgent
    (0x0018, 0x1152),  # Exposure
    (0x0018, 0x115E),  # ImageAndFluoroscopyAreaDoseProduct
    (0x0018, 0x9345),  # CTDIvol
]

# ---------------------------------------------------------------------------
# Phantom indicator patterns: tag_name -> list of regex patterns
# ---------------------------------------------------------------------------
PHANTOM_INDICATORS: dict[str, list[str]] = {
    "PatientName": [
        r"(?i)\bphantom\b",
        r"(?i)\bqc\b",
        r"(?i)\btest\b",
        r"(?i)\bacr\b",
        r"(?i)\bcatphan\b",
        r"(?i)\buniformity\b",
        r"(?i)\bcalibration\b",
        r"(?i)\bdaily\s*qa\b",
    ],
    "PatientID": [
        r"(?i)\bphantom\b",
        r"(?i)\bqc\b",
        r"(?i)\btest\b",
        r"(?i)^QA[\-_]?\d*$",
        r"(?i)^PH[\-_]?\d+$",
    ],
    "InstitutionName": [
        r"(?i)\bservice\b",
        r"(?i)\bengineering\b",
    ],
    "StudyDescription": [
        r"(?i)\bphantom\b",
        r"(?i)\bqc\b",
        r"(?i)\bqa\b",
        r"(?i)\bcalibration\b",
        r"(?i)\bdaily\b",
        r"(?i)\bacr\b",
        r"(?i)\btest\b",
    ],
    "SeriesDescription": [
        r"(?i)\bphantom\b",
        r"(?i)\bqc\b",
        r"(?i)\bacr\b",
    ],
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PHIValidationResult(BaseModel):
    """Result of phantom study validation.

    Attributes
    ----------
    is_phantom : bool
        Whether the study is identified as a phantom scan.
    confidence : float
        Confidence score between 0.0 and 1.0, based on the fraction of
        indicator checks that matched.
    flags : list[str]
        Human-readable descriptions of matched phantom indicators.
    raw_checks : dict[str, Any]
        Detailed per-tag match results for auditing.
    """

    is_phantom: bool
    confidence: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)
    raw_checks: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _get_tag_value(dataset: Any, tag_name: str) -> str | None:
    """Safely retrieve a string value from a DICOM dataset by attribute name.

    Parameters
    ----------
    dataset : pydicom.Dataset
        The DICOM dataset to query.
    tag_name : str
        The attribute name (e.g., ``"PatientName"``).

    Returns
    -------
    str | None
        The string representation of the tag value, or ``None`` if not present.
    """
    if hasattr(dataset, tag_name):
        val = getattr(dataset, tag_name, None)
        if val is not None:
            return str(val)
    return None


def validate_phantom_study(dataset: Any) -> PHIValidationResult:
    """Validate whether a DICOM dataset represents a phantom/QC study.

    Checks ``PatientName``, ``PatientID``, ``InstitutionName``,
    ``StudyDescription``, and ``SeriesDescription`` against known phantom
    indicator patterns. Computes a confidence score based on the fraction
    of tag categories with at least one match.

    Parameters
    ----------
    dataset : pydicom.Dataset
        A pydicom ``Dataset`` object (or any object with the expected
        DICOM attribute names).

    Returns
    -------
    PHIValidationResult
        Validation result with confidence score and matched indicators.

    Examples
    --------
    >>> import pydicom
    >>> ds = pydicom.Dataset()
    >>> ds.PatientName = "ACR Phantom"
    >>> ds.PatientID = "QA-001"
    >>> result = validate_phantom_study(ds)
    >>> result.is_phantom
    True
    >>> result.confidence > 0.5
    True
    """
    try:
        import pydicom as _pydicom  # noqa: F401 — validate availability
    except ImportError:
        pass  # pydicom is optional; we work with duck-typed dataset objects

    flags: list[str] = []
    raw_checks: dict[str, Any] = {}
    tags_checked = 0
    tags_matched = 0

    for tag_name, patterns in PHANTOM_INDICATORS.items():
        tags_checked += 1
        value = _get_tag_value(dataset, tag_name)
        tag_result: dict[str, Any] = {
            "value": value,
            "matched_patterns": [],
            "matched": False,
        }

        if value is not None:
            for pattern in patterns:
                if re.search(pattern, value):
                    tag_result["matched_patterns"].append(pattern)
                    tag_result["matched"] = True

        if tag_result["matched"]:
            tags_matched += 1
            flags.append(
                f"{tag_name}='{value}' matched phantom pattern(s): "
                f"{tag_result['matched_patterns']}"
            )

        raw_checks[tag_name] = tag_result

    # Confidence: fraction of checked tag categories that matched
    confidence = tags_matched / tags_checked if tags_checked > 0 else 0.0

    # Consider it a phantom if at least one indicator matched
    is_phantom = tags_matched >= 1

    return PHIValidationResult(
        is_phantom=is_phantom,
        confidence=round(confidence, 4),
        flags=flags,
        raw_checks=raw_checks,
    )


def is_phantom_safe(dataset: Any, min_confidence: float = 0.5) -> bool:
    """Convenience check: is this dataset a phantom study above a confidence threshold?

    Parameters
    ----------
    dataset : pydicom.Dataset
        The DICOM dataset to validate.
    min_confidence : float
        Minimum confidence score required for the study to be considered
        phantom-safe. Default is ``0.5`` (at least half the indicator
        categories must match).

    Returns
    -------
    bool
        ``True`` if the dataset is identified as a phantom study with
        confidence >= *min_confidence*.
    """
    result = validate_phantom_study(dataset)
    return result.is_phantom and result.confidence >= min_confidence


def deidentify_dataset(
    dataset: Any,
    retain_clinical_tags: bool = True,
) -> Any:
    """Remove PHI tags from a DICOM dataset per PS3.15 Annex E.

    Creates a deep copy of the dataset and removes all tags listed in
    :data:`PHI_TAGS_TO_REMOVE`. If *retain_clinical_tags* is ``True``
    (default), tags in :data:`CLINICAL_TAGS_TO_RETAIN` are preserved even
    if they overlap with removal lists.

    This function is **opt-in** and intended for future real-world clinical
    use. For current phantom-only workflows, de-identification is not
    required but may be applied as defense-in-depth.

    Parameters
    ----------
    dataset : pydicom.Dataset
        The original DICOM dataset. Not modified in place.
    retain_clinical_tags : bool
        If ``True``, clinically relevant acquisition parameters (kVp, mA,
        SliceThickness, etc.) are preserved. Default ``True``.

    Returns
    -------
    pydicom.Dataset
        A deep copy of *dataset* with PHI tags removed.

    Raises
    ------
    ImportError
        If ``pydicom`` is not installed.

    Notes
    -----
    This implementation covers a practical subset of DICOM PS3.15 Annex E
    Basic Profile. A full clinical deployment should use a validated
    de-identification library or service.
    """
    try:
        import pydicom  # noqa: F811
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for DICOM de-identification. "
            "Install it with: pip install pydicom"
        ) from exc

    # Deep copy to avoid mutating the original dataset
    ds = copy.deepcopy(dataset)

    # Build the retain set for fast lookup
    retain_set: set[tuple[int, int]] = set()
    if retain_clinical_tags:
        retain_set = set(CLINICAL_TAGS_TO_RETAIN)

    for tag_tuple in PHI_TAGS_TO_REMOVE:
        if tag_tuple in retain_set:
            continue
        tag = pydicom.tag.Tag(*tag_tuple)
        if tag in ds:
            del ds[tag]

    # Replace PatientName with anonymized value if it was removed
    patient_name_tag = pydicom.tag.Tag(0x0010, 0x0010)
    if patient_name_tag not in ds:
        ds.PatientName = "DEIDENTIFIED"

    # Replace PatientID with anonymized value if it was removed
    patient_id_tag = pydicom.tag.Tag(0x0010, 0x0020)
    if patient_id_tag not in ds:
        ds.PatientID = "DEIDENTIFIED"

    return ds
