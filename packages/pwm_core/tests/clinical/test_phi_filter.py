"""Tests for the PHI safety filter module.

Validates phantom detection logic (``validate_phantom_study``,
``is_phantom_safe``) and module-level constants using duck-typed dataset
objects (SimpleNamespace) so that pydicom is NOT required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pwm_core.clinical.common.phi_filter import (
    CLINICAL_TAGS_TO_RETAIN,
    PHI_TAGS_TO_REMOVE,
    PHANTOM_INDICATORS,
    PHIValidationResult,
    is_phantom_safe,
    validate_phantom_study,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(**kwargs) -> SimpleNamespace:
    """Create a lightweight duck-typed DICOM dataset from keyword args."""
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# 1. Phantom detected by PatientName
# ---------------------------------------------------------------------------

class TestPhantomDetectedByPatientName:
    """validate_phantom_study should flag datasets whose PatientName
    contains known phantom indicator words (e.g. 'ACR Phantom')."""

    @pytest.mark.parametrize("name", [
        "ACR Phantom",
        "Daily QA Phantom",
        "CatPhan 604",
        "QC Slab",
        "Test Object",
        "Uniformity Phantom",
        "Calibration Rod",
    ])
    def test_phantom_detected_by_patient_name(self, name: str):
        ds = _make_dataset(PatientName=name)
        result = validate_phantom_study(ds)

        assert result.is_phantom is True
        assert result.confidence > 0.0
        assert len(result.flags) >= 1
        # At least one flag should reference PatientName
        assert any("PatientName" in f for f in result.flags)


# ---------------------------------------------------------------------------
# 2. Phantom detected by PatientID
# ---------------------------------------------------------------------------

class TestPhantomDetectedByPatientID:
    """validate_phantom_study should flag datasets whose PatientID matches
    known phantom ID patterns (e.g. 'QA-001', 'PH-42')."""

    @pytest.mark.parametrize("pid", [
        "QA-001",
        "QA001",
        "QA_99",
        "PH-42",
        "PH_1",
        "phantom-scan",
        "qc scan",
        "test scan",
    ])
    def test_phantom_detected_by_patient_id(self, pid: str):
        ds = _make_dataset(PatientID=pid)
        result = validate_phantom_study(ds)

        assert result.is_phantom is True
        assert result.confidence > 0.0
        assert any("PatientID" in f for f in result.flags)


# ---------------------------------------------------------------------------
# 3. Phantom detected by StudyDescription
# ---------------------------------------------------------------------------

class TestPhantomDetectedByStudyDescription:
    """validate_phantom_study should flag datasets whose StudyDescription
    matches phantom indicator words."""

    @pytest.mark.parametrize("desc", [
        "Daily QC Phantom",
        "ACR CT Accreditation",
        "Calibration Run",
        "QA Check",
        "Daily Test",
    ])
    def test_phantom_detected_by_study_description(self, desc: str):
        ds = _make_dataset(StudyDescription=desc)
        result = validate_phantom_study(ds)

        assert result.is_phantom is True
        assert result.confidence > 0.0
        assert any("StudyDescription" in f for f in result.flags)


# ---------------------------------------------------------------------------
# 4. Non-phantom study is correctly rejected
# ---------------------------------------------------------------------------

class TestNonPhantomStudyRejected:
    """A dataset with realistic patient information that does NOT contain
    any phantom indicator words should NOT be classified as a phantom."""

    def test_non_phantom_study_rejected(self):
        ds = _make_dataset(
            PatientName="John Doe",
            PatientID="MRN-123456",
            InstitutionName="City General Hospital",
            StudyDescription="CT Abdomen with Contrast",
            SeriesDescription="Axial 5mm",
        )
        result = validate_phantom_study(ds)

        assert result.is_phantom is False
        assert result.confidence == 0.0
        assert result.flags == []

    def test_non_phantom_another_patient(self):
        ds = _make_dataset(
            PatientName="Jane Smith",
            PatientID="PAT-789",
            StudyDescription="Brain MRI",
        )
        result = validate_phantom_study(ds)

        assert result.is_phantom is False
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 5. Confidence scales with the number of matched tag categories
# ---------------------------------------------------------------------------

class TestConfidenceScaling:
    """Confidence should increase as more tag categories match phantom
    indicators. The score equals tags_matched / tags_checked."""

    def test_confidence_scales_with_matched_tags(self):
        # Only PatientName matches -> 1/5 = 0.2
        ds_one = _make_dataset(PatientName="ACR Phantom")
        result_one = validate_phantom_study(ds_one)

        # PatientName + PatientID + StudyDescription -> 3/5 = 0.6
        ds_three = _make_dataset(
            PatientName="ACR Phantom",
            PatientID="QA-001",
            StudyDescription="Daily QC Phantom",
        )
        result_three = validate_phantom_study(ds_three)

        # All five tag categories match -> 5/5 = 1.0
        ds_all = _make_dataset(
            PatientName="ACR Phantom",
            PatientID="QA-001",
            InstitutionName="Service Engineering",
            StudyDescription="Daily QC Phantom",
            SeriesDescription="QC ACR",
        )
        result_all = validate_phantom_study(ds_all)

        assert result_one.confidence < result_three.confidence
        assert result_three.confidence < result_all.confidence
        assert result_all.confidence == 1.0

    def test_two_tags_higher_than_one(self):
        ds_one = _make_dataset(PatientName="Phantom")
        ds_two = _make_dataset(PatientName="Phantom", PatientID="phantom-id")

        r1 = validate_phantom_study(ds_one)
        r2 = validate_phantom_study(ds_two)

        assert r2.confidence > r1.confidence


# ---------------------------------------------------------------------------
# 6. is_phantom_safe returns True when confidence is high
# ---------------------------------------------------------------------------

class TestIsPhantomSafeHighConfidence:
    """is_phantom_safe should return True when the phantom confidence is
    at or above the default threshold (0.5)."""

    def test_is_phantom_safe_true_when_high_confidence(self):
        # 3/5 matched categories -> confidence 0.6 >= 0.5
        ds = _make_dataset(
            PatientName="ACR Phantom",
            PatientID="QA-001",
            StudyDescription="Daily QC",
        )
        assert is_phantom_safe(ds) is True

    def test_all_tags_matched(self):
        ds = _make_dataset(
            PatientName="ACR Phantom",
            PatientID="QA-001",
            InstitutionName="Service Engineering",
            StudyDescription="Daily QC Phantom",
            SeriesDescription="QC ACR",
        )
        assert is_phantom_safe(ds) is True


# ---------------------------------------------------------------------------
# 7. is_phantom_safe returns False when confidence is low
# ---------------------------------------------------------------------------

class TestIsPhantomSafeLowConfidence:
    """is_phantom_safe should return False when only a single weak match
    occurs, giving confidence below the default 0.5 threshold."""

    def test_is_phantom_safe_false_when_low_confidence(self):
        # Only PatientName matches -> confidence = 1/5 = 0.2 < 0.5
        ds = _make_dataset(PatientName="Test Object")
        assert is_phantom_safe(ds) is False

    def test_single_patient_id_match(self):
        # Only PatientID matches -> confidence = 1/5 = 0.2 < 0.5
        ds = _make_dataset(PatientID="QA-001")
        assert is_phantom_safe(ds) is False


# ---------------------------------------------------------------------------
# 8. is_phantom_safe with custom min_confidence
# ---------------------------------------------------------------------------

class TestIsPhantomSafeCustomThreshold:
    """is_phantom_safe should respect a custom ``min_confidence`` parameter."""

    def test_is_phantom_safe_with_custom_min_confidence_low_threshold(self):
        # 1/5 = 0.2, passes with threshold 0.1
        ds = _make_dataset(PatientName="Phantom Scan")
        assert is_phantom_safe(ds, min_confidence=0.1) is True

    def test_is_phantom_safe_with_custom_min_confidence_high_threshold(self):
        # 3/5 = 0.6, fails with threshold 0.8
        ds = _make_dataset(
            PatientName="ACR Phantom",
            PatientID="QA-001",
            StudyDescription="Daily QC",
        )
        assert is_phantom_safe(ds, min_confidence=0.8) is False

    def test_exact_threshold_boundary(self):
        # 1/5 = 0.2, passes when min_confidence is exactly 0.2
        ds = _make_dataset(PatientName="Phantom")
        result = validate_phantom_study(ds)
        assert is_phantom_safe(ds, min_confidence=result.confidence) is True

    def test_just_above_threshold_fails(self):
        # 1/5 = 0.2, fails when min_confidence is 0.21
        ds = _make_dataset(PatientName="Phantom")
        assert is_phantom_safe(ds, min_confidence=0.21) is False


# ---------------------------------------------------------------------------
# 9. Empty dataset is not classified as phantom
# ---------------------------------------------------------------------------

class TestEmptyDataset:
    """An empty dataset (no DICOM attributes) should not be classified as
    a phantom, since no indicator patterns can match."""

    def test_empty_dataset_not_phantom(self):
        ds = _make_dataset()  # No attributes at all
        result = validate_phantom_study(ds)

        assert result.is_phantom is False
        assert result.confidence == 0.0
        assert result.flags == []

    def test_empty_dataset_is_phantom_safe_false(self):
        ds = _make_dataset()
        assert is_phantom_safe(ds) is False

    def test_dataset_with_none_values(self):
        ds = _make_dataset(
            PatientName=None,
            PatientID=None,
            StudyDescription=None,
        )
        result = validate_phantom_study(ds)

        assert result.is_phantom is False
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 10. Case-insensitive phantom detection
# ---------------------------------------------------------------------------

class TestCaseInsensitiveDetection:
    """Phantom indicator matching should be case-insensitive. Patterns
    in PHANTOM_INDICATORS all use the (?i) flag."""

    @pytest.mark.parametrize("name", [
        "acr phantom",
        "ACR PHANTOM",
        "Acr Phantom",
        "aCr PhAnToM",
        "CATPHAN 504",
        "catphan 504",
        "CatPhan 504",
    ])
    def test_case_insensitive_phantom_detection(self, name: str):
        ds = _make_dataset(PatientName=name)
        result = validate_phantom_study(ds)

        assert result.is_phantom is True
        assert any("PatientName" in f for f in result.flags)

    def test_case_insensitive_patient_id(self):
        ds_lower = _make_dataset(PatientID="qa-001")
        ds_upper = _make_dataset(PatientID="QA-001")

        r_lower = validate_phantom_study(ds_lower)
        r_upper = validate_phantom_study(ds_upper)

        assert r_lower.is_phantom is True
        assert r_upper.is_phantom is True

    def test_case_insensitive_study_description(self):
        ds = _make_dataset(StudyDescription="DAILY QC PHANTOM")
        result = validate_phantom_study(ds)

        assert result.is_phantom is True
        assert any("StudyDescription" in f for f in result.flags)


# ---------------------------------------------------------------------------
# 11. PHI_TAGS_TO_REMOVE list is non-empty
# ---------------------------------------------------------------------------

class TestPHITagsToRemove:
    """PHI_TAGS_TO_REMOVE should be a non-empty list of (group, element)
    tuples containing well-known DICOM PHI tags."""

    def test_phi_tags_to_remove_list_nonempty(self):
        assert isinstance(PHI_TAGS_TO_REMOVE, list)
        assert len(PHI_TAGS_TO_REMOVE) > 0

    def test_phi_tags_are_tuples_of_two_ints(self):
        for tag in PHI_TAGS_TO_REMOVE:
            assert isinstance(tag, tuple)
            assert len(tag) == 2
            assert isinstance(tag[0], int)
            assert isinstance(tag[1], int)

    def test_patient_name_in_phi_tags(self):
        # PatientName (0010, 0010) must be in the removal list
        assert (0x0010, 0x0010) in PHI_TAGS_TO_REMOVE

    def test_patient_id_in_phi_tags(self):
        # PatientID (0010, 0020) must be in the removal list
        assert (0x0010, 0x0020) in PHI_TAGS_TO_REMOVE


# ---------------------------------------------------------------------------
# 12. CLINICAL_TAGS_TO_RETAIN list is non-empty
# ---------------------------------------------------------------------------

class TestClinicalTagsToRetain:
    """CLINICAL_TAGS_TO_RETAIN should be a non-empty list of (group, element)
    tuples representing clinically relevant acquisition parameters."""

    def test_clinical_tags_to_retain_list_nonempty(self):
        assert isinstance(CLINICAL_TAGS_TO_RETAIN, list)
        assert len(CLINICAL_TAGS_TO_RETAIN) > 0

    def test_clinical_tags_are_tuples_of_two_ints(self):
        for tag in CLINICAL_TAGS_TO_RETAIN:
            assert isinstance(tag, tuple)
            assert len(tag) == 2
            assert isinstance(tag[0], int)
            assert isinstance(tag[1], int)

    def test_kvp_in_clinical_tags(self):
        # KVP (0018, 0060) is a critical acquisition parameter
        assert (0x0018, 0x0060) in CLINICAL_TAGS_TO_RETAIN

    def test_slice_thickness_in_clinical_tags(self):
        # SliceThickness (0018, 0050)
        assert (0x0018, 0x0050) in CLINICAL_TAGS_TO_RETAIN

    def test_pixel_spacing_in_clinical_tags(self):
        # PixelSpacing (0028, 0030)
        assert (0x0028, 0x0030) in CLINICAL_TAGS_TO_RETAIN


# ---------------------------------------------------------------------------
# Additional edge-case and structural tests
# ---------------------------------------------------------------------------

class TestPHIValidationResultModel:
    """Verify the PHIValidationResult Pydantic model behaves correctly."""

    def test_model_creation(self):
        result = PHIValidationResult(
            is_phantom=True,
            confidence=0.8,
            flags=["PatientName matched"],
            raw_checks={"PatientName": {"matched": True}},
        )
        assert result.is_phantom is True
        assert result.confidence == 0.8
        assert len(result.flags) == 1

    def test_model_defaults(self):
        result = PHIValidationResult(is_phantom=False, confidence=0.0)
        assert result.flags == []
        assert result.raw_checks == {}

    def test_confidence_bounds_low(self):
        """Confidence must be >= 0.0."""
        with pytest.raises(Exception):
            PHIValidationResult(is_phantom=False, confidence=-0.1)

    def test_confidence_bounds_high(self):
        """Confidence must be <= 1.0."""
        with pytest.raises(Exception):
            PHIValidationResult(is_phantom=True, confidence=1.1)


class TestPhantomIndicatorsStructure:
    """PHANTOM_INDICATORS dict should be well-formed and reference the
    expected DICOM tag names."""

    def test_phantom_indicators_nonempty(self):
        assert isinstance(PHANTOM_INDICATORS, dict)
        assert len(PHANTOM_INDICATORS) > 0

    def test_expected_tag_names_present(self):
        expected_tags = {"PatientName", "PatientID", "StudyDescription"}
        assert expected_tags.issubset(set(PHANTOM_INDICATORS.keys()))

    def test_each_tag_has_at_least_one_pattern(self):
        for tag_name, patterns in PHANTOM_INDICATORS.items():
            assert isinstance(patterns, list), f"{tag_name} patterns not a list"
            assert len(patterns) >= 1, f"{tag_name} has no patterns"


class TestRawChecksContent:
    """validate_phantom_study should populate raw_checks with per-tag
    match details for auditing."""

    def test_raw_checks_contains_all_indicator_tags(self):
        ds = _make_dataset(PatientName="ACR Phantom")
        result = validate_phantom_study(ds)

        for tag_name in PHANTOM_INDICATORS:
            assert tag_name in result.raw_checks

    def test_raw_checks_matched_tag_has_patterns(self):
        ds = _make_dataset(PatientName="ACR Phantom")
        result = validate_phantom_study(ds)

        patient_check = result.raw_checks["PatientName"]
        assert patient_check["matched"] is True
        assert len(patient_check["matched_patterns"]) >= 1
        assert patient_check["value"] == "ACR Phantom"

    def test_raw_checks_unmatched_tag_empty_patterns(self):
        ds = _make_dataset(PatientName="John Doe")
        result = validate_phantom_study(ds)

        patient_check = result.raw_checks["PatientName"]
        assert patient_check["matched"] is False
        assert patient_check["matched_patterns"] == []

    def test_raw_checks_missing_attribute_has_none_value(self):
        ds = _make_dataset()  # no attributes
        result = validate_phantom_study(ds)

        for tag_name in PHANTOM_INDICATORS:
            assert result.raw_checks[tag_name]["value"] is None
            assert result.raw_checks[tag_name]["matched"] is False
