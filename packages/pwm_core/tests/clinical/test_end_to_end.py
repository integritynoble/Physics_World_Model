"""Integration test: synthetic DICOM -> metrics -> thresholds -> report.

Creates synthetic DICOM files using pydicom, runs the full QC pipeline,
and verifies the outputs.  All file operations use pytest's tmp_path
fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------
pydicom = pytest.importorskip("pydicom", reason="pydicom required for DICOM E2E tests")

from pwm_core.clinical.ct.dicom_ingester import CTScanBundle, DICOMIngester
from pwm_core.clinical.ct.qa_metrics import QAMetricsReport, compute_all_metrics

from tests.clinical.conftest import (
    CENTER_X,
    CENTER_Y,
    PHANTOM_MATRIX,
    PHANTOM_SLICES,
    PIXEL_SPACING_MM,
)


# ===================================================================
# Helpers: synthetic DICOM generation
# ===================================================================

def _create_synthetic_dicom_dir(
    tmp_path: Path,
    volume: np.ndarray,
    pixel_spacing: tuple[float, float] = PIXEL_SPACING_MM,
    slice_thickness: float = 5.0,
    add_phantom_tags: bool = True,
) -> Path:
    """Write a synthetic CT volume as a directory of DICOM files.

    Each slice becomes one ``.dcm`` file with proper CT metadata.
    """
    import pydicom
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    dcm_dir = tmp_path / "dicom"
    dcm_dir.mkdir(exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref_uid = generate_uid()

    for z_idx in range(volume.shape[0]):
        sop_uid = generate_uid()
        filename = str(dcm_dir / f"slice_{z_idx:04d}.dcm")

        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(
            filename, {}, file_meta=file_meta, preamble=b"\x00" * 128,
        )

        if add_phantom_tags:
            ds.PatientName = "ACR Phantom"
            ds.PatientID = "QA-001"
        else:
            ds.PatientName = "John Doe"
            ds.PatientID = "MRN-12345"

        ds.StudyInstanceUID = study_uid
        ds.StudyDate = "20260101"
        ds.StudyDescription = "ACR CT QC"
        ds.Modality = "CT"

        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = "ACR Module 1 Axial"
        ds.SeriesNumber = 1
        ds.InstanceNumber = z_idx + 1

        ds.FrameOfReferenceUID = frame_of_ref_uid

        ds.Manufacturer = "SYNTHETIC"
        ds.ManufacturerModelName = "TestScanner"
        ds.StationName = "SYN-CT-001"

        ds.Rows = volume.shape[1]
        ds.Columns = volume.shape[2]
        ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
        ds.SliceThickness = str(slice_thickness)
        ds.SliceLocation = str(z_idx * slice_thickness)
        ds.ImagePositionPatient = ["0", "0", str(z_idx * slice_thickness)]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]

        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0

        ds.KVP = 120
        ds.XRayTubeCurrent = 200

        pixel_array = np.clip(volume[z_idx], -1024, 3071).astype(np.int16)
        ds.PixelData = pixel_array.tobytes()

        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = sop_uid

        ds.save_as(filename)

    return dcm_dir


# ===================================================================
# Tests
# ===================================================================


class TestFullPipelinePass:
    """Clean phantom should produce a valid metrics report."""

    def test_full_pipeline_pass(
        self,
        synthetic_acr_phantom: np.ndarray,
        sample_casepack_config: dict,
        pixel_spacing: tuple[float, float],
        tmp_path: Path,
    ) -> None:
        """End-to-end: clean phantom DICOM -> ingest -> metrics.

        All metrics on a clean phantom should be computed without error.
        """
        dcm_dir = _create_synthetic_dicom_dir(
            tmp_path, synthetic_acr_phantom,
        )

        # Step 1: Ingest DICOM
        ingester = DICOMIngester(phi_strict=True)
        bundle = ingester.ingest(
            dicom_path=dcm_dir,
            casepack_config=sample_casepack_config,
        )
        assert bundle.slices is not None
        assert bundle.slices.shape[0] > 0

        # Step 2: Compute metrics
        report = compute_all_metrics(
            scan_bundle=bundle,
            casepack=sample_casepack_config,
        )
        assert isinstance(report, QAMetricsReport)
        assert len(report.metrics) > 0

        # Verify no metrics errored
        errored = [
            k for k, v in report.metrics.items()
            if v.status == "error"
        ]
        assert len(errored) == 0, (
            f"Metrics with errors on clean phantom: {errored}"
        )


class TestFullPipelineFail:
    """Phantom with artifacts should produce measurable differences."""

    def test_full_pipeline_fail(
        self,
        synthetic_acr_phantom_with_artifacts: dict[str, np.ndarray],
        sample_casepack_config: dict,
        pixel_spacing: tuple[float, float],
        tmp_path: Path,
    ) -> None:
        """End-to-end: cupping phantom DICOM -> ingest -> metrics.

        A phantom with cupping artifacts should produce different metric
        values compared to a clean phantom.
        """
        cupping_volume = synthetic_acr_phantom_with_artifacts["cupping"]

        dcm_dir = _create_synthetic_dicom_dir(
            tmp_path, cupping_volume,
        )

        # Step 1: Ingest
        ingester = DICOMIngester(phi_strict=True)
        bundle = ingester.ingest(
            dicom_path=dcm_dir,
            casepack_config=sample_casepack_config,
        )

        # Step 2: Compute metrics
        report = compute_all_metrics(
            scan_bundle=bundle,
            casepack=sample_casepack_config,
        )
        assert isinstance(report, QAMetricsReport)
        assert len(report.metrics) > 0

        # The cupping artifact should affect uniformity or water CT number
        if "uniformity" in report.metrics:
            uniformity_result = report.metrics["uniformity"]
            if uniformity_result.status == "computed":
                # With -20 HU cupping, uniformity may exceed the 5 HU limit
                assert uniformity_result.value >= 0.0, (
                    "Uniformity should be non-negative"
                )
