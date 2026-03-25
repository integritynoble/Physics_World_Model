"""pwm_core.connectors.dicom

DICOM and HL7/FHIR connectors for loading hospital imaging data into PWM.

Examples
--------
Load a single DICOM file::

    from pwm_core.connectors.dicom import DicomConnector

    dc = DicomConnector()
    result = dc.load("/path/to/scan.dcm")
    volume    = result["volume"]      # np.ndarray float32, [0,1]
    metadata  = result["metadata"]   # dict of DICOM tags
    card      = result["dataset_card"]

Load a directory of DICOMs (stacked into a 3-D volume)::

    result = dc.load("/path/to/dicom_series/")

Strip PHI before storing metadata::

    clean_meta = dc.strip_phi(result["metadata"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

# Guard pydicom import so that the module can be imported without pydicom
# installed (a clear error is raised only when DicomConnector.load() is
# actually called).
try:
    import pydicom
    from pydicom.errors import InvalidDicomError

    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False

from pwm_core.cards.dataset_card import DatasetCard

# ---------------------------------------------------------------------------
# PHI tags that must be redacted
# ---------------------------------------------------------------------------

_PHI_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "InstitutionName",
    "ReferringPhysicianName",
    "AccessionNumber",
    "StudyID",
)

# Map DICOM Modality tag values to PWM modality strings
_MODALITY_MAP: dict[str, str] = {
    "CT": "ct",
    "MR": "mri",
    "MRI": "mri",
    "PT": "pet",
    "NM": "spect",
    "US": "ultrasound",
    "OPT": "oct",
    "OP": "fundus",
    "CR": "xray",
    "DX": "xray",
    "MG": "mammography",
    "XA": "angiography",
}


# ---------------------------------------------------------------------------
# DicomConnector
# ---------------------------------------------------------------------------


class DicomConnector:
    """Load DICOM files or series into PWM-compatible NumPy arrays.

    All pixel data is converted to float32 and normalised to [0, 1].
    PHI can be stripped from metadata before further processing.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Union[str, Path]) -> dict:
        """Load a DICOM file or a directory of DICOMs.

        Parameters
        ----------
        path:
            Path to a single ``.dcm`` file **or** a directory containing
            multiple DICOM slices (they will be stacked in instance-number
            order into a 3-D volume).

        Returns
        -------
        dict with keys:
            ``"volume"``      – float32 NumPy array, normalised to [0, 1].
            ``"metadata"``    – dict of selected DICOM tag values.
            ``"dataset_card"``– :class:`~pwm_core.cards.dataset_card.DatasetCard`.
        """
        _require_pydicom()

        target = Path(path)

        if target.is_dir():
            volume, metadata = self._load_series(target)
        elif target.is_file():
            volume, metadata = self._load_single(target)
        else:
            raise FileNotFoundError(f"DICOM path not found: {target}")

        volume = self._normalize(volume)
        card = self._emit_dataset_card(volume, metadata, str(target))

        return {"volume": volume, "metadata": metadata, "dataset_card": card}

    def strip_phi(self, metadata: dict) -> dict:
        """Remove standard PHI tags from a metadata dict.

        Each identified PHI field is replaced with the string
        ``"REDACTED"`` rather than being deleted, to preserve dict
        structure for downstream consumers.

        Parameters
        ----------
        metadata:
            Dict as returned by :meth:`load` (``result["metadata"]``).

        Returns
        -------
        dict
            A *new* dict with PHI fields redacted.
        """
        cleaned = dict(metadata)
        for tag in _PHI_TAGS:
            if tag in cleaned:
                cleaned[tag] = "REDACTED"
        return cleaned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_single(self, path: Path) -> tuple[np.ndarray, dict]:
        """Load pixel data and metadata from a single DICOM file."""
        ds = pydicom.dcmread(str(path), force=True)
        pixel_array = ds.pixel_array.astype(np.float32)
        metadata = self._extract_metadata(ds, str(path))
        return pixel_array, metadata

    def _load_series(self, directory: Path) -> tuple[np.ndarray, dict]:
        """Stack all DICOM slices in *directory* into a 3-D volume.

        Slices are sorted by InstanceNumber when available, falling back
        to filename sort order.
        """
        dcm_files = sorted(
            [f for f in directory.iterdir() if f.is_file()],
            key=lambda f: f.name,
        )
        if not dcm_files:
            raise ValueError(f"No files found in DICOM directory: {directory}")

        datasets = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), force=True)
                if hasattr(ds, "pixel_array"):
                    datasets.append(ds)
            except Exception:
                # Skip non-DICOM files silently
                continue

        if not datasets:
            raise ValueError(
                f"No readable DICOM slices with pixel data found in: {directory}"
            )

        # Sort by InstanceNumber (tag 0020,0013) when present
        def _instance_key(ds: "pydicom.Dataset") -> int:
            try:
                return int(ds.InstanceNumber)
            except Exception:
                return 0

        datasets.sort(key=_instance_key)

        slices = [ds.pixel_array.astype(np.float32) for ds in datasets]
        volume = np.stack(slices, axis=0)  # shape: (n_slices, H, W)

        # Use metadata from the first slice as representative
        metadata = self._extract_metadata(datasets[0], str(directory))
        metadata["n_slices"] = len(datasets)

        return volume, metadata

    def _extract_metadata(self, ds: "pydicom.Dataset", source_path: str) -> dict:
        """Extract a flat dict of useful DICOM tag values from a dataset."""
        _str = lambda tag: str(getattr(ds, tag, "")) if hasattr(ds, tag) else ""

        meta: dict = {
            # Identity / study
            "PatientName": _str("PatientName"),
            "PatientID": _str("PatientID"),
            "PatientBirthDate": _str("PatientBirthDate"),
            "PatientSex": _str("PatientSex"),
            "StudyID": _str("StudyID"),
            "AccessionNumber": _str("AccessionNumber"),
            # Institution
            "InstitutionName": _str("InstitutionName"),
            "ReferringPhysicianName": _str("ReferringPhysicianName"),
            # Acquisition
            "Modality": _str("Modality"),
            "StudyDate": _str("StudyDate"),
            "SeriesDescription": _str("SeriesDescription"),
            "Manufacturer": _str("Manufacturer"),
            "ManufacturerModelName": _str("ManufacturerModelName"),
            "KVP": _str("KVP"),
            "SliceThickness": _str("SliceThickness"),
            "PixelSpacing": str(getattr(ds, "PixelSpacing", "")),
            "Rows": int(getattr(ds, "Rows", 0)),
            "Columns": int(getattr(ds, "Columns", 0)),
            "BitsAllocated": int(getattr(ds, "BitsAllocated", 0)),
            # Source
            "source_path": source_path,
        }
        return meta

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalise *arr* to [0, 1] float32.

        If the array is constant (max == min), returns a zero array of the
        same shape to avoid division by zero.
        """
        arr = arr.astype(np.float32)
        lo = float(arr.min())
        hi = float(arr.max())
        if hi == lo:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    def _emit_dataset_card(
        self,
        volume: np.ndarray,
        metadata: dict,
        source_path: str,
    ) -> DatasetCard:
        """Create a DatasetCard from loaded DICOM data.

        The PWM modality is inferred from the DICOM ``Modality`` tag when
        possible, falling back to ``"generic"``.
        """
        dicom_modality = metadata.get("Modality", "").upper()
        pwm_modality = _MODALITY_MAP.get(dicom_modality, "generic")

        n_samples = volume.shape[0] if volume.ndim == 3 else 1
        description = (
            f"DICOM data loaded from {source_path}. "
            f"DICOM modality: {dicom_modality or 'unknown'}. "
            f"Volume shape: {volume.shape}."
        )

        return DatasetCard(
            dataset_id=f"{pwm_modality}_dicom_imported",
            modality=pwm_modality,
            version="1.0.0",
            split="public",
            n_samples=n_samples,
            source="real",
            license="proprietary",
            description=description,
            tags=["dicom", "imported", pwm_modality],
        )


# ---------------------------------------------------------------------------
# HL7FHIRConnector (stub — planned for P3)
# ---------------------------------------------------------------------------


class HL7FHIRConnector:
    """Stub connector for HL7/FHIR imaging endpoints.

    Full implementation is planned for P3 using the ``fhirclient`` library.

    Parameters
    ----------
    base_url:
        Base URL of the FHIR server (e.g. ``https://fhir.example.org/R4``).
    auth_token:
        Bearer token for authenticated FHIR servers.
    """

    _NOT_IMPLEMENTED_MSG = (
        "HL7/FHIR connector is planned for P3 — implement with fhirclient library"
    )

    def __init__(self, base_url: str, auth_token: str = "") -> None:
        self.base_url = base_url
        self.auth_token = auth_token

    def get_imaging_study(self, study_id: str) -> dict:
        """Retrieve an ImagingStudy resource by ID.

        .. note::
            This is a stub.  Full implementation is planned for P3.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)

    def list_patients(self, **filters) -> list:
        """List patients matching the given FHIR search filters.

        .. note::
            This is a stub.  Full implementation is planned for P3.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(self._NOT_IMPLEMENTED_MSG)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _require_pydicom() -> None:
    """Raise a clear ImportError if pydicom is not available."""
    if not _PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required for DicomConnector. "
            "Install with: pip install pydicom"
        )
