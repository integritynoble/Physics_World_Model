"""pwm_core.cards.instrument_card

InstrumentCard — describes a physical instrument (scanner, sensor, imager).

An InstrumentCard captures the hardware context of a physics experiment:
manufacturer, model, acquisition parameters, and calibration provenance.
This allows reproducibility auditors to flag measurement gaps attributable
to instrument-level variation (Triad Gate G3 — operator/instrument noise).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pwm_core.cards.utils import new_card_id, now_iso


class InstrumentCard(BaseModel):
    """Describes a physical instrument used in a physics experiment.

    Fields
    ------
    instrument_id:
        Unique identifier, e.g. "ct_scanner_siemens_somatom_001".
    card_version:
        Schema version (semver string).
    instrument_type:
        Category: "ct_scanner" | "mri_scanner" | "optical_imager" |
        "spectrometer" | "ultrasound_probe" | "lidar" | "other".
    manufacturer:
        Instrument manufacturer name.
    model:
        Model name / number.
    serial_number:
        Serial number (optional — may be anonymised).
    acquisition_params:
        Key acquisition parameters as free-form dict.
        Examples:
          CT:   {"kVp": 120, "mAs": 200, "slice_thickness_mm": 1.0}
          MRI:  {"field_strength_T": 3.0, "TR_ms": 500, "TE_ms": 10}
          CASSI: {"integration_time_ms": 50, "pixel_pitch_um": 6.5}
    calibration_date:
        ISO-8601 date of last calibration.
    calibration_method:
        Free-text description of calibration method / reference.
    known_artefacts:
        List of known measurement artefacts (ring artefacts, field inhomogeneity, etc.).
    linked_spec_ids:
        SpecCard IDs that reference this instrument.
    linked_dataset_ids:
        DatasetCard IDs collected with this instrument.
    extra:
        Arbitrary extra metadata.
    created_at:
        ISO-8601 creation timestamp.
    """

    instrument_id: str = Field(default_factory=lambda: f"instrument_{new_card_id()}")
    card_version: str = "1.0.0"
    instrument_type: str = "other"
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    acquisition_params: Dict[str, Any] = Field(default_factory=dict)
    calibration_date: Optional[str] = None
    calibration_method: Optional[str] = None
    known_artefacts: List[str] = Field(default_factory=list)
    linked_spec_ids: List[str] = Field(default_factory=list)
    linked_dataset_ids: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
