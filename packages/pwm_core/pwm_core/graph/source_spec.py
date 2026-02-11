"""pwm_core.graph.source_spec
==============================

Source illumination specification for the universal forward model.

The universal rule is: y ~ Noise(Sensor(Transport/Interaction(Source(x))))

This module defines the Source(...) block: what carrier is emitted, how
strong, its spectral and spatial profile, coherence properties, constraints
(dose/damage), and expected exposure budget.

All models inherit from StrictBaseModel (extra="forbid", NaN/Inf rejection).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pwm_core.graph.ir_types import CarrierType, DriftModel


# ---------------------------------------------------------------------------
# StrictBaseModel (local copy for self-containment)
# ---------------------------------------------------------------------------


class StrictBaseModel(BaseModel):
    """Root model with extra='forbid' and NaN/Inf rejection."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        ser_json_inf_nan="constants",
    )

    @model_validator(mode="after")
    def _reject_nan_inf(self) -> "StrictBaseModel":
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(
                    f"Field '{field_name}' contains {val!r}, which is not allowed."
                )
        return self


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


class SpectrumKind(str, Enum):
    """Kind of source spectrum."""

    monochromatic = "monochromatic"
    broadband = "broadband"
    comb = "comb"
    custom = "custom"


class SpectrumSpec(StrictBaseModel):
    """Spectral characteristics of the source.

    Attributes
    ----------
    kind : SpectrumKind
        Type of spectrum (monochromatic, broadband, comb, custom).
    center_nm : float
        Central wavelength in nanometers.
    bandwidth_nm : float
        Full-width half-max spectral bandwidth in nanometers.
    n_lines : int
        Number of spectral lines (for comb sources).
    custom_wavelengths_nm : list[float]
        Custom wavelength list for ``kind="custom"``.
    custom_weights : list[float]
        Corresponding weights for custom wavelengths.
    """

    kind: SpectrumKind = SpectrumKind.monochromatic
    center_nm: float = 500.0
    bandwidth_nm: float = 0.0
    n_lines: int = 1
    custom_wavelengths_nm: List[float] = Field(default_factory=list)
    custom_weights: List[float] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------


class CoherenceSpec(StrictBaseModel):
    """Spatial and temporal coherence properties.

    Attributes
    ----------
    spatial_coherence : float
        Spatial coherence degree [0, 1]. 1 = fully coherent.
    temporal_coherence : float
        Temporal coherence degree [0, 1]. 1 = fully coherent.
    coherence_length_m : float
        Coherence length in meters.
    """

    spatial_coherence: float = 0.0
    temporal_coherence: float = 0.0
    coherence_length_m: float = 0.0


# ---------------------------------------------------------------------------
# Spatial profile
# ---------------------------------------------------------------------------


class SpatialProfileKind(str, Enum):
    """Kind of illumination spatial profile."""

    uniform = "uniform"
    gaussian = "gaussian"
    structured = "structured"
    point = "point"


class SpatialProfile(StrictBaseModel):
    """Spatial illumination profile.

    Attributes
    ----------
    kind : SpatialProfileKind
        Profile type.
    beam_waist_m : float
        Gaussian beam waist in meters (for ``kind="gaussian"``).
    pattern_id : str
        Reference to a pattern specification (for ``kind="structured"``).
    """

    kind: SpatialProfileKind = SpatialProfileKind.uniform
    beam_waist_m: float = 0.0
    pattern_id: str = ""


# ---------------------------------------------------------------------------
# Source constraints
# ---------------------------------------------------------------------------


class SourceConstraints(StrictBaseModel):
    """Physical constraints on the source illumination.

    Attributes
    ----------
    max_intensity_w_per_m2 : float
        Maximum intensity in W/m^2 (0 = unlimited).
    max_dose_j : float
        Maximum total dose in joules (0 = unlimited).
    max_exposure_s : float
        Maximum single-frame exposure time in seconds (0 = unlimited).
    damage_threshold_j_per_cm2 : float
        Sample damage threshold in J/cm^2 (0 = not applicable).
    """

    max_intensity_w_per_m2: float = 0.0
    max_dose_j: float = 0.0
    max_exposure_s: float = 0.0
    damage_threshold_j_per_cm2: float = 0.0


# ---------------------------------------------------------------------------
# SourceSpec (top-level)
# ---------------------------------------------------------------------------


class SourceSpec(StrictBaseModel):
    """Complete source illumination specification.

    Attributes
    ----------
    carrier : CarrierType
        Physical carrier (photon, electron, spin, acoustic, ...).
    strength : float
        Source strength in ``strength_units``.
    strength_units : str
        Units for strength (e.g. ``"photons/s"``, ``"mW"``, ``"T"``).
    spectrum : SpectrumSpec
        Spectral characteristics.
    spatial_profile : SpatialProfile
        Spatial illumination profile.
    coherence : CoherenceSpec
        Coherence properties.
    constraints : SourceConstraints
        Physical constraints (dose, damage, etc.).
    drift : DriftModel
        Source drift model over time.
    """

    carrier: CarrierType = CarrierType.photon
    strength: float = 1.0
    strength_units: str = "arbitrary"
    spectrum: SpectrumSpec = Field(default_factory=SpectrumSpec)
    spatial_profile: SpatialProfile = Field(default_factory=SpatialProfile)
    coherence: CoherenceSpec = Field(default_factory=CoherenceSpec)
    constraints: SourceConstraints = Field(default_factory=SourceConstraints)
    drift: DriftModel = Field(default_factory=DriftModel)


# ---------------------------------------------------------------------------
# ExposureBudget
# ---------------------------------------------------------------------------


class ExposureBudget(StrictBaseModel):
    """Exposure budget for the measurement.

    Attributes
    ----------
    total_photons : float
        Total photon budget across all frames.
    exposure_time_s : float
        Single-frame exposure time in seconds.
    n_frames : int
        Number of measurement frames.
    expected_snr_db : float
        Expected SNR in dB based on source + noise model.
    regime : str
        Operating regime hint: ``"photon_rich"``, ``"photon_starved"``,
        ``"intermediate"``.
    """

    total_photons: float = 0.0
    exposure_time_s: float = 0.0
    n_frames: int = 1
    expected_snr_db: float = 0.0
    regime: str = "intermediate"
