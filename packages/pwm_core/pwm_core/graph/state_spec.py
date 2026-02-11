"""pwm_core.graph.state_spec
=============================

Carrier-specific state specifications for the universal forward model.

Each carrier type (photon, electron, acoustic, spin) has its own state
representation describing what physical quantities are tracked as the
signal propagates through the imaging system.

StateSpec is the union type dispatched by carrier_type.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pwm_core.graph.ir_types import CarrierType


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
# Photon state
# ---------------------------------------------------------------------------


class PhotonState(StrictBaseModel):
    """State spec for photon-based imaging modalities.

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``photon``.
    representation : str
        Signal representation: ``complex_field``, ``intensity``, ``stokes``.
    polarization_resolved : bool
        Whether polarization is tracked.
    wavelength_resolved : bool
        Whether wavelength/spectral channels are tracked.
    time_resolved : bool
        Whether temporal gating/FLIM is tracked.
    """

    carrier_type: CarrierType = CarrierType.photon
    representation: str = "intensity"
    polarization_resolved: bool = False
    wavelength_resolved: bool = False
    time_resolved: bool = False


# ---------------------------------------------------------------------------
# Electron state
# ---------------------------------------------------------------------------


class ElectronState(StrictBaseModel):
    """State spec for electron-based imaging modalities (e.g. TEM, SEM).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``electron``.
    representation : str
        Signal representation: ``wavefunction``, ``intensity``, ``diffraction``.
    energy_resolved : bool
        Whether energy filtering (EELS) is tracked.
    accelerating_voltage_kv : float
        Accelerating voltage in kV.
    """

    carrier_type: CarrierType = CarrierType.electron
    representation: str = "intensity"
    energy_resolved: bool = False
    accelerating_voltage_kv: float = 200.0


# ---------------------------------------------------------------------------
# Acoustic state
# ---------------------------------------------------------------------------


class AcousticState(StrictBaseModel):
    """State spec for acoustic imaging modalities (ultrasound, photoacoustic).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``acoustic``.
    representation : str
        Signal representation: ``pressure``, ``displacement``, ``rf_signal``.
    speed_of_sound_m_per_s : float
        Speed of sound in the medium (m/s).
    center_freq_hz : float
        Transducer center frequency in Hz.
    """

    carrier_type: CarrierType = CarrierType.acoustic
    representation: str = "pressure"
    speed_of_sound_m_per_s: float = 1540.0
    center_freq_hz: float = 5.0e6


# ---------------------------------------------------------------------------
# Spin state
# ---------------------------------------------------------------------------


class SpinState(StrictBaseModel):
    """State spec for spin-based imaging modalities (MRI, NMR).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``spin``.
    representation : str
        Signal representation: ``magnetization``, ``kspace``, ``image``.
    field_strength_t : float
        Main magnetic field strength in Tesla.
    sequence_type : str
        MRI sequence type hint: ``spin_echo``, ``gradient_echo``, ``epi``.
    """

    carrier_type: CarrierType = CarrierType.spin
    representation: str = "magnetization"
    field_strength_t: float = 3.0
    sequence_type: str = "spin_echo"


# ---------------------------------------------------------------------------
# Wave-field state (complex field + wavelength/frequency)
# ---------------------------------------------------------------------------


class WaveFieldState(StrictBaseModel):
    """State spec for complex wave-field imaging (holography, ptychography).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``photon``.
    representation : str
        Signal representation: ``complex_field``.
    wavelength_nm : float
        Wavelength in nanometres.
    frequency_hz : Optional[float]
        Optical frequency in Hz (derived from wavelength if not given).
    """

    carrier_type: CarrierType = CarrierType.photon
    representation: str = "complex_field"
    wavelength_nm: float = 550.0
    frequency_hz: Optional[float] = None


# ---------------------------------------------------------------------------
# Ray-bundle state (rays for rendering / LiDAR)
# ---------------------------------------------------------------------------


class RayBundleState(StrictBaseModel):
    """State spec for ray-based imaging (LiDAR, light-field, NeRF).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``photon``.
    representation : str
        Signal representation: ``ray_bundle``.
    n_rays : int
        Number of rays in the bundle.
    max_bounces : int
        Maximum number of bounces for ray tracing.
    """

    carrier_type: CarrierType = CarrierType.photon
    representation: str = "ray_bundle"
    n_rays: int = 1024
    max_bounces: int = 1


# ---------------------------------------------------------------------------
# Event-list state (time-stamped events: SPAD, PALM/STORM)
# ---------------------------------------------------------------------------


class EventListState(StrictBaseModel):
    """State spec for event-based imaging (SPAD, PALM/STORM, event cameras).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``photon``.
    representation : str
        Signal representation: ``event_list``.
    time_resolved : bool
        Whether events carry time-stamps.
    bin_width_ns : float
        Temporal bin width in nanoseconds.
    """

    carrier_type: CarrierType = CarrierType.photon
    representation: str = "event_list"
    time_resolved: bool = True
    bin_width_ns: float = 1.0


# ---------------------------------------------------------------------------
# K-space state (complex k-space, extends SpinState concept)
# ---------------------------------------------------------------------------


class KSpaceState(StrictBaseModel):
    """State spec for k-space data (MRI trajectories, non-Cartesian sampling).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``spin``.
    representation : str
        Signal representation: ``kspace``.
    trajectory_type : str
        K-space trajectory type: ``cartesian``, ``radial``, ``spiral``.
    undersampling_factor : float
        Undersampling factor (1.0 = fully sampled).
    """

    carrier_type: CarrierType = CarrierType.spin
    representation: str = "kspace"
    trajectory_type: str = "cartesian"
    undersampling_factor: float = 1.0


# ---------------------------------------------------------------------------
# Phase-space state (position + angle + energy for particles)
# ---------------------------------------------------------------------------


class PhaseSpaceState(StrictBaseModel):
    """State spec for particle phase-space imaging (muon, proton, EBSD).

    Attributes
    ----------
    carrier_type : CarrierType
        Always ``abstract``.
    representation : str
        Signal representation: ``phase_space``.
    energy_kev : float
        Particle kinetic energy in keV.
    angular_spread_mrad : float
        Angular spread of the beam in milliradians.
    """

    carrier_type: CarrierType = CarrierType.abstract
    representation: str = "phase_space"
    energy_kev: float = 1.0
    angular_spread_mrad: float = 0.0


# ---------------------------------------------------------------------------
# StateSpec union
# ---------------------------------------------------------------------------

StateSpec = Union[
    PhotonState,
    ElectronState,
    AcousticState,
    SpinState,
    WaveFieldState,
    RayBundleState,
    EventListState,
    KSpaceState,
    PhaseSpaceState,
]


def make_state_spec(carrier_type: CarrierType, **kwargs: Any) -> StateSpec:
    """Factory: create the appropriate state spec for a carrier type.

    Parameters
    ----------
    carrier_type : CarrierType
        The physical carrier.
    **kwargs
        Additional fields passed to the state spec constructor.

    Returns
    -------
    StateSpec
        A carrier-specific state spec instance.
    """
    _MAP = {
        CarrierType.photon: PhotonState,
        CarrierType.electron: ElectronState,
        CarrierType.acoustic: AcousticState,
        CarrierType.spin: SpinState,
        CarrierType.abstract: PhaseSpaceState,
    }
    cls = _MAP.get(carrier_type, PhotonState)
    return cls(carrier_type=carrier_type, **kwargs)
