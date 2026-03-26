"""pwm_core.spec.profile_registry

Formal DomainProfile registry for Priority-1 modalities.

Each of the 13 Priority-1 modalities gets a typed DomainProfile entry
with gates, diagnostic decomposition reference, primitive chain, and
maturity status.

The profiles extend the canonical ``IMAGING_V1`` base profile defined in
``domain_profile.py``, inheriting its triad gates (G1/G2/G3) but narrowing
the primitive chain and noise model to the modality-specific physics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pwm_core.spec.domain_profile import (
    DomainGateSpec,
    DomainProfile,
    GateThreshold,
    IMAGING_V1,
    CT_QC_V1,
)
from pwm_core.core.runbundle.certificate import DomainProfileMaturity


# ---------------------------------------------------------------------------
# Helper: reuse triad gates from IMAGING_V1 by ID
# ---------------------------------------------------------------------------

_IMAGING_GATES_BY_ID: Dict[str, DomainGateSpec] = {
    g.gate_id: g for g in IMAGING_V1.domain_gates
}

# Shorthand aliases used by profile definitions below
_G1 = _IMAGING_GATES_BY_ID["g1_sampling_recoverability"]
_G2 = _IMAGING_GATES_BY_ID["g2_carrier_budget"]
_G3 = _IMAGING_GATES_BY_ID["g3_operator_mismatch"]

_TRIAD_GATES = [_G1, _G2, _G3]
_TRIAD_GATES_NO_G2 = [_G1, _G3]


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def _build_registry() -> Dict[str, DomainProfile]:
    """Build the Priority-1 DomainProfile registry."""
    profiles: Dict[str, DomainProfile] = {}

    # --- Medical Imaging Modalities (12 imaging + 1 QC) ---

    profiles["ct"] = DomainProfile(
        domain_id="ct",
        domain_version="v1.0",
        display_name="Computed Tomography",
        extends="imaging/v1",
        description=(
            "Parallel-beam X-ray CT with Radon forward model and Poisson "
            "photon statistics.  The canonical benchmark modality for PWM."
        ),
        primitive_chain=["Project", "Accumulate", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="ray",
        noise_model="poisson",
        extra={
            "maturity": DomainProfileMaturity.trusted.value,
            "geometry": "parallel_beam",
            "angles": 180,
        },
    )

    profiles["mri"] = DomainProfile(
        domain_id="mri",
        domain_version="v1.0",
        display_name="Magnetic Resonance Imaging",
        extends="imaging/v1",
        description=(
            "Cartesian-undersampled Fourier-encoded MRI with Gaussian "
            "noise in k-space."
        ),
        primitive_chain=["Encode", "Sample", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="wave",
        noise_model="gaussian",
        extra={
            "maturity": DomainProfileMaturity.trusted.value,
            "acquisition": "cartesian_undersampled",
        },
    )

    profiles["pet"] = DomainProfile(
        domain_id="pet",
        domain_version="v1.0",
        display_name="Positron Emission Tomography",
        extends="imaging/v1",
        description=(
            "Coincidence-detection PET with Poisson emission statistics "
            "and attenuation correction."
        ),
        primitive_chain=["Accumulate", "Project", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="ray",
        noise_model="poisson",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    profiles["spect"] = DomainProfile(
        domain_id="spect",
        domain_version="v1.0",
        display_name="Single-Photon Emission CT",
        extends="imaging/v1",
        description=(
            "Gamma-ray SPECT with collimator blur and Poisson photon "
            "statistics."
        ),
        primitive_chain=["Accumulate", "Project", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="ray",
        noise_model="poisson",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    profiles["ultrasound"] = DomainProfile(
        domain_id="ultrasound",
        domain_version="v1.0",
        display_name="Ultrasound Imaging",
        extends="imaging/v1",
        description=(
            "Pulse-echo ultrasound with acoustic wave propagation and "
            "multiplicative speckle noise."
        ),
        primitive_chain=["Propagate", "Scatter", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="wave",
        noise_model="speckle",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    profiles["oct"] = DomainProfile(
        domain_id="oct",
        domain_version="v1.0",
        display_name="Optical Coherence Tomography",
        extends="imaging/v1",
        description=(
            "Low-coherence interferometric OCT with broadband optical "
            "source and Gaussian detection noise."
        ),
        primitive_chain=["Propagate", "Encode", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="wave",
        noise_model="gaussian",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    profiles["mammography"] = DomainProfile(
        domain_id="mammography",
        domain_version="v1.0",
        display_name="X-ray Mammography",
        extends="imaging/v1",
        description=(
            "Low-dose X-ray projection mammography with Poisson photon "
            "statistics and scatter correction."
        ),
        primitive_chain=["Project", "Attenuate", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="ray",
        noise_model="poisson",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    profiles["cbct"] = DomainProfile(
        domain_id="cbct",
        domain_version="v1.0",
        display_name="Cone-Beam CT",
        extends="imaging/v1",
        description=(
            "3D cone-beam geometry CT with Feldkamp-type reconstruction "
            "and Poisson noise."
        ),
        primitive_chain=["Project", "Accumulate", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="ray",
        noise_model="poisson",
        extra={
            "maturity": DomainProfileMaturity.reference.value,
            "geometry": "cone_beam",
        },
    )

    profiles["fundus"] = DomainProfile(
        domain_id="fundus",
        domain_version="v1.0",
        display_name="Retinal Fundus Photography",
        extends="imaging/v1",
        description=(
            "Widefield retinal fundus photography with lens blur PSF "
            "and Gaussian sensor noise."
        ),
        primitive_chain=["Convolve", "Detect"],
        domain_gates=_TRIAD_GATES_NO_G2,
        physics_tier="geometric",
        noise_model="gaussian",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    profiles["endoscopy"] = DomainProfile(
        domain_id="endoscopy",
        domain_version="v1.0",
        display_name="Endoscopic Imaging",
        extends="imaging/v1",
        description=(
            "Fibre-optic or chip-tip endoscopic imaging with scatter "
            "and geometric distortion."
        ),
        primitive_chain=["Scatter", "Sample", "Detect"],
        domain_gates=_TRIAD_GATES_NO_G2,
        physics_tier="geometric",
        noise_model="gaussian",
        extra={"maturity": DomainProfileMaturity.experimental.value},
    )

    profiles["fmri"] = DomainProfile(
        domain_id="fmri",
        domain_version="v1.0",
        display_name="Functional MRI",
        extends="imaging/v1",
        description=(
            "BOLD functional MRI with temporal encoding, Fourier "
            "undersampling, and haemodynamic accumulation."
        ),
        primitive_chain=["Encode", "Sample", "Accumulate", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="wave",
        noise_model="gaussian",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    profiles["diffusion_mri"] = DomainProfile(
        domain_id="diffusion_mri",
        domain_version="v1.0",
        display_name="Diffusion-Weighted MRI",
        extends="imaging/v1",
        description=(
            "Diffusion-weighted MRI with q-space encoding, Gaussian "
            "dispersion kernel, and Rician magnitude noise."
        ),
        primitive_chain=["Encode", "Disperse", "Sample", "Detect"],
        domain_gates=_TRIAD_GATES,
        physics_tier="wave",
        noise_model="rician",
        extra={"maturity": DomainProfileMaturity.reference.value},
    )

    # --- Non-imaging profiles ---

    profiles["ct_qc"] = DomainProfile(
        domain_id="ct_qc",
        domain_version="v1.0",
        display_name="CT Quality Control",
        extends="ct_qc/v1",
        description=(
            "CT scanner QC profile with threshold gates for noise, "
            "uniformity, CNR, and MTF aligned with ACR accreditation."
        ),
        primitive_chain=["Validate"],
        domain_gates=[
            DomainGateSpec(
                gate_id="drift",
                display_name="QC Drift Detection",
                description="Detects systematic drift in QC metrics over time.",
                required=True,
            ),
            DomainGateSpec(
                gate_id="artifact",
                display_name="Artifact Classification",
                description="Classifies and flags reconstruction artifacts.",
                required=True,
            ),
            DomainGateSpec(
                gate_id="threshold_breach",
                display_name="Threshold Breach",
                description="Flags when QC metrics exceed tolerance thresholds.",
                thresholds=[
                    GateThreshold(
                        metric="breach_count",
                        warn=1.0,
                        fail=3.0,
                        direction="high",
                    ),
                ],
                required=True,
            ),
        ],
        physics_tier="statistical",
        noise_model="gaussian",
        extra={"maturity": DomainProfileMaturity.certified_compatible.value},
    )

    return profiles


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

PROFILE_REGISTRY: Dict[str, DomainProfile] = _build_registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_profile(modality: str) -> Optional[DomainProfile]:
    """Look up a registered DomainProfile by modality key."""
    return PROFILE_REGISTRY.get(modality)


def list_profiles() -> List[str]:
    """Return all registered profile keys (sorted)."""
    return sorted(PROFILE_REGISTRY.keys())


def get_profile_maturity(modality: str) -> Optional[str]:
    """Return the maturity stage string of a profile, or None if not found."""
    p = PROFILE_REGISTRY.get(modality)
    if p is not None:
        return p.extra.get("maturity", DomainProfileMaturity.experimental.value)
    return None
