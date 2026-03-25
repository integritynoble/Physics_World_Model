"""pwm_core.spec.domain_profile

DomainProfile — domain-specific extension of CoreSpec.

A DomainProfile layers domain-specific constraints, gates, and metadata on top
of the universal CoreSpec kernel. CoreSpec defines the six-tuple problem
description; DomainProfile adds domain physics, gate extensions, threshold
tables, and modality-specific fields.

Formula:
    CoreSpec + DomainProfile + ProblemInstance = executable PWM object

DomainProfiles are maintained by domain maintainers and versioned
independently from CoreSpec. They extend CoreSpec — they never override
kernel fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GateThreshold(BaseModel):
    metric: str
    warn: Optional[float] = None
    fail: Optional[float] = None
    direction: str = "high"   # "high" = high value is bad; "low" = low value is bad
    description: str = ""


class DomainGateSpec(BaseModel):
    gate_id: str
    display_name: str
    thresholds: List[GateThreshold] = Field(default_factory=list)
    required: bool = True
    description: str = ""


class DomainProfile(BaseModel):
    """Domain-specific extension of CoreSpec.

    Fields
    ------
    domain_id:
        Unique domain identifier, e.g. "imaging", "ct_qc", "acoustics".
    domain_version:
        Semver version of this DomainProfile.
    display_name:
        Human-readable domain name.
    extends:
        Parent DomainProfile ID (e.g. "imaging/v1" for ct_qc). None = extends CoreSpec directly.
    primitive_chain:
        Ordered list of physics primitives defining the canonical forward model
        for this domain (e.g. ["Project", "Attenuate", "Detect"] for CT).
    domain_gates:
        Domain-specific Judge gate extensions. These add to, never replace, S1-S4.
    physics_tier:
        Tier descriptor for the physics complexity
        ("quantum", "wave", "ray", "geometric", "statistical").
    noise_model:
        Canonical noise model for this domain ("poisson", "gaussian", "poisson_gaussian", etc.)
    extra:
        Arbitrary domain-specific metadata.
    """

    domain_id: str
    domain_version: str = "v1"
    display_name: str = ""
    extends: Optional[str] = None
    description: str = ""
    primitive_chain: List[str] = Field(default_factory=list)
    domain_gates: List[DomainGateSpec] = Field(default_factory=list)
    physics_tier: str = "geometric"
    noise_model: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)

    def gate_ids(self) -> List[str]:
        return [g.gate_id for g in self.domain_gates]

    def required_gates(self) -> List[DomainGateSpec]:
        return [g for g in self.domain_gates if g.required]


# ---------------------------------------------------------------------------
# Built-in domain profiles (canonical imaging and CT QC)
# ---------------------------------------------------------------------------

IMAGING_V1 = DomainProfile(
    domain_id="imaging",
    domain_version="v1",
    display_name="Computational Imaging",
    description=(
        "Universal domain profile for computational imaging. Covers the full "
        "spectrum from CT/MRI clinical imaging through compressive sensing, "
        "spectral imaging, and microscopy. Adds Triad gates (G1/G2/G3) on top "
        "of the universal S1-S4 kernel."
    ),
    primitive_chain=["Propagate", "Modulate", "Project", "Encode", "Convolve",
                     "Accumulate", "Detect", "Sample", "Disperse", "Scatter", "Attenuate"],
    physics_tier="geometric",
    noise_model="poisson_gaussian",
    domain_gates=[
        DomainGateSpec(
            gate_id="g1_sampling_recoverability",
            display_name="G1: Sampling Recoverability",
            description=(
                "Checks whether the measurement is theoretically recoverable given "
                "the sampling operator and the signal's assumed sparsity or smoothness."
            ),
            required=False,
        ),
        DomainGateSpec(
            gate_id="g2_carrier_budget",
            display_name="G2: Carrier Budget",
            description=(
                "Checks whether the SNR budget of the physical carrier "
                "(photon count, k-space coverage, etc.) is sufficient for the "
                "claimed reconstruction accuracy."
            ),
            required=False,
        ),
        DomainGateSpec(
            gate_id="g3_operator_mismatch",
            display_name="G3: Operator Mismatch",
            description=(
                "Checks whether the assumed forward model matches the actual "
                "measurement process within claimed tolerance bounds."
            ),
            required=False,
        ),
    ],
)

CT_QC_V1 = DomainProfile(
    domain_id="ct_qc",
    domain_version="v1",
    display_name="CT Quality Control",
    extends="imaging/v1",
    description=(
        "Domain profile for CT scanner quality control. Adds threshold-based "
        "QC gates (noise, uniformity, CNR, MTF), artifact classification, "
        "and drift detection. Aligned with ACR CT Accreditation and AAPM TG-18."
    ),
    primitive_chain=["Project", "Attenuate", "Detect"],
    physics_tier="geometric",
    noise_model="poisson_gaussian",
    domain_gates=[
        DomainGateSpec(
            gate_id="qc_noise",
            display_name="CT Noise Level (HU SD)",
            thresholds=[GateThreshold(metric="HU_noise_sd", warn=8.0, fail=12.0, direction="high")],
        ),
        DomainGateSpec(
            gate_id="qc_uniformity",
            display_name="CT Number Uniformity",
            thresholds=[GateThreshold(metric="HU_uniformity_max_deviation", warn=4.0, fail=8.0, direction="high")],
        ),
        DomainGateSpec(
            gate_id="qc_cnr",
            display_name="Contrast-to-Noise Ratio",
            thresholds=[GateThreshold(metric="CNR", warn=3.0, fail=1.5, direction="low")],
        ),
        DomainGateSpec(
            gate_id="qc_mtf",
            display_name="MTF 10%",
            thresholds=[GateThreshold(metric="MTF10_lpmm", warn=4.5, fail=3.5, direction="low")],
        ),
    ],
)

_BUILT_IN_PROFILES: Dict[str, DomainProfile] = {
    "imaging/v1": IMAGING_V1,
    "ct_qc/v1": CT_QC_V1,
}


def get_domain_profile(domain_id: str) -> Optional[DomainProfile]:
    return _BUILT_IN_PROFILES.get(domain_id)


def list_domain_profiles() -> List[str]:
    return list(_BUILT_IN_PROFILES.keys())
