"""pwm_core.core.runbundle.certificate

Certificate v2 -- trust verdict emitted at the end of every certified run.

Schema defined in docs/dyson_swarm_strategy.md S4 (R1-R4 operational gates).

A Certificate is the output of the Judge after all four R-gates have been
evaluated.  It is serialised to ``certificate.json`` inside the RunBundle
directory alongside ``runbundle_manifest.json``.

Version-stamping fields (kernel_version, profile_version, registry_versions,
product_version) record exactly which code and configuration produced the
certificate, enabling bit-exact reproducibility audits.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TrustTier(str, Enum):
    """Linear trust promotion ladder (no skipping allowed).

    ``rejected`` is a terminal state: one or more hard gates (R1/R2/R3) failed.
    Rejected runs cannot be promoted without a new RunBundle.
    """
    rejected = "rejected"            # hard gate failure -- cannot be promoted
    draft = "draft"
    author_confirmed = "author_confirmed"
    reproduced = "reproduced"
    certified = "certified"


class GateVerdict(str, Enum):
    """Pass / warn / fail for each R-gate."""
    pass_ = "pass"
    warn = "warn"
    fail = "fail"


class RiskFlag(str, Enum):
    """Overlay flags -- can be applied to any tier without demoting it."""
    boundary_risk = "boundary_risk"
    safety_brake = "safety_brake"
    high_variance = "high_variance"
    reviewer_disputed = "reviewer_disputed"


class DomainProfileMaturity(str, Enum):
    """Maturity level of a DomainProfile, used by the registry to decide
    whether a profile may be loaded in production or requires explicit opt-in.
    """
    experimental = "experimental"
    reference = "reference"
    trusted = "trusted"
    certified_compatible = "certified_compatible"
    deprecated = "deprecated"


# ---------------------------------------------------------------------------
# Certificate sub-models
# ---------------------------------------------------------------------------

class GateResult(BaseModel):
    """Pass/fail/warn verdict + optional human-readable message for one gate."""
    verdict: GateVerdict
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ImagingTriadFlags(BaseModel):
    """Imaging-domain triad gate attribution (G1/G2/G3) results.

    These flags summarise the triad decomposition for inverse-imaging
    modalities (CT, MRI, spectral, etc.).  For non-imaging domains,
    domain-specific flag models should be used instead via DomainFlags.
    """
    g1_sampling: Optional[GateResult] = None   # Sampling-domain bottleneck
    g2_noise: Optional[GateResult] = None       # Noise-model mismatch
    g3_operator: Optional[GateResult] = None    # Forward-operator mismatch


# Backward compatibility alias -- existing code imports TriadFlags
TriadFlags = ImagingTriadFlags


class DomainFlags(BaseModel):
    """Domain-specific diagnostic flags.  Universal envelope, domain fills content.

    Each domain gets an Optional field.  When Certificate is issued the
    appropriate domain field is populated while the rest remain None.
    Extensible -- add new domains as Optional fields.
    """
    imaging: Optional[ImagingTriadFlags] = None
    ct_qc: Optional[Dict[str, Any]] = None
    combustion: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Certificate model
# ---------------------------------------------------------------------------

class Certificate(BaseModel):
    """Certificate v2 -- canonical trust record for a RunBundle.

    Emitted by ``issue_certificate()`` in ``runbundle_emitter.py`` after the
    R1-R4 gate checks complete.  Written to ``certificate.json`` in the bundle
    root.

    Version-stamping fields allow downstream consumers to pin results to
    exact code revisions of the kernel, domain profile, primitive registries,
    and (optionally) the hosted product build.
    """

    # Identity
    run_id: str = Field(..., description="Unique run identifier (matches RunBundle directory name)")
    spec_id: str = Field(..., description="Spec identifier that produced this run")
    judge_version: str = Field(default="2.0.0", description="pwm_core version that issued this certificate")

    # Trust outcome
    trust_tier: TrustTier = Field(..., description="Highest tier this run has passed")
    risk_flags: List[RiskFlag] = Field(default_factory=list, description="Overlay risk flags (do not demote tier)")

    # Gate verdicts (R1-R4)
    active_gates: List[str] = Field(default_factory=list, description="Names of gates that were evaluated")
    gate_verdicts: Dict[str, GateResult] = Field(
        default_factory=dict,
        description="Per-gate results; keys are gate names r1/r2/r3/r4",
    )

    # Domain-specific diagnostic flags
    domain_flags: Optional[DomainFlags] = None

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 of provenance.json for cross-reference",
    )
    contributor_attribution: Optional[str] = Field(
        default=None,
        description="Free-text or URI identifying the submitter",
    )

    # Version stamping
    kernel_version: Optional[str] = Field(
        default=None,
        description="Public-kernel commit hash or semver tag (Physics_World_Model)",
    )
    profile_version: Optional[str] = Field(
        default=None,
        description="DomainProfile version used, e.g. 'imaging/v1.2'",
    )
    registry_versions: Optional[Dict[str, str]] = Field(
        default=None,
        description="Primitive registry versions, e.g. {'general': 'v1.3', 'imaging': 'v1.1'}",
    )
    product_version: Optional[str] = Field(
        default=None,
        description="Product build version if issued on hosted platform; absent for local runs",
    )

    # Timestamp
    issued_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this certificate was issued",
    )

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    @property
    def triad_flags(self) -> Optional[ImagingTriadFlags]:
        """Backward-compatible accessor.  Returns imaging triad flags from domain_flags."""
        return self.domain_flags.imaging if self.domain_flags else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_certified(self) -> bool:
        """True if all active gates passed (no failures)."""
        return all(
            v.verdict != GateVerdict.fail
            for v in self.gate_verdicts.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        try:
            return self.model_dump()
        except AttributeError:
            return self.__dict__
