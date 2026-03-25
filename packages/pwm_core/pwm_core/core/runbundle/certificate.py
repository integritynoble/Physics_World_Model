"""pwm_core.core.runbundle.certificate

Certificate v1 — trust verdict emitted at the end of every certified run.

Schema defined in docs/dyson_swarm_strategy.md §3 (S1-S4 gates).

A Certificate is the output of the Judge after all four S-gates have been
evaluated.  It is serialised to ``certificate.json`` inside the RunBundle
directory alongside ``runbundle_manifest.json``.
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
    """Linear trust promotion ladder (no skipping allowed)."""
    draft = "draft"
    author_confirmed = "author_confirmed"
    reproduced = "reproduced"
    certified = "certified"


class GateVerdict(str, Enum):
    """Pass / warn / fail for each S-gate."""
    pass_ = "pass"
    warn = "warn"
    fail = "fail"


class RiskFlag(str, Enum):
    """Overlay flags — can be applied to any tier without demoting it."""
    boundary_risk = "boundary_risk"
    safety_brake = "safety_brake"
    high_variance = "high_variance"
    reviewer_disputed = "reviewer_disputed"


# ---------------------------------------------------------------------------
# Certificate model
# ---------------------------------------------------------------------------

class GateResult(BaseModel):
    """Pass/fail/warn verdict + optional human-readable message for one gate."""
    verdict: GateVerdict
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TriadFlags(BaseModel):
    """Triad gate attribution (G1/G2/G3) results."""
    g1_sampling: Optional[GateResult] = None   # Sampling-domain bottleneck
    g2_noise: Optional[GateResult] = None       # Noise-model mismatch
    g3_operator: Optional[GateResult] = None    # Forward-operator mismatch


class Certificate(BaseModel):
    """Certificate v1 — canonical trust record for a RunBundle.

    Emitted by ``issue_certificate()`` in ``runbundle_emitter.py`` after the
    S1-S4 gate checks complete.  Written to ``certificate.json`` in the bundle
    root.
    """

    # Identity
    run_id: str = Field(..., description="Unique run identifier (matches RunBundle directory name)")
    spec_id: str = Field(..., description="Spec identifier that produced this run")
    judge_version: str = Field(default="1.0.0", description="pwm_core version that issued this certificate")

    # Trust outcome
    trust_tier: TrustTier = Field(..., description="Highest tier this run has passed")
    risk_flags: List[RiskFlag] = Field(default_factory=list, description="Overlay risk flags (do not demote tier)")

    # Gate verdicts (S1-S4)
    active_gates: List[str] = Field(default_factory=list, description="Names of gates that were evaluated")
    gate_verdicts: Dict[str, GateResult] = Field(
        default_factory=dict,
        description="Per-gate results; keys are gate names s1/s2/s3/s4",
    )

    # Triad attribution
    triad_flags: Optional[TriadFlags] = None

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 of provenance.json for cross-reference",
    )
    contributor_attribution: Optional[str] = Field(
        default=None,
        description="Free-text or URI identifying the submitter",
    )

    # Timestamp
    issued_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this certificate was issued",
    )

    # Overall pass/fail
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
