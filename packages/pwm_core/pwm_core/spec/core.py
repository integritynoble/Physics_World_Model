"""pwm_core.spec.core

Three-layer CoreSpec schema (P-1 terminology alignment).

The paper's canonical six-tuple is:

    (Omega, E, B, I, O, epsilon)
    — domain, equations, boundary conditions,
      initial conditions, observables, tolerance.

``ExperimentSpec`` (v0.2.1) is the current implementation substrate.
``CanonicalCoreSpec`` encodes the target schema directly.  See the
ExperimentSpec -> CoreSpec mapping in ``dyson_swarm_strategy.md`` §2.

This module provides:
- ``CanonicalCoreSpec`` — the paper's universal six-tuple as a Pydantic model
- ``CoreSpec`` — a compatibility alias for ``ExperimentSpec`` (no rewrite)
- ``DomainProfile`` — typed stub for per-modality physics constraints
- ``ProblemInstance`` — typed stub for a single experimental realisation
- ``EXPERIMENT_SPEC_MAPPING`` — 8-layer mapping from ExperimentSpec states
  to CanonicalCoreSpec fields

Migration guide
---------------
Existing code that imports ``ExperimentSpec`` continues to work unchanged.
New code should import ``CoreSpec`` from here to align with published nomenclature.
``CanonicalCoreSpec`` is the forward-looking target; adopt it for new subsystems.

When the three-layer structural separation (§1 of the strategy) is fully
implemented, ``DomainProfile`` and ``ProblemInstance`` will become independent
Pydantic models.  Until then they are lightweight wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore

# ExperimentSpec is the canonical implementation of CoreSpec v1.
# Import it here and re-export under both names so callers can use either.
from pwm_core.api.types import ExperimentSpec

# CoreSpec v1 = ExperimentSpec (compatibility alias, no destructive rewrite)
CoreSpec = ExperimentSpec


# ---------------------------------------------------------------------------
# Canonical CoreSpec — the paper's universal six-tuple
# ---------------------------------------------------------------------------


class CanonicalCoreSpec(BaseModel):
    """Canonical CoreSpec v1 — the paper's universal six-tuple.

    (Omega, E, B, I, O, epsilon): domain, equations, boundary conditions,
    initial conditions, observables, tolerance.

    This is the target schema. ExperimentSpec v0.2.1 is the current
    implementation substrate. See ExperimentSpec -> CoreSpec mapping
    in dyson_swarm_strategy.md §2.
    """

    omega: Dict[str, Any] = Field(
        ..., description="Omega — Domain: spatial extent, mesh, material properties"
    )
    equations: Dict[str, Any] = Field(
        ...,
        description="E — Equations: forward model, physics parameters, noise model",
    )
    boundary_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="B — Boundary conditions: environmental constraints",
    )
    initial_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="I — Initial conditions: calibration, prior",
    )
    observables: Dict[str, Any] = Field(
        ..., description="O — Observables: detector geometry, measurement space"
    )
    tolerance: float = Field(
        ...,
        description="epsilon — Tolerance: acceptance threshold for reconstruction quality",
    )

    # Metadata
    spec_version: str = Field(
        default="1.0.0", description="CoreSpec schema version"
    )
    domain_profile_ref: Optional[str] = Field(
        default=None, description="Reference to the DomainProfile used"
    )


# ---------------------------------------------------------------------------
# ExperimentSpec 8-layer -> CanonicalCoreSpec mapping
# ---------------------------------------------------------------------------

EXPERIMENT_SPEC_MAPPING: Dict[str, str] = {
    "PhysicsState": "equations (E)",
    "SampleState": "omega (Omega)",
    "SensorState": "observables (O)",
    "EnvironmentState": "boundary_conditions (B)",
    "CalibrationState": "initial_conditions (I) / DomainProfile",
    "BudgetState": "ProblemInstance / R4",
    "ComputeState": "ProblemInstance",
    "TaskState": "DomainProfile + tolerance (epsilon)",
}
"""Maps each ExperimentSpec state layer to the corresponding
CanonicalCoreSpec field(s).  BudgetState and ComputeState do not
have direct six-tuple counterparts; they live in ProblemInstance."""


class DomainProfile:
    """Per-modality physics constraints and parameter bounds.

    Stub for future structural separation from CoreSpec.  Currently domain
    profiles live as YAML files in ``contrib/`` (e.g. ``modalities.yaml``,
    ``graph_templates.yaml``).

    Parameters
    ----------
    modality : str
        Modality name (e.g. ``"ct"``, ``"cassi"``).
    constraints : dict, optional
        Physics parameter bounds for this modality.
    """

    def __init__(self, modality: str, constraints: Optional[Dict[str, Any]] = None):
        self.modality = modality
        self.constraints = constraints or {}

    def __repr__(self) -> str:
        return f"DomainProfile(modality={self.modality!r})"


class ProblemInstance:
    """A single experimental realisation bound to a DomainProfile.

    Stub for future structural separation from CoreSpec.  Currently problem
    instances are constructed inline during harness execution.

    Parameters
    ----------
    spec : CoreSpec
        The ``ExperimentSpec`` / ``CoreSpec`` for this instance.
    domain : DomainProfile, optional
        Associated domain profile.
    instance_id : str, optional
        Unique identifier for this realisation.
    """

    def __init__(
        self,
        spec: CoreSpec,
        domain: Optional[DomainProfile] = None,
        instance_id: Optional[str] = None,
    ):
        self.spec = spec
        self.domain = domain
        self.instance_id = instance_id

    def __repr__(self) -> str:
        return f"ProblemInstance(id={self.instance_id!r})"
