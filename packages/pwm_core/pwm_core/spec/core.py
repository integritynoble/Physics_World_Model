"""pwm_core.spec.core

Three-layer CoreSpec schema (P-1 terminology alignment).

Dyson Swarm strategy §1 names the six-tuple
  (object, forward model, measurement, noise model, prior, task)
as **CoreSpec v1**.  In the codebase this is already implemented as
``ExperimentSpec`` (v0.2.1).

This module provides:
- ``CoreSpec`` — a compatibility alias for ``ExperimentSpec`` (no rewrite)
- ``DomainProfile`` — typed stub for per-modality physics constraints
- ``ProblemInstance`` — typed stub for a single experimental realisation

Migration guide
---------------
Existing code that imports ``ExperimentSpec`` continues to work unchanged.
New code should import ``CoreSpec`` from here to align with published nomenclature.

When the three-layer structural separation (§1 of the strategy) is fully
implemented, ``DomainProfile`` and ``ProblemInstance`` will become independent
Pydantic models.  Until then they are lightweight wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ExperimentSpec is the canonical implementation of CoreSpec v1.
# Import it here and re-export under both names so callers can use either.
from pwm_core.api.types import ExperimentSpec

# CoreSpec v1 = ExperimentSpec (compatibility alias, no destructive rewrite)
CoreSpec = ExperimentSpec


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
