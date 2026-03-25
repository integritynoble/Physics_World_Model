"""pwm_core.core.primitive_dialect

PrimitiveDialect — typed registry entry bridging domain-specific primitives to
general/v1 primitives.

A PrimitiveDialect describes *how* a single domain-specific primitive (e.g.
``acoustics/v1/pressure_wave_propagation``) decomposes into one or more
``general/v1`` primitives.  It is the bridge between a :class:`PhysicsDialect`
(which groups all primitives for a domain) and the :class:`OperatorGraph`
(which uses only ``general/v1`` nodes that the executor can run).

Every entry in ``contrib/primitive_registry/{domain}/v1/primitives.yaml``
corresponds to one ``PrimitiveDialect`` instance.  The
``from_registry_entry()`` class method constructs a ``PrimitiveDialect``
directly from the parsed YAML dict produced by the registry loader.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore


# ---------------------------------------------------------------------------
# Sub-model
# ---------------------------------------------------------------------------

class DecompositionStep(BaseModel):
    """One step in the decomposition of a domain primitive into general/v1.

    Steps are ordered by ``step`` index.  An ``optional`` step may be skipped
    by an executor that does not support the referenced primitive; mandatory
    steps must succeed for the decomposition to be valid.
    """

    step: int = Field(..., description="Ordinal index of this step (0-based)")
    general_primitive_id: str = Field(
        ...,
        description="Fully-qualified general/v1 primitive ID, e.g. 'general/v1/Transform'",
    )
    role: str = Field(
        ...,
        description="Human-readable description of what this step does in the decomposition",
    )
    optional: bool = Field(
        default=False,
        description="If True, executors may skip this step without invalidating the decomposition",
    )


# ---------------------------------------------------------------------------
# PrimitiveDialect
# ---------------------------------------------------------------------------

class PrimitiveDialect(BaseModel):
    """Typed registry entry that defines how a domain-specific primitive
    decomposes into general/v1 primitives.

    PrimitiveDialect is the bridge between PhysicsDialect (the domain-level
    description of physical behaviour) and OperatorGraph (the symbolic DAG that
    the executor compiles and runs).

    Each ``PrimitiveDialect`` is stored as a YAML block inside
    ``contrib/primitive_registry/{domain}/v1/primitives.yaml`` and can be
    loaded via ``from_registry_entry()``.
    """

    # Identity
    dialect_id: str = Field(
        ...,
        description="Unique dialect identifier, e.g. 'acoustics/v1/pressure_wave_propagation'",
    )
    display_name: str = Field(..., description="Human-readable name for this primitive")
    domain: str = Field(..., description="Physics domain, e.g. 'acoustics', 'imaging'")
    version: str = Field(..., description="Dialect version string, e.g. 'v1'")
    description: str = Field(..., description="Full description of what this primitive models")

    # Decomposition
    decomposition: List[DecompositionStep] = Field(
        default_factory=list,
        description="Ordered steps mapping this primitive to general/v1 primitives",
    )

    # Semantic I/O
    inputs: List[str] = Field(
        default_factory=list,
        description="Semantic input names consumed by this primitive",
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="Semantic output names produced by this primitive",
    )

    # Physics contract
    physics_assumptions: List[str] = Field(
        default_factory=list,
        description=(
            "Key physical assumptions made by this primitive, "
            "e.g. 'linear wave equation', 'far-field approximation'"
        ),
    )

    # Shortcut index
    general_primitive_ids: List[str] = Field(
        default_factory=list,
        description="Shortcut list of general/v1 primitive IDs used in the decomposition",
    )

    # Registry provenance
    registry_path: str = Field(
        default="",
        description=(
            "Path to the YAML file that defines this dialect, "
            "e.g. 'contrib/primitive_registry/acoustics/v1/primitives.yaml'"
        ),
    )
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this entry was created",
    )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        try:
            return self.model_dump()
        except AttributeError:
            return self.__dict__

    @classmethod
    def from_registry_entry(cls, entry: Dict[str, Any]) -> "PrimitiveDialect":
        """Construct a PrimitiveDialect from a parsed YAML registry entry dict.

        The registry entry is expected to follow the schema used in
        ``contrib/primitive_registry/{domain}/v1/primitives.yaml``.  Unknown
        keys in ``entry`` are silently ignored so that forward-compatible YAML
        additions do not break existing code.

        Parameters
        ----------
        entry:
            Parsed YAML dictionary for a single primitive entry.

        Returns
        -------
        PrimitiveDialect
            A fully-validated PrimitiveDialect instance.
        """
        raw_steps: List[Dict[str, Any]] = entry.get("decomposition", [])
        steps = [
            DecompositionStep(
                step=s.get("step", i),
                general_primitive_id=s["general_primitive_id"],
                role=s.get("role", ""),
                optional=s.get("optional", False),
            )
            for i, s in enumerate(raw_steps)
        ]

        # Build shortcut list from decomposition if not explicitly provided.
        general_ids: List[str] = entry.get(
            "general_primitive_ids",
            [s.general_primitive_id for s in steps],
        )

        return cls(
            dialect_id=entry["dialect_id"],
            display_name=entry.get("display_name", entry["dialect_id"]),
            domain=entry.get("domain", ""),
            version=entry.get("version", "v1"),
            description=entry.get("description", ""),
            decomposition=steps,
            inputs=entry.get("inputs", []),
            outputs=entry.get("outputs", []),
            physics_assumptions=entry.get("physics_assumptions", []),
            general_primitive_ids=general_ids,
            registry_path=entry.get("registry_path", ""),
            created_at=entry.get(
                "created_at",
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
