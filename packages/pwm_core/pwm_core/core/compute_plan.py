"""pwm_core.core.compute_plan

ComputePlan — compiled execution plan produced from an OperatorGraph.

A ComputePlan specifies the concrete numerical schedule: which primitives run
in which order, on which hardware, with what memory budget.  It is produced by
the ComputePlanner (which compiles an OperatorGraph against a CoreSpec) and
consumed by the executor that actually runs the computation.

Relationship to other objects
------------------------------
* **OperatorGraph** (symbolic DAG of primitives) → compiled by ComputePlanner
  → **ComputePlan** (ordered, hardware-bound execution steps) → run by executor.
* The ``operator_graph_id`` and ``provenance_hash`` fields preserve a traceable
  link back to the symbolic representation for debugging and audit.
* Each ``ComputeStep`` binds one general/v1 primitive to concrete I/O tensor
  names and a device assignment.
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

class ComputeStep(BaseModel):
    """One concrete execution step within a ComputePlan.

    Each step corresponds to a single general/v1 primitive call with fully
    resolved tensor names and a device assignment.
    """

    step_id: str = Field(..., description="Unique identifier for this step within the plan")
    primitive_id: str = Field(
        ...,
        description="Fully-qualified primitive identifier, e.g. 'general/v1/Transform'",
    )
    inputs: List[str] = Field(
        default_factory=list,
        description="Names of input tensors/arrays consumed by this step",
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="Names of output tensors/arrays produced by this step",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Primitive-specific configuration parameters",
    )
    estimated_runtime_s: Optional[float] = Field(
        default=None,
        description="Estimated wall-clock runtime in seconds (None if unknown)",
    )
    device: str = Field(
        default="cpu",
        description="Hardware device for this step: 'cpu' or 'cuda'",
    )


# ---------------------------------------------------------------------------
# ComputePlan
# ---------------------------------------------------------------------------

class ComputePlan(BaseModel):
    """Compiled execution plan produced from an OperatorGraph.

    ComputePlan is the output of the ComputePlanner.  It lists the ordered
    sequence of ``ComputeStep`` objects that an executor must run to perform the
    computation described by the originating ``OperatorGraph``.

    Fields that begin with ``estimate`` are advisory: executors should record
    actual values but must not reject a plan because an estimate is absent.
    """

    # Identity
    plan_id: str = Field(..., description="Unique plan identifier (UUID hex)")
    operator_graph_id: str = Field(
        ...,
        description="ID of the OperatorGraph this plan was compiled from",
    )
    spec_id: str = Field(..., description="CoreSpec ID that governs this plan")
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this plan was created",
    )

    # Execution schedule
    steps: List[ComputeStep] = Field(
        default_factory=list,
        description="Ordered list of execution steps; must be run in index order",
    )

    # Resource estimates
    total_flops_estimate: Optional[float] = Field(
        default=None,
        description="Estimated total floating-point operations (advisory)",
    )
    peak_memory_bytes_estimate: Optional[int] = Field(
        default=None,
        description="Estimated peak RAM usage in bytes (advisory)",
    )

    # Hardware target
    target_device: str = Field(
        default="cpu",
        description="Primary hardware target: 'cpu', 'cuda', or 'mps'",
    )

    # Provenance
    compiler_version: str = Field(
        ...,
        description="Version string of the ComputePlanner that produced this plan",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 of the OperatorGraph this plan was compiled from",
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
