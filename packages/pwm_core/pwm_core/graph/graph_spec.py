"""pwm_core.graph.graph_spec
============================

Pydantic v2 models for the OperatorGraph intermediate representation.

Classes
-------
NoiseSpec         Noise model specification
GraphNode         One node in the operator DAG (primitive + params)
GraphEdge         Directed edge connecting two nodes
OperatorGraphSpec Complete graph specification (serializable, validatable)

All models inherit from StrictBaseModel (extra="forbid", NaN/Inf rejection).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pwm_core.graph.ir_types import NodeRole, NodeTags, ParameterSpec


# ---------------------------------------------------------------------------
# StrictBaseModel (local copy so graph module is self-contained)
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
# Noise specification
# ---------------------------------------------------------------------------


class NoiseSpec(StrictBaseModel):
    """Explicit noise model for the graph.

    Attributes
    ----------
    noise_type : str
        One of ``poisson``, ``gaussian``, ``poisson_gaussian``, ``fpn``.
    params : dict
        Noise parameters (e.g., ``peak_photons``, ``read_sigma``).
    """

    noise_type: str = "poisson_gaussian"
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------


class GraphNode(StrictBaseModel):
    """One node in the operator DAG.

    Attributes
    ----------
    node_id : str
        Unique identifier within the graph (e.g., ``"modulate"``).
    primitive_id : str
        Must exist in ``primitives.yaml`` / PRIMITIVE_REGISTRY.
    params : dict
        Parameters bound at compile time.
    learnable : list[str]
        Parameter names eligible for calibration / optimisation.
    """

    node_id: str
    primitive_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    learnable: List[str] = Field(default_factory=list)
    tags: Optional[NodeTags] = None
    role: Optional[NodeRole] = None
    parameter_specs: List[ParameterSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph edge
# ---------------------------------------------------------------------------


class GraphEdge(StrictBaseModel):
    """Directed edge in the operator DAG.

    Attributes
    ----------
    source : str
        Source ``node_id``.
    target : str
        Target ``node_id``.
    axes : list[str]
        Axis metadata for compatibility checking (e.g., ``["H", "W", "L"]``).
    """

    source: str
    target: str
    axes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OperatorGraphSpec (top-level)
# ---------------------------------------------------------------------------


class OperatorGraphSpec(StrictBaseModel):
    """Complete operator graph specification (DAG of primitives).

    This is the serializable, human-readable description of a forward model.
    It compiles into a ``GraphOperator`` via ``GraphCompiler.compile()``.

    Attributes
    ----------
    graph_id : str
        Unique identifier, typically ``<modality>_graph_v<N>``
        following registry conventions.
    nodes : list[GraphNode]
        Nodes in the DAG, each referencing a ``primitive_id``.
    edges : list[GraphEdge]
        Directed edges defining data flow.
    noise_model : NoiseSpec | None
        Optional explicit noise specification.
    metadata : dict
        Arbitrary metadata (modality tags, references, etc.).
    """

    graph_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    noise_model: Optional[NoiseSpec] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_edges_reference_existing_nodes(self) -> "OperatorGraphSpec":
        """Validate that all edge endpoints reference existing nodes."""
        node_ids = {n.node_id for n in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(
                    f"Edge source '{edge.source}' not found in nodes. "
                    f"Available: {sorted(node_ids)}"
                )
            if edge.target not in node_ids:
                raise ValueError(
                    f"Edge target '{edge.target}' not found in nodes. "
                    f"Available: {sorted(node_ids)}"
                )
        return self

    @model_validator(mode="after")
    def _check_unique_node_ids(self) -> "OperatorGraphSpec":
        """Validate that all node_ids are unique."""
        seen: set[str] = set()
        for node in self.nodes:
            if node.node_id in seen:
                raise ValueError(f"Duplicate node_id: '{node.node_id}'")
            seen.add(node.node_id)
        return self
