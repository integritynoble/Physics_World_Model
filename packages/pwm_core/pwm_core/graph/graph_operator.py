"""pwm_core.graph.graph_operator
================================

GraphOperator: a compiled operator graph that implements forward(), adjoint(),
serialize(), check_adjoint(), and explain().

The GraphOperator is produced by ``GraphCompiler.compile()`` and should not
be instantiated directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.graph.ir_types import NodeTags
from pwm_core.graph.primitives import BasePrimitive

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AdjointCheckReport (graph-level)
# ---------------------------------------------------------------------------


@dataclass
class GraphAdjointCheckReport:
    """Result of check_adjoint() on a compiled GraphOperator."""

    passed: bool
    n_trials: int
    max_relative_error: float
    mean_relative_error: float
    tolerance: float
    details: List[Dict[str, float]]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"GraphAdjointCheck [{status}]: "
            f"max_rel_err={self.max_relative_error:.2e}, "
            f"tol={self.tolerance:.2e}, trials={self.n_trials}"
        )


# ---------------------------------------------------------------------------
# GraphOperator
# ---------------------------------------------------------------------------


@dataclass
class GraphOperator:
    """Compiled operator graph with forward/adjoint execution plans.

    Attributes
    ----------
    graph_id : str
        Identifier from the OperatorGraphSpec.
    forward_plan : list[tuple[str, BasePrimitive]]
        Topologically sorted (node_id, primitive) pairs for forward pass.
    adjoint_plan : list[tuple[str, BasePrimitive]]
        Reverse order for adjoint pass (only valid for linear graphs).
    node_map : dict[str, BasePrimitive]
        Map from node_id to instantiated primitive.
    all_linear : bool
        True if every primitive in the graph is linear.
    x_shape : tuple[int, ...]
        Expected input shape (informational).
    y_shape : tuple[int, ...]
        Expected output shape (informational).
    metadata : dict[str, Any]
        Graph metadata from the spec.
    edges : list[tuple[str, str]]
        Edge list (source, target) for introspection.
    learnable_params : dict[str, list[str]]
        node_id -> list of learnable parameter names.
    """

    graph_id: str
    forward_plan: List[Tuple[str, BasePrimitive]]
    adjoint_plan: List[Tuple[str, BasePrimitive]]
    node_map: Dict[str, BasePrimitive]
    all_linear: bool
    x_shape: Tuple[int, ...] = (64, 64)
    y_shape: Tuple[int, ...] = (64, 64)
    metadata: Dict[str, Any] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    learnable_params: Dict[str, List[str]] = field(default_factory=dict)
    node_tags: Dict[str, NodeTags] = field(default_factory=dict)

    # ---- Forward ----

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute the forward model in topological order.

        Each node takes the output of the previous node in the sequential plan.
        """
        current = x.copy()
        for node_id, primitive in self.forward_plan:
            try:
                current = primitive.forward(current)
            except Exception as exc:
                raise RuntimeError(
                    f"Forward pass failed at node '{node_id}' "
                    f"(primitive={primitive.primitive_id}): {exc}"
                ) from exc
        return current

    # ---- Adjoint ----

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Execute the adjoint model in reverse topological order.

        Only valid when all primitives are linear.
        """
        if not self.all_linear:
            raise RuntimeError(
                "Adjoint is only available for fully-linear graphs. "
                "This graph contains non-linear primitives: "
                + ", ".join(
                    nid
                    for nid, p in self.forward_plan
                    if not p.is_linear
                )
            )
        current = y.copy()
        for node_id, primitive in self.adjoint_plan:
            try:
                current = primitive.adjoint(current)
            except Exception as exc:
                raise RuntimeError(
                    f"Adjoint pass failed at node '{node_id}' "
                    f"(primitive={primitive.primitive_id}): {exc}"
                ) from exc
        return current

    # ---- Serialize ----

    def serialize(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Serialize the compiled graph, including SHA256 hashes for blobs.

        Returns a JSON-serializable dict suitable for RunBundle inclusion.
        """
        nodes_ser = []
        all_blobs = []
        for node_id, primitive in self.forward_plan:
            prim_data = primitive.serialize()
            prim_data["node_id"] = node_id
            # Include NodeTags if available
            if node_id in self.node_tags:
                prim_data["tags"] = self.node_tags[node_id].model_dump()
            all_blobs.extend(prim_data.pop("blobs", []))
            nodes_ser.append(prim_data)

        # Compute a graph-level hash from the deterministic JSON of node data
        graph_json = json.dumps(nodes_ser, sort_keys=True, default=str)
        graph_hash = hashlib.sha256(graph_json.encode()).hexdigest()

        # Serialize node_tags
        tags_ser = {
            nid: tags.model_dump()
            for nid, tags in self.node_tags.items()
        }

        return {
            "graph_id": self.graph_id,
            "graph_sha256": graph_hash,
            "all_linear": self.all_linear,
            "x_shape": list(self.x_shape),
            "y_shape": list(self.y_shape),
            "n_nodes": len(self.forward_plan),
            "n_edges": len(self.edges),
            "nodes": nodes_ser,
            "edges": [{"source": s, "target": t} for s, t in self.edges],
            "blobs": all_blobs,
            "metadata": self.metadata,
            "learnable_params": self.learnable_params,
            "node_tags": tags_ser,
        }

    # ---- Check Adjoint ----

    def check_adjoint(
        self, n_trials: int = 3, rtol: float = 1e-4, seed: int = 0
    ) -> GraphAdjointCheckReport:
        """Verify <Ax, y> == <x, A^T y> for random vectors.

        Only meaningful for fully linear graphs.
        """
        if not self.all_linear:
            return GraphAdjointCheckReport(
                passed=False,
                n_trials=0,
                max_relative_error=float("inf"),
                mean_relative_error=float("inf"),
                tolerance=rtol,
                details=[{"note": "graph contains non-linear primitives"}],
            )

        rng = np.random.default_rng(seed)
        details: List[Dict[str, float]] = []
        max_err = 0.0

        for trial in range(n_trials):
            x = rng.standard_normal(self.x_shape).astype(np.float64)
            y = rng.standard_normal(self.y_shape).astype(np.float64)

            Ax = self.forward(x).astype(np.float64)
            ATy = self.adjoint(y).astype(np.float64)

            # Reshape for inner product if shapes mismatch
            inner_Ax_y = float(np.sum(Ax.ravel() * y.ravel()))
            inner_x_ATy = float(np.sum(x.ravel() * ATy.ravel()))

            denom = max(abs(inner_Ax_y), abs(inner_x_ATy), 1e-30)
            rel_err = abs(inner_Ax_y - inner_x_ATy) / denom
            max_err = max(max_err, rel_err)

            details.append(
                {
                    "trial": trial,
                    "inner_Ax_y": inner_Ax_y,
                    "inner_x_ATy": inner_x_ATy,
                    "rel_err": rel_err,
                }
            )

        mean_err = float(np.mean([d["rel_err"] for d in details]))

        report = GraphAdjointCheckReport(
            passed=max_err < rtol,
            n_trials=n_trials,
            max_relative_error=max_err,
            mean_relative_error=mean_err,
            tolerance=rtol,
            details=details,
        )

        if not report.passed:
            logger.warning(
                f"Graph adjoint check FAILED for {self.graph_id}: "
                f"{report.summary()}"
            )
        else:
            logger.debug(
                f"Graph adjoint check passed for {self.graph_id}: "
                f"{report.summary()}"
            )

        return report

    # ---- Explain ----

    def explain(self) -> str:
        """Return a deterministic text explanation of graph structure."""
        from pwm_core.graph.introspection import explain_graph

        return explain_graph(self)
