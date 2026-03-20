"""pwm_core.graph.adapter
=========================

GraphOperatorAdapter: wraps a compiled GraphOperator to provide the full
BaseOperator protocol (info, get_theta, set_theta, metadata, check_adjoint,
serialize).

This enables the graph-first Mode 1 pipeline where `build_operator()` returns
a GraphOperatorAdapter backed by the OperatorGraph IR, instead of a
modality-specific operator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.graph.graph_operator import GraphOperator
from pwm_core.physics.base import AdjointCheckReport, BaseOperator, OperatorMetadata

logger = logging.getLogger(__name__)


class GraphOperatorAdapter(BaseOperator):
    """Adapt a compiled GraphOperator to the full BaseOperator protocol.

    Delegates forward/adjoint to the underlying GraphOperator and synthesises
    info(), get_theta(), set_theta(), metadata(), check_adjoint(), and
    serialize() from graph metadata and node parameters.
    """

    def __init__(self, graph_op: GraphOperator, modality: str) -> None:
        self._graph_op = graph_op
        self._modality = modality

        # Collect theta from all graph nodes
        theta: Dict[str, Any] = {}
        for node_id, prim in graph_op.forward_plan:
            for k, v in prim._params.items():
                theta[f"{node_id}.{k}"] = v

        super().__init__(
            operator_id=graph_op.graph_id,
            theta=theta,
            _x_shape=graph_op.x_shape,
            _y_shape=graph_op.y_shape,
            _is_linear=graph_op.all_linear,
            _supports_autodiff=False,
        )

    @property
    def graph_op(self) -> GraphOperator:
        """Access the underlying GraphOperator."""
        return self._graph_op

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._graph_op.forward(x)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self._graph_op.adjoint(y)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self._graph_op.graph_id,
            "modality": self._modality,
            "n_nodes": len(self._graph_op.forward_plan),
            "n_edges": len(self._graph_op.edges),
            "all_linear": self._graph_op.all_linear,
            "x_shape": list(self._graph_op.x_shape),
            "y_shape": list(self._graph_op.y_shape),
            "backend": "graph_operator",
        }

    def get_theta(self) -> Dict[str, Any]:
        theta: Dict[str, Any] = {}
        for node_id, prim in self._graph_op.forward_plan:
            for k, v in prim._params.items():
                if isinstance(v, np.ndarray):
                    theta[f"{node_id}.{k}"] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    theta[f"{node_id}.{k}"] = v.item()
                else:
                    theta[f"{node_id}.{k}"] = v
        return theta

    def set_theta(self, theta: Dict[str, Any]) -> None:
        for key, val in theta.items():
            parts = key.split(".", 1)
            if len(parts) == 2:
                node_id, param_name = parts
                if node_id in self._graph_op.node_map:
                    self._graph_op.node_map[node_id]._params[param_name] = val
        self.theta = theta

    def serialize(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        result = self._graph_op.serialize(data_dir)
        result["modality"] = self._modality
        result["adapter"] = "GraphOperatorAdapter"
        return result

    def check_adjoint(
        self, n_trials: int = 3, tol: float = 1e-4, seed: int = 0
    ) -> AdjointCheckReport:
        graph_report = self._graph_op.check_adjoint(
            n_trials=n_trials, rtol=tol, seed=seed
        )
        return AdjointCheckReport(
            passed=graph_report.passed,
            n_trials=graph_report.n_trials,
            max_relative_error=graph_report.max_relative_error,
            mean_relative_error=graph_report.mean_relative_error,
            tolerance=graph_report.tolerance,
            details=graph_report.details,
        )

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality=self._modality,
            operator_id=self._graph_op.graph_id,
            x_shape=list(self._graph_op.x_shape),
            y_shape=list(self._graph_op.y_shape),
            is_linear=self._graph_op.all_linear,
            supports_autodiff=False,
        )
