"""pwm_core.graph.executor
===========================

GraphExecutor: unified execution engine for Mode S (Simulate), Mode I (Invert),
and Mode C (Calibrate) on a canonical operator graph.

The executor operates on a compiled GraphOperator that follows the canonical
Source -> Element(s) -> Sensor -> Noise chain.

Execution modes
---------------
Mode S (simulate)    x -> forward all nodes -> y_clean -> noise -> y
Mode I (invert)      y -> strip noise -> build A -> solve -> x_hat
Mode C (calibrate)   (x, y) -> fit theta -> update belief -> optionally invert

Operator correction mode: when ``config.provided_operator`` is set, Mode I
uses it instead of the graph-derived forward operator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.core.enums import ExecutionMode
from pwm_core.graph.graph_operator import GraphOperator
from pwm_core.graph.ir_types import NodeRole
from pwm_core.graph.primitives import BasePrimitive, PRIMITIVE_REGISTRY
from pwm_core.mismatch.belief_state import BeliefState, build_belief_from_graph
from pwm_core.objectives.base import ObjectiveSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExecutionConfig:
    """Configuration for GraphExecutor.execute()."""

    mode: ExecutionMode = ExecutionMode.simulate
    seed: int = 0
    add_noise: bool = True
    solver_ids: List[str] = field(default_factory=lambda: ["lsq", "fista"])
    max_iter: int = 100
    objective_spec: Optional[ObjectiveSpec] = None
    calibration_config: Optional[Dict[str, Any]] = None
    provided_operator: Optional[Any] = None  # PhysicsOperator for correction mode


@dataclass
class ExecutionResult:
    """Result from GraphExecutor.execute()."""

    mode: ExecutionMode
    y: Optional[np.ndarray] = None
    x_recon: Optional[np.ndarray] = None
    belief_state: Optional[BeliefState] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GraphExecutor
# ---------------------------------------------------------------------------


class GraphExecutor:
    """Unified execution engine for canonical operator graphs.

    Parameters
    ----------
    graph : GraphOperator
        Compiled graph operator following the canonical chain.
    """

    def __init__(self, graph: GraphOperator) -> None:
        self._graph = graph
        self._belief_state = build_belief_from_graph(graph)

        # Identify noise node (last in forward plan with _node_role="noise")
        self._noise_idx: Optional[int] = None
        self._noise_prim: Optional[BasePrimitive] = None
        for i, (node_id, prim) in enumerate(graph.forward_plan):
            role = getattr(prim, "_node_role", None)
            if role == "noise":
                self._noise_idx = i
                self._noise_prim = prim

    @property
    def belief_state(self) -> BeliefState:
        return self._belief_state

    def execute(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        config: Optional[ExecutionConfig] = None,
    ) -> ExecutionResult:
        """Dispatch to Mode S / I / C based on config."""
        cfg = config or ExecutionConfig()

        if cfg.mode == ExecutionMode.simulate:
            return self._simulate(x, cfg)
        elif cfg.mode == ExecutionMode.invert:
            return self._invert(y, cfg)
        elif cfg.mode == ExecutionMode.calibrate:
            return self._calibrate(x, y, cfg)
        else:
            raise ValueError(f"Unknown execution mode: {cfg.mode}")

    # ------------------------------------------------------------------
    # Mode S: Simulate
    # ------------------------------------------------------------------

    def _simulate(self, x: Optional[np.ndarray], config: ExecutionConfig) -> ExecutionResult:
        if x is None:
            raise ValueError("Mode S requires input x")

        current = x.copy().astype(np.float64)

        # Forward through all nodes except noise -> y_clean
        y_clean = current
        for i, (node_id, prim) in enumerate(self._graph.forward_plan):
            if i == self._noise_idx and not config.add_noise:
                continue
            if i == self._noise_idx:
                # Store y_clean before noise
                y_clean = current.copy()
            current = prim.forward(current)

        return ExecutionResult(
            mode=ExecutionMode.simulate,
            y=current,
            diagnostics={"y_clean": y_clean},
        )

    # ------------------------------------------------------------------
    # Mode I: Invert
    # ------------------------------------------------------------------

    def _invert(self, y: Optional[np.ndarray], config: ExecutionConfig) -> ExecutionResult:
        if y is None:
            raise ValueError("Mode I requires measurement y")

        # Build forward operator A (graph without noise)
        if config.provided_operator is not None:
            # Operator correction mode
            operator = config.provided_operator
        else:
            operator = _StrippedGraphOp(self._graph, self._noise_idx)

        # Infer objective from noise node type
        obj_spec = config.objective_spec or self._infer_objective_from_noise()

        # Run solver portfolio
        from pwm_core.recon.portfolio import run_portfolio
        recon_config = {
            "candidates": config.solver_ids,
            "max_candidates": len(config.solver_ids),
        }
        x_hat, recon_info = run_portfolio(y, operator, recon_config)

        metrics = {"solver": recon_info.get("solver", "unknown")}

        return ExecutionResult(
            mode=ExecutionMode.invert,
            x_recon=x_hat,
            metrics=metrics,
            diagnostics={"recon_info": recon_info, "objective_spec": obj_spec},
        )

    # ------------------------------------------------------------------
    # Mode C: Calibrate
    # ------------------------------------------------------------------

    def _calibrate(
        self,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        config: ExecutionConfig,
    ) -> ExecutionResult:
        if y is None:
            raise ValueError("Mode C requires measurement y")

        from pwm_core.mismatch.calibrators import calibrate, CalibConfig
        from pwm_core.graph.adapter import GraphOperatorAdapter

        # Build theta space from belief state
        theta_space = self._belief_state.to_theta_space()

        # Build adapter for forward function
        adapter = GraphOperatorAdapter(self._graph, modality="calibration")

        # Use x for calibration (or adjoint of y as proxy)
        if x is not None:
            calib_x = x
        else:
            try:
                calib_x = adapter.adjoint(y)
            except Exception:
                calib_x = np.zeros(self._graph.x_shape, dtype=np.float64)

        def forward_fn(theta):
            adapter.set_theta(theta)
            try:
                return adapter.forward(calib_x)
            except (ValueError, RuntimeError):
                return np.full_like(y, fill_value=1e6)

        # Build CalibConfig
        calib_params = config.calibration_config or {}
        calib_cfg = CalibConfig(
            num_candidates=calib_params.get("num_candidates", 16),
            num_refine_steps=calib_params.get("num_refine_steps", 5),
            max_evals=calib_params.get("max_evals", 200),
            seed=config.seed,
        )

        calib_result = calibrate(y, forward_fn, theta_space, calib_cfg)

        # Update belief state
        self._belief_state.update(calib_result.best_theta)

        # Optionally run Mode I with calibrated params
        x_recon = None
        if config.calibration_config and config.calibration_config.get("run_invert", False):
            adapter.set_theta(calib_result.best_theta)
            invert_result = self._invert(y, config)
            x_recon = invert_result.x_recon

        return ExecutionResult(
            mode=ExecutionMode.calibrate,
            x_recon=x_recon,
            belief_state=self._belief_state,
            metrics={
                "best_score": calib_result.best_score,
                "num_evals": calib_result.num_evals,
                "status": calib_result.status,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_objective_from_noise(self) -> ObjectiveSpec:
        """Inspect noise primitive_id to determine the correct NLL."""
        if self._noise_prim is None:
            return ObjectiveSpec(kind="gaussian")

        pid = self._noise_prim.primitive_id
        _NOISE_TO_OBJECTIVE = {
            "poisson_gaussian_sensor": "mixed_poisson_gaussian",
            "complex_gaussian_sensor": "complex_gaussian",
            "poisson_only_sensor": "poisson",
            # Legacy noise primitives
            "poisson_gaussian": "mixed_poisson_gaussian",
            "poisson": "poisson",
            "gaussian": "gaussian",
        }
        kind = _NOISE_TO_OBJECTIVE.get(pid, "gaussian")
        return ObjectiveSpec(kind=kind)


# ---------------------------------------------------------------------------
# _StrippedGraphOp: lightweight operator wrapping graph minus noise
# ---------------------------------------------------------------------------


class _StrippedGraphOp:
    """Graph operator with noise node stripped for Mode I.

    Implements forward() and adjoint() by skipping the noise node.
    """

    def __init__(self, graph: GraphOperator, noise_idx: Optional[int]) -> None:
        self._graph = graph
        self._noise_idx = noise_idx

        # Build stripped forward/adjoint plans
        self._fwd_plan = [
            (nid, prim) for i, (nid, prim) in enumerate(graph.forward_plan)
            if i != noise_idx
        ]
        self._adj_plan = list(reversed(self._fwd_plan))
        self._all_linear = all(p.is_linear for _, p in self._fwd_plan)

    def forward(self, x: np.ndarray) -> np.ndarray:
        current = x.copy()
        for node_id, prim in self._fwd_plan:
            current = prim.forward(current)
        return current

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        if not self._all_linear:
            raise RuntimeError("Adjoint requires all-linear graph (after noise strip)")
        current = y.copy()
        for node_id, prim in self._adj_plan:
            current = prim.adjoint(current)
        return current

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": f"{self._graph.graph_id}_stripped",
            "x_shape": list(self._graph.x_shape),
            "y_shape": list(self._graph.y_shape),
            "is_linear": self._all_linear,
            "n_nodes": len(self._fwd_plan),
        }

    @property
    def _x_shape(self):
        return self._graph.x_shape

    @property
    def _y_shape(self):
        return self._graph.y_shape

    @property
    def _is_linear(self):
        return self._all_linear
