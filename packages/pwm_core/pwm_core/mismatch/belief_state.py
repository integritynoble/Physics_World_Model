"""pwm_core.mismatch.belief_state
==================================

BeliefState: structured container for mismatch parameter estimates,
uncertainty, history, and bounds.

Bridges to existing ThetaSpace / CalibConfig for the calibration loop.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pwm_core.graph.ir_types import ParameterSpec
from pwm_core.mismatch.parameterizations import ThetaSpace


# ---------------------------------------------------------------------------
# StrictBaseModel (local copy)
# ---------------------------------------------------------------------------


class StrictBaseModel(BaseModel):
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
# BeliefState
# ---------------------------------------------------------------------------


class BeliefState(StrictBaseModel):
    """Mismatch parameter belief state.

    Tracks current estimates, uncertainty, bounds, and update history
    for all learnable parameters in a graph.

    Attributes
    ----------
    params : dict
        Parameter name -> ParameterSpec (bounds, prior, drift model).
    theta : dict
        Current best estimate for each parameter.
    uncertainty : dict or None
        Optional per-parameter uncertainty estimates.
    history : list
        List of past theta snapshots (pushed on each update).
    """

    params: Dict[str, ParameterSpec] = Field(default_factory=dict)
    theta: Dict[str, float] = Field(default_factory=dict)
    uncertainty: Optional[Dict[str, float]] = None
    history: List[Dict[str, float]] = Field(default_factory=list)

    def update(
        self,
        new_theta: Dict[str, float],
        uncertainty: Optional[Dict[str, float]] = None,
    ) -> None:
        """Push current theta to history and set new estimate.

        Parameters
        ----------
        new_theta : dict
            New parameter estimates.
        uncertainty : dict, optional
            New uncertainty estimates.
        """
        if self.theta:
            self.history.append(dict(self.theta))
        self.theta = dict(new_theta)
        if uncertainty is not None:
            self.uncertainty = dict(uncertainty)

    def get_bounds(self, name: str) -> tuple:
        """Get (lower, upper) bounds for a named parameter.

        Returns
        -------
        tuple[float, float]
            Parameter bounds from ParameterSpec.

        Raises
        ------
        KeyError
            If parameter not found.
        """
        if name not in self.params:
            raise KeyError(f"Parameter '{name}' not in belief state")
        spec = self.params[name]
        return (spec.lower, spec.upper)

    def to_theta_space(self) -> ThetaSpace:
        """Convert to ThetaSpace for use with existing calibrators.

        Returns
        -------
        ThetaSpace
            Bounded search space derived from BeliefState params.
        """
        ts_params: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.params.items():
            ts_params[name] = {
                "type": "float",
                "low": spec.lower,
                "high": spec.upper,
                "unit": spec.units,
            }
        return ThetaSpace(name="belief_state_theta", params=ts_params)


# ---------------------------------------------------------------------------
# Factory: build from graph
# ---------------------------------------------------------------------------


def build_belief_from_graph(graph_op) -> BeliefState:
    """Extract a BeliefState from a compiled GraphOperator.

    Scans all nodes for learnable parameters and builds ParameterSpec
    entries with default bounds derived from current values.

    Parameters
    ----------
    graph_op : GraphOperator
        Compiled graph operator.

    Returns
    -------
    BeliefState
        Initial belief state with current theta values and default bounds.
    """
    params: Dict[str, ParameterSpec] = {}
    theta: Dict[str, float] = {}

    for node_id, prim in graph_op.forward_plan:
        # Check if this node has learnable params in the graph spec
        learnable = graph_op.learnable_params.get(node_id, [])
        for param_name in learnable:
            if param_name in prim._params:
                val = prim._params[param_name]
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue

                key = f"{node_id}.{param_name}"
                abs_val = abs(fval) if abs(fval) > 1e-6 else 1.0
                params[key] = ParameterSpec(
                    name=key,
                    lower=fval - abs_val * 2.0,
                    upper=fval + abs_val * 2.0,
                    prior="uniform",
                    parameterization="identity",
                    identifiability_hint="unknown",
                )
                theta[key] = fval

    return BeliefState(params=params, theta=theta)
