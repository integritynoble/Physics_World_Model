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

from pwm_core.graph.ir_types import ParameterSpec, StrictBaseModel
from pwm_core.mismatch.parameterizations import ThetaSpace


# ---------------------------------------------------------------------------
# Primitive-aware parameter defaults
# ---------------------------------------------------------------------------

# Primitive-aware parameter defaults (instead of naive ±2x)
_PARAM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "sigma": {"lower": 0.1, "upper": 20.0, "prior": "log_normal", "units": "px",
              "parameterization": "log"},
    "gain": {"lower": 0.01, "upper": 100.0, "prior": "log_uniform", "units": "scalar",
             "parameterization": "log"},
    "quantum_efficiency": {"lower": 0.01, "upper": 1.0, "prior": "uniform", "units": "scalar"},
    "sensitivity": {"lower": 0.01, "upper": 100.0, "prior": "log_uniform", "units": "scalar",
                    "parameterization": "log"},
    "strength": {"lower": 0.01, "upper": 100.0, "prior": "log_uniform", "units": "scalar"},
    "dx": {"lower": -50.0, "upper": 50.0, "prior": "uniform", "units": "px"},
    "dy": {"lower": -50.0, "upper": 50.0, "prior": "uniform", "units": "px"},
    "dx0": {"lower": -50.0, "upper": 50.0, "prior": "uniform", "units": "px"},
    "dy0": {"lower": -50.0, "upper": 50.0, "prior": "uniform", "units": "px"},
    "theta": {"lower": -3.14159, "upper": 3.14159, "prior": "uniform", "units": "rad"},
    "disp_step": {"lower": 0.1, "upper": 10.0, "prior": "uniform", "units": "px/band"},
}


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
        """Convert to ThetaSpace, honoring parameterization transforms."""
        ts_params: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.params.items():
            if spec.parameterization == "log" and spec.lower > 0:
                import math as _math
                ts_params[name] = {
                    "type": "float",
                    "low": _math.log(max(spec.lower, 1e-30)),
                    "high": _math.log(max(spec.upper, 1e-30)),
                    "unit": spec.units,
                    "parameterization": "log",
                }
            elif spec.parameterization == "logit" and 0 < spec.lower and spec.upper < 1:
                import math as _math
                ts_params[name] = {
                    "type": "float",
                    "low": _math.log(spec.lower / (1 - spec.lower)),
                    "high": _math.log(spec.upper / (1 - spec.upper)),
                    "unit": spec.units,
                    "parameterization": "logit",
                }
            else:
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

    Priority:
    1. Use GraphNode.parameter_specs if populated (from spec)
    2. Fall back to primitive-aware defaults (not naive ±2x)
    """
    import math
    params: Dict[str, ParameterSpec] = {}
    theta: Dict[str, float] = {}

    # Try to get parameter_specs from the original spec
    node_param_specs: Dict[str, List[ParameterSpec]] = {}
    if hasattr(graph_op, 'spec') and graph_op.spec is not None:
        for node in graph_op.spec.nodes:
            if node.parameter_specs:
                node_param_specs[node.node_id] = list(node.parameter_specs)

    for node_id, prim in graph_op.forward_plan:
        learnable = graph_op.learnable_params.get(node_id, [])

        # Check if this node has explicit parameter_specs
        explicit_specs = {ps.name: ps for ps in node_param_specs.get(node_id, [])}

        for param_name in learnable:
            if param_name in prim._params:
                val = prim._params[param_name]
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue

                key = f"{node_id}.{param_name}"

                # Priority 1: explicit parameter_specs from graph spec
                if param_name in explicit_specs:
                    ps = explicit_specs[param_name]
                    params[key] = ParameterSpec(
                        name=key,
                        lower=ps.lower,
                        upper=ps.upper,
                        prior=ps.prior,
                        parameterization=ps.parameterization,
                        identifiability_hint=ps.identifiability_hint,
                        drift_model=ps.drift_model,
                        units=ps.units,
                    )
                # Priority 2: primitive-aware defaults
                elif param_name in _PARAM_DEFAULTS:
                    defaults = _PARAM_DEFAULTS[param_name]
                    params[key] = ParameterSpec(
                        name=key,
                        lower=defaults["lower"],
                        upper=defaults["upper"],
                        prior=defaults.get("prior", "uniform"),
                        parameterization=defaults.get("parameterization", "identity"),
                        identifiability_hint="unknown",
                        units=defaults.get("units", "dimensionless"),
                    )
                # Priority 3: generic positive-definite defaults
                else:
                    abs_val = abs(fval) if abs(fval) > 1e-6 else 1.0
                    if fval > 0:
                        lower = max(fval / 10.0, 1e-6)
                        upper = fval * 10.0
                        prior = "log_uniform"
                    else:
                        lower = fval - abs_val * 2.0
                        upper = fval + abs_val * 2.0
                        prior = "uniform"
                    params[key] = ParameterSpec(
                        name=key,
                        lower=lower,
                        upper=upper,
                        prior=prior,
                        parameterization="identity",
                        identifiability_hint="unknown",
                    )
                theta[key] = fval

    return BeliefState(params=params, theta=theta)
