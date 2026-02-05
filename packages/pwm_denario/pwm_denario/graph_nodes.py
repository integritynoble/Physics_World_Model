"""pwm_denario.graph_nodes (optional)

Optional LangGraph-style nodes. You can ignore this file if not using LangGraph.
"""

from __future__ import annotations

from typing import Any, Dict

from pwm_denario.tools import pwm_run, pwm_fit_operator, pwm_calibrate_recon


def node_run(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = state.get("prompt")
    spec = state.get("spec")
    out_dir = state.get("out_dir", "runs")
    state["pwm_result"] = pwm_run(prompt=prompt, spec=spec, out_dir=out_dir)
    return state


def node_fit_operator(state: Dict[str, Any]) -> Dict[str, Any]:
    state["fit_result"] = pwm_fit_operator(state["y_path"], state["operator_id"], state.get("out_dir","runs"))
    return state


def node_calib_recon(state: Dict[str, Any]) -> Dict[str, Any]:
    state["calib_recon_result"] = pwm_calibrate_recon(state["y_path"], state["operator_id"], state.get("out_dir","runs"))
    return state
