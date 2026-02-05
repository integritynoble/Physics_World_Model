"""pwm_denario.policy_bridge

Maps DiagnosisResult.suggested_actions -> next ExperimentSpec patch.
"""

from __future__ import annotations

from typing import Any, Dict, List


def apply_actions_to_spec(spec: Dict[str, Any], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply structured actions to a spec dict.

    Actions are of the form:
      {"knob": "budget.photon_budget.max_photons", "op": "multiply", "val": 2.0}
    """
    out = dict(spec)

    def _get_set(root: Dict[str, Any], path: str):
        parts = path.split(".")
        cur = root
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
        return cur, parts[-1]

    for a in actions:
        knob = a.get("knob", "")
        op = a.get("op", "set")
        val = a.get("val")
        if not knob:
            continue
        tgt, leaf = _get_set(out, knob)
        if op == "set":
            tgt[leaf] = val
        elif op == "multiply":
            tgt[leaf] = float(tgt.get(leaf, 1.0)) * float(val)
        elif op == "add":
            tgt[leaf] = float(tgt.get(leaf, 0.0)) + float(val)
        elif op == "optimize":
            # mark as needing optimization; PWM core may interpret this
            tgt[leaf] = {"optimize": True, "hint": val}
        else:
            tgt[leaf] = val

    return out
