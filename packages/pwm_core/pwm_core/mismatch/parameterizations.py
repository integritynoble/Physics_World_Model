"""pwm_core.mismatch.parameterizations

Defines parameter spaces (theta) for each operator/modality.

Goal:
- Provide a *bounded*, valid search space for fitting theta
- Avoid LLM hallucinations by enforcing ranges + units

This is referenced by calibrators.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ThetaSpace:
    name: str
    # Each entry: key -> {"low":..., "high":..., "type":"float|int|enum", "unit":...}
    params: Dict[str, Dict[str, Any]]

    def clamp(self, theta: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, spec in self.params.items():
            if k not in theta:
                continue
            v = theta[k]
            if spec.get("type", "float") in ("float", "int"):
                lo, hi = float(spec["low"]), float(spec["high"])
                vv = max(lo, min(hi, float(v)))
                out[k] = int(round(vv)) if spec.get("type") == "int" else vv
            elif spec.get("type") == "enum":
                choices = list(spec["choices"])
                out[k] = v if v in choices else choices[0]
        return out


def cassi_theta_space() -> ThetaSpace:
    return ThetaSpace(
        name="cassi_theta_v1",
        params={
            "dx0": {"type": "float", "low": -5.0, "high": 5.0, "unit": "px"},
            "dy0": {"type": "float", "low": -5.0, "high": 5.0, "unit": "px"},
            "disp_poly_x_0": {"type": "float", "low": -10.0, "high": 10.0, "unit": "px"},
            "disp_poly_x_1": {"type": "float", "low": -5.0, "high": 5.0, "unit": "px/band"},
            "disp_poly_x_2": {"type": "float", "low": -1.0, "high": 1.0, "unit": "px/band^2"},
            "L": {"type": "int", "low": 2, "high": 64, "unit": "bands"},
        },
    )


def generic_gain_shift_space() -> ThetaSpace:
    return ThetaSpace(
        name="generic_gain_shift_v1",
        params={
            "gain": {"type": "float", "low": 0.1, "high": 10.0, "unit": "scalar"},
            "bias": {"type": "float", "low": -1.0, "high": 1.0, "unit": "scalar"},
        },
    )
