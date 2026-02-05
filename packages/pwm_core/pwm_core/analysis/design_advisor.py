"""pwm_core.analysis.design_advisor

Turns diagnostics into structured suggested_actions:
- increase dose
- adjust sampling rate / frames
- run fit_operator for alignment/dispersion
- reduce saturation risk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Action:
    knob: str
    op: str  # "set" | "multiply" | "add" | "optimize"
    val: Any
    rationale: str = ""


@dataclass
class DiagnosisResult:
    verdict: str
    confidence: float
    evidence: Dict[str, Any]
    suggested_actions: List[Action]


def recommend_actions(diagnosis: Dict[str, Any]) -> DiagnosisResult:
    verdict = diagnosis.get("verdict", "Unknown")
    conf = float(diagnosis.get("confidence", 0.5))
    evidence = diagnosis.get("evidence", {})

    actions: List[Action] = []

    if verdict.lower().startswith("dose"):
        actions.append(Action(knob="budget.photon_budget.max_photons", op="multiply", val=2.0,
                              rationale="Increase photon budget to improve SNR."))
    if verdict.lower().startswith("sampling"):
        actions.append(Action(knob="budget.measurement_budget.sampling_rate", op="multiply", val=1.5,
                              rationale="Increase sampling rate / number of measurements to reduce ill-posedness."))
    if "drift" in verdict.lower():
        actions.append(Action(knob="sample.motion.drift", op="optimize", val=True,
                              rationale="Estimate/compensate drift during reconstruction or via operator-fit."))

    if not actions:
        actions.append(Action(knob="calibration", op="optimize", val=True, rationale="Run bounded calibration/refinement."))

    return DiagnosisResult(verdict=verdict, confidence=conf, evidence=evidence, suggested_actions=actions)
