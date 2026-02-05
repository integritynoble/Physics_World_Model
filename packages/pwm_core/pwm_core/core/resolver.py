"""pwm_core.core.resolver

Resolve + validate ExperimentSpec:
- fill defaults
- normalize units (best-effort)
- enforce hard constraints
- auto-repair common issues (clamp values)

Conservative: never invent physics.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from pwm_core.api.types import ExperimentSpec, ValidationMessage, ValidationReport, Severity


def _add(report: ValidationReport, sev: Severity, code: str, msg: str, path: str | None = None) -> None:
    report.messages.append(ValidationMessage(severity=sev, code=code, message=msg, path=path))
    if sev == Severity.error:
        report.ok = False


def resolve_and_validate(spec: ExperimentSpec) -> Tuple[ExperimentSpec, ValidationReport]:
    """Return (spec_resolved, report)."""
    report = ValidationReport(ok=True)

    if spec.version != "0.2.1":
        _add(report, Severity.error, "version_mismatch", f"Expected spec version 0.2.1, got {spec.version}", "version")
        return spec, report

    if not spec.states or not spec.states.physics:
        _add(report, Severity.error, "missing_physics", "states.physics is required", "states.physics")
    if not spec.states or not spec.states.task:
        _add(report, Severity.error, "missing_task", "states.task is required", "states.task")

    if spec.input.mode == "measured":
        if not spec.input.y_source:
            _add(report, Severity.error, "missing_y_source", "input.y_source is required for measured mode", "input.y_source")
        if spec.input.operator is None:
            _add(report, Severity.error, "missing_operator", "input.operator is required for measured mode", "input.operator")

    patch: Dict[str, Any] = {}
    budget = spec.states.budget
    if budget and budget.photon_budget:
        mp = budget.photon_budget.get("max_photons")
        if mp is not None:
            try:
                mpv = float(mp)
                if mpv <= 0:
                    patch.setdefault("states", {}).setdefault("budget", {}).setdefault("photon_budget", {})["max_photons"] = 1.0
                    _add(report, Severity.warning, "clamp_max_photons", "max_photons must be > 0; clamped to 1.0",
                         "states.budget.photon_budget.max_photons")
            except Exception:
                _add(report, Severity.error, "invalid_max_photons", "max_photons must be numeric", "states.budget.photon_budget.max_photons")

    sensor = spec.states.sensor
    if sensor and sensor.quantization_bits is not None and sensor.quantization_bits < 1:
        patch.setdefault("states", {}).setdefault("sensor", {})["quantization_bits"] = 8
        _add(report, Severity.warning, "clamp_quant_bits", "quantization_bits must be >=1; set to 8", "states.sensor.quantization_bits")

    if patch:
        report.auto_repair_patch = patch

    return spec, report
