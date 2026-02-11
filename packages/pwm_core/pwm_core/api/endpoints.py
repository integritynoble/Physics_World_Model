"""
pwm_core.api.endpoints

Stable entrypoints for:
- compile_prompt
- simulate
- reconstruct
- analyze
- export
- fit_operator
- calibrate_recon

These are "library-level endpoints" (callable from CLI, AI_Scientist, notebooks).
They should avoid global state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from .types import (
    CalibReconResult,
    CalibResult,
    DiagnosisResult,
    ExperimentSpec,
    ReconResult,
    ValidationReport,
)
from .errors import PWMError

# The actual implementations live in pwm_core.core.runner and helpers
from pwm_core.core.runner import run_pipeline
from pwm_core.core.resolver import resolve_and_validate
from pwm_core.core.prompt_compiler import compile_prompt_to_spec, CompileResult


@dataclass
class ResolveValidateResult:
    """Container for resolve_validate result."""
    spec_resolved: ExperimentSpec
    validation: ValidationReport


def _get_default_casepack_dir() -> str:
    """Return the default casepack directory path."""
    # Get the path relative to this file (endpoints.py is in pwm_core/api/)
    # Structure: packages/pwm_core/pwm_core/api/endpoints.py
    # Casepacks: packages/pwm_core/contrib/casepacks/
    this_file = os.path.abspath(__file__)
    api_dir = os.path.dirname(this_file)  # pwm_core/api
    pwm_core_pkg_dir = os.path.dirname(api_dir)  # pwm_core
    pwm_core_root = os.path.dirname(pwm_core_pkg_dir)  # packages/pwm_core
    return os.path.join(pwm_core_root, "contrib", "casepacks")


def compile_prompt(prompt: str, casepack_dir: Optional[str] = None) -> CompileResult:
    """Compile a natural-language prompt into a draft ExperimentSpec.

    Args:
        prompt: Natural language description of the experiment.
        casepack_dir: Directory containing CasePack JSON files.
                      If None, uses the default contrib/casepacks directory.

    Returns:
        CompileResult with casepack_id, draft_spec, assumptions, and patch_list.
    """
    if casepack_dir is None:
        casepack_dir = _get_default_casepack_dir()
    return compile_prompt_to_spec(prompt, casepack_dir)


def resolve_validate(spec: Union[ExperimentSpec, Dict[str, Any]]) -> ResolveValidateResult:
    """Resolve and validate a spec (dict or ExperimentSpec object).

    Returns ResolveValidateResult with spec_resolved and validation attributes.
    """
    # Convert dict to ExperimentSpec if needed
    if isinstance(spec, dict):
        spec = ExperimentSpec(**spec)

    spec_resolved, vreport = resolve_and_validate(spec)
    return ResolveValidateResult(spec_resolved=spec_resolved, validation=vreport)


def simulate(spec: ExperimentSpec, out_dir: Optional[str] = None) -> CalibReconResult:
    """Simulate y from PhysicsTrue (optionally with mismatch) and run recon/analyze."""
    spec_resolved, vreport = resolve_and_validate(spec)
    if not vreport.ok:
        # still allow if auto_repair_patch exists and caller wants to proceed;
        # by default raise for safety.
        raise PWMError(f"Spec validation failed: {vreport.messages}")
    return run_pipeline(spec_resolved, out_dir=out_dir)


def reconstruct(spec: ExperimentSpec, out_dir: Optional[str] = None) -> CalibReconResult:
    """Measured y + known operator -> reconstruct + analyze."""
    spec_resolved, vreport = resolve_and_validate(spec)
    if not vreport.ok:
        raise PWMError(f"Spec validation failed: {vreport.messages}")
    return run_pipeline(spec_resolved, out_dir=out_dir)


def analyze(spec: ExperimentSpec, out_dir: Optional[str] = None) -> DiagnosisResult:
    """Run diagnosis only. (Shortcut: calls pipeline and returns diagnosis.)"""
    res = run_pipeline(spec, out_dir=out_dir)
    if res.diagnosis is None:
        return DiagnosisResult(verdict="no_diagnosis", confidence=0.0)
    return res.diagnosis


def export_runbundle(spec: ExperimentSpec, out_dir: str) -> str:
    """Run pipeline and export a RunBundle. Returns runbundle path."""
    res = run_pipeline(spec, out_dir=out_dir)
    if not res.runbundle_path:
        raise PWMError("RunBundle path missing from result.")
    return res.runbundle_path


def fit_operator(spec: ExperimentSpec, out_dir: Optional[str] = None) -> CalibResult:
    """Measured y + parametric operator A(theta): fit theta only."""
    spec_resolved, vreport = resolve_and_validate(spec)
    if not vreport.ok:
        raise PWMError(f"Spec validation failed: {vreport.messages}")
    # runner returns CalibReconResult; caller wants calib
    res = run_pipeline(spec_resolved, out_dir=out_dir)
    if res.calib is None:
        raise PWMError("Calibration result missing (did you set task.kind=fit_operator_only?)")
    return res.calib


def calibrate_recon(spec: ExperimentSpec, out_dir: Optional[str] = None) -> CalibReconResult:
    """Measured y + A(theta) -> fit theta + reconstruct + diagnose + report."""
    spec_resolved, vreport = resolve_and_validate(spec)
    if not vreport.ok:
        raise PWMError(f"Spec validation failed: {vreport.messages}")
    return run_pipeline(spec_resolved, out_dir=out_dir)


def run(
    prompt: Optional[str] = None,
    spec: Optional[Union[ExperimentSpec, Dict[str, Any]]] = None,
    out_dir: Optional[str] = None,
    casepack_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full PWM pipeline from either a prompt or a spec.

    Args:
        prompt: Natural language description (mutually exclusive with spec).
        spec: ExperimentSpec dict or object (mutually exclusive with prompt).
        out_dir: Output directory for RunBundle.
        casepack_dir: Directory containing CasePack JSON files (for prompt mode).

    Returns:
        Dict with run results including runbundle_path, metrics, diagnosis.
    """
    if prompt is None and spec is None:
        raise PWMError("Either 'prompt' or 'spec' must be provided.")
    if prompt is not None and spec is not None:
        raise PWMError("Only one of 'prompt' or 'spec' should be provided, not both.")

    # Compile prompt to spec if needed
    if prompt is not None:
        compile_result = compile_prompt(prompt, casepack_dir)
        spec = compile_result.draft_spec

    # Convert dict to ExperimentSpec if needed
    if isinstance(spec, dict):
        spec = ExperimentSpec(**spec)

    # Resolve and validate
    spec_resolved, vreport = resolve_and_validate(spec)
    if not vreport.ok:
        # Log warnings but continue if auto-repair is possible
        pass

    # Run pipeline
    result = run_pipeline(spec_resolved, out_dir=out_dir)

    # Convert to dict for JSON serialization
    return {
        "spec_id": result.spec_id,
        "runbundle_path": result.runbundle_path,
        "diagnosis": {
            "verdict": result.diagnosis.verdict if result.diagnosis else None,
            "confidence": result.diagnosis.confidence if result.diagnosis else None,
            "bottleneck": result.diagnosis.bottleneck if result.diagnosis else None,
        } if result.diagnosis else None,
        "recon": [
            {
                "solver_id": r.solver_id,
                "metrics": r.metrics,
            }
            for r in (result.recon or [])
        ],
        "validation": {
            "ok": vreport.ok,
            "messages": [str(m) for m in vreport.messages],
        },
    }


def view(runbundle_dir: str) -> None:
    """Launch the Streamlit viewer for a RunBundle.

    Args:
        runbundle_dir: Path to the RunBundle directory.
    """
    import subprocess
    import sys

    # Get the path to the viewer app
    this_file = os.path.abspath(__file__)
    api_dir = os.path.dirname(this_file)
    pwm_core_pkg_dir = os.path.dirname(api_dir)
    viewer_app = os.path.join(pwm_core_pkg_dir, "viewer", "app.py")

    if not os.path.exists(viewer_app):
        raise PWMError(f"Viewer app not found at {viewer_app}")

    # Launch streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", viewer_app, "--", runbundle_dir]
    print(f"Launching viewer: {' '.join(cmd)}")
    subprocess.run(cmd)
