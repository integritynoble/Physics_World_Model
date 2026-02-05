"""pwm_core.core.runner

Pipeline orchestrator - wires all components together.

Prompt/Spec -> Resolve -> Build operator(s) -> (Sim or Load) y -> Recon -> Diagnose -> Recommend -> RunBundle

This file provides the stable "spine" connecting:
- Physics factory (operator construction)
- Simulator (phantom generation, forward model, noise)
- Reconstruction portfolio (solver selection and execution)
- Analysis (metrics, residual tests, bottleneck classification, design advisor)
- RunBundle artifacts (saving results)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.api.types import (
    Action,
    ActionOp,
    CalibReconResult,
    CalibResult,
    DiagnosisResult,
    ExperimentSpec,
    InputMode,
    ReconResult,
    TaskKind,
)
from pwm_core.api.errors import PWMError
from pwm_core.core.runbundle.writer import write_runbundle_skeleton
from pwm_core.core.runbundle.provenance import capture_provenance
from pwm_core.core.runbundle.artifacts import save_artifacts, create_recon_result
from pwm_core.core.physics_factory import build_operator
from pwm_core.core.simulator import simulate_measurement
from pwm_core.physics.base import PhysicsOperator
from pwm_core.recon.portfolio import run_portfolio
from pwm_core.analysis.metrics import mse, psnr, no_reference_energy
from pwm_core.analysis.residual_tests import residual_diagnostics, as_dict as residual_as_dict
from pwm_core.analysis.bottleneck import classify_bottleneck
from pwm_core.analysis.design_advisor import recommend_actions


def _get_recon_config(spec: ExperimentSpec) -> Dict[str, Any]:
    """Extract reconstruction configuration from spec."""
    candidates = []

    if spec.recon.portfolio.solvers:
        for solver in spec.recon.portfolio.solvers:
            candidates.append(solver.id)

    # Default candidates if none specified
    if not candidates:
        candidates = ["lsq", "fista"]

    return {
        "candidates": candidates,
        "max_candidates": 3,
    }


def _load_measurements(spec: ExperimentSpec) -> np.ndarray:
    """Load measurements from file for measured mode."""
    if spec.input.y_source is None:
        raise PWMError("input.y_source is required for measured mode")

    source = spec.input.y_source

    if source.endswith('.npy'):
        return np.load(source)
    elif source.endswith('.npz'):
        data = np.load(source)
        # Try common key names
        for key in ['y', 'measurements', 'data', 'arr_0']:
            if key in data:
                return data[key]
        # Fall back to first array
        return data[list(data.keys())[0]]
    else:
        # Try loading as numpy array
        return np.load(source)


def _load_ground_truth(spec: ExperimentSpec) -> Optional[np.ndarray]:
    """Load optional ground truth for benchmarking."""
    if spec.input.x_source is None:
        return None

    source = spec.input.x_source

    if source.endswith('.npy'):
        return np.load(source)
    elif source.endswith('.npz'):
        data = np.load(source)
        for key in ['x', 'ground_truth', 'signal', 'arr_0']:
            if key in data:
                return data[key]
        return data[list(data.keys())[0]]
    else:
        return np.load(source)


def _compute_metrics(
    x_hat: np.ndarray,
    x_true: Optional[np.ndarray],
    y: np.ndarray,
    operator: PhysicsOperator,
) -> Dict[str, Any]:
    """Compute quality metrics for reconstruction."""
    metrics: Dict[str, Any] = {}

    # Reference metrics (if ground truth available)
    if x_true is not None:
        # Normalize for fair comparison
        x_hat_norm = x_hat.reshape(-1).astype(np.float32)
        x_true_norm = x_true.reshape(-1).astype(np.float32)

        # Handle shape mismatch
        if x_hat_norm.shape == x_true_norm.shape:
            # Scale to match ground truth range
            data_range = float(x_true_norm.max() - x_true_norm.min())
            if data_range < 1e-8:
                data_range = 1.0

            metrics["mse"] = mse(x_hat, x_true)
            metrics["psnr"] = psnr(x_hat, x_true, data_range=data_range)
        else:
            metrics["shape_mismatch"] = {
                "x_hat": list(x_hat.shape),
                "x_true": list(x_true.shape),
            }

    # No-reference metrics (always computed)
    metrics["energy"] = no_reference_energy(x_hat)

    # Compute residual
    try:
        y_hat = operator.forward(x_hat)
        residual = y.reshape(-1) - y_hat.reshape(-1)
        metrics["residual_norm"] = float(np.linalg.norm(residual))
        metrics["residual_rms"] = float(np.sqrt(np.mean(residual ** 2)))
    except Exception:
        # Forward may fail for shape mismatch; skip residual metrics
        pass

    return metrics


def _build_evidence(
    res_diag: Any,
    metrics: Dict[str, Any],
    spec: ExperimentSpec,
) -> Dict[str, Any]:
    """Build evidence dictionary for bottleneck classification."""
    evidence: Dict[str, Any] = {}

    # From residual diagnostics
    evidence["residual_rms"] = res_diag.rms
    evidence["autocorr1"] = res_diag.autocorr1
    evidence["hf_energy_ratio"] = res_diag.hf_energy_ratio

    # From metrics
    if "residual_rms" in metrics:
        evidence["residual_rms"] = metrics["residual_rms"]

    # From sensor state (saturation risk)
    sensor = spec.states.sensor
    if sensor is not None and sensor.saturation_level is not None:
        evidence["saturation_risk"] = 0.1  # Placeholder; could compute from y stats

    # From budget state (sampling rate)
    budget = spec.states.budget
    if budget is not None and budget.measurement_budget is not None:
        mb = budget.measurement_budget
        if isinstance(mb, dict):
            sr = mb.get("sampling_rate", 1.0)
            evidence["sampling_rate"] = float(sr) if sr is not None else 1.0

    # Drift detection placeholder
    evidence["drift_detected"] = 0.0

    return evidence


def _convert_advisor_actions(advisor_result: Any) -> list:
    """Convert design_advisor Actions to API Action objects."""
    api_actions = []

    for action in advisor_result.suggested_actions:
        api_action = Action(
            knob=action.knob,
            op=ActionOp.set if action.op == "set" else (
                ActionOp.multiply if action.op == "multiply" else (
                    ActionOp.add if action.op == "add" else (
                        ActionOp.optimize if action.op == "optimize" else ActionOp.set
                    )
                )
            ),
            val=action.val,
            reason=getattr(action, 'rationale', None),
        )
        api_actions.append(api_action)

    return api_actions


def run_pipeline(spec: ExperimentSpec, out_dir: Optional[str] = None) -> CalibReconResult:
    """Run the full PWM pipeline: simulate/load -> reconstruct -> analyze -> save.

    Args:
        spec: ExperimentSpec defining the experiment.
        out_dir: Output directory for RunBundle. Defaults to current directory.

    Returns:
        CalibReconResult with reconstruction results, metrics, and diagnosis.
    """
    out_dir = out_dir or os.getcwd()

    # 1. Create RunBundle skeleton
    rb_dir = write_runbundle_skeleton(out_dir, spec_id=spec.id)
    capture_provenance(rb_dir)

    task = spec.states.task.kind
    res = CalibReconResult(spec_id=spec.id, runbundle_path=rb_dir)

    # 2. Build physics operator
    operator = build_operator(spec)

    # 3. Get or simulate measurements
    x_true: Optional[np.ndarray] = None

    if spec.input.mode == InputMode.simulate:
        x_true, y = simulate_measurement(spec, operator)
    else:
        # Measured mode: load from files
        y = _load_measurements(spec)
        x_true = _load_ground_truth(spec)

    # 4. Handle calibration tasks
    if task in (TaskKind.fit_operator_only, TaskKind.calibrate_and_reconstruct):
        # Placeholder for actual calibration
        res.calib = CalibResult(
            operator_id=operator.info().get("operator_id", "unknown"),
            theta_best=operator.get_theta(),
            best_score=0.0,
            num_evals=0,
            logs=[],
        )

        if task == TaskKind.fit_operator_only:
            # Only fit operator, no reconstruction
            res.diagnosis = DiagnosisResult(
                verdict="calibration_only",
                confidence=0.8,
                evidence={"task": "fit_operator_only"},
                suggested_actions=[],
            )
            return res

    # 5. Run reconstruction
    start_time = time.time()
    recon_config = _get_recon_config(spec)
    x_hat, recon_info = run_portfolio(y, operator, recon_config)
    runtime_s = time.time() - start_time
    recon_info["runtime_s"] = runtime_s

    # 6. Compute metrics
    metrics = _compute_metrics(x_hat, x_true, y, operator)

    # 7. Compute residual diagnostics
    try:
        y_hat = operator.forward(x_hat)
        residual = y.reshape(-1) - y_hat.reshape(-1)
    except Exception:
        # Fallback if forward fails
        residual = np.zeros_like(y.reshape(-1))

    res_diag = residual_diagnostics(residual)

    # 8. Build evidence and classify bottleneck
    evidence = _build_evidence(res_diag, metrics, spec)
    bottleneck_result = classify_bottleneck({"evidence": evidence})

    # 9. Get recommended actions
    advisor_result = recommend_actions(bottleneck_result)

    # 10. Build diagnosis result
    res.diagnosis = DiagnosisResult(
        verdict=advisor_result.verdict,
        confidence=advisor_result.confidence,
        bottleneck=bottleneck_result.get("verdict"),
        evidence=evidence,
        suggested_actions=_convert_advisor_actions(advisor_result),
    )

    # 11. Save artifacts
    save_artifacts(
        rb_dir=rb_dir,
        x_hat=x_hat,
        y=y,
        metrics=metrics,
        diagnosis=res.diagnosis,
        x_true=x_true,
        recon_info=recon_info,
    )

    # 12. Build recon result
    solver_id = recon_info.get("solver", "unknown")
    recon_result = create_recon_result(
        solver_id=solver_id,
        rb_dir=rb_dir,
        metrics=metrics,
        runtime_s=runtime_s,
    )
    res.recon = [recon_result]

    return res
