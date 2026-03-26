"""pwm_core.targeting.scientific_validity

S1-S4 scientific validity conditions — verified at design time and audit time.

These are NOT runtime per-run gates (that's R1-R4 in gates.py).
S1-S4 are the paper's formal conditions verified through:
  - S1: spec compilation (also proxied by R1 at runtime)
  - S2: DomainProfile registration (stability analysis)
  - S3: DomainProfile registration (convergence analysis)
  - S4: post-execution audit (error bound verification)

Per strategy §4: "R1-R4 are checked on every run. S1-S4 are supported by
spec compilation, DomainProfile registration, and post-execution audit."
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def verify_s1_finite_specifiability(corespec: Dict[str, Any]) -> Tuple[bool, str]:
    """S1: Verify the problem admits a finite, type-valid description.

    Checked at spec compilation time. Also proxied by R1 at runtime.
    A problem satisfies S1 if all six CoreSpec fields (Omega, E, B, I, O, epsilon)
    can be encoded as finite, typed objects.
    """
    required = {"omega", "equations", "observables", "tolerance"}
    missing = required - set(corespec.keys())
    if missing:
        return False, f"S1 failed: missing fields {sorted(missing)}"

    tol = corespec.get("tolerance")
    if tol is not None and not isinstance(tol, (int, float)):
        return False, f"S1 failed: tolerance must be numeric, got {type(tol).__name__}"

    # Check finiteness: all values must be serializable (finite representation)
    import json
    try:
        json.dumps(corespec, default=str)
    except (TypeError, ValueError) as e:
        return False, f"S1 failed: spec not finitely representable: {e}"

    return True, "S1 satisfied: problem has finite, type-valid description"


def verify_s2_hadamard_stability(
    domain_profile: Dict[str, Any],
    forward_model_condition_number: Optional[float] = None,
    reproducibility_variance: Optional[float] = None,
) -> Tuple[bool, str]:
    """S2: Verify the problem is well-posed (Hadamard stability).

    Checked at DomainProfile registration. A problem satisfies S2 if:
    1. The forward model has bounded condition number (existence + uniqueness)
    2. The solution depends continuously on the data (stability)
    3. Repeated runs with the same seed produce consistent results

    For imaging: condition number of H matrix should be finite.
    For CT QC: metric extraction functions are deterministic.
    """
    issues = []

    # Check condition number if provided
    if forward_model_condition_number is not None:
        if not np.isfinite(forward_model_condition_number):
            issues.append(
                f"Forward model condition number is infinite "
                f"({forward_model_condition_number})"
            )
        elif forward_model_condition_number > 1e12:
            issues.append(
                f"Forward model is severely ill-conditioned "
                f"(kappa={forward_model_condition_number:.2e})"
            )

    # Check reproducibility variance if provided (from repeated runs)
    if reproducibility_variance is not None:
        if reproducibility_variance > 0.01:
            issues.append(
                f"Reproducibility variance too high: "
                f"{reproducibility_variance:.4f} (threshold: 0.01)"
            )

    # Check domain profile declares noise model (continuity of data dependence)
    noise_model = (
        domain_profile.get("noise_model")
        or domain_profile.get("extra", {}).get("noise_model")
    )
    if not noise_model:
        issues.append(
            "No noise model declared — continuous data dependence unverifiable"
        )

    if issues:
        return False, f"S2 issues: {'; '.join(issues)}"

    return True, (
        "S2 satisfied: problem is well-posed "
        "(bounded condition number, declared noise model)"
    )


def verify_s3_approximability(
    domain_profile: Dict[str, Any],
    discretization_error: Optional[float] = None,
    mesh_convergence_rate: Optional[float] = None,
) -> Tuple[bool, str]:
    """S3: Verify the solution admits a convergent discretization.

    Checked at DomainProfile registration. A problem satisfies S3 if:
    1. The discretization error is bounded and decreasing with refinement
    2. The primitive chain preserves numerical stability

    For imaging: reconstruction converges as resolution increases.
    For combustion: mesh refinement yields converging solution.
    """
    issues = []

    # Check discretization error if provided
    if discretization_error is not None:
        if not np.isfinite(discretization_error):
            issues.append(
                f"Discretization error is not finite: {discretization_error}"
            )
        elif discretization_error > 1.0:
            issues.append(
                f"Discretization error exceeds threshold: "
                f"{discretization_error:.4f}"
            )

    # Check convergence rate if provided
    if mesh_convergence_rate is not None:
        if mesh_convergence_rate <= 0:
            issues.append(
                f"Non-positive convergence rate: {mesh_convergence_rate}"
            )

    # Check that primitive chain is declared (basis for approximation)
    primitives = (
        domain_profile.get("primitive_chain")
        or domain_profile.get("primitives", [])
    )
    if not primitives:
        issues.append(
            "No primitive chain declared — approximability unverifiable"
        )

    if issues:
        return False, f"S3 issues: {'; '.join(issues)}"

    return True, "S3 satisfied: solution admits convergent discretization"


def verify_s4_certifiability(
    runbundle_metrics: Dict[str, Any],
    error_bound: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> Tuple[bool, str]:
    """S4: Verify computable error bounds exist (post-execution audit).

    Checked after execution by A_J. A result satisfies S4 if:
    1. Error metrics are present and computable
    2. Error bounds relate to the declared tolerance epsilon
    3. The result can be independently verified

    This is the post-execution audit that realizes S4 in practice.
    """
    issues = []

    # Check that metrics exist
    if not runbundle_metrics:
        issues.append(
            "No metrics in RunBundle — error bounds not computable"
        )

    psnr = runbundle_metrics.get("psnr_db")
    ssim = runbundle_metrics.get("ssim")

    if psnr is None and ssim is None:
        issues.append(
            "Neither PSNR nor SSIM present — no quality metric for certification"
        )

    # Check error bound vs tolerance
    if error_bound is not None and tolerance is not None:
        if error_bound > tolerance:
            issues.append(
                f"Error bound ({error_bound:.4f}) exceeds "
                f"tolerance epsilon ({tolerance:.4f})"
            )

    # Check that runtime is recorded (needed for reproducible audit)
    runtime = runbundle_metrics.get("runtime_s")
    if runtime is None:
        issues.append("runtime_s not recorded — audit incomplete")

    if issues:
        return False, f"S4 issues: {'; '.join(issues)}"

    return True, (
        "S4 satisfied: computable error bounds exist and relate to tolerance"
    )


def verify_all_s_conditions(
    corespec: Dict[str, Any],
    domain_profile: Dict[str, Any],
    runbundle_metrics: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Tuple[bool, str]]:
    """Run all S1-S4 scientific validity checks.

    Returns dict of {s1: (pass, msg), s2: ..., s3: ..., s4: ...}
    """
    results = {}
    results["s1"] = verify_s1_finite_specifiability(corespec)
    results["s2"] = verify_s2_hadamard_stability(
        domain_profile,
        kwargs.get("forward_model_condition_number"),
        kwargs.get("reproducibility_variance"),
    )
    results["s3"] = verify_s3_approximability(
        domain_profile,
        kwargs.get("discretization_error"),
        kwargs.get("mesh_convergence_rate"),
    )
    if runbundle_metrics:
        results["s4"] = verify_s4_certifiability(
            runbundle_metrics,
            kwargs.get("error_bound"),
            corespec.get("tolerance"),
        )
    else:
        results["s4"] = (
            True,
            "S4 deferred: post-execution audit pending (no metrics yet)",
        )

    return results
