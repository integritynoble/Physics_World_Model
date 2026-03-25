"""pwm_core.targeting.gates

R1-R4 operational run-gates — explicit pass/fail/warn verdicts.

(S1-S4 are reserved for the paper's formal scientific conditions and are
**not** checked here; see ``pwm_core.targeting.scientific_gates`` when it
lands.)

Each gate maps to an existing data source inside a RunBundle:

  R1 spec_completeness  — validation report written at run start
  R2 reproducibility    — provenance.json (seeds, git hash, pip freeze)
  R3 metric_integrity   — SHA-256 hashes on all stored artifacts
  R4 budget_compliance  — BudgetGuard timing log

The functions here formalise these checks into hard verdicts that can block
or flag certification.  Previously they produced only diagnostic data; now
each function returns a ``GateResult`` that the Certificate records.

Usage
-----
>>> from pwm_core.targeting.gates import run_r1_r4
>>> verdicts = run_r1_r4(bundle_dir, manifest, provenance, budget_log)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pwm_core.core.runbundle.certificate import GateResult, GateVerdict

logger = logging.getLogger(__name__)

# Maximum allowed budget overrun ratio before R4 fails (2× = disqualified)
BUDGET_FAIL_RATIO = 2.0
# Warn at 1.5× declared budget
BUDGET_WARN_RATIO = 1.5


# ---------------------------------------------------------------------------
# R1 — Spec completeness
# ---------------------------------------------------------------------------

def check_r1_spec_completeness(manifest: Dict[str, Any]) -> GateResult:
    """R1: Verify the spec is complete and all required fields are present.

    Parameters
    ----------
    manifest : dict
        Parsed ``runbundle_manifest.json`` dict.

    Returns
    -------
    GateResult
    """
    required_keys = {"version", "spec_id", "timestamp", "provenance", "metrics", "artifacts"}
    missing = required_keys - set(manifest.keys())
    if missing:
        return GateResult(
            verdict=GateVerdict.fail,
            message=f"Manifest missing required keys: {sorted(missing)}",
            details={"missing_keys": sorted(missing)},
        )

    provenance = manifest.get("provenance", {})
    prov_required = {"modality", "solver"}
    prov_missing = prov_required - set(provenance.keys())
    if prov_missing:
        return GateResult(
            verdict=GateVerdict.warn,
            message=f"Provenance missing fields: {sorted(prov_missing)}",
            details={"missing_provenance_keys": sorted(prov_missing)},
        )

    # ------------------------------------------------------------------
    # Second pass: canonical CoreSpec six-tuple validation (audit K2)
    #
    # The paper's canonical six-tuple is (Omega, E, B, I, O, epsilon).
    # If the manifest carries a "corespec" (or "spec") block we validate
    # that all six fields are present and that tolerance is numeric.
    # A missing corespec block is only a warning — backward-compatible
    # with v0.3.0 RunBundles that pre-date the six-tuple requirement.
    # ------------------------------------------------------------------
    CANONICAL_FIELDS = frozenset({
        "omega", "equations", "boundary_conditions",
        "initial_conditions", "observables", "tolerance",
    })

    corespec = manifest.get("corespec", manifest.get("spec", {}))
    if corespec:
        missing_canonical = CANONICAL_FIELDS - set(corespec.keys())
        if missing_canonical:
            return GateResult(
                verdict=GateVerdict.warn,
                message=f"CoreSpec six-tuple incomplete: missing {sorted(missing_canonical)}",
                details={"missing_canonical_fields": sorted(missing_canonical)},
            )
        # Validate tolerance is numeric
        tol = corespec.get("tolerance")
        if tol is not None and not isinstance(tol, (int, float)):
            return GateResult(
                verdict=GateVerdict.warn,
                message=f"CoreSpec tolerance must be numeric, got {type(tol).__name__}",
                details={"tolerance_type": type(tol).__name__},
            )
    else:
        # No corespec block — warn to encourage migration
        logger.debug(
            "R1: manifest has no 'corespec' block; "
            "six-tuple validation skipped (backward-compat mode)"
        )

    return GateResult(verdict=GateVerdict.pass_, message="Spec completeness check passed")


# ---------------------------------------------------------------------------
# R2 — Reproducibility
# ---------------------------------------------------------------------------

def check_r2_reproducibility(provenance: Dict[str, Any]) -> GateResult:
    """R2: Verify provenance contains enough information for reproduction.

    Parameters
    ----------
    provenance : dict
        Parsed ``provenance.json`` dict (or manifest['provenance']).

    Returns
    -------
    GateResult
    """
    issues = []

    git_hash = provenance.get("git_hash")
    if not git_hash or git_hash == "unknown":
        issues.append("git_hash missing or unknown")

    if provenance.get("git_dirty") is True:
        issues.append("git working tree was dirty at run time")

    seeds = provenance.get("seeds")
    if not seeds:
        issues.append("no random seeds recorded")

    python_version = provenance.get("python_version")
    if not python_version:
        issues.append("python_version not recorded")

    if issues:
        verdict = GateVerdict.warn if len(issues) == 1 else GateVerdict.fail
        return GateResult(
            verdict=verdict,
            message="Reproducibility issues: " + "; ".join(issues),
            details={"issues": issues},
        )

    return GateResult(verdict=GateVerdict.pass_, message="Reproducibility check passed")


# ---------------------------------------------------------------------------
# R3 — Metric integrity
# ---------------------------------------------------------------------------

def check_r3_metric_integrity(bundle_dir: Path, manifest: Dict[str, Any]) -> GateResult:
    """R3: Re-verify SHA-256 hashes for all artifacts listed in the manifest.

    Parameters
    ----------
    bundle_dir : Path
        Root directory of the RunBundle.
    manifest : dict
        Parsed ``runbundle_manifest.json``.

    Returns
    -------
    GateResult
    """
    hashes = manifest.get("hashes", {})
    artifacts = manifest.get("artifacts", {})
    failures = []
    checked = 0

    for key, rel_path in artifacts.items():
        expected_entry = hashes.get(key, "")
        if not expected_entry.startswith("sha256:"):
            continue  # no hash recorded for this artifact — skip
        expected_hex = expected_entry.removeprefix("sha256:")

        full_path = bundle_dir / rel_path
        if not full_path.exists():
            failures.append(f"{key}: file not found ({rel_path})")
            continue

        actual_hex = _sha256_file(full_path)
        if actual_hex != expected_hex:
            failures.append(f"{key}: hash mismatch (expected {expected_hex[:8]}… got {actual_hex[:8]}…)")
        else:
            checked += 1

    if failures:
        return GateResult(
            verdict=GateVerdict.fail,
            message=f"Hash verification failed for {len(failures)} artifact(s)",
            details={"failures": failures, "checked_ok": checked},
        )

    if checked == 0:
        return GateResult(
            verdict=GateVerdict.warn,
            message="No artifact hashes found to verify",
            details={"checked_ok": 0},
        )

    return GateResult(
        verdict=GateVerdict.pass_,
        message=f"All {checked} artifact hashes verified",
        details={"checked_ok": checked},
    )


# ---------------------------------------------------------------------------
# R4 — Budget compliance
# ---------------------------------------------------------------------------

def check_r4_budget_compliance(manifest: Dict[str, Any]) -> GateResult:
    """R4: Verify solver runtime stayed within declared compute budget.

    Parameters
    ----------
    manifest : dict
        Parsed ``runbundle_manifest.json``.

    Returns
    -------
    GateResult
    """
    metrics = manifest.get("metrics", {})
    runtime_s = metrics.get("runtime_s")

    if runtime_s is None:
        return GateResult(
            verdict=GateVerdict.warn,
            message="runtime_s not recorded in metrics — cannot verify budget compliance",
        )

    # Try to read declared budget from provenance
    provenance = manifest.get("provenance", {})
    declared_s = provenance.get("declared_budget_s")

    if declared_s is None:
        return GateResult(
            verdict=GateVerdict.pass_,
            message="No declared_budget_s in provenance; budget compliance not enforced",
            details={"runtime_s": runtime_s},
        )

    ratio = runtime_s / declared_s if declared_s > 0 else float("inf")

    if ratio >= BUDGET_FAIL_RATIO:
        return GateResult(
            verdict=GateVerdict.fail,
            message=(
                f"Budget exceeded: {runtime_s:.1f}s actual vs {declared_s:.1f}s "
                f"declared ({ratio:.1f}x ≥ {BUDGET_FAIL_RATIO}x disqualification threshold)"
            ),
            details={"runtime_s": runtime_s, "declared_s": declared_s, "ratio": ratio},
        )

    if ratio >= BUDGET_WARN_RATIO:
        return GateResult(
            verdict=GateVerdict.warn,
            message=(
                f"Budget warning: {runtime_s:.1f}s actual vs {declared_s:.1f}s "
                f"declared ({ratio:.1f}x ≥ {BUDGET_WARN_RATIO}x warning threshold)"
            ),
            details={"runtime_s": runtime_s, "declared_s": declared_s, "ratio": ratio},
        )

    return GateResult(
        verdict=GateVerdict.pass_,
        message=f"Budget compliant: {runtime_s:.1f}s / {declared_s:.1f}s ({ratio:.2f}x)",
        details={"runtime_s": runtime_s, "declared_s": declared_s, "ratio": ratio},
    )


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_r1_r4(
    bundle_dir: Path,
    manifest: Dict[str, Any],
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, GateResult]:
    """Run all four R-gates and return a dict of gate name → GateResult.

    Parameters
    ----------
    bundle_dir : Path
        Root directory of the RunBundle.
    manifest : dict
        Parsed ``runbundle_manifest.json``.
    provenance : dict, optional
        Parsed ``provenance.json``.  If None, falls back to ``manifest['provenance']``.

    Returns
    -------
    dict[str, GateResult]
        Keys: ``r1``, ``r2``, ``r3``, ``r4``.
    """
    if provenance is None:
        provenance = manifest.get("provenance", {})

    return {
        "r1": check_r1_spec_completeness(manifest),
        "r2": check_r2_reproducibility(provenance),
        "r3": check_r3_metric_integrity(bundle_dir, manifest),
        "r4": check_r4_budget_compliance(manifest),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Backward compatibility aliases (S->R rename per strategy S4)
# ---------------------------------------------------------------------------
check_s1_spec_completeness = check_r1_spec_completeness
check_s2_reproducibility = check_r2_reproducibility
check_s3_metric_integrity = check_r3_metric_integrity
check_s4_budget_compliance = check_r4_budget_compliance
run_s1_s4 = run_r1_r4
