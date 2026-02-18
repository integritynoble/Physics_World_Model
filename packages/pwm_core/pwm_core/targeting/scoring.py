"""pwm_core.targeting.scoring
==============================

Metrics + Anti-Goodhart scoring from ``docs/targeting_system.md`` S3, S5.

Frozen formulas (per Rail Constitution Article 1.3, 1.4):
- Recovery ratio:  rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)
- Oracle gap:      PSNR_I - PSNR_III
- RoIC:            dB recovered per GPU-hour
- Anti-Goodhart:   S_rank = 0.3 * S_retro + 0.7 * S_prospective
- Track weights:   0.35 / 0.20 / 0.25 / 0.20

Gaming penalties (S5):
- Wrong Triad attribution:  -15%
- Overconfident uncertainty: -10%
- Identifiability inconsistency: -10%
- Compute dishonesty:       DQ
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants (Rail Constitution Article 1.4)
# ---------------------------------------------------------------------------

TRACK_WEIGHTS = {
    "correct": 0.35,
    "diagnose": 0.20,
    "no_gt": 0.25,
    "design": 0.20,
}

ANTI_GOODHART_RETRO_WEIGHT = 0.3
ANTI_GOODHART_PROSPECTIVE_WEIGHT = 0.7

# Track 1 sub-weights (S3 of targeting_system.md)
TRACK1_SUBWEIGHTS = {
    "rho": 0.30,
    "param_recovery": 0.20,
    "uncertainty": 0.15,
    "tail_risk": 0.15,
    "cross_modality": 0.10,
    "roic": 0.10,
}


# ---------------------------------------------------------------------------
# Safety brake thresholds (Rail Constitution Article 1.5)
# ---------------------------------------------------------------------------

@dataclass
class SafetyBrakeViolation:
    """A triggered safety brake."""

    condition: str
    threshold: str
    actual_value: float
    action: str


SAFETY_BRAKES = [
    {"condition": "recovery_ratio_low", "threshold": 0.30, "action": "block"},
    {"condition": "uncertainty_miscalibration", "threshold": 0.15, "action": "flag"},
    {"condition": "wrong_gate", "threshold": None, "action": "retrain"},
    {"condition": "budget_exceeded", "threshold": 2.0, "action": "disqualify"},
    {"condition": "consistency_violation", "threshold": 3.0, "action": "quarantine"},
]


# ---------------------------------------------------------------------------
# Core metric functions (frozen per Rail Constitution)
# ---------------------------------------------------------------------------


def compute_recovery_ratio(
    psnr_i: float, psnr_ii: float, psnr_iii: float
) -> float:
    """Recovery ratio: rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II).

    Clamped to [0, 1+] range. Returns 0.0 if PSNR_I == PSNR_II (no gap).
    """
    denom = psnr_i - psnr_ii
    if abs(denom) < 1e-10:
        return 1.0 if abs(psnr_iii - psnr_i) < 1e-10 else 0.0
    rho = (psnr_iii - psnr_ii) / denom
    return float(rho)


def compute_oracle_gap(psnr_i: float, psnr_iii: float) -> float:
    """Oracle gap: PSNR_I - PSNR_III (dB remaining to perfect)."""
    return float(psnr_i - psnr_iii)


def compute_roic(psnr_gain_db: float, gpu_seconds: float) -> float:
    """Return on Imaging Compute: dB recovered per GPU-hour.

    Parameters
    ----------
    psnr_gain_db : float
        PSNR_III - PSNR_II (dB recovered by calibration).
    gpu_seconds : float
        Total GPU-seconds consumed (calibration + reconstruction).
    """
    if gpu_seconds <= 0:
        return float("inf") if psnr_gain_db > 0 else 0.0
    gpu_hours = gpu_seconds / 3600.0
    return float(psnr_gain_db / gpu_hours)


def compute_ofs(psnr_iii: float, psnr_i: float) -> float:
    """Operator Fidelity Score: PSNR_III / PSNR_I ratio."""
    if psnr_i <= 0:
        return 0.0
    return float(psnr_iii / psnr_i)


def compute_msi(psnr_drop_db: float, mismatch_magnitude: float) -> float:
    """Mask-Sensitivity Index: degradation per unit mismatch (dB/px)."""
    if abs(mismatch_magnitude) < 1e-10:
        return 0.0
    return float(abs(psnr_drop_db) / abs(mismatch_magnitude))


# ---------------------------------------------------------------------------
# Gaming penalty checks (S5 of targeting_system.md)
# ---------------------------------------------------------------------------


@dataclass
class GamingPenalty:
    """A detected gaming attempt."""

    check: str
    penalty_fraction: float
    reason: str


def check_gaming_penalties(
    diagnostics: Dict[str, Any],
) -> List[GamingPenalty]:
    """Check for gaming attempts per S5 of targeting_system.md.

    Parameters
    ----------
    diagnostics : dict
        Must contain keys like 'triad_attribution', 'uncertainty_coverage',
        'declared_budget_s', 'actual_budget_s'.
    """
    penalties: List[GamingPenalty] = []

    # Wrong Triad attribution
    if diagnostics.get("triad_attribution_correct") is False:
        penalties.append(GamingPenalty(
            check="wrong_triad",
            penalty_fraction=0.15,
            reason="Incorrect Triad gate attribution",
        ))

    # Overconfident uncertainty
    coverage = diagnostics.get("uncertainty_coverage")
    if coverage is not None and coverage < 0.75:
        penalties.append(GamingPenalty(
            check="overconfident_uncertainty",
            penalty_fraction=0.10,
            reason=f"Uncertainty coverage {coverage:.2f} < 0.75 at 90% CI",
        ))

    # Identifiability inconsistency
    if diagnostics.get("identifiability_consistent") is False:
        penalties.append(GamingPenalty(
            check="identifiability_inconsistency",
            penalty_fraction=0.10,
            reason="Claimed identifiability inconsistent with evidence",
        ))

    # Compute dishonesty
    declared = diagnostics.get("declared_budget_s", 0)
    actual = diagnostics.get("actual_budget_s", 0)
    if declared > 0 and actual > 0 and declared < 0.5 * actual:
        penalties.append(GamingPenalty(
            check="compute_dishonesty",
            penalty_fraction=1.0,
            reason=f"Declared {declared:.1f}s < 50% of actual {actual:.1f}s → DQ",
        ))

    return penalties


# ---------------------------------------------------------------------------
# Safety brake checks (S6.4 of targeting_system.md)
# ---------------------------------------------------------------------------


def check_safety_brakes(
    rho: float,
    budget_ratio: float = 1.0,
    uncertainty_deviation: float = 0.0,
    wrong_gate: bool = False,
    reprojection_error_ratio: float = 1.0,
) -> List[SafetyBrakeViolation]:
    """Check all 5 pre-committed safety thresholds."""
    violations: List[SafetyBrakeViolation] = []

    if rho < 0.30:
        violations.append(SafetyBrakeViolation(
            condition="recovery_ratio_low",
            threshold="rho < 0.30",
            actual_value=rho,
            action="block",
        ))

    if uncertainty_deviation > 0.15:
        violations.append(SafetyBrakeViolation(
            condition="uncertainty_miscalibration",
            threshold="deviation > 15%",
            actual_value=uncertainty_deviation,
            action="flag",
        ))

    if wrong_gate:
        violations.append(SafetyBrakeViolation(
            condition="wrong_gate",
            threshold="confident wrong diagnosis",
            actual_value=1.0,
            action="retrain",
        ))

    if budget_ratio > 2.0:
        violations.append(SafetyBrakeViolation(
            condition="budget_exceeded",
            threshold="> 2x declared",
            actual_value=budget_ratio,
            action="disqualify",
        ))

    if reprojection_error_ratio > 3.0:
        violations.append(SafetyBrakeViolation(
            condition="consistency_violation",
            threshold="re-projection error > 3x median",
            actual_value=reprojection_error_ratio,
            action="quarantine",
        ))

    return violations


# ---------------------------------------------------------------------------
# Scored result
# ---------------------------------------------------------------------------


@dataclass
class ScoredResult:
    """Aggregate scores for a harness run."""

    rho: float
    oracle_gap: float
    roic: float
    ofs: float
    track: str
    track_score: float
    penalties: List[GamingPenalty] = field(default_factory=list)
    safety_violations: List[SafetyBrakeViolation] = field(default_factory=list)
    final_score: float = 0.0
    disqualified: bool = False
    disqualification_reason: str = ""
    per_scene: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rho": self.rho,
            "oracle_gap": self.oracle_gap,
            "roic": self.roic,
            "ofs": self.ofs,
            "track": self.track,
            "track_score": self.track_score,
            "final_score": self.final_score,
            "disqualified": self.disqualified,
            "disqualification_reason": self.disqualification_reason,
            "penalties": [
                {"check": p.check, "penalty": p.penalty_fraction, "reason": p.reason}
                for p in self.penalties
            ],
            "safety_violations": [
                {"condition": v.condition, "threshold": v.threshold,
                 "actual": v.actual_value, "action": v.action}
                for v in self.safety_violations
            ],
        }


def score_run(
    psnr_i: float,
    psnr_ii: float,
    psnr_iii: float,
    psnr_iv: float,
    total_gpu_seconds: float,
    track: str = "correct",
    diagnostics: Optional[Dict[str, Any]] = None,
    mismatch_magnitude: float = 1.0,
) -> ScoredResult:
    """Compute all metrics and apply anti-Goodhart scoring.

    Parameters
    ----------
    psnr_i, psnr_ii, psnr_iii, psnr_iv : float
        Per-scenario PSNR values.
    total_gpu_seconds : float
        Total compute consumed.
    track : str
        Evaluation track ('correct', 'diagnose', 'no_gt', 'design').
    diagnostics : dict, optional
        Diagnostic data for gaming penalty checks.
    mismatch_magnitude : float
        Magnitude of injected mismatch (for MSI calculation).
    """
    rho = compute_recovery_ratio(psnr_i, psnr_ii, psnr_iii)
    oracle_gap = compute_oracle_gap(psnr_i, psnr_iii)
    psnr_gain = psnr_iii - psnr_ii
    roic = compute_roic(psnr_gain, total_gpu_seconds)
    ofs = compute_ofs(psnr_iii, psnr_i)

    # Track 1 score (simplified: rho-dominated)
    track_score = float(min(rho, 1.0))

    # Gaming penalties
    diagnostics = diagnostics or {}
    penalties = check_gaming_penalties(diagnostics)
    total_penalty = sum(p.penalty_fraction for p in penalties)

    disqualified = total_penalty >= 1.0
    dq_reason = ""
    if disqualified:
        dq_reasons = [p.reason for p in penalties if p.penalty_fraction >= 1.0]
        dq_reason = "; ".join(dq_reasons) if dq_reasons else "Cumulative penalties >= 100%"

    # Apply penalties
    penalized_score = track_score * (1.0 - min(total_penalty, 1.0))

    # Safety brakes
    budget_ratio = 1.0
    declared = diagnostics.get("declared_budget_s", 0)
    if declared > 0:
        budget_ratio = total_gpu_seconds / declared

    safety_violations = check_safety_brakes(
        rho=rho,
        budget_ratio=budget_ratio,
        uncertainty_deviation=diagnostics.get("uncertainty_deviation", 0.0),
        wrong_gate=diagnostics.get("wrong_gate", False),
        reprojection_error_ratio=diagnostics.get("reprojection_error_ratio", 1.0),
    )

    # DQ overrides
    for v in safety_violations:
        if v.action == "disqualify":
            disqualified = True
            dq_reason = f"Safety brake: {v.condition}"

    final_score = 0.0 if disqualified else penalized_score

    return ScoredResult(
        rho=rho,
        oracle_gap=oracle_gap,
        roic=roic,
        ofs=ofs,
        track=track,
        track_score=track_score,
        penalties=penalties,
        safety_violations=safety_violations,
        final_score=final_score,
        disqualified=disqualified,
        disqualification_reason=dq_reason,
    )
