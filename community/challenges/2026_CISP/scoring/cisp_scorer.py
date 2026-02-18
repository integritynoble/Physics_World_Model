"""
CISP 2026 Scoring Module
=========================

Wraps the PWM targeting/scoring.py with CISP-specific weighting and rules.

CISP scoring:
- Per-track rho (primary metric)
- Anti-Goodhart: S_rank = 0.3 * S_retro + 0.7 * S_prospective
- Safety brakes (rho < 0.30 = blocked, budget > 2x = DQ)
- Cross-modal harmonic mean (Track 4)
- Clinical relevance bonus (Track 3)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# CISP track definitions
# ---------------------------------------------------------------------------

TRACKS = {
    "correct": {
        "modalities": ["cassi"],
        "primary_metric": "rho",
        "secondary_metrics": ["oracle_gap", "roic"],
        "weight": 0.35,
    },
    "temporal": {
        "modalities": ["cacti"],
        "primary_metric": "rho",
        "secondary_metrics": ["temporal_consistency", "roic"],
        "weight": 0.25,
    },
    "medical": {
        "modalities": ["mri", "ct"],
        "primary_metric": "rho",
        "secondary_metrics": ["ssim_roi", "safety_compliance"],
        "weight": 0.20,
    },
    "cross": {
        "modalities": ["cassi", "spc", "cacti", "widefield"],
        "primary_metric": "rho_cross",
        "secondary_metrics": ["n_above_050", "n_above_080"],
        "weight": 0.20,
    },
}

# Safety brake thresholds
SAFETY_BRAKES = {
    "min_rho": 0.30,
    "max_budget_ratio": 2.0,
    "max_uncertainty_deviation": 0.15,
    "max_reprojection_ratio": 3.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModalityScore:
    """Score for a single modality within a track."""
    modality: str
    rho: float
    oracle_gap: float
    roic: float
    psnr_i: float
    psnr_ii: float
    psnr_iii: float
    psnr_iv: float
    n_scenes: int
    safety_violations: list = field(default_factory=list)


@dataclass
class TrackScore:
    """Aggregated score for a CISP track."""
    track: str
    modality_scores: list  # List[ModalityScore]
    primary_score: float
    secondary_scores: dict
    penalties: dict = field(default_factory=dict)
    final_score: float = 0.0
    rank: Optional[int] = None
    disqualified: bool = False
    dq_reason: str = ""


@dataclass
class CISPSubmissionScore:
    """Complete CISP score for a submission."""
    team_id: str
    track_scores: dict  # track_name -> TrackScore
    composite_score: float = 0.0
    overall_rank: Optional[int] = None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def compute_rho(psnr_i: float, psnr_ii: float, psnr_iii: float) -> float:
    """Recovery ratio: rho = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II).

    Measures how much of the ideal performance gap is recovered by
    calibration. rho=1.0 means perfect calibration recovery.
    """
    denom = psnr_i - psnr_ii
    if abs(denom) < 1e-10:
        return 0.0
    return (psnr_iii - psnr_ii) / denom


def compute_oracle_gap(psnr_i: float, psnr_iii: float) -> float:
    """Oracle gap (dB): how far calibrated result is from ideal."""
    return psnr_i - psnr_iii


def compute_roic(psnr_gain_db: float, gpu_seconds: float) -> float:
    """Return on Invested Compute: dB recovered per GPU-hour."""
    gpu_hours = gpu_seconds / 3600.0
    if gpu_hours < 1e-10:
        return 0.0
    return psnr_gain_db / gpu_hours


def compute_harmonic_mean(values: list) -> float:
    """Harmonic mean of positive values. Returns 0 if any value is <= 0."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    n = len(values)
    return n / sum(1.0 / v for v in values)


def compute_rho_cross(modality_rhos: dict) -> float:
    """Cross-modal rho: harmonic mean of per-modality rho values.

    Penalizes solvers that fail on any single modality.
    """
    values = list(modality_rhos.values())
    return compute_harmonic_mean(values)


# ---------------------------------------------------------------------------
# Anti-Goodhart scoring
# ---------------------------------------------------------------------------


def apply_anti_goodhart(retro_score: float, prospective_score: float,
                        diagnostics: dict) -> tuple:
    """Apply anti-Goodhart scoring.

    S_rank = 0.3 * S_retro + 0.7 * S_prospective - penalties

    Returns (final_score, penalties_dict).
    """
    penalties = {}

    # Gaming penalty: wrong Triad attribution
    if diagnostics.get("wrong_triad_attribution", False):
        penalties["wrong_triad"] = -0.15

    # Overconfident uncertainty
    coverage = diagnostics.get("uncertainty_coverage", 0.90)
    if coverage < 0.75:
        penalties["overconfident_uncertainty"] = -0.10

    # Identifiability inconsistency
    if diagnostics.get("identifiability_inconsistent", False):
        penalties["identifiability"] = -0.10

    # Compute dishonesty
    budget_ratio = diagnostics.get("budget_ratio", 1.0)
    if budget_ratio > SAFETY_BRAKES["max_budget_ratio"]:
        penalties["compute_dishonesty"] = "DQ"
        return 0.0, penalties

    total_penalty = sum(v for v in penalties.values() if isinstance(v, (int, float)))
    raw = 0.3 * retro_score + 0.7 * prospective_score
    final = max(0.0, raw + total_penalty)
    return final, penalties


# ---------------------------------------------------------------------------
# Safety brakes
# ---------------------------------------------------------------------------


def check_safety_brakes(score: ModalityScore) -> list:
    """Check safety brake conditions. Returns list of violations."""
    violations = []

    if score.rho < SAFETY_BRAKES["min_rho"]:
        violations.append({
            "brake": "recovery_ratio_regression",
            "threshold": SAFETY_BRAKES["min_rho"],
            "actual": score.rho,
            "action": "block_deployment",
        })

    return violations


# ---------------------------------------------------------------------------
# Track scoring
# ---------------------------------------------------------------------------


def score_track(track: str, modality_scores: list,
                diagnostics: dict = None) -> TrackScore:
    """Score a complete CISP track."""
    diagnostics = diagnostics or {}
    track_def = TRACKS[track]

    # Primary metric
    if track == "cross":
        rhos = {ms.modality: ms.rho for ms in modality_scores}
        primary = compute_rho_cross(rhos)
    else:
        primary = sum(ms.rho for ms in modality_scores) / max(len(modality_scores), 1)

    # Secondary metrics
    secondary = {}
    if track == "cross":
        secondary["n_above_050"] = sum(1 for ms in modality_scores if ms.rho > 0.50)
        secondary["n_above_080"] = sum(1 for ms in modality_scores if ms.rho > 0.80)
    else:
        secondary["oracle_gap_mean"] = (
            sum(ms.oracle_gap for ms in modality_scores)
            / max(len(modality_scores), 1)
        )
        secondary["roic_mean"] = (
            sum(ms.roic for ms in modality_scores)
            / max(len(modality_scores), 1)
        )

    # Check for DQ conditions
    all_violations = []
    for ms in modality_scores:
        violations = check_safety_brakes(ms)
        all_violations.extend(violations)

    # Anti-Goodhart (use primary as retro, prospective from diagnostics)
    retro = primary
    prospective = diagnostics.get("prospective_score", primary)
    final, penalties = apply_anti_goodhart(retro, prospective, diagnostics)

    dq = any(v == "DQ" for v in penalties.values())

    return TrackScore(
        track=track,
        modality_scores=modality_scores,
        primary_score=primary,
        secondary_scores=secondary,
        penalties=penalties,
        final_score=final if not dq else 0.0,
        disqualified=dq,
        dq_reason="Compute dishonesty (>2x declared budget)" if dq else "",
    )


def score_submission(team_id: str, track_scores: dict) -> CISPSubmissionScore:
    """Score a complete CISP submission across tracks."""
    composite = 0.0
    for track_name, ts in track_scores.items():
        if not ts.disqualified:
            weight = TRACKS.get(track_name, {}).get("weight", 0.25)
            composite += weight * ts.final_score

    return CISPSubmissionScore(
        team_id=team_id,
        track_scores=track_scores,
        composite_score=composite,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_runbundle_scores(runbundle_path: Path) -> dict:
    """Load scores from a RunBundle's metrics.json."""
    metrics_path = runbundle_path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json in {runbundle_path}")
    with open(metrics_path) as f:
        return json.load(f)


def export_scores(submission_score: CISPSubmissionScore, output_path: Path):
    """Export CISP scores to JSON."""
    result = {
        "team_id": submission_score.team_id,
        "composite_score": submission_score.composite_score,
        "overall_rank": submission_score.overall_rank,
        "tracks": {},
    }
    for track_name, ts in submission_score.track_scores.items():
        result["tracks"][track_name] = {
            "primary_score": ts.primary_score,
            "secondary_scores": ts.secondary_scores,
            "penalties": {
                k: str(v) for k, v in ts.penalties.items()
            },
            "final_score": ts.final_score,
            "disqualified": ts.disqualified,
            "dq_reason": ts.dq_reason,
            "rank": ts.rank,
            "modalities": [
                {
                    "modality": ms.modality,
                    "rho": ms.rho,
                    "oracle_gap": ms.oracle_gap,
                    "roic": ms.roic,
                    "psnr_i": ms.psnr_i,
                    "psnr_ii": ms.psnr_ii,
                    "psnr_iii": ms.psnr_iii,
                    "psnr_iv": ms.psnr_iv,
                }
                for ms in ts.modality_scores
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
