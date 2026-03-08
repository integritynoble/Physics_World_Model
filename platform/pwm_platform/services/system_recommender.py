"""PWM-SyS System Recommender — feasibility gate, TNA scoring, Pareto ranking.

Implements the three benchmark tasks from the PWM-SyS proposal:
  Task 1: Constrained System Retrieval
  Task 2: System + Solver Recommendation
  Task 3: Co-Design Proposal (forward to LLM with catalog context)

All three are unified as Mode 1 in SpecLab: Prompt → Spec → Simulate.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Load system catalog at module level ──────────────────────────────────

_CATALOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "static" / "benchmark-data" / "system_catalog.json"
)

_CATALOG: dict[str, dict[str, Any]] = {}


def _ensure_catalog() -> dict[str, dict[str, Any]]:
    global _CATALOG
    if not _CATALOG:
        try:
            with open(_CATALOG_PATH) as f:
                _CATALOG = json.load(f)
            logger.info("Loaded system catalog: %d modalities", len(_CATALOG))
        except Exception as exc:
            logger.error("Failed to load system catalog: %s", exc)
            _CATALOG = {}
    return _CATALOG


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class TaskQuery:
    """User's system requirement query."""
    purpose: str = ""
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_objectives: dict[str, Any] = field(default_factory=dict)
    latency_budget_s: Optional[float] = None
    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskQuery":
        return cls(
            purpose=d.get("purpose", ""),
            hard_constraints=d.get("hard_constraints", {}),
            soft_objectives=d.get("soft_objectives", {}),
            latency_budget_s=d.get("latency_budget_s"),
            weights=d.get("weights", {}),
        )


@dataclass
class AdequacyScores:
    """8-dimension TNA scores for a system against a task query."""
    D1_acquisition: float = 5.0
    D2_temporal: float = 5.0
    D3_spatial: float = 5.0
    D4_observable: float = 5.0
    D5_recovery: float = 5.0
    D6_budget: float = 5.0
    D7_deployment: float = 5.0
    D8_sample: float = 5.0

    def to_dict(self) -> dict[str, float]:
        return {
            "D1": round(self.D1_acquisition, 1),
            "D2": round(self.D2_temporal, 1),
            "D3": round(self.D3_spatial, 1),
            "D4": round(self.D4_observable, 1),
            "D5": round(self.D5_recovery, 1),
            "D6": round(self.D6_budget, 1),
            "D7": round(self.D7_deployment, 1),
            "D8": round(self.D8_sample, 1),
        }

    def values(self) -> list[float]:
        return [
            self.D1_acquisition, self.D2_temporal, self.D3_spatial,
            self.D4_observable, self.D5_recovery, self.D6_budget,
            self.D7_deployment, self.D8_sample,
        ]

    def weighted_sum(self, weights: dict[str, float]) -> float:
        default_w = 1.0 / 8
        d = self.to_dict()
        return sum(weights.get(k, default_w) * v for k, v in d.items())


@dataclass
class SystemResult:
    """One system's evaluation result."""
    system_id: str
    display_name: str
    category: str
    feasibility: str  # "PASS" or "FAIL"
    fail_reasons: list[str] = field(default_factory=list)
    scores: AdequacyScores = field(default_factory=AdequacyScores)
    recommended_solver: str = ""
    solver_detail: dict[str, Any] = field(default_factory=dict)
    pareto_optimal: bool = False
    preference_rank: int = 0
    hardware_summary: str = ""


@dataclass
class RecommendationResult:
    """Full recommendation output for a task query."""
    query: TaskQuery
    total_systems: int = 0
    feasible_count: int = 0
    pareto_count: int = 0
    results: list[SystemResult] = field(default_factory=list)
    solver_operating_points: list[dict] = field(default_factory=list)


# ── Operator skill ordering ─────────────────────────────────────────────

_SKILL_ORDER = {"untrained": 0, "technician": 1, "expert": 2, "specialist": 3}


def _skill_level(s: str) -> int:
    return _SKILL_ORDER.get(s, 3)


# ── Feasibility Gate ────────────────────────────────────────────────────


def feasibility_gate(system: dict, query: TaskQuery) -> tuple[bool, list[str]]:
    """Check hard constraints. Returns (passed, list_of_fail_reasons)."""
    fails = []
    hc = query.hard_constraints

    # Budget constraint
    if "budget_usd" in hc:
        cap = system.get("capital_cost_k_usd", 0) * 1000
        limit = hc["budget_usd"]
        if isinstance(limit, str):
            limit = float(limit.replace("<=", "").replace(",", "").strip())
        if cap > limit:
            fails.append(f"Capital ${cap/1000:.0f}k > budget ${limit/1000:.0f}k")

    # Spatial resolution
    if "spatial_resolution_um" in hc:
        sys_res = system.get("spatial_resolution_um", 1e6)
        req = hc["spatial_resolution_um"]
        if isinstance(req, str):
            req = float(req.replace("<=", "").strip())
        if sys_res > req:
            fails.append(f"Resolution {sys_res} µm > required {req} µm")

    # Temporal resolution (fps)
    if "temporal_resolution_fps" in hc:
        sys_fps = system.get("max_fps", 0)
        req = hc["temporal_resolution_fps"]
        if isinstance(req, str):
            req = float(req.replace(">=", "").replace(",", "").replace("_", "").strip())
        if sys_fps < req:
            fails.append(f"FPS {sys_fps:,.0f} < required {req:,.0f}")

    # Single-shot / acquisition mode
    if "acquisition_mode" in hc:
        mode = hc["acquisition_mode"]
        if mode in ("single-shot", "single_shot"):
            shots = system.get("shots_per_datacube", 1)
            if shots > 1:
                fails.append(f"Requires {shots} shots, not single-shot")

    # Sample contact
    if "sample_contact" in hc:
        req_contact = hc["sample_contact"]
        sys_contact = system.get("sample_contact", False)
        if isinstance(req_contact, bool) and req_contact is False and sys_contact:
            fails.append("System requires sample contact")

    # In-vivo capability
    if "in_vivo_capable" in hc:
        if hc["in_vivo_capable"] and not system.get("in_vivo_capable", False):
            fails.append("System not in-vivo capable")

    # Operator skill ceiling
    if "operator_skill" in hc:
        req_skill = hc["operator_skill"]
        if isinstance(req_skill, str):
            req_skill = req_skill.replace("<=", "").strip().strip('"').strip("'")
        sys_skill = system.get("operator_skill", "specialist")
        if _skill_level(sys_skill) > _skill_level(req_skill):
            fails.append(f"Requires {sys_skill}, ceiling is {req_skill}")

    # Non-destructive
    if hc.get("non_destructive") or hc.get("acquisition_mode") == "non-destructive":
        if system.get("sample_destructive", False):
            fails.append("System is destructive")

    # Reconstruction latency
    if "max_recon_time_s" in hc:
        solver_lat = system.get("solver_latency_s", 0)
        if solver_lat > hc["max_recon_time_s"]:
            fails.append(f"Solver latency {solver_lat}s > max {hc['max_recon_time_s']}s")

    return len(fails) == 0, fails


# ── TNA Scoring ─────────────────────────────────────────────────────────


def _tna_score(value: float, r_min: float, r_target: float, r_comfort: float,
               higher_is_better: bool = True) -> float:
    """Compute Task-Normalized Adequacy on [0, 10] scale."""
    if higher_is_better:
        if value < r_min:
            return 0.0
        if value >= r_comfort:
            return 10.0
        if value >= r_target:
            return 5.0 + 5.0 * (value - r_target) / max(r_comfort - r_target, 1e-12)
        return 5.0 * (value - r_min) / max(r_target - r_min, 1e-12)
    else:
        # Lower is better (e.g., cost, resolution)
        if value > r_min:
            return 0.0
        if value <= r_comfort:
            return 10.0
        if value <= r_target:
            return 5.0 + 5.0 * (r_target - value) / max(r_target - r_comfort, 1e-12)
        return 5.0 * (r_min - value) / max(r_min - r_target, 1e-12)


def compute_tna(system: dict, query: TaskQuery) -> AdequacyScores:
    """Compute 8-dimension TNA scores for a system against a task query."""
    scores = AdequacyScores()
    hc = query.hard_constraints
    so = query.soft_objectives

    # D1: Acquisition Feasibility
    shots = system.get("shots_per_datacube", 1)
    if hc.get("acquisition_mode") in ("single-shot", "single_shot"):
        scores.D1_acquisition = 10.0 if shots == 1 else 2.0
    else:
        scores.D1_acquisition = _tna_score(1.0 / max(shots, 1), 0, 0.01, 1.0, higher_is_better=True)
        scores.D1_acquisition = max(scores.D1_acquisition, 5.0)  # most systems are fine

    # D2: Temporal Adequacy
    sys_fps = system.get("max_fps", 1)
    if "temporal_resolution_fps" in hc:
        req_fps = float(str(hc["temporal_resolution_fps"]).replace(">=", "").replace(",", "").replace("_", "").strip())
        scores.D2_temporal = _tna_score(
            math.log10(max(sys_fps, 0.001)),
            math.log10(max(req_fps * 0.1, 0.001)),
            math.log10(max(req_fps, 0.001)),
            math.log10(max(req_fps * 10, 0.001)),
            higher_is_better=True,
        )
    else:
        scores.D2_temporal = min(10.0, 5.0 + math.log10(max(sys_fps, 1)) * 0.5)

    # D3: Spatial Adequacy
    sys_res = system.get("spatial_resolution_um", 1000)
    if "spatial_resolution_um" in hc:
        req_res = float(str(hc["spatial_resolution_um"]).replace("<=", "").strip())
        scores.D3_spatial = _tna_score(sys_res, req_res * 2, req_res, req_res * 0.5,
                                        higher_is_better=False)
    else:
        scores.D3_spatial = min(10.0, _tna_score(
            math.log10(max(1.0 / max(sys_res, 0.0001), 0.001)),
            -3, 0, 3, higher_is_better=True,
        ))

    # D4: Observable Sufficiency
    dims = system.get("output_dimensionality", "2D")
    dim_score = 5.0
    if "3D" in dims or "4D" in dims:
        dim_score = 8.0
    if "+spec" in dims or "+λ" in dims or "hyperspectral" in str(so):
        dim_score = 9.0
    if "+phase" in dims or "+chem" in dims:
        dim_score = 8.5
    scores.D4_observable = dim_score

    # D5: Output Recovery Quality (modality-normalized)
    best_psnr = system.get("best_psnr_db", 25)
    worst_psnr = system.get("worst_psnr_db", 15)
    psnr_range = best_psnr - worst_psnr
    # Score based on how good the best solver is relative to range
    scores.D5_recovery = min(10.0, 5.0 + (best_psnr - 30) * 0.5)
    scores.D5_recovery = max(0.0, scores.D5_recovery)

    # D6: Budget Feasibility
    cap_k = system.get("capital_cost_k_usd", 100)
    if "budget_usd" in hc:
        budget_k = float(str(hc["budget_usd"]).replace("<=", "").replace(",", "").strip()) / 1000
        scores.D6_budget = _tna_score(cap_k, budget_k, budget_k * 0.5, budget_k * 0.1,
                                       higher_is_better=False)
    else:
        scores.D6_budget = _tna_score(cap_k, 10000, 500, 10, higher_is_better=False)

    # D7: Deployment Burden
    skill = system.get("operator_skill", "specialist")
    skill_scores = {"untrained": 10, "technician": 8, "expert": 5, "specialist": 3}
    d7_skill = skill_scores.get(skill, 3)

    solver_lat = system.get("solver_latency_s", 10)
    if solver_lat <= 0.1:
        d7_speed = 10
    elif solver_lat <= 10:
        d7_speed = 8
    elif solver_lat <= 60:
        d7_speed = 6
    elif solver_lat <= 600:
        d7_speed = 4
    else:
        d7_speed = 2

    scores.D7_deployment = min(d7_skill, d7_speed)

    # D8: Sample Compatibility
    d8 = 10.0
    if system.get("sample_destructive", False):
        d8 -= 5.0
    if system.get("sample_contact", False):
        if hc.get("sample_contact") is False:
            d8 -= 5.0
        else:
            d8 -= 2.0
    if hc.get("in_vivo_capable") and not system.get("in_vivo_capable", False):
        d8 -= 5.0
    scores.D8_sample = max(0.0, d8)

    return scores


# ── Pareto Ranking ──────────────────────────────────────────────────────


def _dominates(a: list[float], b: list[float]) -> bool:
    """True if a Pareto-dominates b (>= all, > at least one)."""
    at_least_one_strictly_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            at_least_one_strictly_better = True
    return at_least_one_strictly_better


def pareto_rank(results: list[SystemResult]) -> list[SystemResult]:
    """Mark Pareto-optimal systems and sort by preference score."""
    n = len(results)
    scores_list = [r.scores.values() for r in results]

    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and _dominates(scores_list[j], scores_list[i]):
                dominated = True
                break
        results[i].pareto_optimal = not dominated

    # Sort: Pareto-optimal first, then by weighted sum
    results.sort(key=lambda r: (not r.pareto_optimal, -sum(r.scores.values())))

    for rank, r in enumerate(results, 1):
        r.preference_rank = rank

    return results


# ── Main recommendation pipeline ────────────────────────────────────────


def recommend(query: TaskQuery, max_results: int = 20) -> RecommendationResult:
    """Run the full recommendation pipeline: filter → score → rank."""
    catalog = _ensure_catalog()

    result = RecommendationResult(
        query=query,
        total_systems=len(catalog),
    )

    all_results: list[SystemResult] = []

    for sys_id, system in catalog.items():
        passed, fail_reasons = feasibility_gate(system, query)

        sr = SystemResult(
            system_id=sys_id,
            display_name=system.get("display_name", sys_id),
            category=system.get("category", "unknown"),
            feasibility="PASS" if passed else "FAIL",
            fail_reasons=fail_reasons,
            recommended_solver=system.get("solver_name", ""),
            hardware_summary=_build_hardware_summary(system),
        )

        if passed:
            sr.scores = compute_tna(system, query)
            sr.solver_detail = {
                "name": system.get("solver_name", ""),
                "type": system.get("solver_type", ""),
                "latency_s": system.get("solver_latency_s", 0),
                "psnr_db": system.get("best_psnr_db", 0),
                "ssim": system.get("best_ssim", 0),
                "num_algorithms": system.get("num_algorithms_in_catalog", 0),
                "algorithm_types": system.get("algorithm_type_coverage", []),
            }

        all_results.append(sr)

    # Separate feasible and infeasible
    feasible = [r for r in all_results if r.feasibility == "PASS"]
    infeasible = [r for r in all_results if r.feasibility == "FAIL"]

    result.feasible_count = len(feasible)

    # Pareto rank feasible systems
    if feasible:
        feasible = pareto_rank(feasible)
        result.pareto_count = sum(1 for r in feasible if r.pareto_optimal)

    # Combine: feasible (ranked) + top infeasible (sorted by # fail reasons)
    infeasible.sort(key=lambda r: len(r.fail_reasons))

    result.results = feasible[:max_results] + infeasible[:5]

    # Build solver operating points for the top system
    if feasible:
        top_sys = catalog.get(feasible[0].system_id, {})
        result.solver_operating_points = _get_solver_operating_points(top_sys)

    return result


def _build_hardware_summary(system: dict) -> str:
    """One-line hardware summary."""
    parts = []
    shots = system.get("shots_per_datacube", 1)
    if shots == 1:
        parts.append("single-shot")
    else:
        parts.append(f"{shots} shots")

    fps = system.get("max_fps", 0)
    if fps >= 1e9:
        parts.append(f"{fps/1e9:.0f} Gfps")
    elif fps >= 1e6:
        parts.append(f"{fps/1e6:.0f} Mfps")
    elif fps >= 1e3:
        parts.append(f"{fps/1e3:.0f} kfps")
    elif fps > 0:
        parts.append(f"{fps:.0f} fps")

    res = system.get("spatial_resolution_um", 0)
    if res > 0:
        if res < 0.001:
            parts.append(f"{res*1e6:.1f} pm")
        elif res < 1:
            parts.append(f"{res*1000:.0f} nm")
        elif res < 1000:
            parts.append(f"{res:.0f} µm")
        else:
            parts.append(f"{res/1000:.1f} mm")

    cap = system.get("capital_cost_k_usd", 0)
    if cap > 0:
        if cap >= 1000:
            parts.append(f"${cap/1000:.0f}M")
        else:
            parts.append(f"${cap:.0f}k")

    return " · ".join(parts)


def _get_solver_operating_points(system: dict) -> list[dict]:
    """Get solver comparison at different latency budgets."""
    algo_types = system.get("algorithm_type_coverage", [])
    solver_lat = system.get("solver_latency_s", 10)
    best_psnr = system.get("best_psnr_db", 30)

    points = []

    # Real-time point
    if solver_lat <= 0.1 or "Deep Learning" in algo_types:
        points.append({
            "latency_label": "< 100 ms (real-time)",
            "solver": "Fast CNN / E2E",
            "psnr_estimate": round(best_psnr - 3.0, 1),
            "speed_score": 9,
            "robustness_note": "Lower robustness, fast inference",
        })

    # Interactive point
    if "Deep Learning" in algo_types or "PnP" in algo_types:
        points.append({
            "latency_label": "< 10 s (interactive)",
            "solver": system.get("solver_name", "DL-Solver"),
            "psnr_estimate": round(best_psnr - 1.0, 1),
            "speed_score": 7,
            "robustness_note": "Good trade-off",
        })

    # Offline point
    if "Diffusion" in algo_types or "PnP" in algo_types:
        points.append({
            "latency_label": "< 5 min (offline)",
            "solver": "PnP / Diffusion",
            "psnr_estimate": round(best_psnr, 1),
            "speed_score": 3,
            "robustness_note": "Highest quality, slower",
        })

    # Classical fallback
    if "Classical" in algo_types:
        points.append({
            "latency_label": "Any (robust baseline)",
            "solver": "Classical (no training data)",
            "psnr_estimate": round(best_psnr - 8.0, 1),
            "speed_score": 5,
            "robustness_note": "Most robust, no learned components",
        })

    return points


# ── Query parsing from natural language ──────────────────────────────────


def parse_system_query(text: str) -> Optional[TaskQuery]:
    """Try to detect a system-level query from user text.

    Returns TaskQuery if the text looks like a system recommendation request,
    None if it's a regular spec-building prompt.
    """
    lower = text.lower()

    # Keywords that signal a system-level query
    system_keywords = [
        "which system", "which imaging", "what system", "what imaging",
        "recommend a system", "recommend an imaging", "compare systems",
        "best system for", "best imaging for", "system selection",
        "system design", "co-design", "which modality", "what modality",
        "budget", "within budget", "under $", "cost less than",
        "single-shot", "non-destructive", "in-vivo",
        "which should i", "what should i buy", "what should i build",
        "system comparison", "compare imaging",
        "for my application", "for my lab", "for clinical",
    ]

    if any(kw in lower for kw in system_keywords):
        return _extract_constraints_from_text(text)

    return None


def _extract_constraints_from_text(text: str) -> TaskQuery:
    """Best-effort extraction of constraints from natural language."""
    import re

    query = TaskQuery(purpose=text)
    lower = text.lower()

    # Budget extraction
    budget_match = re.search(r'\$\s*([\d,]+)\s*k', lower)
    if budget_match:
        query.hard_constraints["budget_usd"] = int(budget_match.group(1).replace(",", "")) * 1000
    else:
        budget_match = re.search(r'budget[^$]*\$\s*([\d,]+)', lower)
        if budget_match:
            val = int(budget_match.group(1).replace(",", ""))
            query.hard_constraints["budget_usd"] = val

    # Resolution extraction
    res_match = re.search(r'(\d+)\s*[uμµ]m\s*resolution', lower)
    if res_match:
        query.hard_constraints["spatial_resolution_um"] = int(res_match.group(1))
    res_match = re.search(r'(\d+)\s*nm\s*resolution', lower)
    if res_match:
        query.hard_constraints["spatial_resolution_um"] = int(res_match.group(1)) / 1000

    # FPS extraction
    fps_match = re.search(r'(\d+)\s*(m|g|k)?fps', lower)
    if fps_match:
        val = int(fps_match.group(1))
        mult = fps_match.group(2)
        if mult == "g":
            val *= 1e9
        elif mult == "m":
            val *= 1e6
        elif mult == "k":
            val *= 1e3
        query.hard_constraints["temporal_resolution_fps"] = val

    # Single-shot
    if "single-shot" in lower or "single shot" in lower:
        query.hard_constraints["acquisition_mode"] = "single-shot"

    # Sample constraints
    if "non-contact" in lower or "non contact" in lower or "no contact" in lower:
        query.hard_constraints["sample_contact"] = False
    if "in-vivo" in lower or "in vivo" in lower:
        query.hard_constraints["in_vivo_capable"] = True
    if "non-destructive" in lower or "non destructive" in lower or "ndt" in lower:
        query.hard_constraints["non_destructive"] = True

    # Operator skill
    if "untrained" in lower or "push-button" in lower:
        query.hard_constraints["operator_skill"] = "untrained"
    elif "technician" in lower:
        query.hard_constraints["operator_skill"] = "technician"

    return query
