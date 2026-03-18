"""Expanded benchmark result types.

Provides ``CaseResult`` (per-case execution result) and
``ExpandedRunSummary`` (aggregate statistics across a batch of cases).
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaseResult:
    """Result of executing a single ``CaseInstance``."""

    # Identity
    case_id: str = ""
    modality_id: str = ""
    benchmark: str = ""       # B1, B2, B3, B4
    variant_id: str = ""
    status: str = "pending"   # pending | success | skipped | error
    error_message: str = ""

    # Common metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0

    # B1 (Design) fields
    b1_scores: Dict[str, Any] = field(default_factory=dict)

    # B2 (Forward + Reconstruct) fields
    per_algorithm: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mismatch_results: List[Dict[str, Any]] = field(default_factory=list)

    # B3 (System Identification) fields
    true_params: Dict[str, float] = field(default_factory=dict)
    estimated_params: Dict[str, float] = field(default_factory=dict)
    param_errors: Dict[str, float] = field(default_factory=dict)

    # B4 (Correct + Diagnose) fields
    rho: Optional[float] = None
    grid_search_result: Optional[Dict[str, Any]] = None
    psnr_improvement: Optional[float] = None

    # Metadata
    timestamp: str = ""
    run_id: str = ""
    image_size_id: str = ""
    noise_level_id: str = ""
    mismatch_level_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "case_id": self.case_id,
            "modality_id": self.modality_id,
            "benchmark": self.benchmark,
            "variant_id": self.variant_id,
            "status": self.status,
            "metrics": self.metrics,
            "wall_time_s": self.wall_time_s,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "image_size_id": self.image_size_id,
            "noise_level_id": self.noise_level_id,
            "mismatch_level_id": self.mismatch_level_id,
        }
        if self.error_message:
            d["error_message"] = self.error_message
        if self.b1_scores:
            d["b1_scores"] = self.b1_scores
        if self.per_algorithm:
            d["per_algorithm"] = self.per_algorithm
        if self.mismatch_results:
            d["mismatch_results"] = self.mismatch_results
        if self.true_params:
            d["true_params"] = self.true_params
        if self.estimated_params:
            d["estimated_params"] = self.estimated_params
        if self.param_errors:
            d["param_errors"] = self.param_errors
        if self.rho is not None:
            d["rho"] = self.rho
        if self.grid_search_result is not None:
            d["grid_search_result"] = self.grid_search_result
        if self.psnr_improvement is not None:
            d["psnr_improvement"] = self.psnr_improvement
        return d


@dataclass
class ExpandedRunSummary:
    """Aggregate statistics across a batch of ``CaseResult`` objects."""

    modality_id: str = ""
    benchmark: str = ""
    n_total: int = 0
    n_success: int = 0
    n_skipped: int = 0
    n_error: int = 0
    total_wall_time_s: float = 0.0

    # Mean metrics across successful cases
    mean_metrics: Dict[str, float] = field(default_factory=dict)

    # Per-variant breakdown: variant_id -> {n, mean_psnr, mean_ssim}
    per_variant: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-noise-level breakdown
    per_noise: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-image-size breakdown
    per_size: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Per-mismatch-level breakdown
    per_mismatch: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Error details
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality_id": self.modality_id,
            "benchmark": self.benchmark,
            "n_total": self.n_total,
            "n_success": self.n_success,
            "n_skipped": self.n_skipped,
            "n_error": self.n_error,
            "total_wall_time_s": self.total_wall_time_s,
            "mean_metrics": self.mean_metrics,
            "per_variant": self.per_variant,
            "per_noise": self.per_noise,
            "per_size": self.per_size,
            "per_mismatch": self.per_mismatch,
            "errors": self.errors,
        }


def aggregate_results(results: List[CaseResult]) -> ExpandedRunSummary:
    """Compute aggregate statistics from a list of case results.

    Args:
        results: List of ``CaseResult`` from a batch run.

    Returns:
        ``ExpandedRunSummary`` with breakdowns by variant, noise, size, mismatch.
    """
    if not results:
        return ExpandedRunSummary()

    summary = ExpandedRunSummary(
        modality_id=results[0].modality_id,
        benchmark=results[0].benchmark,
        n_total=len(results),
    )

    # Collect successful results for metric aggregation
    successful = [r for r in results if r.status == "success"]
    summary.n_success = len(successful)
    summary.n_skipped = sum(1 for r in results if r.status == "skipped")
    summary.n_error = sum(1 for r in results if r.status == "error")
    summary.total_wall_time_s = sum(r.wall_time_s for r in results)

    # Errors list
    summary.errors = [
        {"case_id": r.case_id, "error": r.error_message}
        for r in results if r.status == "error"
    ]

    if not successful:
        return summary

    # Mean metrics
    metric_sums: Dict[str, float] = defaultdict(float)
    for r in successful:
        for k, v in r.metrics.items():
            metric_sums[k] += v
    summary.mean_metrics = {k: v / len(successful) for k, v in metric_sums.items()}

    # Per-dimension breakdowns
    def _build_breakdown(key_fn):
        groups: Dict[str, List[CaseResult]] = defaultdict(list)
        for r in successful:
            groups[key_fn(r)].append(r)
        breakdown = {}
        for key, group in sorted(groups.items()):
            group_metrics: Dict[str, float] = defaultdict(float)
            for r in group:
                for k, v in r.metrics.items():
                    group_metrics[k] += v
            breakdown[key] = {
                "n": len(group),
                **{f"mean_{k}": v / len(group) for k, v in group_metrics.items()},
            }
        return breakdown

    summary.per_variant = _build_breakdown(lambda r: r.variant_id)
    summary.per_noise = _build_breakdown(lambda r: r.noise_level_id)
    summary.per_size = _build_breakdown(lambda r: r.image_size_id)
    summary.per_mismatch = _build_breakdown(lambda r: r.mismatch_level_id)

    return summary
