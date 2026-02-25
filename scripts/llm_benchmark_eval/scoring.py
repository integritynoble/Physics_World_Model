"""Accuracy computation, aggregation, and leaderboard-compatible output."""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import (
    ALL_VARIANTS,
    BENCHMARKS,
    MODEL_REGISTRY,
    RAW_RESULTS_DIR,
    SUMMARY_FILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw result loading
# ---------------------------------------------------------------------------

def _load_raw(model_key: str, variant: str, benchmark: str) -> list[dict] | None:
    """Load raw per-sample results, or None if not yet evaluated."""
    path = RAW_RESULTS_DIR / model_key / f"{variant}_{benchmark}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _accuracy(results: list[dict]) -> float:
    """Compute accuracy from per-sample results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["is_correct"]) / len(results)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    model_keys: list[str] | None = None,
    variants: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate raw results into a summary and write to summary.json.

    Returns the summary dict with structure:
    {
        "models": {
            "<model_key>": {
                "model_id": "...",
                "b1": {"overall": 0.82, "per_variant": {"mri": 0.90, ...}},
                "b3": {"overall": 0.75, "per_variant": {"mri": 0.80, ...}},
            }
        },
        "leaderboard": {
            "b1": {"mri": [...], ...},
            "b3": {"mri": [...], ...},
        }
    }
    """
    keys = model_keys or list(MODEL_REGISTRY.keys())
    variant_list = variants or ALL_VARIANTS
    benchmark_list = benchmarks or list(BENCHMARKS)

    models_summary: dict[str, dict[str, Any]] = {}

    for mk in keys:
        model = MODEL_REGISTRY[mk]
        entry: dict[str, Any] = {"model_id": model.model_id}

        for bm in benchmark_list:
            per_variant: dict[str, float] = {}
            all_correct = 0
            all_total = 0

            for variant in variant_list:
                results = _load_raw(mk, variant, bm)
                if results is None:
                    continue
                acc = _accuracy(results)
                per_variant[variant] = round(acc, 4)
                all_correct += sum(1 for r in results if r["is_correct"])
                all_total += len(results)

            overall = round(all_correct / all_total, 4) if all_total else 0.0
            entry[bm] = {
                "overall": overall,
                "variants_evaluated": len(per_variant),
                "per_variant": per_variant,
            }

        models_summary[mk] = entry

    # Build per-variant leaderboards
    leaderboard: dict[str, dict[str, list[dict]]] = {}
    for bm in benchmark_list:
        leaderboard[bm] = {}
        for variant in variant_list:
            rows: list[tuple[str, str, float]] = []
            for mk in keys:
                model = MODEL_REGISTRY[mk]
                results = _load_raw(mk, variant, bm)
                if results is None:
                    continue
                acc = _accuracy(results)
                rows.append((mk, model.model_id, acc))

            # Sort by accuracy descending
            rows.sort(key=lambda r: r[2], reverse=True)
            leaderboard[bm][variant] = [
                {
                    "rank": i + 1,
                    "method": model_id,
                    "accuracy": round(acc, 4),
                    "source": "comparegpt.io",
                    "adopted": False,
                }
                for i, (_, model_id, acc) in enumerate(rows)
            ]

    summary = {
        "models": models_summary,
        "leaderboard": leaderboard,
    }

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", SUMMARY_FILE)

    return summary
