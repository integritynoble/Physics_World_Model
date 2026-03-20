"""Core evaluation loop with checkpoint/resume support."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .client import CompareGPTClient
from .config import (
    ALL_VARIANTS,
    BENCHMARKS,
    CHECKPOINT_FILE,
    MODEL_REGISTRY,
    RAW_RESULTS_DIR,
    ModelEntry,
)
from .data_loader import load_b1_samples, load_b3_samples
from .prompts import B1_SYSTEM, B3_SYSTEM, build_b1_user_prompt, build_b3_user_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> set[str]:
    """Return set of completed keys like 'model_key|variant|benchmark'."""
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE) as f:
        data = json.load(f)
    return set(data.get("completed", []))


def _save_checkpoint(completed: set[str]) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


def _ckpt_key(model_key: str, variant: str, benchmark: str) -> str:
    return f"{model_key}|{variant}|{benchmark}"


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _save_raw_results(
    model_key: str,
    variant: str,
    benchmark: str,
    results: list[dict[str, Any]],
) -> None:
    out_dir = RAW_RESULTS_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant}_{benchmark}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Single-sample evaluation
# ---------------------------------------------------------------------------

async def _eval_b1_sample(
    client: CompareGPTClient,
    model: ModelEntry,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one B1 sample. Returns per-sample result dict."""
    user_prompt, correct_letter = build_b1_user_prompt(sample)
    resp = await client.chat(model.model_id, B1_SYSTEM, user_prompt)

    predicted = None
    if resp["parsed"] and "answer" in resp["parsed"]:
        predicted = resp["parsed"]["answer"].strip().upper()

    return {
        "sample_id": sample["id"],
        "correct": correct_letter,
        "predicted": predicted,
        "is_correct": predicted == correct_letter,
        "raw_text": resp["raw_text"],
        "latency_s": resp["latency_s"],
        "error": resp["error"],
    }


async def _eval_b3_sample(
    client: CompareGPTClient,
    model: ModelEntry,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one B3 sample. Returns per-sample result dict."""
    user_prompt, correct_label = build_b3_user_prompt(sample)
    resp = await client.chat(model.model_id, B3_SYSTEM, user_prompt)

    predicted = None
    if resp["parsed"] and "answer" in resp["parsed"]:
        predicted = resp["parsed"]["answer"].strip().lower()

    return {
        "sample_id": sample["id"],
        "correct": correct_label,
        "predicted": predicted,
        "is_correct": predicted == correct_label,
        "raw_text": resp["raw_text"],
        "latency_s": resp["latency_s"],
        "error": resp["error"],
    }


# ---------------------------------------------------------------------------
# Batch evaluation for one (model, variant, benchmark) combination
# ---------------------------------------------------------------------------

async def _eval_combination(
    client: CompareGPTClient,
    model: ModelEntry,
    variant: str,
    benchmark: str,
) -> list[dict[str, Any]]:
    """Run all 50 samples for a single (model, variant, benchmark) combo."""
    if benchmark == "b1":
        samples = load_b1_samples(variant)
        coros = [_eval_b1_sample(client, model, s) for s in samples]
    elif benchmark == "b3":
        samples = load_b3_samples(variant)
        coros = [_eval_b3_sample(client, model, s) for s in samples]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    results = await asyncio.gather(*coros)
    return list(results)


# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------

async def run_evaluation(
    model_keys: list[str] | None = None,
    variants: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full evaluation loop with checkpoint/resume.

    Parameters
    ----------
    model_keys : list of short keys, or None for all 15
    variants : list of variant names, or None for all 65
    benchmarks : list of benchmark ids ("b1", "b3"), or None for both

    Returns
    -------
    dict with summary stats (total combos, completed, skipped, failed).
    """
    models = [MODEL_REGISTRY[k] for k in (model_keys or MODEL_REGISTRY.keys())]
    variant_list = variants or ALL_VARIANTS
    benchmark_list = benchmarks or list(BENCHMARKS)

    completed = _load_checkpoint()
    total = len(models) * len(variant_list) * len(benchmark_list)
    skipped = 0
    done = 0
    failed = 0

    client = CompareGPTClient()
    try:
        for model in models:
            for variant in variant_list:
                for benchmark in benchmark_list:
                    key = _ckpt_key(model.short_key, variant, benchmark)
                    if key in completed:
                        skipped += 1
                        continue

                    logger.info(
                        "Evaluating %s / %s / %s ...",
                        model.short_key, variant, benchmark,
                    )
                    try:
                        results = await _eval_combination(
                            client, model, variant, benchmark,
                        )
                    except FileNotFoundError as exc:
                        logger.error("Data missing, skipping: %s", exc)
                        failed += 1
                        continue

                    _save_raw_results(model.short_key, variant, benchmark, results)

                    n_correct = sum(1 for r in results if r["is_correct"])
                    n_errors = sum(1 for r in results if r["error"])
                    accuracy = n_correct / len(results) if results else 0.0
                    logger.info(
                        "  -> accuracy=%.1f%% (%d/%d correct, %d errors)",
                        accuracy * 100, n_correct, len(results), n_errors,
                    )

                    completed.add(key)
                    _save_checkpoint(completed)
                    done += 1
    finally:
        await client.close()

    return {
        "total_combinations": total,
        "completed_this_run": done,
        "skipped_from_checkpoint": skipped,
        "failed": failed,
    }


def dry_run(
    model_keys: list[str] | None = None,
    variants: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Show what would be evaluated without making any API calls."""
    models = [MODEL_REGISTRY[k] for k in (model_keys or MODEL_REGISTRY.keys())]
    variant_list = variants or ALL_VARIANTS
    benchmark_list = benchmarks or list(BENCHMARKS)
    completed = _load_checkpoint()

    total = len(models) * len(variant_list) * len(benchmark_list)
    already_done = sum(
        1
        for m in models
        for v in variant_list
        for b in benchmark_list
        if _ckpt_key(m.short_key, v, b) in completed
    )
    remaining = total - already_done
    samples_per_combo = 50
    total_api_calls = remaining * samples_per_combo

    return {
        "models": len(models),
        "variants": len(variant_list),
        "benchmarks": len(benchmark_list),
        "total_combinations": total,
        "already_done": already_done,
        "remaining_combinations": remaining,
        "estimated_api_calls": total_api_calls,
    }
