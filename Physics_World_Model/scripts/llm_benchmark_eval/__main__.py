"""CLI entry point: python -m scripts.llm_benchmark_eval"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import ALL_VARIANTS, BENCHMARKS, MODEL_REGISTRY
from .evaluator import dry_run, run_evaluation
from .scoring import aggregate_results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm_benchmark_eval",
        description="Evaluate LLMs on B1 (Spec Selection) & B3 (Spec Validation).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        metavar="KEY",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model short keys to evaluate (default: all 15).",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        metavar="VARIANT",
        help="Variant names to evaluate (default: all 65).",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="BM",
        choices=list(BENCHMARKS),
        help="Benchmarks to run: b1, b3 (default: both).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be evaluated without making API calls.",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip evaluation; just aggregate existing raw results into summary.json.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate variants
    if args.variants:
        bad = [v for v in args.variants if v not in ALL_VARIANTS]
        if bad:
            print(f"Unknown variants: {bad}", file=sys.stderr)
            print(f"Available: {ALL_VARIANTS}", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        info = dry_run(args.models, args.variants, args.benchmarks)
        print(json.dumps(info, indent=2))
        return

    if args.aggregate_only:
        summary = aggregate_results(args.models, args.variants, args.benchmarks)
        print(json.dumps(summary, indent=2))
        return

    # Run evaluation
    stats = asyncio.run(
        run_evaluation(args.models, args.variants, args.benchmarks)
    )
    print("\n=== Evaluation complete ===")
    print(json.dumps(stats, indent=2))

    # Auto-aggregate after eval
    print("\nAggregating results...")
    summary = aggregate_results(args.models, args.variants, args.benchmarks)
    print(f"Summary written with {len(summary.get('models', {}))} models.")


if __name__ == "__main__":
    main()
