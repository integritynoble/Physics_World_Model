#!/usr/bin/env python3
"""Run expanded benchmark cases for a single modality.

Usage:
    python -m benchmarks.runners.run_expanded --modality cassi --benchmark B2
    python -m benchmarks.runners.run_expanded --modality cassi --benchmark B2 --variant sd_cassi --limit 10
    python -m benchmarks.runners.run_expanded --modality ct --benchmark B4 --parallel 4
    python -m benchmarks.runners.run_expanded --modality cassi --benchmark B2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PWM_CORE = ROOT / "packages" / "pwm_core"
if str(PWM_CORE) not in sys.path:
    sys.path.insert(0, str(PWM_CORE))

from benchmarks.framework.expanded_config import load_expanded_config
from benchmarks.framework.expanded_runner import ExpandedBenchmarkRunner
from benchmarks.framework.expanded_result import aggregate_results


def main():
    parser = argparse.ArgumentParser(
        description="Run expanded benchmark cases for a modality",
    )
    parser.add_argument(
        "--modality", "-m",
        required=True,
        help="Modality ID (e.g., cassi, ct, widefield)",
    )
    parser.add_argument(
        "--benchmark", "-b",
        required=True,
        choices=["B1", "B2", "B3", "B4"],
        help="Benchmark type",
    )
    parser.add_argument(
        "--variant", "-v",
        default=None,
        help="Filter to a specific variant ID",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of cases to run",
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and operator build without running reconstruction",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # 1. Load expanded config
    try:
        ecfg = load_expanded_config(args.modality)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Try loading simple config as fallback
    simple_cfg = None
    simple_cfg_path = ROOT / "benchmarks" / "configs" / f"{args.modality}.yaml"
    if simple_cfg_path.exists():
        from benchmarks.framework.benchmark_config import load_config
        try:
            simple_cfg = load_config(simple_cfg_path)
        except Exception as e:
            if not args.quiet:
                print(f"Warning: could not load simple config: {e}")

    # 3. Generate cases
    gen_method = f"generate_{args.benchmark.lower()}_cases"
    gen_fn = getattr(ecfg, gen_method, None)
    if gen_fn is None:
        print(f"Error: no generator for {args.benchmark}", file=sys.stderr)
        sys.exit(1)

    cases = gen_fn()

    # 4. Filter by variant
    if args.variant:
        cases = [c for c in cases if c.variant.id == args.variant]
        if not cases:
            print(f"Error: no cases found for variant '{args.variant}'", file=sys.stderr)
            sys.exit(1)

    # 5. Apply limit
    if args.limit is not None:
        cases = cases[:args.limit]

    # Print header
    if not args.quiet:
        print(f"PWM Expanded Benchmark Runner")
        print(f"  Modality:  {ecfg.display_name} ({ecfg.modality_id})")
        print(f"  Benchmark: {args.benchmark}")
        print(f"  Cases:     {len(cases)}")
        if args.variant:
            print(f"  Variant:   {args.variant}")
        if args.dry_run:
            print(f"  Mode:      dry-run")
        if args.parallel > 1:
            print(f"  Workers:   {args.parallel}")
        print("=" * 60)

    # 6. Create runner and execute
    output_dir = Path(args.output_dir) if args.output_dir else None
    runner = ExpandedBenchmarkRunner(
        expanded_config=ecfg,
        simple_config=simple_cfg,
        output_dir=output_dir,
        verbose=not args.quiet,
        seed=args.seed,
    )

    t0 = time.time()
    results = runner.run_cases(cases, parallel=args.parallel, dry_run=args.dry_run)
    elapsed = time.time() - t0

    # 7. Aggregate and save summary
    summary = aggregate_results(results)
    summary_dir = output_dir or (ROOT / "benchmarks" / "results")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{ecfg.modality_id}_{args.benchmark}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2, default=str)

    # 8. Print summary table
    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"Summary: {ecfg.modality_id} / {args.benchmark}")
        print(f"  Total:     {summary.n_total}")
        print(f"  Success:   {summary.n_success}")
        print(f"  Skipped:   {summary.n_skipped}")
        print(f"  Error:     {summary.n_error}")
        print(f"  Time:      {elapsed:.1f}s")

        if summary.mean_metrics:
            print(f"\n  Mean Metrics:")
            for k, v in sorted(summary.mean_metrics.items()):
                print(f"    {k}: {v:.4f}")

        if summary.per_variant:
            print(f"\n  Per-Variant Breakdown:")
            print(f"    {'Variant':<25} {'N':>4}  {'PSNR':>8}  {'SSIM':>8}")
            print(f"    {'-'*25} {'-'*4}  {'-'*8}  {'-'*8}")
            for vid, info in sorted(summary.per_variant.items()):
                psnr = info.get("mean_psnr", 0)
                ssim = info.get("mean_ssim", 0)
                n = info.get("n", 0)
                print(f"    {vid:<25} {n:>4}  {psnr:>8.2f}  {ssim:>8.4f}")

        if summary.errors:
            print(f"\n  Errors ({len(summary.errors)}):")
            for err in summary.errors[:5]:
                print(f"    {err['case_id']}: {err['error'][:60]}")
            if len(summary.errors) > 5:
                print(f"    ... and {len(summary.errors) - 5} more")

        print(f"\n  Results saved: {summary_path}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
