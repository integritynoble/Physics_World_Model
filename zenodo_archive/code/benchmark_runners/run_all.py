#!/usr/bin/env python3
"""Run benchmarks for all 168 imaging modalities.

Usage:
    python -m benchmarks.runners.run_all --level M0 --dry-run
    python -m benchmarks.runners.run_all --level M0 --parallel 4
    python -m benchmarks.runners.run_all --tier A --level M1
    python -m benchmarks.runners.run_all --level M0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PWM_CORE = ROOT / "packages" / "pwm_core"
if str(PWM_CORE) not in sys.path:
    sys.path.insert(0, str(PWM_CORE))

from benchmarks.framework.benchmark_config import load_all_configs, BenchmarkConfig
from benchmarks.framework.base_benchmark import BaseBenchmark
from benchmarks.framework.report_writer import ReportWriter, RunBundle


def run_single(
    config_path: str,
    level: str,
    dry_run: bool,
    output_dir: Optional[str],
) -> Dict:
    """Run a single benchmark (for parallel execution).

    Returns a serialisable result dict.
    """
    from benchmarks.framework.benchmark_config import load_config
    from benchmarks.framework.base_benchmark import BaseBenchmark

    cfg = load_config(Path(config_path))
    bench = BaseBenchmark(
        config=cfg,
        output_dir=Path(output_dir) if output_dir else None,
        verbose=False,
    )
    try:
        bundle = bench.run(level=level, dry_run=dry_run)
        return bundle.to_dict()
    except Exception as e:
        return {
            "modality_id": cfg.modality_id,
            "level": level,
            "tier": cfg.tier,
            "error": str(e),
            "metrics": {},
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for all 168 imaging modalities",
    )
    parser.add_argument(
        "--level", "-l",
        default="M0",
        choices=["M0", "M1", "M2", "M3", "M4"],
    )
    parser.add_argument(
        "--tier",
        default=None,
        choices=["A", "B", "C"],
        help="Filter by tier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
    )
    args = parser.parse_args()

    configs = load_all_configs()

    # Filter by tier
    if args.tier:
        configs = {k: v for k, v in configs.items() if v.tier == args.tier}

    print(f"PWM Benchmark Suite – {len(configs)} modalities, level {args.level}")
    if args.dry_run:
        print("  [dry-run mode]")
    print("=" * 60)

    output_dir = args.output_dir
    config_dir = ROOT / "benchmarks" / "configs"
    t0 = time.time()

    results: List[Dict] = []
    completed = 0
    failed = 0

    if args.parallel > 1:
        # Parallel execution
        futures = {}
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            for mid in sorted(configs):
                config_path = str(config_dir / f"{mid}.yaml")
                fut = executor.submit(
                    run_single, config_path, args.level, args.dry_run, output_dir,
                )
                futures[fut] = mid

            for fut in as_completed(futures):
                mid = futures[fut]
                try:
                    result = fut.result()
                    results.append(result)
                    if "error" in result:
                        failed += 1
                        if not args.quiet:
                            print(f"  [FAIL] {mid}: {result['error']}")
                    else:
                        completed += 1
                        psnr = result.get("metrics", {}).get("psnr", 0)
                        if not args.quiet:
                            print(f"  [OK]   {mid}: PSNR={psnr:.2f} "
                                  f"(tier {result.get('tier', '?')})")
                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {mid}: {e}")
    else:
        # Sequential execution
        for mid in sorted(configs):
            cfg = configs[mid]
            try:
                bench = BaseBenchmark(
                    config=cfg,
                    output_dir=Path(output_dir) if output_dir else None,
                    verbose=False,
                )
                bundle = bench.run(level=args.level, dry_run=args.dry_run)
                results.append(bundle.to_dict())
                completed += 1
                psnr = bundle.metrics.get("psnr", 0)
                if not args.quiet:
                    print(f"  [OK]   {mid}: PSNR={psnr:.2f} (tier {bundle.tier})")
            except Exception as e:
                failed += 1
                results.append({
                    "modality_id": mid,
                    "level": args.level,
                    "tier": cfg.tier,
                    "error": str(e),
                    "metrics": {},
                })
                if not args.quiet:
                    print(f"  [FAIL] {mid}: {e}")

    elapsed = time.time() - t0

    # Save aggregate results
    output_path = Path(output_dir) if output_dir else ROOT / "benchmarks" / "results"
    output_path.mkdir(parents=True, exist_ok=True)
    agg_path = output_path / "aggregate_results.json"
    with open(agg_path, "w") as f:
        json.dump({
            "n_total": len(configs),
            "n_completed": completed,
            "n_failed": failed,
            "level": args.level,
            "tier_filter": args.tier,
            "dry_run": args.dry_run,
            "elapsed_s": elapsed,
            "results": results,
        }, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total:     {len(configs)} modalities")
    print(f"Completed: {completed}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f}s")
    print(f"Results:   {agg_path}")

    # Tier breakdown
    tier_stats = {"A": [0, 0], "B": [0, 0], "C": [0, 0]}
    for r in results:
        tier = r.get("tier", "C")
        if tier in tier_stats:
            tier_stats[tier][0] += 1
            if "error" not in r:
                tier_stats[tier][1] += 1
    for tier, (total, ok) in sorted(tier_stats.items()):
        if total > 0:
            print(f"  Tier {tier}: {ok}/{total} passed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
