#!/usr/bin/env python3
"""Run benchmarks for all modalities in a category.

Usage:
    python -m benchmarks.runners.run_category --category "Medical Imaging"
    python -m benchmarks.runners.run_category --category Microscopy --level M1
    python -m benchmarks.runners.run_category --list-categories
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PWM_CORE = ROOT / "packages" / "pwm_core"
if str(PWM_CORE) not in sys.path:
    sys.path.insert(0, str(PWM_CORE))

from benchmarks.framework.benchmark_config import load_all_configs
from benchmarks.framework.base_benchmark import BaseBenchmark
from benchmarks.framework.report_writer import ReportWriter


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for all modalities in a category",
    )
    parser.add_argument(
        "--category", "-c",
        default=None,
        help="Category name (e.g. 'Medical Imaging', 'Microscopy'). "
             "Matches as substring.",
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
        help="Filter by tier classification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all categories and exit",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
    )
    args = parser.parse_args()

    configs = load_all_configs()

    if args.list_categories:
        categories = {}
        for cfg in configs.values():
            cat = cfg.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(cfg.modality_id)
        for cat in sorted(categories):
            print(f"\n{cat} ({len(categories[cat])} modalities):")
            for mid in sorted(categories[cat]):
                print(f"  {mid}")
        return

    if not args.category:
        parser.error("--category is required (or use --list-categories)")

    # Filter configs
    selected = {}
    for mid, cfg in configs.items():
        if args.category.lower() in cfg.category.lower():
            if args.tier and cfg.tier != args.tier:
                continue
            selected[mid] = cfg

    if not selected:
        print(f"No modalities found for category '{args.category}'")
        sys.exit(1)

    print(f"Running {len(selected)} modalities for category '{args.category}' "
          f"at level {args.level}")
    print("=" * 60)

    output_dir = Path(args.output_dir) if args.output_dir else None
    bundles = []
    t0 = time.time()

    for mid in sorted(selected):
        cfg = selected[mid]
        try:
            bench = BaseBenchmark(
                config=cfg,
                output_dir=output_dir,
                verbose=not args.quiet,
            )
            bundle = bench.run(level=args.level, dry_run=args.dry_run)
            bundles.append(bundle)
            psnr = bundle.metrics.get("psnr", 0)
            status = "OK" if psnr > 0 else "SKIP"
            if not args.quiet:
                print(f"  [{status}] {mid}: PSNR={psnr:.2f} ({bundle.tier})")
        except Exception as e:
            print(f"  [FAIL] {mid}: {e}")

    elapsed = time.time() - t0

    # Save aggregate
    if bundles:
        writer = ReportWriter(output_dir)
        agg_path = writer.save_aggregate(bundles)
        print(f"\nAggregate results: {agg_path}")

    print(f"\n{'=' * 60}")
    print(f"Category: {args.category}")
    print(f"Level: {args.level}")
    print(f"Completed: {len(bundles)}/{len(selected)}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
