#!/usr/bin/env python3
"""Run a single modality benchmark.

Usage:
    python -m benchmarks.runners.run_benchmark --modality cassi --level M0
    python -m benchmarks.runners.run_benchmark --modality widefield --level M1
    python -m benchmarks.runners.run_benchmark --modality ct --level M0 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PWM_CORE = ROOT / "packages" / "pwm_core"
if str(PWM_CORE) not in sys.path:
    sys.path.insert(0, str(PWM_CORE))

from benchmarks.framework.benchmark_config import load_config, load_all_configs
from benchmarks.framework.base_benchmark import BaseBenchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark for a single imaging modality",
    )
    parser.add_argument(
        "--modality", "-m",
        required=True,
        help="Modality ID (e.g. cassi, widefield, ct)",
    )
    parser.add_argument(
        "--level", "-l",
        default="M0",
        choices=["M0", "M1", "M2", "M3", "M4"],
        help="Maturity level (default: M0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and operator build without running reconstruction",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    # Load config
    config_path = ROOT / "benchmarks" / "configs" / f"{args.modality}.yaml"
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        print(f"Available configs: {len(list((ROOT / 'benchmarks' / 'configs').glob('*.yaml')))} files")
        sys.exit(1)

    cfg = load_config(config_path)
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Run benchmark
    bench = BaseBenchmark(
        config=cfg,
        output_dir=output_dir,
        verbose=not args.quiet,
        seed=args.seed,
    )
    t0 = time.time()
    bundle = bench.run(level=args.level, dry_run=args.dry_run)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Modality:  {bundle.modality_id}")
    print(f"Level:     {bundle.level}")
    print(f"Tier:      {bundle.tier}")
    print(f"Time:      {elapsed:.1f}s")
    if bundle.metrics:
        for k, v in bundle.metrics.items():
            print(f"  {k}: {v:.4f}")
    if bundle.rho is not None:
        print(f"  rho: {bundle.rho:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
