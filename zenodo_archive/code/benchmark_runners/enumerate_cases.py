#!/usr/bin/env python3
"""Enumerate all benchmark cases from expanded configs.

Usage:
    python -m benchmarks.runners.enumerate_cases                 # all modalities
    python -m benchmarks.runners.enumerate_cases --modality cassi # single modality
    python -m benchmarks.runners.enumerate_cases --summary        # summary only
    python -m benchmarks.runners.enumerate_cases --benchmark B1   # single benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.framework.expanded_config import (
    load_expanded_config,
    load_all_expanded_configs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Enumerate benchmark cases")
    parser.add_argument("--modality", type=str, help="Single modality ID")
    parser.add_argument("--benchmark", type=str, choices=["B1", "B2", "B3", "B4"],
                        help="Single benchmark")
    parser.add_argument("--summary", action="store_true", help="Summary only")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--limit", type=int, default=0, help="Limit cases printed")
    args = parser.parse_args()

    if args.modality:
        try:
            config = load_expanded_config(args.modality)
            configs = {args.modality: config}
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        configs = load_all_expanded_configs()

    if not configs:
        logger.warning("No expanded configs found. Create them in benchmarks/expanded_configs/")
        sys.exit(0)

    grand_total = 0
    summaries = []

    for modality_id, config in configs.items():
        summary = config.summary()
        summaries.append(summary)
        grand_total += summary["total"]

        if args.summary:
            print(f"\n{'='*70}")
            print(f"  {summary['display_name']} ({modality_id})")
            print(f"  Category: {summary['category']}")
            print(f"  Variants: {summary['n_variants']}")
            print(f"  Sizes: {summary['n_sizes']}")
            print(f"  Compression Ratios: {summary['n_compression_ratios']}")
            print(f"  Noise Levels: {summary['n_noise_levels']}")
            print(f"  Mismatch Levels: {summary['n_mismatch_levels']}")
            print(f"  Mismatch Params: {summary['n_mismatch_params']}")
            print(f"  Data Sources: {summary['data_source_labels']}")
            print(f"  Cases: B1={summary['cases'].get('B1',0)}, "
                  f"B2={summary['cases'].get('B2',0)}, "
                  f"B3={summary['cases'].get('B3',0)}, "
                  f"B4={summary['cases'].get('B4',0)}")
            print(f"  Total: {summary['total']}")
            continue

        # Generate and list cases
        all_cases = config.generate_all_cases()
        benchmarks_to_show = [args.benchmark] if args.benchmark else ["B1", "B2", "B3", "B4"]

        for bm in benchmarks_to_show:
            cases = all_cases.get(bm, [])
            print(f"\n--- {modality_id} / {bm}: {len(cases)} cases ---")
            shown = 0
            for case in cases:
                if args.limit and shown >= args.limit:
                    print(f"  ... ({len(cases) - shown} more)")
                    break
                ds_label = case.data_source.label if case.data_source else "?"
                extra = ""
                if bm == "B1":
                    extra = f" difficulty={case.prompt_difficulty} round={case.round_number}"
                elif bm in ("B3", "B4") and case.true_spec_params:
                    n_params = len(case.true_spec_params)
                    extra = f" true_spec={n_params} params"
                print(f"  [{ds_label}] {case.case_id}{extra}")
                shown += 1

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print(f"\n{'='*70}")
        print(f"  GRAND TOTAL: {grand_total} cases across {len(configs)} modalities")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
