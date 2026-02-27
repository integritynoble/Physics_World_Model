#!/usr/bin/env python3
"""Verify and inspect auto-generated benchmark data for all imaging modalities.

Usage:
    python3 scripts/generate_benchmark_data.py --all          # verify all variants
    python3 scripts/generate_benchmark_data.py --variant ct   # inspect one variant
    python3 scripts/generate_benchmark_data.py --stats        # summary statistics
    python3 scripts/generate_benchmark_data.py --dry-run      # show what would be generated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure platform package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pwm_platform.services.benchmark_database import VARIANT_DATABASE
from pwm_platform.services.benchmark_database._challenge_data import CHALLENGE_CONFIG
from pwm_platform.services.benchmark_database._leaderboard_data import LEADERBOARD_DATA


def inspect_variant(key: str, verbose: bool = True) -> dict:
    """Inspect a single variant and return a status dict."""
    v = VARIANT_DATABASE.get(key)
    if v is None:
        return {"key": key, "status": "NOT_FOUND"}

    benchmarks = v.get("benchmarks", [])
    benchmark_ids = [b["id"] for b in benchmarks]

    # Check B2
    b2 = next((b for b in benchmarks if b["id"] == "Benchmark 2"), None)
    b2_leaderboard_count = len(b2["leaderboard"]) if b2 else 0
    b2_has_table = b2 is not None and "scenario_table" in b2
    b2_method_count = len(b2["scenario_table"]["methods"]) if b2_has_table else 0
    b2_scene_count = len(b2["scenario_table"]["per_scene"]) if b2_has_table else 0

    # Check Challenge
    challenge = next((b for b in benchmarks if b["id"] == "Challenge"), None)
    ch_leaderboard_count = len(challenge["leaderboard"]) if challenge else 0
    ch_has_baselines = (
        challenge is not None
        and "baselines" in challenge
        and len(challenge["baselines"].get("scenario_ii", [])) > 0
    )
    ch_baseline_count = (
        len(challenge["baselines"]["scenario_ii"]) if ch_has_baselines else 0
    )

    is_handcrafted = key in LEADERBOARD_DATA
    has_challenge_config = key in CHALLENGE_CONFIG

    status = {
        "key": key,
        "display_name": v["display_name"],
        "category": v["category"],
        "is_handcrafted": is_handcrafted,
        "benchmark_ids": benchmark_ids,
        "b2_leaderboard": b2_leaderboard_count,
        "b2_scenario_table": b2_has_table,
        "b2_methods": b2_method_count,
        "b2_scenes": b2_scene_count,
        "challenge_present": challenge is not None,
        "challenge_leaderboard": ch_leaderboard_count,
        "challenge_baselines": ch_baseline_count,
        "has_challenge_config": has_challenge_config,
    }

    # Determine completeness
    complete = (
        b2_leaderboard_count >= 4
        and b2_has_table
        and b2_method_count >= 4
        and ch_leaderboard_count >= 4
        and ch_baseline_count >= 4
    )
    # For variants without mismatch_params, challenge may be absent — that's OK
    if not v.get("mismatch_params"):
        complete = b2_leaderboard_count >= 4 and b2_has_table and b2_method_count >= 4

    status["complete"] = complete

    if verbose:
        flag = "HC" if is_handcrafted else "AG"
        ok = "\u2713" if complete else "\u2717"
        print(
            f"  [{flag}] {ok} {key:40s}  "
            f"B2={b2_leaderboard_count}m/{b2_scene_count}s  "
            f"Ch={ch_leaderboard_count}m/{ch_baseline_count}b  "
            f"({v['category']})"
        )

    return status


def main():
    parser = argparse.ArgumentParser(description="Verify auto-generated benchmark data")
    parser.add_argument("--all", action="store_true", help="Verify all variants")
    parser.add_argument("--variant", type=str, help="Inspect a specific variant")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not any([args.all, args.variant, args.stats, args.dry_run]):
        args.all = True
        args.stats = True

    total = len(VARIANT_DATABASE)
    print(f"Total variants in VARIANT_DATABASE: {total}")
    print(f"Hand-crafted leaderboards: {len(LEADERBOARD_DATA)}")
    print(f"Challenge configs: {len(CHALLENGE_CONFIG)}")
    print()

    if args.variant:
        key = args.variant
        if key not in VARIANT_DATABASE:
            print(f"Variant '{key}' not found. Available keys:")
            for k in sorted(VARIANT_DATABASE.keys()):
                print(f"  {k}")
            return 1

        status = inspect_variant(key, verbose=True)

        v = VARIANT_DATABASE[key]
        benchmarks = v.get("benchmarks", [])

        # Show B2 leaderboard
        b2 = next((b for b in benchmarks if b["id"] == "Benchmark 2"), None)
        if b2 and b2["leaderboard"]:
            print(f"\n  B2 Leaderboard (Scenario I — Ideal):")
            for e in b2["leaderboard"]:
                print(f"    #{e['rank']}  {e['method']:20s}  PSNR={e['psnr']:.2f}  SSIM={e['ssim']:.3f}")

        if b2 and "scenario_table" in b2:
            table = b2["scenario_table"]
            print(f"\n  B2 Scenario Table:")
            for m in table["methods"]:
                print(
                    f"    {m['method']:20s}  "
                    f"S1={m['s1_psnr']:.2f}/{m['s1_ssim']:.3f}  "
                    f"S2={m['s2_psnr']:.2f}/{m['s2_ssim']:.3f}  "
                    f"S3={m['s3_psnr']:.2f}/{m['s3_ssim']:.3f}"
                )

        # Show Challenge leaderboard
        ch = next((b for b in benchmarks if b["id"] == "Challenge"), None)
        if ch and ch["leaderboard"]:
            print(f"\n  Challenge Leaderboard:")
            for e in ch["leaderboard"]:
                print(
                    f"    #{e['rank']}  {e['method']:28s}  "
                    f"Pub={e['public_score']:.3f}  Dev={e['dev_score']:.3f}  "
                    f"Hid={e['hidden_score']:.3f}  Overall={e['overall_score']:.3f}"
                )

        if ch and ch.get("baselines"):
            bl = ch["baselines"]
            if bl.get("scenario_ii"):
                print(f"\n  Challenge Baselines (Scenario II):")
                for b in bl["scenario_ii"]:
                    print(f"    {b['method']:20s}  PSNR={b['psnr']:.2f}  SSIM={b['ssim']:.3f}")

        return 0

    if args.dry_run:
        print("Variants that would receive auto-generated data:")
        auto_count = 0
        for key in sorted(VARIANT_DATABASE.keys()):
            if key not in LEADERBOARD_DATA:
                v = VARIANT_DATABASE[key]
                print(f"  {key:40s}  ({v['category']})")
                auto_count += 1
        print(f"\nTotal: {auto_count} auto-generated, {len(LEADERBOARD_DATA)} hand-crafted")
        return 0

    if args.all:
        print("Verifying all variants:\n")
        results = []
        for key in sorted(VARIANT_DATABASE.keys()):
            status = inspect_variant(key, verbose=True)
            results.append(status)

        if args.stats:
            print()

    if args.stats:
        if not args.all:
            results = [inspect_variant(key, verbose=False) for key in VARIANT_DATABASE]

        complete = sum(1 for r in results if r.get("complete"))
        incomplete = sum(1 for r in results if not r.get("complete"))
        handcrafted = sum(1 for r in results if r.get("is_handcrafted"))
        autogen = sum(1 for r in results if not r.get("is_handcrafted"))

        with_challenge = sum(1 for r in results if r.get("challenge_present"))
        with_baselines = sum(1 for r in results if r.get("challenge_baselines", 0) >= 4)

        # Category breakdown
        categories = {}
        for r in results:
            cat = r.get("category", "unknown")
            categories.setdefault(cat, []).append(r)

        print("=" * 70)
        print(f"SUMMARY")
        print(f"  Total variants:        {len(results)}")
        print(f"  Hand-crafted:          {handcrafted}")
        print(f"  Auto-generated:        {autogen}")
        print(f"  Complete:              {complete}")
        print(f"  Incomplete:            {incomplete}")
        print(f"  With Challenge:        {with_challenge}")
        print(f"  With baselines (4+):   {with_baselines}")
        print()
        print(f"  By category:")
        for cat in sorted(categories):
            cat_results = categories[cat]
            cat_complete = sum(1 for r in cat_results if r.get("complete"))
            print(f"    {cat:30s}  {cat_complete}/{len(cat_results)} complete")
        print("=" * 70)

        if incomplete > 0:
            print(f"\nIncomplete variants ({incomplete}):")
            for r in results:
                if not r.get("complete"):
                    print(f"  {r['key']:40s}  ({r.get('category', '?')})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
