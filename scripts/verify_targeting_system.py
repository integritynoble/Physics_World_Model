#!/usr/bin/env python3
"""Verify the targeting system works end-to-end for all validated modalities.

Runs the Harness in sandbox mode (1 scene, 32x32, mild severity) for each
validated modality and reports pass/fail.

Usage:
    python scripts/verify_targeting_system.py           # 7 validated only
    python scripts/verify_targeting_system.py --all     # all 26 registry modalities
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

# Ensure pwm_core is importable
_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "packages" / "pwm_core"))

from pwm_core.targeting.harness import Harness, load_solver_registry

VALIDATED_MODALITIES = [
    "cassi", "cacti", "spc", "ct", "mri", "ptychography", "lensless",
]


def get_all_modalities() -> list[str]:
    """Return all modalities that have a traditional_cpu solver."""
    registry = load_solver_registry()
    return sorted(
        m for m, tiers in registry.items()
        if isinstance(tiers, dict) and "traditional_cpu" in tiers
    )


def verify_modality(modality: str, verbose: bool = False) -> dict:
    """Run sandbox harness for one modality, return result dict."""
    result = {
        "modality": modality,
        "passed": False,
        "rho": None,
        "oracle_gap": None,
        "error": None,
    }

    try:
        harness = Harness(
            modality=modality,
            solver="traditional_cpu",
            track="correct",
            budget_s=120,
            sandbox=True,
        )
        hr = harness.run(n_scenes=1, seed=42, severity="mild")

        result["rho"] = hr.aggregate.rho
        result["oracle_gap"] = hr.aggregate.oracle_gap
        result["timing_s"] = hr.timing_s

        # Pass criteria: no crash, valid rho
        if hr.aggregate.rho is not None and not hr.aggregate.disqualified:
            result["passed"] = True

        if verbose:
            print(hr.summary_table())

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify targeting system")
    parser.add_argument(
        "--all", action="store_true",
        help="Test all registry modalities (not just 7 validated)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed results per modality",
    )
    parser.add_argument(
        "modalities", nargs="*",
        help="Specific modalities to test (default: validated 7)",
    )
    args = parser.parse_args()

    if args.modalities:
        modalities = args.modalities
    elif args.all:
        modalities = get_all_modalities()
    else:
        modalities = VALIDATED_MODALITIES

    logging.basicConfig(
        level=logging.WARNING if not args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(f"Verifying targeting system for {len(modalities)} modalities\n")
    print(f"{'Modality':<20} {'Status':<8} {'rho':>8} {'gap_dB':>8} {'time_s':>8}")
    print("-" * 56)

    results = []
    for modality in modalities:
        r = verify_modality(modality, verbose=args.verbose)
        results.append(r)

        status = "PASS" if r["passed"] else "FAIL"
        rho_str = f"{r['rho']:.4f}" if r["rho"] is not None else "N/A"
        gap_str = f"{r['oracle_gap']:.2f}" if r.get("oracle_gap") is not None else "N/A"
        time_str = f"{r.get('timing_s', 0):.2f}" if r.get("timing_s") else "N/A"

        print(f"  {modality:<18} {status:<8} {rho_str:>8} {gap_str:>8} {time_str:>8}")

        if r["error"] and not args.verbose:
            print(f"    Error: {r['error']}")

    # Summary
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = len(results) - n_pass
    print(f"\n{'='*56}")
    print(f"  Results: {n_pass} passed, {n_fail} failed out of {len(results)}")
    print(f"{'='*56}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
