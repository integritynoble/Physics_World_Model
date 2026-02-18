"""pwm_core.targeting.cli
==========================

CLI for ``pwm evaluate``, ``pwm scaffold``, ``pwm contrib check``,
``pwm submit``, and ``pwm install``.

Usage::

    pwm evaluate --modality cassi --solver traditional_cpu --track correct
    pwm evaluate --sandbox --modality widefield --solver traditional_cpu
    pwm scaffold solver my_solver
    pwm scaffold modality my_modality
    pwm contrib check my_solver
    pwm submit runbundle.zip
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the targeting harness."""
    from pwm_core.targeting.harness import Harness
    from pwm_core.targeting.runbundle_emitter import emit_runbundle

    try:
        harness = Harness(
            modality=args.modality,
            solver=args.solver,
            track=args.track,
            budget_s=args.budget,
            sandbox=args.sandbox,
            solver_fn=None,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize harness: {e}")
        return 1

    if args.dry_run:
        print(f"Dry run: harness initialized successfully for "
              f"{args.modality}/{args.solver}/{args.track}")
        print(f"  Template: {harness.template_id}")
        print(f"  Budget: {args.budget}s")
        print(f"  Sandbox: {args.sandbox}")
        return 0

    try:
        result = harness.run(
            n_scenes=args.scenes,
            seed=args.seed,
            severity=args.severity,
        )
    except Exception as e:
        print(f"ERROR: Harness execution failed: {e}")
        return 1

    # Print summary
    print(result.summary_table())

    # Emit RunBundle
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(".")

    bundle_path = emit_runbundle(result, output_dir)
    result.runbundle_path = str(bundle_path)
    print(f"\nRunBundle: {bundle_path}")

    # Save JSON results
    results_path = bundle_path / "harness_result.json"
    with open(results_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    return 0


def cmd_scaffold(args: argparse.Namespace) -> int:
    """Scaffold a new solver, calibrator, or modality."""
    from pwm_core.targeting.scaffold import scaffold_solver, scaffold_modality

    if args.type == "solver":
        scaffold_solver(args.name)
        print(f"Solver scaffolded: contrib/solvers/{args.name}/")
    elif args.type == "modality":
        scaffold_modality(args.name)
        print(f"Modality scaffolded: contrib/modalities/{args.name}/")
    elif args.type == "calibrator":
        scaffold_solver(args.name, calibrator=True)
        print(f"Calibrator scaffolded: contrib/calibrators/{args.name}/")
    else:
        print(f"Unknown scaffold type: {args.type}")
        return 1
    return 0


def cmd_contrib_check(args: argparse.Namespace) -> int:
    """Validate a contribution before PR."""
    from pwm_core.targeting.contrib_check import check_contribution

    result = check_contribution(args.name, args.type)
    for line in result["report"]:
        print(line)
    return 0 if result["passed"] else 1


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit a RunBundle for leaderboard scoring."""
    from pwm_core.targeting.submit import submit_runbundle

    result = submit_runbundle(Path(args.path))
    for line in result["report"]:
        print(line)
    return 0 if result["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pwm",
        description="PWM Targeting System CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # --- evaluate ---
    ev = sub.add_parser("evaluate", help="Run the targeting harness")
    ev.add_argument("--modality", "-m", required=True, help="Modality name")
    ev.add_argument("--solver", "-s", default="traditional_cpu", help="Solver name or tier")
    ev.add_argument("--track", "-t", default="correct",
                     choices=["correct", "diagnose", "no_gt", "design"])
    ev.add_argument("--budget", type=int, default=600, help="Budget in seconds")
    ev.add_argument("--scenes", type=int, default=5, help="Number of scenes")
    ev.add_argument("--seed", type=int, default=42, help="Random seed")
    ev.add_argument("--severity", default="moderate",
                     choices=["mild", "moderate", "severe", "catastrophic"])
    ev.add_argument("--sandbox", action="store_true", help="Sandbox mode (fast, tiny)")
    ev.add_argument("--dry-run", action="store_true", help="Validate setup only")
    ev.add_argument("--output", "-o", help="Output directory for RunBundle")

    # --- scaffold ---
    sc = sub.add_parser("scaffold", help="Scaffold a new contribution")
    sc.add_argument("type", choices=["solver", "calibrator", "modality"])
    sc.add_argument("name", help="Name of the new component")

    # --- contrib check ---
    cc = sub.add_parser("contrib", help="Contribution tools")
    cc_sub = cc.add_subparsers(dest="contrib_command")
    ck = cc_sub.add_parser("check", help="Validate a contribution")
    ck.add_argument("name", help="Component name to check")
    ck.add_argument("--type", default="solver", choices=["solver", "calibrator", "modality"])

    # --- submit ---
    sm = sub.add_parser("submit", help="Submit a RunBundle")
    sm.add_argument("path", help="Path to RunBundle zip or directory")

    return parser


def main(argv: Optional[list] = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "scaffold":
        return cmd_scaffold(args)
    elif args.command == "contrib":
        if hasattr(args, "contrib_command") and args.contrib_command == "check":
            return cmd_contrib_check(args)
        else:
            parser.parse_args(["contrib", "--help"])
            return 1
    elif args.command == "submit":
        return cmd_submit(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
