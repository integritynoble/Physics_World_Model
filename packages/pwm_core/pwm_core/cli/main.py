"""pwm_core.cli.main

Entry point for `pwm` CLI.

Commands:
- pwm run --prompt "..."
- pwm run --spec spec.json
- pwm fit-operator --y y.npy --operator cassi --out out_dir
- pwm calib-recon --y y.npy --operator cassi --out out_dir
- pwm view <runbundle_dir>
- pwm demo <modality> [--preset NAME] [--run] [--open-viewer] [--export-sharepack]
- pwm evaluate --modality cassi --solver traditional_cpu --track correct
- pwm scaffold solver my_solver
- pwm contrib check my_solver
- pwm submit runbundle.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from pwm_core.api import endpoints
from pwm_core.api.types import (
    ExperimentSpec,
    ExperimentInput,
    ExperimentStates,
    InputMode,
    OperatorInput,
    OperatorKind,
    OperatorParametric,
    OperatorMatrix,
    PhysicsState,
    TaskState,
    TaskKind,
    MismatchSpec,
    MismatchFitOperator,
)

# Commands delegated to the targeting CLI
_TARGETING_COMMANDS = {"evaluate", "scaffold", "contrib", "submit", "install", "uninstall", "plugins"}


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _build_spec_from_y_operator(
    y_path: str,
    operator_id: str,
    task_kind: TaskKind,
) -> ExperimentSpec:
    """Build an ExperimentSpec from measured y and operator ID.

    Args:
        y_path: Path to measurement file (.npy, .npz, .pt).
        operator_id: Operator identifier (e.g., 'cassi', 'widefield', 'matrix').
        task_kind: The task to perform.

    Returns:
        ExperimentSpec configured for measured mode with operator fitting.
    """
    spec_id = f"cli_{operator_id}_{uuid.uuid4().hex[:8]}"

    # Determine operator kind based on operator_id
    if operator_id in ("cassi", "widefield", "sim", "confocal", "lensless", "spc"):
        operator_input = OperatorInput(
            kind=OperatorKind.parametric,
            parametric=OperatorParametric(
                operator_id=operator_id,
                theta_init={},
                theta_space=None,
            ),
        )
        modality = operator_id
    elif y_path.endswith((".npz", ".pt", ".mat")):
        # Assume matrix mode for matrix files
        operator_input = OperatorInput(
            kind=OperatorKind.matrix,
            matrix=OperatorMatrix(source=y_path),
        )
        modality = "matrix"
    else:
        # Default to parametric
        operator_input = OperatorInput(
            kind=OperatorKind.parametric,
            parametric=OperatorParametric(
                operator_id=operator_id,
                theta_init={},
            ),
        )
        modality = operator_id

    # Build the spec
    spec = ExperimentSpec(
        id=spec_id,
        input=ExperimentInput(
            mode=InputMode.measured,
            y_source=y_path,
            operator=operator_input,
        ),
        states=ExperimentStates(
            physics=PhysicsState(modality=modality),
            task=TaskState(kind=task_kind),
        ),
        mismatch=MismatchSpec(
            enabled=True,
            fit_operator=MismatchFitOperator(
                enabled=True,
                theta_space=None,  # Use defaults for the operator
                search={"method": "random", "max_evals": 50},
            ),
        ),
    )

    return spec


def _serialize_result(result) -> dict:
    """Serialize a result object to JSON-compatible dict."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    elif hasattr(result, "__dict__"):
        return {k: _serialize_result(v) for k, v in result.__dict__.items()}
    elif isinstance(result, list):
        return [_serialize_result(item) for item in result]
    elif isinstance(result, dict):
        return {k: _serialize_result(v) for k, v in result.items()}
    else:
        return result


def cmd_run(args):
    if args.prompt:
        res = endpoints.run(prompt=args.prompt, out_dir=args.out_dir)
    elif args.spec:
        spec = _read_json(Path(args.spec))
        res = endpoints.run(spec=spec, out_dir=args.out_dir)
    else:
        print("Error: Either --prompt or --spec must be provided.")
        return
    print(json.dumps(res, indent=2, default=str))


def cmd_fit_operator(args):
    spec = _build_spec_from_y_operator(
        y_path=args.y,
        operator_id=args.operator,
        task_kind=TaskKind.fit_operator_only,
    )
    result = endpoints.fit_operator(spec=spec, out_dir=args.out_dir)
    print(json.dumps(_serialize_result(result), indent=2, default=str))


def cmd_calib_recon(args):
    spec = _build_spec_from_y_operator(
        y_path=args.y,
        operator_id=args.operator,
        task_kind=TaskKind.calibrate_and_reconstruct,
    )
    result = endpoints.calibrate_recon(spec=spec, out_dir=args.out_dir)
    print(json.dumps(_serialize_result(result), indent=2, default=str))


def cmd_view(args):
    endpoints.view(runbundle_dir=args.runbundle_dir)


def build_parser():
    p = argparse.ArgumentParser(prog="pwm", description="Physics World Model CLI.")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run full pipeline from prompt or spec")
    p_run.add_argument("--prompt", type=str, default=None, help="Natural language prompt")
    p_run.add_argument("--spec", type=str, default=None, help="Path to spec JSON file")
    p_run.add_argument("--out-dir", type=str, default="runs", help="Output directory")
    p_run.set_defaults(func=cmd_run)

    p_fit = sub.add_parser("fit-operator", help="Fit operator parameters theta from measured y")
    p_fit.add_argument("--y", required=True, help="Path to measurement file (.npy, .npz, .pt)")
    p_fit.add_argument("--operator", required=True, help="Operator ID (cassi, widefield, matrix, etc.)")
    p_fit.add_argument("--out-dir", type=str, default="runs", help="Output directory")
    p_fit.set_defaults(func=cmd_fit_operator)

    p_cr = sub.add_parser("calib-recon", help="Fit operator theta and reconstruct")
    p_cr.add_argument("--y", required=True, help="Path to measurement file (.npy, .npz, .pt)")
    p_cr.add_argument("--operator", required=True, help="Operator ID (cassi, widefield, matrix, etc.)")
    p_cr.add_argument("--out-dir", type=str, default="runs", help="Output directory")
    p_cr.set_defaults(func=cmd_calib_recon)

    p_view = sub.add_parser("view", help="Launch local viewer for a RunBundle")
    p_view.add_argument("runbundle_dir", help="Path to RunBundle directory")
    p_view.set_defaults(func=cmd_view)

    # --- demo subcommand ---
    from pwm_core.cli.demo import add_demo_subparser, cmd_demo
    add_demo_subparser(sub)

    # --- doctor subcommand ---
    from pwm_core.cli.doctor import add_doctor_subparser
    add_doctor_subparser(sub)

    # --- next-modality subcommand ---
    from pwm_core.cli.modality_gate import add_next_modality_subparser
    add_next_modality_subparser(sub)

    # --- targeting system placeholders (for help text) ---
    sub.add_parser("evaluate", help="Run the targeting harness (pwm evaluate --modality cassi)")
    sub.add_parser("scaffold", help="Scaffold a new solver/calibrator/modality (pwm scaffold solver my_solver)")
    sub.add_parser("contrib", help="Contribution tools (pwm contrib check my_solver)")
    sub.add_parser("submit", help="Submit a RunBundle for leaderboard scoring")

    # --- synthesize / ingest / install ---
    from pwm_core.cli.synthesize import add_synthesize_subparser
    add_synthesize_subparser(sub)

    from pwm_core.cli.ingest import add_ingest_subparser
    add_ingest_subparser(sub)

    from pwm_core.cli.install import add_install_subparser
    add_install_subparser(sub)

    return p


def main(argv=None):
    # Check if the first argument is a targeting CLI command -- delegate directly
    args_list = argv if argv is not None else sys.argv[1:]
    if args_list and args_list[0] in _TARGETING_COMMANDS:
        from pwm_core.targeting.cli import main as targeting_main
        sys.exit(targeting_main(args_list))

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return
    if hasattr(args, "func"):
        args.func(args)


if __name__ == "__main__":
    main()
