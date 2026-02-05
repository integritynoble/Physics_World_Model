#!/usr/bin/env python3
"""
prompt_to_casepack.py

Minimal example:
- Take a natural-language prompt
- Use the PWM PromptCompiler to select a CasePack
- Produce a draft ExperimentSpec
- Resolve + validate + auto-repair
- Run simulate/recon/analyze
- Export RunBundle

This script assumes you installed pwm_core:
    pip install -e packages/pwm_core

Run:
    python examples/prompt_to_casepack.py --prompt "SIM live-cell, low dose, 9 frames"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pwm_core.api.endpoints import (
    compile_prompt,
    resolve_validate,
    run_pipeline,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Natural language description of what you want to simulate/reconstruct.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="runs/latest",
        help="Output RunBundle directory (will be created).",
    )
    args = ap.parse_args()

    prompt = args.prompt
    out_dir = Path(args.out)

    # 1) Prompt -> (casepack_id, draft_spec, assumptions, patch_list)
    compiled = compile_prompt(prompt)

    print("\n=== PromptCompiler Output ===")
    print(f"Selected CasePack: {compiled.casepack_id}")
    if compiled.assumptions:
        print("Assumptions:")
        for a in compiled.assumptions:
            print(f"  - {a}")
    if compiled.patch_list:
        print("Extracted overrides / patches:")
        for p in compiled.patch_list:
            print(f"  - {p}")

    # 2) Resolve & validate (fill defaults, normalize units, clamp unsafe values)
    resolved = resolve_validate(compiled.draft_spec)

    print("\n=== Resolve & Validate ===")
    print(f"Valid: {resolved.validation.ok}")
    if resolved.validation.messages:
        print("Messages:")
        for m in resolved.validation.messages:
            print(f"  - [{m.severity.value}] {m.message}")
    if resolved.validation.auto_repair_patch:
        print("Auto-repair patch applied.")
        patch = resolved.validation.auto_repair_patch
        if isinstance(patch, dict) and 'ops' in patch:
            print(f"Patched fields: {len(patch.get('ops', []))}")

    # 3) Run full pipeline
    # This will choose simulate/reconstruct/analyze steps based on TaskState / input.mode
    run = run_pipeline(resolved.spec_resolved)

    print("\n=== Run Summary ===")
    print(f"Spec ID: {run.spec_id}")
    if run.recon:
        for r in run.recon:
            print(f"Solver: {r.solver_id}")
            if r.metrics:
                print(f"Metrics: {r.metrics}")
    if run.diagnosis:
        print(f"Diagnosis: {run.diagnosis.verdict} (conf={run.diagnosis.confidence:.2f})")
        if run.diagnosis.suggested_actions:
            print("Suggested actions:")
            for a in run.diagnosis.suggested_actions[:8]:
                print(f"  - {a.knob} {a.op.value} {a.val}")

    # 4) Export RunBundle
    print("\n=== Export ===")
    if run.runbundle_path:
        print(f"RunBundle written to: {run.runbundle_path}")
        print("You can view it with:")
        print(f"  pwm view {run.runbundle_path}")
    else:
        print("No RunBundle path in result (run was in-memory only).")

if __name__ == "__main__":
    main()
