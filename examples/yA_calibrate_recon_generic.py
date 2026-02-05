#!/usr/bin/env python3
"""
yA_calibrate_recon_generic.py

Example: Measured y + explicit matrix A -> (optional) fit simple operator corrections -> reconstruct -> diagnose -> export RunBundle.

This is for "generic" linear inverse problems where the user can provide:
- y: measured vector/tensor
- A: measurement matrix (dense or sparse)

Use cases:
- Single-pixel camera with a measured pattern matrix
- Compressed sensing / inpainting with a known measurement operator
- Any lab where you can export the forward model as an explicit matrix

Run:
    python examples/yA_calibrate_recon_generic.py \
        --y data/measured_y.pt \
        --A data/A_matrix.npz \
        --N 65536 \
        --Hx 256 --Wx 256 \
        --out runs/latest

Notes:
- You can set --fit-gain-shift to enable a simple operator correction:
    y ≈ (g * A x) + b
  where g (gain) and b (bias) are fit from residual evidence.
- More complex operator fitting (e.g., alignment, PSF) should be implemented as
  parametric operator families, like the CASSI example.

Then:
    pwm view runs/latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pwm_core.api.endpoints import (
    resolve_validate,
    calibrate_recon,
    export_runbundle,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", type=str, required=True, help="Path to measured y (pt/npz/mat/tif/h5/zarr).")
    ap.add_argument("--A", type=str, required=True, help="Path to matrix A (npz/pt).")
    ap.add_argument("--M", type=int, default=None, help="Measurement dimension (optional).")
    ap.add_argument("--N", type=int, required=True, help="Latent dimension N (flattened x).")
    ap.add_argument("--Hx", type=int, default=None, help="Optional x height for reshaping.")
    ap.add_argument("--Wx", type=int, default=None, help="Optional x width for reshaping.")
    ap.add_argument("--out", type=str, default="runs/latest", help="RunBundle output directory.")
    ap.add_argument("--casepack", type=str, default="generic_matrix_yA_fit_gain_shift_v1",
                    help="CasePack id for generic matrix y+A workflow.")
    ap.add_argument("--fit-gain-shift", action="store_true",
                    help="Enable a simple gain/bias correction y ≈ g*A*x + b (operator-fit).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    x_shape = None
    if args.Hx is not None and args.Wx is not None:
        x_shape = [args.Hx, args.Wx]

    spec = {
        "version": "0.2.1",
        "id": "generic_matrix_measured_fit_demo",
        "input": {
            "mode": "measured",
            "data": {
                "y_source": {
                    "kind": "file",
                    "path": args.y,
                    "format": args.y.split(".")[-1].lower(),
                    "keys": {"y": "y"}
                }
            },
            "operator": {
                "kind": "matrix",
                "matrix": {
                    "path": args.A,
                    "format": args.A.split(".")[-1].lower(),
                    "keys": {"A": "A"},
                    "shape": [args.M, args.N] if args.M is not None else None,
                    "storage": "csr"  # recommended default; loader can override
                }
            }
        },
        "states": {
            "physics": {
                "modality": "generic_linear",
                "dims": {
                    "x": x_shape if x_shape is not None else [args.N],
                    "y": [args.M] if args.M is not None else None
                },
                "operator": {"model": "linear"}
            },
            "budget": {
                "photon_budget": {"max_photons": 1200.0},
                "measurement_budget": {"sampling_rate": None}
            },
            "sensor": {
                "shot_noise": {"enabled": False},
                "read_noise_sigma": 0.0,
                "quantization_bits": None
            },
            "task": {"kind": "calibrate_and_reconstruct" if args.fit_gain_shift else "reconstruct_only"}
        },
        "mismatch": {
            "fit_operator": {
                "enabled": bool(args.fit_gain_shift),
                "casepack_id": args.casepack,
                # For generic matrix mode, theta is typically a small correction model.
                "theta_space": {
                    "gain_scale": {"min": 0.8, "max": 1.2},
                    "gain_bias": {"min": -0.05, "max": 0.05}
                },
                "search": {"candidates": 10, "refine_top_k": 2, "refine_steps": 6},
                "proxy_recon": {"solver_id": "tv_fista", "budget": {"iters": 30}},
                "scoring": {
                    "terms": [
                        {"name": "data_fidelity", "weight": 1.0},
                        {"name": "residual_whiteness", "weight": 0.3},
                        {"name": "theta_prior", "weight": 0.2}
                    ]
                },
                "stop": {"max_evals": 15, "plateau_delta": 1e-3, "verify_required": False}
            }
        },
        "recon": {
            "portfolio": {
                "solvers": [
                    {"id": "tv_fista", "params": {"lam": 0.02, "iters": 200}},
                    # If deepinv is installed, you can use a PnP solver:
                    # {"id": "pnp_deepinv", "params": {"denoiser": "drunet_gray", "sigma": 0.02, "iters": 50}}
                ],
                "selection": {"policy": "best_score", "score": "residual_whiteness+stability_proxy"}
            },
            "outputs": {"save_xhat": True, "save_uncertainty": False}
        },
        "analysis": {
            "metrics": [],
            "residual_tests": ["whiteness", "fourier_structure"],
            "advisor": {"enabled": True, "knobs_to_sweep": [], "max_candidates": 6, "output_actions": True}
        },
        "export": {
            "runbundle": {
                "path": str(out_dir.parent),
                "name": out_dir.name,
                "data_policy": {"mode": "auto", "copy_threshold_mb": 100},
                "codegen": {"enabled": True, "include_internal_state": True},
                "viewer": {"enabled": True}
            }
        }
    }

    resolved = resolve_validate(spec)
    if not resolved.validation.is_valid:
        print("Spec validation failed. Errors:")
        for e in resolved.validation.errors:
            print(" -", e)
        raise SystemExit(2)

    run = calibrate_recon(resolved.spec_resolved) if args.fit_gain_shift else calibrate_recon(resolved.spec_resolved)

    print("\n=== Run Summary ===")
    print("Run ID:", run.run_id)
    if run.operator_fit is not None:
        print("Operator correction:", run.operator_fit.theta_best)
    if run.diagnosis is not None:
        print("Diagnosis:", run.diagnosis.verdict, f"(conf={run.diagnosis.confidence:.2f})")
        if run.diagnosis.suggested_actions:
            print("Suggested actions:")
            for a in run.diagnosis.suggested_actions[:8]:
                print(f"  - {a.knob} {a.op} {a.val}")

    export = export_runbundle(run, out_dir=str(out_dir))
    print("\nRunBundle written to:", export.runbundle_path)
    print("View with:")
    print(f"  pwm view {export.runbundle_path}")

if __name__ == "__main__":
    main()
