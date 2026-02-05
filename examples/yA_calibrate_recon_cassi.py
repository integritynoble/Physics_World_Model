#!/usr/bin/env python3
"""
yA_calibrate_recon_cassi.py

Example: Measured y + (parametric) CASSI operator A(theta) -> fit theta -> reconstruct -> diagnose -> export RunBundle

This shows the "operator correction mode" for CASSI/SCI-spectral.

Assumptions:
- You installed pwm_core:
    pip install -e packages/pwm_core
- You have a measured y saved as a tensor (pt/npz/mat supported via io)
- You know (or can guess) the latent x dimensions (H,W,L) for the reconstruction target
- You want PWM to fit a small set of operator parameters theta:
    dx, dy, dispersion polynomial coefficients, psf_sigma (optional), gain (optional)

Run:
    python examples/yA_calibrate_recon_cassi.py \
        --y data/measured_y.pt \
        --Hx 256 --Wx 256 --L 31 \
        --out runs/latest

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
    ap.add_argument("--Hx", type=int, required=True, help="Latent x height.")
    ap.add_argument("--Wx", type=int, required=True, help="Latent x width.")
    ap.add_argument("--L", type=int, required=True, help="Spectral bands (latent channels).")
    ap.add_argument("--out", type=str, default="runs/latest", help="RunBundle output directory.")
    ap.add_argument("--casepack", type=str, default="cassi_measured_y_fit_theta_v1",
                    help="CasePack id for operator-fit CASSI workflow.")
    # Optional user hints / overrides:
    ap.add_argument("--low_dose", action="store_true", help="Hint: low dose (reduces max_photons, increases noise).")
    ap.add_argument("--high_compression", action="store_true", help="Hint: high compression / aggressive sampling.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Build a minimal ExperimentSpec for measured y + parametric CASSI operator-fit
    # Note: this is intentionally minimal; CasePack + resolve_validate fill in many defaults.
    spec = {
        "version": "0.2.1",
        "id": "cassi_measured_fit_demo",
        "input": {
            "mode": "measured",
            "data": {
                "y_source": {
                    "kind": "file",
                    "path": args.y,
                    "format": args.y.split(".")[-1].lower()
                }
            },
            "operator": {
                "kind": "parametric",
                "parametric": {
                    "operator_id": "cassi",
                    # theta search space (safe bounds); these are typical starter bounds.
                    "theta_space": {
                        "dx": {"min": -2.0, "max": 2.0},
                        "dy": {"min": -2.0, "max": 2.0},
                        # dispersion polynomial coeff bounds (example for 2 coeffs)
                        "disp_poly": {"coeffs_min": [-0.02, -0.01], "coeffs_max": [0.02, 0.01]},
                        "psf_sigma": {"min": 0.8, "max": 2.2},
                        "gain_scale": {"min": 0.8, "max": 1.2},
                        "gain_bias": {"min": -0.02, "max": 0.02}
                    }
                }
            }
        },
        "states": {
            "physics": {
                "modality": "cassi",
                "dims": {"x": [args.Hx, args.Wx, args.L]}
            },
            "budget": {
                "photon_budget": {"max_photons": 1200.0},
                "measurement_budget": {"sampling_rate": 1.0}
            },
            "sensor": {
                "shot_noise": {"enabled": True, "model": "poisson"},
                "read_noise_sigma": 0.01
            },
            "task": {"kind": "calibrate_and_reconstruct"}
        },
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "casepack_id": args.casepack,
                "search": {"candidates": 12, "refine_top_k": 3, "refine_steps": 8},
                "proxy_recon": {"solver_id": "tv_fista", "budget": {"iters": 40}},
                "scoring": {
                    "terms": [
                        {"name": "data_fidelity", "weight": 1.0},
                        {"name": "residual_whiteness", "weight": 0.5},
                        {"name": "theta_prior", "weight": 0.2}
                    ]
                },
                "stop": {"max_evals": 20, "plateau_delta": 1e-3, "verify_required": True}
            }
        },
        "recon": {
            "portfolio": {
                "solvers": [
                    {"id": "tv_fista", "params": {"lam": 0.02, "iters": 200}},
                    # Optional: if deepinv installed, you can enable PnP:
                    # {"id": "pnp_deepinv", "params": {"denoiser": "drunet_gray", "sigma": 0.02, "iters": 50}}
                ],
                "selection": {"policy": "best_score", "score": "residual_whiteness+stability_proxy"}
            },
            "outputs": {"save_xhat": True, "save_uncertainty": False}
        },
        "analysis": {
            "metrics": ["psnr", "ssim"],
            "residual_tests": ["whiteness", "fourier_structure"],
            "advisor": {
                "enabled": True,
                "knobs_to_sweep": [
                    "states.budget.photon_budget.max_photons",
                    "states.budget.measurement_budget.sampling_rate"
                ],
                "max_candidates": 10,
                "output_actions": True
            }
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

    # Apply optional hints (simple, safe overrides)
    if args.low_dose:
        spec["states"]["budget"]["photon_budget"]["max_photons"] = 400.0
        spec["states"]["sensor"]["read_noise_sigma"] = 0.015
    if args.high_compression:
        spec["states"]["budget"]["measurement_budget"]["sampling_rate"] = 0.3

    # Resolve & validate (fills defaults, clamps, auto-repair)
    resolved = resolve_validate(spec)

    if not resolved.validation.is_valid:
        print("Spec validation failed. Errors:")
        for e in resolved.validation.errors:
            print(" -", e)
        raise SystemExit(2)

    # Run calibration + reconstruction
    run = calibrate_recon(resolved.spec_resolved)

    print("\n=== Calibrate+Recon Summary ===")
    print("Run ID:", run.run_id)
    if run.operator_fit is not None:
        print("Best theta:", run.operator_fit.theta_best)
        print("Fit score:", run.operator_fit.best_score)

    if run.diagnosis is not None:
        print("Diagnosis:", run.diagnosis.verdict, f"(conf={run.diagnosis.confidence:.2f})")
        if run.diagnosis.suggested_actions:
            print("Suggested actions:")
            for a in run.diagnosis.suggested_actions[:8]:
                print(f"  - {a.knob} {a.op} {a.val}")

    # Export RunBundle
    export = export_runbundle(run, out_dir=str(out_dir))
    print("\nRunBundle written to:", export.runbundle_path)
    print("View with:")
    print(f"  pwm view {export.runbundle_path}")

if __name__ == "__main__":
    main()
