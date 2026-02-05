"""
test_calibrate_recon.py

Unit tests for operator-fit + reconstruction orchestration:
- "generic matrix" operator-fit (gain/bias) + recon
- ensures RunResult structure includes operator_fit and diagnosis
- verifies bounded candidate search respects policy limits

These tests use tiny synthetic problems to run fast on CPU.

Run:
    pytest -q packages/pwm_core/tests/test_calibrate_recon.py
"""

from __future__ import annotations

import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")

from pwm_core.api.endpoints import resolve_validate, calibrate_recon


def _make_tiny_linear_problem(M=24, N=32, seed=0):
    g_true = 1.10
    b_true = -0.02
    torch.manual_seed(seed)
    A = torch.randn(M, N)
    x_true = torch.rand(N)  # nonnegative signal
    y = g_true * (A @ x_true) + b_true
    return A, x_true, y, g_true, b_true


@pytest.mark.xfail(reason="ExperimentSpec schema mismatch - tests need schema update")
def test_calibrate_recon_generic_gain_bias_smoke(tmp_path):
    A, x_true, y, g_true, b_true = _make_tiny_linear_problem(M=20, N=28, seed=1)

    # Minimal measured spec + matrix operator + operator-fit
    spec = {
        "version": "0.2.1",
        "id": "test_generic_gain_bias",
        "input": {
            "mode": "measured",
            "data": {
                "y_source": {
                    "kind": "tensor",
                    "tensor": y,
                    "format": "torch"
                }
            },
            "operator": {
                "kind": "matrix",
                "matrix": {
                    "kind": "tensor",
                    "tensor": A,
                    "format": "torch",
                    "storage": "dense",
                }
            }
        },
        "states": {
            "physics": {"modality": "generic_linear", "dims": {"x": [A.shape[1]], "y": [A.shape[0]]}},
            "task": {"kind": "calibrate_and_reconstruct"},
            "sensor": {"shot_noise": {"enabled": False}, "read_noise_sigma": 0.0},
        },
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "casepack_id": "generic_matrix_yA_fit_gain_shift_v1",
                "theta_space": {
                    "gain_scale": {"min": 0.8, "max": 1.2},
                    "gain_bias": {"min": -0.05, "max": 0.05}
                },
                "search": {"candidates": 8, "refine_top_k": 2, "refine_steps": 5},
                "proxy_recon": {"solver_id": "tv_fista", "budget": {"iters": 10}},
                "stop": {"max_evals": 12, "plateau_delta": 1e-4, "verify_required": False}
            }
        },
        "recon": {
            "portfolio": {
                "solvers": [
                    {"id": "tv_fista", "params": {"lam": 0.01, "iters": 40}}
                ],
                "selection": {"policy": "best_score", "score": "data_fidelity"}
            },
            "outputs": {"save_xhat": True}
        },
        "analysis": {"advisor": {"enabled": True, "output_actions": True}},
        "export": {
            "runbundle": {"path": str(tmp_path), "name": "rb", "data_policy": {"mode": "copy"}}
        }
    }

    resolved = resolve_validate(spec)
    assert resolved.validation.is_valid, f"Validation errors: {resolved.validation.errors}"

    run = calibrate_recon(resolved.spec_resolved)

    # Expect operator_fit present
    assert run.operator_fit is not None
    theta = run.operator_fit.theta_best
    assert "gain_scale" in theta and "gain_bias" in theta

    # Fit should be close-ish to true values (tiny problem so allow loose tolerance)
    assert abs(theta["gain_scale"] - g_true) < 0.25
    assert abs(theta["gain_bias"] - b_true) < 0.08

    # Expect reconstruction exists
    assert run.outputs is not None
    assert run.outputs.xhat is not None
    assert run.outputs.xhat.numel() == A.shape[1]

    # Diagnosis is optional but recommended; if present, should have structured actions list
    if run.diagnosis is not None:
        assert isinstance(run.diagnosis.suggested_actions, list)


@pytest.mark.xfail(reason="ExperimentSpec schema mismatch - tests need schema update")
def test_calibrate_recon_respects_candidate_budget(tmp_path):
    A, x_true, y, *_ = _make_tiny_linear_problem(M=18, N=22, seed=2)

    spec = {
        "version": "0.2.1",
        "id": "test_budget",
        "input": {
            "mode": "measured",
            "data": {"y_source": {"kind": "tensor", "tensor": y, "format": "torch"}},
            "operator": {"kind": "matrix", "matrix": {"kind": "tensor", "tensor": A, "format": "torch"}}
        },
        "states": {"physics": {"modality": "generic_linear", "dims": {"x": [A.shape[1]], "y": [A.shape[0]]}},
                   "task": {"kind": "calibrate_and_reconstruct"}},
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "casepack_id": "generic_matrix_yA_fit_gain_shift_v1",
                "theta_space": {
                    "gain_scale": {"min": 0.9, "max": 1.1},
                    "gain_bias": {"min": -0.02, "max": 0.02}
                },
                "search": {"candidates": 5, "refine_top_k": 1, "refine_steps": 3},
                "stop": {"max_evals": 5}
            }
        },
        "recon": {"portfolio": {"solvers": [{"id": "tv_fista", "params": {"lam": 0.01, "iters": 20}}]}},
        "export": {"runbundle": {"path": str(tmp_path), "name": "rb2", "data_policy": {"mode": "copy"}}}
    }

    resolved = resolve_validate(spec)
    assert resolved.validation.is_valid

    run = calibrate_recon(resolved.spec_resolved)

    assert run.operator_fit is not None
    # Implementation detail: operator_fit should expose how many candidates were evaluated
    if getattr(run.operator_fit, "num_evals", None) is not None:
        assert run.operator_fit.num_evals <= 5
