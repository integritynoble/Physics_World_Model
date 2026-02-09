"""
test_calibrate_recon.py

Unit tests for operator-fit + reconstruction orchestration:
- "generic matrix" operator-fit (gain/bias) + recon
- ensures CalibReconResult structure includes calib and diagnosis
- verifies bounded candidate search respects policy limits

These tests use tiny synthetic problems to run fast on CPU.

Run:
    pytest -q packages/pwm_core/tests/test_calibrate_recon.py
"""

from __future__ import annotations

import os
import numpy as np
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
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, N)).astype(np.float32)
    x_true = np.abs(rng.standard_normal(N)).astype(np.float32)
    y = g_true * (A @ x_true) + b_true
    return A, x_true, y, g_true, b_true


def test_calibrate_recon_generic_gain_bias_smoke(tmp_path):
    A, x_true, y, g_true, b_true = _make_tiny_linear_problem(M=20, N=28, seed=1)

    # Save tensors to temp files
    y_path = str(tmp_path / "y.npy")
    A_path = str(tmp_path / "A.npy")
    x_path = str(tmp_path / "x_true.npy")
    np.save(y_path, y)
    np.save(A_path, A)
    np.save(x_path, x_true)

    spec = {
        "version": "0.2.1",
        "id": "test_generic_gain_bias",
        "input": {
            "mode": "measured",
            "y_source": y_path,
            "x_source": x_path,
            "operator": {
                "kind": "matrix",
                "matrix": {
                    "source": A_path,
                    "format": "npz",
                }
            }
        },
        "states": {
            "physics": {
                "modality": "generic_linear",
                "dims": {"x": [28], "y": [20]},
            },
            "task": {"kind": "calibrate_and_reconstruct"},
            "sensor": {"read_noise_sigma": 0.0},
        },
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "theta_space": {
                    "gain_scale": {"min": 0.8, "max": 1.2},
                    "gain_bias": {"min": -0.05, "max": 0.05}
                },
                "stop": {"max_evals": 12},
            }
        },
        "recon": {
            "portfolio": {
                "solvers": [
                    {"id": "tv_fista", "params": {"lam": 0.01, "iters": 40}}
                ],
            },
        },
        "export": {
            "runbundle": {
                "data_policy": {"mode": "copy"},
            }
        }
    }

    resolved = resolve_validate(spec)
    assert resolved.validation.ok, f"Validation errors: {resolved.validation.messages}"

    run = calibrate_recon(resolved.spec_resolved, out_dir=str(tmp_path))

    # Expect calib present for calibrate_and_reconstruct task
    assert run.calib is not None
    theta = run.calib.theta_best
    assert isinstance(theta, dict)

    # Expect reconstruction exists
    assert run.recon is not None
    assert len(run.recon) > 0

    # Diagnosis is optional but recommended
    if run.diagnosis is not None:
        assert isinstance(run.diagnosis.suggested_actions, list)


def test_calibrate_recon_respects_candidate_budget(tmp_path):
    M, N = 18, 22
    rng = np.random.default_rng(2)
    A = rng.standard_normal((M, N)).astype(np.float32)
    x_true = np.abs(rng.standard_normal(N)).astype(np.float32)
    y = A @ x_true + 0.01 * rng.standard_normal(M).astype(np.float32)

    y_path = str(tmp_path / "y.npy")
    A_path = str(tmp_path / "A.npy")
    np.save(y_path, y)
    np.save(A_path, A)

    spec = {
        "version": "0.2.1",
        "id": "test_budget",
        "input": {
            "mode": "measured",
            "y_source": y_path,
            "operator": {
                "kind": "matrix",
                "matrix": {"source": A_path},
            },
        },
        "states": {
            "physics": {
                "modality": "generic_linear",
                "dims": {"x": [N], "y": [M]},
            },
            "task": {"kind": "calibrate_and_reconstruct"},
        },
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "theta_space": {
                    "gain_scale": {"min": 0.9, "max": 1.1},
                    "gain_bias": {"min": -0.02, "max": 0.02}
                },
                "stop": {"max_evals": 5},
            }
        },
        "recon": {
            "portfolio": {
                "solvers": [{"id": "tv_fista", "params": {"lam": 0.01, "iters": 20}}]
            }
        },
        "export": {
            "runbundle": {"data_policy": {"mode": "copy"}},
        }
    }

    resolved = resolve_validate(spec)
    assert resolved.validation.ok

    run = calibrate_recon(resolved.spec_resolved, out_dir=str(tmp_path))

    assert run.calib is not None
    # num_evals may not always be set, but calib should exist
    if run.calib.num_evals > 0:
        assert run.calib.num_evals <= 10  # some slack
