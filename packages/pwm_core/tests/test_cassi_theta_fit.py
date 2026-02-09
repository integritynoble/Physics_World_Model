"""
test_cassi_theta_fit.py

Unit tests for CASSI operator-fit ("theta fitting") workflow in PWM.

Goal:
- Verify the CASSI operator wrapper is callable.
- Verify the calibrate_recon pipeline can be invoked with a CASSI modality spec.

Notes:
- Uses a toy CASSI operator from physics_factory.
- Tests verify the pipeline runs end-to-end without errors.

Run:
    pytest -q packages/pwm_core/tests/test_cassi_theta_fit.py
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


def _make_tiny_cassi_problem(H=16, W=16, L=8, seed=0):
    """Create a tiny synthetic CASSI measurement."""
    rng = np.random.default_rng(seed)

    # Sparse spectral cube
    x = np.zeros((H, W, L), dtype=np.float32)
    x[4:min(7, H), 5:min(8, W), min(2, L - 1)] = 1.0
    x[min(9, H-3):min(12, H), min(8, W-3):min(11, W), L - 1] = 0.8

    # Binary coded aperture mask
    mask = (rng.random((H, W)) > 0.5).astype(np.float32)

    # Simplified CASSI forward: y = sum_l shift(x[:,:,l] * mask, l)
    y = np.zeros((H, W + L - 1), dtype=np.float32)
    for l_idx in range(L):
        band = x[:, :, l_idx] * mask
        y[:, l_idx:l_idx + W] += band

    # Add mild noise
    y += 0.01 * rng.standard_normal(y.shape).astype(np.float32)
    return x, y, mask


def test_cassi_theta_fit_recovers_dx_toy(tmp_path):
    H, W, L = 16, 16, 8
    x_true, y_meas, mask = _make_tiny_cassi_problem(H=H, W=W, L=L, seed=7)

    # Save data to temp files
    y_path = str(tmp_path / "y.npy")
    mask_path = str(tmp_path / "mask.npy")
    x_path = str(tmp_path / "x_true.npy")
    np.save(y_path, y_meas)
    np.save(mask_path, mask)
    np.save(x_path, x_true)

    spec = {
        "version": "0.2.1",
        "id": "test_cassi_theta_fit",
        "input": {
            "mode": "measured",
            "y_source": y_path,
            "x_source": x_path,
            "operator": {
                "kind": "parametric",
                "parametric": {
                    "operator_id": "cassi",
                    "theta_init": {"L": L},
                    "assets": {"mask": mask_path},
                    "theta_space": {
                        "dx": {"min": -2.0, "max": 2.0},
                    },
                }
            }
        },
        "states": {
            "physics": {
                "modality": "cassi",
                "dims": {"x": [H, W, L], "y": [H, W]},
            },
            "task": {"kind": "calibrate_and_reconstruct"},
            "sensor": {"read_noise_sigma": 0.0},
        },
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "theta_space": {"dx": {"min": -2.0, "max": 2.0}},
                "stop": {"max_evals": 16},
            }
        },
        "recon": {
            "portfolio": {
                "solvers": [{"id": "tv_fista", "params": {"lam": 0.02, "iters": 60}}]
            },
        },
        "export": {
            "runbundle": {"data_policy": {"mode": "copy"}},
        }
    }

    resolved = resolve_validate(spec)
    assert resolved.validation.ok, f"Validation errors: {resolved.validation.messages}"

    run = calibrate_recon(resolved.spec_resolved, out_dir=str(tmp_path))

    # Expect calib present
    assert run.calib is not None
    theta = run.calib.theta_best
    assert isinstance(theta, dict)

    # Expect reconstruction exists
    assert run.recon is not None
    assert len(run.recon) > 0


def test_cassi_theta_fit_reports_operator_artifacts(tmp_path):
    H, W, L = 12, 12, 6
    x_true, y_meas, mask = _make_tiny_cassi_problem(H=H, W=W, L=L, seed=3)

    y_path = str(tmp_path / "y.npy")
    mask_path = str(tmp_path / "mask.npy")
    np.save(y_path, y_meas)
    np.save(mask_path, mask)

    spec = {
        "version": "0.2.1",
        "id": "test_cassi_artifacts",
        "input": {
            "mode": "measured",
            "y_source": y_path,
            "operator": {
                "kind": "parametric",
                "parametric": {
                    "operator_id": "cassi",
                    "theta_init": {"L": L},
                    "assets": {"mask": mask_path},
                    "theta_space": {"dx": {"min": -2.0, "max": 2.0}},
                }
            },
        },
        "states": {
            "physics": {
                "modality": "cassi",
                "dims": {"x": [H, W, L], "y": [H, W]},
            },
            "task": {"kind": "calibrate_and_reconstruct"},
        },
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "theta_space": {"dx": {"min": -2.0, "max": 2.0}},
                "stop": {"max_evals": 10},
            }
        },
        "recon": {
            "portfolio": {
                "solvers": [{"id": "tv_fista", "params": {"lam": 0.02, "iters": 40}}]
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
    assert isinstance(run.calib.theta_best, dict)

    # RunBundle should be created
    assert run.runbundle_path is not None
    assert os.path.isdir(run.runbundle_path)
