"""
test_cassi_theta_fit.py

Unit tests for CASSI operator-fit ("theta fitting") workflow in PWM.

Goal:
- Verify the parametric CASSI operator wrapper is callable and differentiable enough
  for coarse search + local refinement.
- Verify the calibrate_recon pipeline can recover a simple theta (e.g., dx shift)
  on a tiny synthetic dataset.

Notes:
- This test uses a *toy* CASSI operator interface.
  In the real implementation, cassi_operator.py will implement A(theta) with:
    - coded aperture mask
    - per-band spatial shift (dispersion)
    - optional PSF blur
- Here we assume PWM exposes an endpoint calibrate_recon that can:
    y + operator(parametric cassi) -> fit theta -> reconstruct
- If your current code doesn't have full CASSI physics yet,
  you can keep this test xfail until cassi_operator is implemented.

Run:
    pytest -q packages/pwm_core/tests/test_cassi_theta_fit.py
"""

from __future__ import annotations

import pytest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")

from pwm_core.api.endpoints import resolve_validate, calibrate_recon


def _make_tiny_cassi_problem(H=16, W=16, L=8, dx_true=1.0, seed=0):
    """
    Create a tiny synthetic "cassi-like" problem by using a simplified forward model:
    y = sum_l shift_x(x[:,:,l], dx_true*l/L) * mask
    This is NOT a physically complete CASSI model—just enough to test theta recovery.
    """
    torch.manual_seed(seed)

    x = torch.zeros(H, W, L)
    # Put a small bright square in a few bands
    x[4:7, 5:8, 2] = 1.0
    x[9:12, 8:11, 6] = 0.8

    mask = (torch.rand(H, W) > 0.5).float()

    def shift_x(img2d, shift):
        # integer shift for toy (round)
        s = int(round(float(shift)))
        if s == 0:
            return img2d
        if s > 0:
            out = torch.zeros_like(img2d)
            out[:, s:] = img2d[:, :-s]
            return out
        else:
            out = torch.zeros_like(img2d)
            out[:, :s] = img2d[:, -s:]
            return out

    y = torch.zeros(H, W)
    for l in range(L):
        y += shift_x(x[:, :, l], dx_true * (l / max(1, (L - 1)))) * mask

    # add mild noise
    y = y + 0.01 * torch.randn_like(y)
    return x, y, mask


@pytest.mark.xfail(reason="CASSI operator + theta-fit may not be implemented yet")
def test_cassi_theta_fit_recovers_dx_toy(tmp_path):
    H, W, L = 16, 16, 8
    dx_true = 1.0
    x_true, y_meas, mask = _make_tiny_cassi_problem(H=H, W=W, L=L, dx_true=dx_true, seed=7)

    spec = {
        "version": "0.2.1",
        "id": "test_cassi_theta_fit",
        "input": {
            "mode": "measured",
            "data": {
                "y_source": {"kind": "tensor", "tensor": y_meas, "format": "torch"}
            },
            "operator": {
                "kind": "parametric",
                "parametric": {
                    "operator_id": "cassi",
                    # Provide minimal operator assets for the toy:
                    # In real CASSI, these would be param packs with mask and dispersion models.
                    "assets": {"mask": {"kind": "tensor", "tensor": mask, "format": "torch"}},
                    "theta_space": {
                        "dx": {"min": -2.0, "max": 2.0},
                        "dy": {"min": 0.0, "max": 0.0},  # fixed
                        "disp_poly": {"coeffs_min": [0.0, 0.0], "coeffs_max": [0.0, 0.0]},  # fixed in toy
                    },
                }
            }
        },
        "states": {
            "physics": {"modality": "cassi", "dims": {"x": [H, W, L], "y": [H, W]}},
            "task": {"kind": "calibrate_and_reconstruct"},
            "sensor": {"shot_noise": {"enabled": False}, "read_noise_sigma": 0.0},
        },
        "mismatch": {
            "fit_operator": {
                "enabled": True,
                "casepack_id": "cassi_measured_y_fit_theta_v1",
                "search": {"candidates": 10, "refine_top_k": 2, "refine_steps": 6},
                "proxy_recon": {"solver_id": "tv_fista", "budget": {"iters": 20}},
                "stop": {"max_evals": 16, "plateau_delta": 1e-4, "verify_required": False}
            }
        },
        "recon": {
            "portfolio": {"solvers": [{"id": "tv_fista", "params": {"lam": 0.02, "iters": 60}}]},
            "outputs": {"save_xhat": True}
        },
        "export": {"runbundle": {"path": str(tmp_path), "name": "rb_cassi", "data_policy": {"mode": "copy"}}}
    }

    resolved = resolve_validate(spec)
    assert resolved.validation.is_valid, f"Validation errors: {resolved.validation.errors}"

    run = calibrate_recon(resolved.spec_resolved)

    assert run.operator_fit is not None
    theta = run.operator_fit.theta_best
    assert "dx" in theta

    # We only require it to be in the vicinity for toy physics
    assert abs(theta["dx"] - dx_true) < 0.8

    # Ensure xhat saved
    assert run.outputs is not None
    assert run.outputs.xhat is not None


@pytest.mark.xfail(reason="CASSI operator may not be fully implemented yet")
def test_cassi_theta_fit_reports_operator_artifacts(tmp_path):
    H, W, L = 12, 12, 6
    x_true, y_meas, mask = _make_tiny_cassi_problem(H=H, W=W, L=L, dx_true=1.5, seed=3)

    spec = {
        "version": "0.2.1",
        "id": "test_cassi_artifacts",
        "input": {"mode": "measured",
                  "data": {"y_source": {"kind": "tensor", "tensor": y_meas, "format": "torch"}},
                  "operator": {"kind": "parametric",
                               "parametric": {"operator_id": "cassi",
                                              "assets": {"mask": {"kind": "tensor", "tensor": mask, "format": "torch"}},
                                              "theta_space": {"dx": {"min": -2.0, "max": 2.0}}}}},
        "states": {"physics": {"modality": "cassi", "dims": {"x": [H, W, L], "y": [H, W]}},
                   "task": {"kind": "calibrate_and_reconstruct"}},
        "mismatch": {"fit_operator": {"enabled": True, "casepack_id": "cassi_measured_y_fit_theta_v1",
                                      "search": {"candidates": 6}, "stop": {"max_evals": 10}}},
        "recon": {"portfolio": {"solvers": [{"id": "tv_fista", "params": {"lam": 0.02, "iters": 40}}]}},
        "export": {"runbundle": {"path": str(tmp_path), "name": "rb_cassi2", "data_policy": {"mode": "copy"}}}
    }

    resolved = resolve_validate(spec)
    assert resolved.validation.is_valid

    run = calibrate_recon(resolved.spec_resolved)
    assert run.operator_fit is not None

    # Implementation detail: operator_fit should provide a path/index of saved artifacts
    if getattr(run.operator_fit, "artifacts", None) is not None:
        assert isinstance(run.operator_fit.artifacts, dict)
