"""The four lifecycle functions (design, simulate, reconstruct, calibrate) on fast modalities."""
import numpy as np
import pytest

from pwm_mode.imaging import core
from pwm_mode.imaging import lifecycle as L

if not core.PWM_CORE_SRC.exists():
    pytest.skip("pwm_core not available", allow_module_level=True)


@pytest.mark.parametrize("m", ["widefield", "lensless"])
def test_all_four_functions_work(m):
    r = L.audit_one(m, budget_s=120)
    assert "error" not in r, r.get("error")
    for f in ("design", "simulate", "reconstruct", "calibrate"):
        assert r[f]["ok"], (f, r[f])
    assert r["reconstruct"]["rel_error"] < r["reconstruct"]["rel_error_backprojection"]


def test_native_adjoint_is_used_when_it_passes_and_complexity_is_native():
    op = L.linear_operator("cassi")  # real-valued chain: must not be probed with complex data
    assert op["adjoint"] == "native" and op["dot_error"] < 1e-4


def test_blind_calibration_refuses_gain_and_reference_mode_recovers_it():
    m = "widefield"
    x = L.test_object(m)
    y = L.simulate(m, x, gain=1.25)["y_clean"]
    blind = {p["param"]: p for p in L.calibrate(m, y, k=5)["parameters"]}
    assert "reference" in blind["gain"]["status"]
    ref = {p["param"]: p for p in L.calibrate(m, y, x_ref=x, k=5)["parameters"]}
    assert abs(ref["gain"]["estimate"] - 1.25) < 1e-9


def test_reference_calibration_recovers_a_shifted_parameter():
    m = "lensless"
    c = next(c for c in L.calibratable(m) if c["node"])
    g = L._grid(c, 9)
    true_v = g[6]
    x = L.test_object(m)
    y = L.simulate(m, x, {c["node"]: {c["param"]: true_v}})["y_clean"]
    est = next(p for p in L.calibrate(m, y, x_ref=x, k=9)["parameters"] if p["param"] == c["param"])
    assert np.isclose(est["estimate"], true_v)
