import json

import numpy as np
import pytest


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("PWM_MODE_LEDGER", str(tmp_path / "ledger.sqlite"))
    monkeypatch.setenv("PWM_MODE_HOME", str(tmp_path / "home"))
    from pwm_mode import mcp_server as m
    monkeypatch.setattr(m, "WORKSPACE", tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    monkeypatch.setattr(m, "_ledger", None)
    monkeypatch.setattr(m, "_session", {"cap_micro": 0, "spent_micro": 0, "id": "s_test"})
    m.ledger().fund(m.PAYER, "10")
    return m


def J(s):
    return json.loads(s)


def test_nothing_charged_without_authorisation(srv):
    out = J(srv.ldct_solve(seed=0))
    assert out["error"] == "not authorised" and out["charged_pwm"] == "0"
    assert srv.ledger().balance(srv.PAYER)["pwm"] == "10"


def test_ldct_solve_settles_with_receipt(srv):
    srv.pwm_authorize("5")
    out = J(srv.ldct_solve(seed=0))
    assert out["result"]["status"] == "delivered" and out["result"]["verdict"]["passed"]
    r = out["receipt"]
    assert r["status"] == "settled" and r["charged_pwm"] == "2"
    accts = {l["account"]: l["pwm"] for l in r["lines"]}
    assert accts["treasury"] == "0.06"
    assert accts["contrib:test-ldct-method"] == "1.164"
    assert J(srv.pwm_receipt(r["run_id"]))["sha256"] == r["sha256"]
    assert srv.ledger().balance(srv.PAYER)["pwm"] == "8"
    assert srv.ledger().books_balance() == 0


def test_failed_solve_costs_nothing(srv, monkeypatch):
    srv.pwm_authorize("5")
    monkeypatch.setenv("PWM_MODE_FAIL_INJECTION", "1")
    out = J(srv.ldct_solve(seed=1))
    assert out["receipt"]["status"] == "released" and out["receipt"]["charged_pwm"] == "0"
    assert srv.ledger().balance(srv.PAYER)["pwm"] == "10"
    assert J(srv.pwm_balance())["session_spent_pwm"] == "0"


def test_problem_then_judge_user_reconstruction(srv):
    srv.pwm_authorize("5")
    p = J(srv.ldct_problem(seed=2, dest="case"))
    pid = p["result"]["problem_id"]
    ws = srv.WORKSPACE / "case"
    assert not (ws / "data" / "full_dose.npy").exists()           # key withheld
    low = np.load(ws / "data" / "low_dose.npy")
    (ws / "results").mkdir(exist_ok=True)
    np.save(ws / "results" / "reconstruction.npy", low)            # "do nothing" submission
    v = J(srv.ldct_judge(pid, "case/results/reconstruction.npy"))
    assert v["result"]["passed"] is False                          # judge refuses no-op
    assert v["receipt"]["status"] == "settled"                     # a verdict was delivered
    bad = J(srv.ldct_judge(pid, "case/data/low_dose.npy".replace("low_dose", "missing")))
    assert bad["receipt"]["charged_pwm"] == "0"                    # no verdict, no charge
    assert srv.ledger().balance(srv.PAYER)["pwm"] == "9.3"         # 10 - 0.2 - 0.5


def test_path_escape_is_refused_and_not_charged(srv):
    srv.pwm_authorize("1")
    out = J(srv.ldct_problem(seed=0, dest="../../etc"))
    assert out["status"] == "failed" and out["receipt"]["charged_pwm"] == "0"


def test_field_tool_and_cap_exhaustion(srv):
    srv.pwm_authorize("0.02")
    a = J(srv.ldct_field("field_benchmark_protocol"))
    assert a["receipt"]["status"] == "settled"
    J(srv.ldct_field("field_principles"))
    third = J(srv.ldct_field("field_principles"))
    assert third["error"] == "not authorised"
