import pytest

from pwm_mode.ledger import (Conflict, InsufficientPWM, Ledger, NotFound, TREASURY, micro,
                             split)

SVC = {"id": "ldct.solve", "price_pwm": "2.0", "agreement": {"id": "t", "version": 1},
       "beneficiaries": [{"account": "contrib:a", "share_bps": 6000},
                         {"account": "contrib:b", "share_bps": 2000},
                         {"account": "contrib:c", "share_bps": 2000}]}


@pytest.fixture
def L(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    led.fund("user:u", "10")
    return led


def test_only_test_network():
    with pytest.raises(PermissionError):
        Ledger(":memory:", network="main")


def test_split_is_exact_and_treasury_is_3_percent():
    lines = split(micro("2.0"), SVC["beneficiaries"])
    assert sum(l["micro_pwm"] for l in lines) == micro("2.0")
    t = [l for l in lines if l["account"] == TREASURY][0]
    assert t["micro_pwm"] == micro("0.06")
    assert [l["micro_pwm"] for l in lines[:3]] == [micro("1.164"), micro("0.388"), micro("0.388")]
    odd = split(7, SVC["beneficiaries"])           # rounding remainder stays inside gross
    assert sum(l["micro_pwm"] for l in odd) == 7


def test_settle_pays_contributors_and_treasury_and_is_final(L):
    q = L.quote("user:u", SVC)
    L.reserve("r1", q["quote_id"])
    assert L.balance_micro("user:u") == micro("8")
    r = L.settle("r1", evidence={"judge": "pass"})
    assert r["status"] == "settled" and r["final"] and r["charged_pwm"] == "2"
    assert L.balance_micro("contrib:a") == micro("1.164")
    assert L.balance_micro(TREASURY) == micro("0.06")
    assert L.balance_micro("hold:r1") == 0
    assert L.books_balance() == 0
    assert L.settle("r1")["sha256"] == r["sha256"]           # replay: same receipt
    with pytest.raises(Conflict):
        L.settle("r1", gross="1.0")                          # different amount: conflict
    with pytest.raises(Conflict):
        L.release("r1", "try to refund")                      # settled is final


def test_failed_run_is_not_charged(L):
    q = L.quote("user:u", SVC)
    L.reserve("r2", q["quote_id"])
    r = L.release("r2", "failed: solver crashed")
    assert r["status"] == "released" and r["charged_pwm"] == "0" and r["lines"] == []
    assert L.balance_micro("user:u") == micro("10")
    assert L.balance_micro(TREASURY) == 0
    with pytest.raises(Conflict):
        L.settle("r2")


def test_metered_charge_capped_with_refund_and_over_cap(L):
    svc = dict(SVC, pricing="metered")
    q = L.quote("user:u", svc, cap="1.0")
    L.reserve("m1", q["quote_id"])
    r = L.settle("m1", gross="0.25", usage={"input": 100, "output": 50})
    assert r["charged_pwm"] == "0.25" and r["refunded_pwm"] == "0.75"
    assert L.balance_micro("user:u") == micro("9.75")
    q = L.quote("user:u", svc, cap="0.1")
    L.reserve("m2", q["quote_id"])
    r = L.settle("m2", gross="0.4")
    assert r["charged_pwm"] == "0.1" and r["over_cap_pwm"] == "0.3"
    assert L.books_balance() == 0


def test_insufficient_balance_and_quote_reuse(L):
    big = dict(SVC, price_pwm="50")
    with pytest.raises(InsufficientPWM):
        L.reserve("x", L.quote("user:u", big)["quote_id"])
    q = L.quote("user:u", SVC)["quote_id"]
    L.reserve("a", q)
    assert L.reserve("a", q)["replay"] is True                # same run: no double hold
    assert L.balance_micro("user:u") == micro("8")
    with pytest.raises(Conflict):
        L.reserve("b", q)                                     # one quote, one run
    with pytest.raises(NotFound):
        L.reserve("c", "q_missing")


def test_open_runs_survive_restart(tmp_path):
    L = Ledger(tmp_path / "l.sqlite")
    L.fund("user:u", "5")
    L.reserve("crash", L.quote("user:u", SVC)["quote_id"])
    L2 = Ledger(tmp_path / "l.sqlite")
    assert [r["run_id"] for r in L2.open_runs()] == ["crash"]
    L2.release("crash", "reconciled: no delivery evidence")
    assert L2.balance_micro("user:u") == micro("5")
