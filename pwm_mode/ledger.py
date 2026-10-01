"""PWM-mode test ledger: quote -> reserve -> settle | release, one receipt per run.

Carried over from the 21 Sep runtime's reference controller
(pwm/task/prototypes/2026-09-21/ai4science-reliable-terminal/demo/controller.py):
integer micro-PWM amounts, a durable SQLite store, reservation before work,
settlement only of a reconciled delivery, idempotent replay and a conflict on a
changed amount, and the unused reservation returned. What changes is the split:
the 21 Sep 99/1 provider/method split becomes the service's contributor
agreement (pwm D9, D16) with a 3% inclusive treasury line (pwm D7, ai4science D6).

Rules enforced here, not by callers:
  - a settled run is final: no refund, no release, a replay returns the receipt;
  - a failed or rejected run is released: 0 PWM charged, on its own receipt;
  - gross never exceeds the reservation the user authorised;
  - only the test ledger exists. Live charging waits for gate G3 and the owner.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

MICRO = 1_000_000
TREASURY = "treasury"
TREASURY_BPS = 300          # 3% inclusive (pwm D7, ai4science D6)


class LedgerError(Exception):
    code = "ledger_error"


class InsufficientPWM(LedgerError):
    code = "insufficient_pwm"


class Conflict(LedgerError):
    code = "conflict"


class NotFound(LedgerError):
    code = "not_found"


def micro(value) -> int:
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("Amount must be finite and nonnegative")
    return int(number * MICRO)


def pwm(micro_amount: int) -> str:
    return str(Decimal(micro_amount) / MICRO)


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def split(gross: int, beneficiaries: List[Dict[str, Any]],
          treasury_bps: int = TREASURY_BPS) -> List[Dict[str, Any]]:
    """Treasury takes its inclusive share of gross; the rest goes to the agreed
    beneficiaries by share_bps (summing to 10000). Integer arithmetic; the
    rounding remainder goes to the first beneficiary so the lines sum to gross."""
    if not beneficiaries or sum(b["share_bps"] for b in beneficiaries) != 10000:
        raise ValueError("Beneficiary shares must sum to 10000 bps")
    treasury = gross * treasury_bps // 10000
    net = gross - treasury
    lines = [{"account": b["account"], "role": b.get("role", "contributor"),
              "micro_pwm": net * b["share_bps"] // 10000} for b in beneficiaries]
    lines[0]["micro_pwm"] += net - sum(l["micro_pwm"] for l in lines)
    lines.append({"account": TREASURY, "role": "treasury", "micro_pwm": treasury})
    return lines


class Ledger:
    """Durable double-entry test ledger. Every movement is an entry; a balance
    is the sum of its entries, so nothing can be moved without a record."""

    def __init__(self, path, *, network: str = "test"):
        if network != "test":
            raise PermissionError("Only the test ledger exists: live charging needs gate G3 "
                                  "and the owner's approval")
        self.network = network
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), isolation_level=None,
                                  check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS entries(id INTEGER PRIMARY KEY, ts REAL, account TEXT,
            delta INTEGER, run TEXT, kind TEXT, memo TEXT);
        CREATE INDEX IF NOT EXISTS entries_account ON entries(account);
        CREATE TABLE IF NOT EXISTS quotes(id TEXT PRIMARY KEY, ts REAL, payer TEXT, service TEXT,
            cap INTEGER, agreement TEXT, detail TEXT, expires REAL);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, quote TEXT UNIQUE, payer TEXT,
            service TEXT, reserved INTEGER, status TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS receipts(run TEXT PRIMARY KEY, body TEXT, sha256 TEXT);
        """)

    # ---- helpers ---------------------------------------------------------

    def _tx(self):
        self.db.execute("BEGIN IMMEDIATE")

    def _entry(self, account, delta, run, kind, memo=""):
        self.db.execute("INSERT INTO entries(ts,account,delta,run,kind,memo) VALUES(?,?,?,?,?,?)",
                        (time.time(), account, delta, run, kind, memo))

    def balance_micro(self, account: str) -> int:
        return self.db.execute("SELECT COALESCE(SUM(delta),0) FROM entries WHERE account=?",
                               (account,)).fetchone()[0]

    def balance(self, account: str) -> Dict[str, Any]:
        return {"account": account, "network": self.network,
                "pwm": pwm(self.balance_micro(account))}

    # ---- test funding ------------------------------------------------------

    def fund(self, account: str, amount, memo="test faucet") -> Dict[str, Any]:
        """Test PWM only. The faucet is the counter-entry, so the books balance."""
        m = micro(amount)
        self._tx()
        try:
            self._entry("faucet:test", -m, None, "fund", memo)
            self._entry(account, m, None, "fund", memo)
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return self.balance(account)

    # ---- quote -> reserve -> settle | release ------------------------------

    def quote(self, payer: str, service: Dict[str, Any], *, cap=None,
              ttl_s: int = 900) -> Dict[str, Any]:
        """A price (fixed services) or a ceiling (metered services) in PWM, with
        the receipt lines it would settle into, before any work is done."""
        cap_m = micro(cap if cap is not None else service["price_pwm"])
        if cap_m <= 0:
            raise ValueError("A quote needs a positive price or cap")
        qid = "q_" + uuid.uuid4().hex[:16]
        preview = split(cap_m, service["beneficiaries"])
        detail = {"quote_id": qid, "payer": payer, "service": service["id"],
                  "pricing": service.get("pricing", "fixed"),
                  "price_pwm": pwm(cap_m), "network": self.network,
                  "agreement": service["agreement"],
                  "lines_at_full_price": [dict(l, pwm=pwm(l["micro_pwm"])) for l in preview],
                  "settles_only": service.get("settles_only", "a delivered, judge-passed run"),
                  "failed_runs": "not charged", "settled_runs": "final, no refund"}
        now = time.time()
        self.db.execute("INSERT INTO quotes VALUES(?,?,?,?,?,?,?,?)",
                        (qid, now, payer, service["id"], cap_m, _canon(service["agreement"]),
                         _canon({"beneficiaries": service["beneficiaries"], "detail": detail}),
                         now + ttl_s))
        detail["expires_at"] = now + ttl_s
        return detail

    def reserve(self, run_id: str, quote_id: str) -> Dict[str, Any]:
        """Hold the quoted amount. A known run id is never re-dispatched."""
        self._tx()
        try:
            prior = self.db.execute("SELECT quote,status,reserved FROM runs WHERE id=?",
                                    (run_id,)).fetchone()
            if prior:
                self.db.execute("ROLLBACK")
                if prior[0] != quote_id:
                    raise Conflict("Run id reused with a different quote")
                return {"run_id": run_id, "status": prior[1], "reserved_pwm": pwm(prior[2]),
                        "replay": True}
            q = self.db.execute("SELECT payer,service,cap,expires FROM quotes WHERE id=?",
                                (quote_id,)).fetchone()
            if q is None:
                raise NotFound("Unknown quote")
            if self.db.execute("SELECT 1 FROM runs WHERE quote=?", (quote_id,)).fetchone():
                raise Conflict("Quote already used by another run")
            payer, service, cap, expires = q
            if time.time() > expires:
                raise Conflict("Quote expired; ask for a new one")
            if self.balance_micro(payer) < cap:
                raise InsufficientPWM("Balance %s PWM is below the quoted %s PWM"
                                      % (pwm(self.balance_micro(payer)), pwm(cap)))
            self._entry(payer, -cap, run_id, "reserve")
            self._entry("hold:" + run_id, cap, run_id, "reserve")
            self.db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
                            (run_id, quote_id, payer, service, cap, "reserved", time.time()))
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        return {"run_id": run_id, "status": "reserved", "reserved_pwm": pwm(cap), "replay": False}

    def _run(self, run_id):
        row = self.db.execute("SELECT quote,payer,service,reserved,status FROM runs WHERE id=?",
                              (run_id,)).fetchone()
        if row is None:
            raise NotFound("Unknown run")
        return row

    def release(self, run_id: str, reason: str, evidence: Optional[Dict] = None) -> Dict[str, Any]:
        """A failed or rejected run: the whole hold returns, 0 PWM charged."""
        self._tx()
        try:
            quote, payer, service, reserved, status = self._run(run_id)
            if status == "released":
                self.db.execute("ROLLBACK")
                return self.receipt(run_id)
            if status == "settled":
                raise Conflict("Run is settled; a settled run is final")
            self._entry("hold:" + run_id, -reserved, run_id, "release")
            self._entry(payer, reserved, run_id, "release", reason)
            self.db.execute("UPDATE runs SET status='released' WHERE id=?", (run_id,))
            body = self._receipt_body(run_id, quote, payer, service, reserved, "released", 0, [],
                                      reason=reason, evidence=evidence)
            self._store_receipt(run_id, body)
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        return self.receipt(run_id)

    def settle(self, run_id: str, *, gross=None, evidence: Optional[Dict] = None,
               usage: Optional[Dict] = None) -> Dict[str, Any]:
        """Settle a delivered run: pay the agreed beneficiaries and the treasury,
        return the unused hold. Final. A replay returns the same receipt; a
        different gross is a conflict."""
        self._tx()
        try:
            quote, payer, service, reserved, status = self._run(run_id)
            g = reserved if gross is None else micro(gross)
            over_cap = 0
            if g > reserved:           # metered: charge min(usage, cap), record the shortfall
                over_cap, g = g - reserved, reserved
            if status == "settled":
                prior = json.loads(self.db.execute("SELECT body FROM receipts WHERE run=?",
                                                   (run_id,)).fetchone()[0])
                self.db.execute("ROLLBACK")
                if gross is not None and prior["gross_micro_pwm"] != g:
                    raise Conflict("Conflicting settlement for the same run")
                return self.receipt(run_id)
            if status != "reserved":
                raise Conflict("Only a reserved run settles (this one is %s)" % status)
            stored = json.loads(self.db.execute("SELECT detail FROM quotes WHERE id=?",
                                                (quote,)).fetchone()[0])
            lines = split(g, stored["beneficiaries"])
            self._entry("hold:" + run_id, -reserved, run_id, "settle")
            for l in lines:
                if l["micro_pwm"]:
                    self._entry(l["account"], l["micro_pwm"], run_id, "settle", l["role"])
            refund = reserved - g
            if refund:
                self._entry(payer, refund, run_id, "refund", "unused reservation")
            self.db.execute("UPDATE runs SET status='settled' WHERE id=?", (run_id,))
            body = self._receipt_body(run_id, quote, payer, service, reserved, "settled", g, lines,
                                      evidence=evidence, usage=usage, over_cap=over_cap)
            self._store_receipt(run_id, body)
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        return self.receipt(run_id)

    # ---- receipts ----------------------------------------------------------

    def _receipt_body(self, run_id, quote, payer, service, reserved, status, gross, lines, *,
                      reason=None, evidence=None, usage=None, over_cap=0):
        agreement = json.loads(self.db.execute("SELECT agreement FROM quotes WHERE id=?",
                                               (quote,)).fetchone()[0])
        return {"receipt": "pwm-mode/v1", "network": self.network, "run_id": run_id,
                "quote_id": quote, "payer": payer, "service": service, "agreement": agreement,
                "status": status, "final": status == "settled",
                "reserved_pwm": pwm(reserved), "charged_pwm": pwm(gross),
                "refunded_pwm": pwm(reserved - gross), "over_cap_pwm": pwm(over_cap),
                "gross_micro_pwm": gross,
                "lines": [dict(l, pwm=pwm(l["micro_pwm"])) for l in lines],
                "reason": reason, "usage": usage, "evidence": evidence,
                "settled_at" if status == "settled" else "closed_at": time.time()}

    def _store_receipt(self, run_id, body):
        text = _canon(body)
        self.db.execute("INSERT OR REPLACE INTO receipts VALUES(?,?,?)",
                        (run_id, text, hashlib.sha256(text.encode()).hexdigest()))

    def receipt(self, run_id: str) -> Dict[str, Any]:
        row = self.db.execute("SELECT body,sha256 FROM receipts WHERE run=?", (run_id,)).fetchone()
        if row is None:
            quote, payer, service, reserved, status = self._run(run_id)
            return {"run_id": run_id, "status": status, "reserved_pwm": pwm(reserved),
                    "final": False, "note": "open run: not settled or released yet"}
        body = json.loads(row[0])
        body["sha256"] = row[1]
        return body

    def open_runs(self) -> List[Dict[str, Any]]:
        """Reserved runs with no outcome: reconcile these, never re-dispatch them."""
        return [{"run_id": r[0], "service": r[1], "reserved_pwm": pwm(r[2])}
                for r in self.db.execute("SELECT id,service,reserved FROM runs "
                                         "WHERE status='reserved' ORDER BY ts")]

    def books_balance(self) -> int:
        """Sum of all entries; zero when the books balance."""
        return self.db.execute("SELECT COALESCE(SUM(delta),0) FROM entries").fetchone()[0]
