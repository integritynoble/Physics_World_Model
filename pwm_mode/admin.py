"""Test-ledger administration: keys, test funding, balances, receipts, reconciliation.

  python -m pwm_mode.admin new-key user:alice          # prints a key once; stores its hash
  python -m pwm_mode.admin fund user:alice 20          # TEST PWM from the faucet
  python -m pwm_mode.admin balance user:alice treasury contrib:test-ldct-method
  python -m pwm_mode.admin receipt r_...
  python -m pwm_mode.admin open-runs
  python -m pwm_mode.admin reconcile --older-than 900  # release runs with no outcome: 0 PWM
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time

from . import catalog, proxy
from .ledger import Ledger


def main(argv=None):
    p = argparse.ArgumentParser(prog="pwm_mode.admin")
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("new-key"); k.add_argument("payer")
    f = sub.add_parser("fund"); f.add_argument("account"); f.add_argument("amount")
    b = sub.add_parser("balance"); b.add_argument("accounts", nargs="+")
    r = sub.add_parser("receipt"); r.add_argument("run_id")
    sub.add_parser("open-runs")
    c = sub.add_parser("reconcile"); c.add_argument("--older-than", type=int, default=900)
    sub.add_parser("books")
    a = p.parse_args(argv)
    L = Ledger(catalog.ledger_path())

    if a.cmd == "new-key":
        key = "pwmt_" + secrets.token_urlsafe(24)
        keys = json.loads(proxy.KEYS_FILE.read_text()) if proxy.KEYS_FILE.exists() else {}
        keys[hashlib.sha256(key.encode()).hexdigest()] = a.payer
        proxy.KEYS_FILE.write_text(json.dumps(keys, indent=1))
        proxy.KEYS_FILE.chmod(0o600)
        print(key)
    elif a.cmd == "fund":
        print(json.dumps(L.fund(a.account, a.amount)))
    elif a.cmd == "balance":
        print(json.dumps([L.balance(x) for x in a.accounts], indent=1))
    elif a.cmd == "receipt":
        print(json.dumps(L.receipt(a.run_id), indent=1))
    elif a.cmd == "open-runs":
        print(json.dumps(L.open_runs(), indent=1))
    elif a.cmd == "reconcile":
        cutoff = time.time() - a.older_than
        for run_id, ts in L.db.execute("SELECT id,ts FROM runs WHERE status='reserved'").fetchall():
            if ts < cutoff:
                print(json.dumps(L.release(run_id, "reconciled: no delivery evidence"))[:200])
    elif a.cmd == "books":
        print(json.dumps({"sum_of_all_entries_micro_pwm": L.books_balance()}))


if __name__ == "__main__":
    main()
