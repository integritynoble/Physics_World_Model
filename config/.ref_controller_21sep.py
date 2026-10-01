"""Offline reference controller. No credentials, network calls, or real money."""
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import math
import sqlite3


def amount(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError('Amount must be finite and nonnegative')
    return int(number * 1_000_000)


@dataclass(frozen=True)
class Selection:
    llm: str = 'own'
    main_agent: str = 'bundled-free'
    paid_services: tuple = ()

    @property
    def requires_pwm(self):
        return self.llm == 'pwm' or self.main_agent != 'bundled-free' or bool(self.paid_services)

    def authorize(self, *, logged_in=False, cap=None):
        if self.llm not in ('own', 'pwm'):
            raise ValueError('Unknown funding route')
        if not self.requires_pwm:
            return {'funding': 'BYOK', 'pwm': 0, 'login_required': False,
                    'fallback': 'stop; never switch to PWM automatically'}
        if not logged_in or cap is None or amount(cap) <= 0:
            raise PermissionError('PWM login and explicit positive quote cap required')
        return {'funding': 'PWM', 'cap': str(cap), 'login_required': True}


class Controller:
    """Durable local state and shared cap. Demonstration, not a production sandbox."""
    def __init__(self, path, cap=None):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS actions(id TEXT PRIMARY KEY, cost INTEGER, status TEXT);
        CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY, source TEXT, value TEXT, approved INTEGER);
        CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY, event TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS settlements(action TEXT PRIMARY KEY, gross INTEGER,
            provider INTEGER, method INTEGER, refund INTEGER,
            provider_account TEXT, method_account TEXT);
        ''')
        with self.db:
            old = self.db.execute("SELECT value FROM meta WHERE key='cap'").fetchone()
            if old is None:
                if cap is None:
                    raise ValueError('Initial cap required')
                self.db.execute("INSERT INTO meta VALUES('cap',?)", (str(amount(cap)),))
            elif cap is not None and int(old[0]) != amount(cap):
                raise ValueError('Restart cannot change the authorized cap')

    def reserve(self, action, cost):
        cost = amount(cost)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute('SELECT cost,status FROM actions WHERE id=?', (action,)).fetchone()
            if prior:
                if prior[0] != cost:
                    raise ValueError('Action identity reused with changed cost')
                raise RuntimeError('Action already known; reconcile or inspect, never redispatch')
            used = self._used()
            cap = int(self.db.execute("SELECT value FROM meta WHERE key='cap'").fetchone()[0])
            if used + cost > cap:
                raise PermissionError('Parent cap exceeded')
            self.db.execute("INSERT INTO actions VALUES(?,?,'unknown')", (action, cost))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def reconcile(self, action, delivered):
        with self.db:
            row = self.db.execute('SELECT status FROM actions WHERE id=?', (action,)).fetchone()
            if row is None:
                raise ValueError('Unknown action id')
            target = 'delivered' if delivered else 'released'
            if row[0] not in ('unknown', target):
                raise ValueError('Conflicting reconciliation')
            self.db.execute('UPDATE actions SET status=? WHERE id=?', (target, action))

    def _used(self):
        """Reserved cost counts against the cap until a settlement replaces it
        with what was actually charged; the difference is the refund."""
        return self.db.execute('''SELECT COALESCE(SUM(COALESCE(s.gross, a.cost)),0)
                                  FROM actions a LEFT JOIN settlements s ON s.action=a.id
                                  WHERE a.status!='released' ''').fetchone()[0]

    def settle(self, action, *, gross, provider_account, method_account='spiritai@platformai.org'):
        """Direct provider-to-user settlement (owner brief, 2026-09-21).

        The provider submits evidence; the platform computes and settles. Only a
        reconciled delivery settles, never a dispatch. Gross cannot exceed the
        reservation, the unused remainder returns to the user's cap, the split is
        99% provider / 1% method account, and the same action settles once: a
        replay returns the existing record, a different gross is a conflict.
        """
        gross = amount(gross)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            prior = self.db.execute('SELECT gross,provider,method,refund,provider_account,method_account '
                                    'FROM settlements WHERE action=?', (action,)).fetchone()
            if prior:
                if prior[0] != gross:
                    raise ValueError('Conflicting settlement for the same action')
                self.db.commit()
                return self._settlement_row(action, prior)
            row = self.db.execute('SELECT cost,status FROM actions WHERE id=?', (action,)).fetchone()
            if row is None:
                raise ValueError('Unknown action id')
            if row[1] != 'delivered':
                raise ValueError('Settle only a reconciled delivery, never a dispatch or a released action')
            if gross > row[0]:
                raise PermissionError('Gross exceeds the reservation the user authorized')
            split = ProviderSettlement(gross, provider_account, method_account).split()
            values = (gross, split['provider_micro_pwm'], split['method_micro_pwm'],
                      row[0] - gross, provider_account, method_account)
            self.db.execute('INSERT INTO settlements VALUES(?,?,?,?,?,?,?)', (action,) + values)
            self.db.commit()
            return self._settlement_row(action, values)
        except Exception:
            self.db.rollback()
            raise

    @staticmethod
    def _settlement_row(action, values):
        keys = ('gross_micro_pwm', 'provider_micro_pwm', 'method_micro_pwm',
                'refund_micro_pwm', 'provider_account', 'method_account')
        return {'action': action, **dict(zip(keys, values))}

    def remember(self, source, value, approved):
        with self.db:
            self.db.execute('INSERT INTO state(source,value,approved) VALUES(?,?,?)',
                            (source, value, int(approved)))

    def current(self):
        row = self.db.execute('SELECT source,value FROM state WHERE approved=1 ORDER BY id DESC LIMIT 1').fetchone()
        return row

    def snapshot(self):
        return {'actions': self.db.execute('SELECT * FROM actions ORDER BY id').fetchall(),
                'settlements': self.db.execute('SELECT * FROM settlements ORDER BY action').fetchall(),
                'used_micro_pwm': self._used(),
                'current': self.current(), 'cap_micro_pwm': int(self.db.execute("SELECT value FROM meta WHERE key='cap'").fetchone()[0])}


class Exchange:
    """Capacity lifecycle, intentionally separate from a research agent and wallet."""
    def __init__(self):
        self.state, self.cap, self.used, self.receipts = 'offline', 0, 0, {}

    def start(self, cap, *, authenticated, healthy):
        if not authenticated or not healthy:
            raise PermissionError('Authentication and provider health required')
        if self.state in ('online', 'draining'):
            raise RuntimeError('Node already active')
        self.cap = amount(cap)
        if self.cap <= 0:
            raise ValueError('Positive resource cap required')
        self.used, self.state = 0, 'online'

    def supply(self, receipt_id, cost):
        cost = amount(cost)
        if receipt_id in self.receipts:
            if self.receipts[receipt_id] != cost:
                raise ValueError('Conflicting receipt')
            return False
        if self.state != 'online' or self.used + cost > self.cap:
            raise PermissionError('Node unavailable or resource cap exceeded')
        self.receipts[receipt_id] = cost
        self.used += cost
        if self.used == self.cap:
            self.state = 'draining'
        return True

    def stop(self):
        self.state = 'offline'


@dataclass(frozen=True)
class ProviderSettlement:
    """Opt-in provider earnings, separate from ordinary free BYOK use."""
    gross_micro_pwm: int
    provider_account: str
    method_account: str = 'spiritai@platformai.org'
    provider_share_bps: int = 9900
    method_share_bps: int = 100

    def __post_init__(self):
        if self.gross_micro_pwm < 0 or self.provider_share_bps + self.method_share_bps != 10000:
            raise ValueError('Invalid settlement amount or split')

    def split(self):
        provider = self.gross_micro_pwm * self.provider_share_bps // 10000
        return {'provider_micro_pwm': provider, 'method_micro_pwm': self.gross_micro_pwm - provider,
                'provider_account': self.provider_account, 'method_account': self.method_account}


def provider_rate(*, expected_cost_pwm, quality, availability, latency, floor_pwm=0.2):
    """Illustrative quote based on verified delivery, not a promised rate."""
    if expected_cost_pwm < 0 or not 0 <= quality <= 1 or not 0 <= availability <= 1 or latency < 0:
        raise ValueError('Invalid provider measurement')
    multiplier = (0.5 + 0.5 * quality) * (0.75 + 0.25 * availability) / (1 + min(latency, 10) / 100)
    return max(float(floor_pwm), round(expected_cost_pwm * multiplier, 6))

@dataclass(frozen=True)
class Provider:
    id: str
    model: str
    cost_micro_pwm: int
    destination: str
    healthy: bool = True
    capacity: bool = True


def assign_provider(providers, *, model, authorized_destinations, remaining_micro_pwm, excluded=()):
    """System route selection; user selects an LLM, not a provider node.

    Used only before dispatch or after a known non-delivery. Unknown actions
    remain reserved and must be reconciled by the controller before failover.
    """
    eligible = [p for p in providers if p.model == model and p.healthy and p.capacity
                and p.destination in authorized_destinations and p.id not in excluded
                and 0 <= p.cost_micro_pwm <= remaining_micro_pwm]
    if not eligible:
        raise RuntimeError('No compatible healthy provider within the approved envelope')
    return min(eligible, key=lambda p: (p.cost_micro_pwm, p.id))
