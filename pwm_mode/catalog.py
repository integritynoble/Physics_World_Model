"""The PWM-mode catalog (config/services.json): services, models, prices, agreements."""
from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
CATALOG = Path(os.environ.get("PWM_MODE_CATALOG", ROOT / "config" / "services.json"))


def load() -> Dict[str, Any]:
    return json.loads(CATALOG.read_text())


def service(sid: str) -> Dict[str, Any]:
    for s in load()["services"]:
        if s["id"] == sid:
            return s
    raise KeyError("unknown service %r" % sid)


def model(mid: str) -> Dict[str, Any]:
    for m in load()["models"]:
        if m["id"] == mid:
            return m
    raise KeyError(mid)


def model_service(m: Dict[str, Any], cap) -> Dict[str, Any]:
    """A model turn as a ledger service: metered, settled at usage up to the cap."""
    return {"id": "model:" + m["id"], "pricing": "metered", "price_pwm": str(cap),
            "agreement": m["agreement"], "beneficiaries": m["beneficiaries"],
            "settles_only": "a delivered completion, at its metered token usage, never above the cap"}


def model_cost_pwm(m: Dict[str, Any], input_tokens: int, output_tokens: int) -> Decimal:
    return (Decimal(input_tokens) * Decimal(m["pwm_per_million_input"])
            + Decimal(output_tokens) * Decimal(m["pwm_per_million_output"])) / Decimal(1_000_000)


def ledger_path() -> Path:
    return Path(os.environ.get("PWM_MODE_LEDGER",
                               Path.home() / ".ai4science-pwm-mode" / "test-ledger.sqlite"))
