"""pwm_core.cards.utils

Utility helpers for Card construction.
"""

import uuid
import datetime


def new_card_id() -> str:
    """Return a new random UUID string."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with trailing Z."""
    return datetime.datetime.utcnow().isoformat() + "Z"


__all__ = ["new_card_id", "now_iso"]
