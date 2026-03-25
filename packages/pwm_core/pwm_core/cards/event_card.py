"""pwm_core.cards.event_card

EventCard — records a discrete physics or experimental event.

An EventCard captures a named occurrence in an experiment timeline:
acquisition start, protocol change, calibration run, anomaly detection,
or publication milestone.  EventCards form an auditable provenance chain
when linked to RunBundles, SpecCards, and ClaimCards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pwm_core.cards.utils import new_card_id, now_iso


class EventCard(BaseModel):
    """Records a discrete event in the physics experiment lifecycle.

    Fields
    ------
    event_id:
        Unique identifier, e.g. "event_acquisition_2026_03_25_001".
    card_version:
        Schema version (semver string).
    event_type:
        Category: "acquisition" | "calibration" | "protocol_change" |
        "anomaly" | "publication" | "certification" | "retraction" | "other".
    title:
        Short human-readable event title.
    description:
        Free-text description of what happened.
    occurred_at:
        ISO-8601 timestamp when the event occurred.
    severity:
        "info" | "warning" | "critical" — for anomaly / retraction events.
    linked_run_ids:
        RunBundle IDs associated with this event.
    linked_spec_ids:
        SpecCard IDs associated with this event.
    linked_claim_ids:
        ClaimCard IDs associated with this event.
    linked_instrument_ids:
        InstrumentCard IDs associated with this event.
    actor:
        Username or system identifier that triggered the event.
    metadata:
        Arbitrary key-value metadata (instrument readings, protocol params, etc.).
    extra:
        Catch-all for non-standard fields.
    created_at:
        ISO-8601 creation timestamp.
    """

    event_id: str = Field(default_factory=lambda: f"event_{new_card_id()}")
    card_version: str = "1.0.0"
    event_type: str = "other"
    title: str = ""
    description: Optional[str] = None
    occurred_at: str = Field(default_factory=now_iso)
    severity: str = "info"
    linked_run_ids: List[str] = Field(default_factory=list)
    linked_spec_ids: List[str] = Field(default_factory=list)
    linked_claim_ids: List[str] = Field(default_factory=list)
    linked_instrument_ids: List[str] = Field(default_factory=list)
    actor: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
