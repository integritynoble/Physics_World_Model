"""pwm_core.cards — Docking artifact schemas.

Cards are the entry points to the Dyson Swarm.  Every submission begins as a
Card that is compiled into a RunBundle and evaluated by the Judge.

Available cards
---------------
- SpecCard       — describes the forward model and measurement process
- MethodCard     — describes the algorithm / solver
- DatasetCard    — describes the dataset used for evaluation
- ClaimCard      — links an arXiv claim to a benchmark result
- InstrumentCard — describes a physical instrument (scanner, sensor, imager)
- EventCard      — records a discrete physics or experimental event
"""

from pwm_core.cards.spec_card import SpecCard
from pwm_core.cards.method_card import MethodCard
from pwm_core.cards.dataset_card import DatasetCard
from pwm_core.cards.claim_card import ClaimCard
from pwm_core.cards.instrument_card import InstrumentCard
from pwm_core.cards.event_card import EventCard

__all__ = ["SpecCard", "MethodCard", "DatasetCard", "ClaimCard", "InstrumentCard", "EventCard"]
