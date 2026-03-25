"""pwm_core.cards — Docking artifact schemas.

Cards are the entry points to the Dyson Swarm.  Every submission begins as a
Card that is compiled into a RunBundle and evaluated by the Judge.

Available cards
---------------
- SpecCard       — describes the forward model and measurement process
- MethodCard     — describes the algorithm / solver
- DatasetCard    — describes the dataset used for evaluation
- ClaimCard      — links an arXiv claim to a benchmark result
"""

from pwm_core.cards.spec_card import SpecCard
from pwm_core.cards.method_card import MethodCard
from pwm_core.cards.dataset_card import DatasetCard
from pwm_core.cards.claim_card import ClaimCard

__all__ = ["SpecCard", "MethodCard", "DatasetCard", "ClaimCard"]
