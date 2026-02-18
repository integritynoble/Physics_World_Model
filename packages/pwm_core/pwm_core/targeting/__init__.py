"""pwm_core.targeting
=====================

The LIP-Arena targeting system: one modality-agnostic harness for all of
computational imaging.

Exports
-------
Harness          The evaluation engine (4-scenario protocol)
ScoredResult     Aggregate scores for a harness run
BudgetGuard      Compute budget enforcement
ScenarioResult   Per-scenario metrics
"""

__version__ = "1.0.0"

from pwm_core.targeting.budget import BudgetGuard
from pwm_core.targeting.harness import Harness, HarnessResult
from pwm_core.targeting.scenarios import ScenarioResult
from pwm_core.targeting.scoring import ScoredResult
