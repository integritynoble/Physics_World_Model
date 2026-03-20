"""pwm_core.targeting.budget
==============================

Compute budget enforcement per ``rails/gear03_compute_escrow.md``.

- Tracks wall-clock time during solver execution
- Raises BudgetExceeded if > 2x declared budget (disqualification)
- Reports actual consumption for RoIC calculation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Raised when solver exceeds 2x declared compute budget."""

    def __init__(self, declared_s: float, actual_s: float):
        self.declared_s = declared_s
        self.actual_s = actual_s
        ratio = actual_s / declared_s if declared_s > 0 else float("inf")
        super().__init__(
            f"Budget exceeded: {actual_s:.1f}s actual vs {declared_s:.1f}s "
            f"declared ({ratio:.1f}x). Submission disqualified."
        )


@dataclass
class BudgetReport:
    """Summary of compute consumption."""

    declared_s: float
    actual_s: float
    fraction_of_declared: float
    exceeded: bool
    dishonest: bool

    def to_dict(self):
        return {
            "declared_s": self.declared_s,
            "actual_s": self.actual_s,
            "fraction_of_declared": self.fraction_of_declared,
            "exceeded": self.exceeded,
            "dishonest": self.dishonest,
        }


class BudgetGuard:
    """Context manager enforcing wall-clock compute limits.

    Parameters
    ----------
    budget_s : float
        Declared compute budget in seconds.
    enforce : bool
        If True, raise BudgetExceeded when > 2x budget.
        If False (sandbox mode), only track and report.
    """

    def __init__(self, budget_s: float = 600.0, enforce: bool = True):
        self.budget_s = budget_s
        self.enforce = enforce
        self._total_elapsed: float = 0.0
        self._start: float = 0.0
        self._active = False

    def __enter__(self):
        self._start = time.perf_counter()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        self._total_elapsed += elapsed
        self._active = False

        if self.enforce and self.budget_s > 0:
            if self._total_elapsed > 2.0 * self.budget_s:
                raise BudgetExceeded(self.budget_s, self._total_elapsed)
        return False

    @property
    def elapsed_s(self) -> float:
        """Total elapsed time across all context-manager uses."""
        if self._active:
            return self._total_elapsed + (time.perf_counter() - self._start)
        return self._total_elapsed

    def report(self) -> BudgetReport:
        """Generate a budget consumption report."""
        fraction = (
            self._total_elapsed / self.budget_s if self.budget_s > 0 else 0.0
        )
        return BudgetReport(
            declared_s=self.budget_s,
            actual_s=self._total_elapsed,
            fraction_of_declared=fraction,
            exceeded=fraction > 2.0,
            dishonest=self.budget_s > 0 and self.budget_s < 0.5 * self._total_elapsed,
        )
