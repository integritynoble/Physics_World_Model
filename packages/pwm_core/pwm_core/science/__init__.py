"""pwm_core.science -- autonomous science loop, hypothesis generation, and transfer engine."""

from pwm_core.science.autonomous_loop import AutonomousScienceLoop, LoopState, run_loop
from pwm_core.science.hypothesis_engine import (
    Hypothesis,
    HypothesisEngine,
    HypothesisPriority,
    HypothesisReport,
    HypothesisType,
)
from pwm_core.science.transfer_engine import (
    CrossModalityTransferEngine,
    TransferSuggestion,
)

__all__ = [
    # autonomous_loop
    "AutonomousScienceLoop",
    "LoopState",
    "run_loop",
    # hypothesis_engine
    "Hypothesis",
    "HypothesisEngine",
    "HypothesisPriority",
    "HypothesisReport",
    "HypothesisType",
    # transfer_engine
    "CrossModalityTransferEngine",
    "TransferSuggestion",
]
