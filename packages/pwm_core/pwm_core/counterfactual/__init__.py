"""pwm_core.counterfactual
=========================

Counterfactual pack generation for LIP-Arena validation.

Each pack contains **probe** (public, moderate) and **hidden** (secret, harder)
splits covering 7 Red Team injection categories.
"""

from pwm_core.counterfactual.schema import (
    CounterfactualPackManifest,
    ExpectedBaseline,
    MismatchConfig,
    NoiseConfig,
    RedTeamCategory,
    RegimeKind,
    ScenarioSpec,
    SplitKind,
)

__all__ = [
    "CounterfactualPackManifest",
    "ExpectedBaseline",
    "MismatchConfig",
    "NoiseConfig",
    "RedTeamCategory",
    "RegimeKind",
    "ScenarioSpec",
    "SplitKind",
]
