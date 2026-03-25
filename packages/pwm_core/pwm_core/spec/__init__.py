"""pwm_core.spec — CoreSpec compatibility layer.

Imports CoreSpec (alias of ExperimentSpec) and DomainProfile / ProblemInstance
stubs for the three-layer schema described in docs/dyson_swarm_strategy.md §1.
"""

from pwm_core.spec.core import CoreSpec, DomainProfile, ProblemInstance

__all__ = ["CoreSpec", "DomainProfile", "ProblemInstance"]
