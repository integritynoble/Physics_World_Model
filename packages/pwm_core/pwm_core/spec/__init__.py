"""pwm_core.spec — CoreSpec compatibility layer.

Imports CoreSpec (alias of ExperimentSpec) and the full DomainProfile /
ProblemInstance implementations for the three-layer schema described in
docs/dyson_swarm_strategy.md §1.
"""

from pwm_core.spec.core import CoreSpec
from pwm_core.spec.domain_profile import (
    DomainProfile,
    DomainGateSpec,
    GateThreshold,
    get_domain_profile,
    list_domain_profiles,
    IMAGING_V1,
    CT_QC_V1,
)
from pwm_core.spec.problem_instance import ProblemInstance

__all__ = [
    "CoreSpec",
    "DomainProfile",
    "DomainGateSpec",
    "GateThreshold",
    "ProblemInstance",
    "get_domain_profile",
    "list_domain_profiles",
    "IMAGING_V1",
    "CT_QC_V1",
]
