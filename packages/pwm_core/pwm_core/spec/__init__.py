"""pwm_core.spec — CoreSpec compatibility layer.

Imports CoreSpec (alias of ExperimentSpec) and the full DomainProfile /
ProblemInstance implementations for the three-layer schema described in
docs/dyson_swarm_strategy.md §1.
"""

from pwm_core.spec.core import (
    CanonicalCoreSpec,
    CoreSpec,
    EXPERIMENT_SPEC_MAPPING,
    to_canonical,
    validate_canonical,
)
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
from pwm_core.spec.parser import (
    parse_spec_md,
    render_spec_md,
    parse_spec_file,
    render_spec_file,
    roundtrip_spec_md,
)
from pwm_core.spec.profile_registry import (
    PROFILE_REGISTRY,
    get_profile,
    list_profiles,
    get_profile_maturity,
)

__all__ = [
    "CanonicalCoreSpec",
    "CoreSpec",
    "EXPERIMENT_SPEC_MAPPING",
    "to_canonical",
    "validate_canonical",
    "DomainProfile",
    "DomainGateSpec",
    "GateThreshold",
    "ProblemInstance",
    "get_domain_profile",
    "list_domain_profiles",
    "IMAGING_V1",
    "CT_QC_V1",
    "parse_spec_md",
    "render_spec_md",
    "parse_spec_file",
    "render_spec_file",
    "roundtrip_spec_md",
    "PROFILE_REGISTRY",
    "get_profile",
    "list_profiles",
    "get_profile_maturity",
]
