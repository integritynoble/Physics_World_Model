"""Constrained Primitive Compiler for the System Design Agent pipeline.

Provides formal validation that agent-generated imaging system designs
are valid compositions of the 11 canonical primitives from the Finite
Primitive Basis Theorem, guaranteeing representation error ε < 0.01.

Modules
-------
nonlinear_constraints   Formal parameter constraints for D, R, Λ
agent_translator        Convert agent JSON → OperatorGraphSpec
primitive_compiler      Validate + compile against 11-primitive basis
scenario_validator      4-scenario validation protocol (I/II/III/IV)
"""

from papers.system_design.compiler.nonlinear_constraints import (
    DETECT_CONSTRAINTS,
    TRANSFORM_CONSTRAINTS,
    validate_detect_params,
    validate_transform_params,
)
from papers.system_design.compiler.agent_translator import AgentToGraphTranslator
from papers.system_design.compiler.primitive_compiler import (
    CompilationReport,
    ConstrainedPrimitiveCompiler,
)
from papers.system_design.compiler.scenario_validator import (
    FourScenarioValidator,
    ScenarioResult,
    ValidationReport,
)

__all__ = [
    "DETECT_CONSTRAINTS",
    "TRANSFORM_CONSTRAINTS",
    "validate_detect_params",
    "validate_transform_params",
    "AgentToGraphTranslator",
    "CompilationReport",
    "ConstrainedPrimitiveCompiler",
    "FourScenarioValidator",
    "ScenarioResult",
    "ValidationReport",
]
