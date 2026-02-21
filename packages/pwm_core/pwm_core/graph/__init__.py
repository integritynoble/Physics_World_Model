"""pwm_core.graph -- OperatorGraph IR module.

Provides the universal intermediate representation for imaging forward models.
Every modality compiles to a GraphOperator with forward(), adjoint(), serialize(),
check_adjoint(), and explain().

Modules
-------
primitives                PrimitiveOp protocol + ~40 primitive implementations
graph_spec                OperatorGraphSpec Pydantic models (DAG of primitives)
ir_types                  NodeTags, TensorSpec, ParameterSpec, NodeRole, PhysicsTier,
                          CanonicalPrimitive, PhysicsStageFamily, DetectFamily
compiler                  GraphCompiler: validate -> bind -> plan -> export
graph_operator            GraphOperator: forward/adjoint over compiled graph
canonical                 Canonical chain validator (Source->Elems->Sensor->Noise)
canonical_decompositions  31-modality canonical DAG decomposition registry (FPB Table 1)
extension_protocol        5-step extension protocol for new canonical primitives
fidelity                  Tier-2 fidelity bound computation
executor                  GraphExecutor: unified Mode S/I/C execution
introspection             Deterministic graph explanation without LLM
"""

from pwm_core.graph.primitives import (
    PrimitiveOp,
    PRIMITIVE_REGISTRY,
    CANONICAL_REGISTRY,
    get_primitive,
)
from pwm_core.graph.ir_types import (
    CanonicalPrimitive,
    PhysicsStageFamily,
    DetectFamily,
    NodeRole,
    NodeTags,
    PhysicsTier,
    TensorSpec,
    ParameterSpec,
)
from pwm_core.graph.canonical_decompositions import (
    CanonicalDecomposition,
    CANONICAL_DECOMPOSITIONS,
)
from pwm_core.graph.graph_spec import (
    GraphEdge,
    GraphNode,
    NoiseSpec,
    OperatorGraphSpec,
)
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_operator import GraphOperator
from pwm_core.graph.adapter import GraphOperatorAdapter
from pwm_core.graph.canonical import validate_canonical_chain
from pwm_core.graph.executor import GraphExecutor, ExecutionConfig, ExecutionResult
from pwm_core.graph.introspection import explain_graph
from pwm_core.graph.tier_policy import TierPolicy, TierBudget

__all__ = [
    "PrimitiveOp",
    "PRIMITIVE_REGISTRY",
    "CANONICAL_REGISTRY",
    "get_primitive",
    "CanonicalPrimitive",
    "PhysicsStageFamily",
    "DetectFamily",
    "NodeRole",
    "NodeTags",
    "PhysicsTier",
    "TensorSpec",
    "ParameterSpec",
    "CanonicalDecomposition",
    "CANONICAL_DECOMPOSITIONS",
    "GraphEdge",
    "GraphNode",
    "NoiseSpec",
    "OperatorGraphSpec",
    "GraphCompiler",
    "GraphOperator",
    "GraphOperatorAdapter",
    "validate_canonical_chain",
    "GraphExecutor",
    "ExecutionConfig",
    "ExecutionResult",
    "explain_graph",
    "TierPolicy",
    "TierBudget",
]
