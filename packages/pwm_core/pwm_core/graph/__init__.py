"""pwm_core.graph -- OperatorGraph IR module.

Provides the universal intermediate representation for imaging forward models.
Every modality compiles to a GraphOperator with forward(), adjoint(), serialize(),
check_adjoint(), and explain().

Modules
-------
primitives       PrimitiveOp protocol + ~30 primitive implementations
graph_spec       OperatorGraphSpec Pydantic models (DAG of primitives)
compiler         GraphCompiler: validate -> bind -> plan -> export
graph_operator   GraphOperator: forward/adjoint over compiled graph
introspection    Deterministic graph explanation without LLM
"""

from pwm_core.graph.primitives import (
    PrimitiveOp,
    PRIMITIVE_REGISTRY,
    get_primitive,
)
from pwm_core.graph.graph_spec import (
    GraphEdge,
    GraphNode,
    NoiseSpec,
    OperatorGraphSpec,
)
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_operator import GraphOperator
from pwm_core.graph.introspection import explain_graph

__all__ = [
    "PrimitiveOp",
    "PRIMITIVE_REGISTRY",
    "get_primitive",
    "GraphEdge",
    "GraphNode",
    "NoiseSpec",
    "OperatorGraphSpec",
    "GraphCompiler",
    "GraphOperator",
    "explain_graph",
]
