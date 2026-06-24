"""pwm_core.forward_compiler

Forward-model compiler: a structured description of an imaging forward model
(an ordered pipeline of primitive ops) compiled into an executable, validated
pwm_core PhysicsOperator.
"""
from __future__ import annotations

from pwm_core.forward_compiler.ir import ForwardModel, Stage
from pwm_core.forward_compiler.primitives import (
    PRIMITIVES, Primitive, get_primitive, register_primitive,
)
from pwm_core.forward_compiler.compiler import (
    CompiledOperator, NativeCompiledOperator, compile_model,
)
from pwm_core.forward_compiler.validate import (
    ForwardModelReport, validate_forward_model, validate_dimensions,
    classify_linearity, probe_conditioning,
)
from pwm_core.forward_compiler.bridge import from_modality, from_spec_fields

__all__ = [
    "ForwardModel", "Stage",
    "PRIMITIVES", "Primitive", "get_primitive", "register_primitive",
    "CompiledOperator", "NativeCompiledOperator", "compile_model",
    "ForwardModelReport", "validate_forward_model", "validate_dimensions",
    "classify_linearity", "probe_conditioning",
    "from_modality", "from_spec_fields",
]


# Register the compiler as a shared-registry operator factory so other pwm_core
# code can build operators from a ForwardModel via the standard registry.
def _register_in_global_registry() -> None:
    try:
        from pwm_core.core.registry import get_registry
    except ImportError:
        # pwm_core.core is absent in vendored/standalone installs (e.g. the
        # AI4Science wheel ships forward_compiler + physics + optics only). The
        # global-registry integration is optional — compile_model / validate and
        # the harness fm_* tools work without it — so skip it gracefully.
        return
    get_registry().register_operator("forward_compiler", compile_model)


_register_in_global_registry()
