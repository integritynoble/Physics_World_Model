"""pwm_core.forward_compiler

Forward-model compiler: a structured description of an imaging forward model
(an ordered pipeline of primitive ops) compiled into an executable, validated
pwm_core PhysicsOperator.
"""
from __future__ import annotations

from pwm_core.forward_compiler.ir import ForwardModel, Stage

__all__ = ["ForwardModel", "Stage"]
