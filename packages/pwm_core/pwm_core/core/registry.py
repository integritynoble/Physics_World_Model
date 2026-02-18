"""pwm_core.core.registry

Lightweight plugin registry (operators, calibrators, solvers, dataset adapters, param packs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Registry:
    operators: Dict[str, Any] = field(default_factory=dict)
    calibrators: Dict[str, Any] = field(default_factory=dict)
    solvers: Dict[str, Any] = field(default_factory=dict)
    dataset_adapters: Dict[str, Any] = field(default_factory=dict)
    param_packs: Dict[str, Any] = field(default_factory=dict)

    def register_operator(self, operator_id: str, cls: Any) -> None:
        self.operators[operator_id] = cls

    def register_calibrator(self, calibrator_id: str, cls: Any) -> None:
        self.calibrators[calibrator_id] = cls

    def register_solver(self, solver_id: str, cls: Any) -> None:
        self.solvers[solver_id] = cls

    def register_dataset_adapter(self, adapter_id: str, cls: Any) -> None:
        self.dataset_adapters[adapter_id] = cls

    def register_param_pack(self, pack_id: str, meta: Any) -> None:
        self.param_packs[pack_id] = meta


GLOBAL_REGISTRY = Registry()


def get_registry() -> Registry:
    return GLOBAL_REGISTRY
