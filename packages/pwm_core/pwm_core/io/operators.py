"""pwm_core.io.operators

Load/store operator inputs:
- explicit A matrix (dense/sparse) stored as npy/npz
- operator descriptors (JSON)
- callable operators are not serializable; store descriptor + codegen references
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class OperatorDescriptor:
    operator_type: str  # "matrix" | "parametric" | "callable"
    operator_id: str
    theta_init: Dict[str, Any]
    A_path: Optional[str] = None
    meta: Dict[str, Any] = None


def load_operator_descriptor(path: str) -> OperatorDescriptor:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return OperatorDescriptor(
        operator_type=d["operator_type"],
        operator_id=d["operator_id"],
        theta_init=d.get("theta_init", {}),
        A_path=d.get("A_path"),
        meta=d.get("meta", {}) or {},
    )


def save_operator_matrix(A: np.ndarray, out_dir: str, name: str = "A") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{name}.npy")
    np.save(path, A)
    return path


def save_operator_descriptor(desc: OperatorDescriptor, out_dir: str, name: str = "operator") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "operator_type": desc.operator_type,
            "operator_id": desc.operator_id,
            "theta_init": desc.theta_init,
            "A_path": desc.A_path,
            "meta": desc.meta or {},
        }, f, indent=2)
    return path
