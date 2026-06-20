"""Forward-model intermediate representation (IR).

A ForwardModel is an ordered list of Stages; each Stage names a primitive op
(see primitives.py) and carries its params. Params may hold numpy arrays in
memory (e.g. a coded-aperture mask); JSON persistence of arrays is handled by
the tool layer, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class Stage:
    op: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stage":
        return cls(op=d["op"], params=dict(d.get("params", {})))


@dataclass
class ForwardModel:
    name: str
    x_shape: Tuple[int, ...]
    stages: List[Stage]
    dtype: str = "float32"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ForwardModel.name must be non-empty")
        if not self.stages:
            raise ValueError("ForwardModel.stages must be non-empty")
        self.x_shape = tuple(int(d) for d in self.x_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "x_shape": list(self.x_shape),
            "stages": [s.to_dict() for s in self.stages],
            "dtype": self.dtype,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ForwardModel":
        return cls(
            name=d["name"],
            x_shape=tuple(d["x_shape"]),
            stages=[Stage.from_dict(s) for s in d["stages"]],
            dtype=d.get("dtype", "float32"),
            metadata=dict(d.get("metadata", {})),
        )
