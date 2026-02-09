"""pwm_core.physics.base

Core PhysicsOperator abstraction used across PWM.

Supports:
- Parametric operators: A(theta) with explicit forward/adjoint
- Matrix-backed operators (dense/sparse/LinearOperator)
- Callable-backed operators (user provides forward/adjoint)

v3 additions:
- x_shape / y_shape properties
- is_linear / supports_autodiff properties
- serialize() with blob support (SHA256 hashes for large arrays)
- deserialize() class method
- check_adjoint() built-in self-test
- metadata() returning OperatorMetadata
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AdjointCheckReport (plain dataclass, no pydantic dependency here)
# ---------------------------------------------------------------------------

@dataclass
class AdjointCheckReport:
    """Result of check_adjoint() self-test."""
    passed: bool
    n_trials: int
    max_relative_error: float
    mean_relative_error: float
    tolerance: float
    details: List[Dict[str, float]]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"AdjointCheck [{status}]: max_rel_err={self.max_relative_error:.2e}, "
            f"tol={self.tolerance:.2e}, trials={self.n_trials}"
        )


# ---------------------------------------------------------------------------
# OperatorMetadata (plain dataclass)
# ---------------------------------------------------------------------------

@dataclass
class OperatorMetadata:
    """Rich metadata for any operator."""
    modality: str
    operator_id: str
    x_shape: List[int]
    y_shape: List[int]
    is_linear: bool
    supports_autodiff: bool
    axes: Dict[str, str] = field(default_factory=dict)
    wavelength_nm: Optional[float] = None
    wavelength_range_nm: Optional[List[float]] = None
    units: Dict[str, str] = field(default_factory=dict)
    sampling_info: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Protocol (structural typing)
# ---------------------------------------------------------------------------

class PhysicsOperator(Protocol):
    """Unified operator interface for all 26 modalities.

    v3: Extended with shape properties, serialize/deserialize, check_adjoint,
    and metadata.
    """

    # --- Core (existing, unchanged) ---
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def adjoint(self, y: np.ndarray) -> np.ndarray: ...
    def set_theta(self, theta: Dict[str, Any]) -> None: ...
    def get_theta(self) -> Dict[str, Any]: ...
    def info(self) -> Dict[str, Any]: ...

    # --- New required properties ---
    @property
    def x_shape(self) -> Tuple[int, ...]: ...

    @property
    def y_shape(self) -> Tuple[int, ...]: ...

    @property
    def is_linear(self) -> bool: ...

    @property
    def supports_autodiff(self) -> bool: ...

    # --- New required methods ---
    def serialize(self, data_dir: Optional[str] = None) -> Dict[str, Any]: ...

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    data_dir: Optional[str] = None) -> "PhysicsOperator": ...

    def metadata(self) -> OperatorMetadata: ...

    def check_adjoint(self, n_trials: int = 3, tol: float = 1e-5,
                      seed: int = 0) -> AdjointCheckReport: ...


# ---------------------------------------------------------------------------
# BaseOperator (convenience base class)
# ---------------------------------------------------------------------------

@dataclass
class BaseOperator:
    """Enhanced convenience base class for operators.

    Provides default implementations for serialize, check_adjoint, metadata.
    Subclasses must implement forward() and adjoint().
    """
    operator_id: str
    theta: Dict[str, Any]
    _x_shape: Tuple[int, ...] = (1,)
    _y_shape: Tuple[int, ...] = (1,)
    _is_linear: bool = True
    _supports_autodiff: bool = False

    # --- Core ---

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement forward()")

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement adjoint()")

    def set_theta(self, theta: Dict[str, Any]) -> None:
        self.theta = dict(theta)

    def get_theta(self) -> Dict[str, Any]:
        return dict(self.theta)

    def info(self) -> Dict[str, Any]:
        return {"operator_id": self.operator_id, "theta": self.theta}

    # --- Properties ---

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return self._x_shape

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return self._y_shape

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    @property
    def supports_autodiff(self) -> bool:
        return self._supports_autodiff

    # --- Serialize / Deserialize ---

    def serialize(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """Serialize with blob support for large arrays.

        Large arrays (>1000 elements) are saved as .npy files in data_dir
        with SHA256 hashes stored in the serialized dict.
        """
        result: Dict[str, Any] = {
            "operator_id": self.operator_id,
            "theta": {},
            "x_shape": list(self._x_shape),
            "y_shape": list(self._y_shape),
            "is_linear": self._is_linear,
            "supports_autodiff": self._supports_autodiff,
            "blobs": [],
        }

        for k, v in self.theta.items():
            if isinstance(v, np.ndarray) and v.size > 1000:
                sha = hashlib.sha256(v.tobytes()).hexdigest()
                if data_dir:
                    os.makedirs(data_dir, exist_ok=True)
                    blob_path = os.path.join(data_dir, f"{k}.npy")
                    np.save(blob_path, v)
                    result["blobs"].append({
                        "name": k,
                        "path": blob_path,
                        "sha256": sha,
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                    })
                else:
                    # Fallback: store inline (large but works)
                    result["theta"][k] = v.tolist()
                    result["blobs"].append({
                        "name": k,
                        "path": None,
                        "sha256": sha,
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                    })
            elif isinstance(v, np.ndarray):
                result["theta"][k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                result["theta"][k] = v.item()
            else:
                result["theta"][k] = v

        return result

    @classmethod
    def deserialize(cls, data: Dict[str, Any],
                    data_dir: Optional[str] = None) -> "BaseOperator":
        """Reconstruct operator from serialized data + blob files."""
        theta = dict(data.get("theta", {}))

        # Load blobs
        for blob in data.get("blobs", []):
            name = blob["name"]
            path = blob.get("path")
            if path and os.path.exists(path):
                arr = np.load(path)
                # Verify hash
                sha = hashlib.sha256(arr.tobytes()).hexdigest()
                if sha != blob.get("sha256"):
                    logger.warning(
                        f"SHA256 mismatch for blob '{name}': "
                        f"expected {blob.get('sha256')}, got {sha}"
                    )
                theta[name] = arr
            elif name in theta and isinstance(theta[name], list):
                theta[name] = np.array(theta[name])

        op = cls(
            operator_id=data["operator_id"],
            theta=theta,
            _x_shape=tuple(data.get("x_shape", (1,))),
            _y_shape=tuple(data.get("y_shape", (1,))),
            _is_linear=data.get("is_linear", True),
            _supports_autodiff=data.get("supports_autodiff", False),
        )
        return op

    # --- Check Adjoint ---

    def check_adjoint(self, n_trials: int = 3, tol: float = 1e-5,
                      seed: int = 0) -> AdjointCheckReport:
        """Built-in adjoint consistency test.

        Verifies <Ax, y> == <x, A^T y> for random vectors.
        Every operator gets this for free. Catches bugs early.
        """
        rng = np.random.default_rng(seed)
        details = []
        max_err = 0.0

        for trial in range(n_trials):
            x = rng.standard_normal(self._x_shape).astype(np.float64)
            y = rng.standard_normal(self._y_shape).astype(np.float64)

            Ax = self.forward(x).astype(np.float64)
            ATy = self.adjoint(y).astype(np.float64)

            inner_Ax_y = float(np.sum(Ax.ravel() * y.ravel()))
            inner_x_ATy = float(np.sum(x.ravel() * ATy.ravel()))

            denom = max(abs(inner_Ax_y), abs(inner_x_ATy), 1e-30)
            rel_err = abs(inner_Ax_y - inner_x_ATy) / denom
            max_err = max(max_err, rel_err)

            details.append({
                "trial": trial,
                "inner_Ax_y": inner_Ax_y,
                "inner_x_ATy": inner_x_ATy,
                "rel_err": rel_err,
            })

        mean_err = float(np.mean([d["rel_err"] for d in details]))

        report = AdjointCheckReport(
            passed=max_err < tol,
            n_trials=n_trials,
            max_relative_error=max_err,
            mean_relative_error=mean_err,
            tolerance=tol,
            details=details,
        )

        if not report.passed:
            logger.warning(
                f"Adjoint check FAILED for {self.operator_id}: {report.summary()}"
            )
        else:
            logger.debug(f"Adjoint check passed for {self.operator_id}")

        return report

    # --- Metadata ---

    def metadata(self) -> OperatorMetadata:
        """Return rich metadata for this operator."""
        parts = self.operator_id.split("_")
        modality = parts[0] if parts else self.operator_id

        return OperatorMetadata(
            modality=modality,
            operator_id=self.operator_id,
            x_shape=list(self._x_shape),
            y_shape=list(self._y_shape),
            is_linear=self._is_linear,
            supports_autodiff=self._supports_autodiff,
            axes={},
        )
