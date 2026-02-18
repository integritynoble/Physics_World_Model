"""pwm_core.physics.adapters.matrix_operator

Wrap an explicit matrix A as an operator with forward/adjoint.
Supports dense numpy arrays, torch tensors, and scipy sparse matrices.

For large A, store as a reference (RunBundle data manifest) and load lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from pwm_core.physics.base import BaseOperator


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_to_torch(result: np.ndarray, like: Any) -> Any:
    """Convert result back to torch if input was torch."""
    if hasattr(like, "detach"):
        import torch
        return torch.from_numpy(result).to(like.device)
    return result


@dataclass
class MatrixOperator(BaseOperator):
    """Operator backed by an explicit matrix A.

    forward(x) = A @ x
    adjoint(y) = A^T @ y

    Supports dense and sparse matrices. Handles torch tensor inputs
    by converting to/from numpy.
    """

    A: np.ndarray | None = None  # shape (m, n)

    @classmethod
    def from_dense(cls, A: Any) -> "MatrixOperator":
        """Create operator from a dense matrix (numpy or torch).

        Parameters
        ----------
        A : array-like, shape (M, N)
            Dense measurement matrix.

        Returns
        -------
        MatrixOperator
        """
        A_np = _to_numpy(A).astype(np.float64)
        if A_np.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {A_np.shape}")
        M, N = A_np.shape
        return cls(
            operator_id="matrix_dense",
            theta={"M": M, "N": N},
            A=A_np,
            _x_shape=(N,),
            _y_shape=(M,),
        )

    @classmethod
    def from_sparse(cls, A_sparse: Any) -> "MatrixOperator":
        """Create operator from a sparse matrix (scipy or torch sparse).

        Parameters
        ----------
        A_sparse : sparse matrix
            Sparse measurement matrix. Converted to dense internally
            for simplicity. For very large matrices, consider a
            dedicated sparse operator.

        Returns
        -------
        MatrixOperator
        """
        if hasattr(A_sparse, "to_dense"):
            # torch sparse
            A_np = _to_numpy(A_sparse.to_dense()).astype(np.float64)
        elif hasattr(A_sparse, "toarray"):
            # scipy sparse
            A_np = A_sparse.toarray().astype(np.float64)
        else:
            A_np = _to_numpy(A_sparse).astype(np.float64)

        if A_np.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {A_np.shape}")
        M, N = A_np.shape
        return cls(
            operator_id="matrix_sparse",
            theta={"M": M, "N": N},
            A=A_np,
            _x_shape=(N,),
            _y_shape=(M,),
        )

    def forward(self, x: Any) -> Any:
        """Compute y = A @ x."""
        if self.A is None:
            raise ValueError("MatrixOperator.A is None")
        x_np = _to_numpy(x).reshape(-1).astype(np.float64)
        M, N = self.A.shape
        if x_np.shape[0] != N:
            raise ValueError(
                f"Shape mismatch: A is ({M}, {N}) but x has {x_np.shape[0]} elements"
            )
        result = self.A @ x_np
        return _maybe_to_torch(result.astype(np.float32), x)

    def adjoint(self, y: Any) -> Any:
        """Compute x = A^T @ y."""
        if self.A is None:
            raise ValueError("MatrixOperator.A is None")
        y_np = _to_numpy(y).reshape(-1).astype(np.float64)
        M, N = self.A.shape
        if y_np.shape[0] != M:
            raise ValueError(
                f"Shape mismatch: A^T is ({N}, {M}) but y has {y_np.shape[0]} elements"
            )
        result = self.A.T @ y_np
        return _maybe_to_torch(result.astype(np.float32), y)

    def info(self) -> Dict[str, Any]:
        info = super().info()
        if self.A is not None:
            info.update({"shape": list(self.A.shape), "dtype": str(self.A.dtype)})
        return info
