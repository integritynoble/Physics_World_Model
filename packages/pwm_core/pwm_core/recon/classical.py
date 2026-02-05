"""pwm_core.recon.classical

Classical reconstruction baselines.
This starter includes a simple FISTA-TV placeholder and a generic least squares fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


def soft_thresh(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def fista_l2(y: np.ndarray, A: np.ndarray, lam: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Simple FISTA for min_x 0.5||Ax - y||^2 + lam||x||_1 (toy)."""
    # NOTE: For real imaging, prefer TV prox and operator forms; this is a safe placeholder.
    m, n = A.shape
    x = np.zeros((n,), dtype=np.float32)
    z = x.copy()
    t = 1.0
    At = A.T
    L = np.linalg.norm(A, 2) ** 2 + 1e-8  # spectral norm squared
    step = 1.0 / L
    for _ in range(iters):
        grad = At @ (A @ z - y)
        x_new = soft_thresh(z - step * grad, lam * step)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def least_squares(A: np.ndarray, y: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    AtA = A.T @ A + reg * np.eye(A.shape[1], dtype=np.float32)
    Aty = A.T @ y
    return np.linalg.solve(AtA, Aty)


def gradient_descent_operator(
    y: np.ndarray,
    forward: Any,
    adjoint: Any,
    x_shape: Tuple[int, ...],
    iters: int = 100,
    step: float = 0.01,
    reg: float = 1e-4,
) -> np.ndarray:
    """Gradient descent using forward/adjoint operators.

    Solves: min_x 0.5||A(x) - y||^2 + 0.5*reg*||x||^2

    Args:
        y: Measurements.
        forward: Forward operator callable.
        adjoint: Adjoint operator callable.
        x_shape: Shape of the signal to reconstruct.
        iters: Number of iterations.
        step: Step size (learning rate).
        reg: Regularization parameter.

    Returns:
        Reconstructed signal x.
    """
    # Initialize with adjoint of y
    x = adjoint(y).reshape(x_shape).astype(np.float32)

    for _ in range(iters):
        # Gradient: A^T(Ax - y) + reg*x
        residual = forward(x) - y
        grad = adjoint(residual).reshape(x_shape) + reg * x
        x = x - step * grad

    return x


def conjugate_gradient_operator(
    y: np.ndarray,
    forward: Any,
    adjoint: Any,
    x_shape: Tuple[int, ...],
    iters: int = 50,
    reg: float = 1e-4,
) -> np.ndarray:
    """Conjugate gradient using forward/adjoint operators.

    Solves: (A^T A + reg*I) x = A^T y

    Args:
        y: Measurements.
        forward: Forward operator callable.
        adjoint: Adjoint operator callable.
        x_shape: Shape of the signal to reconstruct.
        iters: Number of iterations.
        reg: Regularization parameter.

    Returns:
        Reconstructed signal x.
    """
    # Right-hand side: A^T y
    b = adjoint(y).reshape(x_shape).astype(np.float32)

    # Initialize
    x = np.zeros(x_shape, dtype=np.float32)
    r = b - (adjoint(forward(x)).reshape(x_shape) + reg * x)
    p = r.copy()
    rsold = float(np.sum(r * r))

    for _ in range(iters):
        if rsold < 1e-12:
            break

        # A^T A p + reg * p
        Ap = adjoint(forward(p)).reshape(x_shape) + reg * p
        pAp = float(np.sum(p * Ap))
        if pAp < 1e-12:
            break

        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = float(np.sum(r * r))
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x
