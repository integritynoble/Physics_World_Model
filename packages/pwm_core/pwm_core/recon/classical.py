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


def fista_l2(y: np.ndarray, A: np.ndarray, lam: float = 1e-3, iters: int = 50,
             device=None) -> np.ndarray:
    """Simple FISTA for min_x 0.5||Ax - y||^2 + lam||x||_1 (toy)."""
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _fista_l2_torch(y, A, lam, iters, dev)

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


def _fista_l2_torch(y, A, lam, iters, dev):
    """GPU implementation of fista_l2."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy, soft_threshold_torch

    A_t = to_torch(A, dev)
    y_t = to_torch(y, dev)
    At_t = A_t.T
    m, n = A_t.shape

    x = torch.zeros(n, device=dev, dtype=torch.float32)
    z = x.clone()
    t = 1.0

    L = float(torch.linalg.norm(A_t, 2) ** 2) + 1e-8
    step = 1.0 / L

    for _ in range(iters):
        grad = At_t @ (A_t @ z - y_t)
        x_new = soft_threshold_torch(z - step * grad, lam * step)
        t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new

    return to_numpy(x).astype(np.float32)


def least_squares(A: np.ndarray, y: np.ndarray, reg: float = 1e-6,
                  device=None) -> np.ndarray:
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _least_squares_torch(A, y, reg, dev)

    AtA = A.T @ A + reg * np.eye(A.shape[1], dtype=np.float32)
    Aty = A.T @ y
    return np.linalg.solve(AtA, Aty)


def _least_squares_torch(A, y, reg, dev):
    """GPU implementation of least_squares."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy

    A_t = to_torch(A, dev)
    y_t = to_torch(y, dev)
    n = A_t.shape[1]
    AtA = A_t.T @ A_t + reg * torch.eye(n, device=dev, dtype=A_t.dtype)
    Aty = A_t.T @ y_t
    x = torch.linalg.solve(AtA, Aty)
    return to_numpy(x).astype(np.float32)


def run_fista_l2(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Harness-compatible wrapper for FISTA / gradient descent reconstruction.

    Accepts both explicit matrices (with .shape) and GraphOperator objects
    (with .forward/.adjoint).

    Args:
        y: Measurements.
        physics: Explicit matrix or GraphOperator.
        cfg: Configuration with optional 'lam', 'iters', 'reg', 'step'.

    Returns:
        Tuple of (reconstructed signal, info_dict).
    """
    lam = cfg.get("lam", 1e-3)
    iters = cfg.get("iters", 50)
    reg = cfg.get("reg", 1e-4)
    step = cfg.get("step", 0.01)
    device = cfg.get("device", None)
    info: Dict[str, Any] = {"solver": "fista_l2"}

    try:
        if hasattr(physics, 'shape'):
            # Explicit matrix path
            result = fista_l2(y, physics, lam=lam, iters=iters, device=device)
            x_shape = (physics.shape[1],)
        elif hasattr(physics, 'forward') and hasattr(physics, 'adjoint'):
            # Operator path via gradient descent
            x_shape = getattr(physics, 'x_shape', y.shape)
            result = gradient_descent_operator(
                y, physics.forward, physics.adjoint, x_shape,
                iters=iters, step=step, reg=reg, device=device,
            )
            info["method"] = "gradient_descent_operator"
        else:
            info["error"] = "unsupported_physics_type"
            return y.astype(np.float32), info

        result = result.reshape(x_shape).astype(np.float32)
        return result, info
    except Exception as e:
        info["error"] = str(e)
        x_shape = getattr(physics, 'x_shape', y.shape)
        return np.zeros(x_shape, dtype=np.float32), info


def gradient_descent_operator(
    y: np.ndarray,
    forward: Any,
    adjoint: Any,
    x_shape: Tuple[int, ...],
    iters: int = 100,
    step: float = 0.01,
    reg: float = 1e-4,
    device=None,
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
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Reconstructed signal x.
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _gradient_descent_operator_torch(
            y, forward, adjoint, x_shape, iters, step, reg, dev)

    # Initialize with adjoint of y
    x = adjoint(y).reshape(x_shape).astype(np.float32)

    for _ in range(iters):
        # Gradient: A^T(Ax - y) + reg*x
        residual = forward(x) - y
        grad = adjoint(residual).reshape(x_shape) + reg * x
        x = x - step * grad

    return x


def _gradient_descent_operator_torch(y, forward, adjoint, x_shape, iters,
                                     step, reg, dev):
    """GPU implementation of gradient_descent_operator."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy, wrap_operator

    y_t = to_torch(y, dev)
    fwd = wrap_operator(forward, dev)
    adj = wrap_operator(adjoint, dev)

    x = adj(y_t).reshape(x_shape)

    for _ in range(iters):
        residual = fwd(x) - y_t
        grad = adj(residual).reshape(x_shape) + reg * x
        x = x - step * grad

    return to_numpy(x).astype(np.float32)


def conjugate_gradient_operator(
    y: np.ndarray,
    forward: Any,
    adjoint: Any,
    x_shape: Tuple[int, ...],
    iters: int = 50,
    reg: float = 1e-4,
    device=None,
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
        device: Compute device (None=auto, 'cpu', 'cuda', etc.).

    Returns:
        Reconstructed signal x.
    """
    from pwm_core.recon.gpu_utils import resolve_device
    dev, use_gpu = resolve_device(device)

    if use_gpu:
        return _conjugate_gradient_operator_torch(
            y, forward, adjoint, x_shape, iters, reg, dev)

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


def _conjugate_gradient_operator_torch(y, forward, adjoint, x_shape, iters,
                                       reg, dev):
    """GPU implementation of conjugate_gradient_operator."""
    import torch
    from pwm_core.recon.gpu_utils import to_torch, to_numpy, wrap_operator

    y_t = to_torch(y, dev)
    fwd = wrap_operator(forward, dev)
    adj = wrap_operator(adjoint, dev)

    b = adj(y_t).reshape(x_shape)
    x = torch.zeros(x_shape, device=dev, dtype=torch.float32)
    r = b - (adj(fwd(x)).reshape(x_shape) + reg * x)
    p = r.clone()
    rsold = float(torch.sum(r * r))

    for _ in range(iters):
        if rsold < 1e-12:
            break

        Ap = adj(fwd(p)).reshape(x_shape) + reg * p
        pAp = float(torch.sum(p * Ap))
        if pAp < 1e-12:
            break

        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = float(torch.sum(r * r))
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return to_numpy(x).astype(np.float32)
