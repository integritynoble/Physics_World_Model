"""Compressed Sensing Solvers: TVAL3 and ISTA-Net.

Classical algorithms for Single-Pixel Camera and general CS problems.

References:
- Li, C. et al. (2013). "TVAL3: TV minimization by augmented Lagrangian and ALM"
- Zhang, J. & Ghanem, B. (2018). "ISTA-Net: Interpretable Optimization-Inspired Deep Network"

Benchmark: Set11 dataset (256×256) with Hadamard measurements
Expected PSNR (TVAL3):
- 1%: 18.5 dB, 4%: 23.2 dB, 10%: 27.1 dB, 25%: 32.5 dB, 50%: 38.2 dB
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Soft thresholding (proximal operator for L1 norm)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def tv_norm_2d(x: np.ndarray) -> float:
    """Compute isotropic TV norm of 2D image."""
    dx = np.diff(x, axis=1)
    dy = np.diff(x, axis=0)

    # Pad to same size
    dx_pad = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
    dy_pad = np.pad(dy, ((0, 1), (0, 0)), mode='constant')

    return float(np.sum(np.sqrt(dx_pad**2 + dy_pad**2 + 1e-10)))


def tv_prox_2d(
    x: np.ndarray,
    lam: float,
    iterations: int = 20,
) -> np.ndarray:
    """Proximal operator for isotropic TV (Chambolle's algorithm).

    Args:
        x: Input image
        lam: Regularization parameter
        iterations: Number of iterations

    Returns:
        TV-denoised image
    """
    # Chambolle's dual algorithm
    n, m = x.shape
    p = np.zeros((n, m, 2), dtype=np.float32)

    tau = 0.25  # Step size

    for _ in range(iterations):
        # Compute divergence of p
        div_p = np.zeros_like(x)
        div_p[:, :-1] += p[:, :-1, 0]
        div_p[:, 1:] -= p[:, :-1, 0]
        div_p[:-1, :] += p[:-1, :, 1]
        div_p[1:, :] -= p[:-1, :, 1]

        # Gradient of (x - lam * div(p))
        u = x - lam * div_p
        grad_u = np.zeros((n, m, 2), dtype=np.float32)
        grad_u[:, :-1, 0] = u[:, 1:] - u[:, :-1]
        grad_u[:-1, :, 1] = u[1:, :] - u[:-1, :]

        # Update dual variable
        p_new = p + tau * grad_u

        # Project onto unit ball
        norm = np.sqrt(p_new[:, :, 0]**2 + p_new[:, :, 1]**2 + 1e-10)
        norm = np.maximum(norm, 1)
        p = p_new / norm[:, :, np.newaxis]

    # Final result
    div_p = np.zeros_like(x)
    div_p[:, :-1] += p[:, :-1, 0]
    div_p[:, 1:] -= p[:, :-1, 0]
    div_p[:-1, :] += p[:-1, :, 1]
    div_p[1:, :] -= p[:-1, :, 1]

    return (x - lam * div_p).astype(np.float32)


def tval3(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    mu: float = 256.0,
    beta: float = 32.0,
    tol: float = 1e-4,
    max_iters: int = 200,
    inner_iters: int = 10,
) -> np.ndarray:
    """TVAL3: TV minimization by Augmented Lagrangian.

    Solves: min_x ||Ax - y||_2^2 + mu * TV(x)

    Args:
        y: Measurements
        forward: Forward operator A
        adjoint: Adjoint operator A^T
        x_shape: Shape of x to reconstruct
        mu: TV regularization weight
        beta: Augmented Lagrangian parameter
        tol: Convergence tolerance
        max_iters: Maximum outer iterations
        inner_iters: Inner CG iterations

    Returns:
        Reconstructed image
    """
    # Initialize
    x = adjoint(y).reshape(x_shape).astype(np.float32)

    # Normalize
    x_max = np.abs(x).max()
    if x_max > 1:
        x = x / x_max
        y = y / x_max

    n, m = x_shape[:2] if len(x_shape) >= 2 else (int(np.sqrt(np.prod(x_shape))),) * 2

    # Auxiliary variables for TV
    wx = np.zeros((n, m-1), dtype=np.float32)  # Horizontal differences
    wy = np.zeros((n-1, m), dtype=np.float32)  # Vertical differences

    # Dual variables (Lagrange multipliers)
    lam_x = np.zeros_like(wx)
    lam_y = np.zeros_like(wy)

    for outer in range(max_iters):
        x_old = x.copy()

        # Subproblem 1: Update x (with fixed w)
        # Solve: A^T A x + beta * D^T D x = A^T y + beta * D^T (w - lambda/beta)
        rhs = adjoint(y).reshape(x_shape)

        # Add TV term contribution
        dwx = np.zeros_like(x)
        dwy = np.zeros_like(x)
        dwx[:, :-1] += wx + lam_x / beta
        dwx[:, 1:] -= wx + lam_x / beta
        dwy[:-1, :] += wy + lam_y / beta
        dwy[1:, :] -= wy + lam_y / beta

        rhs = rhs + beta * (dwx + dwy)

        # CG for x-update
        for _ in range(inner_iters):
            # A^T A x
            AtAx = adjoint(forward(x)).reshape(x_shape)

            # D^T D x (Laplacian-like)
            DtDx = np.zeros_like(x)
            # Horizontal
            DtDx[:, :-1] += x[:, :-1] - x[:, 1:]
            DtDx[:, 1:] += x[:, 1:] - x[:, :-1]
            # Vertical
            DtDx[:-1, :] += x[:-1, :] - x[1:, :]
            DtDx[1:, :] += x[1:, :] - x[:-1, :]

            grad = AtAx + beta * DtDx - rhs
            x = x - 0.01 * grad

        # Subproblem 2: Update w (shrinkage on TV)
        # w = prox_{mu/beta * || ||_1} (Dx + lambda/beta)
        dx = x[:, 1:] - x[:, :-1]  # Horizontal gradient
        dy = x[1:, :] - x[:-1, :]  # Vertical gradient

        vx = dx + lam_x / beta
        vy = dy + lam_y / beta

        # Isotropic shrinkage
        norm_v = np.sqrt(
            np.pad(vx, ((0, 0), (0, 1)), mode='constant')**2 +
            np.pad(vy, ((0, 1), (0, 0)), mode='constant')**2 +
            1e-10
        )
        shrink = np.maximum(norm_v - mu / beta, 0) / norm_v

        wx = vx * shrink[:, :-1]
        wy = vy * shrink[:-1, :]

        # Update Lagrange multipliers
        lam_x = lam_x + beta * (dx - wx)
        lam_y = lam_y + beta * (dy - wy)

        # Check convergence
        rel_change = np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10)
        if rel_change < tol:
            break

    # Denormalize
    if x_max > 1:
        x = x * x_max

    return x.astype(np.float32)


def ista(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    lam: float = 0.01,
    step: float = 0.01,
    max_iters: int = 100,
    use_tv: bool = True,
) -> np.ndarray:
    """ISTA: Iterative Shrinkage-Thresholding Algorithm.

    Solves: min_x 0.5 * ||Ax - y||_2^2 + lam * R(x)
    where R is either L1 (sparsity) or TV.

    Args:
        y: Measurements
        forward: Forward operator A
        adjoint: Adjoint operator A^T
        x_shape: Shape of x
        lam: Regularization weight
        step: Step size (should be < 1/L where L is Lipschitz constant)
        max_iters: Maximum iterations
        use_tv: Use TV instead of L1

    Returns:
        Reconstructed image
    """
    # Initialize with adjoint
    x = adjoint(y).reshape(x_shape).astype(np.float32)

    for k in range(max_iters):
        # Gradient step
        residual = forward(x) - y
        grad = adjoint(residual).reshape(x_shape)
        x = x - step * grad

        # Proximal step
        if use_tv:
            x = tv_prox_2d(x, lam * step)
        else:
            x = soft_threshold(x, lam * step)

    return x.astype(np.float32)


def fista(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    lam: float = 0.01,
    step: float = 0.01,
    max_iters: int = 100,
    use_tv: bool = True,
) -> np.ndarray:
    """FISTA: Fast ISTA with momentum.

    Faster convergence than ISTA through Nesterov acceleration.

    Args:
        y: Measurements
        forward: Forward operator A
        adjoint: Adjoint operator A^T
        x_shape: Shape of x
        lam: Regularization weight
        step: Step size
        max_iters: Maximum iterations
        use_tv: Use TV instead of L1

    Returns:
        Reconstructed image
    """
    # Initialize
    x = adjoint(y).reshape(x_shape).astype(np.float32)
    z = x.copy()
    t = 1.0

    for k in range(max_iters):
        # Gradient step on z
        residual = forward(z) - y
        grad = adjoint(residual).reshape(x_shape)
        v = z - step * grad

        # Proximal step
        if use_tv:
            x_new = tv_prox_2d(v, lam * step)
        else:
            x_new = soft_threshold(v, lam * step)

        # Momentum update
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

    return x.astype(np.float32)


def run_tval3(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run TVAL3 reconstruction.

    Args:
        y: CS measurements
        physics: Physics operator with forward/adjoint
        cfg: Configuration with:
            - mu: TV weight (default: 256)
            - beta: ALM parameter (default: 32)
            - iters: Max iterations (default: 200)

    Returns:
        Tuple of (reconstructed, info_dict)
    """
    mu = cfg.get("mu", 256.0)
    beta = cfg.get("beta", 32.0)
    iters = cfg.get("iters", 200)

    info = {
        "solver": "tval3",
        "mu": mu,
        "beta": beta,
        "iters": iters,
    }

    try:
        if not (hasattr(physics, 'forward') and hasattr(physics, 'adjoint')):
            info["error"] = "no_forward_adjoint"
            return y.astype(np.float32), info

        x_shape = y.shape
        if hasattr(physics, 'x_shape'):
            x_shape = tuple(physics.x_shape)
        elif hasattr(physics, 'info'):
            op_info = physics.info()
            if 'x_shape' in op_info:
                x_shape = tuple(op_info['x_shape'])

        result = tval3(
            y, physics.forward, physics.adjoint,
            x_shape, mu, beta, max_iters=iters
        )

        return result, info

    except Exception as e:
        info["error"] = str(e)
        if hasattr(physics, 'adjoint'):
            return physics.adjoint(y).reshape(x_shape).astype(np.float32), info
        return y.astype(np.float32), info


def run_ista(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run ISTA/FISTA reconstruction.

    Args:
        y: CS measurements
        physics: Physics operator with forward/adjoint
        cfg: Configuration with:
            - lam: Regularization weight (default: 0.01)
            - step: Step size (default: 0.01)
            - iters: Max iterations (default: 100)
            - fista: Use FISTA acceleration (default: True)
            - use_tv: Use TV regularization (default: True)

    Returns:
        Tuple of (reconstructed, info_dict)
    """
    lam = cfg.get("lam", 0.01)
    step = cfg.get("step", 0.01)
    iters = cfg.get("iters", 100)
    use_fista = cfg.get("fista", True)
    use_tv = cfg.get("use_tv", True)

    solver_name = "fista" if use_fista else "ista"

    info = {
        "solver": solver_name,
        "lam": lam,
        "step": step,
        "iters": iters,
        "use_tv": use_tv,
    }

    try:
        if not (hasattr(physics, 'forward') and hasattr(physics, 'adjoint')):
            info["error"] = "no_forward_adjoint"
            return y.astype(np.float32), info

        x_shape = y.shape
        if hasattr(physics, 'x_shape'):
            x_shape = tuple(physics.x_shape)
        elif hasattr(physics, 'info'):
            op_info = physics.info()
            if 'x_shape' in op_info:
                x_shape = tuple(op_info['x_shape'])

        if use_fista:
            result = fista(
                y, physics.forward, physics.adjoint,
                x_shape, lam, step, iters, use_tv
            )
        else:
            result = ista(
                y, physics.forward, physics.adjoint,
                x_shape, lam, step, iters, use_tv
            )

        return result, info

    except Exception as e:
        info["error"] = str(e)
        if hasattr(physics, 'adjoint'):
            return physics.adjoint(y).reshape(x_shape).astype(np.float32), info
        return y.astype(np.float32), info
