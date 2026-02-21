"""pwm_core.graph.fidelity
=========================

Tier-2 fidelity metrics from the Finite Primitive Basis Theorem.

Implements:
- Operator-norm relative error (power iteration estimate)
- Pointwise fidelity (Eq. S1, FPT supplementary)
- Mean fidelity (Eq. S2, FPT supplementary)
- Canonical-level adjoint consistency check
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def operator_norm_fidelity(
    H_true: Callable[[np.ndarray], np.ndarray],
    H_dag: Callable[[np.ndarray], np.ndarray],
    x_shape: tuple,
    n_trials: int = 20,
    seed: int = 42,
) -> float:
    """Operator-norm relative error: ||H - H_dag|| / ||H|| <= eps.

    Estimated via power iteration on random test vectors.

    Parameters
    ----------
    H_true : callable
        Ground-truth forward operator.
    H_dag : callable
        Canonical DAG forward operator.
    x_shape : tuple
        Shape of the input signal.
    n_trials : int
        Number of random test vectors for estimation.
    seed : int
        Random seed.

    Returns
    -------
    float
        Estimated operator-norm relative error.
    """
    rng = np.random.default_rng(seed)
    max_diff_norm = 0.0
    max_H_norm = 0.0

    for _ in range(n_trials):
        x = rng.standard_normal(x_shape).astype(np.float64)
        y_true = np.asarray(H_true(x), dtype=np.float64).ravel()
        y_dag = np.asarray(H_dag(x), dtype=np.float64).ravel()

        diff_norm = float(np.linalg.norm(y_true - y_dag))
        H_norm = float(np.linalg.norm(y_true))

        max_diff_norm = max(max_diff_norm, diff_norm)
        max_H_norm = max(max_H_norm, H_norm)

    if max_H_norm < 1e-30:
        return 0.0 if max_diff_norm < 1e-30 else float("inf")
    return max_diff_norm / max_H_norm


def pointwise_fidelity(
    H_true: Callable[[np.ndarray], np.ndarray],
    H_dag: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    delta: float = 1e-8,
) -> float:
    """Pointwise fidelity (Eq. S1 of FPT supplementary).

    e_tier2 = sup_x ||H(x) - H_dag(x)||_2 / (||H(x)||_2 + delta)

    Parameters
    ----------
    H_true : callable
        Ground-truth forward operator.
    H_dag : callable
        Canonical DAG forward operator.
    X_test : ndarray
        Test signals, shape (N, *x_shape).
    delta : float
        Denominator regularization.

    Returns
    -------
    float
        Supremum of pointwise relative errors.
    """
    max_err = 0.0
    for i in range(X_test.shape[0]):
        x = X_test[i]
        y_true = np.asarray(H_true(x), dtype=np.float64).ravel()
        y_dag = np.asarray(H_dag(x), dtype=np.float64).ravel()

        diff_norm = float(np.linalg.norm(y_true - y_dag))
        denom = float(np.linalg.norm(y_true)) + delta
        max_err = max(max_err, diff_norm / denom)

    return max_err


def mean_fidelity(
    H_true: Callable[[np.ndarray], np.ndarray],
    H_dag: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    delta: float = 1e-8,
) -> float:
    """Mean pointwise fidelity (Eq. S2 of FPT supplementary).

    mean_e = (1/N) sum_x ||H(x) - H_dag(x)||_2 / (||H(x)||_2 + delta)

    Parameters
    ----------
    H_true : callable
        Ground-truth forward operator.
    H_dag : callable
        Canonical DAG forward operator.
    X_test : ndarray
        Test signals, shape (N, *x_shape).
    delta : float
        Denominator regularization.

    Returns
    -------
    float
        Mean of pointwise relative errors.
    """
    N = X_test.shape[0]
    if N == 0:
        return 0.0

    total = 0.0
    for i in range(N):
        x = X_test[i]
        y_true = np.asarray(H_true(x), dtype=np.float64).ravel()
        y_dag = np.asarray(H_dag(x), dtype=np.float64).ravel()

        diff_norm = float(np.linalg.norm(y_true - y_dag))
        denom = float(np.linalg.norm(y_true)) + delta
        total += diff_norm / denom

    return total / N


def canonical_adjoint_check(
    forward: Callable[[np.ndarray], np.ndarray],
    adjoint: Callable[[np.ndarray], np.ndarray],
    x_shape: tuple,
    y_shape: tuple,
    n_trials: int = 5,
    rtol: float = 1e-4,
    seed: int = 0,
) -> tuple:
    """Verify <Ax, y> == <x, A^T y> for the canonical-level composition.

    Parameters
    ----------
    forward : callable
        Composed canonical forward operator.
    adjoint : callable
        Composed canonical adjoint operator.
    x_shape : tuple
        Input shape.
    y_shape : tuple
        Output shape.
    n_trials : int
        Number of random dot-product trials.
    rtol : float
        Relative tolerance.
    seed : int
        Random seed.

    Returns
    -------
    (passed, max_rel_err, details) : tuple
        Whether the check passed, the maximum relative error,
        and per-trial details.
    """
    rng = np.random.default_rng(seed)
    max_err = 0.0
    details = []

    for trial in range(n_trials):
        x = rng.standard_normal(x_shape).astype(np.float64)
        y = rng.standard_normal(y_shape).astype(np.float64)

        Ax = np.asarray(forward(x), dtype=np.float64)
        ATy = np.asarray(adjoint(y), dtype=np.float64)

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

    return (max_err < rtol, max_err, details)
