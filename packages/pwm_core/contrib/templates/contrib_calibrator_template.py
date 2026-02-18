"""PWM Calibrator Contribution Template
========================================

Copy this file and implement your calibration algorithm.

Protocol (frozen per Rail Constitution Article 1.8):
    calibrate_<method_name>(y, H_nom, budget) -> (H_hat, info)

The ``H_nom`` argument exposes:
    H_nom.get_theta() -> dict    Current parameters
    H_nom.set_theta(theta)       Update parameters
    H_nom.forward(x) -> y        Apply forward model
    H_nom.adjoint(y) -> x        Apply adjoint

Quickstart:
    1. Copy this file to contrib/calibrators/<your_name>/calibrator.py
    2. Implement calibrate_<your_name>()
    3. Run: pwm evaluate --sandbox --modality widefield
    4. Submit PR

Example paper: "Our blind calibrator reduces oracle gap from 12 dB to 2 dB"
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Tuple

import numpy as np


def calibrate_example(
    y: np.ndarray,
    H_nom: Any,
    budget: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Example: grid search calibrator.

    Searches over parameter perturbations to minimize residual.
    Replace with your algorithm.
    """
    H_hat = copy.deepcopy(H_nom)
    theta_orig = H_hat.get_theta()
    best_theta = dict(theta_orig)
    best_residual = float("inf")

    t_start = time.perf_counter()
    n_evals = 0

    # Simple grid search over learnable parameters
    for key, val in theta_orig.items():
        if not isinstance(val, (int, float)):
            continue

        for delta in np.linspace(-1.0, 1.0, 11):
            if time.perf_counter() - t_start > budget * 0.9:
                break

            test_theta = dict(best_theta)
            test_theta[key] = val + delta
            H_hat.set_theta(test_theta)

            try:
                # Use adjoint backprojection as quick reconstruction
                x_approx = H_hat.adjoint(y)
                y_reprojected = H_hat.forward(x_approx)
                residual = float(np.linalg.norm(y_reprojected - y))
            except Exception:
                residual = float("inf")

            n_evals += 1

            if residual < best_residual:
                best_residual = residual
                best_theta = dict(test_theta)

    H_hat.set_theta(best_theta)

    info = {
        "method": "example_grid_search",
        "budget_s": budget,
        "elapsed_s": time.perf_counter() - t_start,
        "n_evaluations": n_evals,
        "best_residual": best_residual,
        "params_found": best_theta,
    }

    return H_hat, info
