"""pwm_core.targeting.scaffold
================================

``pwm scaffold solver my_solver`` / ``pwm scaffold modality my_modality``

Generates ready-to-fill contribution skeletons with correct signatures,
self-tests, and registry entry templates.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONTRIB_DIR = Path(__file__).resolve().parents[2] / "contrib"


# ---------------------------------------------------------------------------
# Solver scaffold
# ---------------------------------------------------------------------------

_SOLVER_TEMPLATE = '''"""Solver: {name}

Implements the PWM solver protocol:
    run_{name}(y, physics, cfg) -> (x_hat, info)

The `physics` argument satisfies the LinearLikeOperator protocol:
    physics.forward(x) -> y
    physics.adjoint(y) -> x
    physics.x_shape -> tuple
    physics.y_shape -> tuple
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def run_{name}(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reconstruct x from measurements y using the {name} algorithm.

    Parameters
    ----------
    y : np.ndarray
        Measurements (shape = physics.y_shape).
    physics : LinearLikeOperator
        Forward model with .forward(), .adjoint(), .x_shape, .y_shape.
    cfg : dict
        Solver configuration. Suggested keys:
        - iters: int (default 100)
        - step_size: float (default 0.01)

    Returns
    -------
    x_hat : np.ndarray
        Reconstructed signal (shape = physics.x_shape).
    info : dict
        Metadata about the reconstruction.
    """
    iters = cfg.get("iters", 100)
    step_size = cfg.get("step_size", 0.01)

    # Initialize with adjoint (A^T y)
    x_hat = physics.adjoint(y)

    # Gradient descent: x <- x - step * A^T(Ax - y)
    for i in range(iters):
        residual = physics.forward(x_hat) - y
        gradient = physics.adjoint(residual)
        x_hat = x_hat - step_size * gradient

    return x_hat, {{
        "solver": "{name}",
        "iters": iters,
        "step_size": step_size,
    }}
'''

_SOLVER_CONFIG_TEMPLATE = '''# Solver configuration for {name}
name: "{name}"
version: "1.0.0"
family: "classical"  # classical | pnp | unrolled | learned
description: "TODO: describe your solver"

# Default parameters
params:
  iters: 100
  step_size: 0.01

# Registry entry (copy this to contrib/solver_registry.yaml)
registry_entry:
  module: "contrib.solvers.{name}.solver"
  function: "run_{name}"
  params: 0
  gpu: false
'''

_SOLVER_TEST_TEMPLATE = '''"""Self-test for {name} solver."""

import numpy as np


def test_{name}_runs():
    """Verify solver runs on a toy operator."""
    from contrib.solvers.{name}.solver import run_{name}

    class ToyOperator:
        x_shape = (32, 32)
        y_shape = (32, 32)
        all_linear = True

        def forward(self, x):
            return x * 0.5  # simple scaling

        def adjoint(self, y):
            return y * 0.5  # self-adjoint for scaling

    physics = ToyOperator()
    x_true = np.random.randn(*physics.x_shape)
    y = physics.forward(x_true) + np.random.randn(*physics.y_shape) * 0.01

    x_hat, info = run_{name}(y, physics, {{"iters": 10}})

    assert x_hat.shape == physics.x_shape, f"Wrong shape: {{x_hat.shape}}"
    assert "solver" in info, "Missing 'solver' in info dict"
    print(f"PASS: {name} solver self-test")


if __name__ == "__main__":
    test_{name}_runs()
'''


def scaffold_solver(name: str, calibrator: bool = False) -> Path:
    """Generate a solver (or calibrator) contribution skeleton.

    Parameters
    ----------
    name : str
        Solver name (lowercase, underscores).
    calibrator : bool
        If True, generate calibrator template instead.

    Returns
    -------
    Path
        Path to the created directory.
    """
    if calibrator:
        return _scaffold_calibrator(name)

    out_dir = _CONTRIB_DIR / "solvers" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "solver.py").write_text(_SOLVER_TEMPLATE.format(name=name))
    (out_dir / "config.yaml").write_text(_SOLVER_CONFIG_TEMPLATE.format(name=name))
    (out_dir / "test_local.py").write_text(_SOLVER_TEST_TEMPLATE.format(name=name))
    (out_dir / "__init__.py").write_text(f'from .solver import run_{name}\n')

    logger.info(f"Scaffolded solver: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Calibrator scaffold
# ---------------------------------------------------------------------------

_CALIBRATOR_TEMPLATE = '''"""Calibrator: {name}

Implements the PWM calibrator protocol:
    calibrate_{name}(y, H_nom, budget) -> (H_hat, info)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def calibrate_{name}(
    y: np.ndarray,
    H_nom: Any,
    budget: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Calibrate operator H_nom using measurements y.

    Parameters
    ----------
    y : np.ndarray
        Measurements.
    H_nom : operator
        Nominal (mismatched) operator with .get_theta(), .set_theta(),
        .forward(), .adjoint().
    budget : float
        Compute budget in seconds.

    Returns
    -------
    H_hat : operator
        Calibrated operator.
    info : dict
        Calibration metadata.
    """
    import copy
    H_hat = copy.deepcopy(H_nom)

    theta = H_hat.get_theta()
    # TODO: implement your calibration algorithm
    # Example: grid search, gradient descent, etc.

    return H_hat, {{
        "method": "{name}",
        "budget_s": budget,
        "params_found": theta,
    }}
'''


def _scaffold_calibrator(name: str) -> Path:
    out_dir = _CONTRIB_DIR / "calibrators" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "calibrator.py").write_text(_CALIBRATOR_TEMPLATE.format(name=name))
    (out_dir / "__init__.py").write_text(f'from .calibrator import calibrate_{name}\n')

    logger.info(f"Scaffolded calibrator: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Modality scaffold
# ---------------------------------------------------------------------------

_MODALITY_GRAPH_TEMPLATE = '''# Graph template for {name}
# Compose existing primitives from PRIMITIVE_REGISTRY.
# See: packages/pwm_core/pwm_core/graph/primitives.py

description: "{name} imaging modality"
metadata:
  modality: {name}
  x_shape: [64, 64]  # TODO: set correct input shape
  y_shape: [64, 64]  # TODO: set correct output shape

nodes:
  - node_id: forward_op
    primitive_id: conv2d          # TODO: choose correct primitive
    params:
      sigma: 2.0                  # TODO: set physics parameters
    learnable: [sigma]            # Parameters eligible for calibration

edges: []  # TODO: add edges if multi-node graph
'''

_MODALITY_MISMATCH_TEMPLATE = '''# Mismatch parameters for {name}
modality: {name}

parameters:
  param1:
    range: [-1.0, 1.0]           # TODO: set physical range
    typical_error: 0.2
    unit: "pixels"
    param_type: "spatial_shift"
    description: "TODO: describe mismatch source"

severity_weights:
  param1: 1.0

correction_method: "grid_search"
'''

_MODALITY_PHOTON_TEMPLATE = '''# Photon/noise model for {name}
modality: {name}

model_id: "generic"
parameters:
  power_w: 0.001
  wavelength_nm: 500

noise_model: "poisson_gaussian"
photon_levels:
  bright:
    n_photons: 1.0e+05
    scenario: "High SNR"
    read_sigma_fraction: 0.005
  standard:
    n_photons: 1.0e+04
    scenario: "Standard"
    read_sigma_fraction: 0.01
  low_light:
    n_photons: 1.0e+03
    scenario: "Low light"
    read_sigma_fraction: 0.02
'''

_MODALITY_METRICS_TEMPLATE = '''# Metrics for {name}
modality: {name}

primary_metrics:
  - psnr
  - ssim

secondary_metrics: []
# Options: sam, ergas, nrmse, lpips, phase_rmse, temporal_consistency

thresholds:
  rho_minimum: 0.30
  oracle_gap_target: 2.0
'''

_MODALITY_META_TEMPLATE = '''# Modality pack metadata for {name}
name: "{name}"
version: "1.0.0"
description: "TODO: describe the imaging modality"
domain: "other"  # spectral | temporal | spatial | medical | microscopy | other
author: "TODO"
affiliation: "TODO"
license: "Apache-2.0"

references:
  - title: "TODO: original paper"
    doi: ""

supported_solvers:
  - traditional_cpu

status: "experimental"
'''


def scaffold_modality(name: str) -> Path:
    """Generate a modality pack skeleton.

    Parameters
    ----------
    name : str
        Modality name (lowercase, underscores).

    Returns
    -------
    Path
        Path to the created directory.
    """
    out_dir = _CONTRIB_DIR / "modalities" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "graph.yaml").write_text(_MODALITY_GRAPH_TEMPLATE.format(name=name))
    (out_dir / "mismatch.yaml").write_text(_MODALITY_MISMATCH_TEMPLATE.format(name=name))
    (out_dir / "photon.yaml").write_text(_MODALITY_PHOTON_TEMPLATE.format(name=name))
    (out_dir / "metrics.yaml").write_text(_MODALITY_METRICS_TEMPLATE.format(name=name))
    (out_dir / "meta.yaml").write_text(_MODALITY_META_TEMPLATE.format(name=name))

    logger.info(f"Scaffolded modality: {out_dir}")
    return out_dir
