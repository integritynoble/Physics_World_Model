"""
new_calibrator_template.py

Contributor template: add a new calibrator / theta-fitting routine to PWM.

A calibrator searches over theta (operator parameters) to reduce mismatch between:
  - measured y (or simulated y_true)
  - predicted y_hat = A_theta(x_hat)

In PWM, calibrators are used for:
- "y + A(theta) -> fit theta -> reconstruct"
- Two-physics mismatch experiments: theta_true vs theta_model

This template implements a bounded coarse search + optional local refine step.
You should keep it deterministic and reproducible.

How to use:
1) Copy this file to:
     packages/pwm_core/pwm_core/contrib/templates/my_calibrator.py
2) Implement TODOs.
3) Register via registry (entrypoints) or pwm_core/core/registry.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from pwm_core.mismatch.calibrators import BaseCalibrator
from pwm_core.mismatch.scoring import ScoreConfig, score_theta_candidate


@dataclass
class MyCalibConfig:
    """Config for your calibrator."""
    candidates: int = 12
    refine_top_k: int = 3
    refine_steps: int = 8
    seed: int = 0


class MyCalibrator(BaseCalibrator):
    """Example calibrator. Replace with your own logic."""

    CALIBRATOR_ID = "my_calibrator"

    def __init__(self, config: MyCalibConfig):
        self.config = config

    def propose_candidates(self, theta_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return a list of theta dicts to evaluate (bounded)."""
        # TODO: implement sampling strategy (grid, latin hypercube, Sobol, etc.)
        # For now: naive random uniform within min/max for scalar params.
        g = torch.Generator().manual_seed(self.config.seed)
        cands: List[Dict[str, Any]] = []
        for _ in range(self.config.candidates):
            th = {}
            for k, spec in theta_space.items():
                if isinstance(spec, dict) and "min" in spec and "max" in spec:
                    th[k] = float((spec["max"] - spec["min"]) * torch.rand((), generator=g).item() + spec["min"])
            cands.append(th)
        return cands

    def refine(self, theta_init: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Optional local refinement."""
        # TODO: implement local search or gradient steps.
        # Keep it bounded and robust. Default: return unchanged.
        return dict(theta_init)

    def fit(
        self,
        operator,                      # PhysicsOperator
        y_meas: torch.Tensor,
        recon_fn,                      # callable that returns x_hat given operator/theta
        theta_space: Dict[str, Any],
        score_cfg: Optional[ScoreConfig] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Main entry: fit theta then return best theta + logs."""
        context = context or {}
        score_cfg = score_cfg or ScoreConfig()

        candidates = self.propose_candidates(theta_space)

        best = None
        best_score = float("inf")
        logs = []

        for th in candidates:
            x_hat = recon_fn(operator, th)
            s = score_theta_candidate(operator, th, y_meas, x_hat, score_cfg)
            logs.append({"theta": th, "score": float(s)})
            if s < best_score:
                best_score = float(s)
                best = dict(th)

        # refine top-k (optional): here we just refine best
        if best is not None and self.config.refine_steps > 0:
            best = self.refine(best, context)

        return {
            "theta_best": best or {},
            "best_score": best_score,
            "num_evals": len(candidates),
            "logs": logs,
        }


def register(registry) -> None:
    """Entry-point style registration."""
    registry.register_calibrator(MyCalibrator.CALIBRATOR_ID, MyCalibrator)
