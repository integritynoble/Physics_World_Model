"""pwm_core.science.autonomous_loop

Autonomous science loop: hypothesis -> evaluate -> certificate -> update.

The loop uses a UCB1 bandit to select solvers from a pool and iterates until
convergence (last 3 runs all certified) or max_iterations is reached.

Usage
-----
    from pwm_core.science.autonomous_loop import run_loop

    state = run_loop(
        modality="ct",
        solver_pool=["tv", "admm", "bm3d"],
        n_scenes=5,
        max_iterations=20,
    )
    print(state.best_solver, state.best_psnr)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_CERTIFIED_TIERS = {"certified", "silver", "gold", "platinum"}


@dataclass
class LoopRecord:
    """Result of a single autonomous loop iteration."""

    iteration: int
    solver: str
    trust_tier: str
    psnr: float
    timestamp: str
    bundle_dir: str


@dataclass
class LoopState:
    """Accumulated state of the autonomous science loop."""

    modality: str
    iteration: int
    records: List[LoopRecord] = field(default_factory=list)
    best_solver: str = ""
    best_psnr: float = -float("inf")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]


class AutonomousScienceLoop:
    """Iterates hypothesis -> evaluate -> certificate -> update until convergence.

    Parameters
    ----------
    modality:
        Physics modality to evaluate (e.g. ``"ct"``).
    solver_pool:
        List of solver names to explore via UCB1 bandit.
    n_scenes:
        Number of scenes per evaluation call.
    max_iterations:
        Hard stop after this many iterations.
    out_dir:
        Directory where run bundles are written (relative to repo root or
        absolute).
    """

    def __init__(
        self,
        modality: str,
        solver_pool: List[str],
        n_scenes: int = 5,
        max_iterations: int = 10,
        out_dir: str = "science_loop_runs",
    ) -> None:
        if not solver_pool:
            raise ValueError("solver_pool must contain at least one solver.")

        self.modality = modality
        self.solver_pool = list(solver_pool)
        self.n_scenes = n_scenes
        self.max_iterations = max_iterations

        out_path = Path(out_dir)
        self.out_dir: Path = out_path if out_path.is_absolute() else _REPO_ROOT / out_path
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._state = LoopState(modality=modality, iteration=0)

        # UCB1 bookkeeping: counts and cumulative PSNR per solver
        self._counts: Dict[str, int] = {s: 0 for s in self.solver_pool}
        self._totals: Dict[str, float] = {s: 0.0 for s in self.solver_pool}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> LoopState:
        """Main loop entry point.

        Iterates until max_iterations or convergence and returns the final
        LoopState.
        """
        while self._state.iteration < self.max_iterations:
            record = self.step()
            self._state.records.append(record)
            self._state.iteration += 1

            # Update best tracker
            if record.psnr > self._state.best_psnr:
                self._state.best_psnr = record.psnr
                self._state.best_solver = record.solver

            if self._is_converged():
                break

        return self._state

    def step(self) -> LoopRecord:
        """Run one autonomous loop iteration.

        Returns
        -------
        LoopRecord
            Result of this iteration.
        """
        solver = self._pick_solver()
        cert = self._run_evaluate(solver)

        trust_tier: str = cert.get("trust_tier", cert.get("tier", "uncertified"))
        psnr: float = float(cert.get("psnr_db", cert.get("psnr", 0.0)))
        bundle_dir: str = cert.get("bundle_dir", "")
        timestamp = datetime.now(timezone.utc).isoformat()

        # Update UCB1 statistics
        self._counts[solver] += 1
        self._totals[solver] += psnr

        return LoopRecord(
            iteration=self._state.iteration,
            solver=solver,
            trust_tier=trust_tier,
            psnr=psnr,
            timestamp=timestamp,
            bundle_dir=bundle_dir,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_solver(self) -> str:
        """UCB1 bandit selection from solver_pool.

        Solvers that have never been tried are prioritised (round-robin
        initialisation).  After all solvers have been tried at least once,
        UCB1 selects based on mean PSNR + exploration bonus.
        """
        total_trials = sum(self._counts.values())

        # Prioritise untried solvers
        for solver in self.solver_pool:
            if self._counts[solver] == 0:
                return solver

        # All tried at least once — apply UCB1
        best_solver = self.solver_pool[0]
        best_ucb = -float("inf")
        for solver in self.solver_pool:
            n = self._counts[solver]
            mean = self._totals[solver] / n
            exploration = math.sqrt(2.0 * math.log(total_trials) / n)
            ucb = mean + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_solver = solver

        return best_solver

    def _run_evaluate(self, solver: str) -> dict:
        """Run `pwm evaluate` for the given solver and return the certificate dict.

        Returns either the parsed certificate JSON or an error dict on failure.
        """
        run_id = str(uuid.uuid4())[:8]
        bundle_dir = self.out_dir / f"{self.modality}_{solver}_{run_id}"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "pwm", "evaluate",
            "--modality", self.modality,
            "--solver", solver,
            "--n-scenes", str(self.n_scenes),
            "--run-id", run_id,
            "--out-dir", str(bundle_dir),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(_REPO_ROOT),
            )
        except FileNotFoundError:
            return {
                "error": "'pwm' CLI not found",
                "trust_tier": "error",
                "psnr_db": 0.0,
                "bundle_dir": str(bundle_dir),
            }
        except subprocess.TimeoutExpired:
            return {
                "error": "Evaluation timed out",
                "trust_tier": "error",
                "psnr_db": 0.0,
                "bundle_dir": str(bundle_dir),
            }

        if result.returncode != 0:
            return {
                "error": result.stderr.strip(),
                "trust_tier": "error",
                "psnr_db": 0.0,
                "bundle_dir": str(bundle_dir),
            }

        # Try to find and parse certificate.json written by the CLI
        cert_path = bundle_dir / "certificate.json"
        if cert_path.exists():
            with cert_path.open("r", encoding="utf-8") as fh:
                cert: dict = json.load(fh)
            cert.setdefault("bundle_dir", str(bundle_dir))
            return cert

        # Fallback: try to parse stdout as JSON
        try:
            cert = json.loads(result.stdout)
            cert.setdefault("bundle_dir", str(bundle_dir))
            return cert
        except (json.JSONDecodeError, ValueError):
            pass

        # Last resort: return a minimal dict with the bundle path
        return {
            "trust_tier": "uncertified",
            "psnr_db": 0.0,
            "bundle_dir": str(bundle_dir),
            "stdout": result.stdout.strip(),
        }

    def _is_converged(self) -> bool:
        """Return True if the last 3 iterations all produced a certified tier."""
        if len(self._state.records) < 3:
            return False
        last_three = self._state.records[-3:]
        return all(r.trust_tier.lower() in _CERTIFIED_TIERS for r in last_three)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def run_loop(
    modality: str,
    solver_pool: List[str],
    n_scenes: int = 5,
    max_iterations: int = 10,
    out_dir: str = "science_loop_runs",
) -> LoopState:
    """Convenience wrapper: create an AutonomousScienceLoop and run it.

    Parameters
    ----------
    modality:
        Physics modality (e.g. ``"ct"``, ``"mri"``).
    solver_pool:
        Solvers to explore.
    n_scenes:
        Scenes per evaluation.
    max_iterations:
        Hard iteration cap.
    out_dir:
        Output directory for run bundles.

    Returns
    -------
    LoopState
        Final state including best_solver and best_psnr.
    """
    loop = AutonomousScienceLoop(
        modality=modality,
        solver_pool=solver_pool,
        n_scenes=n_scenes,
        max_iterations=max_iterations,
        out_dir=out_dir,
    )
    return loop.run()
