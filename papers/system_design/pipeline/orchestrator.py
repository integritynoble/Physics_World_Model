"""Orchestrator — manages the full plan → judge → [redesign | perform] loop.

Flow:
  1. PlanAgent generates a plan from prompt + DB context
  2. JudgeAgent evaluates feasibility
     - If FAIL: PlanAgent redesigns (up to MAX_REDESIGN_ITERATIONS)
     - If PASS: PerformanceAgent executes the plan
  3. Returns (final_plan, performance_result)

If the judge fails the plan MAX_REDESIGN_ITERATIONS times in a row, the
orchestrator returns (last_plan, None) and signals failure to the caller.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any

import numpy as np

from ..config import MAX_REDESIGN_ITERATIONS
from ..agents import PlanAgent, JudgeAgent, PerformanceAgent
from ..agents.performance_agent import ForwardResult, ReconResult
from ..schemas import PlanDocument, JudgmentResult


@dataclass
class OrchestratorResult:
    plan: PlanDocument
    judgment: JudgmentResult
    performance: ForwardResult | ReconResult | None
    iterations: int
    success: bool
    failure_reason: str = ""


class Orchestrator:
    """Runs the three-agent pipeline for a given modality and period."""

    def __init__(self):
        self._plan_agent  = PlanAgent()
        self._judge_agent = JudgeAgent()
        self._perf_agent  = PerformanceAgent()

    def run(
        self,
        modality: str,
        period: Literal["forward", "reconstruction"],
        prompt: str,
        phantom: np.ndarray | None = None,
        measurements: np.ndarray | None = None,
        x_true: np.ndarray | None = None,
        verbose: bool = True,
    ) -> OrchestratorResult:
        """Full pipeline: plan → judge → [redesign loop] → perform.

        Args:
            modality:     Modality key (e.g. 'ct', 'mri', 'widefield')
            period:       'forward' or 'reconstruction'
            prompt:       User's natural language description
            phantom:      Optional ground truth for forward simulation
            measurements: Required for reconstruction period
            x_true:       Optional ground truth for reconstruction metrics
            verbose:      Print progress to stdout
        """
        judge_feedback = ""
        plan: PlanDocument | None = None
        judgment: JudgmentResult | None = None

        for iteration in range(1, MAX_REDESIGN_ITERATIONS + 2):
            # ── Step 1: Plan ─────────────────────────────────────────────────
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Orchestrator] Iteration {iteration}/{MAX_REDESIGN_ITERATIONS + 1}")
                print(f"  Modality={modality}  Period={period}")
                print(f"  Generating plan...")

            plan = self._plan_agent.generate(
                modality=modality,
                period=period,
                prompt=prompt,
                judge_feedback=judge_feedback,
                iteration=iteration,
            )

            # ── Step 2: Judge ─────────────────────────────────────────────────
            if verbose:
                print(f"  Judging plan...")

            judgment = self._judge_agent.judge(plan)

            if judgment.feasible:
                if verbose:
                    print(f"  [PASS] Plan accepted on iteration {iteration}")
                break

            # ── Redesign needed ───────────────────────────────────────────────
            if verbose:
                print(f"  [FAIL] Plan rejected. Issues: {len(judgment.issues)}")
                print(f"  Redesign prompt: {judgment.redesign_prompt[:120]}...")

            judge_feedback = judgment.redesign_prompt

            if iteration > MAX_REDESIGN_ITERATIONS:
                if verbose:
                    print(f"  [ABORT] Exceeded max redesign iterations ({MAX_REDESIGN_ITERATIONS})")
                return OrchestratorResult(
                    plan=plan,
                    judgment=judgment,
                    performance=None,
                    iterations=iteration,
                    success=False,
                    failure_reason=(
                        f"Plan failed judge after {MAX_REDESIGN_ITERATIONS} redesigns. "
                        f"Last issue: {judgment.redesign_prompt}"
                    ),
                )

        # ── Step 3: Perform ───────────────────────────────────────────────────
        if verbose:
            print(f"\n[Orchestrator] Running Performance Agent...")

        try:
            perf_result = self._perf_agent.run(
                plan=plan,
                phantom=phantom,
                measurements=measurements,
                x_true=x_true,
            )
        except Exception as exc:
            return OrchestratorResult(
                plan=plan,
                judgment=judgment,
                performance=None,
                iterations=iteration,
                success=False,
                failure_reason=f"Performance agent error: {exc}",
            )

        return OrchestratorResult(
            plan=plan,
            judgment=judgment,
            performance=perf_result,
            iterations=iteration,
            success=True,
        )
