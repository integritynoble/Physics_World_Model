"""Plan Agent — generates structured plan documents from prompts + database context.

The plan has four sections (per spec):
  1. Task     — what the system does / what to reconstruct
  2. Plan     — numbered high-level steps (use element/algorithm names from DB)
  3. Action   — for forward: full system flowchart with every element, noise, mismatch
               — for reconstruction: detailed algorithm steps + mismatch corrections
  4. Demands  — yes/no feasibility flags + comments
"""
from __future__ import annotations
import json
import os
from typing import Literal

from ..config import PLAN_MODEL, OUTPUT_DIR
from ..database import get_modality_context_for_prompt, get_algorithm_context_for_prompt
from ..schemas import PlanDocument, ForwardAction, ReconAction, FlowchartElement, \
    NoiseSpec, MismatchSpec, AlgorithmStep, JudgmentResult
from ..schemas.plan import PlanDemands
from .base_agent import BaseAgent


_SYSTEM_FORWARD = """\
You are an expert physicist and imaging system engineer.
Your task is to design a complete, rigorous forward model for an imaging modality.

When asked to create a forward model plan, you MUST produce a JSON object with
exactly this structure (all fields required):

{
  "task": "<one-paragraph description of what the system should do>",
  "plan_steps": ["<step 1>", "<step 2>", ...],
  "action": {
    "flowchart_ascii": "<ASCII diagram of signal path>",
    "elements": [
      {
        "id": "<snake_case_id>",
        "name": "<Human Name>",
        "type": "<source|interaction|geometry|detector|digitization|processing|other>",
        "parameters": {"key": value, ...},
        "noise": [{"type": "<noise_type>", "parameters": {"key": value}}],
        "mismatch": [
          {
            "type": "<mismatch_type>",
            "description": "<physical explanation>",
            "severity": "<low|medium|high>",
            "correction_method": "<how to correct>"
          }
        ],
        "connects_to": ["<next_element_id>"],
        "notes": ""
      }
    ],
    "measurement_shape": "<tuple string e.g. (180, 512)>",
    "total_noise_model": "<composite noise equation>"
  },
  "demands": {
    "feasibility": "yes",
    "budget_feasible": "yes",
    "algorithm_convergence": "N/A",
    "comments": ""
  }
}

Rules:
- Include EVERY physical element (source, medium interaction, geometry, detector, ADC)
- List ALL realistic noise sources at each stage
- List ALL known mismatch sources and their correction methods
- The flowchart_ascii must show the full signal path including noise/mismatch branches
- budget_feasible must reflect realistic equipment cost estimates
- If a requested feature is physically impossible, set feasibility="no" and explain in comments
"""

_SYSTEM_RECON = """\
You are an expert in computational imaging and inverse problems.
Your task is to design a detailed reconstruction algorithm plan.

When asked to create a reconstruction plan, produce a JSON object with this structure:

{
  "task": "<one-paragraph description of the reconstruction problem>",
  "plan_steps": ["<step 1>", "<step 2>", ...],
  "action": {
    "algorithm_name": "<Algorithm Name>",
    "algorithm_type": "<Classical|Variational|PnP|Deep Unrolling|Diffusion|...>",
    "steps": [
      {
        "step": 1,
        "name": "<Step Name>",
        "description": "<detailed description>",
        "equation": "<LaTeX or plain-text equation>",
        "parameters": {"key": value}
      }
    ],
    "convergence_criterion": "<criterion description>",
    "mismatch_corrections": [
      {
        "type": "<mismatch_type>",
        "description": "<what the mismatch is>",
        "severity": "<low|medium|high>",
        "correction_method": "<specific correction procedure>"
      }
    ],
    "hyperparameters": {"lambda": 0.001, ...},
    "expected_runtime_s": null,
    "references": ["<citation>"]
  },
  "demands": {
    "feasibility": "yes",
    "budget_feasible": "N/A",
    "algorithm_convergence": "yes",
    "comments": ""
  }
}

Rules:
- Each algorithm step must have a clear mathematical description
- Mismatch corrections must be specific and practical
- Convergence criterion must be testable (e.g. ||x_k+1 - x_k|| / ||x_k|| < 1e-4)
- If algorithm is known to diverge without careful tuning, set algorithm_convergence="no"
  and explain in comments
- Prioritize methods that can run without pre-training (TTO, classical iterative, PnP)
"""


class PlanAgent(BaseAgent):
    """Generates a PlanDocument from a prompt + database context."""

    def __init__(self):
        super().__init__(model=PLAN_MODEL)

    def generate(
        self,
        modality: str,
        period: Literal["forward", "reconstruction"],
        prompt: str,
        judge_feedback: str = "",
        iteration: int = 1,
    ) -> PlanDocument:
        """Generate (or regenerate after judge feedback) a plan document."""

        mod_context  = get_modality_context_for_prompt(modality)
        alg_context  = get_algorithm_context_for_prompt(modality) if period == "reconstruction" else ""

        user_content = f"""Modality: **{modality}**
Period: **{period}**

## Database Context
{mod_context}
{"" if period == "forward" else alg_context}

## User Prompt
{prompt}
"""
        if judge_feedback:
            user_content += f"""
## Judge Feedback (iteration {iteration - 1} was rejected)
{judge_feedback}

Please redesign the plan addressing the judge's concerns.
"""

        system = _SYSTEM_FORWARD if period == "forward" else _SYSTEM_RECON

        raw = self.complete_json(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=8192,
        )

        plan = self._parse_raw(raw, modality, period, iteration)
        plan.markdown_path = self._save_markdown(plan)
        return plan

    # ── Private helpers ────────────────────────────────────────────────────────

    def _parse_raw(
        self,
        raw: dict,
        modality: str,
        period: Literal["forward", "reconstruction"],
        iteration: int,
    ) -> PlanDocument:
        action_raw = raw["action"]

        if period == "forward":
            elements = [
                FlowchartElement(
                    id=el["id"],
                    name=el["name"],
                    type=el["type"],
                    parameters=el.get("parameters", {}),
                    noise=[NoiseSpec(**n) for n in el.get("noise", [])],
                    mismatch=[MismatchSpec(**m) for m in el.get("mismatch", [])],
                    connects_to=el.get("connects_to", []),
                    notes=el.get("notes", ""),
                )
                for el in action_raw["elements"]
            ]
            action: ForwardAction | ReconAction = ForwardAction(
                flowchart_ascii=action_raw.get("flowchart_ascii", ""),
                elements=elements,
                measurement_shape=action_raw.get("measurement_shape", ""),
                total_noise_model=action_raw.get("total_noise_model", ""),
            )
        else:
            steps = [AlgorithmStep(**s) for s in action_raw["steps"]]
            mismatch_corrections = [
                MismatchSpec(**m) for m in action_raw.get("mismatch_corrections", [])
            ]
            action = ReconAction(
                algorithm_name=action_raw["algorithm_name"],
                algorithm_type=action_raw.get("algorithm_type", ""),
                steps=steps,
                convergence_criterion=action_raw.get("convergence_criterion", ""),
                mismatch_corrections=mismatch_corrections,
                hyperparameters=action_raw.get("hyperparameters", {}),
                expected_runtime_s=action_raw.get("expected_runtime_s"),
                references=action_raw.get("references", []),
            )

        demands_raw = raw.get("demands", {})
        demands = PlanDemands(
            feasibility=demands_raw.get("feasibility", "yes"),
            budget_feasible=demands_raw.get("budget_feasible", "yes"),
            algorithm_convergence=demands_raw.get("algorithm_convergence", "N/A"),
            comments=demands_raw.get("comments", ""),
        )

        return PlanDocument(
            modality=modality,
            period=period,
            version=1,
            iteration=iteration,
            task=raw["task"],
            plan_steps=raw["plan_steps"],
            action=action,
            demands=demands,
        )

    def _save_markdown(self, plan: PlanDocument) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = f"{plan.modality}_{plan.period}_v{plan.version}_iter{plan.iteration}.md"
        path  = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w") as f:
            f.write(plan.to_markdown())
        print(f"[PlanAgent] Saved plan → {path}")
        return path
