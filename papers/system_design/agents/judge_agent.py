"""Judge Agent — evaluates whether a plan is physically / algorithmically feasible.

Forward checks:
  - Physical mechanism is correct and achievable
  - Noise levels are within acceptable SNR range
  - Budget is within realistic bounds
  - No contradictions in element parameters

Reconstruction checks:
  - Algorithm is known to converge for this problem class
  - All significant mismatch sources are addressed
  - No numerical instabilities in the proposed steps
  - Method is executable without pre-training (TTO / classical preferred)
"""
from __future__ import annotations
from typing import Literal

from ..config import JUDGE_MODEL
from ..schemas import PlanDocument, ForwardAction, ReconAction, JudgmentResult
from ..schemas.judgment import JudgmentIssue
from .base_agent import BaseAgent


_SYSTEM_JUDGE = """\
You are an expert judge evaluating imaging system plans.
You receive a structured plan (markdown text) and must assess its feasibility.

Return a JSON object with EXACTLY this structure:

{
  "feasible": true or false,
  "confidence": 0.0-1.0,
  "summary": "<one paragraph judgment>",
  "issues": [
    {
      "category": "<physics|noise_level|budget|convergence|algorithm|mismatch|other>",
      "severity": "<warning|critical>",
      "element_id": "<element or step id, or empty>",
      "description": "<what is wrong>",
      "suggestion": "<how to fix>"
    }
  ],
  "snr_estimate_db": <number or null>,
  "cost_estimate_usd": <number or null>,
  "budget_ok": true or false or null,
  "convergence_likely": true or false or null,
  "mismatch_handled": true or false or null,
  "redesign_prompt": "<if not feasible: concise instruction for redesign, else empty string>"
}

Judgment rules:
FORWARD period:
  - feasible=false if: mechanism physically impossible, SNR < 5 dB, budget > 10x stated,
    or required element is completely missing
  - feasible=true (with warnings) for: suboptimal parameters, missing minor corrections
  - Estimate SNR from noise parameters; flag if <15 dB

RECONSTRUCTION period:
  - feasible=false if: algorithm diverges on this problem class without pre-training,
    critical mismatch not addressed, or steps are internally contradictory
  - feasible=true (with warnings) for: slow convergence, mild instability risks
  - convergence_likely=false is a critical issue

Tone: be rigorous but constructive. If feasible=false, redesign_prompt must be
specific enough for the Plan Agent to fix the issue in one iteration.
"""


class JudgeAgent(BaseAgent):
    """Evaluates the feasibility of a PlanDocument."""

    def __init__(self):
        super().__init__(model=JUDGE_MODEL)

    def judge(self, plan: PlanDocument) -> JudgmentResult:
        """Return a JudgmentResult for the given plan."""

        period_hint = self._period_context(plan)
        user_content = f"""Period: **{plan.period}**
Modality: **{plan.modality}**
Iteration: {plan.iteration}

{period_hint}

## Plan to Evaluate

```markdown
{plan.to_markdown()}
```
"""
        raw = self.complete_json(
            system=_SYSTEM_JUDGE,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=4096,
        )

        issues = [JudgmentIssue(**i) for i in raw.get("issues", [])]

        result = JudgmentResult(
            feasible=raw["feasible"],
            confidence=raw["confidence"],
            summary=raw["summary"],
            issues=issues,
            snr_estimate_db=raw.get("snr_estimate_db"),
            cost_estimate_usd=raw.get("cost_estimate_usd"),
            budget_ok=raw.get("budget_ok"),
            convergence_likely=raw.get("convergence_likely"),
            mismatch_handled=raw.get("mismatch_handled"),
            redesign_prompt=raw.get("redesign_prompt", ""),
        )

        verdict = "PASS" if result.feasible else "FAIL"
        print(f"[JudgeAgent] Verdict: {verdict} (confidence={result.confidence:.2f})")
        if result.critical_issues:
            print(f"[JudgeAgent] Critical issues ({len(result.critical_issues)}):")
            for issue in result.critical_issues:
                print(f"  - [{issue.category}] {issue.description}")

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _period_context(self, plan: PlanDocument) -> str:
        if plan.period == "forward":
            return (
                "Focus on:\n"
                "1. Physical correctness of each element\n"
                "2. Whether the measurement SNR is acceptable (>15 dB = ok, <5 dB = fail)\n"
                "3. Budget realism\n"
                "4. Completeness of noise and mismatch modelling"
            )
        else:
            return (
                "Focus on:\n"
                "1. Whether the algorithm converges without pre-training\n"
                "2. Whether all critical mismatch sources are corrected\n"
                "3. Mathematical consistency of algorithm steps\n"
                "4. Numerical stability (condition numbers, step sizes)"
            )
