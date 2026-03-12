"""Judgment schema returned by the Judge Agent."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class JudgmentIssue(BaseModel):
    category: str = Field(...,
        description="e.g. 'physics', 'noise_level', 'budget', 'convergence', 'algorithm'")
    severity: Literal["warning", "critical"] = "warning"
    element_id: str = Field("", description="Which flowchart element or algorithm step")
    description: str
    suggestion: str = ""


class JudgmentResult(BaseModel):
    """Output of the Judge Agent.

    If `feasible=False`, the orchestrator feeds `redesign_prompt` back to the
    Plan Agent for a revised plan.
    """
    feasible: bool
    confidence: float = Field(..., ge=0.0, le=1.0,
        description="Judge's confidence in the feasibility verdict")
    issues: list[JudgmentIssue] = Field(default_factory=list)
    summary: str = Field(...,
        description="One-paragraph summary of the judgment")
    redesign_prompt: str = Field("",
        description="If not feasible: concise instruction for the Plan Agent on what to fix")

    # Forward-specific
    snr_estimate_db: float | None = None
    cost_estimate_usd: float | None = None
    budget_ok: bool | None = None

    # Reconstruction-specific
    convergence_likely: bool | None = None
    mismatch_handled: bool | None = None

    @property
    def critical_issues(self) -> list[JudgmentIssue]:
        return [i for i in self.issues if i.severity == "critical"]
