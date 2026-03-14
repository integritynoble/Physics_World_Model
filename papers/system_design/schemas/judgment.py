"""Judgment schema returned by the Judge Agent.

The Judge produces three layers of validation:
  1. Structural compilation (6-gate prerequisite)
  2. Triad gate evaluation (primary — from flagship paper)
  3. System feasibility and cost assessment
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class JudgmentIssue(BaseModel):
    category: str = Field(...,
        description="e.g. 'physics', 'noise_level', 'budget', 'convergence', "
                    "'algorithm', 'recoverability', 'carrier_budget', 'mismatch', "
                    "'cost', 'feasibility'")
    severity: Literal["warning", "critical"] = "warning"
    element_id: str = Field("", description="Which flowchart element or algorithm step")
    description: str
    suggestion: str = ""


# ── Triad Gate Results ────────────────────────────────────────────────────────

class TriadGateResult(BaseModel):
    """Result of a single Triad gate evaluation."""
    gate: Literal["G1_recoverability", "G2_carrier_budget", "G3_operator_mismatch"]
    passed: bool
    margin_db: float = Field(0.0,
        description="PSNR margin above target (positive = pass)")
    score_db: float = Field(0.0,
        description="Gate-specific PSNR estimate")
    detail: str = Field("",
        description="Human-readable explanation of the evaluation")
    design_action: str = Field("",
        description="If failed: what the Plan Agent should change")


class TriadReport(BaseModel):
    """Aggregate Triad evaluation report."""
    gate1: TriadGateResult
    gate2: TriadGateResult
    gate3: TriadGateResult
    dominant_gate: Literal["G1", "G2", "G3"] = Field("G3",
        description="Which gate contributes most to potential MSE")
    triad_passed: bool = Field(True,
        description="True if all three gates pass")
    expected_deployment_psnr_db: float | None = Field(None,
        description="Predicted PSNR at deployment (after mismatch)")
    recovery_ratio: float | None = Field(None,
        description="Expected rho from autonomous calibration")


# ── Cost Estimation ───────────────────────────────────────────────────────────

class CostItem(BaseModel):
    """One line item in the hardware cost estimate."""
    category: Literal["source", "optics", "detector", "calibration", "other"]
    item: str
    cost_usd: float
    confidence: Literal["quoted", "estimated", "unknown"] = "estimated"
    note: str = ""


class CostEstimate(BaseModel):
    """Itemized hardware cost from system_elements."""
    items: list[CostItem] = Field(default_factory=list)
    total_usd: float = 0.0
    confidence: Literal["quoted", "estimated", "unknown"] = "estimated"


# ── Feasibility Check ─────────────────────────────────────────────────────────

class FeasibilityItem(BaseModel):
    """Feasibility assessment for one physical element."""
    element: str
    feasible: bool = True
    reason: str = ""
    lead_time: str = Field("", description="e.g. 'off-the-shelf', '2 weeks', '3 months custom'")


class FeasibilityReport(BaseModel):
    """Physical feasibility of all system elements."""
    items: list[FeasibilityItem] = Field(default_factory=list)
    all_feasible: bool = True
    blocking_elements: list[str] = Field(default_factory=list,
        description="Elements that prevent system construction")


# ── Top-level Judgment ────────────────────────────────────────────────────────

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

    # ── Triad gate results (primary) ──
    triad: TriadReport | None = Field(None,
        description="Three-gate Triad evaluation from flagship paper")

    # ── Cost and feasibility (system assessment) ──
    cost: CostEstimate | None = Field(None,
        description="Itemized hardware cost estimate")
    feasibility_report: FeasibilityReport | None = Field(None,
        description="Physical feasibility of system elements")

    # ── Legacy fields (backward compatibility) ──
    snr_estimate_db: float | None = None
    cost_estimate_usd: float | None = None
    budget_ok: bool | None = None

    # Reconstruction-specific
    convergence_likely: bool | None = None
    mismatch_handled: bool | None = None

    @property
    def critical_issues(self) -> list[JudgmentIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def triad_passed(self) -> bool:
        """True if Triad evaluation exists and all gates pass."""
        return self.triad is not None and self.triad.triad_passed
