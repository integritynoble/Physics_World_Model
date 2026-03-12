"""Pydantic schemas for plan documents.

The plan markdown has four sections:
  1. Task   — what to design / reconstruct
  2. Plan   — numbered high-level steps (references element / algorithm names)
  3. Action — full flowchart (forward) or algorithm details (reconstruction)
  4. Demands — yes/no feasibility flags + free-text comments
"""
from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field


# ── Forward model primitives ───────────────────────────────────────────────────

class NoiseSpec(BaseModel):
    type: str = Field(..., description="e.g. poisson, gaussian, shot, dark_current, none")
    parameters: dict[str, Any] = Field(default_factory=dict,
        description="e.g. {'I0': 1e5} for Poisson, {'sigma': 5.0} for Gaussian")


class MismatchSpec(BaseModel):
    type: str = Field(..., description="e.g. beam_hardening, center_of_rotation, defocus")
    description: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    correction_method: str = ""


class FlowchartElement(BaseModel):
    """One block in the forward-model flowchart (source, medium, geometry, detector, …)."""
    id: str
    name: str
    type: Literal["source", "interaction", "geometry", "detector",
                  "digitization", "processing", "other"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    noise: list[NoiseSpec] = Field(default_factory=list)
    mismatch: list[MismatchSpec] = Field(default_factory=list)
    connects_to: list[str] = Field(default_factory=list,
        description="IDs of downstream elements")
    notes: str = ""


class ForwardAction(BaseModel):
    """Action section for the *forward* period."""
    flowchart_ascii: str = Field(...,
        description="ASCII diagram of signal path through all elements")
    elements: list[FlowchartElement]
    measurement_shape: str = Field("",
        description="Shape of output y, e.g. '(180, 512)' for CT sinogram")
    total_noise_model: str = Field("",
        description="Composite noise equation, e.g. 'Poisson(I0 * exp(-Hx)) + N(0, sigma)'")


# ── Reconstruction primitives ──────────────────────────────────────────────────

class AlgorithmStep(BaseModel):
    """One step in the reconstruction algorithm."""
    step: int
    name: str
    description: str
    equation: str = ""   # LaTeX or plain-text equation
    parameters: dict[str, Any] = Field(default_factory=dict)


class ReconAction(BaseModel):
    """Action section for the *reconstruction* period."""
    algorithm_name: str
    algorithm_type: str = Field("",
        description="e.g. 'Variational', 'Deep Unrolling', 'PnP', 'Diffusion'")
    steps: list[AlgorithmStep]
    convergence_criterion: str = ""
    mismatch_corrections: list[MismatchSpec] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    expected_runtime_s: float | None = None
    references: list[str] = Field(default_factory=list)


# ── Top-level plan document ────────────────────────────────────────────────────

class PlanDemands(BaseModel):
    """Section 4 — yes/no feasibility flags."""
    feasibility: Literal["yes", "no"] = "yes"
    budget_feasible: Literal["yes", "no", "N/A"] = "yes"
    algorithm_convergence: Literal["yes", "no", "N/A"] = "yes"
    comments: str = ""


class PlanDocument(BaseModel):
    """Complete plan produced by the Plan Agent.

    Serialised to / from a structured markdown file with YAML front-matter.
    """
    # Metadata
    modality: str
    period: Literal["forward", "reconstruction"]
    version: int = 1
    iteration: int = 1   # redesign counter

    # The four sections
    task: str = Field(..., description="Section 1 – task description")
    plan_steps: list[str] = Field(...,
        description="Section 2 – numbered high-level plan steps")
    action: ForwardAction | ReconAction = Field(...,
        description="Section 3 – detailed action (flowchart or algorithm)")
    demands: PlanDemands = Field(default_factory=PlanDemands,
        description="Section 4 – feasibility flags and comments")

    # Path where the markdown file is saved (set by PlanAgent after writing)
    markdown_path: str = ""

    def to_markdown(self) -> str:
        """Render the plan as a structured markdown string."""
        lines: list[str] = []

        # Front-matter
        lines += [
            "---",
            f"modality: {self.modality}",
            f"period: {self.period}",
            f"version: {self.version}",
            f"iteration: {self.iteration}",
            "---",
            "",
        ]

        # Section 1 – Task
        lines += ["# Task", "", self.task, ""]

        # Section 2 – Plan
        lines += ["# Plan", ""]
        for i, step in enumerate(self.plan_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        # Section 3 – Action
        lines += ["# Action", ""]
        if isinstance(self.action, ForwardAction):
            lines += _render_forward_action(self.action)
        else:
            lines += _render_recon_action(self.action)

        # Section 4 – Demands
        lines += ["# Demands", ""]
        lines += [
            f"- **feasibility**: {self.demands.feasibility}",
            f"- **budget_feasible**: {self.demands.budget_feasible}",
            f"- **algorithm_convergence**: {self.demands.algorithm_convergence}",
        ]
        if self.demands.comments:
            lines += ["", f"**Comments**: {self.demands.comments}"]
        lines.append("")

        return "\n".join(lines)


# ── Private rendering helpers ─────────────────────────────────────────────────

def _render_forward_action(action: ForwardAction) -> list[str]:
    lines: list[str] = ["## System Flowchart", ""]
    if action.flowchart_ascii:
        lines += ["```", action.flowchart_ascii, "```", ""]

    for el in action.elements:
        lines += [f"### Element: {el.name} (`{el.id}`)", ""]
        lines.append(f"- **Type**: {el.type}")
        if el.parameters:
            lines.append("- **Parameters**:")
            for k, v in el.parameters.items():
                lines.append(f"  - `{k}`: {v}")
        if el.noise:
            lines.append("- **Noise**:")
            for n in el.noise:
                pstr = ", ".join(f"{k}={v}" for k, v in n.parameters.items())
                lines.append(f"  - {n.type}: {pstr}")
        if el.mismatch:
            lines.append("- **Mismatch sources**:")
            for m in el.mismatch:
                lines.append(
                    f"  - `{m.type}` [{m.severity}]: {m.description}"
                    + (f" → correction: {m.correction_method}" if m.correction_method else "")
                )
        if el.connects_to:
            lines.append(f"- **Connects to**: {', '.join(el.connects_to)}")
        if el.notes:
            lines.append(f"- **Notes**: {el.notes}")
        lines.append("")

    if action.total_noise_model:
        lines += ["## Composite Noise Model", "", f"```\n{action.total_noise_model}\n```", ""]
    if action.measurement_shape:
        lines += [f"**Measurement shape**: `{action.measurement_shape}`", ""]

    return lines


def _render_recon_action(action: ReconAction) -> list[str]:
    lines: list[str] = [
        f"## Algorithm: {action.algorithm_name}",
        "",
        f"**Type**: {action.algorithm_type}",
        "",
    ]
    if action.references:
        lines.append("**References**:")
        for ref in action.references:
            lines.append(f"  - {ref}")
        lines.append("")

    lines += ["### Algorithm Steps", ""]
    for step in action.steps:
        lines += [f"**Step {step.step}: {step.name}**", ""]
        lines.append(step.description)
        if step.equation:
            lines += ["", f"$$\n{step.equation}\n$$"]
        if step.parameters:
            lines.append("Parameters:")
            for k, v in step.parameters.items():
                lines.append(f"  - `{k}`: {v}")
        lines.append("")

    if action.mismatch_corrections:
        lines += ["### Mismatch Corrections", ""]
        for m in action.mismatch_corrections:
            lines.append(
                f"- `{m.type}` [{m.severity}]: {m.description}"
                + (f"\n  Correction: {m.correction_method}" if m.correction_method else "")
            )
        lines.append("")

    if action.convergence_criterion:
        lines += [f"**Convergence**: {action.convergence_criterion}", ""]

    if action.hyperparameters:
        lines += ["### Hyperparameters", ""]
        for k, v in action.hyperparameters.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    return lines
