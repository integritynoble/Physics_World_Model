"""System Design Agent Service — Plan, Judge, Performance agents via Gemini 2.5 Flash.

Three agents:
  1. Plan Agent — generates spec.md (structured plan for forward model or reconstruction)
  2. Judge Agent — validates the spec for physical/algorithmic feasibility
  3. Performance Agent — analyzes expected performance metrics

Uses CompareGPT API (Gemini 2.5 Flash) for LLM calls.
Session state stored in SpecChatSession.dataset_meta (JSONB).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pwm_platform.services.gemini_client import call_gemini

logger = logging.getLogger(__name__)

# ── Compiler integration (lazy import to avoid import errors if pwm_core missing)
_COMPILER_AVAILABLE = False
try:
    from papers.system_design.compiler.agent_translator import AgentToGraphTranslator
    from papers.system_design.compiler.primitive_compiler import (
        CompilationReport,
        ConstrainedPrimitiveCompiler,
    )
    _COMPILER_AVAILABLE = True
except Exception:
    pass

# ── System Prompts ──────────────────────────────────────────────────────────

_PLAN_FORWARD_SYSTEM = """\
You are PWM (Physics World Model), an expert physicist and imaging system engineer.
Your task is to design complete, rigorous forward models for imaging systems.

When asked to design a forward model, produce a JSON object with EXACTLY this structure:
{
  "task": "<one-paragraph description of what the system does>",
  "plan_steps": ["<step 1>", "<step 2>", ...],
  "action": {
    "flowchart_ascii": "<ASCII diagram showing signal path, e.g.:\\n[Source] -> [Medium] -> [Geometry] -> [Detector] -> [ADC] -> y>",
    "elements": [
      {
        "id": "<snake_case_id>",
        "name": "<Human-Readable Name>",
        "type": "<source|interaction|geometry|detector|digitization|processing|other>",
        "parameters": {"key": "value"},
        "noise": [{"type": "<poisson|gaussian|dark_current|shot|none>", "parameters": {"key": "value"}}],
        "mismatch": [{"type": "<mismatch_type>", "description": "<physics explanation>", "severity": "<low|medium|high>", "correction_method": "<how to correct>"}],
        "connects_to": ["<next_element_id>"],
        "notes": ""
      }
    ],
    "measurement_shape": "<e.g. (180, 512)>",
    "total_noise_model": "<composite noise equation>"
  },
  "demands": {
    "feasibility": "yes",
    "budget_feasible": "yes",
    "algorithm_convergence": "N/A",
    "comments": "<assessment>"
  }
}

Rules:
- Include EVERY physical element in the signal chain (source, medium, geometry, detector, ADC)
- List ALL realistic noise sources at each stage with numeric parameters
- List ALL known mismatch sources and their correction methods
- The flowchart_ascii MUST show the full signal path
- Set feasibility="no" if the design is physically impossible
- Be specific about parameters (wavelengths, voltages, pixel counts, etc.)

IMPORTANT: Return ONLY valid JSON. No markdown fences, no extra text before or after.
"""

_PLAN_RECON_SYSTEM = """\
You are PWM (Physics World Model), an expert in computational imaging and inverse problems.
Your task is to design detailed reconstruction algorithm plans.

When asked to design a reconstruction plan, produce a JSON object with EXACTLY this structure:
{
  "task": "<one-paragraph description of the reconstruction problem>",
  "plan_steps": ["<step 1>", "<step 2>", ...],
  "action": {
    "algorithm_name": "<Algorithm Name>",
    "algorithm_type": "<Classical|Variational|PnP|Deep Unrolling|Diffusion>",
    "steps": [
      {
        "step": 1,
        "name": "<Step Name>",
        "description": "<detailed description>",
        "equation": "<LaTeX or plain-text equation>",
        "parameters": {"key": "value"}
      }
    ],
    "convergence_criterion": "<testable criterion, e.g. ||x_{k+1} - x_k|| / ||x_k|| < 1e-4>",
    "mismatch_corrections": [
      {"type": "<mismatch_type>", "description": "<what it is>", "severity": "<low|medium|high>", "correction_method": "<specific correction>"}
    ],
    "hyperparameters": {"lambda": 0.001},
    "expected_runtime_s": null,
    "references": ["<citation>"]
  },
  "demands": {
    "feasibility": "yes",
    "budget_feasible": "N/A",
    "algorithm_convergence": "yes",
    "comments": "<assessment>"
  }
}

Rules:
- Each step must have a clear mathematical description
- Mismatch corrections must be specific and practical
- Convergence criterion must be testable
- Prioritize methods that run without pre-training (classical, variational, PnP, TTO)
- Set algorithm_convergence="no" if the method is known to diverge

IMPORTANT: Return ONLY valid JSON. No markdown fences, no extra text.
"""

_REFINE_SYSTEM = """\
You are PWM (Physics World Model). The user wants to refine an existing system design spec.

Current spec (JSON):
{current_spec}

Modify the spec based on the user's request. Return the COMPLETE updated spec as valid JSON
(same structure as the original). Only change what the user asks for; keep everything else intact.

IMPORTANT: Return ONLY the complete updated JSON. No markdown fences, no extra text.
"""

_JUDGE_SYSTEM = """\
You are an expert reviewer evaluating imaging system designs for physical and algorithmic feasibility.

Evaluate the plan and return a JSON object with EXACTLY this structure:
{
  "feasible": true,
  "confidence": 0.85,
  "summary": "<one-paragraph judgment>",
  "issues": [
    {
      "category": "<physics|noise_level|budget|convergence|algorithm|mismatch|other>",
      "severity": "<warning|critical>",
      "element_id": "<element or step id>",
      "description": "<what is wrong>",
      "suggestion": "<how to fix>"
    }
  ],
  "snr_estimate_db": null,
  "budget_ok": true,
  "convergence_likely": true,
  "mismatch_handled": true,
  "redesign_prompt": ""
}

FORWARD period rules:
- feasible=false if: mechanism physically impossible, SNR < 5 dB, budget > 10x stated,
  or required element completely missing
- feasible=true (with warnings) for: suboptimal parameters, missing minor corrections

RECONSTRUCTION period rules:
- feasible=false if: algorithm diverges without pre-training, critical mismatch unaddressed,
  or steps internally contradictory
- feasible=true (with warnings) for: slow convergence, mild instability risks

If feasible=false, redesign_prompt must be specific enough to fix in one iteration.

IMPORTANT: Return ONLY valid JSON. No markdown fences.
"""

_PERFORMANCE_SYSTEM = """\
You are PWM (Physics World Model) analyzing the expected performance of an imaging system design.

Given the system spec, provide a detailed analysis covering:
1. Expected measurement SNR (dB) with calculation
2. Expected reconstruction quality (PSNR in dB, SSIM) with reasoning
3. Expected computational runtime and memory requirements
4. Key performance bottlenecks and limiting factors
5. Comparison with published benchmarks for this modality
6. Recommendations for improvement

Be specific with numbers. Reference relevant literature where applicable.
Format your response with clear sections and bullet points.
"""


# ── JSON extraction ─────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()

    # Direct parse
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Markdown code fences
    for pat in (r"```json\s*\n?(.*?)\n?```", r"```\s*\n?(.*?)\n?```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Brace extraction
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:300]}...")


# ── Detection helpers ───────────────────────────────────────────────────────

def detect_period(prompt: str) -> str:
    """Detect whether the user wants forward or reconstruction."""
    lower = prompt.lower()
    recon_kw = [
        "reconstruct", "reconstruction", "inverse", "recover", "algorithm",
        "solver", "tv-admm", "pnp", "fbp", "sart", "iterative", "denois",
    ]
    if any(kw in lower for kw in recon_kw):
        return "reconstruction"
    return "forward"


def detect_modality(prompt: str) -> str:
    """Detect modality from the user's prompt."""
    lower = prompt.lower()
    mapping: dict[str, list[str]] = {
        "ct": ["ct ", "ct,", "ct.", "computed tomography", "x-ray ct", "sinogram"],
        "mri": ["mri", "magnetic resonance", "k-space", "kspace"],
        "ultrasound": ["ultrasound", "sonograph", "b-mode"],
        "oct": ["oct ", "optical coherence"],
        "pet": ["pet ", "positron emission"],
        "spect": ["spect", "single photon emission"],
        "fluorescence": ["fluorescen", "confocal"],
        "widefield": ["widefield", "wide-field", "microscop"],
        "hyperspectral": ["hyperspectral", "spectral imaging"],
        "lidar": ["lidar", "time-of-flight", "tof "],
        "radar": ["radar", "sar ", "synthetic aperture radar"],
        "infrared": ["infrared", "thermal imag"],
        "xray": ["x-ray", "xray", "radiograph"],
        "photoacoustic": ["photoacoustic", "optoacoustic"],
        "endoscopy": ["endoscop"],
        "fundus": ["fundus", "retina"],
        "elastography": ["elastograph", "shear wave"],
    }
    for mod, keywords in mapping.items():
        if any(kw in lower for kw in keywords):
            return mod
    return "generic"


# ── Markdown rendering ──────────────────────────────────────────────────────

def spec_to_plan_md(spec: dict, period: str) -> str:
    """Render a high-level plan summary (plan.md)."""
    lines = [f"# Task\n\n{spec.get('task', 'N/A')}\n", "# Plan\n"]
    for i, step in enumerate(spec.get("plan_steps", []), 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    demands = spec.get("demands", {})
    lines.append("# Demands\n")
    lines.append(f"- **feasibility**: {demands.get('feasibility', 'yes')}")
    lines.append(f"- **budget_feasible**: {demands.get('budget_feasible', 'N/A')}")
    lines.append(f"- **algorithm_convergence**: {demands.get('algorithm_convergence', 'N/A')}")
    if demands.get("comments"):
        lines.append(f"\n**Comments**: {demands['comments']}")
    lines.append("")
    return "\n".join(lines)


def spec_to_spec_md(spec: dict, period: str) -> str:
    """Render the full spec (spec.md) with all details."""
    lines = [f"# Task\n\n{spec.get('task', 'N/A')}\n"]

    lines.append("# Plan\n")
    for i, step in enumerate(spec.get("plan_steps", []), 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    lines.append("# Action\n")
    action = spec.get("action", {})

    if period == "forward":
        _render_forward(lines, action)
    else:
        _render_recon(lines, action)

    demands = spec.get("demands", {})
    lines.append("# Demands\n")
    lines.append(f"- **feasibility**: {demands.get('feasibility', 'yes')}")
    lines.append(f"- **budget_feasible**: {demands.get('budget_feasible', 'N/A')}")
    lines.append(f"- **algorithm_convergence**: {demands.get('algorithm_convergence', 'N/A')}")
    if demands.get("comments"):
        lines.append(f"\n**Comments**: {demands['comments']}")
    lines.append("")
    return "\n".join(lines)


def _render_forward(lines: list[str], action: dict) -> None:
    if action.get("flowchart_ascii"):
        lines += ["## System Flowchart\n", f"```\n{action['flowchart_ascii']}\n```\n"]
    for el in action.get("elements", []):
        lines.append(f"### Element: {el.get('name', '')} (`{el.get('id', '')}`)\n")
        lines.append(f"- **Type**: {el.get('type', '')}")
        if el.get("parameters"):
            lines.append("- **Parameters**:")
            for k, v in el["parameters"].items():
                lines.append(f"  - `{k}`: {v}")
        if el.get("noise"):
            lines.append("- **Noise**:")
            for n in el["noise"]:
                pstr = ", ".join(f"{k}={v}" for k, v in n.get("parameters", {}).items())
                lines.append(f"  - {n.get('type', '')}: {pstr}")
        if el.get("mismatch"):
            lines.append("- **Mismatch sources**:")
            for m in el["mismatch"]:
                line = f"  - `{m.get('type', '')}` [{m.get('severity', '')}]: {m.get('description', '')}"
                if m.get("correction_method"):
                    line += f" -> correction: {m['correction_method']}"
                lines.append(line)
        if el.get("connects_to"):
            lines.append(f"- **Connects to**: {', '.join(el['connects_to'])}")
        lines.append("")
    if action.get("total_noise_model"):
        lines += ["## Composite Noise Model\n", f"```\n{action['total_noise_model']}\n```\n"]
    if action.get("measurement_shape"):
        lines.append(f"**Measurement shape**: `{action['measurement_shape']}`\n")


def _render_recon(lines: list[str], action: dict) -> None:
    lines.append(f"## Algorithm: {action.get('algorithm_name', '')}\n")
    lines.append(f"**Type**: {action.get('algorithm_type', '')}\n")
    if action.get("references"):
        lines.append("**References**:")
        for ref in action["references"]:
            lines.append(f"  - {ref}")
        lines.append("")
    lines.append("### Algorithm Steps\n")
    for step in action.get("steps", []):
        lines.append(f"**Step {step.get('step', '')}: {step.get('name', '')}**\n")
        lines.append(step.get("description", ""))
        if step.get("equation"):
            lines.append(f"\n$$\n{step['equation']}\n$$")
        if step.get("parameters"):
            lines.append("Parameters:")
            for k, v in step["parameters"].items():
                lines.append(f"  - `{k}`: {v}")
        lines.append("")
    if action.get("mismatch_corrections"):
        lines.append("### Mismatch Corrections\n")
        for m in action["mismatch_corrections"]:
            line = f"- `{m.get('type', '')}` [{m.get('severity', '')}]: {m.get('description', '')}"
            if m.get("correction_method"):
                line += f"\n  Correction: {m['correction_method']}"
            lines.append(line)
        lines.append("")
    if action.get("convergence_criterion"):
        lines.append(f"**Convergence**: {action['convergence_criterion']}\n")
    if action.get("hyperparameters"):
        lines.append("### Hyperparameters\n")
        for k, v in action["hyperparameters"].items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")


# ── Chat description ────────────────────────────────────────────────────────

def describe_spec(spec: dict, period: str) -> str:
    """Generate a concise flowchart summary for the chat bubble."""
    action = spec.get("action", {})

    if period == "forward":
        elements = action.get("elements", [])
        # Build concise flowchart: just element names joined by arrows
        names = [el.get("name", "?") for el in elements]
        flowchart = " → ".join(names) + " → y"
        shape = action.get("measurement_shape", "")
        parts = [
            f"**Forward Model** ({len(elements)} elements)",
            f"```\n{flowchart}\n```",
        ]
        if shape:
            parts.append(f"Output: `{shape}`")

        # Compiler status
        compilation = spec.get("_compilation")
        if compilation:
            chain = compilation.get("canonical_chain", "")
            if compilation.get("valid"):
                parts.append(f"Compiler: **VALID** `{chain}`")
            else:
                parts.append(f"Compiler: **INVALID** ({len(compilation.get('failures', []))} issues)")
    else:
        algo = action.get("algorithm_name", "Unknown")
        algo_type = action.get("algorithm_type", "")
        steps = action.get("steps", [])
        # Build concise step list: just step names
        step_names = [f"{s.get('step', i+1)}. {s.get('name', '')}" for i, s in enumerate(steps)]
        parts = [
            f"**{algo}** ({algo_type}, {len(steps)} steps)",
            "\n".join(step_names),
        ]

    return "\n".join(parts)


# ── Compiler validation ────────────────────────────────────────────────────

def compile_forward_spec(spec: dict, modality: str = "generic") -> dict | None:
    """Run the Constrained Primitive Compiler on a forward spec.

    Returns a dict with compilation results, or None if compiler unavailable.
    """
    if not _COMPILER_AVAILABLE:
        return None

    action = spec.get("action", {})
    if not action.get("elements"):
        return None

    try:
        translator = AgentToGraphTranslator()
        graph_spec = translator.translate(action, modality=modality)

        compiler = ConstrainedPrimitiveCompiler()
        report = compiler.compile(graph_spec, modality=modality)

        return {
            "valid": report.valid,
            "canonical_chain": report.canonical_chain_str,
            "node_count": report.node_count,
            "depth": report.depth,
            "nonlinear_ok": report.nonlinear_ok,
            "failures": report.failures,
            "warnings": report.warnings,
            "compilation_time_s": round(report.compilation_time_s, 4),
        }
    except Exception as e:
        logger.warning(f"Compiler failed: {e}")
        return {"valid": False, "failures": [str(e)], "warnings": []}


# ── Agent API ───────────────────────────────────────────────────────────────

async def generate_plan(
    prompt: str,
    period: str | None = None,
    modality: str | None = None,
) -> tuple[dict, str, str, str]:
    """Generate a new plan from a user prompt.

    Returns ``(spec_json, description, plan_md, spec_md)``.
    """
    if period is None:
        period = detect_period(prompt)
    if modality is None:
        modality = detect_modality(prompt)

    system = _PLAN_FORWARD_SYSTEM if period == "forward" else _PLAN_RECON_SYSTEM
    user_msg = f"Modality: {modality}\nPeriod: {period}\n\nUser request: {prompt}"
    history = [{"role": "user", "content": user_msg}]

    response = await call_gemini(system, history)
    spec = _extract_json(response)

    # Run compiler on forward specs
    if period == "forward":
        compilation = compile_forward_spec(spec, modality=modality or "generic")
        if compilation:
            spec["_compilation"] = compilation

    description = describe_spec(spec, period)
    plan_md = spec_to_plan_md(spec, period)
    spec_md = spec_to_spec_md(spec, period)
    return spec, description, plan_md, spec_md


async def refine_plan(
    prompt: str,
    current_spec: dict,
    period: str,
    conversation_history: list[dict],
) -> tuple[dict, str, str, str]:
    """Refine an existing plan based on user feedback.

    Returns ``(spec_json, description, plan_md, spec_md)``.
    """
    system = _REFINE_SYSTEM.format(current_spec=json.dumps(current_spec, indent=2))
    history = list(conversation_history) + [{"role": "user", "content": prompt}]

    response = await call_gemini(system, history)
    spec = _extract_json(response)

    # Run compiler on refined forward specs
    if period == "forward":
        compilation = compile_forward_spec(spec, modality="generic")
        if compilation:
            spec["_compilation"] = compilation

    description = describe_spec(spec, period)
    plan_md = spec_to_plan_md(spec, period)
    spec_md = spec_to_spec_md(spec, period)
    return spec, description, plan_md, spec_md


async def judge_plan(spec: dict, period: str) -> tuple[dict, str]:
    """Judge a plan for feasibility.

    Returns ``(judgment_json, judgment_summary_text)``.
    """
    spec_md = spec_to_spec_md(spec, period)
    user_msg = f"Period: {period}\n\nPlan to evaluate:\n\n```markdown\n{spec_md}\n```"
    history = [{"role": "user", "content": user_msg}]

    response = await call_gemini(_JUDGE_SYSTEM, history)
    judgment = _extract_json(response)

    feasible = judgment.get("feasible", False)
    confidence = judgment.get("confidence", 0)
    summary = judgment.get("summary", "")
    issues = judgment.get("issues", [])

    # Include compiler results in judgment
    compilation = spec.get("_compilation")
    if compilation:
        judgment["_compilation"] = compilation
        if not compilation.get("valid", True):
            # Compiler failures are critical
            for fail in compilation.get("failures", []):
                issues.append({
                    "category": "compiler",
                    "severity": "critical",
                    "element_id": "",
                    "description": fail,
                    "suggestion": "Fix the forward model to satisfy the 11-primitive basis",
                })

    verdict = "PASS" if feasible else "FAIL"
    lines = [
        f"**Judge Verdict**: {verdict} (confidence: {confidence:.0%})",
        "",
        summary,
    ]

    # Compiler status line
    if compilation:
        chain = compilation.get("canonical_chain", "?")
        comp_status = "VALID" if compilation.get("valid") else "INVALID"
        lines.append(f"\n**Compiler**: {comp_status} | Chain: `{chain}` | Nodes: {compilation.get('node_count', '?')}")

    if issues:
        lines.append("")
        for issue in issues:
            sev = "WARNING" if issue.get("severity") == "warning" else "CRITICAL"
            lines.append(
                f"[{sev}] **{issue.get('category', '')}**: {issue.get('description', '')}"
            )
            if issue.get("suggestion"):
                lines.append(f"  -> {issue['suggestion']}")

    return judgment, "\n".join(lines)


async def run_performance(spec: dict, period: str) -> str:
    """Analyze expected performance of a plan.

    Returns a natural-language performance analysis.
    """
    spec_md = spec_to_spec_md(spec, period)
    user_msg = (
        f"Period: {period}\n\n"
        f"System spec:\n\n```markdown\n{spec_md}\n```\n\n"
        "Provide a detailed performance analysis."
    )
    history = [{"role": "user", "content": user_msg}]
    return await call_gemini(_PERFORMANCE_SYSTEM, history)
