"""System prompt assembly and response parsing for the spec-builder chat.

build_system_prompt() — assembles the full system prompt from primitives,
example specs (CASSI, SPC, CACTI), and the current variant context.

parse_spec_from_response() — extracts a JSON spec block from LLM prose.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pwm_platform.services.benchmark_database._primitives import SPEC_PRIMITIVES
from pwm_platform.services.benchmark_database._variant_registry import VARIANT_REGISTRY

# ── Example specs (compact, used as few-shot examples in the prompt) ─────────

_EXAMPLE_SPECS: dict[str, dict[str, Any]] = {
    "cassi": {
        "label": "SD-CASSI",
        "spec_notation": VARIANT_REGISTRY["sd_cassi"]["spec_notation"],
        "forward_model": VARIANT_REGISTRY["sd_cassi"]["spec_dag"],
        "mismatch_params": VARIANT_REGISTRY["sd_cassi"]["mismatch_params"],
        "noise_model": "Mixed Poisson-Gaussian (η₄)",
        "measurement_matrix": "Binary coded aperture mask convolved with spectral dispersion.",
    },
    "spc": {
        "label": "SPC-Block",
        "spec_notation": VARIANT_REGISTRY["spc_block"]["spec_notation"],
        "forward_model": VARIANT_REGISTRY["spc_block"]["spec_dag"],
        "mismatch_params": VARIANT_REGISTRY["spc_block"]["mismatch_params"],
        "noise_model": "Gaussian (η₁)",
        "measurement_matrix": "Block-structured sensing matrix Φ with structured illumination.",
    },
    "cacti": {
        "label": "CACTI",
        "spec_notation": VARIANT_REGISTRY["cacti"]["spec_notation"],
        "forward_model": VARIANT_REGISTRY["cacti"]["spec_dag"],
        "mismatch_params": VARIANT_REGISTRY["cacti"]["mismatch_params"],
        "noise_model": "Mixed Poisson-Gaussian (η₄)",
        "measurement_matrix": "Time-varying binary mask sequence m_t modulating each video frame.",
    },
}


def _format_primitives() -> str:
    """Format the 11 primitives as a numbered list for the system prompt."""
    lines = []
    for i, (key, p) in enumerate(SPEC_PRIMITIVES.items(), 1):
        lines.append(f"{i}. {p['symbol']} ({p['name']}): {p['description']}")
    return "\n".join(lines)


def _format_example(name: str, ex: dict) -> str:
    """Format a single example spec as a block."""
    dag_str = " → ".join(
        f"{n['primitive']}({n['params']})" if n.get("params") else n["primitive"]
        for n in ex["forward_model"]
    )
    mm_lines = []
    for p in ex["mismatch_params"]:
        mm_lines.append(
            f"  - {p['symbol']} ({p['name']}): {p['description']} "
            f"[nominal={p['nominal']}, perturbed={p['perturbed']}]"
        )

    return (
        f"### {ex['label']}\n"
        f"- Spec notation: {ex['spec_notation']}\n"
        f"- Pipeline: {dag_str}\n"
        f"- Noise model: {ex['noise_model']}\n"
        f"- Measurement matrix: {ex['measurement_matrix']}\n"
        f"- Mismatch parameters:\n" + "\n".join(mm_lines)
    )


def build_system_prompt(variant: dict) -> str:
    """Assemble the system prompt for the spec-builder chat.

    Parameters
    ----------
    variant : dict
        The current variant record (from VARIANT_DATABASE), containing at
        least ``spec_notation`` and ``display_name``.

    Returns
    -------
    str
        Full system prompt for Gemini.
    """
    primitives_block = _format_primitives()
    examples_block = "\n\n".join(
        _format_example(name, ex) for name, ex in _EXAMPLE_SPECS.items()
    )

    return f"""\
You are the **PWM Spec Builder**, an expert assistant that helps researchers \
design and refine imaging modality specifications using the Physics World Model \
(PWM) primitive library.

## Your Task

Help the user build or refine a spec for an imaging system. Each spec \
describes the forward model as a DAG (directed acyclic graph) of primitive \
operators, along with the noise model, measurement matrix description, and \
mismatch parameters for robustness testing.

## Available Primitives (11 total)

{primitives_block}

## Output Format

When you produce or update a spec, ALWAYS include a JSON block fenced with \
```json ... ``` containing these keys:

```json
{{
  "spec_notation": "P1(...) → P2(...) → ... → D(g, η)",
  "forward_model": [
    {{"primitive": "KEY", "params": "...", "label": "Human-readable label"}},
    ...
  ],
  "mismatch_params": [
    {{"name": "param_id", "symbol": "σ", "description": "...", "nominal": 0, "perturbed": 0.1}},
    ...
  ],
  "noise_model": "Description of the noise model (e.g. Mixed Poisson-Gaussian η₄)",
  "measurement_matrix": "Description of the measurement/sensing matrix"
}}
```

## Example Specs

{examples_block}

## Current Context

The user is currently viewing the **{variant.get('display_name', 'Unknown')}** \
variant page, which has spec notation: `{variant.get('spec_notation', 'N/A')}`.

## Guidelines

1. Always use primitives from the library above. Do not invent new ones.
2. Explain your reasoning before the JSON block — describe what each primitive \
does in the pipeline and why the parameters are chosen.
3. When the user asks to modify a spec, show the complete updated spec (not \
just the diff).
4. For mismatch parameters, choose physically meaningful perturbation values.
5. If the user pastes a JSON spec, validate it against the primitive library \
and describe what the spec represents.
6. Keep explanations concise but technically accurate.
"""


def get_example_spec(name: str) -> dict[str, Any] | None:
    """Return a pre-built example spec dict, or None if not found."""
    return _EXAMPLE_SPECS.get(name)


def parse_spec_from_response(text: str) -> tuple[str, dict[str, Any] | None]:
    """Extract a JSON spec block from LLM prose.

    Parameters
    ----------
    text : str
        Raw LLM response that may contain a ```json ... ``` block.

    Returns
    -------
    tuple[str, dict | None]
        ``(explanation_text, parsed_spec_or_none)``
        The explanation is the text with the JSON block removed (for display).
        The spec dict is None if no valid JSON block was found.
    """
    # Find ```json ... ``` blocks
    pattern = r"```json\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if match is None:
        return text.strip(), None

    json_str = match.group(1).strip()
    explanation = text[:match.start()].strip()
    # Also grab any text after the JSON block
    after = text[match.end():].strip()
    if after:
        explanation = explanation + "\n\n" + after if explanation else after

    try:
        spec = json.loads(json_str)
        if not isinstance(spec, dict):
            return text.strip(), None
        return explanation, spec
    except json.JSONDecodeError:
        return text.strip(), None
