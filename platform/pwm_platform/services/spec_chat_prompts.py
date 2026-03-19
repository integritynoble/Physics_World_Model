"""System prompt assembly and response parsing for the spec-builder chat.

build_system_prompt() — assembles the full system prompt from primitives,
category-appropriate example specs, and the current variant context.

parse_spec_from_response() — extracts a JSON spec block from LLM prose.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pwm_platform.services.benchmark_database._primitives import SPEC_PRIMITIVES
from pwm_platform.services.benchmark_database._variant_registry import VARIANT_REGISTRY

# ── Hardcoded example specs (rich, from the 65 manual variants) ───────────────

_EXAMPLE_SPECS: dict[str, dict[str, Any]] = {
    "cassi": {
        "label": "CASSI",
        "variant_key": "sd_cassi",
        "spec_notation": VARIANT_REGISTRY["sd_cassi"]["spec_notation"],
        "forward_model": VARIANT_REGISTRY["sd_cassi"]["spec_dag"],
        "mismatch_params": VARIANT_REGISTRY["sd_cassi"]["mismatch_params"],
        "noise_model": "Mixed Poisson-Gaussian (η₄)",
        "measurement_matrix": "Binary coded aperture mask convolved with spectral dispersion.",
    },
    "spc": {
        "label": "SPC-Block",
        "variant_key": "spc_block",
        "spec_notation": VARIANT_REGISTRY["spc_block"]["spec_notation"],
        "forward_model": VARIANT_REGISTRY["spc_block"]["spec_dag"],
        "mismatch_params": VARIANT_REGISTRY["spc_block"]["mismatch_params"],
        "noise_model": "Gaussian (η₁)",
        "measurement_matrix": "Block-structured sensing matrix Φ with structured illumination.",
    },
    "cacti": {
        "label": "CACTI",
        "variant_key": "cacti",
        "spec_notation": VARIANT_REGISTRY["cacti"]["spec_notation"],
        "forward_model": VARIANT_REGISTRY["cacti"]["spec_dag"],
        "mismatch_params": VARIANT_REGISTRY["cacti"]["mismatch_params"],
        "noise_model": "Mixed Poisson-Gaussian (η₄)",
        "measurement_matrix": "Time-varying binary mask sequence m_t modulating each video frame.",
    },
}

# ── Category → representative example variant keys ───────────────────────────
# For each category, pick 3 rich variants from the 65-entry registry to use as
# few-shot examples when the user selects a modality from that category.

_CATEGORY_EXAMPLES: dict[str, list[str]] = {
    "compressive": ["cassi", "spc", "cacti"],
    "medical": ["cassi", "spc", "cacti"],  # will use CT/MRI/PET examples below
    "medical_ultrasound": ["cassi", "spc", "cacti"],
    "coherent": ["cassi", "spc", "cacti"],
    "microscopy": ["cassi", "spc", "cacti"],
    "electron_microscopy": ["cassi", "spc", "cacti"],
    "clinical_optics": ["cassi", "spc", "cacti"],
    "computational": ["cassi", "spc", "cacti"],
    "computational_photography": ["cassi", "spc", "cacti"],
    "neural_rendering": ["cassi", "spc", "cacti"],
    "depth_imaging": ["cassi", "spc", "cacti"],
    "remote_sensing": ["cassi", "spc", "cacti"],
    "particle_imaging": ["cassi", "spc", "cacti"],
    "scanning_probe": ["cassi", "spc", "cacti"],
    "industrial_inspection": ["cassi", "spc", "cacti"],
    "spectroscopy": ["cassi", "spc", "cacti"],
    "astronomy": ["cassi", "spc", "cacti"],
    "ultrafast": ["cassi", "spc", "cacti"],
    "quantum": ["cassi", "spc", "cacti"],
    "experimental_science": ["cassi", "spc", "cacti"],
    "scientific_instrumentation": ["cassi", "spc", "cacti"],
    "multi_modal_fusion": ["cassi", "spc", "cacti"],
}

# ── Category context descriptions ────────────────────────────────────────────

_CATEGORY_CONTEXT: dict[str, str] = {
    "compressive": "Compressive imaging systems use coded apertures or structured masks (M) to multiplex the signal before detection. Typical DAGs: M → W → Σ → D or M → Σ → D.",
    "medical": "Medical imaging modalities use projection (Π), Fourier encoding (F), or propagation (P) to form images from X-ray, RF, or gamma carriers. Typical DAGs: Π → D, M → F → S → D.",
    "medical_ultrasound": "Ultrasound modalities use acoustic propagation (P) and sampling (S) for real-time imaging. Typical DAG: P → S → D.",
    "coherent": "Coherent imaging captures phase information through interference (P+P) or diffraction (C). Typical DAGs: C → D, P+P → Σ → D.",
    "microscopy": "Fluorescence and optical microscopy use convolution (C) with point-spread functions and sampling (S). Typical DAGs: C → D, C → S → D, S → C → D.",
    "electron_microscopy": "Electron microscopy uses electron beams with scattering (C), projection (Π), or diffraction. Typical DAGs: C → D, Π → D, S → D.",
    "clinical_optics": "Clinical optical systems (OCT, fundus) use low-coherence interference (P+P) or direct imaging (C). Typical DAGs: P+P → Σ → D, C → D.",
    "computational": "Computational imaging captures multi-view or light-field data and computationally reconstructs 3D structure. Typical DAGs: P → R → D, R → D.",
    "computational_photography": "Computational photography uses lensless optics, coded exposure, or multi-focus fusion. Typical DAGs: C → D, M → C → D.",
    "neural_rendering": "Neural rendering synthesizes novel views from multi-view captures. Typical DAGs: P → R → D, R → D.",
    "depth_imaging": "Depth imaging uses time-of-flight (P), structured light (S → M), or LiDAR (P → S). Typical DAGs: P → D, S → M → D.",
    "remote_sensing": "Remote sensing uses radar (M → F → D) or sonar (P → D) for large-scale imaging. Typical DAGs: M → F → D, P → D.",
    "particle_imaging": "Particle imaging uses neutron, proton, or muon beams for tomographic reconstruction. Typical DAG: Π → D.",
    "scanning_probe": "Scanning probe microscopies (AFM, STM) raster a probe across the surface. Typical DAG: S → D.",
    "industrial_inspection": "Industrial inspection modalities use X-ray, acoustic, or thermal propagation for non-destructive testing. Typical DAGs: P → D, Π → D.",
    "spectroscopy": "Spectroscopy and spectral imaging capture wavelength-resolved information. Typical DAGs: W → D, F → D, M → W → D.",
    "astronomy": "Astronomical imaging captures faint signals through atmosphere or space optics. Typical DAGs: C → D, Π → W → D, P → D.",
    "ultrafast": "Ultrafast imaging captures sub-nanosecond dynamics using temporal coding or streak cameras. Typical DAGs: M → Σ → D, M → D.",
    "quantum": "Quantum imaging exploits photon correlations or entanglement for sub-shot-noise imaging. Typical DAGs: S → D, C → D.",
    "experimental_science": "Broad experimental science imaging includes seismic, gravitational wave, and other specialized modalities. Typical DAGs vary widely.",
    "scientific_instrumentation": "Scientific instrumentation includes mass spectrometry, NMR, and other analytical tools. Typical DAGs: F → D, S → D.",
    "multi_modal_fusion": "Multi-modal fusion combines data from two or more imaging modalities. Typical DAG: (Branch₁) + (Branch₂) → Fusion.",
}

# ── Carrier → default noise model ────────────────────────────────────────────

_CARRIER_NOISE: dict[str, str] = {
    "Photon": "Mixed Poisson-Gaussian (η₄)",
    "X-ray": "Mixed Poisson-Gaussian (η₄)",
    "Gamma": "Poisson (shot noise)",
    "Gamma/X-ray": "Poisson (shot noise)",
    "Electron": "Poisson (shot noise)",
    "Electron/Photon": "Poisson (shot noise)",
    "Photon/Electron": "Poisson (shot noise)",
    "Photon/EUV": "Poisson (shot noise)",
    "Photon/IR": "Mixed Poisson-Gaussian (η₄)",
    "Spin/RF": "Gaussian (η₁)",
    "RF": "Gaussian (η₁)",
    "Acoustic": "Gaussian (η₁)",
    "Seismic": "Gaussian (η₁)",
    "Seismic/Acoustic": "Gaussian (η₁)",
    "Mechanical": "Gaussian (η₁)",
    "Magnetic": "Gaussian (η₁)",
    "Electric": "Gaussian (η₁)",
    "Neutron": "Poisson (shot noise)",
    "Proton": "Poisson (shot noise)",
    "Muon": "Poisson (shot noise)",
    "Ion": "Poisson (shot noise)",
    "Particle": "Poisson (shot noise)",
    "MV": "Poisson (shot noise)",
    "IR": "Gaussian (η₁)",
    "THz": "Gaussian (η₁)",
    "EM": "Gaussian (η₁)",
    "Gravitational": "Gaussian (η₁)",
}


def _default_noise_for_carrier(carrier: str) -> str:
    """Return the default noise model string for a given carrier type."""
    return _CARRIER_NOISE.get(carrier, "Gaussian (η₁)")


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

    vk = ex.get('variant_key', name)
    return (
        f"### {ex['label']}\n"
        f"- variant_key: {vk}\n"
        f"- Spec notation: {ex['spec_notation']}\n"
        f"- Pipeline: {dag_str}\n"
        f"- Noise model: {ex['noise_model']}\n"
        f"- Measurement matrix: {ex['measurement_matrix']}\n"
        f"- Mismatch parameters:\n" + "\n".join(mm_lines)
    )


def _get_examples_for_variant(variant: dict) -> list[tuple[str, dict]]:
    """Select the best example specs for the variant's category."""
    category = variant.get("category", "compressive")
    example_keys = _CATEGORY_EXAMPLES.get(category, ["cassi", "spc", "cacti"])
    result = []
    for key in example_keys:
        ex = _EXAMPLE_SPECS.get(key)
        if ex:
            result.append((key, ex))
    # Ensure we always have at least the 3 defaults
    if not result:
        for key in ["cassi", "spc", "cacti"]:
            ex = _EXAMPLE_SPECS.get(key)
            if ex:
                result.append((key, ex))
    return result


def build_system_prompt(
    variant: dict,
    dataset_meta: dict | None = None,
    matrix_meta: dict | None = None,
) -> str:
    """Assemble the system prompt for the spec-builder chat.

    Parameters
    ----------
    variant : dict
        The current variant record (from VARIANT_DATABASE), containing at
        least ``spec_notation``, ``display_name``, and ``category``.
    dataset_meta : dict | None
        Metadata from an uploaded measurement file (dataset mode).
    matrix_meta : dict | None
        Metadata from an uploaded sensing matrix (dataset mode).

    Returns
    -------
    str
        Full system prompt for Gemini.
    """
    primitives_block = _format_primitives()

    examples = _get_examples_for_variant(variant)
    examples_block = "\n\n".join(
        _format_example(name, ex) for name, ex in examples
    )

    category = variant.get("category", "other")
    category_context = _CATEGORY_CONTEXT.get(category, "")

    # Determine carrier and noise model for context
    carrier = variant.get("carrier", "")
    if not carrier:
        # Try to look up from the catalog
        from pwm_platform.services.benchmark_database._modality_catalog import MODALITY_CATALOG
        parent = variant.get("parent_modality", "")
        cat_entry = MODALITY_CATALOG.get(parent)
        if cat_entry:
            carrier = cat_entry.get("carrier", "Photon")
    noise_model = _default_noise_for_carrier(carrier) if carrier else ""

    carrier_section = ""
    if carrier or category_context:
        parts = []
        if category_context:
            parts.append(category_context)
        if carrier:
            parts.append(f"Carrier type: **{carrier}**. Default noise model: {noise_model}.")
        carrier_section = "\n## Category Context\n\n" + " ".join(parts)

    # Dataset mode section
    dataset_section = ""
    if dataset_meta:
        shape = dataset_meta.get("shape", [])
        stats = dataset_meta.get("stats", {})
        fname = dataset_meta.get("original_filename", "unknown")
        fmt = dataset_meta.get("file_format", "unknown")
        dtype = dataset_meta.get("dtype", "unknown")

        matrix_info = ""
        if matrix_meta:
            m_fname = matrix_meta.get("original_filename", "unknown")
            m_shape = matrix_meta.get("shape", [])
            matrix_info = f"\n- **Sensing matrix**: {m_fname}, shape={m_shape}"

        dataset_section = f"""

## Dataset Mode — Active

The user has uploaded real measurement data:
- **Measurement**: {fname} ({fmt}), shape={shape}, dtype={dtype}
  Value range: [{stats.get('min', '?'):.4g}, {stats.get('max', '?'):.4g}], \
mean={stats.get('mean', '?'):.4g}, std={stats.get('std', '?'):.4g}{matrix_info}

## Your Task in Dataset Mode

1. Design a spec whose y_shape is COMPATIBLE with the uploaded data shape {shape}.
2. If a sensing matrix is provided, validate that Phi maps x_shape → y_shape.
3. Based on data statistics, suggest appropriate noise model parameters.
4. Identify the variant_key that best matches this data.
5. After spec is finalized, the user clicks "Reconstruct from Dataset" to apply \
the best reconstruction method from PWM benchmarks.
"""

    # Upload reminder for non-dataset mode
    upload_reminder = ""
    if not dataset_meta:
        upload_reminder = """

## File Upload Support

If the user mentions having measurement data, remind them they can upload it \
using the "Upload Dataset" button for reconstruction with their actual data.
"""

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
  "variant_key": "modality_id (e.g. sd_cassi, spc_block, cacti, ct, mri, confocal_3d)",
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

The `variant_key` MUST match the imaging modality being described. Common values: \
sd_cassi, dd_cassi, cacti, spc_block, spc_kronecker, ct, cbct, mri, pet, \
confocal_3d, sem, tem, widefield, coded_exposure, ghost_imaging. Use snake_case.

## Example Specs

{examples_block}
{carrier_section}

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
{dataset_section}{upload_reminder}"""


def get_example_spec(name: str) -> dict[str, Any] | None:
    """Return a pre-built example spec dict, or None if not found.

    Falls back to auto-building from MODALITY_CATALOG if the name isn't
    in the hardcoded examples.
    """
    ex = _EXAMPLE_SPECS.get(name)
    if ex is not None:
        return ex

    # Auto-build from catalog
    from pwm_platform.services.benchmark_database._modality_catalog import MODALITY_CATALOG

    cat_entry = MODALITY_CATALOG.get(name)
    if cat_entry is None:
        return None

    carrier = cat_entry.get("carrier", "Photon")
    return {
        "label": cat_entry["display_name"],
        "variant_key": name,
        "spec_notation": cat_entry["spec_notation"],
        "forward_model": cat_entry["spec_dag"],
        "mismatch_params": cat_entry["mismatch_params"],
        "noise_model": _default_noise_for_carrier(carrier),
        "measurement_matrix": f"Forward model for {cat_entry['display_name']} ({cat_entry['canonical_dag']}).",
    }


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
