#!/usr/bin/env python3
"""Generate _modality_catalog.py from all YAML configs in benchmarks/configs/.

Reads every modality YAML (excluding _template.yaml), parses the canonical_dag,
mismatch_params, category, carrier, and has_dedicated_operator fields, and
writes a static Python file that can be imported at runtime.

Usage:
    python platform/scripts/generate_modality_catalog.py
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "configs"
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "pwm_platform" / "services" / "benchmark_database" / "_modality_catalog.py"
)

# ── Category slug mapping (YAML display name → factory slug) ─────────────────

_CATEGORY_SLUG_MAP: dict[str, str] = {
    "Compressive Imaging": "compressive",
    "Medical Imaging": "medical",
    "Coherent Imaging": "coherent",
    "Microscopy": "microscopy",
    "Electron Microscopy": "electron_microscopy",
    "Computational Optics": "computational",
    "Computational Photography": "computational_photography",
    "Neural Rendering": "neural_rendering",
    "Depth Imaging": "depth_imaging",
    "Remote Sensing": "remote_sensing",
    "Multi-Modal Fusion": "multi_modal_fusion",
    "Scanning Probe Microscopy": "scanning_probe",
    "Industrial Inspection": "industrial_inspection",
    "Spectroscopy & Spectral Imaging": "spectroscopy",
    "Astronomy & Space Imaging": "astronomy",
    "Ultrafast Imaging": "ultrafast",
    "Quantum Imaging": "quantum",
    "Broader Experimental Science": "experimental_science",
    "Scientific Instrumentation": "scientific_instrumentation",
    "Particle Imaging": "particle_imaging",
    # Existing slugs that might appear directly
    "medical_ultrasound": "medical_ultrasound",
    "clinical_optics": "clinical_optics",
}

# ── Primitive label lookup ───────────────────────────────────────────────────

_PRIMITIVE_LABELS: dict[str, str] = {
    "C": "Convolution",
    "D": "Detector",
    "F": "Fourier",
    "M": "Modulation",
    "P": "Propagation",
    "Pi": "Projection",
    "R": "Rotation",
    "S": "Sampling",
    "Sigma": "Summation",
    "Src": "Source",
    "W": "Warp",
    "Fusion": "Fusion",
}

# ── Unicode symbol lookup ────────────────────────────────────────────────────

_PRIMITIVE_UNICODE: dict[str, str] = {
    "C": "C",
    "D": "D",
    "F": "F",
    "M": "M",
    "P": "P",
    "Pi": "Π",
    "R": "R",
    "S": "S",
    "Sigma": "Σ",
    "Src": "Src",
    "W": "W",
    "Fusion": "⊕",
}


def _slugify_category(raw: str) -> str:
    """Convert a YAML category string to a factory-compatible slug."""
    raw = raw.strip().strip('"').strip("'")
    if raw in _CATEGORY_SLUG_MAP:
        return _CATEGORY_SLUG_MAP[raw]
    # Fallback: lowercase, replace spaces/special chars with underscores
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return slug


def _parse_dag_branch(branch_str: str) -> list[dict]:
    """Parse a single DAG branch like 'Pi --> D' or 'M --> R,P --> D' into nodes."""
    nodes = []
    parts = [p.strip() for p in branch_str.split("-->")]
    for part in parts:
        # Handle annotation like "D (CT)" or "D (PET)"
        annotation = ""
        ann_match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", part)
        if ann_match:
            part = ann_match.group(1).strip()
            annotation = ann_match.group(2).strip()

        # Handle comma-separated compound primitives like "R,P" or "R,P,R"
        if "," in part:
            sub_prims = [s.strip() for s in part.split(",")]
            for sp in sub_prims:
                nodes.append({
                    "primitive": sp,
                    "params": "",
                    "label": _PRIMITIVE_LABELS.get(sp, sp),
                })
        # Handle addition compound like "P+P"
        elif "+" in part and not part.startswith("P+P"):
            sub_prims = [s.strip() for s in part.split("+")]
            for sp in sub_prims:
                nodes.append({
                    "primitive": sp,
                    "params": "",
                    "label": _PRIMITIVE_LABELS.get(sp, sp),
                })
        elif part == "P+P":
            # Interference: two propagation beams combined
            nodes.append({
                "primitive": "P",
                "params": "",
                "label": "Propagation (reference)",
            })
            nodes.append({
                "primitive": "P",
                "params": "",
                "label": "Propagation (sample)",
            })
        else:
            label = _PRIMITIVE_LABELS.get(part, part)
            if annotation:
                label = f"{label} ({annotation})"
            nodes.append({
                "primitive": part,
                "params": "",
                "label": label,
            })
    return nodes


def _parse_canonical_dag(dag_str: str) -> tuple[list[dict], list[str], str]:
    """Parse a canonical_dag string into (spec_dag, primitives_list, spec_notation).

    Handles simple, compound, and fusion DAGs.

    Returns:
        (spec_dag, primitives, spec_notation)
    """
    dag_str = dag_str.strip()

    # Detect fusion DAGs: "branch1 + branch2 --> Fusion" pattern
    # e.g. "Pi --> D (CT) + Pi --> D (PET) --> Fusion"
    # e.g. "C --> D (LM) + C --> D (EM) --> Fusion"
    fusion_match = re.match(
        r"^(.+?)\s*\+\s*(.+?)\s*-->\s*Fusion\s*$",
        dag_str,
    )
    if fusion_match and "-->" in fusion_match.group(1) and "-->" in fusion_match.group(2):
        branch_a = _parse_dag_branch(fusion_match.group(1).strip())
        branch_b = _parse_dag_branch(fusion_match.group(2).strip())
        fusion_node = {"primitive": "Fusion", "params": "", "label": "Fusion"}
        spec_dag = branch_a + branch_b + [fusion_node]
        primitives = list(dict.fromkeys(n["primitive"] for n in spec_dag))

        # Build notation
        def _branch_notation(nodes: list[dict]) -> str:
            return " → ".join(_PRIMITIVE_UNICODE.get(n["primitive"], n["primitive"]) for n in nodes)

        notation = f"({_branch_notation(branch_a)}) + ({_branch_notation(branch_b)}) → ⊕"
        return spec_dag, primitives, notation

    # Check for P+P compound (OCT-style interference)
    if "P+P" in dag_str:
        parts = [p.strip() for p in dag_str.split("-->")]
        spec_dag = []
        for part in parts:
            if part == "P+P":
                spec_dag.append({"primitive": "P", "params": "", "label": "Propagation (reference)"})
                spec_dag.append({"primitive": "P", "params": "", "label": "Propagation (sample)"})
            else:
                spec_dag.append({
                    "primitive": part,
                    "params": "",
                    "label": _PRIMITIVE_LABELS.get(part, part),
                })
        primitives = list(dict.fromkeys(n["primitive"] for n in spec_dag))
        symbols = []
        for part in parts:
            if part == "P+P":
                symbols.append("P+P")
            else:
                symbols.append(_PRIMITIVE_UNICODE.get(part, part))
        notation = " → ".join(symbols)
        return spec_dag, primitives, notation

    # Simple / multi-step / comma-compound DAGs
    spec_dag = _parse_dag_branch(dag_str)
    primitives = list(dict.fromkeys(n["primitive"] for n in spec_dag))
    notation = " → ".join(
        _PRIMITIVE_UNICODE.get(n["primitive"], n["primitive"]) for n in spec_dag
    )
    return spec_dag, primitives, notation


def _convert_mismatch_params(yaml_params: list[dict]) -> list[dict]:
    """Convert YAML mismatch params {name, nominal, range, unit} to variant-style format."""
    result = []
    for p in yaml_params:
        name = p.get("name", "")
        nominal = p.get("nominal", 0)
        param_range = p.get("range", [0, 0])
        unit = p.get("unit", "-")

        # Generate a symbol from the first letter/word
        words = name.split()
        if len(words) >= 2:
            symbol = words[0][0].lower() + "_" + words[1][0].lower()
        elif words:
            symbol = words[0][0].lower()
        else:
            symbol = "p"

        # Compute a reasonable perturbed value (move 20% toward range boundary)
        if isinstance(param_range, list) and len(param_range) == 2:
            lo, hi = param_range
            if hi > nominal:
                perturbed = round(nominal + 0.2 * (hi - nominal), 4)
            elif lo < nominal:
                perturbed = round(nominal + 0.2 * (lo - nominal), 4)
            else:
                perturbed = nominal
        else:
            perturbed = nominal

        result.append({
            "name": name.lower().replace(" ", "_").replace("-", "_"),
            "symbol": symbol,
            "description": f"{name} ({unit})",
            "nominal": nominal,
            "perturbed": perturbed,
        })
    return result


def load_yaml_config(path: Path) -> dict | None:
    """Load and validate a single YAML config."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        if "modality_id" not in data or "canonical_dag" not in data:
            return None
        return data
    except Exception as e:
        print(f"  WARNING: Failed to parse {path.name}: {e}", file=sys.stderr)
        return None


def build_catalog_entry(data: dict) -> dict:
    """Build a single catalog entry from parsed YAML data."""
    mod_id = data["modality_id"]
    display_name = data.get("display_name", mod_id)
    category = _slugify_category(data.get("category", "other"))
    carrier = data.get("carrier", "Photon")
    dag_str = data["canonical_dag"]
    has_op = data.get("has_dedicated_operator", False)

    spec_dag, primitives, spec_notation = _parse_canonical_dag(dag_str)
    mismatch_params = _convert_mismatch_params(data.get("mismatch_params", []))

    return {
        "modality_id": mod_id,
        "display_name": display_name,
        "category": category,
        "canonical_dag": dag_str,
        "carrier": carrier,
        "primitives": primitives,
        "spec_dag": spec_dag,
        "spec_notation": spec_notation,
        "mismatch_params": mismatch_params,
        "has_dedicated_operator": has_op,
    }


def generate_catalog() -> dict[str, dict]:
    """Read all YAML configs and produce the catalog dict."""
    catalog: dict[str, dict] = {}
    yaml_files = sorted(CONFIGS_DIR.glob("*.yaml"))

    for path in yaml_files:
        if path.name.startswith("_"):
            continue
        data = load_yaml_config(path)
        if data is None:
            continue
        entry = build_catalog_entry(data)
        catalog[entry["modality_id"]] = entry

    return catalog


def write_catalog_file(catalog: dict[str, dict]) -> None:
    """Write the catalog as a Python file."""
    lines = [
        '"""Auto-generated modality catalog — DO NOT EDIT MANUALLY.',
        "",
        f"Generated from {len(catalog)} YAML configs in benchmarks/configs/.",
        'Run `python platform/scripts/generate_modality_catalog.py` to regenerate.',
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "MODALITY_CATALOG: dict[str, dict] = {",
    ]

    for mod_id, entry in sorted(catalog.items()):
        lines.append(f"    {mod_id!r}: {{")
        for key, value in entry.items():
            lines.append(f"        {key!r}: {value!r},")
        lines.append("    },")

    lines.append("}")
    lines.append("")

    # Convenience lookups
    lines.append("")
    lines.append("# ── Convenience lookups ────────────────────────────────────────────────────")
    lines.append("")
    lines.append("ALL_MODALITY_IDS: list[str] = sorted(MODALITY_CATALOG.keys())")
    lines.append("")
    lines.append("")
    lines.append("def get_categories() -> dict[str, list[str]]:")
    lines.append('    """Return {category_slug: [modality_id, ...]} mapping."""')
    lines.append("    cats: dict[str, list[str]] = {}")
    lines.append("    for mod_id, entry in MODALITY_CATALOG.items():")
    lines.append('        cats.setdefault(entry["category"], []).append(mod_id)')
    lines.append("    return cats")
    lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(catalog)} entries to {OUTPUT_PATH}")


def main() -> None:
    print(f"Reading YAML configs from {CONFIGS_DIR}")
    catalog = generate_catalog()
    print(f"Parsed {len(catalog)} modality configs")

    # Print category breakdown
    cats: dict[str, int] = {}
    for entry in catalog.values():
        cats[entry["category"]] = cats.get(entry["category"], 0) + 1
    print("\nCategory breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    write_catalog_file(catalog)


if __name__ == "__main__":
    main()
