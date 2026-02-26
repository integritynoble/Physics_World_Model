#!/usr/bin/env python3
"""Extract all 168 YAML benchmark configs into a Python module.

Reads benchmarks/configs/*.yaml and produces
platform/pwm_platform/services/modality_configs.py
with a pre-built dict — no YAML parsing at runtime.
"""

import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "configs"
OUTPUT = Path(__file__).resolve().parent.parent / "pwm_platform" / "services" / "modality_configs.py"


def extract_config(path: Path) -> dict | None:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not raw or raw.get("modality_id") == "_template":
        return None

    # Extract solver names + types
    solvers = {}
    for role, info in (raw.get("solvers") or {}).items():
        solvers[role] = {
            "name": info.get("name", ""),
            "module": info.get("module", ""),
            "function": info.get("function", ""),
        }

    # Extract mismatch params
    mismatch = []
    for p in raw.get("mismatch_params") or []:
        mismatch.append({
            "name": p.get("name", ""),
            "nominal": p.get("nominal", 0),
            "range": p.get("range", [0, 0]),
            "unit": p.get("unit", ""),
        })

    # Source attribution
    sa = raw.get("source_attribution") or {}
    attribution = {}
    for key in ("ground_truth", "forward_model", "solver", "mismatch_ranges"):
        entry = sa.get(key) or {}
        attribution[key] = entry.get("reference", "")

    return {
        "modality_id": raw["modality_id"],
        "display_name": raw.get("display_name", raw["modality_id"]),
        "category": raw.get("category", ""),
        "category_module": raw.get("category_module", ""),
        "x_shape": raw.get("x_shape", [128, 128]),
        "y_shape": raw.get("y_shape", [128, 128]),
        "theta": raw.get("theta") or {},
        "solvers": solvers,
        "mismatch_params": mismatch,
        "expected_psnr_range": raw.get("expected_psnr_range"),
        "source_attribution": attribution,
    }


def main():
    configs = {}
    for yf in sorted(CONFIGS_DIR.glob("*.yaml")):
        if yf.name.startswith("_"):
            continue
        cfg = extract_config(yf)
        if cfg:
            configs[cfg["modality_id"]] = cfg

    # Generate Python module
    lines = [
        '"""Auto-generated modality config registry.',
        "",
        f"Extracted from {len(configs)} YAML configs in benchmarks/configs/.",
        "Do not edit by hand — regenerate with scripts/generate_modality_configs.py",
        '"""',
        "",
        "from typing import Dict, Optional",
        "",
        "",
        f"_CONFIGS: Dict[str, dict] = {repr(configs)}",
        "",
        "",
        "def get_modality_config(modality_id: str) -> Optional[dict]:",
        '    """Look up a modality config by ID. Returns None if not found."""',
        "    return _CONFIGS.get(modality_id)",
        "",
        "",
        "def all_modality_ids() -> list:",
        '    """Return sorted list of all modality IDs."""',
        "    return sorted(_CONFIGS.keys())",
        "",
        "",
        "def get_category_module(modality_id: str) -> Optional[str]:",
        '    """Return the category_module for a modality, or None."""',
        "    cfg = _CONFIGS.get(modality_id)",
        "    return cfg[\"category_module\"] if cfg else None",
        "",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {OUTPUT} with {len(configs)} modality configs")

    # Summary
    from collections import Counter
    cats = Counter(c["category_module"] for c in configs.values())
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
