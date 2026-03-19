#!/usr/bin/env python3
"""Auto-generate 168 benchmark YAML configs from registries and markdown specs.

Reads:
  - docs/modality_benchmarks/*.md  — header metadata, mismatch params
  - contrib/modalities.yaml        — operator params, element chains
  - contrib/graph_templates.yaml   — template existence check
  - contrib/solver_registry.yaml   — solver tiers
  - contrib/dataset_registry.yaml  — dataset URLs

Writes:
  - benchmarks/configs/<modality_id>.yaml  (one per modality)

Usage:
    python -m benchmarks.runners.generate_configs
    python -m benchmarks.runners.generate_configs --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Paths
ROOT = Path(__file__).parent.parent.parent
SPECS_DIR = ROOT / "docs" / "modality_benchmarks"
CONTRIB_DIR = ROOT / "packages" / "pwm_core" / "contrib"
CONFIGS_DIR = ROOT / "benchmarks" / "configs"


# ---------------------------------------------------------------------------
# DAG → category module mapping
# ---------------------------------------------------------------------------

DAG_TO_CATEGORY = {
    "C --> D": "microscopy_psf",
    "C --> N --> D": "microscopy_psf",
    "C --> C --> D": "microscopy_psf",
    "P --> C --> D": "microscopy_psf",
    "C --> D --> D": "microscopy_psf",
    "P --> S --> D": "microscopy_psf",
    "S --> D": "microscopy_psf",
    "M --> D": "compressive_mask",
    "M --> Sigma --> D": "compressive_mask",
    "M --> W --> Sigma --> D": "compressive_mask",
    "M --> W --> D": "compressive_mask",
    "M --> T --> D": "compressive_mask",
    "Pi --> D": "medical_ct_radon",
    "Pi --> N --> D": "medical_ct_radon",
    "Pi --> I --> D": "medical_ct_radon",
    "F --> S --> D": "medical_mri_kspace",
    "F --> D": "medical_mri_kspace",
    "F --> S --> N --> D": "medical_mri_kspace",
    "F --> M --> D": "medical_mri_kspace",
    "CTF --> D": "electron_ctf",
    "P --> CTF --> D": "electron_ctf",
    "E --> D": "electron_ctf",
    "E --> S --> D": "electron_ctf",
    "T --> C --> D": "scanning_probe",
    "T --> D": "scanning_probe",
    "T --> F --> D": "scanning_probe",
    "Phi --> R --> D": "remote_sensing_sar",
    "Phi --> D": "remote_sensing_sar",
    "Phi --> I --> D": "remote_sensing_sar",
    "R --> D": "remote_sensing_sar",
    "A --> Pi --> D": "nuclear_emission",
    "A --> D": "nuclear_emission",
    "E --> Pi --> D": "nuclear_emission",
}

# Category keyword fallback mapping
CATEGORY_KEYWORD_MAP = {
    "Microscopy": "microscopy_psf",
    "Super-Resolution Microscopy": "microscopy_psf",
    "Spectral Microscopy": "microscopy_psf",
    "Compressive Imaging": "compressive_mask",
    "Computational Photography": "compressive_mask",
    "Medical Imaging -- CT": "medical_ct_radon",
    "Medical Imaging -- MRI": "medical_mri_kspace",
    "Medical Imaging -- Ultrasound": "nuclear_emission",
    "Medical Imaging -- Nuclear": "nuclear_emission",
    "Medical Imaging -- Optical": "microscopy_psf",
    "Medical Imaging -- X-ray": "medical_ct_radon",
    "Medical Imaging": "medical_ct_radon",
    "Electron Microscopy": "electron_ctf",
    "Scanning Probe Microscopy": "scanning_probe",
    "Remote Sensing": "remote_sensing_sar",
    "Nuclear / Particle Imaging": "nuclear_emission",
    "Acoustic / Seismic Imaging": "nuclear_emission",
    "Spectroscopic Imaging": "microscopy_psf",
    "Coherent Imaging": "microscopy_psf",
    "Radio / Microwave Imaging": "remote_sensing_sar",
    "Industrial / NDT": "medical_ct_radon",
    "Astronomical Imaging": "remote_sensing_sar",
    "Ultrafast / Time-Resolved": "compressive_mask",
    "Quantum Imaging": "compressive_mask",
    "3D / Volumetric Reconstruction": "medical_ct_radon",
}

# DAG → default synthetic generator
DAG_TO_GENERATOR = {
    "C --> D": "cell_phantom",
    "Pi --> D": "shepp_logan",
    "F --> S --> D": "shepp_logan",
    "M --> D": "spectral_scene",
    "M --> W --> Sigma --> D": "spectral_scene",
}


# ---------------------------------------------------------------------------
# Markdown spec parser
# ---------------------------------------------------------------------------

def parse_markdown_spec(path: Path) -> Dict[str, Any]:
    """Parse a modality benchmark markdown spec file.

    Returns dict with: modality_id, display_name, category, canonical_dag,
    carrier, maturity, forward_model_type, default_solver, mismatch_params,
    expected_psnr_range.
    """
    text = path.read_text()
    lines = text.split("\n")
    result: Dict[str, Any] = {"modality_id": path.stem}

    # Line 1: # Display Name (`modality_id`)
    m = re.match(r"^#\s+(.+?)\s+\(`(\w+)`\)", lines[0] if lines else "")
    if m:
        result["display_name"] = m.group(1)
        result["modality_id"] = m.group(2)

    # Line 3: **Category**: ... | **Canonical DAG**: ... | **Carrier**: ...
    for line in lines[1:6]:
        cat_m = re.search(r"\*\*Category\*\*:\s*([^|]+)", line)
        if cat_m:
            result["category"] = cat_m.group(1).strip()
        dag_m = re.search(r"\*\*Canonical DAG\*\*:\s*([^|]+)", line)
        if dag_m:
            result["canonical_dag"] = dag_m.group(1).strip()
        carrier_m = re.search(r"\*\*Carrier\*\*:\s*(\S+)", line)
        if carrier_m:
            result["carrier"] = carrier_m.group(1).strip()
        mat_m = re.search(r"\*\*Current Maturity\*\*:\s*(M\d)", line)
        if mat_m:
            result["maturity"] = mat_m.group(1)
        fm_m = re.search(r"\*\*Forward Model\*\*:\s*(\S+)", line)
        if fm_m:
            result["forward_model_type"] = fm_m.group(1).strip()
        sol_m = re.search(r"\*\*Default Solver\*\*:\s*(\S+)", line)
        if sol_m:
            result["default_solver"] = sol_m.group(1).strip()

    # Mismatch Parameters table
    mismatch_params = []
    in_mismatch = False
    for line in lines:
        if "### Mismatch Parameters" in line:
            in_mismatch = True
            continue
        if in_mismatch:
            if line.startswith("###") or line.startswith("---"):
                break
            if "|" in line and not line.strip().startswith("|---"):
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p]
                if len(parts) >= 4 and parts[0] not in ("Parameter", "---"):
                    name = parts[0]
                    # Parse nominal
                    nominal_str = parts[1].rstrip("%")
                    try:
                        nominal = float(nominal_str)
                    except ValueError:
                        nominal = 0.0
                    # Parse range
                    range_str = parts[2] if len(parts) > 2 else "[0, 0]"
                    range_m = re.findall(r"[-+]?\d*\.?\d+", range_str)
                    if len(range_m) >= 2:
                        rng = sorted([float(range_m[0]), float(range_m[1])])
                    else:
                        rng = [0, 0]
                    # Unit
                    unit = parts[-1] if len(parts) >= 4 else ""
                    mismatch_params.append({
                        "name": name,
                        "nominal": nominal,
                        "range": rng,
                        "unit": unit,
                    })
    result["mismatch_params"] = mismatch_params

    # Expected PSNR range from Solvers & Expected Performance
    for line in lines:
        psnr_m = re.search(r"Scenario I PSNR\*\*:\s*([\d.]+)\s*-\s*([\d.]+)", line)
        if psnr_m:
            result["expected_psnr_range"] = [
                float(psnr_m.group(1)), float(psnr_m.group(2))
            ]

    return result


# ---------------------------------------------------------------------------
# Registry loaders
# ---------------------------------------------------------------------------

def load_modalities_yaml() -> Dict[str, Dict]:
    path = CONTRIB_DIR / "modalities.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("modalities", {})


def load_graph_templates() -> Dict[str, Dict]:
    path = CONTRIB_DIR / "graph_templates.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("templates", {})


def load_solver_registry() -> Dict[str, Dict]:
    path = CONTRIB_DIR / "solver_registry.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_dataset_registry() -> Dict[str, Dict]:
    path = CONTRIB_DIR / "dataset_registry.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("datasets", {})


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(
    spec: Dict[str, Any],
    modalities_yaml: Dict[str, Dict],
    templates: Dict[str, Dict],
    solvers: Dict[str, Dict],
    datasets: Dict[str, Dict],
) -> Dict[str, Any]:
    """Build a complete benchmark config dict for one modality."""
    mid = spec["modality_id"]

    # Check registries
    has_yaml = mid in modalities_yaml
    has_template = any(t.startswith(f"{mid}_graph_") for t in templates)
    has_solver = mid in solvers

    # Tier classification
    if has_yaml and has_template:
        tier = "A"
    elif has_yaml or has_template:
        tier = "B"
    else:
        tier = "C"

    # Get signal dims from modalities.yaml if available
    mod_info = modalities_yaml.get(mid, {})
    x_dims = mod_info.get("signal_dims", {}).get("x", [64, 64])
    y_dims = mod_info.get("signal_dims", {}).get("y", [64, 64])
    if isinstance(x_dims, dict):
        x_dims = [x_dims.get("H", 64), x_dims.get("W", 64)]
    if isinstance(y_dims, dict):
        y_dims = [y_dims.get("H", 64), y_dims.get("W", 64)]

    # Category module
    dag = spec.get("canonical_dag", "")
    category = spec.get("category", "")
    cat_module = DAG_TO_CATEGORY.get(dag, "")
    if not cat_module:
        cat_module = CATEGORY_KEYWORD_MAP.get(category, "microscopy_psf")

    # Graph template ID
    template_id = ""
    for t in templates:
        if t.startswith(f"{mid}_graph_"):
            template_id = t
            break

    # Synthetic generator
    gen = DAG_TO_GENERATOR.get(dag, "shepp_logan")

    # Dataset lookup
    dataset_id = ""
    dataset_url = ""
    dataset_citation = ""
    dataset_license = ""
    for did, dinfo in datasets.items():
        if mid in dinfo.get("modalities", []):
            dataset_id = did
            dataset_url = dinfo.get("download", {}).get("url", "")
            dataset_citation = dinfo.get("citation", "")
            dataset_license = dinfo.get("license", "")
            break

    # Solver configs
    solver_configs = {}
    if has_solver:
        solver_data = solvers[mid]
        for tier_name in ("traditional_cpu", "best_quality", "famous_dl", "small_gpu"):
            if tier_name in solver_data:
                s = solver_data[tier_name]
                solver_configs[tier_name] = {
                    "name": s.get("name", tier_name),
                    "module": s.get("module", ""),
                    "function": s.get("function", ""),
                    "params": str(s.get("params", "0")),
                    "gpu": s.get("gpu", False),
                    "reference": s.get("reference", ""),
                }

    # Build theta from modalities.yaml elements
    theta = {}
    if mod_info:
        for elem in mod_info.get("elements", []):
            for k, v in elem.get("parameters", {}).items():
                theta[k] = v

    # Source attribution
    source_attr = {
        "ground_truth": {
            "type": "web" if dataset_url else "generated",
            "reference": dataset_citation or "Synthetic phantom",
        },
        "forward_model": {
            "type": "registry" if template_id else "paper",
            "reference": f"pwm graph_templates.yaml / {template_id}" if template_id else dag,
        },
        "solver": {
            "type": "registry" if has_solver else "generated",
            "reference": "pwm solver_registry.yaml" if has_solver else "adjoint fallback",
        },
        "mismatch_ranges": {
            "type": "registry",
            "reference": "pwm modality_benchmarks spec",
        },
    }

    # Data source priority
    if dataset_url:
        priority = ["web", "experimental", "synthetic_web", "generated"]
    elif has_yaml:
        priority = ["experimental", "synthetic_web", "generated"]
    else:
        priority = ["generated"]

    config = {
        "modality_id": mid,
        "display_name": spec.get("display_name", mid),
        "category": category,
        "canonical_dag": dag,
        "carrier": spec.get("carrier", "Photon"),
        "maturity": spec.get("maturity", "M0"),
        "forward_model_type": spec.get("forward_model_type", "linear_operator"),
        "default_solver": spec.get("default_solver", ""),
        "tier": tier,
        "x_shape": list(x_dims),
        "y_shape": list(y_dims),
        "operator_id": mid,
        "has_dedicated_operator": has_yaml,
        "graph_template_id": template_id,
        "theta": theta,
        "assets": {},
        "category_module": cat_module,
        "data_source": {
            "priority": priority,
            "dataset_id": dataset_id,
            "dataset_url": dataset_url,
            "fallback": "generated",
            "synthetic_generator": gen,
            "citation": dataset_citation,
            "license": dataset_license,
        },
        "mismatch_params": spec.get("mismatch_params", []),
        "solvers": solver_configs,
        "metrics": {
            "names": ["psnr", "ssim"],
            "primary": "psnr",
            "thresholds": {},
        },
        "reference_psnr": None,
        "expected_psnr_range": spec.get("expected_psnr_range"),
        "source_attribution": source_attr,
    }

    # Add SAM for spectral modalities
    if len(x_dims) >= 3 or dag in ("M --> W --> Sigma --> D", "M --> W --> D"):
        config["metrics"]["names"].append("sam")

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 168 benchmark YAML configs")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not write")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load registries
    print("Loading registries...")
    modalities_yaml = load_modalities_yaml()
    templates = load_graph_templates()
    solver_reg = load_solver_registry()
    dataset_reg = load_dataset_registry()
    print(f"  modalities.yaml: {len(modalities_yaml)} entries")
    print(f"  graph_templates.yaml: {len(templates)} entries")
    print(f"  solver_registry.yaml: {len(solver_reg)} entries")
    print(f"  dataset_registry.yaml: {len(dataset_reg)} entries")

    # Parse all markdown specs
    spec_files = sorted(SPECS_DIR.glob("*.md"))
    spec_files = [f for f in spec_files if f.name != "README.md"]
    print(f"\nParsing {len(spec_files)} markdown specs...")

    specs = []
    for path in spec_files:
        try:
            spec = parse_markdown_spec(path)
            specs.append(spec)
            if args.verbose:
                print(f"  {spec['modality_id']}: {spec.get('category', '?')} | "
                      f"{spec.get('canonical_dag', '?')}")
        except Exception as e:
            print(f"  ERROR parsing {path.name}: {e}")

    # Build and write configs
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    tier_counts = {"A": 0, "B": 0, "C": 0}
    written = 0

    for spec in specs:
        config = build_config(spec, modalities_yaml, templates, solver_reg, dataset_reg)
        tier_counts[config["tier"]] += 1

        if not args.dry_run:
            out_path = CONFIGS_DIR / f"{config['modality_id']}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                          allow_unicode=True, width=120)
            written += 1

    print(f"\nGenerated {written} configs ({len(specs)} parsed)")
    print(f"  Tier A (operator+YAML): {tier_counts['A']}")
    print(f"  Tier B (YAML or template): {tier_counts['B']}")
    print(f"  Tier C (spec only): {tier_counts['C']}")

    if args.dry_run:
        print("\n[dry-run] No files written.")


if __name__ == "__main__":
    main()
