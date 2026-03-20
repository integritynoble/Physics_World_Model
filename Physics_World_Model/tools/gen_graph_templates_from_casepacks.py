#!/usr/bin/env python3
"""gen_graph_templates_from_casepacks.py

Scaffold graph_templates.yaml entries by reading existing operators and
modality definitions.

Usage
-----
    python tools/gen_graph_templates_from_casepacks.py

This script:
1. Reads packages/pwm_core/contrib/modalities.yaml for the 26 modality keys.
2. Reads packages/pwm_core/contrib/primitives.yaml for available primitives.
3. Inspects existing operator classes under packages/pwm_core/pwm_core/physics/.
4. Generates a scaffold YAML for each modality that does not already have a
   template in graph_templates.yaml.

Output is written to stdout as valid YAML. Redirect to a file to save:

    python tools/gen_graph_templates_from_casepacks.py > new_templates.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONTRIB = ROOT / "packages" / "pwm_core" / "contrib"
MODALITIES_PATH = CONTRIB / "modalities.yaml"
PRIMITIVES_PATH = CONTRIB / "primitives.yaml"
TEMPLATES_PATH = CONTRIB / "graph_templates.yaml"
PHYSICS_DIR = ROOT / "packages" / "pwm_core" / "pwm_core" / "physics"

# ---------------------------------------------------------------------------
# Modality -> primitive mapping heuristics
# ---------------------------------------------------------------------------

CATEGORY_PRIMITIVES: Dict[str, List[Dict[str, Any]]] = {
    "microscopy": [
        {"node_id": "blur", "primitive_id": "conv2d", "params": {"sigma": 2.0, "mode": "reflect"}},
        {"node_id": "noise", "primitive_id": "poisson_gaussian", "params": {"peak_photons": 10000.0, "read_sigma": 0.01, "seed": 0}},
    ],
    "compressive": [
        {"node_id": "mask", "primitive_id": "coded_mask", "params": {"seed": 42, "H": 64, "W": 64}},
        {"node_id": "noise", "primitive_id": "gaussian", "params": {"sigma": 0.01, "seed": 0}},
    ],
    "spectral": [
        {"node_id": "modulate", "primitive_id": "coded_mask", "params": {"seed": 42, "H": 64, "W": 64}},
        {"node_id": "disperse", "primitive_id": "spectral_dispersion", "params": {"disp_step": 1.0}},
        {"node_id": "integrate", "primitive_id": "frame_integration", "params": {"axis": -1, "T": 8}},
        {"node_id": "noise", "primitive_id": "poisson_gaussian", "params": {"peak_photons": 10000.0, "read_sigma": 0.01, "seed": 0}},
    ],
    "tomography": [
        {"node_id": "project", "primitive_id": "ct_radon", "params": {"n_angles": 180, "H": 64, "W": 64}},
        {"node_id": "noise", "primitive_id": "poisson_gaussian", "params": {"peak_photons": 100000.0, "read_sigma": 0.005, "seed": 0}},
    ],
    "mri": [
        {"node_id": "kspace", "primitive_id": "mri_kspace", "params": {"H": 64, "W": 64, "sampling_rate": 0.25, "seed": 42}},
        {"node_id": "noise", "primitive_id": "gaussian", "params": {"sigma": 0.005, "seed": 0}},
    ],
    "coherent": [
        {"node_id": "propagate", "primitive_id": "fresnel_prop", "params": {"wavelength": 0.5e-6, "distance": 1.0e-3, "pixel_size": 1.0e-6}},
        {"node_id": "detect", "primitive_id": "magnitude_sq", "params": {}},
        {"node_id": "noise", "primitive_id": "poisson", "params": {"peak_photons": 10000.0, "seed": 0}},
    ],
    "default": [
        {"node_id": "forward", "primitive_id": "identity", "params": {}},
        {"node_id": "noise", "primitive_id": "gaussian", "params": {"sigma": 0.01, "seed": 0}},
    ],
}

MODALITY_CATEGORY_MAP: Dict[str, str] = {
    "widefield": "microscopy",
    "widefield_lowdose": "microscopy",
    "confocal_livecell": "microscopy",
    "confocal_3d": "microscopy",
    "sim": "microscopy",
    "lightsheet": "microscopy",
    "cassi": "spectral",
    "spc": "compressive",
    "cacti": "compressive",
    "matrix": "compressive",
    "ct": "tomography",
    "mri": "mri",
    "ptychography": "coherent",
    "holography": "coherent",
    "nerf": "default",
    "gaussian_splatting": "default",
    "lensless": "microscopy",
    "panorama": "default",
    "light_field": "default",
    "dot": "microscopy",
    "photoacoustic": "default",
    "oct": "default",
    "flim": "microscopy",
    "fpm": "coherent",
    "phase_retrieval": "coherent",
    "integral": "default",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its content."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_existing_template_ids(templates_data: Dict[str, Any]) -> set:
    """Extract existing template graph_ids."""
    templates = templates_data.get("templates", {})
    return set(templates.keys())


def build_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build a linear chain of edges from a node list."""
    edges = []
    for i in range(len(nodes) - 1):
        edges.append({
            "source": nodes[i]["node_id"],
            "target": nodes[i + 1]["node_id"],
        })
    return edges


def scaffold_template(modality_key: str, modality_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a scaffold graph template for a modality."""
    category = MODALITY_CATEGORY_MAP.get(modality_key, "default")
    nodes = CATEGORY_PRIMITIVES.get(category, CATEGORY_PRIMITIVES["default"])

    # Extract signal dims if available
    signal_dims = modality_info.get("signal_dims", {})
    x_shape = signal_dims.get("x", [64, 64])
    y_shape = signal_dims.get("y", x_shape)

    template = {
        "description": f"Auto-scaffolded template for {modality_key}",
        "metadata": {
            "modality": modality_key,
            "x_shape": x_shape,
            "y_shape": y_shape,
        },
        "nodes": nodes,
        "edges": build_edges(nodes),
    }
    return template


def main() -> None:
    """Main entry point."""
    # Load existing data
    modalities_data = load_yaml(MODALITIES_PATH)
    modality_entries = modalities_data.get("modalities", {})

    # Load existing templates to avoid duplicates
    existing_ids: set = set()
    if TEMPLATES_PATH.exists():
        templates_data = load_yaml(TEMPLATES_PATH)
        existing_ids = get_existing_template_ids(templates_data)

    # Generate scaffolds for missing modalities
    new_templates: Dict[str, Any] = {}
    for modality_key, modality_info in modality_entries.items():
        template_id = f"{modality_key}_graph_v1"
        if template_id in existing_ids:
            print(
                f"# SKIP: {template_id} already exists in graph_templates.yaml",
                file=sys.stderr,
            )
            continue
        new_templates[template_id] = scaffold_template(
            modality_key, modality_info or {}
        )

    if not new_templates:
        print("# All 26 modalities already have templates.", file=sys.stderr)
        print("# No new templates to generate.", file=sys.stderr)
        return

    # Output as YAML
    output = {
        "version": "1.0",
        "templates": new_templates,
    }
    print("# Auto-generated graph template scaffolds")
    print("# Review and customize before merging into graph_templates.yaml")
    print("#")
    yaml.dump(output, sys.stdout, default_flow_style=False, sort_keys=False)

    print(
        f"\n# Generated {len(new_templates)} new template(s).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
