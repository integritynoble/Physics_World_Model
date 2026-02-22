#!/usr/bin/env python3
"""Generate tiny synthetic test datasets for LIP-Arena evaluation.

Creates 32x32 (or equivalent small) .npy datasets for each target modality
using the PWM graph compiler infrastructure. These datasets are used by
``scripts/verify_targeting_system.py`` and
``tests/test_targeting_harness_e2e.py`` to validate the end-to-end pipeline.

Usage:
    python datasets/lip_arena/generate.py            # generate all
    python datasets/lip_arena/generate.py cassi mri   # generate specific
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure pwm_core is importable
_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo / "packages" / "pwm_core"))

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec

# ---------------------------------------------------------------------------
# Target modalities
# ---------------------------------------------------------------------------

VALIDATED = ["cassi", "cacti", "spc", "ct", "mri", "ptychography", "lensless"]
REPRESENTATIVE = [
    "widefield", "sim", "holography", "oct", "pet", "nerf", "ultrasound",
]
ALL_TARGETS = VALIDATED + REPRESENTATIVE

# Fallback modalities that have graph templates (pet, ultrasound, nerf may
# not have templates -- we gracefully skip those).
FALLBACK_SIMPLE = {
    "pet": {"x_shape": (32, 32), "description": "PET-like blur+noise"},
    "ultrasound": {"x_shape": (32, 32), "description": "Ultrasound-like blur"},
}


def _load_graph_templates() -> Dict[str, Any]:
    import yaml

    path = _repo / "packages" / "pwm_core" / "contrib" / "graph_templates.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("templates", data)


def _generate_scene(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Generate a small synthetic scene (normalised to [0, 1])."""
    x = rng.standard_normal(shape).astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    return x


def _sandbox_spec(
    template: Dict[str, Any],
    template_id: str,
    max_dim: int = 32,
) -> Tuple[OperatorGraphSpec, Tuple[int, ...]]:
    """Build a sandbox-sized OperatorGraphSpec from a YAML template."""
    allowed_keys = {"graph_id", "nodes", "edges", "metadata"}
    spec_data = {"graph_id": template_id}
    for k, v in template.items():
        if k in allowed_keys:
            spec_data[k] = v

    # Strip noise nodes so operator is linear
    if "nodes" in spec_data:
        noise_ids = {
            n["node_id"]
            for n in spec_data["nodes"]
            if n.get("role") == "noise"
            or n.get("primitive_id", "").startswith("noise")
            or "noise" in n.get("node_id", "").lower()
        }
        if noise_ids:
            spec_data["nodes"] = [
                n for n in spec_data["nodes"] if n["node_id"] not in noise_ids
            ]
            if "edges" in spec_data:
                spec_data["edges"] = [
                    e
                    for e in spec_data["edges"]
                    if e.get("source") not in noise_ids
                    and e.get("target") not in noise_ids
                ]

    spec = GraphCompiler.from_dict(spec_data)

    # Determine sandbox x_shape
    orig_x = tuple(spec.metadata.get("x_shape", [64, 64]))
    sandbox_x = tuple(min(d, max_dim) for d in orig_x)

    # Propagate sandbox spatial dims to primitive params
    sb_h = sandbox_x[0]
    sb_w = sandbox_x[1] if len(sandbox_x) > 1 else sandbox_x[0]
    for node in spec.nodes:
        if node.params:
            if "H" in node.params:
                node.params["H"] = sb_h
            if "W" in node.params:
                node.params["W"] = sb_w

    return spec, sandbox_x


def generate_modality(
    modality: str,
    templates: Dict[str, Any],
    out_dir: Path,
    seed: int = 42,
    max_dim: int = 32,
) -> Optional[Path]:
    """Generate dataset for one modality. Returns output path or None on skip."""
    rng = np.random.default_rng(seed)

    # Find template
    template_id = f"{modality}_graph_v1"
    if template_id not in templates:
        candidates = [k for k in templates if k.startswith(modality)]
        if candidates:
            template_id = candidates[0]
        else:
            # Fallback: generate simple blurred data
            if modality in FALLBACK_SIMPLE:
                return _generate_fallback(modality, out_dir, rng)
            print(f"  SKIP {modality}: no graph template found")
            return None

    template = templates[template_id]

    try:
        spec, x_shape = _sandbox_spec(template, template_id, max_dim)
        y_shape = tuple(spec.metadata.get("y_shape", list(x_shape)))
        sandbox_y = tuple(min(d, max_dim) for d in y_shape)

        compiler = GraphCompiler()
        op = compiler.compile(spec, x_shape=x_shape, y_shape=sandbox_y)

        x_gt = _generate_scene(x_shape, rng)
        y = op.forward(x_gt)
    except Exception as e:
        print(f"  FALLBACK {modality}: compile/forward failed ({e})")
        return _generate_fallback(modality, out_dir, rng, x_shape=(max_dim, max_dim))

    # Save
    mod_dir = out_dir / modality
    mod_dir.mkdir(parents=True, exist_ok=True)

    np.save(mod_dir / "x_gt.npy", x_gt.astype(np.float32))

    # Preserve complex dtype for modalities like MRI/holography
    if np.iscomplexobj(y):
        y_save = y.astype(np.complex64)
        y_dtype = "complex64"
        y_range = [float(np.abs(y).min()), float(np.abs(y).max())]
    else:
        y_save = y.astype(np.float32)
        y_dtype = "float32"
        y_range = [float(y.min()), float(y.max())]
    np.save(mod_dir / "y.npy", y_save)

    metadata = {
        "modality": modality,
        "template_id": template_id,
        "x_shape": list(x_gt.shape),
        "y_shape": list(y.shape),
        "seed": seed,
        "x_dtype": "float32",
        "y_dtype": y_dtype,
        "x_range": [float(x_gt.min()), float(x_gt.max())],
        "y_range": y_range,
    }
    with open(mod_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    x_kb = x_gt.nbytes / 1024
    y_kb = y.nbytes / 1024
    print(f"  OK {modality}: x={list(x_gt.shape)} ({x_kb:.1f}KB), y={list(y.shape)} ({y_kb:.1f}KB)")
    return mod_dir


def _generate_fallback(
    modality: str,
    out_dir: Path,
    rng: np.random.Generator,
    x_shape: Tuple[int, ...] = (32, 32),
) -> Path:
    """Generate a simple blurred fallback dataset."""
    from scipy.ndimage import gaussian_filter

    x_gt = _generate_scene(x_shape, rng)
    y = gaussian_filter(x_gt, sigma=2.0)
    # Add mild Gaussian noise
    y = y + rng.normal(0, 0.01, y.shape)

    mod_dir = out_dir / modality
    mod_dir.mkdir(parents=True, exist_ok=True)

    np.save(mod_dir / "x_gt.npy", x_gt.astype(np.float32))
    np.save(mod_dir / "y.npy", y.astype(np.float32))

    metadata = {
        "modality": modality,
        "template_id": f"{modality}_fallback",
        "x_shape": list(x_gt.shape),
        "y_shape": list(y.shape),
        "seed": 42,
        "x_dtype": "float32",
        "y_dtype": "float32",
        "fallback": True,
    }
    with open(mod_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    x_kb = x_gt.nbytes / 1024
    y_kb = y.nbytes / 1024
    print(f"  OK {modality} (fallback): x={list(x_gt.shape)} ({x_kb:.1f}KB), y={list(y.shape)} ({y_kb:.1f}KB)")
    return mod_dir


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ALL_TARGETS
    out_dir = Path(__file__).resolve().parent

    print(f"Generating LIP-Arena test datasets in {out_dir}")
    print(f"Targets: {targets}\n")

    templates = _load_graph_templates()
    results = {}
    total_bytes = 0

    for modality in targets:
        result = generate_modality(modality, templates, out_dir)
        if result is not None:
            results[modality] = str(result)
            for f in result.iterdir():
                if f.suffix == ".npy":
                    total_bytes += f.stat().st_size

    print(f"\nDone: {len(results)}/{len(targets)} modalities generated")
    print(f"Total .npy size: {total_bytes / 1024:.1f} KB ({total_bytes / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
