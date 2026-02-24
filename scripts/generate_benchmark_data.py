#!/usr/bin/env python3
"""Generate benchmark data files for all 65 variants.

Creates 4 files per variant (B1 JSON, B2 HDF5, B3 tar.gz, B4 HDF5) under
platform/pwm_platform/static/benchmark-data/v1.0/.

Existing files are skipped unless --overwrite is passed.

Usage:
    python scripts/generate_benchmark_data.py
    python scripts/generate_benchmark_data.py --overwrite
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile

import h5py
import numpy as np

# Ensure platform package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform"))

from pwm_platform.services.benchmark_database import VARIANT_DATABASE

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "platform",
    "pwm_platform",
    "static",
    "benchmark-data",
    "v1.0",
)

# ── Distractor specs (used across all B1 datasets) ───────────────────────────

DISTRACTOR_POOL = [
    "\u03a0(fan) \u2192 D(g, \u03b7\u2082)",
    "F(spiral) \u2192 C(psf) \u2192 D(g, \u03b7\u2083)",
    "P(fresnel) \u2192 |\u00b7|\u00b2 \u2192 D(g, \u03b7\u2081)",
    "M(\u03a6) \u2192 \u03a3 \u2192 D(g, \u03b7\u2081)",
    "S(grating) \u2192 C(PSF) \u2192 \u03a3 \u2192 D(g, \u03b7\u2083)",
    "R(\u03b8) \u2192 \u03a0(cone) \u2192 D(g, \u03b7\u2081)",
    "C(PSF) \u2192 D(g, \u03b7\u2083)",
    "P(acoustic) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2082)",
    "\u039b(E\u2081,E\u2082) \u2192 \u03a0(proj) \u2192 D(g, \u03b7\u2081)",
    "F(EPI) \u2192 \u03a3_t \u2192 D(g, \u03b7\u2081)",
    "P(e\u207b) \u2192 C(CTF) \u2192 D(g, \u03b7\u2081)",
    "\u03a0(ray) \u2192 \u03a3(volume) \u2192 D(g, \u03b7\u2081)",
    "P(modulated) \u2192 \u03a3(corr) \u2192 D(g, \u03b7\u2081)",
    "M(polarizer) \u2192 C(PSF) \u2192 D(g, \u03b7\u2081)",
    "P(diffuser) \u2192 D(g, \u03b7\u2081)",
]

# ── Description templates for B1 by forward model family ─────────────────────

_DESC_TEMPLATES = {
    "Pi": [
        "An imaging system that projects a {scene_type} through geometric {proj_type} projection onto a detector array.",
        "A {scene_type} is captured by projecting through {proj_type} geometry, collapsing 3D structure onto a 2D sensor.",
        "This system images {scene_type} by casting rays through the object and measuring attenuation on a flat-panel detector.",
    ],
    "F": [
        "An imaging system that acquires data by sampling in the Fourier domain ({detail}), then reconstructing via inverse transform.",
        "This modality samples {detail} in the frequency domain, encoding spatial information into k-space measurements.",
        "Data is collected by traversing k-space trajectories ({detail}), requiring Fourier inversion for image reconstruction.",
    ],
    "C": [
        "A microscopy system where the image is formed by convolving the specimen with a point-spread function ({detail}).",
        "This {detail} microscope images samples through PSF convolution, with diffraction-limited resolution.",
        "An optical system that blurs the specimen through convolution with a {detail} PSF before detection.",
    ],
    "P": [
        "An imaging system based on wave propagation ({detail}), where measurements encode the object's interaction with the propagating field.",
        "This modality captures data by propagating a {detail} wave through/around the specimen and recording the resulting field.",
        "Measurements are formed by {detail} wave propagation and detection of the resulting intensity or phase.",
    ],
    "S": [
        "This system uses structured illumination patterns ({detail}) to modulate the specimen before detection.",
        "A patterned illumination ({detail}) is projected onto the sample, and multiple acquisitions are combined to reconstruct.",
    ],
    "R": [
        "A tomographic imaging system that rotates the {detail} to acquire projections from multiple angles.",
        "This modality acquires data by rotating {detail} and capturing projections at each angle for 3D reconstruction.",
    ],
    "M": [
        "An imaging system that modulates the signal through a {detail} before detection.",
        "This system applies a {detail} modulation pattern to encode the signal prior to measurement.",
    ],
}


def _get_description_template(variant: dict, idx: int) -> str:
    """Generate a natural-language description for a B1 sample."""
    dag = variant["spec_dag"]
    first_prim = dag[0]["primitive"]
    display = variant["display_name"]
    full = variant["full_name"]
    detail = dag[0].get("label", dag[0].get("params", ""))

    templates = _DESC_TEMPLATES.get(first_prim, _DESC_TEMPLATES["P"])
    tmpl = templates[idx % len(templates)]

    # Fill template
    desc = tmpl.format(
        scene_type=full.lower(),
        proj_type=detail.lower(),
        detail=detail,
    )
    return desc


def _pick_distractors(correct_spec: str, rng: np.random.Generator, n: int = 3) -> list[str]:
    """Pick n distractors that don't match the correct spec."""
    candidates = [d for d in DISTRACTOR_POOL if d != correct_spec]
    chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    return chosen.tolist()


# ── Shape configs by category ────────────────────────────────────────────────

_SHAPE_CONFIG = {
    "compressive":              {"x_shape": (64, 64), "y_factor": 0.25},
    "medical":                  {"x_shape": (64, 64, 32), "y_factor": 0.5},
    "medical_ultrasound":       {"x_shape": (64, 64), "y_factor": 0.5},
    "coherent":                 {"x_shape": (64, 64), "y_factor": 1.0},
    "microscopy":               {"x_shape": (64, 64), "y_factor": 1.0},
    "electron_microscopy":      {"x_shape": (64, 64), "y_factor": 1.0},
    "clinical_optics":          {"x_shape": (64, 64, 32), "y_factor": 0.5},
    "computational":            {"x_shape": (64, 64), "y_factor": 0.5},
    "computational_photography": {"x_shape": (64, 64), "y_factor": 1.0},
    "neural_rendering":         {"x_shape": (64, 64, 3), "y_factor": 0.5},
    "depth_imaging":            {"x_shape": (64, 64), "y_factor": 0.5},
    "remote_sensing":           {"x_shape": (64, 64), "y_factor": 1.0},
    "particle_imaging":         {"x_shape": (64, 64, 32), "y_factor": 0.5},
}


def _get_y_shape(x_shape: tuple, y_factor: float) -> tuple:
    """Compute measurement shape from signal shape and compression factor."""
    total = 1
    for s in x_shape:
        total *= s
    y_size = max(16, int(total * y_factor))
    return (y_size,)


# ── File generators ──────────────────────────────────────────────────────────


def generate_b1(key: str, variant: dict, path: str, rng: np.random.Generator):
    """Generate B1 JSON — 50 NL descriptions + correct spec DAGs + distractors."""
    samples = []
    for i in range(50):
        desc = _get_description_template(variant, i)
        # Add variation
        if i % 5 == 1:
            desc = f"Consider a {variant['full_name'].lower()} system: {desc}"
        elif i % 5 == 2:
            desc = f"{desc} The system operates in a {variant['category'].replace('_', ' ')} context."
        elif i % 5 == 3:
            desc = f"A researcher sets up {variant['full_name'].lower()}. {desc}"
        elif i % 5 == 4:
            desc = f"In a typical {variant['category'].replace('_', ' ')} lab, {desc.lower()}"

        samples.append({
            "id": i + 1,
            "description": desc,
            "correct_spec": variant["spec_notation"],
            "correct_dag": variant["spec_dag"],
            "distractors": _pick_distractors(variant["spec_notation"], rng),
        })

    data = {
        "benchmark": "Benchmark 1",
        "variant": key,
        "task": "spec_selection",
        "metric": "spec_match_accuracy",
        "num_samples": 50,
        "samples": samples,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_b2(key: str, variant: dict, path: str, rng: np.random.Generator):
    """Generate B2 HDF5 — synthetic x_true + measurements y."""
    category = variant["category"]
    cfg = _SHAPE_CONFIG.get(category, _SHAPE_CONFIG["compressive"])
    n_samples = variant["benchmarks"][1]["public_dataset"]["num_samples"]

    x_shape = (n_samples,) + cfg["x_shape"]
    y_shape_per = _get_y_shape(cfg["x_shape"], cfg["y_factor"])
    y_shape = (n_samples,) + y_shape_per

    x_true = rng.standard_normal(x_shape).astype(np.float32)
    y = rng.standard_normal(y_shape).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("x_true", data=x_true, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=y, compression="gzip", compression_opts=4)


def generate_b3(key: str, variant: dict, path: str, rng: np.random.Generator):
    """Generate B3 tar.gz — metadata.json + measurements.h5."""
    # Build metadata with 50 spec validation triplets
    samples = []
    for i in range(50):
        if rng.random() < 0.5:
            # Match case
            samples.append({
                "id": i + 1,
                "candidate_spec": variant["spec_notation"],
                "label": "match",
                "true_spec": variant["spec_notation"],
            })
        else:
            # No-match case — pick a distractor as candidate
            distractors = _pick_distractors(variant["spec_notation"], rng, n=1)
            samples.append({
                "id": i + 1,
                "candidate_spec": distractors[0] if distractors else variant["spec_notation"],
                "label": "no_match",
                "true_spec": variant["spec_notation"],
            })

    metadata = {
        "benchmark": "Benchmark 3",
        "variant": key,
        "task": "spec_validation",
        "metric": "classification_accuracy",
        "num_samples": 50,
        "samples": samples,
    }

    # Build small measurements HDF5 in memory
    category = variant["category"]
    cfg = _SHAPE_CONFIG.get(category, _SHAPE_CONFIG["compressive"])
    y_shape_per = _get_y_shape(cfg["x_shape"], cfg["y_factor"])
    y_data = rng.standard_normal((50,) + y_shape_per).astype(np.float32)

    h5_buf = io.BytesIO()
    with h5py.File(h5_buf, "w") as hf:
        hf.create_dataset("y", data=y_data, compression="gzip", compression_opts=4)
    h5_bytes = h5_buf.getvalue()

    # Write tar.gz
    with tarfile.open(path, "w:gz") as tar:
        # metadata.json
        md_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")
        md_info = tarfile.TarInfo(name="metadata.json")
        md_info.size = len(md_bytes)
        tar.addfile(md_info, io.BytesIO(md_bytes))

        # measurements.h5
        h5_info = tarfile.TarInfo(name="measurements.h5")
        h5_info.size = len(h5_bytes)
        tar.addfile(h5_info, io.BytesIO(h5_bytes))


def generate_b4(key: str, variant: dict, path: str, rng: np.random.Generator):
    """Generate B4 HDF5 — drifted measurements + ground truth."""
    category = variant["category"]
    cfg = _SHAPE_CONFIG.get(category, _SHAPE_CONFIG["compressive"])
    n_samples = variant["benchmarks"][3]["public_dataset"]["num_samples"]

    x_shape = (n_samples,) + cfg["x_shape"]
    y_shape_per = _get_y_shape(cfg["x_shape"], cfg["y_factor"])
    y_shape = (n_samples,) + y_shape_per

    x_true = rng.standard_normal(x_shape).astype(np.float32)
    y_drifted = rng.standard_normal(y_shape).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("x_true", data=x_true, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=y_drifted, compression="gzip", compression_opts=4)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark data files for all variants.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    generated = 0
    skipped = 0

    for key, variant in VARIANT_DATABASE.items():
        files = {
            "b1": os.path.join(OUT_DIR, f"{key}_b1_public.json"),
            "b2": os.path.join(OUT_DIR, f"{key}_b2_public.h5"),
            "b3": os.path.join(OUT_DIR, f"{key}_b3_public.tar.gz"),
            "b4": os.path.join(OUT_DIR, f"{key}_b4_public.h5"),
        }

        generators = {
            "b1": generate_b1,
            "b2": generate_b2,
            "b3": generate_b3,
            "b4": generate_b4,
        }

        for bm_key, path in files.items():
            if os.path.exists(path) and not args.overwrite:
                skipped += 1
                continue

            print(f"  Generating {key}_{bm_key}...", end=" ", flush=True)
            generators[bm_key](key, variant, path, rng)
            size_kb = os.path.getsize(path) / 1024
            print(f"{size_kb:.0f} KB")
            generated += 1

    print(f"\nDone: {generated} files generated, {skipped} skipped (already exist).")
    total = len(VARIANT_DATABASE) * 4
    print(f"Total: {total} files for {len(VARIANT_DATABASE)} variants.")


if __name__ == "__main__":
    main()
