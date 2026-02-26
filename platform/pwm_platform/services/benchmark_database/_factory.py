"""Factory — expands compact registry entries into full VARIANT_DATABASE entries."""

from __future__ import annotations

# ── Category-specific dataset defaults ────────────────────────────────────────

_CATEGORY_DEFAULTS: dict[str, dict] = {
    "compressive": {
        "b2_samples": 5, "b2_size": 5, "b4_samples": 5, "b4_size": 5, "b3_size": 3,
        "data_type": "test images (64\u00d764)",
    },
    "medical": {
        "b2_samples": 3, "b2_size": 50, "b4_samples": 3, "b4_size": 50, "b3_size": 15,
        "data_type": "volumetric scans (128\u00d7128\u00d764)",
    },
    "medical_ultrasound": {
        "b2_samples": 3, "b2_size": 40, "b4_samples": 3, "b4_size": 40, "b3_size": 12,
        "data_type": "ultrasound frames (256\u00d7256)",
    },
    "coherent": {
        "b2_samples": 5, "b2_size": 25, "b4_samples": 5, "b4_size": 25, "b3_size": 10,
        "data_type": "complex-valued fields (256\u00d7256)",
    },
    "microscopy": {
        "b2_samples": 5, "b2_size": 20, "b4_samples": 5, "b4_size": 20, "b3_size": 8,
        "data_type": "microscopy images (256\u00d7256)",
    },
    "electron_microscopy": {
        "b2_samples": 3, "b2_size": 15, "b4_samples": 3, "b4_size": 15, "b3_size": 6,
        "data_type": "micrographs (512\u00d7512)",
    },
    "clinical_optics": {
        "b2_samples": 3, "b2_size": 30, "b4_samples": 3, "b4_size": 30, "b3_size": 10,
        "data_type": "optical scans (256\u00d7256)",
    },
    "computational": {
        "b2_samples": 5, "b2_size": 20, "b4_samples": 5, "b4_size": 20, "b3_size": 8,
        "data_type": "multi-view images (128\u00d7128)",
    },
    "computational_photography": {
        "b2_samples": 5, "b2_size": 20, "b4_samples": 5, "b4_size": 20, "b3_size": 8,
        "data_type": "captured images (256\u00d7256)",
    },
    "neural_rendering": {
        "b2_samples": 5, "b2_size": 50, "b4_samples": 5, "b4_size": 50, "b3_size": 15,
        "data_type": "multi-view image sets (800\u00d7800\u00d750 views)",
    },
    "depth_imaging": {
        "b2_samples": 5, "b2_size": 15, "b4_samples": 5, "b4_size": 15, "b3_size": 6,
        "data_type": "depth maps (256\u00d7256)",
    },
    "remote_sensing": {
        "b2_samples": 3, "b2_size": 40, "b4_samples": 3, "b4_size": 40, "b3_size": 12,
        "data_type": "SAR / sonar images (512\u00d7512)",
    },
    "particle_imaging": {
        "b2_samples": 3, "b2_size": 30, "b4_samples": 3, "b4_size": 30, "b3_size": 10,
        "data_type": "tomographic slices (128\u00d7128\u00d764)",
    },
}

_DEFAULT_FALLBACK = {
    "b2_samples": 5, "b2_size": 20, "b4_samples": 5, "b4_size": 20, "b3_size": 8,
    "data_type": "images (256\u00d7256)",
}


def _get_ds_config(category: str, overrides: dict | None = None) -> dict:
    """Merge category defaults with optional per-variant overrides."""
    cfg = _CATEGORY_DEFAULTS.get(category, _DEFAULT_FALLBACK).copy()
    if overrides:
        cfg.update(overrides)
    return cfg


# ── Benchmark builders ────────────────────────────────────────────────────────

def _make_b1(key: str, display: str, leaderboard: list) -> dict:
    return {
        "id": "Benchmark 1",
        "title": "Benchmark 1 \u2014 LLM Spec Router",
        "description": "Given a natural-language modality description, the LLM selects the correct spec from the primitive library. Scored on spec-match accuracy.",
        "input": "Natural-language modality description",
        "output": "Selected spec (primitive DAG)",
        "has_public_dataset": True,
        "has_hidden_dataset": True,
        "public_dataset": {
            "name": f"{display} Benchmark 1 Spec Prompts (Public)",
            "description": "50 natural-language imaging system descriptions paired with their correct spec DAGs from the primitive library.",
            "format": "JSON",
            "num_samples": 50,
            "size_mb": 0.1,
            "gcs_object_path": f"benchmark-data/v1.0/{key}_b1_public.json",
            "download_url": None,
        },
        "hidden_dataset": {
            "name": f"{display} Benchmark 1 Spec Prompts (Hidden)",
            "description": "100 held-out natural-language descriptions used for server-side evaluation of spec-match accuracy.",
            "format": "JSON",
            "num_samples": 100,
            "note": "Hidden test set \u2014 used for leaderboard scoring only.",
        },
        "leaderboard": leaderboard,
        "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
        "credits": None,
    }


def _make_b2(key: str, display: str, ds: dict, leaderboard: list, scenario_table: dict | None = None) -> dict:
    result = {
        "id": "Benchmark 2",
        "title": "Benchmark 2 \u2014 Algorithm Correction + Reconstruction",
        "description": "Given a mismatched forward model and measurements, the algorithm corrects the spec and reconstructs the signal. Scored on PSNR / SSIM.",
        "input": "Measurements y, mismatched forward model H\u0303",
        "output": "Reconstructed signal x\u0302",
        "has_public_dataset": True,
        "has_hidden_dataset": True,
        "public_dataset": {
            "name": f"{display} Benchmark 2 Recon Data (Public)",
            "description": f"{ds['b2_samples']} {ds['data_type']} with measurements y, mismatched forward model H\u0303, and ground-truth signal x.",
            "format": "HDF5",
            "num_samples": ds["b2_samples"],
            "size_mb": ds["b2_size"],
            "gcs_object_path": f"benchmark-data/v1.0/{key}_b2_public.h5",
            "download_url": None,
        },
        "hidden_dataset": {
            "name": f"{display} Benchmark 2 Recon Data (Hidden)",
            "description": f"Remaining {ds['data_type']} held out for server-side PSNR/SSIM evaluation.",
            "format": "HDF5",
            "num_samples": ds["b2_samples"],
            "note": "Hidden test set \u2014 used for leaderboard scoring only.",
        },
        "leaderboard": leaderboard,
        "links": {
            "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
            "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
        },
        "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
    }
    if scenario_table:
        result["scenario_table"] = scenario_table
    return result


def _make_b3(key: str, display: str, ds: dict, leaderboard: list) -> dict:
    return {
        "id": "Benchmark 3",
        "title": "Benchmark 3 \u2014 LLM Spec Router (with ground-truth)",
        "description": "Given real measurements y, true forward model H, and a candidate spec, the LLM evaluates whether the spec matches the true system. Scored on classification accuracy.",
        "input": "Measurements y, true forward model H, candidate spec",
        "output": "Match / no-match classification",
        "has_public_dataset": True,
        "has_hidden_dataset": True,
        "public_dataset": {
            "name": f"{display} Benchmark 3 Spec Validation (Public)",
            "description": "50 triplets of {measurements y, true forward model H, candidate spec} with match/no-match labels for spec validation.",
            "format": "JSON + HDF5",
            "num_samples": 50,
            "size_mb": ds["b3_size"],
            "gcs_object_path": f"benchmark-data/v1.0/{key}_b3_public.tar.gz",
            "download_url": None,
        },
        "hidden_dataset": {
            "name": f"{display} Benchmark 3 Spec Validation (Hidden)",
            "description": "100 held-out triplets used for server-side evaluation of spec classification accuracy.",
            "format": "JSON + HDF5",
            "num_samples": 100,
            "note": "Hidden test set \u2014 used for leaderboard scoring only.",
        },
        "leaderboard": leaderboard,
        "links": {"contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-spec.md"},
        "credits": None,
    }


def _make_b4(key: str, display: str, ds: dict, leaderboard: list, scenario_table: dict | None = None) -> dict:
    result = {
        "id": "Benchmark 4",
        "title": "Benchmark 4 \u2014 Algorithm Reconstruction (with drift)",
        "description": "Given measurements from a drifted system (low-scoring B3 specs), the algorithm reconstructs the signal. Tests robustness to forward-model drift.",
        "input": "Measurements y, drifted forward model H\u0302",
        "output": "Reconstructed signal x\u0302",
        "has_public_dataset": True,
        "has_hidden_dataset": True,
        "public_dataset": {
            "name": f"{display} Benchmark 4 Drift Recon Data (Public)",
            "description": f"{ds['b4_samples']} {ds['data_type']} with measurements from drifted system, drifted forward model H\u0302, and ground-truth signal x.",
            "format": "HDF5",
            "num_samples": ds["b4_samples"],
            "size_mb": ds["b4_size"],
            "gcs_object_path": f"benchmark-data/v1.0/{key}_b4_public.h5",
            "download_url": None,
        },
        "hidden_dataset": {
            "name": f"{display} Benchmark 4 Drift Recon Data (Hidden)",
            "description": f"Remaining drifted {ds['data_type']} held out for server-side PSNR/SSIM evaluation.",
            "format": "HDF5",
            "num_samples": ds["b4_samples"],
            "note": "Hidden test set \u2014 used for leaderboard scoring only.",
        },
        "leaderboard": leaderboard,
        "links": {
            "contribute": "https://github.com/InverseNet/benchmark-data/issues/new?template=contribute-dataset.md",
            "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
        },
        "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
    }
    if scenario_table:
        result["scenario_table"] = scenario_table
    return result


# ── Public factory ────────────────────────────────────────────────────────────

def build_variant(key: str, registry_entry: dict, leaderboard: dict | None = None) -> dict:
    """Expand a compact registry entry into a full VARIANT_DATABASE entry with 4 benchmarks."""
    display = registry_entry["display_name"]
    category = registry_entry["category"]
    ds = _get_ds_config(category, registry_entry.get("dataset_config"))
    lb = leaderboard or {}

    return {
        "display_name": display,
        "full_name": registry_entry["full_name"],
        "parent_modality": registry_entry["parent_modality"],
        "category": category,
        "spec_notation": registry_entry["spec_notation"],
        "spec_dag": registry_entry["spec_dag"],
        "mismatch_params": registry_entry["mismatch_params"],
        "benchmarks": [
            _make_b1(key, display, lb.get("b1", [])),
            _make_b2(key, display, ds, lb.get("b2", []), scenario_table=lb.get("b2_scenario_table")),
            _make_b3(key, display, ds, lb.get("b3", [])),
            _make_b4(key, display, ds, lb.get("b4", []), scenario_table=lb.get("b4_scenario_table")),
        ],
        "credits_config": {
            "profit_pool_pct": 40,
            "winner_share_pct": 30,
            "min_withdrawal_usd": 100,
        },
    }
