"""Generate bootstrap proposal templates from similar modalities."""
from __future__ import annotations


def generate_operator_graph_template(similar: list, all_modalities: dict) -> dict:
    top = similar[:2]
    primitives = []
    seen: set = set()
    for match in top:
        mod = all_modalities.get(match["modality_id"])
        if mod:
            for p in (getattr(mod, 'primitives', None) or []):
                if p not in seen:
                    primitives.append(p)
                    seen.add(p)
    return {
        "type": "OperatorGraphTemplate",
        "source": "bootstrap_generated",
        "primitives": primitives[:8],
        "derived_from": [m["modality_id"] for m in top],
    }


def generate_experiment_spec_template(name: str, similar: list, all_modalities: dict) -> dict:
    best_id = similar[0]["modality_id"] if similar else "unknown"
    return {
        "version": "0.2.1",
        "id": f"bootstrap_{name.lower().replace(' ', '_')}_v1",
        "input": {"mode": "simulate"},
        "states": {
            "physics": {"modality": name.lower().replace(" ", "_")},
            "sensor": {"shot_noise": {"enabled": True}, "read_noise_sigma": 5.0},
            "task": {"kind": "simulate_recon_analyze"},
        },
        "recon": {"portfolio": {"solvers": [{"id": "admm", "params": {}}]}},
        "_bootstrap_note": f"Template derived from {best_id}",
    }


def generate_sim_dataset_plan(similar: list, all_modalities: dict) -> dict:
    return {
        "num_samples_recommended": 5000,
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        "format": "zarr",
        "spatial_dims": [256, 256],
        "derived_from": [m["modality_id"] for m in similar[:3]],
    }


def generate_real_data_checklist(physics_class: str) -> list:
    base = [
        "Document system setup with photos and diagrams",
        "Record all hardware specs (sensor, source wavelength, geometry)",
        "Measure background/dark frames (N>=50)",
        "Collect calibration target data",
        "Record at least 20 independent samples",
        "Document SNR estimate per sample",
        "Export raw data as .zarr or .h5",
        "Compute and verify SHA-256 checksums",
    ]
    if physics_class == "coherent":
        base += ["Measure coherence length", "Record phase reference if applicable"]
    elif physics_class == "tomographic":
        base += ["Record all projection angles", "Measure geometric calibration phantom"]
    return base


def generate_viability_checklist() -> list:
    return [
        "At least one forward operator primitive identified",
        "Forward model (A: x->y) validated on synthetic data",
        "At least one reconstruction solver produces plausible output",
        "Noise model characterized (type + level)",
        "PSNR or equivalent metric computed on test set",
        "ExperimentSpec validated with pwm_core resolve_validate()",
        "Initial simulation dataset generated (>=1000 samples)",
        "Bootstrap proposal reviewed and approved by domain expert",
    ]
