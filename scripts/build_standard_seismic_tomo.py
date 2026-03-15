"""
build_standard_seismic_tomo.py
Build standard seismic tomography dataset for PWM benchmark.

x_true : (70,70) float32 [0,1] — 2-D velocity model (Marmousi-style)
y_ideal : (70,70) float32       — seismic shot gather via ray-based Radon
                                   (70 shots x 70 receivers, flattened/projected)

Source / citation:
  OpenFWI benchmark, Deng et al. IEEE TGRS 2021,
  doi:10.1109/TGRS.2021.3124783  (synthetic)
"""

import sys
import json
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "benchmark" / "seismic_tomo" / "standard"
MODALITY = "seismic_tomo"
N_SAMPLES = 10
NX = 70
NZ = 70
N_SHOTS = 70
N_RECV = 70


# ---------------------------------------------------------------------------
# Velocity-model generator
# ---------------------------------------------------------------------------

def _make_velocity_model(rng):
    """Return (NZ, NX) float32 slowness model in [0,1].

    Layers with sinusoidal perturbations + optional salt-dome anomaly.
    """
    model = np.zeros((NZ, NX), dtype=np.float32)
    x_arr = np.linspace(0, 1, NX)

    # Number of geological layers
    n_layers = rng.integers(3, 7)
    layer_bounds = np.sort(rng.uniform(0.05, 0.95, n_layers - 1))
    layer_vels = rng.uniform(0.2, 0.9, n_layers).astype(np.float32)

    for iz in range(NZ):
        depth = iz / NZ  # normalised depth [0,1]
        layer_idx = np.searchsorted(layer_bounds, depth)
        v_base = layer_vels[layer_idx]
        # Sine-wave lateral perturbation
        freq = rng.uniform(1.5, 4.0)
        amp = rng.uniform(0.0, 0.08)
        model[iz, :] = v_base + amp * np.sin(2 * np.pi * freq * x_arr).astype(np.float32)

    # Optional salt-dome anomaly (high-velocity blob)
    if rng.random() > 0.4:
        cx = rng.integers(15, NX - 15)
        cz = rng.integers(15, NZ - 15)
        rx = rng.integers(5, 15)
        rz = rng.integers(4, 10)
        Z_idx, X_idx = np.ogrid[:NZ, :NX]
        mask = ((X_idx - cx) / rx) ** 2 + ((Z_idx - cz) / rz) ** 2 <= 1
        model[mask] = np.clip(model[mask] + rng.uniform(0.2, 0.5), 0, 1)

    # Normalise
    vmin, vmax = model.min(), model.max()
    if vmax > vmin:
        model = (model - vmin) / (vmax - vmin)
    return model.astype(np.float32)


# ---------------------------------------------------------------------------
# Simplified ray-based forward model
# ---------------------------------------------------------------------------

def _forward_radon(vel_model):
    """Compute shot gather via ray-based travel-time Radon.

    Slowness s = 1/(v+eps); Radon gives integrated slowness (travel time)
    for each (shot, receiver) pair through the velocity model.
    Result shape: (N_SHOTS, N_RECV) -> returned as (NZ, NX) by reshape.
    """
    nz, nx = vel_model.shape
    slowness = 1.0 / (vel_model + 1e-6)

    shots = np.linspace(0, nx - 1, N_SHOTS).astype(int)
    recvs = np.linspace(0, nx - 1, N_RECV).astype(int)

    gather = np.zeros((N_SHOTS, N_RECV), dtype=np.float32)

    for si, sx in enumerate(shots):
        for ri, rx in enumerate(recvs):
            # Straight-ray path from surface (z=0) at sx to surface at rx
            # Parametric path through model, integrating slowness
            n_steps = max(nz, abs(int(rx) - int(sx)) + 1) * 2
            px = np.round(np.linspace(sx, rx, n_steps)).astype(int)
            pz = np.round(np.linspace(0, nz - 1, n_steps)).astype(int)
            px = np.clip(px, 0, nx - 1)
            pz = np.clip(pz, 0, nz - 1)
            ds = np.sqrt(((rx - sx) / (n_steps - 1)) ** 2 + ((nz - 1) / (n_steps - 1)) ** 2)
            gather[si, ri] = float(slowness[pz, px].sum() * ds)

    # Normalise to [0,1]
    gmin, gmax = gather.min(), gather.max()
    if gmax > gmin:
        gather = (gather - gmin) / (gmax - gmin)
    return gather.astype(np.float32)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[seismic_tomo] Output directory: {OUT_DIR}")

    metadata_list = []

    for idx in range(N_SAMPLES):
        rng = np.random.default_rng(seed=idx + 42)
        sample_id = f"sample_{idx:04d}"
        h5_path = OUT_DIR / f"{sample_id}.h5"

        print(f"  [{idx+1}/{N_SAMPLES}] Building {sample_id} ...")

        x_true = _make_velocity_model(rng)   # (70,70)
        y_ideal = _forward_radon(x_true)     # (N_SHOTS, N_RECV) == (70,70)

        assert x_true.shape == (NZ, NX)
        assert y_ideal.shape == (N_SHOTS, N_RECV)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x_true", data=x_true, compression="gzip")
            hf.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = hf.create_group("H_params")
            hp.attrs["n_shots"] = N_SHOTS
            hp.attrs["n_receivers"] = N_RECV
            hp.attrs["nx"] = NX
            hp.attrs["nz"] = NZ
            hp.attrs["forward_model"] = "Ray-based straight-ray Radon (travel time)"
            hp.attrs["velocity_parameterisation"] = "normalised [0,1] slowness"

            meta = hf.create_group("metadata")
            meta.attrs["modality"] = MODALITY
            meta.attrs["sample_id"] = sample_id
            meta.attrs["x_shape"] = str([NZ, NX])
            meta.attrs["y_shape"] = str([N_SHOTS, N_RECV])
            meta.attrs["seed"] = idx + 42
            meta.attrs["created"] = datetime.utcnow().isoformat()
            meta.attrs["source"] = "Synthetic Marmousi-style geological model"
            meta.attrs["reference"] = "Deng et al. IEEE TGRS 2021 doi:10.1109/TGRS.2021.3124783"

        metadata_list.append({
            "sample_id": sample_id,
            "h5_file": f"{sample_id}.h5",
            "x_shape": [NZ, NX],
            "y_shape": [N_SHOTS, N_RECV],
            "seed": idx + 42,
        })

    meta_doc = {
        "modality": MODALITY,
        "n_samples": N_SAMPLES,
        "split": "standard",
        "created": datetime.utcnow().isoformat(),
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
        "source": "Synthetic Marmousi-style velocity models with layering and salt domes",
        "reference": "Deng et al. IEEE TGRS 2021, doi:10.1109/TGRS.2021.3124783",
        "samples": metadata_list,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_doc, f, indent=2)

    spec_doc = {
        "modality": MODALITY,
        "x_description": "2-D seismic velocity model, shape (70,70), float32 [0,1]",
        "y_description": "Seismic shot gather (travel-time Radon), shape (70,70), float32 [0,1]",
        "forward_model": "Straight-ray travel-time integration (Radon of slowness 1/v)",
        "n_shots": N_SHOTS,
        "n_receivers": N_RECV,
        "x_shape": [NZ, NX],
        "y_shape": [N_SHOTS, N_RECV],
        "dtype": "float32",
        "normalisation": "[0,1] for both x_true and y_ideal",
        "reconstruction_algorithms": ["FWI gradient descent", "adjoint-state method",
                                      "TV-regularised inversion", "InversionNet"],
        "reference": "Deng et al. IEEE TGRS 2021, doi:10.1109/TGRS.2021.3124783",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
    }
    with open(OUT_DIR / "spec.json", "w") as f:
        json.dump(spec_doc, f, indent=2)

    print(f"[seismic_tomo] Done. {N_SAMPLES} samples written to {OUT_DIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
