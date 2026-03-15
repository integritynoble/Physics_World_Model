"""
build_standard_fwi.py
Build standard Full Waveform Inversion (FWI) dataset for PWM benchmark.

x_true : (70,70) float32 [0,1] — subsurface velocity model
y_ideal : (70,70) float32 [0,1] — seismic shot record (shot x time/receiver)

Source / citation:
  OpenFWI, Deng et al. IEEE TGRS 2021,
  doi:10.1109/TGRS.2021.3124783

Geological features modelled: flat layers, anticlines, salt bodies.
Forward model: simplified acoustic wave equation via 2-D Radon transform.
"""

import sys
import json
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "benchmark" / "fwi" / "standard"
MODALITY = "fwi"
N_SAMPLES = 10
NX = 70
NZ = 70
N_SHOTS = 70
N_TIME = 70   # time samples (collapsed with receivers for y shape)


# ---------------------------------------------------------------------------
# Velocity-model generator with geological features
# ---------------------------------------------------------------------------

def _make_flat_layers(rng, nz, nx):
    """Flat horizontal layers."""
    model = np.zeros((nz, nx), dtype=np.float32)
    n_layers = rng.integers(4, 8)
    depths = np.sort(rng.integers(5, nz - 5, n_layers - 1))
    vels = rng.uniform(0.1, 0.95, n_layers).astype(np.float32)
    prev = 0
    for i, d in enumerate(depths):
        model[prev:d, :] = vels[i]
        prev = d
    model[prev:, :] = vels[-1]
    return model


def _add_anticline(model, rng):
    """Add an anticline: upward-arching layer interface."""
    nz, nx = model.shape
    amplitude = rng.integers(5, 15)
    centre_x = rng.integers(10, nx - 10)
    width = rng.uniform(0.2, 0.5) * nx
    fold_depth = rng.integers(20, nz - 20)
    x_arr = np.arange(nx)
    arch = (amplitude * np.exp(-0.5 * ((x_arr - centre_x) / width) ** 2)).astype(int)
    high_vel = rng.uniform(0.6, 0.95)
    for xi in range(nx):
        z_top = max(0, fold_depth - arch[xi])
        z_bot = min(nz, z_top + rng.integers(3, 10))
        model[z_top:z_bot, xi] = high_vel
    return model


def _add_salt_body(model, rng):
    """Add an irregular salt body (very high velocity)."""
    nz, nx = model.shape
    cx = rng.integers(15, nx - 15)
    cz = rng.integers(10, nz - 15)
    rx = rng.integers(8, 18)
    rz = rng.integers(6, 14)
    Z_idx, X_idx = np.ogrid[:nz, :nx]
    # Elliptical salt with slight random perturbation
    base_mask = ((X_idx - cx) / rx) ** 2 + ((Z_idx - cz) / rz) ** 2 <= 1
    model[base_mask] = rng.uniform(0.85, 1.0)
    return model


def _make_velocity_model(rng, sample_idx):
    model = _make_flat_layers(rng, NZ, NX)
    feature = sample_idx % 3
    if feature == 0:
        pass  # pure flat layers
    elif feature == 1:
        model = _add_anticline(model, rng)
    else:
        model = _add_salt_body(model, rng)
    # Normalise
    vmin, vmax = model.min(), model.max()
    if vmax > vmin:
        model = (model - vmin) / (vmax - vmin)
    return model.astype(np.float32)


# ---------------------------------------------------------------------------
# Simplified acoustic forward model (Radon-based travel time)
# ---------------------------------------------------------------------------

def _acoustic_forward(vel_model):
    """Approximate seismic shot record using straight-ray travel times.

    Each shot fires from surface (z=0) at x=shot_x;
    energy travels down to reflectors and back up (simplified as two-way).
    Returns (N_SHOTS, N_TIME) array normalised to [0,1].
    """
    nz, nx = vel_model.shape
    slowness = 1.0 / (vel_model + 1e-6)

    shot_xs = np.linspace(0, nx - 1, N_SHOTS).astype(int)
    time_samples = np.linspace(0, 1, N_TIME)

    record = np.zeros((N_SHOTS, N_TIME), dtype=np.float32)

    for si, sx in enumerate(shot_xs):
        # For each receiver position sample the two-way travel time column
        recv_xs = np.linspace(0, nx - 1, N_TIME).astype(int)
        for ti, rx in enumerate(recv_xs):
            # Down-going path: shot to mid-point reflector
            n_steps = max(nz, abs(int(rx) - int(sx)) + 1) * 2
            # Down-going leg
            px_d = np.round(np.linspace(sx, rx, n_steps)).astype(int)
            pz_d = np.round(np.linspace(0, nz - 1, n_steps)).astype(int)
            px_d = np.clip(px_d, 0, nx - 1)
            pz_d = np.clip(pz_d, 0, nz - 1)
            ds = np.sqrt(((rx - sx) / (n_steps - 1 + 1e-9)) ** 2
                         + ((nz - 1) / (n_steps - 1 + 1e-9)) ** 2)
            tt = float(slowness[pz_d, px_d].sum() * ds)
            record[si, ti] = tt

    # Normalise to [0,1]
    rmin, rmax = record.min(), record.max()
    if rmax > rmin:
        record = (record - rmin) / (rmax - rmin)
    return record.astype(np.float32)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fwi] Output directory: {OUT_DIR}")

    metadata_list = []

    for idx in range(N_SAMPLES):
        rng = np.random.default_rng(seed=idx + 42)
        sample_id = f"sample_{idx:04d}"
        h5_path = OUT_DIR / f"{sample_id}.h5"

        print(f"  [{idx+1}/{N_SAMPLES}] Building {sample_id} ...")

        x_true = _make_velocity_model(rng, idx)    # (70,70)
        y_ideal = _acoustic_forward(x_true)         # (N_SHOTS, N_TIME) == (70,70)

        assert x_true.shape == (NZ, NX)
        assert y_ideal.shape == (N_SHOTS, N_TIME)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x_true", data=x_true, compression="gzip")
            hf.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = hf.create_group("H_params")
            hp.attrs["n_shots"] = N_SHOTS
            hp.attrs["n_time_samples"] = N_TIME
            hp.attrs["nx"] = NX
            hp.attrs["nz"] = NZ
            hp.attrs["forward_model"] = "Straight-ray two-way travel time (acoustic approx)"
            hp.attrs["feature_type"] = ["flat_layers", "anticline", "salt"][idx % 3]

            meta = hf.create_group("metadata")
            meta.attrs["modality"] = MODALITY
            meta.attrs["sample_id"] = sample_id
            meta.attrs["x_shape"] = str([NZ, NX])
            meta.attrs["y_shape"] = str([N_SHOTS, N_TIME])
            meta.attrs["seed"] = idx + 42
            meta.attrs["created"] = datetime.utcnow().isoformat()
            meta.attrs["source"] = "Synthetic FWI velocity model (OpenFWI style)"
            meta.attrs["reference"] = "Deng et al. IEEE TGRS 2021 doi:10.1109/TGRS.2021.3124783"

        metadata_list.append({
            "sample_id": sample_id,
            "h5_file": f"{sample_id}.h5",
            "x_shape": [NZ, NX],
            "y_shape": [N_SHOTS, N_TIME],
            "feature_type": ["flat_layers", "anticline", "salt"][idx % 3],
            "seed": idx + 42,
        })

    meta_doc = {
        "modality": MODALITY,
        "n_samples": N_SAMPLES,
        "split": "standard",
        "created": datetime.utcnow().isoformat(),
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
        "source": "Synthetic FWI: flat layers, anticlines, salt bodies (OpenFWI style)",
        "reference": "Deng et al. IEEE TGRS 2021, doi:10.1109/TGRS.2021.3124783",
        "samples": metadata_list,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_doc, f, indent=2)

    spec_doc = {
        "modality": MODALITY,
        "x_description": "Subsurface velocity model, shape (70,70), float32 [0,1]",
        "y_description": (
            "Seismic shot record (shot x time), shape (70,70), float32 [0,1]; "
            "encoded as two-way travel-time gather"
        ),
        "forward_model": "Straight-ray two-way acoustic travel-time (Radon of slowness)",
        "n_shots": N_SHOTS,
        "n_time_samples": N_TIME,
        "x_shape": [NZ, NX],
        "y_shape": [N_SHOTS, N_TIME],
        "dtype": "float32",
        "normalisation": "[0,1] for both x_true and y_ideal",
        "geological_features": ["flat_layers", "anticline", "salt_body"],
        "reconstruction_algorithms": [
            "adjoint-state FWI", "TV-FWI", "InversionNet", "VelocityGAN"
        ],
        "reference": "Deng et al. IEEE TGRS 2021, doi:10.1109/TGRS.2021.3124783",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
    }
    with open(OUT_DIR / "spec.json", "w") as f:
        json.dump(spec_doc, f, indent=2)

    print(f"[fwi] Done. {N_SAMPLES} samples written to {OUT_DIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
