"""
build_standard_pet.py
Build standard PET (Positron Emission Tomography) dataset for PWM benchmark.

x_true : (128,128,16) float32 [0,1] — 3-D activity map (16 axial slices)
y_ideal : (180,128,16) float32 — PET sinogram (Radon of each 2-D slice)

Source / citation:
  TCIA LIDC-IDRI PET, Clark et al. Sci.Data 2013,
  doi:10.1038/sdata.2015.17
  Phantom: Shepp-Logan with simulated FDG hot-spot variations.
"""

import sys
import json
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "benchmark" / "pet" / "standard"
MODALITY = "pet"
N_SAMPLES = 10
N_PIX = 128
N_SLICES = 16
N_ANGLES = 180
N_DET = 128


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _shepp_logan_2d(npix=128):
    params = [
        [1.0, 0.69, 0.92, 0.0, 0.0, 0.0],
        [-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0],
        [-0.2, 0.11, 0.31, 0.22, 0.0, -18.0],
        [-0.2, 0.16, 0.41, -0.22, 0.0, 18.0],
        [0.1, 0.21, 0.25, 0.0, 0.35, 0.0],
        [0.1, 0.046, 0.046, 0.0, 0.1, 0.0],
        [0.1, 0.046, 0.046, 0.0, -0.1, 0.0],
        [-0.02, 0.046, 0.023, -0.08, -0.605, 0.0],
        [-0.02, 0.023, 0.023, 0.0, -0.606, 0.0],
        [-0.02, 0.023, 0.046, 0.06, -0.605, 0.0],
    ]
    phantom = np.zeros((npix, npix), dtype=np.float32)
    x_arr = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(x_arr, x_arr)
    for p in params:
        A, a, b, x0, y0, phi = p
        phi_rad = np.deg2rad(phi)
        cp, sp = np.cos(phi_rad), np.sin(phi_rad)
        Xr = cp * (X - x0) + sp * (Y - y0)
        Yr = -sp * (X - x0) + cp * (Y - y0)
        phantom[(Xr / a) ** 2 + (Yr / b) ** 2 <= 1] += A
    return np.clip(phantom, 0, None).astype(np.float32)


def _radon_2d(image, n_angles=180, n_det=128):
    npix = image.shape[0]
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    t = np.linspace(-1, 1, n_det)
    sino = np.zeros((n_angles, n_det), dtype=np.float32)
    x_arr = np.linspace(-1, 1, npix)
    y_arr = np.linspace(-1, 1, npix)
    X, Y = np.meshgrid(x_arr, y_arr)
    psize = 2.0 / npix
    for i, theta in enumerate(angles):
        ct, st = np.cos(theta), np.sin(theta)
        t_px = ct * X + st * Y
        t_idx = ((t_px - t[0]) / (t[-1] - t[0]) * (n_det - 1)).astype(int)
        t_idx = np.clip(t_idx, 0, n_det - 1)
        np.add.at(sino[i], t_idx.ravel(), image.ravel() * psize)
    return sino


def _make_activity_volume(rng, sample_idx):
    """Return (N_PIX, N_PIX, N_SLICES) float32 activity map in [0,1]."""
    base = _shepp_logan_2d(N_PIX)

    # Add FDG hot spots that vary per sample
    n_hotspots = rng.integers(3, 8)
    hot = np.zeros((N_PIX, N_PIX), dtype=np.float32)
    x_arr = np.linspace(-1, 1, N_PIX)
    X, Y = np.meshgrid(x_arr, x_arr)
    for _ in range(n_hotspots):
        cx = rng.uniform(-0.6, 0.6)
        cy = rng.uniform(-0.6, 0.6)
        r = rng.uniform(0.04, 0.12)
        intensity = rng.uniform(0.3, 1.0)
        hot += intensity * ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2).astype(np.float32)

    base_plus = base + 0.5 * hot

    # Stack axial slices with slight per-slice attenuation envelope
    volume = np.zeros((N_PIX, N_PIX, N_SLICES), dtype=np.float32)
    for z in range(N_SLICES):
        # Gaussian axial envelope centred at mid-slice
        z_norm = (z - N_SLICES / 2.0) / (N_SLICES / 4.0)
        envelope = float(np.exp(-0.5 * z_norm ** 2))
        slice_noise = rng.normal(0, 0.02, (N_PIX, N_PIX)).astype(np.float32)
        volume[:, :, z] = np.clip(base_plus * envelope + slice_noise, 0, None)

    # Normalise to [0,1]
    vmax = volume.max()
    if vmax > 0:
        volume /= vmax
    return volume


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[pet] Output directory: {OUT_DIR}")

    metadata_list = []

    for idx in range(N_SAMPLES):
        rng = np.random.default_rng(seed=idx + 42)
        sample_id = f"sample_{idx:04d}"
        h5_path = OUT_DIR / f"{sample_id}.h5"

        print(f"  [{idx+1}/{N_SAMPLES}] Building {sample_id} ...")

        x_true = _make_activity_volume(rng, idx)  # (128,128,16)
        assert x_true.shape == (N_PIX, N_PIX, N_SLICES)

        # Radon each axial slice
        y_ideal = np.zeros((N_ANGLES, N_DET, N_SLICES), dtype=np.float32)
        for z in range(N_SLICES):
            y_ideal[:, :, z] = _radon_2d(x_true[:, :, z], N_ANGLES, N_DET)

        # Normalise sinogram to [0,1]
        ymax = y_ideal.max()
        if ymax > 0:
            y_ideal /= ymax

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x_true", data=x_true, compression="gzip")
            hf.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = hf.create_group("H_params")
            hp.attrs["n_angles"] = N_ANGLES
            hp.attrs["n_detectors"] = N_DET
            hp.attrs["n_slices"] = N_SLICES
            hp.attrs["angle_range_deg"] = "0-180"
            hp.attrs["detector_range"] = "-1 to 1"
            hp.attrs["forward_model"] = "Radon transform (per axial slice)"

            meta = hf.create_group("metadata")
            meta.attrs["modality"] = MODALITY
            meta.attrs["sample_id"] = sample_id
            meta.attrs["x_shape"] = str(list(x_true.shape))
            meta.attrs["y_shape"] = str(list(y_ideal.shape))
            meta.attrs["seed"] = idx + 42
            meta.attrs["created"] = datetime.utcnow().isoformat()
            meta.attrs["source"] = "Synthetic Shepp-Logan + FDG hot spots"
            meta.attrs["reference"] = "Clark et al. Sci.Data 2013 doi:10.1038/sdata.2015.17"

        metadata_list.append({
            "sample_id": sample_id,
            "h5_file": f"{sample_id}.h5",
            "x_shape": list(x_true.shape),
            "y_shape": list(y_ideal.shape),
            "seed": idx + 42,
        })

    # metadata.json
    meta_doc = {
        "modality": MODALITY,
        "n_samples": N_SAMPLES,
        "split": "standard",
        "created": datetime.utcnow().isoformat(),
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
        "source": "Synthetic Shepp-Logan phantom with FDG hot-spot variations",
        "reference": "Clark et al. Sci.Data 2013, doi:10.1038/sdata.2015.17",
        "samples": metadata_list,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_doc, f, indent=2)

    # spec.json
    spec_doc = {
        "modality": MODALITY,
        "x_description": "3-D PET activity map, shape (128,128,16), float32 [0,1]",
        "y_description": "PET sinogram per axial slice, shape (180,128,16), float32 [0,1]",
        "forward_model": "Radon transform of each 2-D axial slice; 180 angles, 128 detectors",
        "n_angles": N_ANGLES,
        "n_detectors": N_DET,
        "n_slices": N_SLICES,
        "x_shape": [N_PIX, N_PIX, N_SLICES],
        "y_shape": [N_ANGLES, N_DET, N_SLICES],
        "dtype": "float32",
        "normalisation": "[0,1] for both x_true and y_ideal",
        "reconstruction_algorithms": ["FBP", "MLEM", "OSEM", "TV-regularised MLEM"],
        "reference": "Clark et al. Sci.Data 2013, doi:10.1038/sdata.2015.17",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
    }
    with open(OUT_DIR / "spec.json", "w") as f:
        json.dump(spec_doc, f, indent=2)

    print(f"[pet] Done. {N_SAMPLES} samples written to {OUT_DIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
