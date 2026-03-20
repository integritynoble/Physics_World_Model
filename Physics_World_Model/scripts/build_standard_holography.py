"""
build_standard_holography.py
Build standard digital holography dataset for PWM benchmark.

x_true : (256,256,2) float32 — complex object stored as [amplitude, phase] channels
y_ideal : (256,256)   float32 [0,1] — hologram intensity |FT(x_complex)|^2

Source / citation:
  HoloPy simulation benchmark, NIST SRM,
  doi:10.18434/T4/1502702  (synthetic)
"""

import sys
import json
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "benchmark" / "holography" / "standard"
MODALITY = "holography"
N_SAMPLES = 10
NPIX = 256


# ---------------------------------------------------------------------------
# Synthetic object generator
# ---------------------------------------------------------------------------

def _make_complex_object(rng):
    """Return amplitude (NPIX,NPIX) and phase (NPIX,NPIX) arrays in [0,1] and [0,pi]."""
    amp = np.zeros((NPIX, NPIX), dtype=np.float32)
    phase = np.zeros((NPIX, NPIX), dtype=np.float32)

    x_arr = np.linspace(-1, 1, NPIX)
    X, Y = np.meshgrid(x_arr, x_arr)
    R2 = X ** 2 + Y ** 2

    # Background amplitude (soft disk)
    amp += 0.3 * np.exp(-R2 / 0.8).astype(np.float32)

    # Spherical scatterers (Airy-disk-like blobs)
    n_spheres = rng.integers(3, 8)
    for _ in range(n_spheres):
        cx = rng.uniform(-0.7, 0.7)
        cy = rng.uniform(-0.7, 0.7)
        r = rng.uniform(0.02, 0.10)
        intensity = rng.uniform(0.4, 1.0)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        blob = intensity * np.exp(-dist2 / (2 * r ** 2)).astype(np.float32)
        amp += blob

    # Phase object: smooth random phase (simulates refractive-index variation)
    phase_freqs = rng.integers(2, 6, size=(4, 2))
    for fx, fy in phase_freqs:
        coeff = rng.uniform(-0.5, 0.5)
        phase += coeff * np.cos(np.pi * fx * X + np.pi * fy * Y).astype(np.float32)

    # Normalise amplitude to [0,1], phase to [0, pi]
    amp = np.clip(amp, 0, None)
    amax = amp.max()
    if amax > 0:
        amp /= amax
    # Map phase to [0, pi]
    phase = phase - phase.min()
    pmax = phase.max()
    if pmax > 0:
        phase = phase / pmax * np.pi
    return amp.astype(np.float32), phase.astype(np.float32)


def _fraunhofer_hologram(amp, phase):
    """Compute Fraunhofer diffraction intensity |FT(x_complex)|^2.

    Returns real-valued intensity array (NPIX, NPIX), normalised to [0,1].
    """
    x_complex = amp * np.exp(1j * phase).astype(np.complex64)
    ft = np.fft.fftshift(np.fft.fft2(x_complex))
    intensity = (np.abs(ft) ** 2).astype(np.float32)
    imax = intensity.max()
    if imax > 0:
        intensity /= imax
    return intensity


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[holography] Output directory: {OUT_DIR}")

    metadata_list = []

    for idx in range(N_SAMPLES):
        rng = np.random.default_rng(seed=idx + 42)
        sample_id = f"sample_{idx:04d}"
        h5_path = OUT_DIR / f"{sample_id}.h5"

        print(f"  [{idx+1}/{N_SAMPLES}] Building {sample_id} ...")

        amp, phase = _make_complex_object(rng)
        # Stack as (NPIX, NPIX, 2): channel 0 = amplitude, channel 1 = phase
        x_true = np.stack([amp, phase], axis=-1).astype(np.float32)   # (256,256,2)
        y_ideal = _fraunhofer_hologram(amp, phase)                     # (256,256)

        assert x_true.shape == (NPIX, NPIX, 2)
        assert y_ideal.shape == (NPIX, NPIX)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x_true", data=x_true, compression="gzip")
            hf.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = hf.create_group("H_params")
            hp.attrs["forward_model"] = "Fraunhofer diffraction |FT(x_complex)|^2"
            hp.attrs["pixel_size_m"] = 6.5e-6
            hp.attrs["wavelength_m"] = 532e-9
            hp.attrs["propagation_distance_m"] = 0.1
            hp.attrs["x_channels"] = "0=amplitude, 1=phase_rad"
            hp.attrs["phase_range_rad"] = "0 to pi"

            meta = hf.create_group("metadata")
            meta.attrs["modality"] = MODALITY
            meta.attrs["sample_id"] = sample_id
            meta.attrs["x_shape"] = str(list(x_true.shape))
            meta.attrs["y_shape"] = str(list(y_ideal.shape))
            meta.attrs["seed"] = idx + 42
            meta.attrs["created"] = datetime.utcnow().isoformat()
            meta.attrs["source"] = "Synthetic spherical scatterers + phase objects"
            meta.attrs["reference"] = "HoloPy / NIST SRM doi:10.18434/T4/1502702"

        metadata_list.append({
            "sample_id": sample_id,
            "h5_file": f"{sample_id}.h5",
            "x_shape": list(x_true.shape),
            "y_shape": list(y_ideal.shape),
            "seed": idx + 42,
        })

    meta_doc = {
        "modality": MODALITY,
        "n_samples": N_SAMPLES,
        "split": "standard",
        "created": datetime.utcnow().isoformat(),
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
        "source": "Synthetic holography: spherical scatterers + smooth phase objects",
        "reference": "HoloPy / NIST SRM, doi:10.18434/T4/1502702",
        "samples": metadata_list,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_doc, f, indent=2)

    spec_doc = {
        "modality": MODALITY,
        "x_description": (
            "Complex object stored as (256,256,2) float32; "
            "channel 0 = amplitude [0,1], channel 1 = phase [0,pi] rad"
        ),
        "y_description": "Hologram intensity |FT(x_complex)|^2, shape (256,256), float32 [0,1]",
        "forward_model": "Fraunhofer diffraction: y = |FFT(amp * exp(i*phase))|^2",
        "x_shape": [NPIX, NPIX, 2],
        "y_shape": [NPIX, NPIX],
        "dtype": "float32",
        "normalisation": "amplitude [0,1], phase [0,pi], y_ideal [0,1]",
        "reconstruction_algorithms": [
            "Gerchberg-Saxton", "angular spectrum backpropagation",
            "phase retrieval (HIO)", "deep unfolding"
        ],
        "reference": "HoloPy / NIST SRM, doi:10.18434/T4/1502702",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
    }
    with open(OUT_DIR / "spec.json", "w") as f:
        json.dump(spec_doc, f, indent=2)

    print(f"[holography] Done. {N_SAMPLES} samples written to {OUT_DIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
