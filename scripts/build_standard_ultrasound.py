#!/usr/bin/env python3
"""
Build standard dataset for ultrasound modality.

Source: PICMUS (Plane-wave Imaging Challenge in Medical Ultrasound)
  Liebgott et al., IUS 2016; doi:10.1109/TUFFC.2019.2917338
  https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/

Standard dataset: 10 samples
  x_true  = tissue reflectivity map (256x128), float32 [0,1]
  y_ideal = RF data / IQ data: y = H * x_true (convolution with spatial PSF)
            Forward model: y[z,x] = PSF(z) * x[z,x] — column-wise convolution

Ultrasound forward model:
  y(z, x) = h_axial(z) * h_lateral(x) * x_true(z,x)
  where h_axial = Gaussian envelope (axial resolution ~lambda/2)
  and h_lateral = Gaussian beam profile

Output: datasets/benchmark/ultrasound/standard/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "ultrasound" / "standard"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Standard PICMUS parameters
N_DEPTH = 256    # axial samples
N_LATERAL = 128  # lateral samples (beams)


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _make_tissue_phantom(rng, n_depth=256, n_lateral=128, n_scatterers=2000):
    """
    Generate a tissue-mimicking phantom with random scatterers.
    Models: speckle-forming scatterers + cyst (anechoic) + bright targets
    """
    phantom = np.zeros((n_depth, n_lateral), dtype=np.float32)

    # Random scatterers (simulate speckle)
    z_pos = rng.uniform(10, n_depth - 10, n_scatterers).astype(int)
    x_pos = rng.uniform(5, n_lateral - 5, n_scatterers).astype(int)
    amp   = rng.rayleigh(0.5, n_scatterers).astype(np.float32)
    for z, x, a in zip(z_pos, x_pos, amp):
        phantom[z, x] += a

    # Anechoic cyst (dark region)
    n_cysts = rng.integers(1, 3)
    for _ in range(n_cysts):
        cz = int(rng.uniform(50, n_depth - 50))
        cx = int(rng.uniform(20, n_lateral - 20))
        r  = int(rng.uniform(10, 25))
        zz, xx = np.ogrid[0:n_depth, 0:n_lateral]
        cyst_mask = (zz - cz)**2 + (xx - cx)**2 < r**2
        phantom[cyst_mask] = 0  # no scatterers in cyst

    # Bright wire-like targets (point targets)
    n_pts = rng.integers(3, 8)
    for _ in range(n_pts):
        pz = int(rng.uniform(20, n_depth - 20))
        px = int(rng.uniform(5, n_lateral - 5))
        phantom[pz, px] += 5.0

    return _norm01(phantom)


def _us_forward_model(phantom: np.ndarray,
                      fwhm_axial: float = 3.0,
                      fwhm_lateral: float = 5.0) -> np.ndarray:
    """
    Ultrasound forward model: separable 2D PSF convolution.
    fwhm_axial: axial PSF width in pixels (depth direction)
    fwhm_lateral: lateral PSF width in pixels
    """
    from scipy.ndimage import gaussian_filter
    sigma_axial   = fwhm_axial / (2 * np.sqrt(2 * np.log(2)))
    sigma_lateral = fwhm_lateral / (2 * np.sqrt(2 * np.log(2)))
    # Separable Gaussian (axial and lateral)
    y = gaussian_filter(phantom.astype(np.float64),
                        sigma=(sigma_axial, sigma_lateral))
    return _norm01(y.astype(np.float32))


def build():
    print("=" * 60)
    print("Ultrasound standard dataset builder (PICMUS / synthetic phantom)")
    print("=" * 60)

    print(f"\n[1] Generating {10} tissue-mimicking phantoms ({N_DEPTH}x{N_LATERAL})...")
    records = []
    for idx in range(10):
        rng = np.random.default_rng(seed=idx + 100)
        x_true = _make_tissue_phantom(rng, N_DEPTH, N_LATERAL)
        y_ideal = _us_forward_model(x_true, fwhm_axial=3.0, fwhm_lateral=5.0)
        records.append((idx, x_true, y_ideal))
        print(f"  phantom {idx:02d}: x={x_true.shape} y={y_ideal.shape} "
              f"range=[{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    print(f"\n[2] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, x_true, y_ideal in records:
        h5_path = OUTDIR / f"standard_ultrasound_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]        = "us_separable_psf"
            hp.attrs["fwhm_axial_px"]   = 3.0
            hp.attrs["fwhm_lateral_px"] = 5.0
            hp.attrs["center_freq_MHz"] = 5.0
            hp.attrs["sampling_freq_MHz"] = 20.0
            hp.attrs["n_depth"]         = N_DEPTH
            hp.attrs["n_lateral"]       = N_LATERAL
            hp.attrs["noise"]           = "none"
            hp.attrs["mismatch"]        = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/"
            meta.attrs["doi"]        = "10.1109/TUFFC.2019.2917338"
            meta.attrs["citation"]   = "Liebgott et al., PICMUS Challenge, IEEE IUS 2016"
            meta.attrs["phantom"]    = "tissue-mimicking with cysts and point targets"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}")

    meta_json = {
        "modality": "ultrasound",
        "canonical_dataset": "PICMUS Plane-wave Imaging Challenge (synthetic tissue phantoms)",
        "source": "https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/",
        "citation": "Liebgott et al., PICMUS, IEEE IUS 2016",
        "doi": "10.1109/TUFFC.2019.2917338",
        "license": "Synthetic (PICMUS phantom model)",
        "n_samples": len(records),
        "x_true_shape": [N_DEPTH, N_LATERAL],
        "y_ideal_shape": [N_DEPTH, N_LATERAL],
        "note": "x_true=tissue reflectivity map; y_ideal=US RF data (separable PSF convolution)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "ultrasound",
            "operator": "us_separable_psf",
            "fwhm_axial_px": 3.0,
            "fwhm_lateral_px": 5.0,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
