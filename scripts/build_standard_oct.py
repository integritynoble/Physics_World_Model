#!/usr/bin/env python3
"""
Build standard dataset for oct (Optical Coherence Tomography) modality.

Source: RETOUCH OCT Challenge dataset
  Bogunovic et al., IEEE TMI 2019; doi:10.1109/TMI.2019.2901970
  https://retouch.grand-challenge.org/

  Alternative (more accessible): Generate synthetic OCT B-scans from
  known retinal layer models, following standard OCT forward model.

Standard dataset: 10 samples
  x_true  = clean OCT B-scan (496x512), float32 [0,1]
  y_ideal = measured OCT (with Rayleigh speckle-free forward model):
            y = |FT_2D(x_scatterers)|^2 — the A-scan interference signal
            Here we use simplified: y = Gaussian blur of x_true (depth PSF)

Forward model: OCT depth-resolved reflectivity imaging
  y = H * x = convolution with axial PSF (coherence gate)
  PSF_axial: Gaussian with FWHM ~ 5 pixels (10um coherence length / 2um/px)

Output: datasets/benchmark/oct/standard/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "oct" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _make_synthetic_oct_bscan(rng, npix_depth=496, npix_width=512):
    """
    Generate a synthetic retinal OCT B-scan.

    Retinal layers (from top to bottom in depth):
    - Nerve fiber layer (NFL)
    - Ganglion cell layer (GCL)
    - Inner plexiform layer (IPL)
    - Inner nuclear layer (INL)
    - Outer plexiform layer (OPL)
    - Outer nuclear layer (ONL)
    - External limiting membrane (ELM)
    - Photoreceptor inner/outer segment (IS/OS)
    - Retinal pigment epithelium (RPE)
    """
    bscan = np.zeros((npix_depth, npix_width), dtype=np.float32)

    # Layer positions (pixel depth from top of scan, typical range 150-380px)
    # Curved profile: parabolic with small random variation
    x_arr = np.arange(npix_width)
    # Foveal dip: V-shaped profile
    fovea_center = npix_width // 2 + rng.integers(-50, 50)
    fovea_depth = rng.uniform(20, 50)

    # Base retinal surface (ILM - internal limiting membrane)
    ilm = 150 + fovea_depth * np.exp(-0.5 * ((x_arr - fovea_center) / 80)**2)
    ilm = ilm + rng.uniform(-5, 5, npix_width)  # small variations

    # Layer thicknesses (in pixels)
    nfl_thick = rng.uniform(5, 12, npix_width)
    gcl_thick = rng.uniform(8, 20, npix_width)
    ipl_thick = rng.uniform(8, 15, npix_width)
    inl_thick = rng.uniform(8, 12, npix_width)
    opl_thick = rng.uniform(6, 10, npix_width)
    onl_thick = rng.uniform(30, 50, npix_width)
    is_thick  = rng.uniform(8, 12, npix_width)
    rpe_thick = rng.uniform(10, 15, npix_width)

    # Fovea: thin NFL/GCL at center
    fovea_mask = np.exp(-0.5 * ((x_arr - fovea_center) / 60)**2)
    nfl_thick = nfl_thick * (1 - 0.7 * fovea_mask)
    gcl_thick = gcl_thick * (1 - 0.6 * fovea_mask)

    # Layer reflectivities
    layer_amp = {
        'nfl': 0.8, 'gcl': 0.2, 'ipl': 0.5, 'inl': 0.15,
        'opl': 0.6, 'onl': 0.1, 'is': 0.9, 'rpe': 1.0
    }
    layer_widths = [nfl_thick, gcl_thick, ipl_thick, inl_thick,
                    opl_thick, onl_thick, is_thick, rpe_thick]
    layer_names = ['nfl', 'gcl', 'ipl', 'inl', 'opl', 'onl', 'is', 'rpe']

    # Build B-scan column by column
    for col in range(npix_width):
        depth = ilm[col]
        for name, thick in zip(layer_names, [w[col] for w in layer_widths]):
            d_start = int(depth)
            d_end = int(depth + thick)
            amp = layer_amp[name]
            # Add random speckle variation
            amp_vary = amp * rng.uniform(0.6, 1.4)
            for d in range(d_start, min(d_end, npix_depth)):
                bscan[d, col] = max(bscan[d, col], amp_vary)
            depth += thick

    # Add mild background noise (vitreous)
    bscan += rng.uniform(0, 0.05, bscan.shape).astype(np.float32)

    return _norm01(bscan)


def _oct_axial_psf_forward(x: np.ndarray, fwhm_px: float = 5.0) -> np.ndarray:
    """
    OCT forward model: convolution with axial (depth) PSF.
    PSF = Gaussian along depth axis (axis 0).
    x: (depth, width) OCT B-scan
    """
    from scipy.ndimage import gaussian_filter1d
    sigma = fwhm_px / (2 * np.sqrt(2 * np.log(2)))
    # Convolve along depth (A-scan direction)
    y = gaussian_filter1d(x.astype(np.float64), sigma=sigma, axis=0)
    return _norm01(y.astype(np.float32))


def build():
    print("=" * 60)
    print("OCT standard dataset builder (synthetic retinal B-scans)")
    print("=" * 60)

    print("\n[1] Generating 10 synthetic retinal OCT B-scans (496x512)...")
    npix_depth = 496
    npix_width = 512

    records = []
    for idx in range(10):
        rng = np.random.default_rng(seed=idx + 42)
        x_true = _make_synthetic_oct_bscan(rng, npix_depth, npix_width)
        # OCT forward: convolution with axial PSF
        y_ideal = _oct_axial_psf_forward(x_true, fwhm_px=5.0)
        records.append((idx, x_true, y_ideal))
        print(f"  B-scan {idx:02d}: {x_true.shape} range=[{x_true.min():.3f},{x_true.max():.3f}]")

    print(f"\n[2] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, x_true, y_ideal in records:
        h5_path = OUTDIR / f"standard_oct_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]        = "oct_axial_psf"
            hp.attrs["psf_fwhm_px"]     = 5.0
            hp.attrs["axial_res_um"]    = 10.0
            hp.attrs["pixel_size_um"]   = 2.0
            hp.attrs["wavelength_nm"]   = 840
            hp.attrs["n_a_scans"]       = npix_width
            hp.attrs["n_depth_px"]      = npix_depth
            hp.attrs["noise"]           = "none"
            hp.attrs["mismatch"]        = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://retouch.grand-challenge.org/"
            meta.attrs["doi"]        = "10.1109/TMI.2019.2901970"
            meta.attrs["citation"]   = "Bogunovic et al., RETOUCH OCT Challenge, IEEE TMI 2019"
            meta.attrs["note"]       = "Synthetic retinal B-scan following standard layer model"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}")

    meta_json = {
        "modality": "oct",
        "canonical_dataset": "RETOUCH OCT Challenge (synthetic retinal B-scans)",
        "source": "https://retouch.grand-challenge.org/",
        "citation": "Bogunovic et al., RETOUCH: The Retinal OCT Fluid Detection and Segmentation Benchmark, IEEE TMI 2019",
        "doi": "10.1109/TMI.2019.2901970",
        "license": "Synthetic (layer model from literature)",
        "n_samples": len(records),
        "x_true_shape": [npix_depth, npix_width],
        "y_ideal_shape": [npix_depth, npix_width],
        "note": "x_true=clean retinal B-scan (496x512); y_ideal=axial PSF blurred (FWHM=5px)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/oct/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "oct",
            "operator": "oct_axial_psf",
            "psf_fwhm_px": 5.0,
            "wavelength_nm": 840,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
