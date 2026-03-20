#!/usr/bin/env python3
"""
Build standard dataset for particle_calorimetry modality.

Source: Fast Calorimeter Simulation Challenge 2022 - Dataset 1 (photons)
  Zenodo 6368338; https://zenodo.org/record/6368338
  Contains: photon showers in ATLAS-like calorimeter geometry
  particle = photon entering detector, energy = 256 energy values
  calorimeter = 45×16×9 (layer × eta × phi cells) — dataset 1 geometry

Standard dataset: 10 samples
  x_true  = one calorimeter shower event, shape (45,16,9) energy deposition
  y_ideal = H_ideal(x_true) = x_true (identity — the calorimeter measurement IS the shower)
            (The reconstruction task is: given compressed/noisy detector output, recover x_true)

Output: datasets/benchmark/particle_calorimetry/standard/
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
OUTDIR = ROOT / "datasets" / "benchmark" / "particle_calorimetry" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

ZENODO_URL = "https://zenodo.org/api/records/6368338/files/dataset_1_photons_1.hdf5/content"
LOCAL_FILE = SRCDIR / "dataset_1_photons_1.hdf5"


def _download(url, dest):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ({url.split('/')[-2]})... ", end="", flush=True)
    try:
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                written += len(chunk)
        mb = dest.stat().st_size // 1024 // 1024
        print(f"ok ({mb} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def build():
    print("=" * 60)
    print("Particle Calorimetry standard dataset (CaloChallenge 2022)")
    print("=" * 60)

    if not _download(ZENODO_URL, LOCAL_FILE):
        return False

    print("\n[1] Inspecting CaloChallenge HDF5 structure...")
    with h5py.File(str(LOCAL_FILE), "r") as f:
        keys = list(f.keys())
        print(f"  keys: {keys}")
        for k in keys:
            ds = f[k]
            print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}")
            if ds.ndim > 0:
                data_sample = ds[:5]
                print(f"    range=[{float(data_sample.min()):.4f},{float(data_sample.max()):.4f}]")

    # Load structure
    print("\n[2] Loading shower data...")
    with h5py.File(str(LOCAL_FILE), "r") as f:
        # CaloChallenge Dataset 1: keys are 'energy' and 'showers'
        # showers: (N, n_cells) flattened, energy: (N,) incident energy
        if "showers" in f:
            showers = f["showers"][:]
            energies = f["incident_energies"][:, 0]  # shape (N,1) -> (N,)
            print(f"  showers: {showers.shape}, dtype={showers.dtype}")
            print(f"  energies: {energies.shape}, range=[{energies.min():.1f},{energies.max():.1f}] MeV")
        else:
            print(f"  keys: {list(f.keys())}")
            return False

    N = len(showers)
    print(f"  {N} shower events available")

    # Dataset 1 geometry: 45 layers, 16 eta bins, 9 phi bins = 648 cells total
    n_layers_1 = 1    # Layer 0: 1 eta × 1 phi per layer
    n_layers_2 = 12   # Layers 1-12: ...
    # Actually, geometry for Dataset 1 (photons at 256 discrete energies):
    # Total cells = 368: layer0=1*1, layer1=16*9=144, layer2=16*9=144, layer3=7*1=7, etc.
    # Let's just work with the flat representation and reshape to (45,)
    # or use whatever the actual flattened geometry gives us

    n_cells = showers.shape[1]
    print(f"  n_cells per shower: {n_cells}")

    # Standard approach: use sqrt(n_cells) or use as (1, n_cells)
    # For Dataset 1: n_cells = 368 for photons
    # The calorimeter image: reshape to something close to (45,16,9) but exact dims depend on dataset
    # For Dataset 1 photons, the geometry is layers with different granularities:
    # This is complex, let's use (n_cells,) as the shape for x_true

    # Select 10 diverse samples (spread across incident energy spectrum)
    indices = np.linspace(0, N-1, 10, dtype=int)
    print(f"\n[3] Selecting 10 samples at indices {list(indices)}...")

    out_files = []
    for samp_idx, ev_idx in enumerate(indices):
        shower = showers[ev_idx].astype(np.float32)  # (n_cells,)
        energy = float(energies[ev_idx])

        # Normalize shower by incident energy (standard in calorimeter analysis)
        x_true_raw = shower / (energy + 1e-8)  # normalized energy deposition
        x_true = _norm01(x_true_raw)            # further normalize to [0,1]
        y_ideal = x_true.copy()                 # identity forward model

        h5_path = OUTDIR / f"standard_particle_calorimetry_{samp_idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]        = "calorimeter"
            hp.attrs["particle"]        = "photon"
            hp.attrs["incident_energy_MeV"] = energy
            hp.attrs["n_cells"]         = n_cells
            hp.attrs["dataset"]         = "CaloChallenge Dataset 1 photons"
            hp.attrs["noise"]           = "none"
            hp.attrs["mismatch"]        = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]        = "https://zenodo.org/record/6368338"
            meta.attrs["doi"]           = "10.48550/arXiv.2202.07347"
            meta.attrs["citation"]      = "CaloChallenge 2022 Dataset 1 - photons"
            meta.attrs["event_index"]   = int(ev_idx)
            meta.attrs["date_built"]    = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  sample {samp_idx:02d}: event {ev_idx}, E={energy:.0f} MeV, "
              f"x_true={x_true.shape}, range=[{x_true.min():.3f},{x_true.max():.3f}]")

    meta_json = {
        "modality": "particle_calorimetry",
        "canonical_dataset": "CaloChallenge 2022 - Dataset 1 (photons, ATLAS calorimeter)",
        "source": "https://zenodo.org/record/6368338",
        "citation": "Kruse et al., Fast Calorimeter Simulation Challenge 2022, arXiv:2202.07347",
        "doi": "10.48550/arXiv.2202.07347",
        "license": "CC-BY-4.0",
        "n_samples": 10,
        "x_true_shape": [n_cells],
        "y_ideal_shape": [n_cells],
        "particle": "photon",
        "note": "Shower normalized by incident energy; flat (n_cells,) representation",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/particle_calorimetry/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "particle_calorimetry",
            "operator": "calorimeter_detection",
            "particle": "photon",
            "n_cells": n_cells,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: 10 samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
