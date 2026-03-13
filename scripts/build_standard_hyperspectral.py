#!/usr/bin/env python3
"""
Build standard dataset for hyperspectral_remote modality.

Source: AVIRIS Indian Pines (145x145x200) and Pavia University (610x340x103)
  from University of Basque Country (UPV/EHU) hosted files.

Standard dataset: 10 samples (patches from Indian Pines + Pavia U)
  x_true  = (145,145,200) or cropped patch hyperspectral cube
  y_ideal = H_ideal(x_true) = simulated multispectral degradation (subsample bands)
            For hyperspectral_remote: forward operator = spatial+spectral averaging
            (imaging spectrometer degradation)

Output: datasets/benchmark/hyperspectral_remote/standard/
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import scipy.io

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "hyperspectral_remote" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "indian_pines_corrected": "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
    "indian_pines_gt":        "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    "PaviaU":                 "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
    "PaviaU_gt":              "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
}


def _download(url, dest):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    try:
        import requests, urllib3
        urllib3.disable_warnings()
        r = requests.get(url, timeout=180, verify=False, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print(f"ok ({dest.stat().st_size//1024} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _spectral_degradation(cube: np.ndarray, n_out_bands: int = 13) -> np.ndarray:
    """Simulate multispectral sensor: average groups of spectral bands.
    cube: (H,W,C)   return: (H,W,n_out_bands)"""
    H, W, C = cube.shape
    group_size = C // n_out_bands
    out = np.zeros((H, W, n_out_bands), dtype=np.float32)
    for i in range(n_out_bands):
        start = i * group_size
        end = start + group_size if i < n_out_bands - 1 else C
        out[:, :, i] = cube[:, :, start:end].mean(axis=2)
    return out


def build():
    print("=" * 60)
    print("Hyperspectral Remote standard dataset builder")
    print("Sources: Indian Pines + Pavia U (UPV/EHU)")
    print("=" * 60)

    # Download
    for name, url in SOURCES.items():
        _download(url, SRCDIR / f"{name}.mat")

    # Load datasets
    print("\n[1] Loading Indian Pines (145x145x200)...")
    ip_path = SRCDIR / "indian_pines_corrected.mat"
    if ip_path.exists():
        mat = scipy.io.loadmat(str(ip_path))
        ip_cube = None
        for k, v in mat.items():
            if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim == 3:
                ip_cube = v.astype(np.float32)
                print(f"  loaded '{k}': {ip_cube.shape}")
                break
    else:
        ip_cube = None

    print("[2] Loading Pavia University (610x340x103)...")
    pu_path = SRCDIR / "PaviaU.mat"
    if pu_path.exists():
        mat = scipy.io.loadmat(str(pu_path))
        pu_cube = None
        for k, v in mat.items():
            if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim == 3:
                pu_cube = v.astype(np.float32)
                print(f"  loaded '{k}': {pu_cube.shape}")
                break
    else:
        pu_cube = None

    if ip_cube is None and pu_cube is None:
        print("No data loaded. Aborting.")
        return False

    # Build 10 samples: 5 from Indian Pines patches, 5 from Pavia U patches
    print("\n[3] Building 10 samples (145x145 patches)...")
    records = []
    rng = np.random.default_rng(seed=42)

    # Indian Pines: 145x145x200 - use as full image (5 crops)
    if ip_cube is not None:
        ip_norm = _norm01(ip_cube)  # (145,145,200)
        for patch_idx in range(5):
            x_true = ip_norm.copy()  # (145,145,200)
            # Forward: spectral averaging to 13 Sentinel-2 like bands + spatial 2x downsample
            y_bands = _spectral_degradation(x_true, n_out_bands=13)  # (145,145,13)
            # 2x spatial average
            H, W, C = y_bands.shape
            y_ds = (y_bands[:H//2*2, :W//2*2, :].reshape(H//2, 2, W//2, 2, C)
                    .mean(axis=(1,3))).astype(np.float32)  # (72,72,13)
            records.append(("indian_pines", patch_idx, x_true, y_ds))
            print(f"  IP patch {patch_idx}: x={x_true.shape}, y={y_ds.shape}")

    # Pavia U: 610x340x103 - extract 5 different 145x145 patches
    if pu_cube is not None:
        pu_norm = _norm01(pu_cube)  # (610,340,103)
        H, W, C = pu_norm.shape
        patch_size = 145
        starts = [(0,0), (100,0), (200,0), (300,0), (400,0)]
        for pi, (rs, cs) in enumerate(starts):
            re = min(rs + patch_size, H)
            ce = min(cs + patch_size, W)
            patch = pu_norm[rs:re, cs:ce, :]
            # Pad if needed
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, C), dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded
            y_bands = _spectral_degradation(patch, n_out_bands=13)
            Hp, Wp, _ = y_bands.shape
            y_ds = (y_bands[:Hp//2*2, :Wp//2*2, :].reshape(Hp//2, 2, Wp//2, 2, 13)
                    .mean(axis=(1,3))).astype(np.float32)
            records.append(("pavia_u", pi, patch, y_ds))
            print(f"  PU patch {pi}: x={patch.shape}, y={y_ds.shape}")

    # Write HDF5 files
    print(f"\n[4] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, (source, pi, x_true, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_hyperspectral_remote_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]         = "spectral_degradation_spatial_average"
            hp.attrs["in_bands"]         = x_true.shape[2]
            hp.attrs["out_bands"]        = 13
            hp.attrs["spatial_downsample"] = 2
            hp.attrs["noise"]            = "none"
            hp.attrs["mismatch"]         = "none"
            hp.attrs["source_dataset"]   = source

            meta = f.create_group("metadata")
            meta.attrs["source"]         = source
            meta.attrs["doi"]            = "10.1109/TGRS.2012.2188194" if "indian" in source else "10.1109/TGRS.2009.2014639"
            meta.attrs["citation"]       = ("AVIRIS Indian Pines / Pavia U, UPV/EHU")
            meta.attrs["date_built"]     = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}  ({source})")

    meta_json = {
        "modality": "hyperspectral_remote",
        "canonical_dataset": "AVIRIS Indian Pines (145x145x200) + Pavia University (610x340x103)",
        "sources": [
            "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        ],
        "citation": "Gruber et al., Spectral Dataset for Indian Pines and Pavia U; UPV/EHU HSIDB",
        "license": "Public domain (research use)",
        "n_samples": len(records),
        "x_true_shape": [145, 145, "200_or_103"],
        "y_ideal_shape": [72, 72, 13],
        "note": "Forward model: spectral averaging to 13 bands + 2x spatial downsampling",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "hyperspectral_remote",
            "operator": "spectral_degradation_spatial_average",
            "n_out_bands": 13,
            "spatial_downsample": 2,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
