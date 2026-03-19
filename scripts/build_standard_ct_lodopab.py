#!/usr/bin/env python3
"""
Build standard CT dataset from REAL LoDoPaB-CT data.

Downloads ground_truth_test.zip from Zenodo (1.6 GB), extracts 10 real
clinical CT slices, computes clean sinograms, and saves as standard H5.

Source: LoDoPaB-CT (Leuschner et al., Sci. Data 2021)
  doi: 10.1038/s41597-021-00893-z
  Zenodo: https://zenodo.org/records/3384092
  License: CC BY 4.0
  Original data: LIDC/IDRI human chest CT scans

LoDoPaB-CT native parameters:
  - Image size: 362x362 pixels (26cm x 26cm domain)
  - Geometry: parallel beam
  - 1000 projection angles in [0, pi)
  - 513 detector pixels

Output: datasets/benchmark/ct/standard/
  x_true  = real CT image (362x362), float32 [0,1]
  y_ideal = sinogram (1000 angles x 513 detectors), noiseless Radon transform
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests
from skimage.transform import radon

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "ct" / "standard"
OUTDIR.mkdir(parents=True, exist_ok=True)
TMPDIR = ROOT / "tmp_lodopab"
TMPDIR.mkdir(parents=True, exist_ok=True)

ZENODO_URL = "https://zenodo.org/records/3384092/files/ground_truth_test.zip"
N_SAMPLES = 20
N_ANGLES = 1000
N_DETECTORS = 513  # Not directly controllable in skimage; radon auto-computes
NPIX = 362


def download_zenodo(url: str, dest: Path) -> Path:
    """Download file from Zenodo with progress."""
    fname = url.split("/")[-1]
    fpath = dest / fname
    if fpath.exists():
        print(f"  Already downloaded: {fpath} ({fpath.stat().st_size / 1e6:.1f} MB)")
        return fpath

    print(f"  Downloading {fname} from Zenodo...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(fpath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.1f}%)", end="", flush=True)
    print()
    return fpath


def extract_images(zip_path: Path, n_samples: int) -> list[np.ndarray]:
    """Extract n_samples images from LoDoPaB-CT ground truth ZIP."""
    print(f"  Extracting {n_samples} images from {zip_path.name}...")
    images = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # List HDF5 files inside the ZIP
        h5_names = sorted([n for n in zf.namelist() if n.endswith(".hdf5")])
        print(f"  Found {len(h5_names)} HDF5 files in ZIP")

        for h5_name in h5_names:
            if len(images) >= n_samples:
                break
            # Read HDF5 from ZIP into memory
            with zf.open(h5_name) as h5_file:
                h5_bytes = h5_file.read()
                with h5py.File(io.BytesIO(h5_bytes), "r") as f:
                    # LoDoPaB stores images as 'data' dataset with shape (N, 362, 362)
                    if "data" in f:
                        data = f["data"]
                        n_in_file = data.shape[0]
                        for i in range(n_in_file):
                            if len(images) >= n_samples:
                                break
                            img = np.array(data[i], dtype=np.float32)
                            images.append(img)
                            print(f"    Loaded image {len(images)}/{n_samples} from {h5_name} "
                                  f"(shape={img.shape}, range=[{img.min():.4f}, {img.max():.4f}])")
                    else:
                        # Try other common key names
                        for key in f.keys():
                            ds = f[key]
                            if hasattr(ds, "shape") and len(ds.shape) >= 2:
                                if len(ds.shape) == 3:
                                    for i in range(ds.shape[0]):
                                        if len(images) >= n_samples:
                                            break
                                        img = np.array(ds[i], dtype=np.float32)
                                        images.append(img)
                                        print(f"    Loaded image {len(images)}/{n_samples} "
                                              f"from {h5_name}[{key}][{i}]")
                                elif len(ds.shape) == 2:
                                    img = np.array(ds, dtype=np.float32)
                                    images.append(img)
                                    print(f"    Loaded image {len(images)}/{n_samples} "
                                          f"from {h5_name}[{key}]")

    print(f"  Extracted {len(images)} images total")
    return images


def compute_sinogram(image: np.ndarray, n_angles: int) -> np.ndarray:
    """Compute parallel-beam sinogram using skimage radon transform.

    Matches LoDoPaB-CT geometry: 1000 angles in [0, 180) degrees.
    """
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    # skimage radon: image (NxN) -> sinogram (n_detectors x n_angles)
    # circle=False: n_detectors = ceil(sqrt(2)*N); supports non-zero corners
    sino = radon(image, theta=angles, circle=False)
    # Transpose to (n_angles, n_detectors) convention
    return sino.T.astype(np.float32)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def build():
    print("=" * 70)
    print("CT Standard Dataset Builder — Real LoDoPaB-CT Data")
    print("=" * 70)

    # Step 1: Download
    print("\n[1] Downloading LoDoPaB-CT ground truth (test split)...")
    zip_path = download_zenodo(ZENODO_URL, TMPDIR)

    # Step 2: Extract images
    print("\n[2] Extracting real CT images...")
    images = extract_images(zip_path, N_SAMPLES)
    if len(images) < N_SAMPLES:
        print(f"WARNING: Only got {len(images)} images (wanted {N_SAMPLES})")
        if len(images) == 0:
            print("FATAL: No images extracted!")
            return False

    # Step 3: Normalize and compute sinograms
    print(f"\n[3] Computing sinograms ({N_ANGLES} angles, parallel beam)...")
    out_files = []
    for idx, raw_img in enumerate(images):
        # Normalize to [0, 1]
        x_true = normalize_01(raw_img)
        npix = x_true.shape[0]

        # Compute clean sinogram
        sinogram = compute_sinogram(x_true, N_ANGLES)
        n_det_actual = sinogram.shape[1]

        # Save H5
        h5_path = OUTDIR / f"standard_ct_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=sinogram, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"] = "radon_transform"
            hp.attrs["geometry"] = "parallel_beam"
            hp.attrs["n_angles"] = N_ANGLES
            hp.attrs["n_detectors"] = n_det_actual
            hp.attrs["image_size"] = npix
            hp.attrs["angle_range_deg"] = 180.0
            hp.attrs["circle"] = False
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"
            hp.create_dataset("angles_deg",
                              data=np.linspace(0, 180, N_ANGLES, endpoint=False))

            meta = f.create_group("metadata")
            meta.attrs["source"] = "LoDoPaB-CT (Leuschner et al., Sci. Data 2021)"
            meta.attrs["original_data"] = "LIDC/IDRI human chest CT"
            meta.attrs["zenodo"] = "https://zenodo.org/records/3384092"
            meta.attrs["doi"] = "10.1038/s41597-021-00893-z"
            meta.attrs["license"] = "CC BY 4.0"
            meta.attrs["split"] = "test"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  sample {idx:02d}: x_true={x_true.shape} [{x_true.min():.3f},{x_true.max():.3f}] "
              f"sinogram={sinogram.shape} [{sinogram.min():.3f},{sinogram.max():.3f}]")

    # Step 4: Write metadata
    n_det_actual = compute_sinogram(images[0], 2).shape[1]  # just to get detector count
    # Re-read from first file
    with h5py.File(str(OUTDIR / out_files[0]), "r") as f:
        y_shape = list(f["y_ideal"].shape)
        x_shape = list(f["x_true"].shape)

    meta_json = {
        "modality": "ct",
        "canonical_dataset": "LoDoPaB-CT (real clinical CT, LIDC/IDRI)",
        "source": "https://zenodo.org/records/3384092",
        "citation": "Leuschner et al., LoDoPaB-CT, a benchmark dataset for low-dose CT reconstruction, Sci. Data 2021",
        "doi": "10.1038/s41597-021-00893-z",
        "license": "CC BY 4.0",
        "original_data": "LIDC/IDRI human chest CT scans",
        "n_samples": len(out_files),
        "x_true_shape": x_shape,
        "y_ideal_shape": y_shape,
        "geometry": "parallel_beam",
        "n_angles": N_ANGLES,
        "note": f"Real clinical CT from LoDoPaB-CT test split; "
                f"x_true={x_shape[0]}x{x_shape[1]} normalized [0,1]; "
                f"y_ideal=sinogram ({N_ANGLES} angles, {y_shape[1]} detectors)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/ct/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "ct",
            "operator": "radon_transform",
            "geometry": "parallel_beam",
            "n_angles": N_ANGLES,
            "n_detectors": y_shape[1],
            "image_size": x_shape[0],
            "circle": True,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    # Step 5: Cleanup
    print(f"\n[4] Cleaning up temporary files...")
    import shutil
    shutil.rmtree(TMPDIR, ignore_errors=True)
    print(f"  Removed {TMPDIR}")

    print(f"\n{'=' * 70}")
    print(f"Done: {len(out_files)} real LoDoPaB-CT samples in {OUTDIR}")
    print(f"  x_true shape: {x_shape}")
    print(f"  y_ideal shape: {y_shape}")
    print(f"  Source: LIDC/IDRI chest CT via LoDoPaB-CT (CC BY 4.0)")
    print(f"{'=' * 70}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
