#!/usr/bin/env python3
"""
Build standard dataset for NeRF modality.

Source: NeRF tiny demo dataset (UCSD / Mildenhall et al. ECCV 2020)
  URL: http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
  Contains: images (106, 100, 100, 3), poses (106, 4, 4), focal (float)
  Scene: 'lego' bulldozer from 106 viewpoints

Standard dataset: 10 samples
  Each sample = one novel viewpoint rendering problem
    x_true  = one reference view image (100x100x3) from training set
    y_ideal = identity (ideal rendered image = x_true for known views)
  Plus the full camera pose set and focal length stored in H_params.

Forward model: neural radiance field render F(scene, pose) -> image
  In standard dataset: direct image (identity operator for reference views)

Output: datasets/benchmark/nerf/standard/standard_nerf_{N}.h5
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "nerf" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

TINY_NERF_URL = ("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20"
                 "/nerf/tiny_nerf_data.npz")


def _download(url, dest):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"ok ({dest.stat().st_size//1024} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def build():
    print("=" * 60)
    print("NeRF standard dataset builder (tiny_nerf_data)")
    print("=" * 60)

    npz_path = SRCDIR / "tiny_nerf_data.npz"
    if not _download(TINY_NERF_URL, npz_path):
        return False

    print("\n[1] Loading NeRF data...")
    data = np.load(str(npz_path))
    images = data["images"]     # (N, H, W, 3) float32, [0,1]
    poses  = data["poses"]      # (N, 4, 4) camera-to-world transforms
    focal  = float(data["focal"])  # scalar focal length

    N, H, W, C = images.shape
    print(f"  images: {images.shape}, range=[{images.min():.3f},{images.max():.3f}]")
    print(f"  poses:  {poses.shape}")
    print(f"  focal:  {focal:.2f} px")

    # Use 10 evenly spaced training views as samples
    indices = np.linspace(0, N - 1, 10, dtype=int)
    print(f"\n[2] Building 10 samples from indices {list(indices)}...")

    out_files = []
    for samp_idx, view_idx in enumerate(indices):
        x_true = images[view_idx].astype(np.float32)  # (100,100,3)
        y_ideal = x_true.copy()                         # identity: ideal render = x_true
        pose   = poses[view_idx].astype(np.float32)    # (4,4)

        h5_path = OUTDIR / f"standard_nerf_{samp_idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",       data=x_true,  compression="gzip")
            f.create_dataset("y_ideal",      data=y_ideal, compression="gzip")
            f.create_dataset("pose",         data=pose,    compression="gzip")
            f.create_dataset("all_poses",    data=poses.astype(np.float32), compression="gzip")
            f.create_dataset("all_images",   data=images.astype(np.float32), compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]    = "nerf_render"
            hp.attrs["focal_px"]    = focal
            hp.attrs["H"]           = H
            hp.attrs["W"]           = W
            hp.attrs["n_channels"]  = C
            hp.attrs["n_views"]     = N
            hp.attrs["view_idx"]    = int(view_idx)
            hp.attrs["noise"]       = "none"
            hp.attrs["mismatch"]    = "none"

            meta = f.create_group("metadata")
            meta.attrs["scene"]     = "lego (tiny_nerf)"
            meta.attrs["source"]    = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/"
            meta.attrs["doi"]       = "10.1007/978-3-030-58452-8_24"
            meta.attrs["citation"]  = "Mildenhall et al., NeRF, ECCV 2020"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}  (view {view_idx}/{N-1})")

    meta_json = {
        "modality": "nerf",
        "canonical_dataset": "tiny_nerf_data (lego scene)",
        "source": "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/",
        "citation": "Mildenhall et al., NeRF: Representing Scenes as Neural Radiance Fields, ECCV 2020",
        "doi": "10.1007/978-3-030-58452-8_24",
        "license": "MIT",
        "n_samples": 10,
        "x_true_shape": [H, W, C],
        "y_ideal_shape": [H, W, C],
        "n_views_total": N,
        "focal_px": focal,
        "note": "Each sample is one training view; all_poses/all_images stored for reconstruction context",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/nerf/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "nerf",
            "operator": "nerf_render",
            "scene": "lego",
            "H": H, "W": W,
            "n_views": N,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: 10 samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
