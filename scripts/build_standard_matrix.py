#!/usr/bin/env python3
"""
Build standard dataset for matrix (matrix completion) modality.

Source: MovieLens ML-100K dataset (GroupLens, public domain)
  URL: https://files.grouplens.org/datasets/movielens/ml-100k.zip
  943 users x 1682 movies, 100,000 ratings (1-5 scale)

Standard dataset: 10 samples
  Each sample is a (943,1682) matrix with:
    x_true  = full ratings matrix (known entries filled, unknowns=0)
    y_ideal = observed entries only (random mask of observed ratings)

Forward model: random mask M  → y = M .* x_true (entry-wise observation)

Output: datasets/benchmark/matrix/standard/standard_matrix_{N}.h5
"""
from __future__ import annotations

import json
import sys
import io
import zipfile
import urllib.request
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "matrix" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


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
    print("Matrix standard dataset builder (MovieLens ML-100K)")
    print("=" * 60)

    zip_path = SRCDIR / "ml-100k.zip"
    _download(ML100K_URL, zip_path)
    if not zip_path.exists():
        return False

    # Load ratings from u.data
    print("\n[1] Loading ratings...")
    rows, cols, vals = [], [], []
    with zipfile.ZipFile(str(zip_path)) as zf:
        with zf.open("ml-100k/u.data") as f:
            for line in f.read().decode().strip().split("\n"):
                uid, mid, rating, ts = line.split("\t")
                rows.append(int(uid) - 1)   # 0-indexed
                cols.append(int(mid) - 1)
                vals.append(float(rating))

    n_users, n_movies = 943, 1682
    # Build full ratings matrix
    R = np.zeros((n_users, n_movies), dtype=np.float32)
    for r, c, v in zip(rows, cols, vals):
        R[r, c] = v / 5.0  # normalize to [0,1]

    obs_density = (R > 0).mean()
    print(f"  Full matrix: {R.shape}, observed={obs_density:.3f}, "
          f"range=[{R.min():.2f},{R.max():.2f}]")

    # Build 10 samples with different random observation masks
    print("\n[2] Building 10 samples...")
    out_files = []
    records = []
    for idx in range(10):
        rng = np.random.default_rng(seed=1000 + idx * 17)
        # Sample a random subset of observed entries (same density as original)
        mask = (R > 0).astype(np.float32)
        # Apply additional 50% dropout to simulate partial observation
        obs_idx = np.argwhere(mask > 0)
        keep = rng.choice(len(obs_idx), size=int(len(obs_idx) * 0.7), replace=False)
        sparse_mask = np.zeros_like(mask)
        for ki in keep:
            r_, c_ = obs_idx[ki]
            sparse_mask[r_, c_] = 1.0
        y_ideal = R * sparse_mask  # observed entries only

        records.append((R.copy(), sparse_mask, y_ideal))
        print(f"  sample {idx:02d}: x_true={R.shape}, "
              f"obs_frac={sparse_mask.mean():.3f}, "
              f"y_nnz={(y_ideal>0).sum()}")

    print(f"\n[3] Writing {len(records)} HDF5 files...")
    for idx, (x_true, mask, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_matrix_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
            f.create_dataset("mask",    data=mask,    compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]       = "matrix_completion"
            hp.attrs["n_users"]        = 943
            hp.attrs["n_movies"]       = 1682
            hp.attrs["obs_fraction"]   = float(mask.mean())
            hp.attrs["noise"]          = "none"
            hp.attrs["mismatch"]       = "none"
            hp.attrs["source"]         = "MovieLens ML-100K"

            meta = f.create_group("metadata")
            meta.attrs["source"]       = "https://grouplens.org/datasets/movielens/100k/"
            meta.attrs["citation"]     = "Harper & Konstan, The MovieLens Datasets, TIIS 2015"
            meta.attrs["doi"]          = "10.1145/2827872"
            meta.attrs["date_built"]   = datetime.now().strftime("%Y-%m-%d")
            meta.attrs["seed"]         = 1000 + idx * 17

        out_files.append(h5_path.name)

    meta_json = {
        "modality": "matrix",
        "canonical_dataset": "MovieLens ML-100K",
        "source": "https://grouplens.org/datasets/movielens/100k/",
        "citation": "Harper & Konstan, The MovieLens Datasets, ACM TIIS 2015",
        "doi": "10.1145/2827872",
        "license": "Public domain",
        "n_samples": len(records),
        "x_true_shape": [943, 1682],
        "y_ideal_shape": [943, 1682],
        "note": "y_ideal = x_true * random_mask (sparse observation)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "matrix",
            "operator": "random_mask",
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
