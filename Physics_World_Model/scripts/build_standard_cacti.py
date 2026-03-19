#!/usr/bin/env python3
"""
Build standard dataset for CACTI modality.

Source: SCI Video Benchmark - 6 gray scenes (ucaswangls/cacti on GitHub)
  Files: kobe/traffic/runner/drop/crash/aerial _cacti.mat
  Each .mat contains: orig (H,W,T) video, mask (H,W,8) per-frame mask

Standard dataset: 6 samples (one per scene)
  x_true  = first 8 frames of orig, normalized to [0,1], shape (256,256,8)
  y_ideal = ideal CACTI measurement = sum_t mask[:,:,t] * x_true[:,:,t], shape (256,256)
  No noise, no operator mismatch.

Output: datasets/benchmark/cacti/standard/standard_cacti_{N}.h5
        datasets/benchmark/cacti/standard/metadata.json
        datasets/benchmark/cacti/standard/spec.json

Reference:
  Liu et al., Rank Minimization for Snapshot Compressive Imaging,
  IEEE T-PAMI 2019. https://github.com/ucaswangls/cacti
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
OUTDIR = ROOT / "datasets" / "benchmark" / "cacti" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/ucaswangls/cacti/main/test_datasets"

SCENES = ["kobe", "traffic", "runner", "drop", "crash", "aerial"]

SCENE_URLS = {s: f"{BASE_URL}/simulation/{s}_cacti.mat" for s in SCENES}
MASK_URL   = f"{BASE_URL}/mask/mask.mat"


def _download(url: str, dest: Path, label: str = "") -> bool:
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {label or dest.name} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"ok ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _load_scene(path: Path):
    """Return (x_true, mask) from a CACTI .mat scene file.

    x_true: first 8 frames normalized to [0,1], shape (256,256,8)
    mask:   per-frame binary mask,               shape (256,256,8)
    """
    mat = scipy.io.loadmat(str(path))
    orig = mat["orig"].astype(np.float32)  # (H,W,T) video, values 0-255
    mask = mat["mask"].astype(np.float32)  # (H,W,8) per-frame mask, 0/1

    # Take first 8 frames
    if orig.ndim == 3 and orig.shape[2] >= 8:
        x = _norm01(orig[:, :, :8])  # (256,256,8)
    else:
        raise ValueError(f"orig shape unexpected: {orig.shape}")

    return x, mask


def _cacti_forward_ideal(x_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ideal CACTI forward: y[i,j] = sum_t mask[i,j,t] * x[i,j,t].

    x_true: (H,W,T=8)
    mask:   (H,W,T=8)
    return: (H,W)
    """
    y = np.sum(mask * x_true, axis=2).astype(np.float32)
    return y


def build():
    print("=" * 60)
    print("CACTI standard dataset builder")
    print("=" * 60)

    # 1. Download
    print("\n[1] Downloading source files...")
    for scene in SCENES:
        _download(SCENE_URLS[scene], SRCDIR / f"{scene}_cacti.mat", scene)

    # 2. Build samples
    print("\n[2] Building samples...")
    records = []
    for idx, scene in enumerate(SCENES):
        mat_path = SRCDIR / f"{scene}_cacti.mat"
        if not mat_path.exists():
            print(f"  SKIP {scene}: file not downloaded")
            continue
        try:
            x_true, mask = _load_scene(mat_path)
            y_ideal = _cacti_forward_ideal(x_true, mask)
            records.append((scene, x_true, mask, y_ideal))
            print(f"  {scene}: x_true={x_true.shape} [{x_true.min():.2f},{x_true.max():.2f}]"
                  f"  y_ideal={y_ideal.shape} [{y_ideal.min():.1f},{y_ideal.max():.1f}]")
        except Exception as e:
            print(f"  ERROR {scene}: {e}")

    if not records:
        print("No samples built. Aborting.")
        return False

    # 3. Write HDF5 files
    print(f"\n[3] Writing {len(records)} HDF5 files to {OUTDIR}")
    out_files = []
    for idx, (scene, x_true, mask, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_cacti_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
            f.create_dataset("mask",    data=mask,    compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]      = "cacti"
            hp.attrs["n_frames"]      = 8
            hp.attrs["mask_type"]     = "binary_per_frame"
            hp.attrs["noise"]         = "none"
            hp.attrs["mismatch"]      = "none"
            hp.attrs["source_repo"]   = "https://github.com/ucaswangls/cacti"

            meta = f.create_group("metadata")
            meta.attrs["scene_name"]  = scene
            meta.attrs["source"]      = "ucaswangls/cacti"
            meta.attrs["doi"]         = "10.1109/TPAMI.2019.2900167"
            meta.attrs["citation"]    = ("Liu et al., Rank Minimization for "
                                         "Snapshot Compressive Imaging, IEEE T-PAMI 2019")
            meta.attrs["date_built"]  = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}  ({scene})")

    # 4. Write metadata.json
    meta_json = {
        "modality": "cacti",
        "canonical_dataset": "SCI Video Benchmark - 6 gray scenes",
        "source_repo": "https://github.com/ucaswangls/cacti",
        "citation": ("Liu et al., Rank Minimization for Snapshot Compressive Imaging, "
                     "IEEE T-PAMI 2019"),
        "doi": "10.1109/TPAMI.2019.2900167",
        "license": "BSD-3-Clause",
        "n_samples": len(records),
        "x_true_shape": [256, 256, 8],
        "y_ideal_shape": [256, 256],
        "scenes": [r[0] for r in records],
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/standard/",
        "status": "done" if len(records) == 6 else "partial",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    spec = {
        "modality": "cacti",
        "operator": "cacti",
        "n_frames": 8,
        "shift_step_px": 1,
        "shift_type": "vertical",
        "mask_type": "binary_per_frame",
        "noise": "none",
        "mismatch": "none",
        "split": "standard",
    }
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nDone: {len(records)}/6 scenes built in {OUTDIR}")
    return True


if __name__ == "__main__":
    ok = build()
    sys.exit(0 if ok else 1)
