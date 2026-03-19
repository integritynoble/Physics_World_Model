#!/usr/bin/env python3
"""
Build standard dataset for CASSI (and sd_cassi) modality.

Source: KAIST Hyperspectral 10 test scenes, accessed via TSA-Net GitHub mirror.
  Original: Choi et al. KAIST hyperspectral database, used in MST CVPR 2022.
  Mirror:   https://github.com/mengziyi64/TSA-Net (BSD-3-Clause)
  Files:    scene01.mat ... scene10.mat, mask.mat
  Each .mat scene: variable 'img', shape (256,256,28), float32, range [0,1]

Standard dataset: 10 samples (one per scene)
  x_true  = img[scene], shape (256,256,28)
  y_ideal = CASSI ideal measurement, shape (256,310)
            y[i, j + lam*step] += mask[i,j] * x[i,j,lam]
            with dispersion_step=2 (as in cassi.yaml: total_shift_px=54 over 27 steps)
  No noise, no operator mismatch.

Output: datasets/benchmark/cassi/standard/standard_cassi_{N}.h5
        datasets/benchmark/cassi/standard/metadata.json
        datasets/benchmark/cassi/standard/spec.json

Reference:
  Cai et al., MST: Mask-guided Spectral-wise Transformer for Efficient HSI Reconstruction,
  CVPR 2022. https://github.com/caiyuanhao1998/MST
  Choi et al., A New Hyperspectral Dataset, 2017.
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
OUTDIR = ROOT / "datasets" / "benchmark" / "cassi" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

# TSA-Net mirror of KAIST scenes (BSD-3-Clause)
BASE_URL = ("https://raw.githubusercontent.com/mengziyi64/TSA-Net/master"
            "/TSA_Net_simulation/Data")
SCENE_URLS = {
    f"scene{i:02d}": f"{BASE_URL}/Testing_data/scene{i:02d}.mat"
    for i in range(1, 11)
}
MASK_URL = f"{BASE_URL}/mask.mat"

# CASSI forward model parameters (from cassi.yaml)
N_BANDS        = 28
DISPERSION_STEP = 2   # px per spectral band (total_shift_px=54 / (N_BANDS-1) ~ 2)
H_out, W_out   = 256, 256 + (N_BANDS - 1) * DISPERSION_STEP  # (256, 310)


def _download(url: str, dest: Path, label: str = "") -> bool:
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {label or dest.name} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
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


def _cassi_forward_ideal(x: np.ndarray, mask: np.ndarray,
                          step: int = DISPERSION_STEP) -> np.ndarray:
    """
    Ideal CASSI (single-disperser) forward model.

    For each spectral band lam in 0..N_BANDS-1:
      y[row, col + lam*step] += mask[row, col] * x[row, col, lam]

    x:    (H, W, N_BANDS)
    mask: (H, W)
    return: (H, W + (N_BANDS-1)*step) = (256, 310)
    """
    H, W, L = x.shape
    W_meas = W + (L - 1) * step
    y = np.zeros((H, W_meas), dtype=np.float64)
    for lam in range(L):
        shift = lam * step
        y[:, shift:shift + W] += mask * x[:, :, lam]
    return y.astype(np.float32)


def build():
    print("=" * 60)
    print("CASSI standard dataset builder")
    print("Source: KAIST 10 scenes via TSA-Net (GitHub)")
    print("=" * 60)

    # 1. Download scenes and mask
    print("\n[1] Downloading source files...")
    mask_path = SRCDIR / "mask.mat"
    _download(MASK_URL, mask_path, "mask.mat (256x256)")
    for name, url in SCENE_URLS.items():
        _download(url, SRCDIR / f"{name}.mat", f"{name}.mat")

    # 2. Load mask
    print("\n[2] Loading mask...")
    if mask_path.exists():
        mat = scipy.io.loadmat(str(mask_path))
        mask = mat["mask"].astype(np.float32)
        print(f"  mask: shape={mask.shape}, density={float((mask > 0.5).mean()):.3f}")
        mask_bin = (mask > 0.5).astype(np.float32)  # binarize (it's near-binary already)
    else:
        print("  ERROR: mask.mat not downloaded")
        return False

    # 3. Build samples
    print(f"\n[3] Building {len(SCENE_URLS)} samples (y_shape will be {H_out}x{W_out})...")
    records = []
    for name in sorted(SCENE_URLS.keys()):
        mat_path = SRCDIR / f"{name}.mat"
        if not mat_path.exists():
            print(f"  SKIP {name}: not downloaded")
            continue
        mat = scipy.io.loadmat(str(mat_path))
        img = mat["img"].astype(np.float32)  # (256,256,28) in [0,1]
        y_ideal = _cassi_forward_ideal(img, mask_bin, step=DISPERSION_STEP)
        records.append((name, img, y_ideal))
        print(f"  {name}: x={img.shape} [{img.min():.3f},{img.max():.3f}]  "
              f"y={y_ideal.shape} [{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    if not records:
        print("No samples built. Aborting.")
        return False

    # 4. Write HDF5 files
    print(f"\n[4] Writing {len(records)} HDF5 files to {OUTDIR}")
    out_files = []
    for idx, (name, x_true, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_cassi_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",   data=x_true,   compression="gzip")
            f.create_dataset("y_ideal",  data=y_ideal,  compression="gzip")
            f.create_dataset("mask",     data=mask_bin, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]         = "cassi"
            hp.attrs["n_bands"]          = N_BANDS
            hp.attrs["dispersion_step"]  = DISPERSION_STEP
            hp.attrs["mask_type"]        = "binary_random"
            hp.attrs["density"]          = float(mask_bin.mean())
            hp.attrs["y_shape"]          = f"({H_out},{W_out})"
            hp.attrs["noise"]            = "none"
            hp.attrs["mismatch"]         = "none"
            hp.attrs["source_repo"]      = "https://github.com/mengziyi64/TSA-Net"

            meta = f.create_group("metadata")
            meta.attrs["scene_name"]     = name
            meta.attrs["source"]         = "KAIST Hyperspectral via TSA-Net mirror"
            meta.attrs["doi"]            = "10.1109/CVPR52688.2022.01540"
            meta.attrs["citation"]       = ("Cai et al., MST: Mask-guided Spectral-wise "
                                            "Transformer, CVPR 2022")
            meta.attrs["date_built"]     = datetime.now().strftime("%Y-%m-%d")
            meta.attrs["spectral_range"] = "450-650 nm, 28 bands, 7.14 nm spacing"

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}  ({name})")

    # 5. metadata.json
    meta_json = {
        "modality": "cassi",
        "canonical_dataset": "KAIST Hyperspectral - 10 scenes",
        "source_repo": "https://github.com/mengziyi64/TSA-Net",
        "original_source": "https://github.com/caiyuanhao1998/MST",
        "citation": ("Cai et al., MST: Mask-guided Spectral-wise Transformer "
                     "for Efficient HSI Reconstruction, CVPR 2022"),
        "doi": "10.1109/CVPR52688.2022.01540",
        "license": "BSD-3-Clause (TSA-Net mirror)",
        "n_samples": len(records),
        "x_true_shape": [256, 256, 28],
        "y_ideal_shape": [H_out, W_out],
        "spectral_bands": N_BANDS,
        "spectral_range_nm": "450-650",
        "dispersion_step_px": DISPERSION_STEP,
        "mask_density": float(mask_bin.mean()),
        "scenes": [r[0] for r in records],
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/standard/",
        "status": "done" if len(records) == 10 else "partial",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    spec = {
        "modality": "cassi",
        "operator": "cassi",
        "n_bands": N_BANDS,
        "dispersion_step_px": DISPERSION_STEP,
        "mask_type": "binary_random",
        "spectral_range_nm": [450, 650],
        "noise": "none",
        "mismatch": "none",
        "split": "standard",
    }
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nDone: {len(records)}/10 scenes built in {OUTDIR}")
    return True


if __name__ == "__main__":
    ok = build()
    sys.exit(0 if ok else 1)
