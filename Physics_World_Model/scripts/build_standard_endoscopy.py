#!/usr/bin/env python3
"""
Build standard dataset for endoscopy modality.

Source: Kvasir-SEG — polyp segmentation dataset
  Jha et al., Kvasir-SEG: A Segmented Polyp Dataset, MMM 2020
  doi:10.1007/978-3-030-37734-2_37
  https://datasets.simula.no/kvasir-seg/
  Download: https://datasets.simula.no/downloads/kvasir-seg.zip

Standard dataset: 10 samples (polyp images from colonoscopy)
  x_true  = original endoscopy image (256x256x3), RGB float [0,1]
  y_ideal = H_ideal(x_true) = 2x spatial downsampling (endoscopic degradation model)
            Simulates typical compression/transmission artifacts in endoscopy pipelines

Forward model: 2x spatial downsampling (anti-aliased)
  y = downsample(x_true, 2)  -> (128,128,3)

Output: datasets/benchmark/endoscopy/standard/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
import zipfile
import warnings

import h5py
import numpy as np
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "endoscopy" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

KVASIR_URL  = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_ZIP  = SRCDIR / "kvasir-seg.zip"
KVASIR_DIR  = SRCDIR / "kvasir-seg"


def _download(url, dest, verify=True):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=300, stream=True, verify=verify)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print(f"ok ({dest.stat().st_size//1024//1024} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _resize_img(img, target_h, target_w):
    """Simple area-averaging downsampling (anti-aliased)."""
    from scipy.ndimage import zoom
    H, W = img.shape[:2]
    zy = target_h / H
    zx = target_w / W
    if img.ndim == 3:
        out = zoom(img.astype(np.float32), (zy, zx, 1.0), order=1)
    else:
        out = zoom(img.astype(np.float32), (zy, zx), order=1)
    return out.astype(np.float32)


def _load_image(path):
    """Load image as float32 RGB [0,1]."""
    try:
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception:
        pass
    # Fallback: use matplotlib
    try:
        import matplotlib.pyplot as plt
        img = plt.imread(str(path))
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        return img[:,:,:3]
    except Exception as e:
        print(f"  image load failed: {e}")
        return None


def build():
    print("=" * 60)
    print("Endoscopy standard dataset builder (Kvasir-SEG)")
    print("=" * 60)

    print("\n[1] Downloading Kvasir-SEG...")
    ok = _download(KVASIR_URL, KVASIR_ZIP, verify=False)
    if not ok:
        print("Download failed")
        return False

    print("\n[2] Extracting zip...")
    if not KVASIR_DIR.exists():
        with zipfile.ZipFile(str(KVASIR_ZIP), 'r') as zf:
            # Extract only the images directory
            members = [m for m in zf.namelist() if '/images/' in m and m.endswith('.jpg')]
            print(f"  found {len(members)} images in zip")
            zf.extractall(str(SRCDIR))
        print("  extracted")
    else:
        print("  [cache] already extracted")

    # Find image files
    img_dir = SRCDIR / "Kvasir-SEG" / "images"
    if not img_dir.exists():
        # Try alternate path
        img_files = list(SRCDIR.rglob("*.jpg"))
        img_files = [f for f in img_files if "images" in str(f)]
    else:
        img_files = sorted(img_dir.glob("*.jpg"))

    if not img_files:
        img_files = sorted(SRCDIR.rglob("*.jpg"))

    print(f"  found {len(img_files)} image files")
    if not img_files:
        return False

    # Select 10 evenly spaced images
    indices = np.linspace(0, len(img_files) - 1, 10, dtype=int)
    selected = [img_files[i] for i in indices]
    print(f"\n[3] Processing 10 images -> (256x256) and (128x128) downsampled...")

    records = []
    target_size = 256
    for img_path in selected:
        img = _load_image(img_path)
        if img is None:
            continue
        # Resize to 256x256
        x_true = _resize_img(img, target_size, target_size)
        x_true = np.clip(x_true, 0, 1)
        # Forward model: 2x downsampling
        y_ideal = _resize_img(x_true, target_size // 2, target_size // 2)
        y_ideal = np.clip(y_ideal, 0, 1)
        records.append((img_path.stem, x_true, y_ideal))
        print(f"  {img_path.stem[:30]}: x_true={x_true.shape} y_ideal={y_ideal.shape} "
              f"range=[{x_true.min():.3f},{x_true.max():.3f}]")

    if not records:
        return False

    print(f"\n[4] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, (name, x_true, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_endoscopy_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]       = "spatial_downsampling"
            hp.attrs["downsample_factor"] = 2
            hp.attrs["method"]         = "bilinear"
            hp.attrs["image_size_in"]  = target_size
            hp.attrs["image_size_out"] = target_size // 2
            hp.attrs["noise"]          = "none"
            hp.attrs["mismatch"]       = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]       = "https://datasets.simula.no/kvasir-seg/"
            meta.attrs["doi"]          = "10.1007/978-3-030-37734-2_37"
            meta.attrs["citation"]     = "Jha et al., Kvasir-SEG, MMM 2020"
            meta.attrs["image_name"]   = name
            meta.attrs["date_built"]   = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}")

    meta_json = {
        "modality": "endoscopy",
        "canonical_dataset": "Kvasir-SEG polyp segmentation dataset",
        "source": "https://datasets.simula.no/kvasir-seg/",
        "citation": "Jha et al., Kvasir-SEG: A Segmented Polyp Dataset, MMM 2020",
        "doi": "10.1007/978-3-030-37734-2_37",
        "license": "CC BY 4.0",
        "n_samples": len(records),
        "x_true_shape": [target_size, target_size, 3],
        "y_ideal_shape": [target_size // 2, target_size // 2, 3],
        "note": "x_true=256x256 colonoscopy polyp image; y_ideal=2x downsampled (128x128)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/endoscopy/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "endoscopy",
            "operator": "spatial_downsampling",
            "downsample_factor": 2,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
