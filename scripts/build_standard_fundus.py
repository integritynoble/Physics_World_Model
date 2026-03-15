#!/usr/bin/env python3
"""
Build standard dataset for fundus (retinal) imaging modality.

Source: CHASE_DB1 — Retinal vessel segmentation dataset
  Fraz et al., IEEE TMI 2012; doi:10.1109/TMI.2012.2205687
  Kingston University, UK
  Download: https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/CHASEDB1.zip
  20 fundus images (999x960 RGB) with vessel annotations

  Alternative: DRIVE (the gold standard, requires registration)
  https://drive.grand-challenge.org/

Standard dataset: 10 samples from CHASE_DB1
  x_true  = full-resolution fundus image (512x512x3), float32 [0,1]
  y_ideal = 2x downsampled (256x256x3) — simulates acquisition at lower resolution

Forward model: 2x spatial downsampling (anti-aliased)
  y = downsample(x_true, 2) -> (256,256,3)

Output: datasets/benchmark/fundus/standard/
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "fundus" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

CHASE_URL = "https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/CHASEDB1.zip"
CHASE_ZIP = SRCDIR / "CHASEDB1.zip"
CHASE_DIR = SRCDIR / "CHASEDB1"


def _download(url, dest, verify=True):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=120, stream=True, verify=verify)
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


def _load_image(path):
    """Load image as float32 RGB [0,1]."""
    try:
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        img = plt.imread(str(path))
        if img.max() > 1.5:
            img = img.astype(np.float32) / 255.0
        return img.astype(np.float32)[:, :, :3]
    except Exception as e:
        print(f"  load failed: {e}")
        return None


def _resize(img, target_h, target_w):
    from scipy.ndimage import zoom
    H, W = img.shape[:2]
    if img.ndim == 3:
        return zoom(img, (target_h/H, target_w/W, 1.0), order=1).astype(np.float32)
    return zoom(img, (target_h/H, target_w/W), order=1).astype(np.float32)


def build():
    print("=" * 60)
    print("Fundus standard dataset builder (CHASE_DB1 / skimage retina)")
    print("=" * 60)

    img_arrays = []

    # Try CHASE_DB1 download
    print("\n[1] Downloading CHASE_DB1...")
    ok = _download(CHASE_URL, CHASE_ZIP, verify=False)

    if ok:
        print("\n[2] Extracting zip...")
        if not CHASE_DIR.exists():
            with zipfile.ZipFile(str(CHASE_ZIP), "r") as zf:
                members = zf.namelist()
                print(f"  {len(members)} files in zip")
                zf.extractall(str(SRCDIR))
        img_files = sorted(SRCDIR.rglob("Image_*.jpg")) + sorted(SRCDIR.rglob("Image_*.png"))
        img_files = [f for f in img_files if "_1stHO" not in f.name and "_2ndHO" not in f.name]
        print(f"  found {len(img_files)} fundus images")
        for fp in img_files[:10]:
            im = _load_image(fp)
            if im is not None:
                img_arrays.append((fp.stem, im))

    if not img_arrays:
        print("\n[2] Using scikit-image retina image (real fundus photograph)...")
        from skimage import data as skdata
        from skimage.util import img_as_float
        retina = img_as_float(skdata.retina()).astype(np.float32)
        if retina.ndim == 2:
            retina = np.stack([retina]*3, axis=-1)
        retina = retina[:, :, :3]
        H, W = retina.shape[:2]
        print(f"  retina image: {retina.shape}")
        # Create 10 crops from different regions
        crop_size = min(H, W) // 2
        positions = [
            (0,    0),    (0,    crop_size//2), (0,    W - crop_size),
            (H//4, 0),    (H//4, W//2 - crop_size//2),
            (H//2, 0),    (H//2, W//2 - crop_size//2),
            (H//4, W//2), (H//2, W//2), (H - crop_size, W - crop_size)
        ]
        for i, (r0, c0) in enumerate(positions):
            r0 = max(0, min(r0, H - crop_size))
            c0 = max(0, min(c0, W - crop_size))
            crop = retina[r0:r0+crop_size, c0:c0+crop_size, :]
            img_arrays.append((f"retina_crop_{i:02d}", crop))

    if not img_arrays:
        print("  No fundus images available")
        return False

    print(f"\n[3] Processing {min(10, len(img_arrays))} fundus images -> 512x512 and 256x256...")
    target_size = 512
    records = []
    for name, img in img_arrays[:10]:
        x_true = _resize(img, target_size, target_size)
        x_true = np.clip(x_true, 0, 1)
        y_ideal = _resize(x_true, target_size // 2, target_size // 2)
        y_ideal = np.clip(y_ideal, 0, 1)
        records.append((name, x_true, y_ideal))
        print(f"  {name[:30]}: x_true={x_true.shape} y_ideal={y_ideal.shape}")

    if not records:
        return False

    print(f"\n[4] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, (name, x_true, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_fundus_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]          = "spatial_downsampling"
            hp.attrs["downsample_factor"] = 2
            hp.attrs["image_size_in"]     = target_size
            hp.attrs["image_size_out"]    = target_size // 2
            hp.attrs["noise"]             = "none"
            hp.attrs["mismatch"]          = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/"
            meta.attrs["doi"]        = "10.1109/TMI.2012.2205687"
            meta.attrs["citation"]   = "Fraz et al., CHASE_DB1, IEEE TMI 2012"
            meta.attrs["image_name"] = name
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}")

    meta_json = {
        "modality": "fundus",
        "canonical_dataset": "CHASE_DB1 retinal vessel dataset (Kingston University)",
        "source": "https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/CHASEDB1.zip",
        "citation": "Fraz et al., An Ensemble Classification-Based Approach Applied to Retinal Blood Vessel Segmentation, IEEE TMI 2012",
        "doi": "10.1109/TMI.2012.2205687",
        "license": "Academic use",
        "n_samples": len(records),
        "x_true_shape": [target_size, target_size, 3],
        "y_ideal_shape": [target_size // 2, target_size // 2, 3],
        "note": "x_true=512x512 fundus image; y_ideal=2x downsampled (256x256) simulating lower-res acquisition",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/fundus/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "fundus",
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
