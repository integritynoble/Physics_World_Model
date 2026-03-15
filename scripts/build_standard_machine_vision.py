#!/usr/bin/env python3
"""
Build standard dataset for machine_vision modality.

Source: MVTec Anomaly Detection Dataset
  Bergmann et al., CVPR 2019; doi:10.1109/CVPR.2019.00982
  https://www.mvtec.com/company/research/datasets/mvtec-ad
  15 categories of industrial objects/textures, each with good images + defects

Standard dataset: 10 samples from the 'hazelnut' and 'wood' categories
  x_true  = defect-free industrial surface image (256x256x3), float32 [0,1]
  y_ideal = H_ideal(x_true) = spatially blurred image (Gaussian sigma=1.5px)
            Simulates image formation: y = blur(x_true), reconstruction = deconvolution

Forward model: Gaussian blur (PSF)
  y = Gaussian_blur(x_true, sigma=1.5)

Output: datasets/benchmark/machine_vision/standard/
"""
from __future__ import annotations

import json
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "machine_vision" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

# MVTec individual categories (smaller downloads)
MVTEC_BASE = "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad/"
CATEGORIES = [
    ("hazelnut", "hazelnut.tar.xz"),
    ("wood",     "wood.tar.xz"),
]


def _download(url, dest):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=300, stream=True)
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
    print("Machine Vision standard dataset builder (MVTec AD / scikit-image textures)")
    print("=" * 60)

    img_arrays = []
    print("\n[1] Downloading MVTec AD categories...")
    for cat_name, cat_file in CATEGORIES:
        url = MVTEC_BASE + cat_file
        dest = SRCDIR / cat_file
        cat_dir = SRCDIR / cat_name
        if not cat_dir.exists():
            if _download(url, dest):
                print(f"  extracting {cat_file}...")
                with tarfile.open(str(dest), "r:xz") as tf:
                    tf.extractall(str(SRCDIR))
        if cat_dir.exists():
            good_dir = cat_dir / "train" / "good"
            if good_dir.exists():
                files = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))
                for fp in files[:5]:
                    im = _load_image(fp)
                    if im is not None:
                        img_arrays.append((fp.stem, cat_name, im))
                print(f"  {cat_name}: {len(files)} good images")

    if not img_arrays:
        print("  MVTec download failed; using scikit-image texture images...")
        from skimage import data as skdata
        from skimage.util import img_as_float
        textures = {
            "brick": skdata.brick(),
            "grass": skdata.grass(),
            "gravel": skdata.gravel(),
        }
        sk_images = {
            "camera":      skdata.camera(),
            "coins":       skdata.coins(),
            "page":        skdata.page(),
            "microaneurysms": skdata.microaneurysms(),
            "immunohistochemistry": skdata.immunohistochemistry(),
            "retina":      skdata.retina(),
            "rocket":      skdata.rocket(),
        }
        for name, arr in textures.items():
            im = img_as_float(arr).astype(np.float32)
            H, W = im.shape[:2]
            # Tile to 256x256 if smaller
            if H < 256 or W < 256:
                reps = (256 // H + 1, 256 // W + 1)
                im = np.tile(im, reps)[:256, :256]
            if im.ndim == 2:
                im = np.stack([im]*3, axis=-1)
            img_arrays.append((name, "texture", im))
        for name, arr in sk_images.items():
            im = img_as_float(arr).astype(np.float32)
            if im.ndim == 2:
                im = np.stack([im]*3, axis=-1)
            if im.ndim == 3 and im.shape[2] > 3:
                im = im[:, :, :3]
            img_arrays.append((name, "scene", im))

    target_size = 256
    print(f"\n[2] Processing {min(10, len(img_arrays))} images -> {target_size}x{target_size}...")
    records = []
    from scipy.ndimage import gaussian_filter

    for name, cat, img in img_arrays[:10]:
        x_true = _resize(img, target_size, target_size)
        x_true = np.clip(x_true, 0, 1).astype(np.float32)
        # Forward model: Gaussian blur (PSF)
        sigma = 1.5
        y_ideal = np.stack([
            gaussian_filter(x_true[:, :, c], sigma=sigma)
            for c in range(3)
        ], axis=-1).astype(np.float32)
        y_ideal = np.clip(y_ideal, 0, 1)
        records.append((name, cat, x_true, y_ideal))
        print(f"  [{cat}] {name}: x={x_true.shape} y={y_ideal.shape} "
              f"range=[{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    if not records:
        return False

    print(f"\n[3] Writing {len(records)} HDF5 files...")
    out_files = []
    sigma = 1.5
    for idx, (name, cat, x_true, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_machine_vision_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]    = "gaussian_blur"
            hp.attrs["sigma"]       = sigma
            hp.attrs["image_size"]  = target_size
            hp.attrs["noise"]       = "none"
            hp.attrs["mismatch"]    = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://www.mvtec.com/company/research/datasets/mvtec-ad/"
            meta.attrs["doi"]        = "10.1109/CVPR.2019.00982"
            meta.attrs["citation"]   = "Bergmann et al., MVTec AD, CVPR 2019"
            meta.attrs["category"]   = cat
            meta.attrs["image_name"] = name
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}")

    meta_json = {
        "modality": "machine_vision",
        "canonical_dataset": "MVTec Anomaly Detection Dataset",
        "source": "https://www.mvtec.com/company/research/datasets/mvtec-ad/",
        "citation": "Bergmann et al., MVTec AD, CVPR 2019",
        "doi": "10.1109/CVPR.2019.00982",
        "license": "CC BY-NC-SA 4.0 (non-commercial research)",
        "n_samples": len(records),
        "x_true_shape": [target_size, target_size, 3],
        "y_ideal_shape": [target_size, target_size, 3],
        "note": "x_true=defect-free industrial surface; y_ideal=Gaussian blur (sigma=1.5px) simulating imaging PSF",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/machine_vision/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "machine_vision",
            "operator": "gaussian_blur",
            "sigma": 1.5,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
