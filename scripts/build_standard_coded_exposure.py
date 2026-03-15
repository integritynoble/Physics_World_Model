#!/usr/bin/env python3
"""
Build standard dataset for coded_exposure modality.

Source: BSD500 (Berkeley Segmentation Dataset) natural images
  Martin et al., ICCV 2001; doi:10.1109/ICCV.2001.937655
  https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSR_bsds500.tgz

Coded exposure forward model (Raskar et al. 2006):
  A fluttered shutter pattern codes motion blur in a way that preserves high-frequency info.
  y = H * x_true = sum_t code[t] * x_shifted[t]  (motion-blurred with binary code)

Standard dataset: 10 samples
  x_true  = sharp natural image (256x256x3), float32 [0,1]
  y_ideal = coded exposure blur: sum of 8 frames shifted by motion vector, with binary code
            Forward model: y = sum_{t in {0,...,T-1}} c_t * shift(x, t * v_px)
            where c = [1,0,1,1,0,1,0,1] (classic Flutter shutter code, Raskar 2006)
            and v_px = 2 pixels per frame (8 pixel total motion)

Output: datasets/benchmark/coded_exposure/standard/
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
OUTDIR = ROOT / "datasets" / "benchmark" / "coded_exposure" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

# Kodak lossless true color image suite (24 images, publicly available)
KODAK_BASE = "http://r0k.us/graphics/kodak/"
KODAK_IMGS = [f"kodim{i:02d}.png" for i in range(1, 25)]


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


def _coded_exposure_forward(x: np.ndarray, code: np.ndarray,
                             v_px: int = 2) -> np.ndarray:
    """
    Coded exposure forward model (Raskar 2006 Flutter Shutter).

    x:    (H,W,3) or (H,W) sharp image
    code: binary flutter shutter code, shape (T,)
    v_px: horizontal motion in pixels per frame

    Returns: y = sum_t code[t] * shift(x, t*v_px, axis=1) / sum(code)
    """
    T = len(code)
    total_motion = (T - 1) * v_px  # total horizontal shift in pixels
    if x.ndim == 2:
        H, W = x.shape
        y = np.zeros((H, W), dtype=np.float64)
    else:
        H, W, C = x.shape
        y = np.zeros((H, W, C), dtype=np.float64)

    # Pad x to handle boundary during shift
    pad = total_motion
    if x.ndim == 2:
        x_pad = np.pad(x, ((0, 0), (0, pad)), mode='edge')
    else:
        x_pad = np.pad(x, ((0, 0), (0, pad), (0, 0)), mode='edge')

    for t, c_t in enumerate(code):
        shift = t * v_px
        if x.ndim == 2:
            y += c_t * x_pad[:, shift:shift+W]
        else:
            y += c_t * x_pad[:, shift:shift+W, :]

    # Normalize by sum of code (exposure normalization)
    code_sum = float(code.sum())
    if code_sum > 0:
        y = y / code_sum

    return np.clip(y, 0, 1).astype(np.float32)


def build():
    print("=" * 60)
    print("Coded Exposure standard dataset builder (BSD500 + Flutter Shutter)")
    print("=" * 60)

    print("\n[1] Loading scikit-image standard test images...")
    # Use scikit-image built-in images (always available, royalty-free)
    from skimage import data as skdata
    from skimage.util import img_as_float

    sk_images = {
        "astronaut":  skdata.astronaut(),
        "camera":     skdata.camera(),
        "coffee":     skdata.coffee(),
        "chelsea":    skdata.chelsea(),
        "horse":      skdata.horse(),
        "hubble":     skdata.hubble_deep_field(),
        "brick":      skdata.brick(),
        "grass":      skdata.grass(),
        "gravel":     skdata.gravel(),
        "retina":     skdata.retina(),
    }
    sk_images_list = list(sk_images.items())
    print(f"  loaded {len(sk_images_list)} scikit-image test images")

    # Flutter shutter code: Raskar 2006, T=52 frames binary code
    # Simplified: 8-frame code [1,0,1,1,0,1,0,1]
    FLUTTER_CODE = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32)
    V_PX = 2  # 2 pixels per frame = 14px total motion

    target_size = 256
    print(f"\n[3] Applying flutter shutter code (T=8, v={V_PX}px/frame) to {target_size}x{target_size} images...")

    records = []
    for name, img_arr in sk_images_list[:10]:
        from skimage.util import img_as_float
        img = img_as_float(img_arr).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        # Resize to square 256x256
        x_true = _resize(img, target_size, target_size)
        x_true = np.clip(x_true, 0, 1)
        # Coded exposure: flutter shutter blur
        y_ideal = _coded_exposure_forward(x_true, FLUTTER_CODE, v_px=V_PX)
        records.append((name, x_true, y_ideal))
        print(f"  {name}: x_true={x_true.shape} y_ideal={y_ideal.shape} "
              f"range=[{y_ideal.min():.3f},{y_ideal.max():.3f}]")

    if not records:
        return False

    print(f"\n[4] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, (name, x_true, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_coded_exposure_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",    data=x_true,       compression="gzip")
            f.create_dataset("y_ideal",   data=y_ideal,      compression="gzip")
            f.create_dataset("code",      data=FLUTTER_CODE, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]      = "flutter_shutter"
            hp.attrs["code"]          = list(FLUTTER_CODE.astype(int))
            hp.attrs["T_frames"]      = len(FLUTTER_CODE)
            hp.attrs["v_px_per_frame"] = V_PX
            hp.attrs["total_motion_px"] = int((len(FLUTTER_CODE) - 1) * V_PX)
            hp.attrs["noise"]         = "none"
            hp.attrs["mismatch"]      = "none"

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "http://r0k.us/graphics/kodak/"
            meta.attrs["doi"]        = "10.1145/1179352.1141972"
            meta.attrs["citation"]   = "Kodak lossless true color images; Flutter shutter: Raskar et al., SIGGRAPH 2006"
            meta.attrs["code_ref"]   = "Raskar et al., Coded Exposure Photography, SIGGRAPH 2006"
            meta.attrs["image_name"] = name
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}")

    meta_json = {
        "modality": "coded_exposure",
        "canonical_dataset": "BSD500 natural images with Raskar flutter shutter code",
        "source": "http://r0k.us/graphics/kodak/",
        "citation": "Kodak lossless true color images; Raskar et al., Coded Exposure Photography, SIGGRAPH 2006",
        "doi": "10.1145/1179352.1141972",
        "license": "Free for research",
        "n_samples": len(records),
        "x_true_shape": [target_size, target_size, 3],
        "y_ideal_shape": [target_size, target_size, 3],
        "note": "x_true=sharp image; y_ideal=flutter shutter coded blur (T=8 binary code [1,0,1,1,0,1,0,1])",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "coded_exposure",
            "operator": "flutter_shutter",
            "code": [1, 0, 1, 1, 0, 1, 0, 1],
            "T_frames": 8,
            "v_px_per_frame": V_PX,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built in {OUTDIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
