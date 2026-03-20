"""
Build standard CACTI dataset with ALL measurement groups from each scene.

Source: SCI Video Benchmark - 6 gray scenes (ucaswangls/cacti on GitHub)
  kobe:    (256,256,32) -> 4 groups of 8 frames
  traffic: (256,256,48) -> 6 groups of 8 frames
  runner:  (256,256,40) -> 5 groups of 8 frames
  drop:    (256,256,40) -> 5 groups of 8 frames
  crash:   (256,256,32) -> 4 groups of 8 frames
  aerial:  (256,256,32) -> 4 groups of 8 frames
  Total:   28 measurement groups

Each sample:
  x_true  = (256,256,8) video datacube (one group)
  y_ideal = (256,256) = sum_t mask[:,:,t] * x_true[:,:,t]
  mask    = (256,256,8) per-frame binary coded aperture

Reference: Liu et al., IEEE T-PAMI 2019, doi:10.1109/TPAMI.2019.2900167
Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
import scipy.io
import urllib.request
from pathlib import Path
from PIL import Image
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "cacti" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/ucaswangls/cacti/main/test_datasets"
SCENES = ["kobe", "traffic", "runner", "drop", "crash", "aerial"]
SCENE_URLS = {s: f"{BASE_URL}/simulation/{s}_cacti.mat" for s in SCENES}
MASK_URL = f"{BASE_URL}/mask/mask.mat"

T_MASK = 8  # frames per coded aperture group


def download(url, dest, label=""):
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


def norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def to_uint8(arr):
    arr = np.nan_to_num(arr)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def save_images(x_true, y_ideal, mask, img_dir):
    """Save visualization images for one CACTI sample."""
    img_dir.mkdir(parents=True, exist_ok=True)
    # Save each temporal frame
    for t in range(x_true.shape[2]):
        Image.fromarray(to_uint8(x_true[:, :, t]), mode="L").save(
            str(img_dir / f"x_true_frame_{t:02d}.png"))
    # Middle frame as main x_true
    mid = x_true.shape[2] // 2
    Image.fromarray(to_uint8(x_true[:, :, mid]), mode="L").save(
        str(img_dir / "x_true.png"))
    # y_ideal
    Image.fromarray(to_uint8(y_ideal), mode="L").save(
        str(img_dir / "y_ideal.png"))
    # First mask frame
    Image.fromarray(to_uint8(mask[:, :, 0]), mode="L").save(
        str(img_dir / "mask_frame_00.png"))


def build():
    print("=" * 60)
    print("CACTI standard dataset builder (ALL measurement groups)")
    print("=" * 60)

    # 1. Download
    print("\n[1] Downloading source files...")
    for scene in SCENES:
        download(SCENE_URLS[scene], SRCDIR / f"{scene}_cacti.mat", scene)
    download(MASK_URL, SRCDIR / "mask.mat", "mask")

    # Load mask (shared across all scenes)
    mask_path = SRCDIR / "mask.mat"
    if mask_path.exists():
        mat = scipy.io.loadmat(str(mask_path))
        # mask.mat may have key 'mask' or 'mask_3d'
        for key in ["mask", "mask_3d"]:
            if key in mat:
                shared_mask = mat[key].astype(np.float32)
                break
        else:
            shared_mask = None
    else:
        shared_mask = None

    # 2. Extract ALL groups from each scene
    print("\n[2] Extracting measurement groups...")
    records = []
    for scene in SCENES:
        mat_path = SRCDIR / f"{scene}_cacti.mat"
        if not mat_path.exists():
            print(f"  SKIP {scene}: not downloaded")
            continue

        mat = scipy.io.loadmat(str(mat_path))
        orig = mat["orig"].astype(np.float32)  # (H, W, T_total)

        # Get mask: per-scene or shared
        if "mask" in mat:
            mask = mat["mask"].astype(np.float32)
        elif shared_mask is not None:
            mask = shared_mask.copy()
        else:
            print(f"  ERROR {scene}: no mask found")
            continue

        # Ensure mask has T_MASK frames
        if mask.ndim == 2:
            # Single mask -> replicate
            mask = np.stack([mask] * T_MASK, axis=2)
        elif mask.shape[2] > T_MASK:
            mask = mask[:, :, :T_MASK]

        T_total = orig.shape[2]
        n_groups = T_total // T_MASK
        print(f"  {scene}: shape={orig.shape}, {n_groups} groups of {T_MASK} frames")

        for g in range(n_groups):
            t0 = g * T_MASK
            x_true = norm01(orig[:, :, t0:t0 + T_MASK])  # (256, 256, 8)
            # Ideal CACTI forward: y = sum_t mask[:,:,t] * x[:,:,t]
            y_ideal = np.sum(mask * x_true, axis=2).astype(np.float32)
            label = f"{scene}_g{g}"
            records.append((label, x_true, mask.copy(), y_ideal))

    if not records:
        print("No samples built")
        return False

    # 3. Remove old files
    for old in OUTDIR.glob("standard_cacti_*.h5"):
        old.unlink()

    # 4. Write HDF5
    print(f"\n[3] Writing {len(records)} HDF5 files...")
    out_files = []
    scene_names = []
    for idx, (label, x_true, mask, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_cacti_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
            f.create_dataset("mask", data=mask, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"] = "cacti"
            hp.attrs["forward_type"] = "temporal_mask_sum"
            hp.attrs["n_frames"] = T_MASK
            hp.attrs["mask_type"] = "binary_per_frame"
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"

            meta = f.create_group("metadata")
            meta.attrs["scene_name"] = label
            meta.attrs["source"] = "ucaswangls/cacti"
            meta.attrs["doi"] = "10.1109/TPAMI.2019.2900167"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        scene_names.append(label)

        # Save images
        img_dir = OUTDIR / "images" / f"sample_{idx:02d}"
        save_images(x_true, y_ideal, mask, img_dir)

        print(f"  {idx:02d}: {label}  x={x_true.shape} y={y_ideal.shape}")

    # 5. Write metadata
    meta_json = {
        "modality": "cacti",
        "canonical_dataset": "SCI Video Benchmark - 6 gray scenes (28 measurement groups)",
        "source_repo": "https://github.com/ucaswangls/cacti",
        "citation": "Liu et al., Rank Minimization for Snapshot Compressive Imaging, IEEE T-PAMI 2019",
        "doi": "10.1109/TPAMI.2019.2900167",
        "license": "BSD-3-Clause",
        "n_samples": len(records),
        "n_scenes": 6,
        "n_groups_per_scene": {"kobe": 4, "traffic": 6, "runner": 5, "drop": 5, "crash": 4, "aerial": 4},
        "x_true_shape": [256, 256, T_MASK],
        "y_ideal_shape": [256, 256],
        "forward_type": "temporal_mask_sum",
        "forward_model": "y[i,j] = sum_t mask[i,j,t] * x[i,j,t]",
        "scenes": scene_names,
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    spec = {
        "modality": "cacti",
        "operator": "cacti",
        "forward_type": "temporal_mask_sum",
        "n_frames": T_MASK,
        "mask_type": "binary_per_frame",
        "noise": "none",
        "mismatch": "none",
        "split": "standard",
    }
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nDone: {len(records)} samples from {len(SCENES)} scenes "
          f"({len(records)} measurement groups total)")
    return True


if __name__ == "__main__":
    build()
