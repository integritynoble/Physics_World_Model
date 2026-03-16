"""Rebuild NeRF and Gaussian Splatting standards with ALL 106 Tiny NeRF views.

Tiny NeRF: 106 multi-view images of Lego scene (100x100x3), with camera poses.
This is the canonical NeRF benchmark scene from Mildenhall et al., ECCV 2020.

NeRF: All views with camera poses for novel view synthesis
Gaussian Splatting: Same data, different reconstruction approach
"""
import numpy as np
import h5py
import json
import struct
import zlib
from pathlib import Path
from skimage.transform import resize as sk_resize

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"

def normalize_01(x):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)

def write_png_gray(arr_2d, path):
    u = np.nan_to_num(arr_2d, 0)
    lo, hi = float(u.min()), float(u.max())
    if hi - lo < 1e-12:
        u = np.zeros(u.shape, dtype=np.uint8)
    else:
        u = ((u - lo) / (hi - lo) * 255).astype(np.uint8)
    h, w = u.shape
    def chunk(ct, d):
        c = ct + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''
    for row in u:
        raw += b'\x00' + row.tobytes()
    with open(str(path), 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
                + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9))
                + chunk(b'IEND', b''))

# Load all NeRF data
print("Loading Tiny NeRF data...")
data = np.load(str(CACHE / "tiny_nerf_data.npz"))
images = data["images"]    # (106, 100, 100, 3)
poses = data["poses"]      # (106, 4, 4)
focal = float(data["focal"])

n_views = images.shape[0]
print(f"  {n_views} views, shape={images.shape}, focal={focal:.1f}")

# Standard NeRF split: first 100 train, last 6 test
n_train = 100
n_test = n_views - n_train

for mod_name, view_indices, description in [
    ("nerf", list(range(n_views)), "NeRF novel view synthesis"),
    ("gaussian_splatting", list(range(n_views)), "3D Gaussian Splatting"),
]:
    OUT = BASE / mod_name / "standard"
    OUT.mkdir(parents=True, exist_ok=True)

    for old in OUT.glob(f"standard_{mod_name}_*.h5"):
        old.unlink()
    img_dir = OUT / "images"
    img_dir.mkdir(exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()

    print(f"\n=== {mod_name}: {len(view_indices)} views ===")

    # Group views into samples: each sample is a batch of views
    # For NeRF: 10 samples of ~10 views each, plus test views
    # Sample 0-9: training views (10 per sample)
    # Plus include ALL views in the dataset
    batch_size = 10
    n_samples = (len(view_indices) + batch_size - 1) // batch_size

    for idx in range(n_samples):
        start = idx * batch_size
        end = min(start + batch_size, len(view_indices))
        batch_views = view_indices[start:end]
        nb = len(batch_views)

        # Stack all views in this batch
        view_stack = np.stack([
            sk_resize(images[v], (256, 256, 3), order=3, anti_aliasing=True).astype(np.float32)
            for v in batch_views
        ], axis=0)  # (nb, 256, 256, 3)

        pose_stack = poses[batch_views]  # (nb, 4, 4)

        # For NeRF, y_ideal is the set of posed images (input to NeRF training)
        # x_true is the ground truth image to be rendered
        # Use luminance for 2D representation
        lum_stack = np.stack([
            0.299*view_stack[i,:,:,0] + 0.587*view_stack[i,:,:,1] + 0.114*view_stack[i,:,:,2]
            for i in range(nb)
        ], axis=0)  # (nb, 256, 256)

        h5path = OUT / f"standard_{mod_name}_{idx:02d}.h5"
        with h5py.File(str(h5path), "w") as f:
            f.create_dataset("x_true", data=view_stack, compression="gzip")
            f.create_dataset("poses", data=pose_stack, compression="gzip")
            f.create_dataset("luminance", data=lum_stack, compression="gzip")
            f.attrs["modality"] = mod_name
            f.attrs["sample_index"] = idx
            f.attrs["view_indices"] = str(batch_views)
            f.attrs["n_views"] = nb
            f.attrs["focal_length"] = focal
            is_test = start >= n_train
            f.attrs["split"] = "test" if is_test else "train"
            f.attrs["source"] = f"Tiny NeRF Lego scene (views {start}-{end-1})"
            f.attrs["reference"] = "Mildenhall et al., NeRF: Representing Scenes as Neural Radiance Fields, ECCV 2020"
            f.attrs["data_type"] = "real"

        # Save PNGs for each view in this batch
        for i, v in enumerate(batch_views):
            write_png_gray(lum_stack[i], str(img_dir / f"x_true_{idx:02d}_view{v:03d}.png"))

        split_label = "test" if start >= n_train else "train"
        print(f"  [{idx:02d}] views {start}-{end-1} ({nb} views, {split_label})")

    n_total = n_samples
    meta = {
        "modality": mod_name,
        "n_samples": n_total,
        "n_total_views": n_views,
        "views_per_sample": batch_size,
        "x_shape": [256, 256, 3],
        "image_resolution": [100, 100],
        "upsampled_to": [256, 256],
        "focal_length": focal,
        "train_views": n_train,
        "test_views": n_test,
        "source": "Tiny NeRF Lego scene (Mildenhall et al., ECCV 2020)",
        "reference": "Mildenhall et al., NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020",
        "data_type": "real",
        "scene": "Lego bulldozer (synthetic rendered, multi-view)",
    }
    with open(OUT / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(OUT / "spec.json", "w") as f:
        json.dump({"modality": mod_name, "source": meta["source"], "reference": meta["reference"]}, f, indent=2)

    hashes = set()
    for i in range(n_total):
        with h5py.File(str(OUT / f"standard_{mod_name}_{i:02d}.h5"), "r") as f:
            hashes.add(hash(f["x_true"][:].tobytes()))
    n_imgs = len(list(img_dir.glob("*.png")))
    print(f"  Unique hashes: {len(hashes)}/{n_total}, PNGs: {n_imgs}")

print("\nNeRF + 3DGS rebuild DONE")
