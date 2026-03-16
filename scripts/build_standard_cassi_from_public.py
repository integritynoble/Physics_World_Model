"""Build cassi standard dataset from sd_cassi/public 10 scenes.

The cassi YAML expects x_shape [256,256,28] and y_shape [256,310],
matching the sd_cassi public dataset exactly.
"""
import h5py
import numpy as np
from pathlib import Path
from PIL import Image

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")

def normalize_01(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12: return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def write_png(arr, path):
    a = normalize_01(arr)
    a = (a * 255).clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        Image.fromarray(a, 'L').save(path)
    elif a.ndim == 3 and a.shape[2] >= 3:
        # Use bands 5,15,25 as RGB for visualization
        rgb = a[:, :, [5, 15, min(25, a.shape[2]-1)]]
        Image.fromarray(rgb, 'RGB').save(path)

# Source: sd_cassi public dataset
src_h5 = BASE / "sd_cassi" / "public" / "sd_cassi_challenge_public.h5"
out_dir = BASE / "cassi" / "standard"
out_dir.mkdir(parents=True, exist_ok=True)
img_dir = out_dir / "images"
img_dir.mkdir(exist_ok=True)

# Clean old files
for old in out_dir.glob("standard_cassi_*.h5"):
    old.unlink()
for old in img_dir.glob("*.png"):
    old.unlink()

print(f"Reading {src_h5}...")
with h5py.File(str(src_h5), "r") as src:
    n_samples = len([k for k in src.keys() if k.startswith("sample_")])
    print(f"  Found {n_samples} scenes")

    for i in range(n_samples):
        sample_key = f"sample_{i:02d}"
        s = src[sample_key]

        x_true = np.array(s["x_true"], dtype=np.float32)     # (256, 256, 28)
        y_ideal = np.array(s["y_ideal"], dtype=np.float32)    # (256, 310)
        H_ideal = np.array(s["H_ideal"], dtype=np.float32)    # (256, 256) mask

        out_path = out_dir / f"standard_cassi_{i:02d}.h5"
        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("x_true", data=x_true, compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
            f.create_dataset("mask", data=H_ideal, compression="gzip")
            f.attrs["modality"] = "cassi"
            f.attrs["sample_index"] = i
            f.attrs["source"] = "CAVE hyperspectral (sd_cassi public challenge scenes)"
            f.attrs["reference"] = "Wagadarikar et al., Applied Optics 2008; Yasuma et al., ICIP 2010"
            f.attrs["data_type"] = "real"
            f.attrs["n_bands"] = int(x_true.shape[2])

        # Save PNGs
        write_png(x_true, str(img_dir / f"x_true_{i:02d}.png"))
        write_png(y_ideal, str(img_dir / f"y_meas_{i:02d}.png"))

        print(f"  scene {i:02d}: x_true {x_true.shape}, y_ideal {y_ideal.shape}")

print(f"\nSaved {n_samples} cassi standard samples to {out_dir}")
