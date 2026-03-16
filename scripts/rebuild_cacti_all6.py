"""Rebuild CACTI standard dataset using ALL 6 gray DeSCI datasets, ALL shots.

CACTI forward: y = sum_t(frame_t * mask_t), compressing 8 video frames into 1 snapshot.

6 datasets, 20 total shots:
  kobe(4), aerial(4), crash(4), traffic(6), drop(1), runner(1)

Each sample:
  x_true = (256,256,8) video cube (8 ground-truth frames)
  y_ideal = (256,256) coded snapshot measurement

Images: per-shot frame PNGs (frame_00..07) + measurement PNG
"""
import numpy as np
import h5py
import json
import struct
import zlib
from pathlib import Path
from scipy.io import loadmat

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark/cacti/standard")
RAW = BASE / "_raw_src"

def normalize_01(x):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)

def write_png(arr_2d, path):
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

# Dataset order (standard DeSCI benchmark order)
DATASETS = [
    ("kobe",    "kobe_cacti.mat"),
    ("aerial",  "aerial_cacti.mat"),
    ("crash",   "crash_cacti.mat"),
    ("traffic", "traffic_cacti.mat"),
    ("drop",    "drop_cacti.mat"),
    ("runner",  "runner_cacti.mat"),
]

# Clean old files
for old in BASE.glob("standard_cacti_*.h5"):
    old.unlink()
img_dir = BASE / "images"
img_dir.mkdir(exist_ok=True)
for old in img_dir.glob("*.png"):
    old.unlink()
for old in img_dir.glob("*.pgm"):
    old.unlink()

sample_idx = 0
sample_info = []

for dname, fname in DATASETS:
    mat = loadmat(str(RAW / fname))
    orig = mat["orig"].astype(np.float64)
    mask = mat["mask"].astype(np.float64)

    # Normalize orig to [0,1]
    if orig.max() > 1.0:
        orig = orig / 255.0 if orig.max() > 1.0 else orig

    n_frames = orig.shape[2]
    n_masks = mask.shape[2]  # always 8
    n_shots = n_frames // n_masks

    print(f"=== {dname}: {n_frames} frames, {n_shots} shots ===")

    for shot in range(n_shots):
        start = shot * n_masks
        end = start + n_masks

        # x_true: 8-frame video cube (256, 256, 8)
        video_cube = orig[:, :, start:end].astype(np.float32)

        # Normalize each frame independently for cleaner data
        for f in range(video_cube.shape[2]):
            frame = video_cube[:, :, f]
            lo, hi = frame.min(), frame.max()
            if hi - lo > 1e-12:
                video_cube[:, :, f] = (frame - lo) / (hi - lo)
            else:
                video_cube[:, :, f] = 0.0

        # y_ideal: CACTI coded snapshot = sum_t(frame_t * mask_t)
        y = np.zeros((256, 256), dtype=np.float64)
        for t in range(n_masks):
            y += video_cube[:, :, t].astype(np.float64) * mask[:, :, t]
        y = normalize_01(y.astype(np.float32))

        # Save H5
        h5path = BASE / f"standard_cacti_{sample_idx:02d}.h5"
        with h5py.File(str(h5path), "w") as f:
            f.create_dataset("x_true", data=video_cube, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            f.create_dataset("mask", data=mask.astype(np.float32), compression="gzip")
            f.attrs["modality"] = "cacti"
            f.attrs["sample_index"] = sample_idx
            f.attrs["dataset_name"] = dname
            f.attrs["shot_index"] = shot
            f.attrs["n_shots_in_dataset"] = n_shots
            f.attrs["frame_range"] = f"{start}-{end-1}"
            f.attrs["compression_ratio"] = n_masks
            f.attrs["source"] = f"DeSCI {dname} CACTI (shot {shot+1}/{n_shots})"
            f.attrs["reference"] = "Liu et al., Rank Minimization for Snapshot Compressive Imaging, TPAMI 2019"
            f.attrs["data_type"] = "real"

        # Save frame PNGs (all 8 frames per shot)
        for t in range(n_masks):
            write_png(video_cube[:, :, t],
                      str(img_dir / f"x_true_{sample_idx:02d}_frame{t:02d}.png"))

        # Save measurement PNG
        write_png(y, str(img_dir / f"y_meas_{sample_idx:02d}.png"))

        info = f"  [{sample_idx:02d}] {dname} shot {shot+1}/{n_shots} (frames {start}-{end-1})"
        print(info)
        sample_info.append({
            "sample": sample_idx,
            "dataset": dname,
            "shot": shot + 1,
            "total_shots": n_shots,
            "frame_range": f"{start}-{end-1}",
        })
        sample_idx += 1

n_total = sample_idx
print(f"\nTotal samples: {n_total} (from 6 datasets)")

# Metadata
meta = {
    "modality": "cacti",
    "n_samples": n_total,
    "x_shape": [256, 256, 8],
    "y_shape": [256, 256],
    "mask_shape": [256, 256, 8],
    "compression_ratio": 8,
    "source": "DeSCI 6 gray CACTI benchmark (kobe, aerial, crash, traffic, drop, runner)",
    "reference": "Liu et al., Rank Minimization for Snapshot Compressive Imaging, TPAMI 2019",
    "data_type": "real",
    "forward_model": "CACTI: y = sum_t(frame_t * mask_t), 8 frames -> 1 coded snapshot",
    "datasets": {
        "kobe":    {"frames": 32, "shots": 4, "description": "Basketball game (Kobe Bryant)"},
        "aerial":  {"frames": 32, "shots": 4, "description": "Aerial city skyline"},
        "crash":   {"frames": 32, "shots": 4, "description": "Car crash test"},
        "traffic": {"frames": 48, "shots": 6, "description": "Highway traffic"},
        "drop":    {"frames":  8, "shots": 1, "description": "Water drop"},
        "runner":  {"frames":  8, "shots": 1, "description": "Marathon runner"},
    },
    "samples": sample_info,
}
with open(BASE / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

spec = {
    "modality": "cacti",
    "source": "DeSCI 6 gray CACTI benchmark (kobe, aerial, crash, traffic, drop, runner)",
    "reference": "Liu et al., Rank Minimization for Snapshot Compressive Imaging, TPAMI 2019",
}
with open(BASE / "spec.json", "w") as f:
    json.dump(spec, f, indent=2)

# Verify uniqueness
hashes = set()
for i in range(n_total):
    with h5py.File(str(BASE / f"standard_cacti_{i:02d}.h5"), "r") as f:
        h = hash(f["x_true"][:].tobytes())
        hashes.add(h)
print(f"Unique x_true hashes: {len(hashes)}/{n_total}")

# Count images
n_imgs = len(list(img_dir.glob("*.png")))
print(f"Total PNGs: {n_imgs} ({n_total}*8 frames + {n_total} measurements = {n_total*9})")
print("CACTI rebuild DONE - all 6 datasets, all shots")
