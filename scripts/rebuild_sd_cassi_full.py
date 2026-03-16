"""Rebuild SD-CASSI standard with full 24-band hyperspectral cube from DeSCI bird24.

SD-CASSI (Single Disperser CASSI) forward model: the spectral cube is dispersed
by a prism and coded by a mask. Unlike CASSI, SD-CASSI uses a single disperser
so each spectral band shifts in one direction.

x_true = (256, 256, 24) hyperspectral cube
y_ideal = (256, 279) coded dispersed snapshot  (256 + 24 - 1 = 279)
"""
import numpy as np
import h5py
import json
import struct
import zlib
from pathlib import Path
from scipy.io import loadmat
from skimage.transform import resize as sk_resize

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
OUT = BASE / "sd_cassi" / "standard"

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

# Load bird24 SD-CASSI data
print("Loading sd_cassi_bird24.mat...")
mat = loadmat(str(CACHE / "sd_cassi_bird24.mat"))
orig = mat["orig"].astype(np.float64)  # (703, 1021, 24)
mask_full = mat["mask"].astype(np.float64)  # (703, 1021, 24)
wavelength = mat["wavelength"].flatten()

print(f"  orig: {orig.shape}, mask: {mask_full.shape}")

if orig.max() > 1.0:
    orig = orig / orig.max()

n_bands = orig.shape[2]
h_full, w_full = orig.shape[:2]

# Create patches from the 703x1021 scene
# Use 256x256 crops from different locations
patches = [
    (0, 0),
    (0, 256),
    (0, 512),
    (256, 0),
    (256, 256),
    (256, 512),
    (447, 0),     # bottom-left (703-256=447)
    (447, 256),
    (447, 512),
    (200, 400),   # center-ish
]

# Clean
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("standard_sd_cassi_*.h5"):
    old.unlink()
img_dir = OUT / "images"
img_dir.mkdir(exist_ok=True)
for old in img_dir.glob("*.png"):
    old.unlink()

print(f"Building SD-CASSI samples ({n_bands} bands)...")

for idx, (r, c) in enumerate(patches):
    r = min(r, h_full - 256)
    c = min(c, w_full - 256)

    cube = orig[r:r+256, c:c+256, :].astype(np.float32)
    for b in range(n_bands):
        band = cube[:, :, b]
        lo, hi = band.min(), band.max()
        if hi - lo > 1e-12:
            cube[:, :, b] = (band - lo) / (hi - lo)

    # SD-CASSI forward: shift + mask + sum
    shifted_w = 256 + n_bands - 1
    y = np.zeros((256, shifted_w), dtype=np.float64)
    mask_2d = mask_full[r:r+256, c:c+256, 0]

    for b in range(n_bands):
        y[:, b:b+256] += cube[:, :, b] * mask_2d

    y = normalize_01(y.astype(np.float32))

    h5path = OUT / f"standard_sd_cassi_{idx:02d}.h5"
    with h5py.File(str(h5path), "w") as f:
        f.create_dataset("x_true", data=cube, compression="gzip")
        f.create_dataset("y_ideal", data=y, compression="gzip")
        f.create_dataset("mask", data=mask_2d.astype(np.float32), compression="gzip")
        f.create_dataset("wavelength", data=wavelength.astype(np.float32))
        f.attrs["modality"] = "sd_cassi"
        f.attrs["sample_index"] = idx
        f.attrs["patch_offset"] = f"({r},{c})"
        f.attrs["n_bands"] = n_bands
        f.attrs["source"] = f"DeSCI bird24 SD-CASSI 24-band hyperspectral (CAVE dataset, patch {idx})"
        f.attrs["reference"] = "Liu et al., TPAMI 2019; Wagadarikar et al., SD-CASSI, Applied Optics 2008"
        f.attrs["data_type"] = "real"

    # Save representative bands + RGB
    for b in [0, 6, 12, 18, 23]:
        write_png(cube[:, :, b], str(img_dir / f"x_true_{idx:02d}_band{b:02d}.png"))
    rgb_gray = 0.299*cube[:,:,min(20,n_bands-1)] + 0.587*cube[:,:,min(12,n_bands-1)] + 0.114*cube[:,:,min(4,n_bands-1)]
    write_png(rgb_gray, str(img_dir / f"x_true_{idx:02d}_rgb.png"))
    write_png(y, str(img_dir / f"y_meas_{idx:02d}.png"))

    print(f"  [{idx:02d}] patch ({r},{c}): cube {cube.shape} y {y.shape}")

meta = {
    "modality": "sd_cassi",
    "n_samples": len(patches),
    "x_shape": [256, 256, n_bands],
    "y_shape": [256, 256 + n_bands - 1],
    "n_spectral_bands": n_bands,
    "source": "DeSCI bird24 SD-CASSI (CAVE 24-band hyperspectral dataset)",
    "reference": "Liu et al., TPAMI 2019; Wagadarikar et al., SD-CASSI, Applied Optics 2008",
    "data_type": "real",
    "forward_model": "SD-CASSI: y(x, y+lambda) = sum_lambda[mask(x,y) * cube(x,y,lambda)]",
}
with open(OUT / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
with open(OUT / "spec.json", "w") as f:
    json.dump({"modality": "sd_cassi", "source": meta["source"], "reference": meta["reference"]}, f, indent=2)

hashes = set()
for i in range(len(patches)):
    with h5py.File(str(OUT / f"standard_sd_cassi_{i:02d}.h5"), "r") as f:
        hashes.add(hash(f["x_true"][:].tobytes()))
print(f"\nUnique hashes: {len(hashes)}/{len(patches)}")
print(f"Total PNGs: {len(list(img_dir.glob('*.png')))}")
print("SD-CASSI rebuild DONE")
