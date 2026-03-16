"""Rebuild CASSI standard with full 31-band hyperspectral cube from DeSCI toy31.

CASSI forward model: spectral bands are shifted and coded by a binary mask,
then summed into a single 2D coded snapshot measurement.

x_true = (256, 256, 31) hyperspectral cube
y_ideal = (256, 542) coded aperture spectral snapshot
mask = (256, 256) binary coded aperture
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
OUT = BASE / "cassi" / "standard"

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

# Load toy31 CASSI data
print("Loading toy31_cassi.mat...")
mat = loadmat(str(CACHE / "cassi_toy31.mat"))
orig = mat["orig"].astype(np.float64)  # (512, 512, 31)
mask_full = mat["mask"].astype(np.float64)  # (512, 512, 31)
meas = mat["meas"].astype(np.float64)  # (512, 512) - real measurement
wavelength = mat["wavelength"].flatten()  # (31,)

print(f"  orig: {orig.shape}, mask: {mask_full.shape}, meas: {meas.shape}")
print(f"  wavelengths: {wavelength[:5]}...{wavelength[-5:]}")

# Normalize to [0,1]
if orig.max() > 1.0:
    orig = orig / orig.max()

# Create spatial patches from the 512x512 scene for multiple samples
# Standard CASSI benchmark uses different spatial regions
patch_size = 256
patches = [
    (0, 0),       # top-left
    (0, 256),     # top-right
    (256, 0),     # bottom-left
    (256, 256),   # bottom-right
    (128, 128),   # center
    (0, 128),     # top-center
    (128, 0),     # left-center
    (128, 256),   # right-center
    (256, 128),   # bottom-center
    (64, 64),     # offset
]

# Clean old files
for old in OUT.glob("standard_cassi_*.h5"):
    old.unlink()
img_dir = OUT / "images"
img_dir.mkdir(exist_ok=True)
for old in img_dir.glob("*.png"):
    old.unlink()

print("Building CASSI samples with full spectral cubes...")
n_bands = orig.shape[2]

for idx, (r, c) in enumerate(patches):
    # Extract 256x256x31 patch
    cube = orig[r:r+patch_size, c:c+patch_size, :].astype(np.float32)

    # Normalize each band
    for b in range(n_bands):
        band = cube[:, :, b]
        lo, hi = band.min(), band.max()
        if hi - lo > 1e-12:
            cube[:, :, b] = (band - lo) / (hi - lo)

    # CASSI forward model: shift each band by its index, apply mask, sum
    # y(x, y+lambda) = sum_lambda [mask(x,y) * cube(x, y, lambda)]
    # Shifted measurement has width = 256 + 31 - 1 = 286
    shifted_width = patch_size + n_bands - 1
    y = np.zeros((patch_size, shifted_width), dtype=np.float64)
    mask_2d = mask_full[r:r+patch_size, c:c+patch_size, 0]  # Use first mask plane

    for b in range(n_bands):
        y[:, b:b+patch_size] += cube[:, :, b] * mask_2d

    y = normalize_01(y.astype(np.float32))

    # Save H5
    h5path = OUT / f"standard_cassi_{idx:02d}.h5"
    with h5py.File(str(h5path), "w") as f:
        f.create_dataset("x_true", data=cube, compression="gzip")
        f.create_dataset("y_ideal", data=y, compression="gzip")
        f.create_dataset("mask", data=mask_2d.astype(np.float32), compression="gzip")
        f.create_dataset("wavelength", data=wavelength.astype(np.float32))
        f.attrs["modality"] = "cassi"
        f.attrs["sample_index"] = idx
        f.attrs["patch_offset"] = f"({r},{c})"
        f.attrs["n_bands"] = n_bands
        f.attrs["source"] = f"DeSCI toy31 CASSI 31-band hyperspectral (CAVE dataset, patch {idx})"
        f.attrs["reference"] = "Liu et al., TPAMI 2019; CAVE hyperspectral dataset (Yasuma et al. 2010)"
        f.attrs["data_type"] = "real"

    # Save band PNGs (representative bands: blue, green, red, NIR + more)
    band_indices = [0, 5, 10, 15, 20, 25, 30]
    for b in band_indices:
        if b < n_bands:
            write_png(cube[:, :, b], str(img_dir / f"x_true_{idx:02d}_band{b:02d}.png"))

    # Save RGB composite
    rgb = np.stack([cube[:,:,25], cube[:,:,15], cube[:,:,5]], axis=2)
    rgb_gray = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    write_png(rgb_gray, str(img_dir / f"x_true_{idx:02d}_rgb.png"))

    # Save measurement
    write_png(y, str(img_dir / f"y_meas_{idx:02d}.png"))

    print(f"  [{idx:02d}] patch ({r},{c}): cube {cube.shape} y {y.shape}")

# Metadata
meta = {
    "modality": "cassi",
    "n_samples": len(patches),
    "x_shape": [256, 256, 31],
    "y_shape": [256, 286],
    "mask_shape": [256, 256],
    "n_spectral_bands": 31,
    "source": "DeSCI toy31 CASSI (CAVE 31-band hyperspectral dataset)",
    "reference": "Liu et al., Rank Minimization for Snapshot Compressive Imaging, TPAMI 2019; Yasuma et al., Generalized Assorted Pixel Camera, CAVE dataset 2010",
    "data_type": "real",
    "forward_model": "CASSI: y(x, y+lambda) = sum_lambda[mask(x,y) * cube(x,y,lambda)], spectral shift + coded aperture",
    "wavelength_range": f"{wavelength[0]:.0f}-{wavelength[-1]:.0f} nm",
}
with open(OUT / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

spec = {
    "modality": "cassi",
    "source": "DeSCI toy31 CASSI (CAVE 31-band hyperspectral dataset)",
    "reference": "Liu et al., TPAMI 2019; Yasuma et al., CAVE dataset 2010",
}
with open(OUT / "spec.json", "w") as f:
    json.dump(spec, f, indent=2)

# Verify
hashes = set()
for i in range(len(patches)):
    with h5py.File(str(OUT / f"standard_cassi_{i:02d}.h5"), "r") as f:
        h = hash(f["x_true"][:].tobytes())
        hashes.add(h)
n_imgs = len(list(img_dir.glob("*.png")))
print(f"\nUnique hashes: {len(hashes)}/{len(patches)}")
print(f"Total PNGs: {n_imgs}")
print("CASSI rebuild DONE")
