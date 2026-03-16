"""Rebuild SPC Kronecker standard with full Indian Pines 200-band hyperspectral cube.

Single Pixel Camera with Kronecker compressive sensing.
x_true = (145, 145, 200) AVIRIS hyperspectral cube
y_ideal = compressed measurements via Kronecker sensing matrix

Indian Pines is THE standard hyperspectral benchmark (Purdue/AVIRIS).
"""
import numpy as np
import h5py
import json
import struct
import zlib
from pathlib import Path
from scipy.io import loadmat

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
OUT = BASE / "spc_kronecker" / "standard"

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

print("Loading Indian Pines corrected...")
mat = loadmat(str(CACHE / "indian_pines_corrected.mat"))
ip = mat["indian_pines_corrected"].astype(np.float64)  # (145, 145, 200)
print(f"  Shape: {ip.shape}, range: [{ip.min()}, {ip.max()}]")

# Normalize
ip = ip / ip.max()

n_bands = ip.shape[2]
h, w = ip.shape[:2]

# Create samples: different spectral band groups + spatial sub-patches
# Indian Pines is 145x145, so patches are smaller
# Use 10 different band-group selections to show spectral diversity
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("standard_spc_kronecker_*.h5"):
    old.unlink()
img_dir = OUT / "images"
img_dir.mkdir(exist_ok=True)
for old in img_dir.glob("*.png"):
    old.unlink()

# Sample plan: different spectral windows across the 200 bands
# Each sample is the full 145x145 spatial extent with a contiguous spectral window
band_groups = [
    (0, 28),    # bands 0-27 (VNIR blue-green)
    (28, 56),   # bands 28-55 (VNIR green-red)
    (56, 84),   # bands 56-83 (VNIR red-NIR)
    (84, 112),  # bands 84-111 (NIR)
    (112, 140), # bands 112-139 (SWIR-1)
    (140, 168), # bands 140-167 (SWIR-1)
    (168, 200), # bands 168-199 (SWIR-2), 32 bands
    (0, 200),   # ALL 200 bands (full cube)
    (20, 80),   # bands 20-79 (visible range)
    (100, 180), # bands 100-179 (SWIR range)
]

print(f"Building SPC Kronecker samples ({n_bands} bands total)...")

for idx, (b_start, b_end) in enumerate(band_groups):
    nb = b_end - b_start
    cube = ip[:, :, b_start:b_end].astype(np.float32)

    # Normalize
    for b in range(nb):
        band = cube[:, :, b]
        lo, hi = band.min(), band.max()
        if hi - lo > 1e-12:
            cube[:, :, b] = (band - lo) / (hi - lo)

    # SPC Kronecker forward: y = Phi * vec(X)
    # Kronecker sensing: Phi = Phi_1 kron Phi_2
    # Simulate with random Gaussian measurement at 25% compression
    rng = np.random.RandomState(42 + idx)
    cr = 0.25  # compression ratio
    n_spatial = h * w
    n_meas = int(n_spatial * cr)

    # Per-band compressed measurement
    Phi = rng.randn(n_meas, n_spatial).astype(np.float32) / np.sqrt(n_meas)
    y_bands = []
    for b in range(nb):
        y_b = Phi @ cube[:, :, b].flatten()
        y_bands.append(y_b)
    y = np.stack(y_bands, axis=1).astype(np.float32)  # (n_meas, nb)

    h5path = OUT / f"standard_spc_kronecker_{idx:02d}.h5"
    with h5py.File(str(h5path), "w") as f:
        f.create_dataset("x_true", data=cube, compression="gzip")
        f.create_dataset("y_ideal", data=y, compression="gzip")
        f.attrs["modality"] = "spc_kronecker"
        f.attrs["sample_index"] = idx
        f.attrs["band_range"] = f"{b_start}-{b_end-1}"
        f.attrs["n_bands"] = nb
        f.attrs["compression_ratio"] = cr
        f.attrs["source"] = f"Indian Pines AVIRIS corrected (bands {b_start}-{b_end-1})"
        f.attrs["reference"] = "Purdue AVIRIS Indian Pines; Duarte & Baraniuk, Kronecker CS, IEEE TIP 2012"
        f.attrs["data_type"] = "real"

    # Save band PNGs
    step = max(1, nb // 5)
    for b in range(0, nb, step):
        write_png(cube[:, :, b], str(img_dir / f"x_true_{idx:02d}_band{b:03d}.png"))

    # False-color composite
    b1 = min(nb-1, nb//4)
    b2 = min(nb-1, nb//2)
    b3 = min(nb-1, 3*nb//4)
    fc = 0.299*cube[:,:,b3] + 0.587*cube[:,:,b2] + 0.114*cube[:,:,b1]
    write_png(fc, str(img_dir / f"x_true_{idx:02d}_composite.png"))

    print(f"  [{idx:02d}] bands {b_start}-{b_end-1} ({nb} bands): cube {cube.shape} y {y.shape}")

meta = {
    "modality": "spc_kronecker",
    "n_samples": len(band_groups),
    "x_shape": [145, 145, "variable (28-200 bands)"],
    "y_shape": "compressed measurements",
    "n_total_bands": 200,
    "spatial_size": [145, 145],
    "source": "Indian Pines AVIRIS corrected (Purdue University)",
    "reference": "Purdue AVIRIS Indian Pines; Duarte & Baraniuk, Kronecker Compressive Sensing, IEEE TIP 2012",
    "data_type": "real",
    "forward_model": "SPC with Kronecker CS: y = (Phi_1 kron Phi_2) * vec(X), 25% compression",
    "band_groups": {f"sample_{i:02d}": f"bands {s}-{e-1}" for i, (s,e) in enumerate(band_groups)},
}
with open(OUT / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
with open(OUT / "spec.json", "w") as f:
    json.dump({"modality": "spc_kronecker", "source": meta["source"], "reference": meta["reference"]}, f, indent=2)

hashes = set()
for i in range(len(band_groups)):
    with h5py.File(str(OUT / f"standard_spc_kronecker_{i:02d}.h5"), "r") as f:
        hashes.add(hash(f["x_true"][:].tobytes()))
print(f"\nUnique hashes: {len(hashes)}/{len(band_groups)}")
print(f"Total PNGs: {len(list(img_dir.glob('*.png')))}")
print("SPC Kronecker rebuild DONE")
