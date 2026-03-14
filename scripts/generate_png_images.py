"""Generate PNG images from x_true in all standard HDF5 files."""
import numpy as np
import h5py
from pathlib import Path

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")


def to_uint8(x):
    x = np.nan_to_num(x, 0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def write_png(arr_2d, path):
    """Write a grayscale 2D array as PNG using minimal pure-Python PNG writer."""
    import struct, zlib
    u = to_uint8(arr_2d)
    h, w = u.shape

    def chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
    raw = b''
    for row in u:
        raw += b'\x00' + row.tobytes()
    idat = chunk(b'IDAT', zlib.compress(raw, 9))
    iend = chunk(b'IEND', b'')
    with open(str(path), 'wb') as f:
        f.write(sig + ihdr + idat + iend)


if __name__ == "__main__":
    count = 0
    for d in sorted(BASE.iterdir()):
        std = d / "standard"
        if not std.exists():
            continue
        h5s = sorted(std.glob("*.h5"))
        if not h5s:
            continue
        img_dir = std / "images"
        img_dir.mkdir(exist_ok=True)

        for h5_path in h5s:
            with h5py.File(str(h5_path), "r") as f:
                x = f["x_true"][:]
                idx = f.attrs.get("sample_index", 0)

            # Handle non-2D: reduce to 2D
            while x.ndim > 2:
                if x.shape[-1] in [2, 3, 4] and x.ndim == 3:
                    x = x.mean(axis=-1)
                else:
                    x = x[x.shape[0] // 2]
            if x.ndim == 1:
                # 1D signal: reshape to a narrow 2D image
                n = len(x)
                side = int(np.ceil(np.sqrt(n)))
                padded = np.zeros(side * side, dtype=x.dtype)
                padded[:n] = x
                x = padded.reshape(side, side)

            png_path = img_dir / f"x_true_{int(idx):02d}.png"
            write_png(x, png_path)

        count += 1
        if count % 20 == 0:
            print(f"  {count} modalities done...")

    print(f"Generated PNG images for {count} modalities")
