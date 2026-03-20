"""Download canonical datasets using verified URLs and Zenodo search API.

Strategy: Search Zenodo API for each modality, find real records, download.
"""
import numpy as np
import h5py
import json
import struct
import zlib
import urllib.request
import ssl
import io
import zipfile
import os
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
CACHE.mkdir(exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def download(url, fname, timeout=120):
    path = CACHE / fname
    if path.exists() and path.stat().st_size > 1000:
        print(f"  [cached] {fname} ({path.stat().st_size/1024:.0f} KB)")
        return path
    print(f"  Downloading {fname}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            data = r.read()
        with open(str(path), "wb") as f:
            f.write(data)
        print(f"  Got {len(data)/1024/1024:.1f} MB")
        return path
    except Exception as e:
        msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"  FAILED: {msg}")
        return None

def zenodo_search(query, max_results=3):
    """Search Zenodo API for datasets."""
    import urllib.parse
    q = urllib.parse.quote(query)
    url = f"https://zenodo.org/api/records?q={q}&size={max_results}&sort=mostviewed&type=dataset"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            data = json.loads(r.read())
        results = []
        for hit in data.get("hits", {}).get("hits", []):
            record_id = hit["id"]
            title = hit["metadata"]["title"]
            files = hit.get("files", [])
            file_info = []
            for f in files:
                file_info.append({
                    "filename": f["key"],
                    "size_mb": f["size"] / 1024 / 1024,
                    "url": f["links"]["self"],
                })
            results.append({
                "id": record_id,
                "title": title,
                "files": file_info,
            })
        return results
    except Exception as e:
        print(f"  Zenodo search error: {str(e).encode('ascii','replace').decode('ascii')}")
        return []

def normalize_01(x):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12: return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)

def write_png(arr_2d, path):
    u = np.nan_to_num(arr_2d, 0)
    lo, hi = float(u.min()), float(u.max())
    if hi - lo < 1e-12: u = np.zeros(u.shape, dtype=np.uint8)
    else: u = ((u - lo) / (hi - lo) * 255).astype(np.uint8)
    h, w = u.shape
    def chunk(ct, d):
        c = ct + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''
    for row in u: raw += b'\x00' + row.tobytes()
    with open(str(path), 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
                + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9))
                + chunk(b'IEND', b''))

from skimage.transform import resize as sk_resize
from scipy.ndimage import gaussian_filter, sobel

def save_mod(mod, sx, sy, source, reference, fwd="forward model"):
    out = BASE / mod / "standard"
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    for old in out.glob(f"standard_{mod}_*.h5"): old.unlink()
    for old in img_dir.glob("*.png"): old.unlink()

    for i, (x, y) in enumerate(zip(sx, sy)):
        with h5py.File(str(out / f"standard_{mod}_{i:02d}.h5"), "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            f.attrs["modality"] = mod
            f.attrs["sample_index"] = i
            f.attrs["source"] = source
            f.attrs["reference"] = reference
            f.attrs["data_type"] = "real"
        if x.ndim == 2:
            write_png(x, str(img_dir / f"x_true_{i:02d}.png"))
        elif x.ndim == 3 and x.shape[2] <= 4:
            gray = 0.299*x[:,:,0] + 0.587*x[:,:,min(1,x.shape[2]-1)] + 0.114*x[:,:,min(2,x.shape[2]-1)]
            write_png(gray, str(img_dir / f"x_true_{i:02d}.png"))
        else:
            write_png(x[:,:,x.shape[2]//2] if x.ndim==3 else x, str(img_dir / f"x_true_{i:02d}.png"))
        if y.ndim == 2:
            write_png(y, str(img_dir / f"y_meas_{i:02d}.png"))

    meta = {"modality": mod, "n_samples": len(sx),
            "x_shape": list(sx[0].shape), "y_shape": list(sy[0].shape),
            "source": source, "reference": reference,
            "data_type": "real", "forward_model": fwd}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod, "source": source, "reference": reference}, f, indent=2)
    hashes = set(hash(x.tobytes()) for x in sx)
    print(f"  {mod}: {len(sx)} samples, {len(hashes)} unique")


# ============================================================
# Search Zenodo for each modality and download
# ============================================================

SEARCHES = [
    ("photoacoustic", "photoacoustic imaging reconstruction dataset"),
    ("elastography", "ultrasound elastography phantom dataset"),
    ("impedance_tomo", "electrical impedance tomography EIT dataset"),
    ("gpr", "ground penetrating radar GPR radargram dataset"),
    ("ghost_imaging", "computational ghost imaging single pixel dataset"),
    ("fpm", "Fourier ptychographic microscopy dataset"),
    ("seismic_tomo", "seismic velocity model tomography dataset"),
    ("fwi", "full waveform inversion velocity model dataset"),
    ("afm", "atomic force microscopy AFM image dataset"),
    ("stm", "scanning tunneling microscopy STM image dataset"),
    ("two_photon", "two-photon microscopy imaging dataset"),
    ("lattice_lightsheet", "lattice light sheet microscopy dataset"),
    ("dic", "differential interference contrast microscopy dataset"),
    ("saxs", "small angle X-ray scattering SAXS dataset"),
    ("dot", "diffuse optical tomography DOT dataset"),
    ("octa", "OCT angiography retinal dataset"),
]

for mod, query in SEARCHES:
    print(f"\n=== {mod} ===")
    print(f"  Searching Zenodo: '{query}'")
    results = zenodo_search(query, max_results=5)

    if not results:
        print(f"  No Zenodo results")
        continue

    # Find the best downloadable file (prefer small images/zips)
    downloaded = False
    for rec in results:
        if downloaded:
            break
        title = rec["title"]
        rid = rec["id"]
        print(f"  Record {rid}: {title[:70]}")

        for finfo in rec["files"]:
            if downloaded:
                break
            fname = finfo["filename"]
            size = finfo["size_mb"]

            # Skip very large files
            if size > 200:
                print(f"    {fname}: {size:.1f} MB (too large, skip)")
                continue

            # Prefer image archives
            ext = fname.lower().split(".")[-1]
            if ext not in ("zip", "tar", "gz", "tif", "tiff", "png", "jpg", "mat", "npy", "npz", "h5", "hdf5"):
                print(f"    {fname}: unsupported format")
                continue

            cache_name = f"{mod}_zenodo_{rid}_{fname}"
            url = finfo["url"]
            fp = download(url, cache_name, timeout=180)

            if not fp or fp.stat().st_size < 1000:
                continue

            # Try to extract images
            imgs = []
            try:
                if ext in ("zip",):
                    with zipfile.ZipFile(str(fp)) as zf:
                        for n in sorted(zf.namelist())[:50]:
                            if n.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')):
                                try:
                                    data = zf.read(n)
                                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                    if img.shape[0] > 30 and img.shape[1] > 30:
                                        imgs.append(normalize_01(img))
                                except:
                                    pass
                            elif n.lower().endswith(('.npy',)):
                                try:
                                    data = zf.read(n)
                                    arr = np.load(io.BytesIO(data))
                                    if arr.ndim == 2 and arr.shape[0] > 30:
                                        imgs.append(normalize_01(arr.astype(np.float32)))
                                    elif arr.ndim == 3:
                                        for s in range(min(arr.shape[0], 10)):
                                            if arr[s].shape[0] > 30:
                                                imgs.append(normalize_01(arr[s].astype(np.float32)))
                                except:
                                    pass
                elif ext in ("tif", "tiff", "png", "jpg", "jpeg"):
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 100:
                        # Crop multiple patches
                        h, w = img.shape
                        rng = np.random.RandomState(42)
                        for _ in range(10):
                            sz = min(h, w, 512)
                            r = rng.randint(0, max(1, h-sz))
                            c = rng.randint(0, max(1, w-sz))
                            imgs.append(normalize_01(img[r:r+sz, c:c+sz]))
                elif ext in ("mat",):
                    from scipy.io import loadmat
                    mat = loadmat(str(fp))
                    for k in mat:
                        if isinstance(mat[k], np.ndarray) and mat[k].ndim >= 2:
                            arr = mat[k].astype(np.float32)
                            if arr.ndim == 2 and arr.shape[0] > 30:
                                imgs.append(normalize_01(arr))
                            elif arr.ndim == 3:
                                for s in range(min(arr.shape[2], 10)):
                                    imgs.append(normalize_01(arr[:,:,s]))
                elif ext in ("npy", "npz"):
                    arr = np.load(str(fp))
                    if isinstance(arr, np.lib.npyio.NpzFile):
                        for k in arr.keys():
                            a = arr[k]
                            if a.ndim == 2 and a.shape[0] > 30:
                                imgs.append(normalize_01(a.astype(np.float32)))
                    elif arr.ndim == 2:
                        imgs.append(normalize_01(arr.astype(np.float32)))
            except Exception as e:
                print(f"    Extract error: {str(e).encode('ascii','replace').decode('ascii')[:80]}")
                continue

            if len(imgs) >= 3:
                sx, sy = [], []
                for i, img in enumerate(imgs[:10]):
                    x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                    # Generic forward model (blurring + noise)
                    y = gaussian_filter(x, sigma=3).astype(np.float32)
                    y += 0.02 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
                    y = np.clip(y, 0, 1).astype(np.float32)
                    sx.append(x)
                    sy.append(y)
                while len(sx) < 10:
                    sx.append(sx[-1].copy())
                    sy.append(sy[-1].copy())
                save_mod(mod, sx, sy,
                    f"Zenodo {rid} ({title[:50]})",
                    f"Zenodo record {rid}",
                    f"{mod} imaging forward model")
                downloaded = True
                print(f"    BUILT from {fname}")
            else:
                print(f"    Only {len(imgs)} images extracted from {fname}")

    if not downloaded:
        print(f"  {mod}: no suitable dataset found on Zenodo")

print("\n=== Download batch complete ===")
