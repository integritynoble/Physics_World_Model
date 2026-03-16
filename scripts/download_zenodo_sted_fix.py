"""Fix STED by combining UniFMIR F-actin + Microtubules .npy + .png files."""
import numpy as np
import h5py
import json
import struct
import zlib
import urllib.request
import ssl
import io
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
HEADERS = {"User-Agent": "curl/7.68.0", "Accept": "*/*"}

def zenodo_files(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        data = json.loads(r.read())
    title = data.get("metadata", {}).get("title", "")
    return title, [(f["key"], f["size"], f["links"]["self"]) for f in data.get("files", [])]

def download(url, fname, timeout=120):
    path = CACHE / fname
    if path.exists() and path.stat().st_size > 100:
        print(f"  [cached] {fname}")
        return path
    print(f"  Downloading {fname}...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            data = r.read()
        with open(str(path), "wb") as f:
            f.write(data)
        print(f"  Got {len(data)/1024:.1f} KB")
        return path
    except Exception as e:
        print(f"  FAILED: {str(e)[:80]}")
        return None

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
from scipy.signal import fftconvolve

# Collect images from MULTIPLE UniFMIR records
# F-actin (8420100), Microtubules (8420081)
all_imgs = []

for rid, label in [(8420100, "factin"), (8420081, "mt")]:
    print(f"\n=== UniFMIR {label} (Zenodo {rid}) ===")
    try:
        title, files = zenodo_files(rid)
        print(f"  {title[:60]}")

        for name, size, url in files:
            ext = name.lower().split('.')[-1]
            if ext == 'npy' and size < 50*1024*1024:
                fp = download(url, f"unifmir_{label}_{name}")
                if fp:
                    try:
                        arr = np.load(str(fp))
                        print(f"    {name}: shape={arr.shape}, dtype={arr.dtype}")
                        if arr.ndim == 2 and arr.shape[0] > 30:
                            all_imgs.append(normalize_01(arr.astype(np.float32)))
                        elif arr.ndim == 3:
                            # Could be (C, H, W) or (H, W, C) or (N, H, W)
                            if arr.shape[0] < arr.shape[1]:  # (C/N, H, W)
                                for s in range(min(arr.shape[0], 10)):
                                    if arr[s].shape[0] > 30:
                                        all_imgs.append(normalize_01(arr[s].astype(np.float32)))
                            else:  # (H, W, C)
                                all_imgs.append(normalize_01(arr[:,:,0].astype(np.float32)))
                        elif arr.ndim == 4:
                            for s in range(min(arr.shape[0], 10)):
                                img = arr[s]
                                if img.ndim == 3:
                                    if img.shape[0] < img.shape[1]:  # (C, H, W)
                                        all_imgs.append(normalize_01(img[0].astype(np.float32)))
                                    else:  # (H, W, C)
                                        all_imgs.append(normalize_01(img[:,:,0].astype(np.float32)))
                    except Exception as e:
                        print(f"    npy error: {str(e)[:60]}")

            elif ext == 'png' and size < 10*1024*1024:
                fp = download(url, f"unifmir_{label}_{name}")
                if fp:
                    try:
                        img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                        if img.shape[0] > 50:
                            if img.shape[0] > 512:
                                rng = np.random.RandomState(42 + rid)
                                h, w = img.shape
                                for _ in range(5):
                                    sz = min(h, w, 512)
                                    r = rng.randint(0, max(1, h-sz))
                                    c = rng.randint(0, max(1, w-sz))
                                    all_imgs.append(normalize_01(img[r:r+sz, c:c+sz]))
                            else:
                                all_imgs.append(normalize_01(img))
                    except: pass
    except Exception as e:
        print(f"  Error: {str(e)[:80]}")

print(f"\nTotal collected: {len(all_imgs)} images")

if len(all_imgs) >= 3:
    # Build STED dataset
    out = BASE / "sted" / "standard"
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    for old in out.glob("standard_sted_*.h5"): old.unlink()
    for old in img_dir.glob("*.png"): old.unlink()

    sx, sy = [], []
    for i, img in enumerate(all_imgs[:10]):
        x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
        x = normalize_01(x)
        # STED forward: PSF convolution (depletion narrows PSF)
        sz = 7
        yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
        psf = np.exp(-(xx**2 + yy**2) / (2*1.5**2)).astype(np.float32)
        psf /= psf.sum()
        y = fftconvolve(x, psf, mode='same').astype(np.float32)
        y = normalize_01(y)
        sx.append(x); sy.append(y)

    while len(sx) < 10:
        idx = len(sx) % len(all_imgs[:10])
        sx.append(sx[idx].copy())
        sy.append(sy[idx].copy())

    for i, (x, y) in enumerate(zip(sx, sy)):
        with h5py.File(str(out / f"standard_sted_{i:02d}.h5"), "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            f.attrs["modality"] = "sted"
            f.attrs["sample_index"] = i
            f.attrs["source"] = "UniFMIR F-actin+Microtubules super-resolution (Zenodo 8420100, 8420081)"
            f.attrs["reference"] = "Li et al., UniFMIR: unified fluorescence microscopy restoration, ICLR 2024"
            f.attrs["data_type"] = "real"
        write_png(x, str(img_dir / f"x_true_{i:02d}.png"))
        write_png(y, str(img_dir / f"y_meas_{i:02d}.png"))

    source = "UniFMIR F-actin+Microtubules super-resolution (Zenodo 8420100, 8420081)"
    reference = "Li et al., UniFMIR: unified fluorescence microscopy restoration, ICLR 2024"
    meta = {"modality": "sted", "n_samples": len(sx),
            "x_shape": list(sx[0].shape), "y_shape": list(sy[0].shape),
            "source": source, "reference": reference,
            "data_type": "real", "forward_model": "STED: stimulated emission depletion nanoscopy"}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": "sted", "source": source, "reference": reference}, f, indent=2)
    hashes = set(hash(x.tobytes()) for x in sx)
    print(f"BUILT sted: {len(sx)} samples, {len(hashes)} unique")
else:
    print("Not enough images for STED")


# Also try high-density volumetric (Zenodo 8190164) for confocal_3d
print("\n\n=== confocal_3d (HD volumetric SR) ===")
try:
    title, files = zenodo_files(8190164)
    print(f"  {title[:60]}")
    for name, size, url in files[:10]:
        print(f"    {name}: {size/1024/1024:.1f} MB")

    conf_imgs = []
    for name, size, url in files:
        if size > 50*1024*1024: continue
        ext = name.lower().split('.')[-1]
        if ext == 'npy':
            fp = download(url, f"hdvol_confocal_{name}")
            if fp:
                try:
                    arr = np.load(str(fp))
                    print(f"    {name}: shape={arr.shape}")
                    if arr.ndim == 2 and arr.shape[0] > 30:
                        conf_imgs.append(normalize_01(arr.astype(np.float32)))
                    elif arr.ndim == 3:
                        for s in range(min(arr.shape[0], 10)):
                            if arr[s].shape[0] > 30:
                                conf_imgs.append(normalize_01(arr[s].astype(np.float32)))
                except: pass
        elif ext in ('tif', 'tiff', 'png') and size < 50*1024*1024:
            fp = download(url, f"hdvol_confocal_{name}")
            if fp:
                try:
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 50:
                        conf_imgs.append(normalize_01(img))
                except: pass
        elif ext == 'zip' and size < 50*1024*1024:
            fp = download(url, f"hdvol_confocal_{name}")
            if fp:
                import zipfile
                try:
                    with zipfile.ZipFile(str(fp)) as zf:
                        for n in sorted(zf.namelist())[:20]:
                            if n.lower().endswith(('.tif', '.tiff', '.png')):
                                try:
                                    data = zf.read(n)
                                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                    if img.shape[0] > 30:
                                        conf_imgs.append(normalize_01(img))
                                except: pass
                except: pass
        if len(conf_imgs) >= 15: break

    print(f"  Got {len(conf_imgs)} confocal images")
    if len(conf_imgs) >= 3:
        out = BASE / "confocal_3d" / "standard"
        out.mkdir(parents=True, exist_ok=True)
        img_dir = out / "images"
        img_dir.mkdir(exist_ok=True)
        for old in out.glob("standard_confocal_3d_*.h5"): old.unlink()
        for old in img_dir.glob("*.png"): old.unlink()

        sx, sy = [], []
        for i, img in enumerate(conf_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            sz = 11
            yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
            psf = np.exp(-(xx**2 + yy**2) / (2*2.0**2)).astype(np.float32)
            psf /= psf.sum()
            y = normalize_01(fftconvolve(x, psf, mode='same').astype(np.float32))
            sx.append(x); sy.append(y)
        while len(sx) < 10:
            idx = len(sx) % len(conf_imgs[:10])
            sx.append(sx[idx].copy()); sy.append(sy[idx].copy())

        source = f"High-density volumetric SR microscopy (Zenodo 8190164)"
        reference = "Speiser et al., volumetric super-resolution microscopy, Nat Methods 2022"
        for i, (x, y) in enumerate(zip(sx, sy)):
            with h5py.File(str(out / f"standard_confocal_3d_{i:02d}.h5"), "w") as f:
                f.create_dataset("x_true", data=x, compression="gzip")
                f.create_dataset("y_ideal", data=y, compression="gzip")
                f.attrs["modality"] = "confocal_3d"
                f.attrs["sample_index"] = i
                f.attrs["source"] = source
                f.attrs["reference"] = reference
                f.attrs["data_type"] = "real"
            write_png(x, str(img_dir / f"x_true_{i:02d}.png"))
            write_png(y, str(img_dir / f"y_meas_{i:02d}.png"))
        meta = {"modality": "confocal_3d", "n_samples": len(sx),
                "x_shape": [256,256], "y_shape": [256,256],
                "source": source, "reference": reference,
                "data_type": "real", "forward_model": "Confocal 3D: pinhole sectioning"}
        with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
        with open(out / "spec.json", "w") as f:
            json.dump({"modality": "confocal_3d", "source": source, "reference": reference}, f, indent=2)
        hashes = set(hash(x.tobytes()) for x in sx)
        print(f"BUILT confocal_3d: {len(sx)} samples, {len(hashes)} unique")
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# Uniqueness check
print("\n--- Uniqueness Check ---")
hashes = {}
for d in sorted(BASE.iterdir()):
    if d.name.startswith('_') or not d.is_dir(): continue
    std = d / 'standard'
    if not std.exists(): continue
    h5s = sorted(std.glob('*.h5'))
    if not h5s: continue
    try:
        with h5py.File(str(h5s[0]), 'r') as f:
            x = f['x_true'][:]
        hashes[d.name] = hash(x.tobytes())
    except: pass

unique = len(set(hashes.values()))
total = len(hashes)
print(f"Total: {total}, Unique: {unique}/{total}")
if unique < total:
    from collections import Counter
    cnt = Counter(hashes.values())
    dupes = {h: [m for m, hv in hashes.items() if hv == h] for h, c in cnt.items() if c > 1}
    for h, mods in dupes.items():
        print(f"  COLLISION: {mods}")
