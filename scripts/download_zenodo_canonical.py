"""Download canonical datasets from Zenodo using correct curl User-Agent.

Zenodo blocks Mozilla UA but allows curl UA.
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

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {"User-Agent": "curl/7.68.0", "Accept": "*/*"}

def zenodo_files(record_id):
    """Get file list from Zenodo record."""
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        data = json.loads(r.read())
    return [(f["key"], f["size"], f["links"]["self"]) for f in data.get("files", [])]

def download(url, fname, timeout=300):
    path = CACHE / fname
    if path.exists() and path.stat().st_size > 1000:
        print(f"  [cached] {fname} ({path.stat().st_size/1024:.0f} KB)")
        return path
    print(f"  Downloading {fname}...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
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
        write_png(x, str(img_dir / f"x_true_{i:02d}.png"))
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
# 1. AFM - Zenodo 60434: AFM images of various specimens
# ============================================================
print("\n=== afm ===")
try:
    files = zenodo_files(60434)
    png_files = [(name, url) for name, size, url in files if name.endswith('.png')]
    print(f"  Found {len(png_files)} PNG files")

    afm_imgs = []
    for name, url in png_files:
        fp = download(url, f"afm_z60434_{name}")
        if fp:
            try:
                img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                if img.shape[0] > 30 and img.shape[1] > 30:
                    afm_imgs.append(normalize_01(img))
            except: pass

    print(f"  Loaded {len(afm_imgs)} AFM images")
    if afm_imgs:
        sx, sy = [], []
        for i, img in enumerate(afm_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            y = gaussian_filter(x, sigma=2).astype(np.float32)
            y += 0.03 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
        save_mod("afm", sx, sy,
            "Zenodo 60434 AFM images of various specimens (Keysight Technologies)",
            "Oxvig et al., Structure Assisted CS of Undersampled AFM Images, 2017",
            "AFM: tip convolution + thermal noise")
except Exception as e:
    print(f"  AFM error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# 2. PHOTOACOUSTIC - Duke PAM dataset (Zenodo 4042171)
# ============================================================
print("\n=== photoacoustic ===")
try:
    files = zenodo_files(4042171)
    print(f"  Duke PAM files:")
    for name, size, url in files:
        print(f"    {name}: {size/1024/1024:.1f} MB")

    # Download the smallest image files or zip
    pa_imgs = []
    for name, size, url in files:
        if size > 100*1024*1024:  # skip >100MB
            continue
        if name.endswith(('.zip',)):
            fp = download(url, f"pa_duke_{name}")
            if fp:
                try:
                    with zipfile.ZipFile(str(fp)) as zf:
                        for n in sorted(zf.namelist())[:30]:
                            if n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                                try:
                                    data = zf.read(n)
                                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                    if img.shape[0] > 50:
                                        pa_imgs.append(normalize_01(img))
                                except: pass
                except: pass
        elif name.endswith(('.png', '.jpg', '.tif', '.tiff')) and size < 50*1024*1024:
            fp = download(url, f"pa_duke_{name}")
            if fp:
                try:
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 50:
                        pa_imgs.append(normalize_01(img))
                except: pass
        if len(pa_imgs) >= 15:
            break

    print(f"  Loaded {len(pa_imgs)} PA images")
    if pa_imgs:
        sx, sy = [], []
        for i, img in enumerate(pa_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            edges = np.abs(sobel(x, axis=0)) + np.abs(sobel(x, axis=1))
            y = normalize_01(gaussian_filter(edges, sigma=2).astype(np.float32))
            sx.append(x); sy.append(y)
        while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
        save_mod("photoacoustic", sx, sy,
            "Duke PAM dataset (Zenodo 4042171, mouse brain photoacoustic microscopy)",
            "Vu et al., Duke PAM: OR-PAM mouse brain vasculature, Zenodo 2020",
            "Photoacoustic: optical absorption -> acoustic wave -> reconstruct")
except Exception as e:
    print(f"  PA error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# 3. OCTA - ROSE dataset (Zenodo 12775880)
# ============================================================
print("\n=== octa ===")
try:
    files = zenodo_files(12775880)
    print(f"  ROSE files:")
    for name, size, url in files:
        print(f"    {name}: {size/1024/1024:.1f} MB")

    octa_imgs = []
    for name, size, url in files:
        if size > 100*1024*1024: continue
        if name.endswith('.zip'):
            fp = download(url, f"octa_rose_{name}")
            if fp:
                try:
                    with zipfile.ZipFile(str(fp)) as zf:
                        for n in sorted(zf.namelist())[:30]:
                            if n.lower().endswith(('.png', '.tif', '.jpg', '.bmp')):
                                try:
                                    data = zf.read(n)
                                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                    if img.shape[0] > 50:
                                        octa_imgs.append(normalize_01(img))
                                except: pass
                except: pass
        if len(octa_imgs) >= 15: break

    print(f"  Loaded {len(octa_imgs)} OCTA images")
    if octa_imgs:
        sx, sy = [], []
        for i, img in enumerate(octa_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            y = gaussian_filter(x, sigma=1.5).astype(np.float32)
            y += 0.02 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
        save_mod("octa", sx, sy,
            "ROSE retinal OCTA dataset (Zenodo 12775880, vessel segmentation)",
            "Ma et al., ROSE: Retinal OCT-Angiography Vessel Segmentation, IEEE TMI 2021",
            "OCTA: decorrelation-based angiography from OCT volumes")
except Exception as e:
    print(f"  OCTA error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# 4. EIT - KTC 2023 (Zenodo 10986692)
# ============================================================
print("\n=== impedance_tomo (EIT) ===")
try:
    files = zenodo_files(10986692)
    print(f"  KTC 2023 files:")
    for name, size, url in files:
        print(f"    {name}: {size/1024/1024:.1f} MB")

    # Download truth images (phantoms)
    eit_imgs = []
    for name, size, url in files:
        if size > 50*1024*1024: continue
        if name.endswith(('.png', '.jpg', '.mat')):
            fp = download(url, f"eit_ktc_{name}")
            if fp and name.endswith('.png'):
                try:
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 30:
                        eit_imgs.append(normalize_01(img))
                except: pass
        elif name.endswith('.zip'):
            fp = download(url, f"eit_ktc_{name}")
            if fp:
                try:
                    with zipfile.ZipFile(str(fp)) as zf:
                        for n in sorted(zf.namelist())[:20]:
                            if n.lower().endswith(('.png', '.jpg', '.bmp')):
                                try:
                                    data = zf.read(n)
                                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                    if img.shape[0] > 30:
                                        eit_imgs.append(normalize_01(img))
                                except: pass
                except: pass
        if len(eit_imgs) >= 15: break

    print(f"  Loaded {len(eit_imgs)} EIT images")
    if eit_imgs:
        sx, sy = [], []
        for i, img in enumerate(eit_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            y = gaussian_filter(x, sigma=8).astype(np.float32)
            y += 0.05 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
        save_mod("impedance_tomo", sx, sy,
            "KTC 2023 EIT phantom dataset (Zenodo 10986692)",
            "Kuopio Tomography Challenge 2023, Hauptmann et al., 2D EIT reconstruction",
            "EIT: reconstruct conductivity from boundary voltage measurements")
except Exception as e:
    print(f"  EIT error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# 5. Lattice light sheet - napari-lattice (Zenodo 14903188)
# ============================================================
print("\n=== lattice_lightsheet ===")
try:
    files = zenodo_files(14903188)
    print(f"  napari-lattice files:")
    for name, size, url in files:
        if size < 50*1024*1024:
            print(f"    {name}: {size/1024/1024:.1f} MB")

    llsm_imgs = []
    for name, size, url in files:
        if size > 30*1024*1024: continue
        if name.endswith(('.tif', '.tiff', '.png')):
            fp = download(url, f"llsm_{name}")
            if fp:
                try:
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 30:
                        llsm_imgs.append(normalize_01(img))
                except: pass
        elif name.endswith('.zip'):
            fp = download(url, f"llsm_{name}")
            if fp:
                try:
                    with zipfile.ZipFile(str(fp)) as zf:
                        for n in sorted(zf.namelist())[:20]:
                            if n.lower().endswith(('.tif', '.tiff', '.png')):
                                try:
                                    data = zf.read(n)
                                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                    if img.shape[0] > 30:
                                        llsm_imgs.append(normalize_01(img))
                                except: pass
                except: pass
        if len(llsm_imgs) >= 15: break

    print(f"  Loaded {len(llsm_imgs)} LLSM images")
    if llsm_imgs:
        sx, sy = [], []
        for i, img in enumerate(llsm_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            y = gaussian_filter(x, sigma=1.5).astype(np.float32)
            y += 0.02 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
        save_mod("lattice_lightsheet", sx, sy,
            "napari-lattice RBC sample (Zenodo 14903188, Zeiss Lattice LightSheet 7)",
            "Chen et al., Lattice light sheet microscopy, Science 2014",
            "Lattice light sheet: dithered ultrathin sheet excitation")
except Exception as e:
    print(f"  LLSM error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# 6. GPR - TU1208 IFSTTAR (Zenodo 1211173)
# ============================================================
print("\n=== gpr ===")
try:
    files = zenodo_files(1211173)
    print(f"  TU1208 GPR files:")
    for name, size, url in files:
        print(f"    {name}: {size/1024/1024:.1f} MB")

    gpr_imgs = []
    for name, size, url in files:
        if size > 50*1024*1024: continue
        if name.endswith(('.png', '.jpg', '.tif')):
            fp = download(url, f"gpr_tu1208_{name}")
            if fp:
                try:
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 30:
                        gpr_imgs.append(normalize_01(img))
                except: pass
        elif name.endswith('.zip'):
            fp = download(url, f"gpr_tu1208_{name}")
            if fp:
                try:
                    with zipfile.ZipFile(str(fp)) as zf:
                        for n in sorted(zf.namelist())[:30]:
                            if n.lower().endswith(('.png', '.jpg', '.tif', '.bmp')):
                                try:
                                    data = zf.read(n)
                                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                    if img.shape[0] > 30:
                                        gpr_imgs.append(normalize_01(img))
                                except: pass
                except: pass
        if len(gpr_imgs) >= 15: break

    print(f"  Loaded {len(gpr_imgs)} GPR images")
    if gpr_imgs:
        sx, sy = [], []
        for i, img in enumerate(gpr_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            y = gaussian_filter(x, sigma=3).astype(np.float32)
            y += 0.04 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
        save_mod("gpr", sx, sy,
            "TU1208 IFSTTAR GPR radargram dataset (Zenodo 1211173)",
            "Pajewski et al., TU1208 Open Database of Radargrams, Remote Sensing 2018",
            "GPR: electromagnetic wave subsurface reflection profiling")
except Exception as e:
    print(f"  GPR error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# 7. DIC - BBBC028 (broadinstitute.org direct)
# ============================================================
print("\n=== dic ===")
# BBBC028 DIC images - try direct from Broad
fp = download("https://data.broadinstitute.org/bbbc/BBBC028/BBBC028_v1_images.zip",
              "BBBC028_v1_images.zip", timeout=120)

if fp and fp.stat().st_size > 5000:
    try:
        dic_imgs = []
        with zipfile.ZipFile(str(fp)) as zf:
            for n in sorted(zf.namelist()):
                if n.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                    try:
                        data = zf.read(n)
                        img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                        if img.shape[0] > 30:
                            dic_imgs.append(normalize_01(img))
                    except: pass
                if len(dic_imgs) >= 15: break
        print(f"  BBBC028 DIC: {len(dic_imgs)} images")
        if dic_imgs:
            sx, sy = [], []
            for i, img in enumerate(dic_imgs[:10]):
                x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                x = normalize_01(x)
                # DIC forward: phase gradient (Nomarski prism)
                phase = x * np.pi
                y = normalize_01((0.5 + 0.5 * np.cos(phase + 2*np.pi*np.arange(256).reshape(1,-1)/32)).astype(np.float32))
                sx.append(x); sy.append(y)
            while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
            save_mod("dic", sx, sy,
                "BBBC028 DIC polymerized structures (Broad Bioimage Benchmark)",
                "Bray et al., BBBC: image sets for benchmarking, Nat Methods 2012",
                "DIC: differential interference contrast phase gradient imaging")
    except Exception as e:
        print(f"  DIC error: {str(e).encode('ascii','replace').decode('ascii')[:80]}")
else:
    print("  DIC BBBC028: download failed")


print("\n=== Downloads complete ===")

# Final uniqueness check
print("\nFinal uniqueness check:")
hashes = {}
for d in sorted(BASE.iterdir()):
    if d.name.startswith('_') or not d.is_dir(): continue
    std = d / 'standard'
    if not std.exists(): continue
    h5s = sorted(std.glob('*.h5'))
    if not h5s: continue
    with h5py.File(str(h5s[0]), 'r') as f:
        x = f['x_true'][:]
    hashes[d.name] = hash(x.tobytes())

unique = len(set(hashes.values()))
print(f"Total: {len(hashes)}, Unique: {unique}/{len(hashes)}")
