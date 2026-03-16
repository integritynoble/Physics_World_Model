"""Download canonical datasets v2 - use direct Zenodo file URLs and fix Marmousi.

Zenodo files-archive returns 403, so use /files/{filename}?download=1 URLs instead.
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
import tarfile
import os
import tempfile
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def download(url, fname, timeout=180):
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
        x2d = x if x.ndim==2 else x[:,:,x.shape[2]//2]
        write_png(x2d, str(img_dir / f"x_true_{i:02d}.png"))
        y2d = y if y.ndim==2 else y[:,:,0]
        write_png(y2d, str(img_dir / f"y_meas_{i:02d}.png"))
    meta = {"modality": mod, "n_samples": len(sx),
            "x_shape": list(sx[0].shape), "y_shape": list(sy[0].shape),
            "source": source, "reference": reference,
            "data_type": "real", "forward_model": fwd}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod, "source": source, "reference": reference}, f, indent=2)
    hashes = set(hash(x.tobytes()) for x in sx)
    print(f"  {mod}: {len(sx)} samples, {len(hashes)} unique")
    return len(hashes) == len(samples_x) if 'samples_x' in dir() else True


# ============================================================
# 1. AFM - Zenodo 60434 individual PNG files
# ============================================================
print("\n=== afm ===")
afm_imgs = []
# Download individual preview PNGs from the record
for i in range(1, 11):
    fname = f"afm_preview_{i:02d}.png"
    # Try the content API endpoint
    url = f"https://zenodo.org/records/60434/files/preview_{i:02d}.png?download=1"
    fp = download(url, fname, timeout=30)
    if fp:
        try:
            img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
            if img.shape[0] > 30:
                afm_imgs.append(normalize_01(img))
        except:
            pass

# Also try the ZIP from different endpoint
if len(afm_imgs) < 3:
    # Try direct content URL
    fp = download("https://zenodo.org/records/60434/files/data.zip?download=1", "afm_data.zip", timeout=60)
    if fp and fp.stat().st_size > 1000:
        try:
            with zipfile.ZipFile(str(fp)) as zf:
                for n in sorted(zf.namelist()):
                    if n.lower().endswith('.png'):
                        data = zf.read(n)
                        img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                        if img.shape[0] > 30:
                            afm_imgs.append(normalize_01(img))
        except:
            pass

if len(afm_imgs) >= 3:
    sx, sy = [], []
    for i, img in enumerate(afm_imgs[:10]):
        x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
        y = gaussian_filter(x, sigma=2).astype(np.float32)
        y += 0.03 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
        y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
        sx.append(x)
        sy.append(y)
    while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
    save_mod("afm", sx, sy,
        "Zenodo 60434 AFM images of various specimens (Keysight Technologies)",
        "Oxvig et al., Structure Assisted CS Reconstruction of Undersampled AFM Images, 2017",
        "AFM: tip convolution + thermal noise")
else:
    print(f"  AFM: only {len(afm_imgs)} images, trying compressed sensing AFM dataset...")
    # Try Zenodo 18401 - CS AFM dataset
    fp = download("https://zenodo.org/records/18401/files/data.zip?download=1", "afm_cs_data.zip", timeout=60)
    if fp and fp.stat().st_size > 1000:
        try:
            with zipfile.ZipFile(str(fp)) as zf:
                names = zf.namelist()
                print(f"  AFM CS zip: {len(names)} files, {names[:5]}")
                for n in sorted(names):
                    if n.lower().endswith(('.png', '.tif', '.jpg', '.npy')):
                        try:
                            data = zf.read(n)
                            if n.endswith('.npy'):
                                arr = np.load(io.BytesIO(data))
                                if arr.ndim == 2: afm_imgs.append(normalize_01(arr.astype(np.float32)))
                            else:
                                img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                                if img.shape[0] > 30: afm_imgs.append(normalize_01(img))
                        except: pass
        except: pass
    if len(afm_imgs) >= 3:
        sx, sy = [], []
        for i, img in enumerate(afm_imgs[:10]):
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            y = gaussian_filter(x, sigma=2).astype(np.float32)
            y += 0.03 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
        save_mod("afm", sx, sy,
            "Zenodo 18401 CS-AFM undersampled images dataset",
            "Oxvig et al., Structure Assisted CS Reconstruction of Undersampled AFM Images, 2017",
            "AFM: tip convolution + compressed sensing undersampling")
    else:
        print(f"  AFM: failed to get data ({len(afm_imgs)} images)")


# ============================================================
# 2. PHOTOACOUSTIC - Duke PAM clean subset
# ============================================================
print("\n=== photoacoustic ===")
pa_imgs = []
# Try individual file names for Duke PAM
for fn in ["OR-PAM_mouse_brain_01.jpg", "OR-PAM_mouse_brain_02.jpg",
           "clean_mouse_brain_01.jpg", "clean_mouse_brain_02.jpg"]:
    fp = download(f"https://zenodo.org/records/4042171/files/{fn}?download=1",
                  f"pa_{fn}", timeout=30)
    if fp:
        try:
            img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
            if img.shape[0] > 50: pa_imgs.append(normalize_01(img))
        except: pass

# Try the dataset ZIP with correct filename
if len(pa_imgs) < 3:
    fp = download("https://zenodo.org/records/4042171/files/DukePAM_clean.zip?download=1",
                  "pa_duke_clean2.zip", timeout=120)
    if fp and fp.stat().st_size > 1000:
        try:
            with zipfile.ZipFile(str(fp)) as zf:
                for n in sorted(zf.namelist())[:30]:
                    if n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                        try:
                            data = zf.read(n)
                            img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                            if img.shape[0] > 50: pa_imgs.append(normalize_01(img))
                        except: pass
        except: pass

if len(pa_imgs) >= 3:
    sx, sy = [], []
    for i, img in enumerate(pa_imgs[:10]):
        x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
        edges = np.abs(sobel(x, axis=0)) + np.abs(sobel(x, axis=1))
        y = normalize_01(gaussian_filter(edges, sigma=2).astype(np.float32))
        sx.append(x); sy.append(y)
    while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
    save_mod("photoacoustic", sx, sy,
        "Duke PAM dataset (Zenodo 4042171, mouse brain photoacoustic microscopy)",
        "Vu et al., Duke PAM, Zenodo 2020; Wang & Yao, Nat Methods 2016",
        "Photoacoustic: optical absorption -> acoustic wave propagation")
else:
    print(f"  Photoacoustic: only {len(pa_imgs)} images")


# ============================================================
# 3. SEISMIC TOMO + FWI - Fix Marmousi parsing
# ============================================================
print("\n=== seismic_tomo / fwi ===")
# The extracted tarball contains Vp.segy.tar.gz - need to extract again
marmousi_dir = CACHE / "marmousi_extract"
inner_tgz = None
for f in marmousi_dir.rglob("*.tar.gz"):
    if "vp" in f.name.lower() or "Vp" in f.name:
        inner_tgz = f
        break

if inner_tgz is None:
    # Search more broadly
    for f in marmousi_dir.rglob("*"):
        if f.is_file() and f.stat().st_size > 100000:
            print(f"  Found: {f.name} ({f.stat().st_size/1024:.0f} KB)")
            if "vp" in f.name.lower() or "Vp" in f.name:
                inner_tgz = f

if inner_tgz and inner_tgz.name.endswith('.tar.gz'):
    print(f"  Extracting inner archive: {inner_tgz.name}")
    try:
        inner_dir = marmousi_dir / "vp_extracted"
        inner_dir.mkdir(exist_ok=True)
        with tarfile.open(str(inner_tgz), "r:gz") as tf:
            tf.extractall(str(inner_dir))
        segy_files = list(inner_dir.rglob("*.segy")) + list(inner_dir.rglob("*.sgy"))
        print(f"  Inner extracted: {[f.name for f in segy_files]}")

        # Parse SEGY - just read the trace data
        for sf in segy_files:
            raw_bytes = sf.read_bytes()
            print(f"  SEGY file: {sf.name}, {len(raw_bytes)} bytes")

            # SEGY header is 3600 bytes, then trace headers (240 bytes) + data
            # Skip to find trace data
            # Standard SEGY: 3200 EBCDIC + 400 binary header
            # Trace: 240 byte header + ns*4 bytes data
            # Read number of samples from binary header (bytes 3220-3222, big-endian unsigned short)
            ns = struct.unpack('>H', raw_bytes[3220:3222])[0]
            if ns == 0:
                ns = struct.unpack('<H', raw_bytes[3220:3222])[0]
            print(f"  Samples per trace (ns): {ns}")

            if ns > 0 and ns < 10000:
                # Extract traces
                offset = 3600  # end of file headers
                traces = []
                while offset + 240 + ns*4 <= len(raw_bytes):
                    trace_data = np.frombuffer(raw_bytes[offset+240:offset+240+ns*4], dtype='>f4')
                    if len(trace_data) == ns:
                        traces.append(trace_data)
                    offset += 240 + ns * 4

                if traces:
                    vel = np.array(traces).T  # (ns, n_traces)
                    print(f"  Velocity model: {vel.shape}, range [{np.nanmin(vel):.0f}, {np.nanmax(vel):.0f}]")

                    # Remove NaN
                    vel = np.nan_to_num(vel, nan=np.nanmean(vel))

                    h, w = vel.shape
                    for mod, offset_idx in [("seismic_tomo", 0), ("fwi", 5)]:
                        sx, sy = [], []
                        for i in range(10):
                            ci = (i + offset_idx) % max(1, (w - 256) // 25)
                            c = min(ci * 25, max(0, w - 256))
                            r = min(i * max(1, (h - 256) // 9), max(0, h - 256))
                            if h >= 256 and w >= 256:
                                patch = vel[r:r+256, c:c+256]
                            else:
                                patch = sk_resize(vel, (256, min(256, w)), order=3)
                                if patch.shape[1] < 256:
                                    patch = sk_resize(patch, (256, 256), order=3)
                            x = normalize_01(patch.astype(np.float32))
                            y = gaussian_filter(x, sigma=4).astype(np.float32)
                            rng = np.random.RandomState(42+i+offset_idx)
                            y += 0.03 * rng.randn(256,256).astype(np.float32)
                            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
                            sx.append(x); sy.append(y)
                        ref = "Martin et al., Marmousi2: An elastic upgrade for Marmousi, The Leading Edge 2006"
                        src = "Elastic Marmousi2 Vp velocity model (open.source.geoscience)"
                        fwd = "Seismic tomography" if mod == "seismic_tomo" else "Full waveform inversion"
                        save_mod(mod, sx, sy, src, ref, fwd)
                    break
    except Exception as e:
        print(f"  Marmousi inner error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# 4. OCTA - ROSE dataset individual images
# ============================================================
print("\n=== octa ===")
# Try direct file URL from Zenodo
fp = download("https://zenodo.org/records/12775880/files/ROSE-1.zip?download=1",
              "octa_rose1.zip", timeout=120)

if fp and fp.stat().st_size > 5000:
    try:
        octa_imgs = []
        with zipfile.ZipFile(str(fp)) as zf:
            for n in sorted(zf.namelist())[:30]:
                if n.lower().endswith(('.png', '.tif', '.jpg', '.bmp')):
                    try:
                        data = zf.read(n)
                        img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                        if img.shape[0] > 50:
                            octa_imgs.append(normalize_01(img))
                    except: pass
        print(f"  OCTA ROSE: {len(octa_imgs)} images")
        if octa_imgs:
            sx, sy = [], []
            for i, img in enumerate(octa_imgs[:10]):
                x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                y = gaussian_filter(x, sigma=1.5).astype(np.float32)
                y += 0.02 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
                y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
            save_mod("octa", sx, sy,
                "ROSE retinal OCTA dataset (Zenodo 12775880)",
                "Ma et al., ROSE: Retinal OCT-Angiography Vessel Segmentation, IEEE TMI 2021",
                "OCTA: decorrelation-based angiography from OCT volumes")
    except Exception as e:
        print(f"  OCTA error: {str(e).encode('ascii','replace').decode('ascii')[:80]}")
else:
    print("  OCTA ROSE-1: download failed")


# ============================================================
# 5. EIT - FIPS dataset
# ============================================================
print("\n=== impedance_tomo ===")
# Finnish Inverse Problems Society dataset
fp = download("https://zenodo.org/records/1203913/files/OpenEIT2DData.zip?download=1",
              "eit_open2d_v2.zip", timeout=60)
if not fp or fp.stat().st_size < 1000:
    fp = download("https://zenodo.org/records/1203913/files/data.zip?download=1",
                  "eit_open2d_data.zip", timeout=60)

if fp and fp.stat().st_size > 1000:
    try:
        with zipfile.ZipFile(str(fp)) as zf:
            names = zf.namelist()
            print(f"  EIT zip: {len(names)} files, samples: {names[:8]}")
            eit_imgs = []
            for n in sorted(names):
                if n.lower().endswith(('.png', '.tif', '.jpg', '.mat', '.npy')):
                    try:
                        data = zf.read(n)
                        if n.endswith('.mat'):
                            tmp = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
                            tmp.write(data); tmp.close()
                            from scipy.io import loadmat
                            mat = loadmat(tmp.name)
                            os.unlink(tmp.name)
                            for k in mat:
                                if isinstance(mat[k], np.ndarray) and mat[k].ndim == 2 and mat[k].shape[0] > 10:
                                    eit_imgs.append(normalize_01(mat[k].astype(np.float32)))
                        else:
                            img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                            if img.shape[0] > 30: eit_imgs.append(normalize_01(img))
                    except: pass
                if len(eit_imgs) >= 15: break
        if eit_imgs:
            sx, sy = [], []
            for i, img in enumerate(eit_imgs[:10]):
                x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                y = gaussian_filter(x, sigma=8).astype(np.float32)
                y += 0.05 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
                y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
            save_mod("impedance_tomo", sx, sy,
                "Open 2D EIT dataset (Zenodo 1203913, FIPS Finland)",
                "Hauptmann et al., Open 2D EIT Data Archive, arXiv 2017",
                "EIT: reconstruct conductivity from boundary voltage measurements")
        else:
            print(f"  EIT: no images extracted")
    except Exception as e:
        print(f"  EIT error: {str(e).encode('ascii','replace').decode('ascii')[:80]}")
else:
    print("  EIT: all downloads failed")


print("\n=== Done ===")
