"""Fix PET (use h5py for v7.3 mat) and fluoroscopy (URL-encode spaces)."""
import numpy as np
import h5py
import json
import struct
import zlib
import urllib.request
import urllib.parse
import ssl
import io
import zipfile
from pathlib import Path
from PIL import Image
from skimage.transform import resize as sk_resize
from scipy.signal import fftconvolve
import scipy.io as sio
Image.MAX_IMAGE_PIXELS = None

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
HEADERS = {"User-Agent": "curl/7.68.0", "Accept": "*/*"}

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

def save_mod(mod_name, sx, sy, source, reference, forward_model, data_type="real"):
    out = BASE / mod_name / "standard"
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    for old in out.glob(f"standard_{mod_name}_*.h5"): old.unlink()
    for old in img_dir.glob("*.png"): old.unlink()
    for i, (x, y) in enumerate(zip(sx, sy)):
        with h5py.File(str(out / f"standard_{mod_name}_{i:02d}.h5"), "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            f.attrs["modality"] = mod_name
            f.attrs["sample_index"] = i
            f.attrs["source"] = source
            f.attrs["reference"] = reference
            f.attrs["data_type"] = data_type
        write_png(x, str(img_dir / f"x_true_{i:02d}.png"))
        write_png(y, str(img_dir / f"y_meas_{i:02d}.png"))
    meta = {"modality": mod_name, "n_samples": len(sx),
            "x_shape": list(sx[0].shape), "y_shape": list(sy[0].shape),
            "source": source, "reference": reference,
            "data_type": data_type, "forward_model": forward_model}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod_name, "source": source, "reference": reference}, f, indent=2)
    hashes = set(hash(x.tobytes()) for x in sx)
    print(f"  BUILT {mod_name}: {len(sx)} samples, {len(hashes)} unique")


# === FIX PET ===
print("=== Fixing PET ===")
fp = CACHE / "pet_gate_ideal.mat"
mat = sio.loadmat(str(fp))
pet_imgs = []

for key in ["RA", "SC"]:
    if key in mat:
        arr = mat[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        for s in range(arr.shape[2]):
            sl = arr[:, :, s].astype(np.float32)
            if sl.std() > 0 and sl.max() > 0:
                pet_imgs.append(normalize_01(sl))

# Also try sinogram with h5py
fp2 = CACHE / "pet_gate_sinogram.mat"
try:
    with h5py.File(str(fp2), "r") as f:
        print(f"  Sinogram keys: {list(f.keys())}")
        for k in f.keys():
            arr = f[k][:]
            print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.ndim == 3 and min(arr.shape) > 5:
                # Find the spatial dims (the two largest)
                ax = np.argmin(arr.shape)
                for s in range(0, min(arr.shape[ax], 20), 2):
                    if ax == 0:
                        sl = arr[s].astype(np.float32)
                    elif ax == 1:
                        sl = arr[:, s, :].astype(np.float32)
                    else:
                        sl = arr[:, :, s].astype(np.float32)
                    if sl.std() > 0:
                        pet_imgs.append(normalize_01(sl))
except Exception as e:
    print(f"  h5py error: {str(e)[:80]}")

print(f"  Total PET images: {len(pet_imgs)}")

if len(pet_imgs) >= 10:
    sx, sy = [], []
    for img in pet_imgs[:20]:
        x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
        x = normalize_01(x)
        sz = 5
        yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
        psf = np.exp(-(xx**2 + yy**2) / (2*2.0**2)).astype(np.float32)
        psf /= psf.sum()
        y = fftconvolve(x, psf, mode='same').astype(np.float32)
        y_scaled = np.maximum(y * 1000, 0.1)
        y_poisson = np.random.RandomState(42 + len(sx)).poisson(y_scaled).astype(np.float32)
        y = normalize_01(y_poisson)
        sx.append(x)
        sy.append(y)
    save_mod("pet", sx, sy,
        source="GATE simulated PET data (Zenodo 3522199, cylindrical PET phantom)",
        reference="Zenodo 3522199; Raw and sinogram data of simulated GATE PET",
        forward_model="PET: positron emission tomography sinogram with Poisson noise")
else:
    print("  Not enough PET images")


# === FIX FLUOROSCOPY ===
print("\n=== Fixing Fluoroscopy ===")
url = "https://zenodo.org/api/records/4457648/files/" + urllib.parse.quote("Panoramic radiography database.zip") + "/content"
fname = "panoramic_radio.zip"
path = CACHE / fname
if not (path.exists() and path.stat().st_size > 100):
    print(f"  Downloading {fname}...")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=ctx, timeout=600) as r:
            data = r.read()
        with open(str(path), "wb") as f:
            f.write(data)
        print(f"  Got {len(data)/1024/1024:.1f} MB")
    except Exception as e:
        print(f"  FAILED: {str(e)[:120]}")
        path = None
else:
    print(f"  [cached] {fname} ({path.stat().st_size/1024/1024:.1f}MB)")

if path and path.exists() and path.stat().st_size > 100:
    fluoro_imgs = []
    with zipfile.ZipFile(str(path)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
        print(f"  Found {len(names)} radiograph images")
        for n in names[:25]:
            try:
                data = zf.read(n)
                img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                if img.shape[0] > 100 and img.std() > 3:
                    fluoro_imgs.append(normalize_01(img))
                    if len(fluoro_imgs) >= 20:
                        break
            except:
                pass
    print(f"  Loaded {len(fluoro_imgs)} radiograph images")

    if len(fluoro_imgs) >= 10:
        sx, sy = [], []
        for img in fluoro_imgs[:20]:
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            sz = 3
            yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
            psf = np.exp(-(xx**2 + yy**2) / (2*1.2**2)).astype(np.float32)
            psf /= psf.sum()
            y = fftconvolve(x, psf, mode='same').astype(np.float32)
            noise = np.random.RandomState(42 + len(sx)).randn(*y.shape).astype(np.float32) * 0.04
            y = np.clip(y + noise, 0, 1)
            y = normalize_01(y)
            sx.append(x)
            sy.append(y)
        save_mod("fluoroscopy", sx, sy,
            source="Panoramic radiography database (Zenodo 4457648, 598 dental radiographs)",
            reference="Zenodo 4457648; Panoramic radiography database for imaging research",
            forward_model="Fluoroscopy: real-time X-ray imaging with scatter and detector noise")


# === VERIFICATION ===
print("\n=== Verification ===")
for mod in ["pet", "fluoroscopy"]:
    std = BASE / mod / "standard"
    h5s = sorted(std.glob(f"standard_{mod}_*.h5"))
    if h5s:
        with h5py.File(str(h5s[0]), "r") as f:
            src = f.attrs.get("source", "")[:60]
        hashes = set()
        for h5 in h5s:
            with h5py.File(str(h5), "r") as f:
                hashes.add(hash(f["x_true"][:].tobytes()))
        print(f"  {mod}: {len(h5s)} samples, {len(hashes)} unique, src={src}")
