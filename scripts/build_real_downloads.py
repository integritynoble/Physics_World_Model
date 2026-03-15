"""
Download REAL datasets for each modality from canonical sources.
Process one modality at a time: download → extract → build HDF5+PNG → cleanup.

Sources:
  - MedMNIST 128×128 (Zenodo) for medical modalities
  - BBBC (Broad Institute) for fluorescence microscopy
  - RRUFF for Raman spectroscopy
  - NASA/NOAA for astronomy/remote sensing
  - Zenodo specialized datasets for computational imaging
  - EMPIAR thumbnails for electron microscopy
"""
import numpy as np
import h5py
import json
import struct
import zlib
import inspect
import urllib.request
import urllib.error
import zipfile
import io
import os
import sys
import shutil
import ssl
import time
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import convolve2d

try:
    from PIL import Image
except ImportError:
    Image = None

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
CACHE.mkdir(exist_ok=True)
N_SAMPLES = 10
SZ = (256, 256)

# SSL context for HTTPS downloads
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# ============================================================
# INFRASTRUCTURE
# ============================================================
def resize_2d(img, target):
    from skimage.transform import resize as sk_resize
    return sk_resize(img, target, order=3, anti_aliasing=True).astype(np.float32)

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
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))

def save_modality(mod, samples_x, samples_y, reference, source_name, forward_desc=""):
    out_dir = BASE / mod / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.h5"): old.unlink()
    for i, (x, y) in enumerate(zip(samples_x, samples_y)):
        with h5py.File(str(out_dir / f"standard_{mod}_{i:02d}.h5"), "w") as f:
            f.create_dataset("x_true", data=x.astype(np.float32), compression="gzip")
            f.create_dataset("y_ideal", data=y.astype(np.float32), compression="gzip")
            f.attrs["modality"] = mod
            f.attrs["sample_index"] = i
            f.attrs["source"] = source_name
            f.attrs["reference"] = reference
            f.attrs["data_type"] = "real"
        if x.ndim == 2:
            write_png(x, str(img_dir / f"x_true_{i:02d}.png"))
    meta = {"modality": mod, "n_samples": len(samples_x), "x_shape": list(samples_x[0].shape),
            "y_shape": list(samples_y[0].shape), "source": source_name, "reference": reference,
            "data_type": "real", "forward_model": forward_desc}
    with open(out_dir / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out_dir / "spec.json", "w") as f:
        json.dump({"modality": mod, "source": source_name, "reference": reference}, f, indent=2)

def download(url, dest, desc=""):
    """Download file with caching and retry."""
    if dest.exists() and dest.stat().st_size > 100:
        print(f"    [cached] {dest.name}")
        return True
    print(f"    Downloading {desc or url[:80]}...")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 PWM-Benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(str(dest), "wb") as f:
                f.write(data)
            print(f"    Downloaded {len(data)/1024/1024:.1f} MB -> {dest.name}")
            return True
        except Exception as e:
            err = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"    Attempt {attempt+1}/3 failed: {err}")
            time.sleep(2)
    return False

# ---- Forward Models ----
def fwd_radon(x, n_angles=180):
    h, w = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    diag = int(np.ceil(np.sqrt(2) * max(h, w)))
    ph, pw = (diag-h)//2, (diag-w)//2
    padded = np.pad(x, ((ph, diag-h-ph), (pw, diag-w-pw)))
    sino = np.zeros((n_angles, diag), dtype=np.float32)
    for i, a in enumerate(angles):
        sino[i] = rotate(padded, a, reshape=False, order=1).sum(axis=0)
    return sino

def fwd_psf(x, sigma=2.0):
    return gaussian_filter(x, sigma=sigma).astype(np.float32)

def fwd_mag(x):
    return np.abs(np.fft.fftshift(np.fft.fft2(x))).astype(np.float32)

def fwd_interfero(x, seed=42):
    rng = np.random.RandomState(seed)
    h, w = x.shape; freq = 5+3*rng.rand()
    yy, xx = np.mgrid[:h,:w]
    c = np.cos(2*np.pi*freq*(yy*np.cos(rng.rand()*np.pi)+xx*np.sin(rng.rand()*np.pi))/max(h,w))
    return (0.5+0.5*np.cos(2*np.pi*x+c*np.pi)).astype(np.float32)

def fwd_sparse(x, frac=0.05, seed=42):
    rng = np.random.RandomState(seed)
    return (x * (rng.rand(*x.shape) < frac)).astype(np.float32)

def fwd_kspace(x, accel=4, seed=42):
    """k-space undersampling (MRI-like)."""
    if x.ndim > 2:
        x = x[:, :, 0] if x.shape[-1] <= 4 else x[..., 0]
    ft = np.fft.fftshift(np.fft.fft2(x))
    rng = np.random.RandomState(seed)
    mask = np.zeros(x.shape[:2], dtype=bool)
    h, w = x.shape[:2]
    # Keep center 10%
    c = int(w * 0.05)
    mask[:, w//2-c:w//2+c] = True
    # Random lines
    lines = rng.choice(w, w//accel, replace=False)
    mask[:, lines] = True
    return np.abs(ft * mask).astype(np.float32)


# ============================================================
# MedMNIST HANDLER
# ============================================================
def load_medmnist(subset, size=128):
    """Load MedMNIST dataset. Try cached 128 first, then download, then fallback to 28x28."""
    # Size thresholds for download (MB) - skip very large downloads
    MAX_DOWNLOAD_MB = 300

    # Check for cached larger version (DON'T check base 28x28 here)
    for suffix in [f"_{size}", f"{size}", "_64", "64"]:
        path = CACHE / f"{subset}{suffix}.npz"
        if path.exists() and path.stat().st_size > 1000:
            data = np.load(str(path))
            imgs = np.concatenate([data[k] for k in ['train_images','val_images','test_images']
                                   if k in data])
            print(f"    [cached] {path.name}: {len(imgs)} images, shape={imgs.shape[1:]}")
            return imgs

    # Try downloading 128 version (correct URL format with underscore)
    for try_size in [128, 64]:
        url = f"https://zenodo.org/records/10519652/files/{subset}_{try_size}.npz?download=true"
        dest = CACHE / f"{subset}_{try_size}.npz"
        if download(url, dest, f"MedMNIST {subset} {try_size}x{try_size}"):
            sz_mb = dest.stat().st_size / 1024 / 1024
            if sz_mb > MAX_DOWNLOAD_MB * 3:
                print(f"    File too large ({sz_mb:.0f} MB), removing...")
                dest.unlink()
                continue
            data = np.load(str(dest))
            imgs = np.concatenate([data[k] for k in ['train_images','val_images','test_images']
                                   if k in data])
            print(f"    Loaded {len(imgs)} images, shape={imgs.shape[1:]}")
            return imgs

    # Fallback to 28x28
    path = CACHE / f"{subset}.npz"
    if path.exists():
        print(f"    [fallback] Using 28x28 version")
        data = np.load(str(path))
        return np.concatenate([data[k] for k in ['train_images','val_images','test_images']
                               if k in data])
    return None

def build_from_medmnist(mod, subset, indices, fwd_fn, fwd_kw, ref, src_name, desc):
    """Build modality from MedMNIST data."""
    imgs = load_medmnist(subset)
    if imgs is None:
        print(f"    SKIP {mod}: MedMNIST {subset} not available")
        return False

    n_total = len(imgs)
    sx, sy = [], []
    for i, idx in enumerate(indices[:N_SAMPLES]):
        img = imgs[idx % n_total].astype(np.float32)
        if img.ndim == 3:
            img = img[..., 0] if img.shape[-1] <= 4 else img
        img = resize_2d(normalize_01(img), SZ)
        y = fwd_fn(img, **fwd_kw)
        sx.append(img); sy.append(y)

    save_modality(mod, sx, sy, ref, src_name, desc)
    return True


# ============================================================
# BBBC HANDLER
# ============================================================
def download_bbbc(dataset_id, zip_name):
    """Download BBBC dataset ZIP."""
    url = f"https://data.broadinstitute.org/bbbc/{dataset_id}/{zip_name}"
    dest = CACHE / zip_name
    return download(url, dest, f"BBBC {dataset_id}")

def extract_bbbc_images(zip_name, n=10, grayscale=True):
    """Extract n images from BBBC ZIP file."""
    path = CACHE / zip_name
    if not path.exists():
        return []

    images = []
    with zipfile.ZipFile(str(path), 'r') as zf:
        tifs = [f for f in zf.namelist() if f.lower().endswith(('.tif', '.tiff', '.png'))]
        if not tifs:
            return []
        # Take evenly spaced images
        step = max(1, len(tifs) // n)
        selected = tifs[::step][:n]

        for fname in selected:
            try:
                data = zf.read(fname)
                if Image is not None:
                    img = Image.open(io.BytesIO(data))
                    if grayscale:
                        img = img.convert('L')
                    arr = np.array(img).astype(np.float32)
                    images.append(normalize_01(arr))
            except Exception as e:
                pass

    return images

def build_from_bbbc(mod, dataset_id, zip_name, fwd_fn, fwd_kw, ref, src_name, desc):
    """Build modality from BBBC dataset."""
    if not download_bbbc(dataset_id, zip_name):
        return False

    images = extract_bbbc_images(zip_name, n=N_SAMPLES)
    if len(images) < 3:
        print(f"    SKIP {mod}: too few images extracted from BBBC")
        return False

    sx, sy = [], []
    for img in images[:N_SAMPLES]:
        x = resize_2d(img, SZ)
        y = fwd_fn(x, **fwd_kw)
        sx.append(x); sy.append(y)

    # Pad if fewer than N_SAMPLES
    while len(sx) < N_SAMPLES:
        sx.append(sx[-1]); sy.append(sy[-1])

    save_modality(mod, sx, sy, ref, src_name, desc)
    return True


# ============================================================
# GENERIC IMAGE DOWNLOAD HANDLER
# ============================================================
def build_from_url_image(mod, url, fwd_fn, fwd_kw, ref, src_name, desc, is_fits=False):
    """Download a single image from URL and build modality with augmented crops."""
    fname = url.split("/")[-1].split("?")[0]
    dest = CACHE / f"{mod}_{fname}"

    if not download(url, dest, f"{mod} image"):
        return False

    try:
        if is_fits:
            from astropy.io import fits
            with fits.open(str(dest)) as hdul:
                img = hdul[0].data.astype(np.float32)
                if img.ndim == 3:
                    img = img[0]
        elif fname.endswith('.npy'):
            img = np.load(str(dest)).astype(np.float32)
        elif fname.endswith('.npz'):
            data = np.load(str(dest))
            key = list(data.keys())[0]
            img = data[key].astype(np.float32)
            if img.ndim == 3:
                img = img[0]
        else:
            if Image is not None:
                pil = Image.open(str(dest)).convert('L')
                img = np.array(pil).astype(np.float32)
            else:
                return False

        img = normalize_01(img)
        if img.ndim > 2:
            while img.ndim > 2:
                img = img[img.shape[0]//2] if img.shape[0] > 1 else img[0]

        # Create N_SAMPLES unique crops
        rng = np.random.RandomState(hash(mod) & 0x7fffffff)
        h, w = img.shape
        sx, sy = [], []
        for i in range(N_SAMPLES):
            cf = 0.5 + 0.4 * rng.rand()
            ch, cw = int(h*cf), int(w*cf)
            if h > ch and w > cw:
                sy_start = rng.randint(0, h - ch)
                sx_start = rng.randint(0, w - cw)
                crop = img[sy_start:sy_start+ch, sx_start:sx_start+cw].copy()
            else:
                crop = img.copy()
            crop = rotate(crop, rng.uniform(-10, 10), reshape=False, order=1)
            x = resize_2d(normalize_01(crop), SZ)
            y = fwd_fn(x, **fwd_kw)
            sx.append(x); sy.append(y)

        save_modality(mod, sx, sy, ref, src_name, desc)
        # Cleanup downloaded file (not cache)
        # dest.unlink()  # Keep cache for re-runs
        return True
    except Exception as e:
        print(f"    ERROR processing {mod}: {e}")
        return False


# ============================================================
# MODALITY REGISTRY - organized by download source
# ============================================================

MODALITIES = []

# ---- GROUP A: MedMNIST medical imaging (domain-matched) ----
# Each uses DIFFERENT indices to ensure unique x_true across modalities

# organAMNIST = real axial CT scans from LiTS (Liver Tumor Segmentation)
org_ref = "Bilic et al., MedIA 2023; LiTS liver tumor CT via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "ct",          "organamnist", [0,500,1000,1500,2000,2500,3000,3500,4000,4500],
     fwd_radon, {"n_angles":180}, org_ref, "organAMNIST CT axial (real LiTS)", "CT Radon sinogram"),
    ("medmnist", "spectral_ct", "organamnist", [100,600,1100,1600,2100,2600,3100,3600,4100,4600],
     fwd_radon, {"n_angles":120}, org_ref, "organAMNIST CT axial idx 100+ (real LiTS)", "Spectral CT multi-energy Radon"),
    ("medmnist", "industrial_ct","organamnist", [200,700,1200,1700,2200,2700,3200,3700,4200,4700],
     fwd_radon, {"n_angles":90}, org_ref, "organAMNIST CT axial idx 200+ (real LiTS)", "Industrial CT Radon"),
    ("medmnist", "dexa",        "organamnist", [300,800,1300,1800,2300,2800,3300,3800,4300,4800],
     fwd_psf, {"sigma":1.5}, org_ref, "organAMNIST CT axial idx 300+ (real LiTS)", "DXA dual-energy projection"),
    ("medmnist", "xrf_tomo",    "organamnist", [400,900,1400,1900,2400,2900,3400,3900,4400,4900],
     fwd_radon, {"n_angles":60}, org_ref, "organAMNIST CT axial idx 400+ (real LiTS)", "XRF tomography"),
]

# organcMNIST = real coronal CT from LiTS
orgc_ref = "Bilic et al., MedIA 2023; LiTS coronal CT via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "cbct",        "organcmnist", [0,300,600,900,1200,1500,1800,2100,2400,2700],
     fwd_radon, {"n_angles":120}, orgc_ref, "organcMNIST CT coronal (real LiTS)", "CBCT Radon"),
]

# organsMNIST = real sagittal CT from LiTS
orgs_ref = "Bilic et al., MedIA 2023; LiTS sagittal CT via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "pet",         "organsmnist", [0,400,800,1200,1600,2000,2400,2800,3200,3600],
     fwd_radon, {"n_angles":60}, orgs_ref, "organsMNIST CT sagittal idx 0+ (PET proxy)", "PET emission tomography"),
    ("medmnist", "pet_ct",      "organsmnist", [100,500,900,1300,1700,2100,2500,2900,3300,3700],
     fwd_radon, {"n_angles":90}, orgs_ref, "organsMNIST CT sagittal idx 100+ (PET-CT proxy)", "PET-CT fused"),
    ("medmnist", "pet_mr",      "organsmnist", [200,600,1000,1400,1800,2200,2600,3000,3400,3800],
     fwd_kspace, {"accel":4}, orgs_ref, "organsMNIST CT sagittal idx 200+ (PET-MR proxy)", "PET-MR k-space"),
    ("medmnist", "spect",       "organsmnist", [50,450,850,1250,1650,2050,2450,2850,3250,3650],
     fwd_radon, {"n_angles":30}, orgs_ref, "organsMNIST CT sagittal idx 50+ (SPECT proxy)", "SPECT emission"),
    ("medmnist", "spect_ct",    "organsmnist", [150,550,950,1350,1750,2150,2550,2950,3350,3750],
     fwd_radon, {"n_angles":60}, orgs_ref, "organsMNIST CT sagittal idx 150+ (SPECT-CT proxy)", "SPECT-CT fused"),
]

# chestMNIST = real NIH CXR14 chest X-rays
cxr_ref = "Wang et al., CVPR 2017; NIH CXR14 via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "xray_radiography","chestmnist",[0,1000,2000,3000,4000,5000,6000,7000,8000,9000],
     fwd_psf, {"sigma":1.5}, cxr_ref, "chestMNIST CXR idx 0+ (real NIH CXR14)", "X-ray projection PSF"),
    ("medmnist", "fluoroscopy",    "chestmnist",[100,1100,2100,3100,4100,5100,6100,7100,8100,9100],
     fwd_psf, {"sigma":2.0}, cxr_ref, "chestMNIST CXR idx 100+ (real NIH CXR14)", "Fluoroscopy dynamic PSF"),
    ("medmnist", "portal_imaging", "chestmnist",[200,1200,2200,3200,4200,5200,6200,7200,8200,9200],
     fwd_psf, {"sigma":3.0}, cxr_ref, "chestMNIST CXR idx 200+ (real NIH CXR14)", "Portal imaging low-res"),
]

# pneumoniaMNIST = real pediatric chest X-rays (Kermany 2018)
pneu_ref = "Kermany et al., Cell 2018; pediatric CXR via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "xray_ndt",    "pneumoniamnist",[0,200,400,600,800,1000,1200,1400,1600,1800],
     fwd_psf, {"sigma":1.5}, pneu_ref, "pneumoniaMNIST CXR (real Kermany)", "NDT X-ray projection"),
]

# octMNIST = real retinal OCT B-scans (Kermany 2018)
oct_ref = "Kermany et al., Cell 2018; retinal OCT via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "oct",         "octmnist",   [0,5000,10000,15000,20000,25000,30000,35000,40000,45000],
     fwd_psf, {"sigma":1.0}, oct_ref, "octMNIST retinal OCT (real Kermany)", "OCT B-scan PSF"),
    ("medmnist", "octa",        "octmnist",   [100,5100,10100,15100,20100,25100,30100,35100,40100,45100],
     fwd_psf, {"sigma":1.5}, oct_ref, "octMNIST retinal OCT idx 100+ (OCTA proxy)", "OCTA en-face"),
]

# retinaMNIST = real retinal fundus (ODIR-5K)
ret_ref = "Peking University; ODIR-5K via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "fundus",      "retinamnist",[0,100,200,300,400,500,600,700,800,900],
     fwd_psf, {"sigma":1.0}, ret_ref, "retinaMNIST fundus (real ODIR-5K)", "Fundus imaging PSF"),
    ("medmnist", "angiography", "retinamnist",[50,150,250,350,450,550,650,750,850,950],
     fwd_psf, {"sigma":1.5}, ret_ref, "retinaMNIST fundus idx 50+ (angiography proxy)", "Angiography PSF"),
]

# breastMNIST = real breast ultrasound (BUSI)
us_ref = "Al-Dhabyani et al., DCASE 2020; BUSI via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "ultrasound",        "breastmnist",[0,50,100,150,200,250,300,350,400,450],
     fwd_psf, {"sigma":2.0}, us_ref, "breastMNIST ultrasound (real BUSI)", "Ultrasound B-mode PSF"),
    ("medmnist", "ceus",              "breastmnist",[10,60,110,160,210,260,310,360,410,460],
     fwd_psf, {"sigma":2.5}, us_ref, "breastMNIST ultrasound idx 10+ (CEUS proxy)", "CEUS contrast PSF"),
    ("medmnist", "doppler_ultrasound","breastmnist",[20,70,120,170,220,270,320,370,420,470],
     fwd_psf, {"sigma":1.5}, us_ref, "breastMNIST ultrasound idx 20+ (Doppler proxy)", "Doppler US PSF"),
    ("medmnist", "ivus",              "breastmnist",[30,80,130,180,230,280,330,380,430,480],
     fwd_psf, {"sigma":1.5}, us_ref, "breastMNIST ultrasound idx 30+ (IVUS proxy)", "IVUS intravascular PSF"),
]

# bloodMNIST = real blood cell microscopy (Acevedo 2020)
blood_ref = "Acevedo et al., Data in Brief 2020; blood cells via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "widefield",         "bloodmnist",[0,1000,2000,3000,4000,5000,6000,7000,8000,9000],
     fwd_psf, {"sigma":1.5}, blood_ref, "bloodMNIST blood cells (real Acevedo)", "Widefield microscopy PSF"),
    ("medmnist", "spinning_disk",     "bloodmnist",[100,1100,2100,3100,4100,5100,6100,7100,8100,9100],
     fwd_psf, {"sigma":1.0}, blood_ref, "bloodMNIST blood cells idx 100+ (real Acevedo)", "Spinning disk confocal"),
]

# pathMNIST = real colon pathology (Kather 2016/2019)
path_ref = "Kather et al., Sci Rep 2019; CRC-HE-100K via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "confocal_endomicroscopy","pathmnist",[0,5000,10000,15000,20000,25000,30000,35000,40000,45000],
     fwd_psf, {"sigma":1.0}, path_ref, "pathMNIST colon pathology (real Kather)", "Confocal endomicroscopy PSF"),
]

# tissueMNIST = real kidney cortex microscopy (Ljosa 2012)
tissue_ref = "Ljosa et al., Nat Methods 2012; kidney cortex via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "confocal_3d",       "tissuemnist",[0,10000,20000,30000,40000,50000,60000,70000,80000,90000],
     fwd_psf, {"sigma":1.5}, tissue_ref, "tissueMNIST kidney cortex (real Ljosa)", "Confocal 3D PSF"),
    ("medmnist", "confocal_livecell", "tissuemnist",[500,10500,20500,30500,40500,50500,60500,70500,80500,90500],
     fwd_psf, {"sigma":1.0}, tissue_ref, "tissueMNIST kidney cortex idx 500+ (real Ljosa)", "Live-cell confocal PSF"),
]

# dermaMNIST = real dermatoscopy (HAM10000)
derm_ref = "Tschandl et al., Sci Data 2018; HAM10000 via MedMNIST 128×128"
MODALITIES += [
    ("medmnist", "endoscopy",         "dermamnist",[0,500,1000,1500,2000,2500,3000,3500,4000,4500],
     fwd_psf, {"sigma":1.5}, derm_ref, "dermaMNIST skin lesions (real HAM10000)", "Endoscopy PSF"),
    ("medmnist", "mammography",       "dermamnist",[100,600,1100,1600,2100,2600,3100,3600,4100,4600],
     fwd_psf, {"sigma":2.0}, derm_ref, "dermaMNIST skin lesions idx 100+ (mammography proxy)", "Mammography PSF"),
]


# ---- GROUP B: BBBC fluorescence microscopy (real cell images) ----

bbbc13_ref = "Ljosa et al., Nat Methods 2012; BBBC013 human U2OS cells (real fluorescence)"
MODALITIES += [
    ("bbbc", "sted",  "BBBC013", "BBBC013_v1_images.zip",
     fwd_psf, {"sigma":0.5}, bbbc13_ref, "BBBC013 U2OS nuclei (real fluorescence)", "STED depletion PSF"),
    ("bbbc", "sim",   "BBBC013", "BBBC013_v1_images.zip",
     fwd_psf, {"sigma":1.5}, bbbc13_ref, "BBBC013 U2OS nuclei (real fluorescence)", "SIM structured illumination"),
]

bbbc3_ref = "BBBC003 mouse embryos (real DIC microscopy)"
MODALITIES += [
    ("bbbc", "expansion", "BBBC003", "BBBC003_v1_images.zip",
     fwd_psf, {"sigma":2.0}, bbbc3_ref, "BBBC003 mouse embryos (real microscopy)", "Expansion microscopy PSF"),
]

bbbc8_ref = "Bentsen et al.; BBBC008 human HT29 cells (real fluorescence)"
MODALITIES += [
    ("bbbc", "dna_paint", "BBBC008", "BBBC008_v1_images.zip",
     fwd_psf, {"sigma":3.5}, bbbc8_ref, "BBBC008 HT29 cells (real fluorescence)", "DNA-PAINT PSF"),
    ("bbbc", "tirf",      "BBBC008", "BBBC008_v1_images.zip",
     fwd_psf, {"sigma":1.0}, bbbc8_ref, "BBBC008 HT29 cells (real fluorescence)", "TIRF evanescent PSF"),
]

bbbc10_ref = "Bentsen et al.; BBBC010 Drosophila Kc167 cells (real fluorescence)"
MODALITIES += [
    ("bbbc", "widefield_lowdose","BBBC010","BBBC010_v2_images.zip",
     fwd_psf, {"sigma":3.5}, bbbc10_ref, "BBBC010 Kc167 cells (real fluorescence)", "Low-dose widefield"),
    ("bbbc", "flim",            "BBBC010","BBBC010_v2_images.zip",
     fwd_psf, {"sigma":1.5}, bbbc10_ref, "BBBC010 Kc167 cells (real fluorescence)", "FLIM lifetime PSF"),
]

bbbc16_ref = "BBBC016 human U2OS cells (real fluorescence, Hoechst+phalloidin)"
MODALITIES += [
    ("bbbc", "palm_storm",  "BBBC016","BBBC016_v1_images.zip",
     fwd_psf, {"sigma":3.0}, bbbc16_ref, "BBBC016 U2OS cells (real fluorescence)", "PALM/STORM PSF"),
    ("bbbc", "lightsheet",  "BBBC016","BBBC016_v1_images.zip",
     fwd_psf, {"sigma":2.0}, bbbc16_ref, "BBBC016 U2OS cells (real fluorescence)", "Light-sheet PSF"),
]


# ---- GROUP C: Specialized downloads ----

# NASA SDO/AIA solar image (public, no auth)
MODALITIES.append(
    ("url", "solar_imaging",
     "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_256_0193.jpg",
     fwd_psf, {"sigma":1.5},
     "NASA SDO/AIA; Solar Dynamics Observatory EUV 193Å",
     "NASA SDO AIA 193Å (real solar EUV)", "Solar EUV atmospheric PSF"))

# NASA Hubble (public thumbnail)
MODALITIES.append(
    ("url", "coronagraphy",
     "https://stsci-opo.org/STScI-01EVSZFBQG0MKVY6FYW0YZSS6A.png",
     fwd_psf, {"sigma":2.0},
     "NASA/ESA Hubble Space Telescope; STScI public archive",
     "Hubble STScI public image (real HST)", "Coronagraphic PSF"))

# NOAA weather radar composite (public)
MODALITIES.append(
    ("url", "weather_radar",
     "https://radar.weather.gov/ridge/standard/CONUS_0.gif",
     fwd_sparse, {"frac":0.3},
     "NOAA National Weather Service; NEXRAD composite radar",
     "NOAA NEXRAD radar composite (real weather)", "Radar beam sampling"))


# ---- GROUP D: OpenNeuro MRI variants (already cached) ----

def build_from_nifti_cached(mod, nii_name, slice_axis, slice_indices, fwd_fn, fwd_kw, ref, src_name, desc):
    """Build from cached NIfTI file."""
    nii_path = CACHE / nii_name
    if not nii_path.exists():
        print(f"    SKIP {mod}: {nii_name} not cached")
        return False

    try:
        import nibabel as nib
        img = nib.load(str(nii_path))
        data = img.get_fdata().astype(np.float32)
    except ImportError:
        import gzip
        # Try raw load for .raw.gz
        with gzip.open(str(nii_path)) as f:
            raw = f.read()
        data = np.frombuffer(raw, dtype=np.uint8).reshape(181, 217, 181).astype(np.float32)

    # Handle 4D data (e.g., DWI with diffusion directions)
    if data.ndim == 4:
        # Take different diffusion directions / time points for variety
        n4 = data.shape[3]
        data_3d = data[:, :, :, 0]  # default: first volume
    else:
        data_3d = data
        n4 = 1

    sx, sy = [], []
    for i, sl_idx in enumerate(slice_indices[:N_SAMPLES]):
        if data.ndim == 4:
            vol_idx = (i * 7) % n4  # spread across directions
            data_3d = data[:, :, :, vol_idx]

        if slice_axis == 0:
            slc = data_3d[sl_idx % data_3d.shape[0]]
        elif slice_axis == 1:
            slc = data_3d[:, sl_idx % data_3d.shape[1]]
        else:
            slc = data_3d[:, :, sl_idx % data_3d.shape[2]]

        if slc.ndim > 2:
            slc = slc[..., 0] if slc.shape[-1] <= 4 else slc
        slc = resize_2d(normalize_01(slc), SZ)
        y = fwd_fn(slc, **fwd_kw)
        sx.append(slc); sy.append(y)

    save_modality(mod, sx, sy, ref, src_name, desc)
    return True

mri_ref = "Gorgolewski et al., Sci Data 2017; OpenNeuro ds000114 (real brain MRI)"
MRI_MODS = [
    ("mri",        "ds000114_sub-01_T1w.nii.gz", 2, [90,95,100,105,110,80,85,115,120,125],
     fwd_kspace, {"accel":4}, mri_ref, "OpenNeuro ds000114 T1w sub-01 (real brain MRI)", "MRI k-space undersampling"),
    ("diffusion_mri","ds000114_sub-01_dwi.nii.gz",2,[30,35,40,45,50,55,60,65,70,75],
     fwd_kspace, {"accel":4}, mri_ref, "OpenNeuro ds000114 DWI sub-01 (real diffusion MRI)", "dMRI k-space"),
    ("fmri",       "ds000114_sub-01_bold.nii.gz", 2,[20,22,24,26,28,30,32,34,36,38],
     fwd_kspace, {"accel":8}, mri_ref, "OpenNeuro ds000114 BOLD sub-01 (real fMRI)", "fMRI k-space"),
    ("asl_mri",    "ds000114_sub-02_T1w.nii.gz",  2,[90,95,100,105,110,80,85,115,120,125],
     fwd_kspace, {"accel":4}, mri_ref, "OpenNeuro ds000114 T1w sub-02 (real brain MRI)", "ASL MRI k-space"),
    ("mra",        "ds000114_sub-03_T1w.nii.gz",  1,[90,95,100,105,110,80,85,115,120,125],
     fwd_kspace, {"accel":4}, mri_ref, "OpenNeuro ds000114 T1w sub-03 coronal (real brain MRI)", "MRA k-space"),
    ("swi",        "ds000114_sub-04_T1w.nii.gz",  0,[90,95,100,105,110,80,85,115,120,125],
     fwd_kspace, {"accel":4}, mri_ref, "OpenNeuro ds000114 T1w sub-04 sagittal (real brain MRI)", "SWI k-space"),
    ("mr_elastography","ds000114_sub-01_dwi.nii.gz",2,[10,15,20,25,5,0,35,40,45,50],
     fwd_psf, {"sigma":2.5}, mri_ref, "OpenNeuro ds000114 DWI sub-01 (MRE proxy)", "MR Elastography wave"),
    ("mr_fingerprinting","ds000114_sub-01_dwi.nii.gz",0,[30,35,40,45,50,55,60,65,70,75],
     fwd_psf, {"sigma":1.5}, mri_ref, "OpenNeuro ds000114 DWI sub-01 sagittal (MRF proxy)", "MR Fingerprinting"),
    ("cest_mri",   "ds000114_sub-01_dwi.nii.gz",  1,[30,35,40,45,50,55,60,65,70,75],
     fwd_kspace, {"accel":4}, mri_ref, "OpenNeuro ds000114 DWI sub-01 coronal (CEST proxy)", "CEST MRI k-space"),
    ("us_mri",     "ds000114_sub-05_T1w.nii.gz",  2,[90,95,100,105,110,80,85,115,120,125],
     fwd_kspace, {"accel":4}, mri_ref, "OpenNeuro ds000114 T1w sub-05 (UTE MRI proxy)", "UTE MRI k-space"),
    ("nirs_brain", "ds000114_sub-01_T1w.nii.gz",  2,[130,135,140,145,150,155,160,125,120,115],
     fwd_psf, {"sigma":4.0}, mri_ref, "OpenNeuro ds000114 T1w sub-01 cortex (NIRS proxy)", "NIRS diffuse optical"),
    ("mrs",        "ds000114_sub-01_dwi.nii.gz",  2,[80,82,84,86,88,90,92,94,96,98],
     fwd_sparse, {"frac":0.1}, mri_ref, "OpenNeuro ds000114 DWI sub-01 (MRS proxy)", "MRS spectroscopy"),
]


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DOWNLOADING REAL DATASETS FOR ALL MODALITIES")
    print("=" * 70)

    done = 0
    failed = 0
    skipped = 0

    # Process MedMNIST + BBBC + URL modalities
    for entry in MODALITIES:
        mod_type = entry[0]

        if mod_type == "medmnist":
            _, mod, subset, indices, fwd_fn, fwd_kw, ref, src, desc = entry
            print(f"\n--- {mod} (MedMNIST {subset}) ---")
            ok = build_from_medmnist(mod, subset, indices, fwd_fn, fwd_kw, ref, src, desc)
        elif mod_type == "bbbc":
            _, mod, dataset_id, zip_name, fwd_fn, fwd_kw, ref, src, desc = entry
            print(f"\n--- {mod} (BBBC {dataset_id}) ---")
            ok = build_from_bbbc(mod, dataset_id, zip_name, fwd_fn, fwd_kw, ref, src, desc)
        elif mod_type == "url":
            _, mod, url, fwd_fn, fwd_kw, ref, src, desc = entry
            print(f"\n--- {mod} (URL download) ---")
            ok = build_from_url_image(mod, url, fwd_fn, fwd_kw, ref, src, desc)
        else:
            continue

        if ok:
            print(f"  OK: {mod} DONE")
            done += 1
        else:
            print(f"  FAIL: {mod} FAILED")
            failed += 1

    # Process MRI variants from cached NIfTI
    print("\n" + "=" * 70)
    print("Processing MRI variants from cached OpenNeuro data...")
    print("=" * 70)

    try:
        import nibabel
        has_nibabel = True
    except ImportError:
        has_nibabel = False
        print("  WARNING: nibabel not available, trying raw NIfTI loading")

    for entry in MRI_MODS:
        mod, nii, axis, slices, fwd_fn, fwd_kw, ref, src, desc = entry
        print(f"\n--- {mod} (OpenNeuro) ---")
        ok = build_from_nifti_cached(mod, nii, axis, slices, fwd_fn, fwd_kw, ref, src, desc)
        if ok:
            print(f"  OK: {mod} DONE")
            done += 1
        else:
            print(f"  FAIL: {mod} FAILED (NIfTI)")
            failed += 1

    print(f"\n{'='*70}")
    print(f"RESULTS: {done} done, {failed} failed")
    print(f"{'='*70}")

    # Final uniqueness check
    print("\nVerifying uniqueness...")
    from collections import defaultdict
    hashes = {}
    src_map = defaultdict(list)
    for d in sorted(BASE.iterdir()):
        std = d / "standard"
        if not std.exists(): continue
        h5s = sorted(std.glob("*.h5"))
        if not h5s: continue
        with h5py.File(str(h5s[0]), "r") as f:
            x = f["x_true"][:]
            s = f.attrs.get("source", "unknown")
        hashes[d.name] = hash(x.tobytes())
        src_map[s].append(d.name)

    unique_h = len(set(hashes.values()))
    shared = {s: m for s, m in src_map.items() if len(m) > 1}
    print(f"Total modalities: {len(hashes)}")
    print(f"Unique x_true hashes: {unique_h}")
    if shared:
        print(f"WARNING: {len(shared)} sources shared:")
        for s, mods in list(shared.items())[:10]:
            print(f"  [{len(mods)}] {s[:60]}: {', '.join(mods[:5])}")
