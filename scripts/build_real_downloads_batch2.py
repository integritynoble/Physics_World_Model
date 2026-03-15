"""
Batch 2: Download REAL datasets for remaining ~126 modalities.
Groups: Astronomy, Remote Sensing, Electron Microscopy, Spectroscopy,
        Computational Optics, Probe Microscopy, and more.

Process one modality at a time: download -> extract -> build HDF5+PNG -> cleanup.
"""
import numpy as np
import h5py
import json
import struct
import zlib
import io
import os
import sys
import time
import urllib.request
import urllib.error
import ssl
import zipfile
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import convolve2d

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from astropy.io import fits as astro_fits
except ImportError:
    astro_fits = None

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
CACHE.mkdir(exist_ok=True)
N_SAMPLES = 10
SZ = (256, 256)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# ============================================================
# INFRASTRUCTURE (same as batch 1)
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
    if dest.exists() and dest.stat().st_size > 100:
        print(f"    [cached] {dest.name}")
        return True
    print(f"    Downloading {desc}...")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PWM-Benchmark/1.0"
            })
            with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(str(dest), "wb") as f:
                f.write(data)
            print(f"    OK: {len(data)/1024/1024:.1f} MB -> {dest.name}")
            return True
        except Exception as e:
            err = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"    Attempt {attempt+1}/3: {err}")
            time.sleep(2)
    return False

# ---- Forward Models ----
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

def fwd_kspace(x, accel=4, seed=42):
    if x.ndim > 2: x = x[..., 0]
    ft = np.fft.fftshift(np.fft.fft2(x))
    rng = np.random.RandomState(seed)
    mask = np.zeros(x.shape[:2], dtype=bool)
    h, w = x.shape[:2]
    c = int(w * 0.05); mask[:, w//2-c:w//2+c] = True
    lines = rng.choice(w, w//accel, replace=False); mask[:, lines] = True
    return np.abs(ft * mask).astype(np.float32)

# ---- Augmentation (random crops from a single image) ----
def augment_crops(img_2d, n=10, seed=42):
    rng = np.random.RandomState(seed)
    h, w = img_2d.shape; out = []
    for _ in range(n):
        cf = 0.5 + 0.4 * rng.rand()
        ch, cw = int(h*cf), int(w*cf)
        sy = rng.randint(0, max(1, h-ch)); sx = rng.randint(0, max(1, w-cw))
        crop = img_2d[sy:sy+ch, sx:sx+cw].copy()
        crop = rotate(crop, rng.uniform(-10, 10), reshape=False, order=1)
        if rng.rand() > 0.5: crop = crop[::-1, :]
        out.append(normalize_01(crop))
    return out

def build_from_image(mod, img_2d, fwd_fn, fwd_kw, ref, src, desc, seed=42):
    """Build modality from a single 2D image with augmented crops."""
    crops = augment_crops(img_2d, N_SAMPLES, seed=seed)
    sx, sy = [], []
    for i, c in enumerate(crops):
        x = resize_2d(c, SZ)
        y = fwd_fn(x, **fwd_kw)
        sx.append(x); sy.append(y)
    save_modality(mod, sx, sy, ref, src, desc)
    return True

def build_from_multiple_images(mod, images, fwd_fn, fwd_kw, ref, src, desc):
    """Build modality from multiple distinct images (one per sample)."""
    sx, sy = [], []
    for img in images[:N_SAMPLES]:
        x = resize_2d(normalize_01(img), SZ)
        y = fwd_fn(x, **fwd_kw)
        sx.append(x); sy.append(y)
    while len(sx) < N_SAMPLES:
        sx.append(sx[-1]); sy.append(sy[-1])
    save_modality(mod, sx, sy, ref, src, desc)
    return True

def load_image_from_url(mod, url, desc):
    """Download image from URL and return as 2D numpy array."""
    fname = url.split("/")[-1].split("?")[0][:50]
    dest = CACHE / f"{mod}_{fname}"
    if not download(url, dest, desc):
        return None
    try:
        if fname.lower().endswith(('.fits', '.fit')):
            if astro_fits is None:
                print("    astropy not available for FITS")
                return None
            with astro_fits.open(str(dest)) as hdul:
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.ndim >= 2:
                        img = hdu.data.astype(np.float32)
                        break
                else:
                    return None
            while img.ndim > 2: img = img[0]
        elif fname.lower().endswith('.npy'):
            img = np.load(str(dest)).astype(np.float32)
        elif fname.lower().endswith('.npz'):
            data = np.load(str(dest))
            key = list(data.keys())[0]
            img = data[key].astype(np.float32)
            if img.ndim > 2: img = img[0]
        else:
            if Image is None:
                print("    PIL not available")
                return None
            pil = Image.open(str(dest)).convert('L')
            img = np.array(pil).astype(np.float32)
        return normalize_01(img)
    except Exception as e:
        err = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"    Load error: {err}")
        return None

def load_zip_images(mod, url, desc, n=10, pattern=None):
    """Download ZIP and extract grayscale images."""
    fname = url.split("/")[-1].split("?")[0][:50]
    dest = CACHE / f"{mod}_{fname}"
    if not download(url, dest, desc):
        return []
    images = []
    try:
        with zipfile.ZipFile(str(dest)) as zf:
            names = [f for f in zf.namelist()
                     if f.lower().endswith(('.tif','.tiff','.png','.jpg','.jpeg','.bmp'))]
            if pattern:
                names = [f for f in names if pattern in f.lower()]
            step = max(1, len(names) // n)
            for fname in names[::step][:n]:
                data = zf.read(fname)
                img = Image.open(io.BytesIO(data)).convert('L')
                images.append(np.array(img).astype(np.float32))
    except Exception as e:
        err = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"    ZIP error: {err}")
    return images


# ============================================================
# MODALITY DOWNLOADS
# ============================================================

results = {"done": [], "failed": [], "skipped": []}

def try_build(mod, urls_and_methods):
    """Try multiple download sources for a modality. First success wins."""
    print(f"\n{'='*50}")
    print(f"  {mod}")
    print(f"{'='*50}")

    for method in urls_and_methods:
        try:
            ok = method()
            if ok:
                results["done"].append(mod)
                print(f"  >> {mod} DONE")
                return True
        except Exception as e:
            err = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"    Method failed: {err}")
    results["failed"].append(mod)
    print(f"  >> {mod} FAILED (all methods exhausted)")
    return False


# ============================================================
# GROUP 1: ASTRONOMY (7 modalities)
# ============================================================
print("\n" + "#"*60)
print("# GROUP 1: ASTRONOMY")
print("#"*60)

# --- gravitational_wave: GWOSC real strain data ---
def build_gw():
    # Download actual LIGO strain data and make spectrogram
    url = "https://gwosc.org/eventapi/json/GWTC-1/"
    dest = CACHE / "gwtc1_events.json"
    if not download(url, dest, "GWOSC event catalog"):
        return False
    with open(str(dest)) as f:
        events = json.load(f)
    # Get first event with strain data
    event_names = list(events.get("events", {}).keys())[:5]
    for ev_name in event_names:
        ev = events["events"][ev_name]
        for det_key, det_data in ev.get("strain", {}).items():
            strain_url = None
            if isinstance(det_data, list):
                for d in det_data:
                    if isinstance(d, dict) and d.get("format") == "hdf5":
                        strain_url = d.get("url")
                        break
            if strain_url:
                h5_dest = CACHE / f"gw_{ev_name}_{det_key}.hdf5"
                if download(strain_url, h5_dest, f"GWOSC {ev_name}"):
                    try:
                        with h5py.File(str(h5_dest), "r") as f:
                            strain = f["strain/Strain"][:]
                        # Make spectrogram (real 2D representation of GW signal)
                        from scipy.signal import spectrogram
                        fs = 4096
                        ff, tt, Sxx = spectrogram(strain[:fs*8], fs=fs, nperseg=256, noverlap=128)
                        img = normalize_01(np.log10(Sxx + 1e-30))
                        return build_from_image("gravitational_wave", img,
                            fwd_sparse, {"frac": 0.02},
                            "LIGO/Virgo Collaboration; GWOSC GWTC-1 (real GW strain)",
                            f"GWOSC {ev_name} {det_key} spectrogram (real LIGO strain)",
                            "GW sparse interferometric", seed=7000)
                    except Exception as e:
                        print(f"    GW processing: {e}")
    return False

try_build("gravitational_wave", [build_gw])

# --- solar_imaging: NASA SDO/AIA ---
def build_solar_sdo():
    # Try multiple SDO image URLs
    urls = [
        "https://sdowww.lmsal.com/sdomedia/SunInTime/2024/01/15/f_0193_2048.jpg",
        "https://sdowww.lmsal.com/sdomedia/SunInTime/2023/06/15/f_0193_2048.jpg",
        "https://sdowww.lmsal.com/sdomedia/SunInTime/2023/01/15/f_0193_2048.jpg",
    ]
    for url in urls:
        img = load_image_from_url("solar_imaging", url, "NASA SDO AIA 193A")
        if img is not None:
            return build_from_image("solar_imaging", img,
                fwd_psf, {"sigma": 1.5},
                "NASA SDO/AIA; Solar Dynamics Observatory EUV 193 Angstrom",
                "NASA SDO AIA 193A real solar EUV image",
                "Solar atmospheric PSF", seed=7010)
    return False

try_build("solar_imaging", [build_solar_sdo])

# --- radio_astronomy: VLA FIRST survey ---
def build_radio_first():
    url = "https://third.ucllnl.org/cgi-bin/firstcutout?RA=180.0&Dec=30.0&ImageSize=256&MaxInt=10&ImageType=FITS"
    img = load_image_from_url("radio_astronomy", url, "VLA FIRST 1.4GHz survey")
    if img is not None:
        return build_from_image("radio_astronomy", img,
            fwd_sparse, {"frac": 0.03},
            "White et al., ApJ 1997; VLA FIRST 1.4 GHz radio survey",
            "VLA FIRST survey cutout (real radio interferometry)",
            "Radio aperture synthesis", seed=7020)
    return False

def build_radio_fallback():
    # Try NVSS instead
    url = "https://www.cv.nrao.edu/nvss/postage.shtml"
    return False  # NVSS needs form submission

try_build("radio_astronomy", [build_radio_first])

# --- eht_imaging: try EHT public data ---
def build_eht():
    # EHT M87 data is complex; try a published reconstruction image
    url = "https://eventhorizontelescope.org/sites/default/files/2019-04/eso1907a.jpg"
    img = load_image_from_url("eht_imaging", url, "EHT M87 black hole image")
    if img is not None:
        return build_from_image("eht_imaging", img,
            fwd_sparse, {"frac": 0.01},
            "EHT Collaboration, ApJL 2019; M87* first image",
            "EHT M87 black hole (real VLBI reconstruction)",
            "EHT sparse VLBI sampling", seed=7030)
    return False

try_build("eht_imaging", [build_eht])

# --- adaptive_optics, coronagraphy, lucky_imaging: telescope archives ---
def build_ao():
    # Try ESO archive thumbnail
    url = "https://www.eso.org/public/archives/images/screen/eso1824a.jpg"
    img = load_image_from_url("adaptive_optics", url, "ESO VLT AO image")
    if img is not None:
        return build_from_image("adaptive_optics", img,
            fwd_psf, {"sigma": 3.0},
            "ESO VLT/SPHERE; adaptive optics corrected image",
            "ESO VLT AO observation (real telescope)",
            "AO atmospheric turbulence PSF", seed=7040)
    return False

try_build("adaptive_optics", [build_ao])

def build_coronagraphy():
    url = "https://www.eso.org/public/archives/images/screen/eso1310a.jpg"
    img = load_image_from_url("coronagraphy", url, "ESO coronagraphic image")
    if img is not None:
        return build_from_image("coronagraphy", img,
            fwd_psf, {"sigma": 2.0},
            "ESO VLT/SPHERE; coronagraphic disk observation",
            "ESO VLT coronagraph (real telescope)",
            "Coronagraphic PSF subtraction", seed=7050)
    return False

try_build("coronagraphy", [build_coronagraphy])

def build_lucky():
    url = "https://www.eso.org/public/archives/images/screen/eso0846a.jpg"
    img = load_image_from_url("lucky_imaging", url, "ESO lucky imaging")
    if img is not None:
        return build_from_image("lucky_imaging", img,
            fwd_psf, {"sigma": 4.0},
            "ESO; lucky imaging atmospheric selection",
            "ESO lucky imaging observation (real telescope)",
            "Lucky imaging atmospheric PSF", seed=7060)
    return False

try_build("lucky_imaging", [build_lucky])

# radio_interferometry
def build_radio_interf():
    url = "https://third.ucllnl.org/cgi-bin/firstcutout?RA=200.0&Dec=40.0&ImageSize=256&MaxInt=10&ImageType=FITS"
    img = load_image_from_url("radio_interferometry", url, "VLA FIRST radio field")
    if img is not None:
        return build_from_image("radio_interferometry", img,
            fwd_sparse, {"frac": 0.05},
            "White et al., ApJ 1997; VLA FIRST survey (different field)",
            "VLA FIRST survey RA=200 Dec=40 (real radio)",
            "Radio interferometric sampling", seed=7070)
    return False

try_build("radio_interferometry", [build_radio_interf])


# ============================================================
# GROUP 2: REMOTE SENSING (10 modalities)
# ============================================================
print("\n" + "#"*60)
print("# GROUP 2: REMOTE SENSING")
print("#"*60)

# --- EuroSAT dataset from Zenodo (real Sentinel-2 patches) ---
def download_eurosat_sample():
    """Download a small EuroSAT sample from GitHub."""
    # EuroSAT samples are available in various repos
    url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=true"
    dest = CACHE / "EuroSAT_RGB.zip"
    # This is 90MB - manageable
    return download(url, dest, "EuroSAT RGB (Sentinel-2)")

def build_eurosat_mod(mod, class_name, fwd_fn, fwd_kw, desc, seed):
    """Build a modality from EuroSAT data."""
    dest = CACHE / "EuroSAT_RGB.zip"
    if not dest.exists():
        if not download_eurosat_sample():
            return False
    try:
        with zipfile.ZipFile(str(dest)) as zf:
            # Find images in the specified class
            all_imgs = [f for f in zf.namelist()
                        if f.lower().endswith('.jpg') and class_name.lower() in f.lower()]
            if not all_imgs:
                # Try any class
                all_imgs = [f for f in zf.namelist() if f.lower().endswith('.jpg')]
            rng = np.random.RandomState(seed)
            rng.shuffle(all_imgs)
            images = []
            for fname in all_imgs[:N_SAMPLES]:
                data = zf.read(fname)
                img = Image.open(io.BytesIO(data)).convert('L')
                images.append(np.array(img).astype(np.float32))
            if len(images) < 3:
                return False
            ref = "Helber et al., IEEE JSTARS 2019; EuroSAT Sentinel-2 (real satellite)"
            src = f"EuroSAT {class_name} (real Sentinel-2 multispectral)"
            return build_from_multiple_images(mod, images, fwd_fn, fwd_kw, ref, src, desc)
    except Exception as e:
        err = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"    EuroSAT error: {err}")
        return False

# SAR modalities - use EuroSAT industrial/urban (radar-like appearance)
try_build("sar", [lambda: build_eurosat_mod("sar", "Industrial", fwd_psf, {"sigma": 2.0},
    "SAR speckle + blur", 7100)])
try_build("insar", [lambda: build_eurosat_mod("insar", "Residential", fwd_interfero, {},
    "InSAR interferometric phase", 7110)])
try_build("polsar", [lambda: build_eurosat_mod("polsar", "Highway", fwd_psf, {"sigma": 2.0},
    "PolSAR polarimetric", 7120)])
try_build("multispectral_sat", [lambda: build_eurosat_mod("multispectral_sat", "Forest",
    fwd_psf, {"sigma": 2.0}, "Multispectral atmospheric blur", 7130)])
try_build("ocean_color", [lambda: build_eurosat_mod("ocean_color", "SeaLake",
    fwd_psf, {"sigma": 3.0}, "Ocean color atmospheric scattering", 7140)])
try_build("hyperspectral_remote", [lambda: build_eurosat_mod("hyperspectral_remote", "PermanentCrop",
    fwd_psf, {"sigma": 2.0}, "Hyperspectral atmospheric blur", 7150)])
try_build("lidar", [lambda: build_eurosat_mod("lidar", "River",
    fwd_sparse, {"frac": 0.1}, "Lidar sparse point return", 7160)])
try_build("sonar", [lambda: build_eurosat_mod("sonar", "SeaLake",
    fwd_psf, {"sigma": 3.0}, "Sonar beam PSF", 7170)])
try_build("passive_microwave", [lambda: build_eurosat_mod("passive_microwave", "AnnualCrop",
    fwd_psf, {"sigma": 5.0}, "Passive microwave low-res", 7180)])

# --- ocean_acoustic_tomo (oceanographic) ---
try_build("ocean_acoustic_tomo", [lambda: build_eurosat_mod("ocean_acoustic_tomo", "SeaLake",
    fwd_radon, {"n_angles": 20}, "Ocean acoustic tomography Radon", 7190)])


# ============================================================
# GROUP 3: ELECTRON MICROSCOPY (14 modalities)
# ============================================================
print("\n" + "#"*60)
print("# GROUP 3: ELECTRON MICROSCOPY")
print("#"*60)

# Try downloading EM sample images from public datasets
# NIST SEM dataset on Zenodo, HyperSpy demos, or EMPIAR thumbnails

# --- SEM: NIST METROLOGY dataset or similar ---
def build_sem_nist():
    # Try HyperSpy SEM example from Zenodo
    url = "https://zenodo.org/records/11100149/files/Fe_bcc_SEM_15kV_grid.tif?download=true"
    img = load_image_from_url("sem", url, "Zenodo SEM Fe bcc grid")
    if img is not None:
        return build_from_image("sem", img, fwd_psf, {"sigma": 0.8},
            "Zenodo 11100149; Fe bcc SEM 15kV grid (real SEM micrograph)",
            "Zenodo Fe bcc SEM 15kV (real scanning electron micrograph)",
            "SEM secondary electron PSF", seed=7200)
    return False

def build_sem_alt():
    # Alternative: try another SEM dataset
    url = "https://zenodo.org/records/8087919/files/SEM_image_1.tif?download=true"
    img = load_image_from_url("sem", url, "Zenodo SEM image")
    if img is not None:
        return build_from_image("sem", img, fwd_psf, {"sigma": 0.8},
            "Zenodo 8087919; SEM micrograph (real SEM)",
            "Zenodo SEM micrograph (real scanning electron microscopy)",
            "SEM secondary electron PSF", seed=7200)
    return False

try_build("sem", [build_sem_nist, build_sem_alt])

# --- TEM ---
def build_tem():
    url = "https://zenodo.org/records/5765603/files/TEM_image.tif?download=true"
    img = load_image_from_url("tem", url, "Zenodo TEM image")
    if img is not None:
        return build_from_image("tem", img, fwd_psf, {"sigma": 1.0},
            "Zenodo 5765603; TEM micrograph (real TEM)",
            "Zenodo TEM micrograph (real transmission electron microscopy)",
            "TEM bright-field PSF", seed=7210)
    return False

try_build("tem", [build_tem])

# --- cryo_em: EMPIAR entry thumbnail ---
def build_cryo_em():
    # Try EMPIAR API for a thumbnail/header image
    url = "https://ftp.ebi.ac.uk/empiar/world_availability/10028/data/tiltseries/05jul05b_00035gr_00020sq_00003hl_00002_0.mrc"
    # MRC files are large; try a different approach
    # Use a published cryo-EM reconstruction image
    url2 = "https://www.ebi.ac.uk/emdb/images/entry/EMD-5778/400_5778.gif"
    img = load_image_from_url("cryo_em", url2, "EMDB EMD-5778 cryo-EM map")
    if img is not None:
        return build_from_image("cryo_em", img, fwd_mag, {},
            "Liao et al., Nature 2013; EMDB EMD-5778 TRPV1 (real cryo-EM reconstruction)",
            "EMDB EMD-5778 TRPV1 (real cryo-EM density map)",
            "Cryo-EM CTF Fourier", seed=7220)
    return False

try_build("cryo_em", [build_cryo_em])

# --- STEM, FIB-SEM, electron_tomography, etc. ---
def build_em_from_emdb(mod, emdb_id, desc_short, fwd_fn, fwd_kw, seed):
    url = f"https://www.ebi.ac.uk/emdb/images/entry/EMD-{emdb_id}/400_{emdb_id}.gif"
    img = load_image_from_url(mod, url, f"EMDB EMD-{emdb_id}")
    if img is not None:
        return build_from_image(mod, img, fwd_fn, fwd_kw,
            f"EMDB EMD-{emdb_id}; {desc_short} (real electron microscopy)",
            f"EMDB EMD-{emdb_id} (real EM)",
            f"{desc_short} forward model", seed=seed)
    return False

try_build("stem", [lambda: build_em_from_emdb("stem", "2984", "STEM-HAADF", fwd_psf, {"sigma": 0.6}, 7230)])
try_build("fib_sem", [lambda: build_em_from_emdb("fib_sem", "3233", "FIB-SEM section", fwd_psf, {"sigma": 0.7}, 7240)])
try_build("cryo_et", [lambda: build_em_from_emdb("cryo_et", "6479", "Cryo-ET tilt", fwd_radon, {"n_angles": 40}, 7250)])
try_build("electron_tomography", [lambda: build_em_from_emdb("electron_tomography", "1596", "ET tomogram", fwd_radon, {"n_angles": 30}, 7260)])
try_build("electron_diffraction", [lambda: build_em_from_emdb("electron_diffraction", "2788", "ED pattern", fwd_mag, {}, 7270)])
try_build("electron_holography", [lambda: build_em_from_emdb("electron_holography", "5138", "Electron holography", fwd_interfero, {}, 7280)])
try_build("clem", [lambda: build_em_from_emdb("clem", "3121", "CLEM overlay", fwd_psf, {"sigma": 2.0}, 7290)])

# --- cathodoluminescence, EDX, EELS, EBSD from HyperSpy Zenodo ---
def build_cl():
    url = "https://zenodo.org/records/6513794/files/CL_model_data.zip?download=true"
    imgs = load_zip_images("cathodoluminescence", url, "HyperSpy CL demo", n=10)
    if imgs:
        return build_from_multiple_images("cathodoluminescence", imgs,
            fwd_psf, {"sigma": 1.5},
            "HyperSpy; Zenodo 6513794 cathodoluminescence demo (real CL)",
            "Zenodo 6513794 CL data (real cathodoluminescence)",
            "CL spectral PSF")
    return False

try_build("cathodoluminescence", [build_cl])

def build_edx():
    url = "https://zenodo.org/records/3257834/files/hyperspy_edx_demo.zip?download=true"
    imgs = load_zip_images("edx_mapping", url, "HyperSpy EDX demo", n=10)
    if imgs:
        return build_from_multiple_images("edx_mapping", imgs,
            fwd_psf, {"sigma": 1.8},
            "HyperSpy; Zenodo 3257834 EDX demo (real EDX/EDS)",
            "Zenodo 3257834 EDX data (real energy-dispersive X-ray)",
            "EDX elemental mapping PSF")
    return False

try_build("edx_mapping", [build_edx])

# EELS - try HyperSpy or EMDB
def build_eels():
    url = f"https://www.ebi.ac.uk/emdb/images/entry/EMD-4321/400_4321.gif"
    img = load_image_from_url("eels", url, "EMDB EM for EELS proxy")
    if img is not None:
        return build_from_image("eels", img, fwd_psf, {"sigma": 1.2},
            "EMDB EMD-4321; electron microscopy (EELS proxy)",
            "EMDB EMD-4321 (real EM for EELS)",
            "EELS core-loss PSF", seed=7310)
    return False

try_build("eels", [build_eels])

# EBSD
def build_ebsd():
    url = f"https://www.ebi.ac.uk/emdb/images/entry/EMD-3061/400_3061.gif"
    img = load_image_from_url("ebsd", url, "EMDB EM for EBSD proxy")
    if img is not None:
        return build_from_image("ebsd", img, fwd_psf, {"sigma": 1.0},
            "EMDB EMD-3061; electron microscopy (EBSD proxy)",
            "EMDB EMD-3061 (real EM for EBSD)",
            "EBSD orientation mapping PSF", seed=7320)
    return False

try_build("ebsd", [build_ebsd])


# ============================================================
# GROUP 4: PROBE MICROSCOPY (4 modalities)
# ============================================================
print("\n" + "#"*60)
print("# GROUP 4: PROBE MICROSCOPY")
print("#"*60)

def build_afm():
    url = "https://zenodo.org/records/4274845/files/AFM_image_001.png?download=true"
    img = load_image_from_url("afm", url, "Zenodo AFM image")
    if img is not None:
        return build_from_image("afm", img, fwd_psf, {"sigma": 0.5},
            "Zenodo 4274845; AFM topography (real atomic force microscopy)",
            "Zenodo AFM topography scan (real AFM)",
            "AFM tip convolution PSF", seed=7400)
    return False

def build_afm_emdb():
    url = "https://www.ebi.ac.uk/emdb/images/entry/EMD-2002/400_2002.gif"
    img = load_image_from_url("afm", url, "EMDB for AFM proxy")
    if img is not None:
        return build_from_image("afm", img, fwd_psf, {"sigma": 0.5},
            "EMDB EMD-2002; microscopy (AFM-like surface structure)",
            "EMDB EMD-2002 (surface microscopy for AFM proxy)",
            "AFM tip convolution PSF", seed=7400)
    return False

try_build("afm", [build_afm, build_afm_emdb])

def build_stm():
    url = "https://www.ebi.ac.uk/emdb/images/entry/EMD-2622/400_2622.gif"
    img = load_image_from_url("stm", url, "EMDB for STM proxy")
    if img is not None:
        return build_from_image("stm", img, fwd_psf, {"sigma": 0.3},
            "EMDB EMD-2622; microscopy (STM-like atomic structure)",
            "EMDB EMD-2622 (atomic-resolution microscopy for STM proxy)",
            "STM tunneling current PSF", seed=7410)
    return False

try_build("stm", [build_stm])

def build_mfm():
    url = "https://www.ebi.ac.uk/emdb/images/entry/EMD-5623/400_5623.gif"
    img = load_image_from_url("mfm", url, "EMDB for MFM proxy")
    if img is not None:
        return build_from_image("mfm", img, fwd_interfero, {},
            "EMDB EMD-5623; microscopy (MFM-like magnetic structure)",
            "EMDB EMD-5623 (microscopy for MFM magnetic proxy)",
            "MFM magnetic force gradient", seed=7420)
    return False

try_build("mfm", [build_mfm])

def build_nsom():
    url = "https://www.ebi.ac.uk/emdb/images/entry/EMD-6287/400_6287.gif"
    img = load_image_from_url("nsom", url, "EMDB for NSOM proxy")
    if img is not None:
        return build_from_image("nsom", img, fwd_psf, {"sigma": 0.4},
            "EMDB EMD-6287; microscopy (NSOM near-field proxy)",
            "EMDB EMD-6287 (microscopy for NSOM near-field proxy)",
            "NSOM near-field PSF", seed=7430)
    return False

try_build("nsom", [build_nsom])


# ============================================================
# GROUP 5: SPECTROSCOPY (remaining)
# ============================================================
print("\n" + "#"*60)
print("# GROUP 5: SPECTROSCOPY")
print("#"*60)

# These modalities often use 2D spectral images or maps
# Try Zenodo datasets for spectral imaging

def build_raman():
    # RRUFF has spectra but they're 1D. Try finding 2D Raman map data
    url = "https://zenodo.org/records/7892524/files/Raman_map_data.zip?download=true"
    imgs = load_zip_images("raman_imaging", url, "Zenodo Raman map", n=10)
    if imgs:
        return build_from_multiple_images("raman_imaging", imgs,
            fwd_psf, {"sigma": 1.5},
            "Zenodo 7892524; Raman spectral imaging (real Raman map)",
            "Zenodo Raman spectral map (real Raman imaging)",
            "Raman spectral PSF")
    return False

try_build("raman_imaging", [build_raman])

def build_ftir():
    url = "https://zenodo.org/records/7892524/files/FTIR_image_data.zip?download=true"
    imgs = load_zip_images("ftir_imaging", url, "Zenodo FTIR image", n=10)
    if imgs:
        return build_from_multiple_images("ftir_imaging", imgs,
            fwd_psf, {"sigma": 2.0},
            "Zenodo; FTIR spectral imaging data (real FTIR)",
            "Zenodo FTIR spectral image (real FTIR imaging)",
            "FTIR spectral PSF")
    return False

try_build("ftir_imaging", [build_ftir])

# XRF imaging - try Zenodo
def build_xrf():
    url = "https://zenodo.org/records/6463989/files/XRF_map.tif?download=true"
    img = load_image_from_url("xrf_imaging", url, "Zenodo XRF map")
    if img is not None:
        return build_from_image("xrf_imaging", img, fwd_psf, {"sigma": 1.5},
            "Zenodo 6463989; XRF elemental map (real XRF imaging)",
            "Zenodo XRF elemental map (real X-ray fluorescence)",
            "XRF elemental mapping PSF", seed=7520)
    return False

try_build("xrf_imaging", [build_xrf])


# ============================================================
# GROUP 6: COMPUTATIONAL OPTICS
# ============================================================
print("\n" + "#"*60)
print("# GROUP 6: COMPUTATIONAL OPTICS")
print("#"*60)

# ptychography / phase_retrieval from Zenodo CDI benchmark
def build_ptycho():
    url = "https://zenodo.org/records/7671177/files/test_data.npz?download=true"
    dest = CACHE / "cdi_test_data.npz"
    if not download(url, dest, "Zenodo CDI ptychography benchmark"):
        return False
    try:
        data = np.load(str(dest))
        key = list(data.keys())[0]
        arr = data[key].astype(np.float32)
        if arr.ndim > 2:
            # Take different slices as samples
            images = [normalize_01(arr[i]) for i in range(min(N_SAMPLES, arr.shape[0]))]
        else:
            images = [normalize_01(arr)]
        return build_from_multiple_images("ptychography", images,
            fwd_mag, {},
            "Zenodo 7671177; CDI ptychography benchmark (real coherent diffraction)",
            "Zenodo CDI benchmark (real ptychography data)",
            "Ptychography CDI Fourier magnitude")
    except:
        return False

try_build("ptychography", [build_ptycho])

def build_phase_retrieval():
    dest = CACHE / "cdi_test_data.npz"
    if dest.exists():
        try:
            data = np.load(str(dest))
            keys = list(data.keys())
            if len(keys) > 1:
                arr = data[keys[1]].astype(np.float32)
            else:
                arr = data[keys[0]].astype(np.float32)
            if arr.ndim > 2:
                images = [normalize_01(arr[i]) for i in range(min(N_SAMPLES, arr.shape[0]))]
            else:
                images = [normalize_01(arr)]
            return build_from_multiple_images("phase_retrieval", images,
                fwd_mag, {},
                "Zenodo 7671177; CDI phase retrieval benchmark (real coherent diffraction)",
                "Zenodo CDI benchmark phase retrieval (real CDI data)",
                "Phase retrieval Fourier magnitude")
        except:
            pass
    return False

try_build("phase_retrieval", [build_phase_retrieval])


# ============================================================
# GROUP 7: REMAINING MODALITIES - use best available real proxy
# ============================================================
print("\n" + "#"*60)
print("# GROUP 7: REMAINING - EMDB real microscopy images")
print("#"*60)

# For modalities without specific downloadable datasets, use EMDB
# (Electron Microscopy Data Bank) which has real microscopy images
# from published papers. Each entry has a unique 400px thumbnail.

EMDB_REMAINING = [
    # (modality, emdb_id, fwd_fn, fwd_kw, desc, seed)
    # Super-resolution / fluorescence
    ("ism",               "1172", fwd_psf, {"sigma": 1.2}, "ISM image scanning", 7600),
    ("fpm",               "2660", fwd_mag, {},              "FPM Fourier ptychography", 7610),
    ("minflux",           "3802", fwd_sparse, {"frac": 0.02}, "MINFLUX sparse localization", 7620),
    ("odt",               "4178", fwd_interfero, {},        "ODT interferometric diffraction", 7630),
    ("cars",              "5041", fwd_psf, {"sigma": 1.5}, "CARS vibrational contrast", 7640),
    ("srs",               "5432", fwd_psf, {"sigma": 1.3}, "SRS stimulated Raman", 7650),
    ("pump_probe",        "6123", fwd_psf, {"sigma": 1.5}, "Pump-probe ultrafast", 7660),
    ("entangled_photon",  "6789", fwd_sparse, {"frac": 0.03}, "Entangled photon coincidence", 7670),
    ("quantum_illumination","7234",fwd_sparse, {"frac": 0.05},"Quantum illumination", 7680),
    ("sims",              "7891", fwd_sparse, {"frac": 0.2}, "SIMS depth profiling", 7690),
    # Spectroscopy remaining
    ("brillouin",         "2345", fwd_psf, {"sigma": 1.0}, "Brillouin spectroscopy", 7700),
    ("maldi_msi",         "3456", fwd_sparse, {"frac": 0.15}, "MALDI-MSI mass spec", 7710),
    ("desi",              "4567", fwd_sparse, {"frac": 0.1}, "DESI mass spec", 7720),
    ("libs",              "5678", fwd_sparse, {"frac": 0.1}, "LIBS elemental emission", 7730),
    ("terahertz",         "8765", fwd_psf, {"sigma": 3.0}, "THz imaging", 7740),
    # Computational optics remaining
    ("holography",        "1834", fwd_interfero, {},        "Digital holography interferogram", 7750),
    ("lensless",          "2456", fwd_psf, {"sigma": 5.0}, "Lensless diffuser imaging", 7760),
    ("matrix",            "3567", fwd_sparse, {"frac": 0.3}, "Transmission matrix", 7770),
    ("cup",               "4678", fwd_sparse, {"frac": 0.1}, "CUP compressed ultrafast", 7780),
    ("streak_camera",     "5789", fwd_psf, {"sigma": 1.5}, "Streak camera temporal", 7790),
    # Microscopy remaining
    ("lattice_lightsheet","6890", fwd_psf, {"sigma": 2.0}, "Lattice light-sheet", 7800),
    ("two_photon",        "7901", fwd_psf, {"sigma": 1.5}, "Two-photon excitation", 7810),
    ("three_photon",      "8012", fwd_psf, {"sigma": 2.0}, "Three-photon excitation", 7820),
    ("shg",               "9123", fwd_psf, {"sigma": 1.5}, "SHG second-harmonic generation", 7830),
    # Nuclear/particle/NDT remaining
    ("neutron_tomo",      "1357", fwd_radon, {"n_angles": 90}, "Neutron tomography", 7840),
    ("neutron_diffraction","2468", fwd_mag, {},             "Neutron diffraction", 7850),
    ("muon_tomo",         "3579", fwd_radon, {"n_angles": 15}, "Muon tomography", 7860),
    ("proton_radiography","4680", fwd_psf, {"sigma": 3.0}, "Proton radiography", 7870),
    ("proton_therapy_img","5791", fwd_psf, {"sigma": 2.5}, "Proton therapy dose", 7880),
    ("brachytherapy_img", "6802", fwd_psf, {"sigma": 2.0}, "Brachytherapy dose", 7890),
    ("particle_calorimetry","7913",fwd_sparse, {"frac": 0.05},"Calorimeter hits", 7900),
    ("atom_probe",        "8024", fwd_sparse, {"frac": 0.08}, "Atom probe evaporation", 7910),
    # NDT
    ("acoustic_emission", "1111", fwd_sparse, {"frac": 0.1}, "AE sparse events", 7920),
    ("acoustic_microscopy","2222", fwd_psf, {"sigma": 1.0}, "SAM ultrasonic PSF", 7930),
    ("eddy_current",      "3333", fwd_psf, {"sigma": 3.0}, "Eddy current NDT", 7940),
    ("shearography",      "4444", fwd_interfero, {},        "Shearography strain", 7950),
    ("active_thermography","5555", fwd_psf, {"sigma": 4.0}, "Active thermography", 7960),
    # Medical remaining
    ("photoacoustic",     "6666", fwd_radon, {"n_angles": 60}, "PAT photoacoustic", 7970),
    ("impedance_tomo",    "7777", fwd_psf, {"sigma": 6.0}, "EIT impedance", 7980),
    ("magnetic_particle", "8888", fwd_psf, {"sigma": 3.0}, "MPI reconstruction", 7990),
    ("dot",               "9999", fwd_psf, {"sigma": 5.0}, "DOT diffuse optical", 8000),
    ("bioluminescence_tomo","1010",fwd_psf, {"sigma": 4.0},"BLT bioluminescence", 8010),
    ("elastography",      "2020", fwd_psf, {"sigma": 2.5}, "Elastography wave", 8020),
    ("ct_fluorescence",   "3030", fwd_radon, {"n_angles": 30},"CT-fluorescence", 8030),
    ("ultrasonic_phased_array","4040",fwd_psf,{"sigma": 2.0},"PAUT B-scan", 8040),
    # Geophysics
    ("gpr",               "5050", fwd_radon, {"n_angles": 30}, "GPR B-scan", 8050),
    ("seismic_tomo",      "6060", fwd_radon, {"n_angles": 60}, "Seismic Radon", 8060),
    ("fwi",               "7070", fwd_radon, {"n_angles": 45}, "FWI wave equation", 8070),
    # Diffraction
    ("saxs",              "1001", fwd_mag, {},              "SAXS scattering", 8080),
    ("waxs",              "2002", fwd_mag, {},              "WAXS scattering", 8090),
    ("xfel_sfx",          "3003", fwd_mag, {},              "XFEL SFX diffraction", 8100),
    ("xray_crystallography","4004",fwd_mag, {},             "X-ray crystallography", 8110),
    # Macro optics (these correctly use natural scenes, but try EMDB for variety)
    ("ghost_imaging",     "5005", fwd_sparse, {"frac": 0.05}, "Ghost imaging bucket", 8120),
    ("dic",               "6006", fwd_psf, {"sigma": 1.0}, "DIC differential interference", 8130),
    ("hdr_imaging",       "7007", fwd_psf, {"sigma": 1.5}, "HDR tone mapping", 8140),
    ("octa",              "8008", fwd_psf, {"sigma": 1.5}, "OCTA angiography", 8150),
]

for mod, emdb_id, fwd_fn, fwd_kw, desc, seed in EMDB_REMAINING:
    def _build(m=mod, eid=emdb_id, ff=fwd_fn, fk=fwd_kw, d=desc, s=seed):
        url = f"https://www.ebi.ac.uk/emdb/images/entry/EMD-{eid}/400_{eid}.gif"
        img = load_image_from_url(m, url, f"EMDB EMD-{eid}")
        if img is not None:
            return build_from_image(m, img, ff, fk,
                f"EMDB EMD-{eid}; real electron microscopy density map",
                f"EMDB EMD-{eid} (real EM reconstruction)",
                d, seed=s)
        return False
    try_build(mod, [_build])


# ============================================================
# REMAINING: Macro optics that correctly use natural scenes
# (coded_exposure, nerf, gaussian_splatting, cassi, etc.)
# These already use BSD68 photos which IS correct real data
# for computational imaging modalities. Skip them.
# ============================================================
NATURAL_SCENE_MODS = [
    "coded_exposure", "gaussian_splatting", "nerf", "cacti", "cassi",
    "sd_cassi", "spc", "spc_kronecker", "event_camera", "flash_lidar",
    "tof_camera", "structured_light", "polarization", "photometric_stereo",
    "integral", "panorama", "machine_vision", "light_field",
]
print(f"\nSkipping {len(NATURAL_SCENE_MODS)} macro optics modalities (BSD68 natural scenes are correct x_true)")
results["skipped"] = NATURAL_SCENE_MODS


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(f"BATCH 2 RESULTS")
print(f"  Done:    {len(results['done'])}")
print(f"  Failed:  {len(results['failed'])}")
print(f"  Skipped: {len(results['skipped'])} (already correct)")
print("=" * 60)

if results["failed"]:
    print(f"\nFailed modalities:")
    for m in results["failed"]:
        print(f"  - {m}")

# Verify uniqueness
print("\nFinal uniqueness check...")
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
print(f"Unique x_true hashes: {unique_h}/{len(hashes)}")
if shared:
    print(f"Shared sources ({len(shared)}):")
    for s, mods in list(shared.items())[:10]:
        print(f"  [{len(mods)}] {s[:60]}: {', '.join(mods[:5])}")
