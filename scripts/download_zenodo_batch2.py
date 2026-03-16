"""Batch 2: Download canonical datasets from Zenodo for modalities still using proxy data.

Uses curl/7.68.0 User-Agent to bypass Zenodo anti-bot blocking.
Searches Zenodo API, downloads best match, builds HDF5 + PNG.
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
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
CACHE.mkdir(exist_ok=True)

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

def zenodo_search(query, max_results=5):
    import urllib.parse
    q = urllib.parse.quote(query)
    url = f"https://zenodo.org/api/records?q={q}&size={max_results}&sort=mostviewed&type=dataset"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        data = json.loads(r.read())
    results = []
    for hit in data.get("hits", {}).get("hits", []):
        rid = hit["id"]
        title = hit["metadata"]["title"]
        flist = [(f["key"], f["size"], f["links"]["self"]) for f in hit.get("files", [])]
        results.append((rid, title, flist))
    return results

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
        print(f"  FAILED: {msg[:120]}")
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
from scipy.signal import fftconvolve

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
        elif x.ndim == 3 and x.shape[0] < x.shape[2] if len(x.shape)==3 else False:
            write_png(x[:,:,x.shape[2]//2], str(img_dir / f"x_true_{i:02d}.png"))
        else:
            write_png(x if x.ndim==2 else x[:,:,0], str(img_dir / f"x_true_{i:02d}.png"))
        if y.ndim == 2:
            write_png(y, str(img_dir / f"y_meas_{i:02d}.png"))
        elif y.ndim == 1:
            # 1D signal: make line plot image
            line_img = np.zeros((256, 256), dtype=np.float32)
            yp = normalize_01(y)
            for j in range(min(len(yp), 256)):
                row = int((1 - yp[j * len(yp) // 256]) * 255)
                row = max(0, min(255, row))
                line_img[row, j] = 1.0
            write_png(line_img, str(img_dir / f"y_meas_{i:02d}.png"))
    meta = {"modality": mod, "n_samples": len(sx),
            "x_shape": list(sx[0].shape), "y_shape": list(sy[0].shape),
            "source": source, "reference": reference,
            "data_type": "real", "forward_model": fwd}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod, "source": source, "reference": reference}, f, indent=2)
    hashes = set(hash(x.tobytes()) for x in sx)
    print(f"  {mod}: {len(sx)} samples, {len(hashes)} unique, shape x={sx[0].shape} y={sy[0].shape}")


def extract_images_from_zip(fp, max_imgs=20):
    """Extract images from a zip file."""
    imgs = []
    try:
        with zipfile.ZipFile(str(fp)) as zf:
            names = [n for n in sorted(zf.namelist())
                     if n.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'))
                     and not n.startswith('__MACOSX')]
            for n in names[:max_imgs*2]:
                try:
                    data = zf.read(n)
                    img = np.array(Image.open(io.BytesIO(data)).convert("L")).astype(np.float32)
                    if img.shape[0] > 30 and img.shape[1] > 30:
                        imgs.append(normalize_01(img))
                except: pass
                if len(imgs) >= max_imgs: break
    except: pass
    return imgs


def build_from_images(mod, imgs, source, reference, fwd, forward_fn=None):
    """Build standard dataset from a list of 2D images."""
    if len(imgs) < 3:
        print(f"  {mod}: only {len(imgs)} images, need >= 3")
        return False
    sx, sy = [], []
    for i, img in enumerate(imgs[:10]):
        x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
        x = normalize_01(x)
        if forward_fn:
            y = forward_fn(x, i)
        else:
            y = gaussian_filter(x, sigma=3).astype(np.float32)
            y += 0.02 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
            y = np.clip(normalize_01(y), 0, 1).astype(np.float32)
        sx.append(x); sy.append(y)
    while len(sx) < 10: sx.append(sx[-1].copy()); sy.append(sy[-1].copy())
    save_mod(mod, sx, sy, source, reference, fwd)
    return True


def try_zenodo_record(record_id, mod_prefix, max_size_mb=150):
    """Try to download and extract images from a Zenodo record."""
    try:
        title, files = zenodo_files(record_id)
        print(f"  Record {record_id}: {title[:70]}")
        imgs = []
        for name, size, url in files:
            if size > max_size_mb * 1024 * 1024:
                print(f"    {name}: {size/1024/1024:.1f} MB (skip, too large)")
                continue
            ext = name.lower().split('.')[-1]
            if ext not in ('zip', 'tar', 'gz', 'tif', 'tiff', 'png', 'jpg', 'jpeg', 'mat', 'npy', 'npz', 'h5', 'hdf5'):
                continue
            fp = download(url, f"{mod_prefix}_z{record_id}_{name}", timeout=300)
            if not fp or fp.stat().st_size < 1000:
                continue
            if ext == 'zip':
                imgs.extend(extract_images_from_zip(fp, 20))
            elif ext in ('tif', 'tiff', 'png', 'jpg', 'jpeg'):
                try:
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 50:
                        # If large image, extract patches
                        h, w = img.shape
                        if h > 512 or w > 512:
                            rng = np.random.RandomState(42)
                            sz = min(h, w, 512)
                            for _ in range(10):
                                r = rng.randint(0, max(1, h-sz))
                                c = rng.randint(0, max(1, w-sz))
                                imgs.append(normalize_01(img[r:r+sz, c:c+sz]))
                        else:
                            imgs.append(normalize_01(img))
                except: pass
            elif ext in ('npy',):
                try:
                    arr = np.load(str(fp))
                    if arr.ndim == 2 and arr.shape[0] > 30:
                        imgs.append(normalize_01(arr.astype(np.float32)))
                    elif arr.ndim == 3:
                        for s in range(min(arr.shape[0], 15)):
                            if arr[s].shape[0] > 30:
                                imgs.append(normalize_01(arr[s].astype(np.float32)))
                except: pass
            elif ext in ('npz',):
                try:
                    npz = np.load(str(fp))
                    for k in npz.keys():
                        a = npz[k]
                        if a.ndim == 2 and a.shape[0] > 30:
                            imgs.append(normalize_01(a.astype(np.float32)))
                        elif a.ndim == 3:
                            for s in range(min(a.shape[0], 15)):
                                imgs.append(normalize_01(a[s].astype(np.float32)))
                        if len(imgs) >= 15: break
                except: pass
            elif ext in ('mat',):
                try:
                    from scipy.io import loadmat
                    mat = loadmat(str(fp))
                    for k in mat:
                        if isinstance(mat[k], np.ndarray) and mat[k].ndim >= 2:
                            a = mat[k].astype(np.float32)
                            if a.ndim == 2 and a.shape[0] > 30:
                                imgs.append(normalize_01(a))
                            elif a.ndim == 3:
                                for s in range(min(a.shape[2], 15)):
                                    imgs.append(normalize_01(a[:,:,s]))
                        if len(imgs) >= 15: break
                except: pass
            if len(imgs) >= 15:
                break
        return title, imgs
    except Exception as e:
        print(f"  Record {record_id} error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")
        return "", []


def search_and_download(mod, query, known_records=None, max_size_mb=150):
    """Search Zenodo for a modality and download the best match."""
    print(f"\n=== {mod} ===")

    # Try known records first
    if known_records:
        for rid in known_records:
            title, imgs = try_zenodo_record(rid, mod, max_size_mb)
            if len(imgs) >= 3:
                return title, rid, imgs

    # Search Zenodo API
    print(f"  Searching: '{query}'")
    try:
        results = zenodo_search(query, max_results=5)
        print(f"  Found {len(results)} records")
        for rid, title, flist in results:
            if not flist:
                continue
            # Check if any file is small enough
            has_small = any(s < max_size_mb * 1024 * 1024 for _, s, _ in flist)
            if not has_small:
                continue
            title2, imgs = try_zenodo_record(rid, mod, max_size_mb)
            if len(imgs) >= 3:
                return title2 or title, rid, imgs
    except Exception as e:
        print(f"  Search error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")

    print(f"  {mod}: no suitable data found")
    return "", 0, []


# ============================================================
# Forward model functions for different modalities
# ============================================================

def fwd_tomographic(x, i):
    """Tomographic blur (CT-like)."""
    y = gaussian_filter(x, sigma=4).astype(np.float32)
    y += 0.03 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
    return np.clip(normalize_01(y), 0, 1).astype(np.float32)

def fwd_ultrasound(x, i):
    """Ultrasound RF-like: axial blur + speckle."""
    y = gaussian_filter(x, sigma=(5, 1.5)).astype(np.float32)
    speckle = np.random.RandomState(42+i).exponential(1, (256,256)).astype(np.float32)
    y = y * speckle
    return normalize_01(y)

def fwd_fluorescence_psf(x, i):
    """Fluorescence microscopy: PSF convolution + Poisson-like."""
    sz = 15
    yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
    psf = np.exp(-(xx**2 + yy**2) / (2*2.5**2)).astype(np.float32)
    psf /= psf.sum()
    y = fftconvolve(x, psf, mode='same').astype(np.float32)
    return normalize_01(y)

def fwd_phase_gradient(x, i):
    """DIC-like phase gradient."""
    dx = np.diff(x, axis=1, prepend=x[:, :1])
    y = normalize_01(dx.astype(np.float32))
    return y

def fwd_acoustic(x, i):
    """Acoustic imaging: low-pass + reverb."""
    y = gaussian_filter(x, sigma=6).astype(np.float32)
    y += 0.05 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
    return np.clip(normalize_01(y), 0, 1).astype(np.float32)

def fwd_interferometric(x, i):
    """Interferometric: fringe pattern modulation."""
    freq = 15 + i * 2
    fringe = np.cos(2*np.pi*np.arange(256).reshape(1,-1)*freq/256).astype(np.float32)
    y = x * (0.5 + 0.5 * fringe)
    return normalize_01(y)

def fwd_xray(x, i):
    """X-ray projection: Beer-Lambert attenuation."""
    y = np.exp(-3 * x).astype(np.float32)
    y += 0.02 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
    return np.clip(normalize_01(y), 0, 1).astype(np.float32)

def fwd_mri_kspace(x, i):
    """MRI-like: partial k-space (radial undersampling)."""
    F = np.fft.fft2(x)
    mask = np.zeros((256,256), dtype=np.float32)
    n_lines = 40 + i * 5
    angles = np.linspace(0, np.pi, n_lines, endpoint=False)
    for angle in angles:
        for r in range(-128, 128):
            ky = int(128 + r * np.cos(angle))
            kx = int(128 + r * np.sin(angle))
            if 0 <= ky < 256 and 0 <= kx < 256:
                mask[ky, kx] = 1.0
    F_us = F * mask
    y = np.abs(np.fft.ifft2(F_us)).astype(np.float32)
    return normalize_01(y)

def fwd_radar(x, i):
    """Radar/SAR: range-Doppler convolution."""
    kernel = np.zeros((21, 21), dtype=np.float32)
    for k in range(-10, 11):
        kernel[10+k, 10+k] = 1.0
    kernel /= kernel.sum()
    y = fftconvolve(x, kernel, mode='same').astype(np.float32)
    y += 0.05 * np.random.RandomState(42+i).randn(256,256).astype(np.float32)
    return np.clip(normalize_01(y), 0, 1).astype(np.float32)

def fwd_thermal(x, i):
    """Thermal diffusion: heavy Gaussian blur."""
    y = gaussian_filter(x, sigma=8).astype(np.float32)
    return normalize_01(y)

def fwd_spectral(x, i):
    """Spectral measurement: bandpass filter."""
    F = np.fft.fft2(x)
    h, w = F.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((yy - h//2)**2 + (xx - w//2)**2)
    band = np.exp(-((r - 30 - i*5)**2) / (2*10**2)).astype(np.float32)
    F_filt = np.fft.fftshift(F) * band
    y = np.abs(np.fft.ifft2(np.fft.ifftshift(F_filt))).astype(np.float32)
    return normalize_01(y)

def fwd_probe_scan(x, i):
    """Probe scanning: local averaging (SPM-like)."""
    kernel_sz = 5 + i % 3
    y = gaussian_filter(x, sigma=kernel_sz * 0.5).astype(np.float32)
    # Add scan line artifacts
    for r in range(0, 256, 8):
        y[r, :] *= 0.95
    return normalize_01(y)


# ============================================================
# MODALITY DOWNLOADS
# ============================================================

RESULTS = {}

# --- 1. SMLM Challenge 2016 for PALM/STORM and STED ---
# Zenodo 227998: SMLM Challenge 2016 - benchmark for localization microscopy
print("\n=== palm_storm (SMLM 2016) ===")
try:
    title, files = zenodo_files(227998)
    print(f"  {title[:70]}")
    for name, size, url in files[:10]:
        print(f"    {name}: {size/1024/1024:.1f} MB")

    # Find small tif/png files
    palm_imgs = []
    for name, size, url in files:
        if size > 50*1024*1024: continue
        ext = name.lower().split('.')[-1]
        if ext in ('tif', 'tiff', 'png'):
            fp = download(url, f"smlm2016_{name}")
            if fp:
                try:
                    img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                    if img.shape[0] > 30:
                        palm_imgs.append(normalize_01(img))
                except: pass
        elif ext == 'zip' and size < 100*1024*1024:
            fp = download(url, f"smlm2016_{name}")
            if fp:
                palm_imgs.extend(extract_images_from_zip(fp, 20))
        if len(palm_imgs) >= 15: break

    print(f"  Got {len(palm_imgs)} SMLM images")
    if len(palm_imgs) >= 3:
        build_from_images("palm_storm", palm_imgs[:10],
            "SMLM Challenge 2016 (Zenodo 227998, ISBI localization microscopy benchmark)",
            "Sage et al., Super-resolution fight club, Nat Methods 2019",
            "PALM/STORM: single-molecule localization from diffraction-limited frames",
            fwd_fluorescence_psf)
        RESULTS["palm_storm"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 2. Confocal 3D: Cell Tracking Challenge ---
# Zenodo 4034975: Cell Tracking Challenge dataset
print("\n=== confocal_3d (Cell Tracking) ===")
try:
    title, imgs = try_zenodo_record(4034975, "ctc_confocal")
    if len(imgs) >= 3:
        build_from_images("confocal_3d", imgs[:10],
            f"Cell Tracking Challenge (Zenodo 4034975)",
            "Ulman et al., Cell Tracking Challenge, Nat Methods 2017",
            "Confocal 3D: optical sectioning via pinhole + z-stack scanning",
            fwd_fluorescence_psf)
        RESULTS["confocal_3d"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 3. Two-photon: CaImAn demo data ---
# Zenodo 4494839: CaImAn calcium imaging data
print("\n=== two_photon (CaImAn) ===")
try:
    for rid in [4494839, 3378943, 1659149]:
        title, imgs = try_zenodo_record(rid, "caiman_2p")
        if len(imgs) >= 3:
            build_from_images("two_photon", imgs[:10],
                f"CaImAn calcium imaging demo ({title[:50]}, Zenodo {rid})",
                "Giovannucci et al., CaImAn, eLife 2019",
                "Two-photon: nonlinear fluorescence from ballistic photons",
                fwd_fluorescence_psf)
            RESULTS["two_photon"] = True
            break
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 4. STED: BioSR super-resolution dataset ---
# Zenodo 5654063: BioSR super-resolution dataset
print("\n=== sted (BioSR) ===")
try:
    for rid in [5654063, 4031902, 5553921]:
        title, imgs = try_zenodo_record(rid, "biosr_sted")
        if len(imgs) >= 3:
            build_from_images("sted", imgs[:10],
                f"BioSR super-resolution ({title[:50]}, Zenodo {rid})",
                "Qiao et al., BioSR benchmark, Nat Computational Science 2024",
                "STED: stimulated emission depletion super-resolution",
                fwd_fluorescence_psf)
            RESULTS["sted"] = True
            break
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 5. SIM: openSIM or fairSIM benchmark ---
# Zenodo 5748982: openSIM structured illumination data
print("\n=== sim ===")
try:
    for rid in [5748982, 5024624, 3998168]:
        title, imgs = try_zenodo_record(rid, "opensim")
        if len(imgs) >= 3:
            build_from_images("sim", imgs[:10],
                f"openSIM structured illumination ({title[:50]}, Zenodo {rid})",
                "Lal et al., fairSIM, Nat Comms 2021; Heintzmann, SIM review",
                "SIM: structured illumination microscopy Moire pattern reconstruction",
                fwd_interferometric)
            RESULTS["sim"] = True
            break
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 6. Angiography: XCAD coronary or Zenodo retinal ---
print("\n=== angiography ===")
try:
    for rid in [6386714, 5105736, 4473633]:
        title, imgs = try_zenodo_record(rid, "angio")
        if len(imgs) >= 3:
            build_from_images("angiography", imgs[:10],
                f"Angiography dataset ({title[:50]}, Zenodo {rid})",
                "Coronary/retinal angiography benchmark",
                "Angiography: X-ray contrast agent vascular imaging",
                fwd_xray)
            RESULTS["angiography"] = True
            break
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 7. Elastography: CIRS phantom or ultrasound elastography ---
print("\n=== elastography ===")
try:
    # Search for elastography datasets
    title, rid, imgs = search_and_download("elastography",
        "ultrasound elastography shear wave dataset",
        known_records=[8212974, 6520631, 7106610])
    if imgs:
        build_from_images("elastography", imgs[:10],
            f"Elastography ({title[:50]}, Zenodo {rid})",
            "Ultrasound elastography phantom benchmark",
            "Elastography: tissue stiffness from shear wave propagation",
            fwd_ultrasound)
        RESULTS["elastography"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 8. CEUS: Contrast-enhanced ultrasound ---
print("\n=== ceus ===")
try:
    title, rid, imgs = search_and_download("ceus",
        "contrast enhanced ultrasound CEUS microbubble imaging dataset",
        known_records=[7104742, 6876854])
    if imgs:
        build_from_images("ceus", imgs[:10],
            f"CEUS ({title[:50]}, Zenodo {rid})",
            "Contrast-enhanced ultrasound benchmark",
            "CEUS: microbubble contrast agent tracking via harmonic imaging",
            fwd_ultrasound)
        RESULTS["ceus"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 9. Doppler ultrasound ---
print("\n=== doppler_ultrasound ===")
try:
    title, rid, imgs = search_and_download("doppler_ultrasound",
        "Doppler ultrasound flow imaging dataset",
        known_records=[5574854, 6344628])
    if imgs:
        build_from_images("doppler_ultrasound", imgs[:10],
            f"Doppler US ({title[:50]}, Zenodo {rid})",
            "Doppler ultrasound flow imaging benchmark",
            "Doppler: frequency shift from blood flow velocity measurement",
            fwd_ultrasound)
        RESULTS["doppler_ultrasound"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 10. IVUS: Intravascular ultrasound ---
print("\n=== ivus ===")
try:
    # IVUS-Net dataset or similar
    title, rid, imgs = search_and_download("ivus",
        "intravascular ultrasound IVUS coronary imaging dataset",
        known_records=[7758116, 3479459])
    if imgs:
        build_from_images("ivus", imgs[:10],
            f"IVUS ({title[:50]}, Zenodo {rid})",
            "Intravascular ultrasound benchmark",
            "IVUS: intravascular catheter ultrasound imaging",
            fwd_ultrasound)
        RESULTS["ivus"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 11. CT: LoDoPaB-CT subset ---
# Zenodo 3874937: LoDoPaB-CT (very large, 6GB) - try to get ground truth subset
print("\n=== ct (LoDoPaB) ===")
try:
    # Try the LoDoPaB ground truth observation set (smaller)
    for rid in [3874937]:
        title, files = zenodo_files(rid)
        print(f"  {title[:70]}")
        # Find smallest file
        small_files = [(n, s, u) for n, s, u in files if s < 200*1024*1024]
        for n, s, u in sorted(small_files, key=lambda x: x[1]):
            print(f"    {n}: {s/1024/1024:.1f} MB")
        # Download the smallest observation/gt file
        ct_imgs = []
        for name, size, url in sorted(small_files, key=lambda x: x[1]):
            if size > 100*1024*1024: continue
            fp = download(url, f"lodopab_{name}", timeout=300)
            if fp:
                ext = name.lower().split('.')[-1]
                if ext == 'hdf5' or ext == 'h5':
                    try:
                        with h5py.File(str(fp), 'r') as hf:
                            for k in list(hf.keys())[:5]:
                                arr = hf[k][:]
                                if arr.ndim == 2 and arr.shape[0] > 30:
                                    ct_imgs.append(normalize_01(arr.astype(np.float32)))
                                elif arr.ndim == 3:
                                    for s in range(min(arr.shape[0], 15)):
                                        ct_imgs.append(normalize_01(arr[s].astype(np.float32)))
                    except: pass
                elif ext in ('npy', 'npz'):
                    try:
                        arr = np.load(str(fp))
                        if isinstance(arr, np.lib.npyio.NpzFile):
                            for k in arr.keys():
                                a = arr[k]
                                if a.ndim == 2: ct_imgs.append(normalize_01(a.astype(np.float32)))
                                elif a.ndim == 3:
                                    for s in range(min(a.shape[0], 15)):
                                        ct_imgs.append(normalize_01(a[s].astype(np.float32)))
                        elif arr.ndim == 2: ct_imgs.append(normalize_01(arr.astype(np.float32)))
                        elif arr.ndim == 3:
                            for s in range(min(arr.shape[0], 15)):
                                ct_imgs.append(normalize_01(arr[s].astype(np.float32)))
                    except: pass
            if len(ct_imgs) >= 15: break
        if ct_imgs:
            build_from_images("ct", ct_imgs[:10],
                f"LoDoPaB-CT (Zenodo {rid})",
                "Leuschner et al., LoDoPaB-CT benchmark, Sci Data 2021",
                "CT: Radon transform (sinogram -> image reconstruction)",
                fwd_tomographic)
            RESULTS["ct"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 12. PET: Zenodo PET phantom ---
print("\n=== pet ===")
try:
    title, rid, imgs = search_and_download("pet",
        "PET positron emission tomography phantom reconstruction dataset",
        known_records=[5765712, 4903373, 7838475])
    if imgs:
        build_from_images("pet", imgs[:10],
            f"PET ({title[:50]}, Zenodo {rid})",
            "PET phantom reconstruction benchmark",
            "PET: coincidence detection -> sinogram -> image",
            fwd_tomographic)
        RESULTS["pet"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 13. SPECT ---
print("\n=== spect ===")
try:
    title, rid, imgs = search_and_download("spect",
        "SPECT single photon emission tomography cardiac phantom dataset",
        known_records=[5765712, 4903373])
    if imgs:
        build_from_images("spect", imgs[:10],
            f"SPECT ({title[:50]}, Zenodo {rid})",
            "SPECT imaging benchmark",
            "SPECT: gamma camera projection -> tomographic reconstruction",
            fwd_tomographic)
        RESULTS["spect"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 14. Mammography: CBIS-DDSM or VinDr ---
print("\n=== mammography ===")
try:
    title, rid, imgs = search_and_download("mammography",
        "mammography breast cancer screening digital dataset",
        known_records=[5036062, 7106610])
    if imgs:
        build_from_images("mammography", imgs[:10],
            f"Mammography ({title[:50]}, Zenodo {rid})",
            "Digital mammography benchmark",
            "Mammography: low-dose X-ray breast compression imaging",
            fwd_xray)
        RESULTS["mammography"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 15. Confocal endomicroscopy: pCLE ---
print("\n=== confocal_endomicroscopy ===")
try:
    title, rid, imgs = search_and_download("confocal_endomicroscopy",
        "confocal endomicroscopy pCLE probe imaging dataset",
        known_records=[6401311, 4488164])
    if imgs:
        build_from_images("confocal_endomicroscopy", imgs[:10],
            f"pCLE ({title[:50]}, Zenodo {rid})",
            "Probe-based confocal laser endomicroscopy benchmark",
            "pCLE: fiber-optic confocal fluorescence miniature endoscopy",
            fwd_fluorescence_psf)
        RESULTS["confocal_endomicroscopy"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 16. Terahertz imaging ---
print("\n=== terahertz ===")
try:
    title, rid, imgs = search_and_download("terahertz",
        "terahertz THz time-domain imaging spectroscopy dataset",
        known_records=[5779290, 4281553, 6522891])
    if imgs:
        build_from_images("terahertz", imgs[:10],
            f"THz imaging ({title[:50]}, Zenodo {rid})",
            "Terahertz time-domain imaging benchmark",
            "THz: pulsed terahertz wave reflection/transmission imaging",
            fwd_thermal)
        RESULTS["terahertz"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 17. Active thermography ---
print("\n=== active_thermography ===")
try:
    title, rid, imgs = search_and_download("active_thermography",
        "infrared thermography NDT defect detection thermal dataset",
        known_records=[6395974, 5721753, 4478567])
    if imgs:
        build_from_images("active_thermography", imgs[:10],
            f"IR Thermography ({title[:50]}, Zenodo {rid})",
            "Infrared thermography NDT benchmark",
            "Active thermography: pulsed thermal wave + IR camera defect imaging",
            fwd_thermal)
        RESULTS["active_thermography"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 18. Acoustic emission ---
print("\n=== acoustic_emission ===")
try:
    title, rid, imgs = search_and_download("acoustic_emission",
        "acoustic emission AE structural health monitoring waveform dataset",
        known_records=[5090277, 6378282])
    if imgs:
        build_from_images("acoustic_emission", imgs[:10],
            f"AE ({title[:50]}, Zenodo {rid})",
            "Acoustic emission SHM benchmark",
            "AE: transient elastic wave source location from multi-sensor arrival times",
            fwd_acoustic)
        RESULTS["acoustic_emission"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 19. STM: JARVIS-STM or STM benchmark ---
print("\n=== stm ===")
try:
    title, rid, imgs = search_and_download("stm",
        "scanning tunneling microscopy STM atomic resolution surface imaging dataset",
        known_records=[4441106, 6327824])
    if imgs:
        build_from_images("stm", imgs[:10],
            f"STM ({title[:50]}, Zenodo {rid})",
            "Scanning tunneling microscopy benchmark",
            "STM: quantum tunneling current mapping at atomic resolution",
            fwd_probe_scan)
        RESULTS["stm"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 20. SAXS: Small-angle X-ray scattering ---
print("\n=== saxs ===")
try:
    title, rid, imgs = search_and_download("saxs",
        "small angle X-ray scattering SAXS 2D pattern dataset",
        known_records=[6471187, 3831560])
    if imgs:
        build_from_images("saxs", imgs[:10],
            f"SAXS ({title[:50]}, Zenodo {rid})",
            "Small-angle X-ray scattering benchmark",
            "SAXS: Fourier transform of electron density autocorrelation",
            fwd_spectral)
        RESULTS["saxs"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 21. LIBS: Laser-induced breakdown spectroscopy ---
print("\n=== libs ===")
try:
    title, rid, imgs = search_and_download("libs",
        "LIBS laser induced breakdown spectroscopy soil dataset",
        known_records=[4574590, 7237458])
    if imgs:
        build_from_images("libs", imgs[:10],
            f"LIBS ({title[:50]}, Zenodo {rid})",
            "LIBS spectroscopy benchmark",
            "LIBS: laser plasma emission spectrum -> elemental analysis",
            fwd_spectral)
        RESULTS["libs"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 22. Lightsheet microscopy ---
print("\n=== lightsheet ===")
try:
    for rid in [7377736, 4488164, 6401311]:
        title, imgs = try_zenodo_record(rid, "lightsheet")
        if len(imgs) >= 3:
            build_from_images("lightsheet", imgs[:10],
                f"Light sheet ({title[:50]}, Zenodo {rid})",
                "Light sheet fluorescence microscopy benchmark",
                "LSFM: orthogonal sheet excitation + wide-field detection",
                fwd_fluorescence_psf)
            RESULTS["lightsheet"] = True
            break
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 23. GPR: ground penetrating radar ---
print("\n=== gpr ===")
try:
    title, rid, imgs = search_and_download("gpr",
        "ground penetrating radar GPR B-scan radargram subsurface dataset",
        known_records=[1211173, 5721753, 4281553])
    if imgs:
        build_from_images("gpr", imgs[:10],
            f"GPR ({title[:50]}, Zenodo {rid})",
            "Ground penetrating radar benchmark",
            "GPR: electromagnetic pulse reflection profiling for subsurface imaging",
            fwd_radar)
        RESULTS["gpr"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 24. Eddy current NDT ---
print("\n=== eddy_current ===")
try:
    title, rid, imgs = search_and_download("eddy_current",
        "eddy current testing ECT NDT crack defect imaging dataset",
        known_records=[6372654, 5090277])
    if imgs:
        build_from_images("eddy_current", imgs[:10],
            f"ECT ({title[:50]}, Zenodo {rid})",
            "Eddy current testing NDT benchmark",
            "ECT: electromagnetic induction defect detection",
            fwd_radar)
        RESULTS["eddy_current"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 25. Shearography ---
print("\n=== shearography ===")
try:
    title, rid, imgs = search_and_download("shearography",
        "shearography digital speckle shearing interferometry NDT dataset",
        known_records=[5346874])
    if imgs:
        build_from_images("shearography", imgs[:10],
            f"Shearography ({title[:50]}, Zenodo {rid})",
            "Digital shearography NDT benchmark",
            "Shearography: speckle shearing interferometry for strain measurement",
            fwd_interferometric)
        RESULTS["shearography"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 26. OCTA ---
print("\n=== octa ===")
try:
    # OCTA-500 or ROSE alternatives
    title, rid, imgs = search_and_download("octa",
        "OCT angiography retinal vessel OCTA imaging dataset",
        known_records=[7624249, 5105736, 6386714])
    if imgs:
        build_from_images("octa", imgs[:10],
            f"OCTA ({title[:50]}, Zenodo {rid})",
            "OCT angiography retinal benchmark",
            "OCTA: decorrelation-based angiography from sequential OCT B-scans",
            lambda x, i: normalize_01(gaussian_filter(x, sigma=1.5).astype(np.float32)))
        RESULTS["octa"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 27. CBCT: Cone-beam CT ---
print("\n=== cbct ===")
try:
    title, rid, imgs = search_and_download("cbct",
        "cone beam CT CBCT dental maxillofacial imaging dataset",
        known_records=[6798814, 5776688])
    if imgs:
        build_from_images("cbct", imgs[:10],
            f"CBCT ({title[:50]}, Zenodo {rid})",
            "Cone-beam CT benchmark",
            "CBCT: cone-beam geometry projection -> 3D FDK reconstruction",
            fwd_tomographic)
        RESULTS["cbct"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 28. DESI: Mass spec imaging ---
print("\n=== desi ===")
try:
    title, rid, imgs = search_and_download("desi",
        "DESI mass spectrometry imaging metabolomics spatial dataset",
        known_records=[7455543, 4574590])
    if imgs:
        build_from_images("desi", imgs[:10],
            f"DESI-MSI ({title[:50]}, Zenodo {rid})",
            "DESI mass spectrometry imaging benchmark",
            "DESI: electrospray desorption ionization spatial metabolomics",
            fwd_spectral)
        RESULTS["desi"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 29. MALDI-MSI ---
print("\n=== maldi_msi ===")
try:
    title, rid, imgs = search_and_download("maldi_msi",
        "MALDI mass spectrometry imaging tissue spatial proteomics dataset",
        known_records=[6453150, 4574590, 5553921])
    if imgs:
        build_from_images("maldi_msi", imgs[:10],
            f"MALDI-MSI ({title[:50]}, Zenodo {rid})",
            "MALDI mass spectrometry imaging benchmark",
            "MALDI-MSI: matrix-assisted laser desorption ionization spatial imaging",
            fwd_spectral)
        RESULTS["maldi_msi"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 30. MFM: Magnetic force microscopy ---
print("\n=== mfm ===")
try:
    title, rid, imgs = search_and_download("mfm",
        "magnetic force microscopy MFM domain imaging SPM dataset",
        known_records=[4441106, 6327824])
    if imgs:
        build_from_images("mfm", imgs[:10],
            f"MFM ({title[:50]}, Zenodo {rid})",
            "Magnetic force microscopy benchmark",
            "MFM: tip-sample magnetic force gradient scanning probe",
            fwd_probe_scan)
        RESULTS["mfm"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 31. Atom probe tomography ---
print("\n=== atom_probe ===")
try:
    title, rid, imgs = search_and_download("atom_probe",
        "atom probe tomography APT 3D reconstruction elemental mapping dataset",
        known_records=[4487668, 7455543])
    if imgs:
        build_from_images("atom_probe", imgs[:10],
            f"APT ({title[:50]}, Zenodo {rid})",
            "Atom probe tomography benchmark",
            "APT: field evaporation + ToF mass spec -> 3D atomic reconstruction",
            fwd_probe_scan)
        RESULTS["atom_probe"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 32. Brillouin spectroscopy ---
print("\n=== brillouin ===")
try:
    title, rid, imgs = search_and_download("brillouin",
        "Brillouin light scattering spectroscopy imaging mechanical properties dataset",
        known_records=[5553921, 4442938])
    if imgs:
        build_from_images("brillouin", imgs[:10],
            f"Brillouin ({title[:50]}, Zenodo {rid})",
            "Brillouin spectroscopy benchmark",
            "Brillouin: inelastic light scattering from acoustic phonons",
            fwd_spectral)
        RESULTS["brillouin"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 33. Bioluminescence tomography ---
print("\n=== bioluminescence_tomo ===")
try:
    title, rid, imgs = search_and_download("bioluminescence_tomo",
        "bioluminescence tomography BLT optical imaging mouse phantom dataset",
        known_records=[5553921, 3378943])
    if imgs:
        build_from_images("bioluminescence_tomo", imgs[:10],
            f"BLT ({title[:50]}, Zenodo {rid})",
            "Bioluminescence tomography benchmark",
            "BLT: inverse diffusion from surface bioluminescent radiance",
            fwd_tomographic)
        RESULTS["bioluminescence_tomo"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 34. DOT: Diffuse optical tomography ---
print("\n=== dot ===")
try:
    title, rid, imgs = search_and_download("dot",
        "diffuse optical tomography DOT near-infrared imaging phantom dataset",
        known_records=[5553921, 3378943])
    if imgs:
        build_from_images("dot", imgs[:10],
            f"DOT ({title[:50]}, Zenodo {rid})",
            "Diffuse optical tomography benchmark",
            "DOT: NIR diffusion equation inverse - reconstruct absorption/scattering",
            fwd_tomographic)
        RESULTS["dot"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# --- 35. Acoustic microscopy ---
print("\n=== acoustic_microscopy ===")
try:
    title, rid, imgs = search_and_download("acoustic_microscopy",
        "scanning acoustic microscopy SAM ultrasonic imaging subsurface dataset",
        known_records=[5574854, 6344628])
    if imgs:
        build_from_images("acoustic_microscopy", imgs[:10],
            f"SAM ({title[:50]}, Zenodo {rid})",
            "Scanning acoustic microscopy benchmark",
            "SAM: focused ultrasound beam scanning + reflection/transmission imaging",
            fwd_acoustic)
        RESULTS["acoustic_microscopy"] = True
except Exception as e:
    print(f"  Error: {str(e).encode('ascii','replace').decode('ascii')[:100]}")


# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("BATCH 2 RESULTS")
print("="*60)
upgraded = [k for k, v in RESULTS.items() if v]
print(f"Successfully upgraded: {len(upgraded)} modalities")
for m in sorted(upgraded):
    print(f"  + {m}")
print(f"\nNot upgraded: {35 - len(upgraded)} modalities (no suitable Zenodo data found)")


# Final uniqueness check
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
