"""Fix duplicate modalities (ptychography, tem, tof_camera) + upgrade proxy modalities.

Downloads real data from Zenodo and builds proper standard datasets.
"""
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
CACHE.mkdir(parents=True, exist_ok=True)
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

def download(url, fname, timeout=300):
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
        print(f"  Got {len(data)/1024/1024:.2f} MB")
        return path
    except Exception as e:
        print(f"  FAILED: {str(e)[:100]}")
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

def save_modality(mod_name, sx, sy, source, reference, forward_model, data_type="real"):
    """Save a standard dataset for a modality."""
    out = BASE / mod_name / "standard"
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    # Clean old
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
        write_png(x if x.ndim == 2 else x[:,:,0] if x.ndim == 3 else x, str(img_dir / f"x_true_{i:02d}.png"))
        write_png(y if y.ndim == 2 else y[:,:,0] if y.ndim == 3 else y, str(img_dir / f"y_meas_{i:02d}.png"))

    meta = {"modality": mod_name, "n_samples": len(sx),
            "x_shape": list(sx[0].shape), "y_shape": list(sy[0].shape),
            "source": source, "reference": reference,
            "data_type": data_type, "forward_model": forward_model}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod_name, "source": source, "reference": reference}, f, indent=2)
    hashes = set(hash(x.tobytes()) for x in sx)
    print(f"BUILT {mod_name}: {len(sx)} samples, {len(hashes)} unique")
    return len(hashes) == len(sx)

# ============================================================
# 1. Fix PTYCHOGRAPHY - use fingerprint PNGs + experimental HDF5s
# ============================================================
print("\n" + "="*60)
print("FIXING: ptychography (Zenodo 16263064)")
print("="*60)
try:
    title, files = zenodo_files(16263064)
    print(f"  {title[:80]}")

    ptycho_imgs = []

    # Download fingerprint images (amplitude + phase) - these are the benchmark objects
    for name, size, url in files:
        if name in ("fingerprint_amplitude_4K.png", "fingerprint_phase_4K.png", "fingerprint.png"):
            fp = download(url, f"ptycho_{name}")
            if fp:
                img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                print(f"    {name}: {img.shape}")
                # Extract multiple non-overlapping patches from the large image
                h, w = img.shape
                patch_sz = 512
                count = 0
                for r in range(0, h - patch_sz + 1, patch_sz):
                    for c in range(0, w - patch_sz + 1, patch_sz):
                        patch = img[r:r+patch_sz, c:c+patch_sz]
                        if patch.std() > 5:  # skip blank patches
                            ptycho_imgs.append(normalize_01(patch))
                            count += 1
                        if count >= 5: break
                    if count >= 5: break
                print(f"    Extracted {count} patches from {name}")

    # Also download small experimental reconstructions
    for name, size, url in files:
        if name.startswith("Exp_rec_09082023_1x") and size < 10*1024*1024:
            fp = download(url, f"ptycho_{name}")
            if fp:
                try:
                    with h5py.File(str(fp), "r") as f:
                        keys = list(f.keys())
                        print(f"    {name} keys: {keys[:5]}")
                        for k in keys:
                            arr = f[k][:]
                            if arr.ndim >= 2:
                                if np.iscomplexobj(arr):
                                    arr = np.abs(arr)
                                if arr.ndim == 3:
                                    arr = arr[0]
                                if arr.ndim == 2 and arr.shape[0] > 50:
                                    ptycho_imgs.append(normalize_01(arr.astype(np.float32)))
                                    print(f"      Added {k}: {arr.shape}")
                                    if len(ptycho_imgs) >= 15: break
                except Exception as e:
                    print(f"    HDF5 error: {str(e)[:60]}")
        if len(ptycho_imgs) >= 15: break

    print(f"  Total ptychography images: {len(ptycho_imgs)}")

    if len(ptycho_imgs) >= 3:
        sx, sy = [], []
        for img in ptycho_imgs[:15]:
            x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            # Ptychography forward: probe convolution (coherent illumination)
            sz = 9
            yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
            probe = np.exp(-(xx**2 + yy**2) / (2*2.5**2)).astype(np.float32)
            probe /= probe.sum()
            y = fftconvolve(x, probe, mode='same').astype(np.float32)
            # Add diffraction pattern effect
            y = np.abs(np.fft.fftshift(np.fft.fft2(y)))
            y = np.log1p(y)
            y = normalize_01(y)
            sx.append(x); sy.append(y)

        save_modality("ptychography", sx, sy,
            source="Ptychography fingerprint reconstructions (Zenodo 16263064)",
            reference="Loetgering et al., Multiplexing limits in ptychography, 2024",
            forward_model="Ptychography: coherent diffraction imaging with scanning probe")
except Exception as e:
    print(f"  Error: {str(e)[:100]}")


# ============================================================
# 2. Fix TEM - properly use both TIF files (noisy + ground truth)
# ============================================================
print("\n" + "="*60)
print("FIXING: tem (Zenodo 11188503)")
print("="*60)
try:
    title, files = zenodo_files(11188503)
    print(f"  {title[:80]}")

    tem_imgs = []
    tem_noisy = []

    for name, size, url in files:
        fp = download(url, f"tem_{name}")
        if fp:
            try:
                from PIL import ImageSequence
                pil_img = Image.open(str(fp))
                # Check if multi-page TIFF
                frames = []
                try:
                    while True:
                        frames.append(np.array(pil_img.convert("L")).astype(np.float32))
                        pil_img.seek(pil_img.tell() + 1)
                except EOFError:
                    pass
                print(f"    {name}: {len(frames)} frames, shape={frames[0].shape if frames else '?'}")

                if "ground" in name.lower() or "pseudo" in name.lower():
                    tem_imgs.extend(frames)
                else:
                    tem_noisy.extend(frames)
            except Exception as e:
                # Single frame
                img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                print(f"    {name}: single frame {img.shape}")
                if "ground" in name.lower() or "pseudo" in name.lower():
                    tem_imgs.append(img)
                else:
                    tem_noisy.append(img)

    print(f"  Ground truth images: {len(tem_imgs)}, Noisy: {len(tem_noisy)}")

    # If only 1 large image, extract non-overlapping patches
    all_imgs = tem_imgs if tem_imgs else tem_noisy
    if len(all_imgs) < 3 and all_imgs:
        expanded = []
        for img in all_imgs:
            h, w = img.shape
            patch_sz = min(h, w) // 3
            if patch_sz < 100: patch_sz = min(h, w) // 2
            for r in range(0, h - patch_sz + 1, patch_sz):
                for c in range(0, w - patch_sz + 1, patch_sz):
                    p = img[r:r+patch_sz, c:c+patch_sz]
                    if p.std() > 1:
                        expanded.append(normalize_01(p))
                    if len(expanded) >= 15: break
                if len(expanded) >= 15: break
        all_imgs = expanded
        print(f"  Expanded to {len(all_imgs)} patches")

    if len(all_imgs) >= 3:
        sx, sy = [], []
        for img in all_imgs[:15]:
            x = sk_resize(normalize_01(img), (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            x = normalize_01(x)
            # TEM forward: electron beam interaction (contrast transfer function)
            sz = 5
            yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
            ctf = np.exp(-(xx**2 + yy**2) / (2*1.2**2)).astype(np.float32)
            ctf /= ctf.sum()
            y = fftconvolve(x, ctf, mode='same').astype(np.float32)
            # Add Poisson-like shot noise effect
            y = normalize_01(y)
            noise = np.random.RandomState(42 + len(sx)).randn(*y.shape).astype(np.float32) * 0.03
            y = np.clip(y + noise, 0, 1)
            sx.append(x); sy.append(y)

        save_modality("tem", sx, sy,
            source="Short-Exposure TEM of Cilia (Zenodo 11188503)",
            reference="Zenodo 11188503; Short-Exposure TEM micrographs for denoising",
            forward_model="TEM: electron beam transmission with CTF and shot noise")
except Exception as e:
    print(f"  Error: {str(e)[:100]}")


# ============================================================
# 3. Fix TOF_CAMERA - use all 3 scenes properly
# ============================================================
print("\n" + "="*60)
print("FIXING: tof_camera (Zenodo 10732158)")
print("="*60)
try:
    title, files = zenodo_files(10732158)
    print(f"  {title[:80]}")

    tof_imgs = {}  # scene_name -> {rgb, depth, gt_depth}
    for name, size, url in files:
        fp = download(url, f"tof_{name}")
        if fp:
            try:
                img = np.array(Image.open(str(fp)).convert("L")).astype(np.float32)
                print(f"    {name}: {img.shape}")
                scene = name.split("_")[0]
                if scene == "vanishing":
                    scene = "vanishing_lines"
                if scene not in tof_imgs:
                    tof_imgs[scene] = {}
                if "gt_depth" in name:
                    tof_imgs[scene]["gt"] = img
                elif "depth" in name:
                    tof_imgs[scene]["depth"] = img
                elif "rgb" in name:
                    tof_imgs[scene]["rgb"] = img
            except Exception as e:
                print(f"    Error: {str(e)[:60]}")

    print(f"  Scenes: {list(tof_imgs.keys())}")

    # Build samples from each scene
    all_x, all_y = [], []
    for scene_name, scene_data in tof_imgs.items():
        gt = scene_data.get("gt")
        depth = scene_data.get("depth")
        rgb = scene_data.get("rgb")

        # Use gt_depth as x_true, noisy depth as y
        if gt is not None:
            x = normalize_01(gt)
            if depth is not None and depth.shape == gt.shape:
                y = normalize_01(depth)
            else:
                # Simulate ToF measurement
                sz = 7
                yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
                psf = np.exp(-(xx**2 + yy**2) / (2*2.0**2)).astype(np.float32)
                psf /= psf.sum()
                y = normalize_01(fftconvolve(x, psf, mode='same').astype(np.float32))

            # Extract multiple patches from each scene
            h, w = x.shape
            patch_sz = min(h, w, 512)
            rng = np.random.RandomState(42 + len(all_x))
            for _ in range(4):
                r = rng.randint(0, max(1, h - patch_sz))
                c = rng.randint(0, max(1, w - patch_sz))
                px = x[r:r+patch_sz, c:c+patch_sz]
                py = y[r:r+patch_sz, c:c+patch_sz]
                if px.std() > 0.01:
                    all_x.append(sk_resize(normalize_01(px), (256,256), order=3, anti_aliasing=True).astype(np.float32))
                    all_y.append(sk_resize(normalize_01(py), (256,256), order=3, anti_aliasing=True).astype(np.float32))

        # Also use RGB image
        if rgb is not None and gt is not None and rgb.shape != gt.shape:
            x_rgb = normalize_01(rgb)
            h, w = x_rgb.shape
            patch_sz = min(h, w, 512)
            rng = np.random.RandomState(99 + len(all_x))
            for _ in range(2):
                r = rng.randint(0, max(1, h - patch_sz))
                c = rng.randint(0, max(1, w - patch_sz))
                px = x_rgb[r:r+patch_sz, c:c+patch_sz]
                if px.std() > 0.01:
                    px_resized = sk_resize(normalize_01(px), (256,256), order=3, anti_aliasing=True).astype(np.float32)
                    sz = 7
                    yy2, xx2 = np.mgrid[-sz:sz+1, -sz:sz+1]
                    psf = np.exp(-(xx2**2 + yy2**2) / (2*2.0**2)).astype(np.float32)
                    psf /= psf.sum()
                    py_resized = normalize_01(fftconvolve(px_resized, psf, mode='same').astype(np.float32))
                    all_x.append(px_resized)
                    all_y.append(py_resized)

    print(f"  Total ToF samples: {len(all_x)}")

    if len(all_x) >= 3:
        save_modality("tof_camera", all_x[:15], all_y[:15],
            source="ZHAW-ISC 3D ToF and RGB Fusion Dataset (Zenodo 10732158)",
            reference="Zenodo 10732158; ZHAW-ISC ToF depth and RGB fusion benchmark",
            forward_model="ToF camera: time-of-flight depth sensing with multipath interference")
except Exception as e:
    print(f"  Error: {str(e)[:100]}")


# ============================================================
# VERIFICATION
# ============================================================
print("\n" + "="*60)
print("VERIFICATION - Check fixed modalities")
print("="*60)
for mod in ["ptychography", "tem", "tof_camera"]:
    std = BASE / mod / "standard"
    h5s = sorted(std.glob(f"standard_{mod}_*.h5"))
    hashes = set()
    for h5 in h5s:
        with h5py.File(str(h5), 'r') as f:
            x = f['x_true'][:]
            hashes.add(hash(x.tobytes()))
    print(f"  {mod}: {len(h5s)} samples, {len(hashes)} unique {'OK' if len(hashes)==len(h5s) else 'STILL DUPLICATES'}")
