"""Download canonical datasets for needs_canonical modalities - batch 1.
Focus on small/accessible datasets that can be downloaded directly.
"""
import os, sys, zipfile, io, tempfile, shutil
from pathlib import Path
import numpy as np
import h5py
import requests
from PIL import Image

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")

def normalize_01(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12: return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def write_png(arr, path):
    a = normalize_01(arr)
    a = (a * 255).clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        Image.fromarray(a, 'L').save(path)
    elif a.ndim == 3 and a.shape[2] == 3:
        Image.fromarray(a, 'RGB').save(path)
    else:
        Image.fromarray(a[:,:,0] if a.ndim == 3 else a, 'L').save(path)

def save_mod(mod_name, sx, sy, source, reference, data_type="real"):
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
    print(f"  Saved {len(sx)} samples to {out}")

# ============================================================
# 1. HYPERSPECTRAL_REMOTE: Indian Pines (famous, ~5MB)
# ============================================================
def build_hyperspectral_remote():
    print("=== hyperspectral_remote: Indian Pines ===")
    url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"  FAILED to download Indian Pines: {e}")
        return False

    import scipy.io
    mat = scipy.io.loadmat(io.BytesIO(r.content))
    cube = None
    for k in mat:
        if not k.startswith('_'):
            v = mat[k]
            if hasattr(v, 'shape') and len(v.shape) == 3:
                cube = v.astype(np.float32)
                break
    if cube is None:
        print("  FAILED: Could not find hyperspectral cube in .mat")
        return False
    print(f"  Indian Pines cube shape: {cube.shape}")

    h, w, bands = cube.shape
    ps = 32
    sx, sy = [], []
    count = 0
    for row in range(0, h - ps + 1, ps // 2):
        for col in range(0, w - ps + 1, ps // 2):
            if count >= 30: break
            patch = cube[row:row+ps, col:col+ps, :]
            x = normalize_01(patch[:,:,:3])
            noisy = patch + np.random.randn(*patch.shape).astype(np.float32) * (patch.std() * 0.05)
            y = normalize_01(noisy[:,:,:3])
            sx.append(x); sy.append(y)
            count += 1
        if count >= 30: break

    save_mod("hyperspectral_remote", sx, sy,
             "Indian Pines AVIRIS (corrected)",
             "Baumgardner et al., Purdue Univ., 2015",
             "real")
    return True


# ============================================================
# 2. XRAY_NDT: GDXray (direct PNG download)
# ============================================================
def build_xray_ndt():
    print("=== xray_ndt: GDXray ===")
    sx, sy = [], []
    for series in range(1, 70):
        if len(sx) >= 30: break
        sid = f"C{series:04d}"
        img_url = f"https://domingomery.ing.puc.cl/material/gdxray/Castings/{sid}/{sid}_0001.png"
        try:
            r = requests.get(img_url, timeout=15)
            if r.status_code != 200:
                continue
            img = Image.open(io.BytesIO(r.content)).convert('L')
            img = img.resize((256, 256), Image.LANCZOS)
            x = np.array(img, dtype=np.float32) / 255.0
            thickness = x * 3.0
            y = np.exp(-thickness) + np.random.randn(*x.shape).astype(np.float32) * 0.02
            y = np.clip(y, 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
            print(f"    Got {sid} ({len(sx)}/30)")
        except Exception:
            continue
    if len(sx) >= 20:
        save_mod("xray_ndt", sx, sy,
                 "GDXray Castings",
                 "Mery et al., JNDE 2015",
                 "real")
        return True
    else:
        print(f"  Only got {len(sx)} images from GDXray")
        return False


# ============================================================
# 3. PHASE_CONTRAST: ISBI Cell Tracking Challenge
# ============================================================
def build_phase_contrast():
    print("=== phase_contrast: ISBI Cell Tracking ===")
    urls = [
        "http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip",
        "http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip",
    ]
    all_imgs = []
    for url in urls:
        if len(all_imgs) >= 30: break
        try:
            print(f"  Downloading {url.split('/')[-1]}...")
            r = requests.get(url, timeout=300)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for name in sorted(z.namelist()):
                if name.lower().endswith(('.tif', '.tiff', '.png')):
                    try:
                        img_data = z.read(name)
                        img = Image.open(io.BytesIO(img_data)).convert('L')
                        img = img.resize((256, 256), Image.LANCZOS)
                        all_imgs.append(np.array(img, dtype=np.float32) / 255.0)
                        if len(all_imgs) >= 30: break
                    except:
                        pass
        except Exception as e:
            print(f"  Error: {e}")

    if len(all_imgs) >= 20:
        sx, sy = [], []
        for x in all_imgs[:30]:
            phase = x * np.pi * 0.8
            y = (0.5 + 0.3 * np.sin(phase) + np.random.randn(*x.shape).astype(np.float32) * 0.02).astype(np.float32)
            y = np.clip(y, 0, 1)
            sx.append(x); sy.append(y)
        save_mod("phase_contrast", sx, sy,
                 "ISBI Cell Tracking PhC-C2DH-U373",
                 "Ulman et al., Nat Methods 2017",
                 "real")
        return True
    else:
        print(f"  Only got {len(all_imgs)} images")
        return False


# ============================================================
# 4. FUNDUS: Try STARE (no registration needed)
# ============================================================
def build_fundus():
    print("=== fundus: STARE retinal ===")
    import tarfile
    stare_url = "http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar"
    try:
        print("  Downloading STARE retinal images...")
        r = requests.get(stare_url, timeout=120)
        if r.status_code != 200:
            print(f"  STARE HTTP {r.status_code}")
            return False
        tar = tarfile.open(fileobj=io.BytesIO(r.content))
        imgs = []
        for member in tar.getmembers():
            if member.name.lower().endswith(('.ppm', '.png', '.jpg', '.bmp')):
                f = tar.extractfile(member)
                if f:
                    img = Image.open(io.BytesIO(f.read())).convert('RGB')
                    img = img.resize((256, 256), Image.LANCZOS)
                    imgs.append(np.array(img, dtype=np.float32) / 255.0)
                    if len(imgs) >= 30: break
        if len(imgs) >= 20:
            sx, sy = [], []
            for x in imgs[:30]:
                y = x[:,:,1] + np.random.randn(256,256).astype(np.float32) * 0.03
                y = np.clip(y, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("fundus", sx, sy,
                     "STARE retinal fundus",
                     "Hoover et al., IEEE TMI 2000",
                     "real")
            return True
        else:
            print(f"  Only got {len(imgs)} STARE images")
            return False
    except Exception as e:
        print(f"  STARE download failed: {e}")
        return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = {}
    for name, func in [
        ("hyperspectral_remote", build_hyperspectral_remote),
        ("xray_ndt", build_xray_ndt),
        ("phase_contrast", build_phase_contrast),
        ("fundus", build_fundus),
    ]:
        try:
            results[name] = func()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            results[name] = False
        print()

    print("="*60)
    print("RESULTS:")
    for name, ok in results.items():
        status = "SUCCESS" if ok else "SKIPPED"
        print(f"  {name}: {status}")
