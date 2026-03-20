"""Fix: Fundus dataset (was wrong data), try RFMiD nested zips."""
import os, sys, zipfile, io
from pathlib import Path
import numpy as np
import h5py
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter

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


def build_fundus():
    print("=== fundus: RFMiD nested zips ===")
    # RFMiD2_0.zip contains inner zips: Training_set.zip, Test_set.zip, Validation_set.zip
    url = "https://zenodo.org/api/records/7505822/files/RFMiD2_0.zip/content"
    try:
        print("  Downloading RFMiD2_0.zip...")
        r = requests.get(url, timeout=300)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return False
        outer_z = zipfile.ZipFile(io.BytesIO(r.content))
        imgs = []
        for inner_name in outer_z.namelist():
            if inner_name.lower().endswith('.zip'):
                print(f"  Extracting inner zip: {inner_name}")
                inner_data = outer_z.read(inner_name)
                try:
                    inner_z = zipfile.ZipFile(io.BytesIO(inner_data))
                    inner_files = inner_z.namelist()
                    img_files = [n for n in inner_files
                               if n.lower().endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg'))
                               and not n.startswith('__MACOSX')]
                    print(f"    {len(img_files)} images in {inner_name}")
                    for img_name in sorted(img_files)[:20]:
                        try:
                            img = Image.open(io.BytesIO(inner_z.read(img_name))).convert('RGB')
                            img = img.resize((256, 256), Image.LANCZOS)
                            arr = np.array(img, dtype=np.float32) / 255.0
                            if arr.mean() > 0.05 and arr.std() > 0.02:
                                imgs.append(arr)
                                if len(imgs) >= 30: break
                        except:
                            pass
                except zipfile.BadZipFile:
                    pass
            if len(imgs) >= 30: break

        if len(imgs) >= 20:
            sx, sy = [], []
            for x in imgs[:30]:
                # Fundus forward model: green channel extraction + noise
                y = x[:,:,1] + np.random.randn(256,256).astype(np.float32) * 0.03
                y = np.clip(y, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("fundus", sx, sy,
                     "RFMiD retinal fundus multi-disease (Zenodo 7505822)",
                     "Pachade et al., Data 2021",
                     "real")
            return True
        else:
            print(f"  Only got {len(imgs)} retinal images")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def rebuild_fundus_proxy():
    """If canonical download fails, rebuild from MedMNIST as proxy."""
    print("=== fundus: rebuilding proxy from MedMNIST ===")
    try:
        import medmnist
        from medmnist import RetinaMNIST
        # retinaMNIST has retinal fundus images (5 classes)
        ds = RetinaMNIST(split='train', download=True, size=128)
        imgs = []
        for i in range(min(len(ds), 40)):
            img, label = ds[i]
            img = np.array(img)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            pil = Image.fromarray(img).resize((256, 256), Image.LANCZOS)
            imgs.append(np.array(pil, dtype=np.float32) / 255.0)
            if len(imgs) >= 30: break
        if len(imgs) >= 20:
            sx, sy = [], []
            for x in imgs[:30]:
                y = x[:,:,1] + np.random.randn(256,256).astype(np.float32) * 0.03
                y = np.clip(y, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("fundus", sx, sy,
                     "MedMNIST RetinaMNIST fundus (proxy)",
                     "Yang et al., 2021",
                     "proxy")
            return True
    except ImportError:
        print("  medmnist not installed")
    except Exception as e:
        print(f"  Error: {e}")
    return False


if __name__ == "__main__":
    ok = build_fundus()
    if not ok:
        print("  Canonical download failed, trying proxy...")
        ok = rebuild_fundus_proxy()
    if ok:
        print("FUNDUS: FIXED")
    else:
        print("FUNDUS: STILL BROKEN - needs manual fix")
