"""Download canonical datasets - batch 6.
Final attempts: fundus, lightsheet, SEM reclassify, widefield, stm.
"""
import os, sys, zipfile, io, json
from pathlib import Path
import numpy as np
import h5py
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
# 1. FUNDUS: Messidor-2 on ADCIS or Zenodo RFMiD
# ============================================================
def build_fundus():
    print("=== fundus: searching for downloadable retinal datasets ===")
    # Try RFMiD v2 on Zenodo 7505822 - check zip structure more carefully
    url = "https://zenodo.org/api/records/7505822"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        files = data.get('files', [])
        for f in files:
            sz = f['size'] / 1024 / 1024
            print(f"  {f['key']}: {sz:.1f} MB")
            if sz < 300 and f['key'].lower().endswith('.zip'):
                print(f"  Downloading {f['key']}...")
                r2 = requests.get(f['links']['self'], timeout=300)
                if r2.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(r2.content))
                    all_names = z.namelist()
                    print(f"  Zip has {len(all_names)} entries")
                    # Show structure
                    for n in all_names[:30]:
                        info = z.getinfo(n)
                        if info.file_size > 0:
                            print(f"    {n}: {info.file_size/1024:.0f} KB")
                    # Try all image-like files
                    img_files = [n for n in all_names
                               if n.lower().endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg'))
                               and not n.startswith('__MACOSX')]
                    print(f"  Image files: {len(img_files)}")
                    if img_files:
                        imgs = []
                        for name in sorted(img_files)[:40]:
                            try:
                                img = Image.open(io.BytesIO(z.read(name))).convert('RGB')
                                img = img.resize((256, 256), Image.LANCZOS)
                                arr = np.array(img, dtype=np.float32) / 255.0
                                if arr.mean() > 0.05:
                                    imgs.append(arr)
                                    if len(imgs) >= 30: break
                            except:
                                pass
                        if len(imgs) >= 20:
                            sx, sy = [], []
                            for x in imgs[:30]:
                                y = x[:,:,1] + np.random.randn(256,256).astype(np.float32) * 0.03
                                y = np.clip(y, 0, 1).astype(np.float32)
                                sx.append(x); sy.append(y)
                            save_mod("fundus", sx, sy,
                                     "RFMiD retinal fundus (Zenodo 7505822)",
                                     "Pachade et al., Data 2021",
                                     "real")
                            return True
                        print(f"  Only got {len(imgs)} fundus images")
    except Exception as e:
        print(f"  Error: {e}")

    # Try another approach: search for "retinal fundus" with small file size
    print("  Trying broader Zenodo search...")
    api_url = "https://zenodo.org/api/records?q=retinal+fundus+color+photograph&size=10&sort=mostrecent"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            files = h.get('files', [])
            # Only try if total < 500MB and has zip
            total = sum(f.get('size', 0) for f in files) / 1024 / 1024
            zips = [f for f in files if f['key'].lower().endswith('.zip') and f['size'] < 500_000_000]
            if zips and total < 1000:
                print(f"  Zenodo {rid}: {title[:40]} ({total:.0f} MB)")
                for f in zips:
                    try:
                        r2 = requests.get(f['links']['self'], timeout=300)
                        if r2.status_code == 200 and r2.content[:2] == b'PK':
                            z = zipfile.ZipFile(io.BytesIO(r2.content))
                            img_files = [n for n in z.namelist()
                                       if n.lower().endswith(('.jpg', '.png', '.jpeg'))
                                       and not n.startswith('__')]
                            if len(img_files) >= 20:
                                print(f"    {len(img_files)} images in {f['key']}")
                                imgs = []
                                for name in sorted(img_files)[:40]:
                                    try:
                                        img = Image.open(io.BytesIO(z.read(name))).convert('RGB')
                                        img = img.resize((256, 256), Image.LANCZOS)
                                        arr = np.array(img, dtype=np.float32) / 255.0
                                        if arr.mean() > 0.05:
                                            imgs.append(arr)
                                            if len(imgs) >= 30: break
                                    except:
                                        pass
                                if len(imgs) >= 20:
                                    sx, sy = [], []
                                    for x in imgs[:30]:
                                        y = x[:,:,1] + np.random.randn(256,256).astype(np.float32) * 0.03
                                        y = np.clip(y, 0, 1).astype(np.float32)
                                        sx.append(x); sy.append(y)
                                    save_mod("fundus", sx, sy,
                                             f"Zenodo {rid} retinal fundus",
                                             title[:50],
                                             "real")
                                    return True
                    except:
                        pass
    except:
        pass
    return False


# ============================================================
# 2. LIGHTSHEET: Extract from Zenodo .npy test data
# ============================================================
def build_lightsheet():
    print("=== lightsheet: Zenodo Tribolium .npy ===")
    url = "https://zenodo.org/api/records/10664273"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        files = data.get('files', [])
        # Find test-input.npy and test-output.npy
        for f in files:
            if f['key'] == 'test-input.npy':
                print(f"  Downloading test-input.npy ({f['size']/1024/1024:.1f} MB)...")
                r2 = requests.get(f['links']['self'], timeout=120)
                if r2.status_code == 200:
                    arr = np.load(io.BytesIO(r2.content))
                    print(f"  Array shape: {arr.shape}, dtype: {arr.dtype}")
                    # If 3D (Z, H, W) or 4D (N, Z, H, W), extract slices
                    if arr.ndim >= 3:
                        imgs = []
                        if arr.ndim == 3:
                            for z in range(min(arr.shape[0], 40)):
                                sl = arr[z]
                                sl = normalize_01(sl.astype(np.float32))
                                pil = Image.fromarray((sl * 255).astype(np.uint8))
                                pil = pil.resize((256, 256), Image.LANCZOS)
                                imgs.append(np.array(pil, dtype=np.float32) / 255.0)
                                if len(imgs) >= 30: break
                        elif arr.ndim == 4:
                            for n in range(arr.shape[0]):
                                for z in range(arr.shape[1]):
                                    sl = arr[n, z]
                                    sl = normalize_01(sl.astype(np.float32))
                                    pil = Image.fromarray((sl * 255).astype(np.uint8))
                                    pil = pil.resize((256, 256), Image.LANCZOS)
                                    imgs.append(np.array(pil, dtype=np.float32) / 255.0)
                                    if len(imgs) >= 30: break
                                if len(imgs) >= 30: break
                        if len(imgs) >= 20:
                            sx, sy = [], []
                            for x in imgs[:30]:
                                noisy = np.random.poisson(np.clip(x * 50, 0.1, None)).astype(np.float32) / 50.0
                                y = np.clip(noisy, 0, 1).astype(np.float32)
                                sx.append(x); sy.append(y)
                            save_mod("lightsheet", sx, sy,
                                     "Tribolium light-sheet (Zenodo 10664273)",
                                     "Weigert et al., Nat Methods 2018",
                                     "real")
                            return True
                        print(f"  Only {len(imgs)} slices")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 3. CONFOCAL_LIVECELL: Try direct LIVECell image download
# ============================================================
def build_confocal_livecell():
    print("=== confocal_livecell: LIVECell direct ===")
    # Try downloading from the GitHub releases
    # https://github.com/sartorius-research/LIVECell/releases
    api_url = "https://api.github.com/repos/sartorius-research/LIVECell/releases"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            releases = r.json()
            for rel in releases[:3]:
                assets = rel.get('assets', [])
                print(f"  Release: {rel.get('tag_name', 'N/A')}")
                for a in assets:
                    print(f"    {a['name']}: {a['size']/1024/1024:.1f} MB")
    except:
        pass

    # The images are on S3 but might need correct Content-Type header
    base_s3 = "https://livecell-dataset.s3.eu-central-1.amazonaws.com"

    # Try listing the S3 bucket
    try:
        r = requests.get(f"{base_s3}/?prefix=LIVECell_dataset_2021/images/livecell_train_val_images/&max-keys=50", timeout=30)
        if r.status_code == 200:
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.content)
            ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
            keys = [el.text for el in root.findall('.//s3:Key', ns)]
            if not keys:
                # Try without namespace
                keys = [el.text for el in root.iter() if el.tag.endswith('Key')]
            print(f"  Found {len(keys)} keys in S3")
            if keys:
                for k in keys[:5]:
                    print(f"    {k}")
                imgs = []
                for key in keys[:50]:
                    if len(imgs) >= 30: break
                    if key.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                        img_url = f"{base_s3}/{key}"
                        try:
                            r2 = requests.get(img_url, timeout=30)
                            if r2.status_code == 200 and len(r2.content) > 5000:
                                img = Image.open(io.BytesIO(r2.content)).convert('L')
                                img = img.resize((256, 256), Image.LANCZOS)
                                imgs.append(np.array(img, dtype=np.float32) / 255.0)
                                if len(imgs) % 10 == 0:
                                    print(f"    Got {len(imgs)}/30")
                        except:
                            pass
                if len(imgs) >= 20:
                    sx, sy = [], []
                    for x in imgs[:30]:
                        blurred = gaussian_filter(x, sigma=1.5)
                        noisy = np.random.poisson(np.clip(blurred * 100, 0.1, None)).astype(np.float32) / 100.0
                        y = np.clip(noisy, 0, 1).astype(np.float32)
                        sx.append(x); sy.append(y)
                    save_mod("confocal_livecell", sx, sy,
                             "LIVECell phase-contrast cells",
                             "Edlund et al., Nat Methods 2021",
                             "real")
                    return True
    except Exception as e:
        print(f"  S3 list error: {e}")
    return False


# ============================================================
# 4. CONFOCAL_3D: Try BioSR - search for smaller subset
# ============================================================
def build_confocal_3d():
    print("=== confocal_3d: BioSR alternatives ===")
    # BioSR files are all >1GB on Figshare
    # Try Zenodo for confocal microscopy SR datasets
    api_url = "https://zenodo.org/api/records?q=confocal+super+resolution+microscopy+dataset&size=5"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            files = h.get('files', [])
            small = [f for f in files if f.get('size', 0) < 200_000_000]
            if small:
                total = sum(f.get('size', 0) for f in small) / 1024 / 1024
                print(f"  Zenodo {rid}: {title[:40]} ({total:.0f} MB downloadable)")
    except:
        pass
    return False


# ============================================================
# 5. STM: Try smaller STM datasets on Zenodo
# ============================================================
def build_stm():
    print("=== stm: searching for smaller STM dataset ===")
    api_url = "https://zenodo.org/api/records?q=scanning+tunneling+microscopy+STM+images&size=10"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            files = h.get('files', [])
            total = sum(f.get('size', 0) for f in files) / 1024 / 1024
            small_zips = [f for f in files if f.get('size', 0) < 200_000_000 and f['key'].lower().endswith('.zip')]
            if small_zips:
                print(f"  Zenodo {rid}: {title[:50]} ({total:.0f} MB)")
                for f in small_zips:
                    print(f"    Trying {f['key']} ({f['size']/1024/1024:.0f} MB)...")
                    try:
                        r2 = requests.get(f['links']['self'], timeout=300)
                        if r2.status_code == 200 and r2.content[:2] == b'PK':
                            z = zipfile.ZipFile(io.BytesIO(r2.content))
                            img_files = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.tif'))]
                            if len(img_files) >= 20:
                                print(f"    {len(img_files)} images")
                                imgs = []
                                for name in sorted(img_files)[:40]:
                                    try:
                                        img = Image.open(io.BytesIO(z.read(name))).convert('L')
                                        img = img.resize((256, 256), Image.LANCZOS)
                                        imgs.append(np.array(img, dtype=np.float32) / 255.0)
                                        if len(imgs) >= 30: break
                                    except:
                                        pass
                                if len(imgs) >= 20:
                                    sx, sy = [], []
                                    for x in imgs[:30]:
                                        y = x + np.random.randn(*x.shape).astype(np.float32) * 0.05
                                        y = np.clip(y, 0, 1).astype(np.float32)
                                        sx.append(x); sy.append(y)
                                    save_mod("stm", sx, sy,
                                             f"Zenodo {rid} STM",
                                             title[:50],
                                             "real")
                                    return True
                    except:
                        pass
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 6. TERAHERTZ: Search Zenodo for THz imaging datasets
# ============================================================
def build_terahertz():
    print("=== terahertz: searching Zenodo ===")
    api_url = "https://zenodo.org/api/records?q=terahertz+THz+imaging+dataset&size=10"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            files = h.get('files', [])
            total = sum(f.get('size', 0) for f in files) / 1024 / 1024
            small = [f for f in files if f.get('size', 0) < 200_000_000]
            if small:
                print(f"  Zenodo {rid}: {title[:50]} ({total:.0f} MB)")
                for f in small:
                    if f['key'].lower().endswith(('.zip', '.npy', '.mat')):
                        print(f"    {f['key']}: {f['size']/1024/1024:.0f} MB")
    except:
        pass
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = {}
    for name, func in [
        ("fundus", build_fundus),
        ("lightsheet", build_lightsheet),
        ("confocal_livecell", build_confocal_livecell),
        ("stm", build_stm),
        ("terahertz", build_terahertz),
        ("confocal_3d", build_confocal_3d),
    ]:
        try:
            results[name] = func()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = False
        print()

    print("="*60)
    print("RESULTS:")
    for name, ok in results.items():
        status = "SUCCESS" if ok else "SKIPPED"
        print(f"  {name}: {status}")
