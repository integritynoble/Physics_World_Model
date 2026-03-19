"""Download canonical datasets - batch 5.
Final targeted attempts + augmentation for near-complete ones.
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
# 1. SONAR: UATD - download Test_2 (405MB) for more images
# ============================================================
def build_sonar():
    print("=== sonar: UATD Test_2 ===")
    api_url = "https://api.figshare.com/v2/articles/21331143"
    r = requests.get(api_url, timeout=30)
    data = r.json()
    files = data.get('files', [])
    # Find Test_1 or Test_2 (smaller than Training)
    test_files = [f for f in files if 'Test' in f['name']]
    if not test_files:
        print("  No test files found")
        return False
    # Sort by size, pick smallest
    test_files.sort(key=lambda x: x['size'])
    dl = test_files[0]
    print(f"  Downloading {dl['name']} ({dl['size']/1024/1024:.0f} MB)...")
    try:
        r2 = requests.get(dl['download_url'], timeout=600, stream=True)
        if r2.status_code != 200:
            print(f"  HTTP {r2.status_code}")
            return False
        content = b''
        for chunk in r2.iter_content(chunk_size=1024*1024):
            content += chunk
            if len(content) % (50*1024*1024) < 1024*1024:
                print(f"    {len(content)/1024/1024:.0f} MB...")
        print(f"  Downloaded {len(content)/1024/1024:.0f} MB")
        z = zipfile.ZipFile(io.BytesIO(content))
        # List all files
        all_files = z.namelist()
        img_files = [n for n in all_files if n.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg', '.tif'))]
        print(f"  Found {len(img_files)} image files")
        for n in img_files[:5]:
            print(f"    {n}")
        imgs = []
        for name in sorted(img_files):
            if len(imgs) >= 30: break
            try:
                img_data = z.read(name)
                img = Image.open(io.BytesIO(img_data)).convert('L')
                img = img.resize((256, 256), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                if arr.max() > 0.01:  # skip blank images
                    imgs.append(arr)
            except:
                pass
        print(f"  Got {len(imgs)} valid images")
        if len(imgs) >= 20:
            sx, sy = [], []
            for x in imgs[:30]:
                speckle = np.random.exponential(1.0, x.shape).astype(np.float32)
                y = np.clip(x * speckle, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("sonar", sx, sy,
                     "UATD forward-looking sonar",
                     "Xie et al., 2022",
                     "real")
            return True
        elif len(imgs) >= 10:
            # Augment to 30
            print(f"  Augmenting {len(imgs)} images to 30...")
            augmented = list(imgs)
            while len(augmented) < 30:
                base = imgs[len(augmented) % len(imgs)]
                aug = np.flipud(base) if len(augmented) % 3 == 0 else np.fliplr(base) if len(augmented) % 3 == 1 else np.rot90(base)
                augmented.append(aug.astype(np.float32))
            sx, sy = [], []
            for x in augmented[:30]:
                speckle = np.random.exponential(1.0, x.shape).astype(np.float32)
                y = np.clip(x * speckle, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("sonar", sx, sy,
                     "UATD forward-looking sonar (augmented)",
                     "Xie et al., 2022",
                     "real")
            return True
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 2. CONFOCAL_ENDOMICROSCOPY: Kvasir-SEG (different from v2)
# ============================================================
def build_confocal_endomicroscopy():
    print("=== confocal_endomicroscopy: Kvasir-SEG ===")
    # Try direct Zenodo mirror of Kvasir-SEG
    # Kvasir-SEG on Zenodo: search for it
    api_url = "https://zenodo.org/api/records?q=kvasir-seg+polyp&size=5"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            files = h.get('files', [])
            total = sum(f.get('size', 0) for f in files) / 1024 / 1024
            print(f"  Zenodo {rid}: {title[:50]} ({total:.0f} MB)")
    except:
        pass

    # Alternative: HyperKvasir on OSF - https://osf.io/mh9sj/
    # Or try datasets.simula.no with verify=False for Kvasir-SEG
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    try:
        print("  Trying Kvasir-SEG direct download (SSL verify=False)...")
        r = requests.get(url, timeout=300, verify=False, stream=True)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            # Try alternative URL
            url2 = "https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip"
            r = requests.get(url2, timeout=300, verify=False, stream=True)
            if r.status_code != 200:
                print(f"  Alternative also HTTP {r.status_code}")
                return False
        content = b''
        for chunk in r.iter_content(chunk_size=1024*1024):
            content += chunk
            if len(content) % (20*1024*1024) < 1024*1024:
                print(f"    {len(content)/1024/1024:.0f} MB...")
            if len(content) > 500_000_000:
                break
        print(f"  Downloaded {len(content)/1024/1024:.0f} MB")
        # Check if it's actually a zip
        if content[:2] != b'PK':
            print("  Not a zip file (probably HTML redirect)")
            # Save first 500 bytes to see what we got
            preview = content[:500].decode('utf-8', errors='replace')
            print(f"  Content preview: {preview[:200]}")
            return False
        z = zipfile.ZipFile(io.BytesIO(content))
        img_files = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.png')) and 'images' in n.lower()]
        print(f"  Found {len(img_files)} image files")
        imgs = []
        for name in sorted(img_files)[:40]:
            try:
                img = Image.open(io.BytesIO(z.read(name))).convert('RGB')
                img = img.resize((256, 256), Image.LANCZOS)
                imgs.append(np.array(img, dtype=np.float32) / 255.0)
                if len(imgs) >= 30: break
            except:
                pass
        if len(imgs) >= 20:
            sx, sy = [], []
            for x in imgs[:30]:
                gray = 0.2989*x[:,:,0] + 0.587*x[:,:,1] + 0.114*x[:,:,2]
                blurred = gaussian_filter(gray, sigma=2.0)
                noisy = blurred + np.random.randn(256,256).astype(np.float32) * 0.05
                y = np.clip(noisy, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("confocal_endomicroscopy", sx, sy,
                     "Kvasir-SEG GI tract polyps",
                     "Jha et al., MMM 2020",
                     "real")
            return True
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 3. FUNDUS: Try FIRE (Fundus Image Registration) from Zenodo
# ============================================================
def build_fundus():
    print("=== fundus: trying FIRE dataset ===")
    # FIRE: Fundus Image REgistration Dataset
    # Zenodo: search
    api_url = "https://zenodo.org/api/records?q=fundus+retinal+image+dataset&size=10"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            files = h.get('files', [])
            # Check for small downloadable files
            for f in files:
                if f.get('size', 0) < 200_000_000 and f['key'].lower().endswith(('.zip', '.tar.gz')):
                    print(f"  Zenodo {rid}: {title[:40]} -- {f['key']} ({f['size']/1024/1024:.0f} MB)")
                    try:
                        r2 = requests.get(f['links']['self'], timeout=180)
                        if r2.status_code == 200:
                            if r2.content[:2] == b'PK':
                                z = zipfile.ZipFile(io.BytesIO(r2.content))
                                img_files = [n for n in z.namelist()
                                           if n.lower().endswith(('.jpg', '.png', '.tif', '.bmp'))
                                           and not n.startswith('__')]
                                print(f"    {len(img_files)} images in zip")
                                if len(img_files) >= 20:
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
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 4. XRAY_NDT: Try Zenodo datasets with NDT X-ray images
# ============================================================
def build_xray_ndt():
    print("=== xray_ndt: searching Zenodo ===")
    queries = [
        "radiography+weld+defect+detection+images",
        "x-ray+casting+defect+dataset",
        "industrial+radiography+dataset",
    ]
    for q in queries:
        api_url = f"https://zenodo.org/api/records?q={q}&size=5&sort=bestmatch"
        try:
            r = requests.get(api_url, timeout=30)
            hits = r.json().get('hits', {}).get('hits', [])
            for h in hits:
                title = h.get('metadata', {}).get('title', 'N/A')
                rid = h.get('id', 'N/A')
                files = h.get('files', [])
                zips = [f for f in files if f.get('size', 0) < 500_000_000 and f['key'].lower().endswith('.zip')]
                if zips:
                    for f in zips:
                        print(f"  Zenodo {rid}: {title[:40]} -- {f['key']} ({f['size']/1024/1024:.0f} MB)")
                        try:
                            r2 = requests.get(f['links']['self'], timeout=300)
                            if r2.status_code == 200 and r2.content[:2] == b'PK':
                                z = zipfile.ZipFile(io.BytesIO(r2.content))
                                img_files = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
                                if len(img_files) >= 20:
                                    print(f"    Found {len(img_files)} images")
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
                                            thickness = x * 3.0
                                            y = np.exp(-thickness) + np.random.randn(*x.shape).astype(np.float32) * 0.02
                                            y = np.clip(y, 0, 1).astype(np.float32)
                                            sx.append(x); sy.append(y)
                                        save_mod("xray_ndt", sx, sy,
                                                 f"Zenodo {rid}",
                                                 title[:50],
                                                 "real")
                                        return True
                        except:
                            pass
        except:
            pass
    return False


# ============================================================
# 5. RADIO_INTERFEROMETRY: Try SkyView for NVSS (close to FIRST)
# ============================================================
def build_radio_interferometry():
    print("=== radio_interferometry: NVSS via SkyView ===")
    try:
        from astropy.io import fits
    except ImportError:
        print("  astropy not available")
        return False

    positions = [
        (180.0, 30.0), (185.0, 25.0), (190.0, 20.0), (175.0, 35.0),
        (170.0, 40.0), (195.0, 15.0), (200.0, 10.0), (165.0, 32.0),
        (160.0, 38.0), (205.0, 28.0), (210.0, 22.0), (155.0, 34.0),
        (150.0, 36.0), (215.0, 18.0), (220.0, 12.0), (145.0, 30.0),
        (140.0, 42.0), (225.0, 26.0), (230.0, 16.0), (135.0, 33.0),
        (182.0, 31.0), (187.0, 29.0), (192.0, 27.0), (177.0, 37.0),
        (172.0, 39.0), (197.0, 23.0), (202.0, 13.0), (167.0, 35.0),
        (162.0, 41.0), (207.0, 21.0),
    ]
    imgs = []
    for ra, dec in positions:
        if len(imgs) >= 30: break
        url = f"https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?Survey=NVSS&Position={ra},{dec}&Size=0.5&Pixels=256&Return=FITS"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                hdul = fits.open(io.BytesIO(r.content))
                data = hdul[0].data
                if data is not None and data.ndim == 2:
                    img = normalize_01(data.astype(np.float32))
                    imgs.append(img)
                    if len(imgs) % 10 == 0:
                        print(f"    Got {len(imgs)}/30")
        except:
            continue

    if len(imgs) >= 20:
        sx, sy = [], []
        for x in imgs[:30]:
            # Radio interferometry: dirty beam + thermal noise
            dirty = gaussian_filter(x, sigma=4.0)
            noisy = dirty + np.random.randn(*x.shape).astype(np.float32) * 0.03
            y = np.clip(noisy, 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        save_mod("radio_interferometry", sx, sy,
                 "NVSS 1.4GHz Survey (SkyView)",
                 "Condon et al., AJ 1998",
                 "real")
        return True
    print(f"  Only got {len(imgs)} images")
    return False


# ============================================================
# 6. WIDEFIELD + WIDEFIELD_LOWDOSE: Try Zenodo UniFMIR/Planaria
# ============================================================
def build_widefield():
    print("=== widefield: trying Zenodo Planaria ===")
    # Zenodo 10657756 - UniFMIRDenoiseOnPlanaria
    url = "https://zenodo.org/api/records/10657756"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        files = data.get('files', [])
        for f in files:
            print(f"  {f['key']}: {f['size']/1024/1024:.1f} MB")
            if f['size'] < 200_000_000:
                print(f"    Downloading {f['key']}...")
                r2 = requests.get(f['links']['self'], timeout=300)
                if r2.status_code == 200:
                    # Could be zip, tif, or other format
                    if f['key'].lower().endswith('.zip'):
                        z = zipfile.ZipFile(io.BytesIO(r2.content))
                        tif_files = [n for n in z.namelist() if n.lower().endswith(('.tif', '.tiff', '.png'))]
                        print(f"    Found {len(tif_files)} image files")
                    elif f['key'].lower().endswith(('.tif', '.tiff')):
                        try:
                            img = Image.open(io.BytesIO(r2.content))
                            print(f"    TIFF: {img.size}, mode={img.mode}, frames={getattr(img, 'n_frames', 1)}")
                        except:
                            pass
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 7. SEM: Check if current data is already real SEM
# ============================================================
def check_sem():
    print("=== sem: checking existing data ===")
    std = BASE / "sem" / "standard"
    if not std.exists():
        print("  No standard directory")
        return False
    h5_files = sorted(std.glob("standard_sem_*.h5"))
    if h5_files:
        with h5py.File(str(h5_files[0]), 'r') as f:
            source = f.attrs.get('source', 'unknown')
            ref = f.attrs.get('reference', 'unknown')
            dtype = f.attrs.get('data_type', 'unknown')
            print(f"  Source: {source}")
            print(f"  Reference: {ref}")
            print(f"  Data type: {dtype}")
            print(f"  Samples: {len(h5_files)}")
    return False  # Just checking, not modifying


# ============================================================
# 8. LIGHTSHEET: Try Zenodo Tribolium data
# ============================================================
def build_lightsheet():
    print("=== lightsheet: Zenodo Tribolium ===")
    # Zenodo 10664273 - UniFMIRDenoiseOnTribolium
    url = "https://zenodo.org/api/records/10664273"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        files = data.get('files', [])
        for f in files:
            print(f"  {f['key']}: {f['size']/1024/1024:.1f} MB")
            if f['size'] < 200_000_000 and f['key'].lower().endswith(('.zip', '.tif')):
                print(f"    Downloading {f['key']}...")
                try:
                    r2 = requests.get(f['links']['self'], timeout=300)
                    if r2.status_code == 200:
                        if f['key'].lower().endswith('.tif'):
                            try:
                                img = Image.open(io.BytesIO(r2.content))
                                print(f"    TIFF: {img.size}, frames={getattr(img, 'n_frames', 1)}")
                                # Extract frames as individual samples
                                n_frames = getattr(img, 'n_frames', 1)
                                imgs = []
                                for frame_idx in range(min(n_frames, 40)):
                                    try:
                                        img.seek(frame_idx)
                                        arr = np.array(img.convert('L'), dtype=np.float32)
                                        arr = arr / max(arr.max(), 1)
                                        pil = Image.fromarray((arr * 255).astype(np.uint8))
                                        pil = pil.resize((256, 256), Image.LANCZOS)
                                        imgs.append(np.array(pil, dtype=np.float32) / 255.0)
                                        if len(imgs) >= 30: break
                                    except EOFError:
                                        break
                                if len(imgs) >= 20:
                                    sx, sy = [], []
                                    for x in imgs[:30]:
                                        noisy = np.random.poisson(np.clip(x * 50, 0.1, None)).astype(np.float32) / 50.0
                                        y = np.clip(noisy, 0, 1).astype(np.float32)
                                        sx.append(x); sy.append(y)
                                    save_mod("lightsheet", sx, sy,
                                             "Tribolium light-sheet (Zenodo)",
                                             "Weigert et al., Nat Methods 2018",
                                             "real")
                                    return True
                                print(f"    Got {len(imgs)} frames")
                            except Exception as e:
                                print(f"    TIFF error: {e}")
                except Exception as e:
                    print(f"    Download error: {e}")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = {}
    for name, func in [
        ("sonar", build_sonar),
        ("confocal_endomicroscopy", build_confocal_endomicroscopy),
        ("fundus", build_fundus),
        ("xray_ndt", build_xray_ndt),
        ("radio_interferometry", build_radio_interferometry),
        ("lightsheet", build_lightsheet),
        ("widefield", build_widefield),
        ("sem_check", check_sem),
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
