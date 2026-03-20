"""Download canonical datasets - batch 4.
Targeted retries with fixes for close-to-working sources.
"""
import os, sys, zipfile, io, json, warnings
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
# 1. CONFOCAL_ENDOMICROSCOPY: Kvasir v2 with SSL verify=False
# ============================================================
def build_confocal_endomicroscopy():
    print("=== confocal_endomicroscopy: Kvasir v2 ===")
    url = "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip"
    try:
        print("  Downloading Kvasir v2 (SSL verify disabled)...")
        r = requests.get(url, timeout=300, verify=False, stream=True)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return False
        content = b''
        for chunk in r.iter_content(chunk_size=1024*1024):
            content += chunk
            if len(content) % (50*1024*1024) < 1024*1024:
                print(f"    Downloaded {len(content)/1024/1024:.0f} MB...")
            if len(content) > 600_000_000:
                break
        print(f"  Total download: {len(content)/1024/1024:.0f} MB")
        z = zipfile.ZipFile(io.BytesIO(content))
        img_files = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.png'))]
        print(f"  Found {len(img_files)} images in Kvasir")
        imgs = []
        for name in sorted(img_files)[:50]:
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
                     "Kvasir v2 GI tract endoscopy",
                     "Pogorelov et al., ACM MMSys 2017",
                     "real")
            return True
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 2. CONFOCAL_LIVECELL: LIVECell - try direct S3 with correct path
# ============================================================
def build_confocal_livecell():
    print("=== confocal_livecell: LIVECell ===")
    base_s3 = "https://livecell-dataset.s3.eu-central-1.amazonaws.com"
    # First get the train annotations to learn image paths
    ann_url = f"{base_s3}/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"
    try:
        print("  Getting annotation file for image paths...")
        r = requests.get(ann_url, timeout=120)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return False
        ann = r.json()
        print(f"  Total images in annotations: {len(ann['images'])}")

        # The image path is relative - construct full S3 URL
        # Images are at: LIVECell_dataset_2021/images/livecell_train_val_images/{file_name}
        imgs = []
        for img_info in ann['images'][:60]:
            if len(imgs) >= 30: break
            fname = img_info['file_name']
            # Try different path patterns
            for path_pattern in [
                f"LIVECell_dataset_2021/images/livecell_train_val_images/{fname}",
                f"LIVECell_dataset_2021/images/{fname}",
            ]:
                img_url = f"{base_s3}/{path_pattern}"
                try:
                    r2 = requests.get(img_url, timeout=15)
                    if r2.status_code == 200 and len(r2.content) > 5000:
                        img = Image.open(io.BytesIO(r2.content)).convert('L')
                        img = img.resize((256, 256), Image.LANCZOS)
                        imgs.append(np.array(img, dtype=np.float32) / 255.0)
                        if len(imgs) % 10 == 0:
                            print(f"    Got {len(imgs)}/30 images")
                        break
                except:
                    continue

        if len(imgs) >= 20:
            sx, sy = [], []
            for x in imgs[:30]:
                blurred = gaussian_filter(x, sigma=1.5)
                noisy = np.random.poisson(np.clip(blurred * 100, 0.1, None)).astype(np.float32) / 100.0
                y = np.clip(noisy, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("confocal_livecell", sx, sy,
                     "LIVECell phase-contrast cell images",
                     "Edlund et al., Nat Methods 2021",
                     "real")
            return True
        else:
            print(f"  Only got {len(imgs)} images")
            return False
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 3. SONAR: UATD OpenSLT (76MB) - parse it properly
# ============================================================
def build_sonar():
    print("=== sonar: UATD OpenSLT ===")
    api_url = "https://api.figshare.com/v2/articles/21331143"
    try:
        r = requests.get(api_url, timeout=30)
        data = r.json()
        files = data.get('files', [])
        # Find the OpenSLT zip (smallest)
        openslt = [f for f in files if 'OpenSLT' in f['name']]
        if not openslt:
            print("  No OpenSLT file found")
            return False
        dl = openslt[0]
        print(f"  Downloading {dl['name']} ({dl['size']/1024/1024:.1f} MB)...")
        r2 = requests.get(dl['download_url'], timeout=300)
        if r2.status_code != 200:
            print(f"  HTTP {r2.status_code}")
            return False
        z = zipfile.ZipFile(io.BytesIO(r2.content))
        all_files = z.namelist()
        print(f"  Zip contains {len(all_files)} files")
        # Print some examples
        for n in all_files[:20]:
            print(f"    {n}")
        # Try to find images (could be in subdirs)
        img_files = [n for n in all_files if n.lower().endswith(('.png', '.jpg', '.bmp', '.tif', '.jpeg'))]
        print(f"  Image files found: {len(img_files)}")
        # Also check for npy or other data files
        data_files = [n for n in all_files if n.lower().endswith(('.npy', '.mat', '.xml', '.txt'))]
        print(f"  Data files found: {len(data_files)}")

        if img_files:
            imgs = []
            for name in sorted(img_files)[:50]:
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
                    speckle = np.random.exponential(1.0, x.shape).astype(np.float32)
                    y = np.clip(x * speckle, 0, 1).astype(np.float32)
                    sx.append(x); sy.append(y)
                save_mod("sonar", sx, sy,
                         "UATD forward-looking sonar",
                         "Xie et al., 2022",
                         "real")
                return True
            else:
                print(f"  Only got {len(imgs)} parseable images")
        return False
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 4. FUNDUS: Try multiple sources
# ============================================================
def build_fundus():
    print("=== fundus: trying multiple sources ===")
    # Try Kaggle-mirrored STARE
    # Or try HRF (High Resolution Fundus) - but needs registration
    # Or try Messidor: https://www.adcis.net/en/third-party/messidor/
    # Try IDRiD from IEEE DataPort mirror on Kaggle
    # Actually, try a Zenodo fundus dataset
    api_url = "https://zenodo.org/api/records?q=retinal+fundus+images&size=5&sort=bestmatch"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            hits = r.json().get('hits', {}).get('hits', [])
            for h in hits:
                meta = h.get('metadata', {})
                title = meta.get('title', 'N/A')
                rid = h.get('id', 'N/A')
                total = sum(f.get('size', 0) for f in h.get('files', [])) / 1024 / 1024
                print(f"  Zenodo {rid}: {title[:50]} ({total:.0f} MB)")
                # Try to download small fundus datasets
                for f in h.get('files', []):
                    if f['size'] < 200_000_000 and f['key'].lower().endswith('.zip'):
                        print(f"    Trying {f['key']} ({f['size']/1024/1024:.0f} MB)...")
                        try:
                            r2 = requests.get(f['links']['self'], timeout=120)
                            if r2.status_code == 200:
                                try:
                                    z = zipfile.ZipFile(io.BytesIO(r2.content))
                                    img_files = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.png', '.tif', '.bmp'))]
                                    print(f"    Found {len(img_files)} images")
                                    if len(img_files) >= 20:
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
                                                y = x[:,:,1] + np.random.randn(256,256).astype(np.float32) * 0.03
                                                y = np.clip(y, 0, 1).astype(np.float32)
                                                sx.append(x); sy.append(y)
                                            save_mod("fundus", sx, sy,
                                                     f"Zenodo {rid} retinal fundus",
                                                     title[:50],
                                                     "real")
                                            return True
                                except zipfile.BadZipFile:
                                    pass
                        except:
                            pass
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 5. RADIO_ASTRONOMY: Try astropy for FITS or use SkyView
# ============================================================
def build_radio_astronomy():
    print("=== radio_astronomy: FIRST VLA via SkyView ===")
    try:
        from astropy.io import fits
        HAS_ASTROPY = True
        print("  astropy available!")
    except ImportError:
        HAS_ASTROPY = False
        print("  astropy not available")

    # Try SkyView (NASA) virtual observatory which returns FITS
    # https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?Survey=FIRST&Position=180.0,30.0&Size=0.5&Pixels=256&Return=FITS
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
        url = f"https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?Survey=FIRST&Position={ra},{dec}&Size=0.5&Pixels=256&Return=FITS"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                if HAS_ASTROPY:
                    hdul = fits.open(io.BytesIO(r.content))
                    data = hdul[0].data
                    if data is not None and data.ndim == 2:
                        img = normalize_01(data.astype(np.float32))
                        imgs.append(img)
                        print(f"    Got FIRST cutout at ({ra},{dec}) ({len(imgs)}/30)")
                else:
                    # Try to return the FITS as raw binary if no astropy
                    # FITS header is ASCII in 2880-byte blocks
                    # Skip parsing without astropy
                    pass
        except Exception as e:
            continue

    if len(imgs) >= 20:
        sx, sy = [], []
        for x in imgs[:30]:
            dirty = gaussian_filter(x, sigma=3.0)
            noisy = dirty + np.random.randn(*x.shape).astype(np.float32) * 0.05
            y = np.clip(noisy, 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        save_mod("radio_astronomy", sx, sy,
                 "FIRST VLA 1.4GHz Survey (SkyView)",
                 "Becker et al., ApJ 1995",
                 "real")
        return True
    else:
        print(f"  Only got {len(imgs)} FITS cutouts")
        return False


# ============================================================
# 6. WIDEFIELD + WIDEFIELD_LOWDOSE: Try FMD Zenodo mirror
# ============================================================
def build_widefield():
    print("=== widefield: looking for FMD on Zenodo ===")
    # Search Zenodo for fluorescence microscopy denoising
    api_url = "https://zenodo.org/api/records?q=fluorescence+microscopy+denoising&size=5&sort=bestmatch"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            print(f"  Zenodo {rid}: {title[:60]}")
    except:
        pass
    print("  FMD data only on Google Drive - manual download required")
    return False


# ============================================================
# 7. SHG: PSHG-TISS on OSF - try downloading files
# ============================================================
def build_shg():
    print("=== shg: PSHG-TISS on OSF ===")
    api_url = "https://api.osf.io/v2/nodes/k2z8g/files/osfstorage/"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            items = data.get('data', [])
            for item in items:
                attrs = item.get('attributes', {})
                kind = attrs.get('kind', '')
                name = attrs.get('name', 'N/A')
                size = attrs.get('size', 0)
                if kind == 'file' and size > 0:
                    print(f"  File: {name} ({size/1024/1024:.1f} MB)")
                    if size < 200_000_000 and name.lower().endswith(('.zip', '.tar.gz')):
                        dl_url = item.get('links', {}).get('download', '')
                        if dl_url:
                            print(f"  Downloading {name}...")
                            r2 = requests.get(dl_url, timeout=300)
                            if r2.status_code == 200:
                                print(f"  Got {len(r2.content)/1024/1024:.1f} MB")
                elif kind == 'folder':
                    print(f"  Folder: {name}")
                    # List folder contents
                    folder_url = item.get('relationships', {}).get('files', {}).get('links', {}).get('related', {}).get('href', '')
                    if folder_url:
                        r3 = requests.get(folder_url, timeout=30)
                        if r3.status_code == 200:
                            sub_items = r3.json().get('data', [])
                            for si in sub_items[:5]:
                                sa = si.get('attributes', {})
                                print(f"    {sa.get('name', 'N/A')}: {sa.get('size', 0)/1024/1024:.1f} MB")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 8. TERAHERTZ: try GitHub raw with correct branch
# ============================================================
def build_terahertz():
    print("=== terahertz: THz_Dataset GitHub ===")
    # Try main and master branch
    for branch in ['main', 'master']:
        api_url = f"https://api.github.com/repos/LingLIx/THz_Dataset/git/trees/{branch}?recursive=1"
        try:
            r = requests.get(api_url, timeout=30)
            if r.status_code == 200:
                tree = r.json().get('tree', [])
                img_files = [t for t in tree if t['path'].lower().endswith(('.png', '.jpg', '.bmp', '.tif', '.jpeg'))]
                print(f"  Branch {branch}: {len(img_files)} image files")
                if img_files:
                    imgs = []
                    for f in img_files:
                        if len(imgs) >= 30: break
                        raw_url = f"https://raw.githubusercontent.com/LingLIx/THz_Dataset/{branch}/{f['path']}"
                        try:
                            r2 = requests.get(raw_url, timeout=15)
                            if r2.status_code == 200:
                                img = Image.open(io.BytesIO(r2.content)).convert('L')
                                img = img.resize((256, 256), Image.LANCZOS)
                                imgs.append(np.array(img, dtype=np.float32) / 255.0)
                        except:
                            pass
                    if len(imgs) >= 20:
                        sx, sy = [], []
                        for x in imgs[:30]:
                            blurred = gaussian_filter(x, sigma=2.0)
                            noisy = blurred + np.random.randn(*x.shape).astype(np.float32) * 0.05
                            y = np.clip(noisy, 0, 1).astype(np.float32)
                            sx.append(x); sy.append(y)
                        save_mod("terahertz", sx, sy,
                                 "Active THz imaging dataset",
                                 "Ling et al.",
                                 "real")
                        return True
                    print(f"  Only got {len(imgs)} images from {branch}")
                break
        except:
            continue
    return False


# ============================================================
# 9. XRAY_NDT: Try via Zenodo mirror of GDXray
# ============================================================
def build_xray_ndt():
    print("=== xray_ndt: searching Zenodo for X-ray NDT ===")
    api_url = "https://zenodo.org/api/records?q=x-ray+inspection+NDT+radiography&size=5&sort=bestmatch"
    try:
        r = requests.get(api_url, timeout=30)
        hits = r.json().get('hits', {}).get('hits', [])
        for h in hits:
            title = h.get('metadata', {}).get('title', 'N/A')
            rid = h.get('id', 'N/A')
            files = h.get('files', [])
            total = sum(f.get('size', 0) for f in files) / 1024 / 1024
            print(f"  Zenodo {rid}: {title[:50]} ({total:.0f} MB)")
            # Try small zip files
            for f in files:
                if f['size'] < 200_000_000 and f['key'].lower().endswith('.zip'):
                    print(f"    Trying {f['key']}...")
                    try:
                        r2 = requests.get(f['links']['self'], timeout=120)
                        if r2.status_code == 200:
                            z = zipfile.ZipFile(io.BytesIO(r2.content))
                            img_files = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
                            if len(img_files) >= 20:
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
                                             f"Zenodo {rid} X-ray NDT",
                                             title[:50],
                                             "real")
                                    return True
                    except:
                        pass
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = {}
    for name, func in [
        ("confocal_endomicroscopy", build_confocal_endomicroscopy),
        ("confocal_livecell", build_confocal_livecell),
        ("sonar", build_sonar),
        ("fundus", build_fundus),
        ("radio_astronomy", build_radio_astronomy),
        ("terahertz", build_terahertz),
        ("shg", build_shg),
        ("xray_ndt", build_xray_ndt),
        ("widefield", build_widefield),
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

    success = sum(1 for v in results.values() if v)
    print(f"\nSuccessful: {success}/{len(results)}")
