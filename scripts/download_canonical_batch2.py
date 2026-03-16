"""Download canonical datasets - batch 2.
Try more modalities with various download strategies.
"""
import os, sys, zipfile, io, json
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
# 1. CONFOCAL_LIVECELL: LIVECell from S3
# ============================================================
def build_confocal_livecell():
    print("=== confocal_livecell: LIVECell ===")
    # LIVECell images are on S3: s3://livecell-dataset/
    # Direct HTTP access via: https://livecell-dataset.s3.eu-central-1.amazonaws.com/
    # Image list is in annotations
    # Try to download a few images directly
    # From the paper, images are named like: A172_Phase_A7_1_00d00h00m_1.tif
    # Let's try the image folder
    base_s3 = "https://livecell-dataset.s3.eu-central-1.amazonaws.com"

    # First try to list via the annotations file
    ann_url = f"{base_s3}/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"
    try:
        print("  Downloading annotations to find image paths...")
        r = requests.get(ann_url, timeout=120)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return False
        ann = json.loads(r.content)
        image_names = [img['file_name'] for img in ann['images'][:50]]
        print(f"  Found {len(ann['images'])} total images, trying first 50...")

        imgs = []
        for fname in image_names:
            if len(imgs) >= 30: break
            img_url = f"{base_s3}/LIVECell_dataset_2021/images/livecell_train_val_images/{fname}"
            try:
                r2 = requests.get(img_url, timeout=30)
                if r2.status_code == 200:
                    img = Image.open(io.BytesIO(r2.content)).convert('L')
                    img = img.resize((256, 256), Image.LANCZOS)
                    imgs.append(np.array(img, dtype=np.float32) / 255.0)
                    if len(imgs) % 10 == 0:
                        print(f"    Downloaded {len(imgs)}/30")
            except:
                continue

        if len(imgs) >= 20:
            sx, sy = [], []
            for x in imgs[:30]:
                # Confocal forward model: PSF blur + Poisson noise
                from scipy.ndimage import gaussian_filter
                blurred = gaussian_filter(x, sigma=1.5)
                noisy = np.random.poisson(np.clip(blurred * 100, 0.1, None)).astype(np.float32) / 100.0
                y = np.clip(noisy, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("confocal_livecell", sx, sy,
                     "LIVECell phase-contrast cells",
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
# 2. FUNDUS: Try CHASE_DB1 (University of Kingston)
# ============================================================
def build_fundus():
    print("=== fundus: CHASE_DB1 retinal ===")
    # CHASE_DB1 is freely available
    url = "https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip"
    try:
        print("  Downloading CHASE_DB1...")
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            # Try alternative URL
            url2 = "https://blogs.kingston.ac.uk/retinal/chasedb1/"
            print(f"  HTTP {r.status_code}, trying alternative...")
            return False

        z = zipfile.ZipFile(io.BytesIO(r.content))
        imgs = []
        for name in sorted(z.namelist()):
            if name.lower().endswith(('.jpg', '.png', '.bmp')) and 'Image' in name:
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
                     "CHASE_DB1 retinal fundus",
                     "Fraz et al., IEEE TBE 2012",
                     "real")
            return True
    except Exception as e:
        print(f"  Error: {e}")

    # Try HRF (High-Resolution Fundus) - Zenodo
    print("  Trying HRF dataset...")
    # HRF is at: https://www5.cs.fau.de/research/data/fundus-images/
    # Try direct access
    return False


# ============================================================
# 3. XRAY_NDT: Try GDXray with different URL pattern
# ============================================================
def build_xray_ndt():
    print("=== xray_ndt: GDXray Castings ===")
    # Try series-level zip downloads
    # Or try Welds series instead of Castings
    sx, sy = [], []
    # Try Welds series
    for series_type in ['Castings', 'Welds']:
        for series in range(1, 80):
            if len(sx) >= 30: break
            sid = f"{'C' if series_type == 'Castings' else 'W'}{series:04d}"
            for img_num in range(1, 5):
                img_url = f"https://domingomery.ing.puc.cl/material/gdxray/{series_type}/{sid}/{sid}_{img_num:04d}.png"
                try:
                    r = requests.get(img_url, timeout=10)
                    if r.status_code == 200 and len(r.content) > 1000:
                        img = Image.open(io.BytesIO(r.content)).convert('L')
                        img = img.resize((256, 256), Image.LANCZOS)
                        x = np.array(img, dtype=np.float32) / 255.0
                        thickness = x * 3.0
                        y = np.exp(-thickness) + np.random.randn(*x.shape).astype(np.float32) * 0.02
                        y = np.clip(y, 0, 1).astype(np.float32)
                        sx.append(x); sy.append(y)
                        print(f"    Got {sid}_{img_num:04d} ({len(sx)}/30)")
                        break
                except:
                    continue
        if len(sx) >= 30: break

    if len(sx) >= 20:
        save_mod("xray_ndt", sx, sy,
                 "GDXray industrial radiography",
                 "Mery et al., JNDE 2015",
                 "real")
        return True
    else:
        print(f"  Only got {len(sx)} images")
        return False


# ============================================================
# 4. SONAR: UATD from Figshare
# ============================================================
def build_sonar():
    print("=== sonar: UATD ===")
    # UATD dataset on Figshare: https://figshare.com/articles/dataset/UATD_Dataset/21331143
    # Check if direct download is available
    api_url = "https://api.figshare.com/v2/articles/21331143"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code != 200:
            print(f"  Figshare API HTTP {r.status_code}")
            return False
        data = r.json()
        files = data.get('files', [])
        print(f"  Found {len(files)} files on Figshare")
        for f in files:
            sz = f.get('size', 0)
            print(f"    {f['name']}: {sz/1024/1024:.1f} MB")
        # Check if any are small enough to download
        small_files = [f for f in files if f['size'] < 500_000_000]
        if small_files:
            # Download the smallest one
            dl_file = min(small_files, key=lambda x: x['size'])
            print(f"  Downloading {dl_file['name']} ({dl_file['size']/1024/1024:.1f} MB)...")
            r2 = requests.get(dl_file['download_url'], timeout=300)
            if r2.status_code == 200:
                print(f"  Downloaded {len(r2.content)/1024/1024:.1f} MB")
                # Try to open as zip and extract images
                try:
                    z = zipfile.ZipFile(io.BytesIO(r2.content))
                    img_files = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))]
                    print(f"  Found {len(img_files)} image files in zip")
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
                            # Sonar forward model: speckle noise
                            speckle = np.random.exponential(1.0, x.shape).astype(np.float32)
                            y = np.clip(x * speckle, 0, 1).astype(np.float32)
                            sx.append(x); sy.append(y)
                        save_mod("sonar", sx, sy,
                                 "UATD forward-looking sonar",
                                 "Xie et al., 2022",
                                 "real")
                        return True
                except zipfile.BadZipFile:
                    print("  Not a valid zip file")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 5. CONFOCAL_3D: BioSR from Figshare
# ============================================================
def build_confocal_3d():
    print("=== confocal_3d: BioSR ===")
    api_url = "https://api.figshare.com/v2/articles/13264793"
    try:
        r = requests.get(api_url, timeout=30)
        data = r.json()
        files = data.get('files', [])
        print(f"  Found {len(files)} files on Figshare")
        for f in files[:5]:
            print(f"    {f['name']}: {f['size']/1024/1024:.1f} MB")
        # BioSR files are very large (.tif stacks)
        small = [f for f in files if f['size'] < 200_000_000]
        if small:
            print(f"  {len(small)} files under 200MB")
        else:
            print("  All files too large for auto-download")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 6. SEM: Try NFFA-EUROPE from B2SHARE
# ============================================================
def build_sem():
    print("=== sem: NFFA-EUROPE SEM ===")
    # Try the B2SHARE API
    url = "https://b2share.eudat.eu/api/records/f1aa0f5ad38c456eaf7b04d47a65af53"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        files = data.get('files', [])
        if not files:
            # Try 'files' in different location
            meta = data.get('metadata', {})
            print(f"  Record title: {meta.get('titles', [{}])[0].get('title', 'N/A')}")

        # The NFFA-EUROPE dataset has 10 categories of SEM images
        # Try to find downloadable file URLs
        for f in files[:5]:
            print(f"    {f.get('key', 'N/A')}: {f.get('size', 0)/1024/1024:.1f} MB")

    except Exception as e:
        print(f"  Error: {e}")

    # Alternative: use the existing Zenodo SEM nanoparticle data (7986673)
    # which is already used and IS real SEM data
    print("  NFFA-EUROPE B2SHARE access complex - keeping existing SEM data")
    return False


# ============================================================
# 7. FLUOROSCOPY: UCL WEISS dataset
# ============================================================
def build_fluoroscopy():
    print("=== fluoroscopy: UCL WEISS ===")
    # UCL Research Data Repository
    url = "https://rdr.ucl.ac.uk/ndownloader/articles/24624243/versions/1"
    try:
        print("  Trying UCL RDR download...")
        r = requests.get(url, timeout=60, allow_redirects=True)
        print(f"  HTTP {r.status_code}, content-type: {r.headers.get('content-type', 'N/A')}")
        if r.status_code == 200 and len(r.content) > 10000:
            # Check if it's a zip
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                names = z.namelist()
                print(f"  Zip contains {len(names)} files")
                img_files = [n for n in names if n.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
                print(f"  Image files: {len(img_files)}")
            except:
                print("  Not a zip file")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 8. CONFOCAL_ENDOMICROSCOPY: Try alternative CVC-EndoSceneStill
# ============================================================
def build_confocal_endomicroscopy():
    print("=== confocal_endomicroscopy: trying alternatives ===")
    # CVC-ClinicDB requires registration
    # But Kvasir dataset is freely available for GI tract images
    # https://datasets.simula.no/kvasir/
    url = "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip"
    try:
        print("  Downloading Kvasir v2 (endoscopy/endomicroscopy images)...")
        r = requests.get(url, timeout=300, stream=True)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return False
        content = b''
        for chunk in r.iter_content(chunk_size=1024*1024):
            content += chunk
            if len(content) > 500_000_000:
                break
        z = zipfile.ZipFile(io.BytesIO(content))
        # Look for images in any category
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
                from scipy.ndimage import gaussian_filter
                # Endomicroscopy: fiber bundle pattern + blur
                gray = 0.2989*x[:,:,0] + 0.587*x[:,:,1] + 0.114*x[:,:,2]
                blurred = gaussian_filter(gray, sigma=2.0)
                noisy = blurred + np.random.randn(256,256).astype(np.float32) * 0.05
                y = np.clip(noisy, 0, 1).astype(np.float32)
                sx.append(x); sy.append(y)
            save_mod("confocal_endomicroscopy", sx, sy,
                     "Kvasir GI tract endoscopy",
                     "Pogorelov et al., MMM 2017",
                     "real")
            return True
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 9. WIDEFIELD: FMD - try direct GitHub raw files
# ============================================================
def build_widefield():
    print("=== widefield: FMD ===")
    # The FMD data is in Google Drive linked from GitHub
    # https://github.com/yinhaoz/denoising-fluorescence
    # Cannot access Google Drive programmatically without API
    print("  FMD data hosted on Google Drive - requires manual download")
    return False


# ============================================================
# 10. LIGHTSHEET: CARE Tribolium
# ============================================================
def build_lightsheet():
    print("=== lightsheet: CARE Tribolium ===")
    # CARE example data can be downloaded from:
    # https://cloud.mpi-cbg.de/index.php/s/eoQn9KbCLoL0JW0/download
    # This is the Tribolium light-sheet training data
    url = "https://cloud.mpi-cbg.de/index.php/s/eoQn9KbCLoL0JW0/download"
    try:
        print("  Downloading CARE Tribolium data...")
        r = requests.get(url, timeout=300, allow_redirects=True)
        print(f"  HTTP {r.status_code}, size: {len(r.content)/1024/1024:.1f} MB")
        if r.status_code == 200 and len(r.content) > 10000:
            # Try to open as zip or tif
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                print(f"  Zip contains {len(z.namelist())} files")
                for n in z.namelist()[:10]:
                    print(f"    {n}")
            except:
                print("  Not a zip file, trying as TIFF...")
                try:
                    img = Image.open(io.BytesIO(r.content))
                    print(f"  Image size: {img.size}, mode: {img.mode}")
                except:
                    print("  Not a recognized image format")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = {}
    for name, func in [
        ("confocal_livecell", build_confocal_livecell),
        ("fundus", build_fundus),
        ("xray_ndt", build_xray_ndt),
        ("sonar", build_sonar),
        ("confocal_endomicroscopy", build_confocal_endomicroscopy),
        ("confocal_3d", build_confocal_3d),
        ("sem", build_sem),
        ("fluoroscopy", build_fluoroscopy),
        ("widefield", build_widefield),
        ("lightsheet", build_lightsheet),
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
