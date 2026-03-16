"""Download canonical datasets - batch 3.
Try remaining accessible modalities.
"""
import os, sys, zipfile, io, json, struct
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


# ============================================================
# 1. RADIO_ASTRONOMY: FIRST VLA Survey (FITS cutouts)
# ============================================================
def build_radio_astronomy():
    print("=== radio_astronomy: FIRST VLA ===")
    # FIRST survey FITS cutout service
    # https://third.ucllnl.org/cgi-bin/firstimage?RA=12+00+00&Dec=+30+00+00&Equinox=J2000&ImageSize=2&MaxInt=10
    # Try several positions
    positions = [
        ("12+00+00", "+30+00+00"),
        ("13+00+00", "+25+00+00"),
        ("14+00+00", "+20+00+00"),
        ("10+30+00", "+35+00+00"),
        ("11+00+00", "+28+00+00"),
        ("09+30+00", "+40+00+00"),
        ("15+00+00", "+15+00+00"),
        ("08+00+00", "+45+00+00"),
        ("16+00+00", "+10+00+00"),
        ("12+30+00", "+32+00+00"),
        ("13+30+00", "+22+00+00"),
        ("14+30+00", "+18+00+00"),
        ("10+00+00", "+38+00+00"),
        ("11+30+00", "+26+00+00"),
        ("09+00+00", "+42+00+00"),
        ("15+30+00", "+12+00+00"),
        ("08+30+00", "+44+00+00"),
        ("16+30+00", "+08+00+00"),
        ("12+15+00", "+33+00+00"),
        ("13+15+00", "+23+00+00"),
        ("14+15+00", "+19+00+00"),
        ("10+15+00", "+37+00+00"),
        ("11+15+00", "+27+00+00"),
        ("09+15+00", "+41+00+00"),
        ("15+15+00", "+13+00+00"),
        ("08+15+00", "+43+00+00"),
        ("16+15+00", "+09+00+00"),
        ("12+45+00", "+31+00+00"),
        ("13+45+00", "+24+00+00"),
        ("14+45+00", "+17+00+00"),
    ]
    imgs = []
    for ra, dec in positions:
        if len(imgs) >= 30: break
        url = f"https://third.ucllnl.org/cgi-bin/firstimage?RA={ra}&Dec={dec}&Equinox=J2000&ImageSize=2&MaxInt=10&FITS=1"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                # Try to parse FITS
                try:
                    from astropy.io import fits
                    hdu = fits.open(io.BytesIO(r.content))
                    data = hdu[0].data
                    if data is not None and data.ndim == 2:
                        img = normalize_01(data.astype(np.float32))
                        if img.shape[0] >= 64 and img.shape[1] >= 64:
                            pil_img = Image.fromarray((img * 255).astype(np.uint8))
                            pil_img = pil_img.resize((256, 256), Image.LANCZOS)
                            imgs.append(np.array(pil_img, dtype=np.float32) / 255.0)
                            print(f"    Got FIRST cutout {ra} {dec} ({len(imgs)}/30)")
                except ImportError:
                    # No astropy, try manual FITS parsing
                    # Simple FITS: 2880-byte header blocks, then data
                    pass
        except:
            continue

    if len(imgs) >= 20:
        sx, sy = [], []
        for x in imgs[:30]:
            # Radio interferometry: dirty beam convolution + noise
            dirty = gaussian_filter(x, sigma=3.0)
            noisy = dirty + np.random.randn(*x.shape).astype(np.float32) * 0.05
            y = np.clip(noisy, 0, 1).astype(np.float32)
            sx.append(x); sy.append(y)
        save_mod("radio_astronomy", sx, sy,
                 "FIRST VLA 1.4GHz Survey",
                 "Becker et al., ApJ 1995",
                 "real")
        return True
    else:
        print(f"  Only got {len(imgs)} FITS cutouts (need astropy for FITS parsing)")
        return False


# ============================================================
# 2. HDR_IMAGING: Already done with BSD68 (canonical for CI)
#    But was listed as needs_canonical in v5
#    Actually in v6 it's done - skip
# ============================================================


# ============================================================
# 3. CRYO_EM: EMPIAR single-particle micrographs
# ============================================================
def build_cryo_em():
    print("=== cryo_em: EMPIAR ===")
    # EMPIAR has an API: https://www.ebi.ac.uk/empiar/api/
    # Let's find a small dataset entry
    api_url = "https://www.ebi.ac.uk/empiar/api/entry/10028/"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            print(f"  EMPIAR-10028: {data.get('title', 'N/A')[:60]}")
        else:
            print(f"  EMPIAR API HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    print("  EMPIAR micrographs are very large MRC files - skipping auto-download")
    return False


# ============================================================
# 4. CRYO_ET: SHREC 2021
# ============================================================
def build_cryo_et():
    print("=== cryo_et: SHREC 2021 ===")
    print("  SHREC 2021 data on DataverseNL - requires manual download")
    return False


# ============================================================
# 5. CT: LoDoPaB-CT - try to get just the CSV with patient IDs
#    and reconstruct from the test set (still 2.8GB)
# ============================================================
def build_ct():
    print("=== ct: LoDoPaB-CT ===")
    print("  LoDoPaB-CT files are >1.5GB each - too large for auto-download")
    print("  URL: https://zenodo.org/records/3384092")
    return False


# ============================================================
# 6. MRI: fastMRI requires registration
# ============================================================
def build_mri():
    print("=== mri: fastMRI ===")
    print("  fastMRI requires NYU data use agreement")
    print("  URL: https://fastmri.med.nyu.edu/")
    return False


# ============================================================
# 7. SAR: MSTAR requires SDMS access
# ============================================================
def build_sar():
    print("=== sar: MSTAR ===")
    print("  MSTAR requires SDMS account registration")
    # Try alternative: Zenodo SAR datasets
    # Search for SAR on Zenodo
    api_url = "https://zenodo.org/api/records?q=SAR+target+recognition&size=5&sort=bestmatch"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            hits = r.json().get('hits', {}).get('hits', [])
            for h in hits[:3]:
                title = h.get('metadata', {}).get('title', 'N/A')
                print(f"  Zenodo: {title[:60]}")
    except:
        pass
    return False


# ============================================================
# 8. LENSLESS: DiffuserCam Lensless0 from Berkeley
# ============================================================
def build_lensless():
    print("=== lensless: DiffuserCam ===")
    # Check for alternative: PhlatCam or other lensless datasets on Zenodo
    api_url = "https://zenodo.org/api/records?q=lensless+imaging&size=5&sort=bestmatch"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            hits = r.json().get('hits', {}).get('hits', [])
            for h in hits[:3]:
                title = h.get('metadata', {}).get('title', 'N/A')
                rid = h.get('id', 'N/A')
                files = h.get('files', [])
                total_size = sum(f.get('size', 0) for f in files) / 1024 / 1024
                print(f"  Zenodo {rid}: {title[:50]} ({total_size:.0f} MB)")
    except:
        pass
    print("  DiffuserCam data on Google Drive - manual download needed")
    return False


# ============================================================
# 9. CBCT: Walnut CBCT (Zenodo 2686726) - each walnut is ~5.7GB
# ============================================================
def build_cbct():
    print("=== cbct: Walnut CBCT ===")
    print("  Each Walnut zip is ~5.7GB - too large for auto-download")
    return False


# ============================================================
# 10. DIFFUSION_MRI: HCP requires DUA
# ============================================================
def build_diffusion_mri():
    print("=== diffusion_mri: HCP ===")
    print("  HCP requires data use agreement")
    return False


# ============================================================
# 11. PET: UDPET Zenodo 6361846 - only PDF available!
# ============================================================
def build_pet():
    print("=== pet: UDPET ===")
    # UDPET only has a PDF on Zenodo, actual data requires registration
    # Try alternative: QIN-PET-CT on TCIA
    print("  UDPET Zenodo only has PDF, actual PET data requires TCIA access")
    return False


# ============================================================
# 12. PET_CT: autoPET requires TCIA
# ============================================================
def build_pet_ct():
    print("=== pet_ct: autoPET ===")
    print("  autoPET requires TCIA/grand-challenge registration")
    return False


# ============================================================
# 13. STM: Try Zenodo 5799774 partial download
# ============================================================
def build_stm():
    print("=== stm: Zenodo 5799774 ===")
    print("  STM.zip is 3.5GB - too large for auto-download")
    return False


# ============================================================
# 14. SHG: PSHG-TISS on OSF
# ============================================================
def build_shg():
    print("=== shg: PSHG-TISS ===")
    # OSF: https://osf.io/K2Z8G/
    # Check OSF API for files
    api_url = "https://api.osf.io/v2/nodes/k2z8g/files/osfstorage/"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            files = data.get('data', [])
            print(f"  Found {len(files)} files/folders on OSF")
            for f in files[:5]:
                attrs = f.get('attributes', {})
                print(f"    {attrs.get('name', 'N/A')}: {attrs.get('size', 0)/1024/1024:.1f} MB")
        else:
            print(f"  OSF API HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 15. DEXA: NHANES DXA
# ============================================================
def build_dexa():
    print("=== dexa: NHANES DXA ===")
    print("  NHANES DXA data requires complex CDC data access")
    return False


# ============================================================
# 16. OCEAN_ACOUSTIC_TOMO: ACOBAR
# ============================================================
def build_ocean_acoustic_tomo():
    print("=== ocean_acoustic_tomo: ACOBAR ===")
    print("  ACOBAR data is in journal supplementary - requires manual access")
    return False


# ============================================================
# 17. TERAHERTZ: GitHub THz_Dataset
# ============================================================
def build_terahertz():
    print("=== terahertz: THz_Dataset ===")
    # Try to find image data in the GitHub repo
    api_url = "https://api.github.com/repos/LingLIx/THz_Dataset/git/trees/main?recursive=1"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            tree = r.json().get('tree', [])
            img_files = [t for t in tree if t['path'].lower().endswith(('.png', '.jpg', '.bmp', '.mat'))]
            print(f"  Found {len(img_files)} potential data files")
            for f in img_files[:10]:
                print(f"    {f['path']}")

            # Try downloading image files
            imgs = []
            for f in img_files:
                if len(imgs) >= 30: break
                if f['path'].lower().endswith(('.png', '.jpg', '.bmp')):
                    raw_url = f"https://raw.githubusercontent.com/LingLIx/THz_Dataset/main/{f['path']}"
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
                         "Active THz imaging",
                         "Ling et al.",
                         "real")
                return True
            else:
                print(f"  Only got {len(imgs)} images")
        else:
            print(f"  GitHub API HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    return False


# ============================================================
# 18. GPR: CMU-GPR
# ============================================================
def build_gpr():
    print("=== gpr: CMU-GPR ===")
    # Check GitHub repo structure
    api_url = "https://api.github.com/repos/rpl-cmu/CMU-GPR-Dataset/git/trees/main?recursive=1"
    try:
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            tree = r.json().get('tree', [])
            data_files = [t for t in tree if t['path'].lower().endswith(('.png', '.jpg', '.npy', '.mat', '.csv'))]
            print(f"  Found {len(data_files)} data files")
            for f in data_files[:10]:
                print(f"    {f['path']}: {f.get('size', 'dir')}")
        else:
            print(f"  GitHub API HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    print("  CMU-GPR data format requires ROS tools or complex parsing")
    return False


# ============================================================
# 19. RADIO_INTERFEROMETRY: ALMA archive
# ============================================================
def build_radio_interferometry():
    print("=== radio_interferometry: ALMA ===")
    print("  ALMA archive requires complex query interface")
    return False


# ============================================================
# 20. LIDAR: SemanticKITTI requires registration
# ============================================================
def build_lidar():
    print("=== lidar: SemanticKITTI ===")
    print("  SemanticKITTI requires registration at semantic-kitti.org")
    return False


# ============================================================
# 21. MAMMOGRAPHY: Already done in v6 with Zenodo 5084116
# ============================================================


# ============================================================
# 22. WIDEFIELD_LOWDOSE: Same as widefield (FMD)
# ============================================================
def build_widefield_lowdose():
    print("=== widefield_lowdose: FMD ===")
    print("  FMD data on Google Drive - requires manual download")
    return False


# ============================================================
# 23. INDUSTRIAL_CT: Same Walnut CBCT (too large)
# ============================================================
def build_industrial_ct():
    print("=== industrial_ct: Walnut CBCT ===")
    print("  Same as CBCT - too large for auto-download")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    results = {}
    for name, func in [
        ("radio_astronomy", build_radio_astronomy),
        ("terahertz", build_terahertz),
        ("shg", build_shg),
        ("cryo_em", build_cryo_em),
        ("ct", build_ct),
        ("mri", build_mri),
        ("sar", build_sar),
        ("lensless", build_lensless),
        ("cbct", build_cbct),
        ("industrial_ct", build_industrial_ct),
        ("pet", build_pet),
        ("pet_ct", build_pet_ct),
        ("fundus", lambda: False),  # Tried in batch 2
        ("diffusion_mri", build_diffusion_mri),
        ("stm", build_stm),
        ("dexa", build_dexa),
        ("ocean_acoustic_tomo", build_ocean_acoustic_tomo),
        ("gpr", build_gpr),
        ("lidar", build_lidar),
        ("radio_interferometry", build_radio_interferometry),
        ("widefield_lowdose", build_widefield_lowdose),
        ("cryo_et", build_cryo_et),
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

    success = sum(1 for v in results.values() if v)
    print(f"\nSuccessful: {success}/{len(results)}")
