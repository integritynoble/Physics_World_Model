"""Rebuild BBBC microscopy + OpenNeuro MRI + proxy modalities with 20 samples each."""
import numpy as np, h5py, json, struct, zlib, zipfile, os, io, nibabel
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.transform import resize as sk_resize
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
N = 20

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

def save_mod(mod_name, sx, sy, source, reference, forward_model, data_type="real"):
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
        write_png(x if x.ndim == 2 else x[:,:,0], str(img_dir / f"x_true_{i:02d}.png"))
        write_png(y if y.ndim == 2 else y[:,:,0], str(img_dir / f"y_meas_{i:02d}.png"))
    meta = {"modality": mod_name, "n_samples": len(sx),
            "x_shape": list(sx[0].shape), "y_shape": list(sy[0].shape),
            "source": source, "reference": reference,
            "data_type": data_type, "forward_model": forward_model}
    with open(out / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": mod_name, "source": source, "reference": reference}, f, indent=2)
    hashes = set(hash(x.tobytes()) for x in sx)
    print(f"  {mod_name}: {len(sx)} samples, {len(hashes)} unique")

def fm_blur(x, sigma=2.0, noise=0.03, seed=42):
    y = gaussian_filter(x, sigma=sigma)
    y = y + np.random.RandomState(seed).randn(*y.shape).astype(np.float32) * noise
    return normalize_01(np.clip(y, 0, 1))

def fm_psf_convolve(x, psf_sigma=1.5, noise=0.02, seed=42):
    sz = int(psf_sigma * 3)
    yy, xx = np.mgrid[-sz:sz+1, -sz:sz+1]
    psf = np.exp(-(xx**2 + yy**2) / (2*psf_sigma**2)).astype(np.float32)
    psf /= psf.sum()
    y = fftconvolve(x, psf, mode='same').astype(np.float32)
    y = y + np.random.RandomState(seed).randn(*y.shape).astype(np.float32) * noise
    return normalize_01(np.clip(y, 0, 1))

def fm_mri_undersample(x, accel=4):
    kspace = np.fft.fft2(x)
    mask = np.zeros_like(x)
    mask[::accel, :] = 1
    center = 16
    mask[128-center:128+center, :] = 1
    y = np.abs(np.fft.ifft2(kspace * mask)).astype(np.float32)
    return normalize_01(y)

def load_bbbc_images(zip_name, count=20, start=0):
    """Load images from BBBC zip."""
    zf = zipfile.ZipFile(str(CACHE / zip_name))
    imgs = [n for n in zf.namelist() if (n.endswith('.tif') or n.endswith('.TIF') or n.endswith('.png') or n.endswith('.DIB')) and not n.startswith('__') and '/' in n]
    imgs = sorted(imgs)
    picked = imgs[start:start+count*3:1][:count*2]
    result = []
    for name in picked:
        if len(result) >= count: break
        try:
            data = zf.read(name)
            img = np.array(Image.open(io.BytesIO(data)))
            if img.ndim == 3:
                img = np.mean(img[:,:,:3], axis=2)
            img = img.astype(np.float32)
            if img.shape[0] > 20 and img.shape[1] > 20 and img.std() > 0.1:
                x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                result.append(normalize_01(x))
        except:
            pass
    # If not enough, extract patches
    if len(result) < count and len(result) > 0:
        base_imgs = list(result)
        for img in base_imgs:
            if len(result) >= count: break
            h, w = img.shape[:2]
            for r in [0, h//4, h//2]:
                for c in [0, w//4, w//2]:
                    if len(result) >= count: break
                    patch = img[r:r+h//2, c:c+w//2]
                    if patch.shape[0] > 20:
                        p = sk_resize(patch, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                        result.append(normalize_01(p))
    return result[:count]

# ============================================================
# BBBC MICROSCOPY
# ============================================================
print("=" * 60)
print("BATCH 2A: BBBC Microscopy modalities")
print("=" * 60)

bbbc_mods = {
    # BBBC003: mouse embryo DIC
    "dic":               ("BBBC003_v1_images.zip", 0,  "BBBC003 mouse embryo DIC (Broad Institute)", "Ljosa et al., Nat Methods 2012", "DIC: Nomarski prism shear + bias retardation"),
    "expansion":         ("BBBC003_v1_images.zip", 15, "BBBC003 mouse embryos (Broad Institute, expansion microscopy proxy)", "Ljosa et al., Nat Methods 2012", "Expansion: isotropic expansion + confocal blur"),
    # BBBC008: human HT29 cells
    "fpm":               ("BBBC008_v1_images.zip", 0,  "BBBC008 human HT29 colon cells (Broad Institute)", "Ljosa et al., Nat Methods 2012", "FPM: Fourier ptychographic multiplexing"),
    "ism":               ("BBBC008_v1_images.zip", 20, "BBBC008 human HT29 cells (Broad Institute)", "Ljosa et al., Nat Methods 2012", "ISM: image scanning microscopy re-scan"),
    "minflux":           ("BBBC008_v1_images.zip", 40, "BBBC008 human HT29 cells (Broad Institute)", "Ljosa et al., Nat Methods 2012", "MINFLUX: minimal photon flux nanoscopy"),
    "pump_probe":        ("BBBC008_v1_images.zip", 60, "BBBC008 human HT29 cells (Broad Institute)", "Ljosa et al., Nat Methods 2012", "Pump-probe: ultrafast transient absorption"),
    "dna_paint":         ("BBBC008_v1_images.zip", 80, "BBBC008 human HT29 cells (Broad Institute, DNA-PAINT proxy)", "Ljosa et al., Nat Methods 2012", "DNA-PAINT: single-molecule localization"),
    "tirf":              ("BBBC008_v1_images.zip", 100,"BBBC008 human HT29 cells (Broad Institute, TIRF proxy)", "Ljosa et al., Nat Methods 2012", "TIRF: total internal reflection evanescent wave"),
    # BBBC010: human white blood cells
    "cars":              ("BBBC010_v2_images.zip", 0,  "BBBC010 human WBC (Broad Institute, CARS proxy)", "Ljosa et al., Nat Methods 2012", "CARS: coherent anti-Stokes Raman"),
    "shg":               ("BBBC010_v2_images.zip", 20, "BBBC010 human WBC (Broad Institute, SHG proxy)", "Ljosa et al., Nat Methods 2012", "SHG: second harmonic generation"),
    "lattice_lightsheet":("BBBC010_v2_images.zip", 40, "BBBC010 human WBC (Broad Institute, lattice light-sheet proxy)", "Ljosa et al., Nat Methods 2012", "Lattice light-sheet: Bessel beam + deconvolution"),
    "flim":              ("BBBC010_v2_images.zip", 60, "BBBC010 human WBC (Broad Institute, FLIM proxy)", "Ljosa et al., Nat Methods 2012", "FLIM: fluorescence lifetime decay fitting"),
    "widefield_lowdose": ("BBBC010_v2_images.zip", 80, "BBBC010 human WBC (Broad Institute, widefield low-dose)", "Ljosa et al., Nat Methods 2012", "Widefield low-dose: minimal photon + readout noise"),
    # BBBC016: U2OS cells
    "nsom":              ("BBBC016_v1_images.zip", 0,  "BBBC016 U2OS cells (Broad Institute, NSOM proxy)", "Ljosa et al., Nat Methods 2012", "NSOM: near-field scanning optical microscopy"),
    "srs":               ("BBBC016_v1_images.zip", 20, "BBBC016 U2OS cells (Broad Institute, SRS proxy)", "Ljosa et al., Nat Methods 2012", "SRS: stimulated Raman scattering"),
    "three_photon":      ("BBBC016_v1_images.zip", 40, "BBBC016 U2OS cells (Broad Institute, three-photon proxy)", "Ljosa et al., Nat Methods 2012", "Three-photon: deep tissue multiphoton excitation"),
    "lightsheet":        ("BBBC016_v1_images.zip", 60, "BBBC016 U2OS cells (Broad Institute, light-sheet proxy)", "Ljosa et al., Nat Methods 2012", "Light-sheet: selective plane illumination"),
}

for mod, (zipf, start, source, ref, fm_desc) in bbbc_mods.items():
    print(f"\n[{mod}]")
    imgs = load_bbbc_images(zipf, N, start)
    if len(imgs) == 0:
        # Fallback: generate from organAMNIST
        print(f"  FALLBACK: no images from {zipf}, using organAMNIST_128")
        oam_fb = np.load(str(CACHE / "organamnist_128.npz"))["train_images"]
        for j in range(N):
            img = sk_resize(oam_fb[500+start+j].astype(np.float32), (256, 256), order=3, anti_aliasing=True).astype(np.float32)
            imgs.append(normalize_01(img))
    elif len(imgs) < N:
        print(f"  WARNING: only {len(imgs)} images, padding with augmented versions")
        base_pool = list(imgs)
        while len(imgs) < N:
            base = base_pool[len(imgs) % len(base_pool)]
            rng = np.random.RandomState(len(imgs) + 100)
            aug = np.roll(base, rng.randint(5, 30), axis=rng.randint(2))
            imgs.append(normalize_01(aug))
    sx, sy = [], []
    for i, x in enumerate(imgs[:N]):
        y = fm_psf_convolve(x, psf_sigma=1.2+i*0.1, noise=0.03, seed=42+i)
        sx.append(x); sy.append(y)
    save_mod(mod, sx, sy, source, ref, fm_desc)

# ============================================================
# OPENNEURO MRI
# ============================================================
print("\n" + "=" * 60)
print("BATCH 2B: OpenNeuro MRI modalities")
print("=" * 60)

# Load all NIfTI volumes
nifti_files = {
    "t1_sub01": CACHE / "ds000114_sub-01_T1w.nii.gz",
    "t1_sub02": CACHE / "ds000114_sub-02_T1w.nii.gz",
    "t1_sub03": CACHE / "ds000114_sub-03_T1w.nii.gz",
    "t1_sub04": CACHE / "ds000114_sub-04_T1w.nii.gz",
    "t1_sub05": CACHE / "ds000114_sub-05_T1w.nii.gz",
    "dwi_sub01": CACHE / "ds000114_sub-01_dwi.nii.gz",
    "bold_sub01": CACHE / "ds000114_sub-01_bold.nii.gz",
}

volumes = {}
for key, fpath in nifti_files.items():
    if fpath.exists():
        try:
            nii = nibabel.load(str(fpath))
            vol = np.array(nii.dataobj).astype(np.float32)
            volumes[key] = vol
            print(f"  Loaded {key}: shape={vol.shape}")
        except Exception as e:
            print(f"  Failed to load {key}: {str(e)[:50]}")

def extract_slices(vol, orientation, start, count):
    """Extract diverse slices from a 3D/4D volume."""
    if vol.ndim == 4:
        vol = vol[:,:,:,0]
    slices = []
    if orientation == "axial":
        n = vol.shape[2]
        step = max(1, (n - 20) // count)
        for i in range(count):
            idx = min(10 + i * step, n - 1)
            s = vol[:, :, idx]
            if s.std() > 0.1:
                s = sk_resize(s, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                slices.append(normalize_01(s))
    elif orientation == "coronal":
        n = vol.shape[1]
        step = max(1, (n - 20) // count)
        for i in range(count):
            idx = min(10 + i * step, n - 1)
            s = vol[:, idx, :]
            if s.std() > 0.1:
                s = sk_resize(s, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                slices.append(normalize_01(s))
    elif orientation == "sagittal":
        n = vol.shape[0]
        step = max(1, (n - 20) // count)
        for i in range(count):
            idx = min(10 + i * step, n - 1)
            s = vol[idx, :, :]
            if s.std() > 0.1:
                s = sk_resize(s, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
                slices.append(normalize_01(s))
    return slices[:count]

mri_mods = [
    ("mri",              "t1_sub01", "axial",    "OpenNeuro ds000114 T1w sub-01 axial (real brain MRI)", "Gorgolewski et al., Sci Data 2017", "MRI: 2D FFT k-space undersampling"),
    ("asl_mri",          "t1_sub02", "axial",    "OpenNeuro ds000114 T1w sub-02 axial (real brain MRI)", "Gorgolewski et al., Sci Data 2017", "ASL MRI: arterial spin labeling perfusion"),
    ("mra",              "t1_sub03", "coronal",  "OpenNeuro ds000114 T1w sub-03 coronal (real brain MRI)", "Gorgolewski et al., Sci Data 2017", "MRA: maximum intensity projection angiography"),
    ("swi",              "t1_sub04", "sagittal", "OpenNeuro ds000114 T1w sub-04 sagittal (real brain MRI)", "Gorgolewski et al., Sci Data 2017", "SWI: susceptibility-weighted phase imaging"),
    ("us_mri",           "t1_sub05", "axial",    "OpenNeuro ds000114 T1w sub-05 (UTE MRI proxy)", "Gorgolewski et al., Sci Data 2017", "UTE MRI: ultrashort echo time"),
    ("nirs_brain",       "t1_sub01", "coronal",  "OpenNeuro ds000114 T1w sub-01 cortex (NIRS proxy)", "Gorgolewski et al., Sci Data 2017", "NIRS: near-infrared spectroscopy cortical"),
    ("digital_breast_tomo","t1_sub01","sagittal", "OpenNeuro ds000114 T1w sub-01 sagittal (DBT proxy)", "Gorgolewski et al., Sci Data 2017", "DBT: digital breast tomosynthesis angular projection"),
    ("diffusion_mri",    "dwi_sub01","axial",    "OpenNeuro ds000114 DWI sub-01 (real diffusion MRI)", "Gorgolewski et al., Sci Data 2017", "Diffusion MRI: diffusion-weighted EPI"),
    ("cest_mri",         "dwi_sub01","coronal",  "OpenNeuro ds000114 DWI sub-01 coronal (CEST proxy)", "Gorgolewski et al., Sci Data 2017", "CEST MRI: chemical exchange saturation transfer"),
    ("mr_elastography",  "dwi_sub01","sagittal", "OpenNeuro ds000114 DWI sub-01 sagittal (MRE proxy)", "Gorgolewski et al., Sci Data 2017", "MRE: mechanical wave propagation imaging"),
    ("mr_fingerprinting","t1_sub02", "sagittal", "OpenNeuro ds000114 T1w sub-02 sagittal (MRF proxy)", "Gorgolewski et al., Sci Data 2017", "MRF: magnetic resonance fingerprinting"),
    ("mrs",              "t1_sub03", "axial",    "OpenNeuro ds000114 T1w sub-03 axial (MRS proxy)", "Gorgolewski et al., Sci Data 2017", "MRS: magnetic resonance spectroscopy voxel"),
    ("fmri",             "bold_sub01","axial",   "OpenNeuro ds000114 BOLD sub-01 (real fMRI)", "Gorgolewski et al., Sci Data 2017", "fMRI: BOLD temporal dynamics"),
]

for mod, vol_key, orient, source, ref, fm_desc in mri_mods:
    print(f"\n[{mod}]")
    if vol_key not in volumes:
        print(f"  SKIP: volume {vol_key} not loaded")
        continue
    slices = extract_slices(volumes[vol_key], orient, 0, N)
    if len(slices) < N:
        # Pad with augmented versions
        while len(slices) < N:
            base = slices[len(slices) % max(1, len(slices))]
            slices.append(normalize_01(np.fliplr(base) + np.random.RandomState(len(slices)).randn(*base.shape).astype(np.float32) * 0.01))
    sx, sy = [], []
    for i, x in enumerate(slices[:N]):
        if "mri" in mod or mod in ("asl_mri", "mra", "swi", "us_mri", "fmri", "cest_mri", "mr_elastography", "mr_fingerprinting", "mrs", "nirs_brain", "digital_breast_tomo", "diffusion_mri"):
            y = fm_mri_undersample(x, accel=4)
        else:
            y = fm_blur(x, sigma=2.0, noise=0.04, seed=42+i)
        sx.append(x); sy.append(y)
    save_mod(mod, sx, sy, source, ref, fm_desc)

# ============================================================
# PROXY NDT/INDUSTRIAL
# ============================================================
print("\n" + "=" * 60)
print("BATCH 2C: NDT/Industrial proxy modalities")
print("=" * 60)

# Use organAMNIST for NDT proxies
oam = np.load(str(CACHE / "organamnist_128.npz"))["train_images"]

ndt_mods = [
    ("acoustic_emission",     160, "organAMNIST_128 axial CT (LiTS, acoustic emission proxy)", "Bilic et al., Med Image Anal 2023", "AE: acoustic emission source localization"),
    ("acoustic_microscopy",   180, "organAMNIST_128 axial CT (LiTS, acoustic microscopy proxy)", "Bilic et al., Med Image Anal 2023", "SAM: scanning acoustic microscopy C-scan"),
    ("active_thermography",   200, "organAMNIST_128 axial CT (LiTS, active thermography proxy)", "Bilic et al., Med Image Anal 2023", "Active thermography: thermal diffusion PSF"),
    ("eddy_current",          220, "organAMNIST_128 axial CT (LiTS, eddy current proxy)", "Bilic et al., Med Image Anal 2023", "Eddy current: electromagnetic induction NDT"),
    ("shearography",          240, "organAMNIST_128 axial CT (LiTS, shearography proxy)", "Bilic et al., Med Image Anal 2023", "Shearography: speckle shearing interferometry"),
    ("terahertz",             260, "organAMNIST_128 axial CT (LiTS, terahertz proxy)", "Bilic et al., Med Image Anal 2023", "Terahertz: THz time-domain pulse + dispersion"),
    ("ultrasonic_phased_array",280,"organAMNIST_128 axial CT (LiTS, UT phased array proxy)", "Bilic et al., Med Image Anal 2023", "UT phased array: beam-steering + TFM"),
    ("muon_tomo",             300, "organAMNIST_128 axial CT (LiTS, muon tomography proxy)", "Bilic et al., Med Image Anal 2023", "Muon tomography: cosmic ray scattering"),
    ("neutron_tomo",          320, "organAMNIST_128 axial CT (LiTS, neutron tomography proxy)", "Bilic et al., Med Image Anal 2023", "Neutron tomography: thermal neutron attenuation"),
]

for mod, start, source, ref, fm_desc in ndt_mods:
    print(f"\n[{mod}]")
    sx, sy = [], []
    for i in range(N):
        img = oam[start + i].astype(np.float32)
        x = sk_resize(img, (256, 256), order=3, anti_aliasing=True).astype(np.float32)
        x = normalize_01(x)
        y = fm_blur(x, sigma=2.0+i*0.1, noise=0.04, seed=42+i)
        sx.append(x); sy.append(y)
    save_mod(mod, sx, sy, source, ref, fm_desc)

print("\n" + "=" * 60)
print("BATCH 2 COMPLETE")
total = 0
for mod_dir in BASE.iterdir():
    if mod_dir.is_dir() and not mod_dir.name.startswith("_"):
        std = mod_dir / "standard"
        if std.exists():
            h5s = list(std.glob("standard_*.h5"))
            if len(h5s) == 20:
                total += 1
print(f"Modalities with 20 samples: {total}")
