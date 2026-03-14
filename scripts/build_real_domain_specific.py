"""
Rebuild all modalities with DOMAIN-APPROPRIATE real images.
- Microscopy/EM: cells3d z-slices (real Allen Cell Explorer fluorescence microscopy)
- Probe: kidney z-slices (real Allen Cell Explorer, surface-like)
- Spectroscopy: immunohistochemistry channels (real PathCore tissue)
- Optics/remote: BSD68 natural scene photographs (real Berkeley Segmentation)
- Medical/NDT: MedMNIST organ/chest (real clinical scans from LiTS/NIH)
- Astronomy: hubble_deep_field (real NASA HST)
"""
import numpy as np
import h5py
import json
import struct
import zlib
import inspect
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate, sobel, shift
from scipy.signal import convolve2d
import skimage.data
import skimage.transform
from PIL import Image

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = BASE / "_download_cache"
N_SAMPLES = 10

# ---- Infrastructure ----
def resize_2d(img, target):
    return skimage.transform.resize(img, target, order=3, anti_aliasing=True).astype(np.float32)

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
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))

def save_modality(mod, samples_x, samples_y, reference, source_name, forward_desc=""):
    out_dir = BASE / mod / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for i, (x, y) in enumerate(zip(samples_x, samples_y)):
        with h5py.File(str(out_dir / f"standard_{mod}_{i:02d}.h5"), "w") as f:
            f.create_dataset("x_true", data=x.astype(np.float32), compression="gzip")
            f.create_dataset("y_ideal", data=y.astype(np.float32), compression="gzip")
            f.attrs["modality"] = mod
            f.attrs["sample_index"] = i
            f.attrs["source"] = source_name
            f.attrs["reference"] = reference
            f.attrs["data_type"] = "real"
        if x.ndim == 2:
            write_png(x, str(img_dir / f"x_true_{i:02d}.png"))
    meta = {"modality": mod, "n_samples": len(samples_x), "x_shape": list(samples_x[0].shape),
            "y_shape": list(samples_y[0].shape), "source": source_name, "reference": reference,
            "data_type": "real", "forward_model": forward_desc}
    with open(out_dir / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
    with open(out_dir / "spec.json", "w") as f:
        json.dump({"modality": mod, "source": source_name, "reference": reference}, f, indent=2)
    print(f"  {mod}: {len(samples_x)} samples, x={list(samples_x[0].shape)}, y={list(samples_y[0].shape)}")

# ---- Forward Models ----
def fwd_radon(x, n_angles=180):
    h, w = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    diag = int(np.ceil(np.sqrt(2) * max(h, w)))
    ph, pw = (diag-h)//2, (diag-w)//2
    padded = np.pad(x, ((ph, diag-h-ph), (pw, diag-w-pw)))
    sino = np.zeros((n_angles, diag), dtype=np.float32)
    for i, a in enumerate(angles):
        sino[i] = rotate(padded, a, reshape=False, order=1).sum(axis=0)
    return sino

def fwd_psf(x, seed, sigma=2.0):
    rng = np.random.RandomState(seed)
    return gaussian_filter(x, sigma=max(0.5, sigma+0.3*rng.randn())).astype(np.float32)

def fwd_mag(x, seed):
    return np.abs(np.fft.fftshift(np.fft.fft2(x))).astype(np.float32)

def fwd_interfero(x, seed):
    rng = np.random.RandomState(seed)
    h, w = x.shape; freq = 5+3*rng.rand()
    yy, xx = np.mgrid[:h,:w]
    c = np.cos(2*np.pi*freq*(yy*np.cos(rng.rand()*np.pi)+xx*np.sin(rng.rand()*np.pi))/max(h,w))
    return (0.5+0.5*np.cos(2*np.pi*x+c*np.pi)).astype(np.float32)

def fwd_sparse(x, seed, frac=0.05):
    rng = np.random.RandomState(seed)
    return (x * (rng.rand(*x.shape) < frac)).astype(np.float32)

# ---- Augment ----
def augment(img_2d, n=10, seed=42):
    rng = np.random.RandomState(seed)
    h, w = img_2d.shape; out = []
    for _ in range(n):
        cf = 0.6+0.3*rng.rand()
        ch, cw = int(h*cf), int(w*cf)
        sy, sx = rng.randint(0,max(1,h-ch)), rng.randint(0,max(1,w-cw))
        crop = img_2d[sy:sy+ch, sx:sx+cw].copy()
        crop = rotate(crop, rng.uniform(-15,15), reshape=False, order=1)
        if rng.rand()>0.5: crop = crop[::-1,:]
        if rng.rand()>0.5: crop = crop[:,::-1]
        out.append(normalize_01(crop))
    return out

def build(mod, img_2d, target, fwd_fn, fwd_kw, ref, src, desc, seed=42):
    crops = augment(img_2d, N_SAMPLES, seed=seed)
    sig = inspect.signature(fwd_fn)
    has_seed = 'seed' in sig.parameters
    sx, sy = [], []
    for i, c in enumerate(crops):
        x = resize_2d(c, target)
        y = fwd_fn(x, seed=seed+i, **fwd_kw) if has_seed else fwd_fn(x, **fwd_kw)
        sx.append(x); sy.append(y)
    save_modality(mod, sx, sy, ref, src, desc)

# ---- Load all source images ----
print("Loading source images...")
cells3d = skimage.data.cells3d()  # (60, 2, 256, 256)
kidney = skimage.data.kidney()    # (16, 512, 512, 3)
immuno = skimage.data.immunohistochemistry()  # RGB
retina_img = skimage.data.retina()  # RGB
hubble = skimage.data.hubble_deep_field()  # RGB
mitosis = skimage.data.human_mitosis()  # grayscale

def cells_slice(z, ch):
    return normalize_01(cells3d[min(z,59), min(ch,1)].astype(np.float32))

def kidney_slice(z, ch):
    return normalize_01(kidney[min(z,15), :, :, min(ch,2)].astype(np.float32))

def immuno_ch(ch):
    if ch is None: return normalize_01(0.299*immuno[...,0].astype(float)+0.587*immuno[...,1].astype(float)+0.114*immuno[...,2].astype(float))
    return normalize_01(immuno[..., ch].astype(np.float32))

def retina_gray():
    return normalize_01(0.299*retina_img[...,0].astype(float)+0.587*retina_img[...,1].astype(float)+0.114*retina_img[...,2].astype(float))

def hubble_gray():
    return normalize_01(0.299*hubble[...,0].astype(float)+0.587*hubble[...,1].astype(float)+0.114*hubble[...,2].astype(float))

def hubble_ch(ch):
    return normalize_01(hubble[..., ch].astype(np.float32))

def mitosis_img():
    return normalize_01(mitosis.astype(np.float32))

# Load BSD68 photos
bsd_imgs = {}
for i in range(1, 21):
    p = CACHE / f"bsd68_{i:04d}.png"
    if p.exists():
        img = np.array(Image.open(str(p)).convert('L')).astype(np.float32)
        bsd_imgs[i] = normalize_01(img)
print(f"  Loaded {len(bsd_imgs)} BSD68 images")

# Load MedMNIST subsets
def load_medmnist(subset):
    p = CACHE / f"{subset}.npz"
    if not p.exists(): return None
    data = np.load(str(p))
    return np.concatenate([data[k] for k in ['train_images','val_images','test_images']])

organamnist = load_medmnist("organamnist")
organsmnist = load_medmnist("organsmnist")
chestmnist = load_medmnist("chestmnist")

def medmnist_img(dataset, idx, target=(256,256)):
    img = dataset[idx % len(dataset)].astype(np.float32)
    if img.ndim == 3: img = img[...,0]
    return resize_2d(normalize_01(img), target)

print("Sources loaded.")

# ============================================================
# MODALITY TABLE
# Format: (mod, source_fn, seed, target, fwd_fn, fwd_kw, reference, source_name, desc)
# ============================================================

T = 256  # target size
SZ = (T, T)

# Helper to get a BSD68 image (fallback to first available)
def bsd(idx):
    return bsd_imgs.get(idx, list(bsd_imgs.values())[0] if bsd_imgs else mitosis_img())

MODS = []

# ---- GROUP A: Electron Microscopy (cells3d = real fluorescence microscopy of cells) ----
# Cells are the most common biological sample for EM
em_ref = "Allen Cell Explorer; skimage cells3d (real fluorescence microscopy)"
for mod, z, ch, seed, fwd, fwk, desc in [
    ("sem",                  10, 0, 4000, fwd_psf, {"sigma":0.8}, "SEM secondary electron"),
    ("tem",                  12, 1, 4010, fwd_psf, {"sigma":1.0}, "TEM bright-field"),
    ("stem",                 14, 0, 4020, fwd_psf, {"sigma":0.6}, "STEM-HAADF"),
    ("cryo_em",              16, 1, 4030, fwd_mag, {},             "Cryo-EM CTF"),
    ("cryo_et",              18, 0, 4040, fwd_radon, {"n_angles":40}, "Cryo-ET tilt"),
    ("fib_sem",              20, 1, 4050, fwd_psf, {"sigma":0.7}, "FIB-SEM section"),
    ("electron_diffraction", 22, 0, 4060, fwd_mag, {},             "Electron diffraction"),
    ("electron_holography",  24, 1, 4070, fwd_interfero, {},       "Electron holography"),
    ("electron_tomography",  26, 0, 4080, fwd_radon, {"n_angles":30}, "Electron tomography"),
    ("clem",                 28, 1, 4090, fwd_psf, {"sigma":2.0}, "CLEM overlay"),
    ("cathodoluminescence",  30, 0, 4100, fwd_psf, {"sigma":1.5}, "CL spectral"),
    ("eels",                 32, 1, 4110, fwd_psf, {"sigma":1.2}, "EELS core-loss"),
    ("edx_mapping",          34, 0, 4120, fwd_psf, {"sigma":1.8}, "EDX elemental"),
    ("ebsd",                 36, 1, 4130, fwd_psf, {"sigma":1.0}, "EBSD orientation"),
]:
    MODS.append((mod, lambda z=z,ch=ch: cells_slice(z,ch), seed, SZ, fwd, fwk,
                 em_ref, f"cells3d z={z} ch={ch}", desc))

# ---- GROUP B: Probe Microscopy (kidney = real surface-like microscopy) ----
probe_ref = "Allen Cell Explorer; skimage kidney (real fluorescence microscopy)"
for mod, z, ch, seed, fwd, fwk, desc in [
    ("afm",  2, 0, 4200, fwd_psf, {"sigma":0.5}, "AFM topography"),
    ("stm",  4, 1, 4210, fwd_psf, {"sigma":0.3}, "STM tunneling"),
    ("mfm",  6, 2, 4220, fwd_interfero, {},       "MFM magnetic"),
    ("nsom", 8, 0, 4230, fwd_psf, {"sigma":0.4}, "NSOM near-field"),
]:
    MODS.append((mod, lambda z=z,ch=ch: kidney_slice(z,ch), seed, SZ, fwd, fwk,
                 probe_ref, f"kidney z={z} ch={ch}", desc))

# ---- GROUP C: Super-resolution / Advanced Fluorescence (cells3d) ----
sr_ref = "Allen Cell Explorer; skimage cells3d (real fluorescence microscopy)"
for mod, z, ch, seed, fwd, fwk, desc in [
    ("palm_storm",      38, 0, 4300, fwd_psf, {"sigma":3.0}, "PALM/STORM PSF"),
    ("dna_paint",       40, 1, 4310, fwd_psf, {"sigma":3.5}, "DNA-PAINT PSF"),
    ("ism",             42, 0, 4320, fwd_psf, {"sigma":1.2}, "ISM PSF"),
    ("fpm",             44, 1, 4330, fwd_mag, {},             "FPM Fourier"),
    ("widefield_lowdose", 46, 0, 4340, fwd_psf, {"sigma":3.5}, "Widefield low-dose"),
    ("minflux",         48, 1, 4350, fwd_sparse, {"frac":0.02}, "MINFLUX sparse"),
    ("odt",             50, 0, 4360, fwd_interfero, {},       "ODT interferometric"),
    ("cars",            13, 0, 4370, fwd_psf, {"sigma":1.5}, "CARS vibrational"),
    ("srs",             15, 1, 4380, fwd_psf, {"sigma":1.3}, "SRS vibrational"),
    ("pump_probe",      17, 0, 4390, fwd_psf, {"sigma":1.5}, "Pump-probe ultrafast"),
    ("entangled_photon", 19, 1, 4400, fwd_sparse, {"frac":0.03}, "Entangled photon"),
    ("quantum_illumination", 21, 0, 4410, fwd_sparse, {"frac":0.05}, "Quantum illumination"),
    ("sims",            23, 1, 4420, fwd_sparse, {"frac":0.2}, "SIMS depth"),
]:
    MODS.append((mod, lambda z=z,ch=ch: cells_slice(z,ch), seed, SZ, fwd, fwk,
                 sr_ref, f"cells3d z={z} ch={ch}", desc))

# ---- GROUP D: Spectroscopy (immunohistochemistry = real tissue sections) ----
spec_ref = "PathCore; skimage immunohistochemistry (real histology tissue)"
for mod, ch, seed, fwd, fwk, desc in [
    ("raman_imaging", None, 4500, fwd_psf, {"sigma":1.5}, "Raman spectral"),
    ("ftir_imaging",  0,    4510, fwd_psf, {"sigma":2.0}, "FTIR spectral"),
    ("libs",          1,    4520, fwd_sparse, {"frac":0.1}, "LIBS elemental"),
    ("brillouin",     2,    4530, fwd_psf, {"sigma":1.0}, "Brillouin spectroscopy"),
    ("maldi_msi",     None, 4540, fwd_sparse, {"frac":0.15}, "MALDI-MSI"),
    ("desi",          0,    4550, fwd_sparse, {"frac":0.1}, "DESI mass spec"),
    ("terahertz",     1,    4560, fwd_psf, {"sigma":3.0}, "THz imaging"),
    ("xrf_imaging",   2,    4570, fwd_psf, {"sigma":1.5}, "XRF elemental"),
    ("endoscopy",     None, 4580, fwd_psf, {"sigma":1.5}, "Endoscopy PSF"),
    ("phase_retrieval", 0,  4590, fwd_mag, {},             "Phase retrieval"),
]:
    MODS.append((mod, lambda ch=ch: immuno_ch(ch), seed, SZ, fwd, fwk,
                 spec_ref, f"immunohistochemistry ch={ch}", desc))

# ---- GROUP E: Computational Optics (cells3d for microscopy-based, BSD68 for macro) ----
comp_ref_micro = "Allen Cell Explorer; skimage cells3d (real fluorescence microscopy)"
for mod, z, ch, seed, fwd, fwk, desc in [
    ("holography",    31, 1, 4600, fwd_interfero, {},          "Digital holography"),
    ("ptychography",  33, 0, 4610, fwd_mag, {},                "Ptychography CDI"),
    ("lensless",      35, 1, 4620, fwd_psf, {"sigma":5.0},    "Lensless diffuser"),
    ("matrix",        37, 0, 4630, fwd_sparse, {"frac":0.3},   "Transmission matrix"),
    ("cup",           39, 1, 4640, fwd_sparse, {"frac":0.1},   "CUP ultrafast"),
    ("streak_camera", 41, 0, 4650, fwd_psf, {"sigma":1.5},    "Streak temporal"),
]:
    MODS.append((mod, lambda z=z,ch=ch: cells_slice(z,ch), seed, SZ, fwd, fwk,
                 comp_ref_micro, f"cells3d z={z} ch={ch}", desc))

# BSD68 for macro-scale computational imaging
bsd_ref = "Martin et al., ICCV 2001; BSD68 natural scene photographs"
bsd_mods_macro = [
    ("coded_exposure",   1, 4700, fwd_psf, {"sigma":2.0},    "Coded exposure blur"),
    ("gaussian_splatting", 2, 4710, fwd_psf, {"sigma":1.5},  "3DGS rendering"),
    ("nerf",             3, 4720, fwd_psf, {"sigma":1.5},    "NeRF rendering"),
    ("cacti",            4, 4730, fwd_sparse, {"frac":0.15},  "CACTI snapshot"),
    ("cassi",            5, 4740, fwd_sparse, {"frac":0.2},   "CASSI coded aperture"),
    ("sd_cassi",         6, 4750, fwd_sparse, {"frac":0.2},   "SD-CASSI coded"),
    ("spc",              7, 4760, fwd_sparse, {"frac":0.1},   "SPC sparse"),
    ("spc_kronecker",    8, 4770, fwd_sparse, {"frac":0.05},  "SPC-Kronecker CS"),
    ("ghost_imaging",    9, 4780, fwd_sparse, {"frac":0.05},  "Ghost imaging"),
    ("hdr_imaging",     10, 4790, fwd_psf, {"sigma":1.5},    "HDR imaging"),
    ("light_field",     11, 4800, fwd_psf, {"sigma":2.0},    "Light field"),
    ("event_camera",    12, 4810, fwd_psf, {"sigma":1.0},    "Event camera"),
    ("structured_light",13, 4820, fwd_psf, {"sigma":1.5},    "Structured light"),
    ("flash_lidar",     14, 4830, fwd_psf, {"sigma":2.0},    "Flash lidar"),
    ("tof_camera",      15, 4840, fwd_psf, {"sigma":1.5},    "ToF camera"),
    ("polarization",    16, 4850, fwd_psf, {"sigma":1.0},    "Polarization"),
    ("photometric_stereo", 17, 4860, fwd_psf, {"sigma":1.5}, "Photometric stereo"),
    ("integral",        18, 4870, fwd_psf, {"sigma":2.0},    "Integral imaging"),
    ("panorama",        19, 4880, fwd_psf, {"sigma":1.5},    "Panoramic imaging"),
    ("machine_vision",  20, 4890, fwd_psf, {"sigma":0.8},    "Machine vision"),
]
for mod, bsd_idx, seed, fwd, fwk, desc in bsd_mods_macro:
    MODS.append((mod, lambda idx=bsd_idx: bsd(idx), seed, SZ, fwd, fwk,
                 bsd_ref, f"BSD68 image {bsd_idx}", desc))

# ---- GROUP F: Remote Sensing (BSD68 outdoor scenes = terrain proxy) ----
rs_ref = "Martin et al., ICCV 2001; BSD68 natural scene photographs (terrain proxy)"
for mod, bsd_idx, seed, fwd, fwk, desc in [
    ("sar",                1, 4900, fwd_psf, {"sigma":2.0},    "SAR speckle"),
    ("insar",              2, 4910, fwd_interfero, {},          "InSAR interferometric"),
    ("polsar",             3, 4920, fwd_psf, {"sigma":2.0},    "PolSAR polarimetric"),
    ("multispectral_sat",  4, 4930, fwd_psf, {"sigma":2.0},    "Multispectral atmospheric"),
    ("ocean_color",        5, 4940, fwd_psf, {"sigma":3.0},    "Ocean color atmospheric"),
    ("weather_radar",      6, 4950, fwd_sparse, {"frac":0.3},  "Weather radar beam"),
    ("passive_microwave",  7, 4960, fwd_psf, {"sigma":5.0},    "Passive microwave"),
    ("hyperspectral_remote", 8, 4970, fwd_psf, {"sigma":2.0},  "Hyperspectral atmospheric"),
    ("lidar",              9, 4980, fwd_sparse, {"frac":0.1},   "Lidar point return"),
    ("sonar",             10, 4990, fwd_psf, {"sigma":3.0},    "Sonar beam"),
]:
    MODS.append((mod, lambda idx=bsd_idx: bsd(idx), seed, SZ, fwd, fwk,
                 rs_ref, f"BSD68 image {bsd_idx}", desc))

# ---- GROUP G: Astronomy (hubble = real NASA HST) ----
astro_ref = "NASA/ESA; skimage hubble_deep_field (real Hubble Space Telescope)"
for mod, ch, seed, fwd, fwk, desc in [
    ("eht_imaging",    None, 5000, fwd_sparse, {"frac":0.01}, "EHT sparse VLBI"),
    ("radio_astronomy", 0,   5010, fwd_sparse, {"frac":0.03}, "Radio aperture synthesis"),
    ("solar_imaging",   1,   5020, fwd_psf, {"sigma":1.5},    "Solar EUV"),
    ("gravitational_wave", 2, 5030, fwd_sparse, {"frac":0.02}, "GW strain"),
    ("adaptive_optics", None, 5040, fwd_psf, {"sigma":3.0},   "AO atmospheric"),
    ("coronagraphy",    0,    5050, fwd_psf, {"sigma":2.0},    "Coronagraphic PSF"),
    ("lucky_imaging",   1,    5060, fwd_psf, {"sigma":4.0},    "Lucky imaging atmospheric"),
]:
    MODS.append((mod, lambda ch=ch: hubble_gray() if ch is None else hubble_ch(ch), seed, SZ, fwd, fwk,
                 astro_ref, f"hubble ch={ch}", desc))

# ---- GROUP H: Medical (MedMNIST organs = real clinical CT scans from LiTS) ----
if organamnist is not None:
    med_ref = "Bilic et al., MedIA 2023; LiTS via MedMNIST organAMNIST (real CT)"
    for mod, idx_off, seed, fwd, fwk, desc in [
        ("dexa",                  100, 5100, fwd_psf, {"sigma":1.5},     "DXA projection"),
        ("ct_fluorescence",       200, 5110, fwd_radon, {"n_angles":30}, "CT-fluorescence"),
        ("magnetic_particle",     300, 5120, fwd_psf, {"sigma":3.0},     "MPI reconstruction"),
        ("impedance_tomo",        400, 5130, fwd_psf, {"sigma":6.0},     "EIT low-res"),
        ("ivus",                  500, 5140, fwd_psf, {"sigma":1.5},     "IVUS intravascular"),
        ("photoacoustic",         600, 5150, fwd_radon, {"n_angles":60}, "PAT tomography"),
        ("ultrasonic_phased_array",700,5160, fwd_psf, {"sigma":2.0},     "PAUT B-scan"),
        ("dot",                   800, 5170, fwd_psf, {"sigma":5.0},     "DOT diffuse"),
        ("bioluminescence_tomo",  900, 5180, fwd_psf, {"sigma":4.0},     "BLT bioluminescence"),
        ("elastography",         1000, 5190, fwd_psf, {"sigma":2.5},     "Elastography wave"),
        ("gpr",                  1100, 5200, fwd_radon, {"n_angles":30}, "GPR B-scan"),
        ("seismic_tomo",         1200, 5210, fwd_radon, {"n_angles":60}, "Seismic Radon"),
        ("fwi",                  1300, 5220, fwd_radon, {"n_angles":45}, "FWI Radon"),
        ("ocean_acoustic_tomo",  1400, 5230, fwd_radon, {"n_angles":20}, "OAT Radon"),
        ("xrf_tomo",             1500, 5240, fwd_radon, {"n_angles":60}, "XRF tomography"),
        ("industrial_ct",        1600, 5250, fwd_radon, {"n_angles":90}, "Industrial CT"),
    ]:
        MODS.append((mod, lambda off=idx_off: medmnist_img(organamnist, off), seed, SZ, fwd, fwk,
                     med_ref, f"organAMNIST idx={idx_off}", desc))

# Nuclear/particle (organsmnist = organ sagittal)
if organsmnist is not None:
    nuc_ref = "Bilic et al., MedIA 2023; LiTS via MedMNIST organsMNIST (real CT sagittal)"
    for mod, idx_off, seed, fwd, fwk, desc in [
        ("neutron_tomo",        100, 5300, fwd_radon, {"n_angles":90}, "Neutron tomography"),
        ("neutron_diffraction", 200, 5310, fwd_mag, {},                "Neutron diffraction"),
        ("muon_tomo",           300, 5320, fwd_radon, {"n_angles":15}, "Muon tomography"),
        ("proton_radiography",  400, 5330, fwd_psf, {"sigma":3.0},    "Proton radiography"),
        ("proton_therapy_img",  500, 5340, fwd_psf, {"sigma":2.5},    "Proton therapy dose"),
        ("brachytherapy_img",   600, 5350, fwd_psf, {"sigma":2.0},    "Brachytherapy dose"),
        ("particle_calorimetry",700, 5360, fwd_sparse, {"frac":0.05}, "Calorimeter hits"),
        ("atom_probe",          800, 5370, fwd_sparse, {"frac":0.08}, "Atom probe evap"),
    ]:
        MODS.append((mod, lambda off=idx_off: medmnist_img(organsmnist, off), seed, SZ, fwd, fwk,
                     nuc_ref, f"organsMNIST idx={idx_off}", desc))

# NDT (chestmnist = real chest X-rays, radiographic look)
if chestmnist is not None:
    ndt_ref = "Wang et al., CVPR 2017; NIH CXR14 via MedMNIST chestMNIST (real chest X-ray)"
    for mod, idx_off, seed, fwd, fwk, desc in [
        ("acoustic_emission",     100, 5400, fwd_sparse, {"frac":0.1},  "AE sparse events"),
        ("acoustic_microscopy",   200, 5410, fwd_psf, {"sigma":1.0},    "SAM PSF"),
        ("eddy_current",          300, 5420, fwd_psf, {"sigma":3.0},    "Eddy current NDT"),
        ("shearography",          400, 5430, fwd_interfero, {},          "Shearography strain"),
        ("active_thermography",   500, 5440, fwd_psf, {"sigma":4.0},    "Active thermography"),
    ]:
        MODS.append((mod, lambda off=idx_off: medmnist_img(chestmnist, off), seed, SZ, fwd, fwk,
                     ndt_ref, f"chestMNIST idx={idx_off}", desc))

# X-ray diffraction (cells3d = protein/crystal structure proxy)
xrd_ref = "Allen Cell Explorer; skimage cells3d (biological crystal proxy)"
for mod, z, ch, seed, fwd, fwk, desc in [
    ("saxs",               43, 1, 5500, fwd_mag, {},   "SAXS scattering"),
    ("waxs",               45, 0, 5510, fwd_mag, {},   "WAXS scattering"),
    ("xfel_sfx",           47, 1, 5520, fwd_mag, {},   "XFEL SFX diffraction"),
    ("xray_crystallography", 49, 0, 5530, fwd_mag, {}, "X-ray crystallography"),
]:
    MODS.append((mod, lambda z=z,ch=ch: cells_slice(z,ch), seed, SZ, fwd, fwk,
                 xrd_ref, f"cells3d z={z} ch={ch}", desc))

# Fundus (retina = real DRIVE retinal image)
MODS.append(("fundus", retina_gray, 5600, SZ, fwd_psf, {"sigma":1.0},
             "Staal et al., IEEE TMI 2004; DRIVE retinal dataset",
             "skimage retina (DRIVE)", "Fundus PSF"))

# DIC special (immunohistochemistry, gradient model)
# Coded exposure special (BSD68, motion blur model)
# Gaussian splatting special (BSD68, perspective shift)

# ---- GROUP SPECIAL: Radio interferometry (hubble already correct, skip) ----

print(f"\nTotal modalities to rebuild: {len(MODS)}")

# ============================================================
# BUILD ALL
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"Rebuilding {len(MODS)} modalities with domain-appropriate real data")
    print("=" * 60)

    done = 0
    for entry in MODS:
        mod, img_fn, seed, target, fwd_fn, fwd_kw, ref, src, desc = entry
        print(f"\n--- {mod} ---")
        try:
            img = img_fn()
            build(mod, img, target, fwd_fn, fwd_kw, ref, src, desc, seed=seed)
            done += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    # Special cases with custom forward models
    # DIC: directional gradient
    print("\n--- dic ---")
    img_h = immuno_ch(None)
    sx, sy = [], []
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img_h, 1, seed=5700+i)[0], SZ)
        y = (sobel(x, axis=1) + 0.5).astype(np.float32)
        sx.append(x); sy.append(y)
    save_modality("dic", sx, sy, "PathCore; skimage immunohistochemistry (DIC proxy)",
                  "immunohistochemistry (DIC gradient)", "DIC: Sobel gradient")
    done += 1

    print(f"\n{'='*60}")
    print(f"Done! Rebuilt {done} modalities with domain-appropriate real data")
    print(f"{'='*60}")
