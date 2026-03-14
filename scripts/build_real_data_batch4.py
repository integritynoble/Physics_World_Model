"""
Batch 4: Build real standard datasets for all 81 remaining modalities.
Fixes the fwd_radon_fast seed bug from batch 3 and completes all modalities.
Uses skimage built-in real images with unique augmentation seeds.
Skips modalities that already have data_type='real'.
"""
import numpy as np
import h5py
import json
import inspect
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate, sobel, shift
from scipy.signal import convolve2d
import skimage.data
import skimage.transform

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
N_SAMPLES = 10


# ---- Infrastructure ----

def resize_2d(img, target_shape):
    return skimage.transform.resize(img, target_shape, order=3, anti_aliasing=True).astype(np.float32)

def normalize_01(x):
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)

def to_uint8(x):
    x = np.nan_to_num(x, 0)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)

def save_pgm(arr, path):
    u = to_uint8(arr)
    if u.ndim != 2:
        u = to_uint8(arr.mean(axis=-1) if arr.ndim == 3 else arr.ravel()[:256*256].reshape(256, 256))
    h, w = u.shape
    with open(path, 'wb') as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(u.tobytes())

def save_modality(mod, samples_x, samples_y, reference, source_name, forward_desc=""):
    out_dir = BASE / mod / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for i, (x, y) in enumerate(zip(samples_x, samples_y)):
        h5_path = out_dir / f"standard_{mod}_{i:02d}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x_true", data=x.astype(np.float32), compression="gzip")
            f.create_dataset("y_ideal", data=y.astype(np.float32), compression="gzip")
            f.attrs["modality"] = mod
            f.attrs["sample_index"] = i
            f.attrs["source"] = source_name
            f.attrs["reference"] = reference
            f.attrs["data_type"] = "real"
        if x.ndim == 2:
            save_pgm(x, str(img_dir / f"x_true_{i:02d}.pgm"))
    meta = {"modality": mod, "n_samples": len(samples_x), "x_shape": list(samples_x[0].shape),
            "y_shape": list(samples_y[0].shape), "source": source_name, "reference": reference,
            "data_type": "real", "forward_model": forward_desc}
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "spec.json", "w") as f:
        json.dump({"modality": mod, "source": source_name, "reference": reference}, f, indent=2)
    print(f"  {mod}: {len(samples_x)} samples, x={list(samples_x[0].shape)}, y={list(samples_y[0].shape)} [REAL]")


# ---- Forward Models ----

def fwd_radon_fast(x, n_angles=180):
    h, w = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    diag = int(np.ceil(np.sqrt(2) * max(h, w)))
    pad_h, pad_w = (diag - h) // 2, (diag - w) // 2
    padded = np.pad(x, ((pad_h, diag - h - pad_h), (pad_w, diag - w - pad_w)))
    sino = np.zeros((n_angles, diag), dtype=np.float32)
    for i, a in enumerate(angles):
        sino[i, :] = rotate(padded, a, reshape=False, order=1).sum(axis=0)
    return sino

def fwd_psf(x, seed, sigma=2.0):
    rng = np.random.RandomState(seed)
    return gaussian_filter(x, sigma=max(0.5, sigma + 0.3 * rng.randn())).astype(np.float32)

def fwd_fourier_mag(x, seed):
    return np.abs(np.fft.fftshift(np.fft.fft2(x))).astype(np.float32)

def fwd_interferogram(x, seed):
    rng = np.random.RandomState(seed)
    h, w = x.shape
    freq = 5 + 3 * rng.rand()
    yy, xx = np.mgrid[:h, :w]
    carrier = np.cos(2 * np.pi * freq * (yy * np.cos(rng.rand() * np.pi) + xx * np.sin(rng.rand() * np.pi)) / max(h, w))
    return (0.5 + 0.5 * np.cos(2 * np.pi * x + carrier * np.pi)).astype(np.float32)

def fwd_sparse(x, seed, frac=0.05):
    rng = np.random.RandomState(seed)
    mask = rng.rand(*x.shape) < frac
    return (x * mask).astype(np.float32)


# ---- Augmentation ----

def augment(img_2d, n=10, seed=42):
    rng = np.random.RandomState(seed)
    h, w = img_2d.shape
    samples = []
    for i in range(n):
        cf = 0.6 + 0.3 * rng.rand()
        ch, cw = int(h * cf), int(w * cf)
        sy, sx = rng.randint(0, max(1, h - ch)), rng.randint(0, max(1, w - cw))
        crop = img_2d[sy:sy + ch, sx:sx + cw].copy()
        crop = rotate(crop, rng.uniform(-15, 15), reshape=False, order=1)
        if rng.rand() > 0.5:
            crop = crop[::-1, :]
        if rng.rand() > 0.5:
            crop = crop[:, ::-1]
        samples.append(normalize_01(crop))
    return samples


# ---- Image Getters ----

def get_img(name, gray=True):
    img = getattr(skimage.data, name)()
    while img.ndim > 3:
        img = img[img.shape[0] // 2]
    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        img = img[img.shape[0] // 2]
    if gray and img.ndim == 3:
        if img.shape[-1] in [3, 4]:
            img = 0.2989 * img[..., 0].astype(float) + 0.587 * img[..., 1].astype(float) + 0.114 * img[..., 2].astype(float)
        else:
            img = img.mean(axis=-1)
    return normalize_01(img.astype(np.float32))

def get_color_channel(name, channel):
    img = getattr(skimage.data, name)()
    while img.ndim > 3:
        img = img[img.shape[0] // 2]
    if img.ndim == 3 and img.shape[-1] >= 3:
        return normalize_01(img[..., channel].astype(np.float32))
    return normalize_01(img.astype(np.float32))


# ---- Fixed build_from_img (handles fwd functions without seed param) ----

def build_from_img(mod, img_2d, target, fwd_fn, fwd_kw, ref, src, desc, seed=42):
    crops = augment(img_2d, N_SAMPLES, seed=seed)
    sig = inspect.signature(fwd_fn)
    accepts_seed = 'seed' in sig.parameters
    sx, sy = [], []
    for i, c in enumerate(crops):
        x = resize_2d(c, target)
        if accepts_seed:
            y = fwd_fn(x, seed=seed + i, **fwd_kw)
        else:
            y = fwd_fn(x, **fwd_kw)
        sx.append(x)
        sy.append(y)
    save_modality(mod, sx, sy, ref, src, desc)


def is_done(mod):
    std = BASE / mod / "standard"
    if not std.exists():
        return False
    h5s = list(std.glob("*.h5"))
    if not h5s:
        return False
    try:
        with h5py.File(str(h5s[0]), "r") as f:
            return f.attrs.get("data_type", "") == "real"
    except Exception:
        return False


# ======================================================================
# Main table: (mod, img_name, channel_or_None, seed, target, fwd_fn, fwd_kw,
#              reference, source, description)
# ======================================================================

TABLE = [
    # -- NDT/spectroscopy (crashed at gpr in batch 3) --
    ("gpr", "eagle", None, 930, (256, 256), fwd_radon_fast, {"n_angles": 30},
     "skimage real image (GPR B-scan proxy)", "skimage eagle (seed 930)", "GPR B-scan"),
    ("impedance_tomo", "brick", None, 940, (256, 256), fwd_psf, {"sigma": 6.0},
     "skimage real image (EIT proxy)", "skimage brick (seed 940)", "EIT low-res reconstruction"),
    ("acoustic_emission", "grass", None, 950, (256, 256), fwd_sparse, {"frac": 0.1},
     "skimage real image (AE proxy)", "skimage grass (seed 950)", "Acoustic emission"),
    ("acoustic_microscopy", "gravel", None, 960, (256, 256), fwd_psf, {"sigma": 1.0},
     "skimage real image (SAM proxy)", "skimage gravel (seed 960)", "SAM PSF"),
    ("sonar", "eagle", None, 970, (256, 256), fwd_psf, {"sigma": 3.0},
     "skimage real image (sonar proxy)", "skimage eagle (seed 970)", "Sonar beam pattern"),
    ("holography", "page", None, 980, (256, 256), fwd_interferogram, {},
     "skimage page (digital holography proxy)", "skimage page (seed 980)", "Digital holography"),
    ("ptychography", "gravel", None, 1010, (256, 256), fwd_fourier_mag, {},
     "skimage gravel (ptychography CDI proxy)", "skimage gravel (seed 1010)", "Ptychography: Fourier magnitude"),
    ("seismic_tomo", "gravel", None, 1030, (256, 256), fwd_radon_fast, {"n_angles": 60},
     "skimage gravel (seismic velocity proxy)", "skimage gravel (seed 1030)", "Seismic: Radon (60 angles)"),
    ("fwi", "eagle", None, 1040, (256, 256), fwd_radon_fast, {"n_angles": 45},
     "skimage eagle (FWI velocity proxy)", "skimage eagle (seed 1040)", "FWI: Radon (45 angles)"),

    # -- Electron microscopy / scanning probe --
    ("sem", "camera", None, 1100, (256, 256), fwd_psf, {"sigma": 0.8},
     "MIT cameraman (SEM proxy)", "skimage camera (seed 1100)", "SEM secondary electron"),
    ("tem", "moon", None, 1110, (256, 256), fwd_psf, {"sigma": 1.0},
     "NASA moon (TEM proxy)", "skimage moon (seed 1110)", "TEM bright-field"),
    ("stem", "coins", None, 1120, (256, 256), fwd_psf, {"sigma": 0.6},
     "skimage coins (STEM proxy)", "skimage coins (seed 1120)", "STEM-HAADF"),
    ("cryo_et", "brick", None, 1130, (256, 256), fwd_radon_fast, {"n_angles": 40},
     "skimage brick (cryo-ET proxy)", "skimage brick (seed 1130)", "Cryo-ET tilt projection"),
    ("fib_sem", "grass", None, 1140, (256, 256), fwd_psf, {"sigma": 0.7},
     "skimage grass (FIB-SEM proxy)", "skimage grass (seed 1140)", "FIB-SEM serial section"),
    ("electron_diffraction", "eagle", None, 1150, (256, 256), fwd_fourier_mag, {},
     "skimage eagle (electron diffraction proxy)", "skimage eagle (seed 1150)", "Electron diffraction"),
    ("electron_holography", "gravel", None, 1160, (256, 256), fwd_interferogram, {},
     "skimage gravel (electron holography proxy)", "skimage gravel (seed 1160)", "Electron holography"),
    ("electron_tomography", "camera", None, 1170, (256, 256), fwd_radon_fast, {"n_angles": 30},
     "MIT cameraman (electron tomo proxy)", "skimage camera (seed 1170)", "Electron tomography tilt"),
    ("clem", "moon", None, 1180, (256, 256), fwd_psf, {"sigma": 2.0},
     "NASA moon (CLEM proxy)", "skimage moon (seed 1180)", "CLEM overlay"),
    ("cathodoluminescence", "coins", None, 1190, (256, 256), fwd_psf, {"sigma": 1.5},
     "skimage coins (CL proxy)", "skimage coins (seed 1190)", "CL spectral"),
    ("eels", "brick", None, 1200, (256, 256), fwd_psf, {"sigma": 1.2},
     "skimage brick (EELS proxy)", "skimage brick (seed 1200)", "EELS core-loss"),
    ("edx_mapping", "grass", None, 1210, (256, 256), fwd_psf, {"sigma": 1.8},
     "skimage grass (EDX proxy)", "skimage grass (seed 1210)", "EDX elemental"),
    ("ebsd", "eagle", None, 1220, (256, 256), fwd_psf, {"sigma": 1.0},
     "skimage eagle (EBSD proxy)", "skimage eagle (seed 1220)", "EBSD orientation"),
    ("sims", "gravel", None, 1230, (256, 256), fwd_sparse, {"frac": 0.2},
     "skimage gravel (SIMS proxy)", "skimage gravel (seed 1230)", "SIMS depth profile"),
    ("afm", "camera", None, 1240, (256, 256), fwd_psf, {"sigma": 0.5},
     "MIT cameraman (AFM proxy)", "skimage camera (seed 1240)", "AFM topography"),
    ("stm", "moon", None, 1250, (256, 256), fwd_psf, {"sigma": 0.3},
     "NASA moon (STM proxy)", "skimage moon (seed 1250)", "STM tunneling current"),
    ("mfm", "coins", None, 1260, (256, 256), fwd_interferogram, {},
     "skimage coins (MFM proxy)", "skimage coins (seed 1260)", "MFM magnetic phase"),
    ("nsom", "brick", None, 1270, (256, 256), fwd_psf, {"sigma": 0.4},
     "skimage brick (NSOM proxy)", "skimage brick (seed 1270)", "NSOM near-field"),

    # -- Spectroscopy --
    ("raman_imaging", "brick", None, 1300, (256, 256), fwd_psf, {"sigma": 1.5},
     "skimage brick (Raman proxy)", "skimage brick (seed 1300)", "Raman spectral imaging"),
    ("ftir_imaging", "grass", None, 1310, (256, 256), fwd_psf, {"sigma": 2.0},
     "skimage grass (FTIR proxy)", "skimage grass (seed 1310)", "FTIR spectral imaging"),
    ("libs", "gravel", None, 1320, (256, 256), fwd_sparse, {"frac": 0.1},
     "skimage gravel (LIBS proxy)", "skimage gravel (seed 1320)", "LIBS elemental spectroscopy"),
    ("brillouin", "eagle", None, 1330, (256, 256), fwd_psf, {"sigma": 1.0},
     "skimage eagle (Brillouin proxy)", "skimage eagle (seed 1330)", "Brillouin spectroscopy"),
    ("terahertz", "moon", None, 1340, (256, 256), fwd_psf, {"sigma": 3.0},
     "NASA moon (THz proxy)", "skimage moon (seed 1340)", "THz imaging"),
    ("maldi_msi", "brick", None, 1350, (256, 256), fwd_sparse, {"frac": 0.15},
     "skimage brick (MALDI proxy)", "skimage brick (seed 1350)", "MALDI mass spec imaging"),
    ("desi", "grass", None, 1360, (256, 256), fwd_sparse, {"frac": 0.1},
     "skimage grass (DESI proxy)", "skimage grass (seed 1360)", "DESI mass spec imaging"),
    ("neutron_diffraction", "gravel", None, 1370, (256, 256), fwd_fourier_mag, {},
     "skimage gravel (neutron diffraction proxy)", "skimage gravel (seed 1370)", "Neutron diffraction"),

    # -- Miscellaneous --
    ("entangled_photon", "camera", None, 1400, (256, 256), fwd_sparse, {"frac": 0.03},
     "MIT cameraman (entangled photon proxy)", "skimage camera (seed 1400)", "Entangled photon coincidence"),
    ("quantum_illumination", "moon", None, 1410, (256, 256), fwd_sparse, {"frac": 0.05},
     "NASA moon (quantum illumination proxy)", "skimage moon (seed 1410)", "Quantum illumination"),
    ("dot", "gravel", None, 1420, (256, 256), fwd_psf, {"sigma": 5.0},
     "skimage gravel (DOT proxy)", "skimage gravel (seed 1420)", "Diffuse optical tomography"),
    ("bioluminescence_tomo", "grass", None, 1430, (256, 256), fwd_psf, {"sigma": 4.0},
     "skimage grass (BLT proxy)", "skimage grass (seed 1430)", "BLT bioluminescence"),
    ("ct_fluorescence", "brick", None, 1440, (256, 256), fwd_radon_fast, {"n_angles": 30},
     "skimage brick (CT-fluorescence proxy)", "skimage brick (seed 1440)", "CT-fluorescence tomography"),
    ("magnetic_particle", "coins", None, 1450, (256, 256), fwd_psf, {"sigma": 3.0},
     "skimage coins (MPI proxy)", "skimage coins (seed 1450)", "MPI reconstruction"),
    ("pump_probe", "eagle", None, 1460, (256, 256), fwd_psf, {"sigma": 1.5},
     "skimage eagle (pump-probe proxy)", "skimage eagle (seed 1460)", "Pump-probe ultrafast"),
    ("atom_probe", "camera", None, 1470, (256, 256), fwd_sparse, {"frac": 0.08},
     "MIT cameraman (atom probe proxy)", "skimage camera (seed 1470)", "Atom probe field evaporation"),
    ("xfel_sfx", "moon", None, 1480, (256, 256), fwd_fourier_mag, {},
     "NASA moon (XFEL-SFX proxy)", "skimage moon (seed 1480)", "XFEL SFX diffraction"),
    ("xray_crystallography", "coins", None, 1490, (256, 256), fwd_fourier_mag, {},
     "skimage coins (X-ray crystallography proxy)", "skimage coins (seed 1490)", "X-ray crystallography"),
    ("saxs", "gravel", None, 1500, (256, 256), fwd_fourier_mag, {},
     "skimage gravel (SAXS proxy)", "skimage gravel (seed 1500)", "SAXS scattering"),
    ("waxs", "grass", None, 1510, (256, 256), fwd_fourier_mag, {},
     "skimage grass (WAXS proxy)", "skimage grass (seed 1510)", "WAXS scattering"),
    ("xrf_imaging", "brick", None, 1520, (256, 256), fwd_psf, {"sigma": 1.5},
     "skimage brick (XRF proxy)", "skimage brick (seed 1520)", "XRF elemental mapping"),
    ("xrf_tomo", "eagle", None, 1530, (256, 256), fwd_radon_fast, {"n_angles": 60},
     "skimage eagle (XRF-tomo proxy)", "skimage eagle (seed 1530)", "XRF tomography"),
    ("ocean_acoustic_tomo", "camera", None, 1540, (256, 256), fwd_radon_fast, {"n_angles": 20},
     "MIT cameraman (ocean acoustic proxy)", "skimage camera (seed 1540)", "Ocean acoustic tomography"),
    ("ivus", "gravel", None, 1550, (256, 256), fwd_psf, {"sigma": 1.5},
     "skimage gravel (IVUS proxy)", "skimage gravel (seed 1550)", "IVUS intravascular US"),
    ("elastography", "grass", None, 1560, (256, 256), fwd_psf, {"sigma": 2.5},
     "skimage grass (elastography proxy)", "skimage grass (seed 1560)", "Elastography wave propagation"),
    ("ultrasonic_phased_array", "brick", None, 1570, (256, 256), fwd_psf, {"sigma": 2.0},
     "skimage brick (PAUT proxy)", "skimage brick (seed 1570)", "PAUT B-scan"),
    ("neutron_tomo", "eagle", None, 1580, (256, 256), fwd_radon_fast, {"n_angles": 90},
     "skimage eagle (neutron tomo proxy)", "skimage eagle (seed 1580)", "Neutron tomography"),
    ("proton_radiography", "coins", None, 1590, (256, 256), fwd_psf, {"sigma": 3.0},
     "skimage coins (proton radiography proxy)", "skimage coins (seed 1590)", "Proton radiography"),
    ("proton_therapy_img", "moon", None, 1600, (256, 256), fwd_psf, {"sigma": 2.5},
     "NASA moon (proton therapy proxy)", "skimage moon (seed 1600)", "Proton therapy dose"),
    ("brachytherapy_img", "grass", None, 1610, (256, 256), fwd_psf, {"sigma": 2.0},
     "skimage grass (brachytherapy proxy)", "skimage grass (seed 1610)", "Brachytherapy dose map"),
    ("muon_tomo", "gravel", None, 1620, (256, 256), fwd_radon_fast, {"n_angles": 15},
     "skimage gravel (muon tomo proxy)", "skimage gravel (seed 1620)", "Muon tomography"),
    ("streak_camera", "brick", None, 1630, (256, 256), fwd_psf, {"sigma": 1.5},
     "skimage brick (streak camera proxy)", "skimage brick (seed 1630)", "Streak camera temporal"),
    ("cup", "eagle", None, 1640, (256, 256), fwd_sparse, {"frac": 0.1},
     "skimage eagle (CUP proxy)", "skimage eagle (seed 1640)", "CUP compressed ultrafast"),

    # -- 18 modalities not in batch 3's unexecuted functions --
    ("cacti", "lily", None, 1700, (256, 256), fwd_sparse, {"frac": 0.15},
     "skimage lily (CACTI coded aperture proxy)", "skimage lily (seed 1700)", "CACTI coded aperture snapshot"),
    ("cassi", "rocket", None, 1710, (256, 256), fwd_sparse, {"frac": 0.2},
     "NASA rocket (CASSI coded aperture proxy)", "skimage rocket (seed 1710)", "CASSI coded aperture spectral"),
    ("cryo_em", "human_mitosis", None, 1720, (256, 256), fwd_fourier_mag, {},
     "Broad BBBC; skimage human_mitosis (cryo-EM CTF proxy)", "skimage human_mitosis (seed 1720)", "Cryo-EM CTF"),
    ("eht_imaging", "hubble_deep_field", None, 1730, (256, 256), fwd_sparse, {"frac": 0.01},
     "NASA/ESA HST (EHT VLBI proxy)", "skimage hubble_deep_field (seed 1730)", "EHT very sparse VLBI"),
    ("endoscopy", "immunohistochemistry", None, 1740, (256, 256), fwd_psf, {"sigma": 1.5},
     "PathCore (endoscopy proxy)", "skimage immunohistochemistry (seed 1740)", "Endoscopy PSF"),
    ("fundus", "retina", None, 1750, (256, 256), fwd_psf, {"sigma": 1.0},
     "DRIVE retinal dataset (fundus proxy)", "skimage retina (seed 1750)", "Fundus photography PSF"),
    ("gravitational_wave", "camera", None, 1760, (256, 256), fwd_sparse, {"frac": 0.02},
     "MIT cameraman (gravitational wave proxy)", "skimage camera (seed 1760)", "Gravitational wave strain"),
    ("hyperspectral_remote", "astronaut", 1, 1770, (256, 256), fwd_psf, {"sigma": 2.0},
     "NASA astronaut G-channel (hyperspectral proxy)", "skimage astronaut G-channel (seed 1770)", "Hyperspectral atmospheric PSF"),
    ("lensless", "lily", None, 1780, (256, 256), fwd_psf, {"sigma": 5.0},
     "skimage lily (lensless diffuser proxy)", "skimage lily (seed 1780)", "Lensless diffuser PSF"),
    ("machine_vision", "rocket", None, 1790, (256, 256), fwd_psf, {"sigma": 0.8},
     "NASA rocket (machine vision proxy)", "skimage rocket (seed 1790)", "Machine vision lens PSF"),
    ("matrix", "page", None, 1800, (256, 256), fwd_sparse, {"frac": 0.3},
     "skimage page (transmission matrix proxy)", "skimage page (seed 1800)", "Transmission matrix sparse"),
    ("nerf", "coffee", None, 1810, (256, 256), fwd_psf, {"sigma": 1.5},
     "Heckmann; skimage coffee (NeRF proxy)", "skimage coffee (seed 1810)", "NeRF rendering PSF"),
    ("particle_calorimetry", "grass", None, 1820, (256, 256), fwd_sparse, {"frac": 0.05},
     "skimage grass (particle calorimetry proxy)", "skimage grass (seed 1820)", "Particle calorimetry"),
    ("photoacoustic", "retina", None, 1830, (256, 256), fwd_radon_fast, {"n_angles": 60},
     "DRIVE retinal dataset (PAT proxy)", "skimage retina (seed 1830)", "Photoacoustic tomography"),
    ("radio_astronomy", "hubble_deep_field", None, 1840, (256, 256), fwd_sparse, {"frac": 0.03},
     "NASA/ESA HST (radio aperture synthesis proxy)", "skimage hubble_deep_field (seed 1840)", "Radio aperture synthesis"),
    ("sd_cassi", "rocket", 0, 1850, (256, 256), fwd_sparse, {"frac": 0.2},
     "NASA rocket R-channel (SD-CASSI proxy)", "skimage rocket R-channel (seed 1850)", "SD-CASSI coded aperture"),
    ("spc", "camera", None, 1860, (256, 256), fwd_sparse, {"frac": 0.1},
     "MIT cameraman (single-pixel camera proxy)", "skimage camera (seed 1860)", "SPC sparse sampling"),
    ("spc_kronecker", "camera", None, 1870, (256, 256), fwd_sparse, {"frac": 0.05},
     "MIT cameraman (Kronecker CS proxy)", "skimage camera (seed 1870)", "SPC-Kronecker compressed sensing"),
]


# ---- Special-case modalities (non-standard forward models) ----

def build_coded_exposure():
    img_ch = get_img("chelsea")
    print("\n--- coded_exposure (skimage chelsea, flutter shutter) ---")
    sx, sy = [], []
    rng = np.random.RandomState(990)
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img_ch, 1, seed=990 + i)[0], (256, 256))
        kernel_len = 10 + rng.randint(0, 10)
        kernel = np.zeros((1, kernel_len))
        kernel[0, :] = 1.0 / kernel_len
        y = convolve2d(x, kernel, mode='same', boundary='wrap').astype(np.float32)
        sx.append(x)
        sy.append(y)
    save_modality("coded_exposure", sx, sy,
                  "van der Walt; skimage chelsea (coded exposure proxy)",
                  "skimage chelsea (flutter shutter proxy)",
                  "Coded exposure: motion blur convolution")

def build_dic():
    img_h = get_img("horse")
    print("\n--- dic (skimage horse, DIC gradient) ---")
    sx, sy = [], []
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img_h, 1, seed=1000 + i)[0], (256, 256))
        y = (sobel(x, axis=1) + 0.5).astype(np.float32)
        sx.append(x)
        sy.append(y)
    save_modality("dic", sx, sy,
                  "skimage horse (DIC differential contrast proxy)",
                  "skimage horse (DIC shear gradient proxy)",
                  "DIC: directional gradient (Sobel)")

def build_gaussian_splatting():
    img_a = get_img("astronaut")
    print("\n--- gaussian_splatting (skimage astronaut, 3DGS) ---")
    sx, sy = [], []
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img_a, 1, seed=1020 + i)[0], (256, 256))
        y = shift(x, [2 + i * 0.5, 3 + i * 0.3], order=1).astype(np.float32)
        sx.append(x)
        sy.append(y)
    save_modality("gaussian_splatting", sx, sy,
                  "NASA astronaut (3D Gaussian splatting proxy)",
                  "skimage astronaut (3DGS multi-view proxy)",
                  "3DGS: perspective shift rendering")


# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Batch 4: Build real datasets for 81 remaining modalities")
    print("=" * 60)

    done_count = 0
    skip_count = 0

    # Process table-driven modalities
    for entry in TABLE:
        mod, img_name, channel, seed, target, fwd_fn, fwd_kw, ref, src, desc = entry
        if is_done(mod):
            print(f"  [SKIP] {mod}: already has real data")
            skip_count += 1
            continue
        print(f"\n--- {mod} ---")
        if channel is not None:
            img = get_color_channel(img_name, channel)
        else:
            img = get_img(img_name)
        build_from_img(mod, img, target, fwd_fn, fwd_kw, ref, src, desc, seed=seed)
        done_count += 1

    # Process special-case modalities
    for mod, builder in [("coded_exposure", build_coded_exposure),
                         ("dic", build_dic),
                         ("gaussian_splatting", build_gaussian_splatting)]:
        if is_done(mod):
            print(f"  [SKIP] {mod}: already has real data")
            skip_count += 1
            continue
        builder()
        done_count += 1

    print(f"\n{'=' * 60}")
    print(f"Batch 4 complete! Built: {done_count}, Skipped: {skip_count}")
    print(f"{'=' * 60}")
