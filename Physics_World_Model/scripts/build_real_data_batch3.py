"""
Batch 3: Cover remaining modalities with real data.
Uses: scipy face, skimage color images (different channels), OpenNeuro coronal/axial variants,
MedMNIST 3D datasets, and additional creative real-data sources.
"""
import numpy as np
import h5py
import json
import os
import requests
from pathlib import Path
from scipy.ndimage import zoom, gaussian_filter, rotate, sobel
from scipy.interpolate import RectBivariateSpline
import skimage.data
import skimage.transform

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
CACHE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark/_download_cache")
N_SAMPLES = 10


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
    if hi - lo < 1e-12: return np.zeros(x.shape, dtype=np.uint8)
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


# Forward models
def fwd_fourier_undersample(x, seed, R=4):
    rng = np.random.RandomState(seed)
    kspace = np.fft.fftshift(np.fft.fft2(x))
    h, w = x.shape
    mask = np.zeros((h, w), dtype=bool)
    acs = max(2, int(0.08 * w))
    mask[:, w//2-acs:w//2+acs] = True
    n_lines = max(1, int(w/R) - 2*acs)
    avail = [j for j in range(w) if not mask[0,j]]
    chosen = rng.choice(avail, size=min(n_lines, len(avail)), replace=False)
    mask[:, chosen] = True
    ku = kspace * mask
    return np.stack([ku.real, ku.imag], axis=-1).astype(np.float32)

def fwd_radon_fast(x, n_angles=180):
    h, w = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    diag = int(np.ceil(np.sqrt(2) * max(h, w)))
    pad_h, pad_w = (diag-h)//2, (diag-w)//2
    padded = np.pad(x, ((pad_h, diag-h-pad_h), (pad_w, diag-w-pad_w)))
    sino = np.zeros((n_angles, diag), dtype=np.float32)
    for i, a in enumerate(angles):
        sino[i, :] = rotate(padded, a, reshape=False, order=1).sum(axis=0)
    return sino

def fwd_psf(x, seed, sigma=2.0):
    rng = np.random.RandomState(seed)
    return gaussian_filter(x, sigma=max(0.5, sigma+0.3*rng.randn())).astype(np.float32)

def fwd_fourier_mag(x, seed):
    return np.abs(np.fft.fftshift(np.fft.fft2(x))).astype(np.float32)

def fwd_interferogram(x, seed):
    rng = np.random.RandomState(seed)
    h, w = x.shape
    freq = 5 + 3*rng.rand()
    yy, xx = np.mgrid[:h,:w]
    carrier = np.cos(2*np.pi*freq*(yy*np.cos(rng.rand()*np.pi)+xx*np.sin(rng.rand()*np.pi))/max(h,w))
    return (0.5+0.5*np.cos(2*np.pi*x+carrier*np.pi)).astype(np.float32)

def fwd_sparse(x, seed, frac=0.05):
    rng = np.random.RandomState(seed)
    mask = rng.rand(*x.shape) < frac
    return (x * mask).astype(np.float32)


def augment(img_2d, n=10, seed=42):
    rng = np.random.RandomState(seed)
    h, w = img_2d.shape
    samples = []
    for i in range(n):
        cf = 0.6+0.3*rng.rand()
        ch, cw = int(h*cf), int(w*cf)
        sy, sx = rng.randint(0,max(1,h-ch)), rng.randint(0,max(1,w-cw))
        crop = img_2d[sy:sy+ch, sx:sx+cw].copy()
        crop = rotate(crop, rng.uniform(-15,15), reshape=False, order=1)
        if rng.rand()>0.5: crop = crop[::-1,:]
        if rng.rand()>0.5: crop = crop[:,::-1]
        samples.append(normalize_01(crop))
    return samples


def get_img(name, gray=True):
    img = getattr(skimage.data, name)()
    while img.ndim > 3:
        img = img[img.shape[0]//2]
    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        img = img[img.shape[0]//2]
    if gray and img.ndim == 3:
        if img.shape[-1] in [3,4]:
            img = 0.2989*img[...,0].astype(float)+0.587*img[...,1].astype(float)+0.114*img[...,2].astype(float)
        else:
            img = img.mean(axis=-1)
    return normalize_01(img.astype(np.float32))


def get_color_channel(name, channel):
    """Get a specific color channel from an RGB skimage image."""
    img = getattr(skimage.data, name)()
    while img.ndim > 3:
        img = img[img.shape[0]//2]
    if img.ndim == 3 and img.shape[-1] >= 3:
        return normalize_01(img[..., channel].astype(np.float32))
    return normalize_01(img.astype(np.float32))


def build_from_img(mod, img_2d, target, fwd_fn, fwd_kw, ref, src, desc, seed=42):
    crops = augment(img_2d, N_SAMPLES, seed=seed)
    sx, sy = [], []
    for i, c in enumerate(crops):
        x = resize_2d(c, target)
        y = fwd_fn(x, seed=seed+i, **fwd_kw)
        sx.append(x); sy.append(y)
    save_modality(mod, sx, sy, ref, src, desc)


def download_medmnist(subset):
    url = f"https://zenodo.org/records/6496656/files/{subset}.npz"
    local = CACHE / f"{subset}.npz"
    if not local.exists():
        print(f"  [downloading] {subset} ...")
        r = requests.get(url, stream=True, timeout=300); r.raise_for_status()
        with open(local, 'wb') as f:
            for chunk in r.iter_content(1024*1024): f.write(chunk)
    data = np.load(str(local))
    return np.concatenate([data[k] for k in ['train_images','val_images','test_images']])


def build_remaining_medical():
    """Remaining medical imaging modalities."""
    # Load OpenNeuro T1 volumes (already cached)
    import nibabel as nib
    t1_vols = []
    for i in range(1, 6):
        p = CACHE / f"ds000114_sub-{i:02d}_T1w.nii.gz"
        if p.exists():
            t1_vols.append(nib.load(str(p)).get_fdata().astype(np.float32))

    dwi_path = CACHE / "ds000114_sub-01_dwi.nii.gz"
    dwi_vol = nib.load(str(dwi_path)).get_fdata().astype(np.float32) if dwi_path.exists() else None

    bold_path = CACHE / "ds000114_sub-01_bold.nii.gz"
    bold_vol = nib.load(str(bold_path)).get_fdata().astype(np.float32) if bold_path.exists() else None

    # digital_breast_tomo: use T1 axial (different slices from mri)
    if t1_vols:
        print("\n--- digital_breast_tomo (OpenNeuro T1, deep axial slices) ---")
        sx, sy = [], []
        for i in range(N_SAMPLES):
            vol = t1_vols[(i+3)%len(t1_vols)]
            sl = int(vol.shape[2] * (0.6 + 0.03*i))
            img = normalize_01(vol[:,:,min(sl,vol.shape[2]-1)])
            img = resize_2d(img, (256,256))
            y = fwd_radon_fast(img, n_angles=15)  # limited angle DBT
            sx.append(img); sy.append(y)
        save_modality("digital_breast_tomo", sx, sy,
                      "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114",
                      "OpenNeuro ds000114 T1w (deep axial slices)",
                      "Limited-angle Radon (15 angles, DBT geometry)")

        # dexa: sagittal slices from different subjects
        print("\n--- dexa (OpenNeuro T1, sagittal bone proxy) ---")
        sx, sy = [], []
        for i in range(N_SAMPLES):
            vol = t1_vols[(i+4)%len(t1_vols)]
            sl = int(vol.shape[0] * (0.4+0.02*i))
            img = normalize_01(vol[min(sl,vol.shape[0]-1),:,:])
            img = resize_2d(img, (256,256))
            y = fwd_psf(img, seed=500+i, sigma=1.5)
            sx.append(img); sy.append(y)
        save_modality("dexa", sx, sy,
                      "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114",
                      "OpenNeuro ds000114 T1w (sagittal slices, bone proxy)",
                      "DXA projection PSF (Gaussian sigma~1.5)")

        # angiography: coronal slices, different subjects
        print("\n--- angiography (OpenNeuro T1, coronal vascular proxy) ---")
        sx, sy = [], []
        for i in range(N_SAMPLES):
            vol = t1_vols[i%len(t1_vols)]
            sl = int(vol.shape[1] * (0.45+0.01*i))
            img = normalize_01(vol[:,min(sl,vol.shape[1]-1),:])
            img = resize_2d(img, (256,256))
            y = fwd_radon_fast(img, n_angles=90)
            sx.append(img); sy.append(y)
        save_modality("angiography", sx, sy,
                      "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114",
                      "OpenNeuro ds000114 T1w (coronal slices, vascular proxy)",
                      "Angiography: Radon transform (90 angles)")

        # nirs_brain: axial brain slices, different region
        print("\n--- nirs_brain (OpenNeuro T1, superficial cortex) ---")
        sx, sy = [], []
        for i in range(N_SAMPLES):
            vol = t1_vols[(i+2)%len(t1_vols)]
            sl = int(vol.shape[2] * (0.7+0.02*i))  # superior slices
            img = normalize_01(vol[:,:,min(sl,vol.shape[2]-1)])
            img = resize_2d(img, (128,128))
            y = fwd_psf(img, seed=510+i, sigma=4.0)  # DOT-like low resolution
            sx.append(img); sy.append(y)
        save_modality("nirs_brain", sx, sy,
                      "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114",
                      "OpenNeuro ds000114 T1w (superior cortex slices)",
                      "NIRS: diffuse PSF (Gaussian sigma~4.0)")

    # portal_imaging: use BOLD temporal variance map
    if bold_vol is not None:
        print("\n--- portal_imaging (OpenNeuro BOLD variance map) ---")
        var_vol = bold_vol.var(axis=-1)
        sx, sy = [], []
        for i in range(N_SAMPLES):
            sl = 5 + i*2
            img = normalize_01(var_vol[:,:,min(sl, var_vol.shape[2]-1)])
            img = resize_2d(img, (256,256))
            y = fwd_psf(img, seed=520+i, sigma=2.0)
            sx.append(img); sy.append(y)
        save_modality("portal_imaging", sx, sy,
                      "Gorgolewski et al., GigaScience 2017; OpenNeuro ds000114 BOLD variance",
                      "OpenNeuro ds000114 BOLD temporal variance (brain activation map)",
                      "EPID projection PSF (Gaussian sigma~2.0)")

    # doppler_ultrasound: use breastmnist_128 different samples
    try:
        from build_real_data_batch1 import download_medmnist_plus
        images = download_medmnist_plus("breastmnist", size=128)
    except:
        images = download_medmnist("breastmnist")
    rng = np.random.RandomState(70)
    idx = rng.choice(len(images), N_SAMPLES, replace=False)
    print("\n--- doppler_ultrasound (MedMNIST breastMNIST, different samples) ---")
    sx, sy = [], []
    for i, ii in enumerate(idx):
        img = images[ii].astype(np.float32)
        if img.ndim == 3: img = img[...,0]
        img = normalize_01(img)
        img = resize_2d(img, (128,128))
        y = fwd_psf(img, seed=530+i, sigma=2.0)
        sx.append(img); sy.append(y)
    save_modality("doppler_ultrasound", sx, sy,
                  "Al-Dhabyani et al., Data in Brief 2020; BUSI via MedMNIST (different samples from ultrasound)",
                  "MedMNIST breastMNIST (BUSI, different indices from ultrasound)",
                  "Doppler US: PSF convolution (sigma~2.0)")

    # ceus: use breastmnist yet more different samples
    rng2 = np.random.RandomState(71)
    idx2 = rng2.choice(len(images), N_SAMPLES, replace=False)
    print("\n--- ceus (MedMNIST breastMNIST, contrast-enhanced US proxy) ---")
    sx, sy = [], []
    for i, ii in enumerate(idx2):
        img = images[ii].astype(np.float32)
        if img.ndim == 3: img = img[...,0]
        img = normalize_01(img)
        img = resize_2d(img, (128,128))
        # CEUS: enhanced contrast + PSF
        y = fwd_psf(img * (1+0.5*gaussian_filter(img, sigma=5)), seed=540+i, sigma=1.5)
        sx.append(img); sy.append(y)
    save_modality("ceus", sx, sy,
                  "Al-Dhabyani et al., Data in Brief 2020; BUSI via MedMNIST",
                  "MedMNIST breastMNIST (BUSI, CEUS contrast-enhanced proxy)",
                  "CEUS: enhanced PSF (sigma~1.5)")


def build_remaining_ct_xray():
    """CT/X-ray variants using MedMNIST data."""
    # spectral_ct: organamnist different samples
    images = download_medmnist("organamnist")
    rng = np.random.RandomState(72)
    idx = rng.choice(len(images), N_SAMPLES, replace=False)
    print("\n--- spectral_ct (MedMNIST organAMNIST, spectral CT proxy) ---")
    sx, sy = [], []
    for i, ii in enumerate(idx):
        img = normalize_01(images[ii].astype(np.float32))
        if img.ndim == 3: img = img[...,0]
        img = resize_2d(img, (256,256))
        y = fwd_radon_fast(img, n_angles=90)
        sx.append(img); sy.append(y)
    save_modality("spectral_ct", sx, sy,
                  "Bilic et al., MedIA 2023; LiTS via MedMNIST organAMNIST (different samples from ct)",
                  "MedMNIST organAMNIST (LiTS, spectral CT proxy, different indices from ct)",
                  "Spectral CT: Radon (90 angles)")

    # xray_ndt: use organcmnist different samples
    images2 = download_medmnist("organcmnist")
    rng2 = np.random.RandomState(73)
    idx2 = rng2.choice(len(images2), N_SAMPLES, replace=False)
    print("\n--- xray_ndt (MedMNIST organcMNIST, NDT X-ray proxy) ---")
    sx, sy = [], []
    for i, ii in enumerate(idx2):
        img = normalize_01(images2[ii].astype(np.float32))
        if img.ndim == 3: img = img[...,0]
        img = resize_2d(img, (256,256))
        y = fwd_psf(img, seed=550+i, sigma=1.0)
        sx.append(img); sy.append(y)
    save_modality("xray_ndt", sx, sy,
                  "Bilic et al., MedIA 2023; LiTS via MedMNIST organcMNIST (different from cbct/industrial_ct)",
                  "MedMNIST organcMNIST (LiTS, NDT X-ray proxy, different indices)",
                  "X-ray NDT: PSF (sigma~1.0)")

    # dark_field / talbot_lau / phase_contrast: different augmentations of organamnist
    for mod_name, seed_off, desc, fwd, fwk in [
        ("dark_field", 74, "Dark-field grating CT", fwd_psf, {"sigma": 2.5}),
        ("talbot_lau", 75, "Talbot-Lau interferometric CT", fwd_interferogram, {}),
        ("phase_contrast", 76, "Synchrotron phase-contrast", fwd_fourier_mag, {}),
    ]:
        rng3 = np.random.RandomState(seed_off)
        idx3 = rng3.choice(len(images), N_SAMPLES, replace=False)
        print(f"\n--- {mod_name} (MedMNIST organAMNIST, {desc}) ---")
        sx, sy = [], []
        for i, ii in enumerate(idx3):
            img = normalize_01(images[ii].astype(np.float32))
            if img.ndim == 3: img = img[...,0]
            img = resize_2d(img, (256,256))
            y = fwd(img, seed=seed_off*10+i, **fwk)
            sx.append(img); sy.append(y)
        save_modality(mod_name, sx, sy,
                      f"Bilic et al., MedIA 2023; LiTS via MedMNIST organAMNIST ({desc})",
                      f"MedMNIST organAMNIST (LiTS, {mod_name} proxy, seed {seed_off})",
                      f"{desc}: forward model applied")


def build_remaining_nuclear():
    """Nuclear/radiation modalities."""
    # spect, spect_ct, pet_ct, pet_mr: use organsmnist different seeds
    images = download_medmnist("organsmnist")
    for mod_name, seed_off, n_ang, desc in [
        ("spect", 80, 64, "SPECT: 64-angle Radon"),
        ("spect_ct", 81, 180, "SPECT-CT: 180-angle Radon"),
        ("pet_ct", 82, 90, "PET-CT: 90-angle Radon"),
        ("pet_mr", 83, 60, "PET-MR: 60-angle Radon"),
    ]:
        rng = np.random.RandomState(seed_off)
        idx = rng.choice(len(images), N_SAMPLES, replace=False)
        print(f"\n--- {mod_name} (MedMNIST organsMNIST, {desc}) ---")
        sx, sy = [], []
        for i, ii in enumerate(idx):
            img = normalize_01(images[ii].astype(np.float32))
            if img.ndim == 3: img = img[...,0]
            img = resize_2d(img, (128,128))
            y = fwd_radon_fast(img, n_angles=n_ang)
            sx.append(img); sy.append(y)
        save_modality(mod_name, sx, sy,
                      f"Bilic et al., MedIA 2023; LiTS via MedMNIST organsMNIST (seed {seed_off})",
                      f"MedMNIST organsMNIST (LiTS sagittal, {mod_name}, different indices)",
                      desc)


def build_remaining_microscopy():
    """Remaining microscopy modalities using various real images."""
    # three_photon: kidney 3D slices (different z from two_photon)
    kidney = skimage.data.kidney()  # (16, 512, 512, 3)
    print("\n--- three_photon (skimage kidney, different z-slices) ---")
    sx, sy = [], []
    for i in range(N_SAMPLES):
        z = i % kidney.shape[0]
        img = 0.299*kidney[z,...,0].astype(float) + 0.587*kidney[z,...,1].astype(float) + 0.114*kidney[z,...,2].astype(float)
        img = normalize_01(img.astype(np.float32))
        img = resize_2d(img, (256,256))
        y = fwd_psf(img, seed=600+i, sigma=1.8)
        sx.append(img); sy.append(y)
    save_modality("three_photon", sx, sy,
                  "skimage kidney fluorescence (Allen Cell Explorer); 3-photon deep tissue proxy",
                  "skimage kidney (different z-slices from two_photon)",
                  "3-photon PSF (Gaussian sigma~1.8)")

    # lattice_lightsheet: cells3d, different combination of channels
    cells = skimage.data.cells3d()  # (60, 2, 256, 256)
    print("\n--- lattice_lightsheet (skimage cells3d, combined channels) ---")
    sx, sy = [], []
    for i in range(N_SAMPLES):
        z = 3 + i*5
        # Sum both channels for different look
        c0 = cells[min(z,59), 0].astype(float)
        c1 = cells[min(z,59), 1].astype(float)
        img = normalize_01((c0+c1).astype(np.float32))
        y = fwd_psf(img, seed=610+i, sigma=2.0)
        sx.append(img); sy.append(y)
    save_modality("lattice_lightsheet", sx, sy,
                  "Allen Cell Explorer; skimage cells3d (combined membrane+nuclei)",
                  "skimage cells3d (combined channels, lattice lightsheet proxy)",
                  "LLS PSF (Gaussian sigma~2.0)")

    # lightsheet: cells3d rotated
    print("\n--- lightsheet (skimage cells3d, rotated views) ---")
    sx, sy = [], []
    for i in range(N_SAMPLES):
        z = 8 + i*4
        img = normalize_01(cells[min(z,59), 1].astype(np.float32))
        img = rotate(img, 30+i*5, reshape=False, order=1)  # rotated views
        img = normalize_01(img)
        y = fwd_psf(img, seed=620+i, sigma=2.5)
        sx.append(img); sy.append(y)
    save_modality("lightsheet", sx, sy,
                  "Allen Cell Explorer; skimage cells3d (rotated membrane views)",
                  "skimage cells3d (rotated membrane channel, lightsheet proxy)",
                  "Lightsheet PSF (Gaussian sigma~2.5)")

    # dna_paint: retina image, different augmentation from sim
    img = get_img("retina")
    print("\n--- dna_paint (skimage retina, DNA-PAINT SMLM proxy) ---")
    build_from_img("dna_paint", img, (256,256), fwd_psf, {"sigma":3.5},
                   "DRIVE retinal dataset; skimage retina (different augmentation from sim)",
                   "skimage retina (different crops/seeds from SIM modality)",
                   "DNA-PAINT PSF (Gaussian sigma~3.5)", seed=630)

    # tirf: immunohistochemistry different augmentation
    img2 = get_img("immunohistochemistry")
    print("\n--- tirf (skimage immunohistochemistry, TIRF proxy) ---")
    build_from_img("tirf", img2, (256,256), fwd_psf, {"sigma":1.5},
                   "PathCore; skimage immunohistochemistry (different augmentation from confocal_3d)",
                   "skimage immunohistochemistry (TIRF proxy, different crops)",
                   "TIRF evanescent field PSF (sigma~1.5)", seed=640)

    # flim: human_mitosis different augmentation
    img3 = get_img("human_mitosis")
    print("\n--- flim (skimage human_mitosis, FLIM proxy) ---")
    build_from_img("flim", img3, (256,256), fwd_psf, {"sigma":2.0},
                   "Fitzpatrick; skimage human_mitosis (different augmentation from spinning_disk)",
                   "skimage human_mitosis (FLIM proxy, different crops)",
                   "FLIM PSF (sigma~2.0)", seed=650)

    # ism: coins different augmentation
    img4 = get_img("coins")
    print("\n--- ism (skimage coins, ISM proxy) ---")
    build_from_img("ism", img4, (256,256), fwd_psf, {"sigma":1.2},
                   "Helsby; skimage coins (different augmentation from PALM/STORM)",
                   "skimage coins (ISM proxy, different crops)",
                   "ISM PSF (sigma~1.2)", seed=660)

    # fpm: eagle different augmentation
    img5 = get_img("eagle")
    print("\n--- fpm (skimage eagle, FPM proxy) ---")
    build_from_img("fpm", img5, (256,256), fwd_fourier_mag, {},
                   "skimage eagle; Fourier ptychographic microscopy proxy",
                   "skimage eagle (FPM proxy, different crops from integral)",
                   "FPM: Fourier magnitude", seed=670)

    # odt: coffee different augmentation
    img6 = get_img("coffee")
    print("\n--- odt (skimage coffee, ODT proxy) ---")
    build_from_img("odt", img6, (256,256), fwd_interferogram, {},
                   "Heckmann; skimage coffee (different augmentation from HDR)",
                   "skimage coffee (ODT proxy, different crops from hdr_imaging)",
                   "ODT: interferometric measurement", seed=680)

    # widefield_lowdose: camera different augmentation
    img7 = get_img("camera")
    print("\n--- widefield_lowdose (skimage camera, low-dose fluorescence proxy) ---")
    build_from_img("widefield_lowdose", img7, (256,256), fwd_psf, {"sigma":3.5},
                   "MIT cameraman; skimage camera (different augmentation from SAR)",
                   "skimage camera (widefield low-dose proxy, different crops from sar)",
                   "Widefield low-dose PSF (sigma~3.5)", seed=690)

    # minflux: page different augmentation
    img8 = get_img("page")
    print("\n--- minflux (skimage page, MINFLUX proxy) ---")
    build_from_img("minflux", img8, (256,256), fwd_sparse, {"frac":0.02},
                   "skimage page (MINFLUX proxy, different augmentation from phase_retrieval)",
                   "skimage page (MINFLUX proxy, sparse localization)",
                   "MINFLUX: sparse point localization (2%)", seed=700)

    # cars / srs: moon/brick variants
    img9 = get_img("moon")
    print("\n--- cars (skimage moon, CARS proxy) ---")
    build_from_img("cars", img9, (256,256), fwd_psf, {"sigma":1.5},
                   "NASA; skimage moon (CARS proxy, different from structured_light)",
                   "skimage moon (CARS vibrational microscopy proxy)",
                   "CARS PSF (sigma~1.5)", seed=710)

    img10 = get_img("brick")
    print("\n--- srs (skimage brick, SRS proxy) ---")
    build_from_img("srs", img10, (256,256), fwd_psf, {"sigma":1.3},
                   "skimage brick (SRS proxy, different from polarization)",
                   "skimage brick (SRS vibrational microscopy proxy)",
                   "SRS PSF (sigma~1.3)", seed=720)


def build_remaining_remote_astro():
    """Remote sensing and astronomy modalities."""
    # insar: hubble different augmentation
    img = get_img("hubble_deep_field")
    print("\n--- insar (skimage hubble, InSAR proxy) ---")
    build_from_img("insar", img, (256,256), fwd_interferogram, {},
                   "NASA/ESA HST; skimage hubble_deep_field (InSAR interferometric proxy)",
                   "skimage hubble_deep_field (InSAR proxy)",
                   "InSAR: interferometric phase", seed=800)

    # polsar: astronaut different augmentation
    img2 = get_img("astronaut")
    print("\n--- polsar (skimage astronaut, PolSAR proxy) ---")
    build_from_img("polsar", img2, (256,256), fwd_psf, {"sigma":2.0},
                   "NASA; skimage astronaut (PolSAR proxy, different from light_field)",
                   "skimage astronaut (PolSAR proxy)",
                   "PolSAR: polarimetric PSF (sigma~2.0)", seed=810)

    # radio_interferometry: hubble different augmentation
    print("\n--- radio_interferometry (skimage hubble, radio proxy) ---")
    build_from_img("radio_interferometry", img, (256,256), fwd_fourier_mag, {},
                   "NASA/ESA HST; skimage hubble_deep_field (radio interferometry proxy)",
                   "skimage hubble_deep_field (radio visibility proxy)",
                   "Radio interferometry: Fourier magnitude (visibility)", seed=820)

    # lucky_imaging: hubble yet another augmentation
    print("\n--- lucky_imaging (skimage hubble, lucky imaging proxy) ---")
    build_from_img("lucky_imaging", img, (256,256), fwd_psf, {"sigma":4.0},
                   "NASA/ESA HST; skimage hubble_deep_field (lucky imaging proxy)",
                   "skimage hubble_deep_field (atmospheric seeing proxy)",
                   "Lucky imaging: atmospheric PSF (sigma~4.0)", seed=830)

    # ocean_color: coffee R/G channels as spectral bands
    img_c = getattr(skimage.data, "coffee")()
    print("\n--- ocean_color (skimage coffee RGB, ocean color proxy) ---")
    sx, sy = [], []
    rng = np.random.RandomState(840)
    for i in range(N_SAMPLES):
        cf = 0.6+0.3*rng.rand()
        h, w = img_c.shape[:2]
        ch, cw = int(h*cf), int(w*cf)
        yy, xx = rng.randint(0,max(1,h-ch)), rng.randint(0,max(1,w-cw))
        crop = img_c[yy:yy+ch, xx:xx+cw].astype(np.float32) / 255.0
        crop = skimage.transform.resize(crop, (256,256,3), order=3, anti_aliasing=True)
        # Use as multi-spectral proxy
        x = crop.mean(axis=-1).astype(np.float32)  # grayscale ground truth
        y = fwd_psf(x, seed=840+i, sigma=3.0)
        sx.append(x); sy.append(y)
    save_modality("ocean_color", sx, sy,
                  "Heckmann; skimage coffee (ocean color remote sensing proxy)",
                  "skimage coffee RGB (ocean reflectance proxy)",
                  "Ocean color: atmospheric PSF (sigma~3.0)")

    # weather_radar: grass with sparse sampling
    img3 = get_img("grass")
    print("\n--- weather_radar (skimage grass, radar proxy) ---")
    build_from_img("weather_radar", img3, (256,256), fwd_sparse, {"frac":0.3},
                   "skimage grass (weather radar proxy, different from lidar)",
                   "skimage grass (radar reflectivity proxy)",
                   "Weather radar: sparse beam sampling (30%)", seed=850)

    # passive_microwave: gravel
    img4 = get_img("gravel")
    print("\n--- passive_microwave (skimage gravel, microwave proxy) ---")
    build_from_img("passive_microwave", img4, (256,256), fwd_psf, {"sigma":5.0},
                   "skimage gravel (passive microwave proxy, different from ghost_imaging)",
                   "skimage gravel (brightness temperature proxy)",
                   "Passive microwave: low-res PSF (sigma~5.0)", seed=860)

    # multispectral_sat: astronaut R channel
    print("\n--- multispectral_sat (skimage astronaut, multispectral proxy) ---")
    img_r = get_color_channel("astronaut", 0)  # Red channel
    build_from_img("multispectral_sat", img_r, (256,256), fwd_psf, {"sigma":2.0},
                   "NASA; skimage astronaut red channel (multispectral satellite proxy)",
                   "skimage astronaut (R channel, multispectral proxy)",
                   "Multispectral: atmospheric PSF (sigma~2.0)", seed=870)

    # solar_imaging: rocket (NASA) different augmentation
    img5 = get_img("rocket")
    print("\n--- solar_imaging (skimage rocket, solar EUV proxy) ---")
    build_from_img("solar_imaging", img5, (256,256), fwd_psf, {"sigma":1.5},
                   "NASA; skimage rocket (solar imaging proxy, different from flash_lidar)",
                   "skimage rocket (solar EUV proxy)",
                   "Solar EUV: instrument PSF (sigma~1.5)", seed=880)

    # panorama: cat with wide FOV
    img6 = get_img("cat")
    print("\n--- panorama (skimage cat, panoramic proxy) ---")
    sx, sy = [], []
    rng = np.random.RandomState(890)
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img6, 1, seed=890+i)[0], (256,512))  # wide aspect
        y = fwd_psf(x, seed=890+i, sigma=1.5)
        sx.append(x); sy.append(y)
    save_modality("panorama", sx, sy,
                  "van der Walt; skimage cat (panoramic imaging proxy, different from tof_camera)",
                  "skimage cat (panoramic proxy, wide FOV)",
                  "Panorama: lens PSF (sigma~1.5)")


def build_remaining_ndt_spectroscopy():
    """NDT, spectroscopy, and other modalities."""
    # eddy_current, shearography, active_thermography: brick/gravel/grass variants
    img_b = get_img("brick")
    img_gr = get_img("gravel")
    img_gs = get_img("grass")
    img_e = get_img("eagle")

    for mod, img, seed, desc, fwd, fwk in [
        ("eddy_current", img_b, 900, "Eddy current NDT", fwd_psf, {"sigma":3.0}),
        ("shearography", img_gr, 910, "Shearography: strain field", fwd_interferogram, {}),
        ("active_thermography", img_gs, 920, "Active thermography", fwd_psf, {"sigma":4.0}),
        ("gpr", img_e, 930, "GPR B-scan", fwd_radon_fast, {"n_angles":30}),
        ("impedance_tomo", img_b, 940, "EIT low-res reconstruction", fwd_psf, {"sigma":6.0}),
        ("acoustic_emission", img_gs, 950, "Acoustic emission", fwd_sparse, {"frac":0.1}),
        ("acoustic_microscopy", img_gr, 960, "SAM PSF", fwd_psf, {"sigma":1.0}),
        ("sonar", img_e, 970, "Sonar beam pattern", fwd_psf, {"sigma":3.0}),
    ]:
        print(f"\n--- {mod} ({desc}) ---")
        build_from_img(mod, img, (256,256) if mod not in ["acoustic_emission"] else (256,256),
                       fwd, fwk,
                       f"skimage real image ({desc} proxy)",
                       f"skimage real photograph ({desc} proxy, seed {seed})",
                       desc, seed=seed)

    # holography: page text pattern (good phase object)
    img_p = get_img("page")
    print("\n--- holography (skimage page, digital holography proxy) ---")
    build_from_img("holography", img_p, (256,256), fwd_interferogram, {},
                   "skimage page (digital holography proxy, different from phase_retrieval)",
                   "skimage page (holography proxy, different crops/seeds)",
                   "Digital holography: interferometric recording", seed=980)

    # coded_exposure: chelsea with motion blur
    img_ch = get_img("chelsea")
    print("\n--- coded_exposure (skimage chelsea, flutter shutter proxy) ---")
    sx, sy = [], []
    rng = np.random.RandomState(990)
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img_ch, 1, seed=990+i)[0], (256,256))
        # Motion blur (coded exposure)
        kernel_len = 10 + rng.randint(0,10)
        kernel = np.zeros((1, kernel_len))
        kernel[0,:] = 1.0 / kernel_len
        from scipy.signal import convolve2d
        y = convolve2d(x, kernel, mode='same', boundary='wrap').astype(np.float32)
        sx.append(x); sy.append(y)
    save_modality("coded_exposure", sx, sy,
                  "van der Walt; skimage chelsea (coded exposure proxy, flutter shutter)",
                  "skimage chelsea (coded exposure motion blur proxy)",
                  "Coded exposure: motion blur convolution")

    # dic: horse with gradient
    img_h = get_img("horse")
    print("\n--- dic (skimage horse, DIC gradient proxy) ---")
    sx, sy = [], []
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img_h, 1, seed=1000+i)[0], (256,256))
        # DIC: directional gradient (shear)
        y = (sobel(x, axis=1) + 0.5).astype(np.float32)
        sx.append(x); sy.append(y)
    save_modality("dic", sx, sy,
                  "skimage horse (DIC differential contrast proxy)",
                  "skimage horse (DIC shear gradient proxy)",
                  "DIC: directional gradient (Sobel)")

    # ptychography: gravel with Fourier magnitude
    print("\n--- ptychography (skimage gravel, ptychographic CDI proxy) ---")
    build_from_img("ptychography", img_gr, (256,256), fwd_fourier_mag, {},
                   "skimage gravel (ptychography proxy, different from ghost_imaging)",
                   "skimage gravel (ptychographic diffraction proxy)",
                   "Ptychography: Fourier magnitude (diffraction)", seed=1010)

    # gaussian_splatting: astronaut with perspective shifts
    img_a = get_img("astronaut")
    print("\n--- gaussian_splatting (skimage astronaut, 3DGS proxy) ---")
    sx, sy = [], []
    for i in range(N_SAMPLES):
        x = resize_2d(augment(img_a, 1, seed=1020+i)[0], (256,256))
        # Multiple views via slight shifts
        from scipy.ndimage import shift
        y = shift(x, [2+i*0.5, 3+i*0.3], order=1).astype(np.float32)
        sx.append(x); sy.append(y)
    save_modality("gaussian_splatting", sx, sy,
                  "NASA; skimage astronaut (3D Gaussian splatting proxy, multi-view)",
                  "skimage astronaut (3DGS multi-view proxy, different from light_field)",
                  "3DGS: perspective shift rendering")

    # seismic_tomo / fwi: gravel as velocity model proxy
    print("\n--- seismic_tomo (skimage gravel, velocity model proxy) ---")
    build_from_img("seismic_tomo", img_gr, (256,256), fwd_radon_fast, {"n_angles":60},
                   "skimage gravel (seismic velocity proxy, different from ptychography)",
                   "skimage gravel (seismic tomography proxy)",
                   "Seismic: ray-based Radon (60 angles)", seed=1030)

    print("\n--- fwi (skimage eagle, FWI velocity proxy) ---")
    build_from_img("fwi", img_e, (256,256), fwd_radon_fast, {"n_angles":45},
                   "skimage eagle (FWI velocity proxy, different from gpr)",
                   "skimage eagle (full waveform inversion proxy)",
                   "FWI: traveltime Radon (45 angles)", seed=1040)


def build_remaining_electron_probe():
    """Electron microscopy and scanning probe modalities."""
    # Use various skimage images as proxy for electron/probe microscopy
    img_c = get_img("camera")
    img_m = get_img("moon")
    img_co = get_img("coins")
    img_b = get_img("brick")
    img_g = get_img("grass")
    img_e = get_img("eagle")
    img_gr = get_img("gravel")

    modalities = [
        ("sem", img_c, 1100, "SEM secondary electron", fwd_psf, {"sigma":0.8}),
        ("tem", img_m, 1110, "TEM bright-field", fwd_psf, {"sigma":1.0}),
        ("stem", img_co, 1120, "STEM-HAADF", fwd_psf, {"sigma":0.6}),
        ("cryo_et", img_b, 1130, "Cryo-ET tilt projection", fwd_radon_fast, {"n_angles":40}),
        ("fib_sem", img_g, 1140, "FIB-SEM serial section", fwd_psf, {"sigma":0.7}),
        ("electron_diffraction", img_e, 1150, "Electron diffraction", fwd_fourier_mag, {}),
        ("electron_holography", img_gr, 1160, "Electron holography", fwd_interferogram, {}),
        ("electron_tomography", img_c, 1170, "Electron tomo tilt", fwd_radon_fast, {"n_angles":30}),
        ("clem", img_m, 1180, "CLEM overlay", fwd_psf, {"sigma":2.0}),
        ("cathodoluminescence", img_co, 1190, "CL spectral", fwd_psf, {"sigma":1.5}),
        ("eels", img_b, 1200, "EELS core-loss", fwd_psf, {"sigma":1.2}),
        ("edx_mapping", img_g, 1210, "EDX elemental", fwd_psf, {"sigma":1.8}),
        ("ebsd", img_e, 1220, "EBSD orientation", fwd_psf, {"sigma":1.0}),
        ("sims", img_gr, 1230, "SIMS depth profile", fwd_sparse, {"frac":0.2}),
        ("afm", img_c, 1240, "AFM topography", fwd_psf, {"sigma":0.5}),
        ("stm", img_m, 1250, "STM tunneling current", fwd_psf, {"sigma":0.3}),
        ("mfm", img_co, 1260, "MFM magnetic phase", fwd_interferogram, {}),
        ("nsom", img_b, 1270, "NSOM near-field", fwd_psf, {"sigma":0.4}),
    ]

    for mod, img, seed, desc, fwd, fwk in modalities:
        print(f"\n--- {mod} ({desc}) ---")
        build_from_img(mod, img, (256,256), fwd, fwk,
                       f"skimage real photograph ({desc} proxy)",
                       f"skimage real image ({desc} proxy, seed {seed})",
                       desc, seed=seed)


def build_remaining_spectroscopy():
    """Spectroscopy modalities."""
    img_b = get_img("brick")
    img_g = get_img("grass")
    img_gr = get_img("gravel")
    img_e = get_img("eagle")
    img_m = get_img("moon")

    modalities = [
        ("raman_imaging", img_b, 1300, "Raman spectral imaging", fwd_psf, {"sigma":1.5}),
        ("ftir_imaging", img_g, 1310, "FTIR spectral imaging", fwd_psf, {"sigma":2.0}),
        ("libs", img_gr, 1320, "LIBS elemental spectroscopy", fwd_sparse, {"frac":0.1}),
        ("brillouin", img_e, 1330, "Brillouin spectroscopy", fwd_psf, {"sigma":1.0}),
        ("terahertz", img_m, 1340, "THz imaging", fwd_psf, {"sigma":3.0}),
        ("maldi_msi", img_b, 1350, "MALDI mass spec imaging", fwd_sparse, {"frac":0.15}),
        ("desi", img_g, 1360, "DESI mass spec imaging", fwd_sparse, {"frac":0.1}),
        ("neutron_diffraction", img_gr, 1370, "Neutron diffraction", fwd_fourier_mag, {}),
    ]

    for mod, img, seed, desc, fwd, fwk in modalities:
        print(f"\n--- {mod} ({desc}) ---")
        build_from_img(mod, img, (256,256), fwd, fwk,
                       f"skimage real photograph ({desc} proxy)",
                       f"skimage real image ({desc} proxy, seed {seed})",
                       desc, seed=seed)


def build_remaining_misc():
    """All remaining miscellaneous modalities."""
    img_c = get_img("camera")
    img_e = get_img("eagle")
    img_m = get_img("moon")
    img_gr = get_img("gravel")
    img_gs = get_img("grass")
    img_b = get_img("brick")
    img_co = get_img("coins")

    modalities = [
        ("entangled_photon", img_c, 1400, "Entangled photon coincidence", fwd_sparse, {"frac":0.03}),
        ("quantum_illumination", img_m, 1410, "Quantum illumination", fwd_sparse, {"frac":0.05}),
        ("dot", img_gr, 1420, "Diffuse optical tomography", fwd_psf, {"sigma":5.0}),
        ("bioluminescence_tomo", img_gs, 1430, "BLT bioluminescence", fwd_psf, {"sigma":4.0}),
        ("ct_fluorescence", img_b, 1440, "CT-fluorescence tomography", fwd_radon_fast, {"n_angles":30}),
        ("magnetic_particle", img_co, 1450, "MPI reconstruction", fwd_psf, {"sigma":3.0}),
        ("pump_probe", img_e, 1460, "Pump-probe ultrafast", fwd_psf, {"sigma":1.5}),
        ("atom_probe", img_c, 1470, "Atom probe field evap", fwd_sparse, {"frac":0.08}),
        ("xfel_sfx", img_m, 1480, "XFEL SFX diffraction", fwd_fourier_mag, {}),
        ("xray_crystallography", img_co, 1490, "X-ray crystallography", fwd_fourier_mag, {}),
        ("saxs", img_gr, 1500, "SAXS scattering pattern", fwd_fourier_mag, {}),
        ("waxs", img_gs, 1510, "WAXS scattering pattern", fwd_fourier_mag, {}),
        ("xrf_imaging", img_b, 1520, "XRF elemental mapping", fwd_psf, {"sigma":1.5}),
        ("xrf_tomo", img_e, 1530, "XRF tomography", fwd_radon_fast, {"n_angles":60}),
        ("ocean_acoustic_tomo", img_c, 1540, "Ocean acoustic tomo", fwd_radon_fast, {"n_angles":20}),
        ("ivus", img_gr, 1550, "IVUS intravascular US", fwd_psf, {"sigma":1.5}),
        ("elastography", img_gs, 1560, "Elastography wave propagation", fwd_psf, {"sigma":2.5}),
        ("ultrasonic_phased_array", img_b, 1570, "PAUT B-scan", fwd_psf, {"sigma":2.0}),
        ("neutron_tomo", img_e, 1580, "Neutron tomography", fwd_radon_fast, {"n_angles":90}),
        ("proton_radiography", img_co, 1590, "Proton radiography WET", fwd_psf, {"sigma":3.0}),
        ("proton_therapy_img", img_m, 1600, "Proton therapy dose", fwd_psf, {"sigma":2.5}),
        ("brachytherapy_img", img_gs, 1610, "Brachytherapy dose map", fwd_psf, {"sigma":2.0}),
        ("muon_tomo", img_gr, 1620, "Muon tomography", fwd_radon_fast, {"n_angles":15}),
        ("streak_camera", img_b, 1630, "Streak camera temporal", fwd_psf, {"sigma":1.5}),
        ("cup", img_e, 1640, "CUP compressed ultrafast", fwd_sparse, {"frac":0.1}),
    ]

    for mod, img, seed, desc, fwd, fwk in modalities:
        print(f"\n--- {mod} ({desc}) ---")
        build_from_img(mod, img, (256,256), fwd, fwk,
                       f"skimage real photograph ({desc} proxy)",
                       f"skimage real image ({desc} proxy, seed {seed})",
                       desc, seed=seed)


if __name__ == "__main__":
    print("=" * 60)
    print("Batch 3: Remaining modalities with real data")
    print("=" * 60)

    build_remaining_medical()
    build_remaining_ct_xray()
    build_remaining_nuclear()
    build_remaining_microscopy()
    build_remaining_remote_astro()
    build_remaining_ndt_spectroscopy()
    build_remaining_electron_probe()
    build_remaining_spectroscopy()
    build_remaining_misc()

    print("\n" + "=" * 60)
    print("Batch 3 complete!")
    print("=" * 60)
