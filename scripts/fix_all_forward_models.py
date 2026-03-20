"""
Comprehensive forward model fix for ALL standard datasets.

Fixes:
  1. Identity modalities (19) -> add appropriate degradation operators
  2. Phase retrieval group (6) -> |FFT|^2 (magnitude-only measurement)
  3. Holography -> Fresnel propagation (not Gaussian PSF)
  4. Ultrasound group (6) -> anisotropic PSF (depth-dependent resolution)
  5. PSF parameters -> modality-appropriate sigma values
  6. Ghost imaging -> bucket detection (random projections)
  7. Coded exposure -> motion blur kernel (not Gaussian)
  8. X-ray transmission -> Beer-Lambert attenuation (not PSF)
  9. SPC -> compressive random projection (fewer measurements)
  10. Coded_exposure -> coded motion blur

Does NOT touch modalities with dedicated builders:
  cacti, cassi, cup, streak_camera, cryo_em, eht_imaging, gravitational_wave,
  hyperspectral_remote, lensless, machine_vision, matrix, nerf, photoacoustic,
  radio_astronomy, spc_kronecker, sd_cassi, fundus, particle_calorimetry

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
import os
from pathlib import Path
from PIL import Image
from scipy import ndimage

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
DATA_DIR = ROOT / "datasets" / "benchmark"

# Modalities with dedicated builders — skip these
SKIP_MODS = {
    "cacti", "cassi", "cup", "streak_camera", "cryo_em",
    "eht_imaging", "gravitational_wave", "hyperspectral_remote",
    "lensless", "machine_vision", "matrix", "nerf", "photoacoustic",
    "radio_astronomy", "spc_kronecker", "sd_cassi", "fundus",
    "particle_calorimetry",
}


# ============================================================
# Forward model functions
# ============================================================

def fwd_psf(x, sigma=1.5):
    """Gaussian PSF convolution."""
    return ndimage.gaussian_filter(x, sigma=sigma).astype(np.float32)

def fwd_psf_aniso(x, sigma_axial=1.0, sigma_lateral=2.5):
    """Anisotropic PSF (e.g., ultrasound: better axial than lateral)."""
    return ndimage.gaussian_filter(x, sigma=[sigma_axial, sigma_lateral]).astype(np.float32)

def fwd_radon(x, n_angles=180):
    """Radon transform (parallel beam)."""
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    h, w = x.shape
    sino = np.zeros((n_angles, w), dtype=np.float32)
    cx, cy = h / 2.0, w / 2.0
    for i, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated = ndimage.rotate(x, np.degrees(theta), reshape=False, order=1)
        sino[i, :] = rotated.sum(axis=0)
    return sino.astype(np.float32)

def fwd_fourier_undersample(x, ratio=0.3, seed=42):
    """Fourier undersampling (k-space for MRI-like)."""
    rng = np.random.RandomState(seed)
    F = np.fft.fft2(x)
    mask = rng.rand(*F.shape) < ratio
    # Always keep center
    ch, cw = F.shape[0]//2, F.shape[1]//2
    r = max(3, min(F.shape)//10)
    mask[ch-r:ch+r, cw-r:cw+r] = True
    F_under = F * mask
    y = np.stack([F_under.real, F_under.imag], axis=-1)
    return y.astype(np.float32)

def fwd_magnitude_fourier(x):
    """Magnitude-only Fourier: y = |FFT(x)|^2 (phase retrieval forward)."""
    F = np.fft.fftshift(np.fft.fft2(x))
    intensity = np.abs(F) ** 2
    # Log scale for visualization
    return intensity.astype(np.float32)

def fwd_fresnel(x, z_m=0.01, wavelength_m=632.8e-9, pixel_size_m=1e-6):
    """Fresnel propagation for holography."""
    N = x.shape[0]
    k = 2 * np.pi / wavelength_m
    fx = np.fft.fftfreq(N, d=pixel_size_m)
    FY, FX = np.meshgrid(fx, fx, indexing="ij")
    # Fresnel transfer function
    H = np.exp(1j * k * z_m) * np.exp(-1j * np.pi * wavelength_m * z_m * (FX**2 + FY**2))
    F = np.fft.fft2(x.astype(np.complex128))
    U = np.fft.ifft2(F * H)
    # Hologram = intensity
    hologram = np.abs(U) ** 2
    return hologram.astype(np.float32)

def fwd_bucket_detection(x, n_patterns=256, seed=42):
    """Ghost imaging: bucket detector with random illumination patterns.
    y_k = <pattern_k, x> for each pattern k.
    Returns 1D bucket signal of length n_patterns.
    """
    rng = np.random.RandomState(seed)
    flat_x = x.flatten()
    n_pixels = len(flat_x)
    patterns = rng.randn(n_patterns, n_pixels).astype(np.float32)
    patterns = (patterns > 0).astype(np.float32)  # binary speckle patterns
    y = patterns @ flat_x  # (n_patterns,) bucket measurements
    return y.astype(np.float32)

def fwd_motion_blur(x, length=15, angle=0):
    """Motion blur kernel (for coded exposure)."""
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    for i in range(length):
        t = i - center
        r = int(round(center + t * sin_a))
        c = int(round(center + t * cos_a))
        if 0 <= r < length and 0 <= c < length:
            kernel[r, c] = 1.0
    kernel /= kernel.sum() + 1e-10
    from scipy.signal import fftconvolve
    y = fftconvolve(x, kernel, mode="same")
    return y.astype(np.float32)

def fwd_beer_lambert(x, mu_scale=0.02):
    """Beer-Lambert attenuation: I = I0 * exp(-mu * x * thickness).
    For X-ray transmission imaging (radiography, fluoroscopy).
    x = tissue density map, y = transmitted intensity.
    """
    # Line integral (sum) along projection direction
    # For 2D: simulate single projection
    thickness = x * mu_scale  # optical density
    y = np.exp(-thickness)    # transmitted intensity
    return y.astype(np.float32)

def fwd_spectral_broaden(x, fwhm=8):
    """Spectral broadening by instrument response function.
    For 1D spectroscopy modalities. Gaussian kernel.
    """
    sigma = fwhm / 2.355
    return ndimage.gaussian_filter1d(x, sigma=sigma).astype(np.float32)

def fwd_spectral_broaden_nd(x, fwhm=3):
    """Spectral broadening for N-D spectral data (along last axis)."""
    sigma = fwhm / 2.355
    y = ndimage.gaussian_filter1d(x, sigma=sigma, axis=-1)
    # Also add slight spatial PSF
    if x.ndim == 3:
        for ch in range(x.shape[-1]):
            y[:, :, ch] = ndimage.gaussian_filter(y[:, :, ch], sigma=0.8)
    return y.astype(np.float32)

def fwd_downsample(x, factor=2):
    """Spatial downsampling."""
    if x.ndim == 2:
        return x[::factor, ::factor].astype(np.float32)
    elif x.ndim == 3:
        return x[::factor, ::factor, :].astype(np.float32)
    return x.astype(np.float32)

def fwd_compressive_proj(x, n_measurements=64, seed=42):
    """Compressive random projection (SPC).
    y = Phi @ x_flat where Phi is random Gaussian.
    """
    rng = np.random.RandomState(seed)
    flat = x.flatten()
    Phi = rng.randn(n_measurements, len(flat)).astype(np.float32) / np.sqrt(len(flat))
    y = Phi @ flat
    return y.astype(np.float32)

def fwd_depth_projection(x, max_range=50.0, seed=42):
    """LiDAR/ToF depth measurement: sparse + quantized depth map."""
    rng = np.random.RandomState(seed)
    # Sparse sampling (LiDAR doesn't measure every pixel)
    mask = (rng.rand(*x.shape) < 0.1).astype(np.float32)
    # Quantize depth
    y = np.round(x * 255 / max_range) * max_range / 255
    y = y * mask
    return y.astype(np.float32)

def fwd_eit(x, n_electrodes=16, seed=42):
    """Simplified EIT (Electrical Impedance Tomography) forward.
    Approximate: voltage measurements from conductivity distribution.
    Uses simplified sensitivity-weighted projections.
    """
    rng = np.random.RandomState(seed)
    H, W = x.shape
    n_meas = n_electrodes * (n_electrodes - 1) // 2
    y = np.zeros((H, W), dtype=np.float32)
    # Apply current-flow-like weighting (sensitivity map)
    for pair in range(min(n_meas, 30)):
        angle = pair * np.pi / 15
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = H/2, W/2
        # Sensitivity decays with distance from electrode pair axis
        proj_axis = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        perp_axis = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        sensitivity = np.exp(-0.5 * perp_axis**2 / (W/4)**2)
        y += x * sensitivity / n_meas
    return ndimage.gaussian_filter(y, sigma=3.0).astype(np.float32)

def fwd_interference(x, ref_angle=0.02, wavelength_px=20):
    """Electron holography: off-axis interference pattern."""
    H, W = x.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Reference wave tilted at angle
    ref_phase = 2 * np.pi * ref_angle * xx
    # Object phase proportional to image
    obj_phase = x * np.pi  # phase shift
    obj_amp = 0.8 + 0.2 * x  # amplitude modulation
    # Interference: |obj + ref|^2
    obj_wave = obj_amp * np.exp(1j * obj_phase)
    ref_wave = np.exp(1j * ref_phase)
    hologram = np.abs(obj_wave + ref_wave) ** 2
    return hologram.astype(np.float32)

def fwd_event_threshold(x, threshold=0.1, seed=42):
    """Event camera: detect temporal changes above threshold.
    Simulates log-intensity change detection.
    """
    rng = np.random.RandomState(seed)
    # Simulate two frames with slight motion
    shift_x = rng.randint(1, 5)
    x_shifted = np.roll(x, shift_x, axis=1)
    log_diff = np.log(x + 1e-3) - np.log(x_shifted + 1e-3)
    # Threshold to get events
    events = np.zeros_like(x)
    events[log_diff > threshold] = 1.0
    events[log_diff < -threshold] = -1.0
    return events.astype(np.float32)

def fwd_stokes(x, seed=42):
    """Polarization: Stokes parameter measurement."""
    rng = np.random.RandomState(seed)
    # Generate polarization state from intensity
    I = x
    Q = x * (0.3 + 0.2 * rng.randn(*x.shape)).astype(np.float32)
    # Measurement through polarizer at 0 degrees
    y = 0.5 * (I + Q)
    return np.clip(y, 0, 1).astype(np.float32)

def fwd_tone_clip(x, n_exposures=3, gamma=2.2):
    """HDR -> LDR: tone mapping + clipping (simulates camera response)."""
    # Single exposure with limited dynamic range
    exposure = 0.5  # middle exposure
    y = np.clip(x * exposure, 0, 1)
    y = y ** (1.0 / gamma)  # gamma correction
    return y.astype(np.float32)

def fwd_pattern_proj(x, n_patterns=3, seed=42):
    """Structured light: sinusoidal fringe projection."""
    rng = np.random.RandomState(seed)
    H, W = x.shape
    # Project sinusoidal pattern and capture reflected image
    freq = rng.uniform(5, 15)
    phase = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pattern = 0.5 + 0.5 * np.cos(2 * np.pi * freq * xx / W + phase)
    y = x * pattern  # reflected = object * pattern
    return y.astype(np.float32)


# ============================================================
# Modality -> forward model mapping
# ============================================================

def to_uint8(arr):
    arr = np.nan_to_num(arr)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def save_images_generic(x_true, y_ideal, img_dir):
    """Save x_true and y_ideal images (handles 1D, 2D, 3D)."""
    img_dir.mkdir(parents=True, exist_ok=True)

    for prefix, arr in [("x_true", x_true), ("y_ideal", y_ideal)]:
        arr = np.nan_to_num(arr)
        if arr.ndim == 1:
            # 1D: save as thin strip image
            N = len(arr)
            H = 64
            u8 = to_uint8(arr)
            img = np.zeros((H, N), dtype=np.uint8)
            for i, v in enumerate(u8):
                row = H - 1 - int(v / 255.0 * (H - 1))
                img[row:, i] = v
            Image.fromarray(img, mode="L").save(str(img_dir / f"{prefix}.png"))
        elif arr.ndim == 2:
            Image.fromarray(to_uint8(arr), mode="L").save(str(img_dir / f"{prefix}.png"))
        elif arr.ndim == 3:
            if arr.shape[-1] == 2:
                # Complex (re, im) -> magnitude
                mag = np.sqrt(arr[..., 0]**2 + arr[..., 1]**2)
                Image.fromarray(to_uint8(np.log1p(mag)), mode="L").save(
                    str(img_dir / f"{prefix}.png"))
            elif arr.shape[-1] == 3:
                Image.fromarray(to_uint8(arr), mode="RGB").save(
                    str(img_dir / f"{prefix}.png"))
            elif arr.shape[-1] <= 16:
                # Spectral: save middle band as main
                mid = arr.shape[-1] // 2
                Image.fromarray(to_uint8(arr[:, :, mid]), mode="L").save(
                    str(img_dir / f"{prefix}.png"))
                for c in range(min(arr.shape[-1], 8)):
                    Image.fromarray(to_uint8(arr[:, :, c]), mode="L").save(
                        str(img_dir / f"{prefix}_band_{c:02d}.png"))
            else:
                # 3D volume (D, H, W)
                mid = arr.shape[0] // 2
                Image.fromarray(to_uint8(arr[mid]), mode="L").save(
                    str(img_dir / f"{prefix}.png"))
                for s in range(min(arr.shape[0], 8)):
                    idx = s * arr.shape[0] // 8
                    Image.fromarray(to_uint8(arr[idx]), mode="L").save(
                        str(img_dir / f"{prefix}_slice_{idx:02d}.png"))


def apply_forward_and_update(modality, fwd_func, fwd_type_name, fwd_desc, **fwd_kwargs):
    """Apply a new forward model to an existing modality's standard dataset."""
    std_dir = DATA_DIR / modality / "standard"
    if not std_dir.exists():
        return False

    h5_files = sorted(std_dir.glob("standard_*.h5"))
    if not h5_files:
        return False

    updated = 0
    for h5_path in h5_files:
        try:
            # Read x_true
            with h5py.File(str(h5_path), "r") as f:
                x_true = f["x_true"][:]

            # Apply new forward model
            seed = 42 + updated * 137
            if "seed" in fwd_func.__code__.co_varnames:
                y_ideal = fwd_func(x_true, seed=seed, **fwd_kwargs)
            else:
                y_ideal = fwd_func(x_true, **fwd_kwargs)

            # Write back
            with h5py.File(str(h5_path), "a") as f:
                if "y_ideal" in f:
                    del f["y_ideal"]
                f.create_dataset("y_ideal", data=y_ideal, compression="gzip")
                if "H_params" in f:
                    f["H_params"].attrs["forward_type"] = fwd_type_name

            # Update images
            stem = h5_path.stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                sample_id = f"sample_{parts[1]}"
            else:
                sample_id = f"sample_{updated:02d}"
            img_dir = std_dir / "images" / sample_id
            save_images_generic(x_true, y_ideal, img_dir)

            updated += 1
        except Exception as e:
            print(f"    ERROR {modality}/{h5_path.name}: {e}")

    # Update metadata
    meta_path = std_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["forward_type"] = fwd_type_name
        if fwd_desc:
            meta["forward_model"] = fwd_desc
        # Update y_ideal_shape from first file
        first_h5 = h5_files[0]
        with h5py.File(str(first_h5), "r") as f:
            meta["y_ideal_shape"] = list(f["y_ideal"].shape)
        meta["date_built"] = "2026-03-14"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Update spec
    spec_path = std_dir / "spec.json"
    if spec_path.exists():
        with open(spec_path) as f:
            spec = json.load(f)
        spec["forward_type"] = fwd_type_name
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)

    return updated > 0


def main():
    print("=" * 70)
    print("Comprehensive forward model fix for ALL standard datasets")
    print("=" * 70)

    fixes = {}
    total_fixed = 0

    # ================================================================
    # GROUP 1: Phase retrieval — |FFT|^2 (magnitude-only measurement)
    # ================================================================
    print("\n[1] Phase retrieval group: FFT -> |FFT|^2")
    phase_mods = ["phase_retrieval", "ptychography", "fpm",
                  "xfel_sfx", "xray_crystallography"]
    for mod in phase_mods:
        if mod in SKIP_MODS:
            continue
        ok = apply_forward_and_update(
            mod, fwd_magnitude_fourier, "magnitude_fourier",
            "y = |FFT(x)|^2 (intensity-only, phase lost)")
        if ok:
            fixes[mod] = "magnitude_fourier"
            total_fixed += 1
            print(f"    {mod}: -> |FFT|^2")

    # ================================================================
    # GROUP 2: Holography — Fresnel propagation
    # ================================================================
    print("\n[2] Holography: PSF -> Fresnel propagation")
    holo_mods = ["holography"]
    for mod in holo_mods:
        ok = apply_forward_and_update(
            mod, fwd_fresnel, "fresnel_propagation",
            "y = |U(z)|^2 where U = IFFT(FFT(x) * H_fresnel(z))")
        if ok:
            fixes[mod] = "fresnel_propagation"
            total_fixed += 1
            print(f"    {mod}: -> Fresnel propagation")

    # ================================================================
    # GROUP 3: Ghost imaging — bucket detection
    # ================================================================
    print("\n[3] Ghost imaging: mask -> bucket detection")
    for mod in ["ghost_imaging"]:
        ok = apply_forward_and_update(
            mod, fwd_bucket_detection, "bucket_detection",
            "y_k = <pattern_k, x> (scalar bucket measurements)",
            n_patterns=256)
        if ok:
            fixes[mod] = "bucket_detection"
            total_fixed += 1
            print(f"    {mod}: -> bucket detection (256 patterns)")

    # ================================================================
    # GROUP 4: Ultrasound group — anisotropic PSF
    # ================================================================
    print("\n[4] Ultrasound: isotropic PSF -> anisotropic PSF")
    us_mods = {
        "ultrasound": (0.8, 2.5),        # good axial, poor lateral
        "doppler_ultrasound": (0.8, 2.5),
        "ceus": (1.0, 3.0),              # contrast = more scatter
        "ivus": (0.5, 1.5),              # intravascular = higher freq
        "elastography": (1.0, 2.0),
        "ultrasonic_phased_array": (0.6, 2.0),
    }
    for mod, (s_ax, s_lat) in us_mods.items():
        ok = apply_forward_and_update(
            mod, fwd_psf_aniso, "anisotropic_psf",
            f"PSF: sigma_axial={s_ax}, sigma_lateral={s_lat}",
            sigma_axial=s_ax, sigma_lateral=s_lat)
        if ok:
            fixes[mod] = f"aniso_psf({s_ax},{s_lat})"
            total_fixed += 1
            print(f"    {mod}: -> aniso PSF ({s_ax}, {s_lat})")

    # ================================================================
    # GROUP 5: X-ray transmission — Beer-Lambert
    # ================================================================
    print("\n[5] X-ray transmission: PSF -> Beer-Lambert")
    xray_mods = ["fluoroscopy", "angiography", "xray_radiography",
                 "portal_imaging"]
    for mod in xray_mods:
        ok = apply_forward_and_update(
            mod, fwd_beer_lambert, "beer_lambert",
            "y = exp(-mu * x) (X-ray attenuation)",
            mu_scale=0.02)
        if ok:
            fixes[mod] = "beer_lambert"
            total_fixed += 1
            print(f"    {mod}: -> Beer-Lambert attenuation")

    # ================================================================
    # GROUP 6: Coded exposure — motion blur kernel
    # ================================================================
    print("\n[6] Coded exposure: PSF -> motion blur")
    ok = apply_forward_and_update(
        "coded_exposure", fwd_motion_blur, "motion_blur",
        "y = conv(x, motion_kernel)",
        length=15, angle=30)
    if ok:
        fixes["coded_exposure"] = "motion_blur"
        total_fixed += 1
        print("    coded_exposure: -> motion blur kernel")

    # ================================================================
    # GROUP 7: 1D spectroscopy — instrument broadening
    # ================================================================
    print("\n[7] 1D spectroscopy: identity -> instrument broadening")
    spec_1d_mods = {
        "acoustic_emission": 12,
        "brillouin": 6,
        "eels": 8,
        "libs": 10,
        "mrs": 5,
        "nirs_brain": 8,
        "terahertz": 6,
    }
    for mod, fwhm in spec_1d_mods.items():
        ok = apply_forward_and_update(
            mod, fwd_spectral_broaden, "spectral_broadening",
            f"y = conv(x, gaussian_IRF(fwhm={fwhm}))",
            fwhm=fwhm)
        if ok:
            fixes[mod] = f"spectral_broaden(fwhm={fwhm})"
            total_fixed += 1
            print(f"    {mod}: -> spectral broadening (FWHM={fwhm})")

    # ================================================================
    # GROUP 8: 2D/3D spectroscopy with identity -> spectral+spatial broadening
    # ================================================================
    print("\n[8] Spectral imaging: identity -> spectral+spatial broadening")
    spec_nd_mods = {
        "desi": 3, "maldi_msi": 3, "sims": 3, "xrf_imaging": 3,
        "cathodoluminescence": 4, "cars": 3, "srs": 3,
        "edx_mapping": 3, "raman_imaging": 3, "ftir_imaging": 4,
    }
    for mod, fwhm in spec_nd_mods.items():
        ok = apply_forward_and_update(
            mod, fwd_spectral_broaden_nd, "spectral_spatial_broadening",
            f"y = spatial_psf(spectral_broaden(x, fwhm={fwhm}), sigma=0.8)",
            fwhm=fwhm)
        if ok:
            fixes[mod] = f"spectral_spatial_broaden(fwhm={fwhm})"
            total_fixed += 1
            print(f"    {mod}: -> spectral+spatial broadening")

    # ================================================================
    # GROUP 9: Electron microscopy identity -> appropriate models
    # ================================================================
    print("\n[9] Electron microscopy: identity -> correct models")

    # Electron holography -> off-axis interference
    ok = apply_forward_and_update(
        "electron_holography", fwd_interference, "interference_hologram",
        "y = |obj_wave + ref_wave|^2 (off-axis electron holography)")
    if ok:
        fixes["electron_holography"] = "interference"
        total_fixed += 1
        print("    electron_holography: -> interference pattern")

    # EBSD -> Fourier magnitude (diffraction pattern)
    ok = apply_forward_and_update(
        "ebsd", fwd_magnitude_fourier, "diffraction_pattern",
        "y = |FFT(x)|^2 (electron backscatter diffraction)")
    if ok:
        fixes["ebsd"] = "diffraction_pattern"
        total_fixed += 1
        print("    ebsd: -> diffraction pattern")

    # Atom probe -> PSF broadening (mass resolution)
    ok = apply_forward_and_update(
        "atom_probe", fwd_psf, "mass_broadening",
        "y = conv(x, psf) (mass spectrometer resolution)",
        sigma=2.0)
    if ok:
        fixes["atom_probe"] = "mass_broadening"
        total_fixed += 1
        print("    atom_probe: -> mass resolution broadening")

    # ================================================================
    # GROUP 10: Depth/range sensors — sparse depth
    # ================================================================
    print("\n[10] Depth sensors: identity -> sparse depth")
    depth_mods = ["flash_lidar", "lidar", "tof_camera"]
    for mod in depth_mods:
        ok = apply_forward_and_update(
            mod, fwd_depth_projection, "sparse_depth",
            "y = sparse_sample(quantize(x)) (sparse depth measurement)")
        if ok:
            fixes[mod] = "sparse_depth"
            total_fixed += 1
            print(f"    {mod}: -> sparse depth projection")

    # ================================================================
    # GROUP 11: Other identity modalities
    # ================================================================
    print("\n[11] Remaining identity modalities")

    # Event camera -> temporal change detection
    ok = apply_forward_and_update(
        "event_camera", fwd_event_threshold, "event_threshold",
        "y = threshold(log(I_t) - log(I_{t-1}))")
    if ok:
        fixes["event_camera"] = "event_threshold"
        total_fixed += 1
        print("    event_camera: -> event threshold detection")

    # HDR -> LDR tone clipping
    ok = apply_forward_and_update(
        "hdr_imaging", fwd_tone_clip, "tone_clip",
        "y = clip(x * exposure)^(1/gamma)")
    if ok:
        fixes["hdr_imaging"] = "tone_clip"
        total_fixed += 1
        print("    hdr_imaging: -> tone clipping")

    # Impedance tomography -> EIT forward
    ok = apply_forward_and_update(
        "impedance_tomo", fwd_eit, "eit_sensitivity",
        "y = sensitivity_weighted_proj(x) (EIT)")
    if ok:
        fixes["impedance_tomo"] = "eit_sensitivity"
        total_fixed += 1
        print("    impedance_tomo: -> EIT sensitivity model")

    # Polarization -> Stokes measurement
    ok = apply_forward_and_update(
        "polarization", fwd_stokes, "stokes_measurement",
        "y = 0.5 * (I + Q*cos(2*theta))")
    if ok:
        fixes["polarization"] = "stokes"
        total_fixed += 1
        print("    polarization: -> Stokes measurement")

    # Pump-probe -> temporal convolution with PSF
    ok = apply_forward_and_update(
        "pump_probe", fwd_psf, "temporal_psf",
        "y = conv(x, temporal_response)",
        sigma=2.5)
    if ok:
        fixes["pump_probe"] = "temporal_psf"
        total_fixed += 1
        print("    pump_probe: -> temporal response convolution")

    # Structured light -> fringe pattern projection
    ok = apply_forward_and_update(
        "structured_light", fwd_pattern_proj, "fringe_projection",
        "y = x * pattern (reflected fringe)")
    if ok:
        fixes["structured_light"] = "fringe_projection"
        total_fixed += 1
        print("    structured_light: -> fringe projection")

    # Light field & integral -> angular integration (downsample as proxy)
    for mod in ["light_field", "integral"]:
        ok = apply_forward_and_update(
            mod, fwd_psf, "angular_integration",
            "y = conv(x, aperture_psf) (angular integration)",
            sigma=3.0)
        if ok:
            fixes[mod] = "angular_integration"
            total_fixed += 1
            print(f"    {mod}: -> angular integration PSF")

    # ================================================================
    # GROUP 12: PSF parameter fixes (correct model, wrong sigma)
    # ================================================================
    print("\n[12] PSF parameter corrections")
    psf_fixes = {
        # Super-resolution microscopy (sharp PSFs)
        "sted": 0.3,         # STED nanoscopy: ~30nm resolution
        "palm_storm": 0.4,   # PALM/STORM: ~40nm
        "dna_paint": 0.4,    # DNA-PAINT: ~40nm
        "minflux": 0.2,      # MINFLUX: ~1nm tracking (sharpest)
        # Confocal (sharper than widefield)
        "confocal_3d": 0.8,
        "confocal_endomicroscopy": 0.9,
        "confocal_livecell": 0.8,
        "spinning_disk": 0.9,
        "ism": 0.6,          # Image scanning: sharper than confocal
        # Two-photon (between confocal and widefield)
        "two_photon": 1.0,
        "shg": 1.0,
        # TIRF (very thin optical section)
        "tirf": 0.7,
        # SIM (2x resolution improvement)
        "sim": 0.5,
        # Widefield (standard diffraction limit)
        "widefield": 1.5,
        "widefield_lowdose": 1.8,    # Low dose = slightly worse
        "flim": 1.2,
        # Probe microscopy (very sharp tip convolution)
        "afm": 0.3,
        "stm": 0.2,         # STM: atomic resolution
        "mfm": 0.5,
        "nsom": 0.4,        # Near-field: sub-diffraction
        # Electron microscopy
        "sem": 0.4,          # SEM: nanoscale
        "tem": 0.3,          # TEM: sub-nm
        "stem": 0.3,         # STEM: sub-nm
        # Optical (varies)
        "adaptive_optics": 0.6,     # AO corrects turbulence
        "coronagraphy": 1.2,
        "lucky_imaging": 0.8,
        "dic": 1.0,
        "acoustic_microscopy": 1.5,
        "sonar": 3.0,               # Sonar: poor resolution
        "weather_radar": 3.0,       # Radar: poor resolution
        "oct": 1.0,
        "octa": 1.0,
        "solar_imaging": 1.5,
        "shearography": 1.5,
        "eddy_current": 2.0,
        "active_thermography": 2.5, # Thermal: poor resolution
        "photometric_stereo": 1.0,
        "brachytherapy_img": 2.0,
        # CLEM
        "clem": 1.0,
    }

    for mod, sigma in psf_fixes.items():
        if mod in SKIP_MODS:
            continue
        ok = apply_forward_and_update(
            mod, fwd_psf, "psf",
            f"y = conv(x, gaussian(sigma={sigma}))",
            sigma=sigma)
        if ok:
            fixes[mod] = f"psf(sigma={sigma})"
            total_fixed += 1

    print(f"    Updated {len([m for m in psf_fixes if m in fixes])} PSF parameters")

    # ================================================================
    # GROUP 13: SPC -> compressive random projection
    # ================================================================
    print("\n[13] SPC: mask -> compressive projection")
    ok = apply_forward_and_update(
        "spc", fwd_compressive_proj, "compressive_projection",
        "y = Phi @ x (random Gaussian projection)",
        n_measurements=64)
    if ok:
        fixes["spc"] = "compressive_projection"
        total_fixed += 1
        print("    spc: -> compressive random projection (64 meas)")

    # Quantum modalities -> PSF (simplified)
    for mod in ["entangled_photon", "quantum_illumination"]:
        ok = apply_forward_and_update(
            mod, fwd_psf, "quantum_psf",
            "y = conv(x, quantum_psf)",
            sigma=2.0)
        if ok:
            fixes[mod] = "quantum_psf"
            total_fixed += 1
            print(f"    {mod}: -> quantum PSF")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print(f"DONE: Fixed {total_fixed} modalities")
    print("=" * 70)

    # Group by fix type
    by_type = {}
    for mod, fix in fixes.items():
        fix_type = fix.split("(")[0]
        by_type.setdefault(fix_type, []).append(mod)

    for fix_type, mods in sorted(by_type.items()):
        print(f"  {fix_type}: {len(mods)} modalities")
        for m in sorted(mods):
            print(f"    - {m}")

    return True


if __name__ == "__main__":
    main()
