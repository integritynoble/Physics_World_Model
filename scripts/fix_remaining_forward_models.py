"""
Fix remaining modalities with near-identity or incorrect forward models.

Targets:
  1. DIC -> gradient operator (Nomarski shearing)
  2. Photometric stereo -> Lambertian shading
  3. OCT -> coherence gate (anisotropic axial/lateral PSF)
  4. Coronagraphy -> PSF + stellar leakage
  5. Gaussian splatting -> rendering DOF + vignetting
  6. Brachytherapy -> X-ray transmission + scatter
  7. OCTA -> motion contrast (decorrelation)
  8. Weather radar -> range-dependent PSF + attenuation
  9. Sonar -> beam pattern + depth attenuation
  10. Solar imaging -> atmospheric seeing + limb darkening

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage

DATA = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\datasets\benchmark")


def to_uint8(arr):
    arr = np.nan_to_num(arr)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def save_img(arr, path):
    if arr.ndim == 2:
        Image.fromarray(to_uint8(arr), "L").save(str(path))
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        Image.fromarray(to_uint8(arr), "RGB").save(str(path))


def update_mod(modality, fwd_func, fwd_type, fwd_desc):
    std = DATA / modality / "standard"
    h5s = sorted(std.glob("standard_*.h5"))
    if not h5s:
        print(f"  SKIP {modality}: no h5 files")
        return
    n = 0
    for h5p in h5s:
        with h5py.File(str(h5p), "r") as f:
            x = f["x_true"][:]
        y = fwd_func(x, seed=42 + n * 137)
        with h5py.File(str(h5p), "a") as f:
            if "y_ideal" in f:
                del f["y_ideal"]
            f.create_dataset("y_ideal", data=y, compression="gzip")
            if "H_params" in f:
                f["H_params"].attrs["forward_type"] = fwd_type
        # Images
        stem = h5p.stem
        parts = stem.rsplit("_", 1)
        sid = f"sample_{parts[1]}" if len(parts) == 2 and parts[1].isdigit() else f"sample_{n:02d}"
        img_dir = std / "images" / sid
        img_dir.mkdir(parents=True, exist_ok=True)
        save_img(x, img_dir / "x_true.png")
        save_img(y, img_dir / "y_ideal.png")
        n += 1
    # Update metadata
    meta_p = std / "metadata.json"
    if meta_p.exists():
        with open(meta_p) as f:
            meta = json.load(f)
        meta["forward_type"] = fwd_type
        meta["forward_model"] = fwd_desc
        with h5py.File(str(h5s[0]), "r") as f:
            meta["y_ideal_shape"] = list(f["y_ideal"].shape)
        meta["date_built"] = "2026-03-14"
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2)
    spec_p = std / "spec.json"
    if spec_p.exists():
        with open(spec_p) as f:
            spec = json.load(f)
        spec["forward_type"] = fwd_type
        with open(spec_p, "w") as f:
            json.dump(spec, f, indent=2)
    print(f"  {modality}: {n} samples -> {fwd_type}")


# ---------------------------------------------------------------
# Forward model functions
# ---------------------------------------------------------------

def fwd_dic(x, seed=42):
    """DIC: Nomarski shearing = directional gradient of OPD."""
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, np.pi)
    dx = ndimage.sobel(x, axis=1).astype(np.float32)
    dy = ndimage.sobel(x, axis=0).astype(np.float32)
    grad = np.cos(angle) * dx + np.sin(angle) * dy
    y = 0.5 + 0.3 * grad / (np.abs(grad).max() + 1e-10)
    return y.astype(np.float32)


def fwd_photometric_stereo(x, seed=42):
    """Lambertian shading: single directional illumination."""
    rng = np.random.RandomState(seed)
    dx = ndimage.sobel(x, axis=1).astype(np.float32)
    dy = ndimage.sobel(x, axis=0).astype(np.float32)
    norm = np.sqrt(dx**2 + dy**2 + 1)
    nx, ny, nz = -dx / norm, -dy / norm, 1.0 / norm
    theta = rng.uniform(np.pi / 6, np.pi / 3)
    phi = rng.uniform(0, 2 * np.pi)
    lx = np.cos(theta) * np.cos(phi)
    ly = np.cos(theta) * np.sin(phi)
    lz = np.sin(theta)
    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    y = x * ndotl
    return y.astype(np.float32)


def fwd_oct(x, seed=42):
    """OCT: axial coherence gate + lateral scan PSF + speckle."""
    rng = np.random.RandomState(seed)
    y = ndimage.gaussian_filter(x, sigma=[0.5, 2.0])
    H, W = x.shape
    speckle_pattern = ndimage.gaussian_filter(rng.randn(H, W).astype(np.float32), sigma=1.0)
    speckle = 1.0 + 0.2 * speckle_pattern
    y = y * speckle
    return np.clip(y, 0, None).astype(np.float32)


def fwd_coronagraphy(x, seed=42):
    """Coronagraph: PSF + stellar leakage speckle halo."""
    rng = np.random.RandomState(seed)
    H, W = x.shape
    y = ndimage.gaussian_filter(x, sigma=2.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = H // 2, W // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2) + 1e-10
    leakage = 0.1 * np.exp(-r / (W * 0.15)) * np.cos(6 * np.arctan2(yy - cy, xx - cx))
    y = y + leakage
    return np.clip(y, 0, None).astype(np.float32)


def fwd_gaussian_splatting(x, seed=42):
    """3D rendering: depth-of-field blur + vignetting."""
    rng = np.random.RandomState(seed)
    dof_sigma = rng.uniform(1.5, 3.0)
    y = ndimage.gaussian_filter(x, sigma=dof_sigma)
    H, W = x.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = H / 2, W / 2
    r2 = ((xx - cx) / (W / 2))**2 + ((yy - cy) / (H / 2))**2
    vignette = np.clip(1.0 - 0.3 * r2, 0.5, 1.0)
    y = y * vignette
    return y.astype(np.float32)


def fwd_brachytherapy(x, seed=42):
    """Brachytherapy: X-ray transmission + scatter."""
    y = np.exp(-0.015 * x)
    scatter = ndimage.gaussian_filter(y, sigma=5.0) * 0.15
    y = y + scatter
    return y.astype(np.float32)


def fwd_octa(x, seed=42):
    """OCTA: OCT PSF + motion decorrelation contrast."""
    y = ndimage.gaussian_filter(x, sigma=[0.5, 1.5])
    edges = ndimage.gaussian_laplace(x, sigma=1.0)
    y = y + 0.3 * np.abs(edges)
    return np.clip(y, 0, None).astype(np.float32)


def fwd_weather_radar(x, seed=42):
    """Weather radar: range-dependent PSF + atmospheric attenuation."""
    H, W = x.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = H // 2, W // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2) + 1
    y = np.zeros_like(x)
    for sigma_val in [1.0, 2.0, 3.0, 4.0]:
        ring_mask = ((r > (sigma_val - 0.5) * W / 8) & (r <= (sigma_val + 0.5) * W / 8)).astype(np.float32)
        blurred = ndimage.gaussian_filter(x, sigma=sigma_val)
        y += blurred * ring_mask
    center_mask = (r <= W / 16).astype(np.float32)
    y += ndimage.gaussian_filter(x, sigma=1.0) * center_mask
    attenuation = np.exp(-0.003 * r)
    y = y * attenuation
    return np.clip(y, 0, None).astype(np.float32)


def fwd_sonar(x, seed=42):
    """Sonar: anisotropic beam + depth attenuation."""
    H, W = x.shape
    y = ndimage.gaussian_filter(x, sigma=[1.5, 4.0])
    shadow_mask = 1.0 - 0.3 * (np.arange(H, dtype=np.float32) / H)[:, None]
    y = y * shadow_mask
    return y.astype(np.float32)


def fwd_solar(x, seed=42):
    """Solar imaging: atmospheric seeing + limb darkening."""
    y = ndimage.gaussian_filter(x, sigma=2.0)
    H, W = x.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = H // 2, W // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    solar_r = min(H, W) * 0.4
    limb = np.clip(1.0 - 0.3 * (r / solar_r)**2, 0.5, 1.0)
    y = y * limb
    return y.astype(np.float32)


def fwd_shearography(x, seed=42):
    """Shearography: lateral shearing interferometry = spatial derivative."""
    rng = np.random.RandomState(seed)
    shear_px = rng.choice([2, 3, 4])
    # Shearing: difference between original and laterally shifted
    shifted = np.roll(x, shear_px, axis=1)
    y = 0.5 + 0.4 * (x - shifted) / (np.abs(x - shifted).max() + 1e-10)
    return y.astype(np.float32)


def fwd_eddy_current(x, seed=42):
    """Eddy current: electromagnetic induction with depth-dependent sensitivity."""
    rng = np.random.RandomState(seed)
    # Eddy currents penetrate to skin depth (exponential decay with depth)
    # Simulate as depth-weighted blurred image
    y = ndimage.gaussian_filter(x, sigma=3.0)
    # Add phase shift (imaginary component of impedance)
    phase = ndimage.gaussian_filter(ndimage.sobel(x, axis=0), sigma=2.0)
    y = y + 0.15 * phase
    return y.astype(np.float32)


def fwd_active_thermography(x, seed=42):
    """Active thermography: thermal diffusion PSF (very broad)."""
    # Heat diffuses isotropically - very broad PSF
    y = ndimage.gaussian_filter(x, sigma=4.0)
    # Thermal contrast decays with time (simulated by reduced contrast)
    y = 0.3 + 0.7 * y / (y.max() + 1e-10)
    return y.astype(np.float32)


def main():
    print("Fixing remaining forward models...")

    update_mod("dic", fwd_dic, "dic_gradient",
               "y = 0.5 + grad(OPD) along shear axis (Nomarski)")

    update_mod("photometric_stereo", fwd_photometric_stereo, "lambertian_shading",
               "y = albedo * max(n . light_dir, 0)")

    update_mod("oct", fwd_oct, "oct_coherence_gate",
               "y = speckle * PSF_aniso(x); axial=0.5, lateral=2.0")

    update_mod("coronagraphy", fwd_coronagraphy, "coronagraph_psf",
               "y = PSF(x) + stellar_leakage(r,theta)")

    update_mod("gaussian_splatting", fwd_gaussian_splatting, "rendering_dof",
               "y = vignette * depth_blur(x)")

    update_mod("brachytherapy_img", fwd_brachytherapy, "xray_transmission_scatter",
               "y = exp(-mu*x) + scatter(x)")

    update_mod("octa", fwd_octa, "octa_motion_contrast",
               "y = OCT_psf(x) + motion_decorrelation(x)")

    update_mod("weather_radar", fwd_weather_radar, "range_dependent_psf",
               "y = attenuation(r) * range_psf(x)")

    update_mod("sonar", fwd_sonar, "sonar_beam",
               "y = depth_attenuation * beam_pattern(x)")

    update_mod("solar_imaging", fwd_solar, "atmospheric_seeing",
               "y = limb_darkening * seeing_psf(x)")

    update_mod("shearography", fwd_shearography, "shearing_interferometry",
               "y = x - shift(x, shear_px) (lateral shearing)")

    update_mod("eddy_current", fwd_eddy_current, "electromagnetic_induction",
               "y = skin_depth_psf(x) + phase_shift(x)")

    update_mod("active_thermography", fwd_active_thermography, "thermal_diffusion",
               "y = thermal_psf(x, sigma=4.0) (heat diffusion)")

    print("\nDone - all remaining forward models fixed.")


if __name__ == "__main__":
    main()
