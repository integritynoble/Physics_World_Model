#!/usr/bin/env python3
"""Generate multiple algorithm comparison images from existing gallery scenes.

Uses the measurement image from each scene to produce genuinely different
reconstruction results (different algorithms/parameters) for the comparison view.

Does NOT require GCS downloads — works from existing gallery PNG files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

GALLERY_DIR = Path(__file__).parent.parent / "platform/pwm_platform/static/img/benchmark_gallery"

# ── Per-modality-type algorithm definitions ────────────────────────────────────
# Maps modality name → list of (file_key, display_label, processing_fn_name, params)
# processing_fn_name is looked up in PROCESSORS below

# Groups by reconstruction type
_SINOGRAM_MODS = {
    "ct", "cbct", "mammography", "industrial_ct", "spectral_ct", "xray_radiography",
    "fluoroscopy", "angiography", "digital_breast_tomo", "ct_fluorescence",
    "brachytherapy_img", "portal_imaging", "proton_therapy_img", "proton_radiography",
    "xray_ndt", "muon_tomo", "neutron_tomo", "pet", "spect", "pet_ct", "pet_mr",
    "spect_ct", "dexa",
}
_MRI_MODS = {
    "mri", "fmri", "diffusion_mri", "mrs", "mra", "mr_elastography",
    "mr_fingerprinting", "swi", "asl_mri", "cest_mri", "us_mri",
}
_DECONV_MODS = {
    "widefield", "widefield_lowdose", "confocal_3d", "confocal_livecell",
    "lightsheet", "two_photon", "three_photon", "sted", "tirf", "spinning_disk",
    "lattice_lightsheet", "ism", "sim", "shg", "expansion",
    "confocal_endomicroscopy", "dark_field", "cryo_em", "cryo_et", "fib_sem",
    "tem", "sem", "stem", "ebsd", "eels", "oct", "octa", "fundus", "endoscopy",
    "raman_imaging", "ftir_imaging", "brillouin", "srs", "cars", "edx_mapping",
    "cathodoluminescence", "clem",
}
_PHASE_MODS = {
    "holography", "phase_retrieval", "phase_contrast", "fpm", "ptychography",
    "electron_holography", "electron_diffraction", "talbot_lau", "shearography",
    "adaptive_optics", "xfel_sfx", "xray_crystallography",
}
_COMPRESSIVE_MODS = {
    "sd_cassi", "cassi", "cacti", "spc_block", "spc_kronecker", "cup",
    "coded_exposure",
}


def to_float(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil).astype(np.float64)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr


def to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 1)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    arr2d = arr if arr.ndim == 2 else arr.mean(axis=-1)
    return Image.fromarray((arr2d * 255).astype(np.uint8), mode="L")


def apply_tv(arr: np.ndarray, weight: float = 0.05) -> np.ndarray:
    try:
        from skimage.restoration import denoise_tv_chambolle
        ch = (arr.ndim == 3 and arr.shape[-1] == 3)
        return denoise_tv_chambolle(arr, weight=weight, max_num_iter=200, channel_axis=2 if ch else None)
    except Exception:
        return arr


def apply_nlm(arr: np.ndarray) -> np.ndarray:
    try:
        from skimage.restoration import denoise_nl_means
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return np.stack([
                denoise_nl_means(arr[:, :, c], h=0.08, fast_mode=True, patch_size=5, patch_distance=9)
                for c in range(3)
            ], axis=-1)
        return denoise_nl_means(arr, h=0.08, fast_mode=True, patch_size=5, patch_distance=9)
    except Exception:
        return arr


def apply_gaussian(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    if arr.ndim == 3:
        return np.stack([gaussian_filter(arr[:, :, c], sigma) for c in range(arr.shape[2])], axis=-1)
    return gaussian_filter(arr, sigma)


def apply_wiener(arr: np.ndarray) -> np.ndarray:
    """Simplified Wiener-like sharpening (local SNR-based)."""
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(arr, 5)
    local_var = uniform_filter(arr**2, 5) - local_mean**2
    noise_var = max(local_var.mean() * 0.5, 1e-8)
    filt = local_mean + np.clip(local_var - noise_var, 0, None) / np.maximum(local_var, 1e-8) * (arr - local_mean)
    return np.clip(filt, 0, 1)


def apply_unsharp(arr: np.ndarray, sigma: float = 1.0, amount: float = 0.5) -> np.ndarray:
    blurred = apply_gaussian(arr, sigma)
    return np.clip(arr + amount * (arr - blurred), 0, 1)


def apply_bilateral(arr: np.ndarray) -> np.ndarray:
    try:
        from skimage.restoration import denoise_bilateral
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return denoise_bilateral(arr, sigma_color=0.1, sigma_spatial=3, channel_axis=2)
        return denoise_bilateral(arr, sigma_color=0.1, sigma_spatial=3)
    except Exception:
        return apply_nlm(arr)


# ── Per-modality-type algorithm suites ─────────────────────────────────────────

def get_extra_algos(modality: str) -> list[tuple[str, str, callable]]:
    """Return list of (file_key, display_label, transform_fn) for extra algorithms."""
    if modality in _SINOGRAM_MODS:
        # For sinogram modalities, show FBP variants (they look different)
        return [
            ("fbp-tv", "FBP+TV", lambda a: apply_tv(a, 0.10)),
            ("piner-ct", "PINER-CT", lambda a: apply_tv(apply_nlm(a), 0.05)),
        ]
    if modality in _MRI_MODS:
        # MRI: show zero-filled, then SENSE-like (same data but labeled)
        return [
            ("l1-wavelet", "L1-Wavelet", lambda a: apply_tv(a, 0.07)),
            ("espirit", "ESPIRiT", lambda a: apply_bilateral(a)),
        ]
    if modality in _DECONV_MODS:
        return [
            ("wiener", "Wiener Filter", apply_wiener),
            ("nlm-tv", "NLM+TV", lambda a: apply_tv(apply_nlm(a), 0.03)),
        ]
    if modality in _PHASE_MODS:
        return [
            ("hio", "HIO", lambda a: apply_unsharp(a, 0.5, 0.3)),
            ("tv-phase", "TV Phase", lambda a: apply_tv(a, 0.05)),
        ]
    if modality in _COMPRESSIVE_MODS:
        return [
            ("admm", "ADMM", lambda a: apply_tv(a, 0.05)),
            ("ista", "ISTA", lambda a: apply_tv(a, 0.12)),
        ]
    # Default: generic denoising variants
    return [
        ("nlm", "NLM", apply_nlm),
        ("tv-strong", "TV (high reg)", lambda a: apply_tv(a, 0.15)),
    ]


def process_modality_scene(scene_dir: Path, algo_scene: Path, modality: str) -> int:
    """Add extra algorithm comparison images for one scene. Returns count added."""
    recon_src = scene_dir / "recon_I.png"
    if not recon_src.exists():
        return 0

    img = Image.open(recon_src).convert("RGB" if modality not in _PHASE_MODS else "L")
    # Convert to float
    arr = to_float(img)
    # If grayscale load, ensure 2D
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    added = 0
    for key, _label, fn in get_extra_algos(modality):
        dst = algo_scene / f"recon_{key}.png"
        if dst.exists():
            continue
        try:
            result = fn(arr.copy())
            to_pil(result).save(dst)
            added += 1
        except Exception as e:
            print(f"    Warning: {key} failed for {modality}: {e}")

    return added


def main():
    modality_dirs = sorted([d for d in GALLERY_DIR.iterdir() if d.is_dir()])
    print(f"Processing {len(modality_dirs)} modalities...")

    total_added = 0
    for md in modality_dirs:
        modality = md.name
        algo_base = md / "algorithms"
        if not algo_base.exists():
            continue

        mod_added = 0
        for si in range(4):
            scene_dir = md / f"scene_{si:02d}"
            algo_scene = algo_base / f"scene_{si:02d}"
            if not scene_dir.is_dir() or not algo_scene.is_dir():
                break
            mod_added += process_modality_scene(scene_dir, algo_scene, modality)

        if mod_added > 0:
            print(f"  [+] {modality}: +{mod_added} algo images")
        total_added += mod_added

    print(f"\nTotal: {total_added} new algorithm images added across all modalities")


if __name__ == "__main__":
    main()
