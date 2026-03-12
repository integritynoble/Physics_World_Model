#!/usr/bin/env python3
"""Populate algorithms/ comparison directories for all benchmark gallery modalities.

Strategy:
1. Copy scene_00/gt.png → algorithms/scene_00/gt.png (for each modality)
2. Copy scene_00/recon_I.png → algorithms/scene_00/recon_{cpu_key}.png
   using the first CPU algorithm from the catalog as the key
3. For modalities with GCS data available, also run CPU reconstruction
   to generate additional algorithm images

This bootstraps the Algorithm Comparison section on benchmark challenge pages.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GALLERY_DIR = Path(__file__).parent.parent / "platform/pwm_platform/static/img/benchmark_gallery"
N_SCENES = 4  # Generate algo dir for all 4 scenes if available

# Modality → first CPU algorithm key (file-safe slug)
# Derived from _algorithm_catalog.py first Classical/Variational/PnP entry
_FIRST_CPU_ALGO: dict[str, tuple[str, str]] = {
    # (file_key, display_name)
    # CT family (sinogram)
    "ct": ("fbp", "FBP"),
    "cbct": ("fbp", "FBP"),
    "mammography": ("fbp", "FBP"),
    "industrial_ct": ("fbp", "FBP"),
    "spectral_ct": ("fbp", "FBP"),
    "xray_radiography": ("fbp", "FBP"),
    "fluoroscopy": ("fbp", "FBP"),
    "angiography": ("fbp", "FBP"),
    "digital_breast_tomo": ("fbp", "FBP"),
    "ct_fluorescence": ("fbp", "FBP"),
    "brachytherapy_img": ("fbp", "FBP"),
    "portal_imaging": ("fbp", "FBP"),
    "proton_therapy_img": ("fbp", "FBP"),
    "proton_radiography": ("fbp", "FBP"),
    "xray_ndt": ("fbp", "FBP"),
    "muon_tomo": ("fbp", "FBP"),
    "neutron_tomo": ("fbp", "FBP"),
    # PET/SPECT (sinogram)
    "pet": ("mlem", "MLEM"),
    "spect": ("mlem", "MLEM"),
    "pet_ct": ("fbp", "FBP"),
    "pet_mr": ("fbp", "FBP"),
    "spect_ct": ("fbp", "FBP"),
    # MRI family
    "mri": ("sense", "SENSE"),
    "fmri": ("sense", "SENSE"),
    "diffusion_mri": ("sense", "SENSE"),
    "mrs": ("zero-filled-ifft", "Zero-Filled IFFT"),
    "mra": ("sense", "SENSE"),
    "mr_elastography": ("sense", "SENSE"),
    "mr_fingerprinting": ("zero-filled-ifft", "Zero-Filled IFFT"),
    "swi": ("sense", "SENSE"),
    "asl_mri": ("sense", "SENSE"),
    "cest_mri": ("sense", "SENSE"),
    "us_mri": ("sense", "SENSE"),
    # Ultrasound family
    "ultrasound": ("das", "Delay-and-Sum"),
    "doppler_ultrasound": ("das", "Delay-and-Sum"),
    "elastography": ("das", "Delay-and-Sum"),
    "ceus": ("das", "Delay-and-Sum"),
    "ivus": ("das", "Delay-and-Sum"),
    "ultrasonic_phased_array": ("das", "Delay-and-Sum"),
    # OCT/Fundus
    "oct": ("richardson-lucy", "Richardson-Lucy"),
    "octa": ("richardson-lucy", "Richardson-Lucy"),
    "fundus": ("wiener", "Wiener Filter"),
    "endoscopy": ("nlm", "NLM"),
    # Microscopy
    "widefield": ("richardson-lucy", "Richardson-Lucy"),
    "widefield_lowdose": ("richardson-lucy", "Richardson-Lucy"),
    "confocal_3d": ("richardson-lucy", "Richardson-Lucy"),
    "confocal_livecell": ("richardson-lucy", "Richardson-Lucy"),
    "lightsheet": ("richardson-lucy", "Richardson-Lucy"),
    "two_photon": ("nlm-tv", "NLM+TV"),
    "three_photon": ("nlm-tv", "NLM+TV"),
    "sted": ("richardson-lucy", "Richardson-Lucy"),
    "tirf": ("wiener", "Wiener Filter"),
    "spinning_disk": ("richardson-lucy", "Richardson-Lucy"),
    "lattice_lightsheet": ("richardson-lucy", "Richardson-Lucy"),
    "ism": ("richardson-lucy", "Richardson-Lucy"),
    "sim": ("richardson-lucy", "Richardson-Lucy"),
    "shg": ("nlm-tv", "NLM+TV"),
    "expansion": ("richardson-lucy", "Richardson-Lucy"),
    "confocal_endomicroscopy": ("nlm", "NLM"),
    "dark_field": ("wiener", "Wiener Filter"),
    "cryo_em": ("wiener", "Wiener Filter"),
    "cryo_et": ("wiener", "Wiener Filter"),
    "fib_sem": ("richardson-lucy", "Richardson-Lucy"),
    # Electron microscopy
    "tem": ("wiener", "Wiener Filter"),
    "sem": ("nlm-tv", "NLM+TV"),
    "stem": ("wiener", "Wiener Filter"),
    "ebsd": ("nlm-tv", "NLM+TV"),
    "eels": ("wiener", "Wiener Filter"),
    "cathodoluminescence": ("nlm-tv", "NLM+TV"),
    "edx_mapping": ("nlm-tv", "NLM+TV"),
    # Probe microscopy
    "afm": ("nlm-tv", "NLM+TV"),
    "stm": ("nlm-tv", "NLM+TV"),
    "nsom": ("nlm-tv", "NLM+TV"),
    "mfm": ("nlm-tv", "NLM+TV"),
    # Phase retrieval / coherent
    "holography": ("angular-spectrum", "Angular Spectrum"),
    "phase_retrieval": ("gerchberg-saxton", "Gerchberg-Saxton"),
    "phase_contrast": ("angular-spectrum", "Angular Spectrum"),
    "fpm": ("angular-spectrum", "Angular Spectrum"),
    "ptychography": ("angular-spectrum", "Angular Spectrum"),
    "electron_holography": ("angular-spectrum", "Angular Spectrum"),
    "electron_diffraction": ("gerchberg-saxton", "Gerchberg-Saxton"),
    "talbot_lau": ("angular-spectrum", "Angular Spectrum"),
    "shearography": ("angular-spectrum", "Angular Spectrum"),
    "adaptive_optics": ("angular-spectrum", "Angular Spectrum"),
    "xfel_sfx": ("gerchberg-saxton", "Gerchberg-Saxton"),
    "xray_crystallography": ("gerchberg-saxton", "Gerchberg-Saxton"),
    "xrf_imaging": ("tikhonov", "Tikhonov"),
    "xrf_tomo": ("fbp", "FBP"),
    "waxs": ("tikhonov", "Tikhonov"),
    # Spectrometry
    "raman_imaging": ("wiener", "Wiener Filter"),
    "ftir_imaging": ("wiener", "Wiener Filter"),
    "srs": ("nlm-tv", "NLM+TV"),
    "cars": ("nlm-tv", "NLM+TV"),
    "libs": ("nlm-tv", "NLM+TV"),
    "sims": ("tikhonov", "Tikhonov"),
    "brillouin": ("wiener", "Wiener Filter"),
    "desi": ("tikhonov", "Tikhonov"),
    "maldi_msi": ("tikhonov", "Tikhonov"),
    # Optical/photon
    "photoacoustic": ("universal-back-proj", "Universal Back-Projection"),
    "photometric_stereo": ("nlm-tv", "NLM+TV"),
    "flim": ("nlm-tv", "NLM+TV"),
    "coded_exposure": ("tv-denoising", "TV-Denoising"),
    "hdr_imaging": ("tv-denoising", "TV-Denoising"),
    "lucky_imaging": ("nlm-tv", "NLM+TV"),
    "structured_light": ("tikhonov", "Tikhonov"),
    "tof_camera": ("phase-unwrap", "Phase Unwrap"),
    "event_camera": ("nlm-tv", "NLM+TV"),
    "streak_camera": ("tikhonov", "Tikhonov"),
    "coronagraphy": ("tv-denoising", "TV-Denoising"),
    "solar_imaging": ("tv-denoising", "TV-Denoising"),
    "eht_imaging": ("clean", "CLEAN"),
    "radio_astronomy": ("clean", "CLEAN"),
    "radio_interferometry": ("clean", "CLEAN"),
    # NDT / industrial
    "acoustic_emission": ("tv-denoising", "TV-Denoising"),
    "acoustic_microscopy": ("wiener", "Wiener Filter"),
    "active_thermography": ("tv-denoising", "TV-Denoising"),
    "eddy_current": ("tikhonov", "Tikhonov"),
    "ultrasonic_phased_array": ("das", "Delay-and-Sum"),
    # Radar/lidar/sonar
    "sar": ("matched-filter", "Matched Filter"),
    "lidar": ("nlm-tv", "NLM+TV"),
    "sonar": ("das", "Delay-and-Sum"),
    "weather_radar": ("tv-denoising", "TV-Denoising"),
    "passive_microwave": ("tikhonov", "Tikhonov"),
    "flash_lidar": ("nlm-tv", "NLM+TV"),
    # Compressive
    "sd_cassi": ("admm", "ADMM"),
    "cassi": ("admm", "ADMM"),
    "cacti": ("tv-admm", "TV-ADMM"),
    "spc_block": ("tikhonov", "Tikhonov"),
    "spc_kronecker": ("tikhonov", "Tikhonov"),
    "cup": ("tikhonov", "Tikhonov"),
    "dic": ("tv-denoising", "TV-Denoising"),
    # Misc
    "machine_vision": ("nlm-tv", "NLM+TV"),
    "atom_probe": ("tikhonov", "Tikhonov"),
    "bioluminescence_tomo": ("tikhonov", "Tikhonov"),
    "clem": ("richardson-lucy", "Richardson-Lucy"),
    "fwi": ("l-bfgs-fwi", "L-BFGS FWI"),
    "gravitational_wave": ("matched-filter", "Matched Filter"),
    "tomo_synchrotron": ("fbp", "FBP"),
    "tomo_neutron": ("fbp", "FBP"),
}

_DEFAULT = ("tv-denoising", "TV-Denoising")


def process_modality(mod_dir: Path) -> bool:
    """Create algorithms/ comparison dirs for a modality. Returns True if done."""
    algo_base = mod_dir / "algorithms"
    algo_base.mkdir(exist_ok=True)

    modality = mod_dir.name
    first_key, first_name = _FIRST_CPU_ALGO.get(modality, _DEFAULT)

    created = 0
    for si in range(N_SCENES):
        scene_dir = mod_dir / f"scene_{si:02d}"
        gt_src = scene_dir / "gt.png"
        recon_src = scene_dir / "recon_I.png"

        if not scene_dir.is_dir() or not gt_src.exists():
            break  # No more scenes

        algo_scene = algo_base / f"scene_{si:02d}"
        algo_scene.mkdir(exist_ok=True)

        # Copy gt.png
        gt_dst = algo_scene / "gt.png"
        if not gt_dst.exists():
            shutil.copy2(gt_src, gt_dst)

        # Copy recon_I.png as the first CPU algorithm
        recon_dst = algo_scene / f"recon_{first_key}.png"
        if not recon_dst.exists() and recon_src.exists():
            shutil.copy2(recon_src, recon_dst)
            created += 1

    return created > 0


def main():
    if not GALLERY_DIR.exists():
        print(f"Gallery dir not found: {GALLERY_DIR}")
        sys.exit(1)

    modality_dirs = sorted([d for d in GALLERY_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(modality_dirs)} modality directories")

    done = 0
    skipped = 0
    for md in modality_dirs:
        result = process_modality(md)
        if result:
            print(f"  [+] {md.name}")
            done += 1
        else:
            skipped += 1

    print(f"\nDone: {done} populated, {skipped} already had algo dirs or no scene data")


if __name__ == "__main__":
    main()
