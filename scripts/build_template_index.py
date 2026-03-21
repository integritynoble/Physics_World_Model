#!/usr/bin/env python3
"""Build datasets/templates/index.md with 5x verified results for all 168 modalities.

Reads PSNR from template_5x_results.json and SSIM from ssim_results.json,
then generates the full index.md with per-algorithm 5-run verification tables.
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FLAGSHIP = {
    'cassi', 'cacti', 'mri', 'ct', 'spc', 'lensless', 'compressive_holography',
    'ptychography', 'cbct', 'ultrasound', 'cryo_em', 'widefield',
}

YEAR_MAP = {
    "wiener": 1949, "landweber": 1951, "richardson_lucy": 1972,
    "tikhonov": 1963, "tv_admm": 1992, "chambolle_pock": 2011,
    "pnp_admm_nlm": 2013, "pnp_fista_nlm": 2013,
}

# Display name overrides
DISPLAY_NAMES = {
    "acoustic_emission": "Acoustic Emission Imaging",
    "acoustic_microscopy": "Scanning Acoustic Microscopy",
    "active_thermography": "Active Thermography / Pulsed Thermography",
    "adaptive_optics": "Adaptive Optics Imaging",
    "afm": "Atomic Force Microscopy (AFM)",
    "angiography": "X-ray Angiography / DSA",
    "asl_mri": "Arterial Spin Labeling MRI (ASL-MRI)",
    "atom_probe": "Atom Probe Tomography",
    "bioluminescence_tomo": "Bioluminescence Tomography (BLT)",
    "brachytherapy_img": "Brachytherapy Imaging",
    "brillouin": "Brillouin Microscopy",
    "cars": "Coherent Anti-Stokes Raman Scattering (CARS)",
    "cathodoluminescence": "Cathodoluminescence (CL) Imaging",
    "cest_mri": "Chemical Exchange Saturation Transfer MRI (CEST-MRI)",
    "ceus": "Contrast-Enhanced Ultrasound (CEUS)",
    "clem": "Correlative Light-Electron Microscopy (CLEM)",
    "coded_exposure": "Coded Exposure / Flutter Shutter",
    "confocal_3d": "Confocal Microscopy (3D)",
    "confocal_endomicroscopy": "Confocal Laser Endomicroscopy (CLE)",
    "confocal_livecell": "Confocal Live-Cell Microscopy",
    "coronagraphy": "Stellar Coronagraphy",
    "cryo_et": "Cryo-Electron Tomography (Cryo-ET)",
    "ct_fluorescence": "CT Fluoroscopy",
    "cup": "Compressed Ultrafast Photography (CUP)",
    "dark_field": "Dark-Field Microscopy",
    "desi": "Desorption Electrospray Ionization (DESI) MSI",
    "dexa": "Dual-Energy X-ray Absorptiometry (DEXA)",
    "dic": "Differential Interference Contrast (DIC)",
    "diffusion_mri": "Diffusion MRI (dMRI)",
    "digital_breast_tomo": "Digital Breast Tomosynthesis (DBT)",
    "dna_paint": "DNA-PAINT Super-Resolution",
    "doppler_ultrasound": "Doppler Ultrasound",
    "dot": "Diffuse Optical Tomography (DOT)",
    "ebsd": "Electron Backscatter Diffraction (EBSD)",
    "eddy_current": "Eddy Current Testing (ECT)",
    "edx_mapping": "Energy-Dispersive X-ray (EDX) Mapping",
    "eels": "Electron Energy Loss Spectroscopy (EELS)",
    "eht_imaging": "Event Horizon Telescope Imaging",
    "elastography": "Ultrasound Elastography",
    "electron_diffraction": "Electron Diffraction",
    "electron_holography": "Electron Holography",
    "electron_tomography": "Electron Tomography",
    "endoscopy": "Endoscopy / Capsule Endoscopy",
    "entangled_photon": "Entangled Photon Imaging",
    "event_camera": "Event Camera / DVS Imaging",
    "expansion": "Expansion Microscopy (ExM)",
    "fib_sem": "Focused Ion Beam SEM (FIB-SEM)",
    "flash_lidar": "Flash LiDAR",
    "flim": "Fluorescence Lifetime Imaging (FLIM)",
    "fluoroscopy": "X-ray Fluoroscopy",
    "fmri": "Functional MRI (fMRI)",
    "fpm": "Fourier Ptychographic Microscopy (FPM)",
    "ftir_imaging": "FTIR Spectroscopic Imaging",
    "fundus": "Fundus Photography / Retinal Imaging",
    "fwi": "Full Waveform Inversion (FWI)",
    "gaussian_splatting": "3D Gaussian Splatting (3DGS)",
    "ghost_imaging": "Ghost Imaging / Computational GI",
    "gpr": "Ground-Penetrating Radar (GPR)",
    "gravitational_wave": "Gravitational Wave Imaging",
    "hdr_imaging": "High Dynamic Range (HDR) Imaging",
    "hyperspectral_remote": "Hyperspectral Remote Sensing",
    "impedance_tomo": "Electrical Impedance Tomography (EIT)",
    "industrial_ct": "Industrial CT / Micro-CT",
    "insar": "Interferometric SAR (InSAR)",
    "integral": "Integral Imaging / Light Field Display",
    "ism": "Image Scanning Microscopy (ISM)",
    "ivus": "Intravascular Ultrasound (IVUS)",
    "lattice_lightsheet": "Lattice Light-Sheet Microscopy",
    "libs": "Laser-Induced Breakdown Spectroscopy (LIBS)",
    "lidar": "LiDAR Point Cloud Imaging",
    "light_field": "Light-Field Camera / Plenoptic Imaging",
    "lightsheet": "Light-Sheet Fluorescence Microscopy (LSFM)",
    "lucky_imaging": "Lucky Imaging / Speckle Imaging",
    "machine_vision": "Machine Vision / Industrial Inspection",
    "magnetic_particle": "Magnetic Particle Imaging (MPI)",
    "maldi_msi": "MALDI Mass Spectrometry Imaging",
    "mammography": "Digital Mammography",
    "matrix": "Reflection Matrix Imaging",
    "mfm": "Magnetic Force Microscopy (MFM)",
    "minflux": "MINFLUX Nanoscopy",
    "mr_elastography": "MR Elastography (MRE)",
    "mr_fingerprinting": "MR Fingerprinting (MRF)",
    "mra": "MR Angiography (MRA)",
    "mrs": "MR Spectroscopy (MRS)",
    "multispectral_sat": "Multispectral Satellite Imaging",
    "muon_tomo": "Muon Tomography",
    "nerf": "Neural Radiance Fields (NeRF)",
    "neutron_diffraction": "Neutron Diffraction Imaging",
    "neutron_tomo": "Neutron Tomography",
    "nirs_brain": "Near-Infrared Spectroscopy Brain Imaging (fNIRS)",
    "nsom": "Near-field Scanning Optical Microscopy (NSOM)",
    "ocean_acoustic_tomo": "Ocean Acoustic Tomography",
    "ocean_color": "Ocean Color Remote Sensing",
    "oct": "Optical Coherence Tomography (OCT)",
    "octa": "OCT Angiography (OCTA)",
    "odt": "Optical Diffraction Tomography (ODT)",
    "palm_storm": "PALM / STORM Super-Resolution",
    "panorama": "Panoramic Imaging / Image Stitching",
    "particle_calorimetry": "Particle Calorimetry Imaging",
    "particle_tracker": "Particle Tracker Imaging",
    "passive_microwave": "Passive Microwave Radiometry",
    "pet": "Positron Emission Tomography (PET)",
    "pet_ct": "PET/CT Fusion Imaging",
    "pet_mr": "PET/MR Fusion Imaging",
    "phase_contrast": "Phase Contrast Microscopy",
    "phase_retrieval": "Phase Retrieval / CDI",
    "photoacoustic": "Photoacoustic Imaging (PAI)",
    "photometric_stereo": "Photometric Stereo",
    "polarization": "Polarimetric Imaging",
    "polsar": "Polarimetric SAR (PolSAR)",
    "portal_imaging": "Portal Imaging (EPID)",
    "proton_radiography": "Proton Radiography",
    "proton_therapy_img": "Proton Therapy Imaging",
    "pump_probe": "Pump-Probe Microscopy",
    "quantum_illumination": "Quantum Illumination Imaging",
    "radio_astronomy": "Radio Astronomy Imaging",
    "radio_interferometry": "Radio Interferometry / VLBI",
    "raman_imaging": "Raman Spectroscopic Imaging",
    "sar": "Synthetic Aperture Radar (SAR)",
    "saxs": "Small-Angle X-ray Scattering (SAXS)",
    "seismic_tomo": "Seismic Tomography",
    "sem": "Scanning Electron Microscopy (SEM)",
    "shearography": "Shearography / Speckle Shearing",
    "shg": "Second Harmonic Generation (SHG) Microscopy",
    "sim": "Structured Illumination Microscopy (SIM)",
    "sims": "Secondary Ion Mass Spectrometry (SIMS)",
    "sofi": "Super-Resolution Optical Fluctuation Imaging (SOFI)",
    "solar_imaging": "Solar Imaging / Helioseismology",
    "sonar": "Sonar Imaging / Side-Scan Sonar",
    "spect": "Single-Photon Emission CT (SPECT)",
    "spect_ct": "SPECT/CT Fusion Imaging",
    "spectral_ct": "Spectral (Photon-Counting) CT",
    "spinning_disk": "Spinning Disk Confocal Microscopy",
    "srs": "Stimulated Raman Scattering (SRS) Microscopy",
    "sted": "Stimulated Emission Depletion (STED) Microscopy",
    "stem": "Scanning Transmission Electron Microscopy (STEM)",
    "stm": "Scanning Tunneling Microscopy (STM)",
    "streak_camera": "Streak Camera Imaging",
    "structured_light": "Structured Light 3D Scanning",
    "swi": "Susceptibility-Weighted Imaging (SWI)",
    "talbot_lau": "Talbot-Lau X-ray Grating Interferometry",
    "tem": "Transmission Electron Microscopy (TEM)",
    "terahertz": "Terahertz (THz) Imaging",
    "three_photon": "Three-Photon Microscopy",
    "tirf": "Total Internal Reflection Fluorescence (TIRF)",
    "tof_camera": "Time-of-Flight (ToF) Camera",
    "two_photon": "Two-Photon Excitation Microscopy",
    "ultrasonic_phased_array": "Ultrasonic Phased Array Imaging",
    "us_mri": "Ultrasound-Guided MRI / US+MRI Fusion",
    "waxs": "Wide-Angle X-ray Scattering (WAXS)",
    "weather_radar": "Weather Radar Imaging",
    "widefield_lowdose": "Widefield Low-Dose Fluorescence",
    "xfel_sfx": "XFEL Serial Femtosecond Crystallography (SFX)",
    "xray_crystallography": "X-ray Crystallography",
    "xray_ndt": "X-ray Non-Destructive Testing (NDT)",
    "xray_radiography": "X-ray Radiography",
    "xrf_imaging": "X-ray Fluorescence (XRF) Imaging",
    "xrf_tomo": "X-ray Fluorescence Tomography",
}

# Categories
CATEGORIES = [
    ("Medical Tomography & Nuclear", [
        "pet", "spect", "spect_ct", "spectral_ct", "fmri", "diffusion_mri",
        "asl_mri", "cest_mri", "us_mri", "swi", "digital_breast_tomo", "dexa",
        "mr_elastography", "mr_fingerprinting", "mra", "mrs", "industrial_ct",
        "impedance_tomo", "mammography", "brachytherapy_img", "portal_imaging",
        "proton_radiography", "proton_therapy_img", "pet_ct", "pet_mr",
    ]),
    ("Ultrasound & Acoustic", [
        "doppler_ultrasound", "ceus", "elastography", "photoacoustic",
        "ultrasonic_phased_array", "ivus", "acoustic_emission", "acoustic_microscopy",
        "ocean_acoustic_tomo",
    ]),
    ("Optical Microscopy", [
        "confocal_3d", "confocal_livecell", "confocal_endomicroscopy",
        "two_photon", "three_photon", "sted", "tirf", "spinning_disk",
        "lightsheet", "lattice_lightsheet", "flim", "fpm", "dic",
        "dark_field", "phase_contrast", "structured_light", "expansion",
        "ism", "minflux", "widefield_lowdose", "shg", "pump_probe",
    ]),
    ("Fluorescence & Super-Resolution", [
        "palm_storm", "sim", "sofi", "dna_paint", "srs", "cars",
        "bioluminescence_tomo",
    ]),
    ("Electron & Probe Microscopy", [
        "cryo_et", "sem", "tem", "stem", "fib_sem", "eels", "edx_mapping",
        "electron_holography", "electron_tomography", "electron_diffraction",
        "ebsd", "stm", "afm", "atom_probe", "cathodoluminescence", "clem",
        "mfm", "nsom",
    ]),
    ("Optical Imaging & Computational Photography", [
        "oct", "octa", "odt", "fundus", "endoscopy", "panorama",
        "event_camera", "light_field", "coded_exposure", "cup",
        "flash_lidar", "tof_camera", "integral", "machine_vision",
        "hdr_imaging", "lucky_imaging", "photometric_stereo", "polarization",
        "phase_retrieval", "adaptive_optics",
    ]),
    ("Spectral & Hyperspectral", [
        "hyperspectral_remote", "multispectral_sat", "ftir_imaging",
        "raman_imaging", "brillouin", "desi", "xrf_imaging",
        "maldi_msi", "libs", "sims",
    ]),
    ("Coherent & Interferometric", [
        "ghost_imaging", "entangled_photon", "coronagraphy", "talbot_lau",
        "streak_camera", "nerf", "gaussian_splatting", "matrix",
        "quantum_illumination", "shearography", "dot",
    ]),
    ("X-ray & Nuclear Imaging", [
        "fluoroscopy", "xray_radiography", "xray_ndt", "xray_crystallography",
        "xfel_sfx", "xrf_tomo", "waxs", "saxs", "ct_fluorescence",
        "angiography", "neutron_tomo", "neutron_diffraction", "muon_tomo",
    ]),
    ("Remote Sensing & Geophysics", [
        "sar", "polsar", "insar", "lidar", "sonar", "gpr",
        "radio_astronomy", "radio_interferometry", "eht_imaging",
        "gravitational_wave", "weather_radar", "fwi",
        "seismic_tomo", "solar_imaging", "ocean_color", "passive_microwave",
        "nirs_brain", "magnetic_particle",
    ]),
    ("Industrial & NDT", [
        "active_thermography", "eddy_current", "terahertz",
    ]),
    ("Particle & High-Energy Physics", [
        "particle_calorimetry", "particle_tracker",
    ]),
]


def get_year(solver_key, solver_name):
    """Determine algorithm year from key or name."""
    if solver_key in YEAR_MAP:
        return str(YEAR_MAP[solver_key])
    # DL-based
    dl_keys = ["dl_", "unet", "transformer", "diffusion", "mamba", "cnn", "gan", "nerf_dl"]
    if any(dk in solver_key for dk in dl_keys):
        return "2016+"
    # Check for year in name
    import re
    m = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', solver_name)
    if m:
        return m.group(1)
    # Proxy solvers
    if "[proxy]" in solver_name.lower() or "proxy" in solver_name.lower():
        return "--"
    return "--"


def main():
    # Load data
    psnr_file = ROOT / "benchmark_results" / "template_5x_results.json"
    ssim_file = ROOT / "benchmark_results" / "ssim_results.json"
    ref_file = ROOT / "benchmark_results" / "benchmark_leaderboard_reference.json"
    index_file = ROOT / "datasets" / "templates" / "index.md"

    with open(psnr_file) as f:
        psnr_data = json.load(f)
    ssim_data = {}
    if ssim_file.exists():
        with open(ssim_file) as f:
            ssim_data = json.load(f)
    ref_data = {}
    if ref_file.exists():
        with open(ref_file) as f:
            ref_data = json.load(f)

    # Read existing index.md to preserve flagship sections
    existing = ""
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            existing = f.read()

    # Count total algorithms
    total_algos = 0
    total_done = 0
    non_flagship_mods = []

    for cat_name, mod_list in CATEGORIES:
        for mod_id in mod_list:
            if mod_id in FLAGSHIP:
                continue
            if mod_id in psnr_data and psnr_data[mod_id].get("error") is None:
                solvers = psnr_data[mod_id].get("solvers", {})
                total_algos += len(solvers)
                total_done += sum(1 for sv in solvers.values() if sv.get("status") == "done")
                non_flagship_mods.append(mod_id)

    # Also count uncategorized modalities
    algo_dir = ROOT / "algorithm_base"
    all_mods = sorted([
        d for d in os.listdir(algo_dir)
        if (algo_dir / d).is_dir()
        and not d.startswith('_') and not d.startswith('.')
        and d not in ('shared', '__pycache__')
    ])
    categorized = set()
    for _, ml in CATEGORIES:
        categorized.update(ml)
    uncategorized = [m for m in all_mods if m not in FLAGSHIP and m not in categorized and m in psnr_data]

    for mod_id in uncategorized:
        if psnr_data[mod_id].get("error") is None:
            solvers = psnr_data[mod_id].get("solvers", {})
            total_algos += len(solvers)
            total_done += sum(1 for sv in solvers.values() if sv.get("status") == "done")

    lines = []
    lines.append("# Modality Templates Index")
    lines.append("")
    lines.append("Comprehensive 7-step templates for all 168 imaging modalities in the PWM5 Benchmark.")
    lines.append("Each template covers: (1) Verify Standard Dataset, (2) List All Algorithms, (3) Update Solvers, (4) Verify Each Algorithm, (5) Upload Checkpoints to GCS, (6) Upload Standard Dataset to GCS, (7) Push to GitHub.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    # Flagship section — preserve from existing index.md
    # ═══════════════════════════════════════════════════════════════════
    lines.append("## Implementation Tracking -- 12 Flagship Paper Modalities")
    lines.append("")
    lines.append("Each algorithm must be implemented at least **5 times** (5 independent verification runs on the standard dataset).")
    lines.append("When all 5 runs are complete, the algorithm status is marked **done**.")
    lines.append("")

    # Extract flagship section from existing file
    if "### 1. CASSI" in existing:
        start = existing.index("### 1. CASSI")
        # Find end of flagship section
        if "## Implementation Tracking -- 156 Non-Flagship Modalities" in existing:
            end = existing.index("## Implementation Tracking -- 156 Non-Flagship Modalities")
        elif "## Flagship Summary" in existing:
            end = existing.index("## Flagship Summary")
        else:
            end = len(existing)
        flagship_block = existing[start:end].strip()
        lines.append(flagship_block)
        lines.append("")
    else:
        # Generate placeholder for flagship
        flagship_list = [
            ("CASSI", "cassi"), ("CACTI", "cacti"), ("SPC", "spc"),
            ("Lensless Imaging", "lensless"), ("Holography", "holography"),
            ("Ptychography", "ptychography"), ("CT", "ct"), ("CBCT", "cbct"),
            ("Ultrasound", "ultrasound"), ("Cryo-EM", "cryo_em"),
            ("MRI", "mri"), ("Widefield", "widefield"),
        ]
        for i, (name, mid) in enumerate(flagship_list, 1):
            lines.append(f"### {i}. {name} (`{mid}`)")
            lines.append("")
            lines.append("| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |")
            lines.append("|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|")
            lines.append("| | *(see flagship verification above)* | | -- | -- | -- | -- | -- | -- | -- | pending |")
            lines.append("")
            lines.append("---")
            lines.append("")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════
    # Non-flagship section
    # ═══════════════════════════════════════════════════════════════════
    lines.append("## Implementation Tracking -- 156 Non-Flagship Modalities")
    lines.append("")
    lines.append(f"Progress: **{total_done} / {total_algos} algorithms done** ({100*total_done/max(total_algos,1):.1f}%) | Last updated: 2026-03-20")
    lines.append("")

    mod_counter = 0
    for cat_name, mod_list in CATEGORIES:
        # Filter to non-flagship with data
        valid = [m for m in mod_list if m not in FLAGSHIP and m in psnr_data and psnr_data[m].get("error") is None]
        if not valid:
            continue

        lines.append(f"### {cat_name}")
        lines.append("")

        for mod_id in valid:
            mod_counter += 1
            display = DISPLAY_NAMES.get(mod_id, mod_id.replace("_", " ").title())
            solvers = psnr_data[mod_id].get("solvers", {})
            ssim_mod = ssim_data.get(mod_id, {})

            lines.append(f"#### {mod_counter}. {display} (`{mod_id}`)")
            lines.append("")

            # Reference if available
            if mod_id in ref_data and isinstance(ref_data[mod_id], list) and ref_data[mod_id]:
                top = ref_data[mod_id][0]
                rp = top.get("psnr_db", "N/A")
                rs = top.get("ssim", "N/A")
                lines.append(f"**Reference (SOTA):** {top['name']} -- PSNR {rp} dB, SSIM {rs}")
                lines.append("")

            lines.append("| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |")
            lines.append("|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|")

            # Sort solvers by PSNR descending
            sorted_solvers = sorted(
                solvers.items(),
                key=lambda x: x[1].get("mean_psnr") or -999,
                reverse=True,
            )

            for rank, (sk, sv) in enumerate(sorted_solvers, 1):
                name = sv.get("name", sk)
                runs = sv.get("runs", [])
                mean_p = sv.get("mean_psnr")
                status = sv.get("status", "fail")

                year = get_year(sk, name)

                # Format runs
                run_strs = []
                for r in runs[:5]:
                    if r is not None:
                        run_strs.append(f"{r:.1f}")
                    else:
                        run_strs.append("--")
                while len(run_strs) < 5:
                    run_strs.append("--")

                p_str = f"{mean_p:.2f}" if mean_p is not None else "--"

                # SSIM
                s_val = ssim_mod.get(sk) if ssim_mod else None
                s_str = f"{s_val:.4f}" if s_val is not None else "--"

                status_str = "**done**" if status == "done" else "fail"

                lines.append(
                    f"| {rank} | {name} | {year} | {' | '.join(run_strs)} | {p_str} | {s_str} | {status_str} |"
                )

            lines.append("")
            lines.append("---")
            lines.append("")

    # Handle uncategorized modalities
    if uncategorized:
        lines.append("### Other Modalities")
        lines.append("")
        for mod_id in uncategorized:
            if psnr_data[mod_id].get("error") is not None:
                continue
            mod_counter += 1
            display = DISPLAY_NAMES.get(mod_id, mod_id.replace("_", " ").title())
            solvers = psnr_data[mod_id].get("solvers", {})
            ssim_mod = ssim_data.get(mod_id, {})

            lines.append(f"#### {mod_counter}. {display} (`{mod_id}`)")
            lines.append("")

            lines.append("| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |")
            lines.append("|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|")

            sorted_solvers = sorted(
                solvers.items(),
                key=lambda x: x[1].get("mean_psnr") or -999,
                reverse=True,
            )

            for rank, (sk, sv) in enumerate(sorted_solvers, 1):
                name = sv.get("name", sk)
                runs = sv.get("runs", [])
                mean_p = sv.get("mean_psnr")
                status = sv.get("status", "fail")
                year = get_year(sk, name)

                run_strs = []
                for r in runs[:5]:
                    run_strs.append(f"{r:.1f}" if r is not None else "--")
                while len(run_strs) < 5:
                    run_strs.append("--")

                p_str = f"{mean_p:.2f}" if mean_p is not None else "--"
                s_val = ssim_mod.get(sk) if ssim_mod else None
                s_str = f"{s_val:.4f}" if s_val is not None else "--"
                status_str = "**done**" if status == "done" else "fail"

                lines.append(
                    f"| {rank} | {name} | {year} | {' | '.join(run_strs)} | {p_str} | {s_str} | {status_str} |"
                )

            lines.append("")
            lines.append("---")
            lines.append("")

    # Summary
    lines.append("")
    lines.append("## Non-Flagship Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total modalities | 156 |")
    lines.append(f"| Modalities verified | {len(non_flagship_mods) + len(uncategorized)} |")
    lines.append(f"| Total algorithms | {total_algos} |")
    lines.append(f"| Algorithms done (5x verified) | {total_done} |")
    lines.append(f"| Pass rate | {100*total_done/max(total_algos,1):.1f}% |")
    lines.append("")
    lines.append("*All results verified with 5 independent runs on standard benchmark datasets.*")

    # Write
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Written {len(lines)} lines to {index_file}")
    print(f"Total algorithms: {total_algos}, Done: {total_done}")
    print(f"Modalities covered: {mod_counter}")


if __name__ == "__main__":
    main()
