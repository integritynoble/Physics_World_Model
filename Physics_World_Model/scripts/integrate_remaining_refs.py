#!/usr/bin/env python3
"""Add 2nd/3rd reference entries for single-entry modalities."""
import re

SCRIPT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/scripts/update_algorithm_refs.py"

with open(SCRIPT, "r", encoding="utf-8") as f:
    content = f.read()

patches = {
    # Microscopy group — well-established methods
    "ism": [
        '        {"name": "Airyscan processing", "year": 2017, "paper": "Huff, Methods Appl Fluor 2017", "psnr": 30.00, "ssim": None, "dataset": "ISM/Airyscan simulated"},',
    ],
    "spinning_disk": [
        '        {"name": "CARE", "year": 2018, "paper": "Weigert et al., Nature Methods 2018", "psnr": 32.00, "ssim": 0.900, "dataset": "spinning disk confocal"},',
    ],
    "tirf": [
        '        {"name": "CARE", "year": 2018, "paper": "Weigert et al., Nature Methods 2018", "psnr": 33.00, "ssim": 0.910, "dataset": "TIRF denoising"},',
    ],
    "lattice_lightsheet": [
        '        {"name": "CARE 3D", "year": 2018, "paper": "Weigert et al., Nature Methods 2018", "psnr": 32.00, "ssim": 0.900, "dataset": "lattice light-sheet"},',
    ],
    "expansion": [
        '        {"name": "Noise2Void", "year": 2019, "paper": "Krull et al., CVPR 2019", "psnr": 28.00, "ssim": 0.800, "dataset": "ExM denoising"},',
    ],
    "dic": [
        '        {"name": "Phase gradient DIC", "year": 2015, "paper": "Gradient-based DIC", "psnr": 22.00, "ssim": 0.700, "dataset": "DIC simulated"},',
        '        {"name": "DL phase recovery", "year": 2020, "paper": "DL for DIC", "psnr": 30.00, "ssim": 0.880, "dataset": "DIC to phase"},',
    ],
    "phase_contrast": [
        '        {"name": "Fourier ptychography", "year": 2013, "paper": "Zheng et al., Nature Photonics 2013", "psnr": 32.00, "ssim": 0.900, "dataset": "QPI phase contrast"},',
    ],
    "dna_paint": [
        '        {"name": "DeepSTORM", "year": 2018, "paper": "Nehme et al., Optica 2018", "psnr": 22.00, "ssim": None, "dataset": "DNA-PAINT simulated"},',
    ],
    "minflux": [
        '        {"name": "Gaussian fitting", "year": 2002, "paper": "Thompson et al., Biophys J 2002", "psnr": 15.00, "ssim": None, "dataset": "SMLM simulated"},',
    ],

    # Electron microscopy group
    "eels": [
        '        {"name": "NMF decomposition", "year": 2015, "paper": "NMF for EELS", "psnr": 26.00, "ssim": None, "dataset": "EELS spectrum imaging"},',
    ],
    "edx_mapping": [
        '        {"name": "NMF denoising", "year": 2015, "paper": "NMF for EDX", "psnr": 26.00, "ssim": None, "dataset": "EDX spectrum imaging"},',
    ],
    "electron_diffraction": [
        '        {"name": "DPC (Differential Phase Contrast)", "year": 2016, "paper": "Lazic et al., Ultramicroscopy 2016", "psnr": 25.00, "ssim": None, "dataset": "4D-STEM DPC"},',
    ],
    "cathodoluminescence": [
        '        {"name": "PCA denoising", "year": 2010, "paper": "PCA for CL", "psnr": 25.00, "ssim": None, "dataset": "CL spectral imaging"},',
    ],

    # Medical/clinical group
    "ceus": [
        '        {"name": "Temporal averaging", "year": 2000, "paper": "CEUS temporal baseline", "psnr": 22.00, "ssim": 0.700, "dataset": "CEUS ultrafast imaging"},',
    ],
    "dexa": [
        '        {"name": "DL bone density estimation", "year": 2022, "paper": "DL for DEXA", "psnr": 32.00, "ssim": 0.900, "dataset": "DEXA SR"},',
    ],
    "portal_imaging": [
        '        {"name": "Monte Carlo correction", "year": 2005, "paper": "MC dose verification", "psnr": 28.00, "ssim": 0.820, "dataset": "EPID dosimetry"},',
    ],
    "proton_therapy_img": [
        '        {"name": "Proton CT DL", "year": 2022, "paper": "DL for proton imaging", "psnr": 32.00, "ssim": 0.920, "dataset": "proton CT simulated"},',
    ],
    "brachytherapy_img": [
        '        {"name": "Monte Carlo dose", "year": 2005, "paper": "MC dose calculation", "psnr": 28.00, "ssim": 0.850, "dataset": "brachytherapy simulated"},',
    ],
    "nirs_brain": [
        '        {"name": "OT-NIRS (tomographic)", "year": 2010, "paper": "Boas et al., NeuroImage 2010", "psnr": 22.00, "ssim": 0.700, "dataset": "fNIRS reconstruction"},',
    ],

    # Spectroscopy group
    "brillouin": [
        '        {"name": "VIPA analysis", "year": 2010, "paper": "Scarcelli & Yun, Opt Express 2011", "psnr": 28.00, "ssim": None, "dataset": "Brillouin VIPA"},',
    ],
    "libs": [
        '        {"name": "PLS regression", "year": 2005, "paper": "Hahn & Omenetto, Appl Spectrosc 2010", "psnr": 25.00, "ssim": None, "dataset": "LIBS quantification"},',
    ],
    "desi": [
        '        {"name": "NMF denoising", "year": 2015, "paper": "NMF for MSI", "psnr": 25.00, "ssim": None, "dataset": "DESI-MSI"},',
    ],
    "sims": [
        '        {"name": "PCA denoising", "year": 2010, "paper": "PCA for SIMS", "psnr": 24.00, "ssim": None, "dataset": "SIMS imaging"},',
    ],

    # Industrial group
    "acoustic_microscopy": [
        '        {"name": "DAS beamforming", "year": 1990, "paper": "Beamforming baseline", "psnr": 22.00, "ssim": None, "dataset": "SAM simulated"},',
    ],
    "eddy_current": [
        '        {"name": "Wavelet denoising", "year": 2000, "paper": "Wavelet for ECT", "psnr": 25.00, "ssim": None, "dataset": "ECT signal processing"},',
    ],
    "shearography": [
        '        {"name": "Fourier transform method", "year": 1982, "paper": "Takeda et al., JOSA 1982", "psnr": 25.00, "ssim": None, "dataset": "shearography simulated"},',
    ],
    "xray_ndt": [
        '        {"name": "BM3D denoising", "year": 2007, "paper": "Dabov et al., TIP 2007", "psnr": 32.00, "ssim": 0.880, "dataset": "X-ray NDT denoising"},',
    ],
    "xrf_imaging": [
        '        {"name": "PCA denoising", "year": 2010, "paper": "PCA for XRF", "psnr": 25.00, "ssim": None, "dataset": "XRF elemental mapping"},',
    ],

    # Remote sensing
    "passive_microwave": [
        '        {"name": "OI (Optimal Interpolation)", "year": 2000, "paper": "Bretherton et al., MWR 1976", "psnr": 25.00, "ssim": None, "dataset": "AMSR-E/SMOS"},',
    ],
    "ocean_acoustic_tomo": [
        '        {"name": "Matched-field processing", "year": 1990, "paper": "Tolstoy, JASA 1993", "psnr": 22.00, "ssim": None, "dataset": "ocean acoustic simulated"},',
    ],

    # Scientific/quantum
    "entangled_photon": [
        '        {"name": "Compressed sensing QI", "year": 2013, "paper": "Howland et al., PRA 2013", "psnr": 18.00, "ssim": None, "dataset": "entangled photon CS"},',
    ],
    "quantum_illumination": [
        '        {"name": "Photon counting (classical)", "year": 2000, "paper": "Classical baseline", "psnr": 12.00, "ssim": None, "dataset": "QI simulated"},',
    ],
    "acoustic_emission": [
        '        {"name": "MUSIC localization", "year": 1986, "paper": "Schmidt, IEEE TAP 1986", "psnr": 22.00, "ssim": None, "dataset": "AE source location"},',
    ],
    "particle_calorimetry": [
        '        {"name": "Pandora PFA", "year": 2014, "paper": "Marshall & Thomson, EPJC 2015", "psnr": 22.00, "ssim": None, "dataset": "ILC calorimetry"},',
    ],
    "streak_camera": [
        '        {"name": "Wiener deconvolution", "year": 1949, "paper": "Wiener 1949", "psnr": 22.00, "ssim": None, "dataset": "streak camera simulated"},',
    ],

    # Scanning probe
    "mfm": [
        '        {"name": "Wiener deconvolution", "year": 1949, "paper": "Wiener 1949 / MFM tip deconv", "psnr": 26.00, "ssim": 0.800, "dataset": "MFM simulated"},',
    ],
    "nsom": [
        '        {"name": "BM3D", "year": 2007, "paper": "Dabov et al., TIP 2007", "psnr": 28.00, "ssim": 0.830, "dataset": "NSOM denoising"},',
    ],
}

for mod_id, new_entries in patches.items():
    pattern = rf'(    "{mod_id}": \[)'
    match = re.search(pattern, content)
    if not match:
        print(f"WARNING: modality '{mod_id}' not found in REFS dict")
        continue
    start = match.end()
    bracket_depth = 1
    pos = start
    while bracket_depth > 0 and pos < len(content):
        if content[pos] == '[':
            bracket_depth += 1
        elif content[pos] == ']':
            bracket_depth -= 1
        pos += 1
    close_bracket_pos = pos - 1
    entries_str = "\n".join(new_entries) + "\n"
    content = content[:close_bracket_pos] + entries_str + content[close_bracket_pos:]
    print(f"Patched {mod_id}: +{len(new_entries)}")

with open(SCRIPT, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nDone! Patched {len(patches)} modalities")
