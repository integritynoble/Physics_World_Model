#!/usr/bin/env python3
"""Integrate verified results from search agents into REFS dict."""
import re

SCRIPT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/scripts/update_algorithm_refs.py"

with open(SCRIPT, "r", encoding="utf-8") as f:
    content = f.read()

patches = {
    # === From Agent Batch 4 (medical/clinical) ===
    "proton_therapy_img": [
        '        {"name": "Residual GAN (PPI-to-DRR)", "year": 2024, "paper": "Wang et al., PMC 2024", "psnr": 39.14, "ssim": 0.987, "dataset": "head phantom proton portal imaging"},',
        '        {"name": "CycleGAN (CBCT-to-sCT)", "year": 2024, "paper": "MDPI Sensors 2024", "psnr": 34.12, "ssim": 0.860, "dataset": "paediatric CBCT for proton therapy"},',
    ],
    "portal_imaging": [
        '        {"name": "CycleGAN MVCT-to-kVCT", "year": 2021, "paper": "Lee et al., Medical Physics 2021", "psnr": 32.73, "ssim": 0.955, "dataset": "MVCT helical tomotherapy"},',
        '        {"name": "CycleGAN+Attention+Residual", "year": 2024, "paper": "Lv et al., Medical Physics 2024", "psnr": 34.00, "ssim": 0.965, "dataset": "MVCT-to-synthetic-kVCT"},',
    ],
    "nirs_brain": [
        '        {"name": "CNN-LSTM Hybrid", "year": 2024, "paper": "Multimedia Tools Appl 2024", "psnr": 32.15, "ssim": 0.986, "dataset": "simulated DOT phantom"},',
    ],
    "phase_contrast": [
        '        {"name": "GAN (self-attention)", "year": 2024, "paper": "Scientific Reports 2024", "psnr": 38.33, "ssim": 0.880, "dataset": "X-ray phase contrast fringe"},',
        '        {"name": "DL flat-fielding QPC", "year": 2024, "paper": "ResearchGate 2024", "psnr": 29.13, "ssim": 0.865, "dataset": "quantitative phase contrast"},',
    ],
    "shg": [
        '        {"name": "DnCNN", "year": 2023, "paper": "Bai et al., Biomed Opt Express 2023", "psnr": 25.40, "ssim": 0.770, "dataset": "SHG tissue imaging"},',
    ],
    "tirf": [
        '        {"name": "RED-fairSIM", "year": 2021, "paper": "Christensen et al., Photonics Research 2021", "psnr": 33.22, "ssim": 0.900, "dataset": "TIRF-SIM U2OS cells"},',
    ],
    "xray_ndt": [
        '        {"name": "U-Net++", "year": 2025, "paper": "NDT.net DIR 2025", "psnr": 32.32, "ssim": 0.896, "dataset": "industrial X-ray CT 20% noise"},',
    ],
    "ceus": [
        '        {"name": "GAN-RW (Residual Dense)", "year": 2022, "paper": "Lan et al., PeerJ Computer Science 2022", "psnr": 33.91, "ssim": 0.872, "dataset": "US speckle denoising sigma=25"},',
        '        {"name": "Real-time CNN", "year": 2022, "paper": "Choi et al., MBEC 2022", "psnr": 36.13, "ssim": 0.964, "dataset": "obstetric US 5K images"},',
    ],

    # === From Agent Batch 5 (electron/spectro) ===
    "eels": [
        '        {"name": "Deep CNN Denoiser", "year": 2021, "paper": "Mohan et al., Microsc Microanal 2021", "psnr": 42.87, "ssim": 0.990, "dataset": "TEM nanoparticle denoising"},',
    ],
    "acoustic_microscopy": [
        '        {"name": "SwinIR (SAM)", "year": 2024, "paper": "Somani et al., CVPR Workshop 2023", "psnr": 35.13, "ssim": 0.950, "dataset": "SAM biological + industrial"},',
        '        {"name": "HDL-SAM (SwinIR+Hypergraph)", "year": 2024, "paper": "Somani & Banerjee, OpenReview 2024", "psnr": 31.60, "ssim": 0.920, "dataset": "SAM 4x SR"},',
        '        {"name": "Hypergraph Inpainting", "year": 2023, "paper": "Somani et al., CVPR Workshop 2023", "psnr": 27.96, "ssim": 0.820, "dataset": "SAM 4x SR"},',
    ],
    "shearography": [
        '        {"name": "FPD-CNN", "year": 2020, "paper": "Lin et al., Applied Optics 2020", "psnr": 27.88, "ssim": 0.972, "dataset": "ESPI fringe patterns"},',
        '        {"name": "DBDNet", "year": 2021, "paper": "Li et al., Applied Optics 2021", "psnr": 20.56, "ssim": None, "dataset": "ESPI wrapped phase"},',
    ],
    "sims": [
        '        {"name": "De-MSI (DL)", "year": 2025, "paper": "Gank et al., Anal Chem 2025", "psnr": 18.93, "ssim": 0.740, "dataset": "MALDI/DESI MSI"},',
    ],

    # === From Agent Batch 6 (optics/other) ===
    "dic": [
        '        {"name": "TIE-GANs", "year": 2024, "paper": "Poliwoda et al., J Biomed Opt 2024", "psnr": 28.10, "ssim": 0.980, "dataset": "microbeads 4um phase imaging"},',
        '        {"name": "PINN-TIE", "year": 2022, "paper": "Zhang et al., Opt Express 2022", "psnr": 25.23, "ssim": 0.919, "dataset": "quantitative phase cells"},',
    ],
    "streak_camera": [
        '        {"name": "PnP-FFDNet (sim)", "year": 2022, "paper": "Yuan et al., Sensors 2022, PMC9571970", "psnr": 28.37, "ssim": 0.910, "dataset": "simulated CUP 5-scene avg"},',
        '        {"name": "PnP-BM3D (sim)", "year": 2022, "paper": "Yuan et al., Sensors 2022, PMC9571970", "psnr": 29.18, "ssim": 0.920, "dataset": "simulated CUP 5-scene avg"},',
    ],
    "mfm": [
        '        {"name": "Interval-BCS (AFM)", "year": 2019, "paper": "Lu et al., Nanotechnology 2019, PMC6902871", "psnr": 43.20, "ssim": 0.970, "dataset": "AFM noise density 0.4"},',
        '        {"name": "Adaptive Median (AFM)", "year": 2019, "paper": "Lu et al., Nanotechnology 2019, PMC6902871", "psnr": 33.90, "ssim": 0.950, "dataset": "AFM noise density 0.4"},',
    ],
    "xrf_imaging": [
        '        {"name": "DnCNN (XFCT)", "year": 2024, "paper": "J Imaging 2024, PMC11204716", "psnr": 49.35, "ssim": 0.943, "dataset": "XFCT low-noise"},',
        '        {"name": "NLM (XFCT)", "year": 2024, "paper": "J Imaging 2024, PMC11204716", "psnr": 39.94, "ssim": 0.803, "dataset": "XFCT low-noise"},',
    ],
}

for mod_id, new_entries in patches.items():
    pattern = rf'(    "{mod_id}": \[)'
    match = re.search(pattern, content)
    if not match:
        print(f"WARNING: modality '{mod_id}' not found")
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

print(f"\nDone! Patched {len(patches)} modalities with agent results")
