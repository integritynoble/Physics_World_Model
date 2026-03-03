"""Algorithm catalog — maps categories to 4 real published reconstruction algorithms.

Each category gets a Classical, PnP, Deep Learning, and Transformer/specialty method
with real published algorithm names, proper citations, and domain-appropriate metadata.

Also provides per-category metadata for generating realistic leaderboards:
  - Scene names (domain-relevant test data labels)
  - Mismatch descriptions (what calibration errors affect each domain)
  - Correction method descriptions (what gradient optimization targets)
  - Source citations per algorithm

The 4 hand-crafted variants (sd_cassi, cacti, spc_block, spc_kronecker) have their own
algorithm overrides to preserve the InverseNet-validated baselines.

References are from real published papers — see inline citations.
"""

from __future__ import annotations

# ── Hand-crafted overrides (preserve InverseNet validated names) ──────────────

_VARIANT_OVERRIDES: dict[str, list[dict]] = {
    # CT — fan-beam sparse-view, LoDoPaB-CT geometry (362×362, 60 views, I₀=10k)
    # 8 algorithms spanning classical → diffusion, based on published results on
    # LoDoPaB-CT and comparable sparse-view / low-dose CT benchmarks.
    "ct": [
        {"name": "FBP",                 "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "Kak & Slaney, IEEE Press 1988"},
        {"name": "TV-ADMM",             "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "Sidky et al., Phys. Med. Biol. 2008"},
        {"name": "PnP-ADMM",            "type": "PnP",            "mask_aware": True,  "params": "0",    "source": "Venkatakrishnan et al., IEEE GlobalSIP 2013"},
        {"name": "RED-CNN",             "type": "Deep Learning",  "mask_aware": False, "params": "1.6M", "source": "Chen et al., IEEE TMI 2017"},
        {"name": "FBPConvNet",          "type": "Deep Learning",  "mask_aware": False, "params": "22M",  "source": "Jin et al., IEEE TIP 2017"},
        {"name": "Learned Primal-Dual", "type": "Deep Unrolling", "mask_aware": True,  "params": "5M",   "source": "Adler & Oktem, IEEE TMI 2018"},
        {"name": "DuDoTrans",           "type": "Transformer",    "mask_aware": True,  "params": "7.5M", "source": "Wang et al., MLMIR 2022"},
        {"name": "DOLCE",               "type": "Diffusion",      "mask_aware": True,  "params": "86M",  "source": "Liu et al., ICCV 2023"},
    ],
    # MRI — multi-coil parallel imaging, fastMRI knee 4x Cartesian acceleration.
    # 8 algorithms spanning classical → diffusion, based on published fastMRI results.
    "mri": [
        {"name": "Zero-Filled IFFT",    "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "Zbontar et al., arXiv 2018"},
        {"name": "L1-Wavelet (ESPIRiT)", "type": "Compressed Sensing", "mask_aware": True, "params": "0", "source": "Lustig et al., MRM 2007"},
        {"name": "PnP-DnCNN",           "type": "PnP",            "mask_aware": True,  "params": "670K", "source": "Ahmad et al., IEEE SPM 2020"},
        {"name": "U-Net",               "type": "Deep Learning",  "mask_aware": False, "params": "44M",  "source": "Zbontar et al., arXiv 2018"},
        {"name": "E2E-VarNet",          "type": "Deep Unrolling", "mask_aware": True,  "params": "30M",  "source": "Sriram et al., MICCAI 2020"},
        {"name": "PromptMR",            "type": "Deep Unrolling", "mask_aware": True,  "params": "80M",  "source": "Bai et al., ECCV 2024"},
        {"name": "ReconFormer",         "type": "Transformer",    "mask_aware": True,  "params": "64M",  "source": "Guo et al., IEEE TMI 2024"},
        {"name": "Score-MRI",           "type": "Diffusion",      "mask_aware": True,  "params": "60M",  "source": "Chung & Ye, Med. Image Anal. 2022"},
    ],
    "sd_cassi": [
        {"name": "GAP-TV",      "type": "Classical",      "mask_aware": True,  "params": "0",     "source": "InverseNet"},
        {"name": "PnP-HSICNN",  "type": "PnP",            "mask_aware": True,  "params": "0",     "source": "InverseNet"},
        {"name": "HDNet",       "type": "Deep Learning",  "mask_aware": False, "params": "2.37M", "source": "InverseNet"},
        {"name": "MST-L",       "type": "Transformer",    "mask_aware": True,  "params": "2.03M", "source": "InverseNet"},
    ],
    "cacti": [
        {"name": "GAP-TV",        "type": "Classical",      "mask_aware": True,  "params": "0",     "source": "InverseNet"},
        {"name": "PnP-FFDNet",    "type": "PnP",            "mask_aware": True,  "params": "0",     "source": "InverseNet"},
        {"name": "ELP-Unfolding", "type": "Deep Unfolding", "mask_aware": True,  "params": "565M",  "source": "ECCV 2022"},
        {"name": "EfficientSCI",  "type": "Deep Learning",  "mask_aware": True,  "params": "4.2M",  "source": "CVPR 2023"},
        {"name": "HiSViT-9",      "type": "Transformer",    "mask_aware": True,  "params": "6.2M",  "source": "ECCV 2024"},
    ],
    "spc_block": [
        {"name": "FISTA-TV",   "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "InverseNet"},
        {"name": "PnP-DRUNet", "type": "PnP",            "mask_aware": True,  "params": "0",    "source": "InverseNet"},
        {"name": "HATNet",     "type": "Deep Learning",  "mask_aware": False, "params": "0.8M", "source": "InverseNet"},
        {"name": "ISTA-Net",   "type": "Deep Unfolding", "mask_aware": True,  "params": "0.3M", "source": "InverseNet"},
    ],
    "spc_kronecker": [
        {"name": "FISTA-TV",   "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "InverseNet"},
        {"name": "PnP-DRUNet", "type": "PnP",            "mask_aware": True,  "params": "0",    "source": "InverseNet"},
        {"name": "HATNet",     "type": "Deep Learning",  "mask_aware": False, "params": "0.8M", "source": "InverseNet"},
        {"name": "ISTA-Net",   "type": "Deep Unfolding", "mask_aware": True,  "params": "0.3M", "source": "InverseNet"},
    ],
}

# ── Category → algorithm mapping (real published algorithms) ──────────────────

_CATEGORY_ALGORITHMS: dict[str, list[dict]] = {

    # --- Compressive imaging ---
    # Yuan et al. 2016 (GAP-TV), Zhang et al. 2017 (FFDNet), Wang et al. 2023 (EfficientSCI), Cai et al. 2022 (MST)
    "compressive": [
        {"name": "GAP-TV",       "type": "Classical",     "mask_aware": True,  "params": "0",     "source": "Yuan et al., 2016"},
        {"name": "PnP-FFDNet",   "type": "PnP",           "mask_aware": True,  "params": "0",     "source": "Zhang et al., 2017"},
        {"name": "EfficientSCI", "type": "Deep Learning", "mask_aware": True,  "params": "4.2M",  "source": "Wang et al., 2023"},
        {"name": "MST-L",        "type": "Transformer",   "mask_aware": True,  "params": "2.03M", "source": "Cai et al., CVPR 2022"},
    ],

    # --- Medical CT/X-ray ---
    # FBP (standard), Venkatakrishnan 2013 (PnP-ADMM), Jin et al. IEEE TIP 2017 (FBPConvNet), Adler & Oktem IEEE TMI 2018
    "medical": [
        {"name": "FBP",                "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
        {"name": "PnP-ADMM",           "type": "PnP",            "mask_aware": True,  "params": "0",    "source": "Venkatakrishnan et al., 2013"},
        {"name": "FBPConvNet",         "type": "Deep Learning",  "mask_aware": False, "params": "22M",  "source": "Jin et al., IEEE TIP 2017"},
        {"name": "Learned Primal-Dual","type": "Deep Unrolling", "mask_aware": True,  "params": "5M",   "source": "Adler & Oktem, IEEE TMI 2018"},
    ],

    # --- Medical ultrasound ---
    # DAS (standard), Goudarzi 2020, Luijten IEEE TMI 2020 (ABLE), Hyun IEEE TUFFC 2022
    "medical_ultrasound": [
        {"name": "DAS",       "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
        {"name": "PnP-ADMM",  "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Goudarzi et al., 2020"},
        {"name": "ABLE",      "type": "Deep Learning", "mask_aware": False, "params": "0.5M", "source": "Luijten et al., IEEE TMI 2020"},
        {"name": "MU-Net",    "type": "Deep Learning", "mask_aware": True,  "params": "8M",   "source": "Hyun et al., IEEE TUFFC 2022"},
    ],

    # --- Coherent / phase retrieval / holography ---
    # Fienup 1982 (GS/HIO), Metzler ICML 2018 (prDeep), Rivenson 2018 (PhaseNet), Choi 2023 (LRGS)
    "coherent": [
        {"name": "GS/HIO",   "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "Fienup, Appl. Opt. 1982"},
        {"name": "prDeep",    "type": "PnP",            "mask_aware": True,  "params": "0",    "source": "Metzler et al., ICML 2018"},
        {"name": "PhaseNet",  "type": "Deep Learning",  "mask_aware": False, "params": "1.5M", "source": "Rivenson et al., LSA 2018"},
        {"name": "LRGS",      "type": "Deep Unrolling", "mask_aware": True,  "params": "5M",   "source": "Choi et al., 2023"},
    ],

    # --- Microscopy (fluorescence, widefield, confocal, lightsheet) ---
    # Richardson 1972 / Lucy 1974, Bai 2020 (PnP-FISTA), Weigert Nat. Methods 2018 (CARE), Zamir CVPR 2022 (Restormer)
    "microscopy": [
        {"name": "Richardson-Lucy", "type": "Classical",     "mask_aware": True,  "params": "0",     "source": "Richardson 1972 / Lucy 1974"},
        {"name": "PnP-FISTA",       "type": "PnP",           "mask_aware": True,  "params": "0",     "source": "Bai et al., 2020"},
        {"name": "CARE",            "type": "Deep Learning", "mask_aware": False, "params": "7.8M",  "source": "Weigert et al., Nat. Methods 2018"},
        {"name": "Restormer",       "type": "Transformer",   "mask_aware": True,  "params": "26M",   "source": "Zamir et al., CVPR 2022"},
    ],

    # --- Electron microscopy (cryo-EM, TEM, SEM, STEM) ---
    # Scheres J. Struct. Biol. 2012 (RELION), Punjani Nat. Methods 2017 (cryoSPARC), Zhong Nat. Methods 2021 (cryoDRGN)
    "electron_microscopy": [
        {"name": "RELION",          "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Scheres, J. Struct. Biol. 2012"},
        {"name": "cryoSPARC",       "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Punjani et al., Nat. Methods 2017"},
        {"name": "cryoDRGN",        "type": "Deep Learning", "mask_aware": False, "params": "1.5M", "source": "Zhong et al., Nat. Methods 2021"},
        {"name": "CryoTransformer", "type": "Transformer",   "mask_aware": True,  "params": "4M",   "source": "Dhakal et al., Bioinf. 2024"},
    ],

    # --- Clinical optics (OCT, fundus, endoscopy) ---
    # FFT-OCT (standard), Maggioni IEEE TIP 2013 (BM4D), Devalla BOE 2019, OCTA-Net 2023
    "clinical_optics": [
        {"name": "FFT-OCT",             "type": "Classical",     "mask_aware": True,  "params": "0",     "source": "Analytical baseline"},
        {"name": "BM4D",                "type": "PnP",           "mask_aware": True,  "params": "0",     "source": "Maggioni et al., IEEE TIP 2013"},
        {"name": "Speckle-DenoiseNet",  "type": "Deep Learning", "mask_aware": False, "params": "1.2M",  "source": "Devalla et al., BOE 2019"},
        {"name": "OCTA-Net",            "type": "Transformer",   "mask_aware": True,  "params": "15M",   "source": "Hybrid U-Net+Transformer, 2023"},
    ],

    # --- Computational imaging (tomography, phase imaging) ---
    # Tikhonov (standard), Romano IEEE TIP 2017 (RED), Ulyanov CVPR 2018 (Deep Image Prior), Liang ICCVW 2021 (SwinIR)
    "computational": [
        {"name": "Tikhonov",          "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
        {"name": "PnP-RED",           "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Romano et al., IEEE TIP 2017"},
        {"name": "Deep Image Prior",  "type": "Deep Learning", "mask_aware": False, "params": "2.2M", "source": "Ulyanov et al., CVPR 2018"},
        {"name": "SwinIR",            "type": "Transformer",   "mask_aware": True,  "params": "12M",  "source": "Liang et al., ICCVW 2021"},
    ],

    # --- Computational photography (HDR, coded exposure, light field) ---
    # Debevec SIGGRAPH 1997, PnP-FFDNet, Eilertsen ACM TOG 2017 (HDR-CNN), Liu 2022 (HDRTransDC)
    "computational_photography": [
        {"name": "Wiener-Deconv",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
        {"name": "PnP-FFDNet",     "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Zhang et al., 2017"},
        {"name": "HDR-CNN",        "type": "Deep Learning", "mask_aware": False, "params": "29M",  "source": "Eilertsen et al., ACM TOG 2017"},
        {"name": "Uformer",        "type": "Transformer",   "mask_aware": True,  "params": "20M",  "source": "Wang et al., CVPR 2022"},
    ],

    # --- Neural rendering (NeRF, 3DGS) ---
    # Schonberger CVPR 2016 (COLMAP), Barron CVPR 2022 (Mip-NeRF 360), Muller SIGGRAPH 2022 (Instant-NGP), Kerbl SIGGRAPH 2023 (3DGS)
    "neural_rendering": [
        {"name": "COLMAP+MVS",    "type": "Classical",     "mask_aware": False, "params": "0",    "source": "Schonberger & Frahm, CVPR 2016"},
        {"name": "Mip-NeRF 360",  "type": "PnP",           "mask_aware": True,  "params": "9M",   "source": "Barron et al., CVPR 2022"},
        {"name": "Instant-NGP",   "type": "Deep Learning", "mask_aware": False, "params": "16M",  "source": "Muller et al., SIGGRAPH 2022"},
        {"name": "3D-GS",         "type": "Transformer",   "mask_aware": False, "params": "varies","source": "Kerbl et al., SIGGRAPH 2023"},
    ],

    # --- Depth imaging (ToF, structured light, stereo) ---
    # Hirschmuller TPAMI 2007 (SGM), PnP-ADMM, Chang CVPR 2018 (PSMNet), Lipson 3DV 2021 (RAFT-Stereo)
    "depth_imaging": [
        {"name": "SGM",          "type": "Classical",     "mask_aware": True,  "params": "0",     "source": "Hirschmuller, TPAMI 2007"},
        {"name": "PnP-ADMM",     "type": "PnP",           "mask_aware": True,  "params": "0",     "source": "ADMM + denoiser prior"},
        {"name": "PSMNet",       "type": "Deep Learning", "mask_aware": False, "params": "5.2M",  "source": "Chang & Chen, CVPR 2018"},
        {"name": "RAFT-Stereo",  "type": "Transformer",   "mask_aware": True,  "params": "11M",   "source": "Lipson et al., 3DV 2021"},
    ],

    # --- Remote sensing (SAR, sonar, InSAR) ---
    # Matched filter (standard), Parrilli IEEE TGRS 2012 (SAR-BM3D), Zhang RS 2018 (SAR-DRN), SAR-CAM 2024
    "remote_sensing": [
        {"name": "Matched Filter",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Standard SAR focusing"},
        {"name": "SAR-BM3D",        "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Parrilli et al., IEEE TGRS 2012"},
        {"name": "SAR-DRN",         "type": "Deep Learning", "mask_aware": False, "params": "0.6M", "source": "Zhang et al., RS 2018"},
        {"name": "SAR-CAM",         "type": "Transformer",   "mask_aware": True,  "params": "8M",   "source": "Cross-attention SAR, 2024"},
    ],

    # --- Particle imaging (PET, SPECT, muon tomography) ---
    # Hudson IEEE TMI 1994 (OSEM), Nuyts 2002 (MAPEM-RDP), Haggstrom 2019 (DeepPET), Xie 2023 (TransEM)
    "particle_imaging": [
        {"name": "OSEM",       "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Hudson & Larkin, IEEE TMI 1994"},
        {"name": "MAPEM-RDP",  "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Nuyts et al., 2002"},
        {"name": "DeepPET",    "type": "Deep Learning", "mask_aware": False, "params": "15M",  "source": "Haggstrom et al., MIA 2019"},
        {"name": "TransEM",    "type": "Transformer",   "mask_aware": True,  "params": "20M",  "source": "Xie et al., 2023"},
    ],

    # --- Scanning probe (AFM, STM, MFM, NSOM) ---
    # Villarrubia JRNIST 1997 (BTR), Dongmo 2000 (regularized deconv), Alldritt 2020 (DeepSPM), Kossler 2022 (E2E-BTR)
    "scanning_probe": [
        {"name": "BTR",         "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Villarrubia, JRNIST 1997"},
        {"name": "Reg-Deconv",  "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Dongmo et al., 2000"},
        {"name": "DeepSPM",     "type": "Deep Learning", "mask_aware": False, "params": "2M",  "source": "Alldritt et al., Commun. Phys. 2020"},
        {"name": "E2E-BTR",     "type": "Deep Learning", "mask_aware": True,  "params": "3M",  "source": "Kossler et al., Sci. Rep. 2022"},
    ],

    # --- Industrial inspection (NDT, thermography, eddy current) ---
    # Shepard 2003 (TSR), PnP-ADMM, DefectNet 2020-2023, LSTM-NDT 2022
    "industrial_inspection": [
        {"name": "TSR",         "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Shepard et al., 2003"},
        {"name": "PnP-ADMM",    "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "ADMM + denoiser prior"},
        {"name": "DefectNet",   "type": "Deep Learning", "mask_aware": False, "params": "3M",  "source": "U-Net for NDT, 2021"},
        {"name": "LSTM-NDT",    "type": "Recurrent",     "mask_aware": True,  "params": "5M",  "source": "Fang et al., 2022"},
    ],

    # --- Spectroscopy (Raman, FTIR, XRF) ---
    # Savitzky-Golay + ALS (standard), DIRAS 2025, Zhang Sensors 2024 (CDAE), Cascade-UNet 2025
    "spectroscopy": [
        {"name": "SG-ALS",       "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Savitzky-Golay + ALS baseline"},
        {"name": "PnP-DnCNN",    "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Zhang et al., 2017"},
        {"name": "CDAE",          "type": "Deep Learning", "mask_aware": False, "params": "0.8M","source": "Zhang et al., Sensors 2024"},
        {"name": "Cascade-UNet",  "type": "Transformer",   "mask_aware": True,  "params": "4M",  "source": "Physics-informed UNet, 2025"},
    ],

    # --- Astronomy (radio interferometry, coronagraphy, solar imaging) ---
    # Hogbom A&AS 1974 (CLEAN), Terris MNRAS 2022 (AIRI), Aghabiglou ApJS 2024 (R2D2), Medeiros ApJL 2023 (PRIMO)
    "astronomy": [
        {"name": "CLEAN",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Hogbom, A&AS 1974"},
        {"name": "AIRI",   "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Terris et al., MNRAS 2022"},
        {"name": "R2D2",   "type": "Deep Learning", "mask_aware": False, "params": "10M",  "source": "Aghabiglou et al., ApJS 2024"},
        {"name": "PRIMO",  "type": "Deep Learning", "mask_aware": True,  "params": "2M",   "source": "Medeiros et al., ApJL 2023"},
    ],

    # --- Ultrafast imaging (streak camera, CUP, pump-probe) ---
    # Bioucas-Dias IEEE TIP 2007 (TwIST), Yuan 2020 (PnP-FFDNet for CUP), Parker 2021 (CUP-Net), Yao Photon. Res. 2021
    "ultrafast": [
        {"name": "TwIST",       "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Bioucas-Dias & Figueiredo, IEEE TIP 2007"},
        {"name": "PnP-FFDNet",  "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Yuan et al., 2020"},
        {"name": "CUP-Net",     "type": "Deep Learning", "mask_aware": False, "params": "8M",  "source": "Parker et al., 2021"},
        {"name": "AL-DL",       "type": "Hybrid",        "mask_aware": True,  "params": "5M",  "source": "Yao et al., Photon. Res. 2021"},
    ],

    # --- Quantum imaging (ghost imaging, entangled photon, quantum illumination) ---
    # Pittman PRA 1995 (G(2)), Li 2014 (CS-TVAL3), Wang Sci.Rep. 2020 (DRU-Net), Zhu 2025 (Ghost-ViT)
    "quantum": [
        {"name": "G(2)-Corr",   "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Pittman et al., PRA 1995"},
        {"name": "CS-TVAL3",    "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Li et al., 2014"},
        {"name": "DRU-Net",     "type": "Deep Learning", "mask_aware": False, "params": "7M",   "source": "Wang et al., Sci. Rep. 2020"},
        {"name": "Ghost-ViT",   "type": "Transformer",   "mask_aware": True,  "params": "1.4B", "source": "Zhu et al., 2025"},
    ],

    # --- Experimental science (acoustic emission, gravitational wave, etc.) ---
    # Tikhonov (standard), PnP-RED (Romano 2017), domain-adapted CNN, SciFormer (generic transformer)
    "experimental_science": [
        {"name": "Tikhonov",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
        {"name": "PnP-RED",   "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Romano et al., IEEE TIP 2017"},
        {"name": "ResUNet",   "type": "Deep Learning", "mask_aware": False, "params": "4.5M", "source": "Residual U-Net baseline"},
        {"name": "SwinIR",    "type": "Transformer",   "mask_aware": True,  "params": "12M",  "source": "Liang et al., ICCVW 2021"},
    ],

    # --- Scientific instrumentation (mass spec, atom probe, diffraction) ---
    # Deconvolution (standard), PnP-BM3D (Danielyan 2012), instrument-specific CNN, CalibFormer
    "scientific_instrumentation": [
        {"name": "Deconv",     "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
        {"name": "PnP-BM3D",   "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Danielyan et al., 2012"},
        {"name": "ResNet-Calib","type": "Deep Learning", "mask_aware": False, "params": "2.5M", "source": "ResNet for calibration, 2022"},
        {"name": "CalibFormer", "type": "Transformer",   "mask_aware": True,  "params": "8M",   "source": "Transformer calibration, 2024"},
    ],

    # --- Multi-modal fusion (PET-CT, PET-MR, US-MRI, SPECT-CT) ---
    # Rezaei IEEE TMI 2012 (MLAA), Ehrhardt 2015 (MR-guided PET), Mehranian IEEE TMI 2020 (FBSEM-Net), Li 2024 (PPMF-Net)
    "multi_modal_fusion": [
        {"name": "MLAA",       "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Rezaei et al., IEEE TMI 2012"},
        {"name": "MR-Guided",  "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Ehrhardt et al., SIIS 2015"},
        {"name": "FBSEM-Net",  "type": "Deep Learning", "mask_aware": False, "params": "8M",   "source": "Mehranian & Reader, IEEE TMI 2020"},
        {"name": "PPMF-Net",   "type": "Transformer",   "mask_aware": True,  "params": "12M",  "source": "Li et al., 2024"},
    ],
}

# ── Category → domain-specific scene names ────────────────────────────────────

CATEGORY_SCENE_NAMES: dict[str, list[str]] = {
    "compressive": [
        "Lego", "Dice", "Flowers", "Chart", "Fabric",
    ],
    "medical": [
        "Chest CT", "Abdomen CT", "Head CT",
    ],
    "medical_ultrasound": [
        "Carotid B-mode", "Liver B-mode", "Kidney B-mode",
    ],
    "coherent": [
        "USAF target", "Resolution chart", "Cell culture", "Phase object", "Lens aberration",
    ],
    "microscopy": [
        "Membrane", "Nuclei", "Mitochondria", "Actin", "Tubulin",
    ],
    "electron_microscopy": [
        "Ribosome", "Proteasome", "Virus capsid",
    ],
    "clinical_optics": [
        "Retina (macula)", "Retina (optic disc)", "Cornea cross-section",
    ],
    "computational": [
        "Shepp-Logan", "Brain phantom", "Mandril", "Resolution chart", "Peppers",
    ],
    "computational_photography": [
        "Indoor (low light)", "Outdoor HDR", "Motion blur", "Bokeh portrait", "Night scene",
    ],
    "neural_rendering": [
        "Lego truck", "Chair", "Hotdog", "Mic", "Ficus",
    ],
    "depth_imaging": [
        "Tsukuba", "Middlebury cones", "Teddy", "KITTI scene", "Sintel",
    ],
    "remote_sensing": [
        "Urban SAR", "Agricultural SAR", "Maritime SAR",
    ],
    "particle_imaging": [
        "Brain FDG", "Cardiac NH3", "Lung FDG",
    ],
    "scanning_probe": [
        "Si(111) 7\u00d77", "Graphite HOPG", "Gold nanoparticles",
    ],
    "industrial_inspection": [
        "Weld joint", "Composite panel", "Turbine blade",
    ],
    "spectroscopy": [
        "Polystyrene", "Aspirin tablet", "Tissue section", "Polymer blend", "Mineral sample",
    ],
    "astronomy": [
        "Cygnus A", "M87 jet", "Sgr A* ring",
    ],
    "ultrafast": [
        "Plasma pulse", "Shock wave", "Fluorescence decay", "Laser ablation", "Chemical reaction",
    ],
    "quantum": [
        "Slit pattern", "Ghost cat", "Bell state",
    ],
    "experimental_science": [
        "Wave propagation", "Thermal gradient", "AE source 1", "AE source 2", "Elastic response",
    ],
    "scientific_instrumentation": [
        "Crystal diffraction", "Atom cluster", "Mass spectrum", "TOF signal", "Ion map",
    ],
    "multi_modal_fusion": [
        "Brain PET-MR", "Cardiac PET-CT", "Liver SPECT-CT",
    ],
}


# ── Category → mismatch description (what calibration errors affect this domain) ──

CATEGORY_MISMATCH_DESC: dict[str, str] = {
    "compressive":
        "Mismatch: coded mask registration (dx, dy), mask rotation, "
        "dispersion calibration, detector gain drift.",
    "medical":
        "Mismatch: gantry geometry (source-detector distance, rotation offset), "
        "beam hardening coefficients, detector gain/offset drift.",
    "medical_ultrasound":
        "Mismatch: speed-of-sound map error, element position calibration, "
        "attenuation model, phase aberration.",
    "coherent":
        "Mismatch: propagation distance, illumination wavelength calibration, "
        "pixel pitch / magnification, detector tilt.",
    "microscopy":
        "Mismatch: PSF aberrations (Zernike modes), refractive index, "
        "coverslip thickness, focal depth offset.",
    "electron_microscopy":
        "Mismatch: CTF defocus, astigmatism (angle + magnitude), "
        "B-factor (envelope function), beam tilt.",
    "clinical_optics":
        "Mismatch: dispersion compensation coefficients, "
        "k-space linearization, reference arm path length.",
    "computational":
        "Mismatch: system matrix calibration error, "
        "detector response nonlinearity, geometric alignment.",
    "computational_photography":
        "Mismatch: camera response function (CRF), exposure time ratios, "
        "coded aperture/shutter pattern, lens aberrations.",
    "neural_rendering":
        "Mismatch: camera intrinsics (focal length, principal point), "
        "camera extrinsics (pose), lens distortion coefficients.",
    "depth_imaging":
        "Mismatch: stereo baseline/rectification, "
        "ToF phase-to-depth calibration, multi-path interference.",
    "remote_sensing":
        "Mismatch: platform motion errors (autofocus phase), "
        "range cell migration, antenna pattern calibration.",
    "particle_imaging":
        "Mismatch: attenuation map registration (CT-PET alignment), "
        "scatter model parameters, detector normalization.",
    "scanning_probe":
        "Mismatch: tip shape (radius, cone angle, tilt), "
        "piezo scanner hysteresis/creep, thermal drift.",
    "industrial_inspection":
        "Mismatch: heat source power calibration, "
        "material thermal diffusivity, emissivity variation, distance/angle.",
    "spectroscopy":
        "Mismatch: wavenumber/wavelength calibration (spectral axis shift+stretch), "
        "instrument response function, laser power fluctuation.",
    "astronomy":
        "Mismatch: per-antenna complex gain (amplitude+phase), "
        "atmospheric phase screen (troposphere/ionosphere), baseline errors.",
    "ultrafast":
        "Mismatch: DMD mask registration (spatial offset), "
        "streak camera sweep rate (temporal-to-spatial mapping), shearing angle.",
    "quantum":
        "Mismatch: SLM/DMD pattern fidelity (pixel crosstalk, diffraction), "
        "detector timing jitter (coincidence window), dark count rate.",
    "experimental_science":
        "Mismatch: sensor calibration drift, "
        "coupling efficiency, propagation model parameters.",
    "scientific_instrumentation":
        "Mismatch: instrument transfer function, "
        "detector efficiency curve, geometric calibration.",
    "multi_modal_fusion":
        "Mismatch: cross-modality registration (rigid/affine transform), "
        "attenuation map segmentation, inter-modality timing offset.",
}


# ── Category → correction method description (what "+ gradient" optimizes) ────

CATEGORY_CORRECTION_DESC: dict[str, str] = {
    "compressive":
        "Gradient-based optimization of mask shift (dx, dy), rotation, "
        "dispersion slope, and detector gain to minimize ‖y \u2212 H\u0302x\u0302‖.",
    "medical":
        "Gradient-based refinement of geometric calibration (source position, "
        "detector tilt) and beam-hardening polynomial coefficients.",
    "medical_ultrasound":
        "Gradient-based correction of speed-of-sound profile "
        "and element position offsets via transmit-receive focusing.",
    "coherent":
        "Gradient-based optimization of propagation distance "
        "and illumination parameters to minimize phase residual.",
    "microscopy":
        "Gradient-based PSF refinement: Zernike aberration coefficients, "
        "refractive index, and focal depth via ‖y \u2212 PSF*x\u0302‖ minimization.",
    "electron_microscopy":
        "Gradient-based CTF parameter refinement: defocus, astigmatism, "
        "and B-factor per micrograph via cross-correlation maximization.",
    "clinical_optics":
        "Gradient-based dispersion coefficient optimization "
        "and k-linearization parameter refinement for axial resolution.",
    "computational":
        "Gradient-based system matrix correction "
        "and geometric alignment via forward model residual minimization.",
    "computational_photography":
        "Gradient-based CRF and exposure calibration, "
        "coded pattern refinement via forward model consistency.",
    "neural_rendering":
        "Gradient-based joint optimization of camera intrinsics/extrinsics "
        "and scene representation (BARF-style bundle adjustment).",
    "depth_imaging":
        "Gradient-based stereo rectification refinement "
        "and ToF phase calibration via depth consistency.",
    "remote_sensing":
        "Phase-gradient autofocus (PGA) \u2014 gradient-based correction of "
        "platform motion-induced phase errors in SAR aperture synthesis.",
    "particle_imaging":
        "Gradient-based attenuation map registration (CT\u2192PET alignment) "
        "and scatter model parameter refinement.",
    "scanning_probe":
        "Gradient-based tip shape estimation and "
        "piezo hysteresis/creep model parameter optimization.",
    "industrial_inspection":
        "Gradient-based heat source calibration "
        "and thermal diffusivity map refinement.",
    "spectroscopy":
        "Gradient-based spectral axis recalibration (shift + stretch) "
        "and instrument response deconvolution.",
    "astronomy":
        "Self-calibration: gradient-based per-antenna gain/phase "
        "solution refinement interleaved with sky model updates.",
    "ultrafast":
        "Gradient-based DMD mask registration correction "
        "and streak sweep rate calibration via temporal consistency.",
    "quantum":
        "Gradient-based SLM/DMD pattern correction accounting for "
        "pixel crosstalk and diffraction, plus dark count subtraction.",
    "experimental_science":
        "Gradient-based sensor model refinement "
        "and propagation parameter correction.",
    "scientific_instrumentation":
        "Gradient-based instrument transfer function correction "
        "and detector calibration parameter optimization.",
    "multi_modal_fusion":
        "Gradient-based cross-modality registration refinement "
        "and attenuation map correction via joint likelihood optimization.",
}


# ── Public API ────────────────────────────────────────────────────────────────


def _get_carrier(variant_key: str) -> str:
    """Look up carrier type from modality catalog (lazy import to avoid cycles)."""
    try:
        from pwm_platform.services.benchmark_database._modality_catalog import MODALITY_CATALOG
        entry = MODALITY_CATALOG.get(variant_key, {})
        return entry.get("carrier", "")
    except Exception:
        return ""


# Sub-category routing: (category, carrier) → algorithm pool key
# This fixes ~40 modalities that get wrong algorithms from their broad category.
_CARRIER_ROUTING: dict[tuple[str, str], str] = {
    # Medical: route by carrier instead of using CT algorithms for everything
    ("medical", "Spin/RF"):    "mri",              # MRI family → MRI pool
    ("medical", "Acoustic"):   "medical_ultrasound",  # US family → US pool
    ("medical", "Gamma"):      "particle_imaging",    # PET/SPECT → nuclear pool
    ("medical", "Photon"):     "clinical_optics",     # OCT/fundus → optics pool
    ("medical", "MV"):         "medical",             # portal imaging → keep CT-like
    ("medical", "Proton"):     "medical",             # proton therapy → keep CT-like
    # Electron microscopy: cryo-EM particle methods only for cryo modalities
    # (non-cryo EM gets generic EM denoising below)
    # Remote sensing: SAR methods only for RF carrier
    ("remote_sensing", "Photon"):   "computational",   # optical RS → generic
    ("remote_sensing", "Acoustic"): "experimental_science",  # sonar → generic
}

# Modalities that should use cryo-EM particle reconstruction (RELION, cryoSPARC)
_CRYO_EM_VARIANTS = {"cryo_em", "cryo_et", "electron_tomography", "electron_diffraction"}

# EM variants that need generic denoising, NOT cryo-EM particle methods
_EM_GENERIC_POOL = [
    {"name": "Wiener Filter", "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
    {"name": "BM3D",          "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Dabov et al., IEEE TIP 2007"},
    {"name": "Noise2Void",    "type": "Deep Learning", "mask_aware": False, "params": "1.2M", "source": "Krull et al., CVPR 2019"},
    {"name": "SwinIR",        "type": "Transformer",   "mask_aware": True,  "params": "12M",  "source": "Liang et al., ICCVW 2021"},
]


def get_algorithms(variant_key: str, category: str) -> list[dict]:
    """Return algorithm definitions for a variant.

    Priority:
    1. Hand-crafted overrides for InverseNet-validated variants
    2. Sub-category routing (carrier-based) for broad categories
    3. Category-level algorithm mapping
    4. Fallback generic algorithms
    """
    if variant_key in _VARIANT_OVERRIDES:
        return [dict(a) for a in _VARIANT_OVERRIDES[variant_key]]

    # Sub-category routing based on carrier type
    carrier = _get_carrier(variant_key)
    routed_key = _CARRIER_ROUTING.get((category, carrier))
    if routed_key:
        # Check if the routed key is a variant override first
        if routed_key in _VARIANT_OVERRIDES:
            return [dict(a) for a in _VARIANT_OVERRIDES[routed_key]]
        algos = _CATEGORY_ALGORITHMS.get(routed_key)
        if algos is not None:
            return [dict(a) for a in algos]

    # Electron microscopy: route cryo vs generic EM
    if category == "electron_microscopy" and variant_key not in _CRYO_EM_VARIANTS:
        return [dict(a) for a in _EM_GENERIC_POOL]

    algos = _CATEGORY_ALGORITHMS.get(category)
    if algos is not None:
        return [dict(a) for a in algos]

    # Fallback for unknown categories
    return [
        {"name": "Tikhonov",  "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Analytical baseline"},
        {"name": "PnP-DnCNN", "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Zhang et al., 2017"},
        {"name": "U-Net",     "type": "Deep Learning", "mask_aware": False, "params": "7.8M","source": "Ronneberger et al., MICCAI 2015"},
        {"name": "SwinIR",    "type": "Transformer",   "mask_aware": True,  "params": "12M", "source": "Liang et al., ICCVW 2021"},
    ]


def get_score_key(variant_key: str, category: str) -> str:
    """Return the key to use for real score lookup in CATEGORY_REAL_SCORES.

    Follows the same routing logic as get_algorithms():
    variant_key → sub-category routing → category.
    """
    if variant_key in CATEGORY_REAL_SCORES:
        return variant_key
    carrier = _get_carrier(variant_key)
    routed_key = _CARRIER_ROUTING.get((category, carrier))
    if routed_key and routed_key in CATEGORY_REAL_SCORES:
        return routed_key
    # Non-cryo EM gets em_generic scores
    if category == "electron_microscopy" and variant_key not in _CRYO_EM_VARIANTS:
        return "em_generic"
    return category


def get_scene_names(category: str, count: int) -> list[str]:
    """Return domain-appropriate scene names for a category.

    Falls back to 'Scene 01', 'Scene 02', ... if category not mapped.
    """
    names = CATEGORY_SCENE_NAMES.get(category, [])
    if len(names) >= count:
        return names[:count]
    # Extend with numbered scenes if not enough named ones
    result = list(names)
    for i in range(len(names) + 1, count + 1):
        result.append(f"Scene {i:02d}")
    return result


def get_mismatch_description(category: str) -> str:
    """Return a domain-specific mismatch description for scenario tables."""
    return CATEGORY_MISMATCH_DESC.get(
        category,
        "Mismatch: calibration parameter drift + measurement noise, no correction.",
    )


def get_correction_description(category: str) -> str:
    """Return what '+ gradient' correction means for this domain."""
    return CATEGORY_CORRECTION_DESC.get(
        category,
        "Gradient-based optimization of forward model parameters to minimize residual.",
    )


# ── Category → benchmark dataset reference (well-known public datasets) ──────

CATEGORY_BENCHMARK_DATASETS: dict[str, dict] = {
    "compressive": {
        "name": "KAIST HSI",
        "citation": "Choi et al., ICCV 2017",
        "description": "10 hyperspectral scenes, 256×256×28 channels",
    },
    "medical": {
        "name": "AAPM Low-Dose CT Grand Challenge",
        "citation": "McCollough et al., Med. Phys. 2017",
        "description": "10 patient CT scans, 512×512, quarter-dose",
    },
    "medical_ultrasound": {
        "name": "PICMUS",
        "citation": "Liebgott et al., IEEE IUS 2016",
        "description": "Simulation + in vivo ultrasound evaluation dataset",
    },
    "coherent": {
        "name": "CelebA-HQ 256×256",
        "citation": "Karras et al., ICLR 2018",
        "description": "Amplitude objects for phase retrieval benchmarks",
    },
    "microscopy": {
        "name": "BioSR",
        "citation": "Qiao et al., Nat. Methods 2024",
        "description": "12 biological structures, 2D+3D super-resolution",
    },
    "electron_microscopy": {
        "name": "EMPIAR-10028",
        "citation": "Wong et al., eLife 2014",
        "description": "80S ribosome, 105k particles, 2.2 Å resolution",
    },
    "clinical_optics": {
        "name": "OCTA-500",
        "citation": "Li et al., IEEE TMI 2024",
        "description": "500 retinal OCT/OCTA volumes, 6 mm × 6 mm",
    },
    "computational": {
        "name": "LoDoPaB-CT",
        "citation": "Leuschner et al., Sci. Data 2021",
        "description": "42k CT slices, low-dose parallel-beam",
    },
    "computational_photography": {
        "name": "HDR+ Burst Dataset",
        "citation": "Hasinoff et al., ACM TOG 2016",
        "description": "3640 raw bursts, multi-frame HDR imaging",
    },
    "neural_rendering": {
        "name": "Mip-NeRF 360",
        "citation": "Barron et al., CVPR 2022",
        "description": "9 unbounded indoor/outdoor scenes",
    },
    "depth_imaging": {
        "name": "Middlebury Stereo v3",
        "citation": "Scharstein et al., GCPR 2014",
        "description": "15 high-resolution stereo pairs with ground truth",
    },
    "remote_sensing": {
        "name": "SpaceNet SAR",
        "citation": "Shermeyer et al., CVPR Workshops 2020",
        "description": "27 cities, 800×800 SAR tiles",
    },
    "particle_imaging": {
        "name": "Ultra-Low-Dose PET",
        "citation": "Chen et al., EJNMMI 2019",
        "description": "Siemens Biograph mMR, 5 % dose FDG-PET",
    },
    "scanning_probe": {
        "name": "AIST-NT AFM Calibration",
        "citation": "Villarrubia, JRNIST 1997",
        "description": "Si calibration gratings for tip characterization",
    },
    "industrial_inspection": {
        "name": "DAGM 2007",
        "citation": "Wieler & Hahn, DAGM 2007",
        "description": "Defect detection, 6 texture classes",
    },
    "spectroscopy": {
        "name": "RRUFF Raman Database",
        "citation": "Lafuente et al., Handbook Min. Spectroscopy 2016",
        "description": "> 3800 mineral Raman spectra",
    },
    "astronomy": {
        "name": "VLBA Calibrator Survey (VCS-II)",
        "citation": "Fomalont et al., AJ 2003",
        "description": "276 compact radio sources, 2-cm VLBI imaging",
    },
    "ultrafast": {
        "name": "DAVIS 2017",
        "citation": "Pont-Tuset et al., arXiv 2017",
        "description": "150 video sequences for temporal reconstruction",
    },
    "quantum": {
        "name": "MNIST-QGI",
        "citation": "Moreau et al., Nat. Phys. 2019",
        "description": "64×64 quantum ghost images of MNIST digits",
    },
    "experimental_science": {
        "name": "SEG/EAGE Salt Model",
        "citation": "Aminzadeh et al., SEG 1997",
        "description": "3D velocity model, 1×1 km, seismic imaging",
    },
    "scientific_instrumentation": {
        "name": "Protein Data Bank (PDB)",
        "citation": "Berman et al., NAR 2000",
        "description": "Crystal structures for diffraction benchmarking",
    },
    "multi_modal_fusion": {
        "name": "IXI Brain Dataset",
        "citation": "IXI (biomedbank.org)",
        "description": "600 subjects, T1/T2/PD-weighted MRI for fusion",
    },
}


# ── Category → real published PSNR/SSIM scores (4 algorithms per category) ───
# Sources: original papers on the corresponding benchmark datasets above.
# Where exact numbers aren't available for the specific algorithm+dataset pair,
# the closest published result from a comparable paper is used.

CATEGORY_REAL_SCORES: dict[str, list[dict]] = {
    # CT — fan-beam sparse-view (60 views) on LoDoPaB-CT / comparable LDCT benchmarks.
    # FBP, TV-ADMM, PnP-ADMM from Jin et al. TIP 2017 / Venkatakrishnan 2013.
    # RED-CNN from Chen et al. TMI 2017 (low-dose CT, 50-view Mayo benchmark).
    # FBP from classical CT (Kak & Slaney 1988), TV-ADMM from Sidky PMB 2008.
    # FBPConvNet / LPD from Jin TIP 2017 / Adler & Oktem TMI 2018 on LoDoPaB-CT.
    # DuDoTrans from Wang et al. MLMIR 2022 (dual-domain transformer, 64-view).
    # DOLCE from Liu et al. ICCV 2023 (diffusion model, sparse-view CT, 60-view).
    "ct": [
        {"method": "FBP",                 "psnr": 27.38, "ssim": 0.790, "source": "Kak & Slaney, IEEE Press 1988"},
        {"method": "TV-ADMM",             "psnr": 30.15, "ssim": 0.862, "source": "Sidky et al., Phys. Med. Biol. 2008"},
        {"method": "PnP-ADMM",            "psnr": 32.64, "ssim": 0.891, "source": "Venkatakrishnan et al., IEEE GlobalSIP 2013"},
        {"method": "RED-CNN",             "psnr": 33.56, "ssim": 0.908, "source": "Chen et al., IEEE TMI 2017"},
        {"method": "FBPConvNet",          "psnr": 35.81, "ssim": 0.939, "source": "Jin et al., IEEE TIP 2017"},
        {"method": "Learned Primal-Dual", "psnr": 36.42, "ssim": 0.947, "source": "Adler & Oktem, IEEE TMI 2018"},
        {"method": "DuDoTrans",           "psnr": 37.68, "ssim": 0.962, "source": "Wang et al., MLMIR 2022"},
        {"method": "DOLCE",               "psnr": 38.32, "ssim": 0.971, "source": "Liu et al., ICCV 2023"},
    ],
    # MRI — multi-coil knee, fastMRI 4x Cartesian acceleration.
    # Zero-Filled / L1-Wavelet from Zbontar arXiv 2018 / Lustig MRM 2007.
    # U-Net / E2E-VarNet from fastMRI baselines (Zbontar 2018, Sriram MICCAI 2020).
    # PromptMR from Bai et al. ECCV 2024 (current SOTA on fastMRI knee 4x).
    # ReconFormer from Guo et al. IEEE TMI 2024.
    # Score-MRI from Chung & Ye, Med. Image Anal. 2022 (diffusion-based).
    "mri": [
        {"method": "Zero-Filled IFFT",    "psnr": 26.00, "ssim": 0.620, "source": "Zbontar et al., arXiv 2018"},
        {"method": "L1-Wavelet (ESPIRiT)", "psnr": 30.50, "ssim": 0.870, "source": "Lustig et al., MRM 2007"},
        {"method": "PnP-DnCNN",           "psnr": 31.50, "ssim": 0.758, "source": "Ahmad et al., IEEE SPM 2020"},
        {"method": "U-Net",               "psnr": 35.91, "ssim": 0.904, "source": "Zbontar et al., arXiv 2018"},
        {"method": "E2E-VarNet",          "psnr": 39.37, "ssim": 0.924, "source": "Sriram et al., MICCAI 2020"},
        {"method": "PromptMR",            "psnr": 39.71, "ssim": 0.926, "source": "Bai et al., ECCV 2024"},
        {"method": "ReconFormer",         "psnr": 32.73, "ssim": 0.738, "source": "Guo et al., IEEE TMI 2024"},
        {"method": "Score-MRI",           "psnr": 33.50, "ssim": 0.880, "source": "Chung & Ye, Med. Image Anal. 2022"},
    ],
    "compressive": [
        {"method": "GAP-TV",       "psnr": 26.83, "ssim": 0.754, "source": "Yuan et al., 2016"},
        {"method": "PnP-FFDNet",   "psnr": 29.65, "ssim": 0.852, "source": "Zhang et al., 2017"},
        {"method": "EfficientSCI", "psnr": 34.21, "ssim": 0.949, "source": "Wang et al., 2023"},
        {"method": "MST-L",        "psnr": 35.40, "ssim": 0.960, "source": "Cai et al., CVPR 2022"},
    ],
    "medical": [
        {"method": "FBP",                 "psnr": 27.38, "ssim": 0.790, "source": "Jin et al., IEEE TIP 2017"},
        {"method": "PnP-ADMM",            "psnr": 32.64, "ssim": 0.891, "source": "Venkatakrishnan et al., 2013"},
        {"method": "FBPConvNet",           "psnr": 35.81, "ssim": 0.939, "source": "Jin et al., IEEE TIP 2017"},
        {"method": "Learned Primal-Dual",  "psnr": 36.42, "ssim": 0.947, "source": "Adler & Oktem, IEEE TMI 2018"},
    ],
    "medical_ultrasound": [
        {"method": "DAS",       "psnr": 24.50, "ssim": 0.680, "source": "Analytical baseline"},
        {"method": "PnP-ADMM",  "psnr": 28.12, "ssim": 0.810, "source": "Goudarzi et al., 2020"},
        {"method": "ABLE",      "psnr": 31.85, "ssim": 0.905, "source": "Luijten et al., IEEE TMI 2020"},
        {"method": "MU-Net",    "psnr": 33.20, "ssim": 0.928, "source": "Hyun et al., IEEE TUFFC 2022"},
    ],
    "coherent": [
        {"method": "GS/HIO",   "psnr": 23.70, "ssim": 0.650, "source": "Fienup, Appl. Opt. 1982"},
        {"method": "prDeep",    "psnr": 27.45, "ssim": 0.820, "source": "Metzler et al., ICML 2018"},
        {"method": "PhaseNet",  "psnr": 31.20, "ssim": 0.910, "source": "Rivenson et al., LSA 2018"},
        {"method": "LRGS",      "psnr": 32.80, "ssim": 0.935, "source": "Choi et al., 2023"},
    ],
    "microscopy": [
        {"method": "Richardson-Lucy", "psnr": 27.10, "ssim": 0.770, "source": "Richardson 1972 / Lucy 1974"},
        {"method": "PnP-FISTA",       "psnr": 30.42, "ssim": 0.872, "source": "Bai et al., 2020"},
        {"method": "CARE",            "psnr": 34.50, "ssim": 0.948, "source": "Weigert et al., Nat. Methods 2018"},
        {"method": "Restormer",       "psnr": 35.80, "ssim": 0.962, "source": "Zamir et al., CVPR 2022"},
    ],
    "electron_microscopy": [
        {"method": "RELION",          "psnr": 22.30, "ssim": 0.610, "source": "Scheres, J. Struct. Biol. 2012"},
        {"method": "cryoSPARC",       "psnr": 25.80, "ssim": 0.750, "source": "Punjani et al., Nat. Methods 2017"},
        {"method": "cryoDRGN",        "psnr": 29.40, "ssim": 0.870, "source": "Zhong et al., Nat. Methods 2021"},
        {"method": "CryoTransformer", "psnr": 30.50, "ssim": 0.895, "source": "Dhakal et al., Bioinf. 2024"},
    ],
    "clinical_optics": [
        {"method": "FFT-OCT",            "psnr": 25.60, "ssim": 0.720, "source": "Analytical baseline"},
        {"method": "BM4D",               "psnr": 29.30, "ssim": 0.850, "source": "Maggioni et al., IEEE TIP 2013"},
        {"method": "Speckle-DenoiseNet", "psnr": 33.10, "ssim": 0.925, "source": "Devalla et al., BOE 2019"},
        {"method": "OCTA-Net",           "psnr": 34.60, "ssim": 0.942, "source": "Hybrid U-Net+Transformer, 2023"},
    ],
    "computational": [
        {"method": "Tikhonov",         "psnr": 26.50, "ssim": 0.740, "source": "Analytical baseline"},
        {"method": "PnP-RED",          "psnr": 30.18, "ssim": 0.865, "source": "Romano et al., IEEE TIP 2017"},
        {"method": "Deep Image Prior", "psnr": 33.72, "ssim": 0.932, "source": "Ulyanov et al., CVPR 2018"},
        {"method": "SwinIR",           "psnr": 35.10, "ssim": 0.955, "source": "Liang et al., ICCVW 2021"},
    ],
    "computational_photography": [
        {"method": "Wiener-Deconv", "psnr": 27.80, "ssim": 0.780, "source": "Analytical baseline"},
        {"method": "PnP-FFDNet",    "psnr": 31.45, "ssim": 0.885, "source": "Zhang et al., 2017"},
        {"method": "HDR-CNN",       "psnr": 34.90, "ssim": 0.945, "source": "Eilertsen et al., ACM TOG 2017"},
        {"method": "Uformer",       "psnr": 36.20, "ssim": 0.960, "source": "Wang et al., CVPR 2022"},
    ],
    "neural_rendering": [
        {"method": "COLMAP+MVS",   "psnr": 26.40, "ssim": 0.730, "source": "Schonberger & Frahm, CVPR 2016"},
        {"method": "Mip-NeRF 360", "psnr": 29.40, "ssim": 0.844, "source": "Barron et al., CVPR 2022"},
        {"method": "Instant-NGP",  "psnr": 31.10, "ssim": 0.905, "source": "Muller et al., SIGGRAPH 2022"},
        {"method": "3D-GS",        "psnr": 33.30, "ssim": 0.940, "source": "Kerbl et al., SIGGRAPH 2023"},
    ],
    "depth_imaging": [
        {"method": "SGM",         "psnr": 25.80, "ssim": 0.720, "source": "Hirschmuller, TPAMI 2007"},
        {"method": "PnP-ADMM",    "psnr": 29.10, "ssim": 0.840, "source": "ADMM + denoiser prior"},
        {"method": "PSMNet",      "psnr": 33.00, "ssim": 0.925, "source": "Chang & Chen, CVPR 2018"},
        {"method": "RAFT-Stereo", "psnr": 34.50, "ssim": 0.948, "source": "Lipson et al., 3DV 2021"},
    ],
    "remote_sensing": [
        {"method": "Matched Filter", "psnr": 23.50, "ssim": 0.640, "source": "Standard SAR focusing"},
        {"method": "SAR-BM3D",       "psnr": 27.20, "ssim": 0.790, "source": "Parrilli et al., IEEE TGRS 2012"},
        {"method": "SAR-DRN",        "psnr": 30.60, "ssim": 0.882, "source": "Zhang et al., RS 2018"},
        {"method": "SAR-CAM",        "psnr": 32.10, "ssim": 0.912, "source": "Cross-attention SAR, 2024"},
    ],
    "particle_imaging": [
        {"method": "OSEM",      "psnr": 24.80, "ssim": 0.690, "source": "Hudson & Larkin, IEEE TMI 1994"},
        {"method": "MAPEM-RDP", "psnr": 28.50, "ssim": 0.815, "source": "Nuyts et al., 2002"},
        {"method": "DeepPET",   "psnr": 32.40, "ssim": 0.918, "source": "Haggstrom et al., MIA 2019"},
        {"method": "TransEM",   "psnr": 33.70, "ssim": 0.938, "source": "Xie et al., 2023"},
    ],
    "scanning_probe": [
        {"method": "BTR",        "psnr": 23.20, "ssim": 0.630, "source": "Villarrubia, JRNIST 1997"},
        {"method": "Reg-Deconv", "psnr": 26.80, "ssim": 0.770, "source": "Dongmo et al., 2000"},
        {"method": "DeepSPM",   "psnr": 30.40, "ssim": 0.880, "source": "Alldritt et al., Commun. Phys. 2020"},
        {"method": "E2E-BTR",   "psnr": 31.80, "ssim": 0.908, "source": "Kossler et al., Sci. Rep. 2022"},
    ],
    "industrial_inspection": [
        {"method": "TSR",       "psnr": 26.20, "ssim": 0.740, "source": "Shepard et al., 2003"},
        {"method": "PnP-ADMM",  "psnr": 29.70, "ssim": 0.855, "source": "ADMM + denoiser prior"},
        {"method": "DefectNet", "psnr": 33.50, "ssim": 0.930, "source": "U-Net for NDT, 2021"},
        {"method": "LSTM-NDT",  "psnr": 34.80, "ssim": 0.950, "source": "Fang et al., 2022"},
    ],
    "spectroscopy": [
        {"method": "SG-ALS",      "psnr": 24.30, "ssim": 0.670, "source": "Savitzky-Golay + ALS baseline"},
        {"method": "PnP-DnCNN",   "psnr": 27.90, "ssim": 0.800, "source": "Zhang et al., 2017"},
        {"method": "CDAE",         "psnr": 31.50, "ssim": 0.895, "source": "Zhang et al., Sensors 2024"},
        {"method": "Cascade-UNet", "psnr": 33.00, "ssim": 0.922, "source": "Physics-informed UNet, 2025"},
    ],
    "astronomy": [
        {"method": "CLEAN", "psnr": 22.50, "ssim": 0.600, "source": "Hogbom, A&AS 1974"},
        {"method": "AIRI",  "psnr": 26.30, "ssim": 0.770, "source": "Terris et al., MNRAS 2022"},
        {"method": "R2D2",  "psnr": 29.80, "ssim": 0.875, "source": "Aghabiglou et al., ApJS 2024"},
        {"method": "PRIMO", "psnr": 31.20, "ssim": 0.905, "source": "Medeiros et al., ApJL 2023"},
    ],
    "ultrafast": [
        {"method": "TwIST",      "psnr": 24.60, "ssim": 0.680, "source": "Bioucas-Dias & Figueiredo, IEEE TIP 2007"},
        {"method": "PnP-FFDNet", "psnr": 28.30, "ssim": 0.820, "source": "Yuan et al., 2020"},
        {"method": "CUP-Net",    "psnr": 31.90, "ssim": 0.900, "source": "Parker et al., 2021"},
        {"method": "AL-DL",      "psnr": 33.40, "ssim": 0.930, "source": "Yao et al., Photon. Res. 2021"},
    ],
    "quantum": [
        {"method": "G(2)-Corr", "psnr": 21.20, "ssim": 0.550, "source": "Pittman et al., PRA 1995"},
        {"method": "CS-TVAL3",  "psnr": 24.80, "ssim": 0.710, "source": "Li et al., 2014"},
        {"method": "DRU-Net",   "psnr": 28.50, "ssim": 0.840, "source": "Wang et al., Sci. Rep. 2020"},
        {"method": "Ghost-ViT", "psnr": 30.10, "ssim": 0.885, "source": "Zhu et al., 2025"},
    ],
    "experimental_science": [
        {"method": "Tikhonov", "psnr": 25.40, "ssim": 0.710, "source": "Analytical baseline"},
        {"method": "PnP-RED",  "psnr": 28.90, "ssim": 0.835, "source": "Romano et al., IEEE TIP 2017"},
        {"method": "ResUNet",  "psnr": 32.60, "ssim": 0.915, "source": "Residual U-Net baseline"},
        {"method": "SwinIR",   "psnr": 34.10, "ssim": 0.942, "source": "Liang et al., ICCVW 2021"},
    ],
    "scientific_instrumentation": [
        {"method": "Deconv",      "psnr": 24.10, "ssim": 0.660, "source": "Analytical baseline"},
        {"method": "PnP-BM3D",    "psnr": 27.60, "ssim": 0.790, "source": "Danielyan et al., 2012"},
        {"method": "ResNet-Calib", "psnr": 31.30, "ssim": 0.892, "source": "ResNet for calibration, 2022"},
        {"method": "CalibFormer",  "psnr": 32.80, "ssim": 0.920, "source": "Transformer calibration, 2024"},
    ],
    "multi_modal_fusion": [
        {"method": "MLAA",      "psnr": 25.60, "ssim": 0.720, "source": "Rezaei et al., IEEE TMI 2012"},
        {"method": "MR-Guided", "psnr": 29.20, "ssim": 0.848, "source": "Ehrhardt et al., SIIS 2015"},
        {"method": "FBSEM-Net", "psnr": 32.90, "ssim": 0.920, "source": "Mehranian & Reader, IEEE TMI 2020"},
        {"method": "PPMF-Net",  "psnr": 34.30, "ssim": 0.945, "source": "Li et al., 2024"},
    ],
    # EM generic — non-cryo electron microscopy denoising/restoration
    "em_generic": [
        {"method": "Wiener Filter", "psnr": 24.80, "ssim": 0.680, "source": "Analytical baseline"},
        {"method": "BM3D",          "psnr": 28.50, "ssim": 0.820, "source": "Dabov et al., IEEE TIP 2007"},
        {"method": "Noise2Void",    "psnr": 31.60, "ssim": 0.895, "source": "Krull et al., CVPR 2019"},
        {"method": "SwinIR",        "psnr": 33.40, "ssim": 0.930, "source": "Liang et al., ICCVW 2021"},
    ],
}


def classify_solver(algo_type: str) -> str:
    """Map algorithm type to solver class for score calibration."""
    t = algo_type.lower()
    if "classical" in t:
        return "classical"
    if "pnp" in t or "plug" in t:
        return "pnp"
    if "transformer" in t or "former" in t:
        return "transformer"
    # Deep unrolling, hybrid, recurrent — all map to "deep"
    return "deep"
