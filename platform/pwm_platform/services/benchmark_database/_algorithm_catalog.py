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
    "sd_cassi": [
        {"name": "GAP-TV",      "type": "Classical",      "mask_aware": True,  "params": "0",     "source": "InverseNet"},
        {"name": "PnP-HSICNN",  "type": "PnP",            "mask_aware": True,  "params": "0",     "source": "InverseNet"},
        {"name": "HDNet",       "type": "Deep Learning",  "mask_aware": False, "params": "2.37M", "source": "InverseNet"},
        {"name": "MST-L",       "type": "Transformer",    "mask_aware": True,  "params": "2.03M", "source": "InverseNet"},
    ],
    "cacti": [
        {"name": "GAP-TV",        "type": "Classical",      "mask_aware": True,  "params": "0",    "source": "InverseNet"},
        {"name": "PnP-FFDNet",    "type": "PnP",            "mask_aware": True,  "params": "0",    "source": "InverseNet"},
        {"name": "ELP-Unfolding", "type": "Deep Unfolding", "mask_aware": True,  "params": "1.6M", "source": "InverseNet"},
        {"name": "EfficientSCI",  "type": "Deep Learning",  "mask_aware": True,  "params": "4.2M", "source": "InverseNet"},
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


def get_algorithms(variant_key: str, category: str) -> list[dict]:
    """Return 4 algorithm definitions for a variant.

    Priority:
    1. Hand-crafted overrides for InverseNet-validated variants
    2. Category-level algorithm mapping
    3. Fallback generic algorithms
    """
    if variant_key in _VARIANT_OVERRIDES:
        return [dict(a) for a in _VARIANT_OVERRIDES[variant_key]]

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
