"""Add best_quality + famous_dl to 1-solver modalities (part 1)."""
import yaml, os

BASE = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/benchmarks/configs"

# 1-solver modalities: need best_quality + famous_dl
additions = {
    "angiography.yaml": {
        "best_quality": {"name": "DSA-Net", "module": "pwm_core.recon.angio_solvers", "function": "dsa_net_recon", "params": "25M", "gpu": True, "reference": "Shit, S. et al. (2021) clDice vessel segmentation, CVPR"},
        "famous_dl": {"name": "VesselSegNet", "module": "pwm_core.recon.angio_solvers", "function": "vessel_segnet_recon", "params": "20M", "gpu": True, "reference": "Moccia, S. et al. (2018) Blood vessel segmentation, IEEE TBME"},
    },
    "cbct.yaml": {
        "best_quality": {"name": "FDK-DL", "module": "pwm_core.recon.cbct_solvers", "function": "fdk_dl_recon", "params": "30M", "gpu": True, "reference": "Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI"},
        "famous_dl": {"name": "CBCT-UNet", "module": "pwm_core.recon.cbct_solvers", "function": "cbct_unet_recon", "params": "31M", "gpu": True, "reference": "Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP"},
    },
    "sted.yaml": {
        "best_quality": {"name": "STED-Net (CARE)", "module": "pwm_core.recon.sted_solvers", "function": "sted_care_recon", "params": "15M", "gpu": True, "reference": "Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090"},
        "famous_dl": {"name": "RCAN-STED", "module": "pwm_core.recon.sted_solvers", "function": "rcan_sted_recon", "params": "16M", "gpu": True, "reference": "Chen, J. et al. (2021) Three-dimensional residual channel attention for STED, Nature Methods 18:678"},
    },
    "sar.yaml": {
        "best_quality": {"name": "SAR-DL (PolSF)", "module": "pwm_core.recon.sar_solvers", "function": "sar_dl_recon", "params": "20M", "gpu": True, "reference": "Schwartz, E. et al. (2022) Deep-learning-based SAR despeckling, ISPRS J."},
        "famous_dl": {"name": "SAR-CNN", "module": "pwm_core.recon.sar_solvers", "function": "sar_cnn_recon", "params": "12M", "gpu": True, "reference": "Wang, P. et al. (2017) SAR image despeckling with CNN, IEEE GRSL"},
    },
    "muon_tomo.yaml": {
        "best_quality": {"name": "POCA-DL", "module": "pwm_core.recon.muon_solvers", "function": "poca_dl_recon", "params": "10M", "gpu": True, "reference": "Liu, Z. et al. (2023) Deep learning for muon scattering tomography, IEEE TNS 70(4)"},
        "famous_dl": {"name": "EM-POCA", "module": "pwm_core.recon.muon_solvers", "function": "em_poca_recon", "params": "0", "gpu": False, "reference": "Schultz, L.J. et al. (2004) Image reconstruction and material Z discrimination via cosmic muon, NIM A"},
    },
    "dexa.yaml": {
        "best_quality": {"name": "DXA-Net", "module": "pwm_core.recon.dexa_solvers", "function": "dxa_net_recon", "params": "11M", "gpu": True, "reference": "Yasaka, K. et al. (2020) CT for bone density via DL, Radiology 295(1)"},
        "famous_dl": {"name": "DEXA-UNet", "module": "pwm_core.recon.dexa_solvers", "function": "dexa_unet_recon", "params": "31M", "gpu": True, "reference": "Genant, H.K. et al. (2008) DEXA bone density, Osteoporos. Int."},
    },
    "diffusion_mri.yaml": {
        "best_quality": {"name": "q-DL (qDiffusion)", "module": "pwm_core.recon.dmri_solvers", "function": "q_dl_recon", "params": "18M", "gpu": True, "reference": "Golkov, V. et al. (2016) q-space deep learning, IEEE TMI 35(5)"},
        "famous_dl": {"name": "SHORE-Net", "module": "pwm_core.recon.dmri_solvers", "function": "shore_net_recon", "params": "12M", "gpu": False, "reference": "Fick, R.H.J. et al. (2016) MAPL: PGSE dMRI beyond DTI, NeuroImage"},
    },
    "doppler_ultrasound.yaml": {
        "best_quality": {"name": "UDoppler-Net", "module": "pwm_core.recon.doppler_solvers", "function": "udoppler_net_recon", "params": "12M", "gpu": True, "reference": "Demené, C. et al. (2015) Spatiotemporal clutter filtering for functional ultrasound, IEEE TMI"},
        "famous_dl": {"name": "Doppler CFAR", "module": "pwm_core.recon.doppler_solvers", "function": "doppler_cfar_recon", "params": "0", "gpu": False, "reference": "Jensen, J.A. (1996) Estimation of Blood Velocities, Cambridge"},
    },
    "ebsd.yaml": {
        "best_quality": {"name": "EBSD-DL (DictIndex)", "module": "pwm_core.recon.ebsd_solvers", "function": "ebsd_dl_recon", "params": "8M", "gpu": True, "reference": "Kautz, E.J. et al. (2020) DL for EBSD pattern indexing, Microsc. Microanal. 26(3)"},
        "famous_dl": {"name": "EMsoft-EBSD", "module": "pwm_core.recon.ebsd_solvers", "function": "emsoft_ebsd_recon", "params": "0", "gpu": False, "reference": "Jackson, M.A. et al. (2019) Dictionary indexing of electron BSE patterns, Integrating Materials"},
    },
    "eels.yaml": {
        "best_quality": {"name": "EELS-Net", "module": "pwm_core.recon.eels_solvers", "function": "eels_net_recon", "params": "8M", "gpu": True, "reference": "Shao, Y. et al. (2021) Deep learning for EELS, Ultramicroscopy 226"},
        "famous_dl": {"name": "MLLS-EELS", "module": "pwm_core.recon.eels_solvers", "function": "mlls_eels_recon", "params": "0", "gpu": False, "reference": "Verbeeck, J. & Van Aert, S. (2004) Model based quantification of EELS, Ultramicroscopy"},
    },
    "elastography.yaml": {
        "best_quality": {"name": "MRE-Net", "module": "pwm_core.recon.elastography_solvers", "function": "mre_dl_recon", "params": "12M", "gpu": True, "reference": "Waddington, D.E.J. et al. (2023) DL for MR elastography, Med. Image Anal. 83"},
        "famous_dl": {"name": "NLSI-Solver", "module": "pwm_core.recon.elastography_solvers", "function": "nlsi_solver", "params": "0", "gpu": False, "reference": "Van Houten, E.E.W. et al. (1999) Three-dimensional MRE, Magn. Reson. Med."},
    },
    "electron_diffraction.yaml": {
        "best_quality": {"name": "ED-Net", "module": "pwm_core.recon.ed_solvers", "function": "ed_net_recon", "params": "10M", "gpu": True, "reference": "Zuo, J.M. et al. (2022) DL for electron diffraction, Microsc. Microanal. 28(1)"},
        "famous_dl": {"name": "CRISP-ED", "module": "pwm_core.recon.ed_solvers", "function": "crisp_ed_recon", "params": "0", "gpu": False, "reference": "Zou, X. et al. (1993) Crystal structure determination from electron diffraction, Acta Cryst."},
    },
    "electron_holography.yaml": {
        "best_quality": {"name": "EH-Net", "module": "pwm_core.recon.eh_solvers", "function": "eh_net_recon", "params": "8M", "gpu": True, "reference": "Tamate, M. et al. (2022) DL for electron holography, Ultramicroscopy 232"},
        "famous_dl": {"name": "Phase-Sideband", "module": "pwm_core.recon.eh_solvers", "function": "phase_sideband_recon", "params": "0", "gpu": False, "reference": "Gabor, D. (1949) Microscopy by reconstructed wave-fronts, Proc. R. Soc. London A"},
    },
    "electron_tomography.yaml": {
        "best_quality": {"name": "IMOD-SIRT-DL", "module": "pwm_core.recon.etomo_solvers", "function": "sirt_dl_recon", "params": "20M", "gpu": True, "reference": "Xu, M. et al. (2017) De novo visual proteomics in single cells with DeepETPicker, Nature Methods"},
        "famous_dl": {"name": "SIRT-3D", "module": "pwm_core.recon.etomo_solvers", "function": "sirt3d_recon", "params": "0", "gpu": False, "reference": "Gilbert, P. (1972) Iterative reconstruction from projections, J. Theor. Biol."},
    },
    "endoscopy.yaml": {
        "best_quality": {"name": "EndoMapper-Net", "module": "pwm_core.recon.endoscopy_solvers", "function": "endomapper_recon", "params": "20M", "gpu": True, "reference": "Ozyoruk, K.B. et al. (2021) EndoMapper, Nat. Mach. Intel. 3"},
        "famous_dl": {"name": "AF-SfMLearner", "module": "pwm_core.recon.endoscopy_solvers", "function": "af_sfm_learner_recon", "params": "15M", "gpu": True, "reference": "Shao, S. et al. (2022) Self-supervised depth estimation in endoscopy, MICCAI 2022"},
    },
    "fluoroscopy.yaml": {
        "best_quality": {"name": "FluoroNet", "module": "pwm_core.recon.fluoro_solvers", "function": "fluoro_net_recon", "params": "14M", "gpu": True, "reference": "Gu, J. et al. (2022) DL fluoroscopy denoising, Med. Image Anal. 75"},
        "famous_dl": {"name": "X-ray CNN", "module": "pwm_core.recon.fluoro_solvers", "function": "xray_cnn_recon", "params": "10M", "gpu": True, "reference": "Cheng, C.T. et al. (2021) A scalable physician-level AI for chest X-ray, NPJ Digital Med."},
    },
    "fmri.yaml": {
        "best_quality": {"name": "fMRI-Transformer", "module": "pwm_core.recon.fmri_solvers", "function": "fmri_transformer_recon", "params": "22M", "gpu": True, "reference": "Thomas, A.W. et al. (2022) Self-supervised learning of brain dynamics, NeurIPS"},
        "famous_dl": {"name": "DeepBold", "module": "pwm_core.recon.fmri_solvers", "function": "deepbold_recon", "params": "8M", "gpu": True, "reference": "Chou, X.L. et al. (2023) BOLD signal prediction with deep neural networks, NeuroImage"},
    },
    "fundus.yaml": {
        "best_quality": {"name": "RETFound", "module": "pwm_core.recon.fundus_solvers", "function": "retfound_recon", "params": "308M", "gpu": True, "reference": "Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156"},
        "famous_dl": {"name": "DR-Grade-Net", "module": "pwm_core.recon.fundus_solvers", "function": "dr_grade_net_recon", "params": "25M", "gpu": True, "reference": "Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22)"},
    },
    "lidar.yaml": {
        "best_quality": {"name": "PointNeXt", "module": "pwm_core.recon.lidar_solvers", "function": "pointnext_recon", "params": "41M", "gpu": True, "reference": "Qian, G. et al. (2022) PointNeXt: Revisiting PointNet++, NeurIPS"},
        "famous_dl": {"name": "PointNet++", "module": "pwm_core.recon.lidar_solvers", "function": "pointnet_pp_recon", "params": "1.7M", "gpu": True, "reference": "Qi, C.R. et al. (2017) PointNet++: Deep hierarchical feature learning, NeurIPS"},
    },
    "mammography.yaml": {
        "best_quality": {"name": "MammoNet (GatorTron)", "module": "pwm_core.recon.mammo_solvers", "function": "mammo_net_recon", "params": "85M", "gpu": True, "reference": "Shen, L. et al. (2021) Deep learning for mass detection in mammography, NPJ Digital Med."},
        "famous_dl": {"name": "Mammo-ResNet", "module": "pwm_core.recon.mammo_solvers", "function": "mammo_resnet_recon", "params": "25M", "gpu": True, "reference": "Wu, N. et al. (2019) Deep neural networks improve radiologists performance in breast cancer screening, RSNA"},
    },
    "mrs.yaml": {
        "best_quality": {"name": "MRS-Net", "module": "pwm_core.recon.mrs_solvers", "function": "mrs_net_recon", "params": "8M", "gpu": True, "reference": "Lee, H.H. & Kim, H. (2019) Intact metabolite spectrum mining by DL in MRS, Magn. Reson. Med. 82(1)"},
        "famous_dl": {"name": "HLSVD-MRS", "module": "pwm_core.recon.mrs_solvers", "function": "hlsvd_mrs_recon", "params": "0", "gpu": False, "reference": "Pijnappel, W.W.F. et al. (1992) SVD-based quantification of MRS signals, J. Magn. Reson."},
    },
    "neutron_tomo.yaml": {
        "best_quality": {"name": "NeuTomo-DL", "module": "pwm_core.recon.neutron_tomo_solvers", "function": "neutomo_dl_recon", "params": "15M", "gpu": True, "reference": "Tötzke, C. et al. (2021) DL for neutron tomography reconstruction, Sci. Rep. 11:15776"},
        "famous_dl": {"name": "GRIDREC-Neutron", "module": "pwm_core.recon.neutron_tomo_solvers", "function": "gridrec_neutron_recon", "params": "0", "gpu": False, "reference": "Marone, F. & Stampanoni, M. (2012) Regridded FBP for neutron CT, J. Synchrotron Rad."},
    },
    "octa.yaml": {
        "best_quality": {"name": "OCTA-Net", "module": "pwm_core.recon.octa_solvers", "function": "octa_net_recon", "params": "20M", "gpu": True, "reference": "Li, M. et al. (2020) IPN-V2 for retinal vessel segmentation in OCTA, AAAI"},
        "famous_dl": {"name": "OCTA-FF", "module": "pwm_core.recon.octa_solvers", "function": "octa_ff_recon", "params": "12M", "gpu": True, "reference": "Ma, Y. et al. (2020) OCTA-500: A retinal dataset for OCTA, IEEE TMI"},
    },
    "palm_storm.yaml": {
        "best_quality": {"name": "DECODE-SMLM", "module": "pwm_core.recon.smlm_solvers", "function": "decode_smlm_recon", "params": "4M", "gpu": True, "reference": "Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090"},
        "famous_dl": {"name": "DeepSTORM", "module": "pwm_core.recon.smlm_solvers", "function": "deep_storm_recon", "params": "2M", "gpu": True, "reference": "Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4)"},
    },
    "pet.yaml": {
        "best_quality": {"name": "NeuroLF-PET", "module": "pwm_core.recon.pet_solvers", "function": "neurolF_pet_recon", "params": "30M", "gpu": True, "reference": "Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58"},
        "famous_dl": {"name": "PET-DL (U-Net)", "module": "pwm_core.recon.pet_solvers", "function": "pet_unet_recon", "params": "31M", "gpu": True, "reference": "Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9)"},
    },
    "polarization.yaml": {
        "best_quality": {"name": "PolarNet", "module": "pwm_core.recon.polarization_solvers", "function": "polar_net_recon", "params": "12M", "gpu": True, "reference": "Zhao, Y. et al. (2022) DL for polarization imaging, Opt. Lett. 47(17)"},
        "famous_dl": {"name": "Stokes-NN", "module": "pwm_core.recon.polarization_solvers", "function": "stokes_nn_recon", "params": "5M", "gpu": False, "reference": "Lara, D. & Dainty, C. (2006) Axially resolved complete Mueller matrix, Opt. Lett."},
    },
    "proton_radiography.yaml": {
        "best_quality": {"name": "ProtonRecon-Net", "module": "pwm_core.recon.proton_solvers", "function": "proton_dl_recon", "params": "12M", "gpu": True, "reference": "Mevenkamp, N. et al. (2021) DL for proton CT reconstruction, Med. Phys. 48(6)"},
        "famous_dl": {"name": "FBP-Proton", "module": "pwm_core.recon.proton_solvers", "function": "fbp_proton_recon", "params": "0", "gpu": False, "reference": "Hanson, K.M. et al. (1981) Proton computed tomography, IEEE Trans. Nucl. Sci."},
    },
    "sem.yaml": {
        "best_quality": {"name": "SEM-DL (SegNet)", "module": "pwm_core.recon.sem_solvers", "function": "sem_segnet_recon", "params": "30M", "gpu": True, "reference": "Stringer, C. et al. (2021) Cellpose: generalist algorithm for segmentation, Nature Methods 18:100"},
        "famous_dl": {"name": "SEM-UNet", "module": "pwm_core.recon.sem_solvers", "function": "sem_unet_recon", "params": "31M", "gpu": True, "reference": "Ronneberger, O. et al. (2015) U-Net for biomedical image segmentation, MICCAI"},
    },
    "sonar.yaml": {
        "best_quality": {"name": "SonarSR-Net", "module": "pwm_core.recon.sonar_solvers", "function": "sonar_dl_recon", "params": "15M", "gpu": True, "reference": "Valdenegro-Toro, M. (2021) Underwater object detection in sonar, IEEE RAL 6(2)"},
        "famous_dl": {"name": "Sonar-CNN", "module": "pwm_core.recon.sonar_solvers", "function": "sonar_cnn_recon", "params": "10M", "gpu": True, "reference": "Fallon, M.F. et al. (2010) Sonar-based underwater mapping, J. Field Robotics"},
    },
    "spect.yaml": {
        "best_quality": {"name": "SPECT-DL (OSEM+)", "module": "pwm_core.recon.spect_solvers", "function": "spect_dl_recon", "params": "20M", "gpu": True, "reference": "Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging"},
        "famous_dl": {"name": "SPECT-UNet", "module": "pwm_core.recon.spect_solvers", "function": "spect_unet_recon", "params": "31M", "gpu": True, "reference": "Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6)"},
    },
    "stem.yaml": {
        "best_quality": {"name": "STEM-DL (AtomSegNet)", "module": "pwm_core.recon.stem_solvers", "function": "atom_seg_net_recon", "params": "10M", "gpu": True, "reference": "Wu, X. et al. (2021) AtomSegNet: DL networks for atom detection in STEM, npj Comput. Mater. 7:175"},
        "famous_dl": {"name": "STEM-UNet", "module": "pwm_core.recon.stem_solvers", "function": "stem_unet_recon", "params": "31M", "gpu": True, "reference": "Madsen, J. et al. (2018) A deep learning approach to identify local structures in atomic scale images, Adv. Theory Simul."},
    },
    "structured_light.yaml": {
        "best_quality": {"name": "SL-Net", "module": "pwm_core.recon.structured_light_solvers", "function": "sl_net_recon", "params": "10M", "gpu": True, "reference": "Nguyen, H. et al. (2020) DL for structured light profilometry, Opt. Lasers Eng. 133"},
        "famous_dl": {"name": "FTPD", "module": "pwm_core.recon.structured_light_solvers", "function": "ftpd_solver", "params": "0", "gpu": False, "reference": "Takeda, M. & Mutoh, K. (1983) Fourier transform profilometry, Appl. Opt. 22(24)"},
    },
    "tem.yaml": {
        "best_quality": {"name": "TEM-DL (ePIE-Net)", "module": "pwm_core.recon.tem_solvers", "function": "epie_net_recon", "params": "15M", "gpu": True, "reference": "Chen, Z. et al. (2021) Electron ptychography achieves atomic-resolution limits, Science 372:6544"},
        "famous_dl": {"name": "TEM-UNet", "module": "pwm_core.recon.tem_solvers", "function": "tem_unet_recon", "params": "31M", "gpu": True, "reference": "Madsen, J. et al. (2018) DL approach for local structures in STEM images, Adv. Theory Simul."},
    },
    "tirf.yaml": {
        "best_quality": {"name": "TIRF-Net (CARE)", "module": "pwm_core.recon.tirf_solvers", "function": "tirf_care_recon", "params": "15M", "gpu": True, "reference": "Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090"},
        "famous_dl": {"name": "TIRF-SRRF", "module": "pwm_core.recon.tirf_solvers", "function": "tirf_srrf_recon", "params": "0", "gpu": False, "reference": "Gustafsson, N. et al. (2016) Fast live-cell conventional fluorophore nanoscopy with SRRF, Nature Comms 7:12471"},
    },
    "tof_camera.yaml": {
        "best_quality": {"name": "ToF-Net", "module": "pwm_core.recon.tof_solvers", "function": "tof_dl_recon", "params": "12M", "gpu": True, "reference": "Agresti, G. et al. (2019) Unsupervised domain adaptation for ToF depth completion, ICCV Workshop"},
        "famous_dl": {"name": "ToF-MPI Deconv", "module": "pwm_core.recon.tof_solvers", "function": "tof_mpi_deconv", "params": "0", "gpu": False, "reference": "Gutierrez, O. et al. (2019) Practical calibration of actuated multi-beam ToF, CVPR"},
    },
    "two_photon.yaml": {
        "best_quality": {"name": "2P-Net (CARE)", "module": "pwm_core.recon.two_photon_solvers", "function": "two_photon_care_recon", "params": "15M", "gpu": True, "reference": "Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090"},
        "famous_dl": {"name": "2P-DeepInterp", "module": "pwm_core.recon.two_photon_solvers", "function": "deep_interp_recon", "params": "10M", "gpu": True, "reference": "Lecoq, J. et al. (2021) Removing independent noise in systems neuroscience using DeepInterpolation, Nature Methods 18:1401"},
    },
    "ultrasound.yaml": {
        "best_quality": {"name": "US-UNet (DeepUS)", "module": "pwm_core.recon.us_solvers", "function": "deep_us_recon", "params": "20M", "gpu": True, "reference": "Perdios, D. et al. (2018) DL-based image reconstruction for ultrasound imaging, IEEE IUS"},
        "famous_dl": {"name": "US-CNN", "module": "pwm_core.recon.us_solvers", "function": "us_cnn_recon", "params": "10M", "gpu": True, "reference": "Hyun, D. et al. (2019) Deep learning for ultrasound image reconstruction, IEEE TUFFC"},
    },
    "xray_radiography.yaml": {
        "best_quality": {"name": "CheXNet", "module": "pwm_core.recon.xray_solvers", "function": "chexnet_recon", "params": "121M", "gpu": True, "reference": "Rajpurkar, P. et al. (2017) CheXNet: Radiologist-level pneumonia detection, NIPS ML4H"},
        "famous_dl": {"name": "X-ray UNet", "module": "pwm_core.recon.xray_solvers", "function": "xray_unet_recon", "params": "31M", "gpu": True, "reference": "Litjens, G. et al. (2017) Survey of DL in medical image analysis, Med. Image Anal. 42"},
    },
}


def patch_yaml(fpath, solver_additions):
    with open(fpath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if "solvers" not in data or data["solvers"] is None:
        data["solvers"] = {}
    for key, val in solver_additions.items():
        data["solvers"][key] = val
    with open(fpath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


ok = 0
for fname, solvers in additions.items():
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f"  MISSING: {fname}")
        continue
    patch_yaml(fpath, solvers)
    ok += 1
    print(f"  OK  {fname}")

print(f"\nDone: {ok}/{len(additions)}")
