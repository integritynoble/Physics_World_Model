# Benchmark Dataset & Algorithm State Tracker

Last updated: 2026-03-11T18:00Z

## Status Legend

| State | Description |
|-------|-------------|
| **1) Dataset** | Creating public (>=10), dev (20), hidden (20) + spec.json + true_spec.json + per-sample images |
| **2) Benchmark** | Update https://pwm.platformai.org/benchmark -- gallery, data preview, baseline CPU recon |
| **3) Algorithms** | All algorithms tested (1 public sample each): classical CPU + GPU via Modal/server |
| **4) SpecLab** | All algorithm tests integrated into https://pwm.platformai.org/speclab |

Values: `done` / `in-progress` / `pending`

---

## Priority 1 — Core Medical & Imaging

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| ct | done | done | 0/24 tested | pending | 24 algos: FBP, SART, OSEM, ART-TV... +20 more |
| mri | done | done | 0/35 tested | pending | 35 algos: Zero-Filled IFFT, SENSE, GRAPPA, L1-Wavelet... +31 more |
| pet | done | done | 0/10 tested | pending | 10 algos: FBP-PET, OSEM, ML-EM, MAPEM-RDP... +6 more |
| ultrasound | done | done | 0/14 tested | pending | 14 algos: DAS, DAS-CF, PW-DAS, PnP-ADMM... +10 more |
| oct | done | done | 0/13 tested | pending | 13 algos: FFT-OCT, Speckle-Lee, TV-Denoising, BM4D... +9 more |
| mammography | done | done | 0/13 tested | pending | 13 algos: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN... +9 more |
| cbct | done | done | 0/9 tested | pending | 9 algos: FDK, TV-ADMM, FBPConvNet, Metal-AR-Net... +5 more |
| spect | done | done | 0/10 tested | pending | 10 algos: FBP-PET, OSEM, ML-EM, MAPEM-RDP... +6 more |
| fundus | done | done | 0/4 tested | pending | 4 algos: Richardson-Lucy, PnP-BM3D, cofe-Net, Swin-Fundus |
| endoscopy | done | done | 0/9 tested | pending | 9 algos: Histogram-Eq, CLAHE-Endo, BM3D-Endo, DnCNN-Endo... +5 more |
| fmri | done | done | 0/35 tested | pending | 35 algos: Zero-Filled IFFT, SENSE, GRAPPA, L1-Wavelet... +31 more |
| diffusion_mri | done | done | 0/9 tested | pending | 9 algos: DTI-FIT, SHORE, CHARMED, DnCNN-DTI... +5 more |

## Priority 2 — Microscopy & Optical

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| palm_storm | done | done | 0/4 tested | pending | 4 algos: ThunderSTORM, FALCON, Deep-STORM, DECODE |
| sted | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| sim | done | done | 0/4 tested | pending | 4 algos: Wiener-SIM, PnP-SIM, DL-SIM, SIMformer |
| confocal_3d | done | done | 0/9 tested | pending | 9 algos: Richardson-Lucy, Wiener-3D, IRCNN-Confocal, CARE... +5 more |
| lightsheet | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| two_photon | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| cryo_em | done | done | 0/9 tested | pending | 9 algos: CTFFIND4, RELION-3D, cryoSPARC, IsoNet... +5 more |
| sem | done | done | 0/4 tested | pending | 4 algos: Wiener Filter, BM3D, Noise2Void, SwinIR |
| tem | done | done | 0/4 tested | pending | 4 algos: Wiener Filter, BM3D, Noise2Void, SwinIR |
| widefield | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| photoacoustic | done | done | 0/4 tested | pending | 4 algos: Universal Back-Proj, PnP-ADMM, Deep-PAI, PAT-Former |

## Priority 3 — Computational & Advanced

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| holography | done | done | 0/14 tested | pending | 14 algos: Gerchberg-Saxton, GS/HIO, Error Reduction, prDeep... +10 more |
| ptychography | done | done | 0/4 tested | pending | 4 algos: ePIE, sDR, PtychoNN, AutoPhaseNN |
| lensless | done | done | 0/4 tested | pending | 4 algos: Wiener-ADMM, PnP-ADMM, FlatNet, Uformer |
| gaussian_splatting | done | done | 0/11 tested | pending | 11 algos: COLMAP+MVS, Photogrammetry, NeRF, Mip-NeRF 360... +7 more |
| phase_retrieval | done | done | 0/14 tested | pending | 14 algos: Gerchberg-Saxton, GS/HIO, Error Reduction, prDeep... +10 more |
| fpm | done | done | 0/4 tested | pending | 4 algos: Alternating Projections, Gradient Descent FPM, Fourier PtychoNet, PtychoDV |
| odt | done | done | 0/4 tested | pending | 4 algos: Wolf FBP, Born-ADMM, ODT-Net, Rytov-Former |
| ghost_imaging | done | done | 0/10 tested | pending | 10 algos: G(2)-Corr, Photon Counting, CS-TVAL3, Bayesian CS... +6 more |
| nerf | pending | pending | 0/11 tested | pending | 11 algos: COLMAP+MVS, Photogrammetry, NeRF, Mip-NeRF 360... +7 more |

## Priority 4 — Spectroscopy & Remote Sensing

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| raman_imaging | done | done | 0/11 tested | pending | 11 algos: SG-ALS, Baseline Correction, SVD, PnP-DnCNN... +7 more |
| ftir_imaging | done | done | 0/11 tested | pending | 11 algos: SG-ALS, Baseline Correction, SVD, PnP-DnCNN... +7 more |
| sar | done | done | 0/13 tested | pending | 13 algos: Matched Filter, Range-Doppler, Chirp Scaling, SAR-BM3D... +9 more |
| lidar | done | done | 0/4 tested | pending | 4 algos: Bilateral Filter, PnP-ADMM, RandLA-Net, Point Transformer |
| hyperspectral_remote | done | done | 0/4 tested | pending | 4 algos: CNMF, PnP-LTTR, DBIN, MST++ |
| insar | done | done | 0/4 tested | pending | 4 algos: Goldstein-MCF, InSAR-BM3D, PhaseNet, InSAR-Former |
| multispectral_sat | pending | pending | 0/13 tested | pending | 13 algos: Tikhonov, LSQR, ART, PnP-RED... +9 more |

## Priority 5 — Nuclear & Particle / Multimodality

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| pet_ct | done | done | 0/13 tested | pending | 13 algos: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN... +9 more |
| pet_mr | done | done | 0/10 tested | pending | 10 algos: FBP-PET, OSEM, ML-EM, MAPEM-RDP... +6 more |
| spect_ct | done | done | 0/4 tested | pending | 4 algos: OSEM, AC-OSEM, MAP-OSEM, DL-SPECT |
| spectral_ct | done | done | 0/13 tested | pending | 13 algos: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN... +9 more |
| industrial_ct | done | done | 0/4 tested | pending | 4 algos: FDK, PnP-ADMM, FBPConvNet, Learned Primal-Dual |

## Priority 6 — Remaining Modalities

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| acoustic_emission | done | done | 0/9 tested | pending | 9 algos: Time-Reversal Imaging, TDOA-WLS, Sparse TR (L1), PnP-ADMM... +5 more |
| acoustic_microscopy | done | done | 0/8 tested | pending | 8 algos: SAFT, Wiener Deconv, PnP-ADMM, SAM-Net... +4 more |
| active_thermography | done | done | 0/8 tested | pending | 8 algos: TSR, PCT, PnP-ADMM, ThermoNet... +4 more |
| adaptive_optics | done | done | 0/8 tested | pending | 8 algos: Zernike LS, Fried Estimator, PnP-ADMM (WF), WFNet... +4 more |
| afm | done | done | 0/7 tested | pending | 7 algos: Plane Fit, Wiener Deconv, PnP-ADMM, DeepAFM... +3 more |
| angiography | done | done | 0/9 tested | pending | 9 algos: FDK, TV-CS, PnP-ADMM, FBPConvNet... +5 more |
| asl_mri | done | done | 0/9 tested | pending | 9 algos: Zero-Filled IFFT, L1-Wavelet (ESPIRiT), PnP-DnCNN, U-Net (ASL)... +5 more |
| atom_probe | done | done | 0/9 tested | pending | 9 algos: Bas-Protocol, Tikhonov-Trajectory, PnP-BM3D (APT), ResNet-ArtefactCorr... +5 more |
| bioluminescence_tomo | done | done | 0/9 tested | pending | 9 algos: Tikhonov-BLT, Tikhonov-PR, PnP-ADMM (BLT), BLT-CNN... +5 more |
| brachytherapy_img | done | done | 0/9 tested | pending | 9 algos: FDK, TV-ADMM, FBPConvNet, RED-CNN... +5 more |
| brillouin | done | done | 0/9 tested | pending | 9 algos: Lorentzian-Fit, SG-Baseline, CNN-Spectra, DnCNN-Brillouin... +5 more |
| cacti | done | done | 0/9 tested | pending | 9 algos: GAP-TV, DeSCI, PnP-DnCNN, DGSMP... +5 more |
| cars | done | done | 0/9 tested | pending | 9 algos: KK-Retrieval, MEM-CARS, CNN-NRB, U-Net-CARS... +5 more |
| cassi | pending | pending | 0/13 tested | pending | 13 algos: Tikhonov, LSQR, ART, PnP-RED... +9 more |
| cathodoluminescence | done | done | 0/9 tested | pending | 9 algos: Wiener-CL, Richardson-Lucy, DnCNN-CL, U-Net-CL... +5 more |
| cest_mri | done | done | 0/9 tested | pending | 9 algos: MTR-asym, Lorentzian-Fit, WASSR, DnCNN-CEST... +5 more |
| ceus | done | done | 0/9 tested | pending | 9 algos: Pulse-Inversion, AM-CEUS, CNN-Bubble, ULM-Net... +5 more |
| clem | done | done | 0/9 tested | pending | 9 algos: Cross-Correlation, Landmark-Reg, CNN-Reg, VoxelMorph... +5 more |
| coded_exposure | done | done | 0/9 tested | pending | 9 algos: Wiener-Deconv, TV-Deconv, BM3D-Deblur, DnCNN-Deblur... +5 more |
| confocal_endomicroscopy | done | done | 0/9 tested | pending | 9 algos: NLM-Speckle, BM3D-CLE, DnCNN-CLE, U-Net-CLE... +5 more |
| confocal_livecell | done | done | 0/9 tested | pending | 9 algos: VST-Denoise, NLM-Fluorescence, CARE, Noise2Void... +5 more |
| coronagraphy | done | done | 0/9 tested | pending | 9 algos: ADI, KLIP, LOCI, PCA-ADI... +5 more |
| cryo_et | done | done | 0/9 tested | pending | 9 algos: WBP, SART-ET, IMOD, IsoNet... +5 more |
| ct_fluorescence | done | done | 0/9 tested | pending | 9 algos: FBP-XRF, MLEM-XRF, TV-XRFCT, DnCNN-XRF... +5 more |
| cup | done | done | 0/9 tested | pending | 9 algos: TV-CUP, TwIST-CUP, GAP-TV, DeSCI-CUP... +5 more |
| dark_field | done | done | 0/9 tested | pending | 9 algos: Richardson-Lucy, Wiener-DF, TV-DF, BM3D-DF... +5 more |
| desi | done | done | 0/9 tested | pending | 9 algos: MSI-Hotelling, MSI-PCA, MSI-NMF, MSI-TV... +5 more |
| dexa | done | done | 0/9 tested | pending | 9 algos: FBP-DEXA, TV-DEXA, BML-Sep, DXA-CNN... +5 more |
| dic | done | done | 0/9 tested | pending | 9 algos: DIC-Deconv, TV-DIC, Phase-DLSIM, DIC-CNN... +5 more |
| digital_breast_tomo | done | done | 0/9 tested | pending | 9 algos: FBP-DBT, TV-DBT, SART-DBT, DnCNN-DBT... +5 more |
| dna_paint | done | done | 0/9 tested | pending | 9 algos: STORM-2D, PALM, DAOSTORM, DeepSTORM... +5 more |
| doppler_ultrasound | done | done | 0/9 tested | pending | 9 algos: CF-Doppler, VENC-Flow, MV-Doppler, DnCNN-Doppler... +5 more |
| dot | done | done | 0/9 tested | pending | 9 algos: Born-Approx, TV-DOT, FEM-DOT, DnCNN-DOT... +5 more |
| ebsd | done | done | 0/9 tested | pending | 9 algos: Hough-EBSD, DI-EBSD, TV-EBSD, DnCNN-EBSD... +5 more |
| eddy_current | done | done | 0/9 tested | pending | 9 algos: EC-Deconv, TV-EC, MUSIC-EC, DnCNN-EC... +5 more |
| edx_mapping | done | done | 0/9 tested | pending | 9 algos: MLS-EDX, TV-EDX, NMF-EDX, DnCNN-EDX... +5 more |
| eels | done | done | 0/9 tested | pending | 9 algos: PowerLaw-EELS, MLS-EELS, ICA-EELS, DnCNN-EELS... +5 more |
| eht_imaging | done | done | 0/9 tested | pending | 9 algos: CLEAN-VLBI, MEM-VLBI, RESOLVE, eht-imaging... +5 more |
| elastography | done | done | 0/9 tested | pending | 9 algos: LFE-Elasto, DI-Elasto, AIDE, DnCNN-Elasto... +5 more |
| electron_diffraction | done | done | 0/9 tested | pending | 9 algos: Direct-Methods, PEDT, MicroED, DnCNN-ED... +5 more |
| electron_holography | done | done | 0/9 tested | pending | 9 algos: FFT-Holo, WDD-Holo, TV-Phase, DnCNN-Holo... +5 more |
| electron_tomography | done | done | 0/9 tested | pending | 9 algos: WBP-ET, SIRT-ET, CS-ET, DnCNN-ET... +5 more |
| entangled_photon | done | done | 0/9 tested | pending | 9 algos: Coincidence-Count, CS-Ghost, SVD-Ghost, DnCNN-Ghost... +5 more |
| event_camera | done | done | 0/9 tested | pending | 9 algos: Event-Integration, Complementary, E2VID, FireNet... +5 more |
| expansion | done | done | 0/9 tested | pending | 9 algos: Deconv-Exp, RL-ExM, TV-ExM, DnCNN-ExM... +5 more |
| fib_sem | done | done | 0/9 tested | pending | 9 algos: BM3D-FIB, NLM-FIB, TV-FIB, DnCNN-FIB... +5 more |
| flash_lidar | done | done | 0/9 tested | pending | 9 algos: MLE-SPAD, Coates-Hist, NL-Means-LiDAR, DnCNN-LiDAR... +5 more |
| flim | done | done | 0/9 tested | pending | 9 algos: Phasor-FLIM, MLE-FLIM, RLD-FLIM, DnCNN-FLIM... +5 more |
| fluoroscopy | done | done | 0/9 tested | pending | 9 algos: BM3D-Fluoro, NLM-Fluoro, TV-Fluoro, DnCNN-Fluoro... +5 more |
| fwi | done | done | 0/4 tested | pending | 4 algos: L-BFGS FWI, TV-Reg FWI, InversionNet, VelocityGAN |
| gpr | done | done | 0/4 tested | pending | 4 algos: Kirchhoff Migration, RTM, GPR-RCNN, HyperDet |
| gravitational_wave | done | done | 0/4 tested | pending | 4 algos: Matched Filter, BayesWave, GW-CNN, WaveFormer |
| hdr_imaging | done | done | 0/14 tested | pending | 14 algos: Wiener-Deconv, Laplacian Pyramid, Lucy-Richardson, PnP-FFDNet... +10 more |
| impedance_tomo | done | done | 0/4 tested | pending | 4 algos: Gauss-Newton, TV-ADMM, D-bar CNN, EIT-Former |
| integral | done | done | 0/4 tested | pending | 4 algos: Shift-and-Add, PnP-LF, LFAttNet, DistgSSR |
| ism | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| ivus | done | done | 0/14 tested | pending | 14 algos: DAS, DAS-CF, PW-DAS, PnP-ADMM... +10 more |
| lattice_lightsheet | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| libs | done | done | 0/11 tested | pending | 11 algos: SG-ALS, Baseline Correction, SVD, PnP-DnCNN... +7 more |
| light_field | done | done | 0/4 tested | pending | 4 algos: Shift-and-Sum, PnP-LF, LFNet, DistgSSR |
| lucky_imaging | done | done | 0/4 tested | pending | 4 algos: Shift-and-Add, Drizzle, BDI, SpeckleNet |
| machine_vision | done | done | 0/4 tested | pending | 4 algos: Template Match, PnP-ADMM, PatchCore, UniAD |
| magnetic_particle | done | done | 0/11 tested | pending | 11 algos: Tikhonov, Wiener Filter, Matched Filter, PnP-RED... +7 more |
| maldi_msi | done | done | 0/11 tested | pending | 11 algos: Deconv, Calibration-Lookup, Peak Fitting, PnP-BM3D... +7 more |
| matrix | done | done | 0/14 tested | pending | 14 algos: GAP-TV, FISTA-TV, TVAL3, PnP-FFDNet... +10 more |
| mfm | done | done | 0/10 tested | pending | 10 algos: BTR, MLE Reconstruction, Reg-Deconv, TV-Deconvolution... +6 more |
| minflux | done | done | 0/4 tested | pending | 4 algos: MLE Localization, SPARCOM, DECODE, ANNA-PALM |
| mr_elastography | done | done | 0/35 tested | pending | 35 algos: Zero-Filled IFFT, SENSE, GRAPPA, L1-Wavelet... +31 more |
| mr_fingerprinting | done | done | 0/4 tested | pending | 4 algos: SVD-MRF, MANTIS, MRF-Net, MRF-Former |
| mra | done | done | 0/35 tested | pending | 35 algos: Zero-Filled IFFT, SENSE, GRAPPA, L1-Wavelet... +31 more |
| mrs | done | done | 0/35 tested | pending | 35 algos: Zero-Filled IFFT, SENSE, GRAPPA, L1-Wavelet... +31 more |
| muon_tomo | done | done | 0/10 tested | pending | 10 algos: FBP-PET, OSEM, ML-EM, MAPEM-RDP... +6 more |
| neutron_diffraction | done | done | 0/4 tested | pending | 4 algos: Rietveld-GSAS, Le Bail Fit, NeutronNet, DiffFormer |
| neutron_tomo | done | done | 0/10 tested | pending | 10 algos: FBP-PET, OSEM, ML-EM, MAPEM-RDP... +6 more |
| nirs_brain | done | done | 0/4 tested | pending | 4 algos: MBLL, Tikhonov-DOT, PnP-DOT, DL-DOT |
| nsom | done | done | 0/10 tested | pending | 10 algos: BTR, MLE Reconstruction, Reg-Deconv, TV-Deconvolution... +6 more |
| ocean_acoustic_tomo | done | done | 0/11 tested | pending | 11 algos: Tikhonov, Wiener Filter, Matched Filter, PnP-RED... +7 more |
| ocean_color | done | done | 0/4 tested | pending | 4 algos: Gordon AC, MUMM, OC-Net, AquaFormer |
| octa | done | done | 0/13 tested | pending | 13 algos: FFT-OCT, Speckle-Lee, TV-Denoising, BM4D... +9 more |
| panorama | done | done | 0/4 tested | pending | 4 algos: SIFT-RANSAC, APAP, UDIS, PanoFormer |
| particle_calorimetry | done | done | 0/4 tested | pending | 4 algos: PandoraPFA, GARFIELD++, GravNet, CaloDiffusion |
| passive_microwave | done | done | 0/4 tested | pending | 4 algos: Backus-Gilbert, Tikhonov-SMOS, RadioNet, MWR-Former |
| phase_contrast | done | done | 0/4 tested | pending | 4 algos: TIE Solver, DPC-ADMM, QPI-Net, PhaseFormer |
| photometric_stereo | done | done | 0/4 tested | pending | 4 algos: LS Normal Est., Robust PCA, CNN-PS, PS-Transformer |
| polarization | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| polsar | done | done | 0/13 tested | pending | 13 algos: Matched Filter, Range-Doppler, Chirp Scaling, SAR-BM3D... +9 more |
| portal_imaging | done | done | 0/13 tested | pending | 13 algos: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN... +9 more |
| proton_radiography | done | done | 0/4 tested | pending | 4 algos: FBP-MLP, DROP-TVS, ProtonNet, pCT-Former |
| proton_therapy_img | done | done | 0/13 tested | pending | 13 algos: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN... +9 more |
| pump_probe | done | done | 0/4 tested | pending | 4 algos: SVD-GlobFit, MCR-ALS, TAS-Net, DynFormer |
| quantum_illumination | done | done | 0/4 tested | pending | 4 algos: OPA Receiver, FF-SFG, QI-Net, QuantumFormer |
| radio_astronomy | done | done | 0/4 tested | pending | 4 algos: CLEAN, AIRI, R2D2, PRIMO |
| radio_interferometry | done | done | 0/4 tested | pending | 4 algos: CLEAN, AIRI, R2D2, PRIMO |
| saxs | done | done | 0/4 tested | pending | 4 algos: PyFAI-Integrate, McSAS, ScatterNet, ScatterFormer |
| seismic_tomo | done | done | 0/11 tested | pending | 11 algos: Tikhonov, Wiener Filter, Matched Filter, PnP-RED... +7 more |
| shearography | done | done | 0/4 tested | pending | 4 algos: Goldstein MCF, PnP-Phase, ShearNet, PhaseFormer |
| shg | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| sims | done | done | 0/11 tested | pending | 11 algos: SG-ALS, Baseline Correction, SVD, PnP-DnCNN... +7 more |
| solar_imaging | done | done | 0/4 tested | pending | 4 algos: Richardson-Lucy, Pixon, DeepEM, SolarFormer |
| sonar | done | done | 0/4 tested | pending | 4 algos: DAS, MVDR/Capon, SonarNet, AcousticFormer |
| spc | pending | pending | 0/13 tested | pending | 13 algos: Tikhonov, LSQR, ART, PnP-RED... +9 more |
| spinning_disk | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| srs | done | done | 0/11 tested | pending | 11 algos: SG-ALS, Baseline Correction, SVD, PnP-DnCNN... +7 more |
| stem | done | done | 0/4 tested | pending | 4 algos: Wiener Filter, BM3D, Noise2Void, SwinIR |
| stm | done | done | 0/10 tested | pending | 10 algos: BTR, MLE Reconstruction, Reg-Deconv, TV-Deconvolution... +6 more |
| streak_camera | done | done | 0/11 tested | pending | 11 algos: TwIST, Temporal Filtering, PnP-FFDNet, PnP-ADMM... +7 more |
| structured_light | done | done | 0/4 tested | pending | 4 algos: Phase Shifting, Gray Code, FPP-Net, PhaseFormer |
| swi | done | done | 0/35 tested | pending | 35 algos: Zero-Filled IFFT, SENSE, GRAPPA, L1-Wavelet... +31 more |
| talbot_lau | done | done | 0/4 tested | pending | 4 algos: Phase Stepping, PCA Retrieval, DPC-Net, GratingFormer |
| terahertz | done | done | 0/4 tested | pending | 4 algos: Wiener-THz, PnP-SPIRAL, THz-Net, THz-Former |
| three_photon | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| tirf | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| tof_camera | done | done | 0/4 tested | pending | 4 algos: Phase Unwrap, PnP-ToF, DeepToF, MPI-Former |
| ultrasonic_phased_array | done | done | 0/4 tested | pending | 4 algos: TFM, SAFT, UTPA-Net, FMC-Former |
| us_mri | done | done | 0/4 tested | pending | 4 algos: Demons, B-spline FFD, VoxelMorph, TransMorph |
| waxs | done | done | 0/4 tested | pending | 4 algos: PyFAI-Integrate, Rietveld-WAXS, WAXS-Net, CrystalFormer |
| weather_radar | done | done | 0/4 tested | pending | 4 algos: Pulse-Pair Doppler, CLEAN-AP, RainNet, Earthformer |
| widefield_lowdose | done | done | 0/13 tested | pending | 13 algos: Richardson-Lucy, Wiener Filter, TV-Deconvolution, PnP-FISTA... +9 more |
| xfel_sfx | done | done | 0/4 tested | pending | 4 algos: CrystFEL, EMC, CNN Hit-Finder, CrysFormer |
| xray_crystallography | done | done | 0/4 tested | pending | 4 algos: Molecular Replacement, SHELXD, DL-Phase, CrystFormer |
| xray_ndt | done | done | 0/4 tested | pending | 4 algos: FBP, PnP-ADMM, FBPConvNet, DR-GAN |
| xray_radiography | done | done | 0/13 tested | pending | 13 algos: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN... +9 more |
| xrf_imaging | done | done | 0/4 tested | pending | 4 algos: FP-Quantify, PnP-BM3D, XRF-UNet, SpectraFormer |
| xrf_tomo | done | done | 0/11 tested | pending | 11 algos: Deconv, Calibration-Lookup, Peak Fitting, PnP-BM3D... +7 more |

