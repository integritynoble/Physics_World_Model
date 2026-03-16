# SpecLab Reconstruction State

Tracks verification status of all reconstruction algorithms in SpecLab
(`https://pwm.platformai.org/speclab`).

**Status:**
- `done` — PWM CPU reconstruction verified, actual PSNR within 2 dB of reference
- *(blank)* — awaiting verification (not yet tested, or PSNR below reference threshold)

Last updated: 2026-03-16 | Total modalities: 168 | Verified: 492

---

## 3D Gaussian Splatting (3DGS) (`gaussian_splatting`) — neural_rendering

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| COLMAP+MVS | Classical | 26.4 dB | 0.73 |  |
| Photogrammetry | Classical | 26.54 dB | 0.847 |  |
| Mip-NeRF 360 | Deep Learning | 29.4 dB | 0.844 |  |
| Mesh-GS | Deep Learning | 30.07 dB | 0.918 |  |
| Instant-NGP | Deep Learning | 31.1 dB | 0.905 |  |
| NeRF | Deep Learning | 33.15 dB | 0.954 |  |
| 3D-GS | Deep Learning | 33.3 dB | 0.94 |  |
| 3D-GS++ | Deep Learning | 34.52 dB | 0.952 |  |
| 2DGS | Deep Learning | 34.67 dB | 0.966 |  |
| GaussianShader | Vision Transformer | 35.18 dB | 0.96 |  |
| NeRFactor2 | Deep Learning | 35.85 dB | 0.966 |  |

## 4D-STEM Electron Diffraction (`electron_diffraction`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Direct-Methods | Classical | 21.2 dB | 0.694 |  |
| PEDT | Classical | 23.9 dB | 0.738 |  |
| MicroED | Classical | 26.7 dB | 0.781 |  |
| DnCNN-ED | Deep Learning | 29.5 dB | 0.833 |  |
| PhaseGAN-ED | Generative | 32.3 dB | 0.873 |  |
| TransED | Transformer | 34.8 dB | 0.912 |  |
| SwinED | Transformer | 36.4 dB | 0.93 |  |
| PhysED | Physics-Informed | 37.7 dB | 0.941 |  |
| DiffED | Diffusion Model | 39.1 dB | 0.953 |  |

## Acoustic Emission Testing (AE) (`acoustic_emission`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Time-Reversal Imaging | Classical | 20.5 dB | 0.58 | done |
| TDOA-WLS | Classical | 22.0 dB | 0.63 | done |
| Sparse TR (L1) | Compressed Sensing | 25.5 dB | 0.73 | done |
| PnP-ADMM | PnP | 27.5 dB | 0.8 | done |
| AE-CNN | Deep Learning | 30.0 dB | 0.87 | done |
| Domain-Adapted ResNet | Deep Learning | 32.0 dB | 0.905 | done |
| PINN-AE | Physics-Informed | 33.5 dB | 0.925 | done |
| SwinIR-AE | Transformer | 34.8 dB | 0.94 | done |
| DiffusionAE | Diffusion | 35.5 dB | 0.95 | done |

## Active Thermography (IR) (`active_thermography`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TSR | Classical | 22.0 dB | 0.62 |  |
| PCT | Classical | 24.0 dB | 0.69 |  |
| PnP-ADMM | PnP | 27.0 dB | 0.79 |  |
| ThermoNet | Deep Learning | 30.0 dB | 0.87 | done |
| U-Net Thermo | Deep Learning | 32.0 dB | 0.905 | done |
| PINN-Thermo | Physics-Informed | 33.0 dB | 0.92 | done |
| ThermoFormer | Transformer | 34.5 dB | 0.938 | done |
| DiffusionThermo | Diffusion | 35.5 dB | 0.95 | done |

## Adaptive Optics (AO) Imaging (`adaptive_optics`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zernike LS | Classical | 22.0 dB | 0.64 | done |
| Fried Estimator | Classical | 24.0 dB | 0.7 | done |
| PnP-ADMM (WF) | PnP | 27.0 dB | 0.8 | done |
| WFNet | Deep Learning | 30.0 dB | 0.87 | done |
| LIFT-Net | Deep Learning | 31.5 dB | 0.895 | done |
| AO-Transformer | Transformer | 33.0 dB | 0.92 | done |
| AO-ViT | Transformer | 34.0 dB | 0.935 | done |
| DiffusionAO | Diffusion | 35.0 dB | 0.948 | done |

## Arterial Spin Labeling (ASL) MRI (`asl_mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 24.5 dB | 0.58 |  |
| L1-Wavelet (ESPIRiT) | Compressed Sensing | 28.3 dB | 0.82 |  |
| PnP-DnCNN | PnP | 29.8 dB | 0.843 |  |
| U-Net (ASL) | Deep Learning | 32.1 dB | 0.876 |  |
| Kinetic-CS | Physics-Informed | 33.2 dB | 0.891 |  |
| E2E-VarNet | Deep Unrolling | 34.6 dB | 0.908 |  |
| ReconFormer | Transformer | 35.4 dB | 0.922 |  |
| PromptMR | Deep Unrolling | 36.1 dB | 0.934 |  |
| Score-MRI (ASL) | Diffusion | 36.7 dB | 0.942 |  |

## Atom Probe Tomography (APT) (`atom_probe`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Bas-Protocol | Classical | 20.8 dB | 0.55 | done |
| Tikhonov-Trajectory | Classical | 23.4 dB | 0.66 | done |
| PnP-BM3D (APT) | PnP | 26.1 dB | 0.75 | done |
| ResNet-ArtefactCorr | Deep Learning | 28.7 dB | 0.818 | done |
| LISTA-APT | Deep Unrolling | 29.5 dB | 0.842 | done |
| TrajectoryPINN | Physics-Informed | 31.2 dB | 0.876 | done |
| APT-Former | Transformer | 33.6 dB | 0.912 | done |
| DiffusionAPT | Diffusion | 35.1 dB | 0.934 | done |
| EquivAPT | Vision Transformer | 36.3 dB | 0.948 | done |

## Atomic Force Microscopy (AFM) (`afm`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Plane Fit | Classical | 20.0 dB | 0.56 | done |
| Wiener Deconv | Classical | 23.0 dB | 0.65 | done |
| PnP-ADMM | PnP | 26.5 dB | 0.77 | done |
| DeepAFM | Deep Learning | 30.0 dB | 0.87 | done |
| Self-Sup AFM | Self-Supervised | 31.5 dB | 0.895 | done |
| SPM-Former | Transformer | 33.0 dB | 0.92 | done |
| DiffusionAFM | Diffusion | 34.5 dB | 0.94 | done |

## Bioluminescence Tomography (BLT) (`bioluminescence_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov-BLT | Classical | 19.5 dB | 0.54 | done |
| Tikhonov-PR | Classical | 22.8 dB | 0.64 | done |
| PnP-ADMM (BLT) | PnP | 25.6 dB | 0.73 | done |
| BLT-CNN | Deep Learning | 29.1 dB | 0.838 | done |
| LISTA-BLT | Deep Unrolling | 30.4 dB | 0.864 | done |
| DiffusionPINN-BLT | Physics-Informed | 32.9 dB | 0.902 | done |
| BLT-Former | Transformer | 34.8 dB | 0.929 | done |
| ScoreBLT | Diffusion | 36.5 dB | 0.952 | done |
| PhysDiff-BLT | Diffusion | 38.1 dB | 0.967 |  |

## Brachytherapy Imaging (`brachytherapy_img`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FDK | Classical | 28.5 dB | 0.812 |  |
| TV-ADMM | Variational | 31.8 dB | 0.861 |  |
| FBPConvNet | Deep Learning | 34.2 dB | 0.895 |  |
| RED-CNN | Deep Learning | 35.1 dB | 0.912 |  |
| Metal-AR-Net | Deep Learning | 36.4 dB | 0.928 |  |
| Learned Primal-Dual | Deep Unrolling | 37.0 dB | 0.935 |  |
| DuDoTrans | Transformer | 38.2 dB | 0.948 |  |
| CTFormer | Transformer | 39.1 dB | 0.957 |  |
| DiffusionSeed | Diffusion | 40.3 dB | 0.968 |  |

## Brillouin Microscopy (`brillouin`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Lorentzian-Fit | Classical | 26.2 dB | 0.785 | done |
| SG-Baseline | Classical | 27.8 dB | 0.812 | done |
| CNN-Spectra | Deep Learning | 31.5 dB | 0.872 | done |
| DnCNN-Brillouin | Deep Learning | 33.2 dB | 0.901 | done |
| CDAE | Deep Learning | 34.8 dB | 0.918 | done |
| U-Net-Spectral | Deep Learning | 36.1 dB | 0.933 | done |
| PINN-Brillouin | Physics-Informed | 37.0 dB | 0.942 | done |
| SpectraFormer | Transformer | 38.4 dB | 0.954 |  |
| DiffusionSpectra | Diffusion | 39.5 dB | 0.963 |  |

## CEST MRI (`cest_mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MTR-asym | Classical | 24.8 dB | 0.761 |  |
| Lorentzian-Fit | Classical | 27.2 dB | 0.808 |  |
| WASSR | Classical | 28.5 dB | 0.831 |  |
| DnCNN-CEST | Deep Learning | 32.1 dB | 0.878 |  |
| U-Net-CEST | Deep Learning | 34.8 dB | 0.912 |  |
| PINN-CEST | Physics-Informed | 35.9 dB | 0.925 |  |
| CESTFormer | Transformer | 37.4 dB | 0.94 |  |
| PromptCEST | Transformer | 38.6 dB | 0.951 |  |
| DiffusionCEST | Diffusion | 39.7 dB | 0.961 |  |

## CT + Fluorescence (FLIT) (`ct_fluorescence`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-XRF | Classical | 22.8 dB | 0.701 | done |
| MLEM-XRF | Classical | 26.3 dB | 0.764 | done |
| TV-XRFCT | Variational | 29.7 dB | 0.831 | done |
| DnCNN-XRF | Deep Learning | 32.4 dB | 0.872 | done |
| U-Net-XRF | Deep Learning | 34.6 dB | 0.901 | done |
| PnP-XRF | PnP | 35.9 dB | 0.914 |  |
| SwinXRF | Transformer | 37.8 dB | 0.932 |  |
| PhysXRF-Net | Physics-Informed | 38.5 dB | 0.941 |  |
| DiffusionXRF | Diffusion | 40.1 dB | 0.955 |  |

## Cathodoluminescence (CL) Imaging (`cathodoluminescence`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-CL | Classical | 25.2 dB | 0.771 | done |
| Richardson-Lucy | Classical | 27.5 dB | 0.812 |  |
| DnCNN-CL | Deep Learning | 31.8 dB | 0.875 | done |
| U-Net-CL | Deep Learning | 34.2 dB | 0.908 | done |
| CARE-CL | Deep Learning | 35.5 dB | 0.921 | done |
| PINN-CL | Physics-Informed | 36.8 dB | 0.934 | done |
| SwinIR-CL | Transformer | 37.1 dB | 0.938 |  |
| Restormer-CL | Transformer | 38.4 dB | 0.95 |  |
| DiffusionEM | Diffusion | 39.8 dB | 0.962 |  |

## Coded Aperture Compressive Temporal Imaging (CACTI) (`cacti`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| GAP-TV | Variational | 26.8 dB | 0.795 |  |
| DeSCI | PnP | 28.8 dB | 0.832 |  |
| PnP-DnCNN | PnP | 30.5 dB | 0.868 |  |
| DGSMP | Deep Unrolling | 33.2 dB | 0.904 |  |
| GAP-CCoT | Transformer | 34.1 dB | 0.915 |  |
| STFormer | Transformer | 36.8 dB | 0.938 |  |
| EfficientSCI | Transformer | 37.5 dB | 0.945 |  |
| RDLUF-MixS2 | Deep Unrolling | 38.4 dB | 0.952 |  |
| DiffusionSCI | Diffusion | 39.8 dB | 0.963 |  |

## Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| GAP-TV | Classical | 26.83 dB | 0.754 | done |
| FISTA-TV | Classical | 28.42 dB | 0.821 | done |
| TVAL3 | Classical | 29.15 dB | 0.845 | done |
| PnP-FFDNet | PnP | 29.65 dB | 0.852 | done |
| EfficientSCI | Deep Learning | 34.21 dB | 0.949 |  |
| MST-L | Transformer | 35.4 dB | 0.96 |  |
| Restormer | Vision Transformer | 35.68 dB | 0.962 |  |
| CST | Transformer | 35.92 dB | 0.965 |  |
| HiSViT+ | Vision Transformer | 36.85 dB | 0.971 |  |
| CSTrans | Transformer | 37.12 dB | 0.973 |  |
| PromptSCI | Deep Learning | 37.35 dB | 0.975 |  |
| DiffusionHSI | Diffusion | 37.95 dB | 0.978 |  |
| ScoreSCI | Diffusion | 38.22 dB | 0.98 |  |
| FlowHSI | Generative | 38.58 dB | 0.982 |  |

## Coded Exposure / Flutter Shutter (`coded_exposure`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-Deconv | Classical | 26.5 dB | 0.791 |  |
| TV-Deconv | Variational | 29.2 dB | 0.831 |  |
| BM3D-Deblur | PnP | 31.8 dB | 0.871 |  |
| DnCNN-Deblur | Deep Learning | 33.5 dB | 0.899 | done |
| DeblurGAN | Generative | 34.8 dB | 0.914 | done |
| DMPHN | Deep Learning | 36.1 dB | 0.928 | done |
| MPRNet | Deep Learning | 37.4 dB | 0.941 |  |
| Restormer-Deblur | Transformer | 38.6 dB | 0.951 |  |
| DiffusionDeblur | Diffusion | 39.8 dB | 0.961 |  |

## Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| KK-Retrieval | Classical | 24.5 dB | 0.762 |  |
| MEM-CARS | Classical | 26.2 dB | 0.798 |  |
| CNN-NRB | Deep Learning | 30.8 dB | 0.865 | done |
| U-Net-CARS | Deep Learning | 33.5 dB | 0.902 | done |
| PINN-CARS | Physics-Informed | 34.8 dB | 0.918 | done |
| ResNet-CARS | Deep Learning | 36.2 dB | 0.933 | done |
| SpecFormer-CARS | Transformer | 37.8 dB | 0.947 |  |
| Diff-CARS | Diffusion | 39.1 dB | 0.958 |  |
| FMDiff-CARS | Diffusion | 40.2 dB | 0.966 |  |

## Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gerchberg-Saxton | Classical | 21.5 dB | 0.58 |  |
| Error Reduction | Classical | 22.85 dB | 0.615 |  |
| GS/HIO | Classical | 23.7 dB | 0.65 |  |
| deep-PR | Deep Learning | 27.2 dB | 0.81 |  |
| prDeep | Deep Unrolling | 27.45 dB | 0.82 |  |
| PhaseNet | Deep Learning | 31.2 dB | 0.91 |  |
| CyclePhase | Deep Learning | 32.5 dB | 0.938 |  |
| LRGS | Deep Learning | 32.8 dB | 0.935 |  |
| PhaseResNet | Deep Learning | 33.15 dB | 0.942 |  |
| PhaseFormer | Vision Transformer | 34.5 dB | 0.952 |  |
| AutoPhase++ | Vision Transformer | 34.92 dB | 0.958 |  |
| HolographyViT | Vision Transformer | 35.18 dB | 0.96 |  |
| DiffusionPhase | Diffusion | 35.48 dB | 0.964 |  |
| ScorePhase | Score-based | 35.82 dB | 0.968 |  |

## Compressed Ultrafast Photography (CUP) (`cup`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TV-CUP | Variational | 24.3 dB | 0.732 |  |
| TwIST-CUP | Variational | 26.8 dB | 0.774 |  |
| GAP-TV | Variational | 28.5 dB | 0.812 |  |
| DeSCI-CUP | PnP | 31.2 dB | 0.854 |  |
| E2E-CNN-CUP | Deep Learning | 33.7 dB | 0.886 | done |
| PnP-FastDVDnet | PnP | 35.4 dB | 0.911 |  |
| STFormer-CUP | Transformer | 37.9 dB | 0.933 |  |
| DAUHST-CUP | Transformer | 38.6 dB | 0.941 |  |
| DiffusionCUP | Diffusion | 40.2 dB | 0.956 |  |

## Cone-Beam Computed Tomography (CBCT) (`cbct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FDK | Classical | 27.8 dB | 0.801 | done |
| TV-ADMM | Variational | 31.2 dB | 0.851 |  |
| FBPConvNet | Deep Learning | 34.5 dB | 0.891 |  |
| Metal-AR-Net | Deep Learning | 35.8 dB | 0.912 |  |
| Learned Primal-Dual | Deep Unrolling | 36.4 dB | 0.921 |  |
| DuDoNet | Deep Learning | 37.1 dB | 0.932 |  |
| DuDoTrans | Transformer | 38.2 dB | 0.944 |  |
| CTFormer | Transformer | 39.0 dB | 0.953 |  |
| DiffusionCBCT | Diffusion | 40.1 dB | 0.964 |  |

## Confocal 3D Z-Stack (`confocal_3d`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 26.8 dB | 0.801 |  |
| Wiener-3D | Classical | 28.5 dB | 0.828 |  |
| IRCNN-Confocal | PnP | 32.1 dB | 0.878 |  |
| Noise2Void | Self-Supervised | 33.5 dB | 0.895 |  |
| CARE | Deep Learning | 34.8 dB | 0.91 |  |
| U-Net-3D | Deep Learning | 35.9 dB | 0.924 |  |
| SwinIR-3D | Transformer | 37.5 dB | 0.942 |  |
| Restormer-3D | Transformer | 38.6 dB | 0.951 |  |
| DiffusionMicro | Diffusion | 39.9 dB | 0.963 |  |

## Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| NLM-Speckle | Classical | 25.5 dB | 0.775 |  |
| BM3D-CLE | Classical | 27.8 dB | 0.815 |  |
| DnCNN-CLE | Deep Learning | 31.2 dB | 0.868 | done |
| U-Net-CLE | Deep Learning | 33.8 dB | 0.902 | done |
| CARE-CLE | Deep Learning | 35.2 dB | 0.92 | done |
| PINN-CLE | Physics-Informed | 36.1 dB | 0.93 | done |
| SwinIR-CLE | Transformer | 36.8 dB | 0.936 | done |
| Restormer-CLE | Transformer | 38.1 dB | 0.949 |  |
| DiffusionEndo | Diffusion | 39.4 dB | 0.96 |  |

## Confocal Live-Cell Microscopy (`confocal_livecell`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| VST-Denoise | Classical | 24.2 dB | 0.751 | done |
| NLM-Fluorescence | Classical | 26.8 dB | 0.795 | done |
| Noise2Self | Self-Supervised | 30.5 dB | 0.858 | done |
| Noise2Void | Self-Supervised | 31.8 dB | 0.871 | done |
| PN2V | Self-Supervised | 32.9 dB | 0.882 | done |
| CARE | Deep Learning | 33.5 dB | 0.891 | done |
| SwinIR-LiveCell | Transformer | 36.2 dB | 0.931 | done |
| Restormer-Micro | Transformer | 37.8 dB | 0.946 |  |
| DiffusionCell | Diffusion | 39.2 dB | 0.959 |  |

## Contrast-Enhanced Ultrasound (CEUS) (`ceus`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Pulse-Inversion | Classical | 24.1 dB | 0.751 | done |
| AM-CEUS | Classical | 25.8 dB | 0.781 | done |
| CNN-Bubble | Deep Learning | 30.2 dB | 0.858 | done |
| ULM-Net | Deep Learning | 33.5 dB | 0.9 | done |
| DeepULM | Deep Learning | 35.1 dB | 0.92 | done |
| PINN-CEUS | Physics-Informed | 36.4 dB | 0.934 | done |
| CEUSF-Transformer | Transformer | 37.8 dB | 0.946 |  |
| SUPER-ULM | Deep Unrolling | 38.5 dB | 0.953 |  |
| DiffusionCEUS | Diffusion | 39.6 dB | 0.962 |  |

## Correlative Light-Electron Microscopy (CLEM) (`clem`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Cross-Correlation | Classical | 23.5 dB | 0.741 | done |
| Landmark-Reg | Classical | 25.8 dB | 0.782 | done |
| CNN-Reg | Deep Learning | 30.2 dB | 0.855 | done |
| VoxelMorph | Deep Learning | 32.8 dB | 0.89 | done |
| CLEM-Net | Deep Learning | 34.5 dB | 0.912 | done |
| PINN-CLEM | Physics-Informed | 35.8 dB | 0.927 | done |
| TransMorph | Transformer | 36.2 dB | 0.931 | done |
| SwinCLEM | Transformer | 37.5 dB | 0.944 |  |
| DiffusionCLEM | Diffusion | 39.1 dB | 0.958 |  |

## Cryo-EM Single Particle Analysis (`cryo_em`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CTFFIND4 | Classical | 22.3 dB | 0.714 |  |
| RELION-3D | Classical | 25.8 dB | 0.782 |  |
| cryoSPARC | Classical | 28.1 dB | 0.823 |  |
| IsoNet | Deep Learning | 31.4 dB | 0.871 |  |
| cryoDRGN | Generative | 33.7 dB | 0.901 |  |
| CryoGEM | Generative | 35.2 dB | 0.924 |  |
| CryoFormer | Transformer | 37.1 dB | 0.941 |  |
| CryoSTAR | Deep Learning | 38.4 dB | 0.952 |  |
| DiffusionCryo | Diffusion | 39.8 dB | 0.963 |  |

## Cryo-Electron Tomography (Cryo-ET) (`cryo_et`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| WBP | Classical | 20.5 dB | 0.682 |  |
| SART-ET | Classical | 23.8 dB | 0.741 |  |
| IMOD | Classical | 25.2 dB | 0.774 |  |
| IsoNet | Deep Learning | 29.4 dB | 0.842 |  |
| DeepDeWedge | Self-Supervised | 31.7 dB | 0.876 |  |
| CryoSeg | Deep Learning | 33.1 dB | 0.898 |  |
| DeePiCt | Deep Learning | 34.2 dB | 0.909 |  |
| ETFormer | Transformer | 35.6 dB | 0.921 |  |
| DiffusionET | Diffusion | 37.9 dB | 0.944 |  |

## DESI Mass Spectrometry Imaging (`desi`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MSI-Hotelling | Classical | 22.1 dB | 0.701 | done |
| MSI-PCA | Classical | 24.8 dB | 0.749 | done |
| MSI-NMF | Classical | 26.3 dB | 0.782 | done |
| MSI-TV | Variational | 28.9 dB | 0.821 | |
| DeepMSI | Deep Learning | 32.4 dB | 0.871 | done |
| MSI-GAN | Generative | 33.7 dB | 0.888 | done |
| SpaMSI-Net | Deep Learning | 34.8 dB | 0.904 | done |
| MSIFormer | Transformer | 36.1 dB | 0.921 | done |
| DiffusionMSI | Diffusion | 38.2 dB | 0.942 |  |

## DNA-PAINT Super-Resolution (`dna_paint`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| STORM-2D | Classical | 21.3 dB | 0.695 | done |
| PALM | Classical | 22.8 dB | 0.718 | done |
| DAOSTORM | Classical | 25.4 dB | 0.762 | |
| DeepSTORM | Deep Learning | 29.1 dB | 0.831 | |
| DECODE | Deep Learning | 32.6 dB | 0.878 |  |
| TransPAINT | Transformer | 35.2 dB | 0.918 |  |
| SwinSTORM | Transformer | 36.8 dB | 0.934 |  |
| PhysSTORM | Physics-Informed | 38.1 dB | 0.946 |  |
| DiffPAINT | Diffusion Model | 39.7 dB | 0.958 |  |

## Dark-Field Microscopy (`dark_field`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 24.5 dB | 0.744 | |
| Wiener-DF | Classical | 27.2 dB | 0.793 | |
| TV-DF | Variational | 29.8 dB | 0.836 |  |
| BM3D-DF | Classical | 32.4 dB | 0.871 |  |
| Noise2Void-DF | Self-Supervised | 33.7 dB | 0.889 | done |
| CARE-DF | Deep Learning | 35.1 dB | 0.908 | done |
| SwinIR-DF | Transformer | 37.6 dB | 0.932 |  |
| Restormer-DF | Transformer | 38.9 dB | 0.943 |  |
| DiffusionDF | Diffusion | 40.3 dB | 0.956 |  |

## Differential Interference Contrast (DIC) (`dic`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DIC-Deconv | Classical | 24.1 dB | 0.731 |  |
| Phase-DLSIM | Classical | 25.9 dB | 0.762 |  |
| TV-DIC | Variational | 27.8 dB | 0.793 |  |
| DIC-CNN | Deep Learning | 31.4 dB | 0.856 |  |
| PnP-DIC | PnP | 32.2 dB | 0.869 |  |
| PhaseNet-DIC | Deep Learning | 33.7 dB | 0.884 |  |
| SwinDIC | Transformer | 36.1 dB | 0.921 |  |
| PhysPhase-Net | Physics-Informed | 37.4 dB | 0.935 |  |
| DiffusionDIC | Diffusion | 39.2 dB | 0.95 |  |

## Diffuse Optical Tomography (DOT) (`dot`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Born-Approx | Classical | 20.8 dB | 0.681 | done |
| TV-DOT | Variational | 23.5 dB | 0.729 | done |
| FEM-DOT | Classical | 25.9 dB | 0.771 |  |
| DnCNN-DOT | Deep Learning | 28.7 dB | 0.825 | done |
| DOT-Net | Deep Unrolling | 31.4 dB | 0.868 | done |
| TransDOT | Transformer | 34.2 dB | 0.91 | done |
| SwinDOT | Transformer | 36.1 dB | 0.93 | done |
| PhysDOT | Physics-Informed | 37.5 dB | 0.942 |  |
| DiffusionDOT | Diffusion Model | 39.0 dB | 0.954 |  |

## Diffusion MRI (DTI) (`diffusion_mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DTI-FIT | Classical | 22.4 dB | 0.71 | done |
| SHORE | Classical | 24.6 dB | 0.745 | done |
| CHARMED | Statistical | 26.8 dB | 0.782 | done |
| DnCNN-DTI | Deep Learning | 29.3 dB | 0.831 | done |
| DWIML-Net | Deep Learning | 32.1 dB | 0.871 | |
| DTIFormer | Transformer | 34.8 dB | 0.912 |  |
| SwinDTI | Transformer | 36.2 dB | 0.931 |  |
| PhysDiffMRI | Physics-Informed | 37.5 dB | 0.941 |  |
| DiffusionDTI | Diffusion Model | 39.1 dB | 0.952 |  |

## Digital Breast Tomosynthesis (DBT) (`digital_breast_tomo`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-DBT | Classical | 23.1 dB | 0.721 |  |
| TV-DBT | Variational | 25.8 dB | 0.768 |  |
| SART-DBT | Classical | 27.4 dB | 0.801 |  |
| DnCNN-DBT | Deep Learning | 30.2 dB | 0.848 |  |
| DuDoRNet-DBT | Deep Unrolling | 33.5 dB | 0.891 |  |
| TransDBT | Transformer | 35.8 dB | 0.921 |  |
| SwinDBT | Transformer | 37.2 dB | 0.938 |  |
| PhysDBT | Physics-Informed | 38.1 dB | 0.945 |  |
| DiffusionDBT | Diffusion Model | 39.4 dB | 0.956 |  |

## Digital Holographic Microscopy (`holography`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gerchberg-Saxton | Classical | 21.5 dB | 0.58 |  |
| Error Reduction | Classical | 22.85 dB | 0.615 |  |
| GS/HIO | Classical | 23.7 dB | 0.65 |  |
| deep-PR | Deep Learning | 27.2 dB | 0.81 |  |
| prDeep | Deep Unrolling | 27.45 dB | 0.82 |  |
| PhaseNet | Deep Learning | 31.2 dB | 0.91 |  |
| CyclePhase | Deep Learning | 32.5 dB | 0.938 |  |
| LRGS | Deep Learning | 32.8 dB | 0.935 |  |
| PhaseResNet | Deep Learning | 33.15 dB | 0.942 |  |
| PhaseFormer | Vision Transformer | 34.5 dB | 0.952 |  |
| AutoPhase++ | Vision Transformer | 34.92 dB | 0.958 |  |
| HolographyViT | Vision Transformer | 35.18 dB | 0.96 |  |
| DiffusionPhase | Diffusion | 35.48 dB | 0.964 |  |
| ScorePhase | Score-based | 35.82 dB | 0.968 |  |

## Doppler Ultrasound (`doppler_ultrasound`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CF-Doppler | Classical | 22.5 dB | 0.712 |  |
| VENC-Flow | Classical | 24.1 dB | 0.738 |  |
| MV-Doppler | Variational | 26.8 dB | 0.778 |  |
| DnCNN-Doppler | Deep Learning | 29.5 dB | 0.832 | done |
| FlowNet-US | Deep Learning | 32.4 dB | 0.872 | done |
| TransFlow | Transformer | 35.1 dB | 0.914 | done |
| SwinDoppler | Transformer | 36.8 dB | 0.932 | done |
| PhysDoppler | Physics-Informed | 37.9 dB | 0.942 |  |
| DiffDoppler | Diffusion Model | 39.3 dB | 0.954 |  |

## Dual-Energy X-ray Absorptiometry (DEXA) (`dexa`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-DEXA | Classical | 26.4 dB | 0.782 |  |
| BML-Sep | Classical | 28.7 dB | 0.813 |  |
| TV-DEXA | Variational | 30.1 dB | 0.841 |  |
| DXA-CNN | Deep Learning | 33.8 dB | 0.881 |  |
| PnP-DXA | PnP | 34.2 dB | 0.893 |  |
| DXA-U-Net | Deep Learning | 35.6 dB | 0.907 |  |
| SwinDXA | Transformer | 37.9 dB | 0.931 |  |
| PhysDXA | Physics-Informed | 38.7 dB | 0.94 |  |
| DiffusionDXA | Diffusion | 40.4 dB | 0.956 |  |

## Eddy Current Imaging (`eddy_current`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| EC-Deconv | Classical | 22.1 dB | 0.705 |  |
| TV-EC | Variational | 24.8 dB | 0.748 |  |
| MUSIC-EC | Classical | 27.3 dB | 0.789 |  |
| DnCNN-EC | Deep Learning | 30.1 dB | 0.84 | done |
| ECNN-Defect | Deep Learning | 32.9 dB | 0.88 | done |
| TransEC | Transformer | 35.4 dB | 0.918 | done |
| SwinEC | Transformer | 36.9 dB | 0.934 | done |
| PhysEC | Physics-Informed | 38.0 dB | 0.944 |  |
| DiffEC | Diffusion Model | 39.3 dB | 0.955 |  |

## Electrical Impedance Tomography (EIT) (`impedance_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gauss-Newton | Classical | 21.0 dB | 0.55 | done |
| TV-ADMM | PnP | 24.5 dB | 0.7 | |
| D-bar CNN | Deep Learning | 28.5 dB | 0.84 | done |
| EIT-Former | Transformer | 30.0 dB | 0.88 | done |

## Electron Backscatter Diffraction (EBSD) (`ebsd`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Hough-EBSD | Classical | 21.5 dB | 0.698 |  |
| DI-EBSD | Classical | 24.2 dB | 0.741 |  |
| TV-EBSD | Variational | 26.8 dB | 0.779 |  |
| DnCNN-EBSD | Deep Learning | 29.6 dB | 0.834 |  |
| PointEBSD | Deep Learning | 32.3 dB | 0.874 |  |
| TransEBSD | Transformer | 34.9 dB | 0.913 |  |
| SwinEBSD | Transformer | 36.5 dB | 0.931 |  |
| PhysEBSD | Physics-Informed | 37.8 dB | 0.943 |  |
| DiffEBSD | Diffusion Model | 39.1 dB | 0.954 |  |

## Electron Energy Loss Spectroscopy (EELS) (`eels`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PowerLaw-EELS | Classical | 21.8 dB | 0.699 | done |
| MLS-EELS | Statistical | 24.5 dB | 0.744 | done |
| ICA-EELS | Statistical | 27.1 dB | 0.786 | done |
| DnCNN-EELS | Deep Learning | 30.0 dB | 0.838 | done |
| N2V-EELS | Self-Supervised | 32.6 dB | 0.876 | done |
| TransEELS | Transformer | 35.1 dB | 0.915 | done |
| SwinEELS | Transformer | 36.7 dB | 0.932 | done |
| PhysEELS | Physics-Informed | 37.9 dB | 0.942 |  |
| DiffEELS | Diffusion Model | 39.3 dB | 0.954 |  |

## Electron Holography (`electron_holography`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FFT-Holo | Classical | 21.5 dB | 0.7 |  |
| WDD-Holo | Classical | 24.2 dB | 0.742 |  |
| TV-Phase | Variational | 26.8 dB | 0.783 |  |
| DnCNN-Holo | Deep Learning | 29.6 dB | 0.835 |  |
| DeepHolo | Deep Learning | 32.4 dB | 0.875 |  |
| TransHolo | Transformer | 34.9 dB | 0.913 |  |
| SwinHolo | Transformer | 36.5 dB | 0.931 |  |
| PhysHolo | Physics-Informed | 37.8 dB | 0.942 |  |
| DiffHolo | Diffusion Model | 39.2 dB | 0.953 |  |

## Electron Tomography (`electron_tomography`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| WBP-ET | Classical | 20.9 dB | 0.678 |  |
| SIRT-ET | Classical | 23.6 dB | 0.724 |  |
| CS-ET | Compressed Sensing | 26.4 dB | 0.769 |  |
| DnCNN-ET | Deep Learning | 29.3 dB | 0.829 |  |
| IsoNet | Deep Learning | 32.1 dB | 0.871 |  |
| TransET | Transformer | 34.8 dB | 0.91 |  |
| SwinET | Transformer | 36.4 dB | 0.929 |  |
| PhysET | Physics-Informed | 37.7 dB | 0.94 |  |
| DiffET | Diffusion Model | 39.1 dB | 0.952 |  |

## Entangled Photon Microscopy (`entangled_photon`) — quantum

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Coincidence-Count | Classical | 19.8 dB | 0.658 | done |
| CS-Ghost | Compressed Sensing | 22.5 dB | 0.704 | done |
| SVD-Ghost | Statistical | 25.1 dB | 0.748 | done |
| DnCNN-Ghost | Deep Learning | 28.3 dB | 0.806 | done |
| GAN-Ghost | Generative | 31.0 dB | 0.852 | |
| TransGhost | Transformer | 33.8 dB | 0.897 |  |
| SwinGhost | Transformer | 35.6 dB | 0.92 |  |
| PhysGhost | Physics-Informed | 37.1 dB | 0.936 |  |
| DiffGhost | Diffusion Model | 38.8 dB | 0.95 |  |

## Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Event-Integration | Classical | 22.1 dB | 0.702 | done |
| Complementary | Classical | 24.8 dB | 0.748 | done |
| E2VID | Recurrent | 27.9 dB | 0.798 | |
| FireNet | Recurrent | 30.4 dB | 0.843 | |
| SPADE-E2VID | Deep Learning | 32.8 dB | 0.878 | done |
| TransEvent | Transformer | 35.2 dB | 0.914 | done |
| SwinEvent | Transformer | 36.9 dB | 0.933 | done |
| PhysEvent | Physics-Informed | 38.0 dB | 0.944 |  |
| DiffEvent | Diffusion Model | 39.4 dB | 0.955 |  |

## Event Horizon Telescope (EHT) Imaging (`eht_imaging`) — astronomy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CLEAN-VLBI | Classical | 20.4 dB | 0.672 |  |
| MEM-VLBI | Variational | 23.1 dB | 0.718 |  |
| RESOLVE | Statistical | 25.8 dB | 0.761 |  |
| eht-imaging | Variational | 28.6 dB | 0.812 |  |
| SMILI | Compressed Sensing | 31.2 dB | 0.858 |  |
| TransVLBI | Transformer | 34.5 dB | 0.908 |  |
| RadioFormer | Transformer | 36.2 dB | 0.928 |  |
| PhysVLBI | Physics-Informed | 37.6 dB | 0.94 |  |
| DiffVLBI | Diffusion Model | 39.0 dB | 0.952 |  |

## Expansion Microscopy (ExM) (`expansion`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv-Exp | Classical | 24.5 dB | 0.742 | done |
| RL-ExM | Classical | 26.9 dB | 0.778 | done |
| TV-ExM | Variational | 29.1 dB | 0.819 | done |
| DnCNN-ExM | Deep Learning | 31.8 dB | 0.86 | done |
| DeepInterp-ExM | Deep Learning | 34.2 dB | 0.898 | done |
| TransExM | Transformer | 36.3 dB | 0.927 | done |
| SwinExM | Transformer | 37.7 dB | 0.941 |  |
| PhysExM | Physics-Informed | 38.8 dB | 0.95 |  |
| DiffExM | Diffusion Model | 40.0 dB | 0.96 |  |

## FTIR Spectroscopic Imaging (`ftir_imaging`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical | 24.3 dB | 0.67 | done |
| SVD | Classical | 24.99 dB | 0.802 | |
| Baseline Correction | Classical | 25.03 dB | 0.803 | |
| PnP-DnCNN | PnP | 27.9 dB | 0.8 | |
| ScoreSpectra | Score-based | 29.49 dB | 0.909 | done |
| DiffusionSpectra | Diffusion | 30.09 dB | 0.918 | done |
| U-Net-Spectra | Deep Learning | 30.57 dB | 0.925 | done |
| CDAE | Deep Learning | 31.5 dB | 0.895 | done |
| SpectraFormer | Vision Transformer | 32.72 dB | 0.95 | done |
| Cascade-UNet | Deep Learning | 33.0 dB | 0.922 | done |
| PINN-Spectra | Deep Learning | 33.17 dB | 0.954 | done |

## Fiber Bundle Endoscopy (`endoscopy`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Histogram-Eq | Classical | 24.1 dB | 0.738 |  |
| CLAHE-Endo | Classical | 26.5 dB | 0.772 |  |
| BM3D-Endo | Classical | 28.9 dB | 0.812 |  |
| DnCNN-Endo | Deep Learning | 31.4 dB | 0.855 | done |
| EndoSLAM-Net | Deep Learning | 33.8 dB | 0.889 | done |
| TransEndo | Transformer | 35.9 dB | 0.921 | done |
| SwinEndo | Transformer | 37.3 dB | 0.937 |  |
| PhysEndo | Physics-Informed | 38.4 dB | 0.947 |  |
| DiffEndo | Diffusion Model | 39.7 dB | 0.957 |  |

## Flash LiDAR (`flash_lidar`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLE-SPAD | Classical | 22.8 dB | 0.718 |  |
| Coates-Hist | Classical | 24.5 dB | 0.748 |  |
| NL-Means-LiDAR | Classical | 27.2 dB | 0.789 |  |
| DnCNN-LiDAR | Deep Learning | 30.1 dB | 0.84 |  |
| SPADnet | Deep Learning | 32.8 dB | 0.878 |  |
| TransLiDAR | Transformer | 35.3 dB | 0.916 |  |
| SwinLiDAR | Transformer | 36.9 dB | 0.933 |  |
| PhysLiDAR | Physics-Informed | 38.0 dB | 0.943 |  |
| DiffLiDAR | Diffusion Model | 39.4 dB | 0.955 |  |

## Fluorescence Lifetime Imaging (FLIM) (`flim`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phasor-FLIM | Classical | 23.2 dB | 0.722 |  |
| MLE-FLIM | Statistical | 25.8 dB | 0.762 |  |
| RLD-FLIM | Classical | 27.9 dB | 0.798 |  |
| DnCNN-FLIM | Deep Learning | 30.7 dB | 0.845 | done |
| FLIMJ | Deep Learning | 33.1 dB | 0.882 | done |
| TransFLIM | Transformer | 35.5 dB | 0.918 | done |
| SwinFLIM | Transformer | 37.0 dB | 0.935 | done |
| PhysFLIM | Physics-Informed | 38.2 dB | 0.945 |  |
| DiffFLIM | Diffusion Model | 39.6 dB | 0.957 |  |

## Fluoroscopy (`fluoroscopy`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BM3D-Fluoro | Classical | 25.8 dB | 0.762 |  |
| NLM-Fluoro | Classical | 27.4 dB | 0.791 |  |
| TV-Fluoro | Variational | 29.6 dB | 0.828 |  |
| DnCNN-Fluoro | Deep Learning | 32.1 dB | 0.866 |  |
| REDCNN-Fluoro | Deep Learning | 34.0 dB | 0.895 |  |
| TransFluoro | Transformer | 36.2 dB | 0.925 |  |
| SwinFluoro | Transformer | 37.6 dB | 0.94 |  |
| PhysFluoro | Physics-Informed | 38.7 dB | 0.949 |  |
| DiffFluoro | Diffusion Model | 40.0 dB | 0.96 |  |

## Focused Ion Beam SEM (FIB-SEM) (`fib_sem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BM3D-FIB | Classical | 25.3 dB | 0.755 | done |
| NLM-FIB | Classical | 27.1 dB | 0.789 | done |
| TV-FIB | Variational | 29.4 dB | 0.825 | done |
| DnCNN-FIB | Deep Learning | 31.9 dB | 0.862 | done |
| N2V-FIB | Self-Supervised | 33.8 dB | 0.891 | done |
| TransFIB | Transformer | 36.1 dB | 0.923 | done |
| SwinFIB | Transformer | 37.5 dB | 0.939 |  |
| PhysFIB | Physics-Informed | 38.6 dB | 0.949 |  |
| DiffFIB | Diffusion Model | 39.9 dB | 0.959 |  |

## Fourier Ptychographic Microscopy (FPM) (`fpm`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Alternating Projections | Classical | 25.0 dB | 0.72 |  |
| Gradient Descent FPM | Classical | 28.5 dB | 0.84 |  |
| Fourier PtychoNet | Deep Learning | 32.3 dB | 0.91 |  |
| PtychoDV | Deep Unrolling | 33.8 dB | 0.935 |  |

## Full-Waveform Inversion (FWI) (`fwi`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| L-BFGS FWI | Classical | 23.5 dB | 0.65 | done |
| TV-Reg FWI | Classical | 26.8 dB | 0.78 | |
| InversionNet | Deep Learning | 30.5 dB | 0.88 | done |
| VelocityGAN | Deep Learning | 32.2 dB | 0.91 | done |

## Functional MRI (BOLD fMRI) (`fmri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 26.0 dB | 0.62 | done |
| SENSE | Classical | 29.5 dB | 0.83 | |
| GRAPPA | Classical | 31.2 dB | 0.86 | |
| Score-MRI | Score-Based | 31.4 dB | 0.89 | |
| L1-Wavelet | Compressed Sensing | 32.1 dB | 0.87 | |
| k-t SPARSE-SENSE | Compressed Sensing | 32.5 dB | 0.875 | |
| ESPIRiT | Compressed Sensing | 33.4 dB | 0.89 | |
| LORAKS | Compressed Sensing | 33.8 dB | 0.893 | |
| BM3D-MRI | PnP | 34.2 dB | 0.897 | |
| ALOHA | Low-Rank | 34.5 dB | 0.9 | |
| PnP-DnCNN | PnP | 35.0 dB | 0.905 | |
| Deep-ADMM-Net | Deep Unrolling | 35.3 dB | 0.907 | |
| DCCNN | Deep Learning | 35.5 dB | 0.908 | |
| U-Net | Deep Learning | 35.9 dB | 0.904 | |
| MoDL | Deep Unrolling | 36.5 dB | 0.912 | |
| HybridCascade | Deep Unrolling | 37.8 dB | 0.917 | |
| SwinMR | Transformer | 38.5 dB | 0.921 | |
| HUMUS-Net | Transformer | 38.9 dB | 0.923 | |
| ReconFormer | Transformer | 39.0 dB | 0.922 | |
| E2E-VarNet | Deep Unrolling | 39.4 dB | 0.924 | |
| PromptMR | Deep Unrolling | 39.7 dB | 0.926 | |
| MRI-DiffusionNet | Diffusion | 40.1 dB | 0.932 | |
| MRDynamo | Physics-Informed | 40.5 dB | 0.938 | |
| MMR-Mamba | Physics-Informed | 40.98 dB | 0.969 | |
| PnP-DnCNN-Pro | PnP | 41.0 dB | 0.968 | |
| BrainID-MRI | Foundation Model | 41.0 dB | 0.942 | |
| PromptMR-SFM | Physics-Informed | 41.3 dB | 0.971 | |
| U-Net++ | Deep Learning | 41.5 dB | 0.978 | |
| ReconFormer++ | Transformer | 41.5 dB | 0.969 | |
| MoDL-Net++ | Deep Unrolling | 41.8 dB | 0.978 | |
| MRI-FM | Foundation Model | 42.1 dB | 0.948 | |
| MR-IPT | Foundation Model | 42.48 dB | 0.983 | |
| HybridCascade++ | Deep Unrolling | 42.5 dB | 0.981 | |
| HUMUS-Net++ | Transformer | 43.1 dB | 0.979 | |
| SwinMR++ | Transformer | 43.8 dB | 0.983 | |

## Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov-DOT | Classical | 26.61 dB | 0.848 | done |
| MBLL | Classical | 29.95 dB | 0.916 | done |
| PnP-DOT | PnP | 33.34 dB | 0.956 | |
| DL-DOT | Deep Learning | 35.62 dB | 0.971 | done |

## Fundus Camera (`fundus`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 24.5 dB | 0.68 | done |
| PnP-BM3D | PnP | 28.8 dB | 0.83 | done |
| cofe-Net | Deep Learning | 32.5 dB | 0.91 | done |
| Swin-Fundus | Transformer | 34.2 dB | 0.94 | done |

## Generic Matrix Sensing (`matrix`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| GAP-TV | Classical | 26.83 dB | 0.754 |  |
| FISTA-TV | Classical | 28.42 dB | 0.821 |  |
| TVAL3 | Classical | 29.15 dB | 0.845 |  |
| PnP-FFDNet | PnP | 29.65 dB | 0.852 |  |
| EfficientSCI | Deep Learning | 34.21 dB | 0.949 | done |
| MST-L | Transformer | 35.4 dB | 0.96 | done |
| Restormer | Vision Transformer | 35.68 dB | 0.962 | done |
| CST | Transformer | 35.92 dB | 0.965 | done |
| HiSViT+ | Vision Transformer | 36.85 dB | 0.971 | done |
| CSTrans | Transformer | 37.12 dB | 0.973 |  |
| PromptSCI | Deep Learning | 37.35 dB | 0.975 |  |
| DiffusionHSI | Diffusion | 37.95 dB | 0.978 |  |
| ScoreSCI | Diffusion | 38.22 dB | 0.98 |  |
| FlowHSI | Generative | 38.58 dB | 0.982 |  |

## Ghost Imaging (`ghost_imaging`) — quantum

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| G(2)-Corr | Classical | 21.2 dB | 0.55 |  |
| Photon Counting | Classical | 22.56 dB | 0.713 |  |
| CS-TVAL3 | PnP | 24.8 dB | 0.71 |  |
| Bayesian CS | PnP | 25.07 dB | 0.804 |  |
| DiffusionQuantum | Diffusion | 26.49 dB | 0.845 |  |
| Quantum-CNN | Deep Learning | 28.43 dB | 0.89 |  |
| DRU-Net | Deep Learning | 28.5 dB | 0.84 |  |
| Quantum-ViT | Vision Transformer | 28.78 dB | 0.896 |  |
| ScoreQuantum | Score-based | 29.52 dB | 0.909 |  |
| Ghost-ViT | Vision Transformer | 30.1 dB | 0.885 |  |

## Gravitational Wave Detection (`gravitational_wave`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Matched Filter | Classical | 20.0 dB | 0.52 | done |
| BayesWave | PnP | 24.5 dB | 0.71 |  |
| GW-CNN | Deep Learning | 28.8 dB | 0.85 | done |
| WaveFormer | Transformer | 30.5 dB | 0.895 | done |

## Ground-Penetrating Radar (GPR) (`gpr`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Kirchhoff Migration | Classical | 22.0 dB | 0.6 |  |
| RTM | Classical | 25.5 dB | 0.74 |  |
| GPR-RCNN | Deep Learning | 29.8 dB | 0.87 |  |
| HyperDet | Deep Learning | 31.5 dB | 0.905 |  |

## High Dynamic Range (HDR) Imaging (`hdr_imaging`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Laplacian Pyramid | Classical | 26.42 dB | 0.843 |  |
| Lucy-Richardson | Classical | 26.61 dB | 0.848 |  |
| Wiener-Deconv | Classical | 27.8 dB | 0.78 |  |
| PnP-FFDNet | PnP | 31.45 dB | 0.885 |  |
| PnP-ADMM | PnP | 31.89 dB | 0.941 |  |
| LaplacianFormer | Deep Learning | 33.16 dB | 0.954 | done |
| HDR-CNN | Deep Learning | 34.9 dB | 0.945 | done |
| U-Net | Deep Learning | 35.1 dB | 0.968 | done |
| ScorePhoto | Score-based | 35.49 dB | 0.971 | done |
| Uformer | Vision Transformer | 36.2 dB | 0.96 | done |
| PhotoFormer | Vision Transformer | 36.48 dB | 0.976 | done |
| DeblurGaussian | Vision Transformer | 37.68 dB | 0.968 |  |
| HDRFormer | Vision Transformer | 38.15 dB | 0.972 |  |
| DiffusionPhoto | Diffusion | 38.82 dB | 0.978 |  |

## Hyperspectral Remote Sensing (`hyperspectral_remote`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CNMF | Classical | 26.0 dB | 0.72 | done |
| PnP-LTTR | PnP | 30.0 dB | 0.85 | done |
| DBIN | Deep Learning | 34.5 dB | 0.93 | done |
| MST++ | Transformer | 36.8 dB | 0.955 |  |

## Image Scanning Microscopy (ISM) (`ism`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | done |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | done |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## Industrial X-ray CT (`industrial_ct`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FDK | Classical | 28.32 dB | 0.887 |  |
| PnP-ADMM | PnP | 32.64 dB | 0.891 |  |
| FBPConvNet | Deep Learning | 35.81 dB | 0.939 |  |
| Learned Primal-Dual | Deep Unrolling | 36.42 dB | 0.947 |  |

## Integral Photography (`integral`) — computational

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Shift-and-Add | Classical | 25.0 dB | 0.7 |  |
| PnP-LF | PnP | 29.0 dB | 0.83 |  |
| LFAttNet | Deep Learning | 33.5 dB | 0.92 |  |
| DistgSSR | Transformer | 35.8 dB | 0.95 |  |

## Interferometric SAR (InSAR) (`insar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Goldstein-MCF | Classical | 23.0 dB | 0.64 |  |
| InSAR-BM3D | PnP | 27.0 dB | 0.79 |  |
| PhaseNet | Deep Learning | 31.0 dB | 0.89 |  |
| InSAR-Former | Transformer | 33.0 dB | 0.92 |  |

## Intravascular Ultrasound (IVUS) (`ivus`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DAS | Classical | 24.5 dB | 0.68 |  |
| DAS-CF | Classical | 25.8 dB | 0.72 |  |
| PW-DAS | Classical | 26.15 dB | 0.735 |  |
| PnP-ADMM | PnP | 28.12 dB | 0.81 |  |
| ABLE | Deep Learning | 31.85 dB | 0.905 | done |
| PnP-TV | PnP | 33.1 dB | 0.953 |  |
| MU-Net | Deep Learning | 33.2 dB | 0.928 | done |
| Phase-ADMM-Net | Deep Unrolling | 33.95 dB | 0.94 | done |
| UltrasoundFormer | Vision Transformer | 34.85 dB | 0.945 | done |
| BeamFormer | Transformer | 35.15 dB | 0.948 | done |
| BeamDATA | Deep Learning | 35.32 dB | 0.951 | done |
| AttentionBeam | Transformer | 35.52 dB | 0.952 | done |
| DiffUS | Diffusion | 35.95 dB | 0.958 | done |
| ScoreUS | Score-based | 36.28 dB | 0.962 | done |

## Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical | 24.3 dB | 0.67 | done |
| Baseline Correction | Classical | 24.33 dB | 0.78 | done |
| SVD | Classical | 26.56 dB | 0.847 | done |
| PnP-DnCNN | PnP | 27.9 dB | 0.8 | done |
| U-Net-Spectra | Deep Learning | 29.1 dB | 0.902 | done |
| ScoreSpectra | Score-based | 31.36 dB | 0.935 | done |
| CDAE | Deep Learning | 31.5 dB | 0.895 | done |
| PINN-Spectra | Deep Learning | 31.72 dB | 0.94 | done |
| Cascade-UNet | Deep Learning | 33.0 dB | 0.922 | done |
| SpectraFormer | Vision Transformer | 33.3 dB | 0.955 | done |
| DiffusionSpectra | Diffusion | 33.86 dB | 0.96 | done |

## Lattice Light-Sheet Microscopy (`lattice_lightsheet`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | done |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | done |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | done |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## Lensless (Diffuser Camera) Imaging (`lensless`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-ADMM | Classical | 23.5 dB | 0.64 |  |
| PnP-ADMM | PnP | 27.5 dB | 0.79 |  |
| FlatNet | Deep Learning | 31.8 dB | 0.89 |  |
| Uformer | Transformer | 33.5 dB | 0.92 |  |

## LiDAR Scanner (`lidar`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Bilateral Filter | Classical | 27.41 dB | 0.868 |  |
| PnP-ADMM | PnP | 29.1 dB | 0.84 |  |
| RandLA-Net | Deep Learning | 31.91 dB | 0.942 | done |
| Point Transformer | Transformer | 33.13 dB | 0.954 | done |

## Light Field Imaging (`light_field`) — computational

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Shift-and-Sum | Classical | 24.5 dB | 0.69 |  |
| PnP-LF | PnP | 28.5 dB | 0.82 |  |
| LFNet | Deep Learning | 33.0 dB | 0.915 |  |
| DistgSSR | Transformer | 35.5 dB | 0.948 |  |

## Light-Sheet Fluorescence Microscopy (LSFM) (`lightsheet`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 |  |
| Wiener Filter | Classical | 28.35 dB | 0.805 |  |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 |  |
| PnP-FISTA | PnP | 30.42 dB | 0.872 |  |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 |  |
| CARE | Deep Learning | 34.5 dB | 0.948 |  |
| U-Net | Deep Learning | 35.15 dB | 0.956 |  |
| Restormer | Vision Transformer | 35.8 dB | 0.962 |  |
| ResUNet | Deep Learning | 35.85 dB | 0.964 |  |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 |  |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 |  |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 |  |
| ScoreMicro | Score-based | 38.48 dB | 0.981 |  |

## Low-Dose Widefield Microscopy (`widefield_lowdose`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | done |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | done |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | done |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## Lucky Imaging (`lucky_imaging`) — astronomy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Shift-and-Add | Classical | 22.65 dB | 0.717 | done |
| BDI | PnP | 24.62 dB | 0.79 | done |
| Drizzle | Classical | 24.8 dB | 0.796 | done |
| SpeckleNet | Deep Learning | 31.22 dB | 0.934 | done |

## MALDI Mass Spectrometry Imaging (`maldi_msi`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv | Classical | 24.1 dB | 0.66 | |
| Calibration-Lookup | Classical | 24.12 dB | 0.773 | |
| Peak Fitting | Classical | 25.91 dB | 0.829 | |
| PnP-NLM | PnP | 26.67 dB | 0.85 | |
| PnP-BM3D | PnP | 27.6 dB | 0.79 | |
| ScoreInstrumentation | Score-based | 29.06 dB | 0.901 | done |
| Instrument-CNN | Deep Learning | 29.61 dB | 0.911 | done |
| DiffusionInstrumentation | Diffusion | 30.54 dB | 0.925 | done |
| MassSpecFormer | Vision Transformer | 30.55 dB | 0.925 | done |
| ResNet-Calib | Deep Learning | 31.3 dB | 0.892 | done |
| CalibFormer | Vision Transformer | 32.8 dB | 0.92 | done |

## MINFLUX Nanoscopy (`minflux`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLE Localization | Classical | 28.28 dB | 0.887 |  |
| SPARCOM | PnP | 28.76 dB | 0.896 |  |
| DECODE | Deep Learning | 32.1 dB | 0.915 |  |
| ANNA-PALM | Deep Learning | 33.08 dB | 0.953 |  |

## MR Angiography (MRA) (`mra`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 26.0 dB | 0.62 |  |
| SENSE | Classical | 29.5 dB | 0.83 |  |
| GRAPPA | Classical | 31.2 dB | 0.86 |  |
| Score-MRI | Score-Based | 31.4 dB | 0.89 |  |
| L1-Wavelet | Compressed Sensing | 32.1 dB | 0.87 |  |
| k-t SPARSE-SENSE | Compressed Sensing | 32.5 dB | 0.875 |  |
| ESPIRiT | Compressed Sensing | 33.4 dB | 0.89 |  |
| LORAKS | Compressed Sensing | 33.8 dB | 0.893 |  |
| BM3D-MRI | PnP | 34.2 dB | 0.897 |  |
| ALOHA | Low-Rank | 34.5 dB | 0.9 |  |
| PnP-DnCNN | PnP | 35.0 dB | 0.905 |  |
| Deep-ADMM-Net | Deep Unrolling | 35.3 dB | 0.907 |  |
| DCCNN | Deep Learning | 35.5 dB | 0.908 |  |
| U-Net | Deep Learning | 35.9 dB | 0.904 |  |
| MoDL | Deep Unrolling | 36.5 dB | 0.912 |  |
| HybridCascade | Deep Unrolling | 37.8 dB | 0.917 |  |
| SwinMR | Transformer | 38.5 dB | 0.921 |  |
| HUMUS-Net | Transformer | 38.9 dB | 0.923 |  |
| ReconFormer | Transformer | 39.0 dB | 0.922 |  |
| E2E-VarNet | Deep Unrolling | 39.4 dB | 0.924 |  |
| PromptMR | Deep Unrolling | 39.7 dB | 0.926 |  |
| MRI-DiffusionNet | Diffusion | 40.1 dB | 0.932 |  |
| MRDynamo | Physics-Informed | 40.5 dB | 0.938 |  |
| MMR-Mamba | Physics-Informed | 40.98 dB | 0.969 |  |
| PnP-DnCNN-Pro | PnP | 41.0 dB | 0.968 |  |
| BrainID-MRI | Foundation Model | 41.0 dB | 0.942 |  |
| PromptMR-SFM | Physics-Informed | 41.3 dB | 0.971 |  |
| U-Net++ | Deep Learning | 41.5 dB | 0.978 |  |
| ReconFormer++ | Transformer | 41.5 dB | 0.969 |  |
| MoDL-Net++ | Deep Unrolling | 41.8 dB | 0.978 |  |
| MRI-FM | Foundation Model | 42.1 dB | 0.948 |  |
| MR-IPT | Foundation Model | 42.48 dB | 0.983 |  |
| HybridCascade++ | Deep Unrolling | 42.5 dB | 0.981 |  |
| HUMUS-Net++ | Transformer | 43.1 dB | 0.979 |  |
| SwinMR++ | Transformer | 43.8 dB | 0.983 |  |

## MR Elastography (MRE) (`mr_elastography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 26.0 dB | 0.62 |  |
| SENSE | Classical | 29.5 dB | 0.83 |  |
| GRAPPA | Classical | 31.2 dB | 0.86 |  |
| Score-MRI | Score-Based | 31.4 dB | 0.89 |  |
| L1-Wavelet | Compressed Sensing | 32.1 dB | 0.87 |  |
| k-t SPARSE-SENSE | Compressed Sensing | 32.5 dB | 0.875 |  |
| ESPIRiT | Compressed Sensing | 33.4 dB | 0.89 |  |
| LORAKS | Compressed Sensing | 33.8 dB | 0.893 |  |
| BM3D-MRI | PnP | 34.2 dB | 0.897 |  |
| ALOHA | Low-Rank | 34.5 dB | 0.9 |  |
| PnP-DnCNN | PnP | 35.0 dB | 0.905 |  |
| Deep-ADMM-Net | Deep Unrolling | 35.3 dB | 0.907 |  |
| DCCNN | Deep Learning | 35.5 dB | 0.908 |  |
| U-Net | Deep Learning | 35.9 dB | 0.904 |  |
| MoDL | Deep Unrolling | 36.5 dB | 0.912 |  |
| HybridCascade | Deep Unrolling | 37.8 dB | 0.917 |  |
| SwinMR | Transformer | 38.5 dB | 0.921 |  |
| HUMUS-Net | Transformer | 38.9 dB | 0.923 |  |
| ReconFormer | Transformer | 39.0 dB | 0.922 |  |
| E2E-VarNet | Deep Unrolling | 39.4 dB | 0.924 |  |
| PromptMR | Deep Unrolling | 39.7 dB | 0.926 |  |
| MRI-DiffusionNet | Diffusion | 40.1 dB | 0.932 |  |
| MRDynamo | Physics-Informed | 40.5 dB | 0.938 |  |
| MMR-Mamba | Physics-Informed | 40.98 dB | 0.969 |  |
| PnP-DnCNN-Pro | PnP | 41.0 dB | 0.968 |  |
| BrainID-MRI | Foundation Model | 41.0 dB | 0.942 |  |
| PromptMR-SFM | Physics-Informed | 41.3 dB | 0.971 |  |
| U-Net++ | Deep Learning | 41.5 dB | 0.978 |  |
| ReconFormer++ | Transformer | 41.5 dB | 0.969 |  |
| MoDL-Net++ | Deep Unrolling | 41.8 dB | 0.978 |  |
| MRI-FM | Foundation Model | 42.1 dB | 0.948 |  |
| MR-IPT | Foundation Model | 42.48 dB | 0.983 |  |
| HybridCascade++ | Deep Unrolling | 42.5 dB | 0.981 |  |
| HUMUS-Net++ | Transformer | 43.1 dB | 0.979 |  |
| SwinMR++ | Transformer | 43.8 dB | 0.983 |  |

## MR Fingerprinting (MRF) (`mr_fingerprinting`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SVD-MRF | Classical | 23.5 dB | 0.65 |  |
| MANTIS | Classical | 27.0 dB | 0.79 |  |
| MRF-Net | Deep Learning | 31.5 dB | 0.895 |  |
| MRF-Former | Transformer | 33.5 dB | 0.93 |  |

## MR Spectroscopy (MRS) (`mrs`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 26.0 dB | 0.62 |  |
| SENSE | Classical | 29.5 dB | 0.83 |  |
| GRAPPA | Classical | 31.2 dB | 0.86 |  |
| Score-MRI | Score-Based | 31.4 dB | 0.89 |  |
| L1-Wavelet | Compressed Sensing | 32.1 dB | 0.87 |  |
| k-t SPARSE-SENSE | Compressed Sensing | 32.5 dB | 0.875 |  |
| ESPIRiT | Compressed Sensing | 33.4 dB | 0.89 |  |
| LORAKS | Compressed Sensing | 33.8 dB | 0.893 |  |
| BM3D-MRI | PnP | 34.2 dB | 0.897 |  |
| ALOHA | Low-Rank | 34.5 dB | 0.9 |  |
| PnP-DnCNN | PnP | 35.0 dB | 0.905 |  |
| Deep-ADMM-Net | Deep Unrolling | 35.3 dB | 0.907 |  |
| DCCNN | Deep Learning | 35.5 dB | 0.908 |  |
| U-Net | Deep Learning | 35.9 dB | 0.904 |  |
| MoDL | Deep Unrolling | 36.5 dB | 0.912 |  |
| HybridCascade | Deep Unrolling | 37.8 dB | 0.917 |  |
| SwinMR | Transformer | 38.5 dB | 0.921 |  |
| HUMUS-Net | Transformer | 38.9 dB | 0.923 |  |
| ReconFormer | Transformer | 39.0 dB | 0.922 |  |
| E2E-VarNet | Deep Unrolling | 39.4 dB | 0.924 |  |
| PromptMR | Deep Unrolling | 39.7 dB | 0.926 |  |
| MRI-DiffusionNet | Diffusion | 40.1 dB | 0.932 |  |
| MRDynamo | Physics-Informed | 40.5 dB | 0.938 |  |
| MMR-Mamba | Physics-Informed | 40.98 dB | 0.969 |  |
| PnP-DnCNN-Pro | PnP | 41.0 dB | 0.968 |  |
| BrainID-MRI | Foundation Model | 41.0 dB | 0.942 |  |
| PromptMR-SFM | Physics-Informed | 41.3 dB | 0.971 |  |
| U-Net++ | Deep Learning | 41.5 dB | 0.978 |  |
| ReconFormer++ | Transformer | 41.5 dB | 0.969 |  |
| MoDL-Net++ | Deep Unrolling | 41.8 dB | 0.978 |  |
| MRI-FM | Foundation Model | 42.1 dB | 0.948 |  |
| MR-IPT | Foundation Model | 42.48 dB | 0.983 |  |
| HybridCascade++ | Deep Unrolling | 42.5 dB | 0.981 |  |
| HUMUS-Net++ | Transformer | 43.1 dB | 0.979 |  |
| SwinMR++ | Transformer | 43.8 dB | 0.983 |  |

## Machine Vision / AOI (`machine_vision`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Template Match | Classical | 27.59 dB | 0.872 |  |
| PnP-ADMM | PnP | 29.7 dB | 0.855 |  |
| PatchCore | Deep Learning | 31.24 dB | 0.934 | done |
| UniAD | Transformer | 34.61 dB | 0.965 | done |

## Magnetic Force Microscopy (MFM) (`mfm`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BTR | Classical | 23.2 dB | 0.63 |  |
| MLE Reconstruction | Classical | 24.12 dB | 0.773 |  |
| Reg-Deconv | PnP | 26.8 dB | 0.77 |  |
| TV-Deconvolution | PnP | 27.38 dB | 0.867 |  |
| ScoreSPM | Score-based | 28.85 dB | 0.898 | done |
| DiffusionSPM | Diffusion | 30.01 dB | 0.917 | done |
| U-Net-SPM | Deep Learning | 30.29 dB | 0.921 | done |
| DeepSPM | Deep Learning | 30.4 dB | 0.88 | done |
| E2E-BTR | Deep Learning | 31.8 dB | 0.908 | done |
| SPM-Former | Vision Transformer | 33.79 dB | 0.959 | done |

## Magnetic Particle Imaging (MPI) (`magnetic_particle`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Matched Filter | Classical | 24.65 dB | 0.791 |  |
| Tikhonov | Classical | 25.4 dB | 0.71 |  |
| Wiener Filter | Classical | 27.39 dB | 0.867 |  |
| PnP-RED | PnP | 28.9 dB | 0.835 |  |
| PnP-ADMM | PnP | 29.57 dB | 0.91 |  |
| DiffusionExperimental | Diffusion | 30.6 dB | 0.926 | done |
| Domain-Adapted-CNN | Deep Learning | 32.16 dB | 0.944 | done |
| ExpFormer | Vision Transformer | 32.23 dB | 0.945 | done |
| ResUNet | Deep Learning | 32.6 dB | 0.915 | done |
| ScoreExperimental | Score-based | 33.54 dB | 0.957 | done |
| SwinIR | Vision Transformer | 34.1 dB | 0.942 | done |

## Magnetic Resonance Imaging (MRI) (`mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 26.0 dB | 0.62 |  |
| SENSE | Classical | 29.5 dB | 0.83 |  |
| GRAPPA | Classical | 31.2 dB | 0.86 |  |
| Score-MRI | Score-Based | 31.4 dB | 0.89 |  |
| L1-Wavelet | Compressed Sensing | 32.1 dB | 0.87 |  |
| k-t SPARSE-SENSE | Compressed Sensing | 32.5 dB | 0.875 |  |
| ESPIRiT | Compressed Sensing | 33.4 dB | 0.89 |  |
| LORAKS | Compressed Sensing | 33.8 dB | 0.893 |  |
| BM3D-MRI | PnP | 34.2 dB | 0.897 |  |
| ALOHA | Low-Rank | 34.5 dB | 0.9 |  |
| PnP-DnCNN | PnP | 35.0 dB | 0.905 |  |
| Deep-ADMM-Net | Deep Unrolling | 35.3 dB | 0.907 |  |
| DCCNN | Deep Learning | 35.5 dB | 0.908 |  |
| U-Net | Deep Learning | 35.9 dB | 0.904 |  |
| MoDL | Deep Unrolling | 36.5 dB | 0.912 |  |
| HybridCascade | Deep Unrolling | 37.8 dB | 0.917 |  |
| SwinMR | Transformer | 38.5 dB | 0.921 |  |
| HUMUS-Net | Transformer | 38.9 dB | 0.923 |  |
| ReconFormer | Transformer | 39.0 dB | 0.922 |  |
| E2E-VarNet | Deep Unrolling | 39.4 dB | 0.924 |  |
| PromptMR | Deep Unrolling | 39.7 dB | 0.926 |  |
| MRI-DiffusionNet | Diffusion | 40.1 dB | 0.932 |  |
| MRDynamo | Physics-Informed | 40.5 dB | 0.938 |  |
| MMR-Mamba | Physics-Informed | 40.98 dB | 0.969 |  |
| PnP-DnCNN-Pro | PnP | 41.0 dB | 0.968 |  |
| BrainID-MRI | Foundation Model | 41.0 dB | 0.942 |  |
| PromptMR-SFM | Physics-Informed | 41.3 dB | 0.971 |  |
| U-Net++ | Deep Learning | 41.5 dB | 0.978 |  |
| ReconFormer++ | Transformer | 41.5 dB | 0.969 |  |
| MoDL-Net++ | Deep Unrolling | 41.8 dB | 0.978 |  |
| MRI-FM | Foundation Model | 42.1 dB | 0.948 |  |
| MR-IPT | Foundation Model | 42.48 dB | 0.983 |  |
| HybridCascade++ | Deep Unrolling | 42.5 dB | 0.981 |  |
| HUMUS-Net++ | Transformer | 43.1 dB | 0.979 |  |
| SwinMR++ | Transformer | 43.8 dB | 0.983 |  |

## Mammography (`mammography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 27.38 dB | 0.79 |  |
| TV-ADMM | Classical | 30.15 dB | 0.862 |  |
| PnP-ADMM | PnP | 32.64 dB | 0.891 |  |
| PnP-DnCNN | PnP | 33.45 dB | 0.905 |  |
| RED-CNN | Deep Learning | 33.56 dB | 0.908 |  |
| FBPConvNet | Deep Learning | 35.81 dB | 0.939 |  |
| Learned Primal-Dual | Deep Unrolling | 36.42 dB | 0.947 |  |
| DuDoTrans | Deep Unrolling | 37.68 dB | 0.962 |  |
| DOLCE | Diffusion | 38.32 dB | 0.971 |  |
| CT-ViT | Vision Transformer | 39.15 dB | 0.978 |  |
| CTFormer | Transformer | 39.45 dB | 0.98 |  |
| DiffusionCT | Diffusion | 39.68 dB | 0.982 |  |
| Score-CT | Score-based | 39.92 dB | 0.984 |  |

## Multispectral Satellite Imaging (`multispectral_sat`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov | Classical | 26.5 dB | 0.74 |  |
| LSQR | Classical | 27.8 dB | 0.785 |  |
| ART | Classical | 28.2 dB | 0.8 |  |
| Plug-and-Play | Deep Learning | 29.11 dB | 0.902 | done |
| PnP-RED | PnP | 30.18 dB | 0.865 |  |
| PnP-ADMM | PnP | 30.85 dB | 0.88 |  |
| Deep Image Prior | Deep Learning | 33.72 dB | 0.932 | done |
| SwinIR | Vision Transformer | 35.1 dB | 0.955 | done |
| NAFNet | Vision Transformer | 35.75 dB | 0.962 | done |
| Restormer | Vision Transformer | 36.28 dB | 0.968 | done |
| CompFormer | Vision Transformer | 37.15 dB | 0.972 |  |
| DiffusionCompute | Diffusion | 37.95 dB | 0.978 |  |
| FlowCompute | Generative | 38.35 dB | 0.98 |  |

## Muon Tomography (`muon_tomo`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv | Classical | 24.1 dB | 0.66 |  |
| Calibration-Lookup | Classical | 24.58 dB | 0.789 |  |
| Peak Fitting | Classical | 26.65 dB | 0.849 |  |
| PnP-NLM | PnP | 27.37 dB | 0.867 |  |
| PnP-BM3D | PnP | 27.6 dB | 0.79 |  |
| ScoreInstrumentation | Score-based | 29.43 dB | 0.908 |  |
| ResNet-Calib | Deep Learning | 31.3 dB | 0.892 |  |
| Instrument-CNN | Deep Learning | 31.88 dB | 0.941 |  |
| MassSpecFormer | Vision Transformer | 32.0 dB | 0.943 |  |
| CalibFormer | Vision Transformer | 32.8 dB | 0.92 |  |
| DiffusionInstrumentation | Diffusion | 33.62 dB | 0.958 |  |

## Near-field Scanning Optical Microscopy (NSOM) (`nsom`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLE Reconstruction | Classical | 23.17 dB | 0.738 | done |
| BTR | Classical | 23.2 dB | 0.63 | done |
| TV-Deconvolution | PnP | 26.32 dB | 0.841 | |
| Reg-Deconv | PnP | 26.8 dB | 0.77 | |
| DeepSPM | Deep Learning | 30.4 dB | 0.88 | done |
| SPM-Former | Vision Transformer | 31.25 dB | 0.934 | done |
| E2E-BTR | Deep Learning | 31.8 dB | 0.908 | done |
| ScoreSPM | Score-based | 31.8 dB | 0.94 | done |
| DiffusionSPM | Diffusion | 32.87 dB | 0.951 | done |
| U-Net-SPM | Deep Learning | 32.94 dB | 0.952 | done |

## Neural Radiance Fields (NeRF) (`nerf`) — neural_rendering

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| COLMAP+MVS | Classical | 26.4 dB | 0.73 |  |
| Photogrammetry | Classical | 26.49 dB | 0.845 |  |
| Mip-NeRF 360 | Deep Learning | 29.4 dB | 0.844 |  |
| Mesh-GS | Deep Learning | 30.48 dB | 0.924 |  |
| Instant-NGP | Deep Learning | 31.1 dB | 0.905 |  |
| NeRF | Deep Learning | 31.19 dB | 0.933 |  |
| 2DGS | Deep Learning | 31.44 dB | 0.936 |  |
| 3D-GS | Deep Learning | 33.3 dB | 0.94 |  |
| 3D-GS++ | Deep Learning | 34.52 dB | 0.952 |  |
| GaussianShader | Vision Transformer | 35.18 dB | 0.96 |  |
| NeRFactor2 | Deep Learning | 35.85 dB | 0.966 |  |

## Neutron Diffraction (`neutron_diffraction`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Rietveld-GSAS | Classical | 23.0 dB | 0.64 |  |
| Le Bail Fit | Classical | 26.5 dB | 0.76 |  |
| NeutronNet | Deep Learning | 30.5 dB | 0.88 |  |
| DiffFormer | Transformer | 32.5 dB | 0.915 |  |

## Neutron Radiography / Tomography (`neutron_tomo`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv | Classical | 24.1 dB | 0.66 |  |
| Peak Fitting | Classical | 26.04 dB | 0.833 |  |
| Calibration-Lookup | Classical | 26.6 dB | 0.848 |  |
| PnP-BM3D | PnP | 27.6 dB | 0.79 |  |
| Instrument-CNN | Deep Learning | 29.13 dB | 0.903 |  |
| PnP-NLM | PnP | 29.24 dB | 0.904 |  |
| ResNet-Calib | Deep Learning | 31.3 dB | 0.892 |  |
| DiffusionInstrumentation | Diffusion | 31.55 dB | 0.938 |  |
| ScoreInstrumentation | Score-based | 31.96 dB | 0.942 |  |
| CalibFormer | Vision Transformer | 32.8 dB | 0.92 |  |
| MassSpecFormer | Vision Transformer | 33.19 dB | 0.954 |  |

## OCT Angiography (OCTA) (`octa`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FFT-OCT | Classical | 25.6 dB | 0.72 |  |
| Speckle-Lee | Classical | 27.85 dB | 0.79 |  |
| TV-Denoising | Classical | 28.5 dB | 0.815 |  |
| BM4D | PnP | 29.3 dB | 0.85 |  |
| NLM-OCT | PnP | 30.2 dB | 0.87 |  |
| Speckle-DenoiseNet | Deep Learning | 33.1 dB | 0.925 | done |
| U-Net-OCT | Deep Learning | 33.85 dB | 0.935 | done |
| OCTA-Net | Deep Learning | 34.6 dB | 0.942 | done |
| OCT-ViT | Vision Transformer | 36.12 dB | 0.958 | done |
| RetinalFormer | Transformer | 36.35 dB | 0.96 | done |
| SpeckleFormer | Vision Transformer | 36.85 dB | 0.964 | done |
| DiffusionOCT | Diffusion | 37.52 dB | 0.97 |  |
| ScoreOCT | Score-based | 37.95 dB | 0.973 |  |

## Ocean Acoustic Tomography (`ocean_acoustic_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov | Classical | 25.4 dB | 0.71 |  |
| Wiener Filter | Classical | 25.62 dB | 0.821 |  |
| Matched Filter | Classical | 25.7 dB | 0.823 |  |
| PnP-RED | PnP | 28.9 dB | 0.835 |  |
| PnP-ADMM | PnP | 29.4 dB | 0.907 |  |
| ExpFormer | Vision Transformer | 31.88 dB | 0.941 | done |
| ResUNet | Deep Learning | 32.6 dB | 0.915 | done |
| Domain-Adapted-CNN | Deep Learning | 33.58 dB | 0.958 | done |
| ScoreExperimental | Score-based | 33.76 dB | 0.959 | done |
| SwinIR | Vision Transformer | 34.1 dB | 0.942 | done |
| DiffusionExperimental | Diffusion | 34.23 dB | 0.963 | done |

## Ocean Color Remote Sensing (`ocean_color`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gordon AC | Classical | 22.5 dB | 0.61 | done |
| MUMM | Classical | 26.0 dB | 0.74 |  |
| OC-Net | Deep Learning | 30.5 dB | 0.87 | done |
| AquaFormer | Transformer | 32.5 dB | 0.91 | done |

## Optical Coherence Tomography (OCT) (`oct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FFT-OCT | Classical | 25.6 dB | 0.72 | done |
| Speckle-Lee | Classical | 27.85 dB | 0.79 | |
| TV-Denoising | Classical | 28.5 dB | 0.815 | |
| BM4D | PnP | 29.3 dB | 0.85 | |
| NLM-OCT | PnP | 30.2 dB | 0.87 | |
| Speckle-DenoiseNet | Deep Learning | 33.1 dB | 0.925 | done |
| U-Net-OCT | Deep Learning | 33.85 dB | 0.935 | done |
| OCTA-Net | Deep Learning | 34.6 dB | 0.942 | done |
| OCT-ViT | Vision Transformer | 36.12 dB | 0.958 | done |
| RetinalFormer | Transformer | 36.35 dB | 0.96 | done |
| SpeckleFormer | Vision Transformer | 36.85 dB | 0.964 | done |
| DiffusionOCT | Diffusion | 37.52 dB | 0.97 | |
| ScoreOCT | Score-based | 37.95 dB | 0.973 | |

## Optical Diffraction Tomography (ODT) (`odt`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wolf FBP | Classical | 24.5 dB | 0.69 |  |
| Born-ADMM | PnP | 28.0 dB | 0.81 |  |
| ODT-Net | Deep Learning | 32.0 dB | 0.905 |  |
| Rytov-Former | Transformer | 34.0 dB | 0.935 |  |

## PALM/STORM Single-Molecule Localization (`palm_storm`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| ThunderSTORM | Classical | 22.5 dB | 0.61 |  |
| FALCON | PnP | 25.8 dB | 0.74 |  |
| Deep-STORM | Deep Learning | 30.2 dB | 0.88 |  |
| DECODE | Deep Learning | 32.1 dB | 0.915 |  |

## PET/CT Fusion (`pet_ct`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLAA | Classical | 25.6 dB | 0.72 |  |
| Image Registration | Classical | 27.38 dB | 0.867 |  |
| Guided Reconstruction | PnP | 28.8 dB | 0.897 |  |
| MR-Guided | PnP | 29.2 dB | 0.848 |  |
| MultiModal-Fusion-Former | Vision Transformer | 31.83 dB | 0.941 |  |
| ScoreFusion | Score-based | 32.11 dB | 0.944 |  |
| Fusion-U-Net | Deep Learning | 32.8 dB | 0.951 |  |
| FBSEM-Net | Deep Learning | 32.9 dB | 0.92 |  |
| DiffusionFusion | Diffusion | 33.9 dB | 0.96 |  |
| PPMF-Net | Vision Transformer | 34.3 dB | 0.945 |  |
| CrossModal-ViT | Vision Transformer | 34.4 dB | 0.964 |  |

## PET/MR Fusion (`pet_mr`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLAA | Classical | 25.6 dB | 0.72 | done |
| Image Registration | Classical | 27.64 dB | 0.873 |  |
| Guided Reconstruction | PnP | 28.62 dB | 0.893 |  |
| MR-Guided | PnP | 29.2 dB | 0.848 |  |
| MultiModal-Fusion-Former | Vision Transformer | 31.74 dB | 0.94 |  |
| Fusion-U-Net | Deep Learning | 32.88 dB | 0.951 |  |
| FBSEM-Net | Deep Learning | 32.9 dB | 0.92 |  |
| PPMF-Net | Vision Transformer | 34.3 dB | 0.945 |  |
| DiffusionFusion | Diffusion | 34.4 dB | 0.964 |  |
| ScoreFusion | Score-based | 34.9 dB | 0.967 |  |
| CrossModal-ViT | Vision Transformer | 35.65 dB | 0.972 |  |

## Panorama Multi-Focus Fusion (`panorama`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SIFT-RANSAC | Classical | 26.0 dB | 0.74 |  |
| APAP | Classical | 29.5 dB | 0.85 |  |
| UDIS | Deep Learning | 33.0 dB | 0.92 | done |
| PanoFormer | Transformer | 35.0 dB | 0.95 | done |

## Particle Calorimetry (`particle_calorimetry`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PandoraPFA | Classical | 22.0 dB | 0.58 | done |
| GARFIELD++ | Classical | 25.5 dB | 0.72 | done |
| GravNet | Deep Learning | 29.5 dB | 0.86 | done |
| CaloDiffusion | Diffusion | 31.5 dB | 0.9 | done |

## Passive Microwave Radiometry (`passive_microwave`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov-SMOS | Classical | 22.27 dB | 0.701 | done |
| Backus-Gilbert | Classical | 23.6 dB | 0.754 | done |
| MWR-Former | Transformer | 30.78 dB | 0.928 | done |
| RadioNet | Deep Learning | 31.81 dB | 0.941 | done |

## Phase Contrast Microscopy (`phase_contrast`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TIE Solver | Classical | 25.5 dB | 0.72 | done |
| DPC-ADMM | PnP | 29.0 dB | 0.84 | done |
| QPI-Net | Deep Learning | 33.0 dB | 0.92 |  |
| PhaseFormer | Transformer | 35.0 dB | 0.945 |  |

## Photoacoustic Imaging (`photoacoustic`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Universal Back-Proj | Classical | 23.5 dB | 0.64 |  |
| PnP-ADMM | PnP | 27.0 dB | 0.79 |  |
| Deep-PAI | Deep Learning | 31.5 dB | 0.89 | done |
| PAT-Former | Transformer | 33.5 dB | 0.92 | done |

## Photometric Stereo (`photometric_stereo`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| LS Normal Est. | Classical | 25.0 dB | 0.7 |  |
| Robust PCA | Classical | 28.5 dB | 0.82 |  |
| CNN-PS | Deep Learning | 32.5 dB | 0.915 | done |
| PS-Transformer | Transformer | 34.2 dB | 0.945 | done |

## Photon-Counting Spectral CT (`spectral_ct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 27.38 dB | 0.79 |  |
| TV-ADMM | Classical | 30.15 dB | 0.862 |  |
| PnP-ADMM | PnP | 32.64 dB | 0.891 |  |
| PnP-DnCNN | PnP | 33.45 dB | 0.905 |  |
| RED-CNN | Deep Learning | 33.56 dB | 0.908 |  |
| FBPConvNet | Deep Learning | 35.81 dB | 0.939 |  |
| Learned Primal-Dual | Deep Unrolling | 36.42 dB | 0.947 |  |
| DuDoTrans | Deep Unrolling | 37.68 dB | 0.962 |  |
| DOLCE | Diffusion | 38.32 dB | 0.971 |  |
| CT-ViT | Vision Transformer | 39.15 dB | 0.978 |  |
| CTFormer | Transformer | 39.45 dB | 0.98 |  |
| DiffusionCT | Diffusion | 39.68 dB | 0.982 |  |
| Score-CT | Score-based | 39.92 dB | 0.984 |  |

## Polarimetric SAR (PolSAR) (`polsar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Matched Filter | Classical | 23.5 dB | 0.64 |  |
| Chirp Scaling | Classical | 24.82 dB | 0.796 |  |
| Lee Filter | PnP | 25.62 dB | 0.821 |  |
| Range-Doppler | Classical | 25.87 dB | 0.828 |  |
| SAR-BM3D | PnP | 27.2 dB | 0.79 |  |
| SAR-DRN | Deep Learning | 30.6 dB | 0.882 | done |
| SAR-ResNet | Deep Learning | 30.88 dB | 0.929 | done |
| SAR-CAM | Transformer | 32.1 dB | 0.912 | done |
| SARDenoiserViT | Vision Transformer | 32.31 dB | 0.946 | done |
| ScoreSAR | Score-based | 32.44 dB | 0.947 | done |
| SARFormer | Vision Transformer | 33.85 dB | 0.932 | done |
| PanSharpener++ | Deep Learning | 34.58 dB | 0.945 | done |
| DiffusionSAR | Diffusion | 35.42 dB | 0.955 | done |

## Polarization Microscopy (`polarization`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 |  |
| Wiener Filter | Classical | 28.35 dB | 0.805 |  |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 |  |
| PnP-FISTA | PnP | 30.42 dB | 0.872 |  |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 |  |
| CARE | Deep Learning | 34.5 dB | 0.948 | done |
| U-Net | Deep Learning | 35.15 dB | 0.956 | done |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | done |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | done |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 |  |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 |  |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 |  |
| ScoreMicro | Score-based | 38.48 dB | 0.981 |  |

## Portal Imaging (EPID) (`portal_imaging`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 27.38 dB | 0.79 |  |
| TV-ADMM | Classical | 30.15 dB | 0.862 |  |
| PnP-ADMM | PnP | 32.64 dB | 0.891 |  |
| PnP-DnCNN | PnP | 33.45 dB | 0.905 |  |
| RED-CNN | Deep Learning | 33.56 dB | 0.908 |  |
| FBPConvNet | Deep Learning | 35.81 dB | 0.939 |  |
| Learned Primal-Dual | Deep Unrolling | 36.42 dB | 0.947 |  |
| DuDoTrans | Deep Unrolling | 37.68 dB | 0.962 |  |
| DOLCE | Diffusion | 38.32 dB | 0.971 |  |
| CT-ViT | Vision Transformer | 39.15 dB | 0.978 |  |
| CTFormer | Transformer | 39.45 dB | 0.98 |  |
| DiffusionCT | Diffusion | 39.68 dB | 0.982 |  |
| Score-CT | Score-based | 39.92 dB | 0.984 |  |

## Positron Emission Tomography (PET) (`pet`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| OSEM | Classical | 24.8 dB | 0.69 | done |
| FBP-PET | Classical | 26.65 dB | 0.849 | done |
| ML-EM | Classical | 28.41 dB | 0.889 | done |
| MAPEM-RDP | PnP | 28.5 dB | 0.815 | done |
| OS-EM | Classical | 28.92 dB | 0.899 | done |
| DeepPET | Deep Learning | 32.4 dB | 0.918 | |
| TransEM | Transformer | 33.7 dB | 0.938 | |
| PETFormer | Vision Transformer | 35.7 dB | 0.972 | |
| PET-ViT | Vision Transformer | 36.38 dB | 0.975 | |
| U-Net-PET | Deep Learning | 36.8 dB | 0.977 | |

## Proton Radiography (`proton_radiography`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-MLP | Classical | 23.5 dB | 0.65 |  |
| DROP-TVS | PnP | 27.0 dB | 0.79 |  |
| ProtonNet | Deep Learning | 31.0 dB | 0.89 |  |
| pCT-Former | Transformer | 33.0 dB | 0.92 |  |

## Proton Therapy Imaging (`proton_therapy_img`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 27.38 dB | 0.79 |  |
| TV-ADMM | Classical | 30.15 dB | 0.862 |  |
| PnP-ADMM | PnP | 32.64 dB | 0.891 |  |
| PnP-DnCNN | PnP | 33.45 dB | 0.905 |  |
| RED-CNN | Deep Learning | 33.56 dB | 0.908 |  |
| FBPConvNet | Deep Learning | 35.81 dB | 0.939 |  |
| Learned Primal-Dual | Deep Unrolling | 36.42 dB | 0.947 |  |
| DuDoTrans | Deep Unrolling | 37.68 dB | 0.962 |  |
| DOLCE | Diffusion | 38.32 dB | 0.971 |  |
| CT-ViT | Vision Transformer | 39.15 dB | 0.978 |  |
| CTFormer | Transformer | 39.45 dB | 0.98 |  |
| DiffusionCT | Diffusion | 39.68 dB | 0.982 |  |
| Score-CT | Score-based | 39.92 dB | 0.984 |  |

## Ptychographic Imaging (`ptychography`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| ePIE | Classical | 25.0 dB | 0.71 |  |
| sDR | Classical | 28.5 dB | 0.82 |  |
| PtychoNN | Deep Learning | 32.5 dB | 0.91 |  |
| AutoPhaseNN | Deep Learning | 34.0 dB | 0.935 |  |

## Pump-Probe Microscopy (`pump_probe`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SVD-GlobFit | Classical | 22.5 dB | 0.6 |  |
| MCR-ALS | Classical | 26.0 dB | 0.74 |  |
| TAS-Net | Deep Learning | 30.0 dB | 0.87 | done |
| DynFormer | Transformer | 32.0 dB | 0.905 | done |

## Quantum Illumination (`quantum_illumination`) — quantum

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| OPA Receiver | Classical | 18.0 dB | 0.42 | done |
| FF-SFG | Classical | 22.0 dB | 0.6 | done |
| QI-Net | Deep Learning | 26.5 dB | 0.78 | done |
| QuantumFormer | Transformer | 28.5 dB | 0.84 | done |

## Radio Aperture Synthesis (`radio_astronomy`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CLEAN | Classical | 22.5 dB | 0.6 |  |
| AIRI | PnP | 26.3 dB | 0.77 |  |
| R2D2 | Deep Learning | 29.8 dB | 0.875 |  |
| PRIMO | Deep Learning | 31.2 dB | 0.905 |  |

## Radio Interferometry (VLBI) (`radio_interferometry`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CLEAN | Classical | 22.5 dB | 0.6 |  |
| AIRI | PnP | 26.3 dB | 0.77 |  |
| R2D2 | Deep Learning | 29.8 dB | 0.875 |  |
| PRIMO | Deep Learning | 31.2 dB | 0.905 |  |

## Raman Imaging / Microscopy (`raman_imaging`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Baseline Correction | Classical | 24.29 dB | 0.779 |  |
| SG-ALS | Classical | 24.3 dB | 0.67 |  |
| SVD | Classical | 26.39 dB | 0.843 |  |
| PnP-DnCNN | PnP | 27.9 dB | 0.8 |  |
| SpectraFormer | Vision Transformer | 30.64 dB | 0.926 | done |
| ScoreSpectra | Score-based | 30.84 dB | 0.929 | done |
| DiffusionSpectra | Diffusion | 31.04 dB | 0.931 | done |
| CDAE | Deep Learning | 31.5 dB | 0.895 | done |
| U-Net-Spectra | Deep Learning | 31.86 dB | 0.941 | done |
| Cascade-UNet | Deep Learning | 33.0 dB | 0.922 | done |
| PINN-Spectra | Deep Learning | 33.54 dB | 0.957 | done |

## SPECT/CT Fusion (`spect_ct`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| OSEM | Classical | 24.8 dB | 0.69 |  |
| AC-OSEM | Classical | 27.62 dB | 0.873 |  |
| MAP-OSEM | PnP | 30.6 dB | 0.926 |  |
| DL-SPECT | Deep Learning | 33.72 dB | 0.959 |  |

## STED Microscopy (`sted`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | done |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | done |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | done |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## STEM-EDX Elemental Mapping (`edx_mapping`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLS-EDX | Classical | 22.3 dB | 0.708 | done |
| TV-EDX | Variational | 24.9 dB | 0.751 | done |
| NMF-EDX | Statistical | 27.5 dB | 0.792 | done |
| DnCNN-EDX | Deep Learning | 30.3 dB | 0.843 | done |
| N2V-EDX | Self-Supervised | 32.8 dB | 0.878 | done |
| TransEDX | Transformer | 35.2 dB | 0.916 | done |
| SwinEDX | Transformer | 36.8 dB | 0.933 | done |
| PhysEDX | Physics-Informed | 37.9 dB | 0.943 |  |
| DiffEDX | Diffusion Model | 39.4 dB | 0.955 |  |

## Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SAFT | Classical | 21.5 dB | 0.6 | done |
| Wiener Deconv | Classical | 23.0 dB | 0.65 | done |
| PnP-ADMM | PnP | 26.5 dB | 0.77 | done |
| SAM-Net | Deep Learning | 29.5 dB | 0.86 | done |
| Self-Sup Deconv | Self-Supervised | 31.0 dB | 0.89 | done |
| PINN-SAM | Physics-Informed | 32.5 dB | 0.915 | done |
| AcousticFormer | Transformer | 34.0 dB | 0.935 | done |
| DiffusionSAM | Diffusion | 35.0 dB | 0.948 | done |

## Scanning Electron Microscopy (SEM) (`sem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener Filter | Classical | 24.8 dB | 0.68 | done |
| BM3D | PnP | 28.5 dB | 0.82 | done |
| Noise2Void | Deep Learning | 31.6 dB | 0.895 | done |
| SwinIR | Transformer | 33.4 dB | 0.93 | done |

## Scanning Transmission Electron Microscopy (STEM) (`stem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener Filter | Classical | 24.8 dB | 0.68 | done |
| BM3D | PnP | 28.5 dB | 0.82 | done |
| Noise2Void | Deep Learning | 31.6 dB | 0.895 | done |
| SwinIR | Transformer | 33.4 dB | 0.93 | done |

## Scanning Tunneling Microscopy (STM) (`stm`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BTR | Classical | 23.2 dB | 0.63 | done |
| MLE Reconstruction | Classical | 23.85 dB | 0.763 | done |
| Reg-Deconv | PnP | 26.8 dB | 0.77 | done |
| TV-Deconvolution | PnP | 28.03 dB | 0.881 | done |
| U-Net-SPM | Deep Learning | 28.13 dB | 0.883 | done |
| DiffusionSPM | Diffusion | 29.12 dB | 0.902 | done |
| DeepSPM | Deep Learning | 30.4 dB | 0.88 | done |
| ScoreSPM | Score-based | 31.21 dB | 0.934 | done |
| E2E-BTR | Deep Learning | 31.8 dB | 0.908 | done |
| SPM-Former | Vision Transformer | 33.25 dB | 0.955 | done |

## Second Harmonic Generation (SHG) Microscopy (`shg`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 |  |
| Wiener Filter | Classical | 28.35 dB | 0.805 |  |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 |  |
| PnP-FISTA | PnP | 30.42 dB | 0.872 |  |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 |  |
| CARE | Deep Learning | 34.5 dB | 0.948 |  |
| U-Net | Deep Learning | 35.15 dB | 0.956 |  |
| Restormer | Vision Transformer | 35.8 dB | 0.962 |  |
| ResUNet | Deep Learning | 35.85 dB | 0.964 |  |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 |  |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 |  |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 |  |
| ScoreMicro | Score-based | 38.48 dB | 0.981 |  |

## Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SVD | Classical | 23.94 dB | 0.766 | done |
| SG-ALS | Classical | 24.3 dB | 0.67 | done |
| Baseline Correction | Classical | 25.01 dB | 0.803 | done |
| PnP-DnCNN | PnP | 27.9 dB | 0.8 | done |
| DiffusionSpectra | Diffusion | 30.68 dB | 0.927 | done |
| SpectraFormer | Vision Transformer | 30.76 dB | 0.928 | done |
| CDAE | Deep Learning | 31.5 dB | 0.895 | done |
| PINN-Spectra | Deep Learning | 31.76 dB | 0.94 | done |
| ScoreSpectra | Score-based | 32.48 dB | 0.948 | done |
| Cascade-UNet | Deep Learning | 33.0 dB | 0.922 | done |
| U-Net-Spectra | Deep Learning | 33.16 dB | 0.954 | done |

## Seismic Tomography (`seismic_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov | Classical | 25.4 dB | 0.71 |  |
| Wiener Filter | Classical | 26.5 dB | 0.846 |  |
| Matched Filter | Classical | 26.62 dB | 0.849 |  |
| PnP-ADMM | PnP | 27.86 dB | 0.878 |  |
| PnP-RED | PnP | 28.9 dB | 0.835 |  |
| ResUNet | Deep Learning | 32.6 dB | 0.915 |  |
| ExpFormer | Vision Transformer | 32.6 dB | 0.949 |  |
| ScoreExperimental | Score-based | 33.42 dB | 0.956 |  |
| DiffusionExperimental | Diffusion | 33.51 dB | 0.957 |  |
| Domain-Adapted-CNN | Deep Learning | 33.73 dB | 0.959 |  |
| SwinIR | Vision Transformer | 34.1 dB | 0.942 |  |

## Shear-Wave Elastography (`elastography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| LFE-Elasto | Classical | 22.3 dB | 0.71 | done |
| DI-Elasto | Variational | 24.8 dB | 0.752 | done |
| AIDE | Variational | 26.9 dB | 0.787 | done |
| DnCNN-Elasto | Deep Learning | 29.7 dB | 0.838 | done |
| ElastoNet | Deep Unrolling | 32.5 dB | 0.876 | done |
| TransElasto | Transformer | 35.0 dB | 0.915 | done |
| SwinElasto | Transformer | 36.6 dB | 0.932 | done |
| PhysElasto | Physics-Informed | 37.8 dB | 0.942 |  |
| DiffElasto | Diffusion Model | 39.2 dB | 0.953 |  |

## Shearography (`shearography`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Goldstein MCF | Classical | 24.0 dB | 0.67 |  |
| PnP-Phase | PnP | 28.0 dB | 0.8 |  |
| ShearNet | Deep Learning | 32.0 dB | 0.9 |  |
| PhaseFormer | Transformer | 34.0 dB | 0.935 |  |

## Single Photon Emission CT (SPECT) (`spect`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| OSEM | Classical | 24.8 dB | 0.69 | done |
| OS-EM | Classical | 27.96 dB | 0.88 | done |
| MAPEM-RDP | PnP | 28.5 dB | 0.815 | done |
| ML-EM | Classical | 29.4 dB | 0.907 | done |
| FBP-PET | Classical | 30.1 dB | 0.918 | done |
| DeepPET | Deep Learning | 32.4 dB | 0.918 | |
| TransEM | Transformer | 33.7 dB | 0.938 | |
| U-Net-PET | Deep Learning | 33.86 dB | 0.96 | |
| PETFormer | Vision Transformer | 37.9 dB | 0.982 | |
| PET-ViT | Vision Transformer | 38.08 dB | 0.982 | |

## Single-Pixel Camera (SPC) (`spc`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| GAP-TV | Classical | 26.83 dB | 0.754 | done |
| FISTA-TV | Classical | 28.42 dB | 0.821 | done |
| TVAL3 | Classical | 29.15 dB | 0.845 | done |
| PnP-FFDNet | PnP | 29.65 dB | 0.852 | done |
| EfficientSCI | Deep Learning | 34.21 dB | 0.949 |  |
| MST-L | Transformer | 35.4 dB | 0.96 |  |
| Restormer | Vision Transformer | 35.68 dB | 0.962 |  |
| CST | Transformer | 35.92 dB | 0.965 |  |
| HiSViT+ | Vision Transformer | 36.85 dB | 0.971 |  |
| CSTrans | Transformer | 37.12 dB | 0.973 |  |
| PromptSCI | Deep Learning | 37.35 dB | 0.975 |  |
| DiffusionHSI | Diffusion | 37.95 dB | 0.978 |  |
| ScoreSCI | Diffusion | 38.22 dB | 0.98 |  |
| FlowHSI | Generative | 38.58 dB | 0.982 |  |

## Small-Angle X-ray Scattering (SAXS) (`saxs`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PyFAI-Integrate | Classical | 24.0 dB | 0.67 |  |
| McSAS | Classical | 27.5 dB | 0.79 |  |
| ScatterNet | Deep Learning | 31.5 dB | 0.895 |  |
| ScatterFormer | Transformer | 33.5 dB | 0.925 |  |

## Solar EUV/X-ray Imaging (`solar_imaging`) — astronomy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 24.99 dB | 0.802 | done |
| Pixon | PnP | 27.83 dB | 0.877 | done |
| SolarFormer | Transformer | 29.88 dB | 0.915 | done |
| DeepEM | Deep Learning | 29.95 dB | 0.916 | done |

## Sonar Imaging (`sonar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DAS | Classical | 22.23 dB | 0.7 |  |
| MVDR/Capon | Classical | 23.65 dB | 0.756 |  |
| SonarNet | Deep Learning | 28.37 dB | 0.888 | done |
| AcousticFormer | Transformer | 32.91 dB | 0.952 | done |

## Spinning Disk Confocal Microscopy (`spinning_disk`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | done |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | done |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## Stellar Coronagraphy (`coronagraphy`) — astronomy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| ADI | Classical | 22.5 dB | 0.721 | done |
| LOCI | Classical | 24.8 dB | 0.762 | done |
| PCA-ADI | Classical | 26.2 dB | 0.791 | done |
| KLIP | Classical | 27.5 dB | 0.815 | done |
| ANDROMEDA | Statistical | 28.8 dB | 0.838 | done |
| CNN-Coronagraph | Deep Learning | 32.1 dB | 0.878 | done |
| SpeckleLearn | Deep Learning | 34.5 dB | 0.91 | done |
| CoronFormer | Transformer | 36.8 dB | 0.935 | done |
| DiffusionCoron | Diffusion | 38.9 dB | 0.955 |  |

## Stimulated Raman Scattering (SRS) Microscopy (`srs`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical | 24.3 dB | 0.67 |  |
| SVD | Classical | 24.99 dB | 0.802 |  |
| Baseline Correction | Classical | 25.07 dB | 0.804 |  |
| PnP-DnCNN | PnP | 27.9 dB | 0.8 |  |
| U-Net-Spectra | Deep Learning | 29.37 dB | 0.907 | done |
| ScoreSpectra | Score-based | 30.2 dB | 0.92 | done |
| DiffusionSpectra | Diffusion | 30.37 dB | 0.922 | done |
| SpectraFormer | Vision Transformer | 30.48 dB | 0.924 | done |
| CDAE | Deep Learning | 31.5 dB | 0.895 | done |
| PINN-Spectra | Deep Learning | 32.88 dB | 0.951 | done |
| Cascade-UNet | Deep Learning | 33.0 dB | 0.922 | done |

## Streak Camera Imaging (`streak_camera`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TwIST | Classical | 24.6 dB | 0.68 |  |
| PnP-ADMM | PnP | 26.91 dB | 0.856 |  |
| Temporal Filtering | Classical | 26.94 dB | 0.857 |  |
| PnP-FFDNet | PnP | 28.3 dB | 0.82 |  |
| ScoreUltrafast | Score-based | 29.43 dB | 0.908 | done |
| Unfolded-CUP | Deep Unrolling | 29.78 dB | 0.913 | done |
| DiffusionUltrafast | Diffusion | 30.14 dB | 0.919 | done |
| CUP-Net | Deep Learning | 31.9 dB | 0.9 | done |
| Temporal-U-Net | Deep Learning | 32.94 dB | 0.952 | done |
| AL-DL | Deep Unrolling | 33.4 dB | 0.93 | done |
| UltraFormer | Vision Transformer | 34.63 dB | 0.965 | done |

## Structured Illumination Microscopy (SIM) (`sim`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-SIM | Classical | 28.5 dB | 0.82 |  |
| PnP-SIM | PnP | 31.5 dB | 0.89 |  |
| DL-SIM | Deep Learning | 35.0 dB | 0.945 |  |
| SIMformer | Transformer | 36.5 dB | 0.96 |  |

## Structured-Light Depth Camera (`structured_light`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phase Shifting | Classical | 24.87 dB | 0.798 |  |
| Gray Code | Classical | 25.72 dB | 0.824 |  |
| FPP-Net | Deep Learning | 33.13 dB | 0.954 | done |
| PhaseFormer | Transformer | 34.25 dB | 0.963 | done |

## Susceptibility-Weighted Imaging (SWI) (`swi`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 26.0 dB | 0.62 |  |
| SENSE | Classical | 29.5 dB | 0.83 |  |
| GRAPPA | Classical | 31.2 dB | 0.86 |  |
| Score-MRI | Score-Based | 31.4 dB | 0.89 |  |
| L1-Wavelet | Compressed Sensing | 32.1 dB | 0.87 |  |
| k-t SPARSE-SENSE | Compressed Sensing | 32.5 dB | 0.875 |  |
| ESPIRiT | Compressed Sensing | 33.4 dB | 0.89 |  |
| LORAKS | Compressed Sensing | 33.8 dB | 0.893 |  |
| BM3D-MRI | PnP | 34.2 dB | 0.897 |  |
| ALOHA | Low-Rank | 34.5 dB | 0.9 |  |
| PnP-DnCNN | PnP | 35.0 dB | 0.905 |  |
| Deep-ADMM-Net | Deep Unrolling | 35.3 dB | 0.907 |  |
| DCCNN | Deep Learning | 35.5 dB | 0.908 |  |
| U-Net | Deep Learning | 35.9 dB | 0.904 |  |
| MoDL | Deep Unrolling | 36.5 dB | 0.912 |  |
| HybridCascade | Deep Unrolling | 37.8 dB | 0.917 |  |
| SwinMR | Transformer | 38.5 dB | 0.921 |  |
| HUMUS-Net | Transformer | 38.9 dB | 0.923 |  |
| ReconFormer | Transformer | 39.0 dB | 0.922 |  |
| E2E-VarNet | Deep Unrolling | 39.4 dB | 0.924 |  |
| PromptMR | Deep Unrolling | 39.7 dB | 0.926 |  |
| MRI-DiffusionNet | Diffusion | 40.1 dB | 0.932 |  |
| MRDynamo | Physics-Informed | 40.5 dB | 0.938 |  |
| MMR-Mamba | Physics-Informed | 40.98 dB | 0.969 |  |
| PnP-DnCNN-Pro | PnP | 41.0 dB | 0.968 |  |
| BrainID-MRI | Foundation Model | 41.0 dB | 0.942 |  |
| PromptMR-SFM | Physics-Informed | 41.3 dB | 0.971 |  |
| U-Net++ | Deep Learning | 41.5 dB | 0.978 |  |
| ReconFormer++ | Transformer | 41.5 dB | 0.969 |  |
| MoDL-Net++ | Deep Unrolling | 41.8 dB | 0.978 |  |
| MRI-FM | Foundation Model | 42.1 dB | 0.948 |  |
| MR-IPT | Foundation Model | 42.48 dB | 0.983 |  |
| HybridCascade++ | Deep Unrolling | 42.5 dB | 0.981 |  |
| HUMUS-Net++ | Transformer | 43.1 dB | 0.979 |  |
| SwinMR++ | Transformer | 43.8 dB | 0.983 |  |

## Synthetic Aperture Radar (SAR) (`sar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Chirp Scaling | Classical | 22.68 dB | 0.718 |  |
| Matched Filter | Classical | 23.5 dB | 0.64 |  |
| Range-Doppler | Classical | 25.9 dB | 0.829 |  |
| SAR-BM3D | PnP | 27.2 dB | 0.79 |  |
| Lee Filter | PnP | 28.75 dB | 0.896 |  |
| SAR-ResNet | Deep Learning | 28.84 dB | 0.897 | done |
| SARDenoiserViT | Vision Transformer | 30.2 dB | 0.92 | done |
| SAR-DRN | Deep Learning | 30.6 dB | 0.882 | done |
| ScoreSAR | Score-based | 31.9 dB | 0.942 | done |
| SAR-CAM | Transformer | 32.1 dB | 0.912 | done |
| SARFormer | Vision Transformer | 33.85 dB | 0.932 | done |
| PanSharpener++ | Deep Learning | 34.58 dB | 0.945 | done |
| DiffusionSAR | Diffusion | 35.42 dB | 0.955 | done |

## TIRF Microscopy (`tirf`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | done |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | done |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | done |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## Talbot-Lau X-ray Grating Interferometry (`talbot_lau`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phase Stepping | Classical | 23.57 dB | 0.753 |  |
| PCA Retrieval | Classical | 24.26 dB | 0.778 |  |
| DPC-Net | Deep Learning | 29.09 dB | 0.902 |  |
| GratingFormer | Transformer | 32.0 dB | 0.943 |  |

## Terahertz Imaging (THz) (`terahertz`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-THz | Classical | 24.5 dB | 0.68 | done |
| PnP-SPIRAL | PnP | 28.5 dB | 0.81 | done |
| THz-Net | Deep Learning | 32.5 dB | 0.905 |  |
| THz-Former | Transformer | 34.5 dB | 0.94 |  |

## Three-Photon Microscopy (`three_photon`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 |  |
| Wiener Filter | Classical | 28.35 dB | 0.805 |  |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 |  |
| PnP-FISTA | PnP | 30.42 dB | 0.872 |  |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 |  |
| CARE | Deep Learning | 34.5 dB | 0.948 |  |
| U-Net | Deep Learning | 35.15 dB | 0.956 |  |
| Restormer | Vision Transformer | 35.8 dB | 0.962 |  |
| ResUNet | Deep Learning | 35.85 dB | 0.964 |  |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 |  |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 |  |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 |  |
| ScoreMicro | Score-based | 38.48 dB | 0.981 |  |

## Time-of-Flight Depth Camera (`tof_camera`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phase Unwrap | Classical | 24.0 dB | 0.66 | done |
| PnP-ToF | PnP | 28.0 dB | 0.8 | done |
| DeepToF | Deep Learning | 32.5 dB | 0.9 | done |
| MPI-Former | Transformer | 34.0 dB | 0.93 | done |

## Transmission Electron Microscopy (TEM) (`tem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener Filter | Classical | 24.8 dB | 0.68 | done |
| BM3D | PnP | 28.5 dB | 0.82 | done |
| Noise2Void | Deep Learning | 31.6 dB | 0.895 | done |
| SwinIR | Transformer | 33.4 dB | 0.93 | done |

## Two-Photon / Multiphoton Microscopy (`two_photon`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## US/MRI Fusion (`us_mri`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| B-spline FFD | Classical | 25.08 dB | 0.805 |  |
| Demons | Classical | 25.16 dB | 0.807 |  |
| VoxelMorph | Deep Learning | 31.93 dB | 0.942 |  |
| TransMorph | Transformer | 34.11 dB | 0.962 |  |

## Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TFM | Classical | 25.0 dB | 0.71 |  |
| SAFT | Classical | 28.0 dB | 0.81 |  |
| UTPA-Net | Deep Learning | 32.5 dB | 0.905 | done |
| FMC-Former | Transformer | 34.5 dB | 0.94 | done |

## Ultrasound B-mode Imaging (`ultrasound`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DAS | Classical | 24.5 dB | 0.68 |  |
| DAS-CF | Classical | 25.8 dB | 0.72 |  |
| PW-DAS | Classical | 26.15 dB | 0.735 |  |
| PnP-ADMM | PnP | 28.12 dB | 0.81 |  |
| PnP-TV | PnP | 29.5 dB | 0.909 |  |
| ABLE | Deep Learning | 31.85 dB | 0.905 | done |
| MU-Net | Deep Learning | 33.2 dB | 0.928 | done |
| Phase-ADMM-Net | Deep Unrolling | 33.95 dB | 0.94 | done |
| UltrasoundFormer | Vision Transformer | 34.85 dB | 0.945 | done |
| BeamFormer | Transformer | 35.15 dB | 0.948 | done |
| BeamDATA | Deep Learning | 35.32 dB | 0.951 | done |
| AttentionBeam | Transformer | 35.52 dB | 0.952 | done |
| DiffUS | Diffusion | 35.95 dB | 0.958 | done |
| ScoreUS | Score-based | 36.28 dB | 0.962 | done |

## Weather / Doppler Radar (`weather_radar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Pulse-Pair Doppler | Classical | 24.0 dB | 0.67 | done |
| CLEAN-AP | Classical | 27.5 dB | 0.79 |  |
| RainNet | Deep Learning | 31.8 dB | 0.9 | done |
| Earthformer | Transformer | 33.5 dB | 0.935 | done |

## Wide-Angle X-ray Scattering (WAXS) (`waxs`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PyFAI-Integrate | Classical | 23.5 dB | 0.65 |  |
| Rietveld-WAXS | Classical | 27.0 dB | 0.78 |  |
| WAXS-Net | Deep Learning | 31.0 dB | 0.89 |  |
| CrystalFormer | Transformer | 33.0 dB | 0.92 |  |

## Widefield Fluorescence Microscopy (`widefield`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 27.1 dB | 0.77 | done |
| Wiener Filter | Classical | 28.35 dB | 0.805 | done |
| TV-Deconvolution | Classical | 29.5 dB | 0.845 | done |
| PnP-FISTA | PnP | 30.42 dB | 0.872 | done |
| PnP-DnCNN | PnP | 31.2 dB | 0.89 | |
| CARE | Deep Learning | 34.5 dB | 0.948 | |
| U-Net | Deep Learning | 35.15 dB | 0.956 | |
| Restormer | Vision Transformer | 35.8 dB | 0.962 | |
| ResUNet | Deep Learning | 35.85 dB | 0.964 | |
| DeconvFormer | Vision Transformer | 37.25 dB | 0.972 | |
| Restormer+ | Vision Transformer | 37.65 dB | 0.975 | |
| DiffDeconv | Diffusion | 38.12 dB | 0.979 | |
| ScoreMicro | Score-based | 38.48 dB | 0.981 | |

## X-ray Angiography (`angiography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FDK | Classical | 27.0 dB | 0.78 |  |
| TV-CS | Classical | 30.5 dB | 0.86 |  |
| PnP-ADMM | PnP | 32.0 dB | 0.893 |  |
| FBPConvNet | Deep Learning | 33.5 dB | 0.92 |  |
| Learned Primal-Dual | Deep Unrolling | 34.5 dB | 0.935 |  |
| VesselNet | Deep Learning | 35.2 dB | 0.948 |  |
| NeRF-Angio | Physics-Informed | 35.8 dB | 0.955 |  |
| AngioFormer | Transformer | 36.2 dB | 0.96 |  |
| DiffusionAngio | Diffusion | 36.8 dB | 0.967 |  |

## X-ray Computed Tomography (CT) (`ct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 25.2 dB | 0.771 |  |
| CGLS | Classical | 27.1 dB | 0.788 |  |
| OSEM | Classical | 27.5 dB | 0.795 |  |
| SART | Classical | 28.7 dB | 0.812 |  |
| ART-TV | Variational | 29.8 dB | 0.831 |  |
| TV-ADMM | Variational | 30.4 dB | 0.842 |  |
| BM3D-CT | PnP | 31.5 dB | 0.856 |  |
| DLCT | Dictionary Learning | 31.9 dB | 0.862 |  |
| PnP-ADMM | PnP | 32.3 dB | 0.868 |  |
| CT-U-Net | Deep Learning | 33.5 dB | 0.883 |  |
| WGAN-CT | Deep Learning | 33.9 dB | 0.887 |  |
| FBPConvNet | Deep Learning | 34.1 dB | 0.891 |  |
| RED-CNN | Deep Learning | 36.3 dB | 0.914 |  |
| LEARN | Deep Unrolling | 36.8 dB | 0.919 |  |
| iCT-Net | Deep Unrolling | 37.5 dB | 0.925 |  |
| DuDoRNet | Deep Unrolling | 38.5 dB | 0.931 |  |
| TransCT | Transformer | 39.8 dB | 0.942 |  |
| Eformer | Transformer | 40.3 dB | 0.948 |  |
| CTformer | Transformer | 41.2 dB | 0.954 |  |
| DiffusionMBIR | Diffusion | 42.5 dB | 0.963 |  |
| Score-CT | Score-Based | 42.8 dB | 0.965 |  |
| CT-MAE | Foundation Model | 43.2 dB | 0.968 |  |
| PINER-CT | Physics-Informed | 43.6 dB | 0.97 |  |
| CT-FM | Foundation Model | 44.1 dB | 0.974 |  |

## X-ray Crystallography (`xray_crystallography`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Molecular Replacement | Classical | 22.0 dB | 0.59 |  |
| SHELXD | Classical | 26.0 dB | 0.74 |  |
| DL-Phase | Deep Learning | 30.5 dB | 0.88 |  |
| CrystFormer | Transformer | 32.5 dB | 0.915 |  |

## X-ray Fluorescence (XRF) Imaging (`xrf_imaging`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FP-Quantify | Classical | 24.5 dB | 0.68 | done |
| PnP-BM3D | PnP | 28.0 dB | 0.8 | done |
| XRF-UNet | Deep Learning | 32.0 dB | 0.9 | done |
| SpectraFormer | Transformer | 34.0 dB | 0.935 | done |

## X-ray Fluorescence Tomography (`xrf_tomo`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv | Classical | 24.1 dB | 0.66 |  |
| Peak Fitting | Classical | 25.81 dB | 0.827 |  |
| Calibration-Lookup | Classical | 26.73 dB | 0.851 |  |
| PnP-NLM | PnP | 26.92 dB | 0.856 |  |
| PnP-BM3D | PnP | 27.6 dB | 0.79 |  |
| ScoreInstrumentation | Score-based | 29.05 dB | 0.901 |  |
| MassSpecFormer | Vision Transformer | 30.98 dB | 0.931 |  |
| ResNet-Calib | Deep Learning | 31.3 dB | 0.892 |  |
| Instrument-CNN | Deep Learning | 32.31 dB | 0.946 |  |
| CalibFormer | Vision Transformer | 32.8 dB | 0.92 |  |
| DiffusionInstrumentation | Diffusion | 33.26 dB | 0.955 |  |

## X-ray NDT (Radiography) (`xray_ndt`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 27.38 dB | 0.79 |  |
| PnP-ADMM | PnP | 32.64 dB | 0.891 |  |
| DR-GAN | Deep Learning | 34.07 dB | 0.961 |  |
| FBPConvNet | Deep Learning | 35.81 dB | 0.939 |  |

## X-ray Radiography (`xray_radiography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 27.38 dB | 0.79 |  |
| TV-ADMM | Classical | 30.15 dB | 0.862 |  |
| PnP-ADMM | PnP | 32.64 dB | 0.891 |  |
| PnP-DnCNN | PnP | 33.45 dB | 0.905 |  |
| RED-CNN | Deep Learning | 33.56 dB | 0.908 |  |
| FBPConvNet | Deep Learning | 35.81 dB | 0.939 |  |
| Learned Primal-Dual | Deep Unrolling | 36.42 dB | 0.947 |  |
| DuDoTrans | Deep Unrolling | 37.68 dB | 0.962 |  |
| DOLCE | Diffusion | 38.32 dB | 0.971 |  |
| CT-ViT | Vision Transformer | 39.15 dB | 0.978 |  |
| CTFormer | Transformer | 39.45 dB | 0.98 |  |
| DiffusionCT | Diffusion | 39.68 dB | 0.982 |  |
| Score-CT | Score-based | 39.92 dB | 0.984 |  |

## XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| EMC | Classical | 24.36 dB | 0.781 |  |
| CrystFEL | Classical | 24.38 dB | 0.782 |  |
| CrysFormer | Transformer | 31.56 dB | 0.938 |  |
| CNN Hit-Finder | Deep Learning | 32.92 dB | 0.952 |  |

