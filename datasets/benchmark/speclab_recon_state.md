# SpecLab Reconstruction State

Tracks verification status of all reconstruction algorithms in SpecLab
(`https://pwm.platformai.org/speclab`).

**Status:**
- `done` — PWM CPU reconstruction verified, actual PSNR within 2 dB of reference
- *(blank)* — awaiting verification (not yet tested, or PSNR below reference threshold)

Last updated: 2026-03-12 | Total modalities: 169

---

## Acoustic Emission Testing (AE) (`acoustic_emission`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Time-Reversal Imaging | Classical | 20.5 dB | 0.58 | done |
| TDOA-WLS | Classical | 22.0 dB | 0.63 | done |
| Sparse TR (L1) | Compressed Sensing | 25.5 dB | 0.73 | done |
| PnP-ADMM | PnP | 27.5 dB | 0.8 | done |
| AE-CNN | Deep Learning | 30.0 dB | 0.87 |  |
| Domain-Adapted ResNet | Deep Learning | 32.0 dB | 0.905 |  |
| PINN-AE | Physics-Informed | 33.5 dB | 0.925 |  |
| SwinIR-AE | Transformer | 34.8 dB | 0.94 |  |
| DiffusionAE | Diffusion | 35.5 dB | 0.95 |  |

## Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SAFT | Classical | 21.5 dB | 0.6 | done |
| Wiener Deconv | Classical | 23.0 dB | 0.65 | done |
| PnP-ADMM | PnP | 26.5 dB | 0.77 | done |
| SAM-Net | Deep Learning | 29.5 dB | 0.86 |  |
| Self-Sup Deconv | Self-Supervised | 31.0 dB | 0.89 |  |
| PINN-SAM | Physics-Informed | 32.5 dB | 0.915 |  |
| AcousticFormer | Transformer | 34.0 dB | 0.935 |  |
| DiffusionSAM | Diffusion | 35.0 dB | 0.948 |  |

## Active Thermography (IR) (`active_thermography`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TSR | Classical | 22.0 dB | 0.62 | done |
| PCT | Classical | 24.0 dB | 0.69 | done |
| PnP-ADMM | PnP | 27.0 dB | 0.79 | done |
| ThermoNet | Deep Learning | 30.0 dB | 0.87 |  |
| PINN-Thermo | Physics-Informed | 33.0 dB | 0.92 |  |
| U-Net Thermo | Deep Learning | 32.0 dB | 0.905 |  |
| ThermoFormer | Transformer | 34.5 dB | 0.938 |  |
| DiffusionThermo | Diffusion | 35.5 dB | 0.95 |  |

## Adaptive Optics (AO) Imaging (`adaptive_optics`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zernike LS | Classical | 22.0 dB | 0.64 | done |
| Fried Estimator | Classical | 24.0 dB | 0.7 | done |
| PnP-ADMM (WF) | PnP | 27.0 dB | 0.8 | done |
| WFNet | Deep Learning | 30.0 dB | 0.87 |  |
| LIFT-Net | Deep Learning | 31.5 dB | 0.895 |  |
| AO-Transformer | Transformer | 33.0 dB | 0.92 |  |
| AO-ViT | Transformer | 34.0 dB | 0.935 |  |
| DiffusionAO | Diffusion | 35.0 dB | 0.948 |  |

## Atomic Force Microscopy (AFM) (`afm`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Plane Fit | Classical | 20.0 dB | 0.56 | done |
| Wiener Deconv | Classical | 23.0 dB | 0.65 | done |
| PnP-ADMM | PnP | 26.5 dB | 0.77 | done |
| DeepAFM | Deep Learning | 30.0 dB | 0.87 |  |
| Self-Sup AFM | Self-Supervised | 31.5 dB | 0.895 |  |
| SPM-Former | Transformer | 33.0 dB | 0.92 |  |
| DiffusionAFM | Diffusion | 34.5 dB | 0.94 |  |

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

## Arterial Spin Labeling (ASL) MRI (`asl_mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 24.5 dB | 0.58 |  |
| L1-Wavelet (ESPIRiT) | Compressed Sensing | 28.3 dB | 0.82 |  |
| PnP-DnCNN | PnP | 29.8 dB | 0.843 |  |
| U-Net (ASL) | Deep Learning | 32.1 dB | 0.876 |  |
| E2E-VarNet | Deep Unrolling | 34.6 dB | 0.908 |  |
| Kinetic-CS | Physics-Informed | 33.2 dB | 0.891 |  |
| ReconFormer | Transformer | 35.4 dB | 0.922 |  |
| PromptMR | Deep Unrolling | 36.1 dB | 0.934 |  |
| Score-MRI (ASL) | Diffusion | 36.7 dB | 0.942 |  |

## Atom Probe Tomography (APT) (`atom_probe`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Bas-Protocol | Classical | 20.8 dB | 0.55 | done |
| Tikhonov-Trajectory | Classical | 23.4 dB | 0.66 | done |
| PnP-BM3D (APT) | PnP | 26.1 dB | 0.75 | done |
| ResNet-ArtefactCorr | Deep Learning | 28.7 dB | 0.818 |  |
| LISTA-APT | Deep Unrolling | 29.5 dB | 0.842 |  |
| TrajectoryPINN | Physics-Informed | 31.2 dB | 0.876 |  |
| APT-Former | Transformer | 33.6 dB | 0.912 |  |
| DiffusionAPT | Diffusion | 35.1 dB | 0.934 |  |
| EquivAPT | Vision Transformer | 36.3 dB | 0.948 |  |

## Bioluminescence Tomography (BLT) (`bioluminescence_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov-BLT | Classical | 19.5 dB | 0.54 | done |
| Tikhonov-PR | Classical | 22.8 dB | 0.64 | done |
| PnP-ADMM (BLT) | PnP | 25.6 dB | 0.73 | done |
| BLT-CNN | Deep Learning | 29.1 dB | 0.838 |  |
| LISTA-BLT | Deep Unrolling | 30.4 dB | 0.864 |  |
| DiffusionPINN-BLT | Physics-Informed | 32.9 dB | 0.902 |  |
| BLT-Former | Transformer | 34.8 dB | 0.929 |  |
| ScoreBLT | Diffusion | 36.5 dB | 0.952 |  |
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
| CNN-Spectra | Deep Learning | 31.5 dB | 0.872 |  |
| DnCNN-Brillouin | Deep Learning | 33.2 dB | 0.901 |  |
| CDAE | Deep Learning | 34.8 dB | 0.918 |  |
| U-Net-Spectral | Deep Learning | 36.1 dB | 0.933 |  |
| PINN-Brillouin | Physics-Informed | 37.0 dB | 0.942 |  |
| SpectraFormer | Transformer | 38.4 dB | 0.954 |  |
| DiffusionSpectra | Diffusion | 39.5 dB | 0.963 |  |

## CACTI (`cacti`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| GAP-TV | Variational |  |  | done |
| DeSCI | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| DGSMP | Deep Unrolling |  |  |  |
| GAP-CCoT | Transformer |  |  |  |
| STFormer | Transformer |  |  |  |
| EfficientSCI | Transformer |  |  |  |
| RDLUF-MixS2 | Deep Unrolling |  |  |  |
| DiffusionSCI | Diffusion |  |  |  |

## Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| KK-Retrieval | Classical | 24.5 dB | 0.762 | done |
| MEM-CARS | Classical | 26.2 dB | 0.798 | done |
| CNN-NRB | Deep Learning | 30.8 dB | 0.865 |  |
| U-Net-CARS | Deep Learning | 33.5 dB | 0.902 |  |
| PINN-CARS | Physics-Informed | 34.8 dB | 0.918 |  |
| ResNet-CARS | Deep Learning | 36.2 dB | 0.933 |  |
| SpecFormer-CARS | Transformer | 37.8 dB | 0.947 |  |
| Diff-CARS | Diffusion | 39.1 dB | 0.958 |  |
| FMDiff-CARS | Diffusion | 40.2 dB | 0.966 |  |

## Cathodoluminescence (CL) Imaging (`cathodoluminescence`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-CL | Classical | 25.2 dB | 0.771 | done |
| Richardson-Lucy | Classical | 27.5 dB | 0.812 |  |
| DnCNN-CL | Deep Learning | 31.8 dB | 0.875 |  |
| U-Net-CL | Deep Learning | 34.2 dB | 0.908 |  |
| CARE-CL | Deep Learning | 35.5 dB | 0.921 |  |
| SwinIR-CL | Transformer | 37.1 dB | 0.938 |  |
| PINN-CL | Physics-Informed | 36.8 dB | 0.934 |  |
| Restormer-CL | Transformer | 38.4 dB | 0.95 |  |
| DiffusionEM | Diffusion | 39.8 dB | 0.962 |  |

## CBCT (`cbct`) — medical

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

## CEST MRI (`cest_mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MTR-asym | Classical | 24.8 dB | 0.761 | done |
| Lorentzian-Fit | Classical | 27.2 dB | 0.808 | done |
| WASSR | Classical | 28.5 dB | 0.831 | done |
| DnCNN-CEST | Deep Learning | 32.1 dB | 0.878 |  |
| U-Net-CEST | Deep Learning | 34.8 dB | 0.912 |  |
| PINN-CEST | Physics-Informed | 35.9 dB | 0.925 |  |
| CESTFormer | Transformer | 37.4 dB | 0.94 |  |
| PromptCEST | Transformer | 38.6 dB | 0.951 |  |
| DiffusionCEST | Diffusion | 39.7 dB | 0.961 |  |

## Contrast-Enhanced Ultrasound (CEUS) (`ceus`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Pulse-Inversion | Classical | 24.1 dB | 0.751 | done |
| AM-CEUS | Classical | 25.8 dB | 0.781 | done |
| CNN-Bubble | Deep Learning | 30.2 dB | 0.858 |  |
| ULM-Net | Deep Learning | 33.5 dB | 0.9 |  |
| DeepULM | Deep Learning | 35.1 dB | 0.92 |  |
| PINN-CEUS | Physics-Informed | 36.4 dB | 0.934 |  |
| CEUSF-Transformer | Transformer | 37.8 dB | 0.946 |  |
| SUPER-ULM | Deep Unrolling | 38.5 dB | 0.953 |  |
| DiffusionCEUS | Diffusion | 39.6 dB | 0.962 |  |

## Correlative Light-Electron Microscopy (CLEM) (`clem`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Cross-Correlation | Classical | 23.5 dB | 0.741 | done |
| Landmark-Reg | Classical | 25.8 dB | 0.782 | done |
| CNN-Reg | Deep Learning | 30.2 dB | 0.855 |  |
| VoxelMorph | Deep Learning | 32.8 dB | 0.89 |  |
| CLEM-Net | Deep Learning | 34.5 dB | 0.912 |  |
| TransMorph | Transformer | 36.2 dB | 0.931 |  |
| PINN-CLEM | Physics-Informed | 35.8 dB | 0.927 |  |
| SwinCLEM | Transformer | 37.5 dB | 0.944 |  |
| DiffusionCLEM | Diffusion | 39.1 dB | 0.958 |  |

## Coded Exposure / Flutter Shutter (`coded_exposure`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-Deconv | Classical | 26.5 dB | 0.791 | done |
| TV-Deconv | Variational | 29.2 dB | 0.831 | done |
| BM3D-Deblur | PnP | 31.8 dB | 0.871 |  |
| DnCNN-Deblur | Deep Learning | 33.5 dB | 0.899 |  |
| DeblurGAN | Generative | 34.8 dB | 0.914 |  |
| DMPHN | Deep Learning | 36.1 dB | 0.928 |  |
| MPRNet | Deep Learning | 37.4 dB | 0.941 |  |
| Restormer-Deblur | Transformer | 38.6 dB | 0.951 |  |
| DiffusionDeblur | Diffusion | 39.8 dB | 0.961 |  |

## Confocal 3D (`confocal_3d`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 26.8 dB | 0.801 | done |
| Wiener-3D | Classical | 28.5 dB | 0.828 | done |
| IRCNN-Confocal | PnP | 32.1 dB | 0.878 | done |
| CARE | Deep Learning | 34.8 dB | 0.91 |  |
| Noise2Void | Self-Supervised | 33.5 dB | 0.895 |  |
| U-Net-3D | Deep Learning | 35.9 dB | 0.924 |  |
| SwinIR-3D | Transformer | 37.5 dB | 0.942 |  |
| Restormer-3D | Transformer | 38.6 dB | 0.951 |  |
| DiffusionMicro | Diffusion | 39.9 dB | 0.963 |  |

## Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| NLM-Speckle | Classical | 25.5 dB | 0.775 | done |
| BM3D-CLE | Classical | 27.8 dB | 0.815 | done |
| DnCNN-CLE | Deep Learning | 31.2 dB | 0.868 |  |
| U-Net-CLE | Deep Learning | 33.8 dB | 0.902 |  |
| CARE-CLE | Deep Learning | 35.2 dB | 0.92 |  |
| SwinIR-CLE | Transformer | 36.8 dB | 0.936 |  |
| PINN-CLE | Physics-Informed | 36.1 dB | 0.93 |  |
| Restormer-CLE | Transformer | 38.1 dB | 0.949 |  |
| DiffusionEndo | Diffusion | 39.4 dB | 0.96 |  |

## Confocal Live-Cell (`confocal_livecell`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| VST-Denoise | Classical | 24.2 dB | 0.751 | done |
| NLM-Fluorescence | Classical | 26.8 dB | 0.795 | done |
| CARE | Deep Learning | 33.5 dB | 0.891 |  |
| Noise2Void | Self-Supervised | 31.8 dB | 0.871 |  |
| Noise2Self | Self-Supervised | 30.5 dB | 0.858 |  |
| PN2V | Self-Supervised | 32.9 dB | 0.882 |  |
| SwinIR-LiveCell | Transformer | 36.2 dB | 0.931 |  |
| Restormer-Micro | Transformer | 37.8 dB | 0.946 |  |
| DiffusionCell | Diffusion | 39.2 dB | 0.959 |  |

## Stellar Coronagraphy (`coronagraphy`) — astronomy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| ADI | Classical | 22.5 dB | 0.721 | done |
| KLIP | Classical | 27.5 dB | 0.815 | done |
| LOCI | Classical | 24.8 dB | 0.762 | done |
| PCA-ADI | Classical | 26.2 dB | 0.791 | done |
| ANDROMEDA | Statistical | 28.8 dB | 0.838 |  |
| CNN-Coronagraph | Deep Learning | 32.1 dB | 0.878 |  |
| SpeckleLearn | Deep Learning | 34.5 dB | 0.91 |  |
| CoronFormer | Transformer | 36.8 dB | 0.935 |  |
| DiffusionCoron | Diffusion | 38.9 dB | 0.955 |  |

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
| WBP | Classical | 20.5 dB | 0.682 | done |
| SART-ET | Classical | 23.8 dB | 0.741 |  |
| IMOD | Classical | 25.2 dB | 0.774 |  |
| IsoNet | Deep Learning | 29.4 dB | 0.842 |  |
| DeepDeWedge | Self-Supervised | 31.7 dB | 0.876 |  |
| CryoSeg | Deep Learning | 33.1 dB | 0.898 |  |
| ETFormer | Transformer | 35.6 dB | 0.921 |  |
| DeePiCt | Deep Learning | 34.2 dB | 0.909 |  |
| DiffusionET | Diffusion | 37.9 dB | 0.944 |  |

## CT (`ct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical | 25.2 dB | 0.771 |  |
| SART | Classical | 28.7 dB | 0.812 |  |
| OSEM | Classical | 27.5 dB | 0.795 |  |
| ART-TV | Variational | 29.8 dB | 0.831 |  |
| TV-ADMM | Variational | 30.4 dB | 0.842 |  |
| CGLS | Classical | 27.1 dB | 0.788 |  |
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

## CT + Fluorescence (FLIT) (`ct_fluorescence`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-XRF | Classical | 22.8 dB | 0.701 | done |
| MLEM-XRF | Classical | 26.3 dB | 0.764 | done |
| TV-XRFCT | Variational | 29.7 dB | 0.831 | done |
| DnCNN-XRF | Deep Learning | 32.4 dB | 0.872 |  |
| U-Net-XRF | Deep Learning | 34.6 dB | 0.901 |  |
| PnP-XRF | PnP | 35.9 dB | 0.914 |  |
| SwinXRF | Transformer | 37.8 dB | 0.932 |  |
| PhysXRF-Net | Physics-Informed | 38.5 dB | 0.941 |  |
| DiffusionXRF | Diffusion | 40.1 dB | 0.955 |  |

## Compressed Ultrafast Photography (CUP) (`cup`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TV-CUP | Variational | 24.3 dB | 0.732 |  |
| TwIST-CUP | Variational | 26.8 dB | 0.774 |  |
| GAP-TV | Variational | 28.5 dB | 0.812 |  |
| DeSCI-CUP | PnP | 31.2 dB | 0.854 |  |
| E2E-CNN-CUP | Deep Learning | 33.7 dB | 0.886 |  |
| PnP-FastDVDnet | PnP | 35.4 dB | 0.911 |  |
| STFormer-CUP | Transformer | 37.9 dB | 0.933 |  |
| DAUHST-CUP | Transformer | 38.6 dB | 0.941 |  |
| DiffusionCUP | Diffusion | 40.2 dB | 0.956 |  |

## Dark-Field Microscopy (`dark_field`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 24.5 dB | 0.744 | done |
| Wiener-DF | Classical | 27.2 dB | 0.793 | done |
| TV-DF | Variational | 29.8 dB | 0.836 | done |
| BM3D-DF | Classical | 32.4 dB | 0.871 |  |
| CARE-DF | Deep Learning | 35.1 dB | 0.908 |  |
| Noise2Void-DF | Self-Supervised | 33.7 dB | 0.889 |  |
| SwinIR-DF | Transformer | 37.6 dB | 0.932 |  |
| Restormer-DF | Transformer | 38.9 dB | 0.943 |  |
| DiffusionDF | Diffusion | 40.3 dB | 0.956 |  |

## DESI Mass Spectrometry Imaging (`desi`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MSI-Hotelling | Classical | 22.1 dB | 0.701 | done |
| MSI-PCA | Classical | 24.8 dB | 0.749 | done |
| MSI-NMF | Classical | 26.3 dB | 0.782 | done |
| MSI-TV | Variational | 28.9 dB | 0.821 | done |
| DeepMSI | Deep Learning | 32.4 dB | 0.871 |  |
| MSI-GAN | Generative | 33.7 dB | 0.888 |  |
| MSIFormer | Transformer | 36.1 dB | 0.921 |  |
| SpaMSI-Net | Deep Learning | 34.8 dB | 0.904 |  |
| DiffusionMSI | Diffusion | 38.2 dB | 0.942 |  |

## DEXA (`dexa`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-DEXA | Classical | 26.4 dB | 0.782 |  |
| TV-DEXA | Variational | 30.1 dB | 0.841 |  |
| BML-Sep | Classical | 28.7 dB | 0.813 |  |
| DXA-CNN | Deep Learning | 33.8 dB | 0.881 |  |
| DXA-U-Net | Deep Learning | 35.6 dB | 0.907 |  |
| PnP-DXA | PnP | 34.2 dB | 0.893 |  |
| SwinDXA | Transformer | 37.9 dB | 0.931 |  |
| PhysDXA | Physics-Informed | 38.7 dB | 0.94 |  |
| DiffusionDXA | Diffusion | 40.4 dB | 0.956 |  |

## Differential Interference Contrast (DIC) (`dic`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DIC-Deconv | Classical | 24.1 dB | 0.731 | done |
| TV-DIC | Variational | 27.8 dB | 0.793 | done |
| Phase-DLSIM | Classical | 25.9 dB | 0.762 | done |
| DIC-CNN | Deep Learning | 31.4 dB | 0.856 |  |
| PhaseNet-DIC | Deep Learning | 33.7 dB | 0.884 |  |
| PnP-DIC | PnP | 32.2 dB | 0.869 | done |
| SwinDIC | Transformer | 36.1 dB | 0.921 |  |
| PhysPhase-Net | Physics-Informed | 37.4 dB | 0.935 |  |
| DiffusionDIC | Diffusion | 39.2 dB | 0.95 |  |

## Diffusion MRI (`diffusion_mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DTI-FIT | Classical | 22.4 dB | 0.71 | done |
| SHORE | Classical | 24.6 dB | 0.745 | done |
| CHARMED | Statistical | 26.8 dB | 0.782 |  |
| DnCNN-DTI | Deep Learning | 29.3 dB | 0.831 |  |
| DWIML-Net | Deep Learning | 32.1 dB | 0.871 |  |
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

## DNA-PAINT Super-Resolution (`dna_paint`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| STORM-2D | Classical | 21.3 dB | 0.695 | done |
| PALM | Classical | 22.8 dB | 0.718 | done |
| DAOSTORM | Classical | 25.4 dB | 0.762 | done |
| DeepSTORM | Deep Learning | 29.1 dB | 0.831 |  |
| DECODE | Deep Learning | 32.6 dB | 0.878 |  |
| TransPAINT | Transformer | 35.2 dB | 0.918 |  |
| SwinSTORM | Transformer | 36.8 dB | 0.934 |  |
| PhysSTORM | Physics-Informed | 38.1 dB | 0.946 |  |
| DiffPAINT | Diffusion Model | 39.7 dB | 0.958 |  |

## Doppler Ultrasound (`doppler_ultrasound`) — medical_ultrasound

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CF-Doppler | Classical | 22.5 dB | 0.712 | done |
| VENC-Flow | Classical | 24.1 dB | 0.738 | done |
| MV-Doppler | Variational | 26.8 dB | 0.778 | done |
| DnCNN-Doppler | Deep Learning | 29.5 dB | 0.832 |  |
| FlowNet-US | Deep Learning | 32.4 dB | 0.872 |  |
| TransFlow | Transformer | 35.1 dB | 0.914 |  |
| SwinDoppler | Transformer | 36.8 dB | 0.932 |  |
| PhysDoppler | Physics-Informed | 37.9 dB | 0.942 |  |
| DiffDoppler | Diffusion Model | 39.3 dB | 0.954 |  |

## DOT (`dot`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Born-Approx | Classical | 20.8 dB | 0.681 | done |
| TV-DOT | Variational | 23.5 dB | 0.729 | done |
| FEM-DOT | Classical | 25.9 dB | 0.771 | done |
| DnCNN-DOT | Deep Learning | 28.7 dB | 0.825 |  |
| DOT-Net | Deep Unrolling | 31.4 dB | 0.868 |  |
| TransDOT | Transformer | 34.2 dB | 0.91 |  |
| SwinDOT | Transformer | 36.1 dB | 0.93 |  |
| PhysDOT | Physics-Informed | 37.5 dB | 0.942 |  |
| DiffusionDOT | Diffusion Model | 39.0 dB | 0.954 |  |

## EBSD (`ebsd`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Hough-EBSD | Classical | 21.5 dB | 0.698 | done |
| DI-EBSD | Classical | 24.2 dB | 0.741 | done |
| TV-EBSD | Variational | 26.8 dB | 0.779 | done |
| DnCNN-EBSD | Deep Learning | 29.6 dB | 0.834 |  |
| PointEBSD | Deep Learning | 32.3 dB | 0.874 |  |
| TransEBSD | Transformer | 34.9 dB | 0.913 |  |
| SwinEBSD | Transformer | 36.5 dB | 0.931 |  |
| PhysEBSD | Physics-Informed | 37.8 dB | 0.943 |  |
| DiffEBSD | Diffusion Model | 39.1 dB | 0.954 |  |

## Eddy Current Imaging (`eddy_current`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| EC-Deconv | Classical | 22.1 dB | 0.705 | done |
| TV-EC | Variational | 24.8 dB | 0.748 | done |
| MUSIC-EC | Classical | 27.3 dB | 0.789 | done |
| DnCNN-EC | Deep Learning | 30.1 dB | 0.84 |  |
| ECNN-Defect | Deep Learning | 32.9 dB | 0.88 |  |
| TransEC | Transformer | 35.4 dB | 0.918 |  |
| SwinEC | Transformer | 36.9 dB | 0.934 |  |
| PhysEC | Physics-Informed | 38.0 dB | 0.944 |  |
| DiffEC | Diffusion Model | 39.3 dB | 0.955 |  |

## STEM-EDX Elemental Mapping (`edx_mapping`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLS-EDX | Classical | 22.3 dB | 0.708 | done |
| TV-EDX | Variational | 24.9 dB | 0.751 | done |
| NMF-EDX | Statistical | 27.5 dB | 0.792 |  |
| DnCNN-EDX | Deep Learning | 30.3 dB | 0.843 |  |
| N2V-EDX | Self-Supervised | 32.8 dB | 0.878 |  |
| TransEDX | Transformer | 35.2 dB | 0.916 |  |
| SwinEDX | Transformer | 36.8 dB | 0.933 |  |
| PhysEDX | Physics-Informed | 37.9 dB | 0.943 |  |
| DiffEDX | Diffusion Model | 39.4 dB | 0.955 |  |

## EELS (`eels`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PowerLaw-EELS | Classical | 21.8 dB | 0.699 | done |
| MLS-EELS | Statistical | 24.5 dB | 0.744 |  |
| ICA-EELS | Statistical | 27.1 dB | 0.786 |  |
| DnCNN-EELS | Deep Learning | 30.0 dB | 0.838 |  |
| N2V-EELS | Self-Supervised | 32.6 dB | 0.876 |  |
| TransEELS | Transformer | 35.1 dB | 0.915 |  |
| SwinEELS | Transformer | 36.7 dB | 0.932 |  |
| PhysEELS | Physics-Informed | 37.9 dB | 0.942 |  |
| DiffEELS | Diffusion Model | 39.3 dB | 0.954 |  |

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

## Elastography (`elastography`) — medical_ultrasound

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| LFE-Elasto | Classical | 22.3 dB | 0.71 | done |
| DI-Elasto | Variational | 24.8 dB | 0.752 | done |
| AIDE | Variational | 26.9 dB | 0.787 | done |
| DnCNN-Elasto | Deep Learning | 29.7 dB | 0.838 |  |
| ElastoNet | Deep Unrolling | 32.5 dB | 0.876 |  |
| TransElasto | Transformer | 35.0 dB | 0.915 |  |
| SwinElasto | Transformer | 36.6 dB | 0.932 |  |
| PhysElasto | Physics-Informed | 37.8 dB | 0.942 |  |
| DiffElasto | Diffusion Model | 39.2 dB | 0.953 |  |

## 4D-STEM (`electron_diffraction`) — electron_microscopy

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

## Electron Tomo (`electron_tomography`) — electron_microscopy

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

## Endoscopy (`endoscopy`) — clinical_optics

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Histogram-Eq | Classical | 24.1 dB | 0.738 |  |
| CLAHE-Endo | Classical | 26.5 dB | 0.772 |  |
| BM3D-Endo | Classical | 28.9 dB | 0.812 |  |
| DnCNN-Endo | Deep Learning | 31.4 dB | 0.855 |  |
| EndoSLAM-Net | Deep Learning | 33.8 dB | 0.889 |  |
| TransEndo | Transformer | 35.9 dB | 0.921 |  |
| SwinEndo | Transformer | 37.3 dB | 0.937 |  |
| PhysEndo | Physics-Informed | 38.4 dB | 0.947 |  |
| DiffEndo | Diffusion Model | 39.7 dB | 0.957 |  |

## Entangled Photon Microscopy (`entangled_photon`) — quantum

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Coincidence-Count | Classical | 19.8 dB | 0.658 | done |
| CS-Ghost | Compressed Sensing | 22.5 dB | 0.704 | done |
| SVD-Ghost | Statistical | 25.1 dB | 0.748 |  |
| DnCNN-Ghost | Deep Learning | 28.3 dB | 0.806 |  |
| GAN-Ghost | Generative | 31.0 dB | 0.852 |  |
| TransGhost | Transformer | 33.8 dB | 0.897 |  |
| SwinGhost | Transformer | 35.6 dB | 0.92 |  |
| PhysGhost | Physics-Informed | 37.1 dB | 0.936 |  |
| DiffGhost | Diffusion Model | 38.8 dB | 0.95 |  |

## Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Event-Integration | Classical | 22.1 dB | 0.702 | done |
| Complementary | Classical | 24.8 dB | 0.748 | done |
| E2VID | Recurrent | 27.9 dB | 0.798 |  |
| FireNet | Recurrent | 30.4 dB | 0.843 |  |
| SPADE-E2VID | Deep Learning | 32.8 dB | 0.878 |  |
| TransEvent | Transformer | 35.2 dB | 0.914 |  |
| SwinEvent | Transformer | 36.9 dB | 0.933 |  |
| PhysEvent | Physics-Informed | 38.0 dB | 0.944 |  |
| DiffEvent | Diffusion Model | 39.4 dB | 0.955 |  |

## Expansion Microscopy (ExM) (`expansion`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv-Exp | Classical | 24.5 dB | 0.742 | done |
| RL-ExM | Classical | 26.9 dB | 0.778 | done |
| TV-ExM | Variational | 29.1 dB | 0.819 | done |
| DnCNN-ExM | Deep Learning | 31.8 dB | 0.86 |  |
| DeepInterp-ExM | Deep Learning | 34.2 dB | 0.898 |  |
| TransExM | Transformer | 36.3 dB | 0.927 |  |
| SwinExM | Transformer | 37.7 dB | 0.941 |  |
| PhysExM | Physics-Informed | 38.8 dB | 0.95 |  |
| DiffExM | Diffusion Model | 40.0 dB | 0.96 |  |

## Focused Ion Beam SEM (FIB-SEM) (`fib_sem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BM3D-FIB | Classical | 25.3 dB | 0.755 | done |
| NLM-FIB | Classical | 27.1 dB | 0.789 | done |
| TV-FIB | Variational | 29.4 dB | 0.825 | done |
| DnCNN-FIB | Deep Learning | 31.9 dB | 0.862 |  |
| N2V-FIB | Self-Supervised | 33.8 dB | 0.891 |  |
| TransFIB | Transformer | 36.1 dB | 0.923 |  |
| SwinFIB | Transformer | 37.5 dB | 0.939 |  |
| PhysFIB | Physics-Informed | 38.6 dB | 0.949 |  |
| DiffFIB | Diffusion Model | 39.9 dB | 0.959 |  |

## Flash LiDAR (`flash_lidar`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLE-SPAD | Classical | 22.8 dB | 0.718 | done |
| Coates-Hist | Classical | 24.5 dB | 0.748 | done |
| NL-Means-LiDAR | Classical | 27.2 dB | 0.789 | done |
| DnCNN-LiDAR | Deep Learning | 30.1 dB | 0.84 |  |
| SPADnet | Deep Learning | 32.8 dB | 0.878 |  |
| TransLiDAR | Transformer | 35.3 dB | 0.916 |  |
| SwinLiDAR | Transformer | 36.9 dB | 0.933 |  |
| PhysLiDAR | Physics-Informed | 38.0 dB | 0.943 |  |
| DiffLiDAR | Diffusion Model | 39.4 dB | 0.955 |  |

## FLIM (`flim`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phasor-FLIM | Classical | 23.2 dB | 0.722 | done |
| MLE-FLIM | Statistical | 25.8 dB | 0.762 |  |
| RLD-FLIM | Classical | 27.9 dB | 0.798 | done |
| DnCNN-FLIM | Deep Learning | 30.7 dB | 0.845 |  |
| FLIMJ | Deep Learning | 33.1 dB | 0.882 |  |
| TransFLIM | Transformer | 35.5 dB | 0.918 |  |
| SwinFLIM | Transformer | 37.0 dB | 0.935 |  |
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

## fMRI (`fmri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical |  |  | done |
| SENSE | Classical |  |  | done |
| GRAPPA | Classical |  |  | done |
| L1-Wavelet | Compressed Sensing |  |  | done |
| k-t SPARSE-SENSE | Compressed Sensing |  |  | done |
| ESPIRiT | Compressed Sensing |  |  | done |
| LORAKS | Compressed Sensing |  |  | done |
| BM3D-MRI | PnP |  |  | done |
| ALOHA | Low-Rank |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| PnP-DnCNN-Pro | PnP |  |  | done |
| Deep-ADMM-Net | Deep Unrolling |  |  |  |
| DCCNN | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| U-Net++ | Deep Learning |  |  |  |
| MoDL | Deep Unrolling |  |  |  |
| MoDL-Net++ | Deep Unrolling |  |  |  |
| E2E-VarNet | Deep Unrolling |  |  |  |
| HybridCascade | Deep Unrolling |  |  |  |
| HybridCascade++ | Deep Unrolling |  |  |  |
| SwinMR | Transformer |  |  |  |
| SwinMR++ | Transformer |  |  |  |
| HUMUS-Net | Transformer |  |  |  |
| HUMUS-Net++ | Transformer |  |  |  |
| ReconFormer | Transformer |  |  |  |
| ReconFormer++ | Transformer |  |  |  |
| Score-MRI | Score-Based |  |  |  |
| PromptMR | Deep Unrolling |  |  |  |
| MRI-DiffusionNet | Diffusion |  |  |  |
| MRDynamo | Physics-Informed |  |  |  |
| BrainID-MRI | Foundation Model |  |  |  |
| MMR-Mamba | Physics-Informed |  |  |  |
| PromptMR-SFM | Physics-Informed |  |  |  |
| MR-IPT | Foundation Model |  |  |  |
| MRI-FM | Foundation Model |  |  |  |

## FPM (`fpm`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Alternating Projections | Classical | 25.0 dB | 0.72 | done |
| Gradient Descent FPM | Classical | 28.5 dB | 0.84 | done |
| Fourier PtychoNet | Deep Learning | 32.3 dB | 0.91 |  |
| PtychoDV | Deep Unrolling | 33.8 dB | 0.935 |  |

## FTIR Spectroscopic Imaging (`ftir_imaging`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical |  |  | done |
| Baseline Correction | Classical |  |  | done |
| SVD | Classical |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CDAE | Deep Learning |  |  |  |
| U-Net-Spectra | Deep Learning |  |  |  |
| Cascade-UNet | Deep Learning |  |  |  |
| PINN-Spectra | Deep Learning |  |  |  |
| SpectraFormer | Vision Transformer |  |  |  |
| DiffusionSpectra | Diffusion |  |  |  |
| ScoreSpectra | Score-based |  |  |  |

## Fundus (`fundus`) — clinical_optics

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical | 24.5 dB | 0.68 | done |
| PnP-BM3D | PnP | 28.8 dB | 0.83 | done |
| cofe-Net | Deep Learning | 32.5 dB | 0.91 |  |
| Swin-Fundus | Transformer | 34.2 dB | 0.94 |  |

## Full-Waveform Inversion (FWI) (`fwi`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| L-BFGS FWI | Classical | 23.5 dB | 0.65 | done |
| TV-Reg FWI | Classical | 26.8 dB | 0.78 | done |
| InversionNet | Deep Learning | 30.5 dB | 0.88 |  |
| VelocityGAN | Deep Learning | 32.2 dB | 0.91 |  |

## 3DGS (`gaussian_splatting`) — neural_rendering

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| COLMAP+MVS | Classical |  |  | done |
| Photogrammetry | Classical |  |  | done |
| NeRF | Deep Learning |  |  |  |
| Mip-NeRF 360 | Deep Learning |  |  |  |
| Instant-NGP | Deep Learning |  |  |  |
| 3D-GS | Deep Learning |  |  |  |
| 3D-GS++ | Deep Learning |  |  |  |
| 2DGS | Deep Learning |  |  |  |
| GaussianShader | Vision Transformer |  |  |  |
| NeRFactor2 | Deep Learning |  |  |  |
| Mesh-GS | Deep Learning |  |  |  |

## Ghost Imaging (`ghost_imaging`) — quantum

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| G(2)-Corr | Classical |  |  | done |
| Photon Counting | Classical |  |  | done |
| CS-TVAL3 | PnP |  |  | done |
| Bayesian CS | PnP |  |  | done |
| DRU-Net | Deep Learning |  |  |  |
| Quantum-CNN | Deep Learning |  |  |  |
| Ghost-ViT | Vision Transformer |  |  |  |
| Quantum-ViT | Vision Transformer |  |  |  |
| DiffusionQuantum | Diffusion |  |  |  |
| ScoreQuantum | Score-based |  |  |  |

## Ground-Penetrating Radar (GPR) (`gpr`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Kirchhoff Migration | Classical | 22.0 dB | 0.6 |  |
| RTM | Classical | 25.5 dB | 0.74 |  |
| GPR-RCNN | Deep Learning | 29.8 dB | 0.87 |  |
| HyperDet | Deep Learning | 31.5 dB | 0.905 |  |

## Gravitational Wave Detection (`gravitational_wave`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Matched Filter | Classical | 20.0 dB | 0.52 | done |
| BayesWave | PnP | 24.5 dB | 0.71 |  |
| GW-CNN | Deep Learning | 28.8 dB | 0.85 |  |
| WaveFormer | Transformer | 30.5 dB | 0.895 |  |

## High Dynamic Range (HDR) Imaging (`hdr_imaging`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-Deconv | Classical |  |  | done |
| Laplacian Pyramid | Classical |  |  | done |
| Lucy-Richardson | Classical |  |  | done |
| PnP-FFDNet | PnP |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| HDR-CNN | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| LaplacianFormer | Deep Learning |  |  |  |
| Uformer | Vision Transformer |  |  |  |
| DeblurGaussian | Vision Transformer |  |  |  |
| HDRFormer | Vision Transformer |  |  |  |
| PhotoFormer | Vision Transformer |  |  |  |
| DiffusionPhoto | Diffusion |  |  |  |
| ScorePhoto | Score-based |  |  |  |

## Holography (`holography`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gerchberg-Saxton | Classical |  |  | done |
| GS/HIO | Classical |  |  | done |
| Error Reduction | Classical |  |  | done |
| prDeep | Deep Unrolling |  |  |  |
| PhaseNet | Deep Learning |  |  |  |
| deep-PR | Deep Learning |  |  |  |
| LRGS | Deep Learning |  |  |  |
| PhaseResNet | Deep Learning |  |  |  |
| CyclePhase | Deep Learning |  |  |  |
| PhaseFormer | Vision Transformer |  |  |  |
| AutoPhase++ | Vision Transformer |  |  |  |
| HolographyViT | Vision Transformer |  |  |  |
| DiffusionPhase | Diffusion |  |  |  |
| ScorePhase | Score-based |  |  |  |

## Hyperspectral Remote Sensing (`hyperspectral_remote`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CNMF | Classical | 26.0 dB | 0.72 | done |
| PnP-LTTR | PnP | 30.0 dB | 0.85 | done |
| DBIN | Deep Learning | 34.5 dB | 0.93 |  |
| MST++ | Transformer | 36.8 dB | 0.955 |  |

## Electrical Impedance Tomography (EIT) (`impedance_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gauss-Newton | Classical | 21.0 dB | 0.55 | done |
| TV-ADMM | PnP | 24.5 dB | 0.7 | done |
| D-bar CNN | Deep Learning | 28.5 dB | 0.84 |  |
| EIT-Former | Transformer | 30.0 dB | 0.88 |  |

## Industrial CT (`industrial_ct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FDK | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| Learned Primal-Dual | Deep Unrolling |  |  |  |

## Interferometric SAR (InSAR) (`insar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Goldstein-MCF | Classical | 23.0 dB | 0.64 |  |
| InSAR-BM3D | PnP | 27.0 dB | 0.79 |  |
| PhaseNet | Deep Learning | 31.0 dB | 0.89 |  |
| InSAR-Former | Transformer | 33.0 dB | 0.92 |  |

## Integral (`integral`) — computational

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Shift-and-Add | Classical | 25.0 dB | 0.7 | done |
| PnP-LF | PnP | 29.0 dB | 0.83 | done |
| LFAttNet | Deep Learning | 33.5 dB | 0.92 |  |
| DistgSSR | Transformer | 35.8 dB | 0.95 |  |

## Image Scanning Microscopy (ISM) (`ism`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## Intravascular Ultrasound (IVUS) (`ivus`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DAS | Classical |  |  | done |
| DAS-CF | Classical |  |  | done |
| PW-DAS | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-TV | PnP |  |  | done |
| ABLE | Deep Learning |  |  |  |
| MU-Net | Deep Learning |  |  |  |
| Phase-ADMM-Net | Deep Unrolling |  |  |  |
| UltrasoundFormer | Vision Transformer |  |  |  |
| BeamFormer | Transformer |  |  |  |
| AttentionBeam | Transformer |  |  |  |
| BeamDATA | Deep Learning |  |  |  |
| DiffUS | Diffusion |  |  |  |
| ScoreUS | Score-based |  |  |  |

## Lattice Light-Sheet Microscopy (`lattice_lightsheet`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## Lensless (`lensless`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-ADMM | Classical | 23.5 dB | 0.64 |  |
| PnP-ADMM | PnP | 27.5 dB | 0.79 |  |
| FlatNet | Deep Learning | 31.8 dB | 0.89 |  |
| Uformer | Transformer | 33.5 dB | 0.92 |  |

## Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical |  |  | done |
| Baseline Correction | Classical |  |  | done |
| SVD | Classical |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CDAE | Deep Learning |  |  |  |
| U-Net-Spectra | Deep Learning |  |  |  |
| Cascade-UNet | Deep Learning |  |  |  |
| PINN-Spectra | Deep Learning |  |  |  |
| SpectraFormer | Vision Transformer |  |  |  |
| DiffusionSpectra | Diffusion |  |  |  |
| ScoreSpectra | Score-based |  |  |  |

## LiDAR (`lidar`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Bilateral Filter | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| RandLA-Net | Deep Learning |  |  |  |
| Point Transformer | Transformer |  |  |  |

## Light Field (`light_field`) — computational

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Shift-and-Sum | Classical | 24.5 dB | 0.69 | done |
| PnP-LF | PnP | 28.5 dB | 0.82 | done |
| LFNet | Deep Learning | 33.0 dB | 0.915 |  |
| DistgSSR | Transformer | 35.5 dB | 0.948 |  |

## Light-Sheet (`lightsheet`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## Lucky Imaging (`lucky_imaging`) — astronomy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Shift-and-Add | Classical |  |  | done |
| Drizzle | Classical |  |  | done |
| BDI | PnP |  |  | done |
| SpeckleNet | Deep Learning |  |  |  |

## Machine Vision / AOI (`machine_vision`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Template Match | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PatchCore | Deep Learning |  |  |  |
| UniAD | Transformer |  |  |  |

## Magnetic Particle Imaging (MPI) (`magnetic_particle`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| Matched Filter | Classical |  |  | done |
| PnP-RED | PnP |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| ResUNet | Deep Learning |  |  |  |
| Domain-Adapted-CNN | Deep Learning |  |  |  |
| SwinIR | Vision Transformer |  |  |  |
| ExpFormer | Vision Transformer |  |  |  |
| DiffusionExperimental | Diffusion |  |  |  |
| ScoreExperimental | Score-based |  |  |  |

## MALDI Mass Spectrometry Imaging (`maldi_msi`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv | Classical |  |  | done |
| Calibration-Lookup | Classical |  |  | done |
| Peak Fitting | Classical |  |  | done |
| PnP-BM3D | PnP |  |  | done |
| PnP-NLM | PnP |  |  | done |
| ResNet-Calib | Deep Learning |  |  |  |
| Instrument-CNN | Deep Learning |  |  |  |
| CalibFormer | Vision Transformer |  |  |  |
| MassSpecFormer | Vision Transformer |  |  |  |
| DiffusionInstrumentation | Diffusion |  |  |  |
| ScoreInstrumentation | Score-based |  |  |  |

## Mammography (`mammography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical |  |  | done |
| TV-ADMM | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| RED-CNN | Deep Learning |  |  |  |
| Learned Primal-Dual | Deep Unrolling |  |  |  |
| DuDoTrans | Deep Unrolling |  |  |  |
| CT-ViT | Vision Transformer |  |  |  |
| CTFormer | Transformer |  |  |  |
| DOLCE | Diffusion |  |  |  |
| DiffusionCT | Diffusion |  |  |  |
| Score-CT | Score-based |  |  |  |

## Matrix (`matrix`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| GAP-TV | Classical |  |  | done |
| FISTA-TV | Classical |  |  | done |
| TVAL3 | Classical |  |  | done |
| PnP-FFDNet | PnP |  |  | done |
| MST-L | Transformer |  |  |  |
| EfficientSCI | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| CST | Transformer |  |  |  |
| HiSViT+ | Vision Transformer |  |  |  |
| CSTrans | Transformer |  |  |  |
| PromptSCI | Deep Learning |  |  |  |
| DiffusionHSI | Diffusion |  |  |  |
| ScoreSCI | Diffusion |  |  |  |
| FlowHSI | Generative |  |  |  |

## Magnetic Force Microscopy (MFM) (`mfm`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BTR | Classical |  |  | done |
| MLE Reconstruction | Classical |  |  | done |
| Reg-Deconv | PnP |  |  | done |
| TV-Deconvolution | PnP |  |  | done |
| DeepSPM | Deep Learning |  |  |  |
| U-Net-SPM | Deep Learning |  |  |  |
| E2E-BTR | Deep Learning |  |  |  |
| SPM-Former | Vision Transformer |  |  |  |
| DiffusionSPM | Diffusion |  |  |  |
| ScoreSPM | Score-based |  |  |  |

## MINFLUX Nanoscopy (`minflux`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MLE Localization | Classical |  |  | done |
| SPARCOM | PnP |  |  | done |
| DECODE | Deep Learning |  |  |  |
| ANNA-PALM | Deep Learning |  |  |  |

## MR Elastography (MRE) (`mr_elastography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical |  |  | done |
| SENSE | Classical |  |  | done |
| GRAPPA | Classical |  |  | done |
| L1-Wavelet | Compressed Sensing |  |  | done |
| k-t SPARSE-SENSE | Compressed Sensing |  |  | done |
| ESPIRiT | Compressed Sensing |  |  | done |
| LORAKS | Compressed Sensing |  |  | done |
| BM3D-MRI | PnP |  |  | done |
| ALOHA | Low-Rank |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| PnP-DnCNN-Pro | PnP |  |  | done |
| Deep-ADMM-Net | Deep Unrolling |  |  |  |
| DCCNN | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| U-Net++ | Deep Learning |  |  |  |
| MoDL | Deep Unrolling |  |  |  |
| MoDL-Net++ | Deep Unrolling |  |  |  |
| E2E-VarNet | Deep Unrolling |  |  |  |
| HybridCascade | Deep Unrolling |  |  |  |
| HybridCascade++ | Deep Unrolling |  |  |  |
| SwinMR | Transformer |  |  |  |
| SwinMR++ | Transformer |  |  |  |
| HUMUS-Net | Transformer |  |  |  |
| HUMUS-Net++ | Transformer |  |  |  |
| ReconFormer | Transformer |  |  |  |
| ReconFormer++ | Transformer |  |  |  |
| Score-MRI | Score-Based |  |  |  |
| PromptMR | Deep Unrolling |  |  |  |
| MRI-DiffusionNet | Diffusion |  |  |  |
| MRDynamo | Physics-Informed |  |  |  |
| BrainID-MRI | Foundation Model |  |  |  |
| MMR-Mamba | Physics-Informed |  |  |  |
| PromptMR-SFM | Physics-Informed |  |  |  |
| MR-IPT | Foundation Model |  |  |  |
| MRI-FM | Foundation Model |  |  |  |

## MR Fingerprinting (MRF) (`mr_fingerprinting`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SVD-MRF | Classical | 23.5 dB | 0.65 |  |
| MANTIS | Classical | 27.0 dB | 0.79 |  |
| MRF-Net | Deep Learning | 31.5 dB | 0.895 |  |
| MRF-Former | Transformer | 33.5 dB | 0.93 |  |

## MR Angiography (MRA) (`mra`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical |  |  | done |
| SENSE | Classical |  |  | done |
| GRAPPA | Classical |  |  | done |
| L1-Wavelet | Compressed Sensing |  |  | done |
| k-t SPARSE-SENSE | Compressed Sensing |  |  | done |
| ESPIRiT | Compressed Sensing |  |  | done |
| LORAKS | Compressed Sensing |  |  | done |
| BM3D-MRI | PnP |  |  | done |
| ALOHA | Low-Rank |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| PnP-DnCNN-Pro | PnP |  |  | done |
| Deep-ADMM-Net | Deep Unrolling |  |  |  |
| DCCNN | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| U-Net++ | Deep Learning |  |  |  |
| MoDL | Deep Unrolling |  |  |  |
| MoDL-Net++ | Deep Unrolling |  |  |  |
| E2E-VarNet | Deep Unrolling |  |  |  |
| HybridCascade | Deep Unrolling |  |  |  |
| HybridCascade++ | Deep Unrolling |  |  |  |
| SwinMR | Transformer |  |  |  |
| SwinMR++ | Transformer |  |  |  |
| HUMUS-Net | Transformer |  |  |  |
| HUMUS-Net++ | Transformer |  |  |  |
| ReconFormer | Transformer |  |  |  |
| ReconFormer++ | Transformer |  |  |  |
| Score-MRI | Score-Based |  |  |  |
| PromptMR | Deep Unrolling |  |  |  |
| MRI-DiffusionNet | Diffusion |  |  |  |
| MRDynamo | Physics-Informed |  |  |  |
| BrainID-MRI | Foundation Model |  |  |  |
| MMR-Mamba | Physics-Informed |  |  |  |
| PromptMR-SFM | Physics-Informed |  |  |  |
| MR-IPT | Foundation Model |  |  |  |
| MRI-FM | Foundation Model |  |  |  |

## MRI (`mri`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical | 26.0 dB | 0.62 | done |
| SENSE | Classical | 29.5 dB | 0.83 |  |
| GRAPPA | Classical | 31.2 dB | 0.86 |  |
| L1-Wavelet | Compressed Sensing | 32.1 dB | 0.87 |  |
| k-t SPARSE-SENSE | Compressed Sensing | 32.5 dB | 0.875 |  |
| ESPIRiT | Compressed Sensing | 33.4 dB | 0.89 |  |
| LORAKS | Compressed Sensing | 33.8 dB | 0.893 |  |
| BM3D-MRI | PnP | 34.2 dB | 0.897 |  |
| ALOHA | Low-Rank | 34.5 dB | 0.9 |  |
| PnP-DnCNN | PnP | 35.0 dB | 0.905 |  |
| PnP-DnCNN-Pro | PnP | 41.0 dB | 0.968 |  |
| Deep-ADMM-Net | Deep Unrolling | 35.3 dB | 0.907 |  |
| DCCNN | Deep Learning | 35.5 dB | 0.908 |  |
| U-Net | Deep Learning | 35.9 dB | 0.904 |  |
| U-Net++ | Deep Learning | 41.5 dB | 0.978 |  |
| MoDL | Deep Unrolling | 36.5 dB | 0.912 |  |
| MoDL-Net++ | Deep Unrolling | 41.8 dB | 0.978 |  |
| E2E-VarNet | Deep Unrolling | 39.4 dB | 0.924 |  |
| HybridCascade | Deep Unrolling | 37.8 dB | 0.917 |  |
| HybridCascade++ | Deep Unrolling | 42.5 dB | 0.981 |  |
| SwinMR | Transformer | 38.5 dB | 0.921 |  |
| SwinMR++ | Transformer | 43.8 dB | 0.983 |  |
| HUMUS-Net | Transformer | 38.9 dB | 0.923 |  |
| HUMUS-Net++ | Transformer | 43.1 dB | 0.979 |  |
| ReconFormer | Transformer | 39.0 dB | 0.922 |  |
| ReconFormer++ | Transformer | 41.5 dB | 0.969 |  |
| Score-MRI | Score-Based | 31.4 dB | 0.89 |  |
| PromptMR | Deep Unrolling | 39.7 dB | 0.926 |  |
| MRI-DiffusionNet | Diffusion | 40.1 dB | 0.932 |  |
| MRDynamo | Physics-Informed | 40.5 dB | 0.938 |  |
| BrainID-MRI | Foundation Model | 41.0 dB | 0.942 |  |
| MMR-Mamba | Physics-Informed | 40.98 dB | 0.969 |  |
| PromptMR-SFM | Physics-Informed | 41.3 dB | 0.971 |  |
| MR-IPT | Foundation Model | 42.48 dB | 0.983 |  |
| MRI-FM | Foundation Model | 42.1 dB | 0.948 |  |

## MRS (`mrs`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical |  |  | done |
| SENSE | Classical |  |  | done |
| GRAPPA | Classical |  |  | done |
| L1-Wavelet | Compressed Sensing |  |  | done |
| k-t SPARSE-SENSE | Compressed Sensing |  |  | done |
| ESPIRiT | Compressed Sensing |  |  | done |
| LORAKS | Compressed Sensing |  |  | done |
| BM3D-MRI | PnP |  |  | done |
| ALOHA | Low-Rank |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| PnP-DnCNN-Pro | PnP |  |  | done |
| Deep-ADMM-Net | Deep Unrolling |  |  |  |
| DCCNN | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| U-Net++ | Deep Learning |  |  |  |
| MoDL | Deep Unrolling |  |  |  |
| MoDL-Net++ | Deep Unrolling |  |  |  |
| E2E-VarNet | Deep Unrolling |  |  |  |
| HybridCascade | Deep Unrolling |  |  |  |
| HybridCascade++ | Deep Unrolling |  |  |  |
| SwinMR | Transformer |  |  |  |
| SwinMR++ | Transformer |  |  |  |
| HUMUS-Net | Transformer |  |  |  |
| HUMUS-Net++ | Transformer |  |  |  |
| ReconFormer | Transformer |  |  |  |
| ReconFormer++ | Transformer |  |  |  |
| Score-MRI | Score-Based |  |  |  |
| PromptMR | Deep Unrolling |  |  |  |
| MRI-DiffusionNet | Diffusion |  |  |  |
| MRDynamo | Physics-Informed |  |  |  |
| BrainID-MRI | Foundation Model |  |  |  |
| MMR-Mamba | Physics-Informed |  |  |  |
| PromptMR-SFM | Physics-Informed |  |  |  |
| MR-IPT | Foundation Model |  |  |  |
| MRI-FM | Foundation Model |  |  |  |

## Multispectral Satellite Imaging (`multispectral_sat`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov | Classical |  |  | done |
| LSQR | Classical |  |  | done |
| ART | Classical |  |  | done |
| PnP-RED | PnP |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| Deep Image Prior | Deep Learning |  |  |  |
| Plug-and-Play | Deep Learning |  |  |  |
| SwinIR | Vision Transformer |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| NAFNet | Vision Transformer |  |  |  |
| CompFormer | Vision Transformer |  |  |  |
| DiffusionCompute | Diffusion |  |  |  |
| FlowCompute | Generative |  |  |  |

## Muon Tomo (`muon_tomo`) — particle_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-PET | Classical |  |  | done |
| OSEM | Classical |  |  | done |
| ML-EM | Classical |  |  | done |
| MAPEM-RDP | PnP |  |  | done |
| OS-EM | Classical |  |  | done |
| DeepPET | Deep Learning |  |  |  |
| U-Net-PET | Deep Learning |  |  |  |
| TransEM | Transformer |  |  |  |
| PET-ViT | Vision Transformer |  |  |  |
| PETFormer | Vision Transformer |  |  |  |

## NeRF (`nerf`) — neural_rendering

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| COLMAP+MVS | Classical |  |  | done |
| Photogrammetry | Classical |  |  | done |
| NeRF | Deep Learning |  |  |  |
| Mip-NeRF 360 | Deep Learning |  |  |  |
| Instant-NGP | Deep Learning |  |  |  |
| 3D-GS | Deep Learning |  |  |  |
| 3D-GS++ | Deep Learning |  |  |  |
| 2DGS | Deep Learning |  |  |  |
| GaussianShader | Vision Transformer |  |  |  |
| NeRFactor2 | Deep Learning |  |  |  |
| Mesh-GS | Deep Learning |  |  |  |

## Neutron Diffraction (`neutron_diffraction`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Rietveld-GSAS | Classical | 23.0 dB | 0.64 | done |
| Le Bail Fit | Classical | 26.5 dB | 0.76 | done |
| NeutronNet | Deep Learning | 30.5 dB | 0.88 |  |
| DiffFormer | Transformer | 32.5 dB | 0.915 |  |

## Neutron Tomo (`neutron_tomo`) — particle_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-PET | Classical |  |  | done |
| OSEM | Classical |  |  | done |
| ML-EM | Classical |  |  | done |
| MAPEM-RDP | PnP |  |  | done |
| OS-EM | Classical |  |  | done |
| DeepPET | Deep Learning |  |  |  |
| U-Net-PET | Deep Learning |  |  |  |
| TransEM | Transformer |  |  |  |
| PET-ViT | Vision Transformer |  |  |  |
| PETFormer | Vision Transformer |  |  |  |

## Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| MBLL | Classical |  |  | done |
| Tikhonov-DOT | Classical |  |  | done |
| PnP-DOT | PnP |  |  | done |
| DL-DOT | Deep Learning |  |  |  |

## Near-field Scanning Optical Microscopy (NSOM) (`nsom`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BTR | Classical |  |  | done |
| MLE Reconstruction | Classical |  |  | done |
| Reg-Deconv | PnP |  |  | done |
| TV-Deconvolution | PnP |  |  | done |
| DeepSPM | Deep Learning |  |  |  |
| U-Net-SPM | Deep Learning |  |  |  |
| E2E-BTR | Deep Learning |  |  |  |
| SPM-Former | Vision Transformer |  |  |  |
| DiffusionSPM | Diffusion |  |  |  |
| ScoreSPM | Score-based |  |  |  |

## Ocean Acoustic Tomography (`ocean_acoustic_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| Matched Filter | Classical |  |  | done |
| PnP-RED | PnP |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| ResUNet | Deep Learning |  |  |  |
| Domain-Adapted-CNN | Deep Learning |  |  |  |
| SwinIR | Vision Transformer |  |  |  |
| ExpFormer | Vision Transformer |  |  |  |
| DiffusionExperimental | Diffusion |  |  |  |
| ScoreExperimental | Score-based |  |  |  |

## Ocean Color Remote Sensing (`ocean_color`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gordon AC | Classical | 22.5 dB | 0.61 | done |
| MUMM | Classical | 26.0 dB | 0.74 | done |
| OC-Net | Deep Learning | 30.5 dB | 0.87 |  |
| AquaFormer | Transformer | 32.5 dB | 0.91 |  |

## OCT (`oct`) — clinical_optics

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FFT-OCT | Classical |  |  | done |
| Speckle-Lee | Classical |  |  | done |
| TV-Denoising | Classical |  |  | done |
| BM4D | PnP |  |  | done |
| NLM-OCT | PnP |  |  | done |
| Speckle-DenoiseNet | Deep Learning |  |  |  |
| U-Net-OCT | Deep Learning |  |  |  |
| OCTA-Net | Deep Learning |  |  |  |
| OCT-ViT | Vision Transformer |  |  |  |
| SpeckleFormer | Vision Transformer |  |  |  |
| RetinalFormer | Transformer |  |  |  |
| DiffusionOCT | Diffusion |  |  |  |
| ScoreOCT | Score-based |  |  |  |

## OCTA (`octa`) — clinical_optics

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FFT-OCT | Classical |  |  | done |
| Speckle-Lee | Classical |  |  | done |
| TV-Denoising | Classical |  |  | done |
| BM4D | PnP |  |  | done |
| NLM-OCT | PnP |  |  | done |
| Speckle-DenoiseNet | Deep Learning |  |  |  |
| U-Net-OCT | Deep Learning |  |  |  |
| OCTA-Net | Deep Learning |  |  |  |
| OCT-ViT | Vision Transformer |  |  |  |
| SpeckleFormer | Vision Transformer |  |  |  |
| RetinalFormer | Transformer |  |  |  |
| DiffusionOCT | Diffusion |  |  |  |
| ScoreOCT | Score-based |  |  |  |

## Optical Diffraction Tomography (ODT) (`odt`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wolf FBP | Classical | 24.5 dB | 0.69 |  |
| Born-ADMM | PnP | 28.0 dB | 0.81 |  |
| ODT-Net | Deep Learning | 32.0 dB | 0.905 |  |
| Rytov-Former | Transformer | 34.0 dB | 0.935 |  |

## PALM/STORM (`palm_storm`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| ThunderSTORM | Classical |  |  | done |
| FALCON | PnP |  |  | done |
| Deep-STORM | Deep Learning |  |  |  |
| DECODE | Deep Learning |  |  |  |

## Panorama (`panorama`) — computational_photography

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SIFT-RANSAC | Classical | 26.0 dB | 0.74 | done |
| APAP | Classical | 29.5 dB | 0.85 |  |
| UDIS | Deep Learning | 33.0 dB | 0.92 |  |
| PanoFormer | Transformer | 35.0 dB | 0.95 |  |

## Particle Calorimetry (`particle_calorimetry`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PandoraPFA | Classical | 22.0 dB | 0.58 | done |
| GARFIELD++ | Classical | 25.5 dB | 0.72 | done |
| GravNet | Deep Learning | 29.5 dB | 0.86 |  |
| CaloDiffusion | Diffusion | 31.5 dB | 0.9 |  |

## Passive Microwave Radiometry (`passive_microwave`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Backus-Gilbert | Classical |  |  | done |
| Tikhonov-SMOS | Classical |  |  | done |
| RadioNet | Deep Learning |  |  |  |
| MWR-Former | Transformer |  |  |  |

## PET (`pet`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-PET | Classical |  |  | done |
| OSEM | Classical |  |  | done |
| ML-EM | Classical |  |  | done |
| MAPEM-RDP | PnP |  |  | done |
| OS-EM | Classical |  |  | done |
| DeepPET | Deep Learning |  |  |  |
| U-Net-PET | Deep Learning |  |  |  |
| TransEM | Transformer |  |  |  |
| PET-ViT | Vision Transformer |  |  |  |
| PETFormer | Vision Transformer |  |  |  |

## PET/CT (`pet_ct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical |  |  | done |
| TV-ADMM | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| RED-CNN | Deep Learning |  |  |  |
| Learned Primal-Dual | Deep Unrolling |  |  |  |
| DuDoTrans | Deep Unrolling |  |  |  |
| CT-ViT | Vision Transformer |  |  |  |
| CTFormer | Transformer |  |  |  |
| DOLCE | Diffusion |  |  |  |
| DiffusionCT | Diffusion |  |  |  |
| Score-CT | Score-based |  |  |  |

## PET/MR (`pet_mr`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-PET | Classical |  |  | done |
| OSEM | Classical |  |  | done |
| ML-EM | Classical |  |  | done |
| MAPEM-RDP | PnP |  |  | done |
| OS-EM | Classical |  |  | done |
| DeepPET | Deep Learning |  |  |  |
| U-Net-PET | Deep Learning |  |  |  |
| TransEM | Transformer |  |  |  |
| PET-ViT | Vision Transformer |  |  |  |
| PETFormer | Vision Transformer |  |  |  |

## Phase Contrast Microscopy (`phase_contrast`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TIE Solver | Classical | 25.5 dB | 0.72 | done |
| DPC-ADMM | PnP | 29.0 dB | 0.84 | done |
| QPI-Net | Deep Learning | 33.0 dB | 0.92 |  |
| PhaseFormer | Transformer | 35.0 dB | 0.945 |  |

## CDI (`phase_retrieval`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Gerchberg-Saxton | Classical |  |  | done |
| GS/HIO | Classical |  |  | done |
| Error Reduction | Classical |  |  | done |
| prDeep | Deep Unrolling |  |  |  |
| PhaseNet | Deep Learning |  |  |  |
| deep-PR | Deep Learning |  |  |  |
| LRGS | Deep Learning |  |  |  |
| PhaseResNet | Deep Learning |  |  |  |
| CyclePhase | Deep Learning |  |  |  |
| PhaseFormer | Vision Transformer |  |  |  |
| AutoPhase++ | Vision Transformer |  |  |  |
| HolographyViT | Vision Transformer |  |  |  |
| DiffusionPhase | Diffusion |  |  |  |
| ScorePhase | Score-based |  |  |  |

## Photoacoustic (`photoacoustic`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Universal Back-Proj | Classical | 23.5 dB | 0.64 |  |
| PnP-ADMM | PnP | 27.0 dB | 0.79 |  |
| Deep-PAI | Deep Learning | 31.5 dB | 0.89 |  |
| PAT-Former | Transformer | 33.5 dB | 0.92 |  |

## Photometric Stereo (`photometric_stereo`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| LS Normal Est. | Classical | 25.0 dB | 0.7 | done |
| Robust PCA | Classical | 28.5 dB | 0.82 | done |
| CNN-PS | Deep Learning | 32.5 dB | 0.915 |  |
| PS-Transformer | Transformer | 34.2 dB | 0.945 |  |

## Polarization (`polarization`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## Polarimetric SAR (PolSAR) (`polsar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Matched Filter | Classical |  |  | done |
| Range-Doppler | Classical |  |  | done |
| Chirp Scaling | Classical |  |  | done |
| SAR-BM3D | PnP |  |  | done |
| Lee Filter | PnP |  |  | done |
| SAR-DRN | Deep Learning |  |  |  |
| SAR-ResNet | Deep Learning |  |  |  |
| SAR-CAM | Transformer |  |  |  |
| SARFormer | Vision Transformer |  |  |  |
| PanSharpener++ | Deep Learning |  |  |  |
| SARDenoiserViT | Vision Transformer |  |  |  |
| DiffusionSAR | Diffusion |  |  |  |
| ScoreSAR | Score-based |  |  |  |

## Portal Imaging (EPID) (`portal_imaging`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical |  |  | done |
| TV-ADMM | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| RED-CNN | Deep Learning |  |  |  |
| Learned Primal-Dual | Deep Unrolling |  |  |  |
| DuDoTrans | Deep Unrolling |  |  |  |
| CT-ViT | Vision Transformer |  |  |  |
| CTFormer | Transformer |  |  |  |
| DOLCE | Diffusion |  |  |  |
| DiffusionCT | Diffusion |  |  |  |
| Score-CT | Score-based |  |  |  |

## Proton Radiography (`proton_radiography`) — particle_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-MLP | Classical | 23.5 dB | 0.65 |  |
| DROP-TVS | PnP | 27.0 dB | 0.79 |  |
| ProtonNet | Deep Learning | 31.0 dB | 0.89 |  |
| pCT-Former | Transformer | 33.0 dB | 0.92 |  |

## Proton Therapy Imaging (`proton_therapy_img`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical |  |  | done |
| TV-ADMM | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| RED-CNN | Deep Learning |  |  |  |
| Learned Primal-Dual | Deep Unrolling |  |  |  |
| DuDoTrans | Deep Unrolling |  |  |  |
| CT-ViT | Vision Transformer |  |  |  |
| CTFormer | Transformer |  |  |  |
| DOLCE | Diffusion |  |  |  |
| DiffusionCT | Diffusion |  |  |  |
| Score-CT | Score-based |  |  |  |

## Ptychography (`ptychography`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| ePIE | Classical | 25.0 dB | 0.71 |  |
| sDR | Classical | 28.5 dB | 0.82 |  |
| PtychoNN | Deep Learning | 32.5 dB | 0.91 |  |
| AutoPhaseNN | Deep Learning | 34.0 dB | 0.935 |  |

## Pump-Probe Microscopy (`pump_probe`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SVD-GlobFit | Classical | 22.5 dB | 0.6 | done |
| MCR-ALS | Classical | 26.0 dB | 0.74 | done |
| TAS-Net | Deep Learning | 30.0 dB | 0.87 |  |
| DynFormer | Transformer | 32.0 dB | 0.905 |  |

## Quantum Illumination (`quantum_illumination`) — quantum

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| OPA Receiver | Classical | 18.0 dB | 0.42 | done |
| FF-SFG | Classical | 22.0 dB | 0.6 | done |
| QI-Net | Deep Learning | 26.5 dB | 0.78 |  |
| QuantumFormer | Transformer | 28.5 dB | 0.84 |  |

## Radio Aperture Synthesis (`radio_astronomy`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CLEAN | Classical |  |  | done |
| AIRI | PnP |  |  | done |
| R2D2 | Deep Learning |  |  |  |
| PRIMO | Deep Learning |  |  |  |

## Radio Interferometry (VLBI) (`radio_interferometry`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CLEAN | Classical |  |  | done |
| AIRI | PnP |  |  | done |
| R2D2 | Deep Learning |  |  |  |
| PRIMO | Deep Learning |  |  |  |

## Raman Imaging / Microscopy (`raman_imaging`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical |  |  | done |
| Baseline Correction | Classical |  |  | done |
| SVD | Classical |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CDAE | Deep Learning |  |  |  |
| U-Net-Spectra | Deep Learning |  |  |  |
| Cascade-UNet | Deep Learning |  |  |  |
| PINN-Spectra | Deep Learning |  |  |  |
| SpectraFormer | Vision Transformer |  |  |  |
| DiffusionSpectra | Diffusion |  |  |  |
| ScoreSpectra | Score-based |  |  |  |

## SAR (`sar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Matched Filter | Classical |  |  | done |
| Range-Doppler | Classical |  |  | done |
| Chirp Scaling | Classical |  |  | done |
| SAR-BM3D | PnP |  |  | done |
| Lee Filter | PnP |  |  | done |
| SAR-DRN | Deep Learning |  |  |  |
| SAR-ResNet | Deep Learning |  |  |  |
| SAR-CAM | Transformer |  |  |  |
| SARFormer | Vision Transformer |  |  |  |
| PanSharpener++ | Deep Learning |  |  |  |
| SARDenoiserViT | Vision Transformer |  |  |  |
| DiffusionSAR | Diffusion |  |  |  |
| ScoreSAR | Score-based |  |  |  |

## Small-Angle X-ray Scattering (SAXS) (`saxs`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PyFAI-Integrate | Classical | 24.0 dB | 0.67 | done |
| McSAS | Classical | 27.5 dB | 0.79 | done |
| ScatterNet | Deep Learning | 31.5 dB | 0.895 |  |
| ScatterFormer | Transformer | 33.5 dB | 0.925 |  |

## SD-CASSI (`sd_cassi`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| GAP-TV | Classical |  |  | done |
| PnP-HSICNN | PnP |  |  | done |
| HDNet | Deep Learning |  |  |  |
| MST-L | Transformer |  |  |  |

## Seismic Tomography (`seismic_tomo`) — experimental_science

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Tikhonov | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| Matched Filter | Classical |  |  | done |
| PnP-RED | PnP |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| ResUNet | Deep Learning |  |  |  |
| Domain-Adapted-CNN | Deep Learning |  |  |  |
| SwinIR | Vision Transformer |  |  |  |
| ExpFormer | Vision Transformer |  |  |  |
| DiffusionExperimental | Diffusion |  |  |  |
| ScoreExperimental | Score-based |  |  |  |

## SEM (`sem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener Filter | Classical |  |  | done |
| BM3D | PnP |  |  | done |
| Noise2Void | Deep Learning |  |  |  |
| SwinIR | Transformer |  |  |  |

## Shearography (`shearography`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Goldstein MCF | Classical | 24.0 dB | 0.67 |  |
| PnP-Phase | PnP | 28.0 dB | 0.8 |  |
| ShearNet | Deep Learning | 32.0 dB | 0.9 |  |
| PhaseFormer | Transformer | 34.0 dB | 0.935 |  |

## Second Harmonic Generation (SHG) Microscopy (`shg`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## SIM (`sim`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener-SIM | Classical | 28.5 dB | 0.82 |  |
| PnP-SIM | PnP | 31.5 dB | 0.89 |  |
| DL-SIM | Deep Learning | 35.0 dB | 0.945 |  |
| SIMformer | Transformer | 36.5 dB | 0.96 |  |

## Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical |  |  | done |
| Baseline Correction | Classical |  |  | done |
| SVD | Classical |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CDAE | Deep Learning |  |  |  |
| U-Net-Spectra | Deep Learning |  |  |  |
| Cascade-UNet | Deep Learning |  |  |  |
| PINN-Spectra | Deep Learning |  |  |  |
| SpectraFormer | Vision Transformer |  |  |  |
| DiffusionSpectra | Diffusion |  |  |  |
| ScoreSpectra | Score-based |  |  |  |

## Solar EUV/X-ray Imaging (`solar_imaging`) — astronomy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Pixon | PnP |  |  | done |
| DeepEM | Deep Learning |  |  |  |
| SolarFormer | Transformer |  |  |  |

## Sonar (`sonar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DAS | Classical |  |  | done |
| MVDR/Capon | Classical |  |  | done |
| SonarNet | Deep Learning |  |  |  |
| AcousticFormer | Transformer |  |  |  |

## SPC-Block (`spc_block`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FISTA-TV | Classical |  |  | done |
| PnP-DRUNet | PnP |  |  | done |
| HATNet | Deep Learning |  |  |  |
| ISTA-Net | Deep Unfolding |  |  |  |

## SPC-Kronecker (`spc_kronecker`) — compressive

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FISTA-TV | Classical |  |  | done |
| PnP-DRUNet | PnP |  |  | done |
| HATNet | Deep Learning |  |  |  |
| ISTA-Net | Deep Unfolding |  |  |  |

## SPECT (`spect`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP-PET | Classical |  |  | done |
| OSEM | Classical |  |  | done |
| ML-EM | Classical |  |  | done |
| MAPEM-RDP | PnP |  |  | done |
| OS-EM | Classical |  |  | done |
| DeepPET | Deep Learning |  |  |  |
| U-Net-PET | Deep Learning |  |  |  |
| TransEM | Transformer |  |  |  |
| PET-ViT | Vision Transformer |  |  |  |
| PETFormer | Vision Transformer |  |  |  |

## SPECT/CT (`spect_ct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| OSEM | Classical |  |  | done |
| AC-OSEM | Classical |  |  | done |
| MAP-OSEM | PnP |  |  | done |
| DL-SPECT | Deep Learning |  |  |  |

## Spectral CT (`spectral_ct`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical |  |  | done |
| TV-ADMM | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| RED-CNN | Deep Learning |  |  |  |
| Learned Primal-Dual | Deep Unrolling |  |  |  |
| DuDoTrans | Deep Unrolling |  |  |  |
| CT-ViT | Vision Transformer |  |  |  |
| CTFormer | Transformer |  |  |  |
| DOLCE | Diffusion |  |  |  |
| DiffusionCT | Diffusion |  |  |  |
| Score-CT | Score-based |  |  |  |

## Spinning Disk Confocal Microscopy (`spinning_disk`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## Stimulated Raman Scattering (SRS) Microscopy (`srs`) — spectroscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| SG-ALS | Classical |  |  | done |
| Baseline Correction | Classical |  |  | done |
| SVD | Classical |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CDAE | Deep Learning |  |  |  |
| U-Net-Spectra | Deep Learning |  |  |  |
| Cascade-UNet | Deep Learning |  |  |  |
| PINN-Spectra | Deep Learning |  |  |  |
| SpectraFormer | Vision Transformer |  |  |  |
| DiffusionSpectra | Diffusion |  |  |  |
| ScoreSpectra | Score-based |  |  |  |

## STED (`sted`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## STEM (`stem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener Filter | Classical |  |  | done |
| BM3D | PnP |  |  | done |
| Noise2Void | Deep Learning |  |  |  |
| SwinIR | Transformer |  |  |  |

## Scanning Tunneling Microscopy (STM) (`stm`) — scanning_probe

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| BTR | Classical |  |  | done |
| MLE Reconstruction | Classical |  |  | done |
| Reg-Deconv | PnP |  |  | done |
| TV-Deconvolution | PnP |  |  | done |
| DeepSPM | Deep Learning |  |  |  |
| U-Net-SPM | Deep Learning |  |  |  |
| E2E-BTR | Deep Learning |  |  |  |
| SPM-Former | Vision Transformer |  |  |  |
| DiffusionSPM | Diffusion |  |  |  |
| ScoreSPM | Score-based |  |  |  |

## Streak Camera Imaging (`streak_camera`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TwIST | Classical |  |  | done |
| Temporal Filtering | Classical |  |  | done |
| PnP-FFDNet | PnP |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| CUP-Net | Deep Learning |  |  |  |
| Temporal-U-Net | Deep Learning |  |  |  |
| AL-DL | Deep Unrolling |  |  |  |
| Unfolded-CUP | Deep Unrolling |  |  |  |
| UltraFormer | Vision Transformer |  |  |  |
| DiffusionUltrafast | Diffusion |  |  |  |
| ScoreUltrafast | Score-based |  |  |  |

## Structured Light (`structured_light`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phase Shifting | Classical |  |  | done |
| Gray Code | Classical |  |  | done |
| FPP-Net | Deep Learning |  |  |  |
| PhaseFormer | Transformer |  |  |  |

## Susceptibility-Weighted Imaging (SWI) (`swi`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Zero-Filled IFFT | Classical |  |  | done |
| SENSE | Classical |  |  | done |
| GRAPPA | Classical |  |  | done |
| L1-Wavelet | Compressed Sensing |  |  | done |
| k-t SPARSE-SENSE | Compressed Sensing |  |  | done |
| ESPIRiT | Compressed Sensing |  |  | done |
| LORAKS | Compressed Sensing |  |  | done |
| BM3D-MRI | PnP |  |  | done |
| ALOHA | Low-Rank |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| PnP-DnCNN-Pro | PnP |  |  | done |
| Deep-ADMM-Net | Deep Unrolling |  |  |  |
| DCCNN | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| U-Net++ | Deep Learning |  |  |  |
| MoDL | Deep Unrolling |  |  |  |
| MoDL-Net++ | Deep Unrolling |  |  |  |
| E2E-VarNet | Deep Unrolling |  |  |  |
| HybridCascade | Deep Unrolling |  |  |  |
| HybridCascade++ | Deep Unrolling |  |  |  |
| SwinMR | Transformer |  |  |  |
| SwinMR++ | Transformer |  |  |  |
| HUMUS-Net | Transformer |  |  |  |
| HUMUS-Net++ | Transformer |  |  |  |
| ReconFormer | Transformer |  |  |  |
| ReconFormer++ | Transformer |  |  |  |
| Score-MRI | Score-Based |  |  |  |
| PromptMR | Deep Unrolling |  |  |  |
| MRI-DiffusionNet | Diffusion |  |  |  |
| MRDynamo | Physics-Informed |  |  |  |
| BrainID-MRI | Foundation Model |  |  |  |
| MMR-Mamba | Physics-Informed |  |  |  |
| PromptMR-SFM | Physics-Informed |  |  |  |
| MR-IPT | Foundation Model |  |  |  |
| MRI-FM | Foundation Model |  |  |  |

## Talbot-Lau X-ray Grating Interferometry (`talbot_lau`) — coherent

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phase Stepping | Classical |  |  | done |
| PCA Retrieval | Classical |  |  | done |
| DPC-Net | Deep Learning |  |  |  |
| GratingFormer | Transformer |  |  |  |

## TEM (`tem`) — electron_microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Wiener Filter | Classical |  |  | done |
| BM3D | PnP |  |  | done |
| Noise2Void | Deep Learning |  |  |  |
| SwinIR | Transformer |  |  |  |

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
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## TIRF (`tirf`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## ToF Camera (`tof_camera`) — depth_imaging

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Phase Unwrap | Classical | 24.0 dB | 0.66 | done |
| PnP-ToF | PnP | 28.0 dB | 0.8 | done |
| DeepToF | Deep Learning | 32.5 dB | 0.9 |  |
| MPI-Former | Transformer | 34.0 dB | 0.93 |  |

## Two-Photon (`two_photon`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| TFM | Classical | 25.0 dB | 0.71 | done |
| SAFT | Classical | 28.0 dB | 0.81 | done |
| UTPA-Net | Deep Learning | 32.5 dB | 0.905 |  |
| FMC-Former | Transformer | 34.5 dB | 0.94 |  |

## Ultrasound (`ultrasound`) — medical_ultrasound

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| DAS | Classical |  |  | done |
| DAS-CF | Classical |  |  | done |
| PW-DAS | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-TV | PnP |  |  | done |
| ABLE | Deep Learning |  |  |  |
| MU-Net | Deep Learning |  |  |  |
| Phase-ADMM-Net | Deep Unrolling |  |  |  |
| UltrasoundFormer | Vision Transformer |  |  |  |
| BeamFormer | Transformer |  |  |  |
| AttentionBeam | Transformer |  |  |  |
| BeamDATA | Deep Learning |  |  |  |
| DiffUS | Diffusion |  |  |  |
| ScoreUS | Score-based |  |  |  |

## US/MRI Fusion (`us_mri`) — multi_modal_fusion

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Demons | Classical |  |  | done |
| B-spline FFD | Classical |  |  | done |
| VoxelMorph | Deep Learning |  |  |  |
| TransMorph | Transformer |  |  |  |

## Wide-Angle X-ray Scattering (WAXS) (`waxs`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| PyFAI-Integrate | Classical | 23.5 dB | 0.65 | done |
| Rietveld-WAXS | Classical | 27.0 dB | 0.78 | done |
| WAXS-Net | Deep Learning | 31.0 dB | 0.89 |  |
| CrystalFormer | Transformer | 33.0 dB | 0.92 |  |

## Weather / Doppler Radar (`weather_radar`) — remote_sensing

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Pulse-Pair Doppler | Classical | 24.0 dB | 0.67 | done |
| CLEAN-AP | Classical | 27.5 dB | 0.79 | done |
| RainNet | Deep Learning | 31.8 dB | 0.9 |  |
| Earthformer | Transformer | 33.5 dB | 0.935 |  |

## Widefield (`widefield`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## Widefield Low-Dose (`widefield_lowdose`) — microscopy

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | Classical |  |  | done |
| Wiener Filter | Classical |  |  | done |
| TV-Deconvolution | Classical |  |  | done |
| PnP-FISTA | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| CARE | Deep Learning |  |  |  |
| U-Net | Deep Learning |  |  |  |
| ResUNet | Deep Learning |  |  |  |
| Restormer | Vision Transformer |  |  |  |
| DeconvFormer | Vision Transformer |  |  |  |
| Restormer+ | Vision Transformer |  |  |  |
| DiffDeconv | Diffusion |  |  |  |
| ScoreMicro | Score-based |  |  |  |

## XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`) — ultrafast

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| CrystFEL | Classical |  |  | done |
| EMC | Classical |  |  | done |
| CNN Hit-Finder | Deep Learning |  |  |  |
| CrysFormer | Transformer |  |  |  |

## X-ray Crystallography (`xray_crystallography`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Molecular Replacement | Classical | 22.0 dB | 0.59 | done |
| SHELXD | Classical | 26.0 dB | 0.74 | done |
| DL-Phase | Deep Learning | 30.5 dB | 0.88 |  |
| CrystFormer | Transformer | 32.5 dB | 0.915 |  |

## X-ray NDT (Radiography) (`xray_ndt`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| DR-GAN | Deep Learning |  |  |  |

## X-ray Radiography (`xray_radiography`) — medical

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FBP | Classical |  |  | done |
| TV-ADMM | Classical |  |  | done |
| PnP-ADMM | PnP |  |  | done |
| PnP-DnCNN | PnP |  |  | done |
| FBPConvNet | Deep Learning |  |  |  |
| RED-CNN | Deep Learning |  |  |  |
| Learned Primal-Dual | Deep Unrolling |  |  |  |
| DuDoTrans | Deep Unrolling |  |  |  |
| CT-ViT | Vision Transformer |  |  |  |
| CTFormer | Transformer |  |  |  |
| DOLCE | Diffusion |  |  |  |
| DiffusionCT | Diffusion |  |  |  |
| Score-CT | Score-based |  |  |  |

## X-ray Fluorescence (XRF) Imaging (`xrf_imaging`) — industrial_inspection

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| FP-Quantify | Classical | 24.5 dB | 0.68 | done |
| PnP-BM3D | PnP | 28.0 dB | 0.8 | done |
| XRF-UNet | Deep Learning | 32.0 dB | 0.9 |  |
| SpectraFormer | Transformer | 34.0 dB | 0.935 |  |

## X-ray Fluorescence Tomography (`xrf_tomo`) — scientific_instrumentation

| Algorithm | Type | Ref PSNR | Ref SSIM | Status |
|-----------|------|----------|----------|--------|
| Deconv | Classical |  |  | done |
| Calibration-Lookup | Classical |  |  | done |
| Peak Fitting | Classical |  |  | done |
| PnP-BM3D | PnP |  |  | done |
| PnP-NLM | PnP |  |  | done |
| ResNet-Calib | Deep Learning |  |  |  |
| Instrument-CNN | Deep Learning |  |  |  |
| CalibFormer | Vision Transformer |  |  |  |
| MassSpecFormer | Vision Transformer |  |  |  |
| DiffusionInstrumentation | Diffusion |  |  |  |
| ScoreInstrumentation | Score-based |  |  |  |

---

## Summary

| Category | Count |
|----------|-------|
| CPU algorithms verified (done) | 485 |
| CPU algorithms pending verification | 90 |
| GPU algorithms pending | 979 |
| Total | 1554 |
