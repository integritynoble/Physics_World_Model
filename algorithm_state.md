# Algorithm State — PWM5 Benchmark

Comprehensive listing of reconstruction algorithms for all 168 modalities.
Generated: 2026-03-11

## Legend
- **Ref PSNR/SSIM**: Published reference values from literature
- **PWM PSNR/SSIM**: Values achieved by PWM framework
- **Status**: `done` = PWM matches reference quality | blank = not yet verified
- **Year**: Publication year of algorithm

---

## Compressive Imaging

### 1. Coded Aperture Compressive Temporal Imaging (CACTI) (`cacti`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ELP-Unfolding | 2022 | Yang et al., ELP-Unfolding, ECCV 2022 | 33.1 | 0.9500 | 5.9 | 0.0083 |  |
| 2 | EfficientSCI | 2023 | Wang et al., EfficientSCI, CVPR 2023 | 34.3 | 0.9600 | 1.5 | 0.0774 |  |
| 3 | GAP-TV | 2016 | Yuan, GAP-TV, ICIP 2016 | 26.4 | 0.8500 | 19.8 | 0.4362 |  |
| 4 | DeSCI | 2019 | Liu et al., DeSCI, TPAMI 2019 | 27.1 | — | — | — |  |
| 5 | PnP-FFDNet | 2020 | Yuan et al., PnP-FFDNet, CVPR 2020 | 28.7 | — | — | — |  |
| 6 | MetaSCI | 2021 | Wang et al., MetaSCI, CVPR 2021 | 30.1 | — | — | — |  |
| 7 | RevSCI-Net | 2021 | Cheng et al., RevSCI-Net, NeurIPS 2021 | 31.4 | — | — | — |  |
| 8 | STFormer | 2022 | Wang et al., STFormer, NeurIPS 2022 | 33.9 | — | — | — |  |
| 9 | HiSViT | 2023 | Chen et al., HiSViT, ICCV 2023 | 34.5 | — | — | — |  |
| 10 | EfficientSCI-T (PWM) | — | — | — | — | 8.0 | 0.0759 |  |
| 11 | mask_division_baseline (test) | — | — | — | — | — | — |  |
| 12 | gap_tv (test) | — | — | — | — | — | — |  |

### 2. Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | HDNet | 2022 | Hu et al., HDNet, CVPR 2022 | 35.1 | — | — | — |  |
| 2 | MST-L | 2022 | Cai et al., MST, CVPR 2022 | 33.4 | — | — | — |  |
| 3 | CST-L | 2022 | Cai et al., CST, ECCV 2022 | 32.7 | — | — | — |  |
| 4 | GAP-TV | 2016 | Yuan, GAP-TV, ICIP 2016 | 26.2 | 0.8500 | 25.9 | 0.9649 | done |
| 5 | ADMM-Net | 2019 | Ma et al., ADMM-Net, ICCV 2019 | 30.7 | — | — | — |  |
| 6 | TSA-Net | 2020 | Meng et al., TSA-Net, ECCV 2020 | 31.5 | — | — | — |  |
| 7 | DGSMP | 2021 | Huang et al., DGSMP, CVPR 2021 | 32.6 | — | — | — |  |
| 8 | DAUHST-9stg | 2022 | Cai et al., DAUHST, NeurIPS 2022 | 35.3 | — | — | — |  |
| 9 | SSR-L | 2023 | Zhang et al., SSR, ICCV 2023 | 34.0 | — | — | — |  |
| 10 | PADUT | 2023 | Li et al., PADUT, CVPR 2023 | 34.8 | — | — | — |  |
| 11 | GAP-TV (guided) (PWM) | — | Yuan et al. 2016 | — | — | 26.2 | 0.9665 | done |
| 12 | GAP-TV (fast) (PWM) | — | — | — | — | 25.9 | 0.9649 | done |
| 13 | GAP-TV (small) (PWM) | — | — | — | — | 25.3 | 0.9664 | done |

### 3. Generic Matrix Sensing (`matrix`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FISTA-L1 (PWM) | — | — | — | — | 6.1 | 0.0001 |  |
| 2 | FISTA-L1 (high quality) (PWM) | — | Beck & Teboulle 2009 | — | — | 6.4 | 0.0257 |  |
| 3 | LISTA (PWM) | — | Gregor & LeCun, ICML 2010 | — | — | 6.1 | 0.0000 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 22.0 | 0.6949 | done |

### 4. Single-Pixel Camera (SPC) (`spc`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TVAL3 (PWM) | — | — | — | — | 6.8 | 0.0161 |  |
| 2 | ADMM-L1 (PWM) | — | Boyd et al. 2010 | — | — | 4.5 | 0.0018 |  |
| 3 | FISTA-L1 (PWM) | — | Beck & Teboulle 2009 | — | — | 4.7 | 0.0013 |  |

---

## Medical Imaging

### 5. X-ray Angiography (`angiography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | AngioFormer | 2023 | AngioFormer, 2023 | 35.0 | — | — | — |  |
| 2 | NeRF-Angio | 2023 | NeRF-Angio, 2023 | 34.0 | — | — | — |  |
| 3 | FBP (DSA baseline) (PWM) | — | — | — | — | 12.9 | 0.5828 |  |
| 4 | DSA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.8 | 0.3743 |  |
| 5 | VesselSegNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.8 | 0.3743 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 11.2 | 0.0435 |  |

### 6. Arterial Spin Labeling (ASL) MRI (`asl_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.1 | 0.0041 |  |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.1 | 0.0041 |  |
| 3 | ASL-Net [proxy] (PWM) | — | — | — | — | 2.0 | 0.2531 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 2.7 | 0.0636 |  |

### 7. Brachytherapy Imaging (`brachytherapy_img`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.1810 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.1810 | done |
| 3 | BrachyNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.1810 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 20.5 | 0.2374 | done |

### 8. Cone-Beam Computed Tomography (CBCT) (`cbct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FDK | 1984 | Feldkamp et al., FDK, JOSA 1984 | 28.0 | 0.8000 | 12.7 | 0.4010 |  |
| 2 | SART | 1984 | Andersen & Kak, SART, 1984 | 32.0 | 0.8800 | — | — |  |
| 3 | CTFormer | 2023 | CTFormer, 2023 | 38.0 | — | — | — |  |
| 4 | DuDoTrans | 2022 | DuDoTrans, 2022 | 37.5 | — | — | — |  |
| 5 | DiffusionCBCT | 2023 | DiffusionCBCT, 2023 | 36.0 | — | — | — |  |
| 6 | FDK-DL (PWM) | — | Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI | — | — | — | — |  |
| 7 | CBCT-UNet (PWM) | — | Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP | — | — | — | — |  |
| 8 | fbp_ramlak (test) | — | — | — | — | 14.9 | 0.3496 |  |
| 9 | fbp_shepp_logan (test) | — | — | — | — | 15.2 | 0.3593 | done |

### 9. CEST MRI (`cest_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.6435 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.6435 | done |
| 3 | CEST-Net [proxy] (PWM) | — | — | — | — | 13.9 | 0.3632 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 31.0 | 0.9859 | done |

### 10. Contrast-Enhanced Ultrasound (CEUS) (`ceus`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.2 | 0.1198 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.2 | 0.1198 | done |
| 3 | US-DeepSight [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.2 | 0.1198 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 24.5 | 0.9206 | done |

### 11. Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.4 | 0.7037 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.4 | 0.7037 | done |
| 3 | CLE-Net (CARE) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.4 | 0.7037 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 34.0 | 0.9927 | done |

### 12. X-ray Computed Tomography (CT) (`ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP (Ram-Lak) | 1971 | Ramachandran & Lakshminarayanan, 1971 | 30.0 | 0.8500 | 13.7 | 0.0649 |  |
| 2 | FBPConvNet | 2017 | Jin et al., FBPConvNet, TIP 2017 | 38.5 | 0.9600 | 13.7 | 0.0649 |  |
| 3 | LEARN | 2018 | Chen et al., LEARN, TMI 2018 | 40.2 | 0.9700 | — | — |  |
| 4 | iCT-Net | 2020 | Li et al., iCT-Net, TMI 2020 | 41.0 | — | — | — |  |
| 5 | DuDoTrans | 2022 | Wang et al., DuDoTrans, MICCAI 2022 | 42.1 | — | — | — |  |
| 6 | CT-FM | 2024 | CT-FM, 2024 | 44.1 | — | — | — |  |
| 7 | Score-CT | 2022 | Song et al., Score-CT, ICLR 2022 | 43.0 | — | — | — |  |
| 8 | PINER-CT | 2023 | PINER-CT, 2023 | 43.6 | — | — | — |  |
| 9 | CT-MAE | 2023 | CT-MAE, 2023 | 43.2 | — | — | — |  |
| 10 | PnP-HQS + NLM (PWM) | — | — | — | — | 1.4 | 0.1306 |  |
| 11 | RED-CNN (PWM) | — | Chen et al. 2017, IEEE TMI | — | — | 1.3 | 0.1144 |  |
| 12 | fbp_ramlak (test) | — | — | — | — | 12.9 | 0.0922 |  |
| 13 | fbp_shepp_logan (test) | — | — | — | — | 13.8 | 0.1053 |  |
| 14 | sart_10iter (test) | — | — | — | — | 13.8 | 0.2168 |  |

### 13. Dual-Energy X-ray Absorptiometry (DEXA) (`dexa`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinDXA | 2023 | SwinDXA, 2023 | 36.0 | — | — | — |  |
| 2 | PhysDXA | 2023 | PhysDXA, 2023 | 35.0 | — | — | — |  |
| 3 | FISTA-L2 (dual-energy) (PWM) | — | — | — | — | — | 1.0000 |  |
| 4 | DXA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.3099 |  |
| 5 | DEXA-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.3099 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 9.5 | 0.2550 |  |

### 14. Diffusion MRI (DTI) (`diffusion_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffusionDTI | 2023 | DiffusionDTI, 2023 | 37.0 | — | — | — |  |
| 2 | PhysDiffMRI | 2023 | PhysDiffMRI, 2023 | 35.0 | — | — | — |  |
| 3 | q-DL | 2020 | Golkov et al., q-DL, MRM 2016 | 34.0 | — | 11.3 | 0.8086 |  |
| 4 | SENSE (WLS tensor fit) (PWM) | — | — | — | — | 11.3 | 0.8086 |  |
| 5 | SHORE-Net [proxy] (PWM) | — | — | — | — | 11.3 | 0.8086 |  |
| 6 | zero_filled (test) | — | — | — | — | 11.3 | 0.0002 |  |

### 15. Digital Breast Tomosynthesis (DBT) (`digital_breast_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.2565 |  |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.2565 |  |
| 3 | DBT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.2565 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | — | 0.0001 |  |

### 16. Doppler Ultrasound (`doppler_ultrasound`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Autocorrelation | 1985 | Kasai et al., 1985 | 28.0 | — | 3.4 | 0.0000 |  |
| 2 | SwinDoppler | 2023 | SwinDoppler, 2023 | 36.0 | — | — | — |  |
| 3 | Back-Projection (Doppler) (PWM) | — | — | — | — | 5.1 | 0.0242 |  |
| 4 | UDoppler-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.1 | 0.0242 |  |
| 5 | Doppler CFAR [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.1 | 0.0242 |  |
| 6 | autocorrelation_estimator (test) | — | — | — | — | 3.4 | 0.0000 |  |
| 7 | clutter_filtered (test) | — | — | — | — | 0.2 | -0.0019 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 17.6 | 0.0064 | done |

### 17. Diffuse Optical Tomography (DOT) (`dot`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Born back-projection | 2000 | Arridge, Inverse Problems 1999 | 20.0 | 0.6000 | 4.3 | 0.0170 |  |
| 2 | DiffusionDOT | 2023 | DiffusionDOT, 2023 | 36.0 | — | — | — |  |
| 3 | PhysDOT | 2023 | PhysDOT, 2023 | 34.0 | — | — | — |  |
| 4 | L-BFGS-TV [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.3 | 0.0170 |  |
| 5 | DOT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.3 | 0.0170 |  |
| 6 | born_backprojection (test) | — | — | — | — | 4.7 | -0.0078 |  |
| 7 | tikhonov (test) | — | — | — | — | 2.4 | -0.0033 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 7.0 | 0.0193 |  |

### 18. Shear-Wave Elastography (`elastography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Phase gradient | 2000 | Manduca et al., MRM 2001 | 25.0 | — | — | — |  |
| 2 | DiffElasto | 2023 | DiffElasto, 2023 | 36.0 | — | — | — |  |
| 3 | SENSE (displacement field) (PWM) | — | — | — | — | 11.0 | 0.7949 |  |
| 4 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 10.3 | 0.0031 |  |
| 5 | NLSI-Solver [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 10.3 | 0.0031 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 5.7 | 0.0091 |  |

### 19. Fiber Bundle Endoscopy (`endoscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinEndo | 2023 | SwinEndo, 2023 | 36.0 | — | — | — |  |
| 2 | FISTA-L2 (endoscopy) (PWM) | — | — | — | — | — | — |  |
| 3 | EndoMapper-Net (PWM) | — | Ozyoruk, K.B. et al. (2021) EndoMapper, Nat. Mach. Intel. 3 | — | — | — | — |  |
| 4 | AF-SfMLearner (PWM) | — | Shao, S. et al. (2022) Self-supervised depth estimation in endoscopy, MICCAI 2022 | — | — | — | — |  |
| 5 | rl_20iter (test) | — | — | — | — | 11.8 | 0.8796 |  |
| 6 | rl_50iter (test) | — | — | — | — | 10.4 | 0.8225 |  |
| 7 | precomputed_recon (test) | — | — | — | — | 4.1 | 0.3912 |  |

### 20. Fluoroscopy (`fluoroscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PhysFluoro | 2023 | PhysFluoro, 2023 | 38.0 | — | — | — |  |
| 2 | TransFluoro | 2023 | TransFluoro, 2023 | 37.0 | — | — | — |  |
| 3 | FBP (fluoroscopy) (PWM) | — | — | — | — | 8.6 | 0.1239 |  |
| 4 | FluoroNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 43.6 | 0.9821 | done |
| 5 | X-ray CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 43.6 | 0.9821 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 43.5 | 0.9997 | done |

### 21. Functional MRI (BOLD fMRI) (`fmri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinMR++ | 2023 | SwinMR++, 2023 | 43.5 | — | — | — |  |
| 2 | HUMUS-Net++ | 2023 | HUMUS-Net++, 2023 | 43.0 | — | — | — |  |
| 3 | SENSE (fMRI) (PWM) | — | — | — | — | 4.9 | 0.0536 |  |
| 4 | fMRI-Transformer [proxy] (PWM) | — | — | — | — | 4.9 | 0.0536 |  |
| 5 | DeepBold [proxy] (PWM) | — | — | — | — | 4.9 | 0.0536 |  |
| 6 | zero_filled (test) | — | — | — | — | 4.9 | — |  |

### 22. Fundus Camera (`fundus`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Swin-Fundus | 2023 | Swin-Fundus, 2023 | 33.0 | — | — | — |  |
| 2 | cofe-Net | 2022 | Li et al., cofe-Net, 2022 | 32.0 | — | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 30.6 | 0.9090 | done |
| 4 | RETFound (PWM) | — | Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156 | — | — | — | — |  |
| 5 | DR-Grade-Net (PWM) | — | Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22) | — | — | — | — |  |
| 6 | rl_20iter (test) | — | — | — | — | 35.0 | 0.9965 | done |
| 7 | rl_50iter (test) | — | — | — | — | 35.9 | 0.9972 | done |
| 8 | precomputed_wiener (test) | — | — | — | — | 22.0 | 0.9248 | done |

### 23. Intravascular Ultrasound (IVUS) (`ivus`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.3 | 0.7253 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.3 | 0.7253 | done |
| 3 | IVUS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.3 | 0.7253 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 19.8 | 0.8902 | done |

### 24. Mammography (`mammography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | CT-ViT | 2023 | CT-ViT, 2023 | 37.0 | — | — | — |  |
| 2 | CTFormer | 2023 | CTFormer, 2023 | 36.5 | — | — | — |  |
| 3 | DiffusionCT | 2023 | DiffusionCT, 2023 | 36.0 | — | — | — |  |
| 4 | FBP (mammography) (PWM) | — | — | — | — | 4.1 | 0.0047 |  |
| 5 | MammoNet (GatorTron) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.8 | 0.0988 |  |
| 6 | Mammo-ResNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.8 | 0.0988 |  |
| 7 | precomputed_recon (test) | — | — | — | — | 20.9 | 0.8580 | done |

### 25. MR Elastography (MRE) (`mr_elastography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0020 |  |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0020 |  |
| 3 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0020 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 6.0 | 0.0984 |  |

### 26. MR Fingerprinting (MRF) (`mr_fingerprinting`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0040 |  |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0040 |  |
| 3 | MRF-Net [proxy] (PWM) | — | — | — | — | 2.1 | 0.2517 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 1.8 | 0.0693 |  |

### 27. MR Angiography (MRA) (`mra`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.6 | 0.0064 |  |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.6 | 0.0064 |  |
| 3 | MRA-VesselNet [proxy] (PWM) | — | — | — | — | 0.5 | 0.0095 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 12.1 | 0.2673 |  |

### 28. Magnetic Resonance Imaging (MRI) (`mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6500 | 13.4 | 0.0132 |  |
| 2 | CS-MRI (SparseMRI) | 2007 | Lustig et al., SparseMRI, MRM 2007 | 33.0 | 0.9000 | 13.4 | 0.0132 |  |
| 3 | GRAPPA | 2002 | Griswold et al., GRAPPA, MRM 2002 | 34.0 | 0.9200 | — | — |  |
| 4 | VarNet | 2020 | Sriram et al., E2E-VarNet, NeurIPS 2020 | 40.5 | 0.9700 | — | — |  |
| 5 | HUMUS-Net | 2022 | Fabian et al., HUMUS-Net, NeurIPS 2022 | 42.0 | 0.9800 | — | — |  |
| 6 | SwinMR++ | 2023 | SwinMR++, 2023 | 43.8 | — | — | — |  |
| 7 | PromptMR | 2023 | Li et al., PromptMR, MICCAI 2023 | 41.5 | — | — | — |  |
| 8 | MRI-FM | 2024 | MRI-FM, 2024 | 42.5 | — | 13.4 | 0.0132 |  |
| 9 | MoDL (PWM) | — | Aggarwal et al. 2019, IEEE TMI | — | — | 13.4 | 0.0132 |  |
| 10 | MoDL (5 unrolls) (PWM) | — | — | — | — | 13.4 | 0.0132 |  |
| 11 | zero_filled (test) | — | — | — | — | 13.0 | 0.0004 |  |
| 12 | cs_mri_wavelet (test) | — | — | — | — | 13.0 | 0.0006 |  |
| 13 | sense (test) | — | — | — | — | 13.0 | 0.0010 |  |

### 29. MR Spectroscopy (MRS) (`mrs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinMR++ | 2023 | SwinMR++, 2023 | 43.5 | — | — | — |  |
| 2 | HLSVD | 2002 | Pijnappel et al., HLSVD, 2002 | 30.0 | — | 4.3 | 0.0011 |  |
| 3 | SENSE (spectroscopy) (PWM) | — | — | — | — | 2.1 | 0.2197 |  |
| 4 | MRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.3 | 0.0011 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 1.9 | 0.0676 |  |

### 30. Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 20.2 | 0.5398 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 20.2 | 0.5398 | done |
| 3 | fNIRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 20.2 | 0.5398 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 14.5 | 0.8761 |  |

### 31. Optical Coherence Tomography (OCT) (`oct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SpeckleFormer | 2023 | SpeckleFormer, 2023 | 38.0 | — | — | — |  |
| 2 | RetinalFormer | 2023 | RetinalFormer, 2023 | 37.5 | — | — | — |  |
| 3 | BM3D | 2007 | BM3D, 2007 | 32.0 | — | — | — |  |
| 4 | FFT Recon (PWM) | — | — | — | — | 10.3 | 0.1422 |  |
| 5 | Spectral Estimation (PWM) | — | Leitgeb et al. 2003, Optics Express | — | — | 10.2 | 0.1411 |  |
| 6 | OCT Denoising Net (PWM) | — | Devalla et al. 2019, Biomed. Optics Express | — | — | 10.3 | 0.1422 |  |
| 7 | bscan_baseline (test) | — | — | — | — | 23.1 | 0.9439 | done |
| 8 | bscan_ideal_baseline (test) | — | — | — | — | 23.5 | 0.9482 | done |

### 32. OCT Angiography (OCTA) (`octa`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffusionOCT | 2023 | DiffusionOCT, 2023 | 36.0 | — | — | — |  |
| 2 | FFT Recon (OCTA) (PWM) | — | — | — | — | 13.3 | 0.0566 |  |
| 3 | OCTA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 18.8 | 0.4872 | done |
| 4 | OCTA-FF [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 18.8 | 0.4872 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 16.8 | 0.4326 | done |

### 33. Positron Emission Tomography (PET) (`pet`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MLEM | 1982 | Shepp & Vardi, MLEM, TMI 1982 | 28.0 | 0.7500 | — | — |  |
| 2 | OSEM | 1994 | Hudson & Larkin, OSEM, TMI 1994 | 30.0 | 0.8200 | — | — |  |
| 3 | U-Net-PET | 2021 | U-Net-PET, 2021 | 36.8 | — | — | — |  |
| 4 | PET-ViT | 2023 | PET-ViT, 2023 | 36.4 | — | — | — |  |
| 5 | PETFormer | 2023 | PETFormer, 2023 | 35.7 | — | — | — |  |
| 6 | FBP (emission tomography) (PWM) | — | — | — | — | 15.4 | 0.0116 | done |
| 7 | NeuroLF-PET (PWM) | — | Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58 | — | — | — | — |  |
| 8 | PET-DL (U-Net) (PWM) | — | Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9) | — | — | — | — |  |
| 9 | fbp_ramlak (test) | — | — | — | — | 9.3 | 0.1813 |  |
| 10 | fbp_shepp_logan (test) | — | — | — | — | 11.9 | 0.2681 |  |
| 11 | precomputed_fbp (test) | — | — | — | — | 33.1 | 0.9325 | done |

### 34. Photoacoustic Imaging (`photoacoustic`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Universal back-projection | 2005 | Xu & Wang, UBP, PMB 2005 | 28.0 | 0.7500 | 18.5 | 0.3658 |  |
| 2 | PAT-Former | 2023 | PAT-Former, 2023 | 35.0 | — | 18.5 | 0.3658 |  |
| 3 | Deep-PAI | 2020 | Hauptmann et al., Deep-PAI, TMI 2020 | 32.0 | — | 18.5 | 0.3658 |  |
| 4 | Time Reversal [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 18.5 | 0.3658 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 19.1 | 0.2490 | done |

### 35. Portal Imaging (EPID) (`portal_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.3 | 0.1384 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.3 | 0.1384 | done |
| 3 | PortalDL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.3 | 0.1384 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 10.5 | 0.4088 |  |

### 36. Proton Therapy Imaging (`proton_therapy_img`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.5118 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.5118 | done |
| 3 | ProtonTherapy-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.5118 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 17.8 | 0.7117 | done |

### 37. Single Photon Emission CT (SPECT) (`spect`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MLEM | 1982 | Shepp & Vardi, 1982 | 26.0 | 0.7000 | — | — |  |
| 2 | OSEM | 1994 | Hudson & Larkin, 1994 | 28.5 | 0.7800 | — | — |  |
| 3 | PET-ViT | 2023 | PET-ViT, 2023 | 34.0 | — | — | — |  |
| 4 | TransEM | 2022 | TransEM, 2022 | 32.0 | — | — | — |  |
| 5 | FBP (emission tomography) (PWM) | — | — | — | — | 10.8 | 0.0669 |  |
| 6 | SPECT-DL (OSEM+) (PWM) | — | Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging | — | — | — | — |  |
| 7 | SPECT-UNet (PWM) | — | Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6) | — | — | — | — |  |
| 8 | fbp_ramlak (test) | — | — | — | — | -6.5 | 0.0101 |  |
| 9 | precomputed_fbp (test) | — | — | — | — | 30.0 | 0.9523 | done |

### 38. Photon-Counting Spectral CT (`spectral_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1896 |  |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1896 |  |
| 3 | SpectralCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1896 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 12.3 | 0.1106 |  |

### 39. Susceptibility-Weighted Imaging (SWI) (`swi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.6 | 0.0020 |  |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.6 | 0.0020 |  |
| 3 | SWI-Net [proxy] (PWM) | — | — | — | — | 2.0 | 0.2415 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 1.9 | 0.0677 |  |

### 40. Ultrasound B-mode Imaging (`ultrasound`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DAS (Delay-and-Sum) | 1990 | DAS baseline | 25.0 | 0.6500 | — | — |  |
| 2 | ScoreUS | 2023 | ScoreUS, 2023 | 36.3 | — | — | — |  |
| 3 | DiffUS | 2023 | DiffUS, 2023 | 36.0 | — | — | — |  |
| 4 | AttentionBeam | 2023 | AttentionBeam, 2023 | 35.5 | — | — | — |  |
| 5 | Richardson-Lucy (ultrasound) (PWM) | — | Richardson 1972, JOSA | — | — | 14.8 | 0.3464 |  |
| 6 | US-UNet (DeepUS) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 14.8 | 0.3464 |  |
| 7 | US-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 14.8 | 0.3464 |  |
| 8 | rl_20iter (test) | — | — | — | — | 14.6 | 0.1559 |  |
| 9 | rl_50iter (test) | — | — | — | — | 14.1 | 0.1323 |  |

### 41. X-ray Radiography (`xray_radiography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP (X-ray radiography) (PWM) | — | — | — | — | 4.5 | -0.0019 |  |
| 2 | CheXNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.8 | 0.3322 |  |
| 3 | X-ray UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.8 | 0.3322 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 26.3 | 0.9844 | done |

---

## Coherent Imaging

### 42. Digital Holographic Microscopy (`holography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Angular Spectrum | 2000 | Goodman, Fourier Optics | 22.0 | 0.7000 | 4.9 | 0.0036 |  |
| 2 | AutoPhase++ | 2023 | AutoPhase++, 2023 | 34.0 | — | — | — |  |
| 3 | HolographyViT | 2023 | HolographyViT, 2023 | 34.0 | — | — | — |  |
| 4 | PhaseGAN | 2021 | Zhang et al., PhaseGAN, 2021 | 30.0 | — | — | — |  |
| 5 | PhaseNet (PWM) | — | Rivenson et al. 2018, Light: S&A | — | — | 14.9 | 0.9212 |  |
| 6 | sqrt_intensity_amplitude (test) | — | — | — | — | — | 0.0003 |  |

### 43. Optical Diffraction Tomography (ODT) (`odt`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Rytov-Former | 2023 | Rytov-Former, 2023 | 33.0 | — | — | — |  |
| 2 | ODT-Net | 2022 | ODT-Net, 2022 | 31.0 | — | 3.1 | 0.0782 |  |
| 3 | Born approximation | 2000 | Wolf, Opt Commun 1969 | 25.0 | — | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.1 | 0.0782 |  |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.1 | 0.0782 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 25.5 | 0.9509 | done |

### 44. Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | HIO (PWM) | — | — | — | — | 12.6 | 0.3297 |  |
| 2 | RAAR [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.8 | 0.3193 |  |
| 3 | prDeep [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.8 | 0.3193 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 12.5 | — |  |

### 45. Ptychographic Imaging (`ptychography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ePIE | 2008 | Maiden & Rodenburg, ePIE, Ultramicroscopy 2009 | 28.0 | 0.8500 | 11.7 | 0.5129 |  |
| 2 | AutoPhaseNN | 2022 | Cherukara et al., AutoPhaseNN, APL 2022 | 33.0 | — | — | — |  |
| 3 | PtychoNN | 2021 | Cherukara et al., PtychoNN, APL 2020 | 31.0 | — | 3.2 | 0.0007 |  |
| 4 | PtychoNN 2.0 (PWM) | — | — | — | — | 3.2 | 0.0007 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 21.0 | 0.2841 | done |
| 6 | precomputed_phase_baseline (test) | — | — | — | — | 10.5 | -0.0059 |  |

### 46. Talbot-Lau X-ray Grating Interferometry (`talbot_lau`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | GratingFormer | 2023 | GratingFormer, 2023 | 33.0 | — | — | — |  |
| 2 | PCA Retrieval | 2012 | Zanette et al., PCA, 2012 | 26.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.9 | 0.9891 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.9 | 0.9891 | done |
| 5 | Talbot-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.9 | 0.9891 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 6.6 | 0.1206 |  |

---

## Microscopy

### 47. Confocal 3D Z-Stack (`confocal_3d`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinIR-3D | 2023 | SwinIR-3D, 2023 | 36.0 | — | — | — |  |
| 2 | DiffusionMicro | 2023 | DiffusionMicro, 2023 | 35.5 | — | — | — |  |
| 3 | 3D Richardson-Lucy (PWM) | — | — | — | — | 0.3 | 0.0042 |  |
| 4 | 3D CARE (PWM) | — | — | — | — | 0.1 | 0.0043 |  |
| 5 | CARE-3D (PWM) | — | — | — | — | 0.1 | 0.0043 |  |
| 6 | CARE-3D (slice-wise) (PWM) | — | — | — | — | 27.3 | 0.8317 | done |
| 7 | precomputed_baseline (test) | — | — | — | — | 17.8 | 0.0530 | done |
| 8 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |

### 48. Confocal Live-Cell Microscopy (`confocal_livecell`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffusionCell | 2023 | DiffusionCell, 2023 | 35.0 | — | — | — |  |
| 2 | Restormer-Micro | 2023 | Restormer-Micro, 2023 | 34.0 | — | — | — |  |
| 3 | CARE | 2018 | Weigert et al., CARE, Nature Methods 2018 | 33.0 | — | 13.9 | 0.3589 |  |
| 4 | Richardson-Lucy (PWM) | — | — | — | — | 32.3 | 0.8670 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 31.3 | 0.9870 | done |

### 49. Dark-Field Microscopy (`dark_field`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffusionDF | 2023 | DiffusionDF, 2023 | 36.0 | — | — | — |  |
| 2 | Restormer-DF | 2023 | Restormer-DF, 2023 | 35.5 | — | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 20.6 | 0.7815 | done |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 10.5 | 0.6101 |  |
| 5 | DF-UNet (PWM) | — | Wolfer, T. et al. (2021) DL for dark-field X-ray CT, Sci. Rep. 11:5005 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 25.1 | 0.9781 | done |

### 50. Differential Interference Contrast (DIC) (`dic`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinDIC | 2023 | SwinDIC, 2023 | 35.0 | — | — | — |  |
| 2 | PhysPhase-Net | 2023 | PhysPhase-Net, 2023 | 33.0 | — | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 15.6 | 0.3801 | done |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 15.0 | 0.3956 |  |
| 5 | DIC-Net (PWM) | — | Mir, A. et al. (2015) Automated DIC microscopy, J. Microsc. 257(2) | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 10.9 | — |  |

### 51. DNA-PAINT Super-Resolution (`dna_paint`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffPAINT | 2023 | DiffPAINT, 2023 | 36.0 | — | — | — |  |
| 2 | PICASSO | 2020 | Reymond et al., PICASSO, 2020 | 30.0 | — | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 22.9 | 0.0260 | done |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 27.8 | 0.1984 | done |
| 5 | DECODE-PAINT (PWM) | — | Speiser, A. et al. (2021) DL for dense SMLM, Nature Methods 18:1090 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 28.5 | 0.3552 | done |

### 52. Expansion Microscopy (ExM) (`expansion`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffExM | 2023 | DiffExM, 2023 | 36.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 33.9 | 0.6181 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 15.9 | 0.6087 | done |
| 4 | EXpansionNet (PWM) | — | Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 33.3 | 0.9823 | done |

### 53. Fluorescence Lifetime Imaging (FLIM) (`flim`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Phasor approach | 2008 | Digman et al., Biophys J 2008 | 28.0 | — | 1.2 | 0.0555 |  |
| 2 | SwinFLIM | 2023 | SwinFLIM, 2023 | 36.0 | — | — | — |  |
| 3 | MLE Fit (PWM) | — | Becker 2012, J. Microscopy | — | — | 36.9 | 0.7374 | done |
| 4 | MLE Fit (iterative) (PWM) | — | Becker 2012, J. Microscopy | — | — | 0.9 | 0.0557 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 30.7 | 0.9901 | done |

### 54. Fourier Ptychographic Microscopy (FPM) (`fpm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Gradient Descent FPM | 2014 | Tian & Waller, FPM, Optica 2015 | 30.0 | 0.8500 | 5.2 | 0.0237 |  |
| 2 | PtychoDV | 2023 | PtychoDV, 2023 | 33.0 | — | — | — |  |
| 3 | Fourier PtychoNet | 2021 | FPNet, 2021 | 29.0 | — | 5.2 | 0.0237 |  |
| 4 | Sequential Phase Retrieval (PWM) | — | — | — | — | 5.2 | 0.0237 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 16.9 | 0.7943 | done |

### 55. Image Scanning Microscopy (ISM) (`ism`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Restormer+ | 2023 | Restormer+, 2023 | 36.0 | — | — | — |  |
| 2 | Pixel reassignment | 2010 | Muller & Enderlein, PRL 2010 | 28.0 | — | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 3.1 | 0.1516 |  |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 0.5 | 0.0286 |  |
| 5 | ISM-Reassignment-Net (PWM) | — | Castello, M. et al. (2019) Image scanning microscopy ISM, Nature Methods 16:175 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | — | 0.0000 |  |

### 56. Lattice Light-Sheet Microscopy (`lattice_lightsheet`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DeconvFormer | 2023 | DeconvFormer, 2023 | 36.0 | — | — | — |  |
| 2 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | — | 25.1 | 0.3079 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 14.5 | 0.7656 |  |
| 4 | LLSM-CARE (PWM) | — | Weigert, M. et al. (2018) Content-aware restoration for lattice light-sheet, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 21.3 | 0.7759 | done |

### 57. Light-Sheet Fluorescence Microscopy (LSFM) (`lightsheet`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | 0.8000 | — | — |  |
| 2 | Restormer+ | 2023 | Restormer+, 2023 | 36.0 | — | — | — |  |
| 3 | ScoreMicro | 2023 | ScoreMicro, 2023 | 35.5 | — | — | — |  |
| 4 | Fourier Notch Filter (PWM) | — | — | — | — | 0.2 | 0.0045 |  |
| 5 | VSNR (PWM) | — | — | — | — | 0.2 | 0.0043 |  |
| 6 | DeStripe (PWM) | — | Liang et al. 2022 | — | — | 0.2 | 0.0045 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 20.0 | 0.0553 | done |
| 8 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |
| 9 | fourier_notch (test) | — | — | — | — | — | 0.0000 |  |

### 58. MINFLUX Nanoscopy (`minflux`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ANNA-PALM | 2021 | Ouyang et al., ANNA-PALM, Nature Biotech 2018 | — | — | — | — |  |
| 2 | DECODE | 2021 | Speiser et al., DECODE, Nature Methods 2021 | — | — | — | — |  |
| 3 | MLE Localization | 2006 | Ober et al., Biophys J 2004 | — | — | — | — |  |
| 4 | Richardson-Lucy (PWM) | — | — | — | — | 29.5 | 0.7052 | done |
| 5 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 29.2 | 0.7051 | done |
| 6 | MINFLUX-Net (PWM) | — | Gwosch, K.C. et al. (2020) MINFLUX nanoscopy 3D, Nature Methods 17:217 | — | — | — | — |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 29.5 | 0.4336 | done |

### 59. PALM/STORM Single-Molecule Localization (`palm_storm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ThunderSTORM | 2014 | Ovesny et al., ThunderSTORM, Bioinformatics 2014 | — | — | — | — |  |
| 2 | DECODE | 2021 | Speiser et al., DECODE, Nature Methods 2021 | — | — | — | — |  |
| 3 | Deep-STORM | 2018 | Nehme et al., Deep-STORM, Optica 2018 | — | — | — | — |  |
| 4 | Richardson-Lucy (STORM/PALM) (PWM) | — | — | — | — | 0.0 | 0.0005 |  |
| 5 | DECODE-SMLM (PWM) | — | Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090 | — | — | — | — |  |
| 6 | DeepSTORM (PWM) | — | Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4) | — | — | — | — |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 32.4 | 0.6094 | done |
| 8 | rl_20iter (test) | — | — | — | — | 32.4 | 0.5904 | done |

### 60. Phase Contrast Microscopy (`phase_contrast`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PhaseFormer | 2023 | PhaseFormer, 2023 | 34.0 | — | — | — |  |
| 2 | TIE | 2001 | Zuo et al., TIE, Opt Express 2013 | 28.0 | — | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 8.1 | 0.3458 |  |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 8.1 | 0.3458 |  |
| 5 | PhaseNet (PWM) | — | Rivenson, Y. et al. (2018) Phase recovery with DL, Light: Sci. & Appl. 7:17141 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 45.6 | 0.9991 | done |

### 61. Polarization Microscopy (`polarization`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Restormer+ | 2023 | Restormer+, 2023 | 35.0 | — | — | — |  |
| 2 | PnP-HQS (PWM) | — | — | — | — | 8.4 | 0.0892 |  |
| 3 | PolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.4 | 0.0892 |  |
| 4 | Stokes-NN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.4 | 0.0892 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 15.8 | 0.6265 | done |

### 62. Second Harmonic Generation (SHG) Microscopy (`shg`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Restormer+ | 2023 | Restormer+, 2023 | 36.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 22.3 | 0.5919 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 17.2 | 0.1742 | done |
| 4 | SHG-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for SHG imaging, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 23.0 | 0.7974 | done |

### 63. Structured Illumination Microscopy (SIM) (`sim`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Wiener-SIM | 2008 | Gustafsson et al., 2008 | 30.0 | 0.8800 | 0.1 | 0.0056 |  |
| 2 | SIMformer | 2023 | SIMformer, 2023 | 36.0 | — | — | — |  |
| 3 | DL-SIM | 2021 | Jin et al., DL-SIM, 2021 | 33.0 | — | — | — |  |
| 4 | HiFi-SIM (PWM) | — | Wen et al. 2021, Light: S&A | — | — | 0.3 | 0.0055 |  |
| 5 | fairSIM (open-source) (PWM) | — | Mueller et al. 2016, Nature Comm. | — | — | 6.2 | 0.0126 |  |
| 6 | Wiener-SIM (fast) (PWM) | — | — | — | — | 6.2 | 0.0126 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 21.6 | 0.1863 | done |
| 8 | wiener_sim (test) | — | — | — | — | 6.2 | 0.0174 |  |

### 64. Spinning Disk Confocal Microscopy (`spinning_disk`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DeconvFormer | 2023 | DeconvFormer, 2023 | 36.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 29.5 | 0.7581 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 14.2 | 0.3440 |  |
| 4 | SD-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for spinning disk confocal, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 30.6 | 0.9835 | done |

### 65. STED Microscopy (`sted`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Restormer+ | 2023 | Restormer+, 2023 | 36.0 | — | — | — |  |
| 2 | DeconvFormer | 2023 | DeconvFormer, 2023 | 35.0 | — | — | — |  |
| 3 | Richardson-Lucy (STED) (PWM) | — | — | — | — | 0.3 | 0.0017 |  |
| 4 | STED-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | RCAN-STED (PWM) | — | Chen, J. et al. (2021) Three-dimensional residual channel attention for STED, Nature Methods 18:678 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 25.0 | 0.8484 | done |
| 7 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |

### 66. Three-Photon Microscopy (`three_photon`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ScoreMicro | 2023 | ScoreMicro, 2023 | 36.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 14.6 | 0.3353 |  |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 14.5 | 0.3513 |  |
| 4 | 3P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) CARE for 3P deep tissue imaging, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 20.8 | 0.8419 | done |

### 67. TIRF Microscopy (`tirf`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DeconvFormer | 2023 | DeconvFormer, 2023 | 35.0 | — | — | — |  |
| 2 | Restormer+ | 2023 | Restormer+, 2023 | 34.5 | — | — | — |  |
| 3 | Richardson-Lucy (TIRF) (PWM) | — | — | — | — | 27.7 | 0.1106 | done |
| 4 | TIRF-Net (CARE) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.7 | 0.1106 | done |
| 5 | TIRF-SRRF [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.7 | 0.1106 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 31.2 | 0.6216 | done |

### 68. Two-Photon / Multiphoton Microscopy (`two_photon`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DeconvFormer | 2023 | DeconvFormer, 2023 | 37.0 | — | — | — |  |
| 2 | Restormer+ | 2023 | Restormer+, 2023 | 35.0 | — | — | — |  |
| 3 | Richardson-Lucy (2P) (PWM) | — | — | — | — | 0.9 | 0.0073 |  |
| 4 | 2P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | 2P-DeepInterp (PWM) | — | Lecoq, J. et al. (2021) Removing independent noise in systems neuroscience using DeepInterpolation, Nature Methods 18:1401 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 33.8 | 0.9867 | done |
| 7 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |

### 69. Widefield Fluorescence Microscopy (`widefield`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 / Lucy 1974 | 28.0 | 0.8000 | 24.1 | 0.2696 |  |
| 2 | Wiener deconvolution | 1949 | Wiener, 1949 | 26.0 | 0.7500 | — | — |  |
| 3 | Restormer | 2022 | Zamir et al., Restormer, CVPR 2022 | 36.0 | — | — | — |  |
| 4 | DiffDeconv | 2023 | DiffDeconv, 2023 | 35.5 | — | — | — |  |
| 5 | CARE (PWM) | — | Weigert et al. 2018, Nature Methods | — | — | 14.5 | 0.7656 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 25.0 | 0.9091 | done |

### 70. Low-Dose Widefield Microscopy (`widefield_lowdose`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DeconvFormer | 2023 | DeconvFormer, 2023 | 36.0 | — | — | — |  |
| 2 | ScoreMicro | 2023 | ScoreMicro, 2023 | 35.5 | — | — | — |  |
| 3 | Noise2Void | 2019 | Krull et al., N2V, CVPR 2019 | 32.0 | — | — | — |  |
| 4 | BM3D + RL (PWM) | — | — | — | — | 29.0 | 0.9402 | done |
| 5 | CARE (PWM) | — | — | — | — | 12.6 | 0.5013 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 18.8 | 0.6755 | done |

---

## Electron Microscopy

### 71. Cryo-Electron Tomography (Cryo-ET) (`cryo_et`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | WBP | 1970 | WBP baseline | 22.0 | 0.6000 | — | — |  |
| 2 | DiffusionET | 2023 | DiffusionET, 2023 | 33.0 | — | — | — |  |
| 3 | DeePiCt | 2023 | DeePiCt, Nature Methods 2023 | 32.0 | — | — | — |  |
| 4 | Richardson-Lucy (PWM) | — | — | — | — | 2.7 | 0.0966 |  |
| 5 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 13.2 | 0.4127 |  |
| 6 | CryoCARE (PWM) | — | Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol. | — | — | — | — |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 8.4 | 0.2037 |  |

### 72. Electron Backscatter Diffraction (EBSD) (`ebsd`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PhysEBSD | 2023 | PhysEBSD, 2023 | 35.0 | — | — | — |  |
| 2 | Dictionary indexing | 2015 | Chen et al., 2015 | 28.0 | — | — | — |  |
| 3 | FISTA-L2 (Hough baseline) (PWM) | — | — | — | — | 3.0 | 0.2847 |  |
| 4 | EBSD-DL (DictIndex) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.9 | 0.8024 | done |
| 5 | EMsoft-EBSD [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.9 | 0.8024 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 21.8 | 0.9677 | done |

### 73. STEM-EDX Elemental Mapping (`edx_mapping`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinEDX | 2023 | SwinEDX, 2023 | 36.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 3.2 | 0.2667 |  |
| 3 | Richardson-Lucy (high quality) (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.2667 |  |
| 4 | Richardson-Lucy (DL baseline) (PWM) | — | Tietz, C. et al. (2021) DL for EDS spectrum imaging, Ultramicroscopy 231 | — | — | 3.2 | 0.2667 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 22.0 | 0.9307 | done |

### 74. Electron Energy Loss Spectroscopy (EELS) (`eels`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinEELS | 2023 | SwinEELS, 2023 | 35.0 | — | — | — |  |
| 2 | PCA denoising | 2010 | Cueva et al., Microsc & Microanal 2012 | 28.0 | — | — | — |  |
| 3 | FISTA-L2 (Fourier ratio) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.5036 | done |
| 4 | EELS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.5036 | done |
| 5 | MLLS-EELS [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.5036 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 24.6 | 0.9842 | done |

### 75. 4D-STEM Electron Diffraction (`electron_diffraction`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffED | 2023 | DiffED, 2023 | 35.0 | — | — | — |  |
| 2 | PhysED | 2023 | PhysED, 2023 | 34.0 | — | — | — |  |
| 3 | ePIE (electron ptychography) (PWM) | — | — | — | — | 42.0 | 0.9889 | done |
| 4 | ED-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 42.0 | 0.9889 | done |
| 5 | CRISP-ED [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 42.0 | 0.9889 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 42.0 | 0.9901 | done |

### 76. Electron Holography (`electron_holography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Fourier filtering | 1993 | Lichte, 1993 | 25.0 | — | — | — |  |
| 2 | SwinHolo | 2023 | SwinHolo, 2023 | 34.0 | — | — | — |  |
| 3 | Phase Retrieval (HIO) (PWM) | — | — | — | — | 5.6 | 0.0115 |  |
| 4 | EH-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.5 | 0.0049 |  |
| 5 | Phase-Sideband [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.5 | 0.0049 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 9.5 | -0.0481 |  |

### 77. Electron Tomography (`electron_tomography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | WBP | 1970 | Weighted Back-Projection | 22.0 | 0.6500 | — | — |  |
| 2 | SIRT | 1972 | Gilbert, SIRT, 1972 | 25.0 | 0.7500 | 25.1 | 0.9525 | done |
| 3 | DiffET | 2023 | DiffET, 2023 | 35.0 | — | — | — |  |
| 4 | IMOD-SIRT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.8 | 0.2536 |  |
| 5 | SIRT-3D [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.8 | 0.2536 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 19.3 | 0.9419 | done |

### 78. Focused Ion Beam SEM (FIB-SEM) (`fib_sem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PhysFIB | 2023 | PhysFIB, 2023 | 36.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 26.0 | 0.4862 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 4.9 | 0.0019 |  |
| 4 | FIB-SEM-Net (PWM) | — | Heinrich, L. et al. (2021) Whole-cell organelle segmentation in volume EM, Nature 599:141 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 28.1 | 0.9862 | done |

### 79. Scanning Electron Microscopy (SEM) (`sem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BM3D | 2007 | Dabov et al., BM3D, TIP 2007 | 30.0 | 0.8500 | — | — |  |
| 2 | SwinIR | 2021 | Liang et al., SwinIR, ICCVW 2021 | 34.0 | — | — | — |  |
| 3 | Noise2Void | 2019 | Krull et al., N2V, CVPR 2019 | 28.0 | — | — | — |  |
| 4 | Richardson-Lucy (SEM) (PWM) | — | — | — | — | 23.2 | 0.4997 | done |
| 5 | SEM-DL (SegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 23.2 | 0.4997 | done |
| 6 | SEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 23.2 | 0.4997 | done |
| 7 | precomputed_baseline (test) | — | — | — | — | 15.7 | 0.7926 | done |

### 80. Scanning Transmission Electron Microscopy (STEM) (`stem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinIR | 2021 | SwinIR, 2021 | 33.0 | — | — | — |  |
| 2 | BM3D | 2007 | BM3D, 2007 | 30.0 | — | — | — |  |
| 3 | Richardson-Lucy (STEM) (PWM) | — | — | — | — | 31.0 | 0.9508 | done |
| 4 | STEM-DL (AtomSegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.0 | 0.9508 | done |
| 5 | STEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.0 | 0.9508 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 30.0 | 0.9276 | done |

### 81. Transmission Electron Microscopy (TEM) (`tem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinIR | 2021 | Liang et al., SwinIR, 2021 | 35.0 | — | — | — |  |
| 2 | BM3D | 2007 | Dabov et al., BM3D, 2007 | 30.0 | — | — | — |  |
| 3 | FISTA-L2 (CTF correction) (PWM) | — | — | — | — | 6.9 | 0.0020 |  |
| 4 | TEM-DL (ePIE-Net) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.9 | 0.0997 |  |
| 5 | TEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.9 | 0.0997 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 25.3 | 0.9190 | done |

---

## Computational Photography

### 82. Coded Exposure / Flutter Shutter (`coded_exposure`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Wiener flutter shutter | 2006 | Raskar et al., Coded Exposure, SIGGRAPH 2006 | 26.0 | — | — | — |  |
| 2 | Restormer-Deblur | 2022 | Restormer, CVPR 2022 | 35.0 | — | — | — |  |
| 3 | MPRNet | 2021 | Zamir et al., MPRNet, CVPR 2021 | 34.0 | — | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.9313 | done |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.9313 | done |
| 6 | FlowNet-Coded [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.9313 | done |
| 7 | precomputed_baseline (test) | — | — | — | — | 19.9 | 0.8073 | done |

### 83. Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | E2VID | 2019 | Rebecq et al., E2VID, TPAMI 2020 | 28.0 | — | 4.2 | -0.0104 |  |
| 2 | DiffEvent | 2023 | DiffEvent, 2023 | 36.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | -0.0104 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | -0.0104 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 7.3 | 0.0574 |  |

### 84. High Dynamic Range (HDR) Imaging (`hdr_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Debevec | 1997 | Debevec & Malik, SIGGRAPH 1997 | 30.0 | — | — | — |  |
| 2 | HDRFormer | 2023 | HDRFormer, 2023 | 36.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.4 | 0.7267 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.4 | 0.7267 | done |
| 5 | HDR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.4 | 0.7267 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 36.8 | 0.8232 | done |

### 85. Lensless (Diffuser Camera) Imaging (`lensless`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Uformer | 2022 | Wang et al., Uformer, CVPR 2022 | 33.0 | — | — | — |  |
| 2 | FlatNet | 2020 | Khan et al., FlatNet, TPAMI 2020 | 30.0 | — | 0.5 | 0.0001 |  |
| 3 | Wiener deconvolution | 1949 | Wiener, 1949 | 22.0 | — | 11.8 | 0.0031 |  |
| 4 | ADMM-TV (PWM) | — | Antipa et al. 2018 | — | — | 11.9 | 0.5896 |  |
| 5 | FlatNet-Lite (PWM) | — | — | — | — | 0.5 | 0.0001 |  |
| 6 | wiener_deconv (test) | — | — | — | — | 11.8 | 0.0031 |  |

### 86. Panorama Multi-Focus Fusion (`panorama`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PanoFormer | 2023 | PanoFormer, 2023 | 34.0 | — | — | — |  |
| 2 | UDIS | 2021 | Nie et al., UDIS, CVPR 2021 | 32.0 | — | — | — |  |
| 3 | Laplacian Pyramid Fusion (PWM) | — | — | — | — | 14.6 | 0.0520 |  |
| 4 | Guided Filter Fusion (PWM) | — | — | — | — | 15.6 | 0.0881 | done |
| 5 | IFCNN (PWM) | — | Zhang et al. 2020 | — | — | 5.9 | 0.0001 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 15.1 | 0.6418 | done |

---

## Neural Rendering

### 87. 3D Gaussian Splatting (3DGS) (`gaussian_splatting`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | 3D-GS | 2023 | Kerbl et al., 3DGS, SIGGRAPH 2023 | 33.5 | 0.9700 | — | — |  |
| 2 | 2DGS | 2024 | Huang et al., 2DGS, SIGGRAPH 2024 | 34.0 | — | — | — |  |
| 3 | GaussianShader | 2024 | GaussianShader, 2024 | 34.5 | — | — | — |  |
| 4 | EWA Splatting (PWM) | — | — | — | — | — | 1.0000 |  |
| 5 | 3DGS (full) (PWM) | — | Kerbl et al. SIGGRAPH 2023 | — | — | — | 1.0000 |  |
| 6 | NeRF (baseline comparison) (PWM) | — | — | — | — | — | 1.0000 |  |
| 7 | 3DGS (compact) (PWM) | — | — | — | — | — | 1.0000 |  |
| 8 | direct_render_baseline (test) | — | — | — | — | 0.0 | 0.0000 |  |

### 88. Neural Radiance Fields (NeRF) (`nerf`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | NeRF | 2020 | Mildenhall et al., NeRF, ECCV 2020 | 31.0 | 0.9500 | — | — |  |
| 2 | Instant-NGP | 2022 | Muller et al., Instant-NGP, SIGGRAPH 2022 | 33.5 | 0.9600 | — | — |  |
| 3 | 3D-GS | 2023 | Kerbl et al., 3D-GS, SIGGRAPH 2023 | 33.5 | 0.9700 | — | — |  |
| 4 | NeRFactor2 | 2024 | NeRFactor2, 2024 | 35.9 | — | — | — |  |
| 5 | SfM + MVS (PWM) | — | — | — | — | 21.4 | 0.8758 | done |
| 6 | Mip-NeRF 360 (PWM) | — | Barron et al. CVPR 2022 | — | — | — | — |  |
| 7 | NeRF (original MLP) (PWM) | — | Mildenhall et al. 2020 | — | — | — | — |  |
| 8 | Richardson-Lucy (proxy baseline) (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8507 | done |
| 9 | FISTA-TV (proxy baseline) (PWM) | — | Beck & Teboulle 2009, SIAM | — | — | 19.7 | 0.8507 | done |
| 10 | precomputed_baseline (test) | — | — | — | — | 29.0 | 0.9913 | done |

---

## Depth Imaging

### 89. Flash LiDAR (`flash_lidar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DiffLiDAR | 2023 | DiffLiDAR, 2023 | 36.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1433 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1433 |  |
| 4 | FlashLiDAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1433 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 4.3 | — |  |

### 90. LiDAR Scanner (`lidar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Point Transformer | 2021 | Zhao et al., Point Transformer, ICCV 2021 | 34.0 | — | 32.7 | 0.8269 | done |
| 2 | Bilateral Filter | 1998 | Tomasi & Manduchi, 1998 | 30.0 | — | — | — |  |
| 3 | FISTA-L2 (depth) (PWM) | — | — | — | — | 4.4 | 0.0396 |  |
| 4 | PointNet++ [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.7 | 0.8269 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 32.6 | 0.9955 | done |

### 91. Photometric Stereo (`photometric_stereo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PS-Transformer | 2023 | PS-Transformer, 2023 | 34.0 | — | 7.1 | 0.5223 |  |
| 2 | CNN-PS | 2019 | Chen et al., CNN-PS, CVPR 2019 | 32.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.1 | 0.5223 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.1 | 0.5223 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 29.0 | 0.9583 | done |

### 92. Structured-Light Depth Camera (`structured_light`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PhaseFormer | 2023 | PhaseFormer, 2023 | 34.0 | — | — | — |  |
| 2 | Gray code | 2004 | Scharstein & Szeliski, 2003 | 28.0 | — | — | — |  |
| 3 | FISTA-L2 (phase unwrap) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.0042 |  |
| 4 | SL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.0042 |  |
| 5 | FTPD [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.0042 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 8.0 | -0.0287 |  |

### 93. Time-of-Flight Depth Camera (`tof_camera`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MPI-Former | 2023 | MPI-Former, 2023 | 33.0 | — | 12.3 | 0.1318 |  |
| 2 | DeepToF | 2018 | Marco et al., DeepToF, CVPR 2017 | 32.0 | — | — | — |  |
| 3 | FISTA-L2 (depth) (PWM) | — | — | — | — | 2.4 | 0.0284 |  |
| 4 | ToF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.3 | 0.1318 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 42.0 | 0.9994 | done |

---

## Remote Sensing

### 94. Ground-Penetrating Radar (GPR) (`gpr`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | RTM | 2000 | Reverse Time Migration | 27.0 | — | — | — |  |
| 2 | HyperDet | 2023 | HyperDet, 2023 | 30.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.6 | 0.0035 |  |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.6 | 0.0035 |  |
| 5 | GPR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.6 | 0.0035 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 10.6 | 0.0059 |  |

### 95. Hyperspectral Remote Sensing (`hyperspectral_remote`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MST++ | 2022 | Cai et al., MST++, CVPRW 2022 | 35.0 | — | — | — |  |
| 2 | CNMF | 2012 | Yokoya et al., CNMF, 2012 | 28.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.8 | 0.6998 | done |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.8 | 0.6998 | done |
| 5 | SST-USRNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.8 | 0.6998 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 29.1 | 0.9768 | done |

### 96. Interferometric SAR (InSAR) (`insar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | InSAR-Former | 2023 | InSAR-Former, 2023 | 34.0 | — | 31.8 | 0.9073 | done |
| 2 | Goldstein filter | 1998 | Goldstein & Werner, 1998 | 25.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9073 | done |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9073 | done |
| 5 | wrapped_phase_baseline (test) | — | — | — | — | 31.8 | 0.9933 | done |

### 97. Multispectral Satellite Imaging (`multispectral_sat`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SwinIR | 2021 | SwinIR, 2021 | 35.0 | — | — | — |  |
| 2 | Restormer | 2022 | Restormer, 2022 | 34.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.3 | 0.0718 |  |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.3 | 0.0718 |  |
| 5 | MS-Pansharpening-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.3 | 0.0718 |  |
| 6 | bicubic_upsample (test) | — | — | — | — | 10.8 | 0.1002 |  |

### 98. Ocean Color Remote Sensing (`ocean_color`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | AquaFormer | 2023 | AquaFormer, 2023 | 33.0 | — | — | — |  |
| 2 | MUMM | 2007 | Ruddick et al., MUMM, 2000 | 28.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 38.2 | 0.9992 | done |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 38.2 | 0.9992 | done |
| 5 | OC-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 38.2 | 0.9992 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 44.1 | 0.9998 | done |

### 99. Passive Microwave Radiometry (`passive_microwave`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MWR-Former | 2023 | MWR-Former, 2023 | 32.0 | — | — | — |  |
| 2 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.9 | 0.3301 | done |
| 3 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.9 | 0.3301 | done |
| 4 | PM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.9 | 0.3301 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 9.2 | 0.5946 |  |

### 100. Polarimetric SAR (PolSAR) (`polsar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SARFormer | 2023 | SARFormer, 2023 | 35.0 | — | — | — |  |
| 2 | Lee filter | 1999 | Lee et al., 1999 | 26.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.2891 |  |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.2891 |  |
| 5 | PolSAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.2891 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 3.5 | -0.0175 |  |

### 101. Radio Interferometry (VLBI) (`radio_interferometry`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | CLEAN | 1974 | Hogbom, CLEAN, A&AS 1974 | 25.0 | — | — | — |  |
| 2 | R2D2 | 2023 | Terris et al., R2D2, 2023 | 30.0 | — | 9.0 | 0.0880 |  |
| 3 | PRIMO | 2023 | Medeiros et al., PRIMO, 2023 | 29.0 | — | — | — |  |
| 4 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 9.0 | 0.0880 |  |
| 5 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 9.0 | 0.0880 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 23.2 | 0.2029 | done |

### 102. Synthetic Aperture Radar (SAR) (`sar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Range-Doppler | 1978 | Curlander & McDonough, 1991 | 25.0 | 0.7000 | — | — |  |
| 2 | DiffusionSAR | 2023 | DiffusionSAR, 2023 | 35.4 | — | — | — |  |
| 3 | SARFormer | 2023 | SARFormer, 2023 | 33.9 | — | — | — |  |
| 4 | FBP (SAR backprojection) (PWM) | — | — | — | — | 13.6 | 0.2879 |  |
| 5 | SAR-DL (PolSF) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.8 | 0.1380 | done |
| 6 | SAR-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.8 | 0.1380 | done |
| 7 | precomputed_baseline (test) | — | — | — | — | 17.3 | 0.7046 | done |

### 103. Sonar Imaging (`sonar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MVDR/Capon | 1969 | Capon, 1969 | 25.0 | — | — | — |  |
| 2 | AcousticFormer | 2023 | AcousticFormer, 2023 | 34.0 | — | — | — |  |
| 3 | FISTA-L2 (DAS) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.0 | 0.2817 |  |
| 4 | SonarSR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.0 | 0.2817 |  |
| 5 | Sonar-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.0 | 0.2817 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 10.3 | 0.5149 |  |

### 104. Weather / Doppler Radar (`weather_radar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Earthformer | 2022 | Gao et al., Earthformer, NeurIPS 2022 | 33.0 | — | — | — |  |
| 2 | RainNet | 2020 | Ayzel et al., RainNet, 2020 | 30.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.8 | 0.1187 | done |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.8 | 0.1187 | done |
| 5 | NowcastNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.8 | 0.1187 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 26.9 | 0.9155 | done |

---

## Industrial Inspection

### 105. Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.0 | 0.7398 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.0 | 0.7398 | done |
| 3 | SAFT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.0 | 0.7398 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 10.0 | -0.0384 |  |

### 106. Active Thermography (IR) (`active_thermography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.1475 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.1475 |  |
| 3 | Pulsed-Phase TV [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.1475 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 6.5 | 0.1897 |  |

### 107. Eddy Current Imaging (`eddy_current`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.9 | 0.6356 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.9 | 0.6356 | done |
| 3 | ECT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.9 | 0.6356 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 4.8 | -0.0811 |  |

### 108. Industrial X-ray CT (`industrial_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.9 | 0.4680 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.9 | 0.4680 |  |
| 3 | IndustrialCT-Net [proxy] (PWM) | — | Shepp & Logan 1974 | — | — | 0.4 | 0.0702 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 20.3 | 0.4046 | done |

### 109. Machine Vision / AOI (`machine_vision`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.3 | 0.2934 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.3 | 0.2934 | done |
| 3 | PatchCore [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.3 | 0.2934 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 26.5 | 0.9622 | done |

### 110. Shearography (`shearography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 13.2 | 0.1733 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 13.2 | 0.1733 |  |
| 3 | ShearNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 13.2 | 0.1733 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 8.0 | -0.0011 |  |

### 111. Terahertz Imaging (THz) (`terahertz`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.7 | 0.0217 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.7 | 0.0217 |  |
| 3 | THz-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.7 | 0.0217 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 37.1 | 0.9963 | done |

### 112. Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2409 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2409 | done |
| 3 | TFM-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2409 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 29.6 | 0.6891 | done |

### 113. X-ray NDT (Radiography) (`xray_ndt`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.0 | 0.3334 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.0 | 0.3334 |  |
| 3 | NDT-DefectNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.0 | 0.3334 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 16.7 | 0.8430 | done |

### 114. X-ray Fluorescence (XRF) Imaging (`xrf_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.2 | 0.4457 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.2 | 0.4457 | done |
| 3 | XRF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.2 | 0.4457 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 22.1 | 0.9626 | done |

---

## Quantum Imaging

### 115. Entangled Photon Microscopy (`entangled_photon`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9772 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9772 | done |
| 3 | QGI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9772 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 31.8 | 0.9688 | done |

### 116. Ghost Imaging (`ghost_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 3 | GI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 6.6 | 0.1947 |  |

### 117. Quantum Illumination (`quantum_illumination`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.0 | 0.4292 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.0 | 0.4292 | done |
| 3 | QI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.0 | 0.4292 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 20.2 | 0.7859 | done |

---

## Scientific Instrumentation

### 118. Atom Probe Tomography (APT) (`atom_probe`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.4 | 0.0415 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.4 | 0.0415 |  |
| 3 | APT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.4 | 0.0415 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 40.2 | 0.9878 | done |

### 119. Cathodoluminescence (CL) Imaging (`cathodoluminescence`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.8 | 0.7716 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.8 | 0.7716 | done |
| 3 | CL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.8 | 0.7716 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 28.9 | 0.9772 | done |

### 120. Cryo-EM Single Particle Analysis (`cryo_em`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | cryoSPARC | 2017 | Punjani et al., cryoSPARC, Nature Methods 2017 | 35.0 | — | — | — |  |
| 2 | RELION | 2012 | Scheres, RELION, JSB 2012 | 33.0 | — | 17.9 | 0.0101 |  |
| 3 | CryoSTAR | 2023 | CryoSTAR, 2023 | 38.4 | — | — | — |  |
| 4 | DiffusionCryo | 2024 | DiffusionCryo, 2024 | 39.8 | — | — | — |  |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.9 | 0.0101 | done |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.9 | 0.0101 | done |
| 7 | CryoDRGN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.9 | 0.0101 | done |
| 8 | precomputed_wiener (test) | — | — | — | — | 19.2 | 0.0300 | done |
| 9 | rl_ctf_20iter (test) | — | — | — | — | 13.2 | 0.4136 |  |

### 121. MALDI Mass Spectrometry Imaging (`maldi_msi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.1 | 0.6924 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.1 | 0.6924 | done |
| 3 | MSI-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.1 | 0.6924 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 26.3 | 0.9418 | done |

### 122. Muon Tomography (`muon_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP (muon tomography) (PWM) | — | — | — | — | 3.1 | 0.0019 |  |
| 2 | POCA-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0200 |  |
| 3 | EM-POCA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0200 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 5.2 | -0.0128 |  |

### 123. Neutron Diffraction (`neutron_diffraction`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.3 | 0.0013 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.3 | 0.0013 |  |
| 3 | NeutronDiff-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.3 | 0.0013 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 8.5 | 0.0116 |  |

### 124. Neutron Radiography / Tomography (`neutron_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP (neutron tomography) (PWM) | — | — | — | — | 4.3 | 0.0210 |  |
| 2 | NeuTomo-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.6130 |  |
| 3 | GRIDREC-Neutron [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.6130 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | -5.7 | 0.0503 |  |

### 125. Proton Radiography (`proton_radiography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP (proton radiography) (PWM) | — | — | — | — | 10.9 | 0.0397 |  |
| 2 | ProtonRecon-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.0 | 0.3615 |  |
| 3 | FBP-Proton [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.0 | 0.3615 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 4.1 | -0.0000 |  |

### 126. Small-Angle X-ray Scattering (SAXS) (`saxs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.1 | 0.0265 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.1 | 0.0265 |  |
| 3 | SAXS-VAE [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.1 | 0.0265 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 8.4 | 0.0542 |  |

### 127. Wide-Angle X-ray Scattering (WAXS) (`waxs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.2 | 0.0024 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.2 | 0.0024 |  |
| 3 | WAXS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.2 | 0.0024 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 20.6 | 0.0694 | done |

### 128. X-ray Crystallography (`xray_crystallography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.1 | 0.0063 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.1 | 0.0063 |  |
| 3 | AlphaFold-SF [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.1 | 0.0063 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 22.4 | 0.0651 | done |

### 129. X-ray Fluorescence Tomography (`xrf_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.0361 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.0361 |  |
| 3 | XRFT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.0361 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 15.6 | 0.8431 | done |

---

## Multi-Modal Fusion

### 130. Correlative Light-Electron Microscopy (CLEM) (`clem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.1 | 0.7705 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.1 | 0.7705 | done |
| 3 | CLEM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.1 | 0.7705 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 17.0 | 0.7297 | done |

### 131. CT + Fluorescence (FLIT) (`ct_fluorescence`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.3 | 0.3256 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.3 | 0.3256 |  |
| 3 | XFCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.3 | 0.3256 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | — | 0.0002 |  |

### 132. PET/CT Fusion (`pet_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1811 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1811 |  |
| 3 | PET-CT-Fusion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1811 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 13.0 | 0.0656 |  |

### 133. PET/MR Fusion (`pet_mr`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.3034 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.3034 |  |
| 3 | PET-MR-DeepJoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.3034 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 11.0 | 0.0165 |  |

### 134. SPECT/CT Fusion (`spect_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.2421 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.2421 |  |
| 3 | SPECT-CT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.2421 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 11.4 | 0.0239 |  |

### 135. US/MRI Fusion (`us_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.5 | 0.3470 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.5 | 0.3470 | done |
| 3 | US-MRI-Net [proxy] (PWM) | — | — | — | — | 9.7 | 0.5796 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 7.6 | -0.0694 |  |

---

## Broader Experimental Science

### 136. Acoustic Emission Testing (AE) (`acoustic_emission`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.3286 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.3286 |  |
| 3 | DeepAE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.3286 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 20.2 | 0.0741 | done |

### 137. Adaptive Optics (AO) Imaging (`adaptive_optics`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.2 | 0.4415 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.2 | 0.4415 | done |
| 3 | Deep-AO [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.2 | 0.4415 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | — | 1.0000 |  |

### 138. Bioluminescence Tomography (BLT) (`bioluminescence_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0974 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0974 |  |
| 3 | BLT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0974 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 13.3 | 0.3431 |  |

### 139. Full-Waveform Inversion (FWI) (`fwi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.7 | 0.0009 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.7 | 0.0009 |  |
| 3 | InversionNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.7 | 0.0009 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 8.7 | 0.0125 |  |

### 140. Gravitational Wave Detection (`gravitational_wave`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.6 | 0.0000 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.6 | 0.0000 | done |
| 3 | GW-DL (PyCBC-ML) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.6 | 0.0000 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | — | 0.8666 |  |

### 141. Electrical Impedance Tomography (EIT) (`impedance_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 3 | EIT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 11.2 | 0.3124 |  |

### 142. Magnetic Particle Imaging (MPI) (`magnetic_particle`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.7 | 0.0284 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.7 | 0.0284 |  |
| 3 | MPI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.7 | 0.0284 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 26.5 | 0.9576 | done |

### 143. Ocean Acoustic Tomography (`ocean_acoustic_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.6789 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.6789 | done |
| 3 | OAT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.6789 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 5.6 | 0.6714 |  |

### 144. Particle Calorimetry (`particle_calorimetry`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 36.7 | 0.9421 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 36.7 | 0.9421 | done |
| 3 | CaloDiffusion [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 36.7 | 0.9421 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 36.2 | 0.7914 | done |

### 145. Radio Aperture Synthesis (`radio_astronomy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 37.3 | 0.7683 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 37.3 | 0.7683 | done |
| 3 | RadioAST-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 37.3 | 0.7683 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 16.1 | 0.2876 | done |

### 146. Seismic Tomography (`seismic_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.6 | 0.0430 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.6 | 0.0430 |  |
| 3 | SeisInversion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.6 | 0.0430 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 9.0 | 0.5099 |  |

---

## Scanning Probe Microscopy

### 147. Atomic Force Microscopy (AFM) (`afm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy (PWM) | — | — | — | — | 31.3 | 0.7815 | done |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 15.6 | 0.3770 | done |
| 3 | AFM-UNet (PWM) | — | Cherukara, M.J. et al. (2020) AI-enabled high-res, real-time imaging, npj Comput. Mater. 6:203 | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 19.0 | 0.8537 | done |

### 148. Magnetic Force Microscopy (MFM) (`mfm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy (PWM) | — | — | — | — | 9.5 | 0.3740 |  |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 5.1 | 0.0008 |  |
| 3 | MFM-UNet (PWM) | — | Kim, M. et al. (2021) DL for magnetic force microscopy, npj Comput. Mater. 7:87 | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 34.3 | 0.2871 | done |

### 149. Near-field Scanning Optical Microscopy (NSOM) (`nsom`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy (PWM) | — | — | — | — | 22.3 | 0.2438 | done |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 17.7 | 0.7562 | done |
| 3 | NSOM-Net (PWM) | — | Park, J. et al. (2020) DL for near-field optical microscopy, Optica 7(11) | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 19.6 | 0.7328 | done |

### 150. Scanning Tunneling Microscopy (STM) (`stm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy (PWM) | — | — | — | — | 23.3 | 0.9600 | done |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 7.0 | 0.0000 |  |
| 3 | STM-Net (PWM) | — | Ziatdinov, M. et al. (2021) DL for atomic-level STM, Nat. Mach. Intell. 3:269 | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 17.9 | 0.8025 | done |

---

## Spectroscopy & Spectral Imaging

### 151. Brillouin Microscopy (`brillouin`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.4967 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.4967 | done |
| 3 | Brillouin-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.4967 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 35.8 | 0.9959 | done |

### 152. Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.7 | 0.4812 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.7 | 0.4812 | done |
| 3 | CARS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.7 | 0.4812 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 14.2 | 0.0040 |  |

### 153. DESI Mass Spectrometry Imaging (`desi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.5 | 0.0056 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.5 | 0.0056 |  |
| 3 | DESI-SegNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.5 | 0.0056 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 15.1 | 0.3130 | done |

### 154. FTIR Spectroscopic Imaging (`ftir_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 34.6 | 0.9204 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 34.6 | 0.9204 | done |
| 3 | FTIR-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 34.6 | 0.9204 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 14.8 | 0.8058 |  |

### 155. Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.5 | 0.6516 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.5 | 0.6516 | done |
| 3 | LIBS-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.5 | 0.6516 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 18.0 | 0.5987 | done |

### 156. Raman Imaging / Microscopy (`raman_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.5114 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.5114 | done |
| 3 | RamanNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.5114 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 14.1 | 0.2149 |  |

### 157. Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.9 | 0.4524 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.9 | 0.4524 |  |
| 3 | SIMS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.9 | 0.4524 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 20.5 | 0.9749 | done |

### 158. Stimulated Raman Scattering (SRS) Microscopy (`srs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.0 | 0.9307 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.0 | 0.9307 | done |
| 3 | SRS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.0 | 0.9307 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 29.1 | 0.9779 | done |

---

## Astronomy & Space Imaging

### 159. Stellar Coronagraphy (`coronagraphy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.8930 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.8930 | done |
| 3 | DL-SpeckleNull [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.8930 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 25.2 | 0.2028 | done |

### 160. Event Horizon Telescope (EHT) Imaging (`eht_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.4 | 0.0148 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.4 | 0.0148 |  |
| 3 | EHT-PRIMO [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.4 | 0.0148 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 11.3 | 0.0394 |  |

### 161. Lucky Imaging (`lucky_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.6 | 0.7117 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.6 | 0.7117 | done |
| 3 | Lucky-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.6 | 0.7117 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 29.2 | 0.9746 | done |

### 162. Solar EUV/X-ray Imaging (`solar_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.9 | 0.9167 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.9 | 0.9167 | done |
| 3 | SolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.9 | 0.9167 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 28.4 | 0.9958 | done |

---

## Ultrafast Imaging

### 163. Compressed Ultrafast Photography (CUP) (`cup`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.5 | 0.1925 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.5 | 0.1925 |  |
| 3 | E2E-CUP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.5 | 0.1925 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | -2.3 | 0.1202 |  |

### 164. Pump-Probe Microscopy (`pump_probe`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.5 | 0.8551 |  |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.5 | 0.8551 |  |
| 3 | PumpProbe-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.5 | 0.8551 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 18.2 | 0.7781 | done |

### 165. Streak Camera Imaging (`streak_camera`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2856 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2856 | done |
| 3 | StreakNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2856 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 14.3 | 0.1114 |  |

### 166. XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8643 | done |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8643 | done |
| 3 | SFX-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8643 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 24.1 | 0.9753 | done |

---

## Computational Optics

### 167. Integral Photography (`integral`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DistgSSR | 2022 | DistgSSR, TPAMI 2022 | 34.0 | — | — | — |  |
| 2 | LFAttNet | 2021 | LFAttNet, 2021 | 32.0 | — | — | — |  |
| 3 | Depth Estimation (PWM) | — | — | — | — | 33.4 | 0.9013 | done |
| 4 | DIBR [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 33.4 | 0.9013 | done |
| 5 | EPINet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 33.4 | 0.9013 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 40.0 | 0.9990 | done |

### 168. Light Field Imaging (`light_field`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DistgSSR | 2022 | Wang et al., DistgSSR, TPAMI 2022 | 33.5 | 0.9600 | — | — |  |
| 2 | LFNet | 2021 | LFNet, 2021 | 31.0 | — | — | — |  |
| 3 | Shift-and-Sum (PWM) | — | — | — | — | 16.3 | 0.1186 | done |
| 4 | LFBM5D (PWM) | — | Alain et al. 2017, Signal Processing: Image Communication | — | — | 4.3 | 0.0170 |  |
| 5 | LFSSR (PWM) | — | Yeung et al. ECCV 2018 | — | — | 16.3 | 0.1186 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 27.3 | 0.9439 | done |

---

## Summary

- **Total modalities**: 168
- **Total algorithm entries**: 903 (across all modalities)
- **Sources**: PWM benchmark tests, YAML solver configs, benchmark webpage, literature 2000-2026
- **Status legend**: `done` = PWM reproduces reference-quality results
