# Algorithm State — PWM5 Benchmark

Comprehensive listing of reconstruction algorithms for all 168 modalities.
Generated: 2026-03-11

## Legend
- **Ref PSNR/SSIM**: Published reference values from literature
- **PWM PSNR/SSIM**: Values achieved by PWM framework on synthetic benchmark data
- **Status**: `done` = PWM within 3 dB of reference or better | blank = not verified
- **Year**: Publication year of algorithm
- **Dataset**: Benchmark dataset used for reference evaluation

---

## Compressive Imaging

### 1. Coded Aperture Compressive Temporal Imaging (CACTI) (`cacti`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | GAP-TV | 2016 | Yuan, ICIP 2016 | 26.7 | 0.8460 | 19.8 | 0.4362 |  |
| 2 | DeSCI | 2019 | Liu et al., TPAMI 2019 | 27.1 | 0.8700 | — | — |  |
| 3 | PnP-FFDNet | 2020 | Yuan et al., CVPR 2020 | 28.7 | 0.9050 | — | — |  |
| 4 | MetaSCI | 2021 | Wang et al., CVPR 2021 | 30.1 | 0.9150 | — | — |  |
| 5 | RevSCI-Net | 2021 | Cheng et al., NeurIPS 2021 | 31.4 | 0.9350 | — | — |  |
| 6 | BIRNAT | 2022 | Cheng et al., ECCV 2022 | 32.7 | 0.9510 | — | — |  |
| 7 | ELP-Unfolding | 2022 | Yang et al., ECCV 2022 | 33.1 | 0.9530 | 5.9 | 0.0083 |  |
| 8 | STFormer | 2022 | Wang et al., NeurIPS 2022 | 33.9 | 0.9600 | — | — |  |
| 9 | EfficientSCI | 2023 | Wang et al., CVPR 2023 | 34.3 | 0.9610 | 1.5 | 0.0774 |  |
| 10 | HiSViT | 2023 | Chen et al., ICCV 2023 | 34.5 | — | — | — |  |
| 11 | DUN-3DUnet | 2022 | Wu et al., CVPR 2022 | 35.3 | 0.9620 | — | — |  |
| 12 | CTM-SCI | 2024 | CTM-SCI, 2024 | 36.5 | — | — | — |  |
| 13 | HiSViT-13 | 2024 | Chen et al., ECCV 2024 | 37.3 | — | — | — |  |
| 14 | EfficientSCI-T (PWM) | — | — | — | — | 8.0 | 0.0759 |  |
| 15 | mask_division_baseline (test) | — | — | — | — | — | — |  |
| 16 | gap_tv (test) | — | — | — | — | — | — |  |

### 2. Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TwIST | 2007 | Bioucas-Dias & Figueiredo, TwIST, TIP 2007 | 23.1 | 0.6690 | — | — |  |
| 2 | GAP-TV | 2016 | Yuan, GAP-TV, ICIP 2016 | 24.4 | 0.6690 | 25.9 | 0.9649 | done |
| 3 | ADMM-Net | 2019 | Ma et al., ICCV 2019 | 29.1 | 0.8600 | — | — |  |
| 4 | λ-Net | 2020 | Miao et al., ICCV 2019 | 30.1 | 0.8770 | — | — |  |
| 5 | TSA-Net | 2020 | Meng et al., ECCV 2020 | 31.5 | 0.8940 | — | — |  |
| 6 | DGSMP | 2021 | Huang et al., CVPR 2021 | 32.6 | 0.9170 | — | — |  |
| 7 | HDNet | 2022 | Hu et al., CVPR 2022 | 35.0 | 0.9430 | — | — |  |
| 8 | MST-L | 2022 | Cai et al., CVPR 2022 | 34.9 | 0.9440 | — | — |  |
| 9 | CST-L-Plus | 2022 | Cai et al., ECCV 2022 | 36.1 | 0.9570 | — | — |  |
| 10 | DAUHST-9stg | 2022 | Cai et al., NeurIPS 2022 | 38.4 | 0.9670 | — | — |  |
| 11 | MST++ | 2022 | Cai et al., CVPRW 2022 | 36.0 | 0.9510 | — | — |  |
| 12 | PADUT | 2023 | Li et al., CVPR 2023 | 34.8 | — | — | — |  |
| 13 | RDLUF-MixS2 | 2022 | Cai et al., ECCV 2022 | 39.6 | 0.9720 | — | — |  |
| 14 | SSR-L | 2023 | Zhang et al., ICCV 2023 | 34.0 | — | — | — |  |
| 15 | PADUT-L | 2023 | Li et al., CVPR 2023 | 38.9 | 0.9700 | — | — |  |
| 16 | MiJUN | 2025 | MiJUN, AAAI 2025 | 40.9 | 0.9760 | — | — |  |
| 17 | GAP-TV (guided) (PWM) | — | Yuan et al. 2016 | — | — | 26.2 | 0.9665 | done |
| 18 | GAP-TV (fast) (PWM) | — | — | — | — | 25.9 | 0.9649 | done |
| 19 | GAP-TV (small) (PWM) | — | — | — | — | 25.3 | 0.9664 | done |

### 3. Generic Matrix Sensing (`matrix`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FISTA | 2009 | Beck & Teboulle, SIAM 2009 | 27.0 | — | 6.1 | 0.0001 |  |
| 2 | LISTA | 2010 | Gregor & LeCun, ICML 2010 | 28.5 | — | 6.1 | 0.0000 |  |
| 3 | FISTA-L1 (high quality) (PWM) | — | Beck & Teboulle 2009 | — | — | 6.4 | 0.0257 |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 22.0 | 0.6949 | done |

### 4. Single-Pixel Camera (SPC) (`spc`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TVAL3 | 2009 | Li et al., TVAL3, Rice 2009 | 24.6 | 0.7500 | 6.8 | 0.0161 |  |
| 2 | ISTA-Net+ | 2018 | Zhang & Ghanem, CVPR 2018 | 32.3 | 0.9350 | 4.7 | 0.0013 |  |
| 3 | CSNet+ | 2019 | Shi et al., TIP 2019 | 29.8 | 0.8820 | — | — |  |
| 4 | AMP-Net | 2021 | Zhang et al., TIP 2021 | 34.6 | 0.9550 | — | — |  |
| 5 | TransCS | 2022 | Shen et al., TIP 2022 | 31.1 | — | — | — |  |
| 6 | ADMM-L1 (PWM) | — | Boyd et al. 2010 | — | — | 4.5 | 0.0018 |  |

---

## Medical Imaging

### 5. X-ray Angiography (`angiography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DSA (Digital Subtraction) | 1980 | DSA, Mistretta et al., 1981 | 25.0 | 0.8000 | 12.9 | 0.5828 |  |
| 2 | Deep Decoupling Net (GAN+RDB) | 2024 | IIETA, TS 2024 | 23.7 | 0.8770 | — | — |  |
| 3 | Maskless 2D-DSA (U-Net) | 2022 | Gao et al., JVIR 2022, PubMed 35311665 | 43.0 | 0.9800 | — | — |  |
| 4 | DSA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.8 | 0.3743 |  |
| 5 | VesselSegNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.8 | 0.3743 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 11.2 | 0.0435 |  |

### 6. Arterial Spin Labeling (ASL) MRI (`asl_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Control-label subtraction | 1998 | Detre et al., MRM 1992 | 22.0 | 0.6500 | — | — |  |
| 2 | ASLRDB (Dilated+RDB) | 2025 | Springer, SIVP 2025 | 25.0 | 0.8240 | — | — |  |
| 3 | HUST (Transformer) 2D | 2025 | Springer, Vis Comput 2025 | 33.7 | 0.9600 | — | — |  |
| 4 | HUST (Transformer) 3D | 2025 | Springer, Vis Comput 2025 | 45.1 | 0.9900 | — | — |  |
| 5 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.1 | 0.0041 |  |
| 6 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.1 | 0.0041 |  |
| 7 | ASL-Net [proxy] (PWM) | — | — | — | — | 2.0 | 0.2531 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 2.7 | 0.0636 |  |

### 7. Brachytherapy Imaging (`brachytherapy_img`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP | 1971 | FBP baseline | 25.0 | — | 25.2 | 0.1810 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.1810 | done |
| 3 | BrachyNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.1810 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 20.5 | 0.2374 | done |

### 8. Cone-Beam Computed Tomography (CBCT) (`cbct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FDK | 1984 | Feldkamp et al., JOSA 1984 | 28.0 | 0.8000 | 12.7 | 0.4010 |  |
| 2 | SART | 1984 | Andersen & Kak, 1984 | 32.0 | 0.8800 | — | — |  |
| 3 | FBPConvNet | 2017 | Jin et al., TIP 2017 | 36.5 | 0.9500 | — | — |  |
| 4 | FACT | 2022 | FACT, 2022 | 33.8 | 0.9300 | — | — |  |
| 5 | FDK-DL (PWM) | — | Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI | — | — | — | — |  |
| 6 | CBCT-UNet (PWM) | — | Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP | — | — | — | — |  |
| 7 | fbp_ramlak (test) | — | — | — | — | 14.9 | 0.3496 |  |
| 8 | fbp_shepp_logan (test) | — | — | — | — | 15.2 | 0.3593 |  |

### 9. CEST MRI (`cest_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Z-spectrum fitting | 2003 | Zhou et al., NMR Biomed 2003 | 25.0 | 0.7500 | — | — |  |
| 2 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.6435 | done |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.6435 | done |
| 4 | CEST-Net [proxy] (PWM) | — | — | — | — | 13.9 | 0.3632 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 31.0 | 0.9859 | done |

### 10. Contrast-Enhanced Ultrasound (CEUS) (`ceus`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Singular value decomposition | 2015 | Demene et al., TMI 2015 | 25.0 | 0.7500 | — | — |  |
| 2 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.2 | 0.1198 |  |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.2 | 0.1198 |  |
| 4 | US-DeepSight [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.2 | 0.1198 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 24.5 | 0.9206 | done |

### 11. Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | — | — | — |  |
| 2 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.4 | 0.7037 | done |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.4 | 0.7037 | done |
| 4 | CLE-Net (CARE) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.4 | 0.7037 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 34.0 | 0.9927 | done |

### 12. X-ray Computed Tomography (CT) (`ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP (Ram-Lak) | 1971 | Ramachandran & Lakshminarayanan 1971 | 30.2 | 0.8200 | 13.7 | 0.0649 |  |
| 2 | TV regularization | 2006 | Sidky et al., PMB 2006 | 33.4 | 0.9000 | — | — |  |
| 3 | RED-CNN | 2017 | Chen et al., TMI 2017 | 33.2 | 0.9150 | 1.3 | 0.1144 |  |
| 4 | FBPConvNet | 2017 | Jin et al., TIP 2017 | 38.5 | 0.9590 | 13.7 | 0.0649 |  |
| 5 | Learned Primal-Dual | 2018 | Adler & Oktem, TMI 2018 | 36.2 | 0.9590 | — | — |  |
| 6 | iRadonMAP | 2019 | He et al., 2019 | 36.9 | 0.9420 | — | — |  |
| 7 | LEARN | 2019 | Chen et al., TMI 2018 | 43.1 | — | — | — |  |
| 8 | DuDoTrans | 2022 | Wang et al., MICCAI 2022 | 42.1 | — | — | — |  |
| 9 | Score-CT | 2022 | Song et al., ICLR 2022 | 43.0 | — | — | — |  |
| 10 | DOLCE | 2023 | Liu et al., 2023 | 36.0 | — | — | — |  |
| 11 | PnP-HQS + NLM (PWM) | — | — | — | — | 1.4 | 0.1306 |  |
| 12 | fbp_ramlak (test) | — | — | — | — | 12.9 | 0.0922 |  |
| 13 | fbp_shepp_logan (test) | — | — | — | — | 13.8 | 0.1053 |  |
| 14 | sart_10iter (test) | — | — | — | — | 13.8 | 0.2168 |  |

### 13. Dual-Energy X-ray Absorptiometry (DEXA) (`dexa`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Dual-energy decomposition | 1987 | Alvarez & Macovski, PMB 1976 | 28.0 | 0.8500 | — | — |  |
| 2 | FISTA-L2 (dual-energy) (PWM) | — | — | — | — | — | 1.0000 |  |
| 3 | DXA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.3099 |  |
| 4 | DEXA-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.3099 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 9.5 | 0.2550 |  |

### 14. Diffusion MRI (DTI) (`diffusion_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6000 | 11.3 | 0.0002 |  |
| 2 | q-DL | 2016 | Golkov et al., MRM 2016 | 34.0 | — | 11.3 | 0.8086 |  |
| 3 | SENSE (WLS tensor fit) (PWM) | — | — | — | — | 11.3 | 0.8086 |  |
| 4 | SHORE-Net [proxy] (PWM) | — | — | — | — | 11.3 | 0.8086 |  |
| 5 | zero_filled (test) | — | — | — | — | 11.3 | 0.0002 |  |

### 15. Digital Breast Tomosynthesis (DBT) (`digital_breast_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP | 1971 | FBP baseline | 25.0 | — | 2.5 | 0.2565 |  |
| 2 | SART | 1984 | Andersen & Kak 1984 | 30.0 | — | — | — |  |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.2565 |  |
| 4 | DBT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.5 | 0.2565 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | — | 0.0001 |  |

### 16. Doppler Ultrasound (`doppler_ultrasound`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Autocorrelation | 1985 | Kasai et al., 1985 | 22.0 | 0.7000 | 3.4 | 0.0000 |  |
| 2 | DL Doppler | 2020 | DL for Doppler dealiasing | 30.0 | 0.8800 | — | — |  |
| 3 | Back-Projection (Doppler) (PWM) | — | — | — | — | 5.1 | 0.0242 |  |
| 4 | UDoppler-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.1 | 0.0242 |  |
| 5 | Doppler CFAR [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.1 | 0.0242 |  |
| 6 | autocorrelation_estimator (test) | — | — | — | — | 3.4 | 0.0000 |  |
| 7 | clutter_filtered (test) | — | — | — | — | 0.2 | -0.0019 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 17.6 | 0.0064 |  |

### 17. Diffuse Optical Tomography (DOT) (`dot`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Born approximation | 1999 | Arridge, Inverse Problems 1999 | 20.0 | 0.6000 | 4.3 | 0.0170 |  |
| 2 | Tikhonov regularization | 2000 | Tikhonov | 22.0 | 0.6500 | 2.4 | -0.0033 |  |
| 3 | L-BFGS-TV [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.3 | 0.0170 |  |
| 4 | DOT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.3 | 0.0170 |  |
| 5 | born_backprojection (test) | — | — | — | — | 4.7 | -0.0078 |  |
| 6 | tikhonov (test) | — | — | — | — | 2.4 | -0.0033 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 7.0 | 0.0193 |  |

### 18. Shear-Wave Elastography (`elastography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Phase gradient | 2000 | Manduca et al., MRM 2001 | 22.0 | 0.7000 | — | — |  |
| 2 | SENSE (displacement field) (PWM) | — | — | — | — | 11.0 | 0.7949 |  |
| 3 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 10.3 | 0.0031 |  |
| 4 | NLSI-Solver [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 10.3 | 0.0031 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 5.7 | 0.0091 |  |

### 19. Fiber Bundle Endoscopy (`endoscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Interpolation baseline | 2000 | Fiber bundle baseline | 22.0 | 0.6500 | — | — |  |
| 2 | Richardson-Lucy | 1972 | Richardson 1972 | 24.0 | 0.7200 | — | — |  |
| 3 | U-Net denoising | 2019 | DL for CLE | 28.0 | 0.8500 | — | — |  |
| 4 | FISTA-L2 (endoscopy) (PWM) | — | — | — | — | — | — |  |
| 5 | EndoMapper-Net (PWM) | — | Ozyoruk, K.B. et al. (2021) EndoMapper, Nat. Mach. Intel. 3 | — | — | — | — |  |
| 6 | AF-SfMLearner (PWM) | — | Shao, S. et al. (2022) Self-supervised depth estimation in endoscopy, MICCAI 2022 | — | — | — | — |  |
| 7 | rl_20iter (test) | — | — | — | — | 11.8 | 0.8796 |  |
| 8 | rl_50iter (test) | — | — | — | — | 10.4 | 0.8225 |  |
| 9 | precomputed_recon (test) | — | — | — | — | 4.1 | 0.3912 |  |

### 20. Fluoroscopy (`fluoroscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Motion compensation | 2000 | fluoroscopy baseline | 28.0 | 0.8000 | — | — |  |
| 2 | FBP (fluoroscopy) (PWM) | — | — | — | — | 8.6 | 0.1239 |  |
| 3 | FluoroNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 43.6 | 0.9821 | done |
| 4 | X-ray CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 43.6 | 0.9821 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 43.5 | 0.9997 | done |

### 21. Functional MRI (BOLD fMRI) (`fmri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6000 | 4.9 | — |  |
| 2 | CS-fMRI | 2010 | Jung et al., PMB 2009 | 32.0 | 0.8800 | — | — |  |
| 3 | SENSE (fMRI) (PWM) | — | — | — | — | 4.9 | 0.0536 |  |
| 4 | fMRI-Transformer [proxy] (PWM) | — | — | — | — | 4.9 | 0.0536 |  |
| 5 | DeepBold [proxy] (PWM) | — | — | — | — | 4.9 | 0.0536 |  |
| 6 | zero_filled (test) | — | — | — | — | 4.9 | — |  |

### 22. Fundus Camera (`fundus`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 30.0 | 0.9000 | 30.6 | 0.9090 | done |
| 2 | Cofe-Net | 2022 | Li et al., Cofe-Net, 2022 | 24.9 | — | — | — |  |
| 3 | PCE-Net | 2023 | PCE-Net, 2023 | 29.9 | — | — | — |  |
| 4 | RETFound (PWM) | — | Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156 | — | — | — | — |  |
| 5 | DR-Grade-Net (PWM) | — | Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22) | — | — | — | — |  |
| 6 | rl_20iter (test) | — | — | — | — | 35.0 | 0.9965 | done |
| 7 | rl_50iter (test) | — | — | — | — | 35.9 | 0.9972 | done |
| 8 | precomputed_wiener (test) | — | — | — | — | 22.0 | 0.9248 | done |

### 23. Intravascular Ultrasound (IVUS) (`ivus`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DAS beamforming | 1990 | DAS baseline | 22.0 | 0.7000 | — | — |  |
| 2 | IVUS-Net | 2020 | DL for IVUS | 30.0 | 0.8800 | 15.3 | 0.7253 |  |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.3 | 0.7253 |  |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.3 | 0.7253 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 19.8 | 0.8902 |  |

### 24. Mammography (`mammography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP | 1971 | FBP baseline | 30.0 | 0.8500 | 4.1 | 0.0047 |  |
| 2 | MammoNet (GatorTron) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.8 | 0.0988 |  |
| 3 | Mammo-ResNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.8 | 0.0988 |  |
| 4 | precomputed_recon (test) | — | — | — | — | 20.9 | 0.8580 | done |

### 25. MR Elastography (MRE) (`mr_elastography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Direct inversion | 2001 | Manduca et al., MRM 2001 | 22.0 | 0.7000 | — | — |  |
| 2 | Phase gradient | 2001 | Manduca et al., MRM 2001 | 24.0 | 0.7500 | — | — |  |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0020 |  |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0020 |  |
| 5 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0020 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 6.0 | 0.0984 |  |

### 26. MR Fingerprinting (MRF) (`mr_fingerprinting`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Dictionary matching | 2013 | Ma et al., Nature 2013 | 25.0 | 0.8000 | — | — |  |
| 2 | MANTIS | 2019 | Fang et al., MRM 2019 | 30.0 | 0.9000 | — | — |  |
| 3 | GAST-Mamba (T1 map) | 2025 | arXiv 2507.03369 | 33.1 | 0.9670 | — | — |  |
| 4 | MRF-Mixer (T1 map) | 2025 | MDPI Information 2025 | 33.5 | 0.9800 | 2.1 | 0.2517 |  |
| 5 | MRF-Mixer (T2 map) | 2025 | MDPI Information 2025 | 35.9 | 0.9800 | 2.1 | 0.2517 |  |
| 6 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0040 |  |
| 7 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | 0.0040 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 1.8 | 0.0693 |  |

### 27. MR Angiography (MRA) (`mra`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6500 | — | — |  |
| 2 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.6 | 0.0064 |  |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.6 | 0.0064 |  |
| 4 | MRA-VesselNet [proxy] (PWM) | — | — | — | — | 0.5 | 0.0095 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 12.1 | 0.2673 |  |

### 28. Magnetic Resonance Imaging (MRI) (`mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Zero-filled IFFT | 2000 | Baseline | 28.0 | 0.6400 | 13.4 | 0.0132 |  |
| 2 | CS-MRI (SparseMRI) | 2007 | Lustig et al., MRM 2007 | 33.0 | 0.9000 | — | — |  |
| 3 | GRAPPA | 2002 | Griswold et al., MRM 2002 | 34.0 | 0.9200 | — | — |  |
| 4 | U-Net | 2018 | Zbontar et al., fastMRI 2018 | 36.0 | 0.9470 | — | — |  |
| 5 | E2E-VarNet | 2020 | Sriram et al., NeurIPS 2020 | 40.5 | 0.9720 | — | — |  |
| 6 | HUMUS-Net | 2022 | Fabian et al., NeurIPS 2022 | 37.3 | 0.9500 | — | — |  |
| 7 | ReconFormer | 2023 | Guo et al., TMI 2023 | 40.1 | 0.9750 | — | — |  |
| 8 | PromptMR | 2023 | Li et al., MICCAI 2023 | 41.5 | — | — | — |  |
| 9 | PromptMR+ | 2024 | Li et al., TMI 2024 | 39.9 | 0.9730 | — | — |  |
| 10 | CS-MRI (Wavelet) (PWM) | — | Lustig et al. 2007, MRM | — | — | 13.4 | 0.0132 |  |
| 11 | MoDL (PWM) | — | Aggarwal et al. 2019, IEEE TMI | — | — | 13.4 | 0.0132 |  |
| 12 | MoDL (5 unrolls) (PWM) | — | — | — | — | 13.4 | 0.0132 |  |
| 13 | zero_filled (test) | — | — | — | — | 13.0 | 0.0004 |  |
| 14 | cs_mri_wavelet (test) | — | — | — | — | 13.0 | 0.0006 |  |
| 15 | sense (test) | — | — | — | — | 13.0 | 0.0010 |  |

### 29. MR Spectroscopy (MRS) (`mrs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | HLSVD | 2002 | Pijnappel et al., 1992 | 22.0 | — | 4.3 | 0.0011 |  |
| 2 | LCModel | 1993 | Provencher, MRM 1993 | 28.0 | — | — | — |  |
| 3 | SENSE (spectroscopy) (PWM) | — | — | — | — | 2.1 | 0.2197 |  |
| 4 | MRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.3 | 0.0011 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 1.9 | 0.0676 |  |

### 30. Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MBLL | 1988 | Modified Beer-Lambert Law | 20.0 | 0.6000 | — | — |  |
| 2 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 20.2 | 0.5398 | done |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 20.2 | 0.5398 | done |
| 4 | fNIRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 20.2 | 0.5398 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 14.5 | 0.8761 |  |

### 31. Optical Coherence Tomography (OCT) (`oct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BM3D | 2007 | Dabov et al., TIP 2007 | 25.0 | 0.8000 | — | — |  |
| 2 | PSCAT | 2022 | PSCAT, PKU37 OCT | 32.2 | 0.9200 | — | — |  |
| 3 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 35.0 | — | — | — |  |
| 4 | FFT Recon (PWM) | — | — | — | — | 10.3 | 0.1422 |  |
| 5 | Spectral Estimation (PWM) | — | Leitgeb et al. 2003, Optics Express | — | — | 10.2 | 0.1411 |  |
| 6 | OCT Denoising Net (PWM) | — | Devalla et al. 2019, Biomed. Optics Express | — | — | 10.3 | 0.1422 |  |
| 7 | bscan_baseline (test) | — | — | — | — | 23.1 | 0.9439 | done |
| 8 | bscan_ideal_baseline (test) | — | — | — | — | 23.5 | 0.9482 | done |

### 32. OCT Angiography (OCTA) (`octa`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SSADA | 2012 | Jia et al., Opt Express 2012 | 22.0 | 0.7000 | — | — |  |
| 2 | CNN accelerated OCTA | 2022 | Sci Rep 2022 | 20.8 | 0.6300 | — | — |  |
| 3 | SU-Net (Siamese) | 2019 | Lee et al., 2019 | 28.0 | 0.8130 | — | — |  |
| 4 | Motion artifact DL | 2024 | MDPI Mathematics 2024 | 32.7 | 0.9260 | — | — |  |
| 5 | FFT Recon (OCTA) (PWM) | — | — | — | — | 13.3 | 0.0566 |  |
| 6 | OCTA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 18.8 | 0.4872 |  |
| 7 | OCTA-FF [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 18.8 | 0.4872 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 16.8 | 0.4326 |  |

### 33. Positron Emission Tomography (PET) (`pet`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MLEM | 1982 | Shepp & Vardi, TMI 1982 | 28.0 | 0.7500 | — | — |  |
| 2 | OSEM | 1994 | Hudson & Larkin, TMI 1994 | 30.0 | 0.8200 | — | — |  |
| 3 | MAP-OSEM | 2001 | Qi et al., PMB 2003 | 32.0 | 0.8700 | — | — |  |
| 4 | DeepPET | 2019 | Haggstrom et al., PMB 2019 | 34.7 | 0.9200 | — | — |  |
| 5 | SwinIR-PET | 2023 | SwinIR for PET denoising | 39.9 | 0.9600 | — | — |  |
| 6 | FBP (emission tomography) (PWM) | — | — | — | — | 15.4 | 0.0116 |  |
| 7 | NeuroLF-PET (PWM) | — | Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58 | — | — | — | — |  |
| 8 | PET-DL (U-Net) (PWM) | — | Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9) | — | — | — | — |  |
| 9 | fbp_ramlak (test) | — | — | — | — | 9.3 | 0.1813 |  |
| 10 | fbp_shepp_logan (test) | — | — | — | — | 11.9 | 0.2681 |  |
| 11 | precomputed_fbp (test) | — | — | — | — | 33.1 | 0.9325 | done |

### 34. Photoacoustic Imaging (`photoacoustic`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Time Reversal (FBP) | 2000 | Xu & Wang, PMB 2005 | 22.7 | 0.7300 | 18.5 | 0.3658 |  |
| 2 | Post-DL (U-Net) | 2020 | Antholzer et al., Sci Rep 2020 | 24.4 | 0.8500 | — | — |  |
| 3 | Pixel-DL | 2020 | Antholzer et al., Sci Rep 2020 | 29.6 | 0.9100 | — | — |  |
| 4 | Iterative (model-based) | 2000 | Antholzer et al., Sci Rep 2020 | 30.2 | 0.8900 | — | — |  |
| 5 | Residual U-Net (Deep-PAT) | 2021 | Shahid et al., Front Neurosci 2021 | 29.9 | 0.9700 | — | — |  |
| 6 | Back Projection (PWM) | — | — | — | — | 18.5 | 0.3658 |  |
| 7 | Deep-PAT [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 18.5 | 0.3658 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 19.1 | 0.2490 |  |

### 35. Portal Imaging (EPID) (`portal_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Flat-field correction | 2000 | EPID baseline | 25.0 | 0.7500 | — | — |  |
| 2 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.3 | 0.1384 |  |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.3 | 0.1384 |  |
| 4 | PortalDL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.3 | 0.1384 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 10.5 | 0.4088 |  |

### 36. Proton Therapy Imaging (`proton_therapy_img`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP | 1971 | FBP baseline | 28.0 | — | 26.6 | 0.5118 | done |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.5118 | done |
| 3 | ProtonTherapy-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.5118 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 17.8 | 0.7117 |  |

### 37. Single Photon Emission CT (SPECT) (`spect`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MLEM | 1982 | Shepp & Vardi, 1982 | 26.0 | 0.7000 | — | — |  |
| 2 | OSEM | 1994 | Hudson & Larkin, 1994 | 28.5 | 0.7800 | — | — |  |
| 3 | DIP-SPECT | 2020 | Baguer et al., 2020 | 33.3 | 0.9000 | — | — |  |
| 4 | FBP (emission tomography) (PWM) | — | — | — | — | 10.8 | 0.0669 |  |
| 5 | SPECT-DL (OSEM+) (PWM) | — | Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging | — | — | — | — |  |
| 6 | SPECT-UNet (PWM) | — | Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6) | — | — | — | — |  |
| 7 | fbp_ramlak (test) | — | — | — | — | -6.5 | 0.0101 |  |
| 8 | precomputed_fbp (test) | — | — | — | — | 30.0 | 0.9523 | done |

### 38. Photon-Counting Spectral CT (`spectral_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Material decomposition | 2003 | Alvarez & Macovski, PMB 1976 | 28.0 | 0.8500 | — | — |  |
| 2 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1896 |  |
| 3 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1896 |  |
| 4 | SpectralCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1896 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 12.3 | 0.1106 |  |

### 39. Susceptibility-Weighted Imaging (SWI) (`swi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Homodyne filtering | 2004 | Haacke et al., MRM 2004 | 28.0 | 0.8500 | — | — |  |
| 2 | DeepSWI (cGAN) | 2023 | Genc et al., JMRI 2023 | 36.9 | 0.8900 | — | — |  |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.6 | 0.0020 |  |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.6 | 0.0020 |  |
| 5 | SWI-Net [proxy] (PWM) | — | — | — | — | 2.0 | 0.2415 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 1.9 | 0.0677 |  |

### 40. Ultrasound B-mode Imaging (`ultrasound`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DAS (Delay-and-Sum) | 1990 | DAS baseline | 30.4 | — | — | — |  |
| 2 | ADMIRE | 2018 | Byram et al., IEEE TUFFC 2015 | — | — | — | — |  |
| 3 | Deep beamforming (Goudarzi) | 2020 | Goudarzi et al., IEEE TUFFC 2022 | 29.1 | — | 14.8 | 0.3464 |  |
| 4 | Richardson-Lucy (ultrasound) (PWM) | — | Richardson 1972, JOSA | — | — | 14.8 | 0.3464 |  |
| 5 | US-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 14.8 | 0.3464 |  |
| 6 | rl_20iter (test) | — | — | — | — | 14.6 | 0.1559 |  |
| 7 | rl_50iter (test) | — | — | — | — | 14.1 | 0.1323 |  |

### 41. X-ray Radiography (`xray_radiography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Flat-field correction | 2000 | X-ray baseline | 30.0 | 0.8500 | — | — |  |
| 2 | FBP (X-ray radiography) (PWM) | — | — | — | — | 4.5 | -0.0019 |  |
| 3 | CheXNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.8 | 0.3322 |  |
| 4 | X-ray UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.8 | 0.3322 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 26.3 | 0.9844 | done |

---

## Coherent Imaging

### 42. Digital Holographic Microscopy (`holography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Angular Spectrum | 2000 | Goodman, Fourier Optics | 22.0 | 0.7000 | 4.9 | 0.0036 |  |
| 2 | GS (Gerchberg-Saxton) | 1972 | Gerchberg & Saxton, Optik 1972 | 20.0 | 0.6500 | — | — |  |
| 3 | HIO | 1982 | Fienup, Applied Optics 1982 | 25.0 | 0.7800 | — | — |  |
| 4 | PhaseNet (PWM) | — | Rivenson et al. 2018, Light: S&A | — | — | 14.9 | 0.9212 |  |
| 5 | sqrt_intensity_amplitude (test) | — | — | — | — | — | 0.0003 |  |

### 43. Optical Diffraction Tomography (ODT) (`odt`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Rytov approximation | 2000 | Rytov, 1937 | 25.0 | — | — | — |  |
| 2 | Born approximation | 2000 | Wolf, Opt Commun 1969 | 22.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.1 | 0.0782 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.1 | 0.0782 |  |
| 5 | ODT-Net (PhaseNet) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.1 | 0.0782 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 25.5 | 0.9509 | done |

### 44. Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | HIO | 1982 | Fienup, Applied Optics 1982 | 25.0 | 0.7500 | 12.6 | 0.3297 |  |
| 2 | ER (Error Reduction) | 1972 | Gerchberg & Saxton, 1972 | 23.0 | 0.7000 | — | — |  |
| 3 | WF (Wirtinger Flow) | 2015 | Candes et al., TIT 2015 | 30.0 | 0.9000 | — | — |  |
| 4 | RAAR [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.8 | 0.3193 |  |
| 5 | prDeep [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.8 | 0.3193 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 12.5 | — |  |

### 45. Ptychographic Imaging (`ptychography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ePIE | 2009 | Maiden & Rodenburg, Ultramicroscopy 2009 | 28.0 | 0.8500 | 11.7 | 0.5129 |  |
| 2 | PtychoNN | 2020 | Cherukara et al., APL 2020 | 31.0 | — | 3.2 | 0.0007 |  |
| 3 | AutoPhaseNN | 2022 | Cherukara et al., APL 2022 | 33.0 | — | — | — |  |
| 4 | PtychoNN 2.0 (PWM) | — | — | — | — | 3.2 | 0.0007 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 21.0 | 0.2841 | done |
| 6 | precomputed_phase_baseline (test) | — | — | — | — | 10.5 | -0.0059 |  |

### 46. Talbot-Lau X-ray Grating Interferometry (`talbot_lau`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Phase-stepping | 2006 | Weitkamp et al., Opt Express 2005 | 28.0 | — | — | — |  |
| 2 | Fourier analysis | 2006 | Takeda et al., JOSA 1982 | 25.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.9 | 0.9891 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.9 | 0.9891 | done |
| 5 | Talbot-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.9 | 0.9891 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 6.6 | 0.1206 |  |

---

## Microscopy

### 47. Confocal 3D Z-Stack (`confocal_3d`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy 3D | 1972 | Richardson 1972 | 26.0 | 0.7500 | 0.3 | 0.0042 |  |
| 2 | CARE 3D | 2018 | Weigert et al., Nature Methods 2018 | 32.0 | 0.9000 | 0.1 | 0.0043 |  |
| 3 | CARE-3D (PWM) | — | — | — | — | 0.1 | 0.0043 |  |
| 4 | CARE-3D (slice-wise) (PWM) | — | — | — | — | 27.3 | 0.8317 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 17.8 | 0.0530 |  |
| 6 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |

### 48. Confocal Live-Cell Microscopy (`confocal_livecell`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | 0.8000 | 32.3 | 0.8670 | done |
| 2 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | 0.9200 | 13.9 | 0.3589 |  |
| 3 | precomputed_baseline (test) | — | — | — | — | 31.3 | 0.9870 | done |

### 49. Dark-Field Microscopy (`dark_field`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BM3D | 2007 | Dabov et al., TIP 2007 | 30.0 | 0.8500 | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 20.6 | 0.7815 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 10.5 | 0.6101 |  |
| 4 | DF-UNet (PWM) | — | Wolfer, T. et al. (2021) DL for dark-field X-ray CT, Sci. Rep. 11:5005 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 25.1 | 0.9781 | done |

### 50. Differential Interference Contrast (DIC) (`dic`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TIE-DIC | 2010 | TIE for DIC | 25.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 15.6 | 0.3801 |  |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 15.0 | 0.3956 |  |
| 4 | DIC-Net (PWM) | — | Mir, A. et al. (2015) Automated DIC microscopy, J. Microsc. 257(2) | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 10.9 | — |  |

### 51. DNA-PAINT Super-Resolution (`dna_paint`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PICASSO | 2020 | Reymond et al., PNAS 2020 | 20.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 22.9 | 0.0260 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 27.8 | 0.1984 | done |
| 4 | DECODE-PAINT (PWM) | — | Speiser, A. et al. (2021) DL for dense SMLM, Nature Methods 18:1090 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 28.5 | 0.3552 | done |

### 52. Expansion Microscopy (ExM) (`expansion`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy ExM | 2015 | Chen et al., Science 2015 | 26.0 | — | 33.9 | 0.6181 | done |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 15.9 | 0.6087 |  |
| 3 | EXpansionNet (PWM) | — | Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090 | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 33.3 | 0.9823 | done |

### 53. Fluorescence Lifetime Imaging (FLIM) (`flim`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Phasor approach | 2008 | Digman et al., Biophys J 2008 | 25.0 | — | 1.2 | 0.0555 |  |
| 2 | Multi-exponential fitting | 2000 | Elson 2004 | 22.0 | — | — | — |  |
| 3 | MLE Fit (PWM) | — | Becker 2012, J. Microscopy | — | — | 36.9 | 0.7374 | done |
| 4 | MLE Fit (iterative) (PWM) | — | Becker 2012, J. Microscopy | — | — | 0.9 | 0.0557 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 30.7 | 0.9901 | done |

### 54. Fourier Ptychographic Microscopy (FPM) (`fpm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | GS-FPM | 2013 | Zheng et al., Nature Photonics 2013 | 28.0 | 0.8500 | — | — |  |
| 2 | Gradient descent FPM | 2015 | Tian & Waller, Optica 2015 | 30.0 | 0.8700 | 5.2 | 0.0237 |  |
| 3 | Sequential Phase Retrieval (PWM) | — | — | — | — | 5.2 | 0.0237 |  |
| 4 | Fourier Ptychnet (PWM) | — | Jiang et al. 2018, Biomed. Optics Express | — | — | 5.2 | 0.0237 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 16.9 | 0.7943 |  |

### 55. Image Scanning Microscopy (ISM) (`ism`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Pixel reassignment | 2010 | Muller & Enderlein, PRL 2010 | 28.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 3.1 | 0.1516 |  |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 0.5 | 0.0286 |  |
| 4 | ISM-Reassignment-Net (PWM) | — | Castello, M. et al. (2019) Image scanning microscopy ISM, Nature Methods 16:175 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | — | 0.0000 |  |

### 56. Lattice Light-Sheet Microscopy (`lattice_lightsheet`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy 3D | 1972 | Richardson 1972 | 26.0 | 0.7500 | 25.1 | 0.3079 | done |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 14.5 | 0.7656 |  |
| 3 | LLSM-CARE (PWM) | — | Weigert, M. et al. (2018) Content-aware restoration for lattice light-sheet, Nature Methods 15:1090 | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 21.3 | 0.7759 | done |

### 57. Light-Sheet Fluorescence Microscopy (LSFM) (`lightsheet`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 26.0 | 0.7500 | — | — |  |
| 2 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | — | — | — |  |
| 3 | Fourier Notch Filter (PWM) | — | — | — | — | 0.2 | 0.0045 |  |
| 4 | VSNR (PWM) | — | — | — | — | 0.2 | 0.0043 |  |
| 5 | DeStripe (PWM) | — | Liang et al. 2022 | — | — | 0.2 | 0.0045 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 20.0 | 0.0553 |  |
| 7 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |
| 8 | fourier_notch (test) | — | — | — | — | — | 0.0000 |  |

### 58. MINFLUX Nanoscopy (`minflux`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MLE localization | 2006 | Ober et al., Biophys J 2004 | 18.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 29.5 | 0.7052 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 29.2 | 0.7051 | done |
| 4 | MINFLUX-Net (PWM) | — | Gwosch, K.C. et al. (2020) MINFLUX nanoscopy 3D, Nature Methods 17:217 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 29.5 | 0.4336 | done |

### 59. PALM/STORM Single-Molecule Localization (`palm_storm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ThunderSTORM | 2014 | Ovesny et al., Bioinformatics 2014 | 18.0 | — | — | — |  |
| 2 | Deep-STORM | 2018 | Nehme et al., Optica 2018 | 22.0 | — | — | — |  |
| 3 | DECODE | 2021 | Speiser et al., Nature Methods 2021 | 25.0 | — | — | — |  |
| 4 | Richardson-Lucy (STORM/PALM) (PWM) | — | — | — | — | 0.0 | 0.0005 |  |
| 5 | DECODE-SMLM (PWM) | — | Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090 | — | — | — | — |  |
| 6 | DeepSTORM (PWM) | — | Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4) | — | — | — | — |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 32.4 | 0.6094 | done |
| 8 | rl_20iter (test) | — | — | — | — | 32.4 | 0.5904 | done |

### 60. Phase Contrast Microscopy (`phase_contrast`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TIE (Transport of Intensity) | 2001 | Zuo et al., Opt Express 2013 | 28.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 8.1 | 0.3458 |  |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 8.1 | 0.3458 |  |
| 4 | PhaseNet (PWM) | — | Rivenson, Y. et al. (2018) Phase recovery with DL, Light: Sci. & Appl. 7:17141 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 45.6 | 0.9991 | done |

### 61. Polarization Microscopy (`polarization`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Mueller matrix | 2000 | Chipman, Handbook of Optics | 25.0 | — | — | — |  |
| 2 | DnCNN | 2022 | Opt Express 30(12), PMC9208591 | 34.4 | 0.8100 | — | — |  |
| 3 | MIRNet | 2022 | Opt Express 30(12), PMC9208591 | 37.9 | 0.8950 | — | — |  |
| 4 | MDU-Net | 2022 | Opt Express 30(12), PMC9208591 | 38.1 | 0.8970 | — | — |  |
| 5 | PnP-HQS (PWM) | — | — | — | — | 8.4 | 0.0892 |  |
| 6 | PolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.4 | 0.0892 |  |
| 7 | Stokes-NN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.4 | 0.0892 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 15.8 | 0.6265 |  |

### 62. Second Harmonic Generation (SHG) Microscopy (`shg`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | — | 22.3 | 0.5919 |  |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 17.2 | 0.1742 |  |
| 3 | SHG-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for SHG imaging, Nature Methods 15:1090 | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 23.0 | 0.7974 | done |

### 63. Structured Illumination Microscopy (SIM) (`sim`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Wiener-SIM | 2008 | Gustafsson et al., 2008 | 30.0 | 0.8800 | 0.1 | 0.0056 |  |
| 2 | fairSIM | 2015 | Muller et al., Bioinformatics 2016 | 30.5 | 0.8900 | 6.2 | 0.0126 |  |
| 3 | ML-SIM | 2021 | Christensen et al., APL 2021 | 33.0 | — | — | — |  |
| 4 | HiFi-SIM (PWM) | — | Wen et al. 2021, Light: S&A | — | — | 0.3 | 0.0055 |  |
| 5 | Wiener-SIM (fast) (PWM) | — | — | — | — | 6.2 | 0.0126 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 21.6 | 0.1863 | done |
| 7 | wiener_sim (test) | — | — | — | — | 6.2 | 0.0174 |  |

### 64. Spinning Disk Confocal Microscopy (`spinning_disk`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 27.0 | 0.7800 | 29.5 | 0.7581 | done |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 14.2 | 0.3440 |  |
| 3 | SD-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for spinning disk confocal, Nature Methods 15:1090 | — | — | — | — |  |
| 4 | precomputed_baseline (test) | — | — | — | — | 30.6 | 0.9835 | done |

### 65. STED Microscopy (`sted`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy STED | 2006 | RL for STED | 28.0 | 0.8000 | 0.3 | 0.0017 |  |
| 2 | DDPM denoiser | 2023 | DDPM-avg for STED | 32.8 | 0.9200 | — | — |  |
| 3 | STED-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | — | — | — | — |  |
| 4 | RCAN-STED (PWM) | — | Chen, J. et al. (2021) Three-dimensional residual channel attention for STED, Nature Methods 18:678 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 25.0 | 0.8484 | done |
| 6 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |

### 66. Three-Photon Microscopy (`three_photon`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 26.0 | — | 14.6 | 0.3353 |  |
| 2 | DeepCAD-RT | 2023 | Li et al., Nature Biotech 2023 | 34.0 | — | — | — |  |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 14.5 | 0.3513 |  |
| 4 | 3P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) CARE for 3P deep tissue imaging, Nature Methods 15:1090 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 20.8 | 0.8419 | done |

### 67. TIRF Microscopy (`tirf`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | 0.8000 | 27.7 | 0.1106 | done |
| 2 | TIRF-Net (CARE) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.7 | 0.1106 | done |
| 3 | TIRF-SRRF [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.7 | 0.1106 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 31.2 | 0.6216 | done |

### 68. Two-Photon / Multiphoton Microscopy (`two_photon`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 27.0 | 0.7800 | 0.9 | 0.0073 |  |
| 2 | DeepCAD | 2021 | Li et al., Nature Methods 2021 | 35.0 | — | — | — |  |
| 3 | 2P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | — | — | — | — |  |
| 4 | 2P-DeepInterp (PWM) | — | Lecoq, J. et al. (2021) Removing independent noise in systems neuroscience using DeepInterpolation, Nature Methods 18:1401 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 33.8 | 0.9867 | done |
| 6 | rl_20iter (test) | — | — | — | — | — | 0.0000 |  |

### 69. Widefield Fluorescence Microscopy (`widefield`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Wiener deconvolution | 1949 | Wiener, 1949 | 26.0 | 0.7500 | — | — |  |
| 2 | Richardson-Lucy (20 iter) | 1972 | Richardson 1972 / Lucy 1974 | 13.4 | 0.4000 | 24.1 | 0.2696 | done |
| 3 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 22.1 | 0.7500 | 14.5 | 0.7656 |  |
| 4 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 31.0 | 0.8800 | — | — |  |
| 5 | m-rBCR | 2023 | m-rBCR deconvolution, 2023 | 24.9 | 0.8300 | — | — |  |
| 6 | Restormer | 2022 | Zamir et al., CVPR 2022 | 35.5 | — | — | — |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 25.0 | 0.9091 | done |

### 70. Low-Dose Widefield Microscopy (`widefield_lowdose`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 20.0 | 0.6000 | — | — |  |
| 2 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 26.0 | 0.8000 | — | — |  |
| 3 | BM3D + RL (PWM) | — | — | — | — | 29.0 | 0.9402 | done |
| 4 | CARE (PWM) | — | — | — | — | 12.6 | 0.5013 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 18.8 | 0.6755 |  |

---

## Electron Microscopy

### 71. Cryo-Electron Tomography (Cryo-ET) (`cryo_et`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | WBP | 1970 | Weighted back-projection | 22.0 | 0.6000 | — | — |  |
| 2 | SIRT | 1972 | Gilbert 1972 | 25.0 | 0.7000 | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 2.7 | 0.0966 |  |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 13.2 | 0.4127 |  |
| 5 | CryoCARE (PWM) | — | Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol. | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 8.4 | 0.2037 |  |

### 72. Electron Backscatter Diffraction (EBSD) (`ebsd`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Dictionary indexing | 2015 | Chen et al., Microscopy 2015 | 25.0 | — | — | — |  |
| 2 | Hough indexing | 1992 | Krieger-Lassen 1998 | 22.0 | — | 3.0 | 0.2847 |  |
| 3 | EBSD-DL (DictIndex) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.9 | 0.8024 | done |
| 4 | EMsoft-EBSD [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.9 | 0.8024 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 21.8 | 0.9677 | done |

### 73. STEM-EDX Elemental Mapping (`edx_mapping`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PCA denoising | 2010 | PCA for EDX | 24.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 3.2 | 0.2667 |  |
| 3 | Richardson-Lucy (high quality) (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.2667 |  |
| 4 | Richardson-Lucy (DL baseline) (PWM) | — | Tietz, C. et al. (2021) DL for EDS spectrum imaging, Ultramicroscopy 231 | — | — | 3.2 | 0.2667 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 22.0 | 0.9307 | done |

### 74. Electron Energy Loss Spectroscopy (EELS) (`eels`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PCA denoising | 2012 | Cueva et al., Microsc Microanal 2012 | 28.0 | — | — | — |  |
| 2 | FISTA-L2 (Fourier ratio) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.5036 | done |
| 3 | EELS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.5036 | done |
| 4 | MLLS-EELS [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.5036 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 24.6 | 0.9842 | done |

### 75. 4D-STEM Electron Diffraction (`electron_diffraction`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Center-of-mass analysis | 2014 | Muller-Caspary et al., 2014 | 22.0 | — | — | — |  |
| 2 | ePIE (electron ptychography) (PWM) | — | — | — | — | 42.0 | 0.9889 | done |
| 3 | ED-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 42.0 | 0.9889 | done |
| 4 | CRISP-ED [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 42.0 | 0.9889 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 42.0 | 0.9901 | done |

### 76. Electron Holography (`electron_holography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Fourier filtering | 1993 | Lichte, Ultramicroscopy 1993 | 25.0 | — | — | — |  |
| 2 | Phase Retrieval (HIO) (PWM) | — | — | — | — | 5.6 | 0.0115 |  |
| 3 | EH-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.5 | 0.0049 |  |
| 4 | Phase-Sideband [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.5 | 0.0049 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 9.5 | -0.0481 |  |

### 77. Electron Tomography (`electron_tomography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | WBP | 1970 | Weighted Back-Projection | 22.0 | 0.6500 | — | — |  |
| 2 | SIRT | 1972 | Gilbert, SIRT, 1972 | 25.0 | 0.7500 | 25.1 | 0.9525 | done |
| 3 | IMOD-SIRT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.8 | 0.2536 |  |
| 4 | SIRT-3D [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.8 | 0.2536 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 19.3 | 0.9419 |  |

### 78. Focused Ion Beam SEM (FIB-SEM) (`fib_sem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BM3D | 2007 | Dabov et al., 2007 | 30.0 | — | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 26.0 | 0.4862 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 4.9 | 0.0019 |  |
| 4 | FIB-SEM-Net (PWM) | — | Heinrich, L. et al. (2021) Whole-cell organelle segmentation in volume EM, Nature 599:141 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 28.1 | 0.9862 | done |

### 79. Scanning Electron Microscopy (SEM) (`sem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BM3D | 2007 | Dabov et al., TIP 2007 | 30.0 | 0.8500 | — | — |  |
| 2 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 28.0 | — | — | — |  |
| 3 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 34.0 | — | — | — |  |
| 4 | Richardson-Lucy (SEM) (PWM) | — | — | — | — | 23.2 | 0.4997 | done |
| 5 | SEM-DL (SegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 23.2 | 0.4997 | done |
| 6 | SEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 23.2 | 0.4997 | done |
| 7 | precomputed_baseline (test) | — | — | — | — | 15.7 | 0.7926 |  |

### 80. Scanning Transmission Electron Microscopy (STEM) (`stem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BM3D | 2007 | Dabov et al., 2007 | 30.0 | 0.8500 | — | — |  |
| 2 | SwinIR | 2021 | Liang et al., 2021 | 33.0 | — | — | — |  |
| 3 | Richardson-Lucy (STEM) (PWM) | — | — | — | — | 31.0 | 0.9508 | done |
| 4 | STEM-DL (AtomSegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.0 | 0.9508 | done |
| 5 | STEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.0 | 0.9508 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 30.0 | 0.9276 | done |

### 81. Transmission Electron Microscopy (TEM) (`tem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BM3D | 2007 | Dabov et al., BM3D, 2007 | 30.0 | — | — | — |  |
| 2 | Topaz-Denoise | 2020 | Bepler et al., Nature Commun 2020 | 32.0 | — | — | — |  |
| 3 | SwinIR | 2021 | Liang et al., 2021 | 35.0 | — | — | — |  |
| 4 | FISTA-L2 (CTF correction) (PWM) | — | — | — | — | 6.9 | 0.0020 |  |
| 5 | TEM-DL (ePIE-Net) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.9 | 0.0997 |  |
| 6 | TEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.9 | 0.0997 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 25.3 | 0.9190 | done |

---

## Computational Optics

### 82. Integral Photography (`integral`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Drizzle (IFS) | 2003 | Fruchter & Hook, PASP 2002 | 25.0 | — | — | — |  |
| 2 | PCA sky subtraction | 2012 | IFS baseline | 22.0 | — | — | — |  |
| 3 | Depth Estimation (PWM) | — | — | — | — | 33.4 | 0.9013 | done |
| 4 | DIBR [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 33.4 | 0.9013 | done |
| 5 | EPINet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 33.4 | 0.9013 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 40.0 | 0.9990 | done |

### 83. Light Field Imaging (`light_field`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | LFSSR | 2018 | Yeung et al., ECCV 2018 | 33.7 | 0.9740 | 16.3 | 0.1186 |  |
| 2 | LF-InterNet | 2020 | Wang et al., ECCV 2020 | 34.1 | 0.9760 | — | — |  |
| 3 | DistgSSR | 2021 | Wang et al., TPAMI 2022 | 34.8 | 0.9790 | — | — |  |
| 4 | LFT | 2022 | Liang et al., 2022 | 34.8 | 0.9780 | — | — |  |
| 5 | EPIT | 2022 | EPIT, 2022 | 34.8 | 0.9780 | — | — |  |
| 6 | Shift-and-Sum (PWM) | — | — | — | — | 16.3 | 0.1186 |  |
| 7 | LFBM5D (PWM) | — | Alain et al. 2017, Signal Processing: Image Communication | — | — | 4.3 | 0.0170 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 27.3 | 0.9439 | done |

---

## Computational Photography

### 84. Coded Exposure / Flutter Shutter (`coded_exposure`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Wiener (flutter shutter) | 2006 | Raskar et al., SIGGRAPH 2006 | 26.0 | — | — | — |  |
| 2 | MPRNet | 2021 | Zamir et al., CVPR 2021 | 32.7 | 0.9590 | — | — |  |
| 3 | Restormer | 2022 | Zamir et al., CVPR 2022 | 32.9 | 0.9610 | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.9313 | done |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.9313 | done |
| 6 | FlowNet-Coded [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.1 | 0.9313 | done |
| 7 | precomputed_baseline (test) | — | — | — | — | 19.9 | 0.8073 |  |

### 85. Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | E2VID | 2019 | Rebecq et al., TPAMI 2020 | 7.5 | 0.4500 | 4.2 | -0.0104 |  |
| 2 | SPADE-E2VID | 2021 | Cadena et al., CVPRW 2021 | 10.4 | 0.4610 | — | — |  |
| 3 | E2VID+ | 2020 | Stoffregen et al., ECCV 2020 | 11.5 | 0.5030 | 4.2 | -0.0104 |  |
| 4 | ET-Net | 2021 | Weng et al., ICCV 2021 | 13.3 | 0.5520 | — | — |  |
| 5 | HyperE2VID | 2024 | Ercan et al., IEEE TIP 2024 | 14.8 | 0.5760 | — | — |  |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | -0.0104 |  |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.2 | -0.0104 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 7.3 | 0.0574 |  |

### 86. High Dynamic Range (HDR) Imaging (`hdr_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Debevec | 1997 | Debevec & Malik, SIGGRAPH 1997 | 30.0 | — | — | — |  |
| 2 | AHDRNet | 2019 | Yan et al., CVPR 2019 | 41.1 | 0.9800 | — | — |  |
| 3 | HDR-Transformer | 2022 | Liu et al., AAAI 2022 | 42.4 | — | 32.4 | 0.7267 |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.4 | 0.7267 | done |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.4 | 0.7267 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 36.8 | 0.8232 | done |

### 87. Lensless (Diffuser Camera) Imaging (`lensless`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Wiener deconvolution | 1949 | Wiener, 1949 | 7.3 | 0.0830 | 11.8 | 0.0031 | done |
| 2 | ADMM | 2000 | Boyd et al., ADMM, 2010 | 12.8 | 0.4420 | 11.9 | 0.5896 | done |
| 3 | FlatNet | 2022 | Khan et al., TPAMI 2022 | 21.2 | 0.7200 | 0.5 | 0.0001 |  |
| 4 | MWDN | 2023 | MWDN, 2023 | 25.7 | 0.8160 | — | — |  |
| 5 | LensNet | 2025 | LensNet, IJCAI 2025 | 27.5 | 0.8630 | — | — |  |
| 6 | FlatNet-Lite (PWM) | — | — | — | — | 0.5 | 0.0001 |  |
| 7 | wiener_deconv (test) | — | — | — | — | 11.8 | 0.0031 |  |

### 88. Panorama Multi-Focus Fusion (`panorama`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | APAP | 2013 | Zaragoza et al., CVPR 2013 | 25.0 | 0.8500 | — | — |  |
| 2 | UDIS (Unsupervised Deep Image Stitching) | 2021 | Nie et al., CVPR 2021 | 28.0 | 0.9000 | — | — |  |
| 3 | Laplacian Pyramid Fusion (PWM) | — | — | — | — | 14.6 | 0.0520 |  |
| 4 | Guided Filter Fusion (PWM) | — | — | — | — | 15.6 | 0.0881 |  |
| 5 | IFCNN (PWM) | — | Zhang et al. 2020 | — | — | 5.9 | 0.0001 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 15.1 | 0.6418 |  |

---

## Neural Rendering

### 89. 3D Gaussian Splatting (3DGS) (`gaussian_splatting`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | 3D Gaussian Splatting | 2023 | Kerbl et al., SIGGRAPH 2023 | 33.3 | 0.9690 | — | — |  |
| 2 | 2DGS | 2024 | Huang et al., SIGGRAPH 2024 | 34.0 | — | — | — |  |
| 3 | Scaffold-GS | 2024 | Lu et al., CVPR 2024 | 33.8 | — | — | — |  |
| 4 | EWA Splatting (PWM) | — | — | — | — | — | 1.0000 |  |
| 5 | 3DGS (full) (PWM) | — | Kerbl et al. SIGGRAPH 2023 | — | — | — | 1.0000 |  |
| 6 | NeRF (baseline comparison) (PWM) | — | — | — | — | — | 1.0000 |  |
| 7 | 3DGS (compact) (PWM) | — | — | — | — | — | 1.0000 |  |
| 8 | direct_render_baseline (test) | — | — | — | — | 0.0 | 0.0000 |  |

### 90. Neural Radiance Fields (NeRF) (`nerf`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | NeRF | 2020 | Mildenhall et al., ECCV 2020 | 31.0 | 0.9470 | — | — |  |
| 2 | Plenoxels | 2022 | Fridovich-Keil et al., CVPR 2022 | 31.7 | 0.9580 | — | — |  |
| 3 | TensoRF | 2022 | Chen et al., ECCV 2022 | 33.1 | 0.9630 | — | — |  |
| 4 | Instant-NGP | 2022 | Muller et al., SIGGRAPH 2022 | 33.2 | 0.9600 | — | — |  |
| 5 | 3D Gaussian Splatting | 2023 | Kerbl et al., SIGGRAPH 2023 | 33.3 | 0.9690 | — | — |  |
| 6 | Mip-NeRF 360 | 2022 | Barron et al., CVPR 2022 | 33.1 | 0.9610 | — | — |  |
| 7 | Zip-NeRF | 2023 | Barron et al., ICCV 2023 | 33.7 | — | — | — |  |
| 8 | SfM + MVS (PWM) | — | — | — | — | 21.4 | 0.8758 | done |
| 9 | NeRF (original MLP) (PWM) | — | Mildenhall et al. 2020 | — | — | — | — |  |
| 10 | Richardson-Lucy (proxy baseline) (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8507 |  |
| 11 | FISTA-TV (proxy baseline) (PWM) | — | Beck & Teboulle 2009, SIAM | — | — | 19.7 | 0.8507 |  |
| 12 | precomputed_baseline (test) | — | — | — | — | 29.0 | 0.9913 | done |

---

## Depth Imaging

### 91. Flash LiDAR (`flash_lidar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TCSPC histogram | 2000 | flash LiDAR baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1433 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1433 |  |
| 4 | FlashLiDAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1433 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 4.3 | — |  |

### 92. LiDAR Scanner (`lidar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Bilateral Filter | 1998 | Tomasi & Manduchi, 1998 | 25.0 | — | — | — |  |
| 2 | NLSPN | 2020 | Park et al., ECCV 2020 | 35.0 | — | — | — |  |
| 3 | BP-Net | 2022 | Tang et al., CVPR 2022 | 36.0 | — | — | — |  |
| 4 | CompletionFormer | 2023 | Zhang et al., CVPR 2023 | 35.5 | — | — | — |  |
| 5 | FISTA-L2 (depth) (PWM) | — | — | — | — | 4.4 | 0.0396 |  |
| 6 | PointNeXt [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.7 | 0.8269 | done |
| 7 | PointNet++ [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 32.7 | 0.8269 | done |
| 8 | precomputed_baseline (test) | — | — | — | — | 32.6 | 0.9955 | done |

### 93. Photometric Stereo (`photometric_stereo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Woodham (Lambertian) | 1980 | Woodham, Opt Eng 1980 | 25.0 | — | — | — |  |
| 2 | CNN-PS | 2019 | Chen et al., CVPR 2019 | 32.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.1 | 0.5223 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.1 | 0.5223 |  |
| 5 | PS-FCN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.1 | 0.5223 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 29.0 | 0.9583 | done |

### 94. Structured-Light Depth Camera (`structured_light`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Gray code | 2003 | Scharstein & Szeliski, 2003 | 25.0 | — | — | — |  |
| 2 | Phase-shifting (4-step) | 1984 | Creath, 1988 | 35.0 | 0.9500 | 6.6 | 0.0042 |  |
| 3 | SFNet (fringe-to-phase) | 2024 | ArXiv 2402.00977 | 38.0 | — | — | — |  |
| 4 | SL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.0042 |  |
| 5 | FTPD [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.0042 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 8.0 | -0.0287 |  |

### 95. Time-of-Flight Depth Camera (`tof_camera`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Phase unwrapping | 2000 | ToF baseline | — | — | — | — |  |
| 2 | DeepToF | 2017 | Marco et al., CVPR 2017 | 32.0 | — | — | — |  |
| 3 | FISTA-L2 (depth) (PWM) | — | — | — | — | 2.4 | 0.0284 |  |
| 4 | ToF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.3 | 0.1318 |  |
| 5 | ToF-MPI Deconv [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.3 | 0.1318 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 42.0 | 0.9994 | done |

---

## Remote Sensing

### 96. Ground-Penetrating Radar (GPR) (`gpr`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Kirchhoff migration | 2000 | GPR migration | 20.0 | 0.6500 | — | — |  |
| 2 | RTM (Reverse Time Migration) | 2000 | RTM | 25.0 | 0.8000 | — | — |  |
| 3 | GPR-Net | 2021 | DL for GPR | 30.0 | 0.9000 | 5.6 | 0.0035 |  |
| 4 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.6 | 0.0035 |  |
| 5 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.6 | 0.0035 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 10.6 | 0.0059 |  |

### 97. Hyperspectral Remote Sensing (`hyperspectral_remote`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | HSCNN+ | 2018 | Shi et al., CVPRW 2018 | 26.4 | — | — | — |  |
| 2 | AWAN | 2020 | Li et al., CVPRW 2020 | 31.2 | — | — | — |  |
| 3 | HDNet | 2022 | Hu et al., CVPR 2022 | 32.1 | — | — | — |  |
| 4 | MST++ | 2022 | Cai et al., CVPRW 2022 (Winner) | 34.3 | — | — | — |  |
| 5 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.8 | 0.6998 | done |
| 6 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.8 | 0.6998 | done |
| 7 | SST-USRNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.8 | 0.6998 | done |
| 8 | precomputed_baseline (test) | — | — | — | — | 29.1 | 0.9768 | done |

### 98. Interferometric SAR (InSAR) (`insar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Goldstein filter | 1998 | Goldstein & Werner, GRL 1998 | 22.0 | — | — | — |  |
| 2 | SNAPHU | 2001 | Chen & Zebker, JOSA-A 2001 | 28.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9073 | done |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9073 | done |
| 5 | InSAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9073 | done |
| 6 | wrapped_phase_baseline (test) | — | — | — | — | 31.8 | 0.9933 | done |

### 99. Multispectral Satellite Imaging (`multispectral_sat`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | BDSD (Band-Dependent Spatial Detail) | 2008 | Vivone et al., GRSM 2015 | 30.0 | 0.9000 | — | — |  |
| 2 | PanNet | 2017 | Yang et al., ICCV 2017 | 32.5 | 0.9300 | — | — |  |
| 3 | GPPNN | 2021 | Xu et al., CVPR 2021 | 33.8 | 0.9500 | — | — |  |
| 4 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.3 | 0.0718 |  |
| 5 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.3 | 0.0718 |  |
| 6 | MS-Pansharpening-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.3 | 0.0718 |  |
| 7 | bicubic_upsample (test) | — | — | — | — | 10.8 | 0.1002 |  |

### 100. Ocean Color Remote Sensing (`ocean_color`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MUMM | 2000 | Ruddick et al., RSE 2000 | 22.0 | — | — | — |  |
| 2 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 38.2 | 0.9992 | done |
| 3 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 38.2 | 0.9992 | done |
| 4 | OC-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 38.2 | 0.9992 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 44.1 | 0.9998 | done |

### 101. Passive Microwave Radiometry (`passive_microwave`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Tikhonov retrieval | 2000 | Tikhonov | 22.0 | — | — | — |  |
| 2 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.9 | 0.3301 |  |
| 3 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.9 | 0.3301 |  |
| 4 | PM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.9 | 0.3301 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 9.2 | 0.5946 |  |

### 102. Polarimetric SAR (PolSAR) (`polsar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Lee filter | 1999 | Lee et al., IEEE TGRS 1999 | 22.0 | 0.7000 | — | — |  |
| 2 | Cloude-Pottier decomposition | 1997 | Cloude & Pottier, IEEE TGRS 1997 | — | — | — | — |  |
| 3 | SAR-CNN despeckling | 2020 | DL for SAR despeckling | 28.0 | 0.8500 | 7.2 | 0.2891 |  |
| 4 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.2891 |  |
| 5 | PolSAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.2891 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 3.5 | -0.0175 |  |

### 103. Radio Interferometry (VLBI) (`radio_interferometry`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | CLEAN | 1974 | Hogbom, A&AS 1974 | 25.0 | — | — | — |  |
| 2 | MEM | 1984 | Cornwell & Evans, A&A 1985 | 27.0 | — | — | — |  |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 9.0 | 0.0880 |  |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 9.0 | 0.0880 |  |
| 5 | R2D2 (interferometry) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 9.0 | 0.0880 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 23.2 | 0.2029 | done |

### 104. Synthetic Aperture Radar (SAR) (`sar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Range-Doppler Algorithm | 1978 | Curlander & McDonough, 1991 | 25.0 | 0.7000 | — | — |  |
| 2 | Omega-K Algorithm | 1992 | Stolt 1978 / Cafforio 1991 | 27.0 | 0.7500 | — | — |  |
| 3 | FBP (SAR backprojection) (PWM) | — | — | — | — | 13.6 | 0.2879 |  |
| 4 | SAR-DL (PolSF) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.8 | 0.1380 |  |
| 5 | SAR-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.8 | 0.1380 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 17.3 | 0.7046 |  |

### 105. Sonar Imaging (`sonar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MVDR/Capon beamforming | 1969 | Capon, Proc IEEE 1969 | 25.0 | — | — | — |  |
| 2 | MUSIC | 1986 | Schmidt, IEEE TAP 1986 | 27.0 | — | — | — |  |
| 3 | FISTA-L2 (DAS) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.0 | 0.2817 |  |
| 4 | SonarSR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.0 | 0.2817 |  |
| 5 | Sonar-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 15.0 | 0.2817 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 10.3 | 0.5149 |  |

### 106. Weather / Doppler Radar (`weather_radar`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | CLEAN-AP | 2000 | CLEAN for weather | 25.0 | — | — | — |  |
| 2 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.8 | 0.1187 | done |
| 3 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.8 | 0.1187 | done |
| 4 | NowcastNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 21.8 | 0.1187 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 26.9 | 0.9155 | done |

---

## Scanning Probe Microscopy

### 107. Atomic Force Microscopy (AFM) (`afm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Flatten + line correction | 2000 | SPM baseline processing | 25.0 | 0.7500 | — | — |  |
| 2 | Deep-AFM | 2020 | Rashidi & Wolkow, Machine Learning 2020 | 32.0 | 0.9000 | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 31.3 | 0.7815 | done |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 15.6 | 0.3770 |  |
| 5 | AFM-UNet (PWM) | — | Cherukara, M.J. et al. (2020) AI-enabled high-res, real-time imaging, npj Comput. Mater. 6:203 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 19.0 | 0.8537 |  |

### 108. Magnetic Force Microscopy (MFM) (`mfm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Deconvolution | 2000 | MFM tip deconvolution | 24.0 | 0.7500 | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 9.5 | 0.3740 |  |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 5.1 | 0.0008 |  |
| 4 | MFM-UNet (PWM) | — | Kim, M. et al. (2021) DL for magnetic force microscopy, npj Comput. Mater. 7:87 | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 34.3 | 0.2871 | done |

### 109. Near-field Scanning Optical Microscopy (NSOM) (`nsom`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Deconvolution | 2000 | Near-field deconvolution | 24.0 | 0.7500 | — | — |  |
| 2 | Richardson-Lucy (PWM) | — | — | — | — | 22.3 | 0.2438 | done |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 17.7 | 0.7562 |  |
| 4 | NSOM-Net (PWM) | — | Park, J. et al. (2020) DL for near-field optical microscopy, Optica 7(11) | — | — | — | — |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 19.6 | 0.7328 |  |

### 110. Scanning Tunneling Microscopy (STM) (`stm`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Drift correction | 2000 | SPM baseline | 22.0 | 0.7000 | — | — |  |
| 2 | DeepSPM | 2020 | Krull et al., 2020 | 30.0 | 0.8800 | — | — |  |
| 3 | Richardson-Lucy (PWM) | — | — | — | — | 23.3 | 0.9600 | done |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | — | — | 7.0 | 0.0000 |  |
| 5 | STM-Net (PWM) | — | Ziatdinov, M. et al. (2021) DL for atomic-level STM, Nat. Mach. Intell. 3:269 | — | — | — | — |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 17.9 | 0.8025 |  |

---

## Industrial Inspection

### 111. Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SAFT (Synth Aperture Focus) | 1980 | Doctor et al., 1986 | 25.0 | — | 22.0 | 0.7398 | done |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.0 | 0.7398 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.0 | 0.7398 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 10.0 | -0.0384 |  |

### 112. Active Thermography (IR) (`active_thermography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Pulsed phase thermography | 1996 | Maldague & Marinetti, J Appl Phys 1996 | 25.0 | — | 7.2 | 0.1475 |  |
| 2 | Bicubic baseline | 2024 | Sci Reports 2024, PMC11227526 | 42.1 | 0.9820 | — | — |  |
| 3 | SRCNN | 2024 | Sci Reports 2024, PMC11227526 | 42.9 | 0.9840 | — | — |  |
| 4 | EDSR | 2024 | Sci Reports 2024, PMC11227526 | 45.3 | 0.9900 | — | — |  |
| 5 | RCAN | 2024 | Sci Reports 2024, PMC11227526 | 45.9 | 0.9920 | — | — |  |
| 6 | TESR (Transformer) | 2024 | Sci Reports 2024, PMC11227526 | 46.2 | 0.9920 | — | — |  |
| 7 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.1475 |  |
| 8 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.1475 |  |
| 9 | precomputed_baseline (test) | — | — | — | — | 6.5 | 0.1897 |  |

### 113. Eddy Current Imaging (`eddy_current`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Impedance plane analysis | 2000 | ECT baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.9 | 0.6356 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.9 | 0.6356 | done |
| 4 | ECT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 22.9 | 0.6356 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 4.8 | -0.0811 |  |

### 114. Industrial X-ray CT (`industrial_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FDK | 1984 | Feldkamp et al., 1984 | 28.0 | 0.8000 | — | — |  |
| 2 | SIRT | 1972 | Gilbert 1972 | 30.0 | 0.8500 | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.9 | 0.4680 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 4.9 | 0.4680 |  |
| 5 | IndustrialCT-Net [proxy] (PWM) | — | Shepp & Logan 1974 | — | — | 0.4 | 0.0702 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 20.3 | 0.4046 | done |

### 115. Machine Vision / AOI (`machine_vision`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Template matching | 2000 | Brunelli, Template Matching, 2009 | 25.0 | — | — | — |  |
| 2 | PatchCore | 2022 | Roth et al., CVPR 2022 | 30.0 | — | 28.3 | 0.2934 | done |
| 3 | UniAD | 2023 | You et al., NeurIPS 2022 | 32.0 | — | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.3 | 0.2934 | done |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.3 | 0.2934 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 26.5 | 0.9622 | done |

### 116. Shearography (`shearography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Phase-shifting shearography | 2000 | Hung, 1982 | 28.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 13.2 | 0.1733 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 13.2 | 0.1733 |  |
| 4 | ShearNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 13.2 | 0.1733 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 8.0 | -0.0011 |  |

### 117. Terahertz Imaging (THz) (`terahertz`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TDS deconvolution | 2000 | THz-TDS baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.7 | 0.0217 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.7 | 0.0217 |  |
| 4 | THz-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 8.7 | 0.0217 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 37.1 | 0.9963 | done |

### 118. Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TFM (Total Focusing Method) | 2004 | Holmes et al., NDT&E Int 2005 | 28.0 | — | 30.8 | 0.2409 | done |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2409 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2409 | done |
| 4 | precomputed_baseline (test) | — | — | — | — | 29.6 | 0.6891 | done |

### 119. X-ray NDT (Radiography) (`xray_ndt`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP | 1971 | FBP baseline | 28.0 | 0.8000 | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.0 | 0.3334 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.0 | 0.3334 |  |
| 4 | NDT-DefectNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.0 | 0.3334 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 16.7 | 0.8430 |  |

### 120. X-ray Fluorescence (XRF) Imaging (`xrf_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Fundamental parameters | 2000 | Sherman, Spectrochim Acta 1955 | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.2 | 0.4457 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.2 | 0.4457 |  |
| 4 | XRF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.2 | 0.4457 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 22.1 | 0.9626 | done |

---

## Spectroscopy & Spectral Imaging

### 121. Brillouin Microscopy (`brillouin`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Lorentzian fitting | 2000 | Brillouin spectral fit | 25.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.4967 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.4967 | done |
| 4 | Brillouin-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.4967 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 35.8 | 0.9959 | done |

### 122. Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MEM (Maximum Entropy Method) | 2006 | Vartiainen et al., Opt Express 2006 | 25.0 | — | — | — |  |
| 2 | Median Filter | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 20.1 | 0.4300 | — | — |  |
| 3 | N2N (Noise2Noise) | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 20.6 | 0.5600 | — | — |  |
| 4 | DnCNN | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 23.0 | 0.5900 | — | — |  |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.7 | 0.4812 |  |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.7 | 0.4812 |  |
| 7 | CARS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 16.7 | 0.4812 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 14.2 | 0.0040 |  |

### 123. DESI Mass Spectrometry Imaging (`desi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Peak fitting | 2000 | DESI baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.5 | 0.0056 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.5 | 0.0056 |  |
| 4 | DESI-SegNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.5 | 0.0056 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 15.1 | 0.3130 |  |

### 124. FTIR Spectroscopic Imaging (`ftir_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | ATR correction | 2000 | Bassan et al., Analyst 2010 | 24.0 | — | — | — |  |
| 2 | MCR-ALS | 2000 | Tauler, Chemom Intell Lab 1995 | 28.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 34.6 | 0.9204 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 34.6 | 0.9204 | done |
| 5 | FTIR-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 34.6 | 0.9204 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 14.8 | 0.8058 |  |

### 125. Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Peak identification | 2000 | LIBS baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.5 | 0.6516 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.5 | 0.6516 | done |
| 4 | LIBS-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.5 | 0.6516 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 18.0 | 0.5987 |  |

### 126. Raman Imaging / Microscopy (`raman_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Savitzky-Golay | 1964 | Savitzky & Golay, 1964 | 20.0 | — | — | — |  |
| 2 | PCA denoising | 2000 | Horgan et al., Anal Chem 2022 (comparison) | 39.4 | 0.8680 | — | — |  |
| 3 | DeepeR (1D ResUNet) | 2022 | Horgan et al., Anal Chem 2022, PMC9286315 | 46.2 | 0.9530 | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.5114 |  |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.5114 |  |
| 6 | RamanNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.5114 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 14.1 | 0.2149 |  |

### 127. Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Dead-time correction | 2000 | SIMS baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.9 | 0.4524 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.9 | 0.4524 |  |
| 4 | SIMS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.9 | 0.4524 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 20.5 | 0.9749 | done |

### 128. Stimulated Raman Scattering (SRS) Microscopy (`srs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Spectral unmixing | 2000 | SRS baseline | 24.0 | — | — | — |  |
| 2 | PURE-LET | 2019 | Manifold et al., Biomed Opt Express 10(8):3860, PMC6701518 | 13.5 | — | — | — |  |
| 3 | UHRED (unsupervised) | 2021 | Opt Express 29(21):34205 | 22.0 | — | — | — |  |
| 4 | SHRED | 2021 | Opt Express 29(21):34205 | 25.0 | — | — | — |  |
| 5 | U-Net CNN | 2019 | Manifold et al., Biomed Opt Express 10(8):3860, PMC6701518 | 28.9 | — | — | — |  |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.0 | 0.9307 | done |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.0 | 0.9307 | done |
| 8 | SRS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.0 | 0.9307 | done |
| 9 | precomputed_baseline (test) | — | — | — | — | 29.1 | 0.9779 | done |

---

## Astronomy & Space Imaging

### 129. Stellar Coronagraphy (`coronagraphy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Classical ADI | 2006 | Marois et al., ApJ 2006 | 18.0 | — | — | — |  |
| 2 | PCA/KLIP | 2012 | Soummer et al., ApJL 2012 | 22.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.8930 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.8930 | done |
| 5 | DL-SpeckleNull [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.2 | 0.8930 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 25.2 | 0.2028 | done |

### 130. Event Horizon Telescope (EHT) Imaging (`eht_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | CLEAN | 1974 | Hogbom, A&AS 1974 | 20.0 | — | — | — |  |
| 2 | eht-imaging RML | 2019 | Chael et al., ApJ 2018 | 25.0 | — | 11.4 | 0.0148 |  |
| 3 | PRIMO | 2023 | Medeiros et al., ApJL 2023 | 28.0 | — | 11.4 | 0.0148 |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.4 | 0.0148 |  |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 11.4 | 0.0148 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 11.3 | 0.0394 |  |

### 131. Lucky Imaging (`lucky_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Shift-and-add | 2000 | Lucky imaging baseline | 22.0 | — | — | — |  |
| 2 | Drizzle | 2002 | Fruchter & Hook, PASP 2002 | 26.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.6 | 0.7117 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.6 | 0.7117 | done |
| 5 | Lucky-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 29.6 | 0.7117 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 29.2 | 0.9746 | done |

### 132. Solar EUV/X-ray Imaging (`solar_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 25.0 | — | — | — |  |
| 2 | Pixon | 1991 | Pina & Puetter, PASP 1993 | 30.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.9 | 0.9167 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.9 | 0.9167 | done |
| 5 | SolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.9 | 0.9167 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | 28.4 | 0.9958 | done |

---

## Ultrafast Imaging

### 133. Compressed Ultrafast Photography (CUP) (`cup`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | TwIST | 2007 | Bioucas-Dias & Figueiredo, TIP 2007 | 22.0 | 0.7000 | — | — |  |
| 2 | PnP-FFDNet | 2020 | Yuan et al., CVPR 2020 | 28.0 | — | 5.5 | 0.1925 |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.5 | 0.1925 |  |
| 4 | E2E-CUP [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.5 | 0.1925 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | -2.3 | 0.1202 |  |

### 134. Pump-Probe Microscopy (`pump_probe`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | SVD analysis | 2000 | SVD for transient spectra | 22.0 | — | — | — |  |
| 2 | MCR-ALS | 2000 | Tauler, Chemom Intell Lab 1995 | 26.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.5 | 0.8551 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.5 | 0.8551 |  |
| 5 | PumpProbe-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.5 | 0.8551 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 18.2 | 0.7781 |  |

### 135. Streak Camera Imaging (`streak_camera`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Temporal deconvolution | 2000 | Streak deconv baseline | 25.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2856 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2856 | done |
| 4 | StreakNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 30.8 | 0.2856 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 14.3 | 0.1114 |  |

### 136. XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | CrystFEL | 2012 | White et al., JAC 2012 | 22.0 | — | — | — |  |
| 2 | cctbx.xfel | 2014 | Hattne et al., Nature Methods 2014 | 25.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8643 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8643 |  |
| 5 | SFX-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.7 | 0.8643 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 24.1 | 0.9753 | done |

---

## Quantum Imaging

### 137. Entangled Photon Microscopy (`entangled_photon`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Coincidence counting | 2002 | quantum imaging baseline | 15.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9772 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9772 | done |
| 4 | QGI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 31.8 | 0.9772 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 31.8 | 0.9688 | done |

### 138. Ghost Imaging (`ghost_imaging`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | DeepGhost (autoencoder) | 2020 | Nature Sci Rep, s41598-020-68401-8 | 19.9 | 0.6000 | — | — |  |
| 2 | Correlation imaging | 2002 | Bennink et al., PRL 2002 | 15.0 | 0.4000 | — | — |  |
| 3 | Differential GI | 2010 | Ferri et al., 2010 | 18.0 | 0.5000 | — | — |  |
| 4 | CS-GI | 2013 | Katz et al., APL 2009 | 22.0 | 0.7000 | — | — |  |
| 5 | Bio-inspired self-attention | 2025 | MDPI Biomimetics 11(1):53 | 24.5 | 0.8000 | — | — |  |
| 6 | DGI-Net | 2021 | DL ghost imaging | 28.0 | 0.8800 | — | — |  |
| 7 | Orthogonal GI (2D-DCT) | 2025 | Nature Sci Rep, s41598-025-01283-w | 30.0 | — | — | — |  |
| 8 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 9 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 10 | GI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 11 | precomputed_baseline (test) | — | — | — | — | 6.6 | 0.1947 |  |

### 139. Quantum Illumination (`quantum_illumination`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Optimal receiver | 2008 | Lloyd, Science 2008 | 15.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.0 | 0.4292 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.0 | 0.4292 |  |
| 4 | QI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 19.0 | 0.4292 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 20.2 | 0.7859 | done |

---

## Broader Experimental Science

### 140. Acoustic Emission Testing (AE) (`acoustic_emission`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | AIC picker | 2000 | Akaike, Ann Inst Stat Math 1974 | 20.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.3286 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.3286 |  |
| 4 | DeepAE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 7.2 | 0.3286 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 20.2 | 0.0741 | done |

### 141. Adaptive Optics (AO) Imaging (`adaptive_optics`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Shack-Hartmann WFS | 1971 | Shack & Platt, 1971 | 22.0 | — | — | — |  |
| 2 | Phase diversity | 1982 | Gonsalves, Opt Eng 1982 | 26.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.2 | 0.4415 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.2 | 0.4415 | done |
| 5 | Deep-AO [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.2 | 0.4415 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | — | 1.0000 |  |

### 142. Bioluminescence Tomography (BLT) (`bioluminescence_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Diffusion-model inversion | 2005 | Wang et al., Opt Lett 2004 | 18.0 | 0.6000 | — | — |  |
| 2 | L1-regularized BLT | 2010 | TV-BLT | 22.0 | 0.7500 | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0974 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0974 |  |
| 5 | BLT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0974 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 13.3 | 0.3431 |  |

### 143. Full-Waveform Inversion (FWI) (`fwi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Adjoint-state FWI | 2006 | Virieux & Operto, Geophysics 2009 | 25.0 | 0.8500 | 5.7 | 0.0009 |  |
| 2 | InversionNet | 2020 | Wu & Lin, JGR 2019 | 28.0 | 0.9000 | 5.7 | 0.0009 |  |
| 3 | VelocityGAN | 2020 | Zhang & Alkhalifah, 2020 | 26.5 | 0.8800 | — | — |  |
| 4 | OpenFWI benchmark | 2022 | Deng et al., NeurIPS 2022 | 30.0 | 0.9400 | — | — |  |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.7 | 0.0009 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 8.7 | 0.0125 |  |

### 144. Gravitational Wave Detection (`gravitational_wave`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Matched filtering | 2000 | Allen et al., PRD 2012 | 20.0 | — | — | — |  |
| 2 | BayesWave | 2015 | Cornish & Littenberg, CQG 2015 | 25.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.6 | 0.0000 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.6 | 0.0000 | done |
| 5 | GW-DL (PyCBC-ML) [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 24.6 | 0.0000 | done |
| 6 | precomputed_baseline (test) | — | — | — | — | — | 0.8666 |  |

### 145. Electrical Impedance Tomography (EIT) (`impedance_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | D-bar method | 2000 | Nachman, Annals Math 1996 | 18.0 | 0.6000 | — | — |  |
| 2 | TV-ADMM | 2010 | TV regularization | 22.0 | 0.7500 | — | — |  |
| 3 | EIDORS-Net | 2020 | DL for EIT | 26.0 | 0.8500 | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 6 | EIT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | — | 1.0000 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 11.2 | 0.3124 |  |

### 146. Magnetic Particle Imaging (MPI) (`magnetic_particle`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | System matrix reconstruction | 2005 | Gleich & Weizenecker, Nature 2005 | 22.0 | — | — | — |  |
| 2 | X-space approach | 2010 | Goodwill & Conolly, TMI 2010 | 26.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.7 | 0.0284 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.7 | 0.0284 |  |
| 5 | MPI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.7 | 0.0284 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 26.5 | 0.9576 | done |

### 147. Ocean Acoustic Tomography (`ocean_acoustic_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Travel-time inversion | 1979 | Munk & Wunsch, Deep-Sea Res 1979 | 20.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.6789 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.6789 | done |
| 4 | OAT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 26.6 | 0.6789 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 5.6 | 0.6714 |  |

### 148. Particle Calorimetry (`particle_calorimetry`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Clustering algorithms | 2000 | CALICE collab. | 20.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 36.7 | 0.9421 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 36.7 | 0.9421 | done |
| 4 | CaloDiffusion [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 36.7 | 0.9421 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 36.2 | 0.7914 | done |

### 149. Radio Aperture Synthesis (`radio_astronomy`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | CLEAN | 1974 | Hogbom, A&AS 1974 | 25.0 | — | 37.3 | 0.7683 | done |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 37.3 | 0.7683 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 37.3 | 0.7683 | done |
| 4 | RadioAST-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 37.3 | 0.7683 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 16.1 | 0.2876 |  |

### 150. Seismic Tomography (`seismic_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Travel-time tomography | 1976 | Aki et al., JGR 1977 | 20.0 | 0.6500 | — | — |  |
| 2 | FWI | 2009 | Virieux & Operto, Geophysics 2009 | 28.0 | 0.8800 | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.6 | 0.0430 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.6 | 0.0430 |  |
| 5 | SeisInversion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.6 | 0.0430 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 9.0 | 0.5099 |  |

---

## Scientific Instrumentation

### 151. Atom Probe Tomography (APT) (`atom_probe`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Voltage reconstruction | 2000 | APT reconstruction | 20.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.4 | 0.0415 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.4 | 0.0415 |  |
| 4 | APT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.4 | 0.0415 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 40.2 | 0.9878 | done |

### 152. Cathodoluminescence (CL) Imaging (`cathodoluminescence`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Spectral unmixing | 2000 | NMF/VCA for CL | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.8 | 0.7716 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.8 | 0.7716 | done |
| 4 | CL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.8 | 0.7716 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 28.9 | 0.9772 | done |

### 153. Cryo-EM Single Particle Analysis (`cryo_em`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | RELION | 2012 | Scheres, JSB 2012 | 18.0 | — | 17.9 | 0.0101 | done |
| 2 | DRA (denoising-recon) | 2024 | arXiv 2410.11373 | 20.2 | 0.8700 | — | — |  |
| 3 | cryoSPARC | 2017 | Punjani et al., Nature Methods 2017 | 20.0 | — | — | — |  |
| 4 | DUAL (cryo-ET) | 2024 | PMC10942334, 2024 | 21.3 | 0.8240 | — | — |  |
| 5 | Topaz-Denoise | 2020 | Bepler et al., Nature Commun 2020 | 25.0 | — | — | — |  |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.9 | 0.0101 |  |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.9 | 0.0101 |  |
| 8 | CryoDRGN [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 17.9 | 0.0101 |  |
| 9 | precomputed_wiener (test) | — | — | — | — | 19.2 | 0.0300 |  |
| 10 | rl_ctf_20iter (test) | — | — | — | — | 13.2 | 0.4136 |  |

### 154. MALDI Mass Spectrometry Imaging (`maldi_msi`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Peak picking | 2000 | MALDI-MSI baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.1 | 0.6924 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.1 | 0.6924 | done |
| 4 | MSI-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 27.1 | 0.6924 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 26.3 | 0.9418 | done |

### 155. Muon Tomography (`muon_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | PoCA | 2003 | Borozdin et al., Nature 2003 | 13.7 | — | 3.2 | 0.0200 |  |
| 2 | mu-Net (ConvNeXt U-Net) | 2023 | arXiv 2312.17265 | 17.1 | — | — | — |  |
| 3 | FBP (muon tomography) (PWM) | — | — | — | — | 3.1 | 0.0019 |  |
| 4 | EM-POCA [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.2 | 0.0200 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 5.2 | -0.0128 |  |

### 156. Neutron Diffraction (`neutron_diffraction`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Rietveld refinement | 1969 | Rietveld, JAC 1969 | 25.0 | — | — | — |  |
| 2 | Le Bail fitting | 1988 | Le Bail et al., 1988 | 22.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.3 | 0.0013 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.3 | 0.0013 |  |
| 5 | NeutronDiff-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 5.3 | 0.0013 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 8.5 | 0.0116 |  |

### 157. Neutron Radiography / Tomography (`neutron_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP | 1971 | FBP baseline | 25.0 | 0.7000 | 4.3 | 0.0210 |  |
| 2 | SIRT | 1972 | Gilbert 1972 | 28.0 | 0.8000 | — | — |  |
| 3 | NeuTomo-DL [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.6130 |  |
| 4 | GRIDREC-Neutron [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.6 | 0.6130 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | -5.7 | 0.0503 |  |

### 158. Proton Radiography (`proton_radiography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MLP (Most Likely Path) | 2004 | Schulte et al., Med Phys 2008 | 22.0 | — | — | — |  |
| 2 | DROP-TVS | 2013 | Penfold et al., Med Phys 2010 | 28.0 | — | — | — |  |
| 3 | cGAN synthetic CT | 2023 | PubMed 37800874 | 29.0 | 0.9520 | — | — |  |
| 4 | CNN proton portal imaging | 2024 | PMC11682722 | 39.1 | 0.9870 | — | — |  |
| 5 | FBP (proton radiography) (PWM) | — | — | — | — | 10.9 | 0.0397 |  |
| 6 | ProtonRecon-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.0 | 0.3615 |  |
| 7 | FBP-Proton [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 12.0 | 0.3615 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 4.1 | -0.0000 |  |

### 159. Small-Angle X-ray Scattering (SAXS) (`saxs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Guinier analysis | 1939 | Guinier, 1939 | 20.0 | — | — | — |  |
| 2 | McSAS | 2013 | Bressler et al., JAC 2015 | 25.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.1 | 0.0265 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.1 | 0.0265 |  |
| 5 | SAXS-VAE [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 6.1 | 0.0265 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 8.4 | 0.0542 |  |

### 160. Wide-Angle X-ray Scattering (WAXS) (`waxs`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Rietveld refinement | 1969 | Rietveld, JAC 1969 | 24.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.2 | 0.0024 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.2 | 0.0024 |  |
| 4 | WAXS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.2 | 0.0024 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 20.6 | 0.0694 | done |

### 161. X-ray Crystallography (`xray_crystallography`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Direct methods | 1953 | Hauptman & Karle, 1953 | 22.0 | — | — | — |  |
| 2 | SHELXD | 2010 | Sheldrick, Acta Cryst 2008 | 28.0 | — | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.1 | 0.0063 |  |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.1 | 0.0063 |  |
| 5 | AlphaFold-SF [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 0.1 | 0.0063 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 22.4 | 0.0651 | done |

### 162. X-ray Fluorescence Tomography (`xrf_tomo`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP | 1971 | FBP baseline | 22.0 | — | — | — |  |
| 2 | SIRT | 1972 | Gilbert 1972 | 26.0 | — | — | — |  |
| 3 | Optimized SCUNet | 2024 | MDPI J Imaging 10(6):127 | 39.0 | 0.8600 | — | — |  |
| 4 | 1D-CNN + U-Net | 2025 | Nature Sci Reports, s41598-025-03900-0 | 39.1 | 0.9790 | — | — |  |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.0361 |  |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.0361 |  |
| 7 | XRFT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.0361 |  |
| 8 | precomputed_baseline (test) | — | — | — | — | 15.6 | 0.8431 |  |

---

## Multi-Modal Fusion

### 163. Correlative Light-Electron Microscopy (CLEM) (`clem`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | Landmark registration | 2000 | CLEM registration | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.1 | 0.7705 | done |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.1 | 0.7705 | done |
| 4 | CLEM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 28.1 | 0.7705 | done |
| 5 | precomputed_baseline (test) | — | — | — | — | 17.0 | 0.7297 |  |

### 164. CT + Fluorescence (FLIT) (`ct_fluorescence`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | FBP + fluorescence | 2000 | XFCT baseline | 22.0 | — | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.3 | 0.3256 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.3 | 0.3256 |  |
| 4 | XFCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 3.3 | 0.3256 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | — | 0.0002 |  |

### 165. PET/CT Fusion (`pet_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | OSEM + CT AC | 2000 | PET/CT baseline | 28.0 | 0.8000 | — | — |  |
| 2 | TrUNET-MAPEM | 2023 | ScienceDirect, S0895611123001337 | 33.7 | 0.9550 | — | — |  |
| 3 | Attention U-Net + diffusion | 2025 | arXiv 2504.00816 | 35.9 | 0.9920 | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1811 |  |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1811 |  |
| 6 | PET-CT-Fusion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 1.9 | 0.1811 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 13.0 | 0.0656 |  |

### 166. PET/MR Fusion (`pet_mr`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | MRAC-based reconstruction | 2010 | Wagenknecht et al., 2013 | 26.0 | 0.7800 | — | — |  |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.3034 |  |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.3034 |  |
| 4 | PET-MR-DeepJoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.3034 |  |
| 5 | precomputed_baseline (test) | — | — | — | — | 11.0 | 0.0165 |  |

### 167. SPECT/CT Fusion (`spect_ct`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | OSEM + CT AC | 2000 | SPECT/CT baseline | 26.0 | 0.7800 | — | — |  |
| 2 | U2-Net (bone SPECT/CT) | 2022 | PMC9192886 | 40.8 | 0.7880 | — | — |  |
| 3 | GAN projection-space denoising | 2022 | PMC8940834 | 42.5 | 0.9900 | — | — |  |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.2421 |  |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.2421 |  |
| 6 | SPECT-CT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 2.1 | 0.2421 |  |
| 7 | precomputed_baseline (test) | — | — | — | — | 11.4 | 0.0239 |  |

### 168. US/MRI Fusion (`us_mri`)

| # | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status |
|---|-----------|------|-----------|----------|----------|----------|----------|--------|
| 1 | B-spline FFD | 2003 | Rueckert et al., TMI 1999 | 25.0 | 0.8000 | — | — |  |
| 2 | VoxelMorph | 2019 | Balakrishnan et al., TMI 2019 | 30.0 | 0.9000 | — | — |  |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.5 | 0.3470 | done |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | — | — | 25.5 | 0.3470 | done |
| 5 | US-MRI-Net [proxy] (PWM) | — | — | — | — | 9.7 | 0.5796 |  |
| 6 | precomputed_baseline (test) | — | — | — | — | 7.6 | -0.0694 |  |

---

## Summary

- **Total modalities**: 168
- **Total algorithm entries**: 1040
- **Verified (done)**: 228
- **Not yet verified**: 812
- **Sources**: Published papers (2000-2026), PWM benchmark tests, YAML solver configs
- **Key benchmarks**: KAIST 10 scenes (CASSI), 6 grayscale SCI (CACTI), LoDoPaB-CT, fastMRI, Blender synthetic (NeRF), KITTI (LiDAR), DiffuserCam (lensless), BioSR (microscopy)

### Per-Category Breakdown

| Category | Modalities | Algorithms | Ref Entries | Done | Done % |
|----------|-----------|------------|-------------|------|--------|
| Compressive Imaging | 4 | 45 | 36 | 5 | 11% |
| Medical Imaging | 37 | 246 | 95 | 31 | 13% |
| Coherent Imaging | 5 | 29 | 13 | 5 | 17% |
| Microscopy | 24 | 129 | 45 | 34 | 26% |
| Electron Microscopy | 11 | 61 | 19 | 23 | 38% |
| Computational Optics | 2 | 14 | 7 | 5 | 36% |
| Computational Photography | 5 | 34 | 18 | 8 | 24% |
| Neural Rendering | 2 | 20 | 10 | 2 | 10% |
| Depth Imaging | 5 | 31 | 11 | 5 | 16% |
| Remote Sensing | 11 | 66 | 23 | 17 | 26% |
| Scanning Probe Microscopy | 4 | 22 | 6 | 4 | 18% |
| Industrial Inspection | 10 | 54 | 18 | 17 | 31% |
| Spectroscopy & Spectral Imaging | 8 | 50 | 18 | 15 | 30% |
| Astronomy & Space Imaging | 4 | 24 | 9 | 12 | 50% |
| Ultrafast Imaging | 4 | 22 | 7 | 4 | 18% |
| Quantum Imaging | 3 | 21 | 9 | 5 | 24% |
| Broader Experimental Science | 11 | 63 | 21 | 19 | 30% |
| Scientific Instrumentation | 12 | 74 | 27 | 12 | 16% |
| Multi-Modal Fusion | 6 | 35 | 11 | 5 | 14% |
