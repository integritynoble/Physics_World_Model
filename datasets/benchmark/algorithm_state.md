# Algorithm State — PWM5 Benchmark

Comprehensive listing of reconstruction algorithms for all 168 modalities.
Generated: 2026-03-15 | **65 done** (65 within 3 dB of reference) | 112 partial | 280 gap | 39 fail | 0 ran (no ref) | 798 not implemented | 1294 total

## Legend
- **Ref PSNR/SSIM**: Published reference values from literature
- **PWM PSNR/SSIM**: Values from running that specific algorithm in PWM framework
- **Std PSNR**: PSNR on standard dataset (per-solver)
- **Std**: `pass` = Std PSNR >= 15 dB | `low` = 5-15 dB | `fail` = < 5 dB | `—` = not implemented
- **Rank**: Algorithms sorted by Ref PSNR descending (best first)
- **Status**: `done` = PWM within 3 dB of reference | `partial` = runs, 3-10 dB gap | `gap` = runs, >10 dB gap | `fail` = diverged | `ran` = no ref to compare | `—` = not implemented

---

## Compressive Imaging

### 1. Coded Aperture Compressive Temporal Imaging (CACTI) (`cacti`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | HiSViT-13 | 2024 | Chen et al., ECCV 2024 | 37.3 | — | 12.9 | 0.0590 | gap | 12.9 | low |
| 2 | CTM-SCI | 2024 | CTM-SCI, 2024 | 36.5 | — | — | — | — | — | — |
| 3 | DUN-3DUnet | 2022 | Wu et al., CVPR 2022 | 35.3 | 0.9620 | — | — | — | — | — |
| 4 | HiSViT | 2023 | Chen et al., ICCV 2023 | 34.5 | — | 12.9 | 0.0590 | gap | 12.9 | low |
| 5 | EfficientSCI | 2023 | Wang et al., CVPR 2023 | 34.3 | 0.9610 | 3.3 | 0.1733 | gap | 3.3 | fail |
| 6 | STFormer | 2022 | Wang et al., NeurIPS 2022 | 33.9 | 0.9600 | — | — | — | — | — |
| 7 | ELP-Unfolding | 2022 | Yang et al., ECCV 2022 | 33.1 | 0.9530 | 3.3 | 0.1733 | gap | 3.3 | fail |
| 8 | BIRNAT | 2022 | Cheng et al., ECCV 2022 | 32.7 | 0.9510 | — | — | — | — | — |
| 9 | RevSCI-Net | 2021 | Cheng et al., NeurIPS 2021 | 31.4 | 0.9350 | — | — | — | — | — |
| 10 | MetaSCI | 2021 | Wang et al., CVPR 2021 | 30.1 | 0.9150 | — | — | — | — | — |
| 11 | PnP-FFDNet | 2020 | Yuan et al., CVPR 2020 | 28.7 | 0.9050 | 12.9 | 0.0590 | gap | 12.9 | low |
| 12 | DeSCI | 2019 | Liu et al., TPAMI 2019 | 27.1 | 0.8700 | — | — | — | — | — |
| 13 | GAP-TV | 2016 | Yuan, ICIP 2016 | 26.7 | 0.8460 | 3.3 | 0.1733 | gap | 3.3 | fail |
| 14 | GAP-TV (Traffic scene) | 2016 | Yuan, ICIP 2016 / Wu et al. 2022 | 20.9 | 0.7150 | 3.3 | 0.1733 | gap | 3.3 | fail |
| 15 | EfficientSCI-T (PWM) | — | — | 19.8 | — | 3.3 | 0.1733 | gap | 3.3 | fail |
| 16 | mask_division_baseline (test) | — | — | 19.8 | — | — | — | — | — | — |
| 17 | gap_tv (test) | — | — | 19.8 | — | 3.3 | 0.1733 | gap | 3.3 | fail |

### 2. Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | MiJUN | 2025 | MiJUN, AAAI 2025 | 40.9 | 0.9760 | — | — | — | — | — |
| 2 | RDLUF-MixS2 | 2022 | Cai et al., ECCV 2022 | 39.6 | 0.9720 | — | — | — | — | — |
| 3 | PADUT-L | 2023 | Li et al., CVPR 2023 | 38.9 | 0.9700 | — | — | — | — | — |
| 4 | DAUHST-9stg | 2022 | Cai et al., NeurIPS 2022 | 38.4 | 0.9670 | — | — | — | — | — |
| 5 | CST-L-Plus | 2022 | Cai et al., ECCV 2022 | 36.1 | 0.9570 | — | — | — | — | — |
| 6 | MST++ | 2022 | Cai et al., CVPRW 2022 | 36.0 | 0.9510 | — | — | — | — | — |
| 7 | HDNet | 2022 | Hu et al., CVPR 2022 | 35.0 | 0.9430 | 8.3 | 0.0126 | gap | 8.3 | low |
| 8 | MST-L | 2022 | Cai et al., CVPR 2022 | 34.9 | 0.9440 | 8.3 | 0.0126 | gap | 8.3 | low |
| 9 | PADUT | 2023 | Li et al., CVPR 2023 | 34.8 | — | — | — | — | — | — |
| 10 | SSR-L | 2023 | Zhang et al., ICCV 2023 | 34.0 | — | — | — | — | — | — |
| 11 | DGSMP | 2021 | Huang et al., CVPR 2021 | 32.6 | 0.9170 | — | — | — | — | — |
| 12 | TSA-Net | 2020 | Meng et al., ECCV 2020 | 31.5 | 0.8940 | — | — | — | — | — |
| 13 | λ-Net | 2020 | Miao et al., ICCV 2019 | 30.1 | 0.8770 | — | — | — | — | — |
| 14 | ADMM-Net | 2019 | Ma et al., ICCV 2019 | 29.1 | 0.8600 | — | — | — | — | — |
| 15 | GAP-TV (guided) (PWM) | — | Yuan et al. 2016 | 26.2 | — | 20.9 | 0.9559 | partial | 20.9 | pass |
| 16 | GAP-TV (fast) (PWM) | — | — | 26.2 | — | 21.7 | 0.9633 | partial | 21.7 | pass |
| 17 | GAP-TV (small) (PWM) | — | — | 26.2 | — | 21.7 | 0.9633 | partial | 21.7 | pass |
| 18 | GAP-TV | 2016 | Yuan, GAP-TV, ICIP 2016 | 24.4 | 0.6690 | 21.7 | 0.9633 | done | 21.7 | pass |
| 19 | TwIST | 2007 | Bioucas-Dias & Figueiredo, TwIST, TIP 2007 | 23.1 | 0.6690 | — | — | — | — | — |

### 3. Generic Matrix Sensing (`matrix`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | LISTA | 2010 | Gregor & LeCun, ICML 2010 | 28.5 | — | 4.0 | 0.1284 | gap | 4.0 | fail |
| 2 | FISTA | 2009 | Beck & Teboulle, SIAM 2009 | 27.0 | — | 5.2 | 0.1712 | gap | 5.2 | low |
| 3 | OMP | 1993 | Pati et al., 1993 | 24.0 | — | — | — | — | — | — |
| 4 | FISTA-L1 (high quality) (PWM) | — | Beck & Teboulle 2009 | 22.1 | — | 5.2 | 0.1713 | gap | 5.2 | low |
| 5 | precomputed_baseline (test) | — | — | 22.1 | — | — | — | — | — | — |

### 4. Single-Pixel Camera (SPC) (`spc`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | AMP-Net | 2021 | Zhang et al., TIP 2021 | 34.6 | 0.9550 | — | — | — | — | — |
| 2 | ISTA-Net+ | 2018 | Zhang & Ghanem, CVPR 2018 | 32.3 | 0.9350 | 6.4 | 0.0295 | gap | 6.4 | low |
| 3 | TransCS | 2022 | Shen et al., TIP 2022 | 31.1 | — | — | — | — | — | — |
| 4 | CSNet+ | 2019 | Shi et al., TIP 2019 | 29.8 | 0.8820 | — | — | — | — | — |
| 5 | TVAL3 | 2009 | Li et al., TVAL3, Rice 2009 | 24.6 | 0.7500 | 6.0 | 0.0000 | gap | 6.0 | low |
| 6 | Random sampling baseline | 2009 | Baraniuk, IEEE SPM 2007 | 15.0 | 0.4000 | — | — | — | — | — |
| 7 | Pseudoinverse (no regularization) | 2009 | CS pseudoinverse baseline | 8.0 | 0.2000 | — | — | — | — | — |
| 8 | ADMM-L1 (PWM) | — | Boyd et al. 2010 | 6.8 | — | 6.4 | 0.0295 | done | 6.4 | low |

## Medical Imaging

### 5. X-ray Angiography (`angiography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Maskless 2D-DSA (U-Net) | 2022 | Gao et al., JVIR 2022, PubMed 35311665 | 43.0 | 0.9800 | — | — | — | — | — |
| 2 | DSA subtraction (with motion) | 1980 | Ueda et al., Radiology 2021 (motion-free=40.2 dB) | 30.0 | 0.5000 | — | — | — | — | — |
| 3 | DSA (Digital Subtraction) | 1980 | DSA, Mistretta et al., 1981 | 25.0 | 0.8000 | — | — | — | — | — |
| 4 | Deep Decoupling Net (GAN+RDB) | 2024 | IIETA, TS 2024 | 23.7 | 0.8770 | — | — | — | — | — |
| 5 | DSA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.0377 | 41.8 | 0.9998 | gap | 41.8 | pass |
| 6 | VesselSegNet [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.0377 | 41.8 | 0.9998 | gap | 41.8 | pass |
| 7 | precomputed_baseline (test) | — | — | 12.9 | — | — | — | — | — | — |

### 6. Arterial Spin Labeling (ASL) MRI (`asl_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | HUST (Transformer) 3D | 2025 | Springer, Vis Comput 2025 | 45.1 | 0.9900 | — | — | — | — | — |
| 2 | HUST (Transformer) 2D | 2025 | Springer, Vis Comput 2025 | 33.7 | 0.9600 | — | — | — | — | — |
| 3 | ASLRDB (Dilated+RDB) | 2025 | Springer, SIVP 2025 | 25.0 | 0.8240 | — | — | — | — | — |
| 4 | Control-label subtraction | 1998 | Detre et al., MRM 1992 | 22.0 | 0.6500 | — | — | — | — | — |
| 5 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1371 | -35.9 | 0.0000 | fail | -35.9 | fail |
| 6 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1371 | -37.1 | 0.0000 | fail | -37.1 | fail |
| 7 | ASL-Net [proxy] (PWM) | — | — | 12.9 | 0.1371 | 12.9 | -0.1480 | done | 12.9 | low |
| 8 | precomputed_baseline (test) | — | — | 10.9 | — | — | — | — | — | — |

### 7. Brachytherapy Imaging (`brachytherapy_img`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | RL-ARCNN (metal artifact reduction) | 2018 | Huang et al., BioMedical Eng OnLine 2018 | 38.1 | — | — | — | — | — | — |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 33.1 | 0.8307 | 43.7 | 0.9997 | gap | 43.7 | pass |
| 3 | BrachyNet [proxy] (PWM) | — | Richardson 1972, JOSA | 33.1 | 0.8307 | 43.7 | 0.9997 | gap | 43.7 | pass |
| 4 | Monte Carlo dose | 2005 | MC dose calculation | 28.0 | 0.8500 | — | — | — | — | — |
| 5 | precomputed_baseline (test) | — | — | 25.2 | — | — | — | — | — | — |
| 6 | FBP | 1971 | FBP baseline | 25.0 | — | 43.7 | 0.9997 | gap | 43.7 | pass |

### 8. Cone-Beam Computed Tomography (CBCT) (`cbct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | FBPConvNet | 2017 | Jin et al., TIP 2017 | 36.5 | 0.9500 | — | — | — | — | — |
| 2 | FACT | 2022 | FACT, 2022 | 33.8 | 0.9300 | — | — | — | — | — |
| 3 | SART | 1984 | Andersen & Kak, 1984 | 32.0 | 0.8800 | — | — | — | — | — |
| 4 | FDK | 1984 | Feldkamp et al., JOSA 1984 | 28.0 | 0.8000 | 15.1 | 0.9188 | gap | 15.1 | pass |
| 5 | FDK (8 views) | 1984 | Zha et al., MICCAI 2024 | 16.6 | — | — | — | — | — | — |
| 6 | FDK (6 views) | 1984 | Zha et al., MICCAI 2024, arXiv 2407.01090 | 15.3 | — | — | — | — | — | — |
| 7 | FDK-DL (PWM) | — | Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI | 15.2 | — | 4.5 | 0.0000 | gap | 4.5 | fail |
| 8 | CBCT-UNet (PWM) | — | Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP | 15.2 | — | 4.5 | 0.0000 | gap | 4.5 | fail |
| 9 | fbp_ramlak (test) | — | — | 15.2 | — | — | — | — | — | — |
| 10 | fbp_shepp_logan (test) | — | — | 15.2 | — | — | — | — | — | — |

### 9. CEST MRI (`cest_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | -31.9 | 0.0000 | fail | -31.9 | fail |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | -32.6 | 0.0000 | fail | -32.6 | fail |
| 3 | CEST-Net [proxy] (PWM) | — | — | 44.3 | 0.9999 | 17.8 | -0.0531 | gap | 17.8 | pass |
| 4 | ResUNet-NE | 2023 | Muller et al., Diagnostics 13(21):3326, 2023 | 35.0 | — | — | — | — | — | — |
| 5 | precomputed_baseline (test) | — | — | 32.1 | — | — | — | — | — | — |
| 6 | Z-spectrum fitting | 2003 | Zhou et al., NMR Biomed 2003 | 25.0 | 0.7500 | — | — | — | — | — |

### 10. Contrast-Enhanced Ultrasound (CEUS) (`ceus`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Real-time CNN | 2022 | Choi et al., MBEC 2022 | 36.1 | 0.9640 | — | — | — | — | — |
| 2 | GAN-RW (Residual Dense) | 2022 | Lan et al., PeerJ Computer Science 2022 | 33.9 | 0.8720 | — | — | — | — | — |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 26.4 | 0.9801 | 12.9 | 0.1047 | gap | 12.9 | low |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 26.4 | 0.9801 | 12.4 | 0.0898 | gap | 12.4 | low |
| 5 | US-DeepSight [proxy] (PWM) | — | Richardson 1972, JOSA | 26.4 | 0.9801 | 12.4 | 0.0898 | gap | 12.4 | low |
| 6 | Singular value decomposition | 2015 | Demene et al., TMI 2015 | 25.0 | 0.7500 | — | — | — | — | — |
| 7 | precomputed_baseline (test) | — | — | 24.5 | — | — | — | — | — | — |
| 8 | Temporal averaging | 2000 | CEUS temporal baseline | 22.0 | 0.7000 | — | — | — | — | — |

### 11. Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 41.5 | 0.9999 | 44.0 | 0.9988 | done | 44.0 | pass |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 41.5 | 0.9999 | 44.0 | 0.9988 | done | 44.0 | pass |
| 3 | CLE-Net (CARE) [proxy] (PWM) | — | Richardson 1972, JOSA | 41.5 | 0.9999 | 44.0 | 0.9988 | done | 44.0 | pass |
| 4 | Self-supervised denoising | 2024 | Sensors 2024 | 36.1 | 0.8980 | — | — | — | — | — |
| 5 | precomputed_baseline (test) | — | — | 34.0 | — | — | — | — | — | — |
| 6 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | — | — | — | — | — | — |

### 12. X-ray Computed Tomography (CT) (`ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | LEARN | 2019 | Chen et al., TMI 2018 | 43.1 | — | — | — | — | — | — |
| 2 | Score-CT | 2022 | Song et al., ICLR 2022 | 43.0 | — | — | — | — | — | — |
| 3 | DuDoTrans | 2022 | Wang et al., MICCAI 2022 | 42.1 | — | — | — | — | — | — |
| 4 | FBPConvNet | 2017 | Jin et al., TIP 2017 | 38.5 | 0.9590 | 15.1 | 0.9096 | gap | 15.1 | pass |
| 5 | iRadonMAP | 2019 | He et al., 2019 | 36.9 | 0.9420 | — | — | — | — | — |
| 6 | Learned Primal-Dual | 2018 | Adler & Oktem, TMI 2018 | 36.2 | 0.9590 | — | — | — | — | — |
| 7 | DOLCE | 2023 | Liu et al., 2023 | 36.0 | — | — | — | — | — | — |
| 8 | TV regularization | 2006 | Sidky et al., PMB 2006 | 33.4 | 0.9000 | — | — | — | — | — |
| 9 | RED-CNN | 2017 | Chen et al., TMI 2017 | 33.2 | 0.9150 | -33.3 | 0.0001 | fail | -33.3 | fail |
| 10 | FBP (Ram-Lak) | 1971 | Ramachandran & Lakshminarayanan 1971 | 30.2 | 0.8200 | 15.1 | 0.9096 | gap | 15.1 | pass |
| 11 | FBP (10 angles) | 2021 | Leuschner et al., J Imaging 2021, PMC8321320 | 17.1 | — | 15.1 | 0.9096 | done | 15.1 | pass |
| 12 | FBP (5 angles) | 2021 | Leuschner et al., J Imaging 2021, PMC8321320 | 15.5 | — | 15.1 | 0.9096 | done | 15.1 | pass |
| 13 | PnP-HQS + NLM (PWM) | — | — | 13.8 | — | -45.8 | 0.0000 | fail | -45.8 | fail |
| 14 | fbp_ramlak (test) | — | — | 13.8 | — | 15.1 | 0.9096 | done | 15.1 | pass |
| 15 | fbp_shepp_logan (test) | — | — | 13.8 | — | 15.1 | 0.9096 | done | 15.1 | pass |
| 16 | sart_10iter (test) | — | — | 13.8 | — | — | — | — | — | — |
| 17 | FBP (2 angles, scattering) | 2021 | Leuschner et al., J Imaging 2021, PMC8321320 | 13.1 | — | 15.1 | 0.9096 | done | 15.1 | pass |

### 13. Dual-Energy X-ray Absorptiometry (DEXA) (`dexa`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DL bone density estimation | 2022 | DL for DEXA | 32.0 | 0.9000 | — | — | — | — | — |
| 2 | Dual-energy decomposition | 1987 | Alvarez & Macovski, PMB 1976 | 28.0 | 0.8500 | — | — | — | — | — |
| 3 | Bone decomposition baseline | 2020 | DEXA energy subtraction baseline (estimated) | 19.7 | — | — | — | — | — | — |
| 4 | DXA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.7 | 0.4561 | 34.4 | 0.9971 | gap | 34.4 | pass |
| 5 | DEXA-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 11.7 | 0.4561 | 34.4 | 0.9971 | gap | 34.4 | pass |
| 6 | FISTA-L2 (dual-energy) (PWM) | — | — | 10.7 | — | 33.6 | 0.9964 | gap | 33.6 | pass |
| 7 | precomputed_baseline (test) | — | — | 10.7 | — | — | — | — | — | — |

### 14. Diffusion MRI (DTI) (`diffusion_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | q-DL | 2016 | Golkov et al., MRM 2016 | 34.0 | — | 24.7 | 0.1243 | partial | 24.7 | pass |
| 2 | MPR-ViT (ADC maps) | 2024 | Eidex et al., Med Phys 2024 | 31.0 | 0.9500 | — | — | — | — | — |
| 3 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6000 | — | — | — | — | — |
| 4 | Zero-filled (high b-value) | 2000 | dMRI zero-filled baseline | 15.0 | 0.4000 | — | — | — | — | — |
| 5 | SHORE-Net [proxy] (PWM) | — | — | 13.0 | 0.0360 | 24.7 | 0.1243 | gap | 24.7 | pass |
| 6 | Zero-filled (R=4, multi-b) | 2023 | Zhong et al., Bioengineering 2023, PMC10376839 | 12.2 | — | — | — | — | — | — |
| 7 | Zero-filled (R=6, multi-b) | 2023 | Zhong et al., Bioengineering 2023, PMC10376839 | 12.0 | 0.3000 | — | — | — | — | — |
| 8 | SENSE (WLS tensor fit) (PWM) | — | — | 11.3 | — | 24.7 | 0.1243 | gap | 24.7 | pass |
| 9 | zero_filled (test) | — | — | 11.3 | — | — | — | — | — | — |

### 15. Digital Breast Tomosynthesis (DBT) (`digital_breast_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SART | 1984 | Andersen & Kak 1984 | 30.0 | — | — | — | — | — | — |
| 2 | TV-regularized MLEM | 2010 | TV-MLEM for DBT | 28.0 | 0.8700 | — | — | — | — | — |
| 3 | FBP | 1971 | FBP baseline | 25.0 | — | 16.0 | 0.0006 | partial | 16.0 | pass |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 10.5 | 0.4411 | 16.0 | 0.0006 | partial | 16.0 | pass |
| 5 | DBT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 10.5 | 0.4411 | 16.0 | 0.0006 | partial | 16.0 | pass |
| 6 | precomputed_baseline (test) | — | — | 8.8 | — | — | — | — | — | — |

### 16. Doppler Ultrasound (`doppler_ultrasound`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DL Doppler | 2020 | DL for Doppler dealiasing | 30.0 | 0.8800 | — | — | — | — | — |
| 2 | 3D-Res-UNet (95% compression) | 2022 | Blanchard et al., IEEE TUFFC 2022, PMC9247015 | 26.7 | — | — | — | — | — | — |
| 3 | Autocorrelation | 1985 | Kasai et al., 1985 | 22.0 | 0.7000 | — | — | — | — | — |
| 4 | Conventional SVD (90% compression) | 2022 | Blanchard et al., IEEE TUFFC 2022, PMC9247015 | 19.5 | — | — | — | — | — | — |
| 5 | UDoppler-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 18.6 | 0.0164 | 18.7 | 0.6170 | done | 18.7 | pass |
| 6 | Doppler CFAR [proxy] (PWM) | — | Richardson 1972, JOSA | 18.6 | 0.0164 | 18.7 | 0.6170 | done | 18.7 | pass |
| 7 | Wall filter (highpass) | 1985 | Wall filter baseline | 18.0 | 0.6000 | — | — | — | — | — |
| 8 | Back-Projection (Doppler) (PWM) | — | — | 17.6 | — | 22.6 | 0.7702 | partial | 22.6 | pass |
| 9 | autocorrelation_estimator (test) | — | — | 17.6 | — | — | — | — | — | — |
| 10 | clutter_filtered (test) | — | — | 17.6 | — | — | — | — | — | — |
| 11 | precomputed_baseline (test) | — | — | 17.6 | — | — | — | — | — | — |
| 12 | Conventional SVD (95% compression) | 2022 | Blanchard et al., IEEE TUFFC 2022, PMC9247015 | 17.4 | — | — | — | — | — | — |

### 17. Diffuse Optical Tomography (DOT) (`dot`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | BPNN | 2018 | Feng et al., JBO 24(5), PMC6992907 | 27.8 | 0.9100 | — | — | — | — | — |
| 2 | Tikhonov regularization | 2018 | Feng et al., JBO 24(5), PMC6992907 | 24.3 | 0.4600 | — | — | — | — | — |
| 3 | Tikhonov (basic, noisy) | 2000 | Yoo et al., J Biomed Opt 2019, PMC6992907 | 22.0 | 0.3000 | — | — | — | — | — |
| 4 | Born approximation | 1999 | Arridge, Inverse Problems 1999 | 20.0 | 0.6000 | 24.2 | 0.9626 | partial | 24.2 | pass |
| 5 | Rytov + Laplacian | 2000 | Arridge et al., PMB 1999 | 18.0 | 0.4500 | — | — | — | — | — |
| 6 | L-BFGS-TV [proxy] (PWM) | — | Richardson 1972, JOSA | 8.0 | 0.0293 | 22.5 | 0.9450 | gap | 22.5 | pass |
| 7 | DOT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 8.0 | 0.0293 | 22.5 | 0.9450 | gap | 22.5 | pass |
| 8 | born_backprojection (test) | — | — | 7.0 | — | — | — | — | — | — |
| 9 | tikhonov (test) | — | — | 7.0 | — | — | — | — | — | — |
| 10 | precomputed_baseline (test) | — | — | 7.0 | — | — | — | — | — | — |

### 18. Shear-Wave Elastography (`elastography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CNN-LSTM | 2024 | arXiv 2024 | 32.7 | 0.9960 | — | — | — | — | — |
| 2 | Direct inversion | 2001 | Manduca et al., MRM 2001 | 24.0 | 0.7500 | — | — | — | — | — |
| 3 | Phase gradient | 2000 | Manduca et al., MRM 2001 | 22.0 | 0.7000 | — | — | — | — | — |
| 4 | Raw displacement (no filtering) | 2000 | Elastography raw baseline | 14.0 | 0.4000 | — | — | — | — | — |
| 5 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 12.0 | 0.8049 | 11.5 | 0.0785 | done | 11.5 | low |
| 6 | NLSI-Solver [proxy] (PWM) | — | Richardson 1972, JOSA | 12.0 | 0.8049 | 11.5 | 0.0785 | done | 11.5 | low |
| 7 | SENSE (displacement field) (PWM) | — | — | 11.0 | — | 11.8 | 0.0002 | done | 11.8 | low |
| 8 | precomputed_baseline (test) | — | — | 11.0 | — | — | — | — | — | — |

### 19. Fiber Bundle Endoscopy (`endoscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR | 2024 | Heliyon 2024 | 36.8 | 0.9700 | — | — | — | — | — |
| 2 | U-Net denoising | 2019 | DL for CLE | 28.0 | 0.8500 | — | — | — | — | — |
| 3 | Richardson-Lucy | 1972 | Richardson 1972 | 24.0 | 0.7200 | — | — | — | — | — |
| 4 | Interpolation baseline | 2000 | Fiber bundle baseline | 22.0 | 0.6500 | — | — | — | — | — |
| 5 | Raw CLE (honeycomb artifact) | 2022 | Kim et al., Sensors 2022, PMC9824069 | 20.6 | 0.7300 | — | — | — | — | — |
| 6 | Gaussian filter (fiber bundle) | 2023 | Kim et al., Sensors 2023, PMC9824069 | 19.0 | — | — | — | — | — | — |
| 7 | Raw fiber bundle (no processing) | 2019 | Shao et al., Optics Express 2019, PMC6825616 | 14.6 | — | — | — | — | — | — |
| 8 | FISTA-L2 (endoscopy) (PWM) | — | — | 11.8 | — | 31.5 | 0.9808 | gap | 31.5 | pass |
| 9 | EndoMapper-Net (PWM) | — | Ozyoruk, K.B. et al. (2021) EndoMapper, Nat. Mach. Intel. 3 | 11.8 | — | 4.4 | 0.0154 | partial | 4.4 | fail |
| 10 | AF-SfMLearner (PWM) | — | Shao, S. et al. (2022) Self-supervised depth estimation in endoscopy, MICCAI 2022 | 11.8 | — | 4.4 | 0.0157 | partial | 4.4 | fail |
| 11 | rl_20iter (test) | — | — | 11.8 | — | — | — | — | — | — |
| 12 | rl_50iter (test) | — | — | 11.8 | — | — | — | — | — | — |
| 13 | precomputed_recon (test) | — | — | 11.8 | — | — | — | — | — | — |

### 20. Fluoroscopy (`fluoroscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | FluoroNet [proxy] (PWM) | — | Richardson 1972, JOSA | 54.9 | 0.9999 | 29.1 | 0.9823 | gap | 29.1 | pass |
| 2 | X-ray CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 54.9 | 0.9999 | 29.1 | 0.9823 | gap | 29.1 | pass |
| 3 | FBP (fluoroscopy) (PWM) | — | — | 44.5 | — | 2.6 | 0.0000 | gap | 2.6 | fail |
| 4 | precomputed_baseline (test) | — | — | 44.5 | — | — | — | — | — | — |
| 5 | MSR2AU-Net | 2024 | arXiv 2024 | 39.1 | 0.9800 | — | — | — | — | — |
| 6 | RED-CNN | 2017 | Chen et al., TMI 2017 | 33.0 | 0.9000 | — | — | — | — | — |
| 7 | Motion compensation | 2000 | fluoroscopy baseline | 28.0 | 0.8000 | — | — | — | — | — |

### 21. Functional MRI (BOLD fMRI) (`fmri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | E2E-VarNet | 2021 | Sriram et al., fastMRI Challenge 2020 | 41.4 | 0.9590 | — | — | — | — | — |
| 2 | CS-fMRI | 2010 | Jung et al., PMB 2009 | 32.0 | 0.8800 | — | — | — | — | — |
| 3 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6000 | — | — | — | — | — |
| 4 | fMRI-Transformer [proxy] (PWM) | — | — | 9.9 | 0.1054 | 10.4 | -0.2521 | done | 10.4 | low |
| 5 | DeepBold [proxy] (PWM) | — | — | 9.9 | 0.1054 | 10.4 | -0.2521 | done | 10.4 | low |
| 6 | SENSE (fMRI) (PWM) | — | — | 4.9 | — | 10.4 | -0.2521 | partial | 10.4 | low |
| 7 | zero_filled (test) | — | — | 4.9 | — | — | — | — | — | — |

### 22. Fundus Camera (`fundus`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | RETFound (PWM) | — | Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156 | 35.9 | — | 6.9 | 0.0034 | gap | 6.9 | low |
| 2 | DR-Grade-Net (PWM) | — | Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22) | 35.9 | — | 6.7 | 0.0003 | gap | 6.7 | low |
| 3 | rl_20iter (test) | — | — | 35.9 | — | — | — | — | — | — |
| 4 | rl_50iter (test) | — | — | 35.9 | — | — | — | — | — | — |
| 5 | precomputed_wiener (test) | — | — | 35.9 | — | — | — | — | — | — |
| 6 | Richardson-Lucy | 1972 | Richardson 1972 | 30.0 | 0.9000 | 59.0 | 1.0000 | gap | 59.0 | pass |
| 7 | PCE-Net | 2023 | PCE-Net, 2023 | 29.9 | — | — | — | — | — | — |
| 8 | GFE-Net | 2023 | Med Image Anal 2023 | 29.7 | 0.9550 | — | — | — | — | — |
| 9 | Cofe-Net | 2022 | Li et al., Cofe-Net, 2022 | 24.9 | — | — | — | — | — | — |

### 23. Intravascular Ultrasound (IVUS) (`ivus`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | IVUS-Net | 2020 | DL for IVUS | 30.0 | 0.8800 | 7.0 | 0.0463 | gap | 7.0 | low |
| 2 | U-Net segmentation | 2020 | DL for IVUS | 25.0 | 0.8000 | — | — | — | — | — |
| 3 | DAS beamforming | 1990 | DAS baseline | 22.0 | 0.7000 | — | — | — | — | — |
| 4 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 20.8 | 0.9002 | 7.2 | 0.0539 | gap | 7.2 | low |
| 5 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 20.8 | 0.9002 | 7.0 | 0.0463 | gap | 7.0 | low |
| 6 | precomputed_baseline (test) | — | — | 19.8 | — | — | — | — | — | — |

### 24. Mammography (`mammography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DeepTFormer | 2025 | Scientific Reports 2025 | 39.4 | 0.9400 | — | — | — | — | — |
| 2 | RED-CNN | 2017 | Chen et al., TMI 2017 | 35.0 | 0.9200 | — | — | — | — | — |
| 3 | BM3D | 2007 | Dabov et al., TIP 2007 | 32.0 | 0.9000 | — | — | — | — | — |
| 4 | FBP | 1971 | FBP baseline | 30.0 | 0.8500 | 13.7 | 0.0005 | gap | 13.7 | low |
| 5 | NLM denoising | 2005 | Buades et al., CVPR 2005 | 26.0 | 0.8500 | — | — | — | — | — |
| 6 | MammoNet (GatorTron) [proxy] (PWM) | — | Richardson 1972, JOSA | 21.9 | 0.8680 | -0.2 | -0.0441 | fail | -0.2 | fail |
| 7 | Mammo-ResNet [proxy] (PWM) | — | Richardson 1972, JOSA | 21.9 | 0.8680 | -0.2 | -0.0441 | fail | -0.2 | fail |
| 8 | precomputed_recon (test) | — | — | 20.9 | — | — | — | — | — | — |

### 25. MR Elastography (MRE) (`mr_elastography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SW-ViT (simulated) | 2025 | arXiv 2505.18865 | 32.7 | 0.9950 | — | — | — | — | — |
| 2 | Phase gradient | 2001 | Manduca et al., MRM 2001 | 24.0 | 0.7500 | — | — | — | — | — |
| 3 | Direct inversion | 2001 | Manduca et al., MRM 2001 | 22.0 | 0.7000 | — | — | — | — | — |
| 4 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1408 | 36.3 | 0.9526 | gap | 36.3 | pass |
| 5 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1408 | 36.6 | 0.9560 | gap | 36.6 | pass |
| 6 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1408 | 36.6 | 0.9560 | gap | 36.6 | pass |
| 7 | precomputed_baseline (test) | — | — | 11.0 | — | — | — | — | — | — |

### 26. MR Fingerprinting (MRF) (`mr_fingerprinting`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | MRF-Mixer (T2 map) | 2025 | MDPI Information 2025 | 35.9 | 0.9800 | — | — | — | — | — |
| 2 | MRF-Mixer (T1 map) | 2025 | MDPI Information 2025 | 33.5 | 0.9800 | — | — | — | — | — |
| 3 | GAST-Mamba (T1 map) | 2025 | arXiv 2507.03369 | 33.1 | 0.9670 | — | — | — | — | — |
| 4 | MANTIS | 2019 | Fang et al., MRM 2019 | 30.0 | 0.9000 | — | — | — | — | — |
| 5 | Dictionary matching | 2013 | Ma et al., Nature 2013 | 25.0 | 0.8000 | — | — | — | — | — |
| 6 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1551 | 32.9 | 0.9606 | gap | 32.9 | pass |
| 7 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1551 | 32.8 | 0.9607 | gap | 32.8 | pass |
| 8 | precomputed_baseline (test) | — | — | 11.0 | — | — | — | — | — | — |

### 27. MR Angiography (MRA) (`mra`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | 3D CNN SR | 2025 | Nature Scientific Reports 2025 | 36.8 | 0.9830 | — | — | — | — | — |
| 2 | CS-MRA | 2010 | Lustig et al., MRM 2007 | 30.0 | 0.8500 | — | — | — | — | — |
| 3 | Zero-filled (R=7-11) | 2024 | PMC11424428 (verified 25.80 dB) | 25.8 | — | — | — | — | — | — |
| 4 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6500 | — | — | — | — | — |
| 5 | Zero-filled (16x accel) | 2026 | Li et al., MRM 2026 (R=8: 26.8 dB, extrapolated) | 25.0 | 0.3500 | — | — | — | — | — |
| 6 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 18.1 | 0.4218 | -32.3 | 0.0000 | fail | -32.3 | fail |
| 7 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 18.1 | 0.4218 | -32.8 | 0.0000 | fail | -32.8 | fail |
| 8 | MRA-VesselNet [proxy] (PWM) | — | — | 18.1 | 0.4218 | 16.2 | -0.1642 | done | 16.2 | pass |
| 9 | precomputed_baseline (test) | — | — | 14.7 | — | — | — | — | — | — |

### 28. Magnetic Resonance Imaging (MRI) (`mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | PromptMR | 2023 | Li et al., MICCAI 2023 | 41.5 | — | — | — | — | — | — |
| 2 | E2E-VarNet | 2020 | Sriram et al., NeurIPS 2020 | 40.5 | 0.9720 | — | — | — | — | — |
| 3 | ReconFormer | 2023 | Guo et al., TMI 2023 | 40.1 | 0.9750 | — | — | — | — | — |
| 4 | PromptMR+ | 2024 | Li et al., TMI 2024 | 39.9 | 0.9730 | — | — | — | — | — |
| 5 | HUMUS-Net | 2022 | Fabian et al., NeurIPS 2022 | 37.3 | 0.9500 | — | — | — | — | — |
| 6 | U-Net | 2018 | Zbontar et al., fastMRI 2018 | 36.0 | 0.9470 | — | — | — | — | — |
| 7 | GRAPPA | 2002 | Griswold et al., MRM 2002 | 34.0 | 0.9200 | — | — | — | — | — |
| 8 | CS-MRI (SparseMRI) | 2007 | Lustig et al., MRM 2007 | 33.0 | 0.9000 | — | — | — | — | — |
| 9 | Zero-filled IFFT | 2000 | Baseline | 28.0 | 0.6400 | 13.7 | -0.1498 | gap | 13.7 | low |
| 10 | E2E-VarNet (16x) | 2024 | Neural Operators CS-MRI, arXiv 2410.16290 | 23.2 | — | — | — | — | — | — |
| 11 | Zero-filled (32x accel) | 2018 | Zbontar et al., fastMRI 2018 | 15.0 | 0.3000 | — | — | — | — | — |
| 12 | CS-MRI (Wavelet) (PWM) | — | Lustig et al. 2007, MRM | 13.4 | — | 13.7 | -0.1499 | done | 13.7 | low |
| 13 | MoDL (PWM) | — | Aggarwal et al. 2019, IEEE TMI | 13.4 | — | 13.7 | -0.1499 | done | 13.7 | low |
| 14 | MoDL (5 unrolls) (PWM) | — | — | 13.4 | — | 13.7 | -0.1486 | done | 13.7 | low |
| 15 | zero_filled (test) | — | — | 13.4 | — | 13.7 | -0.1498 | done | 13.7 | low |
| 16 | cs_mri_wavelet (test) | — | — | 13.4 | — | — | — | — | — | — |
| 17 | sense (test) | — | — | 13.4 | — | -27.9 | 0.0000 | fail | -27.9 | fail |

### 29. MR Spectroscopy (MRS) (`mrs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DDPM-MRSI (2x SR) | 2025 | J Imaging Inform Med 2025 | 29.7 | 0.9560 | — | — | — | — | — |
| 2 | LCModel | 1993 | Provencher, MRM 1993 | 28.0 | — | — | — | — | — | — |
| 3 | HLSVD | 2002 | Pijnappel et al., 1992 | 22.0 | — | 20.7 | 0.0514 | done | 20.7 | pass |
| 4 | MRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1516 | 20.7 | 0.0514 | partial | 20.7 | pass |
| 5 | SENSE (spectroscopy) (PWM) | — | — | 11.0 | — | 21.2 | 0.0059 | gap | 21.2 | pass |
| 6 | precomputed_baseline (test) | — | — | 11.0 | — | — | — | — | — | — |

### 30. Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CNN-LSTM Hybrid | 2024 | Multimedia Tools Appl 2024 | 32.1 | 0.9860 | — | — | — | — | — |
| 2 | OT-NIRS (tomographic) | 2010 | Boas et al., NeuroImage 2010 | 22.0 | 0.7000 | — | — | — | — | — |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 21.4 | 0.9587 | 26.7 | 0.9253 | partial | 26.7 | pass |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 21.4 | 0.9587 | 26.8 | 0.9278 | partial | 26.8 | pass |
| 5 | fNIRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 21.4 | 0.9587 | 26.7 | 0.9253 | partial | 26.7 | pass |
| 6 | precomputed_baseline (test) | — | — | 20.2 | — | — | — | — | — | — |
| 7 | MBLL | 1988 | Modified Beer-Lambert Law | 20.0 | 0.6000 | — | — | — | — | — |

### 31. Optical Coherence Tomography (OCT) (`oct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 35.0 | — | — | — | — | — | — |
| 2 | PSCAT | 2022 | PSCAT, PKU37 OCT | 32.2 | 0.9200 | — | — | — | — | — |
| 3 | BM3D | 2007 | Dabov et al., TIP 2007 | 25.0 | 0.8000 | — | — | — | — | — |
| 4 | FFT Recon (PWM) | — | — | 23.5 | — | 13.2 | 0.0008 | gap | 13.2 | low |
| 5 | Spectral Estimation (PWM) | — | Leitgeb et al. 2003, Optics Express | 23.5 | — | 12.9 | 0.0078 | gap | 12.9 | low |
| 6 | OCT Denoising Net (PWM) | — | Devalla et al. 2019, Biomed. Optics Express | 23.5 | — | 13.2 | 0.0008 | gap | 13.2 | low |
| 7 | bscan_baseline (test) | — | — | 23.5 | — | — | — | — | — | — |
| 8 | bscan_ideal_baseline (test) | — | — | 23.5 | — | — | — | — | — | — |

### 32. OCT Angiography (OCTA) (`octa`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Motion artifact DL | 2024 | MDPI Mathematics 2024 | 32.7 | 0.9260 | — | — | — | — | — |
| 2 | SU-Net (Siamese) | 2019 | Lee et al., 2019 | 28.0 | 0.8130 | — | — | — | — | — |
| 3 | CNN accelerated OCTA | 2022 | Sci Rep 2022 | 20.8 | 0.6300 | — | — | — | — | — |
| 4 | OCTA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.7049 | 19.5 | 0.8764 | done | 19.5 | pass |
| 5 | OCTA-FF [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.7049 | 19.5 | 0.8764 | done | 19.5 | pass |
| 6 | FFT Recon (OCTA) (PWM) | — | — | 18.8 | — | 9.0 | 0.0004 | partial | 9.0 | low |
| 7 | precomputed_baseline (test) | — | — | 18.8 | — | — | — | — | — | — |
| 8 | SSADA (single-scan) | 2012 | Xu et al. 2021 PMC8221851 (single-scan 12.09 dB) | 12.1 | 0.7000 | — | — | — | — | — |
| 9 | Single-scan OCTA (noisy) | 2021 | Xu et al. 2021, PMC8221851 | 12.1 | — | — | — | — | — | — |

### 33. Positron Emission Tomography (PET) (`pet`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR-PET | 2023 | SwinIR for PET denoising | 39.9 | 0.9600 | — | — | — | — | — |
| 2 | DeepPET | 2019 | Haggstrom et al., PMB 2019 | 34.7 | 0.9200 | — | — | — | — | — |
| 3 | FBP (emission tomography) (PWM) | — | — | 33.1 | — | 20.0 | 0.8977 | gap | 20.0 | pass |
| 4 | NeuroLF-PET (PWM) | — | Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58 | 33.1 | — | 8.6 | 0.0000 | gap | 8.6 | low |
| 5 | PET-DL (U-Net) (PWM) | — | Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9) | 33.1 | — | -26.3 | -0.0074 | fail | -26.3 | fail |
| 6 | fbp_ramlak (test) | — | — | 33.1 | — | — | — | — | — | — |
| 7 | fbp_shepp_logan (test) | — | — | 33.1 | — | — | — | — | — | — |
| 8 | precomputed_fbp (test) | — | — | 33.1 | — | — | — | — | — | — |
| 9 | MAP-OSEM | 2001 | Qi et al., PMB 2003 | 32.0 | 0.8700 | — | — | — | — | — |
| 10 | OSEM | 1994 | Hudson & Larkin, TMI 1994 | 30.0 | 0.8200 | — | — | — | — | — |
| 11 | MLEM | 1982 | Shepp & Vardi, TMI 1982 | 28.0 | 0.7500 | — | — | — | — | — |

### 34. Photoacoustic Imaging (`photoacoustic`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Iterative (model-based) | 2000 | Antholzer et al., Sci Rep 2020 | 30.2 | 0.8900 | — | — | — | — | — |
| 2 | Residual U-Net (Deep-PAT) | 2021 | Shahid et al., Front Neurosci 2021 | 29.9 | 0.9700 | — | — | — | — | — |
| 3 | Pixel-DL | 2020 | Antholzer et al., Sci Rep 2020 | 29.6 | 0.9100 | — | — | — | — | — |
| 4 | Post-DL (U-Net) | 2020 | Antholzer et al., Sci Rep 2020 | 24.4 | 0.8500 | — | — | — | — | — |
| 5 | Time Reversal (FBP) | 2000 | Xu & Wang, PMB 2005 | 22.7 | 0.7300 | — | — | — | — | — |
| 6 | Backprojection (limited view) | 2021 | Shahid et al., PMC8165448 (FD-UNet BP input=21.9) | 21.9 | 0.6500 | — | — | — | — | — |
| 7 | Deep-PAT [proxy] (PWM) | — | Richardson 1972, JOSA | 21.2 | 0.1988 | 3.4 | -0.1238 | gap | 3.4 | fail |
| 8 | Back Projection (PWM) | — | — | 19.8 | — | 4.4 | -0.3367 | gap | 4.4 | fail |
| 9 | precomputed_baseline (test) | — | — | 19.8 | — | — | — | — | — | — |
| 10 | Time Reversal (16 sensors) | 2020 | Tong et al., Scientific Reports 2020, PMC7244747 | 13.9 | 0.5000 | — | — | — | — | — |
| 11 | Tikhonov (32 views) | 2023 | Boink et al., PMC9872879 | 13.9 | — | — | — | — | — | — |

### 35. Portal Imaging (EPID) (`portal_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CycleGAN+Attention+Residual | 2024 | Lv et al., Medical Physics 2024 | 34.0 | 0.9650 | — | — | — | — | — |
| 2 | CycleGAN MVCT-to-kVCT | 2021 | Lee et al., Medical Physics 2021 | 32.7 | 0.9550 | — | — | — | — | — |
| 3 | Monte Carlo correction | 2005 | MC dose verification | 28.0 | 0.8200 | — | — | — | — | — |
| 4 | Flat-field correction | 2000 | EPID baseline | 25.0 | 0.7500 | — | — | — | — | — |
| 5 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 23.8 | 0.8887 | 28.9 | 0.9925 | partial | 28.9 | pass |
| 6 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 23.8 | 0.8887 | 28.9 | 0.9924 | partial | 28.9 | pass |
| 7 | PortalDL [proxy] (PWM) | — | Richardson 1972, JOSA | 23.8 | 0.8887 | 28.9 | 0.9924 | partial | 28.9 | pass |
| 8 | precomputed_baseline (test) | — | — | 17.3 | — | — | — | — | — | — |
| 9 | Raw EPID (uncorrected) | 2000 | Raw EPID baseline | 15.0 | 0.5000 | — | — | — | — | — |

### 36. Proton Therapy Imaging (`proton_therapy_img`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Residual GAN (PPI-to-DRR) | 2024 | Wang et al., PMC 2024 | 39.1 | 0.9870 | — | — | — | — | — |
| 2 | CycleGAN (CBCT-to-sCT) | 2024 | MDPI Sensors 2024 | 34.1 | 0.8600 | — | — | — | — | — |
| 3 | Proton CT DL | 2022 | DL for proton imaging | 32.0 | 0.9200 | — | — | — | — | — |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9843 | 36.2 | 0.9983 | partial | 36.2 | pass |
| 5 | FBP | 1971 | FBP baseline | 28.0 | — | 36.2 | 0.9983 | partial | 36.2 | pass |
| 6 | precomputed_baseline (test) | — | — | 26.6 | — | — | — | — | — | — |

### 37. Single Photon Emission CT (SPECT) (`spect`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DIP-SPECT | 2020 | Baguer et al., 2020 | 33.3 | 0.9000 | — | — | — | — | — |
| 2 | FBP (emission tomography) (PWM) | — | — | 30.0 | — | 12.2 | 0.7878 | gap | 12.2 | low |
| 3 | SPECT-DL (OSEM+) (PWM) | — | Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging | 30.0 | — | 3.1 | 0.0000 | gap | 3.1 | fail |
| 4 | SPECT-UNet (PWM) | — | Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6) | 30.0 | — | 3.1 | 0.0000 | gap | 3.1 | fail |
| 5 | fbp_ramlak (test) | — | — | 30.0 | — | — | — | — | — | — |
| 6 | precomputed_fbp (test) | — | — | 30.0 | — | — | — | — | — | — |
| 7 | OSEM | 1994 | Hudson & Larkin, 1994 | 28.5 | 0.7800 | 3.1 | 0.0000 | gap | 3.1 | fail |
| 8 | MLEM | 1982 | Shepp & Vardi, 1982 | 26.0 | 0.7000 | — | — | — | — | — |

### 38. Photon-Counting Spectral CT (`spectral_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | D3QN | 2024 | Phys Med Biol 2024 | 37.4 | 0.9790 | — | — | — | — | — |
| 2 | Butterfly-Net | 2022 | Li et al., PMB 2022 | 34.0 | 0.9500 | — | — | — | — | — |
| 3 | ADMM-TV | 2010 | TV regularization | 30.0 | 0.8700 | — | — | — | — | — |
| 4 | Material decomposition | 2003 | Alvarez & Macovski, PMB 1976 | 28.0 | 0.8500 | — | — | — | — | — |
| 5 | FBP per bin (lowest energy) | 2024 | Xing et al., 2024, PMC11744124 | 27.0 | 0.5000 | — | — | — | — | — |
| 6 | FBP (30 sparse views) | 2025 | Guo et al., QIMS 2025, PMC12209656 | 15.5 | — | — | — | — | — | — |
| 7 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 13.3 | 0.1206 | 7.2 | 0.0000 | partial | 7.2 | low |
| 8 | SpectralCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.3 | 0.1206 | -43.1 | 0.0000 | fail | -43.1 | fail |
| 9 | precomputed_baseline (test) | — | — | 12.3 | — | — | — | — | — | — |

### 39. Susceptibility-Weighted Imaging (SWI) (`swi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DeepSWI (cGAN) | 2023 | Genc et al., JMRI 2023 | 36.9 | 0.8900 | — | — | — | — | — |
| 2 | Homodyne filtering | 2004 | Haacke et al., MRM 2004 | 28.0 | 0.8500 | — | — | — | — | — |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1521 | -33.5 | 0.0000 | fail | -33.5 | fail |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1521 | -34.7 | 0.0000 | fail | -34.7 | fail |
| 5 | SWI-Net [proxy] (PWM) | — | — | 12.9 | 0.1521 | 14.6 | -0.1560 | done | 14.6 | low |
| 6 | precomputed_baseline (test) | — | — | 10.9 | — | — | — | — | — | — |

### 40. Ultrasound B-mode Imaging (`ultrasound`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | KD-optimized beamformer | 2025 | Scientific Reports 2025 | 39.0 | 0.9530 | — | — | — | — | — |
| 2 | DAS (Delay-and-Sum) | 1990 | DAS baseline | 30.4 | — | — | — | — | — | — |
| 3 | Deep beamforming (Goudarzi) | 2020 | Goudarzi et al., IEEE TUFFC 2022 | 29.1 | — | — | — | — | — | — |
| 4 | DAS single plane wave | 2020 | Li et al., IUS 2020 / CUBDL | 18.6 | — | — | — | — | — | — |
| 5 | DAS single PW (deep target, 8cm) | 2017 | Perdios et al., IEEE TUFFC 2017 | 17.0 | 0.4500 | — | — | — | — | — |
| 6 | ADMIRE | 2018 | Byram et al., IEEE TUFFC 2015 | 15.8 | 0.3564 | — | — | — | — | — |
| 7 | US-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 15.8 | 0.3564 | 18.7 | 0.8521 | done | 18.7 | pass |
| 8 | Richardson-Lucy (ultrasound) (PWM) | — | Richardson 1972, JOSA | 14.8 | — | 20.6 | 0.8977 | partial | 20.6 | pass |
| 9 | rl_20iter (test) | — | — | 14.8 | — | — | — | — | — | — |
| 10 | rl_50iter (test) | — | — | 14.8 | — | — | — | — | — | — |
| 11 | DAS single PW (in vivo) | 2020 | Li et al., IUS 2020 / CUBDL, PMC verified | 13.5 | — | — | — | — | — | — |

### 41. X-ray Radiography (`xray_radiography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CheXNet [proxy] (PWM) | — | Richardson 1972, JOSA | 46.9 | 0.9999 | 36.0 | 0.9984 | gap | 36.0 | pass |
| 2 | X-ray UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 46.9 | 0.9999 | 36.0 | 0.9984 | gap | 36.0 | pass |
| 3 | Improved Restormer | 2025 | Springer 2025 | 37.3 | 0.9360 | — | — | — | — | — |
| 4 | BM3D | 2007 | Dabov et al., TIP 2007 | 32.0 | 0.8800 | — | — | — | — | — |
| 5 | Flat-field + simple filter | 2018 | Kang et al., J X-ray Sci Tech 2018, PMC6130336 (noisy=24.... | 30.0 | 0.8500 | — | — | — | — | — |
| 6 | NLM | 2005 | Buades et al., CVPR 2005 | 28.0 | 0.8600 | — | — | — | — | — |
| 7 | FBP (X-ray radiography) (PWM) | — | — | 27.1 | — | 4.0 | 0.0000 | gap | 4.0 | fail |
| 8 | precomputed_baseline (test) | — | — | 27.1 | — | — | — | — | — | — |
| 9 | Median filter | 2000 | Median denoising baseline | 25.0 | 0.8000 | — | — | — | — | — |
| 10 | Noisy input (flat-field only) | 2018 | Kang et al., J X-ray Sci Tech 2018, PMC6130336 | 24.1 | 0.3870 | — | — | — | — | — |

## Coherent Imaging

### 42. Digital Holographic Microscopy (`holography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Phase distortion DL | 2024 | ScienceDirect 2024 (DHM) | 36.9 | 0.9900 | — | — | — | — | — |
| 2 | CEHAN (CGH) | 2025 | Appl Opt 65(7), 2025 | 35.7 | — | — | — | — | — | — |
| 3 | Wirtinger Holography | 2020 | Peng et al., SIGGRAPH Asia 2020 | 30.0 | — | — | — | — | — | — |
| 4 | HIO | 1982 | Fienup, Applied Optics 1982 | 25.0 | 0.7800 | — | — | — | — | — |
| 5 | Angular Spectrum | 2000 | Goodman, Fourier Optics | 22.0 | 0.7000 | 8.7 | 0.2071 | gap | 8.7 | low |
| 6 | GS (Gerchberg-Saxton) | 1972 | Gerchberg & Saxton, Optik 1972 | 20.0 | 0.6500 | — | — | — | — | — |
| 7 | Direct backpropagation | 1970 | Gabor, Nature 1948 | 15.0 | 0.5000 | — | — | — | — | — |
| 8 | sqrt_intensity_amplitude (test) | — | — | 14.9 | — | — | — | — | — | — |

### 43. Optical Diffraction Tomography (ODT) (`odt`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 30.5 | 0.9608 | 3.5 | -0.0006 | gap | 3.5 | fail |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 30.5 | 0.9608 | 3.2 | -0.0005 | gap | 3.2 | fail |
| 3 | ODT-Net (PhaseNet) [proxy] (PWM) | — | Richardson 1972, JOSA | 30.5 | 0.9608 | 3.2 | -0.0005 | gap | 3.2 | fail |
| 4 | precomputed_baseline (test) | — | — | 27.2 | — | — | — | — | — | — |
| 5 | Rytov approximation | 2000 | Rytov, 1937 | 25.0 | — | — | — | — | — | — |
| 6 | Born approximation | 2000 | Wolf, Opt Commun 1969 | 22.0 | — | — | — | — | — | — |

### 44. Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DLMMPR (coded diffraction) | 2025 | arXiv 2511.12556 | 45.8 | 0.9840 | — | — | — | — | — |
| 2 | NAS-PRNet (bio cells) | 2022 | arXiv 2210.14231 | 36.7 | 0.8660 | — | — | — | — | — |
| 3 | WF (Wirtinger Flow) | 2015 | Candes et al., TIT 2015 | 30.0 | 0.9000 | — | — | — | — | — |
| 4 | HIO | 1982 | Fienup, Applied Optics 1982 | 25.0 | 0.7500 | 3.4 | 0.0006 | gap | 3.4 | fail |
| 5 | ER (Error Reduction) | 1972 | Gerchberg & Saxton, 1972 | 23.0 | 0.7000 | — | — | — | — | — |
| 6 | Wiener (low SNR) | 2000 | Wiener filter baseline | 18.0 | 0.6000 | — | — | — | — | — |
| 7 | HIO (0 dB input SNR) | 2015 | Shechtman et al., IEEE SPM 2015 | 14.0 | 0.3500 | 3.4 | 0.0006 | gap | 3.4 | fail |
| 8 | RAAR [proxy] (PWM) | — | Richardson 1972, JOSA | 13.6 | 0.3397 | -42.6 | 0.0000 | fail | -42.6 | fail |
| 9 | prDeep [proxy] (PWM) | — | Richardson 1972, JOSA | 13.6 | 0.3397 | -42.6 | 0.0000 | fail | -42.6 | fail |
| 10 | precomputed_baseline (test) | — | — | 12.6 | — | — | — | — | — | — |

### 45. Ptychographic Imaging (`ptychography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | AutoPhaseNN | 2022 | Cherukara et al., APL 2022 | 33.0 | — | — | — | — | — | — |
| 2 | PtychoNN | 2020 | Cherukara et al., APL 2020 | 31.0 | — | -33.8 | -0.0001 | fail | -33.8 | fail |
| 3 | ePIE | 2009 | Maiden & Rodenburg, Ultramicroscopy 2009 | 28.0 | 0.8500 | -33.8 | -0.0001 | fail | -33.8 | fail |
| 4 | PIE | 2004 | Rodenburg & Faulkner, APL 2004 | 22.0 | 0.7000 | -33.8 | -0.0001 | fail | -33.8 | fail |
| 5 | PtychoNN 2.0 (PWM) | — | — | 21.0 | — | -33.8 | -0.0001 | fail | -33.8 | fail |
| 6 | precomputed_baseline (test) | — | — | 21.0 | — | — | — | — | — | — |
| 7 | precomputed_phase_baseline (test) | — | — | 21.0 | — | — | — | — | — | — |

### 46. Talbot-Lau X-ray Grating Interferometry (`talbot_lau`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 34.3 | 0.9999 | 2.6 | -0.0568 | gap | 2.6 | fail |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 34.3 | 0.9999 | 2.2 | -0.0521 | gap | 2.2 | fail |
| 3 | Talbot-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 34.3 | 0.9999 | 2.6 | -0.0568 | gap | 2.6 | fail |
| 4 | precomputed_baseline (test) | — | — | 28.9 | — | — | — | — | — | — |
| 5 | Phase-stepping | 2006 | Weitkamp et al., Opt Express 2005 | 28.0 | — | — | — | — | — | — |
| 6 | Fourier analysis | 2006 | Takeda et al., JOSA 1982 | 25.0 | — | — | — | — | — | — |

## Microscopy

### 47. Confocal 3D Z-Stack (`confocal_3d`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CARE 3D | 2018 | Weigert et al., Nature Methods 2018 | 32.0 | 0.9000 | — | — | — | — | — |
| 2 | Noise2Void 3D | 2019 | Krull et al., CVPR 2019 | 28.0 | 0.8200 | — | — | — | — | — |
| 3 | CARE-3D (PWM) | — | — | 27.3 | — | 10.0 | 0.0035 | gap | 10.0 | low |
| 4 | CARE-3D (slice-wise) (PWM) | — | — | 27.3 | — | 9.2 | 0.0003 | gap | 9.2 | low |
| 5 | precomputed_baseline (test) | — | — | 27.3 | — | — | — | — | — | — |
| 6 | rl_20iter (test) | — | — | 27.3 | — | — | — | — | — | — |
| 7 | Richardson-Lucy 3D | 1972 | Richardson 1972 | 26.0 | 0.7500 | — | — | — | — | — |

### 48. Confocal Live-Cell Microscopy (`confocal_livecell`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | 0.9200 | 11.2 | 0.0074 | gap | 11.2 | low |
| 2 | precomputed_baseline (test) | — | — | 32.3 | — | — | — | — | — | — |
| 3 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 29.0 | 0.8600 | — | — | — | — | — |
| 4 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | 0.8000 | 44.0 | 0.9996 | gap | 44.0 | pass |

### 49. Dark-Field Microscopy (`dark_field`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DAPD | 2024 | Nano Letters 2024 | 33.0 | 0.9890 | — | — | — | — | — |
| 2 | BM3D | 2007 | Dabov et al., TIP 2007 | 30.0 | 0.8500 | — | — | — | — | — |
| 3 | Richardson-Lucy (PWM) | — | — | 25.1 | — | 39.7 | 0.9992 | gap | 39.7 | pass |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | 25.1 | — | 5.3 | 0.0007 | gap | 5.3 | low |
| 5 | DF-UNet (PWM) | — | Wolfer, T. et al. (2021) DL for dark-field X-ray CT, Sci. Rep. 11:5005 | 25.1 | — | 5.1 | 0.0011 | gap | 5.1 | low |
| 6 | precomputed_baseline (test) | — | — | 25.1 | — | — | — | — | — | — |
| 7 | Median filter | 2000 | Median denoising baseline | 24.0 | 0.7800 | — | — | — | — | — |

### 50. Differential Interference Contrast (DIC) (`dic`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DL phase recovery | 2020 | DL for DIC | 30.0 | 0.8800 | — | — | — | — | — |
| 2 | TIE-GANs | 2024 | Poliwoda et al., J Biomed Opt 2024 | 28.1 | 0.9800 | — | — | — | — | — |
| 3 | PINN-TIE | 2022 | Zhang et al., Opt Express 2022 | 25.2 | 0.9190 | — | — | — | — | — |
| 4 | TIE-DIC | 2010 | TIE for DIC | 25.0 | — | — | — | — | — | — |
| 5 | Phase gradient DIC | 2015 | Gradient-based DIC | 22.0 | 0.7000 | — | — | — | — | — |
| 6 | Simple deconvolution | 2000 | DIC basic deconv | 18.0 | 0.6000 | — | — | — | — | — |
| 7 | Richardson-Lucy (PWM) | — | — | 15.6 | — | 2.5 | 0.0015 | gap | 2.5 | fail |
| 8 | CARE (PWM) | — | Weigert et al. 2018 | 15.6 | — | 7.3 | 0.0001 | partial | 7.3 | low |
| 9 | DIC-Net (PWM) | — | Mir, A. et al. (2015) Automated DIC microscopy, J. Microsc. 257(2) | 15.6 | — | 7.3 | 0.0001 | partial | 7.3 | low |
| 10 | precomputed_baseline (test) | — | — | 15.6 | — | — | — | — | — | — |

### 51. DNA-PAINT Super-Resolution (`dna_paint`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Richardson-Lucy (PWM) | — | — | 30.9 | — | 25.8 | 0.7652 | partial | 25.8 | pass |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 30.9 | — | 21.4 | 0.0590 | partial | 21.4 | pass |
| 3 | DECODE-PAINT (PWM) | — | Speiser, A. et al. (2021) DL for dense SMLM, Nature Methods 18:1090 | 30.9 | — | 21.7 | 0.1127 | partial | 21.7 | pass |
| 4 | precomputed_baseline (test) | — | — | 30.9 | — | — | — | — | — | — |
| 5 | DeepSTORM | 2018 | Nehme et al., Optica 2018 | 22.0 | — | — | — | — | — | — |
| 6 | PICASSO | 2020 | Reymond et al., PNAS 2020 | 20.0 | — | — | — | — | — | — |

### 52. Expansion Microscopy (ExM) (`expansion`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CARE (PWM) | — | Weigert et al. 2018 | 33.9 | — | 12.3 | 0.1713 | gap | 12.3 | low |
| 2 | EXpansionNet (PWM) | — | Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090 | 33.9 | — | 10.1 | 0.1260 | gap | 10.1 | low |
| 3 | precomputed_baseline (test) | — | — | 33.9 | — | — | — | — | — | — |
| 4 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 28.0 | 0.8000 | — | — | — | — | — |
| 5 | Richardson-Lucy ExM | 2015 | Chen et al., Science 2015 | 26.0 | — | 17.4 | 0.2544 | partial | 17.4 | pass |

### 53. Fluorescence Lifetime Imaging (FLIM) (`flim`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | MLE Fit (PWM) | — | Becker 2012, J. Microscopy | 36.9 | — | 29.6 | 0.9276 | partial | 29.6 | pass |
| 2 | MLE Fit (iterative) (PWM) | — | Becker 2012, J. Microscopy | 36.9 | — | 29.6 | 0.9276 | partial | 29.6 | pass |
| 3 | precomputed_baseline (test) | — | — | 36.9 | — | — | — | — | — | — |
| 4 | Net-FLIM (DL) | 2019 | Smith et al., Biomed Opt Express 2019 | 30.0 | 0.9000 | — | — | — | — | — |
| 5 | Phasor approach | 2008 | Digman et al., Biophys J 2008 | 25.0 | — | — | — | — | — | — |
| 6 | Multi-exponential fitting | 2000 | Elson 2004 | 22.0 | — | — | — | — | — | — |

### 54. Fourier Ptychographic Microscopy (FPM) (`fpm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Gradient descent FPM | 2015 | Tian & Waller, Optica 2015 | 30.0 | 0.8700 | 20.8 | 0.5958 | partial | 20.8 | pass |
| 2 | GS-FPM | 2013 | Zheng et al., Nature Photonics 2013 | 28.0 | 0.8500 | — | — | — | — | — |
| 3 | Sequential Phase Retrieval (PWM) | — | — | 18.2 | — | 22.1 | 0.5132 | partial | 22.1 | pass |
| 4 | Fourier Ptychnet (PWM) | — | Jiang et al. 2018, Biomed. Optics Express | 18.2 | — | 22.1 | 0.5132 | partial | 22.1 | pass |
| 5 | precomputed_baseline (test) | — | — | 18.2 | — | — | — | — | — | — |
| 6 | Single low-res capture | 2013 | FPM single image baseline | 18.0 | 0.6000 | — | — | — | — | — |

### 55. Image Scanning Microscopy (ISM) (`ism`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Richardson-Lucy (PWM) | — | — | 34.0 | — | 13.7 | 0.1903 | gap | 13.7 | low |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 34.0 | — | 21.3 | 0.0137 | gap | 21.3 | pass |
| 3 | ISM-Reassignment-Net (PWM) | — | Castello, M. et al. (2019) Image scanning microscopy ISM, Nature Methods 16:175 | 34.0 | — | 21.3 | 0.0137 | gap | 21.3 | pass |
| 4 | precomputed_baseline (test) | — | — | 34.0 | — | — | — | — | — | — |
| 5 | Airyscan processing | 2017 | Huff, Methods Appl Fluor 2017 | 30.0 | — | — | — | — | — | — |
| 6 | Pixel reassignment | 2010 | Muller & Enderlein, PRL 2010 | 28.0 | — | — | — | — | — | — |

### 56. Lattice Light-Sheet Microscopy (`lattice_lightsheet`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CARE 3D | 2018 | Weigert et al., Nature Methods 2018 | 32.0 | 0.9000 | 13.3 | 0.0002 | gap | 13.3 | low |
| 2 | Richardson-Lucy 3D | 1972 | Richardson 1972 | 26.0 | 0.7500 | 19.2 | 0.8444 | partial | 19.2 | pass |
| 3 | LLSM-CARE (PWM) | — | Weigert, M. et al. (2018) Content-aware restoration for lattice light-sheet, Nature Methods 15:1090 | 25.1 | — | 14.5 | 0.0375 | gap | 14.5 | low |
| 4 | precomputed_baseline (test) | — | — | 25.1 | — | — | — | — | — | — |

### 57. Light-Sheet Fluorescence Microscopy (LSFM) (`lightsheet`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | — | — | — | — | — | — |
| 2 | Richardson-Lucy | 1972 | Richardson 1972 | 26.0 | 0.7500 | — | — | — | — | — |
| 3 | Fourier Notch Filter (PWM) | — | — | 23.0 | — | 20.5 | 0.3595 | done | 20.5 | pass |
| 4 | VSNR (PWM) | — | — | 23.0 | — | 20.5 | 0.3595 | done | 20.5 | pass |
| 5 | DeStripe (PWM) | — | Liang et al. 2022 | 23.0 | — | — | — | — | — | — |
| 6 | precomputed_baseline (test) | — | — | 23.0 | — | — | — | — | — | — |
| 7 | rl_20iter (test) | — | — | 23.0 | — | — | — | — | — | — |
| 8 | fourier_notch (test) | — | — | 23.0 | — | — | — | — | — | — |
| 9 | Gaussian denoising | 2000 | Gaussian filter baseline | 22.0 | 0.7000 | — | — | — | — | — |

### 58. MINFLUX Nanoscopy (`minflux`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Richardson-Lucy (PWM) | — | — | 29.5 | — | 18.5 | 0.4753 | gap | 18.5 | pass |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 29.5 | — | 20.6 | 0.0871 | partial | 20.6 | pass |
| 3 | MINFLUX-Net (PWM) | — | Gwosch, K.C. et al. (2020) MINFLUX nanoscopy 3D, Nature Methods 17:217 | 29.5 | — | 21.8 | 0.0155 | partial | 21.8 | pass |
| 4 | precomputed_baseline (test) | — | — | 29.5 | — | — | — | — | — | — |
| 5 | MLE localization | 2006 | Ober et al., Biophys J 2004 | 18.0 | — | — | — | — | — | — |
| 6 | Gaussian fitting | 2002 | Thompson et al., Biophys J 2002 | 15.0 | — | — | — | — | — | — |

### 59. PALM/STORM Single-Molecule Localization (`palm_storm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Richardson-Lucy (STORM/PALM) (PWM) | — | — | 32.4 | — | 20.1 | 0.8647 | gap | 20.1 | pass |
| 2 | DECODE-SMLM (PWM) | — | Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090 | 32.4 | — | 6.7 | 0.0000 | gap | 6.7 | low |
| 3 | DeepSTORM (PWM) | — | Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4) | 32.4 | — | 6.7 | 0.0000 | gap | 6.7 | low |
| 4 | precomputed_baseline (test) | — | — | 32.4 | — | — | — | — | — | — |
| 5 | rl_20iter (test) | — | — | 32.4 | — | — | — | — | — | — |
| 6 | DECODE | 2021 | Speiser et al., Nature Methods 2021 | 25.0 | — | 6.7 | 0.0000 | gap | 6.7 | low |
| 7 | Deep-STORM | 2018 | Nehme et al., Optica 2018 | 22.0 | — | — | — | — | — | — |
| 8 | ThunderSTORM | 2014 | Ovesny et al., Bioinformatics 2014 | 18.0 | — | — | — | — | — | — |

### 60. Phase Contrast Microscopy (`phase_contrast`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Richardson-Lucy (PWM) | — | — | 45.6 | — | -43.3 | -0.0000 | fail | -43.3 | fail |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 45.6 | — | -40.0 | -0.0001 | fail | -40.0 | fail |
| 3 | PhaseNet (PWM) | — | Rivenson, Y. et al. (2018) Phase recovery with DL, Light: Sci. & Appl. 7:17141 | 45.6 | — | 8.2 | 0.0001 | gap | 8.2 | low |
| 4 | precomputed_baseline (test) | — | — | 45.6 | — | — | — | — | — | — |
| 5 | GAN (self-attention) | 2024 | Scientific Reports 2024 | 38.3 | 0.8800 | — | — | — | — | — |
| 6 | Fourier ptychography | 2013 | Zheng et al., Nature Photonics 2013 | 32.0 | 0.9000 | — | — | — | — | — |
| 7 | DL flat-fielding QPC | 2024 | ResearchGate 2024 | 29.1 | 0.8650 | — | — | — | — | — |
| 8 | TIE (Transport of Intensity) | 2001 | Zuo et al., Opt Express 2013 | 28.0 | — | — | — | — | — | — |

### 61. Polarization Microscopy (`polarization`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | PolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | 47.8 | 0.9999 | 15.1 | 0.8486 | gap | 15.1 | pass |
| 2 | Stokes-NN [proxy] (PWM) | — | Richardson 1972, JOSA | 47.8 | 0.9999 | 15.1 | 0.8486 | gap | 15.1 | pass |
| 3 | MDU-Net | 2022 | Opt Express 30(12), PMC9208591 | 38.1 | 0.8970 | — | — | — | — | — |
| 4 | MIRNet | 2022 | Opt Express 30(12), PMC9208591 | 37.9 | 0.8950 | — | — | — | — | — |
| 5 | DnCNN | 2022 | Opt Express 30(12), PMC9208591 | 34.4 | 0.8100 | — | — | — | — | — |
| 6 | PnP-HQS (PWM) | — | — | 30.9 | — | 15.3 | 0.8588 | gap | 15.3 | pass |
| 7 | precomputed_baseline (test) | — | — | 30.9 | — | — | — | — | — | — |
| 8 | Raw Mueller matrix | 2022 | Ye et al., Biomed Opt Express 2022, PMC9208591 | 29.0 | 0.5000 | — | — | — | — | — |
| 9 | Mueller matrix | 2000 | Chipman, Handbook of Optics | 25.0 | — | — | — | — | — | — |

### 62. Second Harmonic Generation (SHG) Microscopy (`shg`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | — | 24.5 | 0.8524 | partial | 24.5 | pass |
| 2 | DnCNN | 2023 | Bai et al., Biomed Opt Express 2023 | 25.4 | 0.7700 | — | — | — | — | — |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 24.1 | — | 19.0 | 0.0022 | partial | 19.0 | pass |
| 4 | SHG-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for SHG imaging, Nature Methods 15:1090 | 24.1 | — | 20.6 | 0.0915 | partial | 20.6 | pass |
| 5 | precomputed_baseline (test) | — | — | 24.1 | — | — | — | — | — | — |
| 6 | Gaussian denoising | 2000 | Gaussian filter baseline | 22.0 | 0.7000 | — | — | — | — | — |

### 63. Structured Illumination Microscopy (SIM) (`sim`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | ML-SIM | 2021 | Christensen et al., APL 2021 | 33.0 | — | — | — | — | — | — |
| 2 | fairSIM | 2015 | Muller et al., Bioinformatics 2016 | 30.5 | 0.8900 | 5.0 | 0.2392 | gap | 5.0 | low |
| 3 | Wiener-SIM | 2008 | Gustafsson et al., 2008 | 30.0 | 0.8800 | 5.0 | 0.2392 | gap | 5.0 | low |
| 4 | HiFi-SIM (PWM) | — | Wen et al. 2021, Light: S&A | 24.0 | — | 5.0 | 0.2392 | gap | 5.0 | low |
| 5 | Wiener-SIM (fast) (PWM) | — | — | 24.0 | — | 5.0 | 0.2392 | gap | 5.0 | low |
| 6 | precomputed_baseline (test) | — | — | 24.0 | — | — | — | — | — | — |
| 7 | wiener_sim (test) | — | — | 24.0 | — | 5.0 | 0.2392 | gap | 5.0 | low |
| 8 | Bicubic interpolation | 2000 | Interpolation baseline | 22.0 | 0.7000 | — | — | — | — | — |

### 64. Spinning Disk Confocal Microscopy (`spinning_disk`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 32.0 | 0.9000 | 1.9 | 0.0003 | gap | 1.9 | fail |
| 2 | SD-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for spinning disk confocal, Nature Methods 15:1090 | 30.6 | — | 3.0 | 0.0073 | gap | 3.0 | fail |
| 3 | precomputed_baseline (test) | — | — | 30.6 | — | — | — | — | — | — |
| 4 | Richardson-Lucy | 1972 | Richardson 1972 | 27.0 | 0.7800 | 25.3 | 0.9822 | done | 25.3 | pass |

### 65. STED Microscopy (`sted`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DDPM denoiser | 2023 | DDPM-avg for STED | 32.8 | 0.9200 | — | — | — | — | — |
| 2 | STED-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | 29.6 | — | 13.7 | 0.0002 | gap | 13.7 | low |
| 3 | RCAN-STED (PWM) | — | Chen, J. et al. (2021) Three-dimensional residual channel attention for STED, Nature Methods 18:678 | 29.6 | — | 18.5 | 0.0754 | gap | 18.5 | pass |
| 4 | precomputed_baseline (test) | — | — | 29.6 | — | — | — | — | — | — |
| 5 | rl_20iter (test) | — | — | 29.6 | — | — | — | — | — | — |
| 6 | Richardson-Lucy STED | 2006 | RL for STED | 28.0 | 0.8000 | — | — | — | — | — |
| 7 | Gaussian denoising | 2000 | Gaussian filter baseline | 24.0 | 0.7500 | — | — | — | — | — |

### 66. Three-Photon Microscopy (`three_photon`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DeepCAD-RT | 2023 | Li et al., Nature Biotech 2023 | 34.0 | — | — | — | — | — | — |
| 2 | Richardson-Lucy | 1972 | Richardson 1972 | 26.0 | — | 16.2 | 0.5137 | partial | 16.2 | pass |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 22.3 | — | 19.2 | 0.0688 | partial | 19.2 | pass |
| 4 | 3P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) CARE for 3P deep tissue imaging, Nature Methods 15:1090 | 22.3 | — | 17.8 | 0.0015 | partial | 17.8 | pass |
| 5 | precomputed_baseline (test) | — | — | 22.3 | — | — | — | — | — | — |
| 6 | Gaussian denoising | 2000 | Gaussian filter baseline | 20.0 | 0.6000 | — | — | — | — | — |

### 67. TIRF Microscopy (`tirf`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | RED-fairSIM | 2021 | Christensen et al., Photonics Research 2021 | 33.2 | 0.9000 | — | — | — | — | — |
| 2 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | 0.9100 | 35.4 | 0.9818 | done | 35.4 | pass |
| 3 | TIRF-SRRF [proxy] (PWM) | — | Richardson 1972, JOSA | 32.2 | 0.6316 | 35.4 | 0.9818 | partial | 35.4 | pass |
| 4 | precomputed_baseline (test) | — | — | 31.2 | — | — | — | — | — | — |
| 5 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | 0.8000 | 35.6 | 0.9825 | partial | 35.6 | pass |

### 68. Two-Photon / Multiphoton Microscopy (`two_photon`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | UNet-Att (self-supervised) | 2025 | Complex & Intelligent Systems, 2025 | 38.3 | 0.9500 | — | — | — | — | — |
| 2 | DeepCAD | 2021 | Li et al., Nature Methods 2021 | 35.0 | — | — | — | — | — | — |
| 3 | 2P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | 33.8 | — | 9.0 | 0.0132 | gap | 9.0 | low |
| 4 | 2P-DeepInterp (PWM) | — | Lecoq, J. et al. (2021) Removing independent noise in systems neuroscience using DeepInterpolation, Nature Methods 18:1401 | 33.8 | — | 8.0 | 0.0023 | gap | 8.0 | low |
| 5 | precomputed_baseline (test) | — | — | 33.8 | — | — | — | — | — | — |
| 6 | rl_20iter (test) | — | — | 33.8 | — | — | — | — | — | — |
| 7 | Richardson-Lucy | 1972 | Richardson 1972 | 27.0 | 0.7800 | 11.0 | 0.6020 | gap | 11.0 | low |

### 69. Widefield Fluorescence Microscopy (`widefield`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Restormer | 2022 | Zamir et al., CVPR 2022 | 35.5 | — | — | — | — | — | — |
| 2 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 31.0 | 0.8800 | — | — | — | — | — |
| 3 | Wiener deconvolution | 1949 | Wiener, 1949 | 26.0 | 0.7500 | — | — | — | — | — |
| 4 | precomputed_baseline (test) | — | — | 25.0 | — | — | — | — | — | — |
| 5 | m-rBCR | 2023 | m-rBCR deconvolution, 2023 | 24.9 | 0.8300 | — | — | — | — | — |
| 6 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 22.1 | 0.7500 | 2.1 | 0.0067 | gap | 2.1 | fail |
| 7 | Richardson-Lucy (20 iter) | 1972 | Richardson 1972 / Lucy 1974 | 13.4 | 0.4000 | 29.2 | 0.9835 | gap | 29.2 | pass |

### 70. Low-Dose Widefield Microscopy (`widefield_lowdose`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | BM3D + RL (PWM) | — | — | 29.0 | — | 28.2 | 0.8638 | done | 28.2 | pass |
| 2 | CARE (PWM) | — | — | 29.0 | — | 22.5 | 0.1254 | partial | 22.5 | pass |
| 3 | precomputed_baseline (test) | — | — | 29.0 | — | — | — | — | — | — |
| 4 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 26.0 | 0.8000 | — | — | — | — | — |
| 5 | Richardson-Lucy | 1972 | Richardson 1972 | 20.0 | 0.6000 | — | — | — | — | — |

## Electron Microscopy

### 71. Cryo-Electron Tomography (Cryo-ET) (`cryo_et`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | IsoNet | 2022 | Liu et al., Nature Commun 2022 | 28.0 | 0.8500 | — | — | — | — | — |
| 2 | SIRT | 1972 | Gilbert 1972 | 25.0 | 0.7000 | — | — | — | — | — |
| 3 | WBP | 1970 | Weighted back-projection | 22.0 | 0.6000 | — | — | — | — | — |
| 4 | Richardson-Lucy (PWM) | — | — | 13.2 | — | 12.2 | 0.3337 | done | 12.2 | low |
| 5 | CARE (PWM) | — | Weigert et al. 2018 | 13.2 | — | 11.4 | 0.0002 | done | 11.4 | low |
| 6 | CryoCARE (PWM) | — | Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol. | 13.2 | — | 11.4 | 0.0002 | done | 11.4 | low |
| 7 | precomputed_baseline (test) | — | — | 13.2 | — | — | — | — | — | — |
| 8 | WBP (45-deg missing wedge) | 2019 | Zhang et al., Sci Rep 2019, s41598-019-49267-x | 13.1 | 0.2800 | — | — | — | — | — |

### 72. Electron Backscatter Diffraction (EBSD) (`ebsd`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | EBSD-DL (DictIndex) [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9942 | 7.3 | 0.3548 | gap | 7.3 | low |
| 2 | EMsoft-EBSD [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9942 | 7.3 | 0.3548 | gap | 7.3 | low |
| 3 | Dictionary indexing | 2015 | Chen et al., Microscopy 2015 | 25.0 | — | — | — | — | — | — |
| 4 | Hough indexing | 1992 | Krieger-Lassen 1998 | 22.0 | — | — | — | — | — | — |
| 5 | precomputed_baseline (test) | — | — | 21.9 | — | — | — | — | — | — |

### 73. STEM-EDX Elemental Mapping (`edx_mapping`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | NMF denoising | 2015 | NMF for EDX | 26.0 | — | — | — | — | — | — |
| 2 | Richardson-Lucy (PWM) | — | — | 24.1 | — | 29.3 | 0.8672 | partial | 29.3 | pass |
| 3 | Richardson-Lucy (high quality) (PWM) | — | Richardson 1972, JOSA | 24.1 | — | 29.3 | 0.8672 | partial | 29.3 | pass |
| 4 | Richardson-Lucy (DL baseline) (PWM) | — | Tietz, C. et al. (2021) DL for EDS spectrum imaging, Ultramicroscopy 231 | 24.1 | — | 29.3 | 0.8672 | partial | 29.3 | pass |
| 5 | precomputed_baseline (test) | — | — | 24.1 | — | — | — | — | — | — |
| 6 | PCA denoising | 2010 | PCA for EDX | 24.0 | — | — | — | — | — | — |

### 74. Electron Energy Loss Spectroscopy (EELS) (`eels`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Deep CNN Denoiser | 2021 | Mohan et al., Microsc Microanal 2021 | 42.9 | 0.9900 | — | — | — | — | — |
| 2 | FISTA-L2 (Fourier ratio) [proxy] (PWM) | — | Richardson 1972, JOSA | 28.4 | 0.9979 | 12.8 | 0.7790 | gap | 12.8 | low |
| 3 | EELS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 28.4 | 0.9979 | 11.3 | 0.7135 | gap | 11.3 | low |
| 4 | MLLS-EELS [proxy] (PWM) | — | Richardson 1972, JOSA | 28.4 | 0.9979 | 11.3 | 0.7135 | gap | 11.3 | low |
| 5 | PCA denoising | 2012 | Cueva et al., Microsc Microanal 2012 | 28.0 | — | — | — | — | — | — |
| 6 | NMF decomposition | 2015 | NMF for EELS | 26.0 | — | — | — | — | — | — |
| 7 | precomputed_baseline (test) | — | — | 25.2 | — | — | — | — | — | — |

### 75. 4D-STEM Electron Diffraction (`electron_diffraction`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | ED-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 44.4 | 0.9999 | 1.3 | 0.2125 | gap | 1.3 | fail |
| 2 | CRISP-ED [proxy] (PWM) | — | Richardson 1972, JOSA | 44.4 | 0.9999 | 1.3 | 0.2125 | gap | 1.3 | fail |
| 3 | ePIE (electron ptychography) (PWM) | — | — | 42.0 | — | 13.5 | 0.8520 | gap | 13.5 | low |
| 4 | precomputed_baseline (test) | — | — | 42.0 | — | — | — | — | — | — |
| 5 | DPC (Differential Phase Contrast) | 2016 | Lazic et al., Ultramicroscopy 2016 | 25.0 | — | — | — | — | — | — |
| 6 | Center-of-mass analysis | 2014 | Muller-Caspary et al., 2014 | 22.0 | — | — | — | — | — | — |

### 76. Electron Holography (`electron_holography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | FIN (Fourier Imager Network) | 2022 | Huang et al., Light Sci Appl 2022 | 36.1 | 0.7850 | — | — | — | — | — |
| 2 | HoloPhaseNet (cGAN) | 2022 | Terbe et al., Biomed Opt Express 2022 | 35.3 | 0.9900 | — | — | — | — | — |
| 3 | DNN phase unwrapping | 2021 | DL electron holography | 30.0 | 0.8800 | — | — | — | — | — |
| 4 | Fourier filtering | 1993 | Lichte, Ultramicroscopy 1993 | 25.0 | — | — | — | — | — | — |
| 5 | EH-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0936 | 1.7 | 0.1353 | gap | 1.7 | fail |
| 6 | Phase-Sideband [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0936 | 1.7 | 0.1353 | gap | 1.7 | fail |
| 7 | Phase Retrieval (HIO) (PWM) | — | — | 9.5 | — | 1.4 | 0.0000 | partial | 1.4 | fail |
| 8 | precomputed_baseline (test) | — | — | 9.5 | — | — | — | — | — | — |

### 77. Electron Tomography (`electron_tomography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Joint DL model (IRDM) | 2019 | Zhang et al., Sci Rep 2019, s41598-019-49267-x | 27.5 | 0.9530 | — | — | — | — | — |
| 2 | IMOD-SIRT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 26.1 | 0.9625 | 9.5 | -0.0706 | gap | 9.5 | low |
| 3 | SIRT-3D [proxy] (PWM) | — | Richardson 1972, JOSA | 26.1 | 0.9625 | 9.5 | -0.0706 | gap | 9.5 | low |
| 4 | FBP (SIRT baseline) (PWM) | — | — | 25.1 | — | 0.6 | -0.0000 | gap | 0.6 | fail |
| 5 | precomputed_baseline (test) | — | — | 25.1 | — | — | — | — | — | — |
| 6 | SART (missing wedge) | 1972 | Zhang et al., Sci Rep 2019 | 18.6 | 0.3120 | — | — | — | — | — |
| 7 | WBP (missing wedge) | 1970 | Zhang et al., Sci Rep 2019 | 13.1 | 0.2800 | — | — | — | — | — |

### 78. Focused Ion Beam SEM (FIB-SEM) (`fib_sem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 34.0 | — | — | — | — | — | — |
| 2 | NRRN | 2021 | bioRxiv 2021 | 31.0 | 0.9710 | — | — | — | — | — |
| 3 | BM3D | 2007 | Dabov et al., 2007 | 30.0 | — | — | — | — | — | — |
| 4 | Richardson-Lucy (PWM) | — | — | 28.3 | — | 7.1 | 0.4344 | gap | 7.1 | low |
| 5 | CARE (PWM) | — | Weigert et al. 2018 | 28.3 | — | 1.7 | 0.0000 | gap | 1.7 | fail |
| 6 | FIB-SEM-Net (PWM) | — | Heinrich, L. et al. (2021) Whole-cell organelle segmentation in volume EM, Nature 599:141 | 28.3 | — | 2.7 | 0.0079 | gap | 2.7 | fail |
| 7 | precomputed_baseline (test) | — | — | 28.3 | — | — | — | — | — | — |

### 79. Scanning Electron Microscopy (SEM) (`sem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 34.0 | — | — | — | — | — | — |
| 2 | BM3D | 2007 | Dabov et al., TIP 2007 | 30.0 | 0.8500 | — | — | — | — | — |
| 3 | SEM-DL (SegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.9761 | 31.2 | 0.9867 | done | 31.2 | pass |
| 4 | SEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.9761 | 31.2 | 0.9867 | done | 31.2 | pass |
| 5 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 28.0 | — | — | — | — | — | — |
| 6 | NLM | 2005 | Buades et al., CVPR 2005 | 25.0 | 0.7800 | — | — | — | — | — |
| 7 | Richardson-Lucy (SEM) (PWM) | — | — | 23.2 | — | 30.9 | 0.9859 | partial | 30.9 | pass |
| 8 | precomputed_baseline (test) | — | — | 23.2 | — | — | — | — | — | — |
| 9 | Gaussian filter | 2000 | Gaussian baseline | 22.0 | 0.7000 | — | — | — | — | — |

### 80. Scanning Transmission Electron Microscopy (STEM) (`stem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DAE (Denoising AE) | 2023 | ACS Central Science 2023 | 42.9 | 0.9900 | — | — | — | — | — |
| 2 | STEM-DL (AtomSegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9800 | 8.9 | 0.5585 | gap | 8.9 | low |
| 3 | STEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9800 | 8.9 | 0.5585 | gap | 8.9 | low |
| 4 | Richardson-Lucy (STEM) (PWM) | — | — | 34.5 | — | 7.6 | 0.4796 | gap | 7.6 | low |
| 5 | precomputed_baseline (test) | — | — | 34.5 | — | — | — | — | — | — |
| 6 | SwinIR | 2021 | Liang et al., 2021 | 33.0 | — | — | — | — | — | — |
| 7 | BM3D | 2007 | Dabov et al., 2007 | 30.0 | 0.8500 | — | — | — | — | — |

### 81. Transmission Electron Microscopy (TEM) (`tem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CGRDN | 2024 | Lobato et al., npj Comp Mat 2024, s41524-023-01188-0 | 37.0 | — | — | — | — | — | — |
| 2 | SwinIR | 2021 | Liang et al., 2021 | 35.0 | — | — | — | — | — | — |
| 3 | Topaz-Denoise | 2020 | Bepler et al., Nature Commun 2020 | 32.0 | — | — | — | — | — | — |
| 4 | BM3D | 2007 | Lobato et al., npj Comp Mat 2024 (comparison) | 30.4 | — | — | — | — | — | — |
| 5 | TEM-DL (ePIE-Net) [proxy] (PWM) | — | Richardson 1972, JOSA | 26.3 | 0.9290 | 13.9 | 0.0464 | gap | 13.9 | low |
| 6 | TEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 26.3 | 0.9290 | 13.9 | 0.0464 | gap | 13.9 | low |
| 7 | Wiener filter (basic) | 2013 | Lobato & Van Dyck, Ultramicroscopy 2013 | 26.0 | — | — | — | — | — | — |
| 8 | FISTA-L2 (CTF correction) (PWM) | — | — | 25.3 | — | 30.8 | 0.6647 | partial | 30.8 | pass |
| 9 | precomputed_baseline (test) | — | — | 25.3 | — | — | — | — | — | — |
| 10 | NLM | 2005 | Buades et al., CVPR 2005 | 25.0 | 0.7500 | — | — | — | — | — |

## Computational Optics

### 82. Integral Photography (`integral`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DIBR [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | 19.3 | 0.8490 | gap | 19.3 | pass |
| 2 | EPINet [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | 19.3 | 0.8490 | gap | 19.3 | pass |
| 3 | Depth Estimation (PWM) | — | — | 41.1 | — | 22.9 | 0.9191 | gap | 22.9 | pass |
| 4 | precomputed_baseline (test) | — | — | 41.1 | — | — | — | — | — | — |
| 5 | Drizzle (IFS) | 2003 | Fruchter & Hook, PASP 2002 | 25.0 | — | — | — | — | — | — |
| 6 | PCA sky subtraction | 2012 | IFS baseline | 22.0 | — | — | — | — | — | — |

### 83. Light Field Imaging (`light_field`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DistgSSR | 2021 | Wang et al., TPAMI 2022 | 34.8 | 0.9790 | — | — | — | — | — |
| 2 | LFT | 2022 | Liang et al., 2022 | 34.8 | 0.9780 | — | — | — | — | — |
| 3 | EPIT | 2022 | EPIT, 2022 | 34.8 | 0.9780 | — | — | — | — | — |
| 4 | LF-InterNet | 2020 | Wang et al., ECCV 2020 | 34.1 | 0.9760 | — | — | — | — | — |
| 5 | LFSSR | 2018 | Yeung et al., ECCV 2018 | 33.7 | 0.9740 | 38.1 | 0.9965 | partial | 38.1 | pass |
| 6 | DistgEPIT | 2023 | CVPRW 2023 | 30.7 | — | — | — | — | — | — |
| 7 | VDSR (4x SR) | 2016 | Kim et al., CVPR 2016 / BasicLFSR benchmark | 28.6 | — | — | — | — | — | — |
| 8 | Shift-and-Sum (PWM) | — | — | 27.3 | — | 38.1 | 0.9965 | gap | 38.1 | pass |
| 9 | LFBM5D (PWM) | — | Alain et al. 2017, Signal Processing: Image Communication | 27.3 | — | 38.1 | 0.9965 | gap | 38.1 | pass |
| 10 | precomputed_baseline (test) | — | — | 27.3 | — | — | — | — | — | — |
| 11 | Bicubic (4x SR) | 2019 | Cheng et al., CVPRW 2019, BasicLFSR | 26.5 | 0.9200 | — | — | — | — | — |

## Computational Photography

### 84. Coded Exposure / Flutter Shutter (`coded_exposure`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 38.2 | 0.9999 | 6.9 | 0.1019 | gap | 6.9 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 38.2 | 0.9999 | 5.9 | 0.0825 | gap | 5.9 | low |
| 3 | FlowNet-Coded [proxy] (PWM) | — | Richardson 1972, JOSA | 38.2 | 0.9999 | 5.9 | 0.0825 | gap | 5.9 | low |
| 4 | Restormer | 2022 | Zamir et al., CVPR 2022 | 32.9 | 0.9610 | — | — | — | — | — |
| 5 | MPRNet | 2021 | Zamir et al., CVPR 2021 | 32.7 | 0.9590 | — | — | — | — | — |
| 6 | precomputed_baseline (test) | — | — | 32.1 | — | — | — | — | — | — |
| 7 | Wiener (flutter shutter) | 2006 | Raskar et al., SIGGRAPH 2006 | 26.0 | — | — | — | — | — | — |

### 85. Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | HyperE2VID | 2024 | Ercan et al., IEEE TIP 2024 | 14.8 | 0.5760 | — | — | — | — | — |
| 2 | ET-Net | 2021 | Weng et al., ICCV 2021 | 13.3 | 0.5520 | — | — | — | — | — |
| 3 | E2VID+ | 2020 | Stoffregen et al., ECCV 2020 | 11.5 | 0.5030 | 1.4 | 0.0000 | gap | 1.4 | fail |
| 4 | SPADE-E2VID | 2021 | Cadena et al., CVPRW 2021 | 10.4 | 0.4610 | — | — | — | — | — |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 9.7 | 0.1217 | 1.7 | 0.0000 | partial | 1.7 | fail |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 9.7 | 0.1217 | 1.4 | 0.0000 | partial | 1.4 | fail |
| 7 | precomputed_baseline (test) | — | — | 7.6 | — | — | — | — | — | — |
| 8 | E2VID | 2019 | Rebecq et al., TPAMI 2020 | 7.5 | 0.4500 | 1.4 | 0.0000 | partial | 1.4 | fail |
| 9 | Raw event accumulation | 2014 | Lichtsteiner et al., JSSC 2008 | 5.0 | 0.2000 | — | — | — | — | — |

### 86. High Dynamic Range (HDR) Imaging (`hdr_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | HDR-Transformer | 2022 | Liu et al., AAAI 2022 | 42.4 | — | — | — | — | — | — |
| 2 | AHDRNet | 2019 | Yan et al., CVPR 2019 | 41.1 | 0.9800 | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 40.5 | 0.8634 | 16.2 | 0.8425 | gap | 16.2 | pass |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 40.5 | 0.8634 | 14.8 | 0.7959 | gap | 14.8 | low |
| 5 | precomputed_baseline (test) | — | — | 38.6 | — | — | — | — | — | — |
| 6 | Debevec | 1997 | Debevec & Malik, SIGGRAPH 1997 | 30.0 | — | — | — | — | — | — |

### 87. Lensless (Diffuser Camera) Imaging (`lensless`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | LensNet | 2025 | LensNet, IJCAI 2025 | 27.5 | 0.8630 | — | — | — | — | — |
| 2 | MWDN | 2023 | MWDN, 2023 | 25.7 | 0.8160 | — | — | — | — | — |
| 3 | FlatNet | 2022 | Khan et al., TPAMI 2022 | 21.2 | 0.7200 | 17.2 | 0.8750 | partial | 17.2 | pass |
| 4 | ADMM | 2000 | Boyd et al., ADMM, 2010 | 12.8 | 0.4420 | 17.2 | 0.8750 | partial | 17.2 | pass |
| 5 | FlatNet-Lite (PWM) | — | — | 11.9 | — | 17.2 | 0.8750 | partial | 17.2 | pass |
| 6 | wiener_deconv (test) | — | — | 11.9 | — | — | — | — | — | — |
| 7 | Wiener deconvolution | 2025 | LensNet, IJCAI 2025 (DiffuserCam Wiener=7.33) | 7.3 | 0.0830 | — | — | — | — | — |

### 88. Panorama Multi-Focus Fusion (`panorama`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Deep homography | 2023 | DL panorama stitching 2023 | 33.6 | 0.9390 | — | — | — | — | — |
| 2 | UDIS (Unsupervised Deep Image Stitching) | 2021 | Nie et al., CVPR 2021 | 28.0 | 0.9000 | — | — | — | — | — |
| 3 | APAP | 2013 | Zaragoza et al., CVPR 2013 | 25.0 | 0.8500 | — | — | — | — | — |
| 4 | Laplacian Pyramid Fusion (PWM) | — | — | 16.7 | — | 27.6 | 0.9893 | gap | 27.6 | pass |
| 5 | Guided Filter Fusion (PWM) | — | — | 16.7 | — | 27.6 | 0.9893 | gap | 27.6 | pass |
| 6 | IFCNN (PWM) | — | Zhang et al. 2020 | 16.7 | — | 27.6 | 0.9893 | gap | 27.6 | pass |
| 7 | precomputed_baseline (test) | — | — | 16.7 | — | — | — | — | — | — |
| 8 | Single homography stitch | 2024 | Luo et al., arXiv 2406.19922, 2024 | 15.5 | 0.7000 | — | — | — | — | — |

## Neural Rendering

### 89. 3D Gaussian Splatting (3DGS) (`gaussian_splatting`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | 2DGS | 2024 | Huang et al., SIGGRAPH 2024 | 34.0 | — | — | — | — | — | — |
| 2 | Scaffold-GS | 2024 | Lu et al., CVPR 2024 | 33.8 | — | — | — | — | — | — |
| 3 | 3D Gaussian Splatting | 2023 | Kerbl et al., SIGGRAPH 2023 | 33.3 | 0.9690 | — | — | — | — | — |
| 4 | EWA Splatting (PWM) | — | — | 0.0 | — | — | — | — | — | — |
| 5 | 3DGS (full) (PWM) | — | Kerbl et al. SIGGRAPH 2023 | 0.0 | — | — | — | — | — | — |
| 6 | NeRF (baseline comparison) (PWM) | — | — | 0.0 | — | — | — | — | — | — |
| 7 | 3DGS (compact) (PWM) | — | — | 0.0 | — | — | — | — | — | — |
| 8 | direct_render_baseline (test) | — | — | 0.0 | — | — | — | — | — | — |

### 90. Neural Radiance Fields (NeRF) (`nerf`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Zip-NeRF | 2023 | Barron et al., ICCV 2023 | 33.7 | — | — | — | — | — | — |
| 2 | 3D Gaussian Splatting | 2023 | Kerbl et al., SIGGRAPH 2023 | 33.3 | 0.9690 | — | — | — | — | — |
| 3 | Instant-NGP | 2022 | Muller et al., SIGGRAPH 2022 | 33.2 | 0.9600 | — | — | — | — | — |
| 4 | TensoRF | 2022 | Chen et al., ECCV 2022 | 33.1 | 0.9630 | — | — | — | — | — |
| 5 | Mip-NeRF 360 | 2022 | Barron et al., CVPR 2022 | 33.1 | 0.9610 | — | — | — | — | — |
| 6 | Plenoxels | 2022 | Fridovich-Keil et al., CVPR 2022 | 31.7 | 0.9580 | — | — | — | — | — |
| 7 | NeRF | 2020 | Mildenhall et al., ECCV 2020 | 31.0 | 0.9470 | — | — | — | — | — |
| 8 | SfM + MVS (PWM) | — | — | 29.0 | — | — | — | — | — | — |
| 9 | NeRF (original MLP) (PWM) | — | Mildenhall et al. 2020 | 29.0 | — | — | — | — | — | — |
| 10 | Richardson-Lucy (proxy baseline) (PWM) | — | Richardson 1972, JOSA | 29.0 | — | — | — | — | — | — |
| 11 | FISTA-TV (proxy baseline) (PWM) | — | Beck & Teboulle 2009, SIAM | 29.0 | — | — | — | — | — | — |
| 12 | precomputed_baseline (test) | — | — | 29.0 | — | — | — | — | — | — |

## Depth Imaging

### 91. Flash LiDAR (`flash_lidar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Joint depth+reflectivity DNN | 2025 | arXiv 2505.13250 | 29.1 | — | — | — | — | — | — |
| 2 | TCSPC histogram | 2000 | flash LiDAR baseline | 22.0 | — | — | — | — | — | — |
| 3 | Matched filter SPAD | 2010 | SPAD baseline | 18.0 | — | — | — | — | — | — |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 5.3 | -0.6237 | 2.5 | 0.0467 | done | 2.5 | fail |
| 5 | FlashLiDAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 5.3 | -0.6237 | 2.5 | 0.0467 | done | 2.5 | fail |
| 6 | precomputed_baseline (test) | — | — | 4.3 | — | — | — | — | — | — |

### 92. LiDAR Scanner (`lidar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | PointNeXt [proxy] (PWM) | — | Richardson 1972, JOSA | 52.0 | 0.9999 | 10.0 | 0.0262 | gap | 10.0 | low |
| 2 | PointNet++ [proxy] (PWM) | — | Richardson 1972, JOSA | 52.0 | 0.9999 | 10.0 | 0.0262 | gap | 10.0 | low |
| 3 | BP-Net | 2022 | Tang et al., CVPR 2022 | 36.0 | — | — | — | — | — | — |
| 4 | FISTA-L2 (depth) (PWM) | — | — | 35.8 | — | 11.4 | 0.0413 | gap | 11.4 | low |
| 5 | precomputed_baseline (test) | — | — | 35.8 | — | — | — | — | — | — |
| 6 | CompletionFormer | 2023 | Zhang et al., CVPR 2023 | 35.5 | — | — | — | — | — | — |
| 7 | NLSPN | 2020 | Park et al., ECCV 2020 | 35.0 | — | — | — | — | — | — |
| 8 | Bilateral Filter | 1998 | Tomasi & Manduchi, 1998 | 25.0 | — | — | — | — | — | — |

### 93. Photometric Stereo (`photometric_stereo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CNN-PS | 2019 | Chen et al., CVPR 2019 | 32.0 | — | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 30.0 | 0.9683 | 10.3 | 0.1275 | gap | 10.3 | low |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 30.0 | 0.9683 | 10.3 | 0.1278 | gap | 10.3 | low |
| 4 | PS-FCN [proxy] (PWM) | — | Richardson 1972, JOSA | 30.0 | 0.9683 | 10.3 | 0.1275 | gap | 10.3 | low |
| 5 | precomputed_baseline (test) | — | — | 29.0 | — | — | — | — | — | — |
| 6 | Woodham (Lambertian) | 1980 | Woodham, Opt Eng 1980 | 25.0 | — | — | — | — | — | — |

### 94. Structured-Light Depth Camera (`structured_light`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SFNet (fringe-to-phase) | 2024 | ArXiv 2402.00977 | 38.0 | — | — | — | — | — | — |
| 2 | Phase-shifting (4-step) | 1984 | Creath, 1988 | 35.0 | 0.9500 | — | — | — | — | — |
| 3 | Gray code | 2003 | Scharstein & Szeliski, 2003 | 25.0 | — | — | — | — | — | — |
| 4 | SL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0036 | 1.9 | 0.0099 | gap | 1.9 | fail |
| 5 | FTPD [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0036 | 1.9 | 0.0099 | gap | 1.9 | fail |
| 6 | precomputed_baseline (test) | — | — | 8.3 | — | — | — | — | — | — |

### 95. Time-of-Flight Depth Camera (`tof_camera`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Phase unwrapping | 2000 | ToF baseline | 47.6 | 0.9999 | — | — | — | — | — |
| 2 | ToF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 47.6 | 0.9999 | 2.4 | 0.1542 | gap | 2.4 | fail |
| 3 | ToF-MPI Deconv [proxy] (PWM) | — | Richardson 1972, JOSA | 47.6 | 0.9999 | 2.4 | 0.1542 | gap | 2.4 | fail |
| 4 | FISTA-L2 (depth) (PWM) | — | — | 42.2 | — | 14.2 | 0.7217 | gap | 14.2 | low |
| 5 | precomputed_baseline (test) | — | — | 42.2 | — | — | — | — | — | — |
| 6 | DeepToF | 2017 | Marco et al., CVPR 2017 | 32.0 | — | — | — | — | — | — |
| 7 | Bilateral filter (depth) | 2014 | Park et al., Sensors 2014, PMC4168506 | 29.5 | — | — | — | — | — | — |

## Remote Sensing

### 96. Ground-Penetrating Radar (GPR) (`gpr`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | PGCDM (Physics-Guided Diffusion) | 2024 | Remote Sensing 17(23):3837 | 30.1 | 0.8760 | — | — | — | — | — |
| 2 | RTM (Reverse Time Migration) | 2000 | RTM | 25.0 | 0.8000 | — | — | — | — | — |
| 3 | PSTM | 2005 | Pre-stack time migration | 22.0 | 0.7200 | — | — | — | — | — |
| 4 | Kirchhoff migration | 2000 | GPR migration | 20.0 | 0.6500 | — | — | — | — | — |
| 5 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0507 | 32.8 | 0.9973 | gap | 32.8 | pass |
| 6 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0507 | 32.8 | 0.9973 | gap | 32.8 | pass |
| 7 | GPR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0507 | 32.8 | 0.9973 | gap | 32.8 | pass |
| 8 | Raw B-scan (noisy input) | 2021 | MCAE GPR, Electronics 10(11):1269 (noisy=11.23 dB) | 11.2 | 0.4000 | — | — | — | — | — |
| 9 | precomputed_baseline (test) | — | — | 10.9 | — | — | — | — | — | — |

### 97. Hyperspectral Remote Sensing (`hyperspectral_remote`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 49.7 | 0.9999 | 38.4 | 0.9965 | gap | 38.4 | pass |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 49.7 | 0.9999 | 38.3 | 0.9965 | gap | 38.3 | pass |
| 3 | SST-USRNet [proxy] (PWM) | — | Richardson 1972, JOSA | 49.7 | 0.9999 | 38.3 | 0.9965 | gap | 38.3 | pass |
| 4 | precomputed_baseline (test) | — | — | 35.0 | — | — | — | — | — | — |
| 5 | MST++ | 2022 | Cai et al., CVPRW 2022 (Winner) | 34.3 | — | — | — | — | — | — |
| 6 | HDNet | 2022 | Hu et al., CVPR 2022 | 32.1 | — | — | — | — | — | — |
| 7 | AWAN | 2020 | Li et al., CVPRW 2020 | 31.2 | — | — | — | — | — | — |
| 8 | HSCNN+ | 2018 | Shi et al., CVPRW 2018 | 26.4 | — | — | — | — | — | — |

### 98. Interferometric SAR (InSAR) (`insar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9173 | 0.9 | 0.0196 | gap | 0.9 | fail |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9173 | 0.3 | 0.0173 | gap | 0.3 | fail |
| 3 | InSAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9173 | 0.3 | 0.0173 | gap | 0.3 | fail |
| 4 | wrapped_phase_baseline (test) | — | — | 31.8 | — | — | — | — | — | — |
| 5 | SNAPHU | 2001 | Chen & Zebker, JOSA-A 2001 | 28.0 | — | — | — | — | — | — |
| 6 | Goldstein filter | 1998 | Goldstein & Werner, GRL 1998 | 22.0 | — | — | — | — | — | — |

### 99. Multispectral Satellite Imaging (`multispectral_sat`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CDFAN | 2024 | Entropy 27(6):567, PMC12191612 | 42.8 | — | — | — | — | — | — |
| 2 | PanNet | 2017 | Yang et al., ICCV 2017 | 36.1 | 0.9660 | — | — | — | — | — |
| 3 | GPPNN | 2021 | Xu et al., CVPR 2021 | 33.8 | 0.9500 | — | — | — | — | — |
| 4 | BDSD (Band-Dependent Spatial Detail) | 2008 | Vivone et al., GRSM 2015 | 30.0 | 0.9000 | — | — | — | — | — |
| 5 | EXP baseline (bicubic LRMS) | 2022 | Deng et al., IEEE GRSM 2022, PMC12031081 | 27.4 | 0.5000 | — | — | — | — | — |
| 6 | Nearest-neighbor (4x) | 2000 | Deng et al., IEEE GRSM 2022 benchmark | 22.0 | 0.6000 | — | — | — | — | — |
| 7 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 13.9 | 0.5795 | 27.7 | 0.9202 | gap | 27.7 | pass |
| 8 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 13.9 | 0.5795 | 27.2 | 0.9128 | gap | 27.2 | pass |
| 9 | MS-Pansharpening-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 13.9 | 0.5795 | 27.2 | 0.9128 | gap | 27.2 | pass |
| 10 | bicubic_upsample (test) | — | — | 11.3 | — | — | — | — | — | — |

### 100. Ocean Color Remote Sensing (`ocean_color`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 53.5 | 0.9999 | 20.1 | 0.7418 | gap | 20.1 | pass |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 53.5 | 0.9999 | 19.7 | 0.7296 | gap | 19.7 | pass |
| 3 | OC-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 53.5 | 0.9999 | 20.1 | 0.7418 | gap | 20.1 | pass |
| 4 | precomputed_baseline (test) | — | — | 44.2 | — | — | — | — | — | — |
| 5 | SRCNN | 2023 | GIScience & Remote Sensing 2023 | 25.2 | 0.7900 | — | — | — | — | — |
| 6 | MUMM | 2000 | Ruddick et al., RSE 2000 | 22.0 | — | — | — | — | — | — |

### 101. Passive Microwave Radiometry (`passive_microwave`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 28.5 | 0.9418 | 29.5 | 0.9846 | done | 29.5 | pass |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 28.5 | 0.9418 | 29.5 | 0.9846 | done | 29.5 | pass |
| 3 | PM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 28.5 | 0.9418 | 29.5 | 0.9846 | done | 29.5 | pass |
| 4 | OI (Optimal Interpolation) | 2000 | Bretherton et al., MWR 1976 | 25.0 | — | — | — | — | — | — |
| 5 | Tikhonov retrieval | 2000 | Tikhonov | 22.0 | — | — | — | — | — | — |
| 6 | precomputed_baseline (test) | — | — | 18.3 | — | — | — | — | — | — |
| 7 | Linear regression retrieval | 1990 | Statistical retrieval baseline | 18.0 | 0.5500 | — | — | — | — | — |

### 102. Polarimetric SAR (PolSAR) (`polsar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | PAN-DeSpeck | 2023 | CMC 76(3):54373 | 28.4 | 0.9050 | — | — | — | — | — |
| 2 | CNN learnable activation | 2021 | Remote Sensing 13(17):3444 | 26.4 | 0.8300 | — | — | — | — | — |
| 3 | Refined Lee | 2003 | Lee et al., TGRS 2003 | 24.0 | 0.7800 | — | — | — | — | — |
| 4 | Cloude-Pottier decomposition | 1997 | Cloude & Pottier, IEEE TGRS 1997 | 22.3 | 0.5815 | — | — | — | — | — |
| 5 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 22.3 | 0.5815 | 39.0 | 0.9975 | gap | 39.0 | pass |
| 6 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 22.3 | 0.5815 | 39.0 | 0.9975 | gap | 39.0 | pass |
| 7 | PolSAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 22.3 | 0.5815 | 39.0 | 0.9975 | gap | 39.0 | pass |
| 8 | Lee filter | 1999 | Lee et al., IEEE TGRS 1999 | 22.0 | 0.7000 | — | — | — | — | — |
| 9 | precomputed_baseline (test) | — | — | 19.4 | — | — | — | — | — | — |
| 10 | Single-look noisy input | 2017 | Wang et al., TGRS 2017 | 14.5 | — | — | — | — | — | — |

### 103. Radio Interferometry (VLBI) (`radio_interferometry`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CASA tclean | 2007 | McMullin et al., ASP 2007 | 28.0 | — | — | — | — | — | — |
| 2 | MEM | 1984 | Cornwell & Evans, A&A 1985 | 27.0 | — | — | — | — | — | — |
| 3 | CLEAN | 1974 | Hogbom, A&AS 1974 | 25.0 | — | — | — | — | — | — |
| 4 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3142 | 2.6 | 0.0057 | gap | 2.6 | fail |
| 5 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3142 | 2.6 | 0.0054 | gap | 2.6 | fail |
| 6 | R2D2 (interferometry) [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3142 | 2.6 | 0.0054 | gap | 2.6 | fail |
| 7 | precomputed_baseline (test) | — | — | 23.3 | — | — | — | — | — | — |

### 104. Synthetic Aperture Radar (SAR) (`sar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Omega-K Algorithm | 1992 | Stolt 1978 / Cafforio 1991 | 27.0 | 0.7500 | — | — | — | — | — |
| 2 | Range-Doppler Algorithm | 1978 | Curlander & McDonough, 1991 | 25.0 | 0.7000 | — | — | — | — | — |
| 3 | SAR-DL (PolSF) [proxy] (PWM) | — | Richardson 1972, JOSA | 23.0 | 0.8700 | 33.6 | 0.9852 | gap | 33.6 | pass |
| 4 | SAR-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 23.0 | 0.8700 | 33.6 | 0.9852 | gap | 33.6 | pass |
| 5 | Matched Filter (192 pts) | 2024 | Diffusion-Prior SAR, arXiv 2512.02768 | 19.1 | — | — | — | — | — | — |
| 6 | FBP (SAR backprojection) (PWM) | — | — | 18.5 | — | 12.3 | 0.0001 | partial | 12.3 | low |
| 7 | precomputed_baseline (test) | — | — | 18.5 | — | — | — | — | — | — |
| 8 | Matched Filter (24 pts, 2dB SNR) | 2024 | Diffusion-Prior SAR, arXiv 2512.02768 | 8.8 | — | — | — | — | — | — |

### 105. Sonar Imaging (`sonar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR | 2025 | Frontiers in Remote Sensing 2025 | 36.1 | 0.9810 | — | — | — | — | — |
| 2 | MUSIC | 1986 | Schmidt, IEEE TAP 1986 | 27.0 | — | — | — | — | — | — |
| 3 | MVDR/Capon beamforming | 1969 | Capon, Proc IEEE 1969 | 25.0 | — | — | — | — | — | — |
| 4 | FISTA-L2 (DAS) [proxy] (PWM) | — | Richardson 1972, JOSA | 16.0 | 0.2917 | 32.9 | 0.9674 | gap | 32.9 | pass |
| 5 | SonarSR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 16.0 | 0.2917 | 32.9 | 0.9678 | gap | 32.9 | pass |
| 6 | Sonar-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 16.0 | 0.2917 | 32.9 | 0.9678 | gap | 32.9 | pass |
| 7 | precomputed_baseline (test) | — | — | 15.0 | — | — | — | — | — | — |
| 8 | Matched Filter (sparse) | 2024 | SAR analog, arXiv 2512.02768 | 12.0 | — | — | — | — | — | — |

### 106. Weather / Doppler Radar (`weather_radar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Axial-UNet | 2025 | arXiv 2025 | 47.7 | 0.9940 | — | — | — | — | — |
| 2 | U-Net | 2020 | DL weather radar | 35.0 | 0.9500 | — | — | — | — | — |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 30.2 | 0.9754 | 0.0 | 0.0258 | gap | 0.0 | fail |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 30.2 | 0.9754 | -0.5 | 0.0221 | fail | -0.5 | fail |
| 5 | NowcastNet [proxy] (PWM) | — | Richardson 1972, JOSA | 30.2 | 0.9754 | -0.5 | 0.0221 | fail | -0.5 | fail |
| 6 | precomputed_baseline (test) | — | — | 26.9 | — | — | — | — | — | — |
| 7 | CLEAN-AP | 2000 | CLEAN for weather | 25.0 | — | — | — | — | — | — |

## Scanning Probe Microscopy

### 107. Atomic Force Microscopy (AFM) (`afm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Deep-AFM | 2020 | Rashidi & Wolkow, Machine Learning 2020 | 32.0 | 0.9000 | — | — | — | — | — |
| 2 | Richardson-Lucy (PWM) | — | — | 31.3 | — | 6.6 | 0.2546 | gap | 6.6 | low |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 31.3 | — | 5.4 | 0.0000 | gap | 5.4 | low |
| 4 | AFM-UNet (PWM) | — | Cherukara, M.J. et al. (2020) AI-enabled high-res, real-time imaging, npj Comput. Mater. 6:203 | 31.3 | — | 7.5 | 0.0149 | gap | 7.5 | low |
| 5 | precomputed_baseline (test) | — | — | 31.3 | — | — | — | — | — | — |
| 6 | Flatten + line correction | 2000 | SPM baseline processing | 25.0 | 0.7500 | — | — | — | — | — |

### 108. Magnetic Force Microscopy (MFM) (`mfm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Interval-BCS (AFM) | 2019 | Lu et al., Nanotechnology 2019, PMC6902871 | 43.2 | 0.9700 | — | — | — | — | — |
| 2 | Richardson-Lucy (PWM) | — | — | 34.3 | — | 5.6 | 0.1748 | gap | 5.6 | low |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 34.3 | — | 0.3 | 0.0001 | gap | 0.3 | fail |
| 4 | MFM-UNet (PWM) | — | Kim, M. et al. (2021) DL for magnetic force microscopy, npj Comput. Mater. 7:87 | 34.3 | — | 0.4 | 0.0005 | gap | 0.4 | fail |
| 5 | precomputed_baseline (test) | — | — | 34.3 | — | — | — | — | — | — |
| 6 | Adaptive Median (AFM) | 2019 | Lu et al., Nanotechnology 2019, PMC6902871 | 33.9 | 0.9500 | — | — | — | — | — |
| 7 | Wiener deconvolution | 1949 | Wiener 1949 / MFM tip deconv | 26.0 | 0.8000 | — | — | — | — | — |
| 8 | Deconvolution | 2000 | MFM tip deconvolution | 24.0 | 0.7500 | — | — | — | — | — |

### 109. Near-field Scanning Optical Microscopy (NSOM) (`nsom`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | BM3D | 2007 | Dabov et al., TIP 2007 | 28.0 | 0.8300 | — | — | — | — | — |
| 2 | Deconvolution | 2000 | Near-field deconvolution | 24.0 | 0.7500 | — | — | — | — | — |
| 3 | Richardson-Lucy (PWM) | — | — | 24.0 | — | 14.8 | 0.5366 | partial | 14.8 | low |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | 24.0 | — | 17.8 | 0.0575 | partial | 17.8 | pass |
| 5 | NSOM-Net (PWM) | — | Park, J. et al. (2020) DL for near-field optical microscopy, Optica 7(11) | 24.0 | — | 16.4 | 0.0008 | partial | 16.4 | pass |
| 6 | precomputed_baseline (test) | — | — | 24.0 | — | — | — | — | — | — |

### 110. Scanning Tunneling Microscopy (STM) (`stm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DeepSPM | 2020 | Krull et al., 2020 | 30.0 | 0.8800 | — | — | — | — | — |
| 2 | Richardson-Lucy (PWM) | — | — | 23.3 | — | 7.8 | 0.4738 | gap | 7.8 | low |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 23.3 | — | 1.4 | 0.0000 | gap | 1.4 | fail |
| 4 | STM-Net (PWM) | — | Ziatdinov, M. et al. (2021) DL for atomic-level STM, Nat. Mach. Intell. 3:269 | 23.3 | — | 1.4 | 0.0000 | gap | 1.4 | fail |
| 5 | precomputed_baseline (test) | — | — | 23.3 | — | — | — | — | — | — |
| 6 | Drift correction | 2000 | SPM baseline | 22.0 | 0.7000 | — | — | — | — | — |

## Industrial Inspection

### 111. Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR (SAM) | 2024 | Somani et al., CVPR Workshop 2023 | 35.1 | 0.9500 | — | — | — | — | — |
| 2 | HDL-SAM (SwinIR+Hypergraph) | 2024 | Somani & Banerjee, OpenReview 2024 | 31.6 | 0.9200 | — | — | — | — | — |
| 3 | Hypergraph Inpainting | 2023 | Somani et al., CVPR Workshop 2023 | 28.0 | 0.8200 | — | — | — | — | — |
| 4 | SAFT (Synth Aperture Focus) | 1980 | Doctor et al., 1986 | 25.0 | — | — | — | — | — | — |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 24.8 | 0.9483 | 14.5 | 0.7172 | gap | 14.5 | low |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 24.8 | 0.9483 | 13.3 | 0.6609 | gap | 13.3 | low |
| 7 | precomputed_baseline (test) | — | — | 22.6 | — | — | — | — | — | — |
| 8 | DAS beamforming | 1990 | Beamforming baseline | 22.0 | — | — | — | — | — | — |

### 112. Active Thermography (IR) (`active_thermography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | TESR (Transformer) | 2024 | Sci Reports 2024, PMC11227526 | 46.2 | 0.9920 | — | — | — | — | — |
| 2 | RCAN | 2024 | Sci Reports 2024, PMC11227526 | 45.9 | 0.9920 | — | — | — | — | — |
| 3 | EDSR | 2024 | Sci Reports 2024, PMC11227526 | 45.3 | 0.9900 | — | — | — | — | — |
| 4 | SRCNN | 2024 | Sci Reports 2024, PMC11227526 | 42.9 | 0.9840 | — | — | — | — | — |
| 5 | Bicubic baseline | 2024 | Sci Reports 2024, PMC11227526 | 42.1 | 0.9820 | — | — | — | — | — |
| 6 | Pulsed phase thermography | 1996 | Maldague & Marinetti, J Appl Phys 1996 | 25.0 | — | — | — | — | — | — |
| 7 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 8.2 | 0.1575 | 6.6 | 0.2274 | done | 6.6 | low |
| 8 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 8.2 | 0.1575 | 5.1 | 0.1723 | partial | 5.1 | low |
| 9 | precomputed_baseline (test) | — | — | 7.2 | — | — | — | — | — | — |

### 113. Eddy Current Imaging (`eddy_current`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Wavelet denoising | 2000 | Wavelet for ECT | 25.0 | — | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.9 | 0.6456 | 22.7 | 0.9743 | done | 22.7 | pass |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.9 | 0.6456 | 21.8 | 0.9682 | done | 21.8 | pass |
| 4 | ECT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 23.9 | 0.6456 | 22.7 | 0.9743 | done | 22.7 | pass |
| 5 | precomputed_baseline (test) | — | — | 22.9 | — | — | — | — | — | — |
| 6 | Impedance plane analysis | 2000 | ECT baseline | 22.0 | — | — | — | — | — | — |

### 114. Industrial X-ray CT (`industrial_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | ADMM-TransNet | 2025 | MDPI 2025 | 44.6 | 0.9960 | — | — | — | — | — |
| 2 | SIRT | 1972 | Gilbert 1972 | 30.0 | 0.8500 | — | — | — | — | — |
| 3 | FDK | 1984 | Feldkamp et al., 1984 | 28.0 | 0.8000 | — | — | — | — | — |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 21.3 | 0.4146 | 4.0 | 0.0000 | gap | 4.0 | fail |
| 5 | IndustrialCT-Net [proxy] (PWM) | — | Shepp & Logan 1974 | 21.3 | 0.4146 | -46.7 | 0.0000 | fail | -46.7 | fail |
| 6 | precomputed_baseline (test) | — | — | 20.3 | — | — | — | — | — | — |

### 115. Machine Vision / AOI (`machine_vision`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9999 | 30.4 | 0.9934 | partial | 30.4 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9999 | 30.3 | 0.9931 | partial | 30.3 | pass |
| 3 | UniAD | 2023 | You et al., NeurIPS 2022 | 32.0 | — | — | — | — | — | — |
| 4 | PatchCore | 2022 | Roth et al., CVPR 2022 | 30.0 | — | 30.4 | 0.9934 | done | 30.4 | pass |
| 5 | precomputed_baseline (test) | — | — | 28.3 | — | — | — | — | — | — |
| 6 | Template matching | 2000 | Brunelli, Template Matching, 2009 | 25.0 | — | — | — | — | — | — |

### 116. Shearography (`shearography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Phase-shifting shearography | 2000 | Hung, 1982 | 28.0 | — | — | — | — | — | — |
| 2 | FPD-CNN | 2020 | Lin et al., Applied Optics 2020 | 27.9 | 0.9720 | — | — | — | — | — |
| 3 | Fourier transform method | 1982 | Takeda et al., JOSA 1982 | 25.0 | — | — | — | — | — | — |
| 4 | DBDNet | 2021 | Li et al., Applied Optics 2021 | 20.6 | — | — | — | — | — | — |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 19.1 | 0.4833 | 25.5 | 0.9734 | partial | 25.5 | pass |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 19.1 | 0.4833 | 25.2 | 0.9715 | partial | 25.2 | pass |
| 7 | ShearNet [proxy] (PWM) | — | Richardson 1972, JOSA | 19.1 | 0.4833 | 25.2 | 0.9715 | partial | 25.2 | pass |
| 8 | OCPDE (Oriented Coupled PDE) | 2020 | Lin et al., Applied Optics 2020 | 14.1 | — | — | — | — | — | — |
| 9 | precomputed_baseline (test) | — | — | 13.2 | — | — | — | — | — | — |
| 10 | WFLPF (Windowed Fourier LP Filter) | 2020 | Lin et al., Applied Optics 2020 | 12.8 | — | — | — | — | — | — |

### 117. Terahertz Imaging (THz) (`terahertz`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 47.9 | 0.9999 | 21.1 | 0.9359 | gap | 21.1 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 47.9 | 0.9999 | 21.0 | 0.9352 | gap | 21.0 | pass |
| 3 | THz-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 47.9 | 0.9999 | 21.0 | 0.9352 | gap | 21.0 | pass |
| 4 | precomputed_baseline (test) | — | — | 37.1 | — | — | — | — | — | — |
| 5 | J-Net (real THz) | 2023 | Yeo et al., arXiv 2312.01638 | 32.5 | — | — | — | — | — | — |
| 6 | EARDB | 2023 | Hou et al., Entropy 25(3):440, PMC10047599 | 31.3 | 0.8910 | — | — | — | — | — |
| 7 | TDS deconvolution | 2000 | THz-TDS baseline | 22.0 | — | — | — | — | — | — |

### 118. Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CycleSR | 2025 | MSSP 2025 | 39.3 | — | — | — | — | — | — |
| 2 | CinCGAN | 2025 | MSSP 2025 | 36.4 | — | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 35.2 | 0.8974 | 3.7 | 0.0326 | gap | 3.7 | fail |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 35.2 | 0.8974 | 3.3 | 0.0302 | gap | 3.3 | fail |
| 5 | precomputed_baseline (test) | — | — | 31.1 | — | — | — | — | — | — |
| 6 | TFM (Total Focusing Method) | 2004 | Holmes et al., NDT&E Int 2005 | 28.0 | — | — | — | — | — | — |

### 119. X-ray NDT (Radiography) (`xray_ndt`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | U-Net++ | 2025 | NDT.net DIR 2025 | 32.3 | 0.8960 | — | — | — | — | — |
| 2 | BM3D denoising | 2007 | Dabov et al., TIP 2007 | 32.0 | 0.8800 | — | — | — | — | — |
| 3 | FBP | 1971 | FBP baseline | 28.0 | 0.8000 | — | — | — | — | — |
| 4 | Raw projection (no filtering) | 2000 | X-ray raw projection | 18.0 | 0.6000 | — | — | — | — | — |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 17.7 | 0.8530 | 8.0 | 0.3424 | partial | 8.0 | low |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 17.7 | 0.8530 | 8.0 | 0.3424 | partial | 8.0 | low |
| 7 | NDT-DefectNet [proxy] (PWM) | — | Richardson 1972, JOSA | 17.7 | 0.8530 | 8.0 | 0.3424 | partial | 8.0 | low |
| 8 | precomputed_baseline (test) | — | — | 16.7 | — | — | — | — | — | — |

### 120. X-ray Fluorescence (XRF) Imaging (`xrf_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DnCNN (XFCT) | 2024 | J Imaging 2024, PMC11204716 | 49.4 | 0.9430 | — | — | — | — | — |
| 2 | NLM (XFCT) | 2024 | J Imaging 2024, PMC11204716 | 39.9 | 0.8030 | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 29.8 | 0.9997 | 28.9 | 0.9718 | done | 28.9 | pass |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 29.8 | 0.9997 | 28.5 | 0.9696 | done | 28.5 | pass |
| 5 | XRF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 29.8 | 0.9997 | 28.5 | 0.9696 | done | 28.5 | pass |
| 6 | precomputed_baseline (test) | — | — | 26.7 | — | — | — | — | — | — |
| 7 | PCA denoising | 2010 | PCA for XRF | 25.0 | — | — | — | — | — | — |
| 8 | Fundamental parameters | 2000 | Sherman, Spectrochim Acta 1955 | 22.0 | — | — | — | — | — | — |

## Spectroscopy & Spectral Imaging

### 121. Brillouin Microscopy (`brillouin`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 40.4 | 0.9999 | 9.9 | 0.6771 | gap | 9.9 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 40.4 | 0.9999 | 8.2 | 0.5913 | gap | 8.2 | low |
| 3 | Brillouin-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 40.4 | 0.9999 | 8.2 | 0.5913 | gap | 8.2 | low |
| 4 | precomputed_baseline (test) | — | — | 35.8 | — | — | — | — | — | — |
| 5 | VIPA analysis | 2010 | Scarcelli & Yun, Opt Express 2011 | 28.0 | — | — | — | — | — | — |
| 6 | Lorentzian fitting | 2000 | Brillouin spectral fit | 25.0 | — | — | — | — | — | — |

### 122. Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 27.9 | 0.9820 | 27.0 | 0.8471 | done | 27.0 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 27.9 | 0.9820 | 27.0 | 0.8504 | done | 27.0 | pass |
| 3 | CARS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | 27.9 | 0.9820 | 27.0 | 0.8504 | done | 27.0 | pass |
| 4 | MEM (Maximum Entropy Method) | 2006 | Vartiainen et al., Opt Express 2006 | 25.0 | — | — | — | — | — | — |
| 5 | DnCNN | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 23.0 | 0.5900 | — | — | — | — | — |
| 6 | N2N (Noise2Noise) | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 20.6 | 0.5600 | — | — | — | — | — |
| 7 | Median Filter | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 20.1 | 0.4300 | — | — | — | — | — |
| 8 | precomputed_baseline (test) | — | — | 16.7 | — | — | — | — | — | — |
| 9 | Raw CARS (no correction) | 2000 | CARS raw baseline | 15.0 | 0.3500 | — | — | — | — | — |

### 123. DESI Mass Spectrometry Imaging (`desi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | NMF denoising | 2015 | NMF for MSI | 25.0 | — | — | — | — | — | — |
| 2 | Peak fitting | 2000 | DESI baseline | 22.0 | — | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 16.1 | 0.3230 | 11.6 | 0.6558 | partial | 11.6 | low |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 16.1 | 0.3230 | 9.9 | 0.5631 | partial | 9.9 | low |
| 5 | DESI-SegNet [proxy] (PWM) | — | Richardson 1972, JOSA | 16.1 | 0.3230 | 9.9 | 0.5631 | partial | 9.9 | low |
| 6 | Gaussian smoothing | 2000 | DESI-MSI smoothing baseline | 16.0 | 0.5000 | — | — | — | — | — |
| 7 | precomputed_baseline (test) | — | — | 15.1 | — | — | — | — | — | — |

### 124. FTIR Spectroscopic Imaging (`ftir_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 35.6 | 0.9304 | 18.9 | 0.8480 | gap | 18.9 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 35.6 | 0.9304 | 17.6 | 0.8042 | gap | 17.6 | pass |
| 3 | FTIR-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 35.6 | 0.9304 | 18.9 | 0.8480 | gap | 18.9 | pass |
| 4 | precomputed_baseline (test) | — | — | 34.6 | — | — | — | — | — | — |
| 5 | U-Net SR FTIR | 2022 | DL for FTIR imaging | 30.0 | 0.9000 | — | — | — | — | — |
| 6 | MCR-ALS | 2000 | Tauler, Chemom Intell Lab 1995 | 28.0 | — | — | — | — | — | — |
| 7 | ATR correction | 2000 | Bassan et al., Analyst 2010 | 24.0 | — | — | — | — | — | — |

### 125. Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9907 | 11.4 | 0.7647 | gap | 11.4 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9907 | 9.7 | 0.6901 | gap | 9.7 | low |
| 3 | LIBS-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9907 | 9.7 | 0.6901 | gap | 9.7 | low |
| 4 | precomputed_baseline (test) | — | — | 26.5 | — | — | — | — | — | — |
| 5 | PLS regression | 2005 | Hahn & Omenetto, Appl Spectrosc 2010 | 25.0 | — | — | — | — | — | — |
| 6 | Peak identification | 2000 | LIBS baseline | 22.0 | — | — | — | — | — | — |

### 126. Raman Imaging / Microscopy (`raman_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | DeepeR (1D ResUNet) | 2022 | Horgan et al., Anal Chem 2022, PMC9286315 | 46.2 | 0.9530 | — | — | — | — | — |
| 2 | PCA denoising | 2000 | Horgan et al., Anal Chem 2022 (comparison) | 39.4 | 0.8680 | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.8753 | 38.0 | 0.9720 | gap | 38.0 | pass |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.8753 | 37.2 | 0.9671 | gap | 37.2 | pass |
| 5 | RamanNet [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.8753 | 37.2 | 0.9671 | gap | 37.2 | pass |
| 6 | Savitzky-Golay | 1964 | Savitzky & Golay, 1964 | 20.0 | — | — | — | — | — | — |
| 7 | precomputed_baseline (test) | — | — | 19.7 | — | — | — | — | — | — |

### 127. Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | PCA denoising | 2010 | PCA for SIMS | 24.0 | — | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 22.6 | 0.9807 | 11.8 | 0.6547 | gap | 11.8 | low |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 22.6 | 0.9807 | 10.0 | 0.5569 | gap | 10.0 | low |
| 4 | SIMS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 22.6 | 0.9807 | 10.0 | 0.5569 | gap | 10.0 | low |
| 5 | Dead-time correction | 2000 | SIMS baseline | 22.0 | — | — | — | — | — | — |
| 6 | precomputed_baseline (test) | — | — | 20.5 | — | — | — | — | — | — |
| 7 | De-MSI (DL) | 2025 | Gank et al., Anal Chem 2025 | 18.9 | 0.7400 | — | — | — | — | — |

### 128. Stimulated Raman Scattering (SRS) Microscopy (`srs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 45.2 | 0.9999 | 24.8 | 0.9388 | gap | 24.8 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 45.2 | 0.9999 | 24.4 | 0.9344 | gap | 24.4 | pass |
| 3 | SRS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | 45.2 | 0.9999 | 24.4 | 0.9344 | gap | 24.4 | pass |
| 4 | precomputed_baseline (test) | — | — | 30.6 | — | — | — | — | — | — |
| 5 | U-Net CNN | 2019 | Manifold et al., Biomed Opt Express 10(8):3860, PMC6701518 | 28.9 | — | — | — | — | — | — |
| 6 | SHRED | 2021 | Opt Express 29(21):34205 | 25.0 | — | — | — | — | — | — |
| 7 | Spectral unmixing | 2000 | SRS baseline | 24.0 | — | — | — | — | — | — |
| 8 | UHRED (unsupervised) | 2021 | Opt Express 29(21):34205 | 22.0 | — | — | — | — | — | — |
| 9 | PURE-LET | 2019 | Manifold et al., Biomed Opt Express 10(8):3860, PMC6701518 | 13.5 | — | — | — | — | — | — |

## Astronomy & Space Imaging

### 129. Stellar Coronagraphy (`coronagraphy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.3538 | 26.9 | 0.9854 | done | 26.9 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.3538 | 25.7 | 0.9809 | partial | 25.7 | pass |
| 3 | DL-SpeckleNull [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.3538 | 25.7 | 0.9809 | partial | 25.7 | pass |
| 4 | precomputed_baseline (test) | — | — | 27.7 | — | — | — | — | — | — |
| 5 | PCA/KLIP | 2012 | Soummer et al., ApJL 2012 | 22.0 | — | — | — | — | — | — |
| 6 | LOCI | 2007 | Lafreniere et al., ApJ 2007 | 20.0 | — | — | — | — | — | — |
| 7 | Classical ADI | 2006 | Marois et al., ApJ 2006 | 18.0 | — | — | — | — | — | — |

### 130. Event Horizon Telescope (EHT) Imaging (`eht_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | PRIMO | 2023 | Medeiros et al., ApJL 2023 | 28.0 | — | 15.4 | 0.0019 | gap | 15.4 | pass |
| 2 | eht-imaging RML | 2019 | Chael et al., ApJ 2018 | 25.0 | — | — | — | — | — | — |
| 3 | SMILI | 2019 | Akiyama et al., ApJ 2019 | 24.0 | — | — | — | — | — | — |
| 4 | CLEAN | 1974 | Hogbom, A&AS 1974 | 20.0 | — | — | — | — | — | — |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0866 | 15.4 | 0.0019 | done | 15.4 | pass |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0866 | 15.4 | 0.0019 | done | 15.4 | pass |
| 7 | Dirty beam (no deconvolution) | 1974 | Raw visibility FT | 12.0 | — | — | — | — | — | — |
| 8 | precomputed_baseline (test) | — | — | 11.4 | — | — | — | — | — | — |

### 131. Lucky Imaging (`lucky_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 32.7 | 0.9890 | 20.5 | 0.8480 | gap | 20.5 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 32.7 | 0.9890 | 19.1 | 0.8018 | gap | 19.1 | pass |
| 3 | Lucky-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 32.7 | 0.9890 | 19.1 | 0.8018 | gap | 19.1 | pass |
| 4 | precomputed_baseline (test) | — | — | 30.0 | — | — | — | — | — | — |
| 5 | DiffIR2VR-Zero | 2025 | arXiv 2503.15984 (DIPLI) | 27.8 | 0.6200 | — | — | — | — | — |
| 6 | RVRT+ | 2025 | arXiv 2503.15984 (DIPLI) | 26.5 | 0.5200 | — | — | — | — | — |
| 7 | Drizzle | 2002 | Fruchter & Hook, PASP 2002 | 26.0 | — | — | — | — | — | — |
| 8 | Shift-and-add | 2000 | Lucky imaging baseline | 22.0 | — | — | — | — | — | — |

### 132. Solar EUV/X-ray Imaging (`solar_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 33.0 | — | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 31.1 | 0.9999 | 28.9 | 0.9813 | done | 28.9 | pass |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 31.1 | 0.9999 | 28.6 | 0.9803 | done | 28.6 | pass |
| 4 | SolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | 31.1 | 0.9999 | 28.6 | 0.9803 | done | 28.6 | pass |
| 5 | Pixon | 1991 | Pina & Puetter, PASP 1993 | 30.0 | — | — | — | — | — | — |
| 6 | precomputed_baseline (test) | — | — | 28.4 | — | — | — | — | — | — |
| 7 | Richardson-Lucy | 1972 | Richardson 1972 | 25.0 | — | — | — | — | — | — |

## Ultrafast Imaging

### 133. Compressed Ultrafast Photography (CUP) (`cup`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 29.5 | 0.9923 | 7.6 | 0.2752 | gap | 7.6 | low |
| 2 | E2E-CUP [proxy] (PWM) | — | Richardson 1972, JOSA | 29.5 | 0.9923 | 7.6 | 0.2752 | gap | 7.6 | low |
| 3 | PnP-BM3D | 2020 | Liu et al., Sensors 2022, PMC9571970 | 29.2 | 0.9200 | — | — | — | — | — |
| 4 | PnP-FFDNet | 2020 | Liu et al., Sensors 2022, PMC9571970 | 28.4 | 0.9100 | — | — | — | — | — |
| 5 | PnP-DnCNN | 2020 | Liu et al., Sensors 2022, PMC9571970 | 27.1 | 0.8800 | — | — | — | — | — |
| 6 | TwIST | 2007 | Liu et al., Sensors 2022, PMC9571970 | 24.7 | 0.7900 | — | — | — | — | — |
| 7 | Direct inverse (no regularization) | 2014 | Gao et al., Nature 2014 | 12.0 | 0.3000 | — | — | — | — | — |
| 8 | precomputed_baseline (test) | — | — | 8.5 | — | — | — | — | — | — |
| 9 | Direct inverse (1000x compression) | 2014 | Gao et al., Nature 2014 extreme compression | 8.0 | 0.2000 | — | — | — | — | — |

### 134. Pump-Probe Microscopy (`pump_probe`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | MCR-ALS | 2000 | Tauler, Chemom Intell Lab 1995 | 26.0 | — | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.3 | 0.9741 | 18.5 | 0.2958 | partial | 18.5 | pass |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.3 | 0.9741 | 18.1 | 0.2847 | partial | 18.1 | pass |
| 4 | PumpProbe-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 23.3 | 0.9741 | 18.1 | 0.2847 | partial | 18.1 | pass |
| 5 | SVD analysis | 2000 | SVD for transient spectra | 22.0 | — | — | — | — | — | — |
| 6 | precomputed_baseline (test) | — | — | 18.6 | — | — | — | — | — | — |
| 7 | Simple averaging | 2000 | Time-averaging baseline | 18.0 | 0.5000 | — | — | — | — | — |

### 135. Streak Camera Imaging (`streak_camera`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 36.7 | 0.9928 | 13.7 | 0.2872 | gap | 13.7 | low |
| 2 | StreakNet [proxy] (PWM) | — | Richardson 1972, JOSA | 36.7 | 0.9928 | 13.7 | 0.2872 | gap | 13.7 | low |
| 3 | precomputed_baseline (test) | — | — | 30.8 | — | — | — | — | — | — |
| 4 | PnP-BM3D (sim) | 2022 | Yuan et al., Sensors 2022, PMC9571970 | 29.2 | 0.9200 | — | — | — | — | — |
| 5 | PnP-FFDNet (sim) | 2022 | Yuan et al., Sensors 2022, PMC9571970 | 28.4 | 0.9100 | — | — | — | — | — |
| 6 | Temporal deconvolution | 2000 | Streak deconv baseline | 25.0 | — | — | — | — | — | — |
| 7 | Wiener deconvolution | 1949 | Wiener 1949 | 22.0 | — | — | — | — | — | — |

### 136. XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 25.1 | 0.9853 | 5.2 | -0.0234 | gap | 5.2 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 25.1 | 0.9853 | 4.6 | -0.0201 | gap | 4.6 | fail |
| 3 | SFX-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 25.1 | 0.9853 | 5.2 | -0.0234 | gap | 5.2 | low |
| 4 | cctbx.xfel | 2014 | Hattne et al., Nature Methods 2014 | 25.0 | — | — | — | — | — | — |
| 5 | precomputed_baseline (test) | — | — | 24.1 | — | — | — | — | — | — |
| 6 | CrystFEL | 2012 | White et al., JAC 2012 | 22.0 | — | — | — | — | — | — |

## Quantum Imaging

### 137. Entangled Photon Microscopy (`entangled_photon`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9872 | 17.2 | 0.8517 | gap | 17.2 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9872 | 15.5 | 0.7926 | gap | 15.5 | pass |
| 3 | QGI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9872 | 15.5 | 0.7926 | gap | 15.5 | pass |
| 4 | precomputed_baseline (test) | — | — | 31.8 | — | — | — | — | — | — |
| 5 | Compressed sensing QI | 2013 | Howland et al., PRA 2013 | 18.0 | — | — | — | — | — | — |
| 6 | Coincidence counting | 2002 | quantum imaging baseline | 15.0 | — | — | — | — | — | — |

### 138. Ghost Imaging (`ghost_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Orthogonal GI (2D-DCT) | 2025 | Nature Sci Rep, s41598-025-01283-w | 30.0 | — | — | — | — | — | — |
| 2 | DGI-Net | 2021 | DL ghost imaging | 28.0 | 0.8800 | — | — | — | — | — |
| 3 | Bio-inspired self-attention | 2025 | MDPI Biomimetics 11(1):53 | 24.5 | 0.8000 | — | — | — | — | — |
| 4 | CS-GI | 2013 | Katz et al., APL 2009 | 22.0 | 0.7000 | — | — | — | — | — |
| 5 | DeepGhost (autoencoder) | 2020 | Nature Sci Rep, s41598-020-68401-8 | 19.9 | 0.6000 | — | — | — | — | — |
| 6 | Differential GI | 2010 | Ferri et al., 2010 | 18.0 | 0.5000 | — | — | — | — | — |
| 7 | Correlation imaging | 2002 | Bennink et al., PRL 2002 | 15.0 | 0.4000 | — | — | — | — | — |
| 8 | Raw correlation (5% sampling) | 2002 | Bennink et al., PRL 2002 | 10.0 | 0.2500 | — | — | — | — | — |
| 9 | Correlation GI (natural, 128x128) | 2020 | Bian et al., Scientific Reports 2020, PMC7376173 | 9.5 | — | — | — | — | — | — |
| 10 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.3434 | 6.6 | 0.1157 | done | 6.6 | low |
| 11 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.3434 | 5.2 | 0.0826 | partial | 5.2 | low |
| 12 | GI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.3434 | 6.6 | 0.1157 | done | 6.6 | low |
| 13 | Traditional GI (3000 measurements) | 2021 | Kim et al., Optics Express 2021, PMID 34809299 | 7.2 | 0.2800 | — | — | — | — | — |
| 14 | precomputed_baseline (test) | — | — | 6.6 | — | — | — | — | — | — |

### 139. Quantum Illumination (`quantum_illumination`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.5 | 0.9382 | 14.0 | 0.6831 | partial | 14.0 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.5 | 0.9382 | 13.2 | 0.6497 | gap | 13.2 | low |
| 3 | QI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 23.5 | 0.9382 | 13.2 | 0.6497 | gap | 13.2 | low |
| 4 | precomputed_baseline (test) | — | — | 20.2 | — | — | — | — | — | — |
| 5 | Optimal receiver | 2008 | Lloyd, Science 2008 | 15.0 | — | — | — | — | — | — |
| 6 | Photon counting (classical) | 2000 | Classical baseline | 12.0 | — | — | — | — | — | — |

## Broader Experimental Science

### 140. Acoustic Emission Testing (AE) (`acoustic_emission`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CNN Beamformer (1 source) | 2023 | Sensors 2023, PMC10650508 | 39.4 | 0.9780 | — | — | — | — | — |
| 2 | CNN Beamformer (3 sources) | 2023 | Sensors 2023, PMC10650508 | 32.3 | 0.8120 | — | — | — | — | — |
| 3 | MUSIC localization | 1986 | Schmidt, IEEE TAP 1986 | 22.0 | — | — | — | — | — | — |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.0778 | 6.9 | 0.2848 | gap | 6.9 | low |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.0778 | 5.2 | 0.2140 | gap | 5.2 | low |
| 6 | DeepAE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.0778 | 5.2 | 0.2140 | gap | 5.2 | low |
| 7 | precomputed_baseline (test) | — | — | 20.2 | — | — | — | — | — | — |
| 8 | AIC picker | 2000 | Akaike, Ann Inst Stat Math 1974 | 20.0 | — | — | — | — | — | — |

### 141. Adaptive Optics (AO) Imaging (`adaptive_optics`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.9999 | 40.4 | 0.9993 | gap | 40.4 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.9999 | 41.2 | 0.9994 | gap | 41.2 | pass |
| 3 | Deep-AO [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.9999 | 41.2 | 0.9994 | gap | 41.2 | pass |
| 4 | precomputed_baseline (test) | — | — | 100.0 | — | — | — | — | — | — |
| 5 | cGAN wavefront | 2020 | Biomed Opt Express 2020 | 31.0 | 0.9000 | — | — | — | — | — |
| 6 | Phase diversity | 1982 | Gonsalves, Opt Eng 1982 | 26.0 | — | — | — | — | — | — |
| 7 | Shack-Hartmann WFS | 1971 | Shack & Platt, 1971 | 22.0 | — | — | — | — | — | — |

### 142. Bioluminescence Tomography (BLT) (`bioluminescence_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | L1-regularized BLT | 2010 | TV-BLT | 22.0 | 0.7500 | — | — | — | — | — |
| 2 | Diffusion-model inversion | 2005 | Wang et al., Opt Lett 2004 | 18.0 | 0.6000 | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.3 | 0.3531 | 9.9 | 0.3897 | partial | 9.9 | low |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.3 | 0.3531 | 8.3 | 0.3030 | partial | 8.3 | low |
| 5 | BLT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.3 | 0.3531 | 8.3 | 0.3030 | partial | 8.3 | low |
| 6 | precomputed_baseline (test) | — | — | 13.3 | — | — | — | — | — | — |
| 7 | Direct mapping | 2000 | Direct BLT mapping baseline | 12.0 | 0.4000 | — | — | — | — | — |

### 143. Full-Waveform Inversion (FWI) (`fwi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | FCNVMB | 2021 | Yang & Ma, JGR 2021 | 32.0 | 0.9500 | — | — | — | — | — |
| 2 | OpenFWI benchmark | 2022 | Deng et al., NeurIPS 2022 | 30.0 | 0.9400 | — | — | — | — | — |
| 3 | Conventional FWI (gradient descent) | 2009 | Virieux & Operto, Geophysics 2009 (estimated) | 28.4 | — | — | — | — | — | — |
| 4 | InversionNet | 2020 | Wu & Lin, JGR 2019 | 28.0 | 0.9000 | 7.4 | 0.5009 | gap | 7.4 | low |
| 5 | VelocityGAN | 2020 | Zhang & Alkhalifah, 2020 | 26.5 | 0.8800 | — | — | — | — | — |
| 6 | Adjoint-state FWI | 2006 | Virieux & Operto, Geophysics 2009 | 25.0 | 0.8500 | — | — | — | — | — |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 15.2 | 0.0692 | 6.0 | 0.4219 | partial | 6.0 | low |
| 8 | precomputed_baseline (test) | — | — | 12.4 | — | — | — | — | — | — |

### 144. Gravitational Wave Detection (`gravitational_wave`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.8766 | 27.2 | 0.2248 | gap | 27.2 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.8766 | 27.1 | 0.2245 | gap | 27.1 | pass |
| 3 | GW-DL (PyCBC-ML) [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.8766 | 27.1 | 0.2245 | gap | 27.1 | pass |
| 4 | precomputed_baseline (test) | — | — | 100.0 | — | — | — | — | — | — |
| 5 | BayesWave | 2015 | Cornish & Littenberg, CQG 2015 | 25.0 | — | — | — | — | — | — |
| 6 | cWaveNet | 2020 | Wei & Huerta, PLB 2020 | 22.0 | — | — | — | — | — | — |
| 7 | Matched filtering | 2000 | Allen et al., PRD 2012 | 20.0 | — | — | — | — | — | — |

### 145. Electrical Impedance Tomography (EIT) (`impedance_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SA-HFL | 2023 | CMPB 2023, S0169260723005278 | 31.0 | 0.9880 | — | — | — | — | — |
| 2 | EIDORS-Net | 2020 | DL for EIT | 26.0 | 0.8500 | — | — | — | — | — |
| 3 | TV-ADMM | 2010 | TV regularization | 22.0 | 0.7500 | — | — | — | — | — |
| 4 | Linear backprojection | 1990 | EIT backprojection (RS-FISTA=37.5 dB, extrapolated) | 22.0 | 0.4500 | — | — | — | — | — |
| 5 | Newton one-step | 2005 | Cheney et al., SIAM 1999 | 20.0 | 0.7000 | — | — | — | — | — |
| 6 | D-bar method | 2000 | Nachman, Annals Math 1996 | 18.0 | 0.6000 | — | — | — | — | — |
| 7 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 15.9 | 0.1854 | 12.9 | 0.4768 | partial | 12.9 | low |
| 8 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 15.9 | 0.1854 | 12.0 | 0.4311 | partial | 12.0 | low |
| 9 | EIT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 15.9 | 0.1854 | 12.0 | 0.4311 | partial | 12.0 | low |
| 10 | TPINV (Tikhonov Pseudoinverse) | 2023 | Ivanenko et al., Sensors 2023, PMC10538128 | 12.9 | — | — | — | — | — | — |
| 11 | precomputed_baseline (test) | — | — | 12.6 | — | — | — | — | — | — |
| 12 | LBP (Linear Back Projection) | 2023 | Ivanenko et al., Sensors 2023, PMC10538128 | 12.4 | — | — | — | — | — | — |

### 146. Magnetic Particle Imaging (MPI) (`magnetic_particle`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | VRF-Net (recon) | 2026 | Khair et al., BSPC 113, arXiv 2511.02212 | 41.6 | 0.9600 | — | — | — | — | — |
| 2 | SRCNN (MPI) | 2024 | SRCNN for MPI system matrix | 32.9 | 0.9890 | — | — | — | — | — |
| 3 | Hybrid encoder-decoder | 2025 | Phys Med Biol 2025, 10.1088/1361-6560/ae19c9 | 29.1 | 0.9300 | — | — | — | — | — |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 27.5 | 0.9676 | 22.9 | 0.9806 | partial | 22.9 | pass |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 27.5 | 0.9676 | 22.8 | 0.9794 | partial | 22.8 | pass |
| 6 | MPI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 27.5 | 0.9676 | 22.8 | 0.9794 | partial | 22.8 | pass |
| 7 | precomputed_baseline (test) | — | — | 26.5 | — | — | — | — | — | — |
| 8 | X-space approach | 2010 | Goodwill & Conolly, TMI 2010 | 26.0 | — | — | — | — | — | — |
| 9 | System matrix reconstruction | 2005 | Gleich & Weizenecker, Nature 2005 | 22.0 | — | — | — | — | — | — |

### 147. Ocean Acoustic Tomography (`ocean_acoustic_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 27.6 | 0.6889 | -32.8 | -0.0000 | fail | -32.8 | fail |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 27.6 | 0.6889 | -32.8 | -0.0000 | fail | -32.8 | fail |
| 3 | OAT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 27.6 | 0.6889 | -32.8 | -0.0000 | fail | -32.8 | fail |
| 4 | precomputed_baseline (test) | — | — | 26.6 | — | — | — | — | — | — |
| 5 | Matched-field processing | 1990 | Tolstoy, JASA 1993 | 22.0 | — | — | — | — | — | — |
| 6 | Travel-time inversion | 1979 | Munk & Wunsch, Deep-Sea Res 1979 | 20.0 | — | — | — | — | — | — |

### 148. Particle Calorimetry (`particle_calorimetry`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 37.7 | 0.9521 | 8.3 | 0.0053 | gap | 8.3 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 37.7 | 0.9521 | 8.2 | 0.0052 | gap | 8.2 | low |
| 3 | CaloDiffusion [proxy] (PWM) | — | Richardson 1972, JOSA | 37.7 | 0.9521 | 8.3 | 0.0053 | gap | 8.3 | low |
| 4 | precomputed_baseline (test) | — | — | 36.7 | — | — | — | — | — | — |
| 5 | Pandora PFA | 2014 | Marshall & Thomson, EPJC 2015 | 22.0 | — | — | — | — | — | — |
| 6 | Clustering algorithms | 2000 | CALICE collab. | 20.0 | — | — | — | — | — | — |

### 149. Radio Aperture Synthesis (`radio_astronomy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | POLISH | 2022 | MNRAS 2022 | 55.9 | 0.9980 | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 41.0 | 0.9426 | 1.5 | 0.0022 | gap | 1.5 | fail |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 41.0 | 0.9426 | 1.4 | 0.0021 | gap | 1.4 | fail |
| 4 | RadioAST-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 41.0 | 0.9426 | 1.5 | 0.0022 | gap | 1.5 | fail |
| 5 | precomputed_baseline (test) | — | — | 37.3 | — | — | — | — | — | — |
| 6 | U-Net denoising | 2021 | DL radio astronomy | 35.0 | — | — | — | — | — | — |
| 7 | CLEAN | 1974 | Hogbom, A&AS 1974 | 25.0 | — | — | — | — | — | — |

### 150. Seismic Tomography (`seismic_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | TSISTA-Net | 2025 | Applied Sciences 15(23):12700 | 37.3 | 0.9670 | — | — | — | — | — |
| 2 | PhaseNet-DAS | 2023 | Zhu et al., 2023 | 30.0 | 0.9200 | — | — | — | — | — |
| 3 | FWI | 2009 | Virieux & Operto, Geophysics 2009 | 28.0 | 0.8800 | — | — | — | — | — |
| 4 | Travel-time tomography | 1976 | Aki et al., JGR 1977 | 20.0 | 0.6500 | — | — | — | — | — |
| 5 | Simple ray tracing | 1976 | Aki et al., JGR 1977 | 12.0 | 0.4000 | — | — | — | — | — |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.4406 | 6.8 | 0.4649 | partial | 6.8 | low |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.4406 | 5.2 | 0.3727 | partial | 5.2 | low |
| 8 | SeisInversion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.4406 | 5.2 | 0.3727 | partial | 5.2 | low |
| 9 | precomputed_baseline (test) | — | — | 9.8 | — | — | — | — | — | — |

## Scientific Instrumentation

### 151. Atom Probe Tomography (APT) (`atom_probe`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 42.1 | 0.9999 | 9.5 | 0.5219 | gap | 9.5 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 42.1 | 0.9999 | 8.0 | 0.4372 | gap | 8.0 | low |
| 3 | APT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 42.1 | 0.9999 | 8.0 | 0.4372 | gap | 8.0 | low |
| 4 | precomputed_baseline (test) | — | — | 41.1 | — | — | — | — | — | — |
| 5 | ML trajectory correction | 2022 | DL for APT | 24.0 | — | — | — | — | — | — |
| 6 | Voltage reconstruction | 2000 | APT reconstruction | 20.0 | — | — | — | — | — | — |

### 152. Cathodoluminescence (CL) Imaging (`cathodoluminescence`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 38.7 | 0.9999 | 18.9 | 0.8857 | gap | 18.9 | pass |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 38.7 | 0.9999 | 17.6 | 0.8513 | gap | 17.6 | pass |
| 3 | CL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 38.7 | 0.9999 | 17.6 | 0.8513 | gap | 17.6 | pass |
| 4 | precomputed_baseline (test) | — | — | 28.9 | — | — | — | — | — | — |
| 5 | PCA denoising | 2010 | PCA for CL | 25.0 | — | — | — | — | — | — |
| 6 | Spectral unmixing | 2000 | NMF/VCA for CL | 22.0 | — | — | — | — | — | — |

### 153. Cryo-EM Single Particle Analysis (`cryo_em`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Topaz-Denoise | 2020 | Bepler et al., Nature Commun 2020 | 25.0 | — | — | — | — | — | — |
| 2 | DUAL (cryo-ET) | 2024 | PMC10942334, 2024 | 21.3 | 0.8240 | — | — | — | — | — |
| 3 | DRA (denoising-recon) | 2024 | arXiv 2410.11373 | 20.2 | 0.8700 | — | — | — | — | — |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.0400 | 12.9 | 0.8133 | partial | 12.9 | low |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.0400 | 12.9 | 0.8153 | partial | 12.9 | low |
| 6 | CryoDRGN [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.0400 | 12.9 | 0.8153 | partial | 12.9 | low |
| 7 | cryoSPARC | 2017 | Punjani et al., Nature Methods 2017 | 20.0 | — | — | — | — | — | — |
| 8 | precomputed_wiener (test) | — | — | 19.2 | — | — | — | — | — | — |
| 9 | rl_ctf_20iter (test) | — | — | 19.2 | — | — | — | — | — | — |
| 10 | RELION | 2012 | Scheres, JSB 2012 | 18.0 | — | — | — | — | — | — |

### 154. MALDI Mass Spectrometry Imaging (`maldi_msi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9957 | 11.7 | 0.7841 | gap | 11.7 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9957 | 9.9 | 0.7091 | gap | 9.9 | low |
| 3 | MSI-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9957 | 11.7 | 0.7841 | gap | 11.7 | low |
| 4 | precomputed_baseline (test) | — | — | 27.1 | — | — | — | — | — | — |
| 5 | NMF denoising | 2010 | NMF for MSI | 25.0 | — | — | — | — | — | — |
| 6 | Peak picking | 2000 | MALDI-MSI baseline | 22.0 | — | — | — | — | — | — |

### 155. Muon Tomography (`muon_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | EM-POCA [proxy] (PWM) | — | Richardson 1972, JOSA | 19.2 | 0.1257 | 7.1 | 0.1044 | gap | 7.1 | low |
| 2 | mu-Net (ConvNeXt U-Net) | 2023 | arXiv 2312.17265 | 17.1 | — | — | — | — | — | — |
| 3 | PoCA | 2003 | Borozdin et al., Nature 2003 | 13.7 | — | 7.1 | 0.1044 | partial | 7.1 | low |
| 4 | PoCA (1024 muons) | 2023 | mu-Net, arXiv 2312.17265 | 13.7 | — | — | — | — | — | — |
| 5 | FBP (muon tomography) (PWM) | — | — | 13.5 | — | 4.5 | 0.0000 | partial | 4.5 | fail |
| 6 | precomputed_baseline (test) | — | — | 13.5 | — | — | — | — | — | — |
| 7 | Simple FBP (low stats) | 2003 | Borozdin et al., Nature 2003 | 8.0 | — | — | — | — | — | — |

### 156. Neutron Diffraction (`neutron_diffraction`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Rietveld refinement | 1969 | Rietveld, JAC 1969 | 25.0 | — | — | — | — | — | — |
| 2 | Le Bail fitting | 1988 | Le Bail et al., 1988 | 22.0 | — | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 10.3 | 0.0334 | 2.5 | 0.1614 | partial | 2.5 | fail |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 10.3 | 0.0334 | 0.5 | 0.1041 | partial | 0.5 | fail |
| 5 | NeutronDiff-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 10.3 | 0.0334 | 2.5 | 0.1614 | partial | 2.5 | fail |
| 6 | precomputed_baseline (test) | — | — | 8.8 | — | — | — | — | — | — |

### 157. Neutron Radiography / Tomography (`neutron_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SIRT | 1972 | Gilbert 1972 | 28.0 | 0.8000 | — | — | — | — | — |
| 2 | FBP | 1971 | FBP baseline | 25.0 | 0.7000 | 0.7 | 0.0000 | gap | 0.7 | fail |
| 3 | NeuTomo-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.0792 | 9.1 | 0.3697 | done | 9.1 | low |
| 4 | GRIDREC-Neutron [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.0792 | 9.1 | 0.3697 | done | 9.1 | low |
| 5 | precomputed_baseline (test) | — | — | 6.6 | — | — | — | — | — | — |

### 158. Proton Radiography (`proton_radiography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | CNN proton portal imaging | 2024 | PMC11682722 | 39.1 | 0.9870 | — | — | — | — | — |
| 2 | cGAN synthetic CT | 2023 | PubMed 37800874 | 29.0 | 0.9520 | — | — | — | — | — |
| 3 | DROP-TVS | 2013 | Penfold et al., Med Phys 2010 | 28.0 | — | — | — | — | — | — |
| 4 | FBP (straight-line approx) | 2003 | Schulte et al., Med Phys 2005 | 25.0 | — | — | — | — | — | — |
| 5 | MLP (Most Likely Path) | 2004 | Schulte et al., Med Phys 2008 | 22.0 | — | — | — | — | — | — |
| 6 | ProtonRecon-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.3715 | 8.0 | 0.2592 | partial | 8.0 | low |
| 7 | FBP-Proton [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.3715 | 8.0 | 0.2592 | partial | 8.0 | low |
| 8 | precomputed_baseline (test) | — | — | 12.0 | — | — | — | — | — | — |

### 159. Small-Angle X-ray Scattering (SAXS) (`saxs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | McSAS | 2013 | Bressler et al., JAC 2015 | 25.0 | — | — | — | — | — | — |
| 2 | Guinier analysis | 1939 | Guinier, 1939 | 20.0 | — | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 10.1 | 0.0611 | 3.4 | -0.0319 | partial | 3.4 | fail |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 10.1 | 0.0611 | 3.2 | -0.0265 | partial | 3.2 | fail |
| 5 | SAXS-VAE [proxy] (PWM) | — | Richardson 1972, JOSA | 10.1 | 0.0611 | 3.2 | -0.0265 | partial | 3.2 | fail |
| 6 | precomputed_baseline (test) | — | — | 9.0 | — | — | — | — | — | — |

### 160. Wide-Angle X-ray Scattering (WAXS) (`waxs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3264 | 2.0 | -0.0553 | gap | 2.0 | fail |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3264 | 1.9 | -0.0448 | gap | 1.9 | fail |
| 3 | WAXS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3264 | 1.9 | -0.0448 | gap | 1.9 | fail |
| 4 | Rietveld refinement | 1969 | Rietveld, JAC 1969 | 24.0 | — | — | — | — | — | — |
| 5 | precomputed_baseline (test) | — | — | 23.4 | — | — | — | — | — | — |
| 6 | Background subtraction | 2000 | WAXS baseline processing | 20.0 | 0.6500 | — | — | — | — | — |

### 161. X-ray Crystallography (`xray_crystallography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SHELXD | 2010 | Sheldrick, Acta Cryst 2008 | 28.0 | — | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.4 | 0.0751 | 3.3 | -0.0461 | gap | 3.3 | fail |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.4 | 0.0751 | 3.1 | -0.0395 | gap | 3.1 | fail |
| 4 | AlphaFold-SF [proxy] (PWM) | — | Richardson 1972, JOSA | 23.4 | 0.0751 | 3.3 | -0.0461 | gap | 3.3 | fail |
| 5 | precomputed_baseline (test) | — | — | 22.4 | — | — | — | — | — | — |
| 6 | Direct methods | 1953 | Hauptman & Karle, 1953 | 22.0 | — | — | — | — | — | — |

### 162. X-ray Fluorescence Tomography (`xrf_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | 1D-CNN + U-Net | 2025 | Nature Sci Reports, s41598-025-03900-0 | 39.1 | 0.9790 | — | — | — | — | — |
| 2 | Optimized SCUNet | 2024 | MDPI J Imaging 10(6):127 | 39.0 | 0.8600 | — | — | — | — | — |
| 3 | SIRT | 1972 | Gilbert 1972 | 26.0 | — | — | — | — | — | — |
| 4 | FBP reconstruction | 2000 | Sci Rep 2025 (U-Net=39.1, FBP estimated) | 25.0 | 0.5500 | — | — | — | — | — |
| 5 | FBP | 1971 | FBP baseline | 22.0 | — | — | — | — | — | — |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 16.6 | 0.8531 | -45.3 | 0.0000 | fail | -45.3 | fail |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 16.6 | 0.8531 | -45.3 | 0.0000 | fail | -45.3 | fail |
| 8 | XRFT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 16.6 | 0.8531 | -45.3 | 0.0000 | fail | -45.3 | fail |
| 9 | precomputed_baseline (test) | — | — | 15.6 | — | — | — | — | — | — |

## Multi-Modal Fusion

### 163. Correlative Light-Electron Microscopy (CLEM) (`clem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 39.7 | 0.9999 | 10.2 | 0.6706 | gap | 10.2 | low |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 39.7 | 0.9999 | 8.5 | 0.5782 | gap | 8.5 | low |
| 3 | CLEM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 39.7 | 0.9999 | 8.5 | 0.5782 | gap | 8.5 | low |
| 4 | precomputed_baseline (test) | — | — | 28.1 | — | — | — | — | — | — |
| 5 | VoxelMorph registration | 2019 | Balakrishnan et al., TMI 2019 | 26.0 | 0.8300 | — | — | — | — | — |
| 6 | Landmark registration | 2000 | CLEM registration | 22.0 | — | — | — | — | — | — |

### 164. CT + Fluorescence (FLIT) (`ct_fluorescence`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | SIRT | 1972 | Gilbert 1972 | 25.0 | 0.7500 | — | — | — | — | — |
| 2 | FBP + fluorescence | 2000 | XFCT baseline | 22.0 | — | — | — | — | — | — |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.6723 | 8.3 | 0.4463 | done | 8.3 | low |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.6723 | 7.0 | 0.3733 | partial | 7.0 | low |
| 5 | XFCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.6723 | 7.0 | 0.3733 | partial | 7.0 | low |
| 6 | precomputed_baseline (test) | — | — | 10.2 | — | — | — | — | — | — |

### 165. PET/CT Fusion (`pet_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Attention U-Net + diffusion | 2025 | arXiv 2504.00816 | 35.9 | 0.9920 | — | — | — | — | — |
| 2 | TrUNET-MAPEM | 2023 | ScienceDirect, S0895611123001337 | 33.7 | 0.9550 | — | — | — | — | — |
| 3 | OSEM + CT AC | 2000 | PET/CT baseline | 28.0 | 0.8000 | — | — | — | — | — |
| 4 | MLEM | 1982 | Shepp & Vardi, TMI 1982 | 25.0 | 0.7500 | — | — | — | — | — |
| 5 | MLEM (low-count, 2 iter) | 1982 | Shepp & Vardi 1982 | 15.0 | 0.5000 | — | — | — | — | — |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.0 | 0.0756 | -40.2 | 0.0000 | fail | -40.2 | fail |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.0 | 0.0756 | 7.4 | 0.0000 | partial | 7.4 | low |
| 8 | PET-CT-Fusion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.0 | 0.0756 | -40.2 | 0.0000 | fail | -40.2 | fail |
| 9 | precomputed_baseline (test) | — | — | 13.0 | — | — | — | — | — | — |

### 166. PET/MR Fusion (`pet_mr`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | Brain DL PET/MR | 2024 | PubMed 2024 | 42.0 | 0.9650 | — | — | — | — | — |
| 2 | MRAC-based reconstruction | 2010 | Wagenknecht et al., 2013 | 26.0 | 0.7800 | — | — | — | — | — |
| 3 | No-AC reconstruction | 2010 | PET/MR no attenuation correction | 15.0 | 0.5000 | — | — | — | — | — |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.5 | 0.2076 | -28.2 | 0.0003 | fail | -28.2 | fail |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.5 | 0.2076 | -28.2 | 0.0003 | fail | -28.2 | fail |
| 6 | PET-MR-DeepJoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.5 | 0.2076 | -28.2 | 0.0003 | fail | -28.2 | fail |
| 7 | No-AC (1/10 counts) | 2010 | Catana et al., JNM 2010 | 13.0 | 0.4000 | — | — | — | — | — |
| 8 | precomputed_baseline (test) | — | — | 12.5 | — | — | — | — | — | — |

### 167. SPECT/CT Fusion (`spect_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | GAN projection-space denoising | 2022 | PMC8940834 | 42.5 | 0.9900 | — | — | — | — | — |
| 2 | U2-Net (bone SPECT/CT) | 2022 | PMC9192886 | 40.8 | 0.7880 | — | — | — | — | — |
| 3 | OSEM + CT AC | 2000 | SPECT/CT baseline | 26.0 | 0.7800 | — | — | — | — | — |
| 4 | MLEM | 1982 | Shepp & Vardi, TMI 1982 | 24.0 | 0.7400 | — | — | — | — | — |
| 5 | MLEM (low-count, 2 iter) | 1982 | Shepp & Vardi 1982 | 15.0 | 0.5000 | — | — | — | — | — |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.3684 | 2.7 | 0.0000 | gap | 2.7 | fail |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.3684 | 2.7 | 0.0000 | gap | 2.7 | fail |
| 8 | SPECT-CT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.3684 | 2.7 | 0.0000 | gap | 2.7 | fail |
| 9 | MLEM (1 iter, 1/20 counts) | 1982 | Reader et al., PMB 2007 / Shepp-Vardi 1982 | 13.0 | 0.3500 | — | — | — | — | — |
| 10 | precomputed_baseline (test) | — | — | 11.4 | — | — | — | — | — | — |

### 168. US/MRI Fusion (`us_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|
| 1 | VoxelMorph | 2019 | Balakrishnan et al., TMI 2019 | 30.0 | 0.9000 | — | — | — | — | — |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 28.3 | 0.9765 | -36.7 | 0.0000 | fail | -36.7 | fail |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 28.3 | 0.9765 | -37.4 | 0.0000 | fail | -37.4 | fail |
| 4 | US-MRI-Net [proxy] (PWM) | — | — | 28.3 | 0.9765 | 13.2 | -0.2497 | gap | 13.2 | low |
| 5 | precomputed_baseline (test) | — | — | 25.5 | — | — | — | — | — | — |
| 6 | B-spline FFD | 2003 | Rueckert et al., TMI 1999 | 25.0 | 0.8000 | — | — | — | — | — |
| 7 | Demons registration | 1998 | Thirion, MIA 1998 | 22.0 | 0.7500 | — | — | — | — | — |
| 8 | Affine registration | 2000 | Affine US/MRI baseline (estimated) | 21.0 | 0.6000 | — | — | — | — | — |

---

*PWM Benchmark Algorithm State — 168 modalities, 1294 algorithms (65 done) — Generated 2026-03-15*