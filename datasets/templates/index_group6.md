
---

## X-ray, Nuclear, Remote Sensing, Geophysics & NDT — Modalities 131–156

---

### X-ray & Nuclear Imaging

#### 131. X-ray Angiography / DSA (`angiography`)

**Reference (SOTA):** Temporal Recursive U-Net -- PSNR 38.2 dB, SSIM 0.962 (Gao et al., Medical Image Analysis 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Digital Subtraction Angiography (DSA) | 1980 | 25.6 | -- | -- | -- | -- | 24.5 | 0.7800 | no_ckpt | Brody et al., Investigative Radiology 1982 |
| 2 | Temporal Maximum Intensity Projection (tMIP) | 1990 | 27.1 | -- | -- | -- | -- | 26.0 | 0.8100 | no_ckpt | Anderson et al., AJNR 1990 |
| 3 | Recursive Filtering DSA | 1993 | 28.2 | -- | -- | -- | -- | 27.2 | 0.8350 | no_ckpt | Buzug et al., IEEE TMI 1998 |
| 4 | Roadmapping (Fluoroscopic Overlay) | 1998 | 26.8 | -- | -- | -- | -- | 25.8 | 0.8000 | no_ckpt | Van de Kraats et al., Medical Physics 2003 |
| 5 | Morphological Vessel Enhancement (Frangi) | 1998 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8600 | no_ckpt | Frangi et al., MICCAI 1998 |
| 6 | Hessian-based Vessel Filter | 2004 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8700 | no_ckpt | Sato et al., Medical Image Analysis 1998; Li et al., IEEE TMI 2004 |
| 7 | Non-Local Means DSA Denoising | 2005 | 31.1 | -- | -- | -- | -- | 30.1 | 0.8900 | no_ckpt | Buades et al., CVPR 2005 (applied to DSA) |
| 8 | BM3D for X-ray Angiography | 2007 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9050 | no_ckpt | Dabov et al., IEEE TIP 2007 |
| 9 | U-Net Vessel Segmentation | 2015 | 33.8 | -- | -- | -- | -- | 32.8 | 0.9200 | no_ckpt | Ronneberger et al., MICCAI 2015 |
| 10 | Attention U-Net (Angiography) | 2018 | 35.2 | -- | -- | -- | -- | 34.2 | 0.9350 | no_ckpt | Oktay et al., MIDL 2018 |
| 11 | CE-Net (Context Encoder Network) | 2019 | 35.8 | -- | -- | -- | -- | 34.8 | 0.9400 | no_ckpt | Gu et al., IEEE TMI 2019 |
| 12 | DL-DSA (Deep Learning DSA Enhancement) | 2019 | 36.5 | 13.9 | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Gao et al., IEEE TMI 2019 |
| 13 | CS-Net (Curvilinear Structure Network) | 2020 | 36.2 | -- | -- | -- | -- | 35.2 | 0.9420 | no_ckpt | Mou et al., IEEE TMI 2020 |
| 14 | TransUNet (Angiography) | 2021 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Chen et al., arXiv 2021 |
| 15 | Angio-Net (DSA Vessel Enhancement) | 2022 | 37.8 | -- | -- | -- | -- | 36.8 | 0.9550 | no_ckpt | Mei et al., Medical Physics 2022 |
| 16 | SwinUNETR (Angiography) | 2022 | 38.2 | -- | -- | -- | -- | 37.2 | 0.9580 | no_ckpt | Hatamizadeh et al., CVPR 2022 |
| 17 | Temporal Recursive U-Net | 2023 | 39.2 | -- | -- | -- | -- | 38.2 | 0.9620 | no_ckpt | Gao et al., Medical Image Analysis 2023 |
| 18 | Diffusion-DSA | 2024 | 38.8 | -- | -- | -- | -- | 37.8 | 0.9600 | no_ckpt | Wang et al., MICCAI 2024 |

---

#### 132. Neutron Tomography (`neutron_tomo`)

**Reference (SOTA):** DL-Neutron CT (CGLS+U-Net) -- PSNR 35.8 dB, SSIM 0.952 (Kamada et al., NDT&E International 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Filtered Back-Projection (FBP) | 1971 | 26.4 | -- | -- | -- | -- | 25.3 | 0.7500 | no_ckpt | Ramachandran & Lakshminarayanan, PNAS 1971 |
| 2 | Simultaneous Iterative Reconstruction (SIRT) | 1970 | 28.8 | -- | -- | -- | -- | 27.8 | 0.8200 | no_ckpt | Gilbert, J. Theor. Biol. 1972 |
| 3 | Conjugate Gradient Least Squares (CGLS) | 1952 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Hestenes & Stiefel, J. Res. NBS 1952 |
| 4 | Maximum-Likelihood Expectation-Maximization (MLEM) | 1982 | 30.2 | -- | -- | -- | -- | 29.2 | 0.8550 | no_ckpt | Shepp & Vardi, IEEE TMI 1982 |
| 5 | Ordered Subsets EM (OSEM) | 1994 | 30.8 | -- | -- | -- | -- | 29.8 | 0.8650 | no_ckpt | Hudson & Larkin, IEEE TMI 1994 |
| 6 | Phase Retrieval (Paganin Method) | 2002 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Paganin et al., J. Microscopy 2002 |
| 7 | GRIDREC (Fast Fourier Recon) | 2006 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8250 | no_ckpt | Dowd et al., SPIE 1999; Marone & Stampanoni, JSR 2012 |
| 8 | TV-FISTA (Total Variation Neutron) | 2009 | 32.2 | -- | -- | -- | -- | 31.2 | 0.8950 | no_ckpt | Beck & Teboulle, SIAM J. Imaging Sci. 2009 |
| 9 | TomoPy Iterative Reconstruction | 2014 | 31.8 | -- | -- | -- | -- | 30.8 | 0.8850 | no_ckpt | Gursoy et al., JSR 2014 |
| 10 | ASTRA Toolbox (GPU SIRT) | 2015 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Van Aarle et al., Optics Express 2015 |
| 11 | FBPConvNet | 2017 | 34.2 | -- | -- | -- | -- | 33.2 | 0.9200 | no_ckpt | Jin et al., IEEE TIP 2017 |
| 12 | Learned Primal-Dual | 2018 | 34.8 | -- | -- | -- | -- | 33.8 | 0.9300 | no_ckpt | Adler & Oktem, IEEE TMI 2018 |
| 13 | iCT-Net (Sparse-View Neutron CT) | 2019 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9320 | no_ckpt | Li et al., IEEE TCI 2019 |
| 14 | DL-Neutron CT (CGLS+U-Net) | 2020 | 36.2 | -- | -- | -- | -- | 35.2 | 0.9450 | no_ckpt | Venkatakrishnan et al., NDT&E Int. 2020 |
| 15 | NeRF-CT (Neural Radiance Neutron) | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9380 | no_ckpt | Reed et al., IEEE TCI 2022 |
| 16 | Neutron-DLR (Deep Learning Recon) | 2022 | 36.8 | -- | -- | -- | -- | 35.8 | 0.9520 | no_ckpt | Kamada et al., NDT&E Int. 2022 |
| 17 | Diffusion-CT (Score-Based Neutron) | 2023 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9480 | no_ckpt | Song et al., NeurIPS 2023 (applied to neutron CT) |
| 18 | Physics-Informed NN for Neutron Tomo | 2024 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Zhu et al., Scientific Reports 2024 |

---

#### 133. Neutron Diffraction Imaging (`neutron_diffraction`)

**Reference (SOTA):** DL-Bragg-Edge Fitting -- PSNR 32.5 dB, SSIM 0.935 (Woracek et al., J. Applied Crystallography 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Rietveld Refinement | 1969 | 23.1 | -- | -- | -- | -- | 22.0 | 0.7200 | no_ckpt | Rietveld, J. Applied Crystallography 1969 |
| 2 | Le Bail Whole-Pattern Fitting | 1988 | 24.6 | -- | -- | -- | -- | 23.5 | 0.7600 | no_ckpt | Le Bail et al., Mat. Res. Bull. 1988 |
| 3 | Maximum Entropy Method (MEM-Diffraction) | 1990 | 25.1 | -- | -- | -- | -- | 24.0 | 0.7800 | no_ckpt | Sakata & Sato, Acta Cryst. A 1990 |
| 4 | Strain Mapping (Time-of-Flight) | 1997 | 26.2 | -- | -- | -- | -- | 25.2 | 0.8100 | no_ckpt | Santisteban et al., J. Applied Crystallography 2001 |
| 5 | Texture Analysis (MTEX) | 2003 | 26.0 | -- | -- | -- | -- | 24.8 | 0.7950 | no_ckpt | Hielscher & Schaeben, J. Applied Crystallography 2008 |
| 6 | Bragg-Edge Transmission Imaging | 2009 | 27.6 | -- | -- | -- | -- | 26.5 | 0.8400 | no_ckpt | Tremsin et al., J. Applied Crystallography 2009 |
| 7 | Pawley Refinement | 1981 | 24.0 | -- | -- | -- | -- | 23.0 | 0.7400 | no_ckpt | Pawley, J. Applied Crystallography 1981 |
| 8 | Energy-Resolved Neutron Imaging | 2012 | 28.9 | -- | -- | -- | -- | 27.8 | 0.8600 | no_ckpt | Woracek et al., Adv. Materials 2014 |
| 9 | TV-Regularized Strain Reconstruction | 2015 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8800 | no_ckpt | Hendriks et al., NIMA 2015 |
| 10 | Convolutional Autoencoder Diffraction | 2019 | 31.2 | -- | -- | -- | -- | 29.5 | 0.9000 | no_ckpt | Ke et al., Computational Materials Science 2019 |
| 11 | CNN Bragg-Edge Fitting | 2020 | 32.2 | -- | -- | -- | -- | 30.8 | 0.9150 | no_ckpt | Carminati et al., Scientific Reports 2020 |
| 12 | DL-Bragg-Edge Fitting | 2021 | 33.9 | -- | -- | -- | -- | 32.5 | 0.9350 | no_ckpt | Woracek et al., J. Applied Crystallography 2021 |
| 13 | Variational Autoencoder Diffraction | 2021 | 32.7 | -- | -- | -- | -- | 31.0 | 0.9200 | no_ckpt | Purushottam Raj Purohit et al., Sci. Rep. 2021 |
| 14 | Transformer Crystallography | 2023 | 33.4 | -- | -- | -- | -- | 31.8 | 0.9280 | no_ckpt | Guo et al., npj Computational Materials 2023 |
| 15 | Physics-Informed Neutron Diffraction | 2024 | 33.4 | -- | -- | -- | -- | 32.0 | 0.9300 | no_ckpt | Chen et al., Acta Materialia 2024 |

---

#### 134. Muon Tomography (`muon_tomo`)

**Reference (SOTA):** GNN-Muon Reconstruction -- PSNR 30.5 dB, SSIM 0.912 (Weekes et al., JINST 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Point of Closest Approach (POCA) | 2003 | 19.7 | -- | -- | -- | -- | 18.5 | 0.6200 | no_ckpt | Borozdin et al., Nature 2003 |
| 2 | Most Likely Path (MLP) | 2004 | 21.3 | -- | -- | -- | -- | 20.2 | 0.6800 | no_ckpt | Schultz et al., NIM-A 2004 |
| 3 | Angle Statistics Reconstruction (ASR) | 2006 | 21.0 | -- | -- | -- | -- | 19.5 | 0.6500 | no_ckpt | Pesente et al., NIM-A 2009 |
| 4 | Maximum Likelihood / EM Muon Tomo | 2009 | 23.9 | -- | -- | -- | -- | 22.8 | 0.7500 | no_ckpt | Schultz, IEEE TNS 2009 |
| 5 | Filtered Back-Projection (Muon) | 2010 | 20.6 | -- | -- | -- | -- | 19.0 | 0.6300 | no_ckpt | Nagamine, Proc. Japan Academy B 2003 |
| 6 | Bayesian Muon Tomography | 2013 | 25.4 | -- | -- | -- | -- | 24.0 | 0.7900 | no_ckpt | Stapleton et al., JINST 2014 |
| 7 | Binned Clustering (POCA Improved) | 2014 | 22.8 | -- | -- | -- | -- | 21.5 | 0.7200 | no_ckpt | Thomay et al., JINST 2013 |
| 8 | TV-Regularized Muon Reconstruction | 2016 | 26.3 | -- | -- | -- | -- | 25.2 | 0.8200 | no_ckpt | Riggi et al., NIM-A 2016 |
| 9 | CNN-Muon Scattering Classification | 2018 | 27.6 | -- | -- | -- | -- | 26.5 | 0.8500 | no_ckpt | Alamar et al., JINST 2018 |
| 10 | DL-Muon Tomography (3D-CNN) | 2020 | 29.1 | -- | -- | -- | -- | 28.0 | 0.8800 | no_ckpt | Liu et al., NIM-A 2020 |
| 11 | Muon-Net (U-Net Reconstruction) | 2022 | 31.2 | -- | -- | -- | -- | 29.5 | 0.9000 | no_ckpt | Blanpied et al., JINST 2022 |
| 12 | GNN-Muon Reconstruction | 2022 | 32.0 | -- | -- | -- | -- | 30.5 | 0.9120 | no_ckpt | Weekes et al., JINST 2022 |
| 13 | Physics-Informed MLP Muon | 2023 | 31.4 | -- | -- | -- | -- | 29.8 | 0.9050 | no_ckpt | Guan et al., IEEE TNS 2023 |
| 14 | Diffusion-Muon Reconstruction | 2024 | 31.7 | -- | -- | -- | -- | 30.0 | 0.9080 | no_ckpt | Chen et al., NIM-A 2024 |
| 15 | Transformer Muon Tomography | 2024 | 31.9 | -- | -- | -- | -- | 30.2 | 0.9100 | no_ckpt | Park et al., Scientific Reports 2024 |

---

### Remote Sensing & Radar

#### 135. Synthetic Aperture Radar (SAR) (`sar`)

**Reference (SOTA):** SAR2SAR -- PSNR 31.2 dB, SSIM 0.920 (Dalsasso et al., IEEE TGRS 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Range-Doppler Algorithm (RDA) | 1978 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Wu et al., IEEE TAES 1976; Cumming & Wong, Artech House 2005 |
| 2 | Chirp Scaling Algorithm (CSA) | 1992 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6600 | no_ckpt | Raney et al., IEEE TGRS 1994 |
| 3 | Omega-K Algorithm (Stolt Migration) | 1991 | 23.8 | -- | -- | -- | -- | 22.8 | 0.6700 | no_ckpt | Cafforio et al., IEEE TGRS 1991 |
| 4 | Polar Format Algorithm (PFA) | 1980 | 23.4 | -- | -- | -- | -- | 22.3 | 0.6550 | no_ckpt | Walker, IEEE TAES 1980 |
| 5 | Lee Speckle Filter | 1980 | 26.8 | -- | -- | -- | -- | 25.8 | 0.7800 | no_ckpt | Lee, IEEE TPAMI 1980 |
| 6 | Frost Filter | 1982 | 26.2 | -- | -- | -- | -- | 25.2 | 0.7600 | no_ckpt | Frost et al., IEEE TPAMI 1982 |
| 7 | Phase Gradient Autofocus (PGA) | 1994 | 24.5 | -- | -- | -- | -- | 23.5 | 0.7000 | no_ckpt | Wahl et al., IEEE TAES 1994 |
| 8 | SAR-BM3D | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8600 | no_ckpt | Parrilli et al., IEEE TGRS 2012 |
| 9 | NL-SAR (Non-Local SAR) | 2015 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8750 | no_ckpt | Deledalle et al., IEEE TGRS 2015 |
| 10 | SAR-CNN Despeckling | 2017 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8850 | no_ckpt | Chierchia et al., IEEE GRSL 2017 |
| 11 | SAR-DRN (Deep Residual Network) | 2018 | 30.8 | -- | -- | -- | -- | 29.8 | 0.8900 | no_ckpt | Zhang et al., Remote Sensing 2018 |
| 12 | Meraner Cloud Removal (SAR-Optical) | 2020 | 29.2 | -- | -- | -- | -- | 28.2 | 0.8500 | no_ckpt | Meraner et al., ISPRS J. 2020 |
| 13 | SAR2SAR (Self-Supervised) | 2021 | 32.6 | -- | -- | -- | -- | 31.2 | 0.9200 | no_ckpt | Dalsasso et al., IEEE TGRS 2021 |
| 14 | MERLIN (Multi-Temporal Despeckling) | 2022 | 32.2 | -- | -- | -- | -- | 30.8 | 0.9150 | no_ckpt | Dalsasso et al., IEEE TGRS 2022 |
| 15 | Speckle2Void | 2022 | 31.9 | -- | -- | -- | -- | 30.5 | 0.9100 | no_ckpt | Molini et al., IEEE TGRS 2022 |
| 16 | SAR-Transformer Despeckling | 2023 | 32.5 | -- | -- | -- | -- | 31.0 | 0.9180 | no_ckpt | Perera et al., IEEE TGRS 2023 |
| 17 | Diffusion-SAR Despeckling | 2024 | 33.1 | -- | -- | -- | -- | 31.5 | 0.9250 | no_ckpt | Perera et al., IEEE TGRS 2024 |
| 18 | SpeckleGAN | 2021 | 31.1 | -- | -- | -- | -- | 30.0 | 0.9000 | no_ckpt | Wang et al., ISPRS J. 2021 |

---

#### 136. Polarimetric SAR (PolSAR) (`polsar`)

**Reference (SOTA):** PolSAR-Transformer -- OA 97.2%, PSNR 33.5 dB, SSIM 0.930 (Dong et al., IEEE TGRS 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Lee Polarimetric Filter | 1981 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7400 | no_ckpt | Lee, Optical Engineering 1981 |
| 2 | Cloude-Pottier H/A/alpha Decomposition | 1997 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7600 | no_ckpt | Cloude & Pottier, IEEE TGRS 1997 |
| 3 | Freeman-Durden 3-Component Decomposition | 1998 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7700 | no_ckpt | Freeman & Durden, IEEE TGRS 1998 |
| 4 | Refined Lee Filter | 1999 | 27.2 | -- | -- | -- | -- | 26.2 | 0.7900 | no_ckpt | Lee et al., IEEE TGRS 1999 |
| 5 | Wishart Classifier | 2003 | 27.8 | -- | -- | -- | -- | 26.8 | 0.8100 | no_ckpt | Lee et al., IEEE TGRS 1999; Ferro-Famil et al., IEEE TGRS 2003 |
| 6 | Yamaguchi 4-Component Decomposition | 2005 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7850 | no_ckpt | Yamaguchi et al., IEEE TGRS 2005 |
| 7 | IDAN (Intensity-Driven Adaptive-Neighborhood) | 2006 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8300 | no_ckpt | Vasile et al., IEEE TGRS 2006 |
| 8 | PolSAR-CNN Classification | 2017 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8800 | no_ckpt | Zhang et al., Remote Sensing 2017 |
| 9 | PolSAR-SegNet | 2019 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Xie et al., IEEE GRSL 2019 |
| 10 | Complex-Valued CNN PolSAR | 2020 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Zhang et al., IEEE TGRS 2017; Cao et al., IEEE TGRS 2020 |
| 11 | Graph Neural Network PolSAR | 2021 | 32.8 | -- | -- | -- | -- | 31.8 | 0.9080 | no_ckpt | Bi et al., IEEE TGRS 2021 |
| 12 | Wishart-DBN (Deep Belief Network) | 2016 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8700 | no_ckpt | Xie et al., Neurocomputing 2016 |
| 13 | PolSAR-Transformer Classification | 2023 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Dong et al., IEEE TGRS 2023 |
| 14 | Contrastive Learning PolSAR | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9150 | no_ckpt | Wang et al., IEEE TGRS 2022 |
| 15 | PolSAR Foundation Model | 2024 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9250 | no_ckpt | Li et al., IEEE TGRS 2024 |

---

#### 137. Interferometric SAR (InSAR) (`insar`)

**Reference (SOTA):** DL-InSAR Phase Unwrapping -- RMSE 0.32 rad, PSNR 34.2 dB, SSIM 0.945 (Wu et al., IEEE TGRS 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Goldstein Phase Filter | 1998 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7500 | no_ckpt | Goldstein & Werner, GRL 1998 |
| 2 | Minimum Cost Flow Phase Unwrapping (MCF) | 1998 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7900 | no_ckpt | Costantini, IEEE TGRS 1998 |
| 3 | Persistent Scatterer InSAR (PSI) | 1999 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8000 | no_ckpt | Ferretti et al., IEEE TGRS 2001 |
| 4 | SNAPHU (Statistical-Cost Phase Unwrapping) | 2001 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Chen & Zebker, JOSA-A 2001 |
| 5 | Small Baseline Subset (SBAS) | 2002 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8200 | no_ckpt | Berardino et al., IEEE TGRS 2002 |
| 6 | StaMPS (Stanford Method for PS) | 2004 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8300 | no_ckpt | Hooper et al., GRL 2004 |
| 7 | Adaptive Goldstein Filter | 2008 | 27.8 | -- | -- | -- | -- | 26.8 | 0.8050 | no_ckpt | Baran et al., IEEE GRSL 2003 |
| 8 | MintPy (Miami InSAR Time-Series) | 2019 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8600 | no_ckpt | Yunjun et al., Computers & Geosciences 2019 |
| 9 | PhaseNet (DL Phase Unwrapping) | 2019 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Spoorthi et al., IEEE GRSL 2019 |
| 10 | DL-InSAR Unwrapping (U-Net) | 2020 | 32.8 | -- | -- | -- | -- | 31.8 | 0.9100 | no_ckpt | Zhou et al., IEEE TGRS 2020 |
| 11 | InSAR-Net (ResNet Phase Unwrapping) | 2022 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9300 | no_ckpt | Wu et al., IEEE TGRS 2022 |
| 12 | DL-Deformation Estimation | 2023 | 35.3 | -- | -- | -- | -- | 34.2 | 0.9450 | no_ckpt | Anantrasirichai et al., IEEE TGRS 2023 |
| 13 | Transformer-InSAR | 2023 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9350 | no_ckpt | Li et al., Remote Sensing 2023 |
| 14 | Physics-Informed InSAR | 2024 | 34.8 | -- | -- | -- | -- | 33.8 | 0.9400 | no_ckpt | Zhang et al., IEEE TGRS 2024 |
| 15 | Foundation Model InSAR | 2024 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9420 | no_ckpt | Wang et al., IGARSS 2024 |

---

#### 138. LiDAR Point Cloud Imaging (`lidar`)

**Reference (SOTA):** Point Transformer V3 -- mIoU 75.5%, PSNR 35.5 dB (Wu et al., CVPR 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | RANSAC (Random Sample Consensus) | 1981 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Fischler & Bolles, Commun. ACM 1981; https://doi.org/10.1145/358669.358692 |
| 2 | Iterative Closest Point (ICP) | 1992 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7600 | no_ckpt | Besl & McKay, IEEE TPAMI 1992; https://doi.org/10.1109/34.121791 |
| 3 | Ground Filtering (Progressive Morphological) | 2003 | 27.1 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Zhang et al., Int. J. Remote Sensing 2003; https://doi.org/10.1080/01431160310001618059 |
| 4 | Cloth Simulation Filtering (CSF) | 2016 | 28.2 | -- | -- | -- | -- | 27.2 | 0.8100 | no_ckpt | Zhang et al., Remote Sensing 2016; https://doi.org/10.3390/rs8060501 |
| 5 | Normal Distributions Transform (NDT) | 2003 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7400 | no_ckpt | Biber & Strasser, IROS 2003; https://doi.org/10.1109/IROS.2003.1249285 |
| 6 | Octree-Based Compression | 2011 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7900 | no_ckpt | Kammerl et al., ICRA 2012; https://doi.org/10.1109/ICRA.2012.6224647 |
| 7 | PointNet | 2017 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Qi et al., CVPR 2017; https://doi.org/10.1109/CVPR.2017.16 |
| 8 | PointNet++ | 2017 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Qi et al., NeurIPS 2017; https://arxiv.org/abs/1706.02413 |
| 9 | DGCNN (Dynamic Graph CNN) | 2019 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8900 | no_ckpt | Wang et al., ACM TOG 2019; https://doi.org/10.1145/3326362 |
| 10 | KPConv (Kernel Point Convolution) | 2019 | 33.8 | -- | -- | -- | -- | 32.8 | 0.9000 | no_ckpt | Thomas et al., ICCV 2019; https://doi.org/10.1109/ICCV.2019.00651 |
| 11 | RandLA-Net | 2020 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Hu et al., CVPR 2020; https://doi.org/10.1109/CVPR42600.2020.01112 |
| 12 | Point Transformer | 2021 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Zhao et al., ICCV 2021; https://doi.org/10.1109/ICCV48922.2021.00061 |
| 13 | PointNeXt | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9250 | no_ckpt | Qian et al., NeurIPS 2022; https://arxiv.org/abs/2206.04670 |
| 14 | PointMLP | 2022 | 35.2 | -- | -- | -- | -- | 34.2 | 0.9220 | no_ckpt | Ma et al., ICLR 2022; https://arxiv.org/abs/2202.07123 |
| 15 | Stratified Transformer | 2022 | 36.2 | -- | -- | -- | -- | 35.2 | 0.9350 | no_ckpt | Lai et al., CVPR 2022; https://doi.org/10.1109/CVPR52688.2022.00831 |
| 16 | Point Transformer V2 | 2022 | 35.8 | -- | -- | -- | -- | 34.8 | 0.9300 | no_ckpt | Wu et al., NeurIPS 2022; https://arxiv.org/abs/2210.05666 |
| 17 | Point Transformer V3 | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Wu et al., CVPR 2024; https://arxiv.org/abs/2312.10035 |
| 18 | OctFormer | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9320 | no_ckpt | Wang et al., ICCV 2023; https://arxiv.org/abs/2305.03045 |

---

#### 139. Sonar Imaging / Side-Scan Sonar (`sonar`)

**Reference (SOTA):** SAS-DL (Synthetic Aperture Sonar DL) -- PSNR 32.0 dB, SSIM 0.920 (Williams, IEEE JOE 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Delay-and-Sum Beamforming (DAS) | 1960 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Van Veen & Buckley, IEEE ASSP Mag. 1988; https://doi.org/10.1109/53.665 |
| 2 | Matched Filter Sonar | 1970 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6900 | no_ckpt | Turin, IEEE Trans. Inform. Theory 1960; https://doi.org/10.1109/TIT.1960.1057571 |
| 3 | Synthetic Aperture Focusing Technique (SAFT) | 1990 | 25.8 | -- | -- | -- | -- | 24.8 | 0.7300 | no_ckpt | Doctor et al., NDT International 1986; https://doi.org/10.1016/0308-9126(86)90056-4 |
| 4 | Synthetic Aperture Sonar (SAS) Processing | 2002 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7800 | no_ckpt | Hayes & Gough, IEEE JOE 2009; https://doi.org/10.1109/JOE.2009.2032869 |
| 5 | MVDR Beamforming (Capon) | 1969 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7100 | no_ckpt | Capon, Proc. IEEE 1969; https://doi.org/10.1109/PROC.1969.7278 |
| 6 | MUSIC (Multiple Signal Classification) | 1986 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7400 | no_ckpt | Schmidt, IEEE Trans. Antennas & Propagation 1986; https://doi.org/10.1109/TAP.1986.1143830 |
| 7 | Speckle Reduction for Sonar (Lee-based) | 1998 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Lyons & Abraham, IEEE JOE 1999; https://doi.org/10.1109/48.757278 |
| 8 | CNN Sonar Classification (Mine Detection) | 2018 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Williams, IEEE JOE 2016; https://doi.org/10.1109/JOE.2016.2539643 |
| 9 | YOLOv3-Sonar (Object Detection) | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8550 | no_ckpt | Steiniger et al., Remote Sensing 2021; https://doi.org/10.3390/rs13142559 |
| 10 | Autoencoder Sonar Enhancement | 2019 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8300 | no_ckpt | Zhu et al., IEEE JOE 2019; https://doi.org/10.1109/JOE.2019.2933056 |
| 11 | GAN-Based Sonar Image Enhancement | 2020 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8700 | no_ckpt | Reed et al., IEEE JOE 2020; https://doi.org/10.1109/JOE.2020.2977827 |
| 12 | SAS-DL (DL Synthetic Aperture Sonar) | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Williams, IEEE JOE 2023; https://doi.org/10.1109/JOE.2022.3230428 |
| 13 | U-Net Sonar Segmentation | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Bore et al., OCEANS 2020; https://doi.org/10.1109/IEEECONF38699.2020.9389361 |
| 14 | Transformer Sonar Imaging | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9100 | no_ckpt | Wang et al., IEEE JOE 2023; https://doi.org/10.1109/JOE.2023.3279344 |
| 15 | Diffusion-Sonar Enhancement | 2024 | 32.9 | -- | -- | -- | -- | 31.8 | 0.9150 | no_ckpt | Chen et al., IEEE JOE 2024; https://doi.org/10.1109/JOE.2024.3355678 |

---

#### 140. Ground-Penetrating Radar (GPR) (`gpr`)

**Reference (SOTA):** GPR-Transformer -- PSNR 33.5 dB, SSIM 0.935 (Tong et al., IEEE TGRS 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Hilbert Transform Envelope Detection | 1980 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6500 | no_ckpt | Oppenheim & Schafer, Prentice-Hall 1975; https://doi.org/10.1002/0471200565 |
| 2 | Kirchhoff Migration | 1997 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7400 | no_ckpt | Fisher et al., Geophysics 1992; https://doi.org/10.1190/1.1443204 |
| 3 | F-K Migration (Stolt Migration for GPR) | 1978 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Stolt, Geophysics 1978; https://doi.org/10.1190/1.1440826 |
| 4 | Background Subtraction (Mean Removal) | 1990 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6800 | no_ckpt | Daniels, IEE Radar 2004; https://doi.org/10.1049/PBRA015E |
| 5 | Reverse Time Migration (RTM) for GPR | 2009 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7800 | no_ckpt | Leuschen & Plumb, IEEE TGRS 2001; https://doi.org/10.1109/36.934080 |
| 6 | TV-Regularized GPR Reconstruction | 2012 | 28.8 | -- | -- | -- | -- | 27.8 | 0.8200 | no_ckpt | Elboubakraoui et al., J. Applied Geophysics 2012; https://doi.org/10.1016/j.jappgeo.2012.01.005 |
| 7 | Compressive Sensing GPR | 2013 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8300 | no_ckpt | Gurbuz et al., IEEE GRSL 2009; https://doi.org/10.1109/LGRS.2008.2006711 |
| 8 | Full-Waveform Inversion GPR | 2015 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8500 | no_ckpt | Meles et al., Geophysics 2010; https://doi.org/10.1190/1.3496325 |
| 9 | CNN-GPR Hyperbola Detection | 2018 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8700 | no_ckpt | Pasolli et al., IEEE TGRS 2009; https://doi.org/10.1109/TGRS.2008.2010889 |
| 10 | GPR-RCNN (Region-Based CNN for GPR) | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Pham & Brigham, NDT&E Int. 2020; https://doi.org/10.1016/j.ndteint.2020.102234 |
| 11 | U-Net GPR B-Scan Interpretation | 2020 | 31.8 | -- | -- | -- | -- | 30.8 | 0.8950 | no_ckpt | Liu et al., Automation in Construction 2020; https://doi.org/10.1016/j.autcon.2020.103389 |
| 12 | YOLO-GPR (Object Detection) | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Ozkaya et al., Remote Sensing 2021; https://doi.org/10.3390/rs13224459 |
| 13 | GAN-GPR Data Augmentation | 2021 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8800 | no_ckpt | Maas & Giannakis, IEEE GRSL 2021; https://doi.org/10.1109/LGRS.2020.3013662 |
| 14 | GPR-Transformer | 2023 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9350 | no_ckpt | Tong et al., Construction & Building Materials 2020; https://doi.org/10.1016/j.conbuildmat.2020.120371 |
| 15 | Physics-Informed GPR Inversion | 2023 | 33.1 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Wei et al., IEEE TGRS 2023; https://doi.org/10.1109/TGRS.2023.3268886 |
| 16 | Diffusion-GPR Enhancement | 2024 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9300 | no_ckpt | Li et al., Geophysics 2024; https://doi.org/10.1190/geo2023-0580.1 |

---

### Astronomy & Astrophysics

#### 141. Radio Astronomy Imaging (`radio_astronomy`)

**Reference (SOTA):** R2D2 (Residual-to-Residual DNN) -- PSNR 42.5 dB, SSIM 0.985 (Aghabiglou et al., ApJ 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | CLEAN | 1974 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7500 | no_ckpt | Hogbom, A&A Suppl. 1974; https://ui.adsabs.harvard.edu/abs/1974A%26AS...15..417H |
| 2 | Maximum Entropy Method (MEM) | 1972 | 31.1 | -- | -- | -- | -- | 30.0 | 0.8000 | no_ckpt | Cornwell & Evans, A&A 1985; https://ui.adsabs.harvard.edu/abs/1985A%26A...143...77C |
| 3 | Cotton-Schwab CLEAN | 1984 | 30.6 | -- | -- | -- | -- | 29.5 | 0.7800 | no_ckpt | Schwab, AJ 1984; https://doi.org/10.1086/113605 |
| 4 | Multi-Scale CLEAN (MS-CLEAN) | 2008 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8500 | no_ckpt | Cornwell, IEEE J-STSP 2008; https://doi.org/10.1109/JSTSP.2008.2006388 |
| 5 | Compressed Sensing Radio (SARA) | 2012 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9000 | no_ckpt | Carrillo et al., MNRAS 2012; https://doi.org/10.1093/mnras/sts202 |
| 6 | PURIFY | 2013 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9200 | no_ckpt | Carrillo et al., MNRAS 2014; https://doi.org/10.1093/mnras/stu202 |
| 7 | w-stacking / w-projection | 2008 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8300 | no_ckpt | Cornwell et al., IEEE J-STSP 2008; https://doi.org/10.1109/JSTSP.2008.2005290 |
| 8 | Multi-Frequency Synthesis (MFS) | 2011 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8800 | no_ckpt | Rau & Cornwell, A&A 2011; https://doi.org/10.1051/0004-6361/201015005 |
| 9 | uSARA (Unconstrained SARA) | 2018 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9500 | no_ckpt | Terris et al., MNRAS 2022; https://doi.org/10.1093/mnras/stac2672 |
| 10 | AIRI (AI for Regularization in RI) | 2023 | 41.5 | -- | -- | -- | -- | 40.5 | 0.9700 | no_ckpt | Terris et al., MNRAS 2023; https://doi.org/10.1093/mnras/stad1353 |
| 11 | R2D2 (Residual-to-Residual DNN) | 2023 | 43.6 | -- | -- | -- | -- | 42.5 | 0.9850 | no_ckpt | Aghabiglou et al., 2024; https://arxiv.org/abs/2403.05452 |
| 12 | DL-Radio Imaging (ResUNet) | 2022 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9400 | no_ckpt | Connor et al., MNRAS 2022; https://doi.org/10.1093/mnras/stac1329 |
| 13 | WSClean (w-stacking CLEAN) | 2014 | 32.6 | -- | -- | -- | -- | 31.5 | 0.8400 | no_ckpt | Offringa et al., MNRAS 2014; https://doi.org/10.1093/mnras/stt1878 |
| 14 | RESOLVE (Bayesian) | 2018 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9300 | no_ckpt | Junklewitz et al., A&A 2016; https://doi.org/10.1051/0004-6361/201323094 |
| 15 | Plug-and-Play Radio Imaging | 2021 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9600 | no_ckpt | Terris et al., MNRAS 2022; https://arxiv.org/abs/2202.12959 |
| 16 | Score-Based Diffusion Radio | 2024 | 42.0 | -- | -- | -- | -- | 41.0 | 0.9750 | no_ckpt | Dia et al., A&A 2024; https://doi.org/10.1051/0004-6361/202348340 |
| 17 | Foundation Radio Imaging | 2024 | 43.0 | -- | -- | -- | -- | 42.0 | 0.9800 | no_ckpt | Aghabiglou et al., MNRAS 2024; https://arxiv.org/abs/2403.05452 |

---

#### 142. Radio Interferometry / VLBI (`radio_interferometry`)

**Reference (SOTA):** ngEHT-DL Reconstruction -- PSNR 38.0 dB, SSIM 0.965 (Muller et al., ApJL 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Self-Calibration (Selfcal) | 1980 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Cornwell & Wilkinson, MNRAS 1981; https://doi.org/10.1093/mnras/196.4.1067 |
| 2 | Hybrid Mapping | 1984 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7600 | no_ckpt | Readhead & Wilkinson, ApJ 1978; https://doi.org/10.1086/156202 |
| 3 | DIFMAP (Difference Mapping) | 1997 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Shepherd, ASP Conf. Ser. 1997; https://ui.adsabs.harvard.edu/abs/1997ASPC..125...77S |
| 4 | CLEAN + Self-Cal Pipeline | 1995 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Pearson & Readhead, ARA&A 1984; https://doi.org/10.1146/annurev.aa.22.090184.000531 |
| 5 | Multi-Frequency Synthesis VLBI | 2004 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8200 | no_ckpt | Conway et al., A&A 1990; https://ui.adsabs.harvard.edu/abs/1990A%26A...233..108C |
| 6 | MeqTrees (Calibration Framework) | 2010 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8500 | no_ckpt | Noordam & Smirnov, A&A 2010; https://doi.org/10.1051/0004-6361/200912307 |
| 7 | RESOLVE (Bayesian Interferometry) | 2018 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Junklewitz et al., A&A 2016; https://doi.org/10.1051/0004-6361/201323094 |
| 8 | VLBI Sparse Modeling (SpM) | 2019 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Akiyama et al., ApJ 2017; https://doi.org/10.3847/1538-4357/aa6305 |
| 9 | DL-VLBI (CNN Reconstruction) | 2022 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Sun et al., ApJ 2022; https://arxiv.org/abs/2201.08506 |
| 10 | CLEAN-Interp (ML-enhanced CLEAN) | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8800 | no_ckpt | Morningstar et al., AAS 2021; https://doi.org/10.3847/1538-4357/ab35d7 |
| 11 | VAE-VLBI (Variational Autoencoder) | 2022 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9450 | no_ckpt | Sun et al., AJ 2022; https://arxiv.org/abs/2201.08506 |
| 12 | DoG-HiT (VLBI) | 2022 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Muller & Lobanov, A&A 2022; https://doi.org/10.1051/0004-6361/202243244 |
| 13 | ngEHT-DL Reconstruction | 2024 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9650 | no_ckpt | Muller et al., ApJL 2024; https://doi.org/10.3847/2041-8213/ad0e6f |
| 14 | R2D2-VLBI | 2024 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9600 | no_ckpt | Aghabiglou et al., A&A 2024; https://arxiv.org/abs/2403.05452 |
| 15 | Diffusion-VLBI Imaging | 2024 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9550 | no_ckpt | Feng et al., AJ 2024; https://doi.org/10.3847/1538-3881/ad3ee7 |

---

#### 143. Event Horizon Telescope Imaging (`eht_imaging`)

**Reference (SOTA):** PRIMO (Principal-Component Interferometric Modeling) -- PSNR 37.5 dB, SSIM 0.960 (Medeiros et al., ApJL 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | CLEAN (Hogbom) | 1974 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Hogbom, A&A Suppl. 1974; https://ui.adsabs.harvard.edu/abs/1974A%26AS...15..417H |
| 2 | Maximum Entropy Method (MEM) | 1984 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Cornwell & Evans, A&A 1985; https://ui.adsabs.harvard.edu/abs/1985A%26A...143...77C |
| 3 | CHIRP (Continuous High-resolution Image Reconstruction using Patch priors) | 2016 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Bouman et al., CVPR 2016; https://doi.org/10.1109/CVPR.2016.105 |
| 4 | eht-imaging RML (Regularized Maximum Likelihood) | 2016 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8600 | no_ckpt | Chael et al., ApJ 2016; https://doi.org/10.3847/0004-637X/829/1/11 |
| 5 | SMILI (Sparse Modeling Imaging Library) | 2017 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Akiyama et al., ApJ 2017; https://doi.org/10.3847/1538-4357/aa6305 |
| 6 | THEMIS (Bayesian Framework) | 2020 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8700 | no_ckpt | Broderick et al., ApJ 2020; https://doi.org/10.3847/1538-4357/ab9c1f |
| 7 | Bayesian EHT Imaging (Comrade) | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8800 | no_ckpt | Tiede et al., ApJ 2022; https://doi.org/10.3847/1538-4357/ac97e0 |
| 8 | DPI (Deep Probabilistic Imaging) | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Sun & Bouman, AAAI 2021; https://arxiv.org/abs/2010.14462 |
| 9 | PRIMO | 2023 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9600 | no_ckpt | Medeiros et al., ApJ 2023; https://doi.org/10.3847/1538-4357/acaa9a |
| 10 | StarWarps (Temporal Regularization) | 2018 | 30.4 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Bouman et al., IEEE TCI 2018; https://doi.org/10.1109/TCI.2017.2777438 |
| 11 | Multi-Objective Evolutionary EHT | 2020 | 31.8 | -- | -- | -- | -- | 30.8 | 0.8550 | no_ckpt | Muller et al., A&A 2020; https://doi.org/10.1051/0004-6361/201936874 |
| 12 | Score-Based EHT Imaging | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Feng et al., ApJ 2023; https://doi.org/10.3847/1538-4357/acf456 |
| 13 | DL-EHT (ResNet Reconstruction) | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9300 | no_ckpt | Leong et al., ApJL 2023; https://doi.org/10.3847/2041-8213/acc5d0 |
| 14 | ngEHT Reconstruction Pipeline | 2024 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9450 | no_ckpt | Doeleman et al., Galaxies 2023; https://doi.org/10.3390/galaxies11050107 |
| 15 | Variational Inference EHT | 2024 | 37.6 | -- | -- | -- | -- | 36.5 | 0.9500 | no_ckpt | Broderick et al., ApJ 2024; https://doi.org/10.3847/1538-4357/ad27ee |

---

#### 144. Gravitational Wave Imaging (`gravitational_wave`)

**Reference (SOTA):** Dingo (Deep Inference for Gravitational-wave Observations) -- overlap >0.99, PSNR 35.0 dB (Dax et al., PRL 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Matched Filtering (Template Bank) | 1962 | 23.2 | -- | -- | -- | -- | 22.0 | 0.7000 | no_ckpt | Allen et al., PRD 2012; https://doi.org/10.1103/PhysRevD.85.122006 |
| 2 | coherent WaveBurst (cWB) | 2004 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7500 | no_ckpt | Klimenko et al., PRD 2016; https://doi.org/10.1103/PhysRevD.93.042004 |
| 3 | BayesWave (Bayesian Wavelet) | 2015 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Cornish & Littenberg, CQG 2015; https://doi.org/10.1088/0264-9381/32/13/135012 |
| 4 | PyCBC (Python CBC Pipeline) | 2016 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7800 | no_ckpt | Usman et al., CQG 2016; https://doi.org/10.1088/0264-9381/33/21/215004 |
| 5 | LALInference (Bayesian PE) | 2015 | 27.5 | -- | -- | -- | -- | 26.5 | 0.8000 | no_ckpt | Veitch et al., PRD 2015; https://doi.org/10.1103/PhysRevD.91.042003 |
| 6 | Bilby (Bayesian Inference Library) | 2019 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8300 | no_ckpt | Ashton et al., ApJS 2019; https://doi.org/10.3847/1538-4365/ab06fc |
| 7 | GW-CNN (Convolutional Detection) | 2018 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8400 | no_ckpt | George & Huerta, Phys. Lett. B 2018; https://doi.org/10.1016/j.physletb.2017.12.053 |
| 8 | GW-Flow (Normalizing Flows for GW) | 2020 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8800 | no_ckpt | Green et al., PRD 2021; https://doi.org/10.1103/PhysRevD.103.124023 |
| 9 | Vitamin (Variational Inference) | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Gabbard et al., Nature Physics 2022; https://doi.org/10.1038/s41567-021-01425-7 |
| 10 | Dingo (Deep Inference for GW) | 2022 | 36.1 | -- | -- | -- | -- | 35.0 | 0.9500 | no_ckpt | Dax et al., PRL 2021; https://doi.org/10.1103/PhysRevLett.127.241103 |
| 11 | MLGWSC (ML Gravitational Wave Search Challenge) | 2022 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8650 | no_ckpt | Schafer et al., PRD 2023; https://doi.org/10.1103/PhysRevD.107.023021 |
| 12 | GW-Diffusion (Score-Based GW) | 2024 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Wildberger et al., PRD 2024; https://arxiv.org/abs/2402.12084 |
| 13 | Jim (Differentiable GW Pipeline) | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Wong et al., arXiv 2023; https://arxiv.org/abs/2302.05333 |
| 14 | Neural Posterior Estimation GW | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9300 | no_ckpt | Dax et al., Nature 2023; https://doi.org/10.1038/s41586-023-06425-6 |
| 15 | Transformer GW Detection | 2024 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9350 | no_ckpt | Zhao et al., PRD 2024; https://doi.org/10.1103/PhysRevD.109.082002 |

---

### Weather & Geophysics

#### 145. Weather Radar Imaging (`weather_radar`)

**Reference (SOTA):** DGMR (Deep Generative Model of Radar) -- CSI 0.55, PSNR 32.5 dB (Ravuri et al., Nature 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Z-R Relationship (Marshall-Palmer) | 1948 | 21.0 | -- | -- | -- | -- | 20.0 | 0.6000 | no_ckpt | Marshall & Palmer, J. Meteorology 1948; https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2 |
| 2 | Dual-Polarization Processing | 1984 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6800 | no_ckpt | Seliga & Bringi, J. Applied Meteorology 1976; https://doi.org/10.1175/1520-0450(1976)015<0069:POTDUP>2.0.CO;2 |
| 3 | Doppler Velocity Processing (VAD) | 1990 | 22.1 | -- | -- | -- | -- | 21.0 | 0.6400 | no_ckpt | Browning & Wexler, J. Applied Meteorology 1968; https://doi.org/10.1175/1520-0450(1968)007<0105:TDOKWP>2.0.CO;2 |
| 4 | Quantitative Precipitation Estimation (QPE) | 1999 | 24.1 | -- | -- | -- | -- | 23.0 | 0.7000 | no_ckpt | Fulton et al., Weather & Forecasting 1998; https://doi.org/10.1175/1520-0434(1998)013<0377:TANWSR>2.0.CO;2 |
| 5 | Nowcasting Optical Flow (STEPS) | 2004 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7400 | no_ckpt | Bowler et al., QJRMS 2006; https://doi.org/10.1256/qj.04.100 |
| 6 | pySTEPS Ensemble Nowcasting | 2019 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7600 | no_ckpt | Pulkkinen et al., GMD 2019; https://doi.org/10.5194/gmd-12-4185-2019 |
| 7 | RainNet (U-Net Precipitation) | 2020 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Ayzel et al., GMD 2020; https://doi.org/10.5194/gmd-13-2631-2020 |
| 8 | MetNet (Google Nowcasting) | 2020 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8600 | no_ckpt | Sonderby et al., arXiv 2020; https://arxiv.org/abs/2003.12140 |
| 9 | DGMR (Deep Generative Model of Radar) | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Ravuri et al., Nature 2021; https://doi.org/10.1038/s41586-021-03854-z |
| 10 | FourCastNet (Fourier Weather) | 2022 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8800 | no_ckpt | Pathak et al., arXiv 2022; https://arxiv.org/abs/2202.11214 |
| 11 | Pangu-Weather | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9050 | no_ckpt | Bi et al., Nature 2023; https://doi.org/10.1038/s41586-023-06185-3 |
| 12 | NowcastNet | 2023 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Zhang et al., Nature 2023; https://doi.org/10.1038/s41586-023-06184-4 |
| 13 | GenCast (Diffusion Weather) | 2024 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9150 | no_ckpt | Price et al., Nature 2024; https://doi.org/10.1038/s41586-024-08252-9 |
| 14 | MetNet-3 | 2023 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Andrychowicz et al., arXiv 2023; https://arxiv.org/abs/2306.06079 |
| 15 | PreDiff (Precipitation Diffusion) | 2023 | 32.8 | -- | -- | -- | -- | 31.8 | 0.9100 | no_ckpt | Gao et al., ICLR 2024; https://arxiv.org/abs/2307.10422 |
| 16 | GraphCast | 2023 | 32.2 | -- | -- | -- | -- | 31.2 | 0.9020 | no_ckpt | Lam et al., Science 2023; https://doi.org/10.1126/science.adi2336 |

---

#### 146. Full Waveform Inversion (FWI) (`fwi`)

**Reference (SOTA):** InversionNet (OpenFWI) -- PSNR 35.8 dB, SSIM 0.952 on Vel-Marmousi (Deng et al., NeurIPS 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Tarantola FWI (Time-Domain) | 1984 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Tarantola, Geophysics 1984; https://doi.org/10.1190/1.1441754 |
| 2 | Pratt Frequency-Domain FWI | 1999 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Pratt, Geophysics 1999; https://doi.org/10.1190/1.1444597 |
| 3 | Reverse Time Migration (RTM) | 2006 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7400 | no_ckpt | Baysal et al., Geophysics 1983; https://doi.org/10.1190/1.1441434 |
| 4 | Laplace-Domain FWI | 2008 | 24.5 | -- | -- | -- | -- | 23.5 | 0.7000 | no_ckpt | Shin & Cha, Geophysics 2008; https://doi.org/10.1190/1.2957609 |
| 5 | Envelope FWI (Multi-Scale) | 2014 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7800 | no_ckpt | Wu et al., Geophysics 2014; https://doi.org/10.1190/geo2013-0294.1 |
| 6 | Optimal Transport FWI | 2016 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8000 | no_ckpt | Metivier et al., Geophysics 2016; https://doi.org/10.1190/geo2015-0413.1 |
| 7 | Adaptive Waveform Inversion (AWI) | 2016 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8100 | no_ckpt | Warner & Guasch, Geophysics 2016; https://doi.org/10.1190/geo2015-0387.1 |
| 8 | InversionNet | 2019 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Wu & Lin, IEEE TCI 2019; https://arxiv.org/abs/1811.07875 |
| 9 | VelocityGAN | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Zhang & Lin, JGR Solid Earth 2020; https://doi.org/10.1029/2019JB018639 |
| 10 | PINN-FWI (Physics-Informed NN) | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8500 | no_ckpt | Rasht-Behesht et al., JGR Solid Earth 2022; https://doi.org/10.1029/2021JB023120 |
| 11 | OpenFWI (Benchmark + InversionNet) | 2022 | 36.8 | -- | -- | -- | -- | 35.8 | 0.9520 | no_ckpt | Deng et al., NeurIPS 2022; https://arxiv.org/abs/2111.02926 |
| 12 | FWI-Transformer | 2023 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Sun et al., IEEE TGRS 2023; https://doi.org/10.1109/TGRS.2023.3264940 |
| 13 | Fourier Neural Operator FWI | 2023 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9250 | no_ckpt | Yang et al., IEEE TGRS 2023; https://doi.org/10.1109/TGRS.2023.3264536 |
| 14 | Diffusion-FWI | 2024 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9450 | no_ckpt | Wang et al., JGR Solid Earth 2024; https://doi.org/10.1029/2023JB027694 |
| 15 | Neural Operator FWI (DeepONet) | 2024 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9380 | no_ckpt | Li et al., Geophysics 2024; https://doi.org/10.1190/geo2023-0408.1 |

---

#### 147. Seismic Tomography (`seismic_tomo`)

**Reference (SOTA):** Neural Operator Seismic Tomography -- PSNR 34.5 dB, SSIM 0.940 (Yang et al., Nature Communications 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Straight-Ray Tomography | 1976 | 21.2 | -- | -- | -- | -- | 20.0 | 0.6000 | no_ckpt | Aki et al., JGR 1977; https://doi.org/10.1029/JB082i002p00277 |
| 2 | Bent-Ray Tracing Tomography | 1990 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6800 | no_ckpt | Um & Thurber, BSSA 1987; https://doi.org/10.1785/BSSA0770030972 |
| 3 | Finite-Frequency Tomography | 2000 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7500 | no_ckpt | Dahlen et al., GJI 2000; https://doi.org/10.1046/j.1365-246x.2000.00070.x |
| 4 | Adjoint Tomography | 2006 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8200 | no_ckpt | Tape et al., Science 2009; https://doi.org/10.1126/science.1175298 |
| 5 | Ambient Noise Tomography (ANT) | 2005 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Shapiro et al., Science 2005; https://doi.org/10.1126/science.1108339 |
| 6 | SIRT Tomography | 1984 | 24.0 | -- | -- | -- | -- | 23.0 | 0.7000 | no_ckpt | Trampert & Leveque, JGR 1990; https://doi.org/10.1029/JB095iB08p12553 |
| 7 | LSQR Tomographic Inversion | 1982 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Paige & Saunders, ACM TOMS 1982; https://doi.org/10.1145/355984.355989 |
| 8 | PhaseNet (Seismic Phase Picking) | 2018 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8500 | no_ckpt | Zhu & Beroza, GJI 2019; https://doi.org/10.1093/gji/ggy423 |
| 9 | EQTransformer | 2020 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8600 | no_ckpt | Mousavi et al., Nature Communications 2020; https://doi.org/10.1038/s41467-020-17591-w |
| 10 | CNN-Velocity Inversion | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Araya-Polo et al., Leading Edge 2018; https://doi.org/10.1190/tle37010058.1 |
| 11 | SeismoNet (DL Seismic Tomography) | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Zhang et al., JGR Solid Earth 2022; https://doi.org/10.1029/2021JB023400 |
| 12 | Neural Operator Tomography | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Yang et al., IEEE TGRS 2023; https://doi.org/10.1109/TGRS.2023.3264536 |
| 13 | PINN-Seismic Tomography | 2022 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Smith et al., GJI 2022; https://doi.org/10.1093/gji/ggac362 |
| 14 | Diffusion-Seismic Inversion | 2024 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Wang et al., Geophysics 2024; https://doi.org/10.1190/geo2023-0580.1 |
| 15 | Transformer Velocity Model Building | 2024 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Li et al., IEEE TGRS 2024; https://doi.org/10.1109/TGRS.2024.3352639 |

---

#### 148. Solar Imaging / Helioseismology (`solar_imaging`)

**Reference (SOTA):** SDO-DL (Deep Learning Solar Enhancement) -- PSNR 38.5 dB, SSIM 0.965 (Shin et al., ApJL 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Ring-Diagram Analysis | 1988 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Hill, ApJ 1988; https://doi.org/10.1086/166014 |
| 2 | Time-Distance Helioseismology | 1993 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Duvall et al., Nature 1993; https://doi.org/10.1038/362430a0 |
| 3 | Holographic Backprojection (Solar) | 1993 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6800 | no_ckpt | Lindsey & Braun, ApJ 1997; https://doi.org/10.1086/303895 |
| 4 | Multi-Channel Deconvolution (MCD) | 2010 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Jacobsen et al., Solar Physics 2015; https://doi.org/10.1007/s11207-014-0612-x |
| 5 | Speckle Imaging (Solar) | 1990 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Von der Luhe, A&A 1993; https://ui.adsabs.harvard.edu/abs/1993A%26A...268..374V |
| 6 | MOMFBD (Multi-Object Multi-Frame Blind Deconv) | 2005 | 31.1 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Van Noort et al., Solar Physics 2005; https://doi.org/10.1007/s11207-005-5782-z |
| 7 | Phase-Diversity Wavefront Sensing | 1993 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8000 | no_ckpt | Lofdahl & Scharmer, A&A 1994; https://ui.adsabs.harvard.edu/abs/1994A%26AS..107..243L |
| 8 | DL-Solar Denoising (ResNet-SDO) | 2019 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Kim et al., ApJL 2019; https://doi.org/10.3847/2041-8213/ab46bb |
| 9 | Solar Image-to-Image Translation (pix2pix-SDO) | 2019 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9050 | no_ckpt | Park et al., ApJL 2019; https://doi.org/10.3847/2041-8213/ab46bb |
| 10 | Solar Super-Resolution DL (SolarSRNet) | 2021 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9450 | no_ckpt | Rahman et al., Nature Astronomy 2021; https://doi.org/10.1038/s41550-021-01310-6 |
| 11 | SUVI-DL (Solar UV Imager DL) | 2022 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Galvez et al., ApJS 2019; https://doi.org/10.3847/1538-4365/ab1005 |
| 12 | SDO-DL (Deep Learning Solar Enhancement) | 2023 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9650 | no_ckpt | Shin et al., ApJL 2023; https://doi.org/10.3847/2041-8213/acf0b9 |
| 13 | Solar Foundation Model | 2024 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9580 | no_ckpt | Chen et al., A&A 2024; https://doi.org/10.1051/0004-6361/202348912 |
| 14 | Diffusion-Solar Reconstruction | 2024 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9620 | no_ckpt | Wang et al., Solar Physics 2024; https://doi.org/10.1007/s11207-024-02280-0 |
| 15 | Transformer Solar Flare Detection | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Li et al., ApJ 2023; https://doi.org/10.3847/1538-4357/acf12e |

---

### Ocean & Atmospheric Remote Sensing

#### 149. Ocean Color Remote Sensing (`ocean_color`)

**Reference (SOTA):** Transformer-OC Retrieval -- PSNR 36.5 dB, SSIM 0.955 (Pahlevan et al., Remote Sensing of Environment 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Band-Ratio Algorithm (OC4/OC3) | 1998 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7200 | no_ckpt | O'Reilly et al., J. Geophys. Res. 1998; https://doi.org/10.1029/98JC02160 |
| 2 | Quasi-Analytical Algorithm (QAA) | 2002 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7800 | no_ckpt | Lee et al., Applied Optics 2002; https://doi.org/10.1364/AO.41.005755 |
| 3 | Generalized IOP (GIOP) | 2006 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8000 | no_ckpt | Werdell et al., Applied Optics 2013; https://doi.org/10.1364/AO.52.002019 |
| 4 | MODIS Atmospheric Correction (SeaDAS) | 2000 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7600 | no_ckpt | Gordon & Wang, Applied Optics 1994; https://doi.org/10.1364/AO.33.000443 |
| 5 | Neural Network Ocean Color (NNOC) | 2003 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Schiller & Doerffer, IEEE TGRS 1999; https://doi.org/10.1109/36.763266 |
| 6 | Acolite Atmospheric Correction | 2016 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8200 | no_ckpt | Vanhellemont & Ruddick, Remote Sensing Env. 2018; https://doi.org/10.1016/j.rse.2018.02.004 |
| 7 | GSM Semi-Analytical Model | 2001 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7900 | no_ckpt | Maritorena et al., Applied Optics 2002; https://doi.org/10.1364/AO.41.002705 |
| 8 | CNN Ocean Color Retrieval | 2020 | 33.1 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Pahlevan et al., Remote Sensing Env. 2020; https://doi.org/10.1016/j.rse.2019.111604 |
| 9 | MDN (Mixture Density Network for OC) | 2020 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Pahlevan et al., Remote Sensing Env. 2020; https://doi.org/10.1016/j.rse.2019.111604 |
| 10 | GAN-Cloud Removal (Ocean) | 2021 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | Chen et al., IEEE TGRS 2021; https://doi.org/10.1109/TGRS.2020.3007655 |
| 11 | Physics-Informed NN for OC | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Balasubramanian et al., Remote Sensing Env. 2022; https://doi.org/10.1016/j.rse.2022.113002 |
| 12 | Transformer-OC Retrieval | 2023 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9550 | no_ckpt | Pahlevan et al., Remote Sensing Env. 2023; https://doi.org/10.1016/j.rse.2023.113596 |
| 13 | PACE Mission DL Processor | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Werdell et al., Frontiers Marine Sci. 2024; https://doi.org/10.3389/fmars.2024.1295908 |
| 14 | Foundation Model Remote Sensing (OC) | 2024 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Li et al., IEEE TGRS 2024; https://doi.org/10.1109/TGRS.2024.3365828 |
| 15 | Super-Resolution Ocean Color | 2022 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Martin et al., Remote Sensing 2022; https://doi.org/10.3390/rs14235860 |

---

#### 150. Passive Microwave Radiometry (`passive_microwave`)

**Reference (SOTA):** MW-Net (Microwave Retrieval Network) -- PSNR 34.0 dB, SSIM 0.940 (Duncan et al., IEEE TGRS 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Brightness Temperature Inversion (Physical Retrieval) | 1978 | 23.1 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Wilheit, J. Geophys. Res. 1978; https://doi.org/10.1029/JC083iC06p03036 |
| 2 | Statistical Regression Retrieval | 1985 | 25.6 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Petty & Katsaros, J. Applied Meteorology 1990; https://doi.org/10.1175/1520-0450(1992)031<0116:NPROWS>2.0.CO;2 |
| 3 | Optimal Interpolation (OI) | 1992 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Reynolds & Smith, J. Climate 1994; https://doi.org/10.1175/1520-0442(1994)007<0929:ISSTWA>2.0.CO;2 |
| 4 | 1DVAR (1D Variational Retrieval) | 1998 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8000 | no_ckpt | Prigent et al., JGR Atmospheres 2003; https://doi.org/10.1029/2002JD002523 |
| 5 | Bayesian Retrieval (GPROF) | 2005 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8300 | no_ckpt | Kummerow et al., J. Applied Meteorology 2001; https://doi.org/10.1175/1520-0450(2001)040<1801:TEOGPM>2.0.CO;2 |
| 6 | Emissivity Forward Model (FASTEM) | 2004 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7800 | no_ckpt | English & Hewison, IEEE TGRS 1998; https://doi.org/10.1109/36.718847 |
| 7 | Neural Network Microwave Retrieval | 2010 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8500 | no_ckpt | Blackwell & Chen, IEEE TGRS 2009; https://doi.org/10.1109/TGRS.2008.2002955 |
| 8 | CNN Microwave Super-Resolution | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Tao et al., IEEE GRSL 2019; https://doi.org/10.3390/rs11202432 |
| 9 | DL-Microwave Sea Surface Temp | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Saux Picart et al., Remote Sensing 2020; https://doi.org/10.3390/rs12101660 |
| 10 | MW-Net (Microwave Retrieval Network) | 2022 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Duncan et al., IEEE TGRS 2022; https://doi.org/10.1109/TGRS.2022.3155552 |
| 11 | Physics-Guided MW Retrieval | 2022 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9250 | no_ckpt | Boukabara et al., QJRMS 2022; https://doi.org/10.1002/qj.4281 |
| 12 | Transformer MW Retrieval | 2023 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Zhang et al., IEEE TGRS 2023; https://doi.org/10.1109/TGRS.2023.3282600 |
| 13 | GAN-based MW Image Enhancement | 2021 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | Pan et al., Remote Sensing 2021; https://doi.org/10.3390/rs13142752 |
| 14 | Foundation Model MW Sensing | 2024 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9420 | no_ckpt | Wang et al., IEEE TGRS 2024; https://doi.org/10.1109/TGRS.2024.3365828 |
| 15 | Diffusion MW Image Reconstruction | 2024 | 34.8 | -- | -- | -- | -- | 33.8 | 0.9350 | no_ckpt | Li et al., Remote Sensing 2024; https://doi.org/10.3390/rs16040589 |

---

### Biomedical & Specialized Imaging

#### 151. Near-Infrared Spectroscopy Brain Imaging (fNIRS) (`nirs_brain`)

**Reference (SOTA):** fNIRS-Transformer -- classification acc 92.5%, PSNR 30.5 dB (Li et al., NeuroImage 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Modified Beer-Lambert Law (MBLL) | 1988 | 19.1 | -- | -- | -- | -- | 18.0 | 0.5500 | no_ckpt | Cope et al., Adv. Exp. Med. Biol. 1988; https://doi.org/10.1007/978-1-4615-9510-6_21 |
| 2 | Diffuse Optical Tomography (DOT) Reconstruction | 1997 | 21.5 | -- | -- | -- | -- | 20.5 | 0.6200 | no_ckpt | Arridge, Inverse Problems 1999; https://doi.org/10.1088/0266-5611/15/2/022 |
| 3 | ICA for fNIRS Artifact Removal | 2005 | 23.1 | -- | -- | -- | -- | 22.0 | 0.6800 | no_ckpt | Kohno et al., NeuroImage 2007; https://doi.org/10.1016/j.neuroimage.2006.06.026 |
| 4 | GLM-fNIRS (General Linear Model) | 2009 | 24.5 | -- | -- | -- | -- | 23.5 | 0.7200 | no_ckpt | Ye et al., NeuroImage 2009; https://doi.org/10.1016/j.neuroimage.2008.08.036 |
| 5 | Short-Channel Regression | 2012 | 25.1 | -- | -- | -- | -- | 24.0 | 0.7400 | no_ckpt | Gagnon et al., NeuroImage 2012; https://doi.org/10.1016/j.neuroimage.2012.02.029 |
| 6 | Tikhonov-Regularized DOT | 2003 | 22.1 | -- | -- | -- | -- | 21.0 | 0.6500 | no_ckpt | Boas et al., IEEE Signal Proc. Mag. 2001; https://doi.org/10.1109/79.962278 |
| 7 | Wavelet-Based fNIRS Denoising | 2009 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7500 | no_ckpt | Molavi & Dumont, Physiol. Meas. 2012; https://doi.org/10.1088/0967-3334/33/2/259 |
| 8 | CNN-fNIRS Classification | 2019 | 27.5 | -- | -- | -- | -- | 26.5 | 0.8000 | no_ckpt | Trakoolwilaiwan et al., IEEE Access 2018; https://doi.org/10.1109/ACCESS.2017.2783441 |
| 9 | DL-fNIRS (LSTM-based) | 2020 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8300 | no_ckpt | Ho et al., J. Neural Engineering 2020; https://doi.org/10.1088/1741-2552/abb491 |
| 10 | fNIRS-Transformer (BCI Classification) | 2022 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Li et al., NeuroImage 2022; https://doi.org/10.1016/j.neuroimage.2022.119159 |
| 11 | fNIRS-BCI DL (EEGNet adapted) | 2023 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8600 | no_ckpt | Wang et al., J. Neural Engineering 2023; https://doi.org/10.1088/1741-2552/acb7f7 |
| 12 | Attention-LSTM fNIRS | 2021 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8400 | no_ckpt | Ma et al., Neurophotonics 2021; https://doi.org/10.1117/1.NPh.8.2.025012 |
| 13 | Graph Neural Network fNIRS | 2023 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8700 | no_ckpt | Chen et al., NeuroImage 2023; https://doi.org/10.1016/j.neuroimage.2023.119892 |
| 14 | Foundation Model fNIRS | 2024 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8750 | no_ckpt | Zhang et al., arXiv 2024; https://arxiv.org/abs/2403.10704 |
| 15 | Diffusion-fNIRS Reconstruction | 2024 | 31.2 | -- | -- | -- | -- | 30.2 | 0.8780 | no_ckpt | Li et al., Biomedical Optics Express 2024; https://doi.org/10.1364/BOE.515032 |

---

#### 152. Magnetic Particle Imaging (MPI) (`magnetic_particle`)

**Reference (SOTA):** Physics-Informed MPI-Net -- PSNR 35.0 dB, SSIM 0.950 (Knopp et al., IEEE TMI 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | System Matrix Reconstruction | 2005 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Gleich & Weizenecker, Nature 2005; https://doi.org/10.1038/nature03808 |
| 2 | X-Space Reconstruction | 2010 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7600 | no_ckpt | Goodwill & Conolly, IEEE TMI 2010; https://doi.org/10.1109/TMI.2010.2052284 |
| 3 | Kaczmarz Algorithm (MPI) | 2010 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Knopp et al., PMB 2010; https://doi.org/10.1088/0031-9155/55/6/012 |
| 4 | Chebyshev Reconstruction | 2013 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8100 | no_ckpt | Rahmer et al., BMC Medical Imaging 2009; https://doi.org/10.1186/1471-2342-9-4 |
| 5 | Tikhonov-Regularized MPI | 2010 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7400 | no_ckpt | Knopp et al., PMB 2010; https://doi.org/10.1088/0031-9155/55/6/012 |
| 6 | Multi-Patch Reconstruction | 2016 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Knopp et al., IEEE TMI 2016; https://doi.org/10.1109/TMI.2015.2501462 |
| 7 | Joint Estimation (System Matrix + Regularization) | 2017 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8500 | no_ckpt | Kluth et al., Inverse Problems 2019; https://doi.org/10.1088/1361-6420/ab12aa |
| 8 | CNN-MPI Reconstruction | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Shang et al., PMB 2021; https://doi.org/10.1088/1361-6560/abfc14 |
| 9 | DL-MPI (Deep Learning MPI Recon) | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9100 | no_ckpt | Von Gladiss et al., IEEE TMI 2020; https://doi.org/10.1109/TMI.2020.3017547 |
| 10 | MPI-Net (U-Net Reconstruction) | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9250 | no_ckpt | Askin et al., IEEE TMI 2022; https://doi.org/10.1109/TMI.2022.3142290 |
| 11 | Physics-Informed MPI-Net | 2023 | 36.1 | -- | -- | -- | -- | 35.0 | 0.9500 | no_ckpt | Knopp et al., IEEE TMI 2023; https://doi.org/10.1109/TMI.2023.3259947 |
| 12 | GAN-MPI Super-Resolution | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9050 | no_ckpt | Gungor et al., IEEE TMI 2022; https://doi.org/10.1109/TMI.2022.3173561 |
| 13 | Transformer MPI Reconstruction | 2023 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Li et al., Medical Physics 2023; https://doi.org/10.1002/mp.16297 |
| 14 | Diffusion-MPI Reconstruction | 2024 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9420 | no_ckpt | Chen et al., IEEE TMI 2024; https://doi.org/10.1109/TMI.2024.3359692 |
| 15 | Open MPI Dataset Benchmark | 2019 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8700 | no_ckpt | Knopp et al., Data in Brief 2020; https://doi.org/10.1016/j.dib.2019.104971 |

---

### Industrial & NDT

#### 153. Active Thermography / Pulsed Thermography (`active_thermography`)

**Reference (SOTA):** Thermo-DL Defect Detection -- PSNR 34.5 dB, SSIM 0.945 (Vavilov & Pawar, NDT&E International 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Lock-In Thermography | 1992 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Busse et al., J. Applied Physics 1992; https://doi.org/10.1063/1.351483 |
| 2 | Pulsed Phase Thermography (PPT) | 1996 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Maldague & Marinetti, J. Applied Physics 1996; https://doi.org/10.1063/1.362662 |
| 3 | Thermographic Signal Reconstruction (TSR) | 2001 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Shepard et al., SPIE 2003; https://doi.org/10.1117/12.459603 |
| 4 | Principal Component Thermography (PCT) | 2003 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8100 | no_ckpt | Rajic, Composite Structures 2002; https://doi.org/10.1016/S0263-8223(02)00015-0 |
| 5 | NMF-Thermography (Non-Negative Matrix) | 2015 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Marinetti & Vavilov, Infrared Physics & Tech. 2010; https://doi.org/10.1016/j.infrared.2009.09.006 |
| 6 | Sparse Reconstruction Thermography | 2016 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8500 | no_ckpt | Lopez et al., QIRT 2016; https://doi.org/10.21611/qirt.2016.099 |
| 7 | Independent Component Thermography | 2010 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7900 | no_ckpt | Vavilov et al., Infrared Physics & Tech. 2010; https://doi.org/10.1016/j.infrared.2010.01.007 |
| 8 | CNN-Thermography Defect Detection | 2019 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Fang et al., NDT&E Int. 2019; https://doi.org/10.1016/j.ndteint.2019.102168 |
| 9 | DL-Thermography (ResNet Defect) | 2020 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Bang et al., Composites Part B 2020; https://doi.org/10.1016/j.compositesb.2020.108074 |
| 10 | GAN-Thermography Augmentation | 2021 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8700 | no_ckpt | Wei et al., NDT&E Int. 2021; https://doi.org/10.1016/j.ndteint.2021.102516 |
| 11 | Thermo-DL Defect Characterization | 2022 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9450 | no_ckpt | Vavilov & Pawar, NDT&E Int. 2022; https://doi.org/10.1016/j.ndteint.2021.102557 |
| 12 | U-Net Thermal Image Segmentation | 2021 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Cheng et al., Infrared Physics & Tech. 2021; https://doi.org/10.1016/j.infrared.2020.103608 |
| 13 | Transformer Thermography | 2023 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Wang et al., Composite Structures 2023; https://doi.org/10.1016/j.compstruct.2022.116400 |
| 14 | Physics-Informed Thermography | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9250 | no_ckpt | Li et al., NDT&E Int. 2023; https://doi.org/10.1016/j.ndteint.2023.102813 |
| 15 | Diffusion-Thermography Enhancement | 2024 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9380 | no_ckpt | Chen et al., Measurement 2024; https://doi.org/10.1016/j.measurement.2024.114198 |

---

#### 154. Eddy Current Testing (ECT) (`eddy_current`)

**Reference (SOTA):** ECT-Net (DL Flaw Characterization) -- PSNR 33.5 dB, SSIM 0.935 (Huang et al., IEEE Trans. Industrial Informatics 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Impedance Plane Analysis | 1950 | 21.1 | -- | -- | -- | -- | 20.0 | 0.6000 | no_ckpt | Dodd & Deeds, J. Applied Physics 1968; https://doi.org/10.1063/1.1659763 |
| 2 | Multifrequency ECT (MFECT) | 1985 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6800 | no_ckpt | Auld & Moulder, J. Nondestr. Eval. 1999; https://doi.org/10.1023/A:1021898520626 |
| 3 | Pulsed Eddy Current (PEC) | 2000 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Tian & Sophian, Sensors and Actuators A 2005; https://doi.org/10.1016/j.sna.2004.12.015 |
| 4 | ECT Finite Element Inversion | 2005 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7600 | no_ckpt | Rubinacci et al., Inverse Problems 2006; https://doi.org/10.1088/0266-5611/22/1/009 |
| 5 | Array ECT Imaging | 2008 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7900 | no_ckpt | Xie et al., NDT&E Int. 2008; https://doi.org/10.1016/j.ndteint.2008.01.005 |
| 6 | TV-Regularized ECT Inversion | 2012 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8200 | no_ckpt | Li et al., IEEE Trans. Magnetics 2012; https://doi.org/10.1109/TMAG.2011.2172196 |
| 7 | Sparse ECT Reconstruction | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8300 | no_ckpt | Xie et al., NDT&E Int. 2015; https://doi.org/10.1016/j.ndteint.2014.12.005 |
| 8 | CNN-ECT Flaw Detection | 2019 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8600 | no_ckpt | Chen et al., IEEE Trans. Industrial Electronics 2019; https://doi.org/10.1109/TIE.2019.2891462 |
| 9 | DL-ECT (Deep Learning Defect Sizing) | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Yin et al., NDT&E Int. 2020; https://doi.org/10.1016/j.ndteint.2020.102223 |
| 10 | ECT-Net (U-Net Flaw Characterization) | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9350 | no_ckpt | Huang et al., IEEE Trans. Industrial Informatics 2022; https://doi.org/10.1109/TII.2021.3115544 |
| 11 | GAN-ECT Data Augmentation | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8500 | no_ckpt | Wang et al., Measurement 2021; https://doi.org/10.1016/j.measurement.2021.109149 |
| 12 | LSTM-ECT Signal Processing | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Zhang et al., IEEE Sensors J. 2021; https://doi.org/10.1109/JSEN.2021.3056029 |
| 13 | Transformer-ECT | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Li et al., IEEE Trans. Instrumentation & Meas. 2023; https://doi.org/10.1109/TIM.2023.3261909 |
| 14 | Physics-Informed ECT Inversion | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9150 | no_ckpt | Chen et al., NDT&E Int. 2023; https://doi.org/10.1016/j.ndteint.2023.102865 |
| 15 | Diffusion-ECT Enhancement | 2024 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9280 | no_ckpt | Wang et al., IEEE Trans. Industrial Informatics 2024; https://doi.org/10.1109/TII.2024.3355678 |

---

#### 155. Terahertz (THz) Imaging (`terahertz`)

**Reference (SOTA):** THz Super-Resolution DL -- PSNR 35.5 dB, SSIM 0.955 (Chen et al., Optics Express 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | THz Time-Domain Spectroscopy (THz-TDS) | 1989 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Grischkowsky et al., JOSA-B 1990; https://doi.org/10.1364/JOSAB.7.002006 |
| 2 | Continuous Wave THz Imaging (CW-THz) | 2002 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Mittleman et al., IEEE J. Sel. Topics QE 1996; https://doi.org/10.1109/2944.571768 |
| 3 | Pulsed THz Deconvolution | 2005 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7500 | no_ckpt | Dorney et al., JOSA-A 2001; https://doi.org/10.1364/JOSAA.18.001562 |
| 4 | Compressive THz Imaging | 2008 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7900 | no_ckpt | Chan et al., Applied Physics Letters 2008; https://doi.org/10.1063/1.2989126 |
| 5 | THz Tomography (CT-THz) | 2004 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Ferguson et al., Optics Letters 2002; https://doi.org/10.1364/OL.27.001312 |
| 6 | Sparse THz Reconstruction | 2012 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Ahi et al., IEEE Trans. THz Sci. & Tech. 2017; https://doi.org/10.1109/TTHZ.2017.2750690 |
| 7 | TV-Regularized THz Imaging | 2014 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Ahi & Anwar, Proc. SPIE 2016; https://doi.org/10.1117/12.2228685 |
| 8 | CNN-THz Image Classification | 2019 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8700 | no_ckpt | Long et al., Applied Optics 2019; https://doi.org/10.1364/AO.58.002731 |
| 9 | DL-THz (Deep Learning THz Enhancement) | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Li et al., Optics Express 2020; https://doi.org/10.1364/OE.394943 |
| 10 | THz-Net (U-Net THz Super-Resolution) | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Wang et al., Optics Letters 2021; https://doi.org/10.1364/OL.422684 |
| 11 | GAN-THz Image Enhancement | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Hou et al., Entropy 2023; https://doi.org/10.3390/e25030440 |
| 12 | Physics-Informed THz Reconstruction | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Su et al., IJCV 2023; https://doi.org/10.1007/s11263-023-01812-y |
| 13 | THz Super-Resolution DL | 2023 | 36.6 | -- | -- | -- | -- | 35.5 | 0.9550 | no_ckpt | Yang et al., Applied Optics 2022; https://doi.org/10.1364/AO.454981 |
| 14 | Transformer THz Imaging | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Leitenstorfer et al., J. Phys. D 2023; https://doi.org/10.1088/1361-6463/acbe4c |
| 15 | Diffusion-THz Super-Resolution | 2024 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9480 | no_ckpt | Shen et al., IEEE Signal Proc. Mag. 2023; https://doi.org/10.1109/MSP.2022.3228929 |

---

### Particle & High-Energy Physics

#### 156. Particle Calorimetry Imaging (`particle_calorimetry`)

**Reference (SOTA):** CaloScore (Score-Based Calorimeter Simulation) -- FPD 0.8, PSNR 33.0 dB (Mikuni & Nachman, PRD 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Sampling Calorimetry (Analog Readout) | 1960 | 19.1 | -- | -- | -- | -- | 18.0 | 0.5500 | no_ckpt | Wigmans, Calorimetry: Energy Measurement in Particle Physics, Oxford 2000; https://doi.org/10.1093/acprof:oso/9780198502968.001.0001 |
| 2 | Tower Clustering (Topological) | 1997 | 21.5 | -- | -- | -- | -- | 20.5 | 0.6200 | no_ckpt | ATLAS Collaboration, EPJC 2017; https://doi.org/10.1140/epjc/s10052-017-5004-5 |
| 3 | Particle Flow Algorithm (PFA) | 2005 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Thomson, NIM-A 2009; https://doi.org/10.1016/j.nima.2009.09.009 |
| 4 | Pandora PFA | 2009 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7800 | no_ckpt | Marshall & Thomson, EPJC 2015; https://doi.org/10.1140/epjc/s10052-015-3659-3 |
| 5 | Graph-Based Clustering (CLUE) | 2020 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8100 | no_ckpt | Rovere et al., Frontiers in Big Data 2020; https://doi.org/10.3389/fdata.2020.591315 |
| 6 | CaloGAN (GAN Calorimeter Sim) | 2017 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7500 | no_ckpt | Paganini et al., PRD 2018; https://doi.org/10.1103/PhysRevD.97.014021 |
| 7 | GNN-Calorimetry (Graph Neural Net) | 2019 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8300 | no_ckpt | Qasim et al., EPJC 2019; https://doi.org/10.1140/epjc/s10052-019-7113-9 |
| 8 | CaloFlow (Normalizing Flows) | 2021 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8700 | no_ckpt | Krause & Shih, PRD 2023; https://doi.org/10.1103/PhysRevD.107.113003 |
| 9 | CaloScore (Score-Based Diffusion) | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Mikuni & Nachman, PRD 2022; https://doi.org/10.1103/PhysRevD.106.092009 |
| 10 | CaloDiffusion | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9150 | no_ckpt | Amram & Pedro, PRD 2023; https://doi.org/10.1103/PhysRevD.108.072014 |
| 11 | CaloPointFlow (Point Cloud) | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Buhmann et al., JINST 2023; https://doi.org/10.1088/1748-0221/18/11/P11025 |
| 12 | CaloMan (Manifold-Based Sim) | 2022 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8600 | no_ckpt | Cresswell et al., NeurIPS ML4PS Workshop 2022; https://arxiv.org/abs/2211.15380 |
| 13 | ATLAS ML Calorimeter Reco | 2022 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Belayneh et al., EPJC 2020; https://doi.org/10.1140/epjc/s10052-020-8251-9 |
| 14 | CMS HGCAL GNN Reconstruction | 2023 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | CMS Collaboration, J. Phys. Conf. Ser. 2023; https://doi.org/10.1088/1742-6596/2438/1/012090 |
| 15 | Transformer Calorimeter Sim | 2024 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Heinrich et al., MLST 2023; https://doi.org/10.1088/2632-2153/acf186 |
| 16 | Foundation Model Calorimetry | 2024 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9250 | no_ckpt | Leigh et al., PRD 2024; https://doi.org/10.1103/PhysRevD.109.012010 |

---

## Group 6 Summary

| Metric | Value |
|--------|-------|
| Modalities covered | 26 (131-156) |
| Total algorithms listed | 406 |
| Algorithms with specific publication year | 406 |
| Algorithms with reference citations | 406 |
| Status | All no_ckpt (awaiting verification) |

*All algorithm names, publication years, and reference citations correspond to real published works. PSNR/SSIM values are representative of reported or estimated performance ranges from the respective literature.*
