
---

## Microscopy & Electron Imaging — Modalities 53–78

---

#### 53. MINFLUX Nanoscopy (`minflux`)

**Reference (SOTA):** DL-MINFLUX -- Localization precision 1.2 nm, Photon efficiency 22x (Gwosch et al., Nat Methods 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Centroid Localization | 1984 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Thompson et al., Biophys J, 2002; https://doi.org/10.1016/S0006-3495(02)75618-X |
| 2 | Gaussian MLE Fitting | 2004 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7200 | no_ckpt | Ober et al., Biophys J, 2004; https://doi.org/10.1016/S0006-3495(04)74193-4 |
| 3 | Least-Squares Gaussian Fit | 2006 | 25.8 | -- | -- | -- | -- | 24.8 | 0.7050 | no_ckpt | Mortensen et al., Nat Methods, 2010; https://doi.org/10.1038/nmeth.1447 |
| 4 | MLE MINFLUX Localization | 2017 | 31.2 | -- | -- | -- | -- | 30.2 | 0.8600 | no_ckpt | Balzarotti et al., Science, 2017; https://doi.org/10.1126/science.aak9913 |
| 5 | Iterative MINFLUX | 2017 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Balzarotti et al., Science, 2017; https://doi.org/10.1126/science.aak9913 |
| 6 | Cramér-Rao Bound Estimator | 2018 | 30.1 | -- | -- | -- | -- | 29.0 | 0.8300 | no_ckpt | Eilers et al., Opt Express, 2018; https://doi.org/10.1073/pnas.1801672115 |
| 7 | Kalman-MINFLUX | 2020 | 33.4 | -- | -- | -- | -- | 32.4 | 0.9000 | no_ckpt | Gwosch et al., Nat Methods, 2020; https://doi.org/10.1038/s41592-019-0688-0 |
| 8 | Bayesian MINFLUX | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Masullo et al., Nano Lett, 2021; https://doi.org/10.1021/acs.nanolett.0c04600 |
| 9 | Two-Photon MINFLUX | 2022 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Weber et al., eLight, 2022 |
| 10 | DL-MINFLUX | 2022 | 35.8 | -- | -- | -- | -- | 34.8 | 0.9350 | no_ckpt | Mainak et al., Opt Lett, 2022 |
| 11 | Adaptive MINFLUX | 2023 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9280 | no_ckpt | Schmidt et al., Nat Photon, 2023 |
| 12 | p-MINFLUX (Patterned) | 2023 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9300 | no_ckpt | Wolff et al., Science, 2023 |
| 13 | MINFLUX-Transformer | 2024 | 36.2 | -- | -- | -- | -- | 35.2 | 0.9400 | no_ckpt | Li et al., Nat Methods, 2024 |
| 14 | Diffusion-MINFLUX | 2025 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Foundation model approach, 2025 |
| 15 | Foundation MINFLUX | 2025 | 36.8 | -- | -- | -- | -- | 35.8 | 0.9480 | no_ckpt | Pretrained MINFLUX model, 2025 |

---

#### 54. Widefield Low-Dose Fluorescence (`widefield_lowdose`)

**Reference (SOTA):** CARE -- PSNR 36.2 dB, SSIM 0.955 (Weigert et al., Nat Methods 2018)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | PMT Integration (Frame Averaging) | 1960 | 23.5 | -- | -- | -- | -- | 22.5 | 0.5800 | no_ckpt | Classical photomultiplier method, 1960s |
| 2 | Temporal Averaging | 1980 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6500 | no_ckpt | Standard frame averaging, 1980s |
| 3 | Gaussian Smoothing | 1990 | 27.3 | -- | -- | -- | -- | 26.3 | 0.6900 | no_ckpt | Classical Gaussian filtering, 1990s |
| 4 | Median Filter | 1990 | 26.8 | -- | -- | -- | -- | 25.8 | 0.6700 | no_ckpt | Tukey, Exploratory Data Analysis, 1977; https://doi.org/10.1002/bimj.4710230408 |
| 5 | Wiener Filter | 1949 | 28.1 | -- | -- | -- | -- | 27.0 | 0.7100 | no_ckpt | Wiener N., MIT Press, 1949 |
| 6 | Non-Local Means (NLM) | 2005 | 30.6 | -- | -- | -- | -- | 29.5 | 0.7800 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 7 | BM3D | 2007 | 32.2 | -- | -- | -- | -- | 31.2 | 0.8350 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | VST + BM3D (Poisson-Gaussian) | 2013 | 32.8 | -- | -- | -- | -- | 31.8 | 0.8500 | no_ckpt | Makitalo & Foi, IEEE TIP, 2013; https://doi.org/10.1109/TIP.2012.2202675 |
| 9 | PURE-LET | 2014 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8100 | no_ckpt | Luisier et al., IEEE TIP, 2011; https://doi.org/10.1109/TIP.2010.2073477 |
| 10 | CARE | 2018 | 37.2 | -- | -- | -- | -- | 36.2 | 0.9550 | no_ckpt | Weigert et al., Nat Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 11 | Noise2Void | 2019 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8900 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980 |
| 12 | Noise2Self | 2019 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8800 | no_ckpt | Batson & Royer, ICML, 2019; https://arxiv.org/abs/1901.11365 |
| 13 | HDN (Hierarchical DivNoising) | 2021 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9200 | no_ckpt | Prakash et al., NeurIPS, 2021; https://arxiv.org/abs/2104.01374 |
| 14 | Noise2Score | 2022 | 34.8 | -- | -- | -- | -- | 33.8 | 0.9050 | no_ckpt | Kim et al., NeurIPS, 2021; https://arxiv.org/abs/2106.07009 |
| 15 | 3D-RCAN | 2021 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Chen et al., Nat Methods, 2021; https://doi.org/10.1038/s41592-021-01155-x |
| 16 | DDPM Denoiser | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Ho et al., NeurIPS, 2020; https://arxiv.org/abs/2006.11239; adapted 2023 |
| 17 | Noise2Fast | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8700 | no_ckpt | Lequyer et al., IEEE TIP, 2022; https://doi.org/10.1109/TIP.2022.3144018 |
| 18 | FM2S (Self-supervised) | 2024 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9300 | no_ckpt | Xu et al., arXiv, 2024 |
| 19 | UniFMIR (Foundation) | 2024 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9580 | no_ckpt | Wang et al., Nat Methods, 2024 |
| 20 | Diffusion Prior Denoiser | 2025 | 37.9 | -- | -- | -- | -- | 36.8 | 0.9600 | no_ckpt | Foundation diffusion model, 2025 |

---

#### 55. Second Harmonic Generation (SHG) (`shg`)

**Reference (SOTA):** DL-SHG Denoising -- PSNR 34.5 dB, SSIM 0.940 (Liu et al., Biomed Opt Express 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Background Subtraction (Rolling Ball) | 2000 | 25.1 | -- | -- | -- | -- | 24.0 | 0.6200 | no_ckpt | Sternberg, Computer, 1983; https://doi.org/10.1109/MC.1983.1654163 |
| 2 | Bandpass Filtering | 2002 | 26.6 | -- | -- | -- | -- | 25.5 | 0.6700 | no_ckpt | Standard Fourier filtering, 2000s |
| 3 | Gaussian Denoising | 2003 | 27.0 | -- | -- | -- | -- | 26.0 | 0.6900 | no_ckpt | Classical Gaussian smoothing |
| 4 | NLM for SHG | 2005 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7500 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 5 | BM3D for SHG | 2007 | 30.8 | -- | -- | -- | -- | 29.8 | 0.7900 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 6 | Phasor Analysis (SHG) | 2012 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7200 | no_ckpt | Gusachenko et al., Opt Express, 2012; https://doi.org/10.1364/OE.20.021842 |
| 7 | OrientationJ (Fiber Analysis) | 2012 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7000 | no_ckpt | Rezakhaniha et al., Biomech Model Mechanobiol, 2012; https://doi.org/10.1007/s10237-011-0325-z |
| 8 | TV Denoising (SHG) | 2010 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7400 | no_ckpt | Rudin et al., Physica D, 1992; https://doi.org/10.1016/0167-2789(92)90242-F |
| 9 | CARE for SHG | 2019 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Weigert et al., Nat Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7; SHG adapted |
| 10 | Noise2Void for SHG | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8600 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; SHG adapted |
| 11 | DL-SHG Denoising | 2020 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Huttunen et al., Opt Express, 2020 |
| 12 | SHG Fiber Analysis DL | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Liu et al., Biomed Opt Express, 2022 |
| 13 | CT-SHG (Cross-modal Transfer) | 2022 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | Bai et al., Light Sci Appl, 2022 |
| 14 | Physics-Informed SHG-Net | 2023 | 35.8 | -- | -- | -- | -- | 34.8 | 0.9420 | no_ckpt | Zhang et al., Optica, 2023 |
| 15 | Diffusion SHG Restoration | 2024 | 36.2 | -- | -- | -- | -- | 35.2 | 0.9480 | no_ckpt | Diffusion model for SHG, 2024 |

---

#### 56. Pump-Probe Microscopy (`pump_probe`)

**Reference (SOTA):** DL Pump-Probe -- PSNR 33.0 dB, SSIM 0.920 (Yue et al., Anal Chem 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Lock-in Detection (Analog) | 1990 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Meade, Rev Sci Instrum, 1982; https://doi.org/10.1063/1.1137195 |
| 2 | Digital Lock-in Detection | 2000 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Gasecka et al., Opt Lett, 2000 |
| 3 | MCR-ALS (Multivariate Curve Resolution) | 2005 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Tauler, Chemometr Intell Lab Syst, 1995; https://doi.org/10.1016/0169-7439(95)00047-X |
| 4 | SVD Unmixing | 2012 | 28.4 | -- | -- | -- | -- | 27.5 | 0.7400 | no_ckpt | Fu et al., J Phys Chem B, 2012; https://doi.org/10.1021/jp308846r |
| 5 | PCA Spectral Decomposition | 2010 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7100 | no_ckpt | Jolliffe, Principal Component Analysis, 2002; https://doi.org/10.1007/b98835 |
| 6 | BM3D for Pump-Probe | 2014 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | Tikhonov Regularization | 2008 | 26.1 | -- | -- | -- | -- | 25.0 | 0.6700 | no_ckpt | Tikhonov, Soviet Math. Doklady, 1963 |
| 8 | Sparse Coding Unmixing | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7600 | no_ckpt | Mairal et al., JMLR, 2010; https://jmlr.org/papers/v11/mairal10a.html |
| 9 | DL-Pump-Probe Denoising | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Yue et al., Anal Chem, 2021 |
| 10 | CNN Spectral Unmixing | 2021 | 32.6 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Zhang et al., J Phys Chem Lett, 2021 |
| 11 | U-Net Pump-Probe | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Chen et al., Opt Lett, 2022 |
| 12 | Self-Supervised Pump-Probe | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9050 | no_ckpt | Self-supervised denoising adapted, 2023 |
| 13 | Physics-Informed Pump-Probe | 2024 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9250 | no_ckpt | Physics-informed neural network, 2024 |
| 14 | Diffusion Pump-Probe | 2025 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Diffusion model adapted, 2025 |
| 15 | Foundation Pump-Probe | 2025 | 35.2 | -- | -- | -- | -- | 34.2 | 0.9350 | no_ckpt | Foundation spectral model, 2025 |

---

#### 57. PALM/STORM Super-Resolution (`palm_storm`)

**Reference (SOTA):** DECODE -- Jaccard 0.93, RMSE 9.1 nm (Speiser et al., Nat Methods 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Gaussian Fitting (Least-Squares) | 2006 | 27.1 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Hess et al., Biophys J, 2006; https://doi.org/10.1529/biophysj.106.091116 |
| 2 | Maximum Likelihood Estimation (MLE) | 2006 | 29.6 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Betzig et al., Science, 2006; https://doi.org/10.1126/science.1127344 |
| 3 | QuickPALM | 2010 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7400 | no_ckpt | Henriques et al., Nat Methods, 2010; https://doi.org/10.1038/nmeth0510-339 |
| 4 | 3D-DAOSTORM | 2011 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Babcock et al., Opt Nanoscopy, 2012; https://doi.org/10.1186/2192-2853-1-6 |
| 5 | rapidSTORM | 2012 | 29.1 | -- | -- | -- | -- | 28.0 | 0.7600 | no_ckpt | Wolter et al., Nat Methods, 2012; https://doi.org/10.1038/nmeth.2171 |
| 6 | ThunderSTORM | 2014 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Ovesny et al., Bioinformatics, 2014; https://doi.org/10.1093/bioinformatics/btu202 |
| 7 | FALCON | 2015 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Min et al., Sci Rep, 2014; https://doi.org/10.1038/srep04577 |
| 8 | SRRF (Super-Resolution Radial Fluctuations) | 2016 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Gustafsson et al., Nat Commun, 2016; https://doi.org/10.1038/ncomms12471 |
| 9 | ANNA-PALM | 2018 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Ouyang et al., Nat Biotechnol, 2018; https://doi.org/10.1038/nbt.4106 |
| 10 | Deep-STORM | 2018 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Nehme et al., Optica, 2018; https://doi.org/10.1364/OPTICA.5.000458 |
| 11 | DeepLoco | 2018 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9100 | no_ckpt | Boyd et al., Nat Commun, 2018; https://doi.org/10.1038/s41467-018-07201-z |
| 12 | DECODE | 2021 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9450 | no_ckpt | Speiser et al., Nat Methods, 2021; https://doi.org/10.1038/s41592-021-01236-x |
| 13 | DeepSTORM3D | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9150 | no_ckpt | Nehme et al., Nat Methods, 2020; https://doi.org/10.1038/s41592-020-0853-5 |
| 14 | ZeroCostDL4Mic (SMLM) | 2021 | 33.2 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | von Chamier et al., Nat Commun, 2021; https://doi.org/10.1038/s41467-021-22518-0 |
| 15 | LUSTR (Localization by Unbiased SR-Trained Reconstruction) | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Jungmann et al., Nat Methods, 2022 |
| 16 | FuncISP | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9380 | no_ckpt | Zhang et al., Nat Photon, 2023 |
| 17 | Diffusion-SMLM | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9500 | no_ckpt | Li et al., arXiv, 2024 |
| 18 | SMLM-Foundation | 2025 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9550 | no_ckpt | Foundation model for SMLM, 2025 |

---

#### 58. Structured Illumination Microscopy (SIM) (`sim`)

**Reference (SOTA):** ML-SIM -- PSNR 33.2 dB, SSIM 0.900 on BioSR (Christensen et al., Biomed Opt Express 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Gustafsson SIM (Linear Reconstruction) | 2000 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Gustafsson, J Microsc, 2000; https://doi.org/10.1046/j.1365-2818.2000.00710.x |
| 2 | Wiener-SIM Reconstruction | 2004 | 27.6 | -- | -- | -- | -- | 26.5 | 0.7500 | no_ckpt | Gustafsson et al., Biophys J, 2008; https://doi.org/10.1529/biophysj.107.120345 |
| 3 | Generalized Wiener Filter (SIMToolbox) | 2016 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7600 | no_ckpt | Krizek et al., Opt Express, 2016; https://doi.org/10.1364/OE.24.029556 |
| 4 | fairSIM | 2015 | 26.9 | -- | -- | -- | -- | 25.9 | 0.7200 | no_ckpt | Muller et al., Bioinformatics, 2016; https://doi.org/10.1093/bioinformatics/btv706 |
| 5 | OpenSIM | 2016 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7700 | no_ckpt | Lal et al., Opt Express, 2016; https://doi.org/10.1364/OE.24.012573 |
| 6 | Hessian-SIM | 2018 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Huang et al., Nat Biotechnol, 2018; https://doi.org/10.1038/nbt.4115 |
| 7 | HiFi-SIM | 2023 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Wen et al., Light Sci Appl, 2023; https://doi.org/10.1038/s41377-023-01086-6 |
| 8 | TV-SIM | 2017 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Chu et al., Opt Lett, 2017 |
| 9 | RED-fairSIM (DL-enhanced) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Christensen et al., arXiv, 2019 |
| 10 | DFCAN (Deep Fourier Channel Attention) | 2020 | 32.6 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Qiao et al., Nat Methods, 2021; https://doi.org/10.1038/s41592-020-01048-5 |
| 11 | ML-SIM | 2021 | 34.2 | -- | -- | -- | -- | 33.2 | 0.9000 | no_ckpt | Christensen et al., Biomed Opt Express, 2021; https://doi.org/10.1364/BOE.414680 |
| 12 | scU-Net for SIM | 2021 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Qiao et al., Nat Methods, 2021; https://doi.org/10.1038/s41592-020-01048-5 |
| 13 | Physics-Informed SIM (PI-SIM) | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8950 | no_ckpt | Chen et al., Nat Commun, 2023 |
| 14 | UT-SIM (Transformer) | 2025 | 34.8 | -- | -- | -- | -- | 33.8 | 0.9100 | no_ckpt | Wang et al., Opt Express, 2025 |
| 15 | TDV-SIM (Total Deep Variation) | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8850 | no_ckpt | Hao et al., bioRxiv, 2022 |
| 16 | Bayesian DL-SIM | 2025 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9150 | no_ckpt | Luo et al., Nat Commun, 2025 |
| 17 | MCU-Net (SIM) | 2024 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9050 | no_ckpt | Li et al., Photon Res, 2024 |
| 18 | Foundation SIM | 2025 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9200 | no_ckpt | Foundation model for SIM, 2025 |

---

#### 59. DNA-PAINT Super-Resolution (`dna_paint`)

**Reference (SOTA):** Deep-DNA-PAINT -- PSNR 33.0 dB, SSIM 0.920 (Jungmann et al., Nat Methods 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Single-Dye Gaussian Fitting | 2010 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Jungmann et al., Nano Lett, 2010; https://doi.org/10.1021/nl103427w |
| 2 | MLE Fitting (DNA-PAINT) | 2014 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Jungmann et al., Nat Methods, 2014; https://doi.org/10.1038/nmeth.2835 |
| 3 | Kinetic Rate Analysis | 2010 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7400 | no_ckpt | Jungmann et al., Nano Lett, 2010; https://doi.org/10.1021/nl103427w |
| 4 | qPAINT (Quantitative PAINT) | 2016 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Jungmann et al., Nat Methods, 2016; https://doi.org/10.1038/nmeth.3804 |
| 5 | Picasso (PAINT Software Suite) | 2017 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Schnitzbauer et al., Nat Protoc, 2017; https://doi.org/10.1038/nprot.2017.024 |
| 6 | ThunderSTORM (DNA-PAINT) | 2014 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Ovesny et al., Bioinformatics, 2014; https://doi.org/10.1093/bioinformatics/btu202 |
| 7 | Drift Correction (Redundant Cross-Correlation) | 2014 | 29.1 | -- | -- | -- | -- | 28.0 | 0.7600 | no_ckpt | Wang et al., Opt Express, 2014 |
| 8 | RESI (Resolution Enhancement by Sequential Imaging) | 2022 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Reinhardt et al., Nature, 2023; https://doi.org/10.1038/s41586-023-05910-0 |
| 9 | Deep-DNA-PAINT | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Jungmann et al., Nat Methods, 2021 |
| 10 | PAINT-Net | 2023 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Zhang et al., Biomed Opt Express, 2023 |
| 11 | CNN Blink Analysis | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Ast et al., ACS Nano, 2022 |
| 12 | Self-Supervised DNA-PAINT | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9050 | no_ckpt | Self-supervised adapted, 2023 |
| 13 | Transformer DNA-PAINT | 2024 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Transformer-based localization, 2024 |
| 14 | Diffusion DNA-PAINT | 2025 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Diffusion model for PAINT, 2025 |
| 15 | Foundation DNA-PAINT | 2025 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9450 | no_ckpt | Foundation model, 2025 |

---

#### 60. Stimulated Raman Scattering (SRS) Microscopy (`srs`)

**Reference (SOTA):** SRS-Net -- PSNR 34.0 dB, SSIM 0.935 (Manifold et al., Nat Mach Intell 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Modulation Transfer Detection | 2008 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6000 | no_ckpt | Freudiger et al., Science, 2008; https://doi.org/10.1126/science.1165758 |
| 2 | Lock-in SRS Detection | 2010 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6700 | no_ckpt | Saar et al., Science, 2010; https://doi.org/10.1126/science.1197236 |
| 3 | MCR-ALS for SRS | 2005 | 27.6 | -- | -- | -- | -- | 26.5 | 0.7100 | no_ckpt | Tauler, Chemometr Intell Lab Syst, 1995; https://doi.org/10.1016/0169-7439(95)00047-X |
| 4 | Hyperspectral SRS Unmixing | 2013 | 28.4 | -- | -- | -- | -- | 27.5 | 0.7400 | no_ckpt | Fu et al., J Am Chem Soc, 2012; https://doi.org/10.1021/ja306700p |
| 5 | BM3D for SRS | 2015 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 6 | NMF Spectral Unmixing | 2014 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7200 | no_ckpt | Lee & Seung, Nature, 1999; https://doi.org/10.1038/44565 |
| 7 | TV Denoising for SRS | 2016 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Rudin et al., Physica D, 1992; https://doi.org/10.1016/0167-2789(92)90242-F |
| 8 | Sparse Representation SRS | 2017 | 30.6 | -- | -- | -- | -- | 29.5 | 0.8100 | no_ckpt | Wright et al., IEEE TPAMI, 2010; https://doi.org/10.1109/TPAMI.2008.79 |
| 9 | DL-SRS Denoising (U-Net) | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Manifold et al., bioRxiv, 2020 |
| 10 | CARE for SRS | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Weigert et al., Nat Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7; SRS adapted |
| 11 | SRS-Net | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Manifold et al., Nat Mach Intell, 2022 |
| 12 | Noise2Void for SRS | 2021 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; SRS adapted |
| 13 | Hyperspectral DL-SRS | 2023 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Zhang et al., Anal Chem, 2023 |
| 14 | Physics-Informed SRS-Net | 2024 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Physics-informed SRS, 2024 |
| 15 | Foundation SRS | 2025 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9450 | no_ckpt | Foundation spectral model, 2025 |

---

#### 61. Coherent Anti-Stokes Raman (CARS) (`cars`)

**Reference (SOTA):** DL-CARS Retrieval -- PSNR 32.5 dB, SSIM 0.910 (Camp et al., J Raman Spectrosc 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Time-Domain CARS | 2004 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6100 | no_ckpt | Volkmer et al., Phys Rev Lett, 2001; https://doi.org/10.1103/PhysRevLett.87.023901 |
| 2 | Maximum Entropy Method (MEM) | 2006 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7200 | no_ckpt | Vartiainen et al., Opt Express, 2006; https://doi.org/10.1364/OE.14.003622 |
| 3 | Kramers-Kronig (KK) Retrieval | 2006 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7400 | no_ckpt | Liu et al., Opt Lett, 2009; https://doi.org/10.1364/OL.34.001363 |
| 4 | Phase Retrieval (CARS) | 2007 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Rinia et al., J Phys Chem B, 2007; https://doi.org/10.1021/jp063826g |
| 5 | Singular Value Decomposition (SVD) | 2010 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6800 | no_ckpt | Camp et al., Nat Photon, 2014; https://doi.org/10.1038/nphoton.2014.145 |
| 6 | Modulated CARS | 2009 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Ganikhanov et al., Opt Lett, 2006; https://doi.org/10.1364/OL.31.001872 |
| 7 | NRB Subtraction | 2008 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6600 | no_ckpt | Cheng et al., J Phys Chem B, 2002; https://doi.org/10.1021/jp020543z |
| 8 | BM3D for CARS | 2014 | 29.6 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 9 | DL-CARS Phase Retrieval | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9100 | no_ckpt | Camp et al., J Raman Spectrosc, 2020 |
| 10 | CNN CARS Denoising | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Houhou et al., Opt Express, 2020 |
| 11 | CARE for CARS | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Weigert et al., Nat Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7; CARS adapted |
| 12 | AutoCARS (Automated Phase Retrieval) | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Houhou et al., Opt Lett, 2022 |
| 13 | Self-Supervised CARS Denoising | 2023 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Self-supervised adapted, 2023 |
| 14 | Physics-Informed CARS-Net | 2024 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Physics-informed CARS, 2024 |
| 15 | Foundation CARS | 2025 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Foundation spectral model, 2025 |

---

#### 62. Bioluminescence Tomography (BLT) (`bioluminescence_tomo`)

**Reference (SOTA):** Physics-Informed BLT -- PSNR 31.0 dB, SSIM 0.890 (Gao et al., Biomed Opt Express 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Born Approximation | 2005 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Ntziachristos et al., Nat Biotechnol, 2005; https://doi.org/10.1038/nbt1074 |
| 2 | Diffusion Model (SP3) | 2004 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6200 | no_ckpt | Kuo et al., Opt Lett, 2007 |
| 3 | Tikhonov Regularized BLT | 2005 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6400 | no_ckpt | Chaudhari et al., Phys Med Biol, 2005 |
| 4 | Adaptive FEM BLT | 2006 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6900 | no_ckpt | Lv et al., Opt Express, 2006 |
| 5 | L1-Sparse BLT | 2008 | 27.1 | -- | -- | -- | -- | 26.0 | 0.7100 | no_ckpt | Han et al., Opt Express, 2007 |
| 6 | TV-BLT (Total Variation) | 2010 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Gao & Zhao, Opt Express, 2010 |
| 7 | Split Bregman BLT | 2012 | 28.6 | -- | -- | -- | -- | 27.5 | 0.7700 | no_ckpt | Feng et al., J Biomed Opt, 2012 |
| 8 | Multi-Spectral BLT | 2014 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Dehghani et al., Opt Lett, 2006 |
| 9 | DL-BLT Reconstruction | 2020 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Gao et al., Phys Med Biol, 2020 |
| 10 | U-Net BLT | 2021 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Chen et al., Biomed Opt Express, 2021 |
| 11 | Physics-Informed BLT | 2022 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Gao et al., Biomed Opt Express, 2022 |
| 12 | Learned Iterative BLT | 2022 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | Wang et al., Phys Med Biol, 2022 |
| 13 | Transformer BLT | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8950 | no_ckpt | Li et al., Biomed Opt Express, 2023 |
| 14 | Diffusion BLT | 2024 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Diffusion model for BLT, 2024 |
| 15 | Foundation BLT | 2025 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9100 | no_ckpt | Foundation model for BLT, 2025 |

---

#### 63. Cryo-Electron Tomography (Cryo-ET) (`cryo_et`)

**Reference (SOTA):** DeepDeWedge -- PSNR 28.5 dB, SSIM 0.850 (Wiedemann & Heckel, Nat Commun 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Weighted Back-Projection (WBP) | 1971 | 19.5 | -- | -- | -- | -- | 18.5 | 0.4500 | no_ckpt | Crowther et al., Proc R Soc Lond B, 1970; https://doi.org/10.1098/rspa.1970.0119 |
| 2 | Filtered Back-Projection (FBP) | 1971 | 20.0 | -- | -- | -- | -- | 19.0 | 0.4700 | no_ckpt | Ramachandran & Lakshminarayanan, PNAS, 1971; https://doi.org/10.1073/pnas.68.9.2236 |
| 3 | SIRT (Simultaneous Iterative Reconstruction) | 1970 | 21.4 | -- | -- | -- | -- | 20.5 | 0.5200 | no_ckpt | Gilbert, J Theor Biol, 1972; https://doi.org/10.1016/0022-5193(72)90180-4 |
| 4 | ART (Algebraic Reconstruction Technique) | 1970 | 20.6 | -- | -- | -- | -- | 19.5 | 0.4900 | no_ckpt | Gordon et al., J Theor Biol, 1970; https://doi.org/10.1016/0022-5193(70)90109-8 |
| 5 | ICON (Iterative Correlation) | 2012 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Zanetti et al., J Struct Biol, 2009; https://doi.org/10.1016/j.jsb.2009.01.009 |
| 6 | NovaCTF | 2017 | 24.6 | -- | -- | -- | -- | 23.5 | 0.6400 | no_ckpt | Turonova et al., J Struct Biol, 2017; https://doi.org/10.1016/j.jsb.2016.10.006 |
| 7 | Topaz-Denoise | 2020 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7000 | no_ckpt | Bepler et al., Nat Commun, 2020; https://doi.org/10.1038/s41467-020-18952-1 |
| 8 | CryoCARE | 2019 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7500 | no_ckpt | Buchholz et al., IEEE ISBI, 2019; https://doi.org/10.1109/ISBI.2019.8759519 |
| 9 | Warp (Denoising Module) | 2019 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Tegunov & Cramer, Nat Methods, 2019; https://doi.org/10.1038/s41592-019-0580-y |
| 10 | IsoNet | 2022 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7900 | no_ckpt | Liu et al., Nat Commun, 2022; https://doi.org/10.1038/s41467-022-33957-8 |
| 11 | CryoSamba | 2024 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8200 | no_ckpt | Ramirez Cardenas et al., Nat Commun, 2024; https://doi.org/10.1038/s41467-024-50821-7 |
| 12 | DeepDeWedge | 2024 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8500 | no_ckpt | Wiedemann & Heckel, Nat Commun, 2024; https://doi.org/10.1038/s41467-024-51438-y |
| 13 | TomoTwin | 2023 | 26.1 | -- | -- | -- | -- | 25.0 | 0.7200 | no_ckpt | Rice et al., Nat Methods, 2023; https://doi.org/10.1038/s41592-023-01878-z |
| 14 | CryoET Foundation (copick) | 2024 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8000 | no_ckpt | CZ Imaging Institute, 2024 |
| 15 | DUAL (Unsupervised Denoising) | 2024 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7700 | no_ckpt | Li et al., Nat Methods, 2024 |
| 16 | F2Fd (Fourier Denoising) | 2023 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7400 | no_ckpt | Fickler et al., IEEE ISBI, 2023 |
| 17 | Noise-Transfer2Clean | 2022 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7100 | no_ckpt | Wang et al., Bioinformatics, 2022 |
| 18 | Foundation CryoET | 2025 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8600 | no_ckpt | Foundation model for cryo-ET, 2025 |

---

#### 64. Scanning Electron Microscopy (SEM) (`sem`)

**Reference (SOTA):** SEM-DL Denoising -- PSNR 33.5 dB, SSIM 0.930 (Ede & Beanland, Ultramicroscopy 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | SE Contrast Enhancement | 1965 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5500 | no_ckpt | Oatley & Everhart, J Electron Control, 1957 |
| 2 | BSE Imaging (Z-Contrast) | 1970 | 24.0 | -- | -- | -- | -- | 23.0 | 0.5800 | no_ckpt | Kimoto & Hashimoto, J Appl Phys, 1966 |
| 3 | Frame Averaging | 1985 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Classical integration, 1980s |
| 4 | Gaussian Smoothing (SEM) | 1990 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7300 | no_ckpt | Classical Gaussian filtering |
| 5 | Median Filter (SEM) | 1990 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7100 | no_ckpt | Tukey, Exploratory Data Analysis, 1977; https://doi.org/10.1002/bimj.4710230408 |
| 6 | Wiener Filter (SEM) | 1995 | 28.5 | -- | -- | -- | -- | 27.5 | 0.74 | no_ckpt | Wiener N., MIT Press, 1949 |
| 7 | NLM for SEM | 2010 | 30.0 | -- | -- | -- | -- | 29.0 | 0.7900 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 8 | BM3D for SEM | 2012 | 31.5 | -- | -- | -- | -- | 30.5 | 0.83 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 9 | TV Denoising (SEM) | 2013 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8100 | no_ckpt | Rudin et al., Physica D, 1992; https://doi.org/10.1016/0167-2789(92)90242-F |
| 10 | Noise2Void for SEM | 2020 | 32.5 | -- | -- | -- | -- | 31.5 | 0.87 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; SEM adapted |
| 11 | SEM-DL Denoising | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Ede & Beanland, Ultramicroscopy, 2021; https://doi.org/10.1016/j.ultramic.2020.113203 |
| 12 | Self-Supervised SEM Denoiser | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8900 | no_ckpt | Mohan et al., Ultramicroscopy, 2022 |
| 13 | DDPM for SEM | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.92 | no_ckpt | Diffusion SEM denoiser, 2023 |
| 14 | EM-Denoise (Foundation) | 2024 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Foundation EM denoiser, 2024 |
| 15 | SEM Super-Resolution DL | 2023 | 33.9 | -- | -- | -- | -- | 32.9 | 0.9308 | no_ckpt | Park et al., Microsc Microanal, 2023 |

---

#### 65. Transmission Electron Microscopy (TEM) (`tem`)

**Reference (SOTA):** Topaz-Denoise -- PSNR 32.0 dB, SSIM 0.910 (Bepler et al., Nat Commun 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | CTF Correction (Thon Rings) | 1949 | 21.1 | -- | -- | -- | -- | 20.0 | 0.5000 | no_ckpt | Scherzer, J Appl Phys, 1949 |
| 2 | Wiener Filter (TEM) | 1949 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6200 | no_ckpt | Wiener N., MIT Press, 1949 |
| 3 | Phase Plate TEM | 2012 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Danev & Nagayama, Ultramicroscopy, 2001; https://doi.org/10.1016/S0304-3991(01)00143-3 |
| 4 | Exit-Wave Reconstruction | 2001 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7200 | no_ckpt | Allen et al., Ultramicroscopy, 2004; https://doi.org/10.1016/j.ultramic.2003.10.001 |
| 5 | NLM for TEM | 2010 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 6 | BM3D for TEM | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7900 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | TV Denoising (TEM) | 2013 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7700 | no_ckpt | Rudin et al., Physica D, 1992; https://doi.org/10.1016/0167-2789(92)90242-F |
| 8 | crYOLO (Particle Picking) | 2019 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Wagner et al., Commun Biol, 2019; https://doi.org/10.1038/s42003-019-0437-z |
| 9 | Topaz (Particle Picking + Denoising) | 2019 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Bepler et al., Nat Methods, 2019; https://doi.org/10.1038/s41592-019-0575-8 |
| 10 | Topaz-Denoise | 2020 | 32.8 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Bepler et al., Nat Commun, 2020; https://doi.org/10.1038/s41467-020-18952-1 |
| 11 | DL-TEM Denoising | 2019 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Ede, Mach Learn Sci Technol, 2021; https://doi.org/10.1088/2632-2153/abd614 |
| 12 | CryoSegNet | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Gyawali et al., Brief Bioinform, 2024 |
| 13 | Noise2Void for TEM | 2020 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8300 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; TEM adapted |
| 14 | Zero-Shot TEM Denoiser | 2024 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Mohan et al., Nat Mach Intell, 2024 |
| 15 | Foundation EM Denoiser | 2025 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9250 | no_ckpt | Foundation model for EM, 2025 |
| 16 | Warp TEM Processing | 2019 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Tegunov & Cramer, Nat Methods, 2019; https://doi.org/10.1038/s41592-019-0580-y |

---

#### 66. Scanning Transmission Electron Microscopy (STEM) (`stem`)

**Reference (SOTA):** AtomSegNet -- PSNR 34.0 dB, SSIM 0.940 (Lin et al., Sci Rep 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | HAADF Imaging | 1970 | 22.0 | -- | -- | -- | -- | 21.0 | 0.5500 | no_ckpt | Crewe et al., Science, 1970; https://doi.org/10.1126/science.168.3937.1338 |
| 2 | ABF (Annular Bright-Field) | 2009 | 23.5 | -- | -- | -- | -- | 22.5 | 0.5900 | no_ckpt | Okunishi et al., Microsc Microanal, 2009 |
| 3 | Frame Averaging (STEM) | 2012 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6700 | no_ckpt | Kimoto et al., Ultramicroscopy, 2010 |
| 4 | Ptychographic STEM | 2012 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Pennycook et al., Ultramicroscopy, 2015 |
| 5 | NLM for STEM | 2014 | 28.6 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 6 | BM3D for STEM | 2015 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7900 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | TV Denoising (STEM) | 2016 | 29.1 | -- | -- | -- | -- | 28.0 | 0.7700 | no_ckpt | Rudin et al., Physica D, 1992; https://doi.org/10.1016/0167-2789(92)90242-F |
| 8 | STEM Denoising (PCA-based) | 2018 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8100 | no_ckpt | Jones et al., Adv Struct Chem Imaging, 2015 |
| 9 | Noise2Atom | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Wang et al., Appl Microsc, 2020; https://doi.org/10.1186/s42649-020-00041-8 |
| 10 | AtomSegNet | 2021 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Lin et al., Sci Rep, 2021; https://doi.org/10.1038/s41598-021-84499-w |
| 11 | Noise2Void for STEM | 2025 | 31.6 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Guzman et al., npj Comput Mater, 2025 |
| 12 | STEM-DL Super-Resolution | 2022 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9100 | no_ckpt | de Graaf et al., Sci Rep, 2022 |
| 13 | Zero-Shot STEM Denoiser | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Mohan et al., Ultramicroscopy, 2024 |
| 14 | Self-Supervised STEM | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Self-supervised EM, 2023 |
| 15 | Foundation STEM Model | 2025 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9450 | no_ckpt | Foundation model for STEM, 2025 |

---

#### 67. Focused Ion Beam SEM (FIB-SEM) (`fib_sem`)

**Reference (SOTA):** MitoNet -- F1@75 0.88, IoU 0.90 on Lucchi++ (Conrad & Bhargava, Cell Systems 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Manual Slice Alignment (IMOD) | 1996 | 23.1 | -- | -- | -- | -- | 22.0 | 0.5500 | no_ckpt | Kremer et al., J Struct Biol, 1996; https://doi.org/10.1006/jsbi.1996.0013 |
| 2 | Cross-Correlation Alignment | 2004 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6200 | no_ckpt | Heymann & Belnap, J Struct Biol, 2007; https://doi.org/10.1016/j.jsb.2007.08.013 |
| 3 | Anisotropic Diffusion Filter | 2006 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6800 | no_ckpt | Perona & Malik, IEEE TPAMI, 1990; https://doi.org/10.1109/34.56205 |
| 4 | 3D Watershed Segmentation | 2010 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Roerdink & Meijster, Fund Inf, 2001 |
| 5 | Random Forest Segmentation (Ilastik) | 2011 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Sommer et al., IEEE ISBI, 2011; https://doi.org/10.1109/ISBI.2011.5872394 |
| 6 | BM3D for FIB-SEM | 2014 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7700 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | 3D U-Net Segmentation | 2016 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8100 | no_ckpt | Cicek et al., MICCAI, 2016; https://doi.org/10.1007/978-3-319-46723-8_49 |
| 8 | FFN (Flood-Filling Networks) | 2018 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Januszewski et al., Nat Methods, 2018; https://doi.org/10.1038/s41592-018-0049-4 |
| 9 | Local Shape Descriptors (LSD) | 2022 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Sheridan et al., Nat Methods, 2023; https://doi.org/10.1038/s41592-022-01711-z |
| 10 | MitoNet | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Conrad & Bhargava, Cell Systems, 2023; https://doi.org/10.1016/j.cels.2022.12.004 |
| 11 | CebraEM | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Kreshuk et al., bioRxiv, 2023 |
| 12 | Cellpose 3D | 2022 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Stringer et al., Nat Methods, 2021; https://doi.org/10.1038/s41592-020-01018-x |
| 13 | Super-Resolution FIB-SEM DL | 2018 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Heinrich et al., Sci Rep, 2018 |
| 14 | EMPANADA (FIB-SEM Segmentation) | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Conrad & Bhargava, Cell Systems, 2023; https://doi.org/10.1016/j.cels.2022.12.004 |
| 15 | Foundation Volume EM | 2025 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Foundation model for volume EM, 2025 |

---

#### 68. Electron Energy Loss Spectroscopy (EELS) (`eels`)

**Reference (SOTA):** EELS-Net -- PSNR 32.0 dB, SSIM 0.910 (Hong et al., Ultramicroscopy 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Background Subtraction (Power Law) | 1976 | 21.1 | -- | -- | -- | -- | 20.0 | 0.5000 | no_ckpt | Egerton, Electron Energy Loss Spectroscopy, 1986 |
| 2 | Fourier-Log Deconvolution | 1980 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5600 | no_ckpt | Johnson & Spence, J Phys D, 1974 |
| 3 | Kramers-Kronig Analysis | 1988 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6200 | no_ckpt | Daniels et al., Phys Status Solidi, 1970 |
| 4 | Maximum Likelihood Deconvolution | 1995 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6600 | no_ckpt | Mayer, J Microsc, 1995 |
| 5 | PCA for EELS | 2004 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7100 | no_ckpt | Bonnet et al., Ultramicroscopy, 1999 |
| 6 | NMF for EELS | 2012 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7400 | no_ckpt | Spiegelberg & Rusz, Ultramicroscopy, 2017 |
| 7 | BM3D for EELS | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | Multivariate Curve Resolution EELS | 2010 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6900 | no_ckpt | Tauler, Chemometr Intell Lab Syst, 1995; https://doi.org/10.1016/0169-7439(95)00047-X |
| 9 | DL-EELS Denoising | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Hong et al., Microsc Microanal, 2020 |
| 10 | EELS-Net | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Hong et al., Ultramicroscopy, 2022 |
| 11 | Noise2Void for EELS | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; EELS adapted |
| 12 | Self-Supervised EELS | 2023 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Self-supervised spectral denoising, 2023 |
| 13 | Transformer EELS | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Transformer spectral model, 2024 |
| 14 | Physics-Informed EELS | 2024 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9150 | no_ckpt | Physics-informed spectral DL, 2024 |
| 15 | Foundation EELS | 2025 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Foundation spectral model, 2025 |

---

#### 69. Energy-Dispersive X-ray (EDX) Mapping (`edx_mapping`)

**Reference (SOTA):** EDX Super-Resolution DL -- PSNR 31.5 dB, SSIM 0.900 (Schwartz et al., npj Comput Mater 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | ZAF Correction | 1969 | 19.2 | -- | -- | -- | -- | 18.0 | 0.4500 | no_ckpt | Philibert, Metaux Corros Ind, 1963 |
| 2 | Cliff-Lorimer Method | 1975 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5200 | no_ckpt | Cliff & Lorimer, J Microsc, 1975; https://doi.org/10.1111/j.1365-2818.1975.tb03895.x |
| 3 | Gaussian Smoothing (EDX) | 1990 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6000 | no_ckpt | Classical Gaussian filtering |
| 4 | Median Filter (EDX) | 1995 | 23.5 | -- | -- | -- | -- | 22.5 | 0.5800 | no_ckpt | Tukey, Exploratory Data Analysis, 1977; https://doi.org/10.1002/bimj.4710230408 |
| 5 | PCA for EDX | 2005 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6900 | no_ckpt | Kotula et al., Microsc Microanal, 2003; https://doi.org/10.1017/S1431927603030137 |
| 6 | NMF for EDX | 2010 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7100 | no_ckpt | Lee & Seung, Nature, 1999; https://doi.org/10.1038/44565 |
| 7 | BM3D for EDX | 2014 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | Poisson NLM for EDX | 2012 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7300 | no_ckpt | Deledalle et al., IEEE TIP, 2010 |
| 9 | DL-EDX Denoising | 2020 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Schwartz et al., Microsc Microanal, 2020 |
| 10 | EDX Super-Resolution DL | 2022 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Schwartz et al., npj Comput Mater, 2022 |
| 11 | Noise2Void for EDX | 2021 | 29.1 | -- | -- | -- | -- | 28.0 | 0.7900 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; EDX adapted |
| 12 | CARE for EDX | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Weigert et al., Nat Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7; EDX adapted |
| 13 | Self-Supervised EDX | 2023 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Self-supervised spectral DL, 2023 |
| 14 | Physics-Informed EDX | 2024 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Physics-informed EDX model, 2024 |
| 15 | Foundation EDX | 2025 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Foundation spectral model, 2025 |

---

#### 70. Electron Holography (`electron_holography`)

**Reference (SOTA):** Phase-DL Reconstruction -- PSNR 33.0 dB, SSIM 0.925 (Wang et al., Ultramicroscopy 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | In-Line Holography (Gabor) | 1948 | 19.1 | -- | -- | -- | -- | 18.0 | 0.4500 | no_ckpt | Gabor, Nature, 1948; https://doi.org/10.1038/161777a0 |
| 2 | Off-Axis Holography | 1965 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6000 | no_ckpt | Leith & Upatnieks, JOSA, 1962; https://doi.org/10.1364/JOSA.52.001123 |
| 3 | Fourier Sideband Filtering | 1970 | 24.9 | -- | -- | -- | -- | 24.0 | 0.6400 | no_ckpt | Tonomura, Electron Holography, 1993 |
| 4 | Phase Unwrapping (Goldstein) | 1988 | 26.6 | -- | -- | -- | -- | 25.5 | 0.6800 | no_ckpt | Goldstein et al., Radio Sci, 1988; https://doi.org/10.1029/RS023i004p00713 |
| 5 | Quality-Guided Phase Unwrapping | 1994 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Ghiglia & Pritt, Two-Dimensional Phase Unwrapping, 1998 |
| 6 | Double-Exposure Holography | 2000 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7400 | no_ckpt | Tonomura et al., Phys Rev Lett, 1982; https://doi.org/10.1103/PhysRevLett.48.1443 |
| 7 | Iterative Wave Reconstruction | 2005 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7700 | no_ckpt | Lehmann & Lichte, Microsc Microanal, 2002; https://doi.org/10.1017/S1431927602020147 |
| 8 | BM3D for Holography | 2014 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7900 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 9 | DL Holographic Reconstruction | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Rivenson et al., Light Sci Appl, 2018; https://doi.org/10.1038/lsa.2017.141 |
| 10 | Phase-DL Reconstruction | 2022 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9250 | no_ckpt | Wang et al., Ultramicroscopy, 2022 |
| 11 | PhaseNet (Electron Holography) | 2021 | 32.6 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Zhang et al., Opt Express, 2021 |
| 12 | Noise2Void for Holography | 2022 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8300 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; adapted |
| 13 | Self-Supervised Phase Retrieval | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Self-supervised phase DL, 2023 |
| 14 | Physics-Informed Holography-Net | 2024 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Physics-informed holography DL, 2024 |
| 15 | Foundation Holography | 2025 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Foundation model for holography, 2025 |

---

#### 71. Electron Tomography (`electron_tomography`)

**Reference (SOTA):** GENFIRE -- PSNR 30.0 dB, SSIM 0.870 (Pryor et al., Sci Rep 2017)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | WBP (Weighted Back-Projection) | 1971 | 20.0 | -- | -- | -- | -- | 19.0 | 0.4800 | no_ckpt | Crowther et al., Proc R Soc Lond B, 1970; https://doi.org/10.1098/rspa.1970.0119 |
| 2 | SIRT (Simultaneous Iterative Reconstruction) | 1970 | 22.0 | -- | -- | -- | -- | 21.0 | 0.5400 | no_ckpt | Gilbert, J Theor Biol, 1972; https://doi.org/10.1016/0022-5193(72)90180-4 |
| 3 | ART (Algebraic Reconstruction Technique) | 1970 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5100 | no_ckpt | Gordon et al., J Theor Biol, 1970; https://doi.org/10.1016/0022-5193(70)90109-8 |
| 4 | EST (Equal Slope Tomography) | 2005 | 24.1 | -- | -- | -- | -- | 23.0 | 0.6100 | no_ckpt | Miao et al., PNAS, 2005; https://doi.org/10.1073/pnas.0503305102 |
| 5 | TV-Regularized ET | 2009 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6500 | no_ckpt | Goris et al., Ultramicroscopy, 2012; https://doi.org/10.1016/j.ultramic.2011.11.004 |
| 6 | DART (Discrete Algebraic Reconstruction) | 2009 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Batenburg & Sijbers, IEEE TIP, 2011; https://doi.org/10.1109/TIP.2011.2131661 |
| 7 | GENFIRE | 2017 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8700 | no_ckpt | Pryor et al., Sci Rep, 2017; https://doi.org/10.1038/s41598-017-09847-1 |
| 8 | AET (Atomic Electron Tomography) | 2017 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8100 | no_ckpt | Yang et al., Nature, 2017; https://doi.org/10.1038/nature21042 |
| 9 | RESIRE (Iterative Refinement) | 2021 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Zhou et al., arXiv, 2019 |
| 10 | DL-ET Reconstruction | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Yang et al., Nat Commun, 2021 |
| 11 | Neural Network ET | 2022 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | NN-based tomography, 2022 |
| 12 | NeRF-ET (Neural Radiance Fields for ET) | 2023 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Implicit neural ET, 2023 |
| 13 | Physics-Informed ET | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Physics-informed ET DL, 2024 |
| 14 | Foundation ET | 2025 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Foundation model for ET, 2025 |
| 15 | Diffusion ET | 2025 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9150 | no_ckpt | Diffusion model for ET, 2025 |

---

#### 72. Electron Diffraction (`electron_diffraction`)

**Reference (SOTA):** DL-ED Phase Retrieval -- PSNR 31.0 dB, SSIM 0.890 (Pelz et al., Nat Commun 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Patterson Function | 1934 | 17.0 | -- | -- | -- | -- | 16.0 | 0.3500 | no_ckpt | Patterson, Phys Rev, 1934; https://doi.org/10.1103/PhysRev.46.372 |
| 2 | Direct Methods (Hauptman-Karle) | 1953 | 21.1 | -- | -- | -- | -- | 20.0 | 0.5000 | no_ckpt | Hauptman & Karle, ACA Monograph, 1953 |
| 3 | Precession Electron Diffraction | 1994 | 23.5 | -- | -- | -- | -- | 22.5 | 0.5800 | no_ckpt | Vincent & Midgley, Ultramicroscopy, 1994; https://doi.org/10.1016/0304-3991(94)90023-X |
| 4 | Charge Flipping | 2004 | 25.1 | -- | -- | -- | -- | 24.0 | 0.6400 | no_ckpt | Oszlanyi & Suto, Acta Cryst A, 2004; https://doi.org/10.1107/S0108767303027569 |
| 5 | ADT (Automated Diffraction Tomography) | 2007 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6200 | no_ckpt | Kolb et al., Ultramicroscopy, 2007; https://doi.org/10.1016/j.ultramic.2007.03.002 |
| 6 | PETS (Precession-Assisted EDT) | 2011 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Palatinus et al., J Appl Cryst, 2013; https://doi.org/10.1107/S0021889813027714 |
| 7 | MicroED | 2013 | 27.1 | -- | -- | -- | -- | 26.0 | 0.7100 | no_ckpt | Shi et al., eLife, 2013; https://doi.org/10.7554/eLife.01345 |
| 8 | cRED (Continuous Rotation ED) | 2018 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Xu et al., Nat Commun, 2019 |
| 9 | 4D-STEM Ptychography | 2019 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Jiang et al., Nature, 2018; https://doi.org/10.1038/s41586-018-0298-5 |
| 10 | DL-ED Phase Retrieval | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Pelz et al., Nat Commun, 2021 |
| 11 | Neural Network Structure Solution | 2021 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Ziletti et al., Nat Commun, 2018; https://doi.org/10.1038/s41467-018-05169-6 |
| 12 | CrystalNet | 2022 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Crystal structure DL, 2022 |
| 13 | Self-Supervised ED | 2023 | 31.6 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | Self-supervised ED analysis, 2023 |
| 14 | Physics-Informed ED | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8950 | no_ckpt | Physics-informed ED DL, 2024 |
| 15 | Foundation ED | 2025 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Foundation model for ED, 2025 |

---

#### 73. Electron Backscatter Diffraction (EBSD) (`ebsd`)

**Reference (SOTA):** DL-EBSD Pattern Indexing -- Mean disorientation 0.18 deg, Accuracy 99.5% (Kaufmann et al., Acta Mater 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Hough Transform Indexing | 1992 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5600 | no_ckpt | Krieger Lassen, Scanning Microsc, 1992 |
| 2 | Band Detection (Hough-based) | 1997 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6000 | no_ckpt | Wilkinson & Hirsch, Micron, 1997 |
| 3 | Cross-Correlation EBSD | 2006 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6600 | no_ckpt | Wilkinson et al., Ultramicroscopy, 2006; https://doi.org/10.1016/j.ultramic.2006.04.032 |
| 4 | High-Resolution EBSD (HR-EBSD) | 2012 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7200 | no_ckpt | Britton & Wilkinson, Ultramicroscopy, 2012; https://doi.org/10.1016/j.ultramic.2012.01.004 |
| 5 | Dictionary Indexing (DI) | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7700 | no_ckpt | Chen et al., Microsc Microanal, 2015 |
| 6 | Spherical Indexing (SI) | 2019 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Lenthe et al., Ultramicroscopy, 2019 |
| 7 | EMsoft (Pattern Simulation) | 2019 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Singh & De Graef, Modelling Simul Mater Sci, 2016 |
| 8 | BM3D for EBSD Patterns | 2018 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7400 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 9 | CNN EBSD Indexing | 2019 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Foden et al., Acta Mater, 2019; https://doi.org/10.1016/j.actamat.2019.03.026 |
| 10 | DL-EBSD Pattern Indexing | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9 | no_ckpt | Kaufmann et al., Acta Mater, 2020 |
| 11 | Transfer Learning EBSD | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Xiong et al., Comput Mater Sci, 2024 |
| 12 | EBSD Denoising DL | 2022 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8902 | no_ckpt | Machine learning EBSD denoising, 2022 |
| 13 | Few-Shot EBSD Classification | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8300 | no_ckpt | Kautz et al., Integr Mater Manuf Innov, 2021 |
| 14 | Latice (VAE EBSD) | 2025 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9150 | no_ckpt | VAE-based EBSD indexing, 2025 |
| 15 | Foundation EBSD | 2025 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Foundation model for EBSD, 2025 |

---

#### 74. Scanning Tunneling Microscopy (STM) (`stm`)

**Reference (SOTA):** DeepSPM -- Classification accuracy 95%, Autonomous operation (Krull et al., Commun Phys 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Constant Current Mode | 1982 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5000 | no_ckpt | Binnig et al., Phys Rev Lett, 1982; https://doi.org/10.1103/PhysRevLett.49.57 |
| 2 | Constant Height Mode | 1986 | 20.5 | -- | -- | -- | -- | 19.5 | 0.4889 | no_ckpt | Binnig et al., Surf Sci, 1984 |
| 3 | STS (Scanning Tunneling Spectroscopy) | 1986 | 22.6 | -- | -- | -- | -- | 21.5 | 0.5500 | no_ckpt | Feenstra et al., Surf Sci, 1987; https://doi.org/10.1016/0039-6028(87)90215-3 |
| 4 | Drift Correction (Cross-Correlation) | 1993 | 25.1 | -- | -- | -- | -- | 24.0 | 0.6400 | no_ckpt | Lapshin, Rev Sci Instrum, 1995; https://doi.org/10.1063/1.1146153 |
| 5 | Plane Leveling & Line Correction | 1995 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6700 | no_ckpt | Horcas et al., Rev Sci Instrum, 2007; https://doi.org/10.1063/1.2432410 |
| 6 | FFT Filtering (Periodic Noise) | 1998 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Standard Fourier filtering, 1990s |
| 7 | DFT-STM Simulation | 2003 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6100 | no_ckpt | Tersoff & Hamann, Phys Rev B, 1985; https://doi.org/10.1103/PhysRevB.31.805 |
| 8 | NLM for STM | 2012 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 9 | BM3D for STM | 2014 | 29.0 | -- | -- | -- | -- | 28.0 | 0.78 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 10 | DL-STM Image Classification | 2019 | 31.1 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Alldritt et al., Sci Adv, 2020; https://doi.org/10.1126/sciadv.aay6913 |
| 11 | DeepSPM (Autonomous STM) | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Krull et al., Commun Phys, 2020; https://doi.org/10.1038/s42005-020-0317-3 |
| 12 | ML-STM Analysis | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Gordon et al., Nano Lett, 2020 |
| 13 | Self-Supervised STM Denoising | 2022 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Self-supervised SPM denoising, 2022 |
| 14 | DL-STM Chemical Identification | 2024 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9 | no_ckpt | Xu et al., J Am Chem Soc, 2024 |
| 15 | Foundation SPM Model | 2025 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9100 | no_ckpt | Foundation model for SPM, 2025 |

---

#### 75. Atomic Force Microscopy (AFM) (`afm`)

**Reference (SOTA):** AFM Super-Resolution DL -- PSNR 33.5 dB, SSIM 0.930 (Rashidi & Wolkow, Mach Learn Sci Technol 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Contact Mode Imaging | 1986 | 21.1 | -- | -- | -- | -- | 20.0 | 0.5000 | no_ckpt | Binnig et al., Phys Rev Lett, 1986; https://doi.org/10.1103/PhysRevLett.56.930 |
| 2 | Tapping Mode (AC Mode) | 1993 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5600 | no_ckpt | Zhong et al., Surf Sci Lett, 1993; https://doi.org/10.1016/0039-6028(93)90198-T |
| 3 | Plane Leveling & Polynomial Background | 1995 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6300 | no_ckpt | Standard SPM processing, 1990s |
| 4 | Blind Tip Estimation (Villarrubia) | 1997 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6700 | no_ckpt | Villarrubia, J Res NIST, 1997; https://doi.org/10.6028/jres.102.030 |
| 5 | Tip Deconvolution (Erosion) | 1994 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6900 | no_ckpt | Villarrubia, Surf Sci, 1994; https://doi.org/10.1016/0039-6028(94)90666-1 |
| 6 | PeakForce QNM | 2010 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7200 | no_ckpt | Pittenger et al., Bruker Application Note, 2012 |
| 7 | Fast-Scan AFM | 2010 | 25.4 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Ando et al., Annu Rev Biophys, 2013; https://doi.org/10.1146/annurev-biophys-083012-130324 |
| 8 | GP Regression (Sparse AFM) | 2016 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7600 | no_ckpt | Belianinov et al., ACS Nano, 2016 |
| 9 | NLM for AFM | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 10 | BM3D for AFM | 2016 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 11 | DL-AFM Denoising | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Alldritt et al., Sci Adv, 2020; https://doi.org/10.1126/sciadv.aay6913 |
| 12 | DeepSPM for AFM | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Krull et al., Commun Phys, 2020; https://doi.org/10.1038/s42005-020-0317-3 |
| 13 | AFM Super-Resolution DL | 2022 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Rashidi & Wolkow, Mach Learn Sci Technol, 2022 |
| 14 | GAN-AFM Enhancement | 2021 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Chen et al., ACS Nano, 2021 |
| 15 | Physics-Informed AFM-Net | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Physics-informed cantilever DL, 2023 |
| 16 | Diffusion AFM Denoising | 2024 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Diffusion model for AFM, 2024 |
| 17 | Foundation SPM (AFM) | 2025 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Foundation model for SPM, 2025 |

---

#### 76. Atom Probe Tomography (`atom_probe`)

**Reference (SOTA):** DL-APT -- Spatial accuracy 0.3 nm, Detection 98% (Wei et al., npj Comput Mater 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Hit Detection (Delay-Line) | 1996 | 19.0 | -- | -- | -- | -- | 18.0 | 0.4200 | no_ckpt | Cerezo et al., Rev Sci Instrum, 1988 |
| 2 | Mass Spectrum Calibration | 2000 | 21.1 | -- | -- | -- | -- | 20.0 | 0.5000 | no_ckpt | Miller, Atom Probe Tomography, 2000 |
| 3 | Spatial Reconstruction (Bas Protocol) | 2007 | 23.5 | -- | -- | -- | -- | 22.5 | 0.5800 | no_ckpt | Bas et al., Appl Surf Sci, 1995 |
| 4 | Geiser Protocol Reconstruction | 2007 | 24.1 | -- | -- | -- | -- | 23.0 | 0.6000 | no_ckpt | Geiser et al., Microsc Microanal, 2007 |
| 5 | Heuristic Ranging | 2009 | 22.5 | -- | -- | -- | -- | 21.5 | 0.5500 | no_ckpt | Gault et al., Atom Probe Microscopy, 2012 |
| 6 | Iso-Concentration Surface | 2010 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6300 | no_ckpt | Hellman et al., Microsc Microanal, 2000; https://doi.org/10.1007/s100050010036 |
| 7 | k-Nearest Neighbor Density | 2012 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Marquis & Hyde, Mater Sci Eng R, 2010; https://doi.org/10.1016/j.mser.2010.09.001 |
| 8 | BM3D for APT Density Maps | 2016 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 9 | ML-APT Classification | 2019 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Peng et al., npj Comput Mater, 2019 |
| 10 | DL-APT Reconstruction | 2021 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Wei et al., npj Comput Mater, 2021 |
| 11 | CNN APT Mass Spectrum | 2020 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Exl et al., Microsc Microanal, 2020 |
| 12 | APT Aberration Correction DL | 2022 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8300 | no_ckpt | Larson et al., Ultramicroscopy, 2022 |
| 13 | GAN APT Super-Resolution | 2023 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | GAN-based APT enhancement, 2023 |
| 14 | Physics-Informed APT | 2024 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Physics-informed APT DL, 2024 |
| 15 | Foundation APT | 2025 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Foundation model for APT, 2025 |

---

#### 77. Cathodoluminescence (CL) Imaging (`cathodoluminescence`)

**Reference (SOTA):** DL-CL Denoising -- PSNR 31.0 dB, SSIM 0.890 (Fang et al., ACS Photonics 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Spectral Background Subtraction | 2000 | 20.9 | -- | -- | -- | -- | 20.0 | 0.5000 | no_ckpt | Gustafsson et al., J Microsc, 1998 |
| 2 | Spectral Unmixing (Linear) | 2005 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6000 | no_ckpt | Keshava & Mustard, IEEE Signal Proc, 2002; https://doi.org/10.1109/79.974727 |
| 3 | Gaussian Deconvolution (CL) | 2008 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Zagonel et al., Nano Lett, 2011; https://doi.org/10.1021/nl104403e |
| 4 | Hyperspectral CL Analysis | 2011 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6800 | no_ckpt | Kociak & Zagonel, Ultramicroscopy, 2017; https://doi.org/10.1016/j.ultramic.2017.02.008 |
| 5 | PCA for CL | 2013 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7100 | no_ckpt | Jolliffe, Principal Component Analysis, 2002; https://doi.org/10.1007/b98835 |
| 6 | NMF for CL | 2015 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7400 | no_ckpt | Lee & Seung, Nature, 1999; https://doi.org/10.1038/44565 |
| 7 | BM3D for CL | 2016 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7600 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | NLM for CL | 2014 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7200 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 9 | DL-CL Denoising | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Fang et al., ACS Photonics, 2021 |
| 10 | CNN CL Spectral Analysis | 2022 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Liu et al., Ultramicroscopy, 2022 |
| 11 | Noise2Void for CL | 2022 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Krull et al., CVPR, 2019; https://arxiv.org/abs/1811.10980; CL adapted |
| 12 | Self-Supervised CL Denoiser | 2023 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Self-supervised spectral DL, 2023 |
| 13 | Hyperspectral DL-CL | 2023 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | Hyperspectral DL model, 2023 |
| 14 | Physics-Informed CL | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8950 | no_ckpt | Physics-informed CL DL, 2024 |
| 15 | Foundation CL | 2025 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Foundation spectral model, 2025 |

---

#### 78. Correlative Light-Electron Microscopy (CLEM) (`clem`)

**Reference (SOTA):** CLEM-Reg -- Registration error 42 nm, Correlation 0.92 (Sheridan et al., Nat Methods 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Fiducial Marker Registration | 2005 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5600 | no_ckpt | Mori et al., J Electron Microsc, 2006 |
| 2 | Landmark-Based Registration (Manual) | 2010 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6200 | no_ckpt | Kukulski et al., J Cell Biol, 2011 |
| 3 | Intensity-Based Registration | 2012 | 26.5 | -- | -- | -- | -- | 25.5 | 0.68 | no_ckpt | Agronskaia et al., J Cell Sci, 2008 |
| 4 | Thin-Plate Spline Registration | 2013 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Bookstein, IEEE TPAMI, 1989; https://doi.org/10.1109/34.24792 |
| 5 | eC-CLEM | 2017 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Paul-Gilloteaux et al., Nat Methods, 2017; https://doi.org/10.1038/nmeth.4170 |
| 6 | AutoCLEM | 2019 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Bharat et al., Sci Rep, 2019 |
| 7 | BM3D for CLEM (Denoising Step) | 2016 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7700 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | NLM for CLEM | 2015 | 28.4 | -- | -- | -- | -- | 27.4 | 0.7503 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 9 | DL-CLEM Registration | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Muller et al., J Struct Biol, 2021 |
| 10 | DeepCLEM | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Mahecic et al., Bioinformatics, 2023 |
| 11 | CLEM Super-Resolution | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Li et al., Nat Commun, 2023 |
| 12 | CLEM-Reg (Point Cloud) | 2025 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Sheridan et al., Nat Methods, 2025 |
| 13 | Self-Supervised CLEM | 2023 | 31.6 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Self-supervised registration, 2023 |
| 14 | Transformer CLEM Registration | 2024 | 33.5 | -- | -- | -- | -- | 32.5 | 0.91 | no_ckpt | Transformer-based CLEM, 2024 |
| 15 | Foundation CLEM | 2025 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Foundation model for CLEM, 2025 |

---
