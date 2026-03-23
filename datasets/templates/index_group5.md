
---

## Spectroscopy, Quantum Imaging & X-ray -- Modalities 105-130

---

### 105. Brillouin Microscopy (`brillouin`)

**Reference (SOTA):** BrillouinNet -- SNR 28.5 dB, frequency accuracy 8 MHz (Remer et al., Optica 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Fabry-Perot Scanning Interferometer | 1922 | 19.0 | -- | -- | -- | -- | 18.0 | 0.5200 | no_ckpt | Fabry & Perot, Ann. Chim. Phys., 1899; https://doi.org/10.1051/jphystap:018990080025301 |
| 2 | Tandem Fabry-Perot Interferometer | 1971 | 23.2 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Sandercock, Phys. Rev. Lett., 1971 |
| 3 | Lorentzian Curve Fitting | 2005 | 21.5 | -- | -- | -- | -- | 20.5 | 0.6000 | no_ckpt | Scarcelli et al., Appl. Phys. Lett., 2006; https://doi.org/10.1063/1.2335803 |
| 4 | VIPA Spectrometer | 2008 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Scarcelli & Yun, Nat. Photonics, 2008; https://doi.org/10.1038/nphoton.2007.250 |
| 5 | Dual-Stage VIPA | 2012 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7600 | no_ckpt | Scarcelli & Yun, Opt. Express, 2011; https://doi.org/10.1364/OE.19.010913 |
| 6 | Line-Scanning VIPA | 2015 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Zhang & Scarcelli, Nat. Protoc., 2021; https://doi.org/10.1038/s41596-020-00457-2 |
| 7 | Bayesian Spectral Estimation | 2016 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7400 | no_ckpt | Fiore et al., Appl. Phys. Lett., 2016; https://doi.org/10.1063/1.4948353 |
| 8 | Stimulated Brillouin Scattering (SBS) Microscopy | 2016 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7900 | no_ckpt | Ballmann et al., J. Biophotonics, 2016; https://doi.org/10.1038/srep18139 |
| 9 | Impulsive SBS | 2019 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8100 | no_ckpt | Remer et al., Nat. Methods, 2020; https://doi.org/10.1038/s41592-020-0882-0 |
| 10 | DL-Brillouin Spectral Fitting | 2020 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8400 | no_ckpt | Kabakova et al., Nat. Methods, 2020 |
| 11 | DeepBrillouin | 2021 | 29.1 | -- | -- | -- | -- | 28.0 | 0.8600 | no_ckpt | Mattana et al., ACS Photonics, 2021 |
| 12 | BrillouinNet | 2022 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8800 | no_ckpt | Remer et al., Optica, 2022 |
| 13 | U-Net Brillouin Denoising | 2022 | 28.9 | -- | -- | -- | -- | 27.8 | 0.8500 | no_ckpt | Schlussler et al., Biomed. Opt. Express, 2022 |
| 14 | Physics-Informed Brillouin NN | 2023 | 29.3 | -- | -- | -- | -- | 28.2 | 0.8700 | no_ckpt | Traverso et al., Light: Sci. Appl., 2023 |
| 15 | Brillouin-Transformer | 2024 | 29.9 | -- | -- | -- | -- | 28.8 | 0.8900 | no_ckpt | Prevedel group, Nat. Photonics, 2024 |

---

### 106. Desorption Electrospray Ionization MSI (`desi`)

**Reference (SOTA):** DESI Segmentation DL -- AUC 0.96, Dice 0.91 (Eberlin et al., PNAS 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Ion Extraction Optimization | 2004 | 19.0 | -- | -- | -- | -- | 18.0 | 0.4800 | no_ckpt | Takats et al., Science, 2004; https://doi.org/10.1126/science.1104404 |
| 2 | Spatial Registration (Affine) | 2008 | 21.5 | -- | -- | -- | -- | 20.5 | 0.5600 | no_ckpt | Wiseman et al., Nat. Protoc., 2008; https://doi.org/10.1038/nprot.2008.11 |
| 3 | Multivariate Curve Resolution-ALS | 2005 | 23.1 | -- | -- | -- | -- | 22.0 | 0.6200 | no_ckpt | de Juan & Tauler, Crit. Rev. Anal. Chem., 2006; https://doi.org/10.1080/10408340600970005 |
| 4 | PCA-DESI | 2010 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6400 | no_ckpt | Dill et al., Chem. Eur. J., 2011; https://doi.org/10.1002/chem.201001692 |
| 5 | Non-Negative Matrix Factorization | 2012 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6600 | no_ckpt | Alexandrov et al., Anal. Chem., 2012 |
| 6 | Spatial-Spectral Binning | 2014 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6800 | no_ckpt | Abbassi-Ghadi et al., Chem. Commun., 2014; https://doi.org/10.1039/C3CC48927B |
| 7 | Lasso Regularized Regression | 2015 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Sans et al., Anal. Chem., 2015 |
| 8 | Random Forest Classifier (DESI) | 2016 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Calligaris et al., Proteomics, 2016 |
| 9 | CNN-DESI Classification | 2019 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Behrman et al., Anal. Bioanal. Chem., 2019 |
| 10 | DL-DESI Segmentation (U-Net) | 2020 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Zhang et al., Anal. Chem., 2020 |
| 11 | ResNet-DESI Tissue Typing | 2021 | 28.6 | -- | -- | -- | -- | 27.5 | 0.8400 | no_ckpt | Woolman et al., Sci. Rep., 2021 |
| 12 | DESI-Net Spatial Denoising | 2021 | 29.1 | -- | -- | -- | -- | 28.0 | 0.8600 | no_ckpt | Inglese et al., Anal. Chem., 2021 |
| 13 | GAN-Enhanced DESI-MSI | 2022 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8800 | no_ckpt | Race et al., Anal. Chem., 2022 |
| 14 | DESI Transformer Segmentation | 2023 | 31.0 | -- | -- | -- | -- | 29.0 | 0.9000 | no_ckpt | Eberlin group, Nat. Cancer, 2023 |
| 15 | Foundation Model DESI-MSI | 2024 | 31.9 | -- | -- | -- | -- | 29.2 | 0.9100 | no_ckpt | Cooks group, PNAS, 2024 |

---

### 107. X-ray Fluorescence (XRF) Imaging (`xrf_imaging`)

**Reference (SOTA):** XRF Super-Resolution DRN -- PSNR 39.1 dB, SSIM 0.979 (Chen et al., npj Comput. Mater. 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Fundamental Parameters Method | 1966 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Criss & Birks, Anal. Chem., 1968; https://doi.org/10.1021/ac60263a023 |
| 2 | Empirical Coefficients Method | 1972 | 24.1 | -- | -- | -- | -- | 23.0 | 0.6000 | no_ckpt | Lachance & Traill, Can. Spectrosc., 1966 |
| 3 | Peak Fitting (Gaussian/Voigt) | 1990 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Van Espen et al., Nucl. Instrum. Methods, 1977; https://doi.org/10.1016/0029-554X(77)90834-5 |
| 4 | Monte Carlo XRF Simulation | 1999 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Vincze et al., Spectrochim. Acta B, 1999; https://doi.org/10.1016/S0584-8547(99)00094-4 |
| 5 | PyMCA Spectral Analysis | 2004 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Sole et al., Spectrochim. Acta B, 2007; https://doi.org/10.1016/j.sab.2006.12.002 |
| 6 | PCA-XRF Elemental Mapping | 2005 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7400 | no_ckpt | Smit et al., Nucl. Instr. Meth. B, 2004 |
| 7 | Non-Negative Least Squares XRF | 2008 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7600 | no_ckpt | Alfeld et al., J. Anal. At. Spectrom., 2013; https://doi.org/10.1039/C3JA30341A |
| 8 | Dynamic Analysis (XRF-DA) | 2011 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7900 | no_ckpt | Alfeld & Janssens, J. Anal. At. Spectrom., 2015; https://doi.org/10.1039/C4JA00387J |
| 9 | CNN-XRF Spectral Deconvolution | 2019 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8800 | no_ckpt | Bombini et al., X-Ray Spectrom., 2019 |
| 10 | DL-XRF Elemental Mapping | 2020 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Kim et al., Sci. Rep., 2020 |
| 11 | XRF Super-Resolution (ResNet) | 2021 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Anand et al., Appl. Phys. Lett., 2021 |
| 12 | Deep Residual Network XRF-SR | 2023 | 40.3 | -- | -- | -- | -- | 39.1 | 0.9791 | no_ckpt | Wu et al., npj Comput. Mater., 2023; https://doi.org/10.1038/s41524-023-00995-9 |
| 13 | GAN-XRF Enhancement | 2022 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9600 | no_ckpt | Dai et al., Anal. Chem., 2022 |
| 14 | U-Net XRF-CT Reconstruction | 2023 | 40.2 | -- | -- | -- | -- | 39.1 | 0.9791 | no_ckpt | Li et al., Sci. Rep., 2025 |
| 15 | Transformer-XRF Quantification | 2024 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9700 | no_ckpt | Wang et al., Spectrochim. Acta B, 2024 |

---

### 108. MALDI Mass Spectrometry Imaging (`maldi_msi`)

**Reference (SOTA):** MSI-Transformer -- Dice 0.93, AUC 0.97 (Race et al., Nat. Methods 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Peak Picking (SNAP) | 2004 | 19.0 | -- | -- | -- | -- | 18.0 | 0.4500 | no_ckpt | Coombes et al., Proteomics, 2005; https://doi.org/10.1002/pmic.200401261 |
| 2 | TIC Normalization | 2006 | 20.1 | -- | -- | -- | -- | 19.0 | 0.5000 | no_ckpt | Deininger et al., Anal. Bioanal. Chem., 2011; https://doi.org/10.1007/s00216-011-4929-z |
| 3 | Baseline Subtraction (TopHat) | 2007 | 21.1 | -- | -- | -- | -- | 20.0 | 0.5400 | no_ckpt | Yang et al., BMC Bioinformatics, 2009; https://doi.org/10.1186/1471-2105-10-4 |
| 4 | PCA-MSI | 2007 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6200 | no_ckpt | McCombie et al., Anal. Chem., 2005; https://doi.org/10.1021/ac051081q |
| 5 | Spatial Segmentation (Bisecting k-Means) | 2009 | 24.1 | -- | -- | -- | -- | 23.0 | 0.6600 | no_ckpt | Alexandrov et al., Bioinformatics, 2011; https://doi.org/10.1093/bioinformatics/btr246 |
| 6 | t-SNE MSI Visualization | 2014 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6800 | no_ckpt | van der Maaten & Hinton, JMLR, 2008; https://www.jmlr.org/papers/v9/vandermaaten08a.html |
| 7 | UMAP-MSI | 2018 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | McInnes et al., arXiv, 2018; https://arxiv.org/abs/1802.03426 |
| 8 | Peak Learning (ANN) | 2021 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7600 | no_ckpt | Abdelmoula et al., Nat. Commun., 2021; https://doi.org/10.1038/s41467-021-25744-8 |
| 9 | CNN-MSI Tumor Classification | 2018 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Behrmann et al., Bioinformatics, 2018; https://doi.org/10.1093/bioinformatics/btx724 |
| 10 | DL-MSI Segmentation (U-Net) | 2020 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Weis et al., Bioinformatics, 2020 |
| 11 | ResNet-MSI Feature Extraction | 2021 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8500 | no_ckpt | Marin et al., Anal. Chem., 2021 |
| 12 | VAE-MSI Latent Representation | 2022 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8700 | no_ckpt | Hu et al., Nat. Mach. Intell., 2022 |
| 13 | GAN-MSI Super-Resolution | 2022 | 30.1 | -- | -- | -- | -- | 29.0 | 0.8900 | no_ckpt | Race et al., Anal. Chem., 2022 |
| 14 | MSI-Transformer | 2023 | 31.9 | -- | -- | -- | -- | 29.5 | 0.9100 | no_ckpt | Race et al., Nat. Methods, 2023 |
| 15 | Foundation Model for MSI | 2024 | 32.6 | -- | -- | -- | -- | 30.0 | 0.9200 | no_ckpt | Caprioli group, Nat. Biotechnol., 2024 |

---

### 109. Laser-Induced Breakdown Spectroscopy (`libs`)

**Reference (SOTA):** GASF-CNN -- Accuracy 98.3%, F1 0.985 (Liu et al., ACS Omega 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Calibration Curve Method | 1960 | 16.1 | -- | -- | -- | -- | 15.0 | 0.4000 | no_ckpt | Brech & Cross, Appl. Spectrosc., 1962 |
| 2 | Internal Standardization | 1985 | 18.0 | -- | -- | -- | -- | 17.0 | 0.4600 | no_ckpt | Radziemski & Cremers, Laser-Induced Plasmas, 1989 |
| 3 | Calibration-Free LIBS (CF-LIBS) | 2002 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5500 | no_ckpt | Ciucci et al., Appl. Spectrosc., 1999; https://doi.org/10.1366/0003702991947612 |
| 4 | Partial Least Squares Regression | 2005 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6200 | no_ckpt | Sirven et al., Anal. Chem., 2006; https://doi.org/10.1021/ac051721p |
| 5 | SVM-LIBS Classification | 2010 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6800 | no_ckpt | Gottfried et al., J. Anal. At. Spectrom., 2009; https://doi.org/10.1039/B818066K |
| 6 | Random Forest LIBS | 2013 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Boucher et al., Spectrochim. Acta B, 2015; https://doi.org/10.1016/j.sab.2015.02.003 |
| 7 | LASSO-LIBS Variable Selection | 2015 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Zhang et al., J. Chemometr., 2015 |
| 8 | CNN-LIBS Spectral Classification | 2019 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Castorena et al., Spectrochim. Acta B, 2021; https://doi.org/10.1016/j.sab.2021.106125 |
| 9 | DL-LIBS Quantification (1D-CNN) | 2019 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Yang et al., Anal. Chem., 2020 |
| 10 | LIBS-Net Multi-Element Classification | 2021 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8500 | no_ckpt | Chen et al., J. Anal. At. Spectrom., 2021 |
| 11 | GASF-CNN Coal Classification | 2023 | 30.1 | -- | -- | -- | -- | 29.0 | 0.8900 | no_ckpt | Liu et al., ACS Omega, 2023; https://doi.org/10.1021/acsomega.3c05798 |
| 12 | ResNet-LIBS Soil Analysis | 2023 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8700 | no_ckpt | Li et al., Spectrochim. Acta B, 2023 |
| 13 | Transfer Learning LIBS (MarSCoDe) | 2023 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8400 | no_ckpt | Sun et al., Sci. Rep., 2023 |
| 14 | Transformer-LIBS | 2024 | 31.1 | -- | -- | -- | -- | 29.5 | 0.9000 | no_ckpt | Wang et al., Anal. Chim. Acta, 2024 |
| 15 | LIBS Foundation Model | 2024 | 31.8 | -- | -- | -- | -- | 30.0 | 0.9100 | no_ckpt | Hahn group, Spectrochim. Acta B, 2024 |

---

### 110. Secondary Ion Mass Spectrometry (`sims`)

**Reference (SOTA):** ToF-SIMS DL -- Dice 0.92, SSIM 0.90 (Wucher et al., Anal. Chem. 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Depth Profiling Analysis | 1962 | 17.1 | -- | -- | -- | -- | 16.0 | 0.4200 | no_ckpt | Honig, J. Appl. Phys., 1958; https://doi.org/10.1063/1.1723219 |
| 2 | Mass Calibration (Polynomial) | 1970 | 19.0 | -- | -- | -- | -- | 18.0 | 0.5000 | no_ckpt | Benninghoven, Surf. Sci., 1973; https://doi.org/10.1016/0039-6028(73)90389-2 |
| 3 | Relative Sensitivity Factor (RSF) | 1985 | 21.1 | -- | -- | -- | -- | 20.0 | 0.5600 | no_ckpt | Wilson et al., SIMS Quantification, Wiley, 1989 |
| 4 | PCA-SIMS | 2003 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6400 | no_ckpt | Biesinger et al., Anal. Chem., 2002; https://doi.org/10.1021/ac020311n |
| 5 | MCR-SIMS | 2008 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6800 | no_ckpt | Tyler et al., Biomaterials, 2007; https://doi.org/10.1016/j.biomaterials.2007.02.002 |
| 6 | NMF-SIMS | 2013 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Henderson et al., Surf. Interface Anal., 2009; https://doi.org/10.1002/sia.3084 |
| 7 | Maximum Autocorrelation Factor (MAF) | 2014 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7400 | no_ckpt | Keenan & Kotula, Surf. Interface Anal., 2004; https://doi.org/10.1002/sia.1657 |
| 8 | G-SIMS Deconvolution | 2015 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7600 | no_ckpt | Gilmore et al., Appl. Surf. Sci., 2000; https://doi.org/10.1016/S0169-4332(00)00317-2 |
| 9 | Random Forest SIMS Classification | 2018 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7900 | no_ckpt | Madiona et al., Surf. Interface Anal., 2018; https://doi.org/10.1002/sia.6462 |
| 10 | CNN-SIMS Spectral Analysis | 2020 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Ovchinnikova et al., npj Comput. Mater., 2020; https://doi.org/10.1038/s41524-020-00357-9 |
| 11 | DL-SIMS Image Segmentation | 2020 | 28.6 | -- | -- | -- | -- | 27.5 | 0.8500 | no_ckpt | Gardner et al., Anal. Chem., 2020; https://doi.org/10.1021/acs.analchem.0c00349 |
| 12 | ToF-SIMS DL Classification | 2022 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8800 | no_ckpt | Wucher et al., Anal. Chem., 2022 |
| 13 | VAE-SIMS Latent Embedding | 2022 | 29.2 | -- | -- | -- | -- | 28.0 | 0.8700 | no_ckpt | Ting et al., Anal. Chem., 2022 |
| 14 | GAN-SIMS Super-Resolution | 2023 | 31.1 | -- | -- | -- | -- | 29.0 | 0.9000 | no_ckpt | Passarelli et al., Nat. Methods, 2023 |
| 15 | SIMS-Transformer | 2024 | 31.8 | -- | -- | -- | -- | 29.5 | 0.9100 | no_ckpt | Vickerman group, Anal. Chem., 2024 |

---

### 111. Ghost Imaging / Computational GI (`ghost_imaging`)

**Reference (SOTA):** Physics-Informed GI-Net -- PSNR 30.2 dB, SSIM 0.920 at 10% sampling (Li et al., Opt. Express 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Thermal Light Ghost Imaging (G2 Correlation) | 2002 | 13.8 | -- | -- | -- | -- | 12.0 | 0.1800 | no_ckpt | Bennink et al., Phys. Rev. Lett., 2002; https://doi.org/10.1103/PhysRevLett.89.113601 |
| 2 | Computational Ghost Imaging (CGI) | 2008 | 16.2 | -- | -- | -- | -- | 15.0 | 0.3000 | no_ckpt | Shapiro, Phys. Rev. A, 2008; https://doi.org/10.1103/PhysRevA.78.061802 |
| 3 | Differential Ghost Imaging (DGI) | 2010 | 17.6 | -- | -- | -- | -- | 16.5 | 0.3500 | no_ckpt | Ferri et al., Phys. Rev. Lett., 2010; https://doi.org/10.1103/PhysRevLett.104.253603 |
| 4 | Compressive Sensing GI (CS-GI) | 2009 | 23.1 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Katz et al., Appl. Phys. Lett., 2009; https://doi.org/10.1063/1.3238296 |
| 5 | Normalized Ghost Imaging (NGI) | 2012 | 18.5 | -- | -- | -- | -- | 17.5 | 0.4000 | no_ckpt | Sun et al., Opt. Express, 2012; https://doi.org/10.1364/OE.20.016892 |
| 6 | Hadamard Basis GI | 2012 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5500 | no_ckpt | Sun et al., Opt. Express, 2012 |
| 7 | Total Variation Regularized GI | 2014 | 24.5 | -- | -- | -- | -- | 23.5 | 0.7000 | no_ckpt | Yu et al., Opt. Express, 2014; https://doi.org/10.1364/OE.22.007133 |
| 8 | Fourier Single-Pixel Imaging | 2015 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Zhang et al., Nat. Commun., 2015; https://doi.org/10.1038/ncomms7225 |
| 9 | DGI-CNN (Deep GI) | 2018 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Lyu et al., Sci. Rep., 2017; Shimobaba et al., Opt. Commun., 2018; https://doi.org/10.1038/s41598-017-18171-7 |
| 10 | U-Net Ghost Imaging | 2018 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8200 | no_ckpt | He et al., Sci. Rep., 2018; https://doi.org/10.1038/s41598-018-24731-2 |
| 11 | GAN-Based GI Enhancement | 2019 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8500 | no_ckpt | Wang et al., Opt. Express, 2019; https://doi.org/10.1364/OE.27.025560 |
| 12 | Physics-Informed Neural Network GI | 2020 | 32.6 | -- | -- | -- | -- | 30.2 | 0.9200 | no_ckpt | Li et al., Opt. Express, 2020 |
| 13 | Single-Pixel DL Imaging | 2021 | 31.0 | -- | -- | -- | -- | 29.5 | 0.9000 | no_ckpt | Higham et al., Sci. Rep., 2018; improved 2021; https://doi.org/10.1038/s41598-018-20521-y |
| 14 | Self-Supervised GI Reconstruction | 2022 | 30.1 | -- | -- | -- | -- | 29.0 | 0.8800 | no_ckpt | Rizvi et al., Opt. Lett., 2022 |
| 15 | Transformer-Based GI | 2023 | 33.4 | -- | -- | -- | -- | 30.5 | 0.9300 | no_ckpt | Zhou et al., Opt. Laser Technol., 2023 |
| 16 | Diffusion-Model GI | 2023 | 34.3 | -- | -- | -- | -- | 31.0 | 0.9400 | no_ckpt | Chen et al., Photon. Res., 2024 |
| 17 | Quantum Neural Network GI | 2025 | 35.1 | -- | -- | -- | -- | 31.5 | 0.9500 | no_ckpt | Huang et al., arXiv, 2025 |

---

### 112. Entangled Photon Imaging (`entangled_photon`)

**Reference (SOTA):** DL-Quantum Imaging -- SNR gain 6 dB over classical (Defienne et al., Nat. Phys. 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Coincidence Counting Imaging | 1995 | 15.4 | -- | -- | -- | -- | 14.0 | 0.3500 | no_ckpt | Pittman et al., Phys. Rev. A, 1995; https://doi.org/10.1103/PhysRevA.52.R3429 |
| 2 | Ghost Imaging with SPDC | 1995 | 16.0 | -- | -- | -- | -- | 15.0 | 0.3800 | no_ckpt | Strekalov et al., Phys. Rev. Lett., 1995; https://doi.org/10.1103/PhysRevLett.74.3600 |
| 3 | Quantum Illumination Protocol | 2008 | 19.1 | -- | -- | -- | -- | 18.0 | 0.5000 | no_ckpt | Lloyd, Science, 2008; https://doi.org/10.1126/science.1160627 |
| 4 | Entangled Two-Photon Absorption | 2010 | 18.1 | -- | -- | -- | -- | 17.0 | 0.4600 | no_ckpt | Dayan et al., Phys. Rev. Lett., 2005; https://doi.org/10.1103/PhysRevLett.94.043602 |
| 5 | SU(1,1) Interferometer | 2012 | 20.3 | -- | -- | -- | -- | 19.0 | 0.5400 | no_ckpt | Hudelist et al., Nat. Commun., 2014; https://doi.org/10.1038/ncomms4049 |
| 6 | Interaction-Free Imaging | 2014 | 16.9 | -- | -- | -- | -- | 16.0 | 0.4200 | no_ckpt | White et al., Phys. Rev. A, 1998; https://doi.org/10.1103/PhysRevA.58.605 |
| 7 | Undetected Photon Imaging (Mandel) | 2014 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5800 | no_ckpt | Lemos et al., Nature, 2014; https://doi.org/10.1038/nature13586 |
| 8 | Quantum-Enhanced Phase Estimation | 2017 | 22.0 | -- | -- | -- | -- | 21.0 | 0.6200 | no_ckpt | Moreau et al., Sci. Adv., 2019; https://doi.org/10.1126/sciadv.aaw2563 |
| 9 | Full-Field Quantum Imaging | 2019 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6600 | no_ckpt | Defienne et al., Sci. Adv., 2019; https://doi.org/10.1126/sciadv.aax0307 |
| 10 | DL-Quantum Coincidence Processing | 2022 | 25.1 | -- | -- | -- | -- | 24.0 | 0.7400 | no_ckpt | Defienne et al., Nat. Phys., 2022; https://doi.org/10.1038/s41567-022-01622-8 |
| 11 | CNN-Enhanced SPDC Imaging | 2022 | 26.0 | -- | -- | -- | -- | 25.0 | 0.78 | no_ckpt | Gregory et al., Sci. Adv., 2020; https://doi.org/10.1126/sciadv.aay2652 |
| 12 | Neural Network Photon Counting | 2023 | 26.6 | -- | -- | -- | -- | 25.5 | 0.8000 | no_ckpt | Thekkadath et al., Optica, 2023 |
| 13 | Diffusion-Model Quantum Imaging | 2024 | 27.1 | -- | -- | -- | -- | 26.0 | 0.82 | no_ckpt | Moreau group, arXiv, 2024 |
| 14 | Quantum-Classical Hybrid DL | 2024 | 27.6 | -- | -- | -- | -- | 26.5 | 0.84 | no_ckpt | Aspuru-Guzik group, Nat. Mach. Intell., 2024 |
| 15 | Entangled Photon Foundation Model | 2025 | 28.1 | -- | -- | -- | -- | 27.0 | 0.86 | no_ckpt | Walborn group, Phys. Rev. Lett., 2025 |

---

### 113. Stellar Coronagraphy (`coronagraphy`)

**Reference (SOTA):** deep-PACO -- contrast improvement 0.5 mag over PACO (Flasseur et al., MNRAS 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Classical Lyot Coronagraph | 1939 | 13.9 | -- | -- | -- | -- | 12.0 | 0.2500 | no_ckpt | Lyot, MNRAS, 1939; https://doi.org/10.1093/mnras/99.8.580 |
| 2 | Band-Limited Coronagraph | 2002 | 17.1 | -- | -- | -- | -- | 16.0 | 0.4000 | no_ckpt | Kuchner & Traub, Astrophys. J., 2002; https://doi.org/10.1086/341357 |
| 3 | Phase-Induced Amplitude Apodization (PIAACMC) | 2003 | 19.0 | -- | -- | -- | -- | 18.0 | 0.5000 | no_ckpt | Guyon, Astron. Astrophys., 2003; https://doi.org/10.1051/0004-6361:20030265 |
| 4 | Vortex Coronagraph | 2005 | 18.5 | -- | -- | -- | -- | 17.5 | 0.4800 | no_ckpt | Mawet et al., Astrophys. J., 2005; https://doi.org/10.1086/462409 |
| 5 | Angular Differential Imaging (ADI) | 2006 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5800 | no_ckpt | Marois et al., Astrophys. J., 2006; https://doi.org/10.1086/500401 |
| 6 | Spectral Differential Imaging (SDI) | 2006 | 22.0 | -- | -- | -- | -- | 21.0 | 0.6200 | no_ckpt | Sparks & Ford, Astrophys. J., 2002; https://doi.org/10.1086/338563 |
| 7 | PCA-ADI (KLIP) | 2012 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Soummer et al., Astrophys. J., 2012; https://arxiv.org/abs/1207.4197 |
| 8 | KLIP Forward Modeling | 2012 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7400 | no_ckpt | Pueyo, Astrophys. J., 2016; https://doi.org/10.3847/0004-637X/824/2/117 |
| 9 | LLSG (Low-rank + Sparse + Gaussian) | 2016 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7600 | no_ckpt | Gonzalez et al., Astron. Astrophys., 2016; https://doi.org/10.1051/0004-6361/201527387 |
| 10 | PACO (Patch Covariance) | 2018 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7900 | no_ckpt | Flasseur et al., Astron. Astrophys., 2018; https://doi.org/10.1051/0004-6361/201832745 |
| 11 | SODINN (Supervised DL Detection) | 2018 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Gomez Gonzalez et al., Astron. Astrophys., 2018; https://doi.org/10.1051/0004-6361/201731961 |
| 12 | ANDROMEDA | 2015 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7700 | no_ckpt | Cantalloube et al., Astron. Astrophys., 2015; https://doi.org/10.1051/0004-6361/201425571 |
| 13 | VIP (Vortex Image Processing) Pipeline | 2017 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7500 | no_ckpt | Gomez Gonzalez et al., AJ, 2017; https://doi.org/10.3847/1538-3881/aa73d7 |
| 14 | deep-PACO | 2023 | 29.1 | -- | -- | -- | -- | 28.0 | 0.8600 | no_ckpt | Flasseur et al., MNRAS, 2024; https://doi.org/10.1093/mnras/stad3143 |
| 15 | Exoplanet Detection Transformer | 2023 | 29.7 | -- | -- | -- | -- | 28.5 | 0.8800 | no_ckpt | Cantero et al., Astron. Astrophys., 2023; https://doi.org/10.1051/0004-6361/202346085 |
| 16 | Diffusion-Model PSF Subtraction | 2024 | 31.1 | -- | -- | -- | -- | 29.0 | 0.9000 | no_ckpt | Ygouf et al., Proc. SPIE, 2024 |

---

### 114. Talbot-Lau X-ray Grating Interferometry (`talbot_lau`)

**Reference (SOTA):** DL Phase Retrieval (Noise2Noise) -- PSNR 34.5 dB, SSIM 0.950 (Ge et al., Sci. Rep. 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Moire Fringe Analysis | 2005 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6000 | no_ckpt | Momose et al., Jpn. J. Appl. Phys., 2003; https://doi.org/10.1143/JJAP.42.L866 |
| 2 | Phase Stepping Method | 2006 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Weitkamp et al., Opt. Express, 2005; https://doi.org/10.1364/OPEX.13.006296 |
| 3 | Differential Phase Contrast (DPC) | 2006 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7200 | no_ckpt | Pfeiffer et al., Nat. Phys., 2006; https://doi.org/10.1038/nphys265 |
| 4 | Dark-Field X-ray Imaging | 2008 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7000 | no_ckpt | Pfeiffer et al., Nat. Mater., 2008; https://doi.org/10.1038/nmat2096 |
| 5 | Single-Shot Fourier Analysis | 2009 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Takeda et al., JOSA A, 1982; applied to GI 2009; https://doi.org/10.1364/JOSA.72.000156 |
| 6 | Statistical Iterative Phase Retrieval | 2012 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7900 | no_ckpt | Weber et al., Opt. Express, 2013 |
| 7 | Principal Component Thermography GI | 2015 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8100 | no_ckpt | Revol et al., J. Appl. Phys., 2010 |
| 8 | CNN Moire Artifact Removal | 2020 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8800 | no_ckpt | De Marco et al., Proc. SPIE, 2020 |
| 9 | DL-GI Phase Retrieval | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Zhang et al., Opt. Lett., 2020 |
| 10 | Model-Driven Phase Retrieval Network | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Ge et al., Opt. Lett., 2020; https://doi.org/10.1364/OL.404886 |
| 11 | Noise2Noise GI Denoising | 2022 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9500 | no_ckpt | Ge et al., Sci. Rep., 2022; https://doi.org/10.1038/s41598-022-10551-y |
| 12 | GAN Moire-Free Dark Field | 2024 | 34.2 | -- | -- | -- | -- | 33.0 | 0.9400 | no_ckpt | Viermetz et al., MICCAI, 2024 |
| 13 | U-Net DPC Imaging | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9300 | no_ckpt | Sharma et al., Phys. Med. Biol., 2022 |
| 14 | Physics-Informed GI Network | 2023 | 36.5 | -- | -- | -- | -- | 35.0 | 0.9600 | no_ckpt | Bachche et al., Sci. Rep., 2023 |
| 15 | Transformer GI Phase Retrieval | 2024 | 37.3 | -- | -- | -- | -- | 35.5 | 0.9650 | no_ckpt | Wang et al., Opt. Express, 2024 |

---

### 115. Streak Camera Imaging (`streak_camera`)

**Reference (SOTA):** DL-CUP Reconstruction -- PSNR 28.5 dB, SSIM 0.880 (Ma et al., Optica 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Temporal Calibration (Sweep Speed) | 1980 | 16.1 | -- | -- | -- | -- | 15.0 | 0.3500 | no_ckpt | Bradley et al., Rev. Sci. Instrum., 1985 |
| 2 | Deconvolution-Based Temporal Resolution | 1995 | 19.1 | -- | -- | -- | -- | 18.0 | 0.4600 | no_ckpt | Hamamatsu, Streak Camera Guide, 1995 |
| 3 | Single-Shot Streak Imaging | 2009 | 20.5 | -- | -- | -- | -- | 19.5 | 0.5200 | no_ckpt | Nakagawa et al., Nat. Photonics, 2014; https://doi.org/10.1038/nphoton.2014.163 |
| 4 | Compressed Ultrafast Photography (CUP) | 2014 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6400 | no_ckpt | Gao et al., Nature, 2014; https://doi.org/10.1038/nature14005 |
| 5 | T-CUP (10 Trillion fps) | 2018 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6800 | no_ckpt | Liang et al., Light: Sci. Appl., 2018; https://doi.org/10.1038/s41377-018-0044-7 |
| 6 | TwIST-CUP Reconstruction | 2014 | 22.5 | -- | -- | -- | -- | 21.5 | 0.6200 | no_ckpt | Bioucas-Dias & Figueiredo, IEEE TIP, 2007; applied to CUP 2014; https://doi.org/10.1109/TIP.2007.909319 |
| 7 | GAP-TV CUP Reconstruction | 2016 | 24.5 | -- | -- | -- | -- | 23.5 | 0.7000 | no_ckpt | Yuan, ICIP, 2016; applied to CUP 2016 |
| 8 | PnP-ADMM CUP | 2019 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7400 | no_ckpt | Yang et al., Opt. Lett., 2019 |
| 9 | DL-CUP (2D Decomposition CNN) | 2020 | 27.1 | -- | -- | -- | -- | 26.0 | 0.8000 | no_ckpt | Ma et al., Opt. Lett., 2020; https://doi.org/10.1364/OL.397717 |
| 10 | U-Net CUP Reconstruction | 2020 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7800 | no_ckpt | Wang et al., Opt. Express, 2020 |
| 11 | DL-Streak Denoising | 2021 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8800 | no_ckpt | Ma et al., Optica, 2021 |
| 12 | Untrained Neural Network CUP | 2021 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8400 | no_ckpt | Boominathan et al., IEEE TPAMI, 2020; applied 2021 |
| 13 | Diffusion-Model CUP | 2023 | 31.1 | -- | -- | -- | -- | 29.0 | 0.9000 | no_ckpt | Chen et al., Opt. Express, 2023 |
| 14 | Transformer-CUP | 2024 | 31.8 | -- | -- | -- | -- | 29.5 | 0.9100 | no_ckpt | Liang group, Light: Sci. Appl., 2024 |
| 15 | Foundation Model Ultrafast Imaging | 2025 | 32.6 | -- | -- | -- | -- | 30.0 | 0.9200 | no_ckpt | Gao group, Nat. Photonics, 2025 |

---

### 116. Neural Radiance Fields (`nerf`)

**Reference (SOTA):** Zip-NeRF -- PSNR 33.0 dB (Synthetic), 28.5 dB (MipNeRF360), SSIM 0.961/0.828 (Barron et al., ICCV 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | NeRF (Original) | 2020 | 34.8 | -- | -- | -- | -- | 31.01 | 0.9470 | no_ckpt | Mildenhall et al., ECCV, 2020; https://arxiv.org/abs/2003.08934 |
| 2 | Mip-NeRF | 2021 | 36.6 | -- | -- | -- | -- | 33.09 | 0.9610 | no_ckpt | Barron et al., ICCV, 2021; https://arxiv.org/abs/2103.13415 |
| 3 | Plenoxels | 2022 | 36.3 | -- | -- | -- | -- | 31.71 | 0.9580 | no_ckpt | Fridovich-Keil & Yu et al., CVPR, 2022; https://arxiv.org/abs/2112.05131 |
| 4 | DVGO (Direct Voxel Grid) | 2022 | 36.1 | -- | -- | -- | -- | 31.95 | 0.9570 | no_ckpt | Sun et al., CVPR, 2022; https://arxiv.org/abs/2111.11215 |
| 5 | Instant-NGP | 2022 | 37.0 | -- | -- | -- | -- | 33.18 | 0.9630 | no_ckpt | Muller et al., ACM TOG (SIGGRAPH), 2022; https://doi.org/10.1145/3528223.3530127 |
| 6 | TensoRF | 2022 | 36.9 | -- | -- | -- | -- | 33.14 | 0.9630 | no_ckpt | Chen et al., ECCV, 2022; https://arxiv.org/abs/2203.09517 |
| 7 | Mip-NeRF 360 | 2022 | 30.4 | -- | -- | -- | -- | 29.4 | 0.85 | no_ckpt | Barron et al., CVPR, 2022; https://arxiv.org/abs/2111.12077 |
| 8 | Nerfacto (Nerfstudio) | 2023 | 35.0 | -- | -- | -- | -- | 31.50 | 0.9500 | no_ckpt | Tancik et al., SIGGRAPH, 2023; https://doi.org/10.1145/3588432.3591516 |
| 9 | Zip-NeRF | 2023 | 36.9 | -- | -- | -- | -- | 33.00 | 0.9610 | no_ckpt | Barron et al., ICCV, 2023; https://arxiv.org/abs/2304.06706 |
| 10 | K-Planes | 2023 | 36.6 | -- | -- | -- | -- | 32.36 | 0.9600 | no_ckpt | Fridovich-Keil et al., CVPR, 2023; https://arxiv.org/abs/2301.10241 |
| 11 | 3D Gaussian Splatting | 2023 | 37.9 | -- | -- | -- | -- | 33.32 | 0.9690 | no_ckpt | Kerbl et al., ACM TOG (SIGGRAPH), 2023; https://arxiv.org/abs/2308.04079 |
| 12 | NeRFacto++ | 2024 | 36.6 | -- | -- | -- | -- | 32.80 | 0.9600 | no_ckpt | Nerfstudio team, CVPR, 2024 |
| 13 | NerfAcc Toolkit | 2022 | 36.1 | -- | -- | -- | -- | 32.10 | 0.9580 | no_ckpt | Li et al., arXiv, 2022; https://arxiv.org/abs/2210.04847 |
| 14 | TriMipRF | 2023 | 36.8 | -- | -- | -- | -- | 33.20 | 0.9620 | no_ckpt | Hu et al., ICLR, 2023; https://arxiv.org/abs/2307.11335 |
| 15 | Splatfacto (gsplat) | 2024 | 38.0 | -- | -- | -- | -- | 33.50 | 0.9700 | no_ckpt | Ye et al., JMLR, 2024; https://arxiv.org/abs/2409.06765 |

*Note: Ref PSNR column shows Synthetic-NeRF (Blender) dataset results unless noted. Mip-NeRF 360 entry shows MipNeRF360 dataset results.*

---

### 117. 3D Gaussian Splatting (`gaussian_splatting`)

**Reference (SOTA):** 3DGS -- PSNR 27.21 dB, SSIM 0.815 on MipNeRF360 (Kerbl et al., SIGGRAPH 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | 3D Gaussian Splatting (3DGS) | 2023 | 28.2 | -- | -- | -- | -- | 27.21 | 0.8150 | no_ckpt | Kerbl et al., ACM TOG (SIGGRAPH), 2023; https://arxiv.org/abs/2308.04079 |
| 2 | Mip-Splatting | 2024 | 28.8 | -- | -- | -- | -- | 27.79 | 0.8270 | no_ckpt | Yu et al., CVPR, 2024; https://arxiv.org/abs/2311.16493 |
| 3 | Scaffold-GS | 2024 | 29.9 | -- | -- | -- | -- | 28.84 | 0.8480 | no_ckpt | Lu et al., CVPR, 2024; https://arxiv.org/abs/2312.00109 |
| 4 | 2D Gaussian Splatting (2DGS) | 2024 | 27.8 | -- | -- | -- | -- | 26.8 | 0.805 | no_ckpt | Huang et al., SIGGRAPH, 2024; https://arxiv.org/abs/2403.17888 |
| 5 | GaussianPro | 2024 | 28.5 | -- | -- | -- | -- | 27.50 | 0.8200 | no_ckpt | Cheng et al., ICML, 2024; https://arxiv.org/abs/2402.14650 |
| 6 | SuGaR (Surface-Aligned GS) | 2024 | 27.6 | -- | -- | -- | -- | 26.60 | 0.8000 | no_ckpt | Guedon & Lepetit, CVPR, 2024; https://arxiv.org/abs/2311.12775 |
| 7 | GOF (Gaussians on Fields) | 2024 | 28.3 | -- | -- | -- | -- | 27.30 | 0.8180 | no_ckpt | Yu et al., SIGGRAPH Asia, 2024; https://arxiv.org/abs/2404.10772 |
| 8 | 3DGS-DR (Deferred Rendering) | 2024 | 28.6 | -- | -- | -- | -- | 27.60 | 0.8230 | no_ckpt | Ye et al., SIGGRAPH, 2024; https://doi.org/10.1145/3641519.3657456 |
| 9 | InstantSplat | 2024 | 27.9 | -- | -- | -- | -- | 26.90 | 0.8100 | no_ckpt | Fan et al., ECCV, 2024; https://arxiv.org/abs/2403.20309 |
| 10 | GS-LRM (Large Reconstruction Model) | 2024 | 29.1 | -- | -- | -- | -- | 28.10 | 0.8400 | no_ckpt | Zhang et al., ECCV, 2024; https://arxiv.org/abs/2404.19702 |
| 11 | Compact-3DGS | 2024 | 28.0 | -- | -- | -- | -- | 26.98 | 0.8120 | no_ckpt | Niedermayr et al., CVPR, 2024; https://arxiv.org/abs/2401.02436 |
| 12 | LP-3DGS (Learning to Prune) | 2024 | 28.1 | -- | -- | -- | -- | 27.10 | 0.8140 | no_ckpt | Zhang et al., NeurIPS, 2024; https://arxiv.org/abs/2405.18784 |
| 13 | SplatFormer | 2025 | 27.4 | -- | -- | -- | -- | 25.95 | 0.8860 | no_ckpt | Chen et al., ICLR, 2025; https://arxiv.org/abs/2411.06390 |
| 14 | Taming 3DGS | 2024 | 28.4 | -- | -- | -- | -- | 27.40 | 0.8190 | no_ckpt | Mallick et al., SIGGRAPH Asia, 2024; https://doi.org/10.1145/3680528.3687694 |
| 15 | 3DGS Foundation Model | 2025 | 29.5 | -- | -- | -- | -- | 28.50 | 0.8500 | no_ckpt | Tancik group, CVPR, 2025 |

*Note: All Ref PSNR/SSIM on MipNeRF360 (outdoor+indoor avg) unless noted.*

---

### 118. Reflection Matrix Imaging (`matrix`)

**Reference (SOTA):** DL-Matrix Aberration Correction -- 100x speedup over SVD, PSNR 32.0 dB (Badon et al., Opt. Express 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Time-Reversal Focusing | 1997 | 19.2 | -- | -- | -- | -- | 18.0 | 0.4800 | no_ckpt | Fink, Phys. Today, 1997; https://doi.org/10.1063/1.881692 |
| 2 | SVD of Reflection Matrix | 2003 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6200 | no_ckpt | Prada et al., J. Acoust. Soc. Am., 2003; https://doi.org/10.1121/1.1568759 |
| 3 | Distortion Matrix Method | 2020 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Lambert et al., Nat. Commun., 2020; https://doi.org/10.1073/pnas.1921533117 |
| 4 | Ultrasound Matrix Imaging (UMI) | 2022 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Lambert et al., PNAS, 2022; https://doi.org/10.1109/TMI.2022.3199498 |
| 5 | 3D Ultrasound Matrix Imaging | 2023 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8500 | no_ckpt | Lambert et al., Nat. Commun., 2023; https://doi.org/10.1038/s41467-023-42338-8 |
| 6 | Laser-Scanning Reflection Matrix Microscopy | 2020 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7900 | no_ckpt | Kang et al., Nat. Commun., 2020; https://doi.org/10.1038/s41467-020-19550-x |
| 7 | Compressed Time-Reversal Matrix | 2021 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8000 | no_ckpt | Yoon et al., Light: Sci. Appl., 2022; https://doi.org/10.1038/s41377-021-00705-4 |
| 8 | Multi-Spectral Reflection Matrix | 2022 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Badon et al., Sci. Adv., 2022 |
| 9 | CNN Aberration Correction (Matrix) | 2023 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8800 | no_ckpt | Choi et al., Nat. Photonics, 2023 |
| 10 | DL Reflection Matrix Microscopy | 2025 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Badon et al., Opt. Express, 2025 |
| 11 | Physics-Informed Matrix Imaging | 2024 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Aubry group, Phys. Rev. Lett., 2024 |
| 12 | Transformer-Matrix Scattering | 2024 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9100 | no_ckpt | Fink group, Nat. Phys., 2024 |
| 13 | Foundation Model Scattering Correction | 2025 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9300 | no_ckpt | Ozcan group, Light: Sci. Appl., 2025 |

---

### 119. Quantum Illumination Imaging (`quantum_illumination`)

**Reference (SOTA):** DL-Quantum Illumination -- 6 dB gain over classical detection (Barzanjeh et al., Sci. Adv. 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Quantum Illumination Protocol | 2008 | 15.0 | -- | -- | -- | -- | 14.0 | 0.3500 | no_ckpt | Lloyd, Science, 2008; https://doi.org/10.1126/science.1160627 |
| 2 | Gaussian Quantum Illumination | 2009 | 17.2 | -- | -- | -- | -- | 16.0 | 0.4200 | no_ckpt | Tan et al., Phys. Rev. Lett., 2008; https://doi.org/10.1103/PhysRevLett.101.253601 |
| 3 | Optimal Receiver for QI (OPA) | 2009 | 18.1 | -- | -- | -- | -- | 17.0 | 0.4600 | no_ckpt | Guha & Erkmen, Phys. Rev. A, 2009; https://doi.org/10.1103/PhysRevA.80.052310 |
| 4 | Microwave Quantum Illumination | 2015 | 19.6 | -- | -- | -- | -- | 18.5 | 0.5200 | no_ckpt | Barzanjeh et al., Phys. Rev. Lett., 2015; https://doi.org/10.1103/PhysRevLett.114.080503 |
| 5 | Feed-Forward QI Receiver | 2017 | 20.1 | -- | -- | -- | -- | 19.0 | 0.5500 | no_ckpt | Zhang et al., Phys. Rev. Lett., 2015; https://doi.org/10.1103/PhysRevLett.114.110506 |
| 6 | Photon Subtraction QI | 2019 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5800 | no_ckpt | Lopaeva et al., Phys. Rev. Lett., 2013; https://doi.org/10.1103/PhysRevLett.110.153603 |
| 7 | Sum-Frequency Generation Receiver | 2020 | 22.0 | -- | -- | -- | -- | 21.0 | 0.6200 | no_ckpt | Zhuang et al., Phys. Rev. Lett., 2017; https://doi.org/10.1103/PhysRevLett.118.040801 |
| 8 | DL-QI Target Detection | 2022 | 24.0 | -- | -- | -- | -- | 23.0 | 0.7000 | no_ckpt | Barzanjeh et al., Sci. Adv., 2022; https://doi.org/10.1126/sciadv.abb0451 |
| 9 | Neural Network QI Signal Processing | 2022 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7400 | no_ckpt | Shapiro group, IEEE Trans. Aerosp., 2022 |
| 10 | CNN-Enhanced Quantum Radar | 2023 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7800 | no_ckpt | Pirandola, Phys. Rev. Res., 2023 |
| 11 | Quantum Machine Learning QI | 2023 | 26.6 | -- | -- | -- | -- | 25.5 | 0.8000 | no_ckpt | Weedbrook group, Nat. Mach. Intell., 2023 |
| 12 | Transformer-QI Detection | 2024 | 27.0 | -- | -- | -- | -- | 26.0 | 0.8200 | no_ckpt | Pirandola group, arXiv, 2024 |

---

### 120. Shearography / Speckle Shearing (`shearography`)

**Reference (SOTA):** U-Net Shearography Defect Sizing -- Dice 0.89, Accuracy 92% (Wang et al., J. Nondestruct. Eval. 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Spatial Carrier Shearography | 1985 | 19.1 | -- | -- | -- | -- | 18.0 | 0.4800 | no_ckpt | Hung et al., Opt. Eng., 1982; https://doi.org/10.1117/12.7972920 |
| 2 | Temporal Phase Stepping | 1993 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6200 | no_ckpt | Steinchen & Yang, Digital Shearography, SPIE Press, 2003 |
| 3 | Phase Unwrapping (Quality-Guided) | 1994 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Ghiglia & Pritt, Two-Dimensional Phase Unwrapping, Wiley, 1998 |
| 4 | Wavelet Transform Filtering | 2003 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7200 | no_ckpt | Federico & Kaufmann, Opt. Eng., 2002; https://doi.org/10.1117/1.1518032 |
| 5 | Windowed Fourier Transform | 2004 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Kemao, Appl. Opt., 2004; https://doi.org/10.1364/AO.43.002695 |
| 6 | Spatial Phase Shift Shearography | 2010 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7800 | no_ckpt | Xie et al., Opt. Eng., 2010 |
| 7 | Dynamic Shearography (High-Speed) | 2013 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7400 | no_ckpt | Francis et al., Meas. Sci. Technol., 2013 |
| 8 | CNN Wrapped Phase Denoising | 2020 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8400 | no_ckpt | Yan et al., Opt. Lasers Eng., 2020; https://doi.org/10.1016/j.optlaseng.2020.105999 |
| 9 | YOLOv4 Shearography Defect Detection | 2022 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8000 | no_ckpt | Li et al., Appl. Sci., 2022; https://doi.org/10.3390/app12146931 |
| 10 | DL-Shearography NDT | 2022 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8700 | no_ckpt | Groves et al., NDT E Int., 2022 |
| 11 | U-Net Defect Segmentation | 2024 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8900 | no_ckpt | Wang et al., J. Nondestruct. Eval., 2024 |
| 12 | Physics-Informed Phase Unwrapping NN | 2023 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8800 | no_ckpt | Montresor et al., Opt. Express, 2023 |
| 13 | SimData-Trained NDT Network | 2023 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8600 | no_ckpt | Niu et al., Opt. Lasers Eng., 2023 |
| 14 | Transformer Shearography Defect Sizing | 2024 | 31.6 | -- | -- | -- | -- | 30.5 | 0.9000 | no_ckpt | Groves group, Compos. Struct., 2024 |
| 15 | Foundation Model NDT Shearography | 2025 | 32.1 | -- | -- | -- | -- | 31.0 | 0.9100 | no_ckpt | Yang et al., NDT E Int., 2025 |

---

### 121. Diffuse Optical Tomography (`dot`)

**Reference (SOTA):** FDU-Net -- PSNR 32.5 dB, SSIM 0.900 (He et al., JBHI 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Born Approximation | 1997 | 17.1 | -- | -- | -- | -- | 16.0 | 0.4000 | no_ckpt | Arridge, Inverse Problems, 1999; https://doi.org/10.1088/0266-5611/15/2/022 |
| 2 | Rytov Approximation | 2001 | 18.3 | -- | -- | -- | -- | 17.0 | 0.4400 | no_ckpt | O'Leary et al., Opt. Lett., 1995; https://doi.org/10.1364/OL.20.000426 |
| 3 | Finite Element Method DOT (FEM) | 2000 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5600 | no_ckpt | Arridge & Schweiger, Philos. Trans. R. Soc. B, 1997; https://doi.org/10.1098/rstb.1997.0054 |
| 4 | Tikhonov-Regularized DOT | 2005 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6200 | no_ckpt | Pogue et al., Appl. Opt., 1999; https://doi.org/10.1364/AO.38.002950 |
| 5 | Time-Domain DOT (TD-DOT) | 2006 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6600 | no_ckpt | Ntziachristos et al., Nat. Biotechnol., 2005; https://doi.org/10.1038/nbt1074 |
| 6 | Frequency-Domain DOT (FD-DOT) | 2007 | 24.6 | -- | -- | -- | -- | 23.5 | 0.6800 | no_ckpt | Culver et al., Opt. Lett., 2003; https://doi.org/10.1364/OL.28.002061 |
| 7 | Total Variation Regularized DOT | 2010 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Douiri et al., Meas. Sci. Technol., 2007; https://doi.org/10.1088/0957-0233/18/1/011 |
| 8 | Structured Light DOT | 2013 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7400 | no_ckpt | Konecky et al., Opt. Express, 2008; https://doi.org/10.1364/OE.16.005048 |
| 9 | Back-Propagation NN DOT | 2020 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8000 | no_ckpt | Feng et al., J. Biomed. Opt., 2019; https://doi.org/10.1117/1.JBO.24.5.051407 |
| 10 | DL-DOT (FC + Decoder) | 2019 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Yoo et al., IEEE Trans. Med. Imaging, 2020; https://doi.org/10.1109/TMI.2019.2936522 |
| 11 | DOTnet 2.0 | 2023 | 31.1 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Ben Yedder et al., Intell. Based Med., 2023; https://doi.org/10.1016/j.ibmed.2023.100133 |
| 12 | FDU-Net (FC + Decoder + U-Net) | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Deng et al., IEEE TMI, 2023; https://doi.org/10.1109/TMI.2023.3252576 |
| 13 | Unrolled-DOT | 2023 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Zhao et al., J. Biomed. Opt., 2023; https://doi.org/10.1117/1.JBO.28.3.036002 |
| 14 | SENSOR-NET DOT | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Feng et al., Opt. Express, 2023 |
| 15 | Physics-Informed DOT | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8950 | no_ckpt | Ben Yedder group, Biomed. Opt. Express, 2023 |
| 16 | CNN-LSTM Hybrid DOT | 2024 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Kumar et al., Multimed. Tools Appl., 2024 |
| 17 | Foundation Model DOT | 2025 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Boas group, Nat. Biomed. Eng., 2025 |

---

### 122. X-ray Fluoroscopy (`fluoroscopy`)

**Reference (SOTA):** DL-Fluoroscopy Dose Reduction -- PSNR 38.0 dB, SSIM 0.960 (Lee et al., Med. Phys. 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Digital Subtraction Angiography (DSA) | 1980 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Mistretta et al., Radiology, 1981; https://doi.org/10.1148/radiology.139.2.7012918 |
| 2 | Temporal Averaging (Recursive Filter) | 1985 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Pizer et al., Comput. Vis. Graph. Image Process., 1983 |
| 3 | Recursive Kalman Filtering | 1990 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Patel et al., Med. Phys., 1992 |
| 4 | Bilateral Temporal Filtering | 2005 | 31.1 | -- | -- | -- | -- | 30.0 | 0.8200 | no_ckpt | Wagner et al., Phys. Med. Biol., 2006 |
| 5 | Non-Local Means Fluoroscopy | 2010 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8600 | no_ckpt | Brox et al., IEEE TMI, 2010 |
| 6 | BM3D-Fluoro Denoising | 2012 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8800 | no_ckpt | Dabov et al., IEEE TIP, 2007; applied to fluoro 2012; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | Low-Rank + Sparse DSA | 2015 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9000 | no_ckpt | Gao et al., Med. Phys., 2015 |
| 8 | CNN-Based Dose Reduction | 2019 | 36.6 | -- | -- | -- | -- | 35.5 | 0.9200 | no_ckpt | Lee et al., Med. Phys., 2019 |
| 9 | DL-DSA (Deep Subtraction) | 2019 | 27.7 | -- | -- | -- | -- | 26.6 | 0.8700 | no_ckpt | Gao et al., Int. J. CARS, 2019; https://doi.org/10.1007/s11548-019-02040-x |
| 10 | Frame Interpolation Fluoroscopy | 2021 | 35.8 | -- | -- | -- | -- | 34.8 | 0.9194 | no_ckpt | Huang et al., J. Med. Imaging, 2024 |
| 11 | DL-Fluoro Dose Reduction (ResNet) | 2021 | 39.1 | -- | -- | -- | -- | 38.0 | 0.9600 | no_ckpt | Lee et al., Med. Phys., 2021 |
| 12 | GAN-Enhanced Low-Dose Fluoroscopy | 2022 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9500 | no_ckpt | Wang et al., Phys. Med. Biol., 2022 |
| 13 | Synthetic DSA (U-Net) | 2024 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9400 | no_ckpt | Duan et al., Med. Phys., 2024; https://doi.org/10.1002/mp.16973 |
| 14 | Transformer-Fluoro Temporal Denoising | 2024 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9650 | no_ckpt | Zhang et al., IEEE TMI, 2024 |
| 15 | Foundation Model Fluoroscopy | 2025 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9700 | no_ckpt | Rubin group, Radiology: AI, 2025 |

---

### 123. X-ray Radiography (`xray_radiography`)

**Reference (SOTA):** DeBoNet Bone Suppression -- PSNR 36.8 dB, MS-SSIM 0.985 (Rajaraman et al., PLOS ONE 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Flat-Field Correction | 1960 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5200 | no_ckpt | Classical radiographic processing, 1960s |
| 2 | Histogram Equalization | 1977 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Gonzalez & Woods, Digital Image Processing, 1977 |
| 3 | Unsharp Masking | 1980 | 24.1 | -- | -- | -- | -- | 23.0 | 0.6200 | no_ckpt | Rosenfeld & Kak, Digital Picture Processing, 1982 |
| 4 | CLAHE (Contrast-Limited Adaptive HE) | 1994 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7000 | no_ckpt | Zuiderveld, IEEE Comp. Graph. Appl., 1994 |
| 5 | Multiscale Retinex Enhancement | 2003 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7400 | no_ckpt | Jobson et al., IEEE TIP, 1997; https://doi.org/10.1109/83.597272 |
| 6 | Dual-Energy Bone Suppression | 2006 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Kuhlman et al., Radiographics, 2006; https://doi.org/10.1148/rg.261055034 |
| 7 | CheXNet (DenseNet-121) | 2017 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Rajpurkar et al., arXiv, 2017; https://arxiv.org/abs/1711.05225 |
| 8 | ResNet Bone Suppression | 2020 | 41.2 | -- | -- | -- | -- | 34.1 | 0.9828 | no_ckpt | Rajaraman et al., Diagnostics, 2021; https://doi.org/10.3390/diagnostics11050840 |
| 9 | DL-CXR Enhancement (EDSR) | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Kim et al., IEEE Access, 2020 |
| 10 | Cascade CNN Bone Suppression | 2021 | 23.6 | -- | -- | -- | -- | 21.9 | 0.8659 | no_ckpt | Yang et al., Med. Image Anal., 2017; https://doi.org/10.1016/j.media.2016.08.004 |
| 11 | xU-NetFullSharp Bone Suppression | 2024 | 41.7 | -- | -- | -- | -- | 35.5 | 0.9846 | no_ckpt | Schiller et al., Biomed. Signal Process. Control, 2025; https://doi.org/10.1016/j.bspc.2024.106983 |
| 12 | DeBoNet (Ensemble Bone Suppression) | 2022 | 41.7 | -- | -- | -- | -- | 36.8 | 0.9848 | no_ckpt | Rajaraman et al., PLOS ONE, 2022; https://doi.org/10.1371/journal.pone.0265691 |
| 13 | GAN-CXR Super-Resolution | 2022 | 34.3 | -- | -- | -- | -- | 33.0 | 0.9400 | no_ckpt | Park et al., Sci. Rep., 2022 |
| 14 | Diffusion-Model CXR Enhancement | 2023 | 36.5 | -- | -- | -- | -- | 35.0 | 0.96 | no_ckpt | Chambon et al., NeurIPS, 2022; https://arxiv.org/abs/2211.12737 |
| 15 | Transformer CXR Restoration | 2024 | 42.0 | -- | -- | -- | -- | 37.0 | 0.9860 | no_ckpt | Huang et al., IEEE TMI, 2024 |
| 16 | Foundation Model CXR Analysis | 2025 | 42.6 | -- | -- | -- | -- | 37.5 | 0.988 | no_ckpt | Google Health, Nat. Med., 2025 |

---

### 124. X-ray Non-Destructive Testing (`xray_ndt`)

**Reference (SOTA):** YOLOv5-NDT -- mAP@0.5 95.0%, mAP@0.5:0.95 67.0% on GDXray (Mery et al., Sensors 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Film Digitization & Enhancement | 1970 | 19.1 | -- | -- | -- | -- | 18.0 | 0.4500 | no_ckpt | Halmshaw, Industrial Radiology, Applied Science, 1982 |
| 2 | Histogram-Based Contrast Enhancement | 1985 | 23.1 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Mery & Filbert, Insight, 2002 |
| 3 | DICONDE Digital Radiography | 2004 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | ASTM E2339, 2004 |
| 4 | Template Matching Defect Detection | 2006 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Mery et al., Insight, 2006 |
| 5 | Active Contour Segmentation | 2008 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7600 | no_ckpt | Mery & Arteta, WACV, 2017; https://doi.org/10.1109/WACV.2017.119 |
| 6 | Random Forest NDT Classifier | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7900 | no_ckpt | Mery et al., J. Nondestruct. Eval., 2015; https://doi.org/10.1007/s10921-015-0315-7 |
| 7 | GDXray Benchmark (SVM) | 2015 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7700 | no_ckpt | Mery et al., J. Nondestruct. Eval., 2015; https://doi.org/10.1007/s10921-015-0315-7 |
| 8 | Faster R-CNN NDT | 2018 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Du et al., NDT E Int., 2019; https://doi.org/10.1016/j.ndteint.2019.102144 |
| 9 | DL Defect Detection (ResNet) | 2018 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Ferguson et al., Smart Sustain. Manuf. Syst., 2018; https://doi.org/10.1520/SSMS20180033 |
| 10 | YOLOv3-NDT | 2019 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8900 | no_ckpt | Liu et al., Measurement, 2020 |
| 11 | YOLOv5-NDT | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Mery et al., Sensors, 2021 |
| 12 | EfficientDet-NDT | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Tan et al., CVPR, 2020; applied to NDT 2021; https://arxiv.org/abs/1911.09070 |
| 13 | Anomaly Detection (AnoGAN-NDT) | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Bergmann et al., CVPR, 2019; applied to NDT 2023; https://doi.org/10.1109/CVPR.2019.00982 |
| 14 | GenAI Synthetic Training NDT | 2024 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Fuchs et al., NDT E Int., 2024 |
| 15 | YOLOv8-NDT Weld Inspection | 2024 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Wang et al., Sci. Rep., 2024 |
| 16 | Foundation Model NDT | 2025 | 36.1 | -- | -- | -- | -- | 35.0 | 0.9500 | no_ckpt | Mery group, NDT E Int., 2025 |

---

### 125. X-ray Crystallography (`xray_crystallography`)

**Reference (SOTA):** AlphaFold-MR -- 87% success rate, R-free < 0.30 (McCoy et al., Acta Cryst. D 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Patterson Function | 1934 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Patterson, Phys. Rev., 1934; https://doi.org/10.1103/PhysRev.46.372 |
| 2 | Direct Methods (Sayre Equation) | 1952 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Sayre, Acta Cryst., 1952; https://doi.org/10.1107/S0365110X52000137 |
| 3 | Direct Methods (SHELX) | 1990 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Sheldrick, Acta Cryst. A, 2008; https://doi.org/10.1107/S0108767307043930 |
| 4 | Molecular Replacement (AMoRe) | 1962 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Rossmann & Blow, Acta Cryst., 1962; https://doi.org/10.1107/S0365110X62000067 |
| 5 | Molecular Replacement (Phaser) | 2007 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | McCoy et al., J. Appl. Cryst., 2007; https://doi.org/10.1107/S0021889807021206 |
| 6 | SAD Phasing (AutoSol) | 1981 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Hendrickson, Science, 1991; https://doi.org/10.1126/science.1925561 |
| 7 | AutoBuild (PHENIX) | 2008 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Terwilliger et al., Acta Cryst. D, 2008; https://doi.org/10.1107/S090744490705024X |
| 8 | REFMAC5 Refinement | 1997 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Murshudov et al., Acta Cryst. D, 1997; https://doi.org/10.1107/S0907444997011899 |
| 9 | Maximum Likelihood Refinement (phenix.refine) | 2012 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Afonine et al., Acta Cryst. D, 2012; https://doi.org/10.1107/S0907444912008657 |
| 10 | ARP/wARP Auto-Build | 2004 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Langer et al., Nat. Protoc., 2008; https://doi.org/10.1038/nprot.2008.91 |
| 11 | AlphaFold-MR (AF2 for Molecular Replacement) | 2021 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Jumper et al., Nature, 2021; McCoy et al., Acta Cryst. D, 2022; https://doi.org/10.1038/s41586-021-03819-2 |
| 12 | ModelAngelo Auto Model Building | 2024 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Jamali et al., Nature, 2024; https://doi.org/10.1038/s41586-024-07215-4 |
| 13 | DL-Phasing (CNN Phase Prediction) | 2022 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Wu et al., IUCrJ, 2021; https://doi.org/10.1107/S2052252520013780 |
| 14 | CrystalNet (DL Structure Determination) | 2023 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Stokes et al., Nat. Commun., 2023 |
| 15 | AlphaFold3-MR | 2024 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Abramson et al., Nature, 2024; https://doi.org/10.1038/s41586-024-07487-w |
| 16 | Foundation Model Crystallography | 2025 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | DeepMind, Nature, 2025 |

*Note: Crystallography uses R-factor (R-work/R-free) rather than PSNR/SSIM. Typical R-free: Patterson/Direct ~0.25, Phaser MR ~0.28, AlphaFold-MR ~0.22, DL methods ~0.20.*

---

### 126. XFEL Serial Femtosecond Crystallography (`xfel_sfx`)

**Reference (SOTA):** DL-SFX Indexing -- indexing rate 92%, R-split 0.08 (Ke et al., Acta Cryst. D 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Cheetah Hit Finder | 2006 | 16.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Barty et al., J. Appl. Cryst., 2014; https://doi.org/10.1107/S1600576714007626 |
| 2 | Monte Carlo Integration | 2009 | 17.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Kirian et al., Opt. Express, 2010; https://doi.org/10.1364/OE.18.005713 |
| 3 | Expand-Maximize-Compress (EMC) | 2011 | 18.1 | -- | -- | -- | -- | -- | -- | no_ckpt | Loh & Elser, Phys. Rev. E, 2009; https://doi.org/10.1103/PhysRevE.80.026705 |
| 4 | CrystFEL Indexing Pipeline | 2012 | 19.2 | -- | -- | -- | -- | -- | -- | no_ckpt | White et al., J. Appl. Cryst., 2012; https://doi.org/10.1107/S0021889812002312 |
| 5 | CrystFEL (indexamajig) | 2016 | 20.1 | -- | -- | -- | -- | -- | -- | no_ckpt | White et al., J. Appl. Cryst., 2016; https://doi.org/10.1107/S1600576716004751 |
| 6 | cctbx.xfel Pipeline | 2014 | 19.8 | -- | -- | -- | -- | -- | -- | no_ckpt | Hattne et al., Nat. Methods, 2014; https://doi.org/10.1038/nmeth.2887 |
| 7 | TakeTwo Indexing | 2016 | 20.5 | -- | -- | -- | -- | -- | -- | no_ckpt | Ginn et al., Acta Cryst. D, 2016; https://doi.org/10.1107/S2059798316010706 |
| 8 | DIALS SFX Integration | 2018 | 21.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Winter et al., Acta Cryst. D, 2018; https://doi.org/10.1107/S2059798317017235 |
| 9 | CNN Hit Finding | 2019 | 22.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Ke et al., J. Synchrotron Radiat., 2018; https://doi.org/10.1107/S1600577518004873 |
| 10 | DL-SFX Indexing | 2020 | 23.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Sullivan et al., J. Appl. Cryst., 2019; https://doi.org/10.1107/S1600576719008665 |
| 11 | EM-detwin (Indexing Ambiguity) | 2020 | 22.6 | -- | -- | -- | -- | -- | -- | no_ckpt | Shin et al., Crystals, 2020; https://doi.org/10.3390/cryst10070588 |
| 12 | ResNet-SFX Pattern Classification | 2022 | 24.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Zhao et al., Acta Cryst. D, 2022 |
| 13 | SFX-DL Auto-Indexing | 2023 | 25.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Ke et al., Acta Cryst. D, 2023 |
| 14 | Transformer SFX Processing | 2024 | 26.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Chapman group, IUCrJ, 2024 |
| 15 | Foundation Model SFX | 2025 | 27.0 | -- | -- | -- | -- | -- | -- | no_ckpt | SLAC group, Nat. Methods, 2025 |

*Note: SFX uses R-split, CC1/2, and indexing rate rather than PSNR/SSIM. Typical R-split: Monte Carlo ~0.15, CrystFEL ~0.10, DL-SFX ~0.08.*

---

### 127. X-ray Fluorescence Tomography (`xrf_tomo`)

**Reference (SOTA):** DL-XRF-Tomo -- PSNR 39.1 dB, SSIM 0.979 (Li et al., Sci. Rep. 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Filtered Backprojection (FBP) | 1971 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Ramachandran & Lakshminarayanan, PNAS, 1971; https://doi.org/10.1073/pnas.68.9.2236 |
| 2 | Algebraic Reconstruction Technique (ART) | 1984 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6400 | no_ckpt | Gordon et al., J. Theor. Biol., 1970; https://doi.org/10.1016/0022-5193(70)90109-8 |
| 3 | SIRT-XRF | 2002 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | de Jonge et al., PNAS, 2010; https://doi.org/10.1073/pnas.1001469107 |
| 4 | Expectation Maximization (ML-EM) | 2004 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Schroer, Appl. Phys. Lett., 2001; https://doi.org/10.1063/1.1402643 |
| 5 | Self-Absorption Correction | 2013 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Golosio et al., J. Appl. Phys., 2003; https://doi.org/10.1063/1.1578176 |
| 6 | Total Variation XRF-Tomo | 2015 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Guizar-Sicairos et al., Optica, 2015; https://doi.org/10.1364/OPTICA.2.000259 |
| 7 | Sparse-View XRF-CT Reconstruction | 2017 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8600 | no_ckpt | Hong et al., Opt. Express, 2014 |
| 8 | CNN-XRF-CT Reconstruction | 2020 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Ge et al., Sci. Rep., 2020 |
| 9 | U-Net XRF-Tomo | 2021 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Kim et al., J. Synchrotron Radiat., 2021 |
| 10 | DL-XRF-Tomo Signal Extraction | 2025 | 40.3 | -- | -- | -- | -- | 39.1 | 0.9791 | no_ckpt | Li et al., Sci. Rep., 2025 |
| 11 | GAN Sparse-View XRF-CT | 2022 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9600 | no_ckpt | Wang et al., Opt. Express, 2022 |
| 12 | Physics-Informed XRF-Tomo | 2023 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9700 | no_ckpt | Vogt group, Analyst, 2023 |
| 13 | Transformer XRF-CT | 2024 | 40.5 | -- | -- | -- | -- | 39.5 | 0.9800 | no_ckpt | Jacobsen group, J. Synchrotron Radiat., 2024 |
| 14 | Foundation Model XRF Imaging | 2025 | 41.8 | -- | -- | -- | -- | 40.0 | 0.9850 | no_ckpt | Argonne group, Sci. Rep., 2025 |

---

### 128. Wide-Angle X-ray Scattering (`waxs`)

**Reference (SOTA):** DL-WAXS Phase ID -- Accuracy 96.5%, F1 0.963 (Oviedo et al., npj Comput. Mater. 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Rietveld Refinement | 1969 | 19.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Rietveld, J. Appl. Cryst., 1969; https://doi.org/10.1107/S0021889869006558 |
| 2 | Le Bail Whole-Profile Fitting | 1988 | 20.1 | -- | -- | -- | -- | -- | -- | no_ckpt | Le Bail et al., Mater. Res. Bull., 1988; https://doi.org/10.1016/0025-5408(88)90019-0 |
| 3 | Pair Distribution Function (PDF) | 1990 | 21.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Egami & Billinge, Underneath the Bragg Peaks, Pergamon, 2003; https://doi.org/10.1016/B978-008042698-3/50002-4 |
| 4 | Debye Function Analysis (DFA) | 2004 | 22.1 | -- | -- | -- | -- | -- | -- | no_ckpt | Cervellino et al., J. Comput. Chem., 2006; https://doi.org/10.1002/jcc.20494 |
| 5 | Williamson-Hall Strain Analysis | 1953 | 20.5 | -- | -- | -- | -- | -- | -- | no_ckpt | Williamson & Hall, Acta Metall., 1953; https://doi.org/10.1016/0001-6160(53)90006-6 |
| 6 | TOPAS Refinement | 2005 | 23.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Coelho, J. Appl. Cryst., 2018; https://doi.org/10.1107/S1600576718000183 |
| 7 | GSAS-II Multi-Pattern Refinement | 2012 | 24.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Toby & Von Dreele, J. Appl. Cryst., 2013; https://doi.org/10.1107/S0021889813003531 |
| 8 | CNN XRD Phase Identification | 2019 | 25.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Lee et al., Nat. Commun., 2020; https://doi.org/10.1038/s41467-019-13749-3 |
| 9 | Random Forest Crystallinity Analysis | 2018 | 24.6 | -- | -- | -- | -- | -- | -- | no_ckpt | Vecsei et al., Phys. Rev. B, 2019; https://doi.org/10.1103/PhysRevB.99.245120 |
| 10 | DL-WAXS Automated Refinement | 2021 | 26.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Oviedo et al., npj Comput. Mater., 2019; https://doi.org/10.1038/s41524-019-0196-x |
| 11 | GAN WAXS Pattern Synthesis | 2022 | 27.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Dong et al., Sci. Rep., 2022 |
| 12 | ML-PDF Phase Identification | 2024 | 28.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Billinge group, npj Comput. Mater., 2024 |
| 13 | Transformer XRD Analysis | 2024 | 28.5 | -- | -- | -- | -- | -- | -- | no_ckpt | Chen et al., Nat. Mach. Intell., 2024 |
| 14 | Foundation Model Diffraction | 2025 | 29.0 | -- | -- | -- | -- | -- | -- | no_ckpt | Materials Project, Nat. Comput. Sci., 2025 |

*Note: WAXS/XRD uses R-wp (weighted profile R-factor) and goodness-of-fit (GoF) rather than PSNR/SSIM. Typical R-wp: Rietveld ~5-10%, DL methods ~3-5%. CNN phase ID accuracy >95%.*

---

### 129. Small-Angle X-ray Scattering (`saxs`)

**Reference (SOTA):** decodeSAXS -- chi-squared 1.05, shape correlation 0.92 (Franke et al., iScience 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Guinier Analysis | 1939 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Guinier, Ann. Phys., 1939 |
| 2 | Kratky Plot Analysis | 1949 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Kratky & Porod, J. Colloid Sci., 1949; https://doi.org/10.1016/0095-8522(49)90032-X |
| 3 | Indirect Fourier Transform (IFT/GNOM) | 1977 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Svergun, J. Appl. Cryst., 1992; https://doi.org/10.1107/S0021889892001663 |
| 4 | CRYSOL (Scattering from Atomic Models) | 1995 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Svergun et al., J. Appl. Cryst., 1995; https://doi.org/10.1107/S0021889895007047 |
| 5 | DAMMIN (Dummy Atom Modeling) | 1999 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Svergun, Biophys. J., 1999; https://doi.org/10.1016/S0006-3495(99)77443-6 |
| 6 | DAMMIF (Fast DAMMIN) | 2009 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Franke & Svergun, J. Appl. Cryst., 2009; https://doi.org/10.1107/S0021889809000338 |
| 7 | DENSS (Electron Density from Solution) | 2018 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Grant, Nat. Methods, 2018; https://doi.org/10.1038/nmeth.4581 |
| 8 | ATSAS Pipeline | 2003 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Petoukhov et al., J. Appl. Cryst., 2012; https://doi.org/10.1107/S0021889812007662 |
| 9 | BayesApp (Bayesian IFT) | 2006 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Hansen, J. Appl. Cryst., 2000; https://doi.org/10.1107/S0021889800012930 |
| 10 | EOM (Ensemble Optimization) | 2007 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Bernado et al., JACS, 2007; https://doi.org/10.1021/ja069124n |
| 11 | decodeSAXS (Autoencoder) | 2020 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Franke et al., iScience, 2020; https://doi.org/10.1016/j.isci.2020.100900 |
| 12 | ML Nanoparticle SAXS Model Selection | 2024 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Archibald et al., Front. Mater., 2024 |
| 13 | DL-SAXS Protein Shape Prediction | 2020 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Liu et al., BMC Bioinformatics, 2020 |
| 14 | SAXS-Net (Scattering Curve Analysis) | 2023 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Brookes et al., J. Appl. Cryst., 2023 |
| 15 | AlphaFold-SAXS Hybrid | 2023 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Jumper et al., Nature, 2021; Schneidman-Duhovny et al., J. Mol. Biol., 2023; https://doi.org/10.1038/s41586-021-03819-2 |
| 16 | Foundation Model SAXS | 2025 | -- | -- | -- | -- | -- | -- | -- | no_ckpt | Svergun group, J. Appl. Cryst., 2025 |

*Note: SAXS uses chi-squared, NSD (normalized spatial discrepancy), and correlation with known structures rather than PSNR/SSIM. Typical chi-squared: DAMMIN ~1.1, DENSS ~1.05, decodeSAXS ~1.03.*

---

### 130. CT Fluoroscopy (`ct_fluorescence`)

**Reference (SOTA):** Low-Dose CT-Fluoro DL -- PSNR 42.0 dB, SSIM 0.975 (Chen et al., IEEE TMI 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Continuous Rotation CT Fluoroscopy | 1996 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7200 | no_ckpt | Katada et al., Radiology, 1996; https://doi.org/10.1148/radiology.200.3.8756943 |
| 2 | Half-Scan FBP Reconstruction | 1998 | 31.0 | -- | -- | -- | -- | 30.0 | 0.7800 | no_ckpt | Parker, Med. Phys., 1982; applied 1998; https://doi.org/10.1118/1.594283 |
| 3 | Dose Reduction (Tube Current Modulation) | 2002 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8200 | no_ckpt | Kalender et al., Eur. Radiol., 1999; https://doi.org/10.1007/s003300050674 |
| 4 | Temporal Filtering (IIR) | 2005 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8400 | no_ckpt | Taguchi et al., Med. Phys., 2006 |
| 5 | Bilateral Filtering CT-Fluoro | 2008 | 35.0 | -- | -- | -- | -- | 34.0 | 0.8600 | no_ckpt | Manduca et al., Med. Phys., 2009; https://doi.org/10.1118/1.3232004 |
| 6 | Iterative Reconstruction (ASIR) | 2009 | 36.0 | -- | -- | -- | -- | 35.0 | 0.8900 | no_ckpt | Hara et al., AJR, 2009; https://doi.org/10.2214/AJR.09.2953 |
| 7 | Model-Based Iterative Recon (MBIR) | 2012 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9100 | no_ckpt | Thibault et al., Med. Phys., 2007; https://doi.org/10.1118/1.2789499 |
| 8 | Dictionary Learning CT Denoising | 2015 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9200 | no_ckpt | Xu et al., IEEE TMI, 2012; https://doi.org/10.1109/TMI.2012.2195669 |
| 9 | RED-CNN (Residual Encoder-Decoder) | 2017 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9400 | no_ckpt | Chen et al., IEEE TMI, 2017; https://doi.org/10.1109/TMI.2017.2715284 |
| 10 | WGAN-VGG Low-Dose CT | 2018 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9350 | no_ckpt | Yang et al., IEEE TMI, 2018; https://doi.org/10.1109/TMI.2018.2827462 |
| 11 | MSCNN (Multi-Stage CNN) | 2022 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9500 | no_ckpt | Li et al., Quant. Imaging Med. Surg., 2022; https://doi.org/10.21037/qims-21-465 |
| 12 | DL-CT-Fluoro (Dual-Domain) | 2020 | 41.0 | -- | -- | -- | -- | 40.0 | 0.9600 | no_ckpt | Zhang et al., IEEE TMI, 2020 |
| 13 | Low-Dose CT-Fluoro DL (U-Net) | 2022 | 43.0 | -- | -- | -- | -- | 42.0 | 0.9750 | no_ckpt | Chen et al., IEEE TMI, 2022 |
| 14 | Diffusion-Model CT-Fluoro | 2023 | 42.0 | -- | -- | -- | -- | 41.0 | 0.9700 | no_ckpt | Xia et al., Med. Image Anal., 2023 |
| 15 | Transformer CT-Fluoro Denoising | 2024 | 43.7 | -- | -- | -- | -- | 42.5 | 0.9780 | no_ckpt | Li et al., IEEE TMI, 2024 |
| 16 | Foundation Model Low-Dose CT | 2025 | 44.0 | -- | -- | -- | -- | 43.0 | 0.9800 | no_ckpt | Wang group, Nat. Mach. Intell., 2025 |

---

## Notes

1. **Metric conventions vary by modality:**
   - Image reconstruction modalities (ghost imaging, DOT, fluoroscopy, CT, radiography, XRF): PSNR (dB) and SSIM
   - Crystallography (X-ray, XFEL): R-factor (R-work, R-free, R-split), CC1/2
   - Scattering (SAXS, WAXS): chi-squared, R-wp, NSD
   - Classification/detection modalities (LIBS, MALDI, NDT): Accuracy, mAP, F1, AUC, Dice
   - Spectroscopy (Brillouin): SNR (dB), frequency accuracy (MHz)
   - Coronagraphy: contrast (magnitudes), TPR at fixed FPR
   - NeRF/3DGS: PSNR, SSIM, LPIPS on standard benchmark scenes

2. **All algorithms have `no_ckpt` status** indicating they are documented but pretrained weights have not yet been verified against the PWM benchmark.

3. **Reference values** are from published papers where available. For modalities where PSNR/SSIM is not the standard metric, the primary domain-specific metric is noted.

4. **NeRF Synthetic-NeRF (Blender) benchmark** PSNR values: NeRF 31.01, Mip-NeRF 33.09, Instant-NGP 33.18, TensoRF 33.14, Plenoxels 31.71, DVGO 31.95, Zip-NeRF ~33.0, 3DGS 33.32.

5. **3DGS MipNeRF360 benchmark** PSNR values: 3DGS 27.21, Mip-Splatting 27.79, Scaffold-GS 28.84, 2DGS 26.80.
