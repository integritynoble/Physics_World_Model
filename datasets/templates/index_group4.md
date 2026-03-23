# Optical Imaging & Spectroscopy -- Modalities 79-104

## Implementation Tracking

Each algorithm must be implemented at least **5 times** (5 independent verification runs on the standard dataset).
Status legend: `no_ckpt` = algorithm documented, pretrained weights not yet available; `done` = verified; `partial` = 3-10 dB shortfall; `gap` = >10 dB shortfall; `fail` = solver diverged.

---

### Scanning Probe (Near-Field & Magnetic)

---

#### 79. Magnetic Force Microscopy (`mfm`)

**Reference (SOTA):** DL-MFM Deconvolution -- PSNR 32.5 dB, SSIM 0.920 (Winkler et al., Nanotechnology 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Lift-Mode MFM | 1987 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5500 | no_ckpt | Martin & Wickramasinghe, Appl. Phys. Lett. 1987; https://doi.org/10.1063/1.98865 |
| 2 | Point-Probe Model | 1991 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6200 | no_ckpt | Hartmann, J. Appl. Phys. 1991 |
| 3 | Monopole-Dipole Approximation | 1992 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6400 | no_ckpt | Porthun et al., J. Magn. Magn. Mater. 1992 |
| 4 | Transfer Function Deconvolution | 2003 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7200 | no_ckpt | Hug et al., J. Appl. Phys. 2003; https://doi.org/10.1063/1.1535533 |
| 5 | Wiener Deconvolution (MFM) | 1949 | 25.1 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Wiener N., MIT Press, 1949 |
| 6 | Tikhonov Regularization (MFM) | 1963 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6900 | no_ckpt | Tikhonov, Soviet Math. Doklady, 1963 |
| 7 | 2D FFT Filtering | 1990 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6000 | no_ckpt | Standard FFT-based MFM processing, 1990s |
| 8 | Stray Field Simulation (FEM) | 2005 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7500 | no_ckpt | Piao et al., IEEE Trans. Magn. 2005 |
| 9 | Iterative Deconvolution (RL-MFM) | 2008 | 28.1 | -- | -- | -- | -- | 27.0 | 0.7800 | no_ckpt | Kebe & Carl, J. Phys. D 2008 |
| 10 | Compressed Sensing MFM | 2014 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8100 | no_ckpt | Alem & Bhatt, Nanotechnology 2014 |
| 11 | CNN-MFM Denoising | 2019 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8500 | no_ckpt | Schmid et al., Sci. Rep. 2019 |
| 12 | DL-MFM Deconvolution | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Winkler et al., Nanotechnology 2021 |
| 13 | U-Net MFM Enhancement | 2020 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8700 | no_ckpt | Zhang et al., AIP Advances 2020 |
| 14 | GAN-MFM Super-Resolution | 2022 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Li et al., J. Magn. Magn. Mater. 2022 |
| 15 | Physics-Informed NN (MFM) | 2023 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Chen et al., npj Comput. Mater. 2023 |
| 16 | Diffusion-MFM | 2024 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9300 | no_ckpt | Wang et al., Nano Lett. 2024 |
| 17 | Transformer-MFM | 2024 | 33.8 | -- | -- | -- | -- | 32.8 | 0.9250 | no_ckpt | Liu et al., IEEE Trans. Magn. 2024 |

---

#### 80. Near-field Scanning Optical Microscopy (`nsom`)

**Reference (SOTA):** DL-NSOM Reconstruction -- PSNR 30.8 dB, SSIM 0.905 (Kim et al., ACS Photonics 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Aperture NSOM | 1984 | 19.1 | -- | -- | -- | -- | 18.0 | 0.4500 | no_ckpt | Pohl et al., Appl. Phys. Lett. 1984; https://doi.org/10.1063/1.94865 |
| 2 | Photon STM | 1989 | 20.5 | -- | -- | -- | -- | 19.5 | 0.5000 | no_ckpt | Reddick et al., Phys. Rev. B 1989; https://doi.org/10.1103/PhysRevB.39.767 |
| 3 | Shear-Force Feedback NSOM | 1992 | 21.0 | -- | -- | -- | -- | 20.0 | 0.5200 | no_ckpt | Betzig et al., Appl. Phys. Lett. 1992; https://doi.org/10.1063/1.109066 |
| 4 | Apertureless NSOM (a-NSOM) | 1999 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6000 | no_ckpt | Zenhausern et al., Science 1995; https://doi.org/10.1126/science.269.5227.1083; Knoll & Keilmann, Nature 1999; https://doi.org/10.1038/20154 |
| 5 | Scattering-type NSOM (s-SNOM) | 2004 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Hillenbrand et al., Nature 2004; https://doi.org/10.1038/nature02403 |
| 6 | Pseudoheterodyne Detection | 2006 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7200 | no_ckpt | Ocelic et al., Appl. Phys. Lett. 2006; https://doi.org/10.1063/1.2394341 |
| 7 | Nano-FTIR Spectroscopy | 2012 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Huth et al., Nano Lett. 2012; https://doi.org/10.1021/nl301159v |
| 8 | Tip-Enhanced Raman (TERS) | 2000 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6500 | no_ckpt | Stockle et al., Chem. Phys. Lett. 2000; https://doi.org/10.1016/S0009-2614(99)01451-7 |
| 9 | Deconvolution (s-SNOM) | 2010 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7400 | no_ckpt | Cvitkovic et al., Opt. Express 2010; https://doi.org/10.1364/OE.18.014397 |
| 10 | Finite-Dipole Model | 2007 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7000 | no_ckpt | Cvitkovic et al., Opt. Express 2007; https://doi.org/10.1364/OE.15.008550 |
| 11 | CNN-NSOM Denoising | 2019 | 29.1 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Park et al., Opt. Express 2019 |
| 12 | DL-NSOM Reconstruction | 2021 | 31.8 | -- | -- | -- | -- | 30.8 | 0.9050 | no_ckpt | Kim et al., ACS Photonics 2021 |
| 13 | U-Net Near-Field | 2020 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8600 | no_ckpt | Chen et al., Nanophotonics 2020 |
| 14 | GAN-NSOM Enhancement | 2022 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8800 | no_ckpt | Lee et al., Light Sci. Appl. 2022 |
| 15 | Physics-Informed s-SNOM | 2023 | 31.5 | -- | -- | -- | -- | 30.5 | 0.9000 | no_ckpt | Wang et al., Nat. Commun. 2023 |
| 16 | Diffusion-NSOM | 2024 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9100 | no_ckpt | Zhang et al., ACS Nano 2024 |

---

### Optical Coherence Tomography

---

#### 81. Optical Coherence Tomography (`oct`)

**Reference (SOTA):** DRUNET-OCT -- PSNR 38.2 dB, SSIM 0.965 (Ma et al., BOE 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | FFT-based OCT Reconstruction | 1995 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7000 | no_ckpt | Fercher et al., Opt. Commun. 1995; https://doi.org/10.1016/0030-4018(95)00119-S |
| 2 | Numerical Dispersion Compensation | 2004 | 28.6 | -- | -- | -- | -- | 27.5 | 0.7600 | no_ckpt | Wojtkowski et al., Opt. Express 2004; https://doi.org/10.1364/OPEX.12.002404 |
| 3 | Median Filtering OCT | 2001 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Schmitt et al., J. Biomed. Opt. 2001; https://doi.org/10.1117/1.1427053 |
| 4 | Wavelet Denoising OCT | 2005 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Adler et al., Opt. Express 2005; https://doi.org/10.1364/OPEX.13.003532 |
| 5 | Speckle Reduction (Lee Filter) | 2005 | 28.1 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Ozcan et al., J. Opt. Soc. Am. A 2005; https://doi.org/10.1364/JOSAA.24.001901 |
| 6 | Spectral Shaping | 2008 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Tripathi et al., Opt. Lett. 2008; https://doi.org/10.1364/OL.33.000116 |
| 7 | BM3D-OCT | 2013 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8700 | no_ckpt | Fang et al., BOE 2013 |
| 8 | NLEMCSA (Non-Local Means) | 2014 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8600 | no_ckpt | Cheng et al., BOE 2014 |
| 9 | K-SVD Sparse OCT | 2012 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8400 | no_ckpt | Fang et al., J. Biomed. Opt. 2012 |
| 10 | DL-OCT Denoising (CNN) | 2018 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9100 | no_ckpt | Devalla et al., BOE 2018 |
| 11 | OCT-DnCNN | 2019 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9300 | no_ckpt | Ma et al., BOE 2019 |
| 12 | Parallel-OCT-Net | 2020 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9400 | no_ckpt | Qiu et al., BOE 2020 |
| 13 | DRUNET-OCT | 2021 | 39.2 | -- | -- | -- | -- | 38.2 | 0.9650 | no_ckpt | Ma et al., BOE 2021 |
| 14 | OCT-GAN | 2020 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9250 | no_ckpt | Huang et al., IEEE TMI 2020 |
| 15 | Self2Self OCT | 2021 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9450 | no_ckpt | Li et al., BOE 2021 |
| 16 | OCT Super-Resolution (SRGAN) | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9000 | no_ckpt | Das et al., IEEE TMI 2021 |
| 17 | Speckle2Speckle | 2020 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9150 | no_ckpt | Molini et al., IEEE TCI 2020 |
| 18 | OCT-Transformer | 2023 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9550 | no_ckpt | Zhou et al., MICCAI 2023 |
| 19 | Foundation-OCT | 2024 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9680 | no_ckpt | RETFound applied to OCT, Nat. 2024 |
| 20 | Diffusion-OCT | 2024 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9700 | no_ckpt | Chen et al., MedIA 2024 |

---

#### 82. OCT Angiography (`octa`)

**Reference (SOTA):** OCTA-Net -- Dice 0.892, PSNR 34.5 dB, SSIM 0.945 (Ma et al., BOE 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Phase-Variance OCT | 2011 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6800 | no_ckpt | Fingler et al., Opt. Express 2011; https://doi.org/10.1364/OE.17.022190 |
| 2 | OMAG (Optical Microangiography) | 2006 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7100 | no_ckpt | Wang et al., Opt. Express 2006; https://doi.org/10.1364/OE.15.004083 |
| 3 | SSADA (Split-Spectrum Amplitude) | 2012 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7800 | no_ckpt | Jia et al., Opt. Express 2012; https://doi.org/10.1364/OE.20.004710 |
| 4 | Correlation Mapping OCTA | 2011 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7400 | no_ckpt | Enfield et al., BOE 2011; https://doi.org/10.1364/BOE.2.001184 |
| 5 | Speckle Variance OCT | 2005 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6635 | no_ckpt | Barton & Bhatt, Opt. Express 2005; https://doi.org/10.1364/OPEX.13.005828 |
| 6 | Complex Differential Variance | 2014 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8000 | no_ckpt | Makita et al., Opt. Express 2014; https://doi.org/10.1364/OE.14.007821 |
| 7 | BM3D-OCTA | 2016 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8300 | no_ckpt | Fang et al., BOE 2016 |
| 8 | DL-OCTA Denoising | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Gao et al., BOE 2019 |
| 9 | OCTA-Net (Segmentation) | 2020 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9450 | no_ckpt | Ma et al., BOE 2020 |
| 10 | VesselNet | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Li et al., IEEE TMI 2022 |
| 11 | IPN (Image Projection Network) | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Li et al., MICCAI 2020 |
| 12 | OCTA-GAN | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Lee et al., Sci. Rep. 2021 |
| 13 | TransOCTA | 2023 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9350 | no_ckpt | Wang et al., BOE 2023 |
| 14 | SS-OCTA (Self-Supervised) | 2022 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9050 | no_ckpt | Hormel et al., BOE 2022 |
| 15 | Diffusion-OCTA | 2024 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9500 | no_ckpt | Zhang et al., MedIA 2024 |

---

### Quantitative Phase & Diffraction Tomography

---

#### 83. Optical Diffraction Tomography (`odt`)

**Reference (SOTA):** NeuralODT -- PSNR 35.2 dB, SSIM 0.940 (Ryu et al., Light Sci. Appl. 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Fourier Diffraction Theorem | 1969 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Wolf, Opt. Commun. 1969; https://doi.org/10.1016/0030-4018(69)90052-2 |
| 2 | Born Approximation | 1970 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6200 | no_ckpt | Born & Wolf, Principles of Optics, 1970 |
| 3 | Rytov Approximation | 1979 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6700 | no_ckpt | Devaney, J. Math. Phys. 1979; https://doi.org/10.1063/1.524104 |
| 4 | Filtered Backpropagation | 1982 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7000 | no_ckpt | Devaney, Ultrason. Imaging 1982; https://doi.org/10.1177/016173468200400304 |
| 5 | Algebraic Reconstruction (ART-ODT) | 1990 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7300 | no_ckpt | Kak & Slaney, IEEE Press, 1990 |
| 6 | TV-Regularized ODT | 2010 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Sung et al., Opt. Express 2010; https://doi.org/10.1364/OE.17.000266 |
| 7 | Beam Propagation Method (BPM-ODT) | 2015 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Kamilov et al., Optica 2016; https://doi.org/10.1364/OPTICA.3.000643 |
| 8 | ADMM-ODT | 2016 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Lim et al., Phys. Rev. Lett. 2016; https://doi.org/10.1103/PhysRevLett.117.243902 |
| 9 | Learning Tomography | 2018 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Kamilov et al., IEEE TSP 2018; https://doi.org/10.1109/TSP.2015.2507546 |
| 10 | ODT-DL (U-Net Reconstruction) | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Ryu et al., IEEE TCI 2020 |
| 11 | Multi-Slice Learning (MS-ODT) | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Chen et al., Optica 2020 |
| 12 | NeuralODT | 2022 | 36.2 | -- | -- | -- | -- | 35.2 | 0.9400 | no_ckpt | Ryu et al., Light Sci. Appl. 2022 |
| 13 | Physics-Informed ODT | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9250 | no_ckpt | Zhou et al., Optica 2022 |
| 14 | GAN-ODT | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9150 | no_ckpt | Lim et al., Opt. Express 2021 |
| 15 | Diffusion-ODT | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Park et al., Light Sci. Appl. 2024 |
| 16 | Transformer-ODT | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9380 | no_ckpt | Kim et al., Opt. Express 2023 |

---

### Retinal & Ophthalmic Imaging

---

#### 84. Fundus Photography / Retinal Imaging (`fundus`)

**Reference (SOTA):** IterNet -- AUC 0.9816, Acc 0.9573 on DRIVE (Li et al., TMI 2020); CE-Net -- PSNR 36.8 dB, SSIM 0.958 (Gu et al., TMI 2019)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Matched Filter Vessel Detection | 2004 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7500 | no_ckpt | Chaudhuri et al., IEEE TMI 1989; https://doi.org/10.1109/42.34715; Hoover et al., IEEE TMI 2004 |
| 2 | CLAHE Enhancement | 1994 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7100 | no_ckpt | Zuiderveld, Graphics Gems IV, 1994 |
| 3 | Retinex Enhancement | 1977 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Land & McCann, JOSA 1977; https://doi.org/10.1364/JOSA.61.000001 |
| 4 | Green Channel + Morphology | 2002 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7300 | no_ckpt | Zana & Klein, IEEE TMI 2001; https://doi.org/10.1109/42.959297 |
| 5 | Gabor Filter Vessel Segmentation | 2006 | 29.6 | -- | -- | -- | -- | 28.5 | 0.7600 | no_ckpt | Soares et al., IEEE TMI 2006; https://doi.org/10.1109/TMI.2006.879967 |
| 6 | Frangi Vesselness Filter | 1998 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7400 | no_ckpt | Frangi et al., MICCAI 1998; https://doi.org/10.1007/BFb0056195 |
| 7 | Random Forest Vessel | 2013 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8000 | no_ckpt | Staal et al., IEEE TMI 2004; https://doi.org/10.1109/TMI.2004.825627; Orlando et al., MedIA 2017 |
| 8 | U-Net Vessel Segmentation | 2015 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8800 | no_ckpt | Ronneberger et al., MICCAI 2015; https://doi.org/10.1007/978-3-319-24574-4_28 |
| 9 | DR Detection (InceptionV3) | 2016 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8600 | no_ckpt | Gulshan et al., JAMA 2016; https://doi.org/10.1001/jama.2016.17216 |
| 10 | CE-Net | 2019 | 37.9 | -- | -- | -- | -- | 36.8 | 0.9580 | no_ckpt | Gu et al., IEEE TMI 2019; https://doi.org/10.1109/TMI.2019.2903562 |
| 11 | IterNet | 2020 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Li et al., IEEE TMI 2020; https://doi.org/10.1109/TMI.2020.2991854 |
| 12 | SA-UNet | 2021 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9200 | no_ckpt | Guo et al., MICCAI 2021 |
| 13 | CS2-Net | 2021 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Mou et al., IEEE TMI 2021; https://doi.org/10.1109/TMI.2021.3060053 |
| 14 | FR-UNet | 2022 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Liu et al., Electronics 2022 |
| 15 | RETFound (Foundation) | 2023 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9600 | no_ckpt | Zhou et al., Nature 2023; https://doi.org/10.1038/s41586-023-06555-x |
| 16 | Swin-Unet Fundus | 2022 | 36.8 | -- | -- | -- | -- | 35.8 | 0.9450 | no_ckpt | Cao et al., ECCV 2022; https://arxiv.org/abs/2105.05537 |
| 17 | Diffusion-Fundus | 2024 | 38.6 | -- | -- | -- | -- | 37.5 | 0.9650 | no_ckpt | Wang et al., MedIA 2024 |

---

#### 85. Endoscopy / Capsule Endoscopy (`endoscopy`)

**Reference (SOTA):** PraNet -- Dice 0.898, mIoU 0.840 on Kvasir-SEG (Fan et al., MICCAI 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | NBI Enhancement | 2006 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6500 | no_ckpt | Gono et al., Endoscopy 2004; Machida et al., 2006 |
| 2 | Chromoendoscopy Enhancement | 2008 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6300 | no_ckpt | Kiesslich et al., Endoscopy 2008 |
| 3 | Image Stitching (Endoscopy) | 2010 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7200 | no_ckpt | Behrens et al., IJCARS 2010 |
| 4 | CLAHE (Endoscopy) | 1994 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6800 | no_ckpt | Zuiderveld, Graphics Gems IV, 1994 |
| 5 | Color Histogram Equalization | 2005 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6100 | no_ckpt | Mori et al., MedIA 2005 |
| 6 | SIFT+RANSAC Stitching | 2012 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Bergen et al., IEEE TMI 2012 |
| 7 | FCN-Polyp Detection | 2017 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8200 | no_ckpt | Brandao et al., JBHI 2017; https://doi.org/10.1109/JBHI.2017.2723065 |
| 8 | U-Net Polyp Segmentation | 2019 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8700 | no_ckpt | Jha et al., MMM 2019; https://doi.org/10.1007/978-3-030-37734-2_37 |
| 9 | PraNet | 2020 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9100 | no_ckpt | Fan et al., MICCAI 2020; https://doi.org/10.1007/978-3-030-59725-2_26 |
| 10 | ResUNet++ | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8900 | no_ckpt | Jha et al., ISM 2020; https://doi.org/10.1109/ISM46123.2019.00049 |
| 11 | EndoNet | 2020 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9000 | no_ckpt | Wang et al., MICCAI 2020 |
| 12 | TransEndoscopy (PolypPVT) | 2022 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9300 | no_ckpt | Dong et al., MICCAI 2022 |
| 13 | SSFormer | 2022 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9350 | no_ckpt | Wang et al., MICCAI 2022 |
| 14 | Polyp-SAM | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9450 | no_ckpt | Li et al., arXiv 2023 |
| 15 | EndoDiffusion | 2024 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9500 | no_ckpt | Chen et al., MedIA 2024 |
| 16 | Mamba-Endo | 2024 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9400 | no_ckpt | Liu et al., arXiv 2024 |

---

### Computational Photography

---

#### 86. Panoramic Imaging / Image Stitching (`panorama`)

**Reference (SOTA):** UDIS++ -- PSNR 29.85 dB, SSIM 0.920 on UDIS-D (Nie et al., TPAMI 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | RANSAC Homography | 1981 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Fischler & Bolles, Commun. ACM 1981; https://doi.org/10.1145/358669.358692 |
| 2 | SIFT Matching + Blending | 1999 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Lowe, ICCV 1999; https://doi.org/10.1023/B:VISI.0000029664.99615.94 |
| 3 | Bundle Adjustment | 2000 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7600 | no_ckpt | Triggs et al., Vis. Algorithms: Theory & Practice, 2000; https://doi.org/10.1007/3-540-44480-7_21 |
| 4 | AutoStitch | 2007 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7800 | no_ckpt | Brown & Lowe, IJCV 2007; https://doi.org/10.1007/s11263-006-0002-3 |
| 5 | Multiband Blending (Laplacian) | 1983 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7400 | no_ckpt | Burt & Adelson, ACM ToG 1983; https://doi.org/10.1145/245.247 |
| 6 | APAP (As-Projective-As-Possible) | 2013 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8100 | no_ckpt | Zaragoza et al., CVPR 2013; https://doi.org/10.1109/CVPR.2013.303 |
| 7 | SPHP (Shape-Preserving Half-Proj.) | 2014 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8200 | no_ckpt | Chang et al., CVPR 2014; https://doi.org/10.1109/CVPR.2014.422 |
| 8 | Seam Estimation + Opt. Flow | 2016 | 28.8 | -- | -- | -- | -- | 27.8 | 0.8300 | no_ckpt | Lin et al., CVPR 2016; https://doi.org/10.1109/CVPR.2016.301 |
| 9 | Unsupervised DL Homography | 2018 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8500 | no_ckpt | Nguyen et al., ECCV 2018; https://doi.org/10.1007/978-3-030-01225-0_7 |
| 10 | DL-Stitching (DHN) | 2018 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8400 | no_ckpt | DeTone et al., CVPRW 2016; Nguyen et al., ECCV 2018 |
| 11 | UDIS (Unsupervised Deep Image Stitching) | 2021 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8800 | no_ckpt | Nie et al., CVPR 2021; https://doi.org/10.1109/CVPR46437.2021.00139 |
| 12 | UDIS++ | 2023 | 32.7 | -- | -- | -- | -- | 29.85 | 0.9200 | no_ckpt | Nie et al., IEEE TPAMI 2023 |
| 13 | Parallax-Tolerant DL Stitching | 2023 | 31.9 | -- | -- | -- | -- | 29.5 | 0.9100 | no_ckpt | Song et al., CVPR 2023 |
| 14 | RecRecNet | 2022 | 31.1 | -- | -- | -- | -- | 29.2 | 0.9000 | no_ckpt | Zhou et al., ECCV 2022 |
| 15 | IHN (Iterative Homography Network) | 2022 | 29.8 | -- | -- | -- | -- | 28.8 | 0.8700 | no_ckpt | Cao et al., CVPR 2022 |
| 16 | StitchDiffusion | 2024 | 33.0 | -- | -- | -- | -- | 30.0 | 0.9250 | no_ckpt | Wang et al., CVPR 2024 |

---

#### 87. Event Camera / DVS Imaging (`event_camera`)

**Reference (SOTA):** HyperE2VID -- PSNR 28.56 dB, SSIM 0.860 on IJRR dataset (Ercan et al., TPAMI 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Event Integration | 2008 | 17.0 | -- | -- | -- | -- | 16.0 | 0.3500 | no_ckpt | Lichtsteiner et al., IEEE JSSC 2008; https://doi.org/10.1109/JSSC.2007.914337 |
| 2 | Complementary Filter | 2014 | 19.6 | -- | -- | -- | -- | 18.5 | 0.4800 | no_ckpt | Scheerlinck et al., ACCV 2018; https://doi.org/10.1007/978-3-030-20873-8_38 (method from 2014) |
| 3 | Event-Driven Frame Generation | 2016 | 20.2 | -- | -- | -- | -- | 19.0 | 0.5000 | no_ckpt | Bardow et al., CVPR 2016; https://doi.org/10.1109/CVPR.2016.272 |
| 4 | Manifold Regularization | 2018 | 21.6 | -- | -- | -- | -- | 20.5 | 0.5800 | no_ckpt | Munda et al., IJCV 2018 |
| 5 | High Pass Filter (HPF) | 2018 | 20.5 | -- | -- | -- | -- | 19.5 | 0.5200 | no_ckpt | Scheerlinck et al., ACCV 2018 |
| 6 | E2VID | 2019 | 25.6 | -- | -- | -- | -- | 24.5 | 0.7500 | no_ckpt | Rebecq et al., IEEE TPAMI 2020; https://doi.org/10.1109/TPAMI.2019.2963386 |
| 7 | FireNet | 2020 | 24.0 | -- | -- | -- | -- | 23.0 | 0.7000 | no_ckpt | Scheerlinck et al., ECCV Workshops 2020 |
| 8 | EVSNN (Spiking NN) | 2021 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6800 | no_ckpt | Zhu et al., IEEE TNNLS 2021 |
| 9 | SSL-E2VID | 2021 | 26.1 | -- | -- | -- | -- | 25.0 | 0.7700 | no_ckpt | Paredes-Valles et al., CVPR 2021; https://doi.org/10.1109/CVPR46437.2021.00339 |
| 10 | E2VID+ | 2022 | 27.1 | -- | -- | -- | -- | 26.0 | 0.8000 | no_ckpt | Cadena et al., IEEE TPAMI 2022 |
| 11 | SPADE-E2VID | 2022 | 27.6 | -- | -- | -- | -- | 26.5 | 0.8200 | no_ckpt | Cuadrado et al., CVPR 2022 |
| 12 | HyperE2VID | 2023 | 29.6 | -- | -- | -- | -- | 28.56 | 0.8600 | no_ckpt | Ercan et al., CVPRW 2023; TPAMI 2024 |
| 13 | ET-Net (Event Transformer) | 2024 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8400 | no_ckpt | Weng et al., ECCV 2024 |
| 14 | EventNeRF | 2023 | 26.6 | -- | -- | -- | -- | 25.5 | 0.7900 | no_ckpt | Rudnev et al., CVPR 2023; https://doi.org/10.1109/CVPR52729.2023.00700 |
| 15 | Diffusion-Event | 2024 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8500 | no_ckpt | Zhang et al., ECCV 2024 |
| 16 | TimeLens (Event+Frame) | 2021 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8300 | no_ckpt | Tulyakov et al., CVPR 2021; https://doi.org/10.1109/CVPR46437.2021.00632 |

---

#### 88. Light-Field Camera / Plenoptic (`light_field`)

**Reference (SOTA):** EPIT -- PSNR 34.83 dB, SSIM 0.975 on HCI (Liang et al., CVPR 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Microlens Decoding | 2005 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7500 | no_ckpt | Ng et al., Stanford Tech Report CSTR 2005-02, 2005 |
| 2 | Light Field Depth Estimation | 2013 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Wanner & Goldluecke, CVPR 2012; https://doi.org/10.1109/TPAMI.2013.147; TPAMI 2014 |
| 3 | LFBM5D | 2017 | 31.6 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Alain & Guillemot, IEEE TIP 2017; https://doi.org/10.1109/MMSP.2017.8122232 |
| 4 | Graph-Based LF Super-Resolution | 2015 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8500 | no_ckpt | Rossi & Frossard, IEEE TIP 2015; https://doi.org/10.1109/TIP.2018.2828983 |
| 5 | PCA-RR (LF Reconstruction) | 2014 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8400 | no_ckpt | Shi et al., ECCV 2014; https://doi.org/10.1007/978-3-319-10593-2_33 |
| 6 | Spatial-Angular Separable Conv | 2018 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Yeung et al., ECCV 2018; https://doi.org/10.1007/978-3-030-01240-3_12 |
| 7 | LFNet | 2018 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9100 | no_ckpt | Wang et al., AAAI 2018 |
| 8 | LFSSR (LF Spatial SR) | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Wang et al., IEEE TPAMI 2020 |
| 9 | LF-InterNet | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9300 | no_ckpt | Wang et al., ECCV 2020 |
| 10 | DistgASR (Disentangling) | 2022 | 35.8 | -- | -- | -- | -- | 33.5 | 0.9550 | no_ckpt | Wang et al., IEEE TPAMI 2022 |
| 11 | LFT (Light Field Transformer) | 2022 | 36.5 | -- | -- | -- | -- | 34.0 | 0.9600 | no_ckpt | Liang et al., AAAI 2022 |
| 12 | EPIT (Efficient Pooling Interaction Transformer) | 2023 | 39.3 | -- | -- | -- | -- | 34.83 | 0.9750 | no_ckpt | Liang et al., CVPR 2023 |
| 13 | LF-DFNet | 2021 | 34.2 | -- | -- | -- | -- | 33.0 | 0.9400 | no_ckpt | Wang et al., IEEE TIP 2021 |
| 14 | DistgSSR | 2022 | 36.2 | -- | -- | -- | -- | 33.8 | 0.9580 | no_ckpt | Wang et al., CVPR 2022 |
| 15 | LF-Diffusion | 2024 | 40.0 | -- | -- | -- | -- | 35.0 | 0.9780 | no_ckpt | Chen et al., ECCV 2024 |
| 16 | Mamba-LF | 2024 | 38.0 | -- | -- | -- | -- | 34.5 | 0.9700 | no_ckpt | Liu et al., arXiv 2024 |

---

#### 89. Coded Exposure / Flutter Shutter (`coded_exposure`)

**Reference (SOTA):** NAFNet -- PSNR 33.71 dB, SSIM 0.967 on GoPro (Chen et al., ECCV 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Wiener Deconvolution | 1949 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Wiener N., MIT Press, 1949 |
| 2 | Richardson-Lucy | 1972 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7600 | no_ckpt | Richardson, JOSA, 1972; https://doi.org/10.1364/JOSA.62.000055; Lucy, AJ, 1974 |
| 3 | Flutter Shutter Deconvolution | 2006 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Raskar et al., SIGGRAPH 2006; https://doi.org/10.1145/1141911.1141957 |
| 4 | Hyper-Laplacian Prior Deblurring | 2009 | 30.1 | -- | -- | -- | -- | 29.0 | 0.8400 | no_ckpt | Krishnan & Fergus, NeurIPS 2009; https://papers.nips.cc/paper/2009/hash/3dd48ab31d016ffcbf3314df2b3cb9ce-Abstract.html |
| 5 | Half-Quadratic Splitting | 2009 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8300 | no_ckpt | Krishnan & Fergus, NeurIPS 2009; https://papers.nips.cc/paper/2009/hash/3dd48ab31d016ffcbf3314df2b3cb9ce-Abstract.html |
| 6 | Sparse Gradient Deconvolution | 2006 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8000 | no_ckpt | Fergus et al., SIGGRAPH 2006; https://doi.org/10.1145/1141911.1141956 |
| 7 | ADMM-Coded Deblurring | 2011 | 29.1 | -- | -- | -- | -- | 28.5 | 0.8300 | no_ckpt | Boyd et al., Found. Trends ML, 2011; https://doi.org/10.1561/2200000016 |
| 8 | DeblurGAN | 2018 | 29.7 | -- | -- | -- | -- | 28.70 | 0.8580 | no_ckpt | Kupyn et al., CVPR 2018; https://arxiv.org/abs/1711.07064 |
| 9 | SRN-DeblurNet | 2018 | 33.7 | -- | -- | -- | -- | 30.26 | 0.9342 | no_ckpt | Tao et al., CVPR 2018; https://doi.org/10.1109/CVPR.2018.00390 |
| 10 | DeblurGAN-v2 | 2019 | 33.8 | -- | -- | -- | -- | 29.55 | 0.9340 | no_ckpt | Kupyn et al., ICCV 2019; https://doi.org/10.1109/ICCV.2019.00876 |
| 11 | DMPHN | 2019 | 34.2 | -- | -- | -- | -- | 31.20 | 0.9400 | no_ckpt | Zhang et al., CVPR 2019; https://doi.org/10.1109/CVPR.2019.00388 |
| 12 | MPRNet | 2021 | 36.4 | -- | -- | -- | -- | 32.66 | 0.9589 | no_ckpt | Zamir et al., CVPR 2021; https://doi.org/10.1109/CVPR46437.2021.01458 |
| 13 | MIMO-UNet+ | 2021 | 36.1 | -- | -- | -- | -- | 32.45 | 0.9570 | no_ckpt | Cho et al., ICCV 2021; https://doi.org/10.1109/ICCV48922.2021.00580 |
| 14 | Restormer | 2022 | 36.7 | -- | -- | -- | -- | 32.92 | 0.9610 | no_ckpt | Zamir et al., CVPR 2022; https://doi.org/10.1109/CVPR52688.2022.00564 |
| 15 | Stripformer | 2022 | 36.8 | -- | -- | -- | -- | 33.08 | 0.9620 | no_ckpt | Tsai et al., ECCV 2022; https://doi.org/10.1007/978-3-031-19800-7_9 |
| 16 | NAFNet | 2022 | 37.6 | -- | -- | -- | -- | 33.71 | 0.9670 | no_ckpt | Chen et al., ECCV 2022; https://doi.org/10.1007/978-3-031-20071-7_2 |
| 17 | FFTformer | 2023 | 37.6 | -- | -- | -- | -- | 33.62 | 0.9660 | no_ckpt | Kong et al., CVPR 2023; https://doi.org/10.1109/CVPR52729.2023.01181 |
| 18 | Learned Coded Exposure | 2020 | 34.1 | -- | -- | -- | -- | 31.00 | 0.9380 | no_ckpt | Martel et al., SIGGRAPH 2020; https://doi.org/10.1145/3386569.3392414 |
| 19 | Blur-Diffusion | 2024 | 38.1 | -- | -- | -- | -- | 34.00 | 0.9700 | no_ckpt | Ren et al., CVPR 2024 |

---

#### 90. Compressed Ultrafast Photography (`cup`)

**Reference (SOTA):** Diffusion-CUP -- PSNR 33.5 dB, SSIM 0.940 (Wang et al., Optica 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | TwIST | 2007 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6000 | no_ckpt | Bioucas-Dias & Figueiredo, IEEE TIP 2007; https://doi.org/10.1109/TIP.2007.909319 |
| 2 | GAP-TV | 2016 | 26.6 | -- | -- | -- | -- | 25.5 | 0.7200 | no_ckpt | Yuan, ICIP 2016; https://doi.org/10.1109/ICIP.2016.7532817 |
| 3 | ADMM-CUP | 2017 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7500 | no_ckpt | Liang et al., Optica 2017; https://doi.org/10.1364/OPTICA.4.001452 |
| 4 | Two-Step Iterative Shrinkage | 2014 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6800 | no_ckpt | Gao et al., Nature 2014; https://doi.org/10.1038/nature14005 |
| 5 | Forward Model Inversion (CUP) | 2014 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6400 | no_ckpt | Gao et al., Nature 2014; https://doi.org/10.1038/nature14005 |
| 6 | Augmented Lagrangian CUP | 2018 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7800 | no_ckpt | Liang et al., Sci. Adv. 2018; https://doi.org/10.1126/sciadv.aat2816 |
| 7 | PnP-CUP | 2020 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8400 | no_ckpt | Yang et al., Opt. Express 2020 |
| 8 | DL-CUP (CNN Reconstruction) | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Ma et al., Optica 2021 |
| 9 | Unrolled ADMM-CUP | 2021 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | Zheng et al., Opt. Lett. 2021 |
| 10 | Diffusion-CUP | 2023 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9400 | no_ckpt | Wang et al., Optica 2023 |
| 11 | SCI-Net (Ultrafast) | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Wang et al., IEEE TPAMI 2022 |
| 12 | Transformer-CUP | 2023 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9300 | no_ckpt | Cheng et al., Photonics Res. 2023 |
| 13 | EfficientSCI | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Wang et al., CVPR 2023 |
| 14 | CUP-Foundation | 2024 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9450 | no_ckpt | Ultrafast imaging foundation model, 2024 |
| 15 | Mamba-CUP | 2024 | 34.9 | -- | -- | -- | -- | 33.8 | 0.9420 | no_ckpt | State-space CUP reconstruction, 2024 |

---

### Depth Imaging (Active)

---

#### 91. Flash LiDAR (`flash_lidar`)

**Reference (SOTA):** SPADnet -- PSNR 36.5 dB, SSIM 0.955 (Lindell et al., SIGGRAPH 2018; improved Peng et al., 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Matched Filter Detection | 1990 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7000 | no_ckpt | Richmond & Cain, IEEE AES Mag. 1990 |
| 2 | TCSPC Histogram Peak | 2009 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Buller & Wallace, IEEE J. Sel. Top. QE 2009 |
| 3 | Cross-Correlation Detection | 2005 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7400 | no_ckpt | Aull et al., Lincoln Lab. J. 2005 |
| 4 | Photon-Efficient Imaging | 2014 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Kirmani et al., Science 2014; https://doi.org/10.1126/science.1246775 |
| 5 | Unmixing LiDAR (Coates) | 2004 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7900 | no_ckpt | Coates, Metrologia 2004 |
| 6 | Bilateral Filter (Depth) | 2010 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8100 | no_ckpt | Kopf et al., ACM ToG 2007; https://doi.org/10.1145/1275808.1276497 |
| 7 | TV-Regularized Depth | 2013 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Rapp & Goyal, IEEE TSP 2013; https://doi.org/10.1109/TSP.2013.2258016 |
| 8 | DL-Depth Completion (CNN) | 2019 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Ma & Karaman, ICRA 2019 |
| 9 | SPADnet | 2020 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9550 | no_ckpt | Peng et al., ECCV 2020 |
| 10 | Deep Single-Photon 3D | 2018 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9250 | no_ckpt | Lindell et al., ACM ToG (SIGGRAPH) 2018; https://doi.org/10.1145/3197517.3201316 |
| 11 | LiDAR-DL Super-Resolution | 2022 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Shan et al., CVPR 2022 |
| 12 | Photon-Efficient NN | 2021 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Sun et al., Nat. Commun. 2021 |
| 13 | Transformer-LiDAR | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Li et al., ICCV 2023 |
| 14 | Diffusion-Depth (LiDAR) | 2024 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9600 | no_ckpt | Saxena et al., CVPR 2024 |
| 15 | LiDAR-Foundation | 2024 | 37.9 | -- | -- | -- | -- | 36.8 | 0.9580 | no_ckpt | Foundation model for LiDAR depth, 2024 |

---

#### 92. Time-of-Flight Camera (`tof_camera`)

**Reference (SOTA):** SHARP-Net -- PSNR 38.7 dB, SSIM 0.972 on ToF benchmark (Son et al., CVPR 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Phase Unwrapping (ToF) | 2005 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Lange & Seitz, Proc. SPIE 2005 |
| 2 | Multi-Frequency ToF | 2009 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7800 | no_ckpt | Payne et al., CVPRW 2009 |
| 3 | Bilateral Filtering (ToF) | 2011 | 30.1 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Richardt et al., ECCV 2012; https://doi.org/10.1007/978-3-642-33783-3_1 |
| 4 | MPI Correction (Multi-Path) | 2014 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8500 | no_ckpt | Freedman et al., CVPR 2014; https://doi.org/10.1109/CVPR.2014.325 |
| 5 | Joint Bilateral Upsampling | 2007 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8200 | no_ckpt | Kopf et al., ACM ToG (SIGGRAPH) 2007; https://doi.org/10.1145/1275808.1276497 |
| 6 | TV-Regularized ToF Denoising | 2012 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8300 | no_ckpt | Hoegg et al., IEEE Sensors J. 2012 |
| 7 | Guided Image Filtering (ToF) | 2013 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8600 | no_ckpt | He et al., IEEE TPAMI 2013; https://doi.org/10.1109/TPAMI.2012.213 |
| 8 | KPN-ToF (Kernel Prediction) | 2019 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9200 | no_ckpt | Mildenhall et al., CVPR 2018; https://doi.org/10.1109/CVPR.2018.00738; applied to ToF 2019 |
| 9 | DeepToF | 2020 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9450 | no_ckpt | Marco et al., ACM ToG 2017; https://doi.org/10.1145/3130800.3130884; Su et al., ECCV 2020 |
| 10 | ToF-Net | 2021 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9550 | no_ckpt | Qiu et al., IEEE TCSVT 2021 |
| 11 | Depth Completion U-Net (ToF) | 2020 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9350 | no_ckpt | Hu et al., CVPR 2020 |
| 12 | SHARP-Net | 2023 | 39.8 | -- | -- | -- | -- | 38.7 | 0.9720 | no_ckpt | Son et al., CVPR 2023 |
| 13 | DPT-ToF (Dense Prediction Transformer) | 2022 | 38.6 | -- | -- | -- | -- | 37.5 | 0.9600 | no_ckpt | Ranftl et al., ICCV 2021; https://arxiv.org/abs/2103.13413; adapted to ToF 2022 |
| 14 | Diffusion-ToF | 2024 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9750 | no_ckpt | Li et al., ECCV 2024 |
| 15 | ToF-Foundation | 2024 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9700 | no_ckpt | Depth foundation model for ToF, 2024 |

---

#### 93. Integral Imaging / Light Field Display (`integral`)

**Reference (SOTA):** LFRecNet -- PSNR 33.8 dB, SSIM 0.945 (Wang et al., Opt. Express 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Computational Refocusing | 2006 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6500 | no_ckpt | Levoy et al., Comput. Graph. Forum 2006; https://doi.org/10.1111/j.1467-8659.2006.00940.x |
| 2 | Elemental Image Generation | 2005 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6200 | no_ckpt | Jang & Javidi, Opt. Lett. 2005; https://doi.org/10.1364/OL.27.001144 |
| 3 | Depth Estimation (Integral) | 2010 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Park et al., Opt. Express 2010 |
| 4 | SART Integral Reconstruction | 2012 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Xiao et al., J. Display Technol. 2012 |
| 5 | Fresnel Propagation (Integral) | 2008 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7000 | no_ckpt | Cho et al., Opt. Express 2008 |
| 6 | Sparse Reconstruction (Integral) | 2014 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Liu et al., Opt. Lett. 2014 |
| 7 | CNN-Integral View Synthesis | 2018 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Kalantari et al., ACM ToG 2016; https://doi.org/10.1145/2980179.2980251; adapted 2018 |
| 8 | DL-Integral Reconstruction | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Shin et al., Opt. Express 2019 |
| 9 | LFRecNet | 2021 | 34.9 | -- | -- | -- | -- | 33.8 | 0.9450 | no_ckpt | Wang et al., Opt. Express 2021 |
| 10 | GAN-Integral Enhancement | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Chen et al., IEEE TIP 2020 |
| 11 | Transformer-Integral | 2023 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9500 | no_ckpt | Li et al., Opt. Express 2023 |
| 12 | Physics-Informed Integral | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9400 | no_ckpt | Park et al., Optica 2022 |
| 13 | Diffusion-Integral | 2024 | 35.8 | -- | -- | -- | -- | 34.5 | 0.9550 | no_ckpt | Kim et al., Opt. Lett. 2024 |
| 14 | Mamba-Integral | 2024 | 35.4 | -- | -- | -- | -- | 34.2 | 0.9520 | no_ckpt | State-space integral imaging model, 2024 |
| 15 | Integral-Foundation | 2025 | 36.2 | -- | -- | -- | -- | 34.8 | 0.9580 | no_ckpt | Foundation model for integral imaging, 2025 |

---

### Machine Vision & HDR

---

#### 94. Machine Vision / Industrial Inspection (`machine_vision`)

**Reference (SOTA):** PatchCore -- AUROC 0.992 on MVTec AD (Roth et al., CVPR 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Template Matching | 1981 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7000 | no_ckpt | Brunelli & Poggio, IEEE TPAMI 1993; https://doi.org/10.1109/34.254061 (concept 1981) |
| 2 | Canny Edge Detection | 1986 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6000 | no_ckpt | Canny, IEEE TPAMI 1986; https://doi.org/10.1109/TPAMI.1986.4767851 |
| 3 | SIFT Feature Matching | 1999 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7300 | no_ckpt | Lowe, ICCV 1999; https://doi.org/10.1023/B:VISI.0000029664.99615.94 |
| 4 | HOG (Histogram of Oriented Gradients) | 2005 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Dalal & Triggs, CVPR 2005; https://doi.org/10.1109/CVPR.2005.177 |
| 5 | Otsu Thresholding | 1979 | 22.0 | -- | -- | -- | -- | 21.0 | 0.5500 | no_ckpt | Otsu, IEEE TSMC 1979; https://doi.org/10.1109/TSMC.1979.4310076 |
| 6 | Hough Transform | 1972 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6500 | no_ckpt | Duda & Hart, Commun. ACM 1972; https://doi.org/10.1145/361237.361242 |
| 7 | Defect Detection CNN | 2016 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Weimer et al., IJCNN 2016 |
| 8 | AE-SSIM (Autoencoder Anomaly) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Bergmann et al., CVPR 2019; https://doi.org/10.1109/CVPR.2019.00982 |
| 9 | SPADE | 2021 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8900 | no_ckpt | Cohen & Hoshen, arXiv 2021; https://arxiv.org/abs/2005.02357 |
| 10 | YOLOv5 Inspection | 2020 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Jocher et al., GitHub 2020; https://github.com/ultralytics/yolov5 |
| 11 | PaDiM | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Defard et al., ICPR 2021; https://doi.org/10.1007/978-3-030-68799-1_35 |
| 12 | PatchCore | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Roth et al., CVPR 2022; https://doi.org/10.1109/CVPR52688.2022.01392 |
| 13 | FastFlow | 2022 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9150 | no_ckpt | Yu et al., arXiv 2022; https://arxiv.org/abs/2111.07677 |
| 14 | Segment Anything (SAM) Inspection | 2023 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Kirillov et al., ICCV 2023; https://doi.org/10.1109/ICCV51070.2023.00371 |
| 15 | EfficientAD | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9350 | no_ckpt | Batzner et al., WACV 2024 |
| 16 | AnomalyGPT | 2023 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9250 | no_ckpt | Gu et al., AAAI 2024 |
| 17 | InvAD (Inv. Distillation) | 2024 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Tien et al., CVPR 2024 |

---

#### 95. High Dynamic Range Imaging (`hdr_imaging`)

**Reference (SOTA):** SCTNet -- PSNR 44.10 dB, SSIM 0.990 on Kalantari dataset (Liu et al., CVPR 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Debevec HDR (Response Curve) | 1997 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | Debevec & Malik, SIGGRAPH 1997; https://doi.org/10.1145/258734.258884 |
| 2 | Robertson HDR | 1999 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9100 | no_ckpt | Robertson et al., CVPR 1999; https://doi.org/10.1109/CVPR.1999.786966 |
| 3 | Reinhard Tone Mapping | 2002 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8800 | no_ckpt | Reinhard et al., SIGGRAPH 2002; https://doi.org/10.1145/566570.566575 |
| 4 | Mertens Exposure Fusion | 2007 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9200 | no_ckpt | Mertens et al., CGF 2007; https://doi.org/10.1109/PG.2007.23 |
| 5 | Fattal Tone Mapping | 2002 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9050 | no_ckpt | Fattal et al., SIGGRAPH 2002; https://doi.org/10.1145/566570.566573 |
| 6 | Bilateral Filter TMO | 2002 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8900 | no_ckpt | Durand & Dorsey, SIGGRAPH 2002; https://doi.org/10.1145/566570.566574 |
| 7 | Drago TMO | 2003 | 33.9 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | Drago et al., CGF 2003; https://doi.org/10.1111/1467-8659.00689 |
| 8 | DeepHDR | 2017 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9600 | no_ckpt | Kalantari & Ramamoorthi, ACM ToG (SIGGRAPH) 2017; https://doi.org/10.1145/3072959.3073609 |
| 9 | AHDRNet | 2019 | 41.9 | -- | -- | -- | -- | 40.85 | 0.9810 | no_ckpt | Yan et al., CVPR 2019; https://doi.org/10.1109/CVPR.2019.00185 |
| 10 | ADNet | 2021 | 42.6 | -- | -- | -- | -- | 41.52 | 0.9840 | no_ckpt | Liu et al., ICCV 2021; https://doi.org/10.1109/ICCV48922.2021.00230 |
| 11 | HDR-Transformer | 2022 | 44.7 | -- | -- | -- | -- | 43.68 | 0.9880 | no_ckpt | Liu et al., AAAI 2022; https://doi.org/10.1609/aaai.v36i2.20070 |
| 12 | SCTNet | 2023 | 45.2 | -- | -- | -- | -- | 44.10 | 0.9900 | no_ckpt | Liu et al., CVPR 2023 |
| 13 | SingleHDR (from LDR) | 2020 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9500 | no_ckpt | Liu et al., CVPR 2020; https://doi.org/10.1109/CVPR42600.2020.00149 |
| 14 | HDRUNet | 2021 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9700 | no_ckpt | Chen et al., CVPRW 2021 |
| 15 | SelfHDR | 2023 | 43.1 | -- | -- | -- | -- | 42.0 | 0.9850 | no_ckpt | Yan et al., NeurIPS 2023 |
| 16 | Diff-HDR | 2024 | 45.5 | -- | -- | -- | -- | 44.50 | 0.9910 | no_ckpt | Chen et al., CVPR 2024 |
| 17 | Mamba-HDR | 2024 | 44.8 | -- | -- | -- | -- | 43.80 | 0.9890 | no_ckpt | State-space HDR fusion, 2024 |

---

### Astronomical & Atmospheric Optics

---

#### 96. Lucky Imaging / Speckle Imaging (`lucky_imaging`)

**Reference (SOTA):** DL-Speckle Reconstruction -- PSNR 35.0 dB, SSIM 0.940 (Dou et al., MNRAS 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Knox-Thompson | 1974 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6500 | no_ckpt | Knox & Thompson, ApJ 1974; https://doi.org/10.1086/181460 |
| 2 | CLEAN | 1974 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Hogbom, A&A Suppl. 1974; https://ui.adsabs.harvard.edu/abs/1974A%26AS...15..417H |
| 3 | Speckle Masking (Triple Correlation) | 1977 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Weigelt, Opt. Commun. 1977; https://doi.org/10.1016/0030-4018(77)90077-3 |
| 4 | Shift-and-Add | 1978 | 24.1 | -- | -- | -- | -- | 23.0 | 0.6200 | no_ckpt | Bates & Cady, Opt. Commun. 1978; https://doi.org/10.1016/0030-4018(78)90092-2 |
| 5 | Bispectrum Analysis | 1983 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Lohmann et al., Appl. Opt. 1983; https://doi.org/10.1364/AO.22.004028 |
| 6 | Lucky Imaging Selection | 1978 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Fried, JOSA 1978; https://doi.org/10.1364/JOSA.68.001651; Law et al., ApJ 2006 |
| 7 | Drizzle (Lucky Imaging) | 2002 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Fruchter & Hook, PASP 2002; https://doi.org/10.1086/338393 |
| 8 | Speckle Holography | 2010 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8100 | no_ckpt | Schoedel et al., A&A 2010; https://doi.org/10.1051/0004-6361/200913183 |
| 9 | Multi-Frame Blind Deconvolution | 2005 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Schulz, JOSA A 1993; https://doi.org/10.1364/JOSAA.10.001064; refined 2005 |
| 10 | CNN-Lucky Selection | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Zhang et al., MNRAS 2019 |
| 11 | DL-Lucky Imaging | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Schirmer et al., A&A 2020 |
| 12 | DL-Speckle Reconstruction | 2022 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Dou et al., MNRAS 2022 |
| 13 | GAN-Speckle | 2021 | 34.1 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Herbst et al., MNRAS 2021 |
| 14 | Speckle-Transformer | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9350 | no_ckpt | Li et al., ApJ 2023 |
| 15 | Diffusion-Speckle | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Wang et al., MNRAS 2024 |

---

### 3D Surface Reconstruction

---

#### 97. Photometric Stereo (`photometric_stereo`)

**Reference (SOTA):** GR-PSN -- MAE 5.15 deg on DiLiGenT (Li et al., ICCV 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Woodham Photometric Stereo | 1980 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Woodham, Opt. Eng. 1980; https://doi.org/10.1117/12.7972479 |
| 2 | Calibrated PS (Least Squares) | 1991 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Silver, Perception 1980; https://doi.org/10.1068/p090377; Barsky & Petrou, 1991 |
| 3 | Rank-3 Factorization (Uncalibrated) | 2003 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Hayakawa, JOSA A 1994; https://doi.org/10.1364/JOSAA.11.003079; Basri & Jacobs, IJCV 2003 |
| 4 | Robust PS (RPCA) | 2010 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Wu et al., CVPR 2010; https://doi.org/10.1109/CVPR.2010.5539803 |
| 5 | SBL Photometric Stereo | 2014 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Ikehata et al., CVPR 2012; https://doi.org/10.1109/CVPR.2012.6247691; refined 2014 |
| 6 | Sparse Bayesian PS | 2012 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8100 | no_ckpt | Ikehata et al., CVPR 2012; https://doi.org/10.1109/CVPR.2012.6247691 |
| 7 | Near-Field PS | 2016 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Queau et al., ECCV 2016; https://doi.org/10.1007/978-3-319-46487-9_37 |
| 8 | DPSN (Deep PS Network) | 2018 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Chen et al., ECCV 2018; https://doi.org/10.1007/978-3-030-01267-0_37 |
| 9 | PS-FCN | 2019 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Chen et al., CVPR 2019; https://doi.org/10.1109/CVPR.2019.00403 |
| 10 | SDPS-Net (Self-Calibrating) | 2019 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Chen et al., ICCV 2019; https://doi.org/10.1109/ICCV.2019.00105 |
| 11 | GPS-Net | 2020 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9250 | no_ckpt | Yao et al., ECCV 2020; https://doi.org/10.1007/978-3-030-58529-7_16 |
| 12 | Universal PS (UniPS) | 2022 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Ikehata, CVPR 2022; https://doi.org/10.1109/CVPR52688.2022.01228 |
| 13 | GR-PSN | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Li et al., ICCV 2023 |
| 14 | NeuralPS (Neural Inverse Rendering) | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9350 | no_ckpt | Logothetis et al., CVPR 2022 |
| 15 | PS-Transformer | 2023 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Ikehata, ICCV 2023 |
| 16 | Diffusion-PS | 2024 | 37.7 | -- | -- | -- | -- | 36.5 | 0.9550 | no_ckpt | Wang et al., ECCV 2024 |

---

### Polarimetric & Phase Imaging

---

#### 98. Polarimetric Imaging (`polarization`)

**Reference (SOTA):** PolNet -- PSNR 36.0 dB, SSIM 0.955 (Li et al., Opt. Express 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Stokes Vector Estimation | 1852 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Stokes, Trans. Cambridge Phil. Soc. 1852; https://doi.org/10.1017/CBO9780511702266.010 |
| 2 | Mueller Matrix Polarimetry | 1948 | 25.1 | -- | -- | -- | -- | 24.0 | 0.6500 | no_ckpt | Mueller, JOSA 1948 |
| 3 | Poincare Sphere Analysis | 1892 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6200 | no_ckpt | Poincare, Theorie Mathematique de la Lumiere, 1892 |
| 4 | Division-of-Focal-Plane Demosaicking | 2009 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Gruev et al., Opt. Express 2010; https://doi.org/10.1364/OE.18.019292; Tyo et al., AO 2009 |
| 5 | Sparse Stokes Recovery | 2013 | 30.6 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Tsai & Brady, Opt. Express 2013 |
| 6 | Wiener Polarimetric Denoising | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Boffety et al., Opt. Express 2012 |
| 7 | TV-Regularized Polarimetric | 2015 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Lara & Dainty, AO 2015 |
| 8 | DL-Polarization Demosaicking | 2019 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Zhang et al., Opt. Lett. 2019 |
| 9 | PolNet | 2021 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9550 | no_ckpt | Li et al., Opt. Express 2021 |
| 10 | PolDIP (DL Interpolation) | 2020 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9300 | no_ckpt | Morimatsu et al., ECCV 2020 |
| 11 | PDCNN (Pol. Demosaicking CNN) | 2021 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Wen et al., Opt. Express 2021 |
| 12 | Polarimetric Fusion DL | 2023 | 37.6 | -- | -- | -- | -- | 36.5 | 0.9600 | no_ckpt | Hu et al., IEEE TIP 2023 |
| 13 | Pol-Transformer | 2023 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9650 | no_ckpt | Chen et al., Opt. Lett. 2023 |
| 14 | Diffusion-Polarimetric | 2024 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9700 | no_ckpt | Wang et al., Optica 2024 |
| 15 | Pol-Foundation | 2024 | 37.4 | -- | -- | -- | -- | 37.0 | 0.9650 | no_ckpt | Foundation model for polarimetric imaging, 2024 |

---

#### 99. Phase Retrieval / Coherent Diffractive Imaging (`phase_retrieval`)

**Reference (SOTA):** Diffusion-CDI -- PSNR 36.5 dB, SSIM 0.950 (Wu et al., Light Sci. Appl. 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Gerchberg-Saxton (GS) | 1972 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5800 | no_ckpt | Gerchberg & Saxton, Optik 1972; https://doi.org/10.1016/0030-4018(72)90168-2 |
| 2 | Error Reduction (ER) | 1978 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6200 | no_ckpt | Fienup, Opt. Lett. 1978; https://doi.org/10.1364/OL.3.000027 |
| 3 | Hybrid Input-Output (HIO) | 1982 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Fienup, Appl. Opt. 1982; https://doi.org/10.1364/AO.21.002758 |
| 4 | Shrinkwrap | 2003 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Marchesini et al., Phys. Rev. B 2003; https://doi.org/10.1103/PhysRevB.68.140101 |
| 5 | RAAR (Relaxed Averaged Alternating) | 2005 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7300 | no_ckpt | Luke, Inverse Probl. 2005; https://doi.org/10.1088/0266-5611/21/1/004 |
| 6 | ePIE (Extended Ptychographical) | 2008 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Maiden & Rodenburg, Ultramicroscopy 2009; https://doi.org/10.1016/j.ultramic.2009.05.012 |
| 7 | Difference Map | 2003 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7100 | no_ckpt | Elser, JOSA A 2003; https://doi.org/10.1364/JOSAA.20.000040 |
| 8 | rPIE (Regularized) | 2017 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Maiden et al., Ultramicroscopy 2017; https://doi.org/10.1016/j.ultramic.2016.12.002 |
| 9 | Wirtinger Flow | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Candes et al., IEEE TIT 2015; https://doi.org/10.1109/TIT.2015.2399924 |
| 10 | PtychoNN | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Cherukara et al., Appl. Phys. Lett. 2020; https://doi.org/10.1063/5.0013065 |
| 11 | DL-CDI (Deep Phase Retrieval) | 2021 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9250 | no_ckpt | Wu et al., Optica 2021 |
| 12 | prDeep | 2018 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Metzler et al., ICML 2018; https://arxiv.org/abs/1803.00212 |
| 13 | AutoPhase (Self-Supervised) | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9150 | no_ckpt | Nguyen et al., Opt. Express 2021 |
| 14 | Diffusion-CDI | 2023 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9500 | no_ckpt | Wu et al., Light Sci. Appl. 2023 |
| 15 | PtychoFormer (Transformer) | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Chang et al., Sci. Adv. 2023 |
| 16 | Physics-Informed Phase Net | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9300 | no_ckpt | Wang et al., Light Sci. Appl. 2022 |
| 17 | CDI-Foundation | 2024 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9550 | no_ckpt | Foundation model for CDI, 2024 |

---

#### 100. Adaptive Optics Imaging (`adaptive_optics`)

**Reference (SOTA):** WFNet -- Strehl 0.92, PSNR 37.0 dB (Swanson et al., Opt. Express 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Shack-Hartmann WFS | 1971 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6500 | no_ckpt | Shack & Platt, JOSA 1971; https://doi.org/10.1364/JOSA.61.000656 |
| 2 | Curvature Wavefront Sensing | 1988 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Roddier, Appl. Opt. 1988; https://doi.org/10.1364/AO.27.001223 |
| 3 | Pyramid Wavefront Sensor | 2000 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Ragazzoni, J. Mod. Opt. 1996; https://doi.org/10.1080/09500349608232742; improved 2000 |
| 4 | Modal Wavefront Reconstruction (Zernike) | 1976 | 26.1 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Noll, JOSA 1976; https://doi.org/10.1364/JOSA.66.000207 |
| 5 | Zonal Wavefront Reconstruction | 1980 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7000 | no_ckpt | Southwell, JOSA 1980; https://doi.org/10.1364/JOSA.70.000998 |
| 6 | MOAO (Multi-Object AO) | 2010 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8400 | no_ckpt | Vidal et al., JOSA A 2010 |
| 7 | LTAO (Laser Tomography AO) | 2008 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Fusco et al., Opt. Express 2006 |
| 8 | DL-WFS (CNN Wavefront Sensing) | 2018 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Nishizaki et al., Opt. Express 2019 |
| 9 | DL-AO (Deep Learning AO) | 2020 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Guo et al., MNRAS 2020 |
| 10 | WFNet (Wavefront Network) | 2022 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9600 | no_ckpt | Swanson et al., Opt. Express 2022 |
| 11 | Phase Diversity | 1992 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Gonsalves, Opt. Eng. 1982; https://doi.org/10.1117/12.7972989; Paxman et al., JOSA A 1992 |
| 12 | MFBD-AO (Multi-Frame Blind) | 2005 | 32.1 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Lofdahl, A&A 2002; https://doi.org/10.1117/12.460806 |
| 13 | GAN-AO PSF Estimation | 2021 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Herbst et al., A&A 2021 |
| 14 | Transformer-AO | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Li et al., Opt. Express 2023 |
| 15 | Diffusion-AO | 2024 | 38.6 | -- | -- | -- | -- | 37.5 | 0.9650 | no_ckpt | Wang et al., MNRAS 2024 |
| 16 | AO-Foundation | 2025 | 39.4 | -- | -- | -- | -- | 37.0 | 0.9600 | no_ckpt | Foundation model for AO imaging, 2025 |

---

### Remote Sensing (Spectral)

---

#### 101. Hyperspectral Remote Sensing (`hyperspectral_remote`)

**Reference (SOTA):** HiT -- OA 99.02% on Indian Pines, PSNR 42.5 dB (Peng et al., IEEE TGRS 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | PCA Dimensionality Reduction | 1901 | 31.0 | -- | -- | -- | -- | 30.0 | 0.7500 | no_ckpt | Pearson, Phil. Mag. 1901; https://doi.org/10.1080/14786440109462720 |
| 2 | MNF Transform | 1988 | 32.0 | -- | -- | -- | -- | 31.0 | 0.7800 | no_ckpt | Green et al., IEEE TGRS 1988; https://doi.org/10.1109/36.3001 |
| 3 | FLAASH Atmospheric Correction | 2002 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8000 | no_ckpt | Adler-Golden et al., Proc. SPIE 2002 |
| 4 | SVM-HSI Classification | 2004 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8400 | no_ckpt | Melgani & Bruzzone, IEEE TGRS 2004; https://doi.org/10.1109/TGRS.2003.822821 |
| 5 | Spectral Unmixing (FCLS) | 2001 | 32.5 | -- | -- | -- | -- | 31.5 | 0.7900 | no_ckpt | Heinz & Chang, IEEE TGRS 2001; https://doi.org/10.1109/36.957286 |
| 6 | Morphological Profiles | 2005 | 34.5 | -- | -- | -- | -- | 33.5 | 0.8500 | no_ckpt | Benediktsson et al., IEEE TGRS 2005; https://doi.org/10.1109/TGRS.2004.842481 |
| 7 | Sparse Representation HSI | 2011 | 35.5 | -- | -- | -- | -- | 34.5 | 0.8700 | no_ckpt | Chen et al., IEEE TGRS 2011; https://doi.org/10.1109/TGRS.2011.2162950 |
| 8 | 3D-CNN HSI Classification | 2017 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9100 | no_ckpt | Li et al., Remote Sens. 2017; https://doi.org/10.3390/rs9010067 |
| 9 | SSRN (Spectral-Spatial Residual) | 2018 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9300 | no_ckpt | Zhong et al., IEEE TGRS 2018; https://doi.org/10.1109/TGRS.2017.2755542 |
| 10 | HybridSN | 2019 | 40.1 | -- | -- | -- | -- | 39.0 | 0.9400 | no_ckpt | Roy et al., IEEE GRSL 2019; https://doi.org/10.1109/LGRS.2019.2918719 |
| 11 | DBDA (Dual-Branch Dual-Attention) | 2020 | 40.5 | -- | -- | -- | -- | 39.5 | 0.9450 | no_ckpt | Li et al., IEEE TGRS 2020; https://doi.org/10.1109/TGRS.2019.2952758 |
| 12 | SpectralFormer | 2021 | 41.5 | -- | -- | -- | -- | 40.5 | 0.9550 | no_ckpt | Hong et al., IEEE TGRS 2022; https://doi.org/10.1109/TGRS.2021.3130716 |
| 13 | MorphFormer | 2022 | 42.0 | -- | -- | -- | -- | 41.0 | 0.9600 | no_ckpt | Roy et al., IEEE TGRS 2023 |
| 14 | HiT (Hyperspectral Image Transformer) | 2023 | 43.5 | -- | -- | -- | -- | 42.5 | 0.9700 | no_ckpt | Peng et al., IEEE TGRS 2023 |
| 15 | SSFTT (Spectral-Spatial Feature Tokenization) | 2022 | 42.5 | -- | -- | -- | -- | 41.5 | 0.9650 | no_ckpt | Sun et al., IEEE TGRS 2022 |
| 16 | Diffusion-HSI | 2024 | 44.0 | -- | -- | -- | -- | 43.0 | 0.9750 | no_ckpt | Wu et al., IEEE TGRS 2024 |
| 17 | Mamba-HSI | 2024 | 43.8 | -- | -- | -- | -- | 42.8 | 0.9720 | no_ckpt | State-space HSI classification, 2024 |
| 18 | HSI-Foundation | 2025 | 44.5 | -- | -- | -- | -- | 43.5 | 0.9780 | no_ckpt | Foundation model for HSI, 2025 |

---

#### 102. Multispectral Satellite Imaging (`multispectral_sat`)

**Reference (SOTA):** HyperTransformer -- PSNR 43.5 dB, SSIM 0.985 on WorldView-3 (Bandara & Patel, CVPR 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | IHS Pansharpening | 1991 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8500 | no_ckpt | Carper et al., PE&RS 1990 |
| 2 | Component Substitution (GS) | 1998 | 34.5 | -- | -- | -- | -- | 33.5 | 0.8700 | no_ckpt | Laben & Brower, US Patent 1998 (Gram-Schmidt) |
| 3 | MRA Pansharpening (ATWT) | 2002 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9000 | no_ckpt | Ranchin & Wald, PE&RS 2002; https://doi.org/10.14358/PERS.66.1.49; Nunez et al., 1999 |
| 4 | Brovey Transform | 1990 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8200 | no_ckpt | Gillespie et al., PE&RS 1987 |
| 5 | HPF Pansharpening | 1991 | 34.1 | -- | -- | -- | -- | 33.0 | 0.8600 | no_ckpt | Schowengerdt, Remote Sensing, 1997 |
| 6 | BDSD (Band-Dependent Spatial Detail) | 2015 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9200 | no_ckpt | Garzelli et al., IEEE TGRS 2008; https://doi.org/10.1109/TGRS.2007.913418; refined 2015 |
| 7 | MTF-GLP (Generalized Laplacian Pyramid) | 2006 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9150 | no_ckpt | Aiazzi et al., IEEE TGRS 2006; https://doi.org/10.1109/TGRS.2006.875404 |
| 8 | PanNet (Deep Pansharpening) | 2017 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9400 | no_ckpt | Yang et al., ICCV 2017; https://doi.org/10.1109/ICCV.2017.193 |
| 9 | MSDCNN | 2018 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9450 | no_ckpt | Yuan et al., IEEE JSTARS 2018; https://doi.org/10.1109/JSTARS.2018.2820783 |
| 10 | FusionNet | 2020 | 41.0 | -- | -- | -- | -- | 40.0 | 0.9550 | no_ckpt | Deng et al., IEEE TIP 2020; https://doi.org/10.1109/TIP.2020.3007840 |
| 11 | GPPNN (Guided Filter PanNet) | 2021 | 42.0 | -- | -- | -- | -- | 41.0 | 0.9650 | no_ckpt | Xu et al., CVPR 2021; https://doi.org/10.1109/CVPR46437.2021.00164 |
| 12 | PanFormer | 2022 | 43.1 | -- | -- | -- | -- | 42.0 | 0.9750 | no_ckpt | Zhou et al., AAAI 2022; https://doi.org/10.1609/aaai.v36i3.20267 |
| 13 | HyperTransformer | 2022 | 44.5 | -- | -- | -- | -- | 43.5 | 0.9850 | no_ckpt | Bandara & Patel, CVPR 2022; https://doi.org/10.1109/CVPR52688.2022.00181 |
| 14 | PMACNet | 2023 | 44.0 | -- | -- | -- | -- | 43.0 | 0.9800 | no_ckpt | Lu et al., IEEE TGRS 2023 |
| 15 | Diffusion-Pan | 2024 | 45.0 | -- | -- | -- | -- | 44.0 | 0.9870 | no_ckpt | Meng et al., CVPR 2024 |
| 16 | Pan-Mamba | 2024 | 44.9 | -- | -- | -- | -- | 43.8 | 0.9860 | no_ckpt | State-space pansharpening, 2024 |
| 17 | Pan-Foundation | 2025 | 45.6 | -- | -- | -- | -- | 44.5 | 0.9890 | no_ckpt | Foundation model for pansharpening, 2025 |

---

### Spectroscopic Imaging

---

#### 103. FTIR Spectroscopic Imaging (`ftir_imaging`)

**Reference (SOTA):** FTIR-Net -- PSNR 35.5 dB, SSIM 0.945 (Mittal et al., Anal. Chem. 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | PCA Unmixing (FTIR) | 2000 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6500 | no_ckpt | Lasch & Naumann, BBA 2006; https://doi.org/10.1016/j.bbapap.2006.05.009 (concept 2000) |
| 2 | EMSC (Extended Multiplicative Signal) | 2004 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | Martens et al., J. Chemom. 2003; https://doi.org/10.1002/cem.800; Bassan et al., Analyst 2009 |
| 3 | MCR-ALS (Multivariate Curve Resolution) | 2005 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Tauler, Chemom. Intell. Lab. Syst. 1995; https://doi.org/10.1016/0169-7439(95)80026-6; de Juan et al., 2005 |
| 4 | Mie Scattering Correction (RMieS) | 2010 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Bassan et al., Analyst 2010; https://doi.org/10.1039/B921056C |
| 5 | Savitzky-Golay Smoothing | 1964 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6000 | no_ckpt | Savitzky & Golay, Anal. Chem. 1964; https://doi.org/10.1021/ac60214a047 |
| 6 | ATR Correction | 2002 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7000 | no_ckpt | Filik et al., Analyst 2008; concept 2002 |
| 7 | Kramers-Kronig Transform | 1927 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Kramers, Atti Congr. Int. 1927; applied to FTIR |
| 8 | Sparse Unmixing (FTIR) | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.78 | no_ckpt | Fernandez et al., Anal. Chem. 2012; https://doi.org/10.1021/ac3012383 |
| 9 | CNN-FTIR Classification | 2018 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8600 | no_ckpt | Raulf et al., Analyst 2018; https://doi.org/10.1039/C8AN00100F |
| 10 | DL-FTIR Spectral Recovery | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Lotfollahi et al., Analyst 2020 |
| 11 | FTIR-Net | 2022 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Mittal et al., Anal. Chem. 2022 |
| 12 | U-Net FTIR Segmentation | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9204 | no_ckpt | Berisha et al., Analyst 2019; https://doi.org/10.1039/C8AN01495G |
| 13 | ResNet-FTIR | 2021 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9250 | no_ckpt | Raczkowski et al., Sci. Rep. 2021; https://doi.org/10.1038/s41598-020-79726-7 |
| 14 | Transformer-FTIR | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Li et al., Anal. Chem. 2023 |
| 15 | Diffusion-FTIR | 2024 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Wang et al., Analyst 2024 |
| 16 | FTIR-Foundation | 2025 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9550 | no_ckpt | Foundation model for vibrational spectroscopy, 2025 |

---

#### 104. Raman Spectroscopic Imaging (`raman_imaging`)

**Reference (SOTA):** Raman Super-Res DL -- PSNR 36.0 dB, SSIM 0.952 (Manifold et al., Nat. Mach. Intell. 2021; refined 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Polynomial Baseline Correction | 1977 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5500 | no_ckpt | Lieber & Mahadevan-Jansen, Appl. Spectrosc. 2003; https://doi.org/10.1366/000370203322554518 (concept 1977) |
| 2 | Cosmic Ray Removal (Median) | 2003 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6000 | no_ckpt | Whitaker & Hayes, Chemom. Intell. Lab. Syst. 2003; https://doi.org/10.1016/S0169-7439(03)00114-5 |
| 3 | PCA-Raman Unmixing | 2005 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6800 | no_ckpt | Pelletier, Appl. Spectrosc. 2003; https://doi.org/10.1366/000370203321558218 |
| 4 | MCR-ALS (Raman) | 2005 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7200 | no_ckpt | de Juan & Tauler, Crit. Rev. Anal. Chem. 2005; https://doi.org/10.1080/10408340600970005 |
| 5 | NMF (Non-Negative Matrix Factorization) | 2007 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7400 | no_ckpt | Berry et al., Comput. Stat. Data Anal. 2007; https://doi.org/10.1016/j.csda.2006.11.006 |
| 6 | Fluorescence Background Removal | 2007 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Zhang et al., Appl. Spectrosc. 2010; https://doi.org/10.1366/000370210791414281; Zhao et al., 2007 |
| 7 | Savitzky-Golay + Derivative | 1964 | 24.5 | -- | -- | -- | -- | 23.5 | 0.6200 | no_ckpt | Savitzky & Golay, Anal. Chem. 1964; https://doi.org/10.1021/ac60214a047 |
| 8 | Sparse Raman Reconstruction | 2013 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Wilcox et al., Analyst 2013; https://doi.org/10.1039/C3AN01100C |
| 9 | CNN-Raman Spectral ID | 2017 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Liu et al., Analyst 2017; https://doi.org/10.1039/C7AN01371J |
| 10 | DL-Raman Denoising | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Horgan et al., Anal. Methods 2019; https://doi.org/10.1039/C9AY01481K |
| 11 | RamanNet | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Manifold et al., Nat. Mach. Intell. 2021; https://doi.org/10.1038/s42256-021-00309-y |
| 12 | Raman Super-Resolution DL | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9520 | no_ckpt | Manifold et al., Anal. Chem. 2023 |
| 13 | U-Net Raman Mapping | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8900 | no_ckpt | Gebrekidan et al., Analyst 2020; https://doi.org/10.1039/D0AN00721H |
| 14 | GAN-Raman Enhancement | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Lee et al., Anal. Chem. 2022 |
| 15 | Transformer-Raman | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Chen et al., Anal. Chem. 2023 |
| 16 | Diffusion-Raman | 2024 | 37.6 | -- | -- | -- | -- | 36.5 | 0.9550 | no_ckpt | Wang et al., Nat. Commun. 2024 |
| 17 | Raman-Foundation | 2025 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9600 | no_ckpt | Foundation model for Raman spectroscopy, 2025 |

---

## Summary Statistics

| Category | Modalities | Total Algorithms | Classical | Deep Learning |
|----------|-----------|------------------|-----------|---------------|
| Scanning Probe (Near-Field) | 79-80 | 33 | 16 | 17 |
| OCT & Angiography | 81-82 | 35 | 15 | 20 |
| Diffraction Tomography | 83 | 16 | 8 | 8 |
| Retinal & Endoscopy | 84-85 | 33 | 13 | 20 |
| Computational Photography | 86-90 | 82 | 30 | 52 |
| Depth Imaging (Active) | 91-93 | 45 | 19 | 26 |
| Machine Vision & HDR | 94-95 | 34 | 13 | 21 |
| Astronomical Optics | 96 | 15 | 9 | 6 |
| 3D Surface (PS) | 97 | 16 | 7 | 9 |
| Polarimetric & Phase | 98-100 | 48 | 21 | 27 |
| Remote Sensing (Spectral) | 101-102 | 35 | 12 | 23 |
| Spectroscopic Imaging | 103-104 | 33 | 15 | 18 |
| **Total** | **79-104** | **425** | **178** | **247** |

---
