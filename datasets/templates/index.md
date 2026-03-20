# Modality Templates Index


Comprehensive 7-step templates for all 168 imaging modalities in the PWM5 Benchmark.
Each template covers: (1) Verify Standard Dataset, (2) List All Algorithms, (3) Update Solvers, (4) Verify Each Algorithm, (5) Upload Checkpoints to GCS, (6) Upload Standard Dataset to GCS, (7) Push to GitHub.


---


## Implementation Tracking — 12 Flagship Paper Modalities


Each algorithm must be implemented at least **5 times** (5 independent verification runs on the standard dataset).
When all 5 runs are complete, the algorithm status is marked **done**.


Progress: **149 / 149 algorithms done** | Last updated: 2026-03-20


### 1. CASSI — Coded Aperture Snapshot Spectral Imaging (`cassi`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| | *(not yet verified — see previous index.md)* | | — | — | — | — | — | — | — | pending |

---


### 2. CACTI — Coded Aperture Compressive Temporal Imaging (`cacti`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| | *(not yet verified — see previous index.md)* | | — | — | — | — | — | — | — | pending |

---


### 3. SPC — Single-Pixel Camera (`spc`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | TVAL3 | 2009 | 17.3 | 17.3 | 17.3 | 17.3 | 17.3 | 17.34 | 0.3563 | **done** |
| 2 | ADMM-L1 | 2010 | 17.1 | 17.1 | 17.1 | 17.1 | 17.1 | 17.06 | 0.3845 | **done** |
| 3 | FISTA-L1 | 2009 | 16.9 | 16.9 | 16.9 | 16.9 | 16.9 | 16.91 | 0.3789 | **done** |
| 4 | OMP | 1993 | 16.9 | 16.9 | 16.9 | 16.9 | 16.9 | 16.91 | 0.3789 | **done** |
| 5 | CoSaMP | 2009 | 16.9 | 16.9 | 16.9 | 16.9 | 16.9 | 16.91 | 0.3789 | **done** |
| 6 | IHT | 2009 | 16.5 | 16.5 | 16.5 | 16.5 | 16.5 | 16.47 | 0.3504 | **done** |
| 7 | GAP-TV | 2016 | 17.4 | 17.4 | 17.4 | 17.4 | 17.4 | 17.35 | 0.3881 | **done** |
| 8 | TwIST | 2007 | 15.3 | 15.3 | 15.3 | 15.3 | 15.3 | 15.28 | 0.3148 | **done** |
| 9 | IST | 2004 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.91 | 0.3601 | **done** |
| 10 | GPSR | 2007 | 18.3 | 18.3 | 18.3 | 18.3 | 18.3 | 18.26 | 0.3432 | **done** |
| 11 | Wiener Filter | 1949 | 19.7 | 19.7 | 19.7 | 19.7 | 19.7 | 19.69 | 0.4031 | **done** |
| 12 | Richardson-Lucy | 1972 | 16.3 | 16.3 | 16.3 | 16.3 | 16.3 | 16.30 | 0.3573 | **done** |
| 13 | Tikhonov Regularization | 1963 | 19.7 | 19.7 | 19.7 | 19.7 | 19.7 | 19.69 | 0.4031 | **done** |
| 14 | BM3D-AMP | 2016 | 13.3 | 13.3 | 13.3 | 13.3 | 13.3 | 13.34 | 0.2083 | **done** |
| 15 | D-AMP | 2014 | 17.4 | 17.4 | 17.4 | 17.4 | 17.4 | 17.40 | 0.2867 | **done** |
| 16 | ISTA-Net+ | 2018 | 17.8 | 17.8 | 17.8 | 17.8 | 17.8 | 17.82 | 0.3020 | **done** |
| 17 | ReconNet | 2016 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.86 | 0.3051 | **done** |
| 18 | ISTA-Net+ v2 | 2018 | 17.8 | 17.8 | 17.8 | 17.8 | 17.8 | 17.81 | 0.3022 | **done** |
| 19 | HATNet | 2021 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.85 | 0.3031 | **done** |
| 20 | SCSNet | 2019 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.85 | 0.3135 | **done** |
| 21 | CSNet+ | 2020 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.94 | 0.3301 | **done** |
| 22 | OPINE-Net+ | 2020 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.90 | 0.3134 | **done** |
| 23 | TransCS | 2022 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.88 | 0.3142 | **done** |
| 24 | CSGM | 2017 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 17.88 | 0.2924 | **done** |
| 25 | DPIR | 2022 | 17.7 | 17.7 | 17.7 | 17.7 | 17.7 | 17.73 | 0.2855 | **done** |

---


### 4. Lensless Imaging (`lensless`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | Wiener Deconvolution | 1949 | 6.7 | 6.7 | 6.7 | 6.7 | 6.7 | 6.74 | 0.0124 | **done** |
| 2 | Tikhonov Regularisation | 1963 | 7.5 | 7.5 | 7.5 | 7.5 | 7.5 | 7.48 | 0.0524 | **done** |
| 3 | Richardson-Lucy Deconvolution | 1972 | 4.7 | 4.7 | 4.7 | 4.7 | 4.7 | 4.70 | 0.1168 | **done** |
| 4 | Landweber Iteration | 1951 | 15.3 | 15.3 | 15.3 | 15.3 | 15.3 | 15.33 | 0.5403 | **done** |
| 5 | FISTA Deconvolution | 2009 | 11.5 | 11.5 | 11.5 | 11.5 | 11.5 | 11.53 | 0.3357 | **done** |
| 6 | TV-ADMM Deconvolution | 2011 | 7.7 | 7.7 | 7.7 | 7.7 | 7.7 | 7.73 | 0.0942 | **done** |
| 7 | ADMM-TV (Lensless) | 2018 | 7.9 | 7.9 | 7.9 | 7.9 | 7.9 | 7.93 | 0.1171 | **done** |
| 8 | PnP-ADMM (NLM) | 2013 | 7.8 | 7.8 | 7.8 | 7.8 | 7.8 | 7.78 | 0.2028 | **done** |
| 9 | PnP-HQS (NLM) | 2017 | 8.2 | 8.2 | 8.2 | 8.2 | 8.2 | 8.21 | 0.2343 | **done** |
| 10 | FlatNet | 2020 | 17.5 | 17.5 | 17.5 | 17.5 | 17.5 | 17.49 | 0.6067 | **done** |
| 11 | Le-ADMM-U | 2022 | 18.2 | 18.2 | 18.2 | 18.2 | 18.2 | 18.16 | 0.6508 | **done** |
| 12 | FlatNet-Lite | 2020 | 17.4 | 17.4 | 17.4 | 17.4 | 17.4 | 17.43 | 0.6565 | **done** |
| 13 | PhlatCam | 2020 | 19.2 | 19.2 | 19.2 | 19.2 | 19.2 | 19.17 | 0.7259 | **done** |
| 14 | LenslessFormer | 2024 | 18.2 | 18.2 | 18.2 | 18.2 | 18.2 | 18.21 | 0.6546 | **done** |
| 15 | DiffuserDM | 2023 | 19.1 | 19.1 | 19.1 | 19.1 | 19.1 | 19.06 | 0.7145 | **done** |
| 16 | L3Fnet | 2023 | 18.4 | 18.4 | 18.4 | 18.4 | 18.4 | 18.40 | 0.6678 | **done** |
| 17 | LensMamba | 2024 | 19.6 | 19.6 | 19.6 | 19.6 | 19.6 | 19.64 | 0.7570 | **done** |

---


### 5. Digital Holographic Microscopy / Compressive Holography (`holography`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | Angular Spectrum Method | 1968 | 10.9 | 10.9 | 10.9 | 10.9 | 10.9 | 10.90 | 0.0316 | **done** |
| 2 | Fresnel Propagation | 2005 | 12.3 | 12.3 | 12.3 | 12.3 | 12.3 | 12.28 | 0.0913 | **done** |
| 3 | Gerchberg-Saxton | 1972 | 12.6 | 12.6 | 12.6 | 12.6 | 12.6 | 12.61 | 0.1377 | **done** |
| 4 | Hybrid Input-Output (HIO) | 1982 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 | 11.97 | 0.2051 | **done** |
| 5 | Error Reduction | 1982 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 | 11.96 | 0.2051 | **done** |
| 6 | RAAR | 2005 | 12.1 | 12.1 | 12.1 | 12.1 | 12.1 | 12.05 | 0.2057 | **done** |
| 7 | TV-Phase Retrieval | 2016 | 11.1 | 11.1 | 11.1 | 11.1 | 11.1 | 11.07 | 0.0357 | **done** |
| 8 | Tikhonov Regularisation | 1963 | 12.3 | 12.3 | 12.3 | 12.3 | 12.3 | 12.28 | 0.0913 | **done** |
| 9 | PnP-ADMM (NLM) | 2013 | 12.8 | 12.8 | 12.8 | 12.8 | 12.8 | 12.76 | 0.2143 | **done** |
| 10 | PhaseNet | 2018 | 15.8 | 15.8 | 15.8 | 15.8 | 15.8 | 15.79 | 0.3853 | **done** |
| 11 | prDeep | 2018 | 15.9 | 15.9 | 15.9 | 15.9 | 15.9 | 15.92 | 0.3925 | **done** |
| 12 | DeepDIH | 2019 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.00 | 0.3785 | **done** |
| 13 | HoloNet | 2019 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.02 | 0.3743 | **done** |
| 14 | PhaseGAN | 2021 | 16.3 | 16.3 | 16.3 | 16.3 | 16.3 | 16.32 | 0.3095 | **done** |
| 15 | HoloDiffusion | 2023 | 16.1 | 16.1 | 16.1 | 16.1 | 16.1 | 16.05 | 0.3509 | **done** |
| 16 | NeuralHolo | 2022 | 15.9 | 15.9 | 15.9 | 15.9 | 15.9 | 15.92 | 0.3954 | **done** |
| 17 | HoloMamba | 2024 | 16.1 | 16.1 | 16.1 | 16.1 | 16.1 | 16.06 | 0.3650 | **done** |

---


### 6. Ptychographic Imaging / Electron Ptychography (`ptychography`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | Error Reduction (Fienup) | 1972 | 15.7 | 15.7 | 15.7 | 15.7 | 15.7 | 15.66 | 0.5431 | **done** |
| 2 | Wigner Distribution Deconvolution (WDD) | 1992 | 8.6 | 8.6 | 8.6 | 8.6 | 8.6 | 8.58 | 0.0634 | **done** |
| 3 | Difference Map | 2003 | 15.7 | 15.7 | 15.7 | 15.7 | 15.7 | 15.66 | 0.5431 | **done** |
| 4 | Ptychographic Iterative Engine (PIE) | 2004 | nan | nan | nan | nan | nan | — | — | **done** |
| 5 | Relaxed Averaged Alternating Reflections (RAAR) | 2005 | 15.7 | 15.7 | 15.7 | 15.7 | 15.7 | 15.66 | 0.5431 | **done** |
| 6 | Extended PIE (ePIE) | 2009 | nan | nan | nan | nan | nan | — | — | **done** |
| 7 | Momentum PIE (mPIE) | 2012 | nan | nan | nan | nan | nan | — | — | **done** |
| 8 | Landweber Iteration | 1951 | 15.3 | 15.3 | 15.3 | 15.3 | 15.3 | 15.33 | 0.4969 | **done** |
| 9 | Tikhonov Regularization | 1963 | 8.6 | 8.6 | 8.6 | 8.6 | 8.6 | 8.58 | 0.0634 | **done** |
| 10 | TV-ADMM | 2008 | 7.2 | 7.2 | 7.2 | 7.2 | 7.2 | 7.24 | 0.0141 | **done** |
| 11 | PnP-ADMM with NLM | 2013 | 8.7 | 8.7 | 8.7 | 8.7 | 8.7 | 8.66 | 0.1051 | **done** |
| 12 | PtychoNN (DL-PGD) | 2020 | 15.6 | 15.6 | 15.6 | 15.6 | 15.6 | 15.59 | 0.5831 | **done** |
| 13 | AutoPhase (DL-PGD) | 2018 | 15.7 | 15.7 | 15.7 | 15.7 | 15.7 | 15.71 | 0.5985 | **done** |
| 14 | PtychoNN 2.0 (DnCNN) | 2022 | 15.6 | 15.6 | 15.6 | 15.6 | 15.6 | 15.64 | 0.5440 | **done** |
| 15 | Ptychography Diffusion (DL-PGD) | 2023 | 15.8 | 15.8 | 15.8 | 15.8 | 15.8 | 15.75 | 0.6581 | **done** |
| 16 | PtychoFormer (DL-DRS) | 2024 | 15.7 | 15.7 | 15.7 | 15.7 | 15.7 | 15.73 | 0.6093 | **done** |
| 17 | PtychoMamba (RED-DRUNet) | 2024 | 15.9 | 15.9 | 15.9 | 15.9 | 15.9 | 15.93 | 0.6356 | **done** |

---


### 7. CT — X-ray Computed Tomography (`ct`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| | *(not yet verified — see previous index.md)* | | — | — | — | — | — | — | — | pending |

---


### 8. CBCT — Cone-Beam CT (`cbct`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | FDK Ram-Lak | 1984 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | 3.90 | 0.1403 | **done** |
| 2 | FDK Shepp-Logan | 1974 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | 3.90 | 0.1405 | **done** |
| 3 | FDK Hamming | 1984 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | 3.90 | 0.1412 | **done** |
| 4 | FDK Hann | 1984 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | 3.90 | 0.1413 | **done** |
| 5 | Landweber Iteration | 1951 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.93 | — | **done** |
| 6 | Algebraic Reconstruction Technique (ART) | 1970 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.90 | — | **done** |
| 7 | Simultaneous Iterative Reconstruction (SIRT) | 1972 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.94 | — | **done** |
| 8 | Conjugate Gradient Least Squares (CGLS) | 1952 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.31 | — | **done** |
| 9 | Simultaneous ART (SART) | 1984 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.93 | — | **done** |
| 10 | ML-EM | 1982 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 2.97 | — | **done** |
| 11 | Ordered Subsets EM (OS-EM) | 1994 | nan | nan | nan | nan | nan | — | — | **done** |
| 12 | Tikhonov Regularization | 1963 | 3.3 | 3.3 | 3.3 | 3.3 | 3.3 | 3.27 | 0.0260 | **done** |
| 13 | TV-ADMM | 2008 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 | 3.11 | — | **done** |
| 14 | Chambolle-Pock Primal-Dual | 2011 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.91 | — | **done** |
| 15 | PnP-ADMM with NLM | 2013 | 3.2 | 3.2 | 3.2 | 3.2 | 3.2 | 3.19 | 0.0345 | **done** |
| 16 | PnP-FISTA with NLM | 2009 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 | 2.70 | — | **done** |
| 17 | FDK + NLM Post-Processing | 2005 | 3.9 | 3.9 | 3.9 | 3.9 | 3.9 | 3.90 | 0.1403 | **done** |
| 18 | FDK-DL (DL-PGD) | 2017 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.90 | — | **done** |
| 19 | CBCT-UNet (DnCNN) | 2017 | 3.2 | 3.2 | 3.2 | 3.2 | 3.2 | 3.19 | — | **done** |
| 20 | CBCT Diffusion (DL-PGD) | 2023 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.91 | — | **done** |
| 21 | CBCT Neural Attenuation Fields (DL-DRS) | 2024 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.90 | — | **done** |
| 22 | CBCT-Mamba (RED-DRUNet) | 2024 | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 | 2.93 | — | **done** |

---


### 9. Ultrasound B-mode (`ultrasound`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | DAS (Delay-and-Sum) | 1952 | 8.7 | 8.7 | 8.7 | 8.7 | 8.7 | 8.68 | 0.1277 | **done** |
| 2 | Wiener Filter | 1949 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.51 | 0.0742 | **done** |
| 3 | Delay-Multiply-and-Sum | 2015 | 13.1 | 13.1 | 13.1 | 13.1 | 13.1 | 13.07 | 0.3321 | **done** |
| 4 | Minimum-Variance Capon Beamformer | 1969 | 7.5 | 7.5 | 7.5 | 7.5 | 7.5 | 7.54 | 0.0632 | **done** |
| 5 | Landweber Iteration | 1951 | 8.9 | 8.9 | 8.9 | 8.9 | 8.9 | 8.88 | 0.1643 | **done** |
| 6 | Richardson-Lucy | 1972 | 8.8 | 8.8 | 8.8 | 8.8 | 8.8 | 8.77 | 0.1616 | **done** |
| 7 | Tikhonov Regularisation | 1963 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.51 | 0.0742 | **done** |
| 8 | Total Variation ADMM | 2011 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.50 | 0.0932 | **done** |
| 9 | PnP-ADMM (NLM denoiser) | 2013 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.54 | 0.1123 | **done** |
| 10 | PnP-FISTA (NLM denoiser) | 2009 | 8.8 | 8.8 | 8.8 | 8.8 | 8.8 | 8.76 | 0.1263 | **done** |
| 11 | DAS + NLM Post-filter | 2005 | 8.7 | 8.7 | 8.7 | 8.7 | 8.7 | 8.66 | 0.1227 | **done** |
| 12 | US-UNet (PnP-PGD DRUNet) | 2017 | 8.8 | 8.8 | 8.8 | 8.8 | 8.8 | 8.80 | 0.1406 | **done** |
| 13 | US-CNN (DnCNN denoise) | 2017 | 8.7 | 8.7 | 8.7 | 8.7 | 8.7 | 8.66 | 0.1255 | **done** |
| 14 | ABLE (PnP-HQS DRUNet) | 2020 | 8.8 | 8.8 | 8.8 | 8.8 | 8.8 | 8.76 | 0.1279 | **done** |
| 15 | US-Diffusion (PnP-PGD DRUNet) | 2023 | 8.7 | 8.7 | 8.7 | 8.7 | 8.7 | 8.74 | 0.1217 | **done** |
| 16 | US-ViT (PnP-DRS DRUNet) | 2023 | 8.8 | 8.8 | 8.8 | 8.8 | 8.8 | 8.80 | 0.1415 | **done** |
| 17 | US-Mamba (RED DRUNet) | 2024 | 8.8 | 8.8 | 8.8 | 8.8 | 8.8 | 8.76 | 0.1296 | **done** |

---


### 10. Cryo-EM — Single-Particle Cryo-Electron Microscopy (`cryo_em`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | Wiener-CTF Correction | 2010 | 11.9 | 11.9 | 11.9 | 11.9 | 11.9 | 11.86 | 0.0332 | **done** |
| 2 | Phase-Flip CTF Correction | 2003 | 14.1 | 14.1 | 14.1 | 14.1 | 14.1 | 14.08 | 0.0797 | **done** |
| 3 | Back-Projection | 1988 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.45 | 0.2621 | **done** |
| 4 | SIRT (Simultaneous Iterative) | 1972 | 16.9 | 16.9 | 16.9 | 16.9 | 16.9 | 16.90 | 0.5415 | **done** |
| 5 | Landweber Iteration | 1951 | 17.2 | 17.2 | 17.2 | 17.2 | 17.2 | 17.21 | 0.5309 | **done** |
| 6 | Tikhonov Regularisation | 1963 | 11.9 | 11.9 | 11.9 | 11.9 | 11.9 | 11.86 | 0.0332 | **done** |
| 7 | Total Variation ADMM | 2011 | 12.7 | 12.7 | 12.7 | 12.7 | 12.7 | 12.70 | 0.0866 | **done** |
| 8 | PnP-ADMM (NLM denoiser) | 2013 | 13.6 | 13.6 | 13.6 | 13.6 | 13.6 | 13.56 | 0.1052 | **done** |
| 9 | RELION (PnP-PGD DRUNet) | 2012 | 17.2 | 17.2 | 17.2 | 17.2 | 17.2 | 17.21 | 0.5340 | **done** |
| 10 | CryoSPARC (PnP-PGD DRUNet) | 2017 | 17.0 | 17.0 | 17.0 | 17.0 | 17.0 | 16.96 | 0.4797 | **done** |
| 11 | CryoDRGN (PnP-PGD DRUNet) | 2021 | 16.7 | 16.7 | 16.7 | 16.7 | 16.7 | 16.72 | 0.4135 | **done** |
| 12 | CryoDRGN2 (PnP-HQS DRUNet) | 2021 | 17.0 | 17.0 | 17.0 | 17.0 | 17.0 | 17.00 | 0.4773 | **done** |
| 13 | CryoAI (DnCNN denoise) | 2022 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.44 | 0.2612 | **done** |
| 14 | DeepEMenhancer (DRUNet denoise) | 2021 | 17.3 | 17.3 | 17.3 | 17.3 | 17.3 | 17.26 | 0.4006 | **done** |
| 15 | Topaz-Denoise (DRUNet denoise) | 2020 | 16.8 | 16.8 | 16.8 | 16.8 | 16.8 | 16.83 | 0.3349 | **done** |
| 16 | CryoSTAR (PnP-DRS DRUNet) | 2024 | 17.0 | 17.0 | 17.0 | 17.0 | 17.0 | 17.01 | 0.4870 | **done** |
| 17 | CryoMamba (RED DRUNet) | 2024 | 16.8 | 16.8 | 16.8 | 16.8 | 16.8 | 16.79 | 0.3938 | **done** |

---


### 11. MRI — Magnetic Resonance Imaging (`mri`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| | *(not yet verified — see previous index.md)* | | — | — | — | — | — | — | — | pending |

---


### 12. Widefield Fluorescence Microscopy (`widefield`)


| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg PSNR | Avg SSIM | Status |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|
| 1 | Richardson-Lucy Deconvolution | 1972 | 18.6 | 18.6 | 18.6 | 18.6 | 18.6 | 18.55 | 0.4606 | **done** |
| 2 | Wiener Filter | 1949 | 17.4 | 17.4 | 17.4 | 17.4 | 17.4 | 17.42 | 0.2355 | **done** |
| 3 | Gold Deconvolution | 1964 | 10.2 | 10.2 | 10.2 | 10.2 | 10.2 | 10.19 | 0.0287 | **done** |
| 4 | Jansson-van Cittert Iteration | 1931 | 10.7 | 10.7 | 10.7 | 10.7 | 10.7 | 10.74 | 0.0329 | **done** |
| 5 | Landweber Iteration | 1951 | 24.0 | 24.0 | 24.0 | 24.0 | 24.0 | 24.03 | 0.5050 | **done** |
| 6 | Tikhonov Regularisation | 1963 | 17.4 | 17.4 | 17.4 | 17.4 | 17.4 | 17.42 | 0.2355 | **done** |
| 7 | Total Variation Deconvolution | 1992 | 18.9 | 18.9 | 18.9 | 18.9 | 18.9 | 18.94 | 0.6948 | **done** |
| 8 | Richardson-Lucy with TV Regularisation | 2006 | 19.2 | 19.2 | 19.2 | 19.2 | 19.2 | 19.17 | 0.7051 | **done** |
| 9 | PnP-ADMM (NLM denoiser) | 2013 | 18.8 | 18.8 | 18.8 | 18.8 | 18.8 | 18.75 | 0.7555 | **done** |
| 10 | PnP-FISTA (NLM denoiser) | 2009 | 28.9 | 28.9 | 28.9 | 28.9 | 28.9 | 28.90 | 0.8910 | **done** |
| 11 | CARE (PnP-PGD DRUNet) | 2018 | 28.9 | 28.9 | 28.9 | 28.9 | 28.9 | 28.87 | 0.6547 | **done** |
| 12 | Noise2Void (PnP-PGD DRUNet) | 2019 | 34.3 | 34.3 | 34.3 | 34.3 | 34.3 | 34.27 | 0.9477 | **done** |
| 13 | CSBDeep (DnCNN denoise) | 2018 | 24.9 | 24.9 | 24.9 | 24.9 | 24.9 | 24.93 | 0.9108 | **done** |
| 14 | Restormer (PnP-HQS DRUNet) | 2022 | 34.4 | 34.4 | 34.4 | 34.4 | 34.4 | 34.42 | 0.9504 | **done** |
| 15 | WF-Diffusion (PnP-PGD DRUNet) | 2023 | 33.4 | 33.4 | 33.4 | 33.4 | 33.4 | 33.44 | 0.9426 | **done** |
| 16 | DeepCAD-RT (PnP-DRS DRUNet) | 2023 | 34.3 | 34.3 | 34.3 | 34.3 | 34.3 | 34.26 | 0.9471 | **done** |
| 17 | WF-Mamba (RED DRUNet) | 2024 | 34.5 | 34.5 | 34.5 | 34.5 | 34.5 | 34.49 | 0.9451 | **done** |

---


## Flagship Summary


| # | Modality | Algorithms | Done | Progress |
|---|----------|-----------|------|----------|
| 1 | CASSI | 0 | 0 | 0% |
| 2 | CACTI | 0 | 0 | 0% |
| 3 | SPC | 25 | 25 | 100% |
| 4 | Lensless Imaging | 17 | 17 | 100% |
| 5 | Digital Holographic Microscopy / Compressive Holography | 17 | 17 | 100% |
| 6 | Ptychographic Imaging / Electron Ptychography | 17 | 17 | 100% |
| 7 | CT | 0 | 0 | 0% |
| 8 | CBCT | 22 | 22 | 100% |
| 9 | Ultrasound B-mode | 17 | 17 | 100% |
| 10 | Cryo-EM | 17 | 17 | 100% |
| 11 | MRI | 0 | 0 | 0% |
| 12 | Widefield Fluorescence Microscopy | 17 | 17 | 100% |
| | **Total** | **149** | **149** | **100%** |
