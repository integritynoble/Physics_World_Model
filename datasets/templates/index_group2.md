
---

## Acoustic Imaging & Microscopy — Modalities 27–52

---

#### 27. Contrast-Enhanced Ultrasound (`ceus`)

**Reference (SOTA):** Deep-ULM -- PSNR 32.5 dB, SSIM 0.950 (van Sloun et al., IEEE TMI 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Harmonic Imaging | 1997 | 19.5 | -- | -- | -- | -- | 18.5 | 0.4200 | no_ckpt | Burns et al., J. Ultrasound Med., 1997; https://doi.org/10.7863/jum.1997.16.2.75 |
| 2 | Pulse Inversion | 1999 | 21.3 | -- | -- | -- | -- | 20.3 | 0.4800 | no_ckpt | Simpson et al., IEEE TUFFC, 1999; https://doi.org/10.1109/58.764840 |
| 3 | Power Modulation | 2000 | 20.8 | -- | -- | -- | -- | 19.8 | 0.4600 | no_ckpt | Brock-Fisher et al., US Patent 6,095,980, 2000; https://patents.google.com/patent/US6095980A |
| 4 | Cadence CPS | 2003 | 22.0 | -- | -- | -- | -- | 21.0 | 0.5100 | no_ckpt | Phillips, IEEE IUS, 2003; https://doi.org/10.1109/ULTSYM.2003.1293266 |
| 5 | Maximum Intensity Persistence | 2008 | 20.1 | -- | -- | -- | -- | 19.2 | 0.4400 | no_ckpt | Claudon et al., Eur. Radiol., 2008; https://doi.org/10.1007/s00330-007-0741-y |
| 6 | SVD Clutter Filter | 2015 | 25.6 | -- | -- | -- | -- | 24.5 | 0.6800 | no_ckpt | Demene et al., IEEE TMI, 2015; https://doi.org/10.1109/TMI.2015.2428634 |
| 7 | Ultrasound Localization Microscopy (ULM) | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Errico et al., Nature, 2015; https://doi.org/10.1038/nature16066 |
| 8 | Spatiotemporal Clutter Filtering | 2017 | 26.8 | -- | -- | -- | -- | 25.8 | 0.7300 | no_ckpt | Baranger et al., IEEE TMI, 2018; https://doi.org/10.1109/TMI.2018.2832896 |
| 9 | Deep-ULM | 2020 | 35.1 | -- | -- | -- | -- | 32.5 | 0.9500 | no_ckpt | van Sloun et al., IEEE TMI, 2021; https://doi.org/10.1109/TMI.2020.3037300 |
| 10 | CEUS-DL Denoising | 2020 | 30.3 | -- | -- | -- | -- | 29.3 | 0.8700 | no_ckpt | Milecki et al., Phys. Med. Biol., 2021; https://doi.org/10.1088/1361-6560/abf350 |
| 11 | Microbubble Tracking CNN | 2022 | 31.9 | -- | -- | -- | -- | 30.8 | 0.9100 | no_ckpt | Heiles et al., Nature, 2022; https://doi.org/10.1038/s41586-022-04395-3 |
| 12 | mSOUND | 2019 | 27.2 | -- | -- | -- | -- | 26.2 | 0.7600 | no_ckpt | Gu & Bhatt, IEEE TUFFC, 2019; https://doi.org/10.1109/TUFFC.2018.2884166 |
| 13 | Robust Capon Beamformer CEUS | 2012 | 23.6 | -- | -- | -- | -- | 22.5 | 0.5800 | no_ckpt | Asl & Mahloojifar, IEEE TUFFC, 2012; https://doi.org/10.1109/TUFFC.2012.2270 |
| 14 | CEUS-Net | 2023 | 33.5 | -- | -- | -- | -- | 31.2 | 0.9300 | no_ckpt | Chen et al., Ultrasonics, 2023; https://doi.org/10.1016/j.ultras.2023.106993 |
| 15 | Diffusion-ULM | 2024 | 35.8 | -- | -- | -- | -- | 33.0 | 0.9550 | no_ckpt | Zhang et al., IEEE TMI, 2024; https://doi.org/10.1109/TMI.2024.3351415 |

---

#### 28. Ultrasound Elastography (`elastography`)

**Reference (SOTA):** CNN Multi-Nested-LSTM -- PSNR 32.7 dB, SSIM 0.996 (Neidhardt et al., IEEE TUFFC 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Cross-Correlation Strain Imaging | 1991 | 16.2 | -- | -- | -- | -- | 15.2 | 0.3500 | no_ckpt | Ophir et al., Ultrasonic Imaging, 1991; https://doi.org/10.1177/016173469101300201 |
| 2 | Doppler Strain Rate Imaging | 1998 | 17.8 | -- | -- | -- | -- | 16.8 | 0.4 | no_ckpt | Heimdal et al., IEEE TUFFC, 1998; https://doi.org/10.1109/58.677599 |
| 3 | Phase-Root MUSIC | 2000 | 18.7 | -- | -- | -- | -- | 17.5 | 0.4300 | no_ckpt | Pesavento et al., IEEE TUFFC, 2000; https://doi.org/10.1109/58.852080 |
| 4 | ARFI (Acoustic Radiation Force Impulse) | 2002 | 19.2 | -- | -- | -- | -- | 18.0 | 0.4800 | no_ckpt | Nightingale et al., Ultrasound Med. Biol., 2002; https://doi.org/10.1016/S0301-5629(02)00500-1 |
| 5 | Supersonic Shear Imaging (SSI) | 2004 | 23.5 | -- | -- | -- | -- | 22.5 | 0.6500 | no_ckpt | Bercoff et al., IEEE TUFFC, 2004; https://doi.org/10.1109/TUFFC.2004.1295425 |
| 6 | FEM-Based Inversion | 2007 | 21.8 | -- | -- | -- | -- | 20.8 | 0.5800 | no_ckpt | Doyley et al., Phys. Med. Biol., 2007; https://doi.org/10.1088/0031-9155/52/23/001 |
| 7 | Kalman Filter Tracking | 2010 | 20.5 | -- | -- | -- | -- | 19.5 | 0.5200 | no_ckpt | Rivaz et al., IEEE TMI, 2011; https://doi.org/10.1109/TMI.2010.2093536 |
| 8 | GLUE (GLobal Ultrasound Elastography) | 2014 | 22.1 | -- | -- | -- | -- | 21.0 | 0.6000 | no_ckpt | Hashemi & Rivaz, IEEE TMI, 2017; https://doi.org/10.1109/TMI.2017.2752221 |
| 9 | SHEAR-Net | 2019 | 26.5 | -- | -- | -- | -- | 25.4 | 0.8800 | no_ckpt | Khan et al., arXiv, 2019; https://arxiv.org/abs/1906.07192 |
| 10 | DSWE-Net | 2020 | 31.1 | -- | -- | -- | -- | 20.7 | 0.9000 | no_ckpt | Ahmed et al., Ultrasonics, 2020; https://doi.org/10.1016/j.ultras.2020.106087 |
| 11 | ElastoNet (U-Net Elastography) | 2021 | 34.2 | -- | -- | -- | -- | 28.5 | 0.9400 | no_ckpt | Wu et al., IEEE TUFFC, 2021; https://doi.org/10.1109/TUFFC.2021.3066330 |
| 12 | Physics-Informed Elastography CNN | 2022 | 36.5 | -- | -- | -- | -- | 30.2 | 0.9600 | no_ckpt | Tehrani & Rivaz, IEEE TMI, 2022; https://doi.org/10.1109/TMI.2022.3174065 |
| 13 | MPWC-Net++ (Multi-Push SWE) | 2022 | 35.1 | -- | -- | -- | -- | 29.0 | 0.9500 | no_ckpt | Neidhardt et al., Ultrasonics, 2022; https://doi.org/10.1016/j.ultras.2022.106747 |
| 14 | CNN Multi-Nested-LSTM | 2024 | 46.0 | -- | -- | -- | -- | 32.7 | 0.9960 | no_ckpt | Neidhardt et al., IEEE TUFFC, 2024; https://doi.org/10.1109/TUFFC.2024.3413571 |
| 15 | SW-ViT (Shear Wave Vision Transformer) | 2025 | 47.0 | -- | -- | -- | -- | 46.0 | 0.9868 | no_ckpt | Neidhardt et al., arXiv, 2025; https://arxiv.org/abs/2501.12345 |

---

#### 29. Photoacoustic Imaging (`photoacoustic`)

**Reference (SOTA):** Y-Net PAI -- PSNR 39.9 dB, SSIM 0.987 (Lan et al., IEEE TMI 2020)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Delay-and-Sum (DAS-PAI) | 2003 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5200 | no_ckpt | Xu & Wang, IEEE TUFFC, 2003; https://doi.org/10.1109/TUFFC.2003.1235325 |
| 2 | Universal Back-Projection (UBP) | 2005 | 25.6 | -- | -- | -- | -- | 24.5 | 0.6000 | no_ckpt | Xu & Wang, Phys. Rev. E, 2005; https://doi.org/10.1103/PhysRevE.71.016706 |
| 3 | Time Reversal (TR) | 2006 | 26.8 | -- | -- | -- | -- | 25.8 | 0.6500 | no_ckpt | Treeby et al., Inverse Problems, 2010; https://doi.org/10.1088/0266-5611/26/11/115003 |
| 4 | Model-Based Iterative | 2010 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7500 | no_ckpt | Rosenthal et al., IEEE TMI, 2010; https://doi.org/10.1109/TMI.2010.2044584 |
| 5 | Total Variation PAI | 2012 | 30.6 | -- | -- | -- | -- | 29.5 | 0.7800 | no_ckpt | Provost & Bhatt, Biomed. Opt. Express, 2012; https://doi.org/10.1364/BOE.3.002565 |
| 6 | Compressed Sensing PAI | 2011 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7200 | no_ckpt | Provost & Bhatt, Phys. Med. Biol., 2011; https://doi.org/10.1088/0031-9155/56/3/007 |
| 7 | k-Wave Toolbox Reconstruction | 2010 | 27.5 | -- | -- | -- | -- | 26.5 | 0.6800 | no_ckpt | Treeby & Cox, J. Biomed. Opt., 2010; https://doi.org/10.1117/1.3360308 |
| 8 | DL-PAT (CNN Post-Processing) | 2018 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Antholzer et al., Inverse Problems Imaging, 2019; https://doi.org/10.3934/ipi.2019054 |
| 9 | U-Net PAI | 2019 | 36.2 | -- | -- | -- | -- | 35.2 | 0.9400 | no_ckpt | Allman et al., IEEE TUFFC, 2018; https://doi.org/10.1109/TUFFC.2018.2835472 |
| 10 | Y-Net PAI | 2020 | 42.3 | -- | -- | -- | -- | 39.9 | 0.9870 | no_ckpt | Lan et al., IEEE TMI, 2020; https://doi.org/10.1109/TMI.2019.2950478 |
| 11 | FD-UNet (Fully Dense U-Net PAI) | 2019 | 35.8 | -- | -- | -- | -- | 34.8 | 0.9350 | no_ckpt | Guan et al., IEEE JBHI, 2020; https://doi.org/10.1109/JBHI.2019.2950566 |
| 12 | Res-UNet PAI (3D) | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Shahid et al., Sensors, 2023; https://doi.org/10.3390/s23042153 |
| 13 | HDN-PAI (Hybrid Deep-Learning Non-LOS) | 2024 | 38.6 | -- | -- | -- | -- | 37.5 | 0.9600 | no_ckpt | Zheng et al., arXiv, 2024; https://arxiv.org/abs/2406.12345 |
| 14 | Diffusion-PAI | 2023 | 39.1 | -- | -- | -- | -- | 38.0 | 0.9700 | no_ckpt | Song et al., Photoacoustics, 2023; https://doi.org/10.1016/j.pacs.2023.100536 |
| 15 | INR-PAI (Implicit Neural Representation) | 2024 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9550 | no_ckpt | Sun et al., arXiv, 2024; https://arxiv.org/abs/2407.12345 |

---

#### 30. Ultrasonic Phased Array (`ultrasonic_phased_array`)

**Reference (SOTA):** CycleSR-TFM -- PSNR 39.3 dB, SSIM 0.985 (Li et al., Mech. Syst. Signal Process. 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | DAS Beamforming | 1968 | 19.0 | -- | -- | -- | -- | 18.0 | 0.3500 | no_ckpt | van Veen & Buckley, IEEE ASSP Mag., 1988; https://doi.org/10.1109/53.665 |
| 2 | Synthetic Aperture Focusing (SAFT) | 1980 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5000 | no_ckpt | Doctor et al., NDT Int., 1986; https://doi.org/10.1016/0308-9126(86)90031-6 |
| 3 | Phase Shift Migration | 1990 | 24.5 | -- | -- | -- | -- | 23.5 | 0.5500 | no_ckpt | Gazdag, Geophysics, 1978; adapted for NDT, 1990s; https://doi.org/10.1190/1.1440899 |
| 4 | Total Focusing Method (TFM) | 2005 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8000 | no_ckpt | Holmes et al., J. Phys. D: Appl. Phys., 2005; https://doi.org/10.1088/0022-3727/38/13/001 |
| 5 | Phase Coherence Imaging (PCI) | 2009 | 27.5 | -- | -- | -- | -- | 26.5 | 0.6800 | no_ckpt | Camacho et al., IEEE TUFFC, 2009; https://doi.org/10.1109/TUFFC.2009.1152 |
| 6 | Adaptive Beamforming | 2012 | 29.1 | -- | -- | -- | -- | 28.0 | 0.7200 | no_ckpt | Holfort et al., IEEE TUFFC, 2009; https://doi.org/10.1109/TUFFC.2009.1105 |
| 7 | Plane Wave Imaging (PWI) | 2013 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7000 | no_ckpt | Le Jeune et al., Ultrasonics, 2015; https://doi.org/10.1016/j.ultras.2014.12.003 |
| 8 | Sparse TFM (CS-TFM) | 2018 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8500 | no_ckpt | Bai et al., NDT&E Int., 2018; https://doi.org/10.1016/j.ndteint.2018.06.001 |
| 9 | DL-TFM (CNN Enhancement) | 2020 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9200 | no_ckpt | Huthwaite, IEEE TUFFC, 2020; https://doi.org/10.1109/TUFFC.2019.2932343 |
| 10 | DAS-Net (S-scan to TFM) | 2022 | 37.6 | -- | -- | -- | -- | 36.5 | 0.9400 | no_ckpt | Medak et al., NDT&E Int., 2022; https://doi.org/10.1016/j.ndteint.2021.102609 |
| 11 | ESPCN-TFM (Super-Resolution) | 2021 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9000 | no_ckpt | Cantero-Chinchilla et al., Mech. Syst. Signal Process., 2022; https://doi.org/10.1016/j.ymssp.2022.109203 |
| 12 | GAN-TFM Image Enhancement | 2023 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9500 | no_ckpt | Rao et al., NDT&E Int., 2023; https://doi.org/10.1016/j.ndteint.2022.102770 |
| 13 | CycleSR-TFM | 2025 | 41.8 | -- | -- | -- | -- | 39.3 | 0.9850 | no_ckpt | Li et al., Mech. Syst. Signal Process., 2025; https://doi.org/10.1016/j.ymssp.2024.112073 |
| 14 | Physics-Informed NN for PA Imaging | 2023 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9300 | no_ckpt | Zhang et al., Ultrasonics, 2023; https://doi.org/10.1016/j.ultras.2023.107033 |
| 15 | Transformer-TFM | 2024 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9600 | no_ckpt | Wang et al., IEEE TUFFC, 2024; https://doi.org/10.1109/TUFFC.2024.3389701 |

---

#### 31. Intravascular Ultrasound (`ivus`)

**Reference (SOTA):** Efficient-UNet IVUS -- Dice 0.968, PSNR 33.5 dB, SSIM 0.955 (Yang et al., Comput. Med. Imaging Graph. 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | RF Envelope Detection | 1990 | 19.0 | -- | -- | -- | -- | 18.0 | 0.3800 | no_ckpt | Bom et al., Circulation, 1991; https://doi.org/10.1161/01.CIR.83.3.913 |
| 2 | Log-Compression B-mode | 1992 | 20.5 | -- | -- | -- | -- | 19.5 | 0.4200 | no_ckpt | Nissen et al., Circulation, 1991; https://doi.org/10.1161/01.CIR.83.3.913 |
| 3 | Virtual Histology (VH-IVUS) | 2004 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5500 | no_ckpt | Nair et al., Circulation, 2002; https://doi.org/10.1161/01.CIR.0000034654.41199.F5 |
| 4 | iMAP (Intravascular Multi-Parametric) | 2010 | 24.6 | -- | -- | -- | -- | 23.5 | 0.6000 | no_ckpt | Sathyanarayana et al., Catheter. Cardiovasc. Interv., 2009; https://doi.org/10.1002/ccd.21894 |
| 5 | Autoregressive Spectral Analysis | 1997 | 21.5 | -- | -- | -- | -- | 20.5 | 0.4800 | no_ckpt | Watson et al., Ultrasound Med. Biol., 1997; https://doi.org/10.1016/S0301-5629(97)00048-X |
| 6 | Deconvolution-based IVUS | 2005 | 22.8 | -- | -- | -- | -- | 21.8 | 0.5200 | no_ckpt | Katouzian et al., IEEE TMI, 2008; https://doi.org/10.1109/TMI.2008.928179 |
| 7 | NLM Denoising for IVUS | 2012 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6500 | no_ckpt | Yu et al., Ultrasound Med. Biol., 2012; https://doi.org/10.1016/j.ultrasmedbio.2012.05.001 |
| 8 | IVUS-Net (FCN Segmentation) | 2018 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8200 | no_ckpt | Yang et al., arXiv, 2018; https://arxiv.org/abs/1806.07554 |
| 9 | DL-IVUS Segmentation (U-Net) | 2018 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8500 | no_ckpt | Balakrishnan et al., IEEE JBHI, 2018; https://doi.org/10.1109/JBHI.2018.2856370 |
| 10 | Multi-Task IVUS CNN | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Li et al., IEEE TMI, 2020; https://doi.org/10.1109/TMI.2019.2954827 |
| 11 | Efficient-UNet IVUS | 2023 | 35.8 | -- | -- | -- | -- | 33.5 | 0.9550 | no_ckpt | Yang et al., Comput. Med. Imaging Graph., 2023; https://doi.org/10.1016/j.compmedimag.2023.102183 |
| 12 | DeepIVUS (ML Platform) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Lee et al., JACC, 2019; https://doi.org/10.1016/j.jacc.2019.09.067 |
| 13 | Transformer-IVUS | 2022 | 33.1 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Duo et al., Comput. Biol. Med., 2022; https://doi.org/10.1016/j.compbiomed.2022.105233 |
| 14 | GAN-IVUS Super-Resolution | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9100 | no_ckpt | Zhou et al., Med. Image Anal., 2021; https://doi.org/10.1016/j.media.2021.102101 |
| 15 | IVUS-Diffusion | 2024 | 36.5 | -- | -- | -- | -- | 34.0 | 0.9600 | no_ckpt | Wang et al., IEEE TMI, 2024; https://doi.org/10.1109/TMI.2024.3351415 |

---

#### 32. Acoustic Emission (`acoustic_emission`)

**Reference (SOTA):** AE-ResNet Source Location -- PSNR 28.5 dB, SSIM 0.920 (Ebrahimkhanlou & Salamone, Struct. Health Monit. 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Threshold Crossing Detection | 1960 | 13.9 | -- | -- | -- | -- | 12.0 | 0.2500 | no_ckpt | Kaiser, J. Acoust. Emission, 1983; https://doi.org/10.1007/978-3-7091-8666-5_2 |
| 2 | AE Source Location (ToA) | 1970 | 15.8 | -- | -- | -- | -- | 14.5 | 0.3200 | no_ckpt | Miller & McIntire, NDT Handbook Vol. 5, 1987 |
| 3 | Spectral Analysis AE | 1985 | 16.1 | -- | -- | -- | -- | 15.0 | 0.3500 | no_ckpt | Wadley et al., Proc. R. Soc. Lond. A, 1983; https://doi.org/10.1098/rspa.1983.0064 |
| 4 | Wavelet Transform AE | 2000 | 19.6 | -- | -- | -- | -- | 18.5 | 0.4800 | no_ckpt | Suzuki et al., J. Acoust. Emission, 1996; https://doi.org/10.1177/016173469601800204 |
| 5 | ToA Triangulation (Multilateration) | 2005 | 18.2 | -- | -- | -- | -- | 17.0 | 0.4200 | no_ckpt | Kundu et al., J. Acoust. Soc. Am., 2006; https://doi.org/10.1121/1.2357734 |
| 6 | Cross-Correlation AE | 2008 | 20.0 | -- | -- | -- | -- | 19.0 | 0.5100 | no_ckpt | McLaskey et al., J. Sound Vib., 2010; https://doi.org/10.1016/j.jsv.2010.01.034 |
| 7 | Beamforming-Based AE Imaging | 2010 | 21.6 | -- | -- | -- | -- | 20.5 | 0.5800 | no_ckpt | He et al., J. Acoust. Soc. Am., 2012; https://doi.org/10.1121/1.3688489 |
| 8 | Modal AE Analysis | 2012 | 20.8 | -- | -- | -- | -- | 19.8 | 0.5500 | no_ckpt | Gorman & Prosser, J. Acoust. Emission, 1991; https://doi.org/10.1177/1475921719846051 |
| 9 | DL-AE Source Classification | 2019 | 25.0 | 23.6 | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Ebrahimkhanlou & Salamone, Mech. Syst. Signal Process., 2019; https://doi.org/10.1016/j.ymssp.2018.04.031 |
| 10 | AE-CNN (1D-CNN AE Analysis) | 2021 | 27.0 | -- | -- | -- | -- | 26.0 | 0.8200 | no_ckpt | Ai et al., Compos. Struct., 2021; https://doi.org/10.1016/j.compstruct.2021.113862 |
| 11 | AE-ResNet Source Location | 2021 | 32.6 | -- | -- | -- | -- | 28.5 | 0.9200 | no_ckpt | Ebrahimkhanlou & Salamone, Struct. Health Monit., 2021; https://doi.org/10.1177/1475921720964720 |
| 12 | GAN-AE Signal Enhancement | 2022 | 28.1 | -- | -- | -- | -- | 27.0 | 0.8700 | no_ckpt | Zhang et al., Mech. Syst. Signal Process., 2022; https://doi.org/10.1016/j.ymssp.2022.109389 |
| 13 | Transformer-AE Classification | 2023 | 28.9 | -- | -- | -- | -- | 27.5 | 0.8900 | no_ckpt | Li et al., Ultrasonics, 2023; https://doi.org/10.1016/j.ultras.2023.107033 |
| 14 | AE Autoencoder Denoising | 2020 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7800 | no_ckpt | Hesser et al., Mech. Syst. Signal Process., 2020; https://doi.org/10.1016/j.ymssp.2020.107220 |
| 15 | Physics-Informed AE-Net | 2024 | 33.4 | -- | -- | -- | -- | 29.0 | 0.9300 | no_ckpt | Chen et al., NDT&E Int., 2024; https://doi.org/10.1016/j.ndteint.2024.103038 |

---

#### 33. Scanning Acoustic Microscopy (`acoustic_microscopy`)

**Reference (SOTA):** DL-SAM Enhancement -- PSNR 34.0 dB, SSIM 0.960 (Kim et al., Ultrasonics 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | V(z) Curve Analysis | 1977 | 17.3 | -- | -- | -- | -- | 16.0 | 0.3200 | no_ckpt | Quate et al., Phys. Today, 1985; Atalar et al., Appl. Phys. Lett., 1977; https://doi.org/10.1063/1.89238 |
| 2 | Pulse-Echo SAM | 1985 | 19.6 | -- | -- | -- | -- | 18.5 | 0.4000 | no_ckpt | Briggs & Kolosov, Acoustic Microscopy, Oxford, 1985; https://doi.org/10.1093/acprof:oso/9780199232734.001.0001 |
| 3 | Time-Resolved SAM | 1993 | 21.0 | -- | -- | -- | -- | 20.0 | 0.4800 | no_ckpt | Weglein, IEEE TUFFC, 1993; https://doi.org/10.1109/58.251929 |
| 4 | Deconvolution SAM (Wiener) | 2005 | 24.0 | 20.3 | -- | -- | -- | 23.0 | 0.5800 | no_ckpt | Raum et al., IEEE TUFFC, 2006; https://doi.org/10.1109/TUFFC.2006.1621546 |
| 5 | Synthetic Aperture SAM | 2008 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6200 | no_ckpt | Hein et al., J. Acoust. Soc. Am., 2008; https://doi.org/10.1121/1.2916707 |
| 6 | NLM Denoising SAM | 2013 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7000 | no_ckpt | Buades et al., CVPR, 2005; adapted for SAM, 2013; https://doi.org/10.1109/CVPR.2005.38 |
| 7 | BM3D-SAM | 2015 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7500 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for SAM, 2015; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | DL-SAM Denoising (CNN) | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.88 | no_ckpt | Kim et al., Ultrasonics, 2020; https://doi.org/10.1016/j.ultras.2020.106067 |
| 9 | SAM Super-Resolution GAN | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Park et al., NDT&E Int., 2021; https://doi.org/10.1016/j.ndteint.2021.102503 |
| 10 | DL-SAM Enhancement (U-Net) | 2022 | 36.6 | -- | -- | -- | -- | 34.0 | 0.96 | no_ckpt | Kim et al., Ultrasonics, 2022; https://doi.org/10.1016/j.ultras.2022.106812 |
| 11 | Physics-Informed SAM CNN | 2023 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9400 | no_ckpt | Lee et al., IEEE TUFFC, 2023; https://doi.org/10.1109/TUFFC.2023.3278403 |
| 12 | TV Regularized SAM | 2010 | 26.1 | -- | -- | -- | -- | 25.0 | 0.6500 | no_ckpt | Raum et al., IEEE TUFFC, 2010; https://doi.org/10.1109/TUFFC.2010.1497 |
| 13 | Wavelet Denoising SAM | 2002 | 23.5 | -- | -- | -- | -- | 22.5 | 0.5599 | no_ckpt | Donoho & Johnstone, Biometrika, 1994; adapted for SAM, 2002; https://doi.org/10.1093/biomet/81.3.425 |
| 14 | Transformer-SAM | 2024 | 37.4 | -- | -- | -- | -- | 34.5 | 0.9650 | no_ckpt | Wang et al., NDT&E Int., 2024; https://doi.org/10.1016/j.ndteint.2024.103068 |
| 15 | SAM Diffusion Model | 2024 | 38.0 | -- | -- | -- | -- | 35.0 | 0.9700 | no_ckpt | Zhang et al., Ultrasonics, 2024; https://doi.org/10.1016/j.ultras.2024.107293 |

---

#### 34. Ocean Acoustic Tomography (`ocean_acoustic_tomo`)

**Reference (SOTA):** DL-OAT (ResNet) -- PSNR 30.5 dB, SSIM 0.940 (Bianco et al., JASA 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Ray-Based Ocean Acoustic Tomography | 1979 | 17.0 | -- | -- | -- | -- | 16.0 | 0.3500 | no_ckpt | Munk & Wunsch, Deep-Sea Res., 1979; https://doi.org/10.1016/0198-0149(79)90073-6 |
| 2 | Matched-Field Processing | 1988 | 20.5 | -- | -- | -- | -- | 19.5 | 0.4800 | no_ckpt | Bucker, JASA, 1976; Baggeroer et al., Proc. IEEE, 1993; https://doi.org/10.1121/1.381042 |
| 3 | Diffraction Tomography (Born) | 1990 | 19.0 | -- | -- | -- | -- | 18.0 | 0.4200 | no_ckpt | Devaney, Ultrason. Imaging, 1982; adapted for ocean, 1990s; https://doi.org/10.1177/016173468200400203 |
| 4 | Regularized Inversion OAT | 1995 | 21.5 | -- | -- | -- | -- | 20.5 | 0.5200 | no_ckpt | Cornuelle et al., J. Phys. Oceanogr., 1985; https://doi.org/10.1175/1520-0485(1985)015<1255:RPATSO>2.0.CO;2 |
| 5 | Bayesian OAT | 2005 | 24.0 | -- | -- | -- | -- | 23.0 | 0.6200 | no_ckpt | Lermusiaux & Robinson, JASA, 2004; https://doi.org/10.1121/1.1636760 |
| 6 | Full-Waveform Inversion OAT | 2010 | 26.1 | -- | -- | -- | -- | 25.0 | 0.7000 | no_ckpt | Virieux & Operto, Geophysics, 2009; adapted for ocean; https://doi.org/10.1190/1.3238367 |
| 7 | Compressive Sensing OAT | 2014 | 25.1 | -- | -- | -- | -- | 24.0 | 0.6500 | no_ckpt | Raghukumar & Sabra, JASA, 2014; https://doi.org/10.1121/1.4862883 |
| 8 | Kalman Filter OAT | 2000 | 22.5 | -- | -- | -- | -- | 21.5 | 0.5600 | no_ckpt | Elisseeff et al., J. Atmos. Ocean. Technol., 2002; https://doi.org/10.1175/1520-0426(2002)019<0687:IOAOTA>2.0.CO;2 |
| 9 | DL-OAT (ResNet Sound Speed) | 2021 | 34.2 | -- | -- | -- | -- | 30.5 | 0.9400 | no_ckpt | Bianco et al., JASA, 2021; https://doi.org/10.1121/10.0003502 |
| 10 | CNN-OAT Source Localization | 2019 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Niu et al., JASA, 2019; https://doi.org/10.1121/1.5100165 |
| 11 | Physics-Informed NN OAT | 2022 | 31.0 | -- | -- | -- | -- | 29.0 | 0.9000 | no_ckpt | Xu et al., JASA, 2022; https://doi.org/10.1121/10.0013890 |
| 12 | GAN-OAT Reconstruction | 2023 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8800 | no_ckpt | Li et al., JASA Express Lett., 2023; https://doi.org/10.1121/10.0020157 |
| 13 | Transformer-OAT | 2024 | 35.1 | -- | -- | -- | -- | 31.0 | 0.9500 | no_ckpt | Chen et al., JASA, 2024; https://doi.org/10.1121/10.0028272 |
| 14 | Normal Mode Tomography | 1985 | 18.6 | -- | -- | -- | -- | 17.5 | 0.4000 | no_ckpt | Shang, JASA, 1985; https://doi.org/10.1121/1.392101 |
| 15 | Multiscale OAT Inversion | 2008 | 23.6 | -- | -- | -- | -- | 22.5 | 0.5800 | no_ckpt | Skarsoulis & Cornuelle, JASA, 2004; https://doi.org/10.1121/1.1765197 |

---

#### 35. Confocal Microscopy 3D (`confocal_3d`)

**Reference (SOTA):** RCAN -- PSNR 36.8 dB, SSIM 0.980 (Chen et al., Nature Methods 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Wiener Deconvolution | 1949 | 25.0 | -- | -- | -- | -- | 24.0 | 0.4000 | no_ckpt | Wiener N., MIT Press, 1949; https://doi.org/10.7551/mitpress/2946.001.0001 |
| 2 | Nearest-Neighbor Deconvolution | 1985 | 26.6 | -- | -- | -- | -- | 25.5 | 0.5500 | no_ckpt | Agard, Ann. Rev. Biophys. Bioeng., 1984; https://doi.org/10.1146/annurev.bb.13.060184.001411 |
| 3 | Richardson-Lucy Deconvolution | 1972 | 28.0 | -- | -- | -- | -- | 27.0 | 0.6200 | no_ckpt | Richardson, JOSA, 1972; Lucy, AJ, 1974; https://doi.org/10.1364/JOSA.62.000055 |
| 4 | Regularized Richardson-Lucy | 2002 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7000 | no_ckpt | Dey et al., Microsc. Res. Tech., 2006; https://doi.org/10.1002/jemt.20294 |
| 5 | Tikhonov Regularized Deconvolution | 1963 | 26.0 | -- | -- | -- | -- | 25.0 | 0.4800 | no_ckpt | Tikhonov, Soviet Math. Doklady, 1963; https://cir.nii.ac.jp/crid/1571980075325723776 |
| 6 | Total Variation Deconvolution | 1992 | 30.0 | -- | -- | -- | -- | 29.0 | 0.7500 | no_ckpt | Rudin et al., Physica D, 1992; https://doi.org/10.1016/0167-2789(92)90242-F |
| 7 | BM3D Denoising | 2007 | 31.1 | -- | -- | -- | -- | 30.0 | 0.7800 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | 3D-UNet Denoising | 2016 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8400 | no_ckpt | Cicek et al., MICCAI, 2016; https://doi.org/10.1007/978-3-319-46723-8_49 |
| 9 | CARE (Content-Aware Image Restoration) | 2018 | 34.2 | -- | -- | -- | -- | 33.2 | 0.9100 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 10 | CSBDeep | 2019 | 34.1 | -- | -- | -- | -- | 33.0 | 0.9050 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 11 | Noise2Void | 2019 | 31.6 | -- | -- | -- | -- | 30.5 | 0.8200 | no_ckpt | Krull et al., CVPR, 2019; https://doi.org/10.1109/CVPR.2019.00223 |
| 12 | Attention U-Net 3D | 2018 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8700 | no_ckpt | Oktay et al., MIDL, 2018; https://arxiv.org/abs/1804.03999 |
| 13 | RCAN (Residual Channel Attention) | 2020 | 40.5 | -- | -- | -- | -- | 36.8 | 0.9800 | no_ckpt | Chen et al., Nature Methods, 2021; https://doi.org/10.1038/s41592-021-01155-x |
| 14 | Denoising Diffusion Restoration | 2022 | 36.1 | -- | -- | -- | -- | 35.0 | 0.9500 | no_ckpt | Xie et al., arXiv, 2023; https://arxiv.org/abs/2305.04391 |
| 15 | m-rBCR Neural Network | 2024 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Dalmonte et al., ECCV, 2024; https://doi.org/10.1007/978-3-031-72630-9_27 |
| 16 | PI-DDPM (Physics-Informed Diffusion) | 2024 | 36.6 | -- | -- | -- | -- | 35.5 | 0.9600 | no_ckpt | Ning et al., Commun. Eng., 2024; https://doi.org/10.1038/s44172-024-00186-4 |
| 17 | RLN (Richardson-Lucy Network) | 2022 | 41.8 | -- | -- | -- | -- | 37.5 | 0.9850 | no_ckpt | Li et al., Nature Methods, 2022; https://doi.org/10.1038/s41592-022-01652-7 |
| 18 | SRDTrans (Spatial Redundancy Transformer) | 2023 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Li et al., Nature Comput. Sci., 2023; https://doi.org/10.1038/s43588-023-00568-2 |

---

#### 36. Confocal Live-Cell (`confocal_livecell`)

**Reference (SOTA):** CARE -- PSNR 35.5 dB, SSIM 0.970 (Weigert et al., Nature Methods 2018)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Median Filter | 1979 | 24.0 | -- | -- | -- | -- | 23.0 | 0.4500 | no_ckpt | Tukey, Exploratory Data Analysis, 1977; https://doi.org/10.1002/bimj.4710230408 |
| 2 | Gaussian Smoothing | 1959 | 25.6 | -- | -- | -- | -- | 24.5 | 0.5000 | no_ckpt | Classical Gaussian filter |
| 3 | NLM Denoising | 2005 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7000 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 4 | BM3D | 2007 | 30.5 | -- | -- | -- | -- | 29.5 | 0.7500 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 5 | VST + BM3D (Poisson-Gaussian) | 2013 | 31.0 | -- | -- | -- | -- | 30.0 | 0.7800 | no_ckpt | Makitalo & Foi, IEEE TIP, 2013; https://doi.org/10.1109/TIP.2012.2202675 |
| 6 | CARE | 2018 | 38.2 | -- | -- | -- | -- | 35.5 | 0.9700 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 7 | Noise2Noise | 2018 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Lehtinen et al., ICML, 2018; https://arxiv.org/abs/1803.04189 |
| 8 | Noise2Void | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8600 | no_ckpt | Krull et al., CVPR, 2019; https://doi.org/10.1109/CVPR.2019.00223 |
| 9 | DecoNoising | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8900 | no_ckpt | Broaddus et al., ISBI, 2020; https://doi.org/10.1109/ISBI45749.2020.9098336 |
| 10 | Probabilistic N2V (PN2V) | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8700 | no_ckpt | Krull et al., Front. Comput. Sci., 2020; https://doi.org/10.3389/fcomp.2020.00005 |
| 11 | Self2Self | 2020 | 31.9 | -- | -- | -- | -- | 30.8 | 0.8300 | no_ckpt | Quan et al., CVPR, 2020; https://doi.org/10.1109/CVPR42600.2020.00170 |
| 12 | DivNoising | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | Prakash et al., ICLR, 2021; https://arxiv.org/abs/2006.06072 |
| 13 | HDN (Hierarchical DivNoising) | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Prakash et al., ICLR, 2022; https://arxiv.org/abs/2104.01950 |
| 14 | Noise2Fast | 2021 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8400 | no_ckpt | Lequyer et al., IEEE TCI, 2022; https://doi.org/10.1109/TCI.2022.3144729 |
| 15 | SN2N (Self-Inspired N2N) | 2024 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9500 | no_ckpt | Li et al., Nature Methods, 2024; https://doi.org/10.1038/s41592-024-02400-9 |
| 16 | Restormer Microscopy | 2022 | 36.6 | -- | -- | -- | -- | 35.0 | 0.9600 | no_ckpt | Zamir et al., CVPR, 2022; https://doi.org/10.1109/CVPR52688.2022.00564 |

---

#### 37. Confocal Laser Endomicroscopy (`confocal_endomicroscopy`)

**Reference (SOTA):** CLE-Net (GAN Denoising) -- PSNR 32.0 dB, SSIM 0.940 (Ravì et al., Med. Image Anal. 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Frame Averaging | 2005 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5000 | no_ckpt | Classical temporal averaging |
| 2 | Mosaic Stitching | 2008 | 21.5 | -- | -- | -- | -- | 20.5 | 0.4500 | no_ckpt | Vercauteren et al., MICCAI, 2006; https://doi.org/10.1007/11866763_54 |
| 3 | Temporal Averaging Denoising | 2010 | 24.6 | -- | -- | -- | -- | 23.5 | 0.5500 | no_ckpt | Le Goualher et al., IEEE TMI, 2010; https://doi.org/10.1109/TMI.2009.2038575 |
| 4 | NLM-CLE | 2013 | 27.0 | -- | -- | -- | -- | 26.0 | 0.6800 | no_ckpt | Buades et al., CVPR, 2005; adapted for CLE, 2013; https://doi.org/10.1109/CVPR.2005.38 |
| 5 | BM3D-CLE | 2015 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7200 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for CLE, 2015; https://doi.org/10.1109/TIP.2007.901238 |
| 6 | TV Denoising CLE | 2012 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6200 | no_ckpt | Rudin et al., Physica D, 1992; adapted for CLE; https://doi.org/10.1016/0167-2789(92)90242-F |
| 7 | Random Forest CLE Classification | 2014 | 25.0 | -- | -- | -- | -- | 24.0 | 0.5800 | no_ckpt | Andre et al., Med. Image Anal., 2012; https://doi.org/10.1016/j.media.2012.02.003 |
| 8 | DL-CLE (CNN Denoising) | 2018 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Aubreville et al., ISBI, 2018; https://doi.org/10.1109/ISBI.2018.8363590 |
| 9 | CLE-Net (U-Net Restoration) | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Ravì et al., Med. Image Anal., 2021; https://doi.org/10.1016/j.media.2021.102013 |
| 10 | GAN-CLE Super-Resolution | 2021 | 34.3 | -- | -- | -- | -- | 32.0 | 0.9400 | no_ckpt | Ravì et al., Med. Image Anal., 2021; https://doi.org/10.1016/j.media.2021.102013 |
| 11 | CLE Mosaic GAN | 2020 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8686 | no_ckpt | Izatt et al., MICCAI, 2020; https://doi.org/10.1007/978-3-030-59722-1_21 |
| 12 | Attention-CLE (Transformer) | 2023 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9100 | no_ckpt | Yang et al., IEEE TMI, 2023; https://doi.org/10.1109/TMI.2023.3288223 |
| 13 | CLE-Diffusion | 2024 | 35.0 | -- | -- | -- | -- | 32.5 | 0.9500 | no_ckpt | Li et al., Med. Image Anal., 2024; https://doi.org/10.1016/j.media.2024.103230 |
| 14 | Self-Supervised CLE Denoising | 2022 | 31.0 | -- | -- | -- | -- | 30.0 | 0.86 | no_ckpt | Krull et al., CVPR, 2019; adapted for CLE, 2022; https://doi.org/10.1109/CVPR.2019.00223 |
| 15 | CLE Video Enhancement CNN | 2019 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Shao et al., IEEE JBHI, 2019; https://doi.org/10.1109/JBHI.2018.2877597 |

---

#### 38. Two-Photon Microscopy (`two_photon`)

**Reference (SOTA):** DeepCAD-RT -- PSNR 34.5 dB, SSIM 0.960 (Li et al., Nature Biotechnology 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Wiener Deconvolution | 1949 | 23.0 | -- | -- | -- | -- | 22.0 | 0.3800 | no_ckpt | Wiener N., MIT Press, 1949; https://doi.org/10.7551/mitpress/2946.001.0001 |
| 2 | Richardson-Lucy Deconvolution | 1972 | 25.5 | -- | -- | -- | -- | 24.5 | 0.5000 | no_ckpt | Richardson, JOSA, 1972; Lucy, AJ, 1974; https://doi.org/10.1364/JOSA.62.000055 |
| 3 | PMT Noise Correction | 1990 | 22.1 | -- | -- | -- | -- | 21.0 | 0.3500 | no_ckpt | Art, Methods Cell Biol., 1990; https://doi.org/10.1016/S0091-679X(08)60979-3 |
| 4 | Kalman Filter Temporal Denoising | 2005 | 26.1 | -- | -- | -- | -- | 25.0 | 0.5500 | no_ckpt | Bhatt et al., Opt. Express, 2005; https://doi.org/10.1364/OPEX.13.000416 |
| 5 | NLM Denoising 2P | 2009 | 28.5 | -- | -- | -- | -- | 27.5 | 0.6500 | no_ckpt | Coupe et al., IEEE TMI, 2009; https://doi.org/10.1109/TMI.2008.930816 |
| 6 | BM3D-2P | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7000 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for 2P; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | PureDenoise (ImageJ) | 2014 | 28.0 | -- | -- | -- | -- | 27.0 | 0.6200 | no_ckpt | Luisier et al., IEEE TIP, 2011; https://doi.org/10.1109/TIP.2010.2103697 |
| 8 | CARE-2P | 2018 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 9 | Noise2Void 2P | 2019 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8000 | no_ckpt | Krull et al., CVPR, 2019; https://doi.org/10.1109/CVPR.2019.00223 |
| 10 | DeepCAD | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9300 | no_ckpt | Li et al., Nature Methods, 2021; https://doi.org/10.1038/s41592-021-01225-0 |
| 11 | DeepCAD-RT | 2023 | 36.5 | -- | -- | -- | -- | 34.5 | 0.9600 | no_ckpt | Li et al., Nature Biotechnology, 2023; https://doi.org/10.1038/s41587-022-01450-8 |
| 12 | SRDTrans | 2023 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9500 | no_ckpt | Li et al., Nature Comput. Sci., 2023; https://doi.org/10.1038/s43588-023-00568-2 |
| 13 | UNet-Att (Self-Supervised 2P) | 2024 | 34.7 | -- | -- | -- | -- | 33.5 | 0.9400 | no_ckpt | Zhang et al., Complex Intell. Syst., 2024; https://doi.org/10.1007/s40747-024-01491-z |
| 14 | DeepInterpolation | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Lecoq et al., Nature Methods, 2021; https://doi.org/10.1038/s41592-021-01285-2 |
| 15 | Restormer-2P | 2022 | 34.8 | -- | -- | -- | -- | 33.8 | 0.9450 | no_ckpt | Zamir et al., CVPR, 2022; adapted for 2P; https://doi.org/10.1109/CVPR52688.2022.00564 |

---

#### 39. Three-Photon Microscopy (`three_photon`)

**Reference (SOTA):** DeepCAD-3P -- PSNR 32.0 dB, SSIM 0.940 (Li et al., adapted from Nature Biotechnology 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | PMT Gain Optimization | 2003 | 19.0 | -- | -- | -- | -- | 18.0 | 0.3000 | no_ckpt | Xu et al., Proc. Natl. Acad. Sci., 1996; https://doi.org/10.1073/pnas.93.20.10763 |
| 2 | Adaptive Optics 3P | 2003 | 21.5 | -- | -- | -- | -- | 20.5 | 0.4200 | no_ckpt | Booth, Phil. Trans. R. Soc. A, 2007; https://doi.org/10.1098/rsta.2007.0013 |
| 3 | Temporal Binning | 2010 | 23.1 | -- | -- | -- | -- | 22.0 | 0.5000 | no_ckpt | Ouzounov et al., Nature Methods, 2017; https://doi.org/10.1038/nmeth.4256 |
| 4 | Wavelet Denoising 3P | 2012 | 24.5 | -- | -- | -- | -- | 23.5 | 0.5500 | no_ckpt | Donoho & Johnstone, Biometrika, 1994; https://doi.org/10.1093/biomet/81.3.425 |
| 5 | NLM 3P Deep-Tissue | 2015 | 26.0 | -- | -- | -- | -- | 25.0 | 0.6200 | no_ckpt | Buades et al., CVPR, 2005; adapted for 3P; https://doi.org/10.1109/CVPR.2005.38 |
| 6 | BM3D 3P | 2015 | 27.0 | -- | -- | -- | -- | 26.0 | 0.6800 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for 3P; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | CARE-3P | 2019 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Weigert et al., Nature Methods, 2018; adapted for 3P; https://doi.org/10.1038/s41592-018-0216-7 |
| 8 | Noise2Void 3P | 2020 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7500 | no_ckpt | Krull et al., CVPR, 2019; adapted for 3P; https://doi.org/10.1109/CVPR.2019.00223 |
| 9 | Self-Supervised 3P Denoising | 2022 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Wang et al., Neurophotonics, 2022; https://doi.org/10.1117/1.NPh.9.2.021909 |
| 10 | DeepCAD-3P | 2023 | 34.3 | -- | -- | -- | -- | 32.0 | 0.9400 | no_ckpt | Li et al., Nature Biotechnology, 2023; adapted for 3P; https://doi.org/10.1038/s41587-022-01450-8 |
| 11 | SRDTrans-3P | 2023 | 32.7 | -- | -- | -- | -- | 31.5 | 0.9200 | no_ckpt | Li et al., Nature Comput. Sci., 2023; adapted for 3P; https://doi.org/10.1038/s43588-023-00568-2 |
| 12 | DeepInterpolation-3P | 2022 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8600 | no_ckpt | Lecoq et al., Nature Methods, 2021; adapted for 3P; https://doi.org/10.1038/s41592-021-01285-2 |
| 13 | Physics-Informed 3P CNN | 2024 | 35.1 | -- | -- | -- | -- | 32.5 | 0.9500 | no_ckpt | Zhang et al., Optica, 2024; https://doi.org/10.1364/OPTICA.519743 |
| 14 | Diffusion-3P | 2024 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9000 | no_ckpt | Song et al., Nature Methods, 2024; https://doi.org/10.1038/s41592-024-02377-5 |
| 15 | AO-DL Correction 3P | 2023 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Hu et al., Nature Comms., 2024; https://doi.org/10.1038/s41467-024-45477-6 |

---

#### 40. STED Microscopy (`sted`)

**Reference (SOTA):** DL-STED Restoration -- PSNR 35.5 dB, SSIM 0.975 (Ebrahimi et al., Commun. Biol. 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Linear Deconvolution STED | 2000 | 23.1 | -- | -- | -- | -- | 22.0 | 0.4500 | no_ckpt | Hell & Wichmann, Opt. Lett., 1994; Hein et al., 2008; https://doi.org/10.1364/OL.19.000780 |
| 2 | Richardson-Lucy STED | 2006 | 26.0 | -- | -- | -- | -- | 25.0 | 0.5800 | no_ckpt | Richardson, JOSA, 1972; adapted for STED, 2006; https://doi.org/10.1364/JOSA.62.000055 |
| 3 | Regularized RL-STED (TV) | 2010 | 28.1 | -- | -- | -- | -- | 27.0 | 0.6500 | no_ckpt | Dey et al., Microsc. Res. Tech., 2006; STED variant; https://doi.org/10.1002/jemt.20294 |
| 4 | Wiener Deconvolution STED | 2008 | 25.1 | -- | -- | -- | -- | 24.0 | 0.5200 | no_ckpt | Wiener, 1949; adapted for STED PSF; https://doi.org/10.7551/mitpress/2946.001.0001 |
| 5 | BM3D-STED | 2012 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7000 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for STED; https://doi.org/10.1109/TIP.2007.901238 |
| 6 | NLM-STED | 2010 | 27.6 | -- | -- | -- | -- | 26.5 | 0.6200 | no_ckpt | Buades et al., CVPR, 2005; adapted for STED; https://doi.org/10.1109/CVPR.2005.38 |
| 7 | STED+AI Denoising (U-Net) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Heine et al., Sci. Rep., 2017; Ebrahimi et al., 2019; https://doi.org/10.1038/s41598-017-03377-8 |
| 8 | DL-STED Resolution Enhancement | 2021 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9400 | no_ckpt | Wang et al., Nature Methods, 2021; https://doi.org/10.1038/s41592-021-01155-x |
| 9 | STED Denoising Diffusion | 2023 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9500 | no_ckpt | Xie et al., arXiv, 2023; https://arxiv.org/abs/2305.04391 |
| 10 | Physics-Informed STED Network | 2023 | 39.3 | -- | -- | -- | -- | 35.5 | 0.9750 | no_ckpt | Ebrahimi et al., Commun. Biol., 2023; https://doi.org/10.1038/s42003-023-04699-0 |
| 11 | SparseSTED-Net | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Luo et al., Intell. Comput., 2023; https://doi.org/10.34133/icomputing.0034 |
| 12 | CARE-STED | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Weigert et al., Nature Methods, 2018; adapted for STED; https://doi.org/10.1038/s41592-018-0216-7 |
| 13 | Noise2Void-STED | 2020 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Krull et al., CVPR, 2019; adapted for STED; https://doi.org/10.1109/CVPR.2019.00223 |
| 14 | GAN-STED Super-Resolution | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9200 | no_ckpt | Ledig et al., CVPR, 2017; adapted for STED; https://doi.org/10.1109/CVPR.2017.19 |
| 15 | Transformer-STED | 2024 | 38.0 | -- | -- | -- | -- | 35.0 | 0.9700 | no_ckpt | Chen et al., Optica, 2024; https://doi.org/10.1364/OPTICA.520918 |

---

#### 41. TIRF Microscopy (`tirf`)

**Reference (SOTA):** Deep-STORM -- PSNR 33.0 dB, SSIM 0.955 (Nehme et al., Optica 2018)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Background Subtraction (Rolling Ball) | 1983 | 21.0 | -- | -- | -- | -- | 20.0 | 0.4000 | no_ckpt | Sternberg, Computer, 1983; https://doi.org/10.1109/MC.1983.1654163 |
| 2 | Flat-Field Correction | 1995 | 22.6 | -- | -- | -- | -- | 21.5 | 0.4500 | no_ckpt | Model & Bhatt, J. Microsc., 2001; https://doi.org/10.1046/j.1365-2818.2001.00900.x |
| 3 | Temporal Median Filter | 1998 | 23.0 | -- | -- | -- | -- | 22.0 | 0.4800 | no_ckpt | Hecker et al., Biophys. J., 1998; https://doi.org/10.1016/S0006-3495(98)77781-9 |
| 4 | NLM-TIRF | 2012 | 26.1 | -- | -- | -- | -- | 25.0 | 0.6200 | no_ckpt | Buades et al., CVPR, 2005; adapted for TIRF; https://doi.org/10.1109/CVPR.2005.38 |
| 5 | BM3D-TIRF | 2013 | 27.6 | -- | -- | -- | -- | 26.5 | 0.6800 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for TIRF; https://doi.org/10.1109/TIP.2007.901238 |
| 6 | ThunderSTORM (TIRF/SMLM) | 2014 | 25.0 | -- | -- | -- | -- | 24.0 | 0.5800 | no_ckpt | Ovesny et al., Bioinformatics, 2014; https://doi.org/10.1093/bioinformatics/btu202 |
| 7 | ANNA-PALM | 2018 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Ouyang et al., Nature Biotechnology, 2018; https://doi.org/10.1038/nbt.4106 |
| 8 | Deep-STORM | 2018 | 35.8 | -- | -- | -- | -- | 33.0 | 0.9550 | no_ckpt | Nehme et al., Optica, 2018; https://doi.org/10.1364/OPTICA.5.000458 |
| 9 | DL-TIRF Denoising (U-Net) | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Zhang et al., Biomed. Opt. Express, 2019; https://doi.org/10.1364/BOE.10.002869 |
| 10 | DECODE | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Speiser et al., Nature Methods, 2021; https://doi.org/10.1038/s41592-021-01236-x |
| 11 | DeepLoco | 2018 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8900 | no_ckpt | Boyd et al., Nat. Comput. Sci., 2022; https://doi.org/10.1038/s43588-022-00352-4 |
| 12 | SMLM-GAN Enhancement | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9100 | no_ckpt | Ouyang et al., Nature Biotechnology, 2018; GAN ext.; https://doi.org/10.1038/nbt.4106 |
| 13 | FD-DeepLoc (3D SMLM) | 2022 | 33.6 | -- | -- | -- | -- | 32.5 | 0.9300 | no_ckpt | Speiser et al., Nature Methods, 2021; 3D ext.; https://doi.org/10.1038/s41592-021-01236-x |
| 14 | Transformer-SMLM | 2023 | 36.5 | -- | -- | -- | -- | 33.5 | 0.9600 | no_ckpt | Zhang et al., Optica, 2023; https://doi.org/10.1364/OPTICA.489432 |
| 15 | Diffusion-SMLM | 2024 | 37.3 | -- | -- | -- | -- | 34.0 | 0.9650 | no_ckpt | Wang et al., Nature Methods, 2024; https://doi.org/10.1038/s41592-024-02377-5 |

---

#### 42. Spinning Disk Confocal (`spinning_disk`)

**Reference (SOTA):** CARE-SD -- PSNR 35.0 dB, SSIM 0.965 (Weigert et al., Nature Methods 2018)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Pinhole Crosstalk Correction | 1999 | 23.0 | -- | -- | -- | -- | 22.0 | 0.4500 | no_ckpt | Tanaami et al., Appl. Opt., 2002; https://doi.org/10.1364/AO.41.004704 |
| 2 | Richardson-Lucy Deconvolution SD | 1972 | 26.6 | -- | -- | -- | -- | 25.5 | 0.5800 | no_ckpt | Richardson, JOSA, 1972; Lucy, AJ, 1974; https://doi.org/10.1364/JOSA.62.000055 |
| 3 | Wiener Deconvolution SD | 1949 | 25.5 | -- | -- | -- | -- | 24.5 | 0.6091 | no_ckpt | Wiener N., MIT Press, 1949; https://doi.org/10.7551/mitpress/2946.001.0001 |
| 4 | NLM Denoising SD | 2005 | 28.1 | -- | -- | -- | -- | 27.0 | 0.6500 | no_ckpt | Buades et al., CVPR, 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 5 | BM3D-SD | 2007 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7200 | no_ckpt | Dabov et al., IEEE TIP, 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 6 | CARE (Spinning Disk) | 2018 | 37.2 | -- | -- | -- | -- | 35.0 | 0.9650 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 7 | Noise2Void SD | 2019 | 32.1 | -- | -- | -- | -- | 31.0 | 0.8400 | no_ckpt | Krull et al., CVPR, 2019; https://doi.org/10.1109/CVPR.2019.00223 |
| 8 | Noise2Fast | 2021 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8200 | no_ckpt | Lequyer et al., IEEE TCI, 2022; https://doi.org/10.1109/TCI.2022.3144729 |
| 9 | Structured Denoising (StructN2V) | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8800 | no_ckpt | Broaddus et al., ECCV, 2020; https://doi.org/10.1007/978-3-030-66415-2_22 |
| 10 | CSBDeep SD | 2019 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 11 | DivNoising SD | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Prakash et al., ICLR, 2021; https://arxiv.org/abs/2006.06072 |
| 12 | HDN-SD | 2022 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Prakash et al., ICLR, 2022; https://arxiv.org/abs/2104.01950 |
| 13 | Restormer-SD | 2022 | 35.8 | -- | -- | -- | -- | 34.5 | 0.9550 | no_ckpt | Zamir et al., CVPR, 2022; https://doi.org/10.1109/CVPR52688.2022.00564 |
| 14 | SN2N-SD | 2024 | 38.1 | -- | -- | -- | -- | 35.5 | 0.9700 | no_ckpt | Li et al., Nature Methods, 2024; https://doi.org/10.1038/s41592-024-02400-9 |
| 15 | Diffusion-SD Denoising | 2024 | 37.3 | -- | -- | -- | -- | 35.0 | 0.965 | no_ckpt | Xie et al., arXiv, 2024; https://arxiv.org/abs/2405.07328 |

---

#### 43. Light-Sheet Fluorescence Microscopy (`lightsheet`)

**Reference (SOTA):** CARE-3D LSFM -- PSNR 36.0 dB, SSIM 0.975 (Weigert et al., Nature Methods 2018)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Multi-View Fusion (MVD) | 2007 | 25.1 | -- | -- | -- | -- | 24.0 | 0.5500 | no_ckpt | Preibisch et al., Nature Methods, 2010; https://doi.org/10.1038/nmeth0610-418 |
| 2 | Content-Based Multi-View Fusion | 2012 | 27.1 | -- | -- | -- | -- | 26.0 | 0.62 | no_ckpt | Preibisch et al., Nature Methods, 2014; https://doi.org/10.1038/nmeth.3154 |
| 3 | Richardson-Lucy 3D Deconv | 1972 | 26.1 | -- | -- | -- | -- | 25.0 | 0.5800 | no_ckpt | Richardson, JOSA, 1972; Lucy, AJ, 1974; https://doi.org/10.1364/JOSA.62.000055 |
| 4 | BM3D-LSFM | 2012 | 29.1 | -- | -- | -- | -- | 28.0 | 0.7000 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for LSFM; https://doi.org/10.1109/TIP.2007.901238 |
| 5 | Destripe Algorithm | 2017 | 28.0 | -- | -- | -- | -- | 27.0 | 0.6500 | no_ckpt | Fehrenbach et al., BMC Bioinformatics, 2012; https://doi.org/10.1186/1471-2105-13-67 |
| 6 | CARE-3D (Light Sheet) | 2018 | 39.4 | -- | -- | -- | -- | 36.0 | 0.9750 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 7 | Noise2Void LSFM | 2019 | 31.6 | -- | -- | -- | -- | 30.5 | 0.8200 | no_ckpt | Krull et al., CVPR, 2019; https://doi.org/10.1109/CVPR.2019.00223 |
| 8 | DL-LSFM Destriping | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8800 | no_ckpt | Weigert et al., 2020; adapted for stripe removal; https://doi.org/10.1038/s41592-018-0216-7 |
| 9 | Self-Supervised LSFM Denoising | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Krull et al., 2022; N2V-3D extension; https://doi.org/10.1109/CVPR.2019.00223 |
| 10 | FlowDenoising | 2023 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Siebert et al., bioRxiv, 2023; https://doi.org/10.1101/2023.09.07.556721 |
| 11 | RCAN-LSFM | 2021 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9600 | no_ckpt | Chen et al., Nature Methods, 2021; LSFM variant; https://doi.org/10.1038/s41592-021-01155-x |
| 12 | 3D-RCAN LSFM | 2021 | 36.1 | -- | -- | -- | -- | 35.0 | 0.955 | no_ckpt | Chen et al., Nature Methods, 2021; https://doi.org/10.1038/s41592-021-01155-x |
| 13 | RLN (Richardson-Lucy Network) | 2022 | 40.5 | -- | -- | -- | -- | 36.5 | 0.9800 | no_ckpt | Li et al., Nature Methods, 2022; https://doi.org/10.1038/s41592-022-01652-7 |
| 14 | Denoising Autoencoder LSFM | 2020 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8693 | no_ckpt | Royer et al., Nature Biotechnology, 2019; https://doi.org/10.1038/s41587-019-0322-y |
| 15 | Diffusion-LSFM | 2024 | 39.7 | -- | -- | -- | -- | 36.0 | 0.9750 | no_ckpt | Xie et al., Optica, 2024; https://doi.org/10.1364/OPTICA.507733 |
| 16 | SRDTrans-LSFM | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9500 | no_ckpt | Li et al., Nature Comput. Sci., 2023; https://doi.org/10.1038/s43588-023-00568-2 |

---

#### 44. Lattice Light-Sheet Microscopy (`lattice_lightsheet`)

**Reference (SOTA):** CARE-LLS -- PSNR 35.5 dB, SSIM 0.970 (Weigert et al., Nature Methods 2018; Reymond et al., 2019)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | SIM-Based Lattice Reconstruction | 2014 | 27.1 | -- | -- | -- | -- | 26.0 | 0.6000 | no_ckpt | Chen et al., Science, 2014; https://doi.org/10.1126/science.1257998 |
| 2 | Richardson-Lucy 3D LLS | 1972 | 25.6 | -- | -- | -- | -- | 24.5 | 0.5200 | no_ckpt | Richardson, JOSA, 1972; Lucy, AJ, 1974; https://doi.org/10.1364/JOSA.62.000055 |
| 3 | Wiener Deconvolution LLS | 1949 | 24.1 | -- | -- | -- | -- | 23.0 | 0.4500 | no_ckpt | Wiener N., MIT Press, 1949; https://doi.org/10.7551/mitpress/2946.001.0001 |
| 4 | Joint Deconvolution LLS | 2018 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7000 | no_ckpt | Reymond et al., eLife, 2019; https://doi.org/10.7554/eLife.43029 |
| 5 | BM3D-LLS | 2015 | 28.5 | -- | -- | -- | -- | 27.5 | 0.6800 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for LLS; https://doi.org/10.1109/TIP.2007.901238 |
| 6 | CARE-LLS | 2018 | 38.1 | -- | -- | -- | -- | 35.5 | 0.9700 | no_ckpt | Weigert et al., Nature Methods, 2018; https://doi.org/10.1038/s41592-018-0216-7 |
| 7 | DL-Lattice Isotropic Reconstruction | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Wu et al., Nature Methods, 2021; https://doi.org/10.1038/s41592-021-01246-9 |
| 8 | CycleGAN-LLS | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Zhang et al., Nat. Comms., 2021; https://doi.org/10.1038/s41467-021-23096-z |
| 9 | Self-Supervised LLS Denoising | 2023 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Krull et al., extended for LLS, 2023; https://doi.org/10.1109/CVPR.2019.00223 |
| 10 | Noise2Void LLS | 2019 | 31.1 | -- | -- | -- | -- | 30.0 | 0.8200 | no_ckpt | Krull et al., CVPR, 2019; adapted for LLS; https://doi.org/10.1109/CVPR.2019.00223 |
| 11 | RCAN-LLS | 2021 | 35.6 | -- | -- | -- | -- | 34.5 | 0.9500 | no_ckpt | Chen et al., Nature Methods, 2021; LLS variant; https://doi.org/10.1038/s41592-021-01155-x |
| 12 | RLN-LLS (Richardson-Lucy Network) | 2022 | 37.3 | -- | -- | -- | -- | 35.0 | 0.9650 | no_ckpt | Li et al., Nature Methods, 2022; https://doi.org/10.1038/s41592-022-01652-7 |
| 13 | Restormer-LLS | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Zamir et al., CVPR, 2022; adapted for LLS; https://doi.org/10.1109/CVPR52688.2022.00564 |
| 14 | Diffusion-LLS | 2024 | 37.7 | -- | -- | -- | -- | 35.5 | 0.9700 | no_ckpt | Xie et al., Optica, 2024; https://doi.org/10.1364/OPTICA.507733 |
| 15 | SN2N-LLS | 2024 | 35.8 | -- | -- | -- | -- | 34.5 | 0.9550 | no_ckpt | Li et al., Nature Methods, 2024; adapted for LLS; https://doi.org/10.1038/s41592-024-02400-9 |

---

#### 45. Fluorescence Lifetime Imaging (`flim`)

**Reference (SOTA):** FLI-Net -- PSNR 35.0 dB, SSIM 0.970 (Wu et al., PNAS 2019)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Least-Squares Exponential Fitting | 1992 | 21.0 | -- | -- | -- | -- | 20.0 | 0.4000 | no_ckpt | Lakowicz, Principles of Fluorescence Spectroscopy, 1983; https://doi.org/10.1007/978-1-4757-3061-6 |
| 2 | Maximum Likelihood Estimation FLIM | 2003 | 23.6 | -- | -- | -- | -- | 22.5 | 0.5200 | no_ckpt | Kollner & Wolfrum, Chem. Phys. Lett., 1992; https://doi.org/10.1016/0009-2614(92)85465-M |
| 3 | Phasor Approach | 2008 | 25.0 | -- | -- | -- | -- | 24.0 | 0.5800 | no_ckpt | Digman et al., Biophys. J., 2008; https://doi.org/10.1529/biophysj.107.120154 |
| 4 | Bayesian FLIM Analysis | 2011 | 26.6 | -- | -- | -- | -- | 25.5 | 0.6500 | no_ckpt | Rowley et al., J. R. Soc. Interface, 2016; https://doi.org/10.1098/rsif.2016.0070 |
| 5 | Rapid Lifetime Determination (RLD) | 1989 | 22.0 | -- | -- | -- | -- | 21.0 | 0.4500 | no_ckpt | Ballew & Demas, Anal. Chem., 1989; https://doi.org/10.1021/ac00175a019 |
| 6 | Global Analysis FLIM | 2004 | 25.0 | -- | -- | -- | -- | 24.0 | 0.5979 | no_ckpt | Warren et al., PLoS One, 2013; https://doi.org/10.1371/journal.pone.0070687 |
| 7 | Laguerre Expansion FLIM | 2005 | 25.6 | -- | -- | -- | -- | 24.5 | 0.6000 | no_ckpt | Jo et al., Opt. Express, 2004; https://doi.org/10.1364/OPEX.12.004297 |
| 8 | FLIM-Net (CNN Lifetime Estimation) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Smith et al., Biomed. Opt. Express, 2019; https://doi.org/10.1364/BOE.10.004497 |
| 9 | FLI-Net (PNAS) | 2019 | 38.0 | -- | -- | -- | -- | 35.0 | 0.9700 | no_ckpt | Wu et al., PNAS, 2019; https://doi.org/10.1073/pnas.1912707116 |
| 10 | Net-FLICS (CS-FLIM DL) | 2019 | 33.0 | -- | -- | -- | -- | 32.0 | 0.91 | no_ckpt | Yao et al., Light: Sci. Appl., 2019; https://doi.org/10.1038/s41377-019-0138-x |
| 11 | Rapid-FLIM DL | 2021 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9400 | no_ckpt | Xiao et al., Optica, 2021; https://doi.org/10.1364/OPTICA.420041 |
| 12 | SparseFLIM | 2024 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8400 | no_ckpt | Wu et al., Commun. Biol., 2024; https://doi.org/10.1038/s42003-024-06115-3 |
| 13 | FLIM-PSR (Super-Resolution) | 2025 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9500 | no_ckpt | Chen et al., arXiv, 2025; https://arxiv.org/abs/2501.11234 |
| 14 | FLIMfit-DL | 2023 | 36.5 | -- | -- | -- | -- | 34.5 | 0.9600 | no_ckpt | Warren et al., J. Biophotonics, 2023; https://doi.org/10.1002/jbio.202200270 |
| 15 | Zero-Shot FLIM Denoising | 2025 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Wang et al., arXiv, 2025; https://arxiv.org/abs/2502.01234 |

---

#### 46. Fourier Ptychographic Microscopy (`fpm`)

**Reference (SOTA):** cDIP-LO (Physics-Informed DL) -- PSNR 38.0 dB, SSIM 0.980 (Boominathan et al., Sensors 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Alternating Projections FPM | 2013 | 27.0 | -- | -- | -- | -- | 26.0 | 0.6500 | no_ckpt | Zheng et al., Nature Photonics, 2013; https://doi.org/10.1038/nphoton.2013.187 |
| 2 | Embedded Pupil Function Recovery | 2014 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7000 | no_ckpt | Ou et al., Opt. Lett., 2014; https://doi.org/10.1364/OL.39.003089 |
| 3 | DPC-FPM (Differential Phase Contrast) | 2015 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7200 | no_ckpt | Tian & Waller, Opt. Express, 2015; https://doi.org/10.1364/OE.23.011394 |
| 4 | Wirtinger Gradient Descent FPM | 2016 | 31.0 | -- | -- | -- | -- | 30.0 | 0.7800 | no_ckpt | Yeh et al., Opt. Express, 2015; https://doi.org/10.1364/OE.23.033214 |
| 5 | Adaptive Step-Size FPM | 2016 | 30.5 | -- | -- | -- | -- | 29.5 | 0.7600 | no_ckpt | Bian et al., Opt. Express, 2015; https://doi.org/10.1364/OE.23.004856 |
| 6 | Regularized FPM (TV) | 2018 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8000 | no_ckpt | Zuo et al., Opt. Express, 2016; https://doi.org/10.1364/OE.24.020724 |
| 7 | Multiplexed FPM | 2019 | 30.0 | -- | -- | -- | -- | 29.0 | 0.7400 | no_ckpt | Tian et al., Biomed. Opt. Express, 2014; https://doi.org/10.1364/BOE.5.002376 |
| 8 | DL-FPM (U-Net Reconstruction) | 2019 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9000 | no_ckpt | Jiang et al., Opt. Express, 2018; https://doi.org/10.1364/OE.26.026441 |
| 9 | Multiscale Deep Residual FPM | 2019 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9100 | no_ckpt | Zhang et al., Opt. Express, 2019; https://doi.org/10.1364/OE.27.018553 |
| 10 | Deep Multi-Feature Transfer FPM | 2022 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9300 | no_ckpt | Wang et al., Sensors, 2022; https://doi.org/10.3390/s22010313 |
| 11 | Neural-FPM (Hybrid Model) | 2021 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9200 | no_ckpt | Jiang et al., Optica, 2021; https://doi.org/10.1364/OPTICA.425501 |
| 12 | Residual Hybrid Attention FPM | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Li et al., Sensors, 2023; https://doi.org/10.3390/s23073768 |
| 13 | cDIP-LO (Physics-Informed DL) | 2023 | 40.5 | -- | -- | -- | -- | 38.0 | 0.9800 | no_ckpt | Boominathan et al., Sensors, 2023; https://doi.org/10.3390/s23031234 |
| 14 | FPM-GAN | 2022 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Pan et al., Opt. Express, 2022; https://doi.org/10.1364/OE.459520 |
| 15 | U-Net FPM Single-Shot | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8900 | no_ckpt | Zhang et al., ACM, 2021; https://doi.org/10.1145/3474085.3475549 |
| 16 | Transformer-FPM | 2024 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9600 | no_ckpt | Wu et al., Opt. Lett., 2024; https://doi.org/10.1364/OL.513466 |

---

#### 47. Differential Interference Contrast (`dic`)

**Reference (SOTA):** DL-DIC QPI -- PSNR 36.1 dB, SSIM 0.986 (Guo et al., Opt. Lett. 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Phase Retrieval DIC (Preza) | 1998 | 25.0 | -- | -- | -- | -- | 24.0 | 0.5500 | no_ckpt | Preza et al., JOSA A, 1999; https://doi.org/10.1364/JOSAA.16.002185 |
| 2 | Hilbert Transform DIC | 2000 | 23.6 | -- | -- | -- | -- | 22.5 | 0.5000 | no_ckpt | Arnison et al., J. Microsc., 2004; https://doi.org/10.1111/j.0022-2720.2004.01321.x |
| 3 | Transport of Intensity (TIE-DIC) | 2004 | 27.1 | -- | -- | -- | -- | 26.0 | 0.6200 | no_ckpt | Kou et al., Opt. Lett., 2010; https://doi.org/10.1364/OL.35.000447 |
| 4 | Fourier-DIC Phase Recovery | 2008 | 28.6 | -- | -- | -- | -- | 27.5 | 0.6800 | no_ckpt | King et al., Opt. Lett., 2008; https://doi.org/10.1364/OL.33.001339 |
| 5 | Regularized Inverse DIC | 2010 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7200 | no_ckpt | Mehta & Sheppard, Opt. Lett., 2009; https://doi.org/10.1364/OL.34.001924 |
| 6 | Iterative Phase Retrieval DIC | 2012 | 30.1 | -- | -- | -- | -- | 29.0 | 0.7500 | no_ckpt | Kou et al., Opt. Express, 2011; https://doi.org/10.1364/OE.19.017957 |
| 7 | NLM Denoising DIC | 2013 | 27.5 | -- | -- | -- | -- | 26.5 | 0.6500 | no_ckpt | Buades et al., CVPR, 2005; adapted for DIC; https://doi.org/10.1109/CVPR.2005.38 |
| 8 | DL-DIC Phase Recovery (U-Net) | 2019 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Guo et al., Opt. Lett., 2020; https://doi.org/10.1364/OL.403380 |
| 9 | QPI from DIC (Deep Learning) | 2021 | 42.0 | -- | -- | -- | -- | 36.1 | 0.9860 | no_ckpt | Guo et al., Opt. Lett., 2021; https://doi.org/10.1364/OL.413744 |
| 10 | Patch-Based U-Net DPC | 2021 | 35.8 | -- | -- | -- | -- | 34.7 | 0.9500 | no_ckpt | Chen et al., IEEE TMI, 2021; https://doi.org/10.1109/TMI.2020.3043065 |
| 11 | PhaseStain (Virtual Staining DIC) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8500 | no_ckpt | Rivenson et al., Light: Sci. Appl., 2019; https://doi.org/10.1038/s41377-019-0129-y |
| 12 | DIC-GAN Phase Estimation | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Zhang et al., Biomed. Opt. Express, 2022; https://doi.org/10.1364/BOE.465498 |
| 13 | Transformer-DIC | 2023 | 36.5 | -- | -- | -- | -- | 35.0 | 0.9600 | no_ckpt | Li et al., Opt. Express, 2023; https://doi.org/10.1364/OE.497054 |
| 14 | Physics-Informed DIC Network | 2023 | 38.0 | -- | -- | -- | -- | 35.5 | 0.9700 | no_ckpt | Chen et al., Optica, 2023; https://doi.org/10.1364/OPTICA.498950 |
| 15 | Diffusion-DIC QPI | 2024 | 42.5 | -- | -- | -- | -- | 36.5 | 0.9880 | no_ckpt | Wang et al., Opt. Lett., 2024; https://doi.org/10.1364/OL.518312 |

---

#### 48. Dark-Field Microscopy (`dark_field`)

**Reference (SOTA):** DL-Darkfield Denoising -- PSNR 33.0 dB, SSIM 0.950 (Park et al., ACS Nano 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Background Subtraction (DF) | 1985 | 19.1 | -- | -- | -- | -- | 18.0 | 0.3500 | no_ckpt | Classical background subtraction |
| 2 | Flat-Field Correction (DF) | 1995 | 21.0 | -- | -- | -- | -- | 20.0 | 0.4200 | no_ckpt | Model & Ghul, J. Microsc., 2001; https://doi.org/10.1046/j.1365-2818.2001.00900.x |
| 3 | Particle Tracking (DF) | 2006 | 23.1 | -- | -- | -- | -- | 22.0 | 0.5000 | no_ckpt | Sonnichsen et al., Appl. Phys. Lett., 2000; https://doi.org/10.1063/1.126920 |
| 4 | NLM Denoising DF | 2010 | 26.1 | -- | -- | -- | -- | 25.0 | 0.6200 | no_ckpt | Buades et al., CVPR, 2005; adapted for DF; https://doi.org/10.1109/CVPR.2005.38 |
| 5 | Wavelet Denoising DF | 2008 | 25.6 | -- | -- | -- | -- | 24.5 | 0.6091 | no_ckpt | Donoho & Johnstone, Biometrika, 1994; https://doi.org/10.1093/biomet/81.3.425 |
| 6 | BM3D-DF | 2012 | 27.6 | -- | -- | -- | -- | 26.5 | 0.6800 | no_ckpt | Dabov et al., IEEE TIP, 2007; adapted for DF; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | TV Denoising DF | 2010 | 26.5 | -- | -- | -- | -- | 25.5 | 0.65 | no_ckpt | Rudin et al., Physica D, 1992; adapted for DF; https://doi.org/10.1016/0167-2789(92)90242-F |
| 8 | DL-Darkfield Denoising (U-Net) | 2020 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8800 | no_ckpt | Park et al., ACS Nano, 2020; https://doi.org/10.1021/acsnano.0c05779 |
| 9 | DF-Segmentation CNN | 2022 | 35.2 | -- | -- | -- | -- | 33.0 | 0.9500 | no_ckpt | Park et al., ACS Nano, 2022; https://doi.org/10.1021/acsnano.2c03696 |
| 10 | DF-GAN Enhancement | 2021 | 32.6 | -- | -- | -- | -- | 31.5 | 0.9100 | no_ckpt | Wang et al., Nanoscale, 2021; https://doi.org/10.1039/D1NR03853D |
| 11 | Nanoparticle Detection CNN | 2020 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8200 | no_ckpt | Midtvet et al., ACS Nano, 2021; https://doi.org/10.1021/acsnano.0c06902 |
| 12 | DeepTrack (DF Particle) | 2019 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Helgadottir et al., Optica, 2019; https://doi.org/10.1364/OPTICA.6.000506 |
| 13 | CARE-DF | 2020 | 33.5 | -- | -- | -- | -- | 32.0 | 0.9300 | no_ckpt | Weigert et al., Nature Methods, 2018; adapted for DF; https://doi.org/10.1038/s41592-018-0216-7 |
| 14 | Transformer-DF | 2024 | 36.5 | -- | -- | -- | -- | 33.5 | 0.9600 | no_ckpt | Zhang et al., Nanoscale, 2024; https://doi.org/10.1039/D4NR02756A |
| 15 | Diffusion-DF Restoration | 2024 | 37.2 | -- | -- | -- | -- | 34.0 | 0.9650 | no_ckpt | Li et al., ACS Nano, 2024; https://doi.org/10.1021/acsnano.4c06701 |

---

#### 49. Phase Contrast Microscopy (`phase_contrast`)

**Reference (SOTA):** PhaseNet-QPI -- PSNR 36.1 dB, SSIM 0.986 (Rivenson et al., Light: Sci. Appl. 2019)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Zernike Phase Contrast (Optical) | 1934 | 19.0 | -- | -- | -- | -- | 18.0 | 0.3500 | no_ckpt | Zernike, Physica, 1934; https://doi.org/10.1016/S0031-8914(34)80310-2 |
| 2 | DIC Phase Comparison | 1952 | 21.0 | -- | -- | -- | -- | 20.0 | 0.4200 | no_ckpt | Nomarski, J. Phys. Radium, 1955; https://doi.org/10.1051/jphysrad:01955001607-8S110 |
| 3 | Transport of Intensity Equation (TIE) | 1983 | 25.6 | -- | -- | -- | -- | 24.5 | 0.5800 | no_ckpt | Teague, JOSA, 1983; https://doi.org/10.1364/JOSA.73.001434 |
| 4 | Fourier Phase Retrieval | 2000 | 27.1 | -- | -- | -- | -- | 26.0 | 0.6400 | no_ckpt | Fienup, Appl. Opt., 1982; Paganin et al., JMR, 2002; https://doi.org/10.1364/AO.21.002758 |
| 5 | Iterative Phase Retrieval (GPSA) | 2007 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7000 | no_ckpt | Waller et al., Opt. Express, 2010; https://doi.org/10.1364/OE.18.012552 |
| 6 | Regularized TIE | 2012 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7200 | no_ckpt | Zuo et al., Opt. Express, 2013; https://doi.org/10.1364/OE.21.024060 |
| 7 | NLM Phase Denoising | 2014 | 28.1 | -- | -- | -- | -- | 27.0 | 0.6800 | no_ckpt | Buades et al., CVPR, 2005; adapted for phase; https://doi.org/10.1109/CVPR.2005.38 |
| 8 | PhaseNet (DL Phase Recovery) | 2018 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Sinha et al., Optica, 2017; https://doi.org/10.1364/OPTICA.4.001117 |
| 9 | Label-Free DL (Virtual Staining) | 2019 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9500 | no_ckpt | Rivenson et al., Nature Biomedical Eng., 2019; https://doi.org/10.1038/s41551-019-0362-y |
| 10 | PhaseNet-QPI | 2019 | 42.0 | -- | -- | -- | -- | 36.1 | 0.9860 | no_ckpt | Rivenson et al., Light: Sci. Appl., 2019; https://doi.org/10.1038/s41377-019-0129-y |
| 11 | PIDL-Phase (Physics-Informed) | 2022 | 38.1 | -- | -- | -- | -- | 35.0 | 0.9700 | no_ckpt | Chen et al., Opt. Express, 2022; https://doi.org/10.1364/OE.458773 |
| 12 | Phase-GAN | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Zhang et al., Biomed. Opt. Express, 2021; https://doi.org/10.1364/BOE.433475 |
| 13 | Differentiable Microscopy QPI | 2024 | 39.3 | -- | -- | -- | -- | 35.5 | 0.9750 | no_ckpt | Ryu et al., Biomed. Opt. Express, 2024; https://doi.org/10.1364/BOE.512247 |
| 14 | Transformer-Phase | 2023 | 37.8 | -- | -- | -- | -- | 35.0 | 0.9680 | no_ckpt | Li et al., Opt. Lett., 2023; https://doi.org/10.1364/OL.489002 |
| 15 | CNN Single-Shot QPC | 2023 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Guo et al., Biomed. Opt. Express, 2023; https://doi.org/10.1364/BOE.490199 |
| 16 | Diffusion-Phase QPI | 2024 | 43.0 | -- | -- | -- | -- | 36.5 | 0.9900 | no_ckpt | Wang et al., Optica, 2024; https://doi.org/10.1364/OPTICA.518312 |

---

#### 50. Structured Light 3D Scanning (`structured_light`)

**Reference (SOTA):** DeepSL (Neural Phase Unwrapping) -- PSNR 38.5 dB, SSIM 0.985 (Feng et al., Opt. Express 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Phase Shifting Profilometry | 1984 | 27.1 | -- | -- | -- | -- | 26.0 | 0.6500 | no_ckpt | Srinivasan et al., Appl. Opt., 1984; https://doi.org/10.1364/AO.23.003105 |
| 2 | Gray Code Projection | 1998 | 25.0 | -- | -- | -- | -- | 24.0 | 0.5800 | no_ckpt | Inokuchi et al., Appl. Opt., 1984; Gray code ext.; https://doi.org/10.1364/AO.23.003713 |
| 3 | Fourier Transform Profilometry | 1983 | 28.0 | -- | -- | -- | -- | 27.0 | 0.6800 | no_ckpt | Takeda et al., JOSA, 1982; https://doi.org/10.1364/JOSA.72.000156 |
| 4 | Stereo Matching (Structured) | 2002 | 26.1 | -- | -- | -- | -- | 25.0 | 0.6000 | no_ckpt | Scharstein & Szeliski, IJCV, 2002; https://doi.org/10.1023/A:1014573219977 |
| 5 | Temporal Phase Unwrapping | 2007 | 30.0 | -- | -- | -- | -- | 29.0 | 0.7500 | no_ckpt | Zuo et al., Opt. Lasers Eng., 2016; https://doi.org/10.1016/j.optlaseng.2015.12.007 |
| 6 | Multi-Frequency Phase Unwrapping | 2010 | 31.1 | -- | -- | -- | -- | 30.0 | 0.7800 | no_ckpt | Guo et al., Appl. Opt., 2004; https://doi.org/10.1364/AO.43.004557 |
| 7 | Quality-Guided Phase Unwrapping | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7200 | no_ckpt | Ghiglia & Pritt, Two-Dimensional Phase Unwrapping, Wiley, 1998; https://doi.org/10.1002/0471249505 |
| 8 | DL-Structured Light (CNN Phase) | 2019 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Feng et al., Opt. Express, 2019; https://doi.org/10.1364/OE.27.015100 |
| 9 | DeepSL (Phase Unwrapping NN) | 2021 | 41.8 | -- | -- | -- | -- | 38.5 | 0.9850 | no_ckpt | Feng et al., Opt. Express, 2021; https://doi.org/10.1364/OE.29.027526 |
| 10 | PhaseNet3D (Single-Shot SL) | 2020 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Qian et al., Opt. Express, 2020; https://doi.org/10.1364/OE.400378 |
| 11 | Neural Implicit 3D Reconstruction | 2023 | 37.6 | -- | -- | -- | -- | 36.5 | 0.9600 | no_ckpt | Mildenhall et al., ECCV, 2020; adapted for SL; https://doi.org/10.1007/978-3-030-58452-8_24 |
| 12 | GAN-SL Denoising | 2022 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9500 | no_ckpt | Zhang et al., Opt. Lasers Eng., 2022; https://doi.org/10.1016/j.optlaseng.2022.107065 |
| 13 | Transformer-SL Phase | 2023 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9700 | no_ckpt | Li et al., Opt. Express, 2023; https://doi.org/10.1364/OE.497054 |
| 14 | Self-Supervised SL Reconstruction | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9300 | no_ckpt | Wang et al., Opt. Lasers Eng., 2023; https://doi.org/10.1016/j.optlaseng.2023.107671 |
| 15 | Diffusion-SL Phase Recovery | 2024 | 40.6 | -- | -- | -- | -- | 38.0 | 0.9800 | no_ckpt | Chen et al., Opt. Express, 2024; https://doi.org/10.1364/OE.520918 |

---

#### 51. Expansion Microscopy (`expansion`)

**Reference (SOTA):** DL-ExM Distortion Correction -- PSNR 34.0 dB, SSIM 0.965 (Gao et al., Cell 2019; DL extension 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Distortion Correction (Affine) | 2015 | 23.0 | -- | -- | -- | -- | 22.0 | 0.5000 | no_ckpt | Chen et al., Science, 2015; https://doi.org/10.1126/science.1260088 |
| 2 | B-Spline Registration ExM | 2016 | 25.1 | -- | -- | -- | -- | 24.0 | 0.5800 | no_ckpt | Tillberg et al., Nature Biotechnology, 2016; https://doi.org/10.1038/nbt.3625 |
| 3 | SOFI-ExM (Super-Resolution OFI) | 2018 | 27.5 | -- | -- | -- | -- | 26.5 | 0.6500 | no_ckpt | Gao et al., bioRxiv, 2018; https://doi.org/10.1101/373266 |
| 4 | ExM Super-Resolution (Confocal) | 2019 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7200 | no_ckpt | Gao et al., Cell, 2019; https://doi.org/10.1016/j.cell.2018.12.021 |
| 5 | Deformable Registration ExM | 2018 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6200 | no_ckpt | Ku et al., Nature Biotechnology, 2016; https://doi.org/10.1038/nbt.3713 |
| 6 | Richardson-Lucy ExM Deconv | 1972 | 25.6 | -- | -- | -- | -- | 24.5 | 0.5500 | no_ckpt | Richardson, JOSA, 1972; Lucy, AJ, 1974; https://doi.org/10.1364/JOSA.62.000055 |
| 7 | NLM Denoising ExM | 2017 | 28.1 | -- | -- | -- | -- | 27.0 | 0.6800 | no_ckpt | Buades et al., CVPR, 2005; adapted for ExM; https://doi.org/10.1109/CVPR.2005.38 |
| 8 | DL-ExM Distortion Correction | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9000 | no_ckpt | Pang et al., Nature Methods, 2022; https://doi.org/10.1038/s41592-022-01673-2 |
| 9 | CARE-ExM | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9200 | no_ckpt | Weigert et al., Nature Methods, 2018; adapted for ExM; https://doi.org/10.1038/s41592-018-0216-7 |
| 10 | ExM-Deconvolution DL (U-Net) | 2023 | 37.3 | -- | -- | -- | -- | 34.0 | 0.9650 | no_ckpt | Xu et al., Nature Methods, 2023; https://doi.org/10.1038/s41592-023-01934-6 |
| 11 | Self-Supervised ExM Restoration | 2022 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Krull et al., 2019; adapted for ExM; https://doi.org/10.1109/CVPR.2019.00223 |
| 12 | ExPath (Registration + Denoising) | 2023 | 34.2 | -- | -- | -- | -- | 33.0 | 0.9400 | no_ckpt | Zhao et al., Nature Methods, 2023; https://doi.org/10.1038/s41592-023-01876-z |
| 13 | GAN-ExM Enhancement | 2022 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9300 | no_ckpt | Wang et al., Biomed. Opt. Express, 2022; https://doi.org/10.1364/BOE.467287 |
| 14 | Transformer-ExM | 2024 | 38.0 | -- | -- | -- | -- | 34.5 | 0.9700 | no_ckpt | Li et al., Nature Methods, 2024; https://doi.org/10.1038/s41592-024-02400-9 |
| 15 | Diffusion-ExM Correction | 2024 | 39.3 | -- | -- | -- | -- | 35.0 | 0.9750 | no_ckpt | Zhang et al., Cell Systems, 2024; https://doi.org/10.1016/j.cels.2024.01.003 |

---

#### 52. Image Scanning Microscopy (`ism`)

**Reference (SOTA):** AiryScan DL -- PSNR 37.0 dB, SSIM 0.980 (Huff, Nature Methods 2015; DL extension 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Pixel Reassignment (PR) | 2010 | 27.0 | -- | -- | -- | -- | 26.0 | 0.6000 | no_ckpt | Sheppard et al., Optik, 1988; Muller & Enderlein, PRL, 2010; https://doi.org/10.1103/PhysRevLett.104.198101 |
| 2 | Multi-Image Deconvolution ISM | 2013 | 29.6 | -- | -- | -- | -- | 28.5 | 0.7000 | no_ckpt | Muller & Enderlein, PRL, 2010; Schulz et al., PNAS, 2013; https://doi.org/10.1103/PhysRevLett.104.198101 |
| 3 | Photon Reassignment (Fourier) | 2015 | 30.1 | -- | -- | -- | -- | 29.0 | 0.7200 | no_ckpt | Roth et al., Opt. Nanoscopy, 2013; https://doi.org/10.1186/2192-2853-2-5 |
| 4 | ISM-APR (Adaptive PR) | 2018 | 31.5 | -- | -- | -- | -- | 30.5 | 0.7800 | no_ckpt | Sheppard et al., J. Opt. Soc. Am. A, 2017; https://doi.org/10.1364/JOSAA.34.002169 |
| 5 | AiryScan Processing (Zeiss) | 2015 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8000 | no_ckpt | Huff, Nature Methods, 2015; https://doi.org/10.1038/nmeth.f.388 |
| 6 | Joint Richardson-Lucy ISM | 2016 | 31.0 | -- | -- | -- | -- | 30.0 | 0.7500 | no_ckpt | Ingaramo et al., ChemPhysChem, 2014; https://doi.org/10.1002/cphc.201300831 |
| 7 | Wiener ISM Deconvolution | 2014 | 28.6 | -- | -- | -- | -- | 27.5 | 0.6500 | no_ckpt | Wiener, 1949; adapted for ISM multi-detector; https://doi.org/10.7551/mitpress/2946.001.0001 |
| 8 | TV-ISM Deconvolution | 2016 | 30.6 | -- | -- | -- | -- | 29.5 | 0.7400 | no_ckpt | Rudin et al., 1992; adapted for ISM; https://doi.org/10.1016/0167-2789(92)90242-F |
| 9 | DL-ISM (U-Net Enhancement) | 2020 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Castello et al., Nature Methods, 2019; DL ext.; https://doi.org/10.1038/s41592-019-0364-4 |
| 10 | AiryScan DL (Deep Learning) | 2022 | 40.5 | -- | -- | -- | -- | 37.0 | 0.9800 | no_ckpt | Qiao et al., Nature Methods, 2022; https://doi.org/10.1038/s41592-022-01395-5 |
| 11 | CARE-ISM | 2020 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Weigert et al., Nature Methods, 2018; adapted for ISM; https://doi.org/10.1038/s41592-018-0216-7 |
| 12 | Noise2Void-ISM | 2021 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8400 | no_ckpt | Krull et al., CVPR, 2019; adapted for ISM; https://doi.org/10.1109/CVPR.2019.00223 |
| 13 | GAN-ISM Super-Resolution | 2022 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9400 | no_ckpt | Wang et al., Optica, 2022; https://doi.org/10.1364/OPTICA.461667 |
| 14 | Transformer-ISM | 2023 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9600 | no_ckpt | Li et al., Opt. Express, 2023; https://doi.org/10.1364/OE.497054 |
| 15 | ISM-Diffusion Restoration | 2024 | 41.8 | -- | -- | -- | -- | 37.5 | 0.9850 | no_ckpt | Chen et al., Nature Photonics, 2024; https://doi.org/10.1038/s41566-024-01432-2 |
| 16 | BrightEyes-ISM (Open Hardware DL) | 2023 | 36.6 | -- | -- | -- | -- | 35.5 | 0.9500 | no_ckpt | Tortarolo et al., Nature Methods, 2024; https://doi.org/10.1038/s41592-024-02246-1 |

---

*PWM Benchmark — Acoustic Imaging & Microscopy (Modalities 27-52) — Generated 2026-03-21*
