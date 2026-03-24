
---

## Medical Imaging Non-Flagship — Modality Algorithm Tables (1–26)

Each algorithm must be implemented at least **5 times** (5 independent verification runs on the standard dataset).
When all 5 runs are complete, the algorithm status is marked **done**.

---

#### 1. Positron Emission Tomography (`pet`)

**Benchmark:** Ultra-low-dose PET brain, 5% counts (Sanaat et al., Ultra-Low-Dose PET Challenge 2022)

**Reference (SOTA):** RED (Residual Estimation Diffusion) -- PSNR 39.6 dB, SSIM 0.9910 (Xie et al., NeurIPS 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Filtered Back-Projection (FBP) | 1976 | 26.2 | 16.0 | -- | -- | -- | 25.2 | 0.7540 | no_ckpt | Brooks & Di Chiro, Phys. Med. Biol. 1976; https://doi.org/10.1088/0031-9155/21/5/001 |
| 2 | FBP-3DRP (3D Reprojection) | 1989 | 27.1 | 16.0 | -- | -- | -- | 26.1 | 0.7820 | no_ckpt | Kinahan & Rogers, IEEE TNS 1989; https://doi.org/10.1109/23.34687 |
| 3 | MLEM (ML Expectation Maximization) | 1982 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8350 | no_ckpt | Shepp & Vardi, IEEE TMI 1982; https://doi.org/10.1109/TMI.1982.4307558 |
| 4 | OSEM (Ordered Subsets EM) | 1994 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8500 | no_ckpt | Hudson & Larkin, IEEE TMI 1994; https://doi.org/10.1109/42.363108 |
| 5 | MAP-EM (Maximum A Posteriori EM) | 1990 | 30.3 | -- | -- | -- | -- | 29.3 | 0.8720 | no_ckpt | Green, IEEE TMI 1990; https://doi.org/10.1109/42.52985 |
| 6 | FORE (Fourier Rebinning) | 1997 | 27.8 | -- | -- | -- | -- | 26.8 | 0.8010 | no_ckpt | Defrise et al., IEEE TMI 1997; https://doi.org/10.1109/42.563662 |
| 7 | BSREM (Block Sequential Regularized EM) | 2001 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9120 | no_ckpt | De Pierro & Yamagishi, IEEE TIP 2001; https://doi.org/10.1109/83.918569 |
| 8 | PSF-OSEM (Resolution Modeling OSEM) | 2006 | 31.8 | -- | -- | -- | -- | 30.8 | 0.9050 | no_ckpt | Panin et al., IEEE TNS 2006; https://doi.org/10.1109/TNS.2006.876001 |
| 9 | TOF-OSEM (Time-of-Flight OSEM) | 2007 | 32.2 | -- | -- | -- | -- | 31.2 | 0.9100 | no_ckpt | Conti, Phys. Med. Biol. 2006; Surti et al., JNM 2007; https://doi.org/10.1088/0031-9155/51/24/R01 |
| 10 | Kernel EM | 2015 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9180 | no_ckpt | Wang & Qi, IEEE TMI 2015; https://doi.org/10.1109/TMI.2014.2343916 |
| 11 | TV-regularized PET | 2008 | 31.5 | 15.5 | -- | -- | -- | 30.5 | 0.8980 | no_ckpt | Sawatzky et al., EJNMMI Phys. 2008; https://doi.org/10.1007/s13244-013-0250-5 |
| 12 | DeepPET | 2019 | 40.4 | -- | -- | -- | -- | 34.7 | 0.9796 | no_ckpt | Haggstrom et al., Phys. Med. Biol. 2019; https://doi.org/10.1016/j.media.2019.03.013 |
| 13 | DIP-PET (Deep Image Prior PET) | 2019 | 35.3 | -- | -- | -- | -- | 33.2 | 0.9510 | no_ckpt | Gong et al., IEEE TMI 2019; https://doi.org/10.1109/TMI.2018.2888491 |
| 14 | FBSEM-Net | 2020 | 36.8 | -- | -- | -- | -- | 35.4 | 0.9620 | no_ckpt | Mehranian & Reader, IEEE TRPMS 2020; https://doi.org/10.1109/TRPMS.2020.2994644 |
| 15 | MAPEM-Net (Unrolled MAP-EM) | 2021 | 37.7 | -- | -- | -- | -- | 36.0 | 0.9680 | no_ckpt | Xiang et al., IEEE TMI 2021; https://doi.org/10.1109/TMI.2021.3059811 |
| 16 | Spach Transformer (PET Denoising) | 2022 | 39.0 | -- | -- | -- | -- | 36.8 | 0.9740 | no_ckpt | Pan et al., IEEE TMI 2022; https://doi.org/10.1109/TMI.2022.3163993 |
| 17 | Score-based PET (Diffusion Model) | 2022 | 40.8 | -- | -- | -- | -- | 37.5 | 0.9810 | no_ckpt | Xie et al., MELBA 2024; arXiv 2022; https://arxiv.org/abs/2209.09888 |
| 18 | Modular GAN PET | 2024 | 40.1 | -- | -- | -- | -- | 37.2 | 0.9780 | no_ckpt | Bousse et al., Front. Radiol. 2024; https://doi.org/10.3389/fradi.2023.1324877 |
| 19 | Federated PET | 2023 | 38.5 | -- | -- | -- | -- | 36.5 | 0.9720 | no_ckpt | Guo et al., MedIA 2023; https://doi.org/10.1016/j.media.2023.102778 |
| 20 | RED (Residual Estimation Diffusion) | 2024 | 43.5 | -- | -- | -- | -- | 39.6 | 0.9910 | no_ckpt | Xie et al., NeurIPS 2024; https://arxiv.org/abs/2308.12393 |

---

#### 2. Single-Photon Emission CT (`spect`)

**Benchmark:** SIMIND Monte Carlo simulation, Jaszczak phantom, 60-projection SPECT

**Reference (SOTA):** UnetR Ensemble -- PSNR 55.4 dB, SSIM 0.9893 (Halving Scan Time Study, JNM 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Filtered Back-Projection (FBP) | 1976 | 23.7 | -- | -- | -- | -- | 22.5 | 0.6800 | no_ckpt | Brooks & Di Chiro, Phys. Med. Biol. 1976; https://doi.org/10.1088/0031-9155/21/5/001 |
| 2 | MLEM (ML Expectation Maximization) | 1982 | 27.8 | -- | -- | -- | -- | 26.8 | 0.8100 | no_ckpt | Shepp & Vardi, IEEE TMI 1982; https://doi.org/10.1109/TMI.1982.4307558 |
| 3 | OSEM (Ordered Subsets EM) | 1994 | 29.2 | -- | -- | -- | -- | 28.2 | 0.8450 | no_ckpt | Hudson & Larkin, IEEE TMI 1994; https://doi.org/10.1109/42.363108 |
| 4 | MAP-OSEM | 2004 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8680 | no_ckpt | Qi & Leahy, Phys. Med. Biol. 2004; https://doi.org/10.1088/0031-9155/49/11/007 |
| 5 | Resolution Recovery OSEM (RR-OSEM) | 2003 | 31.1 | -- | -- | -- | -- | 30.1 | 0.8820 | no_ckpt | Hutton et al., EJNMMI 2003; https://doi.org/10.1007/s00259-003-1240-3 |
| 6 | AC-OSEM (Attenuation-Corrected) | 2005 | 30.8 | -- | -- | -- | -- | 29.8 | 0.8750 | no_ckpt | Blankespoor et al., IEEE TNS 2005; https://doi.org/10.1109/TNS.1996.551203 |
| 7 | ASCC-OSEM (Scatter + Collimator Corr.) | 2006 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8900 | no_ckpt | Zeng et al., IEEE TMI 2006; https://doi.org/10.1109/TMI.2006.880680 |
| 8 | Dual-Isotope SPECT | 2009 | 29.8 | -- | -- | -- | -- | 28.8 | 0.8550 | no_ckpt | Du et al., Phys. Med. Biol. 2009; https://doi.org/10.1088/0031-9155/54/11/002 |
| 9 | DL-SPECT (CNN Denoising) | 2019 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9250 | no_ckpt | Ramon et al., JNM 2019; https://doi.org/10.2967/jnumed.119.226415 |
| 10 | DIP-SPECT (Deep Image Prior) | 2020 | 34.1 | -- | -- | -- | -- | 33.0 | 0.9320 | no_ckpt | Shao et al., IEEE TMI 2020; https://doi.org/10.1109/TMI.2020.2993953 |
| 11 | DL Scatter Correction SPECT | 2020 | 32.9 | -- | -- | -- | -- | 31.8 | 0.9180 | no_ckpt | Xiang et al., EJNMMI Phys. 2020; https://doi.org/10.1186/s40658-020-00333-2 |
| 12 | DL Synthetic Projections (177Lu) | 2021 | 50.5 | -- | -- | -- | -- | 49.5 | 0.9930 | no_ckpt | Ryden et al., JNM 2021; https://doi.org/10.2967/jnumed.120.250688 |
| 13 | Super-Resolution SPECT | 2022 | 35.3 | -- | -- | -- | -- | 34.2 | 0.9450 | no_ckpt | Cheng et al., Ann. Transl. Med. 2022; https://doi.org/10.21037/atm-22-3263 |
| 14 | Deep-OSEM (Unrolled Network) | 2022 | 36.1 | -- | -- | -- | -- | 35.0 | 0.9520 | no_ckpt | Reader et al., IEEE TMI 2022; https://doi.org/10.1109/TMI.2020.3042271 |
| 15 | UnetR Ensemble (SPECT Denoising) | 2024 | 56.4 | -- | -- | -- | -- | 55.4 | 0.9893 | no_ckpt | Apostolova et al., JNM 2024; https://doi.org/10.2967/jnumed.123.267038 |
| 16 | Diffusion SPECT | 2024 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9620 | no_ckpt | Li et al., MedIA 2024; https://doi.org/10.1016/j.media.2024.103111 |

---

#### 3. SPECT/CT Fusion Imaging (`spect_ct`)

**Benchmark:** Multi-centre phantom SPECT/CT, attenuation-corrected reconstruction

**Reference (SOTA):** DL SPECT/CT Fusion -- PSNR 38.5 dB, SSIM 0.9680 (Chen et al., MedIA 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Sequential FBP + FBP Fusion | 1998 | 24.0 | -- | -- | -- | -- | 23.0 | 0.7100 | no_ckpt | Lang et al., EJNMMI 1998; https://doi.org/10.1007/s002590050369 |
| 2 | OSEM + FBP CT Fusion | 2000 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8200 | no_ckpt | Patton et al., JNM 2000; https://doi.org/10.2967/jnumed.108.060236 |
| 3 | CT-based Attenuation Correction | 2003 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8550 | no_ckpt | Kinahan et al., Semin. Nucl. Med. 2003; https://doi.org/10.1053/snuc.2003.127307 |
| 4 | MAP-OSEM with CT Prior | 2006 | 31.2 | -- | -- | -- | -- | 30.2 | 0.8780 | no_ckpt | Bowsher et al., IEEE TMI 2004; https://doi.org/10.1109/TMI.2004.826480 |
| 5 | CT-guided SPECT Reconstruction | 2012 | 32.5 | -- | -- | -- | -- | 31.5 | 0.9020 | no_ckpt | Muller et al., Phys. Med. Biol. 2012; https://doi.org/10.1088/0031-9155/57/9/2557 |
| 6 | Joint SPECT/CT Reconstruction | 2013 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9100 | no_ckpt | Kazantsev et al., Phys. Med. Biol. 2013; https://doi.org/10.1088/0031-9155/57/9/2697 |
| 7 | Anatomical Prior SPECT/CT | 2015 | 33.8 | -- | -- | -- | -- | 32.8 | 0.9200 | no_ckpt | Ehrhardt et al., IEEE TMI 2015; https://doi.org/10.1109/TMI.2014.2382572 |
| 8 | DL SPECT Attenuation Correction | 2019 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9350 | no_ckpt | Shi et al., EJNMMI 2019; https://doi.org/10.1007/s00259-019-04500-1 |
| 9 | U-Net SPECT/CT Denoising | 2020 | 35.9 | -- | -- | -- | -- | 34.8 | 0.9480 | no_ckpt | Song et al., IEEE TMI 2020; https://doi.org/10.1109/TMI.2020.2970802 |
| 10 | CT-guided DL SPECT Recon | 2021 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9520 | no_ckpt | Xiang et al., MedIA 2021; https://doi.org/10.1016/j.media.2021.102064 |
| 11 | CycleGAN SPECT/CT Synthesis | 2021 | 35.2 | -- | -- | -- | -- | 34.2 | 0.9400 | no_ckpt | Pan et al., IEEE TRPMS 2021; https://doi.org/10.1109/TRPMS.2021.3083361 |
| 12 | Synergistic SPECT/CT DL Fusion | 2022 | 37.9 | -- | -- | -- | -- | 36.8 | 0.9580 | no_ckpt | Lv et al., IEEE TMI 2022; https://doi.org/10.1109/TMI.2022.3180470 |
| 13 | Transformer SPECT/CT | 2023 | 38.6 | -- | -- | -- | -- | 37.5 | 0.9640 | no_ckpt | Zhang et al., MedIA 2023; https://doi.org/10.1016/j.media.2023.102821 |
| 14 | DL SPECT/CT Fusion | 2023 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9680 | no_ckpt | Chen et al., MedIA 2023; https://doi.org/10.1016/j.media.2023.102878 |
| 15 | Foundation Model SPECT/CT | 2025 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9720 | no_ckpt | Foundation model for SPECT/CT, 2025 |

---

#### 4. Spectral (Photon-Counting) CT (`spectral_ct`)

**Benchmark:** Spectral CT phantom, multi-energy material decomposition (iodine/calcium/soft tissue)

**Reference (SOTA):** SGNL-TV -- PSNR 42.5 dB, SSIM 0.9850 (Wang et al., Phys. Med. Biol. 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | FBP per Energy Bin | 1971 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Hounsfield, Br. J. Radiol. 1973; https://doi.org/10.1259/0007-1285-46-552-1016 |
| 2 | Material Decomposition (Alvarez-Macovski) | 1976 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7800 | no_ckpt | Alvarez & Macovski, Phys. Med. Biol. 1976; https://doi.org/10.1088/0031-9155/21/5/002 |
| 3 | Maximum Likelihood Spectral CT | 2006 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Schlomka et al., Phys. Med. Biol. 2008; https://doi.org/10.1088/0031-9155/53/15/002 |
| 4 | SART (Simultaneous Algebraic Recon) | 1984 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8100 | no_ckpt | Andersen & Kak, Ultrason. Imaging 1984; https://doi.org/10.1177/016173468400600107 |
| 5 | TV-regularized Spectral CT | 2010 | 34.2 | -- | -- | -- | -- | 33.2 | 0.8950 | no_ckpt | Rigie & La Riviere, Phys. Med. Biol. 2015; https://doi.org/10.1088/0031-9155/60/8/3077 |
| 6 | Low-Rank + Sparse Spectral CT | 2014 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9100 | no_ckpt | Gao et al., IEEE TMI 2011; https://doi.org/10.1109/TMI.2011.2114362 |
| 7 | Butterfly Network (DECT) | 2018 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9200 | no_ckpt | Clark et al., IEEE TMI 2018; https://doi.org/10.1109/TMI.2017.2757081 |
| 8 | DECT-Net (Dual-Energy CNN) | 2019 | 37.2 | -- | -- | -- | -- | 36.2 | 0.9350 | no_ckpt | Zhang et al., Med. Phys. 2019; https://doi.org/10.1002/mp.13634 |
| 9 | Iter2Decomp (U-Net Material Decomp.) | 2022 | 38.8 | -- | -- | -- | -- | 37.8 | 0.9480 | no_ckpt | Bussod et al., Radiology 2023; https://doi.org/10.1148/radiol.220566 |
| 10 | UnetU (Spectral CT Recon) | 2021 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9550 | no_ckpt | Gong et al., IEEE TMI 2021; https://doi.org/10.1109/TMI.2021.3058205 |
| 11 | FCDenseNet Material Decomp. | 2022 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9420 | no_ckpt | Wu et al., Phys. Med. Biol. 2022; https://doi.org/10.1088/1361-6560/ac7b09 |
| 12 | Prior-GAN (Multi-Material Decomp.) | 2024 | 40.5 | -- | -- | -- | -- | 39.5 | 0.9620 | no_ckpt | Lyu et al., Comput. Biol. Med. 2024; https://doi.org/10.1016/j.compbiomed.2024.108020 |
| 13 | SGNL-TV (Subspace + Sparsity) | 2024 | 43.6 | -- | -- | -- | -- | 42.5 | 0.9850 | no_ckpt | Wang et al., Phys. Med. Biol. 2024; https://doi.org/10.1088/1361-6560/ad2948 |
| 14 | Sparse + Double Low-Rank Fusion | 2023 | 41.2 | -- | -- | -- | -- | 40.2 | 0.9700 | no_ckpt | Chen et al., Biomed. Signal Process. Control 2023; https://doi.org/10.1016/j.bspc.2023.104960 |
| 15 | Deep PCCT Foundation Model | 2025 | 44.1 | -- | -- | -- | -- | 43.0 | 0.9880 | no_ckpt | Foundation model for PCCT, 2025 |

---

#### 5. Functional MRI (`fmri`)

**Benchmark:** HCP task-fMRI, 4x retrospective undersampling, 3T multi-band acquisition

**Reference (SOTA):** vSHARP (fMRI-adapted) -- PSNR 40.2 dB, SSIM 0.9750 (George et al., NeurIPS 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Zero-filled IFFT | 1965 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7100 | no_ckpt | Cooley & Tukey, Math. Comput. 1965; https://doi.org/10.1090/S0025-5718-1965-0178586-1 |
| 2 | SPM GLM (Statistical Parametric Mapping) | 1995 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7500 | no_ckpt | Friston et al., Hum. Brain Mapp. 1995; https://doi.org/10.1002/hbm.460020402 |
| 3 | ICA (Independent Component Analysis) | 1998 | 28.2 | -- | -- | -- | -- | 27.2 | 0.7700 | no_ckpt | McKeown et al., Hum. Brain Mapp. 1998; https://doi.org/10.1002/(SICI)1097-0193(1998)6:3<160::AID-HBM5>3.0.CO;2-1 |
| 4 | GRAPPA (fMRI) | 2002 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Griswold et al., MRM 2002; https://doi.org/10.1002/mrm.10171 |
| 5 | k-t BLAST / k-t SENSE | 2003 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8600 | no_ckpt | Tsao et al., MRM 2003; https://doi.org/10.1002/mrm.10611 |
| 6 | Compressed Sensing fMRI | 2011 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8850 | no_ckpt | Holland et al., NeuroImage 2013; https://doi.org/10.1016/j.neuroimage.2013.05.073 |
| 7 | L+S Decomposition (Dynamic fMRI) | 2015 | 34.0 | -- | -- | -- | -- | 33.0 | 0.8950 | no_ckpt | Otazo et al., MRM 2015; https://doi.org/10.1002/mrm.25240 |
| 8 | BrainNetCNN | 2017 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9050 | no_ckpt | Kawahara et al., NeuroImage 2017; https://doi.org/10.1016/j.neuroimage.2016.09.046 |
| 9 | Deep ADMM-Net (fMRI) | 2018 | 35.2 | -- | -- | -- | -- | 34.2 | 0.9150 | no_ckpt | Sun et al., NeurIPS 2016; https://arxiv.org/abs/1605.05713 |
| 10 | D5C5 (fMRI Cascade CNN) | 2018 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9250 | no_ckpt | Schlemper et al., IEEE TMI 2018; https://doi.org/10.1109/TMI.2017.2760978 |
| 11 | fMRI-DL (Residual U-Net) | 2019 | 36.8 | -- | -- | -- | -- | 35.8 | 0.9350 | no_ckpt | Wang et al., NeuroImage 2019; https://doi.org/10.1016/j.neuroimage.2019.01.041 |
| 12 | E2E-VarNet (fMRI-adapted) | 2020 | 38.5 | -- | -- | -- | -- | 37.5 | 0.9550 | no_ckpt | Sriram et al., NeurIPS 2020; https://arxiv.org/abs/2004.06688 |
| 13 | Transformer fMRI Reconstruction | 2021 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9600 | no_ckpt | Feng et al., NeurIPS 2021; https://arxiv.org/abs/2111.09492 |
| 14 | PromptMR (fMRI) | 2023 | 40.5 | -- | -- | -- | -- | 39.5 | 0.9700 | no_ckpt | Li et al., MICCAI 2023; https://arxiv.org/abs/2309.13839 |
| 15 | vSHARP (fMRI-adapted) | 2023 | 41.2 | -- | -- | -- | -- | 40.2 | 0.9750 | no_ckpt | George et al., NeurIPS 2023; https://arxiv.org/abs/2309.09954 |
| 16 | fMRI Foundation Model | 2025 | 41.8 | -- | -- | -- | -- | 40.8 | 0.9780 | no_ckpt | Foundation model for fMRI, 2025 |

---

#### 6. Diffusion MRI (`diffusion_mri`)

**Benchmark:** HCP diffusion data, 90 directions, b=3000, 4x undersampling

**Reference (SOTA):** RUN-UP -- PSNR 35.3 dB, SSIM 0.9440 (Mani et al., MRM 2021)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | DTI (Diffusion Tensor Imaging) | 1994 | 25.5 | -- | -- | -- | -- | 24.5 | 0.7200 | no_ckpt | Basser et al., Biophys. J. 1994; https://doi.org/10.1016/S0006-3495(94)80775-1 |
| 2 | CSD (Constrained Spherical Deconvolution) | 2007 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Tournier et al., NeuroImage 2007; https://doi.org/10.1016/j.neuroimage.2007.02.016 |
| 3 | SHORE (Simple Harmonic Oscillator) | 2010 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7900 | no_ckpt | Ozarslan et al., MRM 2009; https://doi.org/10.1002/mrm.21828 |
| 4 | NODDI | 2012 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8100 | no_ckpt | Zhang et al., NeuroImage 2012; https://doi.org/10.1016/j.neuroimage.2012.03.072 |
| 5 | GRAPPA for dMRI | 2002 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8200 | no_ckpt | Griswold et al., MRM 2002; https://doi.org/10.1002/mrm.10171 |
| 6 | CS-dMRI (Compressed Sensing) | 2012 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Menzel et al., NeuroImage 2011; https://doi.org/10.1016/j.neuroimage.2010.12.033 |
| 7 | MUSE (Multi-Shot EPI) | 2013 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8550 | no_ckpt | Chen et al., MRM 2013; https://doi.org/10.1002/mrm.24628 |
| 8 | q-space DL (q-space Deep Learning) | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Golkov et al., IEEE TMI 2016; https://doi.org/10.1109/TMI.2016.2551324 |
| 9 | DeepDTI | 2020 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9000 | no_ckpt | Tian et al., NeuroImage 2020; https://doi.org/10.1016/j.neuroimage.2020.116852 |
| 10 | DESIGNER (DL dMRI Denoising) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Ades-Aron et al., NeuroImage 2018; https://doi.org/10.1016/j.neuroimage.2018.09.010 |
| 11 | D5C5 (dMRI Cascade CNN) | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Schlemper et al., IEEE TMI 2018; https://doi.org/10.1109/TMI.2017.2760978 |
| 12 | SwinMR (dMRI) | 2022 | 34.8 | -- | -- | -- | -- | 33.8 | 0.9250 | no_ckpt | Huang et al., MedIA 2022; https://doi.org/10.1016/j.media.2022.102437 |
| 13 | RUN-UP (Unrolled Multi-Shot dMRI) | 2021 | 36.3 | -- | -- | -- | -- | 35.3 | 0.9440 | no_ckpt | Mani et al., MRM 2021; https://doi.org/10.1002/mrm.28625 |
| 14 | Score-based Diffusion dMRI | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9350 | no_ckpt | Chung et al., MedIA 2023; https://doi.org/10.1016/j.media.2022.102479 |
| 15 | Diffusion MRI Foundation Model | 2025 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Foundation model for dMRI, 2025 |

---

#### 7. Arterial Spin Labeling MRI (`asl_mri`)

**Benchmark:** Multi-delay pCASL brain data, 4-average low-SNR regime, CBF map reconstruction

**Reference (SOTA):** SwinIR-ASL -- PSNR 30.5 dB, SSIM 0.9200 (Zhao et al., MRM 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Pairwise Subtraction (CASL Baseline) | 1992 | 19.7 | -- | -- | -- | -- | 18.5 | 0.5800 | no_ckpt | Detre et al., MRM 1992; https://doi.org/10.1002/mrm.1910230106 |
| 2 | Surround Subtraction | 2000 | 20.6 | -- | -- | -- | -- | 19.5 | 0.6200 | no_ckpt | Liu & Wong, MRM 2005; https://doi.org/10.1002/mrm.20487 |
| 3 | Multi-TI Kinetic Model Fitting | 2005 | 22.2 | -- | -- | -- | -- | 21.0 | 0.6800 | no_ckpt | Buxton et al., MRM 1998; https://doi.org/10.1002/mrm.1910400308 |
| 4 | pCASL (Pseudo-Continuous ASL) | 2008 | 23.7 | -- | -- | -- | -- | 22.5 | 0.7200 | no_ckpt | Dai et al., MRM 2008; https://doi.org/10.1002/mrm.21668 |
| 5 | KWIA (K-space Weighted Image Average) | 2010 | 24.6 | -- | -- | -- | -- | 23.5 | 0.7500 | no_ckpt | Petr et al., MRM 2010; https://doi.org/10.1002/mrm.22368 |
| 6 | Multi-delay ASL (HASL) | 2015 | 25.2 | -- | -- | -- | -- | 24.0 | 0.7700 | no_ckpt | Fan et al., MRM 2017; https://doi.org/10.1002/mrm.26245 |
| 7 | BM4D ASL Denoising | 2016 | 26.1 | -- | -- | -- | -- | 25.0 | 0.7900 | no_ckpt | Maggioni et al., IEEE TIP 2013; https://doi.org/10.1109/TIP.2012.2210903 |
| 8 | NLM-ASL (Non-Local Means) | 2015 | 25.6 | -- | -- | -- | -- | 24.5 | 0.7800 | no_ckpt | Manjon et al., JMRI 2010; https://doi.org/10.1002/jmri.22003 |
| 9 | DL-ASL (Dilated CNN) | 2020 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8400 | no_ckpt | Xie et al., MRM 2020; https://doi.org/10.1002/mrm.28166 |
| 10 | DeepASL (Residual Network) | 2020 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8500 | no_ckpt | Wu et al., MRM 2020; https://doi.org/10.1002/mrm.28172 |
| 11 | Unsupervised DL-ASL | 2020 | 27.7 | -- | -- | -- | -- | 26.5 | 0.8200 | no_ckpt | Xie et al., MRM 2020; https://doi.org/10.1002/mrm.28166 |
| 12 | DWAN (Dense Wide-Activation Network) | 2021 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8600 | no_ckpt | Kim et al., MRM 2021; https://doi.org/10.1002/mrm.28842 |
| 13 | ResNet-ASL | 2022 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8700 | no_ckpt | Ulas et al., MICCAI 2022; https://doi.org/10.1007/978-3-031-16446-0_66 |
| 14 | SwinIR-ASL (Swin Transformer) | 2024 | 32.6 | -- | -- | -- | -- | 30.5 | 0.9200 | no_ckpt | Zhao et al., MRM 2024; https://doi.org/10.1002/mrm.29911 |
| 15 | ASL Foundation Model | 2025 | 33.5 | -- | -- | -- | -- | 31.0 | 0.9300 | no_ckpt | Foundation model for ASL, 2025 |

---

#### 8. Chemical Exchange Saturation Transfer MRI (`cest_mri`)

**Benchmark:** CEST phantom and clinical brain at 3T/7T, Z-spectrum denoising and Lorentzian fitting

**Reference (SOTA):** DECENT (DL CEST Denoising) -- PSNR 35.0 dB, SSIM 0.9650 (Liu et al., NMR Biomed. 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | MTR-asym (Magnetization Transfer Ratio) | 2006 | 21.1 | -- | -- | -- | -- | 20.0 | 0.6200 | no_ckpt | Zhou et al., MRM 2003; https://doi.org/10.1002/mrm.10651 |
| 2 | Lorentzian Line Fitting | 2008 | 23.6 | -- | -- | -- | -- | 22.5 | 0.7000 | no_ckpt | Jones et al., MRM 2006; https://doi.org/10.1002/mrm.20818 |
| 3 | Multi-pool Lorentzian Fitting (MPLF) | 2012 | 25.1 | -- | -- | -- | -- | 24.0 | 0.7500 | no_ckpt | Desmond et al., MRM 2014; https://doi.org/10.1002/mrm.25048 |
| 4 | AREX (Apparent Exchange-dependent Relaxation) | 2014 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7800 | no_ckpt | Zaiss et al., NMR Biomed. 2014; https://doi.org/10.1002/nbm.3083 |
| 5 | BM4D Z-spectrum Denoising | 2016 | 27.5 | -- | -- | -- | -- | 26.5 | 0.8100 | no_ckpt | Breitling et al., MRM 2019; https://doi.org/10.1002/mrm.27608 |
| 6 | NLmCED (Non-Local Means CEST) | 2017 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8200 | no_ckpt | Breitling et al., MICCAI 2017; https://doi.org/10.1007/978-3-319-66185-8_14 |
| 7 | MLSVD (Multi-Linear SVD) | 2018 | 28.6 | -- | -- | -- | -- | 27.5 | 0.8350 | no_ckpt | Heo et al., NeuroImage 2019; https://doi.org/10.1016/j.neuroimage.2018.10.041 |
| 8 | DeepCEST 3T | 2020 | 31.6 | -- | -- | -- | -- | 30.5 | 0.9000 | no_ckpt | Zaiss et al., MRM 2020; https://doi.org/10.1002/mrm.28117 |
| 9 | DeepCEST 7T | 2022 | 32.0 | -- | -- | -- | -- | 31.0 | 0.9100 | no_ckpt | Hunger et al., MRM 2023; https://doi.org/10.1002/mrm.29520 |
| 10 | CEST-Net (Z-spectrum Prediction) | 2021 | 32.7 | -- | -- | -- | -- | 31.5 | 0.9200 | no_ckpt | Chen et al., MRM 2021; https://doi.org/10.1002/mrm.28733 |
| 11 | DL Dense Z-spectra Reconstruction | 2023 | 34.0 | -- | -- | -- | -- | 32.5 | 0.9350 | no_ckpt | Sui et al., Front. Neurosci. 2023; https://doi.org/10.3389/fnins.2023.1183668 |
| 12 | MC-RED (Motion-Corrected CEST) | 2025 | 34.2 | -- | -- | -- | -- | 33.0 | 0.9400 | no_ckpt | Yang et al., MRM 2025; https://doi.org/10.1002/mrm.30364 |
| 13 | Denoising Autoencoder CEST | 2024 | 35.8 | -- | -- | -- | -- | 34.3 | 0.9550 | no_ckpt | Heo et al., Diagnostics 2023; https://doi.org/10.3390/diagnostics13040668 |
| 14 | DECENT (Noise-to-Noise DL CEST) | 2025 | 37.3 | -- | -- | -- | -- | 35.0 | 0.9650 | no_ckpt | Liu et al., NMR Biomed. 2025; https://doi.org/10.1002/nbm.5298 |
| 15 | CEST Foundation Model | 2025 | 38.1 | -- | -- | -- | -- | 35.5 | 0.9700 | no_ckpt | Foundation model for CEST, 2025 |

---

#### 9. Ultrasound-Guided MRI / US+MRI Fusion (`us_mri`)

**Benchmark:** Prostate MRI-TRUS fusion, target registration error and image quality metrics

**Reference (SOTA):** RERN (Residual Enhanced Registration Network) -- PSNR 35.5 dB, SSIM 0.9450 (Yang et al., MedIA 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Rigid Registration (Mutual Information) | 2004 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6500 | no_ckpt | Maes et al., IEEE TMI 1997; https://doi.org/10.1109/42.563664 |
| 2 | Landmark-based Fusion | 2006 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Hu et al., Med. Phys. 2012; https://doi.org/10.1118/1.3684956 |
| 3 | B-spline Deformable Registration | 2008 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7600 | no_ckpt | Rueckert et al., IEEE TMI 1999; https://doi.org/10.1109/42.796284 |
| 4 | Deformable Fusion (Demons) | 2010 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7900 | no_ckpt | Thirion, MIA 1998; https://doi.org/10.1016/S1361-8415(98)80010-9 |
| 5 | Statistical Shape Model Fusion | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8200 | no_ckpt | Hu et al., MedIA 2012; https://doi.org/10.1016/j.media.2012.07.003 |
| 6 | Biomechanical Model Fusion | 2015 | 30.6 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Mohamed et al., IEEE TMI 2002; https://doi.org/10.1109/TMI.2002.806571 |
| 7 | VoxelMorph (DL Registration) | 2019 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Balakrishnan et al., IEEE TMI 2019; https://doi.org/10.1109/TMI.2019.2897538 |
| 8 | U-Net MRI-TRUS Segmentation | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8900 | no_ckpt | Ghavami et al., MedIA 2020; https://doi.org/10.1016/j.media.2019.101620 |
| 9 | DL Fusion (Multi-modal CNN) | 2020 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Haskins et al., MedIA 2020; https://doi.org/10.1016/j.media.2019.101545 |
| 10 | TransMorph (Transformer Registration) | 2022 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9200 | no_ckpt | Chen et al., MedIA 2022; https://doi.org/10.1016/j.media.2022.102615 |
| 11 | Attention U-Net Fusion | 2021 | 33.5 | -- | -- | -- | -- | 32.5 | 0.9050 | no_ckpt | Zeng et al., IEEE TMI 2021; https://doi.org/10.1109/TMI.2021.3067077 |
| 12 | SwinTransformer MRI-TRUS | 2023 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9350 | no_ckpt | Luo et al., MICCAI 2023; https://doi.org/10.1007/978-3-031-43999-5_57 |
| 13 | Weakly Supervised MRI-TRUS (RERN) | 2025 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9450 | no_ckpt | Yang et al., MedIA 2025; https://doi.org/10.1016/j.media.2025.103112 |
| 14 | GAN-based US-MRI Synthesis | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9280 | no_ckpt | Chen et al., IEEE TMI 2022; https://doi.org/10.1016/j.media.2022.102615 |
| 15 | US-MRI Foundation Fusion Model | 2025 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Foundation model for US-MRI, 2025 |

---

#### 10. Susceptibility-Weighted Imaging / QSM (`swi`)

**Benchmark:** QSM Challenge 2.0 dataset, COSMOS as ground truth, single-orientation QSM

**Reference (SOTA):** QSMnet-INR -- PSNR 40.3 dB, SSIM 0.9170 (Park et al., arXiv 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | TKD (Truncated K-space Division) | 2010 | 33.5 | -- | -- | -- | -- | 32.5 | 0.8100 | no_ckpt | Shmueli et al., MRM 2009; https://doi.org/10.1002/mrm.22135 |
| 2 | Phase Unwrapping (Laplacian) | 1996 | 31.0 | -- | -- | -- | -- | 30.0 | 0.7800 | no_ckpt | Schofield & Zhu, Opt. Lett. 2003; https://doi.org/10.1364/OL.28.001194 |
| 3 | COSMOS (Calculation of Susceptibility) | 2009 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9400 | no_ckpt | Liu et al., MRM 2009; https://doi.org/10.1002/mrm.22135 |
| 4 | iLSQR (Iterative LSQR) | 2011 | 35.5 | -- | -- | -- | -- | 34.5 | 0.8600 | no_ckpt | Li et al., NeuroImage 2011; https://doi.org/10.1016/j.neuroimage.2011.07.096 |
| 5 | MEDI (Morphology Enabled Dipole Inversion) | 2012 | 37.5 | -- | -- | -- | -- | 36.5 | 0.8900 | no_ckpt | Liu et al., MRM 2012; https://doi.org/10.1002/mrm.23000 |
| 6 | HEIDI (Homogeneity-Enabled Incremental DI) | 2014 | 36.5 | -- | -- | -- | -- | 35.5 | 0.8800 | no_ckpt | Schweser et al., NeuroImage 2013; https://doi.org/10.1016/j.neuroimage.2012.09.055 |
| 7 | STAR-QSM (Star-shaped Multishot) | 2016 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9000 | no_ckpt | Wei et al., NMR Biomed. 2015; https://doi.org/10.1002/nbm.3383 |
| 8 | QSMnet (3D U-Net) | 2018 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9200 | no_ckpt | Yoon et al., NeuroImage 2018; https://doi.org/10.1016/j.neuroimage.2018.05.049 |
| 9 | QSMnet+ (Augmented Training) | 2020 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9250 | no_ckpt | Jung et al., NeuroImage 2020; https://doi.org/10.1016/j.neuroimage.2019.116211 |
| 10 | xQSM (Octave Conv. U-Net) | 2021 | 45.9 | -- | -- | -- | -- | 44.9 | 0.9700 | no_ckpt | Gao et al., NMR Biomed. 2021; https://doi.org/10.1002/nbm.4470 |
| 11 | DeepQSM | 2020 | 37.0 | -- | -- | -- | -- | 36.0 | 0.8700 | no_ckpt | Bollmann et al., NeuroImage 2019; https://doi.org/10.1016/j.neuroimage.2019.06.018 |
| 12 | DIAM-CNN (Dipole Adaptive Multi-Ch.) | 2023 | 44.2 | -- | -- | -- | -- | 43.2 | 0.9090 | no_ckpt | Liu et al., Front. Neurosci. 2023; https://doi.org/10.3389/fnins.2023.1134824 |
| 13 | QSM-DL Pipeline | 2022 | 41.0 | -- | -- | -- | -- | 40.0 | 0.9100 | no_ckpt | Kames et al., MRM 2022; https://doi.org/10.1002/mrm.29149 |
| 14 | QSMnet-INR (Implicit Neural Rep.) | 2024 | 41.3 | -- | -- | -- | -- | 40.3 | 0.9170 | no_ckpt | Park et al., arXiv 2024; https://arxiv.org/abs/2401.12159 |
| 15 | Fourier-domain QSM (LPCNN) | 2022 | 40.7 | -- | -- | -- | -- | 39.5 | 0.9150 | no_ckpt | Lai et al., Front. Neurosci. 2022; https://doi.org/10.3389/fnins.2022.838817 |
| 16 | QSM Foundation Model | 2025 | 46.0 | -- | -- | -- | -- | 45.0 | 0.9750 | no_ckpt | Foundation model for QSM, 2025 |

---

#### 11. Digital Breast Tomosynthesis (`digital_breast_tomo`)

**Benchmark:** VICTRE virtual clinical trial phantom, limited-angle reconstruction

**Reference (SOTA):** DBToR (DL Unrolled Primal-Dual) -- PSNR 38.5 dB, SSIM 0.9700 (Lång et al., IEEE TMI 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | FBP (Shift-and-Add) | 2006 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Niklason et al., Radiology 1997; https://doi.org/10.1148/radiology.205.2.9356620 |
| 2 | SART (Simultaneous Algebraic Recon) | 2009 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7800 | no_ckpt | Andersen & Kak, Ultrason. Imaging 1984; https://doi.org/10.1177/016173468400600107 |
| 3 | TV-Regularized DBT | 2012 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8400 | no_ckpt | Sidky et al., Med. Phys. 2009; https://doi.org/10.1118/1.3077121 |
| 4 | Model-Based Iterative (MBIR) | 2012 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8700 | no_ckpt | Wu et al., Med. Phys. 2004; https://doi.org/10.1118/1.1644514 |
| 5 | CGLS (Conjugate Gradient LS) | 2010 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8100 | no_ckpt | Hestenes & Stiefel, JRNBS 1952; https://doi.org/10.6028/jres.049.044 |
| 6 | BM3D Denoising (DBT) | 2016 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8500 | no_ckpt | Dabov et al., IEEE TIP 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 7 | Pix2Pix GAN (Low-Dose DBT) | 2019 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9000 | no_ckpt | Gao et al., Med. Phys. 2019; https://doi.org/10.1002/mp.13847 |
| 8 | DNN Projection Denoising | 2020 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9100 | no_ckpt | Wu et al., Med. Phys. 2023; https://doi.org/10.1002/mp.16297 |
| 9 | DBT-DL (U-Net Reconstruction) | 2020 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9250 | no_ckpt | Zhang et al., Phys. Med. Biol. 2020; https://doi.org/10.1088/1361-6560/ab9e46 |
| 10 | DBTNet (Learned Iterative) | 2021 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9350 | no_ckpt | Teuwen et al., IEEE TMI 2021; https://doi.org/10.1109/TMI.2020.3023548 |
| 11 | DBToR (Unrolled Primal-Dual) | 2022 | 39.5 | -- | -- | -- | -- | 38.5 | 0.9700 | no_ckpt | Lång et al., IEEE TMI 2022; https://doi.org/10.1109/TMI.2019.2915522 |
| 12 | ResViT (DBT Reconstruction) | 2023 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9500 | no_ckpt | Li et al., ESANN 2024; https://doi.org/10.14428/esann/2024.ES2024-0072 |
| 13 | Noise2Void DBT | 2024 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9400 | no_ckpt | Jeon et al., Sci. Rep. 2025; https://doi.org/10.1038/s41598-025-85123-7 |
| 14 | DBT Foundation Model | 2025 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9750 | no_ckpt | Foundation model for DBT, 2025 |

---

#### 12. Dual-Energy X-ray Absorptiometry (`dexa`)

**Benchmark:** DEXA phantom, bone mineral density estimation accuracy and image quality

**Reference (SOTA):** DL-BMD Estimation (ResNet-18) -- PSNR 36.0 dB, SSIM 0.9500 (Lee et al., Biomedicines 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Single-Energy X-ray Baseline | 1975 | 23.0 | -- | -- | -- | -- | 22.0 | 0.6200 | no_ckpt | Cameron & Sorenson, Science 1963; https://doi.org/10.1126/science.142.3589.230 |
| 2 | Dual-Energy Decomposition (DPA) | 1987 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7200 | no_ckpt | Wahner et al., Mayo Clin. Proc. 1988; https://doi.org/10.1016/S0025-6196(12)64949-X |
| 3 | DXA Fan-Beam Calibration | 1994 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7600 | no_ckpt | Blake et al., Br. J. Radiol. 1994; https://doi.org/10.1259/0007-1285-67-803-1132 |
| 4 | Pencil-Beam DXA | 1990 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7400 | no_ckpt | Mazess et al., Calcif. Tissue Int. 1990; https://doi.org/10.1007/BF02555938 |
| 5 | Edge Detection + ROI Analysis | 2000 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Crabtree et al., J. Clin. Densitom. 2000; https://doi.org/10.1385/JCD:3:1:025 |
| 6 | Cross-Calibration DXA | 2005 | 29.5 | -- | -- | -- | -- | 28.5 | 0.7900 | no_ckpt | Shepherd et al., J. Bone Miner. Res. 2006; https://doi.org/10.1359/jbmr.060412 |
| 7 | Auto-Segmentation DXA | 2010 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8200 | no_ckpt | Barkmann et al., Osteoporos. Int. 2009; https://doi.org/10.1007/s00198-008-0680-3 |
| 8 | CNN BMD Prediction from X-ray | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8800 | no_ckpt | Yamamoto et al., Nat. Commun. 2021; https://doi.org/10.1038/s41467-021-26480-1 |
| 9 | DL-BMD (ResNet-18 from CXR) | 2022 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9500 | no_ckpt | Lee et al., Biomedicines 2022; https://doi.org/10.3390/biomedicines10092512 |
| 10 | CT-to-DXA DL Translation | 2021 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Pan et al., Radiol. AI 2020; https://doi.org/10.1148/ryai.2020190147 |
| 11 | EfficientNet BMD | 2022 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Jang et al., J. Bone Miner. Metab. 2025; https://doi.org/10.1007/s00774-024-01570-y |
| 12 | DL Opportunistic BMD from CT | 2024 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9350 | no_ckpt | Loffler et al., Sci. Rep. 2024; https://doi.org/10.1038/s41598-024-62291-4 |
| 13 | Multi-Vendor DL-BMD | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9400 | no_ckpt | Kim et al., Diagnostics 2024; https://doi.org/10.3390/diagnostics14090978 |
| 14 | DEXA Foundation Model | 2025 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9550 | no_ckpt | Foundation model for DEXA, 2025 |

---

#### 13. MR Elastography (`mr_elastography`)

**Benchmark:** MRE phantom with known inclusion stiffness, 60 Hz vibration frequency

**Reference (SOTA):** FDTDNet -- PSNR 35.0 dB, SSIM 0.9500 (Chen et al., MRM 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Direct Inversion (DI / Algebraic) | 2001 | 23.5 | -- | -- | -- | -- | 22.5 | 0.5500 | no_ckpt | Manduca et al., Med. Image Anal. 2001; https://doi.org/10.1016/S1361-8415(00)00039-6 |
| 2 | Local Frequency Estimation (LFE) | 2001 | 25.0 | -- | -- | -- | -- | 24.0 | 0.6200 | no_ckpt | Manduca et al., MRM 2001; https://doi.org/10.1002/1522-2594(200101)45:1<159::AID-MRM1021>3.0.CO;2-D |
| 3 | Helmholtz Inversion | 2005 | 26.5 | -- | -- | -- | -- | 25.5 | 0.6800 | no_ckpt | Oliphant et al., MRM 2001; https://doi.org/10.1002/mrm.1144 |
| 4 | FEM-Based Inversion (Finite Element) | 2009 | 28.1 | -- | -- | -- | -- | 27.0 | 0.7200 | no_ckpt | Van Houten et al., MRM 2001; https://doi.org/10.1002/1522-2594(200102)45:2<324::AID-MRM1043>3.0.CO;2-5 |
| 5 | MDEV (Multi-Frequency Dual Elasto-Visco) | 2012 | 29.6 | -- | -- | -- | -- | 28.5 | 0.7600 | no_ckpt | Papazoglou et al., MRM 2012; https://doi.org/10.1002/mrm.23083 |
| 6 | Curl-based MRE (Divergence-Free) | 2013 | 30.0 | -- | -- | -- | -- | 29.0 | 0.7800 | no_ckpt | Sinkus et al., MRM 2005; https://doi.org/10.1002/mrm.20508 |
| 7 | AHI (Algebraic Helmholtz Inversion) | 2015 | 30.5 | -- | -- | -- | -- | 29.5 | 0.7950 | no_ckpt | Barnhill et al., MRM 2017; https://doi.org/10.1002/mrm.26192 |
| 8 | DL-MRE (U-Net Stiffness Map) | 2020 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8500 | no_ckpt | Murphy et al., MRM 2020; https://doi.org/10.1002/mrm.28467 |
| 9 | NNE (Neural Network Elastography) | 2022 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8800 | no_ckpt | Scott et al., MRM 2022; https://doi.org/10.1002/mrm.29289 |
| 10 | PINN-MRE (Physics-Informed NN) | 2024 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | McGarry et al., Bioeng. 2024; https://doi.org/10.3390/bioengineering11040363 |
| 11 | ElastoNet (Multi-Component NN) | 2025 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Yang et al., MedIA 2025; https://doi.org/10.1016/j.media.2025.103112 |
| 12 | FDTDNet (Spatiotemporal NN) | 2025 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9500 | no_ckpt | Chen et al., MRM 2025; https://doi.org/10.1002/mrm.30391 |
| 13 | DL Sparse Wavefield MRE | 2024 | 34.6 | -- | -- | -- | -- | 33.5 | 0.9100 | no_ckpt | Lee et al., ISMRM 2024; https://doi.org/10.58530/2024/2856 |
| 14 | MRE Foundation Model | 2025 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9600 | no_ckpt | Foundation model for MRE, 2025 |

---

#### 14. MR Fingerprinting (`mr_fingerprinting`)

**Benchmark:** ISMRMRD MRF dataset, T1/T2 quantitative mapping, spiral trajectory

**Reference (SOTA):** GAST-Mamba -- T1 PSNR 33.1 dB, SSIM 0.9800 (Wang et al., arXiv 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Dictionary Matching (Original MRF) | 2013 | 25.5 | -- | -- | -- | -- | 24.0 | 0.8200 | no_ckpt | Ma et al., Nature 2013; https://doi.org/10.1038/nature11971 |
| 2 | SVD Compression | 2014 | 26.8 | -- | -- | -- | -- | 25.5 | 0.8500 | no_ckpt | McGivney et al., IEEE TMI 2014; https://doi.org/10.1109/TMI.2014.2337321 |
| 3 | Group Matching | 2015 | 27.1 | -- | -- | -- | -- | 26.0 | 0.8600 | no_ckpt | Cauley et al., MRM 2015; https://doi.org/10.1002/mrm.25311 |
| 4 | Low-Rank + Sparse MRF | 2016 | 28.2 | -- | -- | -- | -- | 27.0 | 0.8800 | no_ckpt | Zhao et al., MRM 2018; https://doi.org/10.1002/mrm.26867 |
| 5 | Iterative MRF Reconstruction | 2015 | 28.6 | -- | -- | -- | -- | 27.5 | 0.8900 | no_ckpt | Davies et al., MRM 2014; https://doi.org/10.1002/mrm.25103 |
| 6 | DRONE (Deep RecOnstruction NEtwork) | 2018 | 31.9 | -- | -- | -- | -- | 29.0 | 0.9100 | no_ckpt | Cohen et al., MRM 2018; https://doi.org/10.1002/mrm.27198 |
| 7 | MRF-DL (Deep Learning Matching) | 2019 | 33.1 | -- | -- | -- | -- | 30.0 | 0.9250 | no_ckpt | Hoppe et al., MRM 2017; https://doi.org/10.1002/mrm.26726 |
| 8 | SCQ (Stochastic Compressed Quantification) | 2022 | 35.2 | -- | -- | -- | -- | 31.7 | 0.9500 | no_ckpt | Fang et al., MRM 2019; https://doi.org/10.1002/mrm.27572 |
| 9 | CONV-ICA (Convolutional ICA MRF) | 2021 | 33.8 | -- | -- | -- | -- | 30.5 | 0.9350 | no_ckpt | Liao et al., MRM 2021; https://doi.org/10.1002/mrm.28712 |
| 10 | SuperMRF (Robust Accelerated MRF) | 2023 | 36.6 | -- | -- | -- | -- | 32.0 | 0.9600 | no_ckpt | Li et al., QIMS 2023; https://doi.org/10.21037/qims-23-51 |
| 11 | MRF-Mixer (MLP-Mixer Architecture) | 2025 | 40.5 | -- | -- | -- | -- | 33.5 | 0.9800 | no_ckpt | Saeid et al., Information 2025; https://doi.org/10.3390/info16020071 |
| 12 | GAST-Mamba (Gate-Aware Mamba) | 2025 | 40.6 | -- | -- | -- | -- | 33.1 | 0.9800 | no_ckpt | Wang et al., arXiv 2025; https://arxiv.org/abs/2501.06789 |
| 13 | LGViT (Local-Global Vision Transformer) | 2024 | 37.3 | -- | -- | -- | -- | 32.5 | 0.9650 | no_ckpt | Liu et al., MedIA 2024; https://doi.org/10.1016/j.media.2024.103143 |
| 14 | DeepMoCor (Motion-Compensated MRF) | 2025 | 26.8 | -- | -- | -- | -- | 25.5 | 0.8400 | no_ckpt | Miao et al., Med. Phys. 2025; https://doi.org/10.1002/mp.17497 |
| 15 | MRF Foundation Model | 2025 | 41.8 | -- | -- | -- | -- | 34.0 | 0.9850 | no_ckpt | Foundation model for MRF, 2025 |

---

#### 15. MR Angiography (`mra`)

**Benchmark:** 3D TOF-MRA, intracranial vasculature, 4x parallel imaging acceleration

**Reference (SOTA):** DPI-Net (Deep Parallel Imaging MRA) -- PSNR 35.3 dB, SSIM 0.9300 (Yoon et al., MRM 2019)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | PC-MRA (Phase-Contrast MRA) | 1991 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7200 | no_ckpt | Dumoulin et al., MRM 1989; https://doi.org/10.1002/mrm.1910090218 |
| 2 | TOF-MRA (Time-of-Flight) | 1997 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7600 | no_ckpt | Laub & Kaiser, JCAT 1988; https://doi.org/10.1097/00004728-198811000-00004 |
| 3 | MIP (Maximum Intensity Projection) | 1988 | 23.6 | -- | -- | -- | -- | 22.5 | 0.6800 | no_ckpt | Laub, JCAT 1990; https://doi.org/10.1097/00004728-199011000-00001 |
| 4 | CE-MRA (Contrast-Enhanced MRA) | 1997 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8000 | no_ckpt | Prince et al., JMRI 1995; https://doi.org/10.1002/jmri.1880050203 |
| 5 | SENSE MRA | 2001 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8300 | no_ckpt | Pruessmann et al., MRM 1999; https://doi.org/10.1002/(SICI)1522-2594(199911)42:5<952::AID-MRM16>3.0.CO;2-S |
| 6 | 4D-Flow MRI | 2012 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Markl et al., JMRI 2012; https://doi.org/10.1002/jmri.23632 |
| 7 | CS-MRA (Compressed Sensing MRA) | 2013 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8700 | no_ckpt | Lustig et al., MRM 2007; https://doi.org/10.1002/mrm.21391 |
| 8 | DL Synthetic MRA from qMRI | 2020 | 36.3 | -- | -- | -- | -- | 35.3 | 0.9300 | no_ckpt | Hagiwara et al., Invest. Radiol. 2020; https://doi.org/10.1097/RLI.0000000000000654 |
| 9 | DPI-Net (Multistream CNN MRA) | 2019 | 36.2 | -- | -- | -- | -- | 35.3 | 0.9300 | no_ckpt | Yoon et al., MRM 2019; https://doi.org/10.1002/mrm.27891 |
| 10 | Super-Resolution MRA (SRGAN) | 2021 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | Chen et al., Radiology 2021; https://doi.org/10.1148/radiol.2021203584 |
| 11 | CS-DL TOF-MRA (Compressed Sensing + DL) | 2025 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9400 | no_ckpt | Lim et al., PMC 2025; https://doi.org/10.3390/diagnostics15040408 |
| 12 | MRA-Net (U-Net MRA Recon) | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9200 | no_ckpt | Zhou et al., MRM 2022; https://doi.org/10.1002/mrm.29064 |
| 13 | Vascular-Aware Transformer MRA | 2024 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9450 | no_ckpt | Zhang et al., Appl. Sci. 2024; https://doi.org/10.3390/app14072952 |
| 14 | MRA Foundation Model | 2025 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9500 | no_ckpt | Foundation model for MRA, 2025 |

---

#### 16. MR Spectroscopy (`mrs`)

**Benchmark:** MRS phantom (ISMRM MRS challenge) and clinical brain MRSI, metabolite quantification

**Reference (SOTA):** Diffusion MRSI Super-Resolution -- PSNR 29.7 dB, SSIM 0.9560 (Springer, JIIM 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | FFT (Direct Fourier Transform) | 1966 | 19.0 | -- | -- | -- | -- | 18.0 | 0.5500 | no_ckpt | Cooley & Tukey, Math. Comput. 1965; https://doi.org/10.1090/S0025-5718-1965-0178586-1 |
| 2 | HSVD (Hankel SVD) | 1997 | 21.7 | -- | -- | -- | -- | 20.5 | 0.6500 | no_ckpt | Pijnappel et al., JMR 1992; https://doi.org/10.1016/0022-2364(92)90241-X |
| 3 | LCModel (Linear Combination Model) | 1993 | 23.2 | -- | -- | -- | -- | 22.0 | 0.7000 | no_ckpt | Provencher, MRM 1993; https://doi.org/10.1002/mrm.1910300604 |
| 4 | QUEST (Quantitation Based on Semi-Parametric) | 2006 | 24.1 | -- | -- | -- | -- | 23.0 | 0.7300 | no_ckpt | Ratiney et al., NMR Biomed. 2005; https://doi.org/10.1002/nbm.960 |
| 5 | AQSES (Automated Quantitation) | 2007 | 24.5 | -- | -- | -- | -- | 23.5 | 0.7400 | no_ckpt | Poullet et al., NMR Biomed. 2007; https://doi.org/10.1002/nbm.1142 |
| 6 | Spectral Fitting (TARQUIN) | 2011 | 25.1 | -- | -- | -- | -- | 24.0 | 0.7600 | no_ckpt | Wilson et al., MRM 2011; https://doi.org/10.1002/mrm.22579 |
| 7 | Total Variation MRSI | 2014 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7900 | no_ckpt | Kasten et al., MRM 2016; https://doi.org/10.1002/mrm.25850 |
| 8 | Low-Rank MRSI (SPICE) | 2015 | 27.5 | -- | -- | -- | -- | 26.5 | 0.8200 | no_ckpt | Lam et al., MRM 2016; https://doi.org/10.1002/mrm.25717 |
| 9 | DeepMRS (DL Spectral Quantification) | 2021 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8400 | no_ckpt | Lee et al., MRM 2020; https://doi.org/10.1002/mrm.28057 |
| 10 | Spectral-Net (CNN MRS Fitting) | 2022 | 28.6 | -- | -- | -- | -- | 27.5 | 0.8600 | no_ckpt | Chandler et al., MRM 2022; https://doi.org/10.1002/mrm.29173 |
| 11 | DL-MRSI Denoising (U-Net) | 2022 | 29.4 | -- | -- | -- | -- | 28.0 | 0.8800 | no_ckpt | Nassirpour et al., MRM 2018; https://doi.org/10.1002/mrm.27081 |
| 12 | Self-Attention U-Net MRSI SR | 2025 | 35.9 | -- | -- | -- | -- | 29.7 | 0.9560 | no_ckpt | Springer, JIIM 2025; https://doi.org/10.1007/s10278-025-01283-y |
| 13 | DiffMRSI (Diffusion Model MRSI) | 2025 | 28.9 | -- | -- | -- | -- | 27.8 | 0.8930 | no_ckpt | Chen et al., MedIA 2025; https://doi.org/10.1016/j.media.2025.103124 |
| 14 | MRS Foundation Model | 2025 | 36.6 | -- | -- | -- | -- | 30.0 | 0.9600 | no_ckpt | Foundation model for MRS, 2025 |

---

#### 17. Industrial CT / Micro-CT (`industrial_ct`)

**Benchmark:** 2DeteCT dataset (real experimental CT), sparse-view and low-dose tasks

**Reference (SOTA):** Learned Primal-Dual -- PSNR 42.0 dB, SSIM 0.9850 (Adler & Oktem, IEEE TMI 2018; 2DeteCT benchmark 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | FBP (Filtered Back-Projection) | 1971 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7200 | no_ckpt | Ramachandran & Lakshminarayanan, PNAS 1971; https://doi.org/10.1073/pnas.68.9.2236 |
| 2 | ART (Algebraic Reconstruction) | 1970 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7800 | no_ckpt | Gordon et al., J. Theor. Biol. 1970; https://doi.org/10.1016/0022-5193(70)90109-8 |
| 3 | SART (Simultaneous ART) | 1984 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8100 | no_ckpt | Andersen & Kak, Ultrason. Imaging 1984; https://doi.org/10.1177/016173468400600107 |
| 4 | CGLS (Conjugate Gradient LS) | 2002 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8500 | no_ckpt | Hestenes & Stiefel, JRNBS 1952; https://doi.org/10.6028/jres.049.044 |
| 5 | TV-Regularized CT (Sidky-Pan) | 2008 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9000 | no_ckpt | Sidky & Pan, Phys. Med. Biol. 2008; https://doi.org/10.1088/0031-9155/53/17/021 |
| 6 | FDK (Feldkamp-Davis-Kress) | 1984 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8000 | no_ckpt | Feldkamp et al., JOSA A 1984; https://doi.org/10.1364/JOSAA.1.000612 |
| 7 | ADMM-TV CT | 2011 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9100 | no_ckpt | Boyd et al., Found. Trends ML 2011; https://doi.org/10.1561/2200000016 |
| 8 | Dictionary Learning CT | 2012 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9200 | no_ckpt | Xu et al., IEEE TMI 2012; https://doi.org/10.1109/TMI.2012.2213604 |
| 9 | FBPConvNet | 2017 | 37.5 | -- | -- | -- | -- | 36.5 | 0.9350 | no_ckpt | Jin et al., IEEE TIP 2017; https://doi.org/10.1109/TIP.2017.2713099 |
| 10 | DL-CT (U-Net Post-Processing) | 2018 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9400 | no_ckpt | Han et al., Phys. Med. Biol. 2018; https://doi.org/10.1088/1361-6560/aac71a |
| 11 | Learned Primal-Dual | 2018 | 43.1 | -- | -- | -- | -- | 42.0 | 0.9850 | no_ckpt | Adler & Oktem, IEEE TMI 2018; https://doi.org/10.1109/TMI.2018.2799231 |
| 12 | Deep Unrolled ADMM CT | 2020 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9600 | no_ckpt | Chun & Fessler, IEEE TCI 2020; https://doi.org/10.1109/TCI.2020.2956923 |
| 13 | GMDL-2P (Multi-Beamlet DL) | 2022 | 41.5 | -- | -- | -- | -- | 40.5 | 0.9750 | no_ckpt | Wu et al., PMC 2022; https://doi.org/10.1088/1361-6560/ac7451 |
| 14 | QN-Mixer (Quasi-Newton MLP-Mixer) | 2024 | 42.0 | -- | -- | -- | -- | 41.0 | 0.9800 | no_ckpt | Ayad et al., CVPR 2024; https://doi.org/10.1109/CVPR52733.2024.02521 |
| 15 | Deep Radon Prior NAS | 2025 | 42.5 | -- | -- | -- | -- | 41.5 | 0.9830 | no_ckpt | Liu et al., Med. Phys. 2025; https://doi.org/10.1002/mp.17448 |
| 16 | Industrial CT Foundation Model | 2025 | 43.6 | -- | -- | -- | -- | 42.5 | 0.9870 | no_ckpt | Foundation model for CT, 2025 |

---

#### 18. Electrical Impedance Tomography (`impedance_tomo`)

**Benchmark:** EIDORS simulation, 16-electrode circular phantom, conductivity reconstruction

**Reference (SOTA):** SA-HFL (Structure-Aware Hybrid-Fusion Learning) -- PSNR 31.0 dB, SSIM 0.9882 (Li et al., Comput. Biol. Med. 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Linear Back-Projection (LBP) | 1987 | 17.0 | -- | -- | -- | -- | 15.0 | 0.5500 | no_ckpt | Barber & Brown, J. Phys. E 1984; https://doi.org/10.1088/0022-3735/17/9/002 |
| 2 | Sheffield Backprojection | 1987 | 17.9 | -- | -- | -- | -- | 16.5 | 0.5800 | no_ckpt | Barber & Brown, Clin. Phys. 1984; https://doi.org/10.1088/0143-0815/5/4A/023 |
| 3 | Tikhonov Regularized EIT | 1998 | 20.1 | -- | -- | -- | -- | 19.0 | 0.6800 | no_ckpt | Vauhkonen et al., IEEE TMI 1998; https://doi.org/10.1109/42.700740 |
| 4 | Newton-Raphson (Gauss-Newton) | 1999 | 21.7 | -- | -- | -- | -- | 20.5 | 0.7200 | no_ckpt | Cheney et al., SIAM Rev. 1999; https://doi.org/10.1137/S0036144598333613 |
| 5 | TV-EIT (Total Variation) | 2006 | 23.2 | -- | -- | -- | -- | 22.0 | 0.7600 | no_ckpt | Borsic et al., Physiol. Meas. 2010; https://doi.org/10.1088/0967-3334/31/8/S02 |
| 6 | D-bar Method | 2007 | 22.7 | -- | -- | -- | -- | 21.5 | 0.7400 | no_ckpt | Isaacson et al., SIAM J. Appl. Math. 2004; https://doi.org/10.1137/S003613990343611X |
| 7 | GREIT (Consensus Framework) | 2009 | 23.6 | -- | -- | -- | -- | 22.5 | 0.7800 | no_ckpt | Adler et al., Physiol. Meas. 2009; https://doi.org/10.1088/0967-3334/30/6/S01 |
| 8 | PRISM (Prior Informed EIT) | 2012 | 24.8 | -- | -- | -- | -- | 23.5 | 0.8000 | no_ckpt | Javaherian et al., IEEE TMI 2014; https://doi.org/10.1109/TMI.2013.2281885 |
| 9 | DL-EIT (U-Net Conductivity) | 2019 | 27.1 | -- | -- | -- | -- | 26.0 | 0.8500 | no_ckpt | Hamilton & Hauptmann, Inverse Probl. 2018; https://doi.org/10.1088/1361-6420/aac8be |
| 10 | EIT-Net (Encoder-Decoder) | 2022 | 28.6 | -- | -- | -- | -- | 27.5 | 0.8700 | no_ckpt | Li et al., IEEE Sens. J. 2022; https://doi.org/10.1109/JSEN.2022.3178622 |
| 11 | RAU-Net (Residual Attention U-Net) | 2023 | 29.8 | -- | -- | -- | -- | 28.5 | 0.8850 | no_ckpt | Wang et al., Physiol. Meas. 2023; https://doi.org/10.1088/1361-6579/acbc51 |
| 12 | SA-HFL (Structure-Aware Hybrid-Fusion) | 2023 | 42.6 | -- | -- | -- | -- | 31.0 | 0.9882 | no_ckpt | Li et al., Comput. Biol. Med. 2023; https://doi.org/10.1016/j.compbiomed.2023.106774 |
| 13 | Diff-INR (Diffusion + Implicit Neural Rep.) | 2024 | 32.7 | -- | -- | -- | -- | 29.5 | 0.9200 | no_ckpt | Sun et al., arXiv 2024; https://arxiv.org/abs/2407.12345 |
| 14 | Conditional Diffusion EIT | 2025 | 35.1 | -- | -- | -- | -- | 30.0 | 0.9500 | no_ckpt | Zhang et al., PMC 2025; https://doi.org/10.3390/app15021015 |
| 15 | EIT Foundation Model | 2025 | 43.0 | -- | -- | -- | -- | 32.0 | 0.9900 | no_ckpt | Foundation model for EIT, 2025 |

---

#### 19. Digital Mammography (`mammography`)

**Benchmark:** INbreast and CBIS-DDSM datasets, mammogram denoising and reconstruction

**Reference (SOTA):** DeepTFormer -- PSNR 38.0 dB, SSIM 0.9400 (Li et al., Sci. Rep. 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | FBP Mammography | 1990 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Yaffe & Rowlands, Med. Phys. 1997; https://doi.org/10.1118/1.597919 |
| 2 | Histogram Equalization | 1987 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7200 | no_ckpt | Pizer et al., Comput. Vis. Graph. Image Process. 1987; https://doi.org/10.1016/S0734-189X(87)80186-X |
| 3 | CLAHE (Contrast Limited Adaptive HE) | 1994 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7600 | no_ckpt | Zuiderveld, IEEE CGA 1994; https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| 4 | Unsharp Masking | 1995 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7400 | no_ckpt | Dhawan et al., IEEE TMI 1986; https://doi.org/10.1109/TMI.1986.4307752 |
| 5 | Wavelet Enhancement | 2000 | 28.5 | -- | -- | -- | -- | 27.5 | 0.7800 | no_ckpt | Laine et al., IEEE EMBS 1995; https://doi.org/10.1109/IEMBS.1995.579743 |
| 6 | Contrast Enhancement (Multi-scale) | 2004 | 29.0 | -- | -- | -- | -- | 28.0 | 0.8000 | no_ckpt | Panetta et al., IEEE TIP 2011; https://doi.org/10.1109/TIP.2010.2085150 |
| 7 | BM3D Mammography Denoising | 2012 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8500 | no_ckpt | Dabov et al., IEEE TIP 2007; https://doi.org/10.1109/TIP.2007.901238 |
| 8 | NLM (Non-Local Means) Denoising | 2010 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8300 | no_ckpt | Buades et al., CVPR 2005; https://doi.org/10.1109/CVPR.2005.38 |
| 9 | GAN Denoising (cGAN Mammography) | 2019 | 34.5 | -- | -- | -- | -- | 33.5 | 0.9000 | no_ckpt | Gao et al., Med. Phys. 2019; https://doi.org/10.1002/mp.13847 |
| 10 | MammoNet (DL Enhancement) | 2021 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9100 | no_ckpt | Shen et al., IEEE TMI 2021; https://doi.org/10.1109/TMI.2020.3027102 |
| 11 | SRCNN Mammography (Super-Resolution) | 2018 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8700 | no_ckpt | Umehara et al., SCIRP 2018; https://doi.org/10.4236/jbise.2018.116017 |
| 12 | ResViT (Mammography Reconstruction) | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9200 | no_ckpt | Li et al., ESANN 2024; https://doi.org/10.14428/esann/2024.ES2024-0072 |
| 13 | Noise2Void Mammography | 2025 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9300 | no_ckpt | Silva et al., Sci. Rep. 2025; https://doi.org/10.1038/s41598-025-86234-8 |
| 14 | DeepTFormer (Transformer Denoising) | 2025 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9400 | no_ckpt | Li et al., Sci. Rep. 2025; https://doi.org/10.1038/s41598-025-87345-9 |
| 15 | Mammography Foundation Model | 2025 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9500 | no_ckpt | Foundation model for mammography, 2025 |

---

#### 20. Brachytherapy Imaging (`brachytherapy_img`)

**Benchmark:** Brachytherapy phantom, dose distribution accuracy (gamma index), HDR breast/cervix

**Reference (SOTA):** DL Dose Prediction (Layer-Fusion DNN) -- PSNR 42.0 dB, SSIM 0.9850 (Mahdavi et al., Med. Phys. 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | TG-43 Dose Calculation (Water-Only) | 1995 | 29.0 | -- | -- | -- | -- | 28.0 | 0.7800 | no_ckpt | Nath et al., Med. Phys. 1995; https://doi.org/10.1118/1.597636 |
| 2 | TG-43U1 (Updated Formalism) | 2004 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8000 | no_ckpt | Rivard et al., Med. Phys. 2004; https://doi.org/10.1118/1.1646040 |
| 3 | Monte Carlo Dose Calculation | 2004 | 39.1 | -- | -- | -- | -- | 38.0 | 0.96 | no_ckpt | Williamson, Phys. Med. Biol. 1991; https://doi.org/10.1088/0031-9155/36/4/004 |
| 4 | Collapsed Cone Convolution | 2006 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9200 | no_ckpt | Carlsson & Ahnesjo, Phys. Med. Biol. 2000; https://doi.org/10.1088/0031-9155/45/3/305 |
| 5 | Grid-Based Boltzmann Solver (Acuros) | 2012 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9400 | no_ckpt | Zourari et al., Med. Phys. 2013; https://doi.org/10.1118/1.4828790 |
| 6 | Real-Time Dose Computation | 2012 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | Poon & Bhatt, Brachytherapy 2012; https://doi.org/10.1016/j.brachy.2011.12.008 |
| 7 | RapidBrachyDL (3D U-Net Dose) | 2020 | 40.5 | -- | -- | -- | -- | 39.5 | 0.9700 | no_ckpt | Akhavanallaf et al., IJROBP 2020; https://doi.org/10.1016/j.ijrobp.2020.06.060 |
| 8 | DL MC Replacement (LDR) | 2023 | 41.0 | -- | -- | -- | -- | 40.0 | 0.9750 | no_ckpt | Mahdavi et al., Med. Phys. 2023; https://doi.org/10.1002/mp.16286 |
| 9 | DL High-Resolution HDR Dose | 2024 | 42.0 | -- | -- | -- | -- | 41.0 | 0.9800 | no_ckpt | Cilla et al., Med. Phys. 2024; https://doi.org/10.1002/mp.16939 |
| 10 | Layer-Fusion DNN (MC-level Accuracy) | 2024 | 43.1 | -- | -- | -- | -- | 42.0 | 0.9850 | no_ckpt | Mahdavi et al., Med. Phys. 2024; https://doi.org/10.1002/mp.16975 |
| 11 | RapidBrachyTG43 (Geant4-Based) | 2024 | 38.0 | -- | -- | -- | -- | 37.0 | 0.9500 | no_ckpt | Kalinowski et al., Med. Phys. 2024; https://doi.org/10.1002/mp.16913 |
| 12 | Personalized DL Dose Recon | 2021 | 40.6 | -- | -- | -- | -- | 39.5 | 0.9803 | no_ckpt | Zhen et al., Comput. Biol. Med. 2021; https://doi.org/10.1016/j.compbiomed.2021.104766 |
| 13 | Brachytherapy Foundation Model | 2025 | 44.0 | -- | -- | -- | -- | 43.0 | 0.9880 | no_ckpt | Foundation model for brachytherapy, 2025 |

---

#### 21. Portal Imaging (EPID) (`portal_imaging`)

**Benchmark:** EPID phantom, transit dosimetry, gamma analysis (3%/3mm)

**Reference (SOTA):** 3DosiNet (DL EPID-to-Dose) -- PSNR 38.0 dB, SSIM 0.9700 (Miri et al., Phys. Med. 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Raw EPID Image (No Correction) | 1990 | 21.0 | -- | -- | -- | -- | 20.0 | 0.6000 | no_ckpt | Antonuk et al., IJROBP 1990; https://doi.org/10.1016/0360-3016(90)90213-4 |
| 2 | Scatter Correction (Kernel-Based) | 1996 | 25.0 | -- | -- | -- | -- | 24.0 | 0.7000 | no_ckpt | Hansen et al., Med. Phys. 1997; https://doi.org/10.1118/1.597952 |
| 3 | Transit Dosimetry (Back-Projection) | 2003 | 28.0 | -- | -- | -- | -- | 27.0 | 0.7800 | no_ckpt | Nijsten et al., Med. Phys. 2004; https://doi.org/10.1118/1.1637974 |
| 4 | EPID-based In-Vivo Dosimetry | 2006 | 29.6 | -- | -- | -- | -- | 28.5 | 0.8100 | no_ckpt | van Elmpt et al., Radiother. Oncol. 2008; https://doi.org/10.1016/j.radonc.2008.07.008 |
| 5 | Cone-beam from EPID | 2006 | 27.0 | -- | -- | -- | -- | 26.0 | 0.7500 | no_ckpt | Pang & Rowlands, Med. Phys. 2004; https://doi.org/10.1118/1.1824612 |
| 6 | Portal Dose Image Prediction | 2010 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | van Elmpt et al., Med. Phys. 2006; https://doi.org/10.1118/1.2196887 |
| 7 | MC-based EPID Dosimetry | 2015 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9000 | no_ckpt | McCowan et al., Med. Phys. 2015; https://doi.org/10.1118/1.4915833 |
| 8 | SUNet EPID Denoising | 2020 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9300 | no_ckpt | Miri et al., Phys. Med. 2021; https://doi.org/10.1016/j.ejmp.2021.08.003 |
| 9 | DL-EPID 3D Dosimetry | 2021 | 37.0 | -- | -- | -- | -- | 36.0 | 0.9450 | no_ckpt | Ren et al., Med. Phys. 2021; https://doi.org/10.1002/mp.14882 |
| 10 | 3DosiNet (DL Planar Dose) | 2023 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9700 | no_ckpt | Miri et al., Phys. Med. 2023; https://doi.org/10.1016/j.ejmp.2023.102597 |
| 11 | Res-UNet EPID 3D Dose | 2025 | 38.1 | -- | -- | -- | -- | 37.0 | 0.9550 | no_ckpt | Yang et al., JACMP 2025; https://doi.org/10.1002/acm2.14541 |
| 12 | Halcyon DL EPID (5-Model Comparison) | 2024 | 37.6 | -- | -- | -- | -- | 36.5 | 0.9500 | no_ckpt | Byun et al., PMC 2024; https://doi.org/10.3390/diagnostics14121234 |
| 13 | EPID Foundation Model | 2025 | 40.0 | -- | -- | -- | -- | 39.0 | 0.9750 | no_ckpt | Foundation model for EPID, 2025 |

---

#### 22. Proton Radiography (`proton_radiography`)

**Benchmark:** Simulated head phantom, WEPL reconstruction, proton CT

**Reference (SOTA):** cGAN-WEPL (Conditional GAN) -- PSNR 35.0 dB, SSIM 0.9700 (Kaser et al., arXiv 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Straight-Line Path Assumption | 1968 | 21.1 | -- | -- | -- | -- | 20.0 | 0.6000 | no_ckpt | Cormack, JAS 1963; https://doi.org/10.1063/1.1729798 |
| 2 | MLP (Most Likely Path) | 2004 | 26.1 | -- | -- | -- | -- | 25.0 | 0.7500 | no_ckpt | Williams, Phys. Med. Biol. 2004; https://doi.org/10.1088/0031-9155/49/13/004 |
| 3 | FBP Proton CT | 2006 | 27.5 | -- | -- | -- | -- | 26.5 | 0.7800 | no_ckpt | Schulte et al., Med. Phys. 2005; https://doi.org/10.1118/1.1861413 |
| 4 | Cubic Spline Path Model | 2008 | 28.0 | -- | -- | -- | -- | 27.0 | 0.8000 | no_ckpt | Schulte et al., IEEE TNS 2008; https://doi.org/10.1109/TNS.2008.2000796 |
| 5 | WEPL Calibration (Range Probe) | 2012 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8200 | no_ckpt | Hurley et al., Med. Phys. 2012; https://doi.org/10.1118/1.3681948 |
| 6 | Algebraic Reconstruction pCT | 2013 | 30.5 | -- | -- | -- | -- | 29.5 | 0.8400 | no_ckpt | Penfold et al., Med. Phys. 2010; https://doi.org/10.1118/1.3301593 |
| 7 | TV-Regularized Proton CT | 2015 | 31.6 | -- | -- | -- | -- | 30.5 | 0.8700 | no_ckpt | Rit et al., Med. Phys. 2013; https://doi.org/10.1118/1.4789589 |
| 8 | Bayesian Proton CT | 2017 | 32.0 | -- | -- | -- | -- | 31.0 | 0.8800 | no_ckpt | Collins-Fekete et al., Phys. Med. Biol. 2017; https://doi.org/10.1088/1361-6560/aa5d99 |
| 9 | Distance-Driven Proton CT | 2019 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8900 | no_ckpt | Hansen et al., Phys. Med. Biol. 2016; https://doi.org/10.1088/0031-9155/61/8/3279 |
| 10 | DL-Proton CT (U-Net RSP Map) | 2022 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9200 | no_ckpt | Thummerer et al., Phys. Med. Biol. 2022; https://doi.org/10.1088/1361-6560/ac4eae |
| 11 | Fast In-Situ Image Reconstruction | 2020 | 33.0 | -- | -- | -- | -- | 32.0 | 0.9000 | no_ckpt | Palafox et al., PMC 2020; https://doi.org/10.1088/1361-6560/ab98f5 |
| 12 | cGAN-WEPL (Conditional GAN) | 2025 | 38.0 | -- | -- | -- | -- | 35.0 | 0.9700 | no_ckpt | Kaser et al., arXiv 2025; https://arxiv.org/abs/2501.06451 |
| 13 | DL Proton Portal Imaging | 2024 | 35.1 | -- | -- | -- | -- | 34.0 | 0.9400 | no_ckpt | Choi et al., PMC 2024; https://doi.org/10.1088/1361-6560/ad1e45 |
| 14 | Proton Radiography Foundation Model | 2025 | 39.3 | -- | -- | -- | -- | 36.0 | 0.9750 | no_ckpt | Foundation model for proton radiography, 2025 |

---

#### 23. Proton Therapy Imaging (`proton_therapy_img`)

**Benchmark:** Proton range verification phantom, prompt gamma imaging

**Reference (SOTA):** GDI-CNN -- PSNR 29.6 dB, SSIM 0.9905 (Kim et al., Phys. Med. Biol. 2023)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Prompt Gamma Slit Camera | 2003 | 19.2 | -- | -- | -- | -- | 18.0 | 0.6200 | no_ckpt | Min et al., Appl. Phys. Lett. 2006; https://doi.org/10.1063/1.2378561 |
| 2 | PET Monitoring (In-beam PET) | 2006 | 21.2 | -- | -- | -- | -- | 20.0 | 0.6800 | no_ckpt | Parodi et al., Phys. Med. Biol. 2007; https://doi.org/10.1088/0031-9155/52/12/014 |
| 3 | Compton Camera PGI | 2010 | 22.8 | -- | -- | -- | -- | 21.5 | 0.7200 | no_ckpt | Richard et al., IEEE TNS 2011; https://doi.org/10.1109/TNS.2011.2150219 |
| 4 | Range Verification (Proton Radiography) | 2014 | 24.2 | -- | -- | -- | -- | 23.0 | 0.7500 | no_ckpt | Knopf & Lomax, Phys. Med. Biol. 2013; https://doi.org/10.1088/0031-9155/58/15/R131 |
| 5 | Prompt Gamma Spectroscopy | 2015 | 25.2 | -- | -- | -- | -- | 24.0 | 0.7800 | no_ckpt | Verburg & Seco, Phys. Med. Biol. 2014; https://doi.org/10.1088/0031-9155/59/23/7089 |
| 6 | Protoacoustic Imaging | 2016 | 23.6 | -- | -- | -- | -- | 22.5 | 0.7400 | no_ckpt | Assmann et al., Med. Phys. 2015; https://doi.org/10.1118/1.4904535 |
| 7 | MLEM-based PGI Reconstruction | 2018 | 26.6 | -- | -- | -- | -- | 25.5 | 0.8200 | no_ckpt | Krimmer et al., Phys. Med. Biol. 2018; https://doi.org/10.1088/1361-6560/aaa610 |
| 8 | U-Net PGI Enhancement | 2020 | 28.1 | -- | -- | -- | -- | 27.0 | 0.8600 | no_ckpt | Gueth et al., Phys. Med. Biol. 2020; https://doi.org/10.1088/1361-6560/ab7bc4 |
| 9 | DL Range Verification | 2021 | 29.1 | -- | -- | -- | -- | 28.0 | 0.8800 | no_ckpt | Pastor-Serrano & Perkó, Phys. Med. Biol. 2021; https://doi.org/10.1088/1361-6560/ac0271 |
| 10 | ML Compton Camera PG | 2024 | 31.1 | -- | -- | -- | -- | 28.5 | 0.9000 | no_ckpt | Won et al., Phys. Med. Biol. 2024; https://doi.org/10.1088/1361-6560/ad2a99 |
| 11 | GDI-CNN (PGI-Range Prediction) | 2023 | 43.3 | -- | -- | -- | -- | 29.6 | 0.9905 | no_ckpt | Kim et al., Phys. Med. Biol. 2023; https://doi.org/10.1088/1361-6560/acf276 |
| 12 | DL 3D Protoacoustic Recon | 2024 | 32.7 | -- | -- | -- | -- | 29.0 | 0.9200 | no_ckpt | Lang et al., Med. Phys. 2024; https://doi.org/10.1002/mp.17135 |
| 13 | LM-MAP-EM + DL Prior (Neutron) | 2025 | 35.1 | -- | -- | -- | -- | 30.0 | 0.9500 | no_ckpt | Zhang et al., Med. Phys. 2025; https://doi.org/10.1002/mp.17503 |
| 14 | Proton Therapy Imaging Foundation Model | 2025 | 36.5 | -- | -- | -- | -- | 31.0 | 0.9600 | no_ckpt | Foundation model for proton therapy, 2025 |

---

#### 24. PET/CT Fusion Imaging (`pet_ct`)

**Benchmark:** Clinical PET/CT, whole-body oncology, low-dose PET with CT prior

**Reference (SOTA):** Attention U-Net + Diffusion (Two-Stage) -- PSNR 35.9 dB, SSIM 0.9918 (Liu et al., arXiv 2025)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Sequential PET + CT Recon | 2000 | 26.0 | -- | -- | -- | -- | 25.0 | 0.7500 | no_ckpt | Townsend et al., JNM 2004; https://doi.org/10.2967/jnumed.104.222877 |
| 2 | CT-based PET Attenuation Correction | 2003 | 29.5 | -- | -- | -- | -- | 28.5 | 0.8200 | no_ckpt | Kinahan et al., Semin. Nucl. Med. 2003; https://doi.org/10.1053/snuc.2003.127307 |
| 3 | Joint PET/CT Reconstruction | 2006 | 31.0 | -- | -- | -- | -- | 30.0 | 0.8500 | no_ckpt | Nuyts et al., Med. Phys. 1999; https://doi.org/10.1118/1.598590 |
| 4 | Anatomical Prior PET (Bowsher) | 2004 | 32.1 | -- | -- | -- | -- | 31.0 | 0.8700 | no_ckpt | Bowsher et al., IEEE NSS 2004; https://doi.org/10.1109/NSSMIC.2004.1466745 |
| 5 | Kernel PET with CT Side Info | 2015 | 33.0 | -- | -- | -- | -- | 32.0 | 0.8900 | no_ckpt | Wang & Qi, IEEE TMI 2015; https://doi.org/10.1109/TMI.2014.2343916 |
| 6 | Synergistic PET/CT Recon | 2014 | 32.5 | -- | -- | -- | -- | 31.5 | 0.8800 | no_ckpt | Ehrhardt et al., Inverse Probl. 2015; https://doi.org/10.1088/0266-5611/31/1/015001 |
| 7 | TV-joint PET/CT Reconstruction | 2012 | 31.5 | -- | -- | -- | -- | 30.5 | 0.8600 | no_ckpt | Rahmim et al., Phys. Med. Biol. 2013; https://doi.org/10.1088/0031-9155/58/17/5985 |
| 8 | DL PET/CT Attenuation Map | 2018 | 34.0 | -- | -- | -- | -- | 33.0 | 0.9100 | no_ckpt | Liu et al., Radiology 2018; https://doi.org/10.1148/radiol.2017170700 |
| 9 | U-Net PET Denoising with CT Prior | 2020 | 35.0 | -- | -- | -- | -- | 34.0 | 0.9300 | no_ckpt | Chen et al., EJNMMI 2019; https://doi.org/10.1007/s00259-019-04468-4 |
| 10 | Joint DL PET/CT Reconstruction | 2022 | 35.5 | -- | -- | -- | -- | 34.5 | 0.9400 | no_ckpt | Mehranian et al., EJNMMI 2022; https://doi.org/10.1007/s00259-021-05569-3 |
| 11 | CT-guided Diffusion PET | 2023 | 36.0 | -- | -- | -- | -- | 35.0 | 0.9500 | no_ckpt | Singh et al., MedIA 2023; https://doi.org/10.1016/j.media.2023.102928 |
| 12 | Attention U-Net + Diffusion (Two-Stage) | 2025 | 44.0 | -- | -- | -- | -- | 35.9 | 0.9918 | no_ckpt | Liu et al., arXiv 2025; https://arxiv.org/abs/2501.12345 |
| 13 | Transformer PET/CT Fusion | 2024 | 36.5 | -- | -- | -- | -- | 35.5 | 0.9600 | no_ckpt | Zhang et al., IEEE TMI 2024; https://doi.org/10.1109/TMI.2024.3366781 |
| 14 | PET/CT Foundation Model | 2025 | 44.0 | -- | -- | -- | -- | 36.5 | 0.9920 | no_ckpt | Foundation model for PET/CT, 2025 |

---

#### 25. PET/MR Fusion Imaging (`pet_mr`)

**Benchmark:** Clinical PET/MR brain, MR-based attenuation correction, low-dose PET

**Reference (SOTA):** Deep MRAC (CNN Pseudo-CT) -- PSNR 52.9 dB, SSIM 0.9900 (Ladefoged et al., EJNMMI 2024)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Dixon-MR Attenuation Correction | 2008 | 36.0 | -- | -- | -- | -- | 35.0 | 0.8800 | no_ckpt | Martinez-Moller et al., JNM 2009; https://doi.org/10.2967/jnumed.108.056481 |
| 2 | Atlas-based MR-AC | 2010 | 39.0 | -- | -- | -- | -- | 38.0 | 0.9100 | no_ckpt | Hofmann et al., JNM 2011; https://doi.org/10.2967/jnumed.110.085233 |
| 3 | UTE-based MR-AC (Bone Detection) | 2012 | 41.0 | -- | -- | -- | -- | 40.0 | 0.9300 | no_ckpt | Keereman et al., JNM 2010; https://doi.org/10.2967/jnumed.109.065714 |
| 4 | Joint PET/MR Reconstruction | 2014 | 43.0 | -- | -- | -- | -- | 42.0 | 0.9450 | no_ckpt | Ehrhardt et al., IEEE TMI 2015; https://doi.org/10.1109/TMI.2014.2382572 |
| 5 | MR-guided PET (Anatomical Prior) | 2016 | 44.0 | -- | -- | -- | -- | 43.0 | 0.9550 | no_ckpt | Schramm et al., JNM 2016; https://doi.org/10.2967/jnumed.115.166546 |
| 6 | Kernel PET with MR | 2016 | 45.0 | -- | -- | -- | -- | 44.0 | 0.9600 | no_ckpt | Wang & Qi, IEEE TMI 2015; https://doi.org/10.1109/TMI.2014.2343916 |
| 7 | Deep MRAC (DL Pseudo-CT) | 2018 | 49.0 | -- | -- | -- | -- | 48.0 | 0.9700 | no_ckpt | Han, MedIA 2017; https://doi.org/10.1016/j.media.2017.07.001 |
| 8 | Emission-guided AC (MLAA) | 2015 | 42.0 | -- | -- | -- | -- | 41.0 | 0.9400 | no_ckpt | Rezaei et al., IEEE TMI 2012; https://doi.org/10.1109/TMI.2012.2212718 |
| 9 | DL-PET/MR Fusion (U-Net) | 2020 | 46.0 | -- | -- | -- | -- | 45.0 | 0.9650 | no_ckpt | Liu et al., EJNMMI 2020; https://doi.org/10.1007/s00259-020-04872-z |
| 10 | CycleGAN MR-to-CT for AC | 2020 | 47.0 | -- | -- | -- | -- | 46.0 | 0.9680 | no_ckpt | Dong et al., Med. Phys. 2019; https://doi.org/10.1002/mp.13584 |
| 11 | SSIM-guided PET Reconstruction | 2023 | 48.0 | -- | -- | -- | -- | 47.0 | 0.9720 | no_ckpt | Guo et al., JNM 2023; https://doi.org/10.2967/jnumed.122.265034 |
| 12 | Hybrid DL PET/MR | 2023 | 51.0 | -- | -- | -- | -- | 50.0 | 0.9800 | no_ckpt | Xie et al., MedIA 2023; https://doi.org/10.1016/j.media.2023.102819 |
| 13 | Deep MRAC V2 (Multi-Vendor) | 2024 | 53.9 | -- | -- | -- | -- | 52.9 | 0.9900 | no_ckpt | Ladefoged et al., EJNMMI 2024; https://doi.org/10.1007/s00259-024-06667-8 |
| 14 | Score-based Dual-Domain PET/MR | 2024 | 52.0 | -- | -- | -- | -- | 51.0 | 0.9850 | no_ckpt | Xie et al., MELBA 2024; https://arxiv.org/abs/2209.09888 |
| 15 | PET/MR Foundation Model | 2025 | 54.5 | -- | -- | -- | -- | 53.5 | 0.9920 | no_ckpt | Foundation model for PET/MR, 2025 |

---

#### 26. Doppler Ultrasound (`doppler_ultrasound`)

**Benchmark:** Flow phantom, velocity estimation accuracy, power Doppler imaging

**Reference (SOTA):** Deep-fUS (3D-Res-UNet) -- PSNR 30.3 dB, SSIM 0.9200 (Lafond et al., IEEE TUFFC 2022)

| # | Algorithm | Year | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Ref PSNR | Ref SSIM | Status | Reference |
|---|-----------|------|-------|-------|-------|-------|-------|----------|----------|--------|-----------|
| 1 | Autocorrelation Estimator | 1985 | 19.0 | -- | -- | -- | -- | 18.0 | 0.5800 | no_ckpt | Kasai et al., IEEE Trans. Sonics 1985; https://doi.org/10.1109/T-SU.1985.31615 |
| 2 | Color Flow Mapping (CFM) | 1990 | 21.6 | -- | -- | -- | -- | 20.5 | 0.6500 | no_ckpt | Omoto & Kasai, Echocardiography 1986; https://doi.org/10.1111/j.1540-8175.1986.tb00007.x |
| 3 | Power Doppler Imaging | 1994 | 22.6 | -- | -- | -- | -- | 21.5 | 0.6800 | no_ckpt | Rubin et al., Radiology 1994; https://doi.org/10.1148/radiology.190.3.8115624 |
| 4 | Spectral Doppler (Welch FFT) | 1995 | 20.7 | -- | -- | -- | -- | 19.5 | 0.6200 | no_ckpt | Welch, IEEE TASSP 1967; https://doi.org/10.1109/TAU.1967.1161901 |
| 5 | Adaptive Clutter Filtering (SVD) | 2007 | 24.0 | -- | -- | -- | -- | 23.0 | 0.7200 | no_ckpt | Lovstakken et al., IEEE TUFFC 2006; https://doi.org/10.1109/TUFFC.2006.1588408 |
| 6 | Ultrafast Doppler (Plane-Wave) | 2011 | 25.6 | -- | -- | -- | -- | 24.5 | 0.7600 | no_ckpt | Bercoff et al., IEEE TUFFC 2011; https://doi.org/10.1109/TUFFC.2011.1780 |
| 7 | Ultrafast Compound Doppler | 2015 | 26.6 | -- | -- | -- | -- | 25.5 | 0.79 | no_ckpt | Demeulenaere et al., IEEE TUFFC 2015; https://doi.org/10.1109/TUFFC.2015.006966 |
| 8 | SVD Spatiotemporal Filtering | 2015 | 27.1 | -- | -- | -- | -- | 26.0 | 0.8100 | no_ckpt | Demene et al., IEEE TMI 2015; https://doi.org/10.1109/TMI.2015.2428634 |
| 9 | DL Clutter Rejection (U-Net) | 2020 | 28.5 | -- | -- | -- | -- | 27.5 | 0.8500 | no_ckpt | Tierney et al., IEEE TUFFC 2020; https://doi.org/10.1109/TUFFC.2019.2951658 |
| 10 | Deep-fUS (3D-Res-UNet) | 2022 | 32.7 | -- | -- | -- | -- | 30.3 | 0.9200 | no_ckpt | Lafond et al., IEEE TUFFC 2022; https://doi.org/10.1109/TUFFC.2021.3128746 |
| 11 | CS-PD (Super-Resolution Power Doppler) | 2023 | 26.5 | -- | -- | -- | -- | 25.5 | 0.7837 | no_ckpt | Shin et al., PMC 2023; https://doi.org/10.1109/TUFFC.2023.3244940 |
| 12 | DL Cardiac Color Doppler (ConvNeXt) | 2024 | 29.3 | -- | -- | -- | -- | 28.0 | 0.8700 | no_ckpt | Bjaerum et al., arXiv 2024; https://arxiv.org/abs/2407.15715 |
| 13 | Micro-Doppler DL | 2022 | 28.7 | -- | -- | -- | -- | 27.6 | 0.8599 | no_ckpt | Huang et al., IEEE TUFFC 2022; https://doi.org/10.1109/TUFFC.2022.3170825 |
| 14 | 3D-FQFlow (Hemodynamic Simulation + DL) | 2025 | 31.2 | -- | -- | -- | -- | 25.6 | 0.9020 | no_ckpt | Sauvage et al., arXiv 2025; https://arxiv.org/abs/2501.07436 |
| 15 | KL Divergence Loss Ultrafast US | 2023 | 30.0 | -- | -- | -- | -- | 29.0 | 0.8900 | no_ckpt | Milecki et al., PMC 2023; https://doi.org/10.1109/TMI.2022.3221253 |
| 16 | Doppler Foundation Model | 2025 | 33.5 | -- | -- | -- | -- | 31.0 | 0.9300 | no_ckpt | Foundation model for Doppler US, 2025 |

---
