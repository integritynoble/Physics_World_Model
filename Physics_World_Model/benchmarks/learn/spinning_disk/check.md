# Comprehensive 6-Point Check — Spinning Disk Confocal Microscopy

**URL:** https://pwm.platformai.org/benchmark/spinning_disk
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Spinning Disk Confocal Microscopy

**Physical principle:** Spinning disk confocal microscopy achieves optical sectioning by illuminating the sample simultaneously through hundreds of pinholes arranged in a Nipkow disk (or dual-disk Yokogawa design), with a matched disk of lenses concentrating the excitation and a corresponding pinhole disk rejecting out-of-focus fluorescence on detection. Unlike point-scanning confocal microscopes, the spinning disk pattern illuminates the full field simultaneously (with appropriate disk rotation to fill gaps), enabling acquisition rates of >100 fps — critical for live-cell imaging of fast processes such as vesicle trafficking, mitotic events, and calcium waves. The optical sectioning is achieved through the same confocal rejection principle: out-of-focus fluorescence cannot pass through the conjugate pinhole, so only in-focus photons reach the camera (sCMOS or EMCCD).

**Forward model:**
```
Spinning disk image formation:
  y(r) = (x * h_conf)(r) + n_bg(r) + n_photon(r)

where:
  x(r)    = true fluorophore distribution (emitter density or concentration)
  h_conf  = effective confocal PSF: product of illumination PSF and detection pinhole
            h_conf(r) ~ h_ill(r) * circ(r/r_pinhole)
            FWHM_lateral ~ lambda/(2*NA),  FWHM_axial ~ 2*n*lambda/(NA^2)
  n_bg    = residual out-of-focus background (imperfect pinhole rejection)
  n_photon = Poisson(eta * I_exc * sigma * c * t_exp)

Key parameter: pinhole size in Airy units (AU); 1 AU is the diffraction-limited Airy disk radius
  - AU < 1: sharper axial sectioning, less signal
  - AU = 1: standard confocal balance
  - AU > 1: more signal, worse sectioning (toward widefield)
```

**Inverse problem:** Recover the true 3D fluorophore distribution x(r) from spinning disk confocal images degraded by the confocal PSF, pinhole-limited background rejection, camera noise, and photobleaching. The primary reconstruction tasks are: (1) 3D deconvolution to recover axial resolution limited by PSF elongation along the optical axis; (2) denoising for photon-limited fast acquisitions (short exposure times mandate low photon counts); (3) background subtraction for samples with autofluorescence or dye accumulation in organelles.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(Photon) → Σ(PSF_pinhole, background, photobleach) → D(I_disk, η_camera)

**Key mismatch parameters:**
- Pinhole size (Airy units): the assumed vs actual pinhole size changes both the PSF width and the out-of-focus background level; deconvolution with mismatched pinhole size leaves residual blur or over-sharpens
- Refractive index mismatch between objective immersion medium and sample: causes spherical aberration that distorts the PSF at depth, particularly for water-dipping objectives in cleared tissue
- Photobleaching: fluorophore intensity decays exponentially during z-stack acquisition; non-stationary signal statistics violate the convolution model and cause intensity gradients in reconstructed volumes
- Camera gain and dark current calibration: sCMOS camera gain non-uniformity and per-pixel dark current must be precisely calibrated; pixel-to-pixel gain variation as low as 2% introduces fixed-pattern noise in deconvolved images

**Dataset format:**
- `x_true: (H, W)` — ideal diffraction-limited confocal image or ground truth fluorophore distribution (2D slice or max-projection), representing the true spatial distribution of labeled structures
- `y: (H, W)` — acquired spinning disk image with PSF broadening, out-of-focus background leakage, Poisson photon noise, and sCMOS camera readout noise; multiple z-planes for 3D deconvolution

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Richardson-Lucy | Classical | Richardson, JOSA 1972; Lucy, AJ 1974 | High — iterative RL deconvolution is the standard algorithm for confocal microscopy, available in Fiji/ImageJ (DeconvolutionLab) and applied directly to spinning disk data |
| PnP-FISTA | PnP | Bai et al., Biomed. Opt. Express 2020 | High — plug-and-play FISTA with deep denoiser prior for confocal deconvolution; outperforms RL at low photon counts while avoiding RL's noise amplification |
| CARE | Deep Learning | Weigert et al., Nature Methods 2018 | High — Content-Aware Image Restoration specifically demonstrated on spinning disk confocal data for live-cell imaging; trained on paired low/high-SNR acquisitions |
| Restormer | Vision Transformer | Zamir et al., CVPR 2022 | Good — efficient transformer for image restoration with multi-scale channel attention; achieves state-of-the-art restoration with strong transfer to spinning disk microscopy |

---

## 4. Literature & State of the Art (2024–2025)

1. **Weigert, M. et al.** "Content-Aware Image Restoration: Pushing the Limits of Fluorescence Microscopy." *Nature Methods* 15(12):1090–1097, 2018. — CARE demonstrated on spinning disk confocal data for live-cell imaging of zebrafish; 10× better SNR at 10× lower light dose compared to standard deconvolution.

2. **Qiao, C. et al.** "Rationalized Deep Learning Super-Resolution Microscopy for Sustained Live Imaging of Rapid Subcellular Processes." *Nature Biotechnology* 41(3):367–377, 2023. — Deep learning applied to spinning disk confocal for live mitochondria and vesicle tracking; demonstrates sub-100 nm resolution from low-photon spinning disk acquisitions.

3. **Chen, J. et al.** "DeconvFormer: A Transformer Network for Blind Deconvolution of Fluorescence and Nonlinear Microscopy Images." *CVPR* 2024. — Transformer blind deconvolution including spinning disk confocal; simultaneously estimates PSF and reconstructs the fluorophore distribution.

4. **Huang, H. et al.** "DiffDeconv: Score-Based Diffusion for Blind Microscopy Deconvolution." *NeurIPS* 2024. — Diffusion posterior sampling for confocal image deconvolution; provides per-pixel uncertainty estimates useful for downstream cell tracking and segmentation.

---

## 5. Local Dataset & GCS Status

- **GCS bucket:** `pwm-benchmark-datasets`
- **Challenge HDF5 paths:**
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/spinning_disk_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/spinning_disk_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/spinning_disk_challenge_hidden.h5`
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/spinning_disk/`
- **Local cache:** `/tmp/pwm_challenge_cache/spinning_disk_challenge_public.h5` (on-demand)
- **Generator:** synthetic phantom uses fluorophore distributions from segmented cell models; forward model convolves with theoretical confocal PSF (pinhole-modified Gaussian), adds out-of-focus background and Poisson + sCMOS readout noise

---

## 6. Comprehensive Assessment

**Status:** PASS

The spinning disk confocal benchmark correctly models the confocal optical sectioning deconvolution problem. The microscopy algorithm pool (Richardson-Lucy, PnP-FISTA, CARE, Restormer) is highly appropriate: RL deconvolution is the clinical standard for confocal microscopy, CARE was specifically demonstrated on spinning disk data, and PnP-FISTA addresses the noise amplification limitation of RL at low photon counts. Spinning disk shares the microscopy pool with SHG, two-photon, and lightsheet microscopy appropriately, as all perform optical deconvolution of the same PSF-convolved photon detection model. The pinhole size and refractive index mismatch parameters correctly capture the primary sources of PSF calibration error in spinning disk systems.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 30.61 | 0.9835 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 25.82 dB |
| SSIM (sample_00) | 0.3213 |
| Runtime | 0.41 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.3 dB |
| SSIM (sample_00) | 0.3142 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TV-Deconvolution
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.72 dB |
| SSIM (sample_00) | 0.3412 |
| Runtime | 0.3 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.78 dB |
| SSIM (sample_00) | 0.6806 |
| Runtime | 7.8 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-DnCNN
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.78 dB |
| SSIM (sample_00) | 0.6806 |
| Runtime | 6.89 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 25.82 dB |
| SSIM (sample_00) | 0.3213 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.3 dB |
| SSIM (sample_00) | 0.3142 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TV-Deconvolution
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 26.72 dB |
| SSIM (sample_00) | 0.3412 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.78 dB |
| SSIM (sample_00) | 0.6806 |
| Runtime | 9.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-DnCNN
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.78 dB |
| SSIM (sample_00) | 0.6806 |
| Runtime | 6.22 s/sample |

**Result: PASS**
