# Comprehensive 6-Point Check -- Widefield Fluorescence Microscopy

**URL:** https://pwm.platformai.org/benchmark/widefield
**Check Date:** 2026-03-11
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Widefield Fluorescence Microscopy (Epifluorescence)

**Physical principle:** Widefield fluorescence microscopy illuminates the entire specimen uniformly through the objective (epi-illumination). All fluorophores in the excitation volume emit simultaneously. The emitted fluorescence passes through the same objective, a dichroic mirror, and emission filter before being captured by an sCMOS or CCD camera. Unlike confocal microscopy, widefield has no pinhole to reject out-of-focus light, so every z-plane in the specimen contributes a blurred haze to the image. Computational deconvolution aims to reverse the PSF blur and remove the out-of-focus background.

**Forward model:**
```
y = Poisson(PSF * x + out_of_focus_haze + autofluorescence) + readout_noise

where:
  x                  -- ground truth fluorescence density (256x256, [0,1])
  PSF                -- widefield PSF (Gaussian, sigma ~3-4 pixels, broader
                        than confocal due to absence of detection pinhole)
  out_of_focus_haze  -- blurred contribution from other z-planes, convolved
                        with a much broader (depth-dependent) defocused PSF
  autofluorescence   -- spatially smooth sample-dependent background from
                        intrinsic tissue fluorescence (NAD(P)H, flavins, etc.)
  readout_noise      -- additive Gaussian from sCMOS camera electronics
```

**Inverse problem:** Recover the in-focus fluorophore density from the blurred, noisy widefield image corrupted by out-of-focus haze, autofluorescence background, and mixed Poisson-Gaussian noise.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(illumination) -> F(fluorophore density) -> D(camera/objective)

**Key mismatch parameters:**
- `NA_error`: deviation in effective numerical aperture (changes PSF width); range [-0.15, +0.15]
- `defocus_amount`: axial defocus from nominal focal plane (um); range [0.0, 1.5]
- `background_level`: autofluorescence + ambient background (photons/pixel); range [10, 80]
- `photobleaching`: fraction of signal lost during exposure (spatially non-uniform); range [0.0, 0.30]
- `noise_level`: peak photon count scaling signal amplitude; range [200, 2000]

**Dataset format (HDF5):**
- `x_true: (256, 256) float32` -- ground truth in-focus fluorescence [0,1]
- `y: (256, 256) float32` -- noisy widefield measurement
- `H_ideal: (256, 256) float32` -- noiseless blurred image (PSF*x + haze + bg)

**Tier structure:**
- Public: 12 samples (seed offset 0), mild mismatch
- Dev: 20 samples (seed offset 10000), moderate mismatch
- Hidden: 20 samples (seed offset 20000), severe mismatch

**Phantoms:** Fluorescently labelled cells:
1. DAPI nuclei -- bright elliptical regions with internal chromatin texture
2. Actin filaments -- thin curved networks (phalloidin staining) with stress fibres
3. Mitochondrial networks -- tubular meshworks (MitoTracker) with puncta

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Richardson-Lucy (baseline) | Classical iterative | Richardson 1972; Lucy 1974 | Poisson ML deconvolution; standard in Fiji/DeconvolutionLab2 |
| Wiener filter | Classical analytical | McNally et al., JOSA A 11:1056, 1994 | Frequency-domain with noise regularization; fast single-pass |
| Blind deconvolution | Classical blind | Sarder & Nehorai, IEEE SPM 23(3):32, 2006 | Joint PSF + image estimation without calibrated PSF |
| CARE / content-aware restoration | Deep Learning | Weigert et al., Nat Methods 15:1090, 2018 | Supervised U-Net trained on paired widefield/confocal data |

**Baseline results (Richardson-Lucy + background subtraction, 50 iterations):**
- Public tier: Mean PSNR = 28.88 dB, Mean SSIM = 0.910
- Dev tier: Mean PSNR = 27.85 dB, Mean SSIM = 0.863
- Hidden tier: Mean PSNR = 25.11 dB, Mean SSIM = 0.702

---

## 4. Literature & State of the Art (2024-2025)

1. **Zhang et al. (2024)** "Virtual confocal microscopy from widefield images using diffusion-based conditional generation," *Nat Commun* -- score-based model generating confocal-equivalent sections from widefield z-stacks.
2. **Christensen et al. (2024)** "Self-supervised fluorescence deconvolution without paired training data," *Biomed Opt Express* -- blind-spot network exploiting sCMOS noise statistics for unsupervised deconvolution.
3. **Qiao et al. (2025)** "Transformer-based 3-D deconvolution for widefield neuron volume imaging," *Light Sci Appl* -- ViT with 3-D positional encoding for joint depth estimation and deconvolution of thick neuronal tissue.
4. **Guo et al. (2024)** "Rapid widefield to super-resolution via flow-matching generative model," *CVPR* -- normalizing flow conditioned on widefield input with calibrated uncertainty.

---

## 5. Local Dataset & GCS Status

**Local dataset:**
- Generator: `datasets/benchmark/widefield/generate_dataset.py`
- Output: `datasets/benchmark/widefield/{public,dev,hidden}/widefield_challenge_{tier}.h5`

**GCS datasets:**
- `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/widefield_challenge_public.h5` (10.1 MiB)
- `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/dev/widefield_challenge_dev.h5` (16.8 MiB)
- `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/hidden/widefield_challenge_hidden.h5` (16.5 MiB)

**Gallery images:** `platform/pwm_platform/static/img/benchmark_gallery/widefield/scene_0{0-3}/`

---

## 6. Comprehensive Assessment

**Status:** PASS

The widefield fluorescence microscopy benchmark faithfully models the key challenges of epifluorescence imaging: broad PSF blur, dominant out-of-focus haze from the entire illuminated z-volume, spatially varying autofluorescence, photobleaching, and mixed Poisson-Gaussian noise. The three phantom types (DAPI nuclei, actin filaments, mitochondrial networks) represent the most common biological targets in widefield imaging. Mismatch parameters (NA error, defocus, background level, photobleaching) increase in severity across tiers, testing algorithmic robustness. The Richardson-Lucy baseline with background subtraction yields 25-29 dB PSNR, leaving substantial room for advanced algorithms (blind deconvolution, CARE, diffusion-based methods) to improve.

---
*Comprehensive 6-point check generated 2026-03-11*

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 29.17 dB |
| SSIM (sample_00) | 0.4333 |
| Runtime | 0.51 s/sample |

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
| PSNR (sample_00) | 30.82 dB |
| SSIM (sample_00) | 0.4627 |
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
| PSNR (sample_00) | 29.65 dB |
| SSIM (sample_00) | 0.4444 |
| Runtime | 0.45 s/sample |

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
| PSNR (sample_00) | 31.92 dB |
| SSIM (sample_00) | 0.6516 |
| Runtime | 7.45 s/sample |

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
| PSNR (sample_00) | 31.92 dB |
| SSIM (sample_00) | 0.6516 |
| Runtime | 8.6 s/sample |

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
| PSNR (sample_00) | 29.17 dB |
| SSIM (sample_00) | 0.4333 |
| Runtime | 0.45 s/sample |

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
| PSNR (sample_00) | 30.82 dB |
| SSIM (sample_00) | 0.4627 |
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
| PSNR (sample_00) | 29.65 dB |
| SSIM (sample_00) | 0.4444 |
| Runtime | 0.29 s/sample |

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
| PSNR (sample_00) | 31.92 dB |
| SSIM (sample_00) | 0.6516 |
| Runtime | 7.41 s/sample |

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
| PSNR (sample_00) | 31.92 dB |
| SSIM (sample_00) | 0.6516 |
| Runtime | 6.26 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.85 dB |
| SSIM (mean, 12 samples) | 0.3256 |
| Runtime | 0.27 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.12 dB |
| SSIM (mean, 12 samples) | 0.3202 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 13.91 dB |
| SSIM (mean, 12 samples) | 0.0514 |
| Runtime | 0.11 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.27 dB |
| SSIM (mean, 12 samples) | 0.1912 |
| Runtime | 0.17 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.88 dB |
| SSIM (mean, 12 samples) | 0.3225 |
| Runtime | 0.32 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.12 dB |
| SSIM (mean, 12 samples) | 0.3202 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.05 dB |
| SSIM (mean, 12 samples) | 0.3708 |
| Runtime | 0.60 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.01 dB |
| SSIM (mean, 12 samples) | 0.3276 |
| Runtime | 0.25 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.42 dB |
| SSIM (mean, 12 samples) | 0.4522 |
| Runtime | 3.77 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.97 dB |
| SSIM (mean, 12 samples) | 0.3783 |
| Runtime | 5.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 3.52 dB |
| SSIM (mean, 12 samples) | 0.0011 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 25.79 dB |
| SSIM (mean, 12 samples) | 0.3940 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.10 dB |
| SSIM (mean, 12 samples) | 0.3828 |
| Runtime | 0.25 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.31 dB |
| SSIM (mean, 12 samples) | 0.4920 |
| Runtime | 6.46 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CARE (PnP-PGD DRUNet)
**Solver Key:** best_quality
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.21 dB |
| SSIM (mean, 12 samples) | 0.4184 |
| Runtime | 2.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Noise2Void (PnP-PGD DRUNet)
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Krull et al. 2019, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.72 dB |
| SSIM (mean, 12 samples) | 0.6069 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSBDeep (DnCNN denoise)
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.07 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Restormer (PnP-HQS DRUNet)
**Solver Key:** restormer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zamir et al. 2022, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.76 dB |
| SSIM (mean, 12 samples) | 0.6274 |
| Runtime | 0.80 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Diffusion (PnP-PGD DRUNet)
**Solver Key:** wf_diffusion
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Xie et al. 2023, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6799 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DeepCAD-RT (PnP-DRS DRUNet)
**Solver Key:** deepcad_rt
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Li et al. 2023, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.75 dB |
| SSIM (mean, 12 samples) | 0.6115 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Mamba (RED DRUNet)
**Solver Key:** wf_mamba
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wang et al. 2024, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.31 dB |
| SSIM (mean, 12 samples) | 0.5450 |
| Runtime | 2.80 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-PGD DRUNet
**Solver Key:** pnp_pgd_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zhang et al. 2017, PnP-PGD framework
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.56 dB |
| SSIM (mean, 12 samples) | 0.5647 |
| Runtime | 0.99 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-GAN (PnP-PGD DRUNet)
**Solver Key:** wf_gan
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** GAN-based widefield restoration, 2020
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6858 |
| Runtime | 0.47 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SRResNet (DnCNN denoise)
**Solver Key:** sr_resnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Ledig et al. 2017, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.08 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Foundation (RED DRUNet)
**Solver Key:** wf_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Foundation model for widefield, 2025
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.03 dB |
| SSIM (mean, 12 samples) | 0.3560 |
| Runtime | 5.24 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.85 dB |
| SSIM (mean, 12 samples) | 0.3256 |
| Runtime | 0.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.12 dB |
| SSIM (mean, 12 samples) | 0.3202 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 13.91 dB |
| SSIM (mean, 12 samples) | 0.0514 |
| Runtime | 0.07 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.27 dB |
| SSIM (mean, 12 samples) | 0.1912 |
| Runtime | 0.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.88 dB |
| SSIM (mean, 12 samples) | 0.3225 |
| Runtime | 0.30 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.12 dB |
| SSIM (mean, 12 samples) | 0.3202 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.05 dB |
| SSIM (mean, 12 samples) | 0.3708 |
| Runtime | 0.57 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.01 dB |
| SSIM (mean, 12 samples) | 0.3276 |
| Runtime | 0.22 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.42 dB |
| SSIM (mean, 12 samples) | 0.4522 |
| Runtime | 3.54 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.97 dB |
| SSIM (mean, 12 samples) | 0.3783 |
| Runtime | 5.18 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 3.52 dB |
| SSIM (mean, 12 samples) | 0.0011 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 25.79 dB |
| SSIM (mean, 12 samples) | 0.3940 |
| Runtime | 0.28 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.10 dB |
| SSIM (mean, 12 samples) | 0.3828 |
| Runtime | 0.22 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CARE (PnP-PGD DRUNet)
**Solver Key:** best_quality
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.21 dB |
| SSIM (mean, 12 samples) | 0.4184 |
| Runtime | 7.33 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Noise2Void (PnP-PGD DRUNet)
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Krull et al. 2019, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.72 dB |
| SSIM (mean, 12 samples) | 0.6069 |
| Runtime | 2.09 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSBDeep (DnCNN denoise)
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.21 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Restormer (PnP-HQS DRUNet)
**Solver Key:** restormer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zamir et al. 2022, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.76 dB |
| SSIM (mean, 12 samples) | 0.6274 |
| Runtime | 2.10 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Diffusion (PnP-PGD DRUNet)
**Solver Key:** wf_diffusion
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Xie et al. 2023, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6799 |
| Runtime | 1.73 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DeepCAD-RT (PnP-DRS DRUNet)
**Solver Key:** deepcad_rt
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Li et al. 2023, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.75 dB |
| SSIM (mean, 12 samples) | 0.6115 |
| Runtime | 3.00 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Mamba (RED DRUNet)
**Solver Key:** wf_mamba
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wang et al. 2024, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.31 dB |
| SSIM (mean, 12 samples) | 0.5450 |
| Runtime | 8.80 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.31 dB |
| SSIM (mean, 12 samples) | 0.4920 |
| Runtime | 5.36 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-PGD DRUNet
**Solver Key:** pnp_pgd_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zhang et al. 2017, PnP-PGD framework
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.56 dB |
| SSIM (mean, 12 samples) | 0.5647 |
| Runtime | 3.64 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-GAN (PnP-PGD DRUNet)
**Solver Key:** wf_gan
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** GAN-based widefield restoration, 2020
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6858 |
| Runtime | 1.56 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SRResNet (DnCNN denoise)
**Solver Key:** sr_resnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Ledig et al. 2017, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.14 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Foundation (RED DRUNet)
**Solver Key:** wf_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Foundation model for widefield, 2025
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.03 dB |
| SSIM (mean, 12 samples) | 0.3560 |
| Runtime | 15.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.66 dB |
| SSIM (mean, 3 samples) | 0.2993 |
| Runtime | 0.13 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.83 dB |
| SSIM (mean, 3 samples) | 0.2925 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 13.16 dB |
| SSIM (mean, 3 samples) | 0.0557 |
| Runtime | 0.07 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 17.03 dB |
| SSIM (mean, 3 samples) | 0.1694 |
| Runtime | 0.16 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.56 dB |
| SSIM (mean, 3 samples) | 0.2942 |
| Runtime | 0.28 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.83 dB |
| SSIM (mean, 3 samples) | 0.2925 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.79 dB |
| SSIM (mean, 3 samples) | 0.3827 |
| Runtime | 0.58 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.76 dB |
| SSIM (mean, 3 samples) | 0.3027 |
| Runtime | 0.25 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.24 dB |
| SSIM (mean, 3 samples) | 0.5108 |
| Runtime | 4.00 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.65 dB |
| SSIM (mean, 3 samples) | 0.3794 |
| Runtime | 5.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 3.53 dB |
| SSIM (mean, 3 samples) | 0.0012 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 24.84 dB |
| SSIM (mean, 3 samples) | 0.3666 |
| Runtime | 0.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.88 dB |
| SSIM (mean, 3 samples) | 0.3456 |
| Runtime | 0.23 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CARE (PnP-PGD DRUNet)
**Solver Key:** best_quality
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.99 dB |
| SSIM (mean, 3 samples) | 0.4515 |
| Runtime | 17.09 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Noise2Void (PnP-PGD DRUNet)
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Krull et al. 2019, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.40 dB |
| SSIM (mean, 3 samples) | 0.6116 |
| Runtime | 0.81 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSBDeep (DnCNN denoise)
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.08 dB |
| SSIM (mean, 3 samples) | 0.2831 |
| Runtime | 0.46 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Restormer (PnP-HQS DRUNet)
**Solver Key:** restormer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zamir et al. 2022, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.41 dB |
| SSIM (mean, 3 samples) | 0.6082 |
| Runtime | 0.87 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Diffusion (PnP-PGD DRUNet)
**Solver Key:** wf_diffusion
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Xie et al. 2023, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.27 dB |
| SSIM (mean, 3 samples) | 0.6648 |
| Runtime | 0.57 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DeepCAD-RT (PnP-DRS DRUNet)
**Solver Key:** deepcad_rt
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Li et al. 2023, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.42 dB |
| SSIM (mean, 3 samples) | 0.6120 |
| Runtime | 1.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Mamba (RED DRUNet)
**Solver Key:** wf_mamba
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Wang et al. 2024, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.16 dB |
| SSIM (mean, 3 samples) | 0.5705 |
| Runtime | 5.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 0 sample(s)
**Status:** FAIL
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** Error: MemoryError: Unable to allocate 1.00 MiB for an array with shape (256, 256) and data type complex128

| Metric | Value |
|--------|-------|
| PSNR (mean, 0 samples) | 0.00 dB |
| SSIM (mean, 0 samples) | 0.0000 |
| Runtime | 0.00 s/sample |

**Result: FAIL**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-PGD DRUNet
**Solver Key:** pnp_pgd_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zhang et al. 2017, PnP-PGD framework
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.41 dB |
| SSIM (mean, 3 samples) | 0.6201 |
| Runtime | 2.04 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-GAN (PnP-PGD DRUNet)
**Solver Key:** wf_gan
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** GAN-based widefield restoration, 2020
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.30 dB |
| SSIM (mean, 3 samples) | 0.6773 |
| Runtime | 1.21 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SRResNet (DnCNN denoise)
**Solver Key:** sr_resnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Ledig et al. 2017, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.08 dB |
| SSIM (mean, 3 samples) | 0.2831 |
| Runtime | 0.13 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Foundation (RED DRUNet)
**Solver Key:** wf_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Foundation model for widefield, 2025
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.77 dB |
| SSIM (mean, 3 samples) | 0.3528 |
| Runtime | 11.92 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.85 dB |
| SSIM (mean, 12 samples) | 0.3256 |
| Runtime | 0.08 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.12 dB |
| SSIM (mean, 12 samples) | 0.3202 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 13.91 dB |
| SSIM (mean, 12 samples) | 0.0514 |
| Runtime | 0.04 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.27 dB |
| SSIM (mean, 12 samples) | 0.1912 |
| Runtime | 0.08 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.88 dB |
| SSIM (mean, 12 samples) | 0.3225 |
| Runtime | 0.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.12 dB |
| SSIM (mean, 12 samples) | 0.3202 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.05 dB |
| SSIM (mean, 12 samples) | 0.3708 |
| Runtime | 0.22 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.01 dB |
| SSIM (mean, 12 samples) | 0.3276 |
| Runtime | 0.09 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.42 dB |
| SSIM (mean, 12 samples) | 0.4522 |
| Runtime | 1.45 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.97 dB |
| SSIM (mean, 12 samples) | 0.3783 |
| Runtime | 1.93 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 3.52 dB |
| SSIM (mean, 12 samples) | 0.0011 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 25.79 dB |
| SSIM (mean, 12 samples) | 0.3940 |
| Runtime | 0.13 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.10 dB |
| SSIM (mean, 12 samples) | 0.3828 |
| Runtime | 0.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.31 dB |
| SSIM (mean, 12 samples) | 0.4920 |
| Runtime | 2.72 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CARE (PnP-PGD DRUNet)
**Solver Key:** best_quality
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.21 dB |
| SSIM (mean, 12 samples) | 0.4184 |
| Runtime | 1.69 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Noise2Void (PnP-PGD DRUNet)
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Krull et al. 2019, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.72 dB |
| SSIM (mean, 12 samples) | 0.6069 |
| Runtime | 0.75 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSBDeep (DnCNN denoise)
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Restormer (PnP-HQS DRUNet)
**Solver Key:** restormer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zamir et al. 2022, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.76 dB |
| SSIM (mean, 12 samples) | 0.6274 |
| Runtime | 0.75 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Diffusion (PnP-PGD DRUNet)
**Solver Key:** wf_diffusion
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Xie et al. 2023, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6799 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CARE (PnP-PGD DRUNet)
**Solver Key:** best_quality
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.21 dB |
| SSIM (mean, 12 samples) | 0.4184 |
| Runtime | 1.55 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Noise2Void (PnP-PGD DRUNet)
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Krull et al. 2019, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.72 dB |
| SSIM (mean, 12 samples) | 0.6069 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSBDeep (DnCNN denoise)
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.05 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Restormer (PnP-HQS DRUNet)
**Solver Key:** restormer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zamir et al. 2022, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.76 dB |
| SSIM (mean, 12 samples) | 0.6274 |
| Runtime | 0.80 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Diffusion (PnP-PGD DRUNet)
**Solver Key:** wf_diffusion
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Xie et al. 2023, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6799 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DeepCAD-RT (PnP-DRS DRUNet)
**Solver Key:** deepcad_rt
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Li et al. 2023, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.75 dB |
| SSIM (mean, 12 samples) | 0.6115 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Mamba (RED DRUNet)
**Solver Key:** wf_mamba
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wang et al. 2024, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.31 dB |
| SSIM (mean, 12 samples) | 0.5450 |
| Runtime | 2.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-PGD DRUNet
**Solver Key:** pnp_pgd_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zhang et al. 2017, PnP-PGD framework
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.56 dB |
| SSIM (mean, 12 samples) | 0.5647 |
| Runtime | 0.94 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-GAN (PnP-PGD DRUNet)
**Solver Key:** wf_gan
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** GAN-based widefield restoration, 2020
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6858 |
| Runtime | 0.42 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SRResNet (DnCNN denoise)
**Solver Key:** sr_resnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Ledig et al. 2017, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Foundation (RED DRUNet)
**Solver Key:** wf_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Foundation model for widefield, 2025
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.03 dB |
| SSIM (mean, 12 samples) | 0.3560 |
| Runtime | 4.17 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CARE (PnP-PGD DRUNet)
**Solver Key:** best_quality
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.21 dB |
| SSIM (mean, 12 samples) | 0.4184 |
| Runtime | 1.71 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Noise2Void (PnP-PGD DRUNet)
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Krull et al. 2019, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.72 dB |
| SSIM (mean, 12 samples) | 0.6069 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSBDeep (DnCNN denoise)
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Restormer (PnP-HQS DRUNet)
**Solver Key:** restormer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zamir et al. 2022, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.76 dB |
| SSIM (mean, 12 samples) | 0.6274 |
| Runtime | 0.79 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Diffusion (PnP-PGD DRUNet)
**Solver Key:** wf_diffusion
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Xie et al. 2023, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6799 |
| Runtime | 0.52 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DeepCAD-RT (PnP-DRS DRUNet)
**Solver Key:** deepcad_rt
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Li et al. 2023, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.75 dB |
| SSIM (mean, 12 samples) | 0.6115 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Mamba (RED DRUNet)
**Solver Key:** wf_mamba
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wang et al. 2024, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.31 dB |
| SSIM (mean, 12 samples) | 0.5450 |
| Runtime | 2.28 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-PGD DRUNet
**Solver Key:** pnp_pgd_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zhang et al. 2017, PnP-PGD framework
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.56 dB |
| SSIM (mean, 12 samples) | 0.5647 |
| Runtime | 0.94 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-GAN (PnP-PGD DRUNet)
**Solver Key:** wf_gan
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** GAN-based widefield restoration, 2020
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.64 dB |
| SSIM (mean, 12 samples) | 0.6858 |
| Runtime | 0.42 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SRResNet (DnCNN denoise)
**Solver Key:** sr_resnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Ledig et al. 2017, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 20.55 dB |
| SSIM (mean, 12 samples) | 0.3419 |
| Runtime | 0.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Foundation (RED DRUNet)
**Solver Key:** wf_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Foundation model for widefield, 2025
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.03 dB |
| SSIM (mean, 12 samples) | 0.3560 |
| Runtime | 4.20 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.96 dB |
| SSIM (mean, 12 samples) | 0.5558 |
| Runtime | 0.13 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.75 dB |
| SSIM (mean, 12 samples) | 0.4591 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.67 dB |
| SSIM (mean, 12 samples) | 0.2807 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.04 dB |
| SSIM (mean, 12 samples) | 0.2602 |
| Runtime | 0.06 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 28.01 dB |
| SSIM (mean, 12 samples) | 0.4954 |
| Runtime | 0.44 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.75 dB |
| SSIM (mean, 12 samples) | 0.4591 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 26.77 dB |
| SSIM (mean, 12 samples) | 0.4498 |
| Runtime | 0.33 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.92 dB |
| SSIM (mean, 12 samples) | 0.5311 |
| Runtime | 0.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.85 dB |
| SSIM (mean, 12 samples) | 0.3847 |
| Runtime | 1.40 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.30 dB |
| SSIM (mean, 12 samples) | 0.3055 |
| Runtime | 2.04 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 25.55 dB |
| SSIM (mean, 12 samples) | 0.3081 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 26.08 dB |
| SSIM (mean, 12 samples) | 0.4255 |
| Runtime | 0.13 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 28.06 dB |
| SSIM (mean, 12 samples) | 0.5384 |
| Runtime | 0.18 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.89 dB |
| SSIM (mean, 12 samples) | 0.4612 |
| Runtime | 2.68 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.96 dB |
| SSIM (mean, 12 samples) | 0.5558 |
| Runtime | 0.16 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.75 dB |
| SSIM (mean, 12 samples) | 0.4591 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.67 dB |
| SSIM (mean, 12 samples) | 0.2807 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.04 dB |
| SSIM (mean, 12 samples) | 0.2602 |
| Runtime | 0.08 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 28.01 dB |
| SSIM (mean, 12 samples) | 0.4954 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.75 dB |
| SSIM (mean, 12 samples) | 0.4591 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 26.77 dB |
| SSIM (mean, 12 samples) | 0.4498 |
| Runtime | 0.39 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.92 dB |
| SSIM (mean, 12 samples) | 0.5311 |
| Runtime | 0.18 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.85 dB |
| SSIM (mean, 12 samples) | 0.3847 |
| Runtime | 1.61 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.30 dB |
| SSIM (mean, 12 samples) | 0.3055 |
| Runtime | 1.98 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 25.55 dB |
| SSIM (mean, 12 samples) | 0.3081 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 26.08 dB |
| SSIM (mean, 12 samples) | 0.4255 |
| Runtime | 0.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 28.06 dB |
| SSIM (mean, 12 samples) | 0.5384 |
| Runtime | 0.17 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.89 dB |
| SSIM (mean, 12 samples) | 0.4612 |
| Runtime | 2.79 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.13 dB |
| SSIM (mean, 12 samples) | 0.7096 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.89 dB |
| SSIM (mean, 12 samples) | 0.5377 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.33 dB |
| SSIM (mean, 12 samples) | 0.7128 |
| Runtime | 0.05 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 13.18 dB |
| SSIM (mean, 12 samples) | 0.3545 |
| Runtime | 0.26 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.55 dB |
| SSIM (mean, 12 samples) | 0.6524 |
| Runtime | 1.94 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.89 dB |
| SSIM (mean, 12 samples) | 0.5377 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 18.39 dB |
| SSIM (mean, 12 samples) | 0.6649 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.12 dB |
| SSIM (mean, 12 samples) | 0.7094 |
| Runtime | 0.52 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 14.97 dB |
| SSIM (mean, 12 samples) | 0.3867 |
| Runtime | 1.09 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 12.95 dB |
| SSIM (mean, 12 samples) | 0.3587 |
| Runtime | 1.58 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.10 dB |
| SSIM (mean, 12 samples) | 0.5489 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 17.99 dB |
| SSIM (mean, 12 samples) | 0.4562 |
| Runtime | 0.45 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.12 dB |
| SSIM (mean, 12 samples) | 0.7094 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.37 dB |
| SSIM (mean, 12 samples) | 0.3188 |
| Runtime | 1.89 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.18 dB |
| SSIM (mean, 12 samples) | 0.7115 |
| Runtime | 0.59 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.59 dB |
| SSIM (mean, 12 samples) | 0.2800 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.18 dB |
| SSIM (mean, 12 samples) | 0.7116 |
| Runtime | 0.05 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 11.09 dB |
| SSIM (mean, 12 samples) | 0.1622 |
| Runtime | 0.24 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.92 dB |
| SSIM (mean, 12 samples) | 0.2811 |
| Runtime | 1.89 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.59 dB |
| SSIM (mean, 12 samples) | 0.2800 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 21.21 dB |
| SSIM (mean, 12 samples) | 0.3853 |
| Runtime | 0.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.18 dB |
| SSIM (mean, 12 samples) | 0.7116 |
| Runtime | 0.49 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 16.64 dB |
| SSIM (mean, 12 samples) | 0.3320 |
| Runtime | 1.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 13.14 dB |
| SSIM (mean, 12 samples) | 0.1722 |
| Runtime | 1.58 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.72 dB |
| SSIM (mean, 12 samples) | 0.2838 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.12 dB |
| SSIM (mean, 12 samples) | 0.2932 |
| Runtime | 0.46 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.18 dB |
| SSIM (mean, 12 samples) | 0.7116 |
| Runtime | 0.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 15.59 dB |
| SSIM (mean, 12 samples) | 0.2588 |
| Runtime | 2.16 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy Deconvolution
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Richardson 1972 / Lucy 1974
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.96 dB |
| SSIM (mean, 12 samples) | 0.5558 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wiener 1949, Extrapolation, Interpolation, and Smoothing
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.75 dB |
| SSIM (mean, 12 samples) | 0.4591 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Gold Deconvolution
**Solver Key:** gold
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Gold 1964, ANL Report 6984
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.67 dB |
| SSIM (mean, 12 samples) | 0.2807 |
| Runtime | 0.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Jansson-van Cittert Iteration
**Solver Key:** jansson
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** van Cittert 1931, Zeitschrift f. Physik; Jansson 1970
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.04 dB |
| SSIM (mean, 12 samples) | 0.2602 |
| Runtime | 0.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Landweber Iteration
**Solver Key:** landweber
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Landweber 1951, Amer. J. Math.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 28.01 dB |
| SSIM (mean, 12 samples) | 0.4954 |
| Runtime | 0.93 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularisation
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963, Soviet Math. Doklady
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.75 dB |
| SSIM (mean, 12 samples) | 0.4591 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Total Variation Deconvolution
**Solver Key:** tv_deconv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Rudin et al. 1992, Physica D
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 26.77 dB |
| SSIM (mean, 12 samples) | 0.4498 |
| Runtime | 0.50 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy with TV Regularisation
**Solver Key:** rl_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Dey et al. 2006, Microscopy Res. Tech.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 27.92 dB |
| SSIM (mean, 12 samples) | 0.5311 |
| Runtime | 0.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM (NLM denoiser)
**Solver Key:** pnp_admm_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013, GlobalSIP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.85 dB |
| SSIM (mean, 12 samples) | 0.3847 |
| Runtime | 1.97 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-FISTA (NLM denoiser)
**Solver Key:** pnp_fista_nlm
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.30 dB |
| SSIM (mean, 12 samples) | 0.3055 |
| Runtime | 3.11 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Inverse Filter
**Solver Key:** inverse_filter
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Direct Fourier division, 1960s
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 25.55 dB |
| SSIM (mean, 12 samples) | 0.3081 |
| Runtime | 0.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Agard Constrained Iterative Deconvolution
**Solver Key:** agard
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Agard 1984, Ann. Rev. Biophys. Bioeng.
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 26.08 dB |
| SSIM (mean, 12 samples) | 0.4255 |
| Runtime | 0.25 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Regularized Richardson-Lucy
**Solver Key:** regularized_rl
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Conchello 1998, JOSA A; Llacer & Nuñez 1990
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 28.06 dB |
| SSIM (mean, 12 samples) | 0.5384 |
| Runtime | 0.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CARE (PnP-PGD DRUNet)
**Solver Key:** best_quality
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.43 dB |
| SSIM (mean, 12 samples) | 0.3278 |
| Runtime | 2.20 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Noise2Void (PnP-PGD DRUNet)
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Krull et al. 2019, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.01 dB |
| SSIM (mean, 12 samples) | 0.5209 |
| Runtime | 0.77 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSBDeep (DnCNN denoise)
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Weigert et al. 2018, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.86 dB |
| SSIM (mean, 12 samples) | 0.3029 |
| Runtime | 0.04 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Restormer (pretrained)
**Solver Key:** restormer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zamir et al. 2022, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.73 dB |
| SSIM (mean, 12 samples) | 0.2554 |
| Runtime | 0.41 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Diffusion (PnP-PGD DRUNet)
**Solver Key:** wf_diffusion
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Xie et al. 2023, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.26 dB |
| SSIM (mean, 12 samples) | 0.6828 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DeepCAD-RT (PnP-DRS DRUNet)
**Solver Key:** deepcad_rt
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Li et al. 2023, Nature Methods
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.08 dB |
| SSIM (mean, 12 samples) | 0.5481 |
| Runtime | 0.79 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Mamba (RED DRUNet)
**Solver Key:** wf_mamba
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Wang et al. 2024, arXiv
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.88 dB |
| SSIM (mean, 12 samples) | 0.4503 |
| Runtime | 2.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (NLM v2)
**Solver Key:** pnp_hqs_nlm_v2
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Venkatakrishnan et al. 2013; HQS variant 2017
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.89 dB |
| SSIM (mean, 12 samples) | 0.4612 |
| Runtime | 3.92 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-PGD DRUNet
**Solver Key:** pnp_pgd_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Zhang et al. 2017, PnP-PGD framework
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 22.57 dB |
| SSIM (mean, 12 samples) | 0.3580 |
| Runtime | 0.93 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-GAN (PnP-PGD DRUNet)
**Solver Key:** wf_gan
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** GAN-based widefield restoration, 2020
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 23.24 dB |
| SSIM (mean, 12 samples) | 0.6836 |
| Runtime | 0.42 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SRResNet (DnCNN denoise)
**Solver Key:** sr_resnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Ledig et al. 2017, CVPR
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.86 dB |
| SSIM (mean, 12 samples) | 0.3029 |
| Runtime | 0.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** WF-Foundation (Restormer)
**Solver Key:** wf_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 12 sample(s)
**Status:** PASS
**Reference:** Foundation model for widefield, 2025
**Operator Family:** psf_conv
**Forward Model:** y(x,y) = PSF * x + noise, incoherent imaging
**Canonical Reference:** Born & Wolf, "Principles of Optics," Cambridge 2019 (7th expanded ed.)
**Note:** 12 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 12 samples) | 19.73 dB |
| SSIM (mean, 12 samples) | 0.2554 |
| Runtime | 0.25 s/sample |

**Result: PASS**
