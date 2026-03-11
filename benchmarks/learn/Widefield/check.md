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
