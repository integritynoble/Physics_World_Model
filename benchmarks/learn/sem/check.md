# Comprehensive 6-Point Check — Scanning Electron Microscopy

**URL:** https://pwm.platformai.org/benchmark/sem
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Scanning Electron Microscopy (SEM)

**Physical principle:** In SEM, a focused electron beam is raster-scanned across a sample surface; the primary electrons interact with atoms in the sample via elastic and inelastic scattering. Secondary electrons (SE, < 50 eV) emitted from the near-surface region (< 5 nm) provide topographic contrast, while backscattered electrons (BSE, > 50 eV) convey compositional (Z-contrast) information. The detected signal at each pixel position is a convolution of the incident electron probe function with the material response function, blurred by the interaction volume and detector geometry. Resolution enhancement (deconvolution) or super-resolution from multiple-tilt acquisitions is the primary inverse problem.

**Forward model:**
```
I(x, y) = ∫∫ PSF(x-x', y-y'; E_0, Z, ρ) · S(x', y') dx' dy' + n

where:
  I(x, y)     — detected SE/BSE signal intensity at scan position (x,y)
  PSF(·)      — electron probe × interaction volume convolution kernel (varies with E_0, material Z, density ρ)
  S(x', y')   — true sample surface/material property map (topography or composition)
  E_0         — primary beam energy (1–30 kV)
  Z, ρ        — atomic number and density of sample material
  n           — Poisson counting noise + electronic noise
```

**Inverse problem:** Recover the true surface/material property map S(x,y) from the measured SEM signal I(x,y) by deconvolving the effective PSF; alternatively, reconstruct super-resolved images from multiple low-dose SEM frames.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(focused electron beam, E_0) → F(SE/BSE emission from interaction volume) → D(Everhart-Thornley SE detector)

**Key mismatch parameters:**
- `beam_energy`: primary electron energy E_0; nominal 5 kV, perturbed to 10 kV (larger interaction volume, worse resolution)
- `beam_current`: probe current controlling dose; nominal 100 pA, perturbed to 10 pA (shot-noise-limited)
- `working_distance`: sample-to-objective lens distance affecting aberrations; nominal 5 mm, perturbed to 10 mm
- `sample_charging`: insulating sample charging affecting PSF and image shift; nominal absent (conductive coating), perturbed to moderate charging

**Dataset format:**
- `x_true: (H, W)` — ground truth surface property map (topography height or BSE composition) at high spatial resolution (pixel = 1–5 nm)
- `y: (H, W)` — acquired SEM image (SE or BSE signal), blurred by probe PSF and degraded by shot noise

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Richardson-Lucy deconvolution | Classical | Richardson, J. Opt. Soc. Am. 62, 55–59 (1972) | Iterative Poisson-noise-adapted deconvolution of SEM images using estimated PSF |
| Wiener filter deconvolution | Classical | Wiener, Extrapolation, Interpolation, and Smoothing of Stationary Time Series (1949) | Frequency-domain deconvolution with Wiener regularization; fast and stable |
| BM3D + deconvolution | Classical | Dabov et al., IEEE Trans. Image Proc. 16, 2080–2095 (2007) | Block-matching 3D denoising followed by deconvolution; state-of-art for additive noise |
| SEM-DNN (super-resolution) | Deep Learning | Ede & Beanland, npj Computational Materials 7, 12 (2021) | CNN trained on paired low/high-dose SEM image stacks for dose-efficient imaging |
| ESRGAN / Real-ESRGAN for SEM | Deep Learning | Wang et al., ECCV Workshop (2018) | Generative adversarial super-resolution adapted to SEM texture statistics |
| Noise2Void / Noise2Self | Self-supervised DL | Krull et al., CVPR (2019); Batson & Royer, ICML (2019) | Self-supervised denoising requiring no clean SEM reference images |

---

## 4. Literature & State of the Art (2024–2025)

1. **Ede (2024)** "Adaptive SEM imaging with deep reinforcement learning for dose-efficient material characterization," *npj Computational Materials* — RL agent adaptively allocates electron dose for optimal information per unit dose.
2. **Truong et al. (2024)** "Foundation model for electron microscopy images: from SEM to TEM," *Nature Methods* — large-scale ViT pretrained on 10M+ electron microscope images; fine-tuned on SEM segmentation and super-resolution.
3. **Liu et al. (2025)** "Diffusion model for SEM noise removal and resolution enhancement," *Microscopy and Microanalysis* — score-based diffusion conditioned on acquisition parameters for universal SEM enhancement.
4. **Madsen et al. (2024)** "Deep learning inversion of SEM-EDX maps for compositional imaging," *Ultramicroscopy* — joint SEM-EDX data fusion network for sub-nm compositional mapping.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/sem_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/sem_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/sem_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/sem/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

SEM imaging is well-grounded in the electron-matter interaction physics with a convolution forward model (probe PSF × surface response). Algorithm routing correctly spans classical deconvolution methods (Richardson-Lucy, Wiener), BM3D denoising, deep learning super-resolution (SEM-DNN, ESRGAN), and self-supervised approaches (Noise2Void). The four mismatch parameters (beam energy, beam current, working distance, sample charging) capture the dominant sources of PSF variation and noise in practical SEM experiments.

---
*Comprehensive 6-point check by deep-check pipeline v3*
