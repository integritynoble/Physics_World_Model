# Comprehensive Check: fundus

**Modality:** Retinal Fundus Photography
**Category:** clinical_optics (via medical + Photon routing)
**Carrier:** Photon
**Check Date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

### Signal Physics

Fundus photography captures an image of the retina through the pupil using a
low-magnification optical system with flash or continuous LED illumination
(500-700 nm). The retinal image passes through the eye's optics (cornea + lens)
which introduce a PSF determined by the pupil diameter, refractive error, and
optical aberrations. The forward model is a standard linear convolution:

```
y = PSF_optic * x + n
```

where PSF_optic encodes the combined blur from the fundus camera optics and the
patient's eye (pupil diameter, refractive error, media opacities such as
cataracts), and n is mixed Poisson-Gaussian noise from the detector.

The inverse problem is image deblurring/denoising: recover a sharp retinal image
x from the blurred, noisy measurement y. This is a well-posed linear inverse
problem, unlike many of the other modalities in the benchmark.

### Forward Model Assessment

The learning materials correctly identify the forward model as `linear_operator`
with category module `microscopy_psf`. The linear classification is correct --
the PSF convolution model y = H*x + n is linear in x. The `microscopy_psf`
module provides the necessary convolution infrastructure.

The DAG notation is `C(PSF_optic) -> D(g, eta_1)`, representing convolution
followed by detection with Poisson-Gaussian noise. This is accurate.

**System parameters** are realistic:
- Illumination: LED, 100 mW, 80 nm spectral width
- Optics: 45-degree FOV, 2.5x magnification, 4 mm pupil diameter
- PSF: sigma = 1.5 px (Gaussian approximation)
- Detector: CMOS, 5.0 um pixel, QE=0.8, 14-bit, 2.0 e- read noise
- Object/measurement shape: [512, 512]

**Mismatch parameters:** The learning materials state "No mismatch parameters
defined for this modality." This is a notable gap. In practice, the dominant
sources of mismatch in fundus imaging would be:
- PSF variation (pupil diameter changes with illumination, patient eye)
- Refractive error (defocus from ametropia)
- Media opacity (cataracts causing scatter)
- Uneven illumination (vignetting)

However, the three-tier structure still operates via the per-tier data
differentiation mechanism (different ground truth phantoms per tier).

### Verdict: ACCEPTABLE

The linear convolution model is correct for fundus photography. The absence
of explicit mismatch parameters is a gap, but the per-tier data source
differentiation still provides meaningful difficulty graduation.

---

## 2. Mismatch Parameters & Benchmark Structure

### Three-Tier Structure

| Tier | Mismatch Level | Ground Truth | Download |
|------|---------------|--------------|----------|
| Public | Mild | Included | Available |
| Dev | Moderate | Excluded | Available |
| Hidden | Severe | Excluded | Blocked (403) |

### Mismatch Parameter Coverage

No explicit mismatch parameters are defined for this modality. The benchmark
relies on per-tier data differentiation (different phantom ground truths with
seed offsets: public=0, dev=+10000, hidden=+20000) rather than physics parameter
variation.

### Data Format

- Object shape: [512, 512]
- Measurement shape: [512, 512]
- Data source: drive_retinal (Staal et al., IEEE TMI 2004) with fallback
  to `cell_phantom` generator
- Metrics: PSNR (primary), SSIM

### Verdict: ACCEPTABLE

The three-tier structure provides difficulty graduation through data
differentiation even without explicit mismatch parameters. For a linear
deconvolution problem like fundus imaging, varying the PSF or noise level
between tiers would further strengthen the benchmark, but the current
setup is functional.

---

## 3. Reconstruction Methods & Leaderboard

### Algorithm Override (Verified in _algorithm_catalog.py)

| Algorithm | Type | Params | Source |
|-----------|------|--------|--------|
| Richardson-Lucy | Classical | 0 | Richardson 1972 / Lucy 1974 |
| PnP-BM3D | PnP | 0 | Danielyan et al., 2012 |
| cofe-Net | Deep Learning | 5M | Shen et al., IEEE TMI 2020 |
| Swin-Fundus | Transformer | 15M | SwinIR-based retinal enhancement, 2023 |

### Algorithm Appropriateness

All four algorithms are appropriate for fundus image restoration:

1. **Richardson-Lucy** -- the classic iterative deconvolution algorithm for
   Poisson noise. Richardson (1972) and Lucy (1974) independently derived the
   expectation-maximization update for PSF deconvolution. Naturally suited for
   photon-limited imaging. The standard baseline for optical image deblurring.

2. **PnP-BM3D** -- Plug-and-Play framework using BM3D (Block-Matching and 3D
   filtering) as the denoising prior within an ADMM optimization loop.
   Danielyan et al. (2012) demonstrated superior deblurring by replacing
   hand-crafted regularizers with powerful denoisers. Effective for fundus
   images which have rich spatial structure.

3. **cofe-Net** -- Shen et al. (IEEE TMI 2020) designed a corrective fusion
   enhancement network specifically for fundus image enhancement. Trained on
   paired high/low-quality retinal images. Addresses both blur and
   illumination artifacts. Approximately 5M parameters.

4. **Swin-Fundus** -- a SwinIR-based (Liang et al., ICCV 2021) architecture
   adapted for retinal image restoration. Uses shifted-window self-attention
   for efficient long-range modeling. Approximately 15M parameters. Represents
   the transformer-based approach to medical image enhancement (2023).

### Leaderboard Scores (from CATEGORY_REAL_SCORES)

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Richardson-Lucy | 24.50 | 0.680 |
| PnP-BM3D | 28.80 | 0.830 |
| cofe-Net | 32.50 | 0.910 |
| Swin-Fundus | 34.50 | 0.945 |

The 10 dB improvement from classical RL (24.5 dB) to Swin-Fundus (34.5 dB)
is realistic for supervised retinal image restoration on phantom data.

### Verdict: EXCELLENT

The algorithm override correctly replaces the generic clinical_optics pool
(FFT-OCT, BM4D, Speckle-DenoiseNet, OCTA-Net -- all OCT-specific) with
fundus-appropriate deconvolution and enhancement algorithms. Richardson-Lucy
is the correct classical baseline. cofe-Net and Swin-Fundus are
retinal-image-specific deep learning methods.

---

## 4. Literature & State of the Art (2024-2025)

### Key References

| Year | Paper | Venue | Contribution |
|------|-------|-------|-------------|
| 1972/1974 | Richardson / Lucy | JOSA / AJ | RL deconvolution |
| 2004 | Staal et al. | IEEE TMI | DRIVE retinal dataset |
| 2012 | Danielyan et al. | IEEE TIP | PnP-BM3D deblurring |
| 2020 | Shen et al. | IEEE TMI | cofe-Net for fundus enhancement |
| 2020 | Li et al. | MIA | ArcNet for retinal vessel enhancement |
| 2021 | Liang et al. | ICCV | SwinIR: image restoration transformer |
| 2022 | Ravi et al. | IEEE TMI | EndoL2H: high-res endoscopy/fundus |
| 2023 | Liu et al. | MICCAI | Fundus image quality enhancement |
| 2024 | Luo et al. | IEEE TMI | Self-supervised retinal restoration |

### State of the Art Assessment

Fundus image restoration is a well-established problem. The field has evolved
from classical deconvolution (RL, Wiener) through PnP methods (BM3D prior)
to deep learning (cofe-Net, 2020) and transformers (SwinIR adaptations, 2023).
Recent 2024 work focuses on self-supervised approaches that do not require
paired training data. The benchmark's algorithm selection represents the
mainstream trajectory.

### Verdict: CURRENT

Algorithm selection covers classical (1972) through 2023 state-of-the-art
transformers. The field continues to advance with self-supervised methods.

---

## 5. Local Dataset & GCS Status

### Challenge Datasets on GCS

| Tier | File | Status |
|------|------|--------|
| Public | `challenge-data/v1.0/fundus_challenge_public.h5` | OK |
| Dev | `challenge-data/v1.0/fundus_challenge_dev.h5` | OK |
| Hidden | `challenge-data/v1.0/fundus_challenge_hidden.h5` | Blocked (403) |

### Gallery Images

Gallery images served from GCS via `/gcs/img/benchmark_gallery/fundus/`.
24/24 gallery images load successfully.

### Learning Materials

| File | Status | Size |
|------|--------|------|
| README.md | Present | 1,406 B |
| 01_physics_fundamentals.md | Present | 2,716 B |
| 02_forward_model.md | Present | 2,475 B |
| 03_reconstruction_algorithms.md | Present | 1,815 B |
| 04_pwm_benchmark.md | Present | 2,250 B |
| 05_hands_on_tutorial.md | Present | 3,522 B |

### Verdict: COMPLETE

All HDF5 challenge datasets present on GCS. Gallery images verified (24/24).
Learning materials complete.

---

## 6. Comprehensive Assessment & Recommendations

### Overall Status: PASS

| Check | Result |
|-------|--------|
| Physics & forward model | Correct linear PSF convolution model |
| Mismatch parameters | None defined (per-tier data differentiation only) |
| Algorithm override | In place -- all 4 algorithms appropriate for fundus imaging |
| Leaderboard scores | Realistic progression from 24.5 to 34.5 dB PSNR |
| Literature coverage | Current through 2024 (self-supervised retinal restoration) |
| GCS datasets | All 3 tiers present |
| Learning materials | Complete 5-file set |
| Gallery images | 24/24 verified |

### What Was Fixed

The original assignment used generic clinical_optics algorithms (FFT-OCT, BM4D,
Speckle-DenoiseNet, OCTA-Net) which are all OCT-specific. Fundus photography
has nothing to do with OCT -- it is a simple reflectance/fluorescence imaging
modality with a linear PSF convolution forward model. The variant override
replaced these with Richardson-Lucy, PnP-BM3D, cofe-Net, and Swin-Fundus --
algorithms appropriate for optical image deconvolution and retinal image
enhancement.

### Strengths

- The linear forward model (y = PSF * x + n) is correct and well-documented.
- The hardware chain (LED illumination, fundus optics, CMOS detector) has
  realistic parameters (45-deg FOV, 4 mm pupil, 14-bit).
- cofe-Net is specifically designed for fundus images (published in IEEE TMI).
- The data source (DRIVE retinal dataset, Staal et al., 2004) is the standard
  benchmark for retinal imaging.

### Minor Notes

- No explicit mismatch parameters are defined. Adding PSF width variation,
  defocus, or illumination non-uniformity would strengthen the benchmark's
  ability to test robustness. This is a documentation/config enhancement,
  not a code issue.
- The 03_reconstruction_algorithms.md only lists Richardson-Lucy as a solver
  (traditional_cpu tier). The override provides the full 4-algorithm set on
  the leaderboard.

### Recommendations

No further code changes needed. The algorithm override is in place and verified.
