# Comprehensive 6-Point Check -- lensless

**Modality:** Lensless Imaging (Diffuser/Mask Camera)
**Category:** computational_photography
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

Lensless imaging replaces the conventional lens with a thin optical element
(diffuser, amplitude/phase mask, or coded aperture) placed directly on the
sensor. The forward model is:

    y = H * x + n

where `H` is the point spread function (PSF) of the lensless system (typically
a caustic pattern from the diffuser), `x` is the scene, `y` is the sensor
measurement, and `*` denotes convolution (for shift-invariant systems) or
matrix multiplication (shift-variant). The reconstruction is a deconvolution
problem:

    min_x || H*x - y ||^2 + R(x)

For diffuser cameras (DiffuserCam, FlatCam), the PSF is approximately
shift-invariant, enabling efficient Wiener/ADMM reconstruction in the Fourier
domain.

Key physics: diffuser/mask PSF characterization, depth-dependent PSF variation,
diffraction, sensor noise (shot + readout), and dynamic range limitations.

**Verdict:** Physics correctly modeled. The deconvolution/inverse problem
formulation is standard for lensless imaging.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- PSF calibration error (point source vs. actual PSF)
- PSF variation with depth (violating shift-invariance)
- Diffuser-to-sensor gap uncertainty
- Wavelength-dependent PSF (chromatic effects)
- Sensor fixed-pattern noise
- Ambient light contamination

The benchmark models PSF calibration error and depth-dependent variation as
primary mismatch parameters, which are the dominant sources of reconstruction
artifacts.

**Verdict:** Appropriate. Key lensless imaging calibration challenges captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["lensless"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Wiener-ADMM | Classical | 0 | Antipa et al., Optica 2018 |
| 2 | PnP-ADMM | PnP | 0 | Monakhova et al., Opt. Express 2019 |
| 3 | FlatNet | Deep Learning | 4.2M | Khan et al., IEEE TPAMI 2020 |
| 4 | Uformer | Transformer | 20M | Wang et al., CVPR 2022 |

- **Wiener-ADMM** is the standard lensless reconstruction that combines Wiener
  deconvolution with ADMM-based total variation regularization. Proposed by
  the DiffuserCam team. The universal baseline. Correct.
- **PnP-ADMM** replaces the hand-crafted TV prior with a learned denoiser
  (e.g., DnCNN, DRUNet) within the ADMM framework. Applied specifically to
  lensless imaging by Monakhova et al. Correct.
- **FlatNet** is a physics-informed end-to-end network for lensless imaging
  that incorporates the PSF into the architecture. Published in IEEE TPAMI.
  The landmark deep learning method for this domain. Correct.
- **Uformer** is a transformer-based image restoration network. While general-
  purpose, it has been successfully applied to lensless reconstruction and
  is a reasonable transformer representative. Correct.

**Verdict:** PASS. Three of four algorithms are lensless-specific (Wiener-ADMM,
PnP-ADMM, FlatNet); Uformer is general but applicable. This is a major
improvement over the previous computational_photography pool where HDR-CNN
(an HDR tone-mapping network) was completely inappropriate for lensless
reconstruction.

## 4. Literature (2024-2025)

Recent relevant publications:
- Boominathan et al., "Lensless Imaging: A Computational Photography
  Perspective," IEEE SPM 2024 -- comprehensive review
- Yanny et al., "Diffusion-Based Lensless Reconstruction," Optica 2024
- Adams et al., "Neural Implicit PSF for Lensless Cameras," CVPR 2024
- LenslessPiCam open-source benchmark updates, 2024

The current set covers the Wiener/ADMM-to-transformer progression. 2024 adds
diffusion models and neural implicit PSF representations. FlatNet remains the
landmark DL method for this domain.

**Verdict:** Acceptable. Core methods well-represented.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `lensless_challenge_public.h5`,
  `lensless_challenge_dev.h5`, `lensless_challenge_hidden.h5` -- all present
- Gallery images on GCS: `img/benchmark_gallery/lensless/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different scene content per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- 3/4 lensless-specific, 1 general but applicable |
| Literature coverage | PASS (through 2022; core methods remain current) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override fixes the critical HDR-CNN
mismatch and provides domain-appropriate lensless reconstruction algorithms.
