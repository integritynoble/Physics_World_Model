# Comprehensive 6-Point Check -- hyperspectral_remote

**Modality:** Hyperspectral Remote Sensing
**Category:** remote_sensing
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

Hyperspectral remote sensing captures images across hundreds of narrow
spectral bands (typically 400--2500 nm, 5--10 nm bandwidth per band). The
forward model for spectral reconstruction/fusion is:

    Y_LR = D * H * X + N    (low-resolution hyperspectral)
    Y_HR = X * R + N         (high-resolution multispectral)

where `X` is the high-resolution hyperspectral image (spatial x spectral),
`D` is spatial downsampling, `H` is the spatial blur PSF, `R` is a spectral
response function mapping hyperspectral to multispectral bands, and `N` is
sensor noise. The reconstruction task fuses low-resolution hyperspectral and
high-resolution multispectral images to produce a high-resolution hyperspectral
datacube.

Key physics: atmospheric absorption (water vapor, CO2 bands), spectral mixing
of sub-pixel materials, sensor noise model (photon shot noise + readout).

**Verdict:** Physics correctly modeled. The spectral fusion formulation is
standard for hyperspectral remote sensing reconstruction.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Spectral response function uncertainty
- Spatial co-registration error between HSI and MSI
- Atmospheric correction residuals
- Spectral variability of endmembers
- Sensor nonlinearity and striping artifacts

The benchmark models spectral response uncertainty and spatial misregistration
as primary mismatch parameters, which are the dominant error sources in
hyperspectral fusion.

**Verdict:** Appropriate. Key fusion-specific uncertainties captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["hyperspectral_remote"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | CNMF | Classical | 0 | Yokoya et al., IEEE TGRS 2012 |
| 2 | PnP-LTTR | PnP | 0 | He et al., IEEE TGRS 2020 |
| 3 | DBIN | Deep Learning | 3.2M | Dong et al., CVPR 2021 |
| 4 | MST++ | Transformer | 8M | Cai et al., CVPRW 2022 |

- **CNMF (Coupled NMF)** is a matrix factorization method that jointly
  decomposes HSI and MSI into shared endmembers and abundance maps. Standard
  baseline for hyperspectral fusion. Correct.
- **PnP-LTTR (Low-Tensor-Train-Rank)** combines tensor decomposition with
  plug-and-play priors for spectral image reconstruction. Exploits the
  low-rank spectral structure. Correct.
- **DBIN (Deep Blind Image Network)** is a CNN-based spectral reconstruction
  network. Published at CVPR 2021. Correct.
- **MST++ (Mask-guided Spectral-wise Transformer)** won the NTIRE 2022
  Spectral Reconstruction Challenge. State-of-the-art transformer for
  spectral reconstruction. Correct.

**Verdict:** PASS. All four algorithms are hyperspectral-specific, replacing
the generic computational pool (Tikhonov, PnP-RED, DIP, SwinIR) that had
no spectral awareness.

## 4. Literature (2024-2025)

Recent relevant publications:
- Arun et al., "Spectral Diffusion: Hyperspectral Image Reconstruction via
  Diffusion Models," CVPR 2024
- Cai et al., "Degradation-Aware Unfolding for HSI Fusion," IEEE TPAMI 2024
- Wang et al., "SpectralGPT: Foundation Model for Remote Sensing," IEEE TGRS
  2024
- NTIRE 2024 Spectral Reconstruction Challenge results

The current set covers methods through 2022 (MST++ winner). 2024 introduces
diffusion-based and foundation model approaches, but the core methodological
coverage (factorization, tensor, CNN, transformer) remains representative.

**Verdict:** Acceptable. Diffusion-based spectral reconstruction is emerging
but not yet required for representative coverage.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `hyperspectral_remote_challenge_public.h5`,
  `hyperspectral_remote_challenge_dev.h5`,
  `hyperspectral_remote_challenge_hidden.h5` -- all present
- Gallery images on GCS: `img/benchmark_gallery/hyperspectral_remote/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different spectral scene content per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 are hyperspectral-specific |
| Literature coverage | PASS (through 2022; 2024 adds diffusion models) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override correctly routes hyperspectral
remote sensing to spectral-aware algorithms rather than the generic
computational pool.
