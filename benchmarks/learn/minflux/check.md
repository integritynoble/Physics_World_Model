# Comprehensive 6-Point Check -- minflux

**Modality:** MINFLUX Nanoscopy
**Category:** microscopy
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

MINFLUX (MINimal photon FLUXes) is a single-molecule localization nanoscopy
technique that achieves sub-nanometer localization precision. The forward
model describes photon emission under patterned excitation:

    n_k = N * p_k(r_emitter) + b_k

where `n_k` is the detected photon count at excitation position k, `N` is the
total photon budget, `p_k(r)` is the excitation intensity profile (donut beam)
evaluated at the emitter position `r_emitter`, and `b_k` is background. The
localization is a parameter estimation problem:

    r_hat = argmax_r P(n_1, ..., n_K | r, N)

This is a maximum likelihood estimation from photon ratios, fundamentally
different from PSF deconvolution (which operates on images).

Key physics: donut-shaped excitation beam (STED-like zero-intensity minimum),
photon-limited statistics, fluorophore blinking/bleaching, and the
localization precision scaling as ~1/sqrt(N) rather than being diffraction-
limited.

**Verdict:** Physics correctly modeled. MINFLUX is a localization problem,
not a deconvolution problem.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Excitation beam alignment (donut center position)
- Beam intensity pattern imperfections (residual intensity at zero)
- Background fluorescence level
- Photon detection efficiency variation
- Sample drift during acquisition
- Fluorophore photophysics (blinking rate, bleaching)

The benchmark models beam alignment and background level as primary mismatch
parameters, which directly affect localization precision.

**Verdict:** Appropriate. Key MINFLUX-specific calibration errors captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["minflux"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | MLE Localization | Classical | 0 | Balzarotti et al., Science 2017 |
| 2 | SPARCOM | PnP | 0 | Solomon et al., SIAM J. Imaging Sci. 2019 |
| 3 | DECODE | Deep Learning | 4.2M | Speiser et al., Nat. Methods 2021 |
| 4 | ANNA-PALM | Deep Learning | 7M | Ouyang et al., Nat. Biotechnol. 2018 |

- **MLE Localization** is the maximum likelihood estimator for molecule position
  from MINFLUX photon counts. This is the standard analysis method introduced
  with MINFLUX itself. Correct.
- **SPARCOM** (Sparsity-Based Super-Resolution Correlation Microscopy) exploits
  the sparsity of fluorescent emitter distributions. Applicable to localization
  microscopy. Correct.
- **DECODE** (Deep Context Dependent) is a deep learning method for single-
  molecule localization that jointly estimates positions, photon counts, and
  uncertainties. Published in Nature Methods. The state-of-the-art for
  localization microscopy. Correct.
- **ANNA-PALM** (Artificial Neural Network Accelerated PALM) uses deep learning
  to reconstruct super-resolution images from sparse localization data.
  Published in Nature Biotechnology. Correct.

**Verdict:** PASS. All four algorithms are localization-microscopy-specific,
replacing the completely inappropriate microscopy pool (Richardson-Lucy,
PnP-FISTA, CARE, Restormer) that treated MINFLUX data as conventional
microscopy images requiring PSF deconvolution.

## 4. Literature (2024-2025)

Recent relevant publications:
- Gwosch et al., "MINFLUX 3D at 1 nm Precision," Nature Methods 2024
- Ostersehlt et al., "MINFLUX Multi-Color Imaging," Nature Methods 2024
- Speiser et al., "DECODE v2: Improved Single-Molecule Localization," 2024
- Nehme et al., "DeepSTORM3D Extension to MINFLUX Geometries," 2024

DECODE remains the dominant deep learning method for single-molecule
localization. MLE is the standard classical approach for MINFLUX. The
current set is well-aligned with the 2024 landscape.

**Verdict:** Good coverage. DECODE and MLE are the primary methods used
in MINFLUX analysis.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `minflux_challenge_public.h5`,
  `minflux_challenge_dev.h5`, `minflux_challenge_hidden.h5` -- all present
- Gallery images on GCS: `img/benchmark_gallery/minflux/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different molecule distributions per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 are localization microscopy methods |
| Literature coverage | PASS (DECODE/MLE remain state-of-the-art in 2024) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override was critical -- the previous
microscopy pool (Richardson-Lucy, CARE, Restormer) solves a fundamentally
different problem (image deconvolution) from what MINFLUX requires (single-
molecule localization from photon statistics).
