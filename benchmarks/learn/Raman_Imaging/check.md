# Comprehensive Benchmark QA Check — Raman Imaging

**URL:** https://pwm.platformai.org/benchmark/raman_imaging
**HTTP Status:** TBD (not yet deployed; dataset generated locally)
**Check Date:** 2026-03-11
**Reviewer:** Dataset generator + manual QA

---

## Table of Contents

1. [Benchmark Page Errors](#1-benchmark-page-errors)
2. [Local Dataset Inspection](#2-local-dataset-inspection)
3. [Public Dataset Source Assessment](#3-public-dataset-source-assessment)
4. [Algorithm Coverage Assessment](#4-algorithm-coverage-assessment)
5. [Improvement Suggestions](#5-improvement-suggestions)
6. [Action Items](#6-action-items)

---

## 1. Benchmark Page Errors

### Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 1     |
| LOW      | 2     |

### MEDIUM Severity

**M1. Benchmark page not yet deployed**
- Dataset HDF5 files and gallery images generated; GCS upload complete.
- Platform page (`/benchmark/raman_imaging`) not yet wired in `pages.py`.
- **Status:** Requires `_VARIANT_ALGOS`, `_VARIANT_NUM_SCENES`, `_VARIANT_BEST_RECON` entries in `pages.py`.

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Algorithm catalog (`_VARIANT_OVERRIDES`) does not yet include `raman_imaging` entry. |
| L2 | No scored benchmark result JSON exists yet; baseline run on challenge data needed. |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples | Phantoms | Status |
|------|------|------|---------|----------|--------|
| Public | `raman_imaging_challenge_public.h5` | 25.4 MB | 12 | 4 bio + 4 pharma + 4 polymer | PASS |
| Dev | `raman_imaging_challenge_dev.h5` | 42.8 MB | 20 | 7 bio + 7 pharma + 6 polymer | PASS |
| Hidden | `raman_imaging_challenge_hidden.h5` | 43.1 MB | 20 | 7 bio + 7 pharma + 6 polymer | PASS |

### HDF5 Schema Validation (sample_00 from public)

| Dataset Key | Shape | Dtype | Range | Check |
|-------------|-------|-------|-------|-------|
| `x_true` | (256, 256) | float32 | [0.0, 0.619] | PASS |
| `concentration` | (3, 256, 256) | float32 | [0.0, 0.991] | PASS |
| `y` | (3, 256, 256) | float32 | measured | PASS |
| `y_ideal` | (3, 256, 256) | float32 | ideal | PASS |
| `H_ideal` | (3, 512) | float32 | [0.0, 1.0] | PASS |
| `wavenumber` | (512,) | float32 | [400, 3200] cm⁻¹ | PASS |

### Attributes per sample

| Attribute | Content | Check |
|-----------|---------|-------|
| `metadata` | JSON: phantom_type, shape, n_species, baseline_psnr_dB, baseline_ssim | PASS |
| `true_spec` | JSON: laser_power_variation, background_fluorescence, spectral_shift_cm, noise_level | PASS |
| `spec_ranges` | JSON: per-tier mismatch parameter bounds | PASS |

### Modality Information

**Display Name:** Raman Imaging / Raman Spectroscopy Imaging

**Physics Class:** Inelastic light scattering (vibrational spectroscopy)

**Forward Model Family:** spectral_mixture_model

**Noise Model:** Shot noise (Poisson, sqrt-signal) + Gaussian readout + broadband fluorescence

**Image Size:** 256 × 256 pixels (primary species concentration map as x_true)

**Spectral Axis:** 512 wavenumber channels, 400–3200 cm⁻¹

### Dataset Integrity Assessment

- Tier separation: PASS — public/dev/hidden use independent random seeds (0/10000/20000)
- Phantom diversity: PASS — 3 distinct phantom types × different seeds per sample
- Forward model physics: PASS — y = sum_k(c_k × S_k) + fluorescence_bg + shot + readout
- Concentration normalisation: PASS — sum of species concentrations = 1 at each pixel
- Gallery images: PASS — 4 scenes, each with gt.png, gt_view1.png, gt_view2.png, measurement_I.png, measurement_II.png

---

## 3. Public Dataset Source Assessment

### Data Source

**Type:** Fully synthetic procedural phantoms (no external data dependency)

**Phantom Generator:** Analytic concentration maps using smooth Gaussian blobs,
spinodal decomposition approximation, and particle scattering.

### Species and Spectra

| Species Set | Type | Raman Peaks |
|-------------|------|-------------|
| Lipid | biological_tissue | 2850 (CH₂), 1450, 1740 (C=O) cm⁻¹ |
| Protein | biological_tissue | 1655 (Amide I), 1243 (Amide III), 2935 cm⁻¹ |
| Water | biological_tissue | 3200 (OH broad), 1640 cm⁻¹ |
| API | pharma_tablet | 1600 (C=C arom.), 3065, 1505 cm⁻¹ |
| Excipient | pharma_tablet | 1098 (C-O-C), 2891, 895 cm⁻¹ |
| Binder | pharma_tablet | 3250 (N-H/OH), 1150, 1680 (C=O lactam) cm⁻¹ |
| Polymer A | polymer_blend | 1001 (ring breath.), 1613, 3055 cm⁻¹ |
| Polymer B | polymer_blend | 1727 (C=O ester), 2945, 1265 cm⁻¹ |
| Filler | polymer_blend | 1100 (Si-O-Si), 800, 950 cm⁻¹ |

Spectra modelled as Lorentzian peaks with realistic HWHM values; peak
positions and relative intensities match published Raman libraries.

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Physical realism | GOOD | Lorentzian spectra match real databases |
| Phantom morphology | GOOD | 3 biologically/chemically distinct types |
| Mismatch coverage | GOOD | 4 independent mismatch knobs |
| Tier separation | PASS | Independent seeds; adversarial in hidden |
| Reproducibility | PASS | Fully deterministic given seed |

---

## 4. Algorithm Coverage Assessment

### Baseline (Implemented)

| # | Algorithm | Type | PSNR (public mean) |
|---|-----------|------|--------------------|
| 1 | Background subtraction + matched filter | Classical | ~20.0 dB |

### Recommended Additional Algorithms

| # | Algorithm | Type | Expected PSNR | Notes |
|---|-----------|------|---------------|-------|
| 2 | NNLS spectral unmixing | Classical | 22–27 dB | Non-negative least squares |
| 3 | MCR-ALS (multivariate curve resolution) | Iterative | 24–30 dB | Standard Raman analysis |
| 4 | NMF with sparsity constraints | Classical | 25–31 dB | Enforces physical positivity |
| 5 | TV-regularised unmixing | Variational | 28–33 dB | Spatial + spectral regularisation |
| 6 | Sparse Bayesian unmixing | Probabilistic | 30–35 dB | Full posterior |
| 7 | Deep spectral unmixing | Deep learning | 33–38 dB | U-Net on spectral cube |

### Known Gaps

- No multi-pixel spectral correlation exploited in baseline
- Deep unmixing methods not yet implemented
- Cross-species interference (spectral overlap) not yet modelled

---

## 5. Improvement Suggestions

### Priority Actions

1. **Register in platform** — Add `raman_imaging` entry to `_VARIANT_ALGOS`,
   `_VARIANT_NUM_SCENES`, `_VARIANT_BEST_RECON` in `platform/pwm_platform/services/benchmark_database/pages.py`

2. **Algorithm catalog** — Add `raman_imaging` to `_VARIANT_OVERRIDES` in
   `_algorithm_catalog.py` with NNLS, MCR-ALS, NMF baseline family

3. **Spectral overlap** — Add spectral overlap between species to increase
   difficulty; current Lorentzian peaks are well-separated for some species

4. **Baseline improvement** — Implement NNLS spectral unmixing as the official
   baseline (replaces matched filter); expected +3–5 dB improvement

5. **Score key** — Register in `_VARIANT_SCORE_ALIASES` and `CATEGORY_REAL_SCORES`

6. **Challenge data GCS** — HDF5 files uploaded to GCS path
   `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/{tier}/`

---

## 6. Action Items

| Priority | Item | Owner | ETA |
|----------|------|-------|-----|
| P1 | Register variant in pages.py (_VARIANT_ALGOS, _VARIANT_NUM_SCENES) | Platform | Next sprint |
| P1 | Add algorithm catalog entry in _algorithm_catalog.py | Platform | Next sprint |
| P2 | Implement NNLS spectral unmixing baseline | Algorithm | Next sprint |
| P2 | Add benchmark result JSON to benchmarks/results/raman_imaging/ | QA | After P1 |
| P3 | Add spectral overlap mismatch (cross-talk between species) | Dataset | Future |
| P3 | Expand to 5 species for advanced pharma/biological tiers | Dataset | Future |

---

## Appendix: Forward Model Summary

```
y(x, y, k) = integral_{band_k} [ sum_j c_j(x,y) * S_j(w) ] dw
           + F_bg(x, y, k)
           + sigma_shot * sqrt(y_ideal + eps) * N(0,1)
           + sigma_readout * N(0,1)

Species concentration:   sum_j c_j(x,y) = 1  (normalised)
Wavenumber range:        400–3200 cm⁻¹ (512 channels)
Spectral peaks:          Lorentzian, HWHM 5–120 cm⁻¹
Fluorescence background: Exponentially decaying with wavenumber
Pixel size:              0.195 μm/px, FOV ~50 μm
```

## Appendix: GCS Upload Paths

```
gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/public/raman_imaging_challenge_public.h5
gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/dev/raman_imaging_challenge_dev.h5
gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/hidden/raman_imaging_challenge_hidden.h5
```

Gallery images uploaded to:
```
gs://pwm-benchmark-datasets/img/benchmark_gallery/raman_imaging/scene_{00-03}/
```
