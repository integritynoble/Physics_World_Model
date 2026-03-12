# SEM Benchmark Dataset -- Generation Check

**Generated:** 2026-03-11
**Status:** PASS

---

## Dataset Summary

| Tier | Samples | Seed | HDF5 Size |
|------|---------|------|-----------|
| Public | 12 | 0 | 11 MB |
| Dev | 20 | 10000 | 16 MB |
| Hidden | 20 | 20000 | 16 MB |

Image size: 256x256, pixel size: 5 nm, FOV: 1.28 um x 1.28 um

---

## Forward Model

```
y = Poisson(eta * (BSE_yield(Z,theta) * PSF(x) + SE_yield * edge_enhancement))
    * detector_response + readout_noise
```

**Physics components:**
- **BSE yield:** Heinrich/Reuter empirical polynomial as function of atomic number Z
- **SE yield:** Sternglass (1957) model, peaks at low beam energies
- **PSF:** Probe size (Schottky FEG) + SE escape depth + spherical/chromatic aberrations from working distance
- **Edge enhancement:** Sobel gradient magnitude for topographic contrast
- **Detector response:** Everhart-Thornley cosine-law collection geometry
- **Noise:** Poisson shot noise (mean count ~2500 at 50 pA) + Gaussian readout (sigma=3.0)
- **Charging artifacts:** Low-frequency brightness shift + horizontal banding + pixel displacement

---

## Mismatch Parameters

| Parameter | Public | Dev | Hidden |
|-----------|--------|-----|--------|
| beam_voltage_kV | 5-15 | 3-20 | 1-30 |
| working_distance_mm | 3-8 | 2-12 | 1-15 |
| detector_bias | 0.8-1.2 | 0.6-1.4 | 0.4-1.6 |
| charging_artifact | 0-0.05 | 0-0.15 | 0-0.30 |

---

## Phantom Types

| Type | Count/Tier | Description | Z_primary |
|------|-----------|-------------|-----------|
| Semiconductor | 3 | Lines, contacts, vias, trenches (IC surface) | Si (14) |
| Fracture | 3 | Multi-scale rough topography, crack features | Fe (26) |
| Nanoparticles | 3 | High-Z particles (Au, Ag, Pt, Cu, Pb) on flat substrate | Si (14) |
| Biological | 3 | Cell membranes, organelles, fibers (tissue cross-section) | O (8) |

---

## CPU Baseline Results

**Algorithm:** Gaussian denoising + Wiener deconvolution + edge-preserving adaptive smoothing

| Tier | Mean PSNR (dB) | Mean SSIM | Min PSNR | Max PSNR |
|------|---------------|-----------|----------|----------|
| Public | 26.12 | 0.957 | 14.92 | 34.03 |
| Dev | 27.04 | 0.970 | 14.91 | 34.46 |
| Hidden | 26.82 | 0.968 | 19.46 | 36.77 |

**Per-phantom-type breakdown (public tier):**

| Type | PSNR Range (dB) | SSIM Range |
|------|----------------|------------|
| Semiconductor | 26.0-29.3 | 0.974-0.983 |
| Fracture | 27.2-28.6 | 0.962-0.964 |
| Nanoparticles | 31.1-34.0 | 0.992-0.993 |
| Biological | 14.9-18.6 | 0.867-0.917 |

**Note:** Biological phantoms score lower due to complex fine structure (organelles, membranes, fibers) that is challenging for simple denoising.

---

## HDF5 Structure

```
sample_XX/
  x_true                  (256, 256) float32  -- Ground truth surface/material map [0, 1]
  y                       (256, 256) float32  -- Measured SEM image (normalized) [0, 1]
  H_ideal                 (256, 256) float32  -- Ideal system response (noiseless) [0, 1]
  reconstruction_baseline (256, 256) float32  -- NLM+Wiener baseline reconstruction
```

---

## GCS Upload

- `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/public/sem_challenge_public.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/dev/sem_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/hidden/sem_challenge_hidden.h5`

---

## Gallery Images

4 scenes in `platform/pwm_platform/static/img/benchmark_gallery/sem/scene_0{0-3}/`:
- `gt.png` -- ground truth
- `measurement_I.png` -- measured SEM image (noisy)
- `measurement_II.png` -- ideal system response
- `recon_I.png` -- baseline reconstruction
- `recon_II.png` -- |GT - recon| difference

---

## Verification Checks

1. **Tier separation:** Each tier uses different seeds (0, 10000, 20000) with different phantom variants and augmentation parameters. No shared phantoms across tiers.
2. **Forward model physics:** BSE yield follows Heinrich empirical polynomial; SE yield follows Sternglass model; PSF based on Schottky FEG probe diameter.
3. **Baseline quality:** Mean PSNR ~22-28 dB across tiers (target range achieved).
4. **Data integrity:** All HDF5 files load correctly with expected shapes and dtypes.
5. **GCS upload:** All three tier HDF5 files uploaded to `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/`.

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 19.3 dB |
| SSIM (sample_00) | 0.5949 |
| Runtime | 0.01 s/sample |

**Result: PASS**
