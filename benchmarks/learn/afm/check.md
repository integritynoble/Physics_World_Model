# Comprehensive Benchmark QA Check — AFM (Atomic Force Microscopy)
**URL:** https://pwm.platformai.org/benchmark/afm
**Check Date:** 2026-03-03 (comprehensive 6-point review)

---

## 1. Benchmark Page Errors

### 1.1 Forward Model Over-Simplification (HIGH)
The DAG shows only "S --> D" (Sampling --> Detector), which is a generic imaging
pipeline. Real AFM forward models require:
- **Cantilever dynamics** — resonance frequency, Q-factor, amplitude setpoint,
  approach/retract force curves
- **Tip-sample force interaction** — van der Waals, electrostatic, adhesion,
  capillary forces (Derjaguin approximation)
- **Z-feedback control loop** — closed-loop PID dynamics that couple scanner
  motion to the detected signal
- **Morphological dilation** — the true forward operation is a dilation of the
  surface by the tip geometry, not a PSF convolution

The `category_module: microscopy_psf` label and the signal equation
`z(x,y) = h(x,y) * tip(x,y) + noise` treat AFM as if it were an optical
microscope with a point-spread function. In reality AFM tip-sample interaction
is a morphological operation (Minkowski sum / dilation), not a linear
convolution. This is a fundamental physics error.

**Fix:** Replace `microscopy_psf` with a dedicated `scanning_probe` category
module. Model the forward operator as morphological dilation plus nonlinear
scanner distortions.

### 1.2 Gallery Section Empty (HIGH)
JavaScript `selectGalleryScene()` and panel DOM elements exist. Gallery images
are present in the repo under
`platform/pwm_platform/static/img/benchmark_gallery/afm/scene_0{0..3}/` (4
scenes with gt.png, measurement_I/II.png, recon_I/II/III.png). However, these
do not render on the live page.

**Fix:** Verify image serving route and gallery JavaScript; populate the gallery
or provide static fallback.

### 1.3 PSNR_norm Undefined in Scoring Formula (HIGH)
The scoring formula `0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - Hx||/||y||)`
uses PSNR_norm but never defines:
- Normalization bounds (min/max PSNR for mapping to [0,1])
- Whether the normalization is per-scene or global
- A worked numerical example

This makes scores non-reproducible by external teams.

**Fix:** Define PSNR_norm = (PSNR - PSNR_min) / (PSNR_max - PSNR_min) with
explicit bounds (e.g., PSNR_min = 10 dB, PSNR_max = 50 dB) and show a
calculation for one method.

### 1.4 Noise Model Missing AFM-Specific Sources (HIGH)
The noise model lists generic "Poisson (photon counting), speckle (coherent),
multiplicative" noise sources. AFM does not use photons. AFM-specific noise:
- Cantilever thermal (Brownian) fluctuations — the dominant noise at small
  amplitudes, governed by kBT/k (spring constant)
- Optical beam deflection (OBD) detector noise — laser shot noise, photodiode
  electronics noise
- 1/f (flicker) noise in piezo actuators
- Acoustic/vibrational coupling from the environment

**Fix:** Replace generic photon-counting noise model with cantilever thermal
noise + OBD detector noise + 1/f scanner noise.

### 1.5 Mismatch Parameter Ranges Inconsistent (MEDIUM)
The **config YAML** defines global ranges (piezo_nonlinearity: [0.0, 5.0],
scanner_hysteresis: [0.0, 10.0]) but the **live leaderboard page** reports
per-tier signed ranges:
- `piezo_nonlinearity`: Public [-1.0, 2.0] / Dev [-1.2, 1.8] / Hidden [-0.7, 2.3]
- `scanner_hysteresis`: Public [-2.0, 4.0] / Dev [-2.4, 3.6] / Hidden [-1.4, 4.6]
- Negative piezo nonlinearity is physically questionable

The config and the website disagree. No hardware reference is cited for any
range (e.g., typical AFM scanner nonlinearity is 1-5% of full range).

**Fix:** Reconcile YAML config with live page. Cite actual AFM hardware specs
(e.g., Nanosurf FlexAFM piezo nonlinearity spec sheet).

### 1.6 Tip Shape Convolution Range Is [0, 0] (MEDIUM)
The mismatch parameter `tip_shape_convolution` has range [0, 0] across all
tiers, meaning it is never perturbed. This is the single most important
AFM-specific mismatch parameter — tip wear, contamination, and double-tip
artifacts are the dominant error sources in real AFM measurements.

**Fix:** Assign a nontrivial range to tip_shape_convolution. Consider
parameterizing tip radius (5-50 nm), half-cone angle (10-35 deg), and
asymmetry factor.

### 1.7 Only 3 Scenes Per Tier (MEDIUM)
Three scenes per tier cannot produce statistically meaningful confidence
intervals. Most imaging benchmarks use 50-200+ test images.

**Fix:** Increase to at least 20 scenes per tier, or add bootstrapped
confidence intervals to scores.

### 1.8 References Incomplete (MEDIUM)
- "Villarrubia, JRNIST 1997" — no exact title, volume, pages, or DOI. The
  canonical reference is: J.S. Villarrubia, "Algorithms for Scanned Probe
  Microscope Image Simulation, Surface Reconstruction, and Tip Estimation,"
  J. Res. Natl. Inst. Stand. Technol. 102(4), 425-454, 1997.
  DOI: 10.6028/jres.102.030
- "Probe Transformer, 2024" — no authors, venue, or DOI
- Missing DOIs for Alldritt et al. (2020), Zhang et al. (2017)

### 1.9 HDF5 Schema Undocumented (LOW)
No documentation of key names, array shapes, data types, or compression for the
benchmark HDF5 files.

### 1.10 Spec Primitives Generic (LOW)
The spec primitive alphabet (P, M, Pi, F, C, Sigma, D, S, W, R, Lambda) is
generic imaging. AFM-specific primitives (cantilever model, tip geometry, force
model, feedback loop) are absent.

---

## 2. Local Dataset Inspection

**Result: No local AFM dataset exists.**

```
$ ls datasets/benchmark/afm 2>/dev/null
(directory does not exist — exit code 2)
```

Gallery images exist at:
```
platform/pwm_platform/static/img/benchmark_gallery/afm/scene_0{0..3}/
```
These contain gt.png, measurement_I.png, measurement_II.png, recon_I.png,
recon_II.png, recon_III.png for 4 scenes (scene_00 through scene_03).

The config YAML declares `data_source.fallback: generated` with
`synthetic_generator: shepp_logan`. This means when no web data is downloaded,
the benchmark falls back to the Shepp-Logan phantom — a medical CT phantom that
is completely unrelated to AFM surface topography.

**Assessment:** The dataset pipeline is non-functional for real AFM data. The
fallback to Shepp-Logan phantoms means any benchmark run without manual data
preparation will produce physically meaningless results.

**Fix:** Either (a) ship curated AFM calibration images in
`datasets/benchmark/afm/`, or (b) implement a realistic synthetic AFM surface
generator (e.g., random rough surfaces with specified RMS roughness and
correlation length, or step/grating calibration structures).

---

## 3. Public Dataset Source Assessment

### Declared Source
| Property | Value |
|----------|-------|
| Dataset ID | `afm_dataset` |
| URL | https://www.nanosurf.com/en/application/afm-images |
| Citation | "Nanosurf sample images" |
| License | "Public domain" |

### Issues

1. **License Mischaracterization**: Nanosurf is a commercial AFM manufacturer.
   Their website images are marketing material and almost certainly NOT public
   domain. They are likely copyrighted by Nanosurf AG with at best a
   limited-use license.

2. **No Ground Truth**: Web-scraped AFM images have no known ground truth
   surface geometry. Without a calibration standard of known dimensions, PSNR
   and SSIM have no physical meaning.

3. **No Controlled Acquisition Parameters**: Web images lack metadata: scan
   speed, setpoint, resolution, tip type, environment (air/liquid/vacuum),
   imaging mode (contact/tapping/non-contact).

4. **Source Contradiction**: The leaderboard cites "AIST-NT AFM Calibration
   (Villarrubia, JRNIST 1997)" but the config YAML points to Nanosurf — these
   are different sources. Which is actually used?

### Recommended Alternative Sources

| Source | Description | License |
|--------|-------------|---------|
| NIST AFM Calibration Standards | Tip characterizer gratings with known geometry | Public (US Gov) |
| Gwyddion sample files | Open-source SPM analysis software samples | GPL |
| AFM Open Data (Figshare) | Community-uploaded AFM datasets | CC-BY |
| Synthetic calibration gratings | Generated step/line gratings with known RMS | N/A |

**Fix:** Replace Nanosurf web scraping with NIST calibration standard data or
well-documented synthetic surfaces with known ground truth.

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard

| Rank | Method | Overall | Public PSNR/SSIM | Dev PSNR/SSIM | Hidden PSNR/SSIM |
|------|--------|---------|------------------|---------------|------------------|
| 1 | DeepSPM + gradient | 0.617 | 28.33 dB / 0.888 | 22.81 dB / 0.724 | 21.13 dB / 0.652 |
| 2 | E2E-BTR + gradient | 0.589 | 30.68 dB / 0.927 | 20.76 dB / 0.635 | 18.55 dB / 0.527 |
| 3 | Reg-Deconv + gradient | 0.543 | 24.47 dB / 0.785 | 20.92 dB / 0.642 | 18.85 dB / 0.542 |
| 4 | BTR + gradient | 0.530 | 21.92 dB / 0.687 | 21.43 dB / 0.665 | 18.43 dB / 0.521 |

### Currently in Config (Solvers)

| Tier | Name | Notes |
|------|------|-------|
| traditional_cpu | Richardson-Lucy | Generic deconvolution — not AFM-specific |
| best_quality | CARE (Weigert 2018) | Fluorescence microscopy denoiser — not AFM-specific |

### Critical Gap: Leaderboard vs. Config Mismatch
The 4 leaderboard algorithms (DeepSPM, E2E-BTR, Reg-Deconv, BTR) do not
appear in the solver config. The 2 config solvers (Richardson-Lucy, CARE) do
not appear on the leaderboard. There is zero overlap.

### Missing Famous/Recent Algorithms

The following well-known AFM reconstruction and image restoration algorithms
are absent from both the leaderboard and the config:

| Algorithm | Year | Type | Why Include |
|-----------|------|------|-------------|
| **Differentiable BTR** (Matsunaga et al.) | 2023 | Blind tip reconstruction | End-to-end differentiable morphological erosion/dilation in Julia/Flux; robust to noise. Published in Scientific Reports. GitHub: matsunagalab/differentiable_BTR |
| **Restormer** (Zamir et al.) | 2022 | Transformer restoration | SOTA efficient transformer for high-res image restoration; shown effective for AFM denoising (Jung et al. 2022) |
| **HINet** (Chen et al.) | 2021 | Half-instance normalization | Most effective model for AFM denoising per comparative study (Jung et al. 2022) |
| **AFM-SRNN** (ACS Appl. Nano Mater.) | 2024 | Super-resolution neural net | Dedicated AFM super-resolution network; 3.5-7.5x imaging speedup |
| **ResU-Net for AFM** (BJNANO 2025) | 2025 | CNN denoising | Trained on synthetic data, validated on 82 real AFM images of 10 samples; handles bidirectional scan artifacts |
| **Cross-Module SR** (Xu et al., J. Microscopy) | 2025 | GAN-based super-resolution | +1.65 dB PSNR, +0.041 SSIM for AFM cell images via frequency division + adaptive fusion |
| **MPRNet** (Zamir et al.) | 2021 | Multi-stage progressive | CVPR 2021; tested on AFM denoising (Jung et al. 2022) |
| **Uformer** (Wang et al.) | 2022 | U-shaped transformer | CVPR 2022; general image restoration with AFM applications |
| **Gwyddion classical** | 2004+ | Open-source SPM analysis | Standard open-source tool for AFM image processing; includes tip deconvolution, leveling, noise filtering |
| **Morphological erosion** (Villarrubia) | 1997 | Classical BTR | The foundational algorithm — should be the primary classical baseline, not Richardson-Lucy |

### Key Concern
**Richardson-Lucy is the wrong classical baseline.** RL is designed for
Poisson-noise linear convolution (optical/fluorescence microscopy). The
correct AFM classical baseline is morphological erosion (Villarrubia BTR),
which is already partially represented on the leaderboard but absent from
the config.

**CARE is the wrong deep learning baseline.** CARE was designed for
fluorescence microscopy denoising with paired low-SNR/high-SNR training data.
It does not address AFM-specific artifacts (tip convolution, scanner
nonlinearity, thermal drift).

---

## 5. Improvement Suggestions

### Priority 1 — Physics Correctness (Weeks 1-2)
1. **Replace `microscopy_psf` with `scanning_probe` module** — implement
   morphological dilation as the forward operator instead of PSF convolution.
2. **Enable `tip_shape_convolution` mismatch** — make this the primary
   mismatch parameter with realistic tip geometry parameterization (radius,
   half-cone angle, asymmetry).
3. **Replace noise model** — remove Poisson/photon noise; add cantilever
   thermal noise (kBT/k), OBD detector noise, and 1/f scanner noise.
4. **Fix forward model equation** — change from `z = h * tip + noise` to
   `z_meas = dilation(z_true, tip) + scanner_distortion + noise`.

### Priority 2 — Dataset Quality (Weeks 2-4)
5. **Create or source real AFM calibration data** with known ground truth
   (e.g., NIST TGT1 or TGG1 calibration gratings, silicon step standards).
6. **Replace Shepp-Logan fallback** with realistic synthetic AFM surfaces
   (random rough surfaces, step gratings, nanoparticles on flat substrates).
7. **Increase scene count** from 3 to at least 20 per tier, or implement
   bootstrapped confidence intervals.
8. **Resolve Nanosurf license** — verify actual license terms or switch to
   properly licensed data.

### Priority 3 — Algorithm Baselines (Weeks 3-5)
9. **Replace Richardson-Lucy with Villarrubia BTR** as `traditional_cpu`
   baseline.
10. **Replace CARE with a dedicated AFM method** (e.g., Differentiable BTR or
    ResU-Net for AFM) as `best_quality` baseline.
11. **Add famous_dl tier** — include Restormer and HINet, both validated for
    AFM denoising.
12. **Reconcile leaderboard and config** — ensure all leaderboard algorithms
    have corresponding solver entries and vice versa.

### Priority 4 — Documentation (Week 5)
13. **Define PSNR_norm** — specify normalization bounds with a worked example.
14. **Complete references** — add DOIs for all citations (Villarrubia 1997:
    DOI 10.6028/jres.102.030).
15. **Document HDF5 schema** — key names, shapes, dtypes, compression.
16. **Fix gallery rendering** — images exist in repo but do not display.

---

## 6. Action Items

| # | Priority | Severity | Action | Owner |
|---|----------|----------|--------|-------|
| A1 | P1 | HIGH | Replace `microscopy_psf` with `scanning_probe` morphological dilation forward model | Physics team |
| A2 | P1 | HIGH | Enable tip_shape_convolution mismatch with realistic parameterization | Physics team |
| A3 | P1 | HIGH | Replace photon noise model with cantilever thermal + OBD noise | Physics team |
| A4 | P1 | HIGH | Define PSNR_norm bounds and add worked example to scoring docs | Platform team |
| A5 | P2 | HIGH | Source or create AFM calibration dataset with known ground truth | Data team |
| A6 | P2 | HIGH | Replace Shepp-Logan fallback with realistic AFM surface generator | Data team |
| A7 | P2 | MEDIUM | Increase scene count from 3 to 20+ per tier | Data team |
| A8 | P2 | MEDIUM | Verify Nanosurf image license or replace with properly licensed data | Legal/Data |
| A9 | P3 | HIGH | Replace Richardson-Lucy with Villarrubia BTR as traditional_cpu solver | Recon team |
| A10 | P3 | HIGH | Replace CARE with AFM-specific DL method as best_quality solver | Recon team |
| A11 | P3 | MEDIUM | Add Restormer/HINet as famous_dl solver tier | Recon team |
| A12 | P3 | MEDIUM | Reconcile leaderboard algorithms with solver config entries | Platform team |
| A13 | P4 | MEDIUM | Complete all reference DOIs (Villarrubia DOI: 10.6028/jres.102.030) | Docs team |
| A14 | P4 | LOW | Document HDF5 schema (keys, shapes, dtypes) | Docs team |
| A15 | P4 | LOW | Fix gallery image rendering on live benchmark page | Frontend team |
| A16 | P4 | LOW | Reconcile YAML config ranges with live page per-tier ranges | Platform team |

---

## Appendix: Key References

1. **Villarrubia (1997)** — J.S. Villarrubia, "Algorithms for Scanned Probe
   Microscope Image Simulation, Surface Reconstruction, and Tip Estimation,"
   J. Res. Natl. Inst. Stand. Technol. 102(4), 425-454.
   DOI: [10.6028/jres.102.030](https://doi.org/10.6028/jres.102.030)

2. **Matsunaga et al. (2023)** — "End-to-end differentiable blind tip
   reconstruction for noisy atomic force microscopy images," Scientific Reports.
   DOI: [10.1038/s41598-022-27057-2](https://doi.org/10.1038/s41598-022-27057-2)
   Code: [github.com/matsunagalab/differentiable_BTR](https://github.com/matsunagalab/differentiable_BTR)

3. **Jung et al. (2022)** — "Comparative study of deep learning algorithms for
   atomic force microscopy image denoising," Ultramicroscopy.
   Code: [github.com/hoichanjung/AFM_Image_Denoising](https://github.com/hoichanjung/AFM_Image_Denoising)

4. **Xu et al. (2025)** — "Enhanced reconstruction of atomic force microscopy
   cell images to super-resolution," Journal of Microscopy.
   DOI: [10.1111/jmi.13423](https://doi.org/10.1111/jmi.13423)

5. **ACS Appl. Nano Mater. (2024)** — "AFM Super-Resolution Reconstruction
   Neural Network for Imaging Nanomaterials."
   DOI: [10.1021/acsanm.4c04427](https://doi.org/10.1021/acsanm.4c04427)

6. **Zamir et al. (2022)** — "Restormer: Efficient Transformer for
   High-Resolution Image Restoration," CVPR 2022.

7. **Chen et al. (2021)** — "HINet: Half Instance Normalization Network for
   Image Restoration," CVPRW 2021.

8. **Krull et al. (2020)** — "Artificial-intelligence-driven scanning probe
   microscopy," Commun. Phys. 3, 54.
   DOI: [10.1038/s42005-020-0317-3](https://doi.org/10.1038/s42005-020-0317-3)

9. **Weigert et al. (2018)** — "Content-aware image restoration: pushing the
   limits of fluorescence microscopy," Nature Methods 15, 1090-1097.

10. **Necas & Klapetek (2012)** — "Gwyddion: an open-source software for SPM
    data analysis," Central European Journal of Physics 10(1), 181-188.

---

*Comprehensive 6-point review on 2026-03-03 — covering benchmark page errors, local dataset inspection, public dataset source assessment, algorithm coverage assessment, improvement suggestions, and prioritized action items.*
