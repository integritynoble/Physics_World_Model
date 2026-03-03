# Stellar Coronagraphy -- PWM Benchmark Review

**Modality ID**: `coronagraphy`
**Category**: Astronomy & Space Imaging
**Carrier**: Photon
**Canonical DAG**: M --> P --> D (Modulation --> Propagation --> Detector)
**Current Maturity**: M0
**Review Date**: 2026-03-03

---

## 1. Platform Specification & Forward Model

The PWM Stellar Coronagraphy benchmark addresses blind reconstruction of
astronomical signals corrupted by instrumental mismatches in a coronagraphic
imaging pipeline. The forward model follows a three-stage DAG:

- **M (Modulation)**: Coronagraph mask applies spatial amplitude modulation
  (Lyot stop or vortex type) with target parameters: 1e-8 contrast ratio,
  3 lambda/D inner working angle (IWA).
- **P (Propagation)**: Free-space propagation via Fresnel or
  Rayleigh-Sommerfeld diffraction kernels.
- **D (Detector)**: Sensor readout with gain g and noise model eta (Gaussian,
  Poisson, or mixed).

The forward model type is `nonlinear_operator` with a dedicated operator
(`coronagraphy`) registered in the PWM graph templates as
`coronagraphy_graph_v1`. The default reconstruction solver is
`adi_speckle_subtraction`; the best-quality solver is `PnP-ADMM` (2M params,
GPU-required).

### Mismatch Parameters (4 degrees of freedom)

| Parameter                   | Nominal | Perturbed Range    | Unit      |
|-----------------------------|---------|--------------------|-----------|
| Coronagraph mask centering  | 0.0     | [0.0, 0.1]         | lambda/D  |
| Wavefront error (WFE)       | 0.0     | [0.0, 100] rms     | --        |
| Stellar leakage             | 1e-6    | [1e-7, 1e-4]       | contrast  |
| Speckle lifetime            | static  | [0.1, 100]         | s         |

### Mismatch Maturity Ladder

| Level | Description                                        | Params Perturbed |
|-------|----------------------------------------------------|------------------|
| M0    | No mismatch -- perfect forward model               | 0                |
| M1    | Single parameter perturbed                          | 1                |
| M2    | Compound mismatch (3+ simultaneously)               | 3+               |
| M3    | Real calibration/experimental errors                | all              |
| M4    | Adversarial worst-case mismatch (max failure)       | all              |

### Image Sizes & Noise Levels

Expanded configs support three spatial resolutions (128x128, 256x256, 512x512)
and four noise levels (clean 60 dB, low 40 dB, medium 30 dB, high 20 dB). The
base config uses 64x64 for rapid iteration. Total planned cases across all
benchmark tiers: B1=12, B2=60, B3=60, B4=60, grand total=192.

### Dataset Tiers (from platform)

| Tier   | Scenes | Access                                  |
|--------|--------|-----------------------------------------|
| Public | 3      | Full download (x_true, y, H, true spec) |
| Dev    | 3      | Blind -- only y, H, spec ranges         |
| Hidden | 3      | Fully blind -- containerized submission  |

Data format is HDF5. Ground truth is currently generated (synthetic phantom,
`shepp_logan` fallback generator). The `data_source` priority list is:
experimental > synthetic_web > generated.

### Scoring Metric

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - H_hat * x_hat|| / ||y||)
```

PSNR (40%) and SSIM (40%) measure reconstruction quality; the consistency
term (20%) penalizes solutions that fail to explain the measurements.

---

## 2. Benchmark Task Tiers (B1--B4)

**B1 -- Design (Prompt + Original-Spec --> Spec)**
Given a natural-language prompt ("Design stellar coronagraph for exoplanet
imaging: Lyot stop, 1e-8 contrast, 3 lambda/D IWA"), the model must produce a
valid instrument specification. Levels M0-M4 progressively increase from
template completion to adversarial/contradictory requirement handling.

**B2 -- Forward + Reconstruct (Spec --> Reconstruction + Feedback)**
Core inverse-problem benchmark. Given noisy measurements y and the (possibly
mismatched) forward operator H, reconstruct the underlying scene x while
estimating correction parameters. Challenges: speckle residuals, wind-driven
halo. Mismatch levels M0-M4 range from nominal to adversarial perturbation.

**B3 -- System Identification (Dataset + Prompt --> Spec)**
Estimate the speckle field and quasi-static aberration parameters from data.
The model must infer instrument configuration (DAG template, mismatch
parameters) from raw measurements alone. Progresses from M0 template ID
to M4 adversarial identification under unknown configuration.

**B4 -- Correct + Diagnose (Dataset + Spec --> Correction + Feedback)**
Apply speckle subtraction (ADI/SDI) and wavefront correction, then
provide diagnostic feedback. Target rho >= 0.80 at M3 level. Improvement
roadmap: post-processing comparison (ADI, SDI, RDI); wavefront sensing
and control loop integration.

---

## 3. State of the Art in Coronagraphic Post-Processing

### Classical Methods

- **ADI (Angular Differential Imaging)**: Exploits field rotation to
  disentangle quasi-static speckles from astrophysical sources. The telescope
  derotator keeps the pupil stable while the field rotates; objects of interest
  follow deterministic circular trajectories while speckles remain quasi-static.
- **SDI (Spectral Differential Imaging)**: Exploits chromatic scaling of
  speckles vs. companions across wavelength channels.
- **RDI (Reference Differential Imaging)**: Uses a library of reference PSFs
  from other observations to model and subtract speckles.
- **PCA / KLIP**: Principal component analysis on the ADI cube to build a
  low-rank speckle model. KLIP (Karhunen-Loeve Image Projection) is the
  standard implementation.
- **PACO**: Probabilistic Algorithm for COmpanion detection -- a
  statistics-based approach that models speckle noise locally as a multivariate
  Gaussian and performs matched filtering with a known PSF template.
- **ANDROMEDA**: ANgular DiffeRential OptiMal Exoplanet Detection Algorithm --
  uses maximum likelihood estimation on pairs of ADI frames.

### Deep Learning Methods (2024-2025)

- **deep PACO** (Flasseur et al., MNRAS 2024): Combines PACO's statistical
  preprocessing (centering + whitening) with a supervised CNN for detection
  and a second network for photometry estimation. Three-stage pipeline:
  (1) PACO-based centering/whitening, (2) CNN detection map, (3) photometry
  network for flux estimation. Uses custom data augmentation to generate
  large training sets from single spatio-temporo-spectral datasets. Achieves
  better precision-recall trade-off than PACO alone on VLT/SPHERE-IRDIS data.
  Ref: https://academic.oup.com/mnras/article/527/1/1534/7313648

- **MODEL&CO** (Bodrito et al., MNRAS 2024): Supervised deep learning that
  builds nuisance models from archives of multiple observations rather than
  the target observation alone. Casts detection as a reconstruction task
  using two complementary data representations and a highly non-linear model
  without explicit image-to-image subtraction. Superior precision-recall
  vs. PACO, with the largest gains when ADI diversity is most limited.
  Tested on VLT/SPHERE data.
  Ref: https://academic.oup.com/mnras/article/534/2/1569/7762210

- **Machine learning for HCI spectroscopy** (A&A 2024): Combines
  cross-correlation maps with deep learning on medium-resolution integral-field
  spectra for improved detection sensitivity.
  Ref: https://www.aanda.org/articles/aa/full_html/2024/09/aa49150-24/aa49150-24.html

### PWM Platform Leaderboard (current top methods)

| Rank | Method              | Overall | Public | Dev   | Hidden |
|------|---------------------|---------|--------|-------|--------|
| 1    | ANDROMEDA + gradient | 0.543   | 0.654  | 0.521 | 0.455  |
| 2    | KLIP + gradient      | 0.485   | --     | --    | --     |
| 3    | SODINN + gradient    | 0.451   | --     | --    | --     |
| 4    | cADI + gradient      | 0.384   | --     | --    | --     |

The "+ gradient" suffix indicates these methods couple classical ADI-family
algorithms with gradient-based mismatch parameter estimation, which is the
key innovation of the PWM formulation -- jointly reconstructing the scene
and correcting the forward model.

---

## 4. Local Repository Status

**Dataset directory**: `datasets/benchmark/coronagraphy/` -- DOES NOT EXIST.
No local data has been downloaded or generated for this modality.

**Configuration files present**:
- `benchmarks/configs/coronagraphy.yaml` -- base benchmark config
- `benchmarks/expanded_configs/coronagraphy_expanded.yaml` -- expanded config
  with image sizes, noise levels, mismatch levels, and total case counts
- `docs/modality_benchmarks/coronagraphy.md` -- detailed benchmark specification

**Key observations from local configs**:
- Maturity is M0 (lowest level); no real data integration yet.
- `data_source.dataset_id` and `data_source.dataset_url` are both empty strings.
- `data_source.fallback` is `generated` with `shepp_logan` phantom generator.
- `reference_psnr` and `expected_psnr_range` are both null (not yet calibrated).
- `category_module` is set to `microscopy_psf`, which appears to be a
  placeholder or cross-category module reuse.
- The `theta.density` is 0.5, controlling phantom sparsity.
- Expanded config defines 192 total evaluation cases but none are populated.
- `data_source.citation` and `data_source.license` are empty.

---

## 5. Gaps, Risks & Recommendations

### Critical Gaps

1. **No real data**: The benchmark is entirely synthetic (generated). The
   platform page references "VLBA Calibrator Survey (VCS-II) (Fomalont et al.,
   AJ 2003)" radio interferometric data, but the local config shows no actual
   dataset URL or ID. Real coronagraphic data from VLT/SPHERE, Gemini/GPI,
   or Keck/NIRC2 archives would dramatically improve ecological validity.

2. **Shepp-Logan phantom fallback**: The synthetic generator uses
   `shepp_logan`, a medical imaging phantom with no astrophysical relevance.
   Coronagraphic scenes should feature point-source companions embedded in
   a structured speckle field at realistic contrast ratios (1e-4 to 1e-8).

3. **Missing reference PSNR**: Without calibrated `reference_psnr` and
   `expected_psnr_range`, it is impossible to validate whether reconstruction
   quality is physically meaningful or to set pass/fail thresholds.

4. **category_module mismatch**: `category_module: microscopy_psf` is
   incorrect for a coronagraphy operator -- this should reference an
   astronomy-specific module with diffraction-limited PSF generation,
   coronagraph mask models, and atmospheric turbulence simulation.

### Moderate Risks

5. **Leaderboard scores are low**: The top method (ANDROMEDA + gradient)
   achieves only 0.543 overall, and 0.455 on hidden data. This could
   indicate that the benchmark is appropriately challenging, or that the
   forward model / mismatch parameterization needs refinement. Without
   reference baselines calibrated on real data, it is difficult to
   distinguish between these explanations.

6. **No deep-learning baselines**: The leaderboard shows only classical
   methods (ANDROMEDA, KLIP, SODINN, cADI) coupled with gradient estimation.
   Recent work (deep PACO, MODEL&CO) demonstrates that supervised deep
   learning substantially outperforms classical approaches on real
   VLT/SPHERE data. Adding these methods as baselines would strengthen
   the benchmark's credibility and relevance.

7. **Noise model underspecification**: The expanded config defines SNR
   levels (20-60 dB) but does not specify the photon noise / read noise
   decomposition, background sky brightness, or atmospheric seeing
   conditions, all of which are critical for realistic coronagraphic
   simulation.

### Recommendations

- **Short-term**: Replace `shepp_logan` with an astrophysically motivated
  synthetic generator (point sources + speckle field at realistic contrasts).
  Set `category_module` to an astronomy-specific module. Calibrate
  `reference_psnr` using the Adjoint solver on nominal (M0) data.

- **Medium-term**: Integrate public VLT/SPHERE-IRDIS ADI datasets (e.g.,
  from the Exoplanet Imaging Data Challenge or the SPHERE Data Centre).
  Add deep PACO and MODEL&CO as leaderboard baselines. Specify the noise
  model with photon/read/background decomposition.

- **Long-term**: Incorporate wavefront sensing and control loop simulation
  for B4-level correction tasks. Target JWST coronagraphic data and
  Roman Coronagraph simulated datasets for M3/M4 real-data tiers.

---

## 6. References

- Flasseur, O. et al. "deep PACO: combining statistical models with deep
  learning for exoplanet detection and characterization in direct imaging at
  high contrast." MNRAS 527.1 (2024): 1534.
  https://academic.oup.com/mnras/article/527/1/1534/7313648

- Bodrito, T. et al. "MODEL&CO: Exoplanet detection in angular differential
  imaging by learning across multiple observations." MNRAS 534.2 (2024): 1569.
  https://academic.oup.com/mnras/article/534/2/1569/7762210
  Also: https://arxiv.org/abs/2409.17178

- Flasseur, O. et al. "Combining statistical learning with deep learning for
  improved exoplanet detection and characterization." arXiv:2409.13031 (2024).
  https://arxiv.org/abs/2409.13031

- "Machine learning for exoplanet detection in high-contrast spectroscopy."
  A&A (2024).
  https://www.aanda.org/articles/aa/full_html/2024/09/aa49150-24/aa49150-24.html

- Males, J. R. "The mysterious lives of speckles. I. Residual atmospheric
  speckle lifetimes in ground-based coronagraphs." (2021).

- Malin, M. "ExoCAT: Exoplanet mid-infrared Coronagraphy & Analysis Tools."
  https://github.com/mathildemalin/ExoCAT

- Fomalont, E. B. et al. "VLBA Calibrator Survey (VCS-II)." AJ (2003).

- PWM Platform benchmark page:
  https://pwm.platformai.org/benchmark/coronagraphy

---

*Comprehensive 6-point review on 2026-03-03: platform spec & forward model
verified; four benchmark tiers (B1-B4) mapped across five mismatch maturity
levels (M0-M4) for 192 total cases; state-of-the-art surveyed covering
classical ADI/KLIP/PACO/ANDROMEDA and recent deep learning (deep PACO,
MODEL&CO); local repository confirmed at M0 maturity with no real data and
critical gaps in synthetic generator, category module, and reference
calibration; seven gaps/risks identified with short-, medium-, and long-term
remediation roadmap; six primary references catalogued.*