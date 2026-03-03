# Benchmark Review -- confocal_3d (3D Confocal Microscopy)

**URL:** <https://pwm.platformai.org/benchmark/confocal_3d>
**Review Date:** 2026-03-03
**Modality:** Confocal 3D Z-Stack | Category: Microscopy | Carrier: Photon
**Maturity:** M1 | **Tier:** A

---

## 1. Benchmark Design and Task Definition

### Task

Reconstruct a 3D fluorescence volume from confocal z-stack measurements acquired
with unknown system parameter mismatches. Given measurements **y**, an ideal
forward operator **H**, and specification parameter ranges (but not exact values),
the algorithm must simultaneously recover the original signal **x** and correct
the mismatched system parameters.

### Forward Model

```
y(x,y,z) = PSF_3d *** x(x,y,z) + n
```

where `***` denotes 3D convolution with an anisotropic confocal PSF and `n`
encompasses Poisson shot noise and Gaussian read noise.

**Spec DAG pipeline:** `C(PSF_3D) --> D(g, eta_3)`
- **C:** 3D Confocal PSF operator (microscopy_psf module)
- **D:** PMT/HyD detector with gain (g) and noise model (eta_3)

### Mismatch Parameters (4 variables in config, 3 surfaced on web)

| Parameter             | Nominal | Config Range    | Web Public Range  | Unit       |
|-----------------------|---------|-----------------|-------------------|------------|
| Axial PSF sigma       | 3.0     | 1.5 -- 6.0      | --                | px         |
| Refractive index      | 1.515   | 1.33 -- 1.56    | 1.505 -- 1.535    | --         |
| Attenuation coeff     | 0.03    | 0.0 -- 0.1      | --                | per slice  |
| Spherical aberration  | 0.0     | 0.0 -- 0.5      | -0.1 -- 0.2       | waves      |
| Z-step error          | --      | --              | -50 -- 100        | nm         |

**Observation:** There is a discrepancy between the local YAML config
(4 mismatch parameters: axial PSF sigma, refractive index, attenuation coeff,
spherical aberration) and the web page (3 parameters: z-step error, spherical
aberration, refractive index). The ranges also differ -- the config uses wider
ranges while the web page uses narrower, tier-specific ranges. This may reflect
a simplification for the public challenge vs. the internal benchmark, but it
should be reconciled or documented explicitly.

### Three-Tier Evaluation Structure

| Tier       | Scenes | Access              | Mismatch Severity |
|------------|--------|---------------------|-------------------|
| **Public** | 5      | Full (GT + params)  | Mild              |
| **Dev**    | 5      | Blind (no GT)       | Moderate          |
| **Hidden** | 5      | Server-side only    | Severe            |

Total: 15 test scenes across three evaluation splits.

### Data Specifications

| Property               | Value                              |
|------------------------|------------------------------------|
| Signal shape (x)       | [256, 256, 64]                     |
| Measurement shape (y)  | [256, 256, 64]                     |
| Processing volume      | (32, 64, 64) voxels (web)          |
| Lateral pixel size     | 80 nm                              |
| Lateral resolution     | 180 nm                             |
| Excitation wavelength  | 561 nm DPSS                        |
| Objective              | Plan Apo 63x / 1.40 NA oil        |
| Pinhole                | 1.0 Airy unit                      |
| Z-step                 | 300 nm                             |
| Z-slices               | 64                                 |
| Dwell time             | 8 us                               |
| Bit depth              | 12-bit                             |
| Data format            | HDF5                               |

**Data Source:** CARE Tribolium dataset (Weigert et al., Nature Methods 2018),
with fallback to synthetic cell_phantom generator. License: CC-BY-4.0.

---

## 2. Evaluation Metrics and Baselines

### Composite Score Formula (from web)

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - H_hat * x_hat|| / ||y||)
```

- PSNR (40%) -- peak signal-to-noise ratio
- SSIM (40%) -- structural similarity index
- Consistency (20%) -- forward model residual (physics fidelity)

**Note:** The local config lists metrics as `[psnr, ssim, sam]` with `psnr` as
primary. The web page uses a composite score with consistency weighting. The
inclusion of SAM (spectral angle mapper) in the config is unusual for
single-channel fluorescence data -- this should be reviewed.

### Challenge Leaderboard (from web)

| Rank | Method                    | Overall | Public Score | Dev Score  | Hidden Score |
|------|---------------------------|---------|--------------|------------|--------------|
| 1    | Restormer + gradient      | 0.767   | 0.796 (33.66 dB / 0.958) | 0.770 (31.79 dB / 0.940) | 0.736 (30.71 dB / 0.927) |
| 2    | CARE + gradient           | 0.674   | 0.777 (32.43 dB / 0.947) | 0.673 (26.48 dB / 0.845) | 0.572 (23.07 dB / 0.734) |
| 3    | PnP-FISTA + gradient      | 0.672   | 0.711 (28.04 dB / 0.882) | 0.677 (27.06 dB / 0.860) | 0.629 (25.45 dB / 0.816) |
| 4    | Richardson-Lucy + gradient| 0.645   | 0.669 (25.34 dB / 0.813) | 0.653 (25.64 dB / 0.822) | 0.613 (24.39 dB / 0.782) |

### Reconstruction Gallery (3D Richardson-Lucy)

| Scenario                   | PSNR (dB) | SSIM   |
|----------------------------|-----------|--------|
| Ideal (no mismatch)        | 36.31     | 0.9475 |
| With mismatch (uncorrected)| 32.67     | 0.8807 |
| Oracle correction          | 31.46     | 0.8428 |

**Key insight:** Mismatch causes ~3.6 dB degradation. The gap between uncorrected
and oracle is only ~1.2 dB, indicating that parameter correction alone does not
fully recover performance -- the mismatch introduces non-trivial structural
artifacts that persist even with known parameters.

### Local Solver Comparison (from config)

| Tier            | Solver                    | Module                         | GPU  | Params |
|-----------------|---------------------------|--------------------------------|------|--------|
| traditional_cpu | 3D Richardson-Lucy        | pwm_core.recon.richardson_lucy | No   | 0      |
| best_quality    | 3D CARE                   | pwm_core.recon.care_unet       | Yes  | 2M     |
| famous_dl       | CARE-3D                   | pwm_core.recon.care_unet       | No   | 2M     |
| small_gpu       | CARE-3D (slice-wise)      | pwm_core.recon.care_unet       | No   | 0      |

**Observation:** The leaderboard features Restormer and PnP-FISTA, but the local
solver registry only includes Richardson-Lucy and CARE variants. The config should
be updated to include the leaderboard algorithms.

---

## 3. Physics and Forward Model Assessment

### Physics Fidelity

The benchmark accurately models the core physics of confocal microscopy:

- **Anisotropic 3D PSF:** Correctly modeled with separate lateral (sigma=1.0 px)
  and axial (sigma=3.5 px) components. The ~3.5:1 axial-to-lateral ratio is
  physically realistic for a 1.4 NA oil immersion objective at 561 nm.

- **Depth-dependent degradation:** Attenuation coefficient (0.03 per slice) and
  spherical aberration parameters capture the key depth-dependent effects in
  confocal imaging -- signal loss and PSF broadening with increasing depth.

- **Noise model:** Poisson (shot noise) + Gaussian (read noise, dark current)
  is the standard and correct noise model for PMT-based confocal detection.

- **Pinhole effect:** 1.0 AU pinhole correctly models the standard confocal
  configuration, balancing sectioning strength against signal throughput.

### Identified Physics Gaps

1. **Photobleaching:** Not modeled. In real confocal z-stacks, fluorophore
   photobleaching during acquisition causes signal reduction in later-acquired
   slices. This is a significant effect for thick specimens.

2. **Chromatic aberration:** Not included. For multi-channel confocal data this
   would be important, though single-channel (561 nm) mitigates this.

3. **Scattering:** The attenuation coefficient captures absorption but does not
   model scattering-induced PSF broadening at depth, which is significant in
   biological tissue beyond ~50 um.

4. **Detector nonlinearity:** PMT response can saturate at high photon rates.
   The current model assumes linear detection.

### Common Pitfalls (from web page)

The benchmark documentation correctly warns against:
- Z-step larger than Nyquist causing axial aliasing
- 2D slice-by-slice deconvolution instead of true 3D processing
- Neglecting depth-dependent spherical aberration
- Using incorrect PSF models (2D Gaussian vs. 3D Born & Wolf)
- Ignoring signal attenuation with depth

These are well-chosen and reflect real-world confocal imaging errors.

---

## 4. Literature and State-of-the-Art Comparison

### Classical Methods

- **Richardson-Lucy (1972/1974):** The gold standard for fluorescence microscopy
  deconvolution. Correctly included as the traditional baseline. Maximum
  likelihood estimator under Poisson noise -- an excellent fit for photon-limited
  confocal data.

- **PnP-FISTA (Bai et al., 2020):** Plug-and-Play with FISTA optimization.
  Appropriate for microscopy deconvolution where the forward model (PSF
  convolution) is well-defined and differentiable.

- **Wiener/Tikhonov deconvolution:** Mentioned in the web documentation but not
  on the leaderboard. Simple linear methods that serve as useful lower-bound
  baselines.

### Deep Learning Methods (2024--2025 literature)

Recent advances that are relevant to this benchmark:

1. **Zero-shot deconvolution (ZS-DeconvNet, Nature Comms 2024):** Enhances
   resolution by >1.5x beyond the diffraction limit without ground truth or
   additional acquisitions. Applicable to confocal microscopy. This self-
   supervised approach could be a strong competitor on the hidden tier where
   no training data is available.

2. **Physics-informed diffusion models (Comms Engineering 2024):** Incorporates
   the physics of light propagation into conditional diffusion models for
   microscopy image reconstruction. Outperforms CARE and other supervised
   baselines on deconvolution tasks.

3. **Deep learning aberration compensation (Nature Comms 2024, DeAbe):**
   Content-aware aberration correction for volumetric confocal, light-sheet,
   and multi-photon data. Directly addresses the spherical aberration mismatch
   parameter in this benchmark.

4. **Physics-guided autoencoders for CLSM (Scientific Reports 2025):**
   Incorporates the confocal PSF and noise model as physics-based constraints
   in a convolutional autoencoder. Demonstrates superiority over Richardson-Lucy,
   NNLS, and total variation methods.

5. **Restormer (Zamir et al., CVPR 2022):** Currently leads the benchmark
   leaderboard. A general-purpose image restoration Transformer that has been
   successfully applied to microscopy. Its strong performance here validates
   the transferability of vision Transformers to scientific imaging.

6. **CARE (Weigert et al., Nature Methods 2018):** The field-defining paper
   for deep learning in fluorescence microscopy. ~2500+ citations. Directly
   designed for confocal z-stack restoration. Appropriately included as
   both a benchmark solver and the data source (Tribolium dataset).

### Gap Analysis vs. State of the Art

The benchmark is **missing several 2024--2025 methods** that could significantly
improve upon the current leader (0.767):

- Physics-informed approaches (diffusion models, physics-guided autoencoders)
  that explicitly encode the confocal PSF into the reconstruction
- Self-supervised methods (ZS-DeconvNet) that do not require paired training data
- Aberration-aware networks (DeAbe) purpose-built for the aberration mismatch
  that this benchmark tests
- 3D U-Net and RCAN3D architectures referenced in the web documentation but
  absent from the leaderboard

---

## 5. Local Data and Infrastructure Status

### Local Dataset

**No local dataset directory found** at `datasets/benchmark/confocal_3d/`.
The benchmark config specifies the CARE Tribolium dataset (Weigert et al., 2018)
from `https://publications.mpi-cbg.de/publications-sites/7207/` with a fallback
to synthetic generation via `cell_phantom`.

### Local Infrastructure

| Component                        | Status     | Path / Notes                              |
|----------------------------------|------------|-------------------------------------------|
| Benchmark config                 | EXISTS     | `benchmarks/configs/confocal_3d.yaml`     |
| Learning materials (6 files)     | EXISTS     | `benchmarks/learn/confocal_3d/`           |
| -- README.md                     | OK (1,426 B) |                                        |
| -- 01_physics_fundamentals.md    | OK (2,873 B) |                                        |
| -- 02_forward_model.md           | OK (2,727 B) |                                        |
| -- 03_reconstruction_algorithms.md | OK (2,549 B) |                                      |
| -- 04_pwm_benchmark.md           | OK (2,584 B) |                                        |
| -- 05_hands_on_tutorial.md       | OK (3,572 B) |                                        |
| Dataset directory                | MISSING    | `datasets/benchmark/confocal_3d/`         |
| Expanded config                  | UNVERIFIED | `benchmarks/expanded_configs/confocal_3d_expanded.yaml` |
| Forward model operator           | DECLARED   | `has_dedicated_operator: true`            |

### Web Page QA (from automated check)

| Check                              | Result |
|-------------------------------------|--------|
| Main page loads (HTTP 200)          | PASS   |
| Title correct                       | PASS   |
| Spec notation matches DAG           | PASS   |
| Challenge leaderboard present       | PASS (4 entries) |
| Gallery images load                 | PASS (24/24) |
| Challenge public/dev pages          | PASS   |
| HDF5 files on GCS                   | PASS   |
| Compete/Contribute pages            | PASS   |
| Forward model reference             | PASS   |
| Learning materials                  | PASS (all 6 files) |
| Errors / Warnings                   | 0 errors, 0 warnings, 2 info |

### Discrepancy: check.md vs. modify_plan.md

The automated check.md previously listed "Noise2Void + gradient" and "Wiener
Deconv + gradient" as leaderboard methods, but the modify_plan.md notes these
are stale -- the actual catalog algorithms are Richardson-Lucy, PnP-FISTA, CARE,
and Restormer (confirmed by the web page leaderboard).

---

## 6. Findings, Recommendations, and Action Items

### Overall Assessment: SOLID BENCHMARK WITH MINOR GAPS

The confocal_3d benchmark is one of the strongest modality-algorithm fits in the
PWM suite. The physics model is faithful, the mismatch parameters target
real-world calibration errors, the three-tier evaluation structure provides
rigorous generalization testing, and the leaderboard has meaningful spread
(0.645 to 0.767) indicating room for improvement without ceiling effects.

### Findings

| # | Severity | Finding |
|---|----------|---------|
| F1 | WARNING | Mismatch parameter discrepancy between config (4 params) and web (3 params) |
| F2 | WARNING | SAM metric in config is inappropriate for single-channel fluorescence |
| F3 | INFO    | Local solver registry missing Restormer and PnP-FISTA (leaderboard methods) |
| F4 | INFO    | No local dataset -- depends on remote download or synthetic fallback |
| F5 | INFO    | Processing volume on web (32,64,64) differs from config shape (256,256,64) |
| F6 | INFO    | reference_psnr and expected_psnr_range are null in config |
| F7 | INFO    | Stale method names in prior automated check (Noise2Void, Wiener vs actual) |

### Recommendations

1. **Reconcile mismatch parameters** (F1): Document why the web page shows 3
   parameters (z-step error, spherical aberration, refractive index) while the
   config has 4 (axial PSF sigma, refractive index, attenuation coeff, spherical
   aberration). If they are intentionally different, add a mapping note.

2. **Remove or replace SAM metric** (F2): SAM is designed for multi-spectral
   data and is meaningless for single-channel confocal fluorescence. Replace
   with a consistency metric (forward model residual) to match the web page
   composite score.

3. **Add leaderboard solvers to config** (F3): Register Restormer and PnP-FISTA
   in the solver registry so local benchmarking can reproduce the leaderboard.

4. **Populate reference_psnr** (F6): Set `reference_psnr` to the Richardson-Lucy
   baseline (~25.34 dB public) and `expected_psnr_range` to [25, 35] based on
   the leaderboard spread.

5. **Add 2024--2025 baselines:** Consider adding ZS-DeconvNet (zero-shot),
   physics-informed diffusion models, or DeAbe (aberration-aware) to the
   leaderboard to keep the benchmark current with the literature.

6. **Download / generate local data:** Create `datasets/benchmark/confocal_3d/`
   with at least the public tier for local development and CI testing.

### Key References

- McNally et al., "Three-dimensional imaging by deconvolution microscopy," Methods 1999
- Weigert et al., "Content-Aware Image Restoration (CARE)," Nature Methods 2018
- Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR 2022
- Li et al., "Zero-shot denoising and super-resolution (ZS-DeconvNet)," Nature Comms 2024
- Aschenbrenner & Bhatt, "Physics-informed denoising diffusion for microscopy," Comms Engineering 2024
- Kang et al., "Deep learning-based aberration compensation (DeAbe)," Nature Comms 2024
- Alqahtani et al., "Enhanced confocal microscopy with physics-guided autoencoders," Scientific Reports 2025

---

*Comprehensive 6-point review on 2026-03-03. Sources: [PWM benchmark page](https://pwm.platformai.org/benchmark/confocal_3d), web literature search (2024--2025), local config/learning materials analysis, and automated QA checks.*