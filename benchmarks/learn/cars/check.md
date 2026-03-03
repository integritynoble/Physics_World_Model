# Comprehensive Benchmark QA Check -- CARS

**Modality**: Coherent Anti-Stokes Raman (CARS) Microscopy
**URL**: https://pwm.platformai.org/benchmark/cars
**Review Date**: 2026-03-03
**Reviewer**: Claude Opus 4.6 (automated)
**Previous check**: 2026-03-03 14:36 UTC (scripts/check_modality.py v2 -- PASS, 0 errors)

---

## 1. Benchmark Page Errors

### 1.1 Extracted Page Data

| Field | Value |
|-------|-------|
| Title | Coherent Anti-Stokes Raman (CARS) Microscopy -- Physics World Model |
| DAG | M --> R --> D (Modulation -> Rotation -> Detector) |
| Forward Model | nonlinear_operator (microscopy_psf category module) |
| Scoring Formula | 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - \|\|y - H_hat x_hat\|\| / \|\|y\|\|) |
| Dataset Source | RRUFF Raman Database (Lafuente et al., Handbook of Mineral Spectroscopy, 2016; Univ. Arizona) |
| Dataset URL | https://rruff.info/ |
| License | Public domain |

#### Leaderboard (from benchmark page)

| Rank | Method | Overall | Public PSNR/SSIM | Dev PSNR/SSIM | Hidden PSNR/SSIM |
|------|--------|---------|------------------|---------------|-------------------|
| 1 | Cascade-UNet + gradient | 0.677 | 31.15 dB / 0.933 | 26.9 dB / 0.856 | 23.75 dB / 0.760 |
| 2 | CDAE + gradient | 0.615 | 29.01 dB / 0.900 | 23.6 dB / 0.754 | 20.65 dB / 0.629 |
| 3 | SG-ALS + gradient | 0.561 | 22.41 dB / 0.707 | 21.54 dB / 0.670 | 21.72 dB / 0.678 |
| 4 | PnP-DnCNN + gradient | 0.560 | 25.39 dB / 0.814 | 21.95 dB / 0.688 | 18.61 dB / 0.530 |

#### Samples per Tier

| Tier | Scenes | Access |
|------|--------|--------|
| Public | 5 | Full (ground truth + ideal H + spectral ranges) |
| Dev | 5 | Blind evaluation (measurements y, H, ranges only) |
| Hidden | 5 | Fully blind, server-side containerized evaluation |

#### Mismatch Parameters (from benchmark page vs. local config)

| Parameter | Page Public | Page Dev | Page Hidden | Local Config Range | Unit |
|-----------|-------------|----------|-------------|--------------------|------|
| Pump-Stokes freq offset (p_f) | [-1.0, 2.0] | [-1.2, 1.8] | [-0.7, 2.3] | [-5.0, 5.0] | cm^-1 |
| Non-resonant background (n_b) | [-10.0, 20.0] | [-12.0, 18.0] | [-7.0, 23.0] | [0.0, 50.0] | dimensionless |
| Chirp mismatch (c_m) | [-100.0, 200.0] | [-120.0, 180.0] | [-70.0, 230.0] | [0.0, 500.0] | fs^2 |

### 1.2 Issues Found

#### HIGH Severity

**H1. Scoring formula inconsistency between page and local config.**
The benchmark page uses a 3-component composite metric (40% PSNR + 40% SSIM + 20% consistency)
but the local config (benchmarks/configs/cars.yaml lines 69-74) lists only psnr and ssim
as metrics, with psnr as primary. The consistency term (1 - ||y - H_hat x_hat|| / ||y||) is
not referenced in the config at all. This makes it unclear whether the composite score is actually
computed or whether the page description is aspirational.

**H2. Mismatch parameter ranges: page vs. config discrepancy.**
The benchmark page shows per-tier mismatch ranges (e.g., p_f in [-1.0, 2.0] for Public) that are
much narrower and include negative values, while the local YAML config shows a single global range
(e.g., p_f in [-5.0, 5.0]). Notably, the config has NRB range [0.0, 50.0] (non-negative) while
the page shows negative NRB values (e.g., [-10.0, 20.0]). Negative non-resonant background is
physically questionable -- NRB is an intensity contribution and should be non-negative. This could
indicate a sign convention error on the page or a modelling issue.

**H3. Forward model equation is generic, not CARS-specific.**
The forward model equation in 02_forward_model.md is listed as y = PSF * x + noise (simple
PSF convolution), which is the generic microscopy_psf template. CARS has a specific forward model
involving the third-order nonlinear susceptibility chi^(3), the non-resonant background chi_NR,
and the pump/Stokes/anti-Stokes electric fields. The true CARS signal is proportional to
|chi^(3)_R + chi_NR|^2, making it inherently nonlinear in the resonant susceptibility. The
current equation is misleading for CARS.

**H4. Algorithm type mislabeling: Cascade-UNet labeled as "Transformer".**
In modify_plan.md, Cascade-UNet is listed with type "Transformer" but it is a UNet architecture.
This metadata error propagates to the platform if the catalog is the source of truth.

#### MEDIUM Severity

**M1. Stale algorithm names in previous check.md.**
The prior check.md (from automated script) listed "RamanNet" and "MCR-ALS" as leaderboard
methods, but the actual benchmark page shows "CDAE" and "SG-ALS". Either the page was updated
after the automated check or the script extracted names incorrectly.

**M2. "Coded Mask" hardware element is incorrect for CARS.**
In 01_physics_fundamentals.md (line 51), the imaging chain includes a "Coded Mask" modulation
element with 0.5 throughput. CARS microscopy does not use coded masks -- it uses a two-beam
(pump + Stokes) excitation geometry. This appears to be a templated artifact from the
coded-aperture modality (e.g., CASSI/CACTI) that was not customized for CARS.

**M3. Wavelength range is too narrow for CARS.**
The physics fundamentals file lists 400-700 nm (visible only), but CARS typically uses near-IR
pump lasers (750-1064 nm) and Stokes beams, with anti-Stokes signal generated at shorter
wavelengths. The spectral range should be approximately 500-1100 nm to cover the full CARS
excitation/detection window.

**M4. Image shape inconsistency across configs.**
The base config (cars.yaml) uses x_shape [64, 64], but the expanded config
(cars_expanded.yaml) offers small [128, 128], standard [256, 256], and large [512, 512]. It is
unclear which shape the benchmark page leaderboard scores correspond to.

**M5. Default solver naming mismatch.**
The config declares default_solver: nrb_removal, but the solver table in
03_reconstruction_algorithms.md lists traditional_cpu: Adjoint and best_quality: PnP-ADMM.
There is no solver called nrb_removal in the solver comparison table. Either the default solver
is not listed or it maps to one of the two named solvers.

#### LOW Severity

**L1. Density parameter (theta.density = 0.5) is unexplained.**
The config includes theta: density: 0.5 but there is no documentation explaining what "density"
means in the CARS context. In coded-aperture systems, density refers to mask transparency, but
CARS does not have a coded mask.

**L2. Missing reference_psnr and expected_psnr_range.**
The config has reference_psnr: null and expected_psnr_range: null. These should be populated
based on the leaderboard baseline (e.g., reference_psnr ~22-31 dB based on current results).

**L3. Maturity M0 is low for a modality with a functioning leaderboard.**
If the page has 4 algorithms with real scores across 3 tiers, the maturity should be at least M1
(single-parameter mismatch validated) or M2.

---

## 2. Local Dataset Inspection

**Result**: No local dataset directory found.

```
$ ls datasets/benchmark/cars 2>/dev/null
(no output -- directory does not exist)
```

The datasets/benchmark/ directory contains: cacti, cbct, cryo_em, ct, mri, sd_cassi,
spc_kronecker, ultrasound. CARS is absent.

**Assessment**: The benchmark page references HDF5 files on GCS (Google Cloud Storage) for public
and dev tiers, confirmed accessible by the automated check. However, there are no local dataset
files for offline development or testing. This is consistent with other spectroscopy modalities
that may use web-sourced data (RRUFF), but it means the benchmark cannot be run locally without
first downloading from GCS.

**Data source chain**: RRUFF Raman Database (https://rruff.info/) -> synthetic generation via
shepp_logan fallback -> HDF5 packaging -> GCS hosting. The "generated" fallback using
shepp_logan phantoms is problematic because Shepp-Logan is an X-ray CT phantom, not a
spectroscopic phantom -- the spectral characteristics of Shepp-Logan have no physical relationship
to Raman/CARS spectra.

---

## 3. Public Dataset Source Assessment

### RRUFF Raman Database

| Property | Details |
|----------|---------|
| URL | https://rruff.info/ |
| Maintainer | University of Arizona |
| License | Public domain |
| Content | ~4,000+ mineral Raman spectra (oriented/unoriented, multiple excitation wavelengths) |
| Format | XY text files (wavenumber vs. intensity) |
| Relevance | MEDIUM -- contains Raman spectra but NOT CARS spectra specifically |

**Assessment**: RRUFF provides high-quality spontaneous Raman spectra of minerals. These are
**not** CARS spectra. CARS spectra differ from spontaneous Raman because:
1. CARS signal is proportional to |chi^(3)_R + chi_NR|^2, introducing the non-resonant background
2. CARS spectra have dispersive line shapes (not purely Lorentzian like spontaneous Raman)
3. CARS spectra require phase retrieval to extract the Raman-equivalent information

Using RRUFF Raman spectra as ground truth for a CARS benchmark is reasonable IF the forward model
correctly simulates the CARS process (adding NRB, applying spectral phase distortion, etc.) and
the reconstruction goal is to recover the underlying Raman spectrum. However, this should be
explicitly documented -- the ground truth is the Raman spectrum, not the raw CARS spectrum.

### Alternative/Complementary Datasets

| Dataset | Content | Access |
|---------|---------|--------|
| RRUFF (current) | Mineral Raman spectra | Public, https://rruff.info/ |
| SERS-active nanostructures | SERS/CARS biomedical spectra | Various publications |
| Biomedical CARS datasets | Lipid/protein CARS imaging | Request-based (Camp et al.) |
| KK-CARS reference data | Phase-retrieved CARS spectra with NRB ground truth | Various publications |

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard

| Algorithm | Type | CARS-Specific? | Notes |
|-----------|------|----------------|-------|
| Cascade-UNet + gradient | Deep Learning (UNet) | No | Generic image reconstruction; mislabeled as Transformer |
| CDAE + gradient | Deep Learning (Autoencoder) | No | Generic spectral denoising |
| SG-ALS + gradient | Classical | Partially | Savitzky-Golay + ALS baseline removal is used in spectroscopy but not the canonical CARS method |
| PnP-DnCNN + gradient | Plug-and-Play | No | Generic PnP denoiser |

### Missing Algorithms (Should Be Added)

| Algorithm | Type | Why Important | Key Reference |
|-----------|------|---------------|---------------|
| **Kramers-Kronig (KK) Transform** | Classical | THE standard method for CARS NRB removal; transforms complex CARS spectrum to retrieve Raman-equivalent | Liu et al., Opt. Lett. 34, 1363 (2009) |
| **Maximum Entropy Method (MEM)** | Classical | Second standard method for CARS phase retrieval; better noise robustness than KK | Vartiainen et al., Opt. Express 14, 3622 (2006) |
| **Time-Domain Kramers-Kronig (TDKK)** | Classical | Improved KK variant for broadband CARS | Camp et al., J. Raman Spectrosc. 47, 408 (2016) |
| **VECTOR (deep autoencoder for NRB removal)** | Deep Learning | Purpose-built DL method for CARS NRB removal | Referenced in Springer Nature 2024 |
| **GAN-based NRB removal** | Deep Learning | Generative adversarial network for NRB removal | Yao et al., Laser Photonics Rev. 2024 |
| **MCR-ALS (Multivariate Curve Resolution)** | Classical/Statistical | Standard chemometric decomposition used in CARS hyperspectral analysis | de Juan & Tauler, Crit. Rev. Anal. Chem. 36, 163 (2006) |
| **Phase-retrieval + total variation (TV)** | Optimization | Combines phase retrieval with TV regularization for CARS image denoising | Various 2020-2024 |

### Gap Analysis

The leaderboard has **zero CARS-specific** algorithms. All four methods are generic spectral
processing or image reconstruction techniques. For a CARS benchmark, the absence of KK transform
and MEM is a critical gap -- these are the two universally recognized standard methods in the CARS
community. Any CARS researcher visiting this benchmark would immediately notice their absence.

---

## 5. Improvement Suggestions

### Priority 1 (Critical -- Must Fix)

1. **Add KK-transform and MEM as baseline algorithms.** These are non-negotiable for a credible
   CARS benchmark. They are fast, well-understood, and form the baseline against which all DL
   methods are compared in the CARS literature.

2. **Replace generic forward model equation with CARS-specific physics.** The signal equation
   should be: I_CARS(omega) = |chi^(3)_NR + sum_j A_j / (omega - Omega_j + i*Gamma_j)|^2
   where chi^(3)_NR is the non-resonant background, A_j are oscillator strengths, Omega_j are
   resonance frequencies, and Gamma_j are linewidths.

3. **Remove "Coded Mask" from the hardware imaging chain.** Replace with the actual CARS optical
   elements: pump laser, Stokes laser, dichroic beamsplitter, bandpass filter, PMT/spectrometer.

### Priority 2 (High -- Should Fix)

4. **Reconcile scoring formula.** Either implement the 3-component composite score in the local
   config/metrics code, or update the benchmark page to reflect the actual scoring (PSNR primary).

5. **Fix mismatch range discrepancies.** Align the per-tier ranges on the page with the global
   range in the config. Clarify the sign convention for NRB (should be non-negative).

6. **Correct Cascade-UNet type label** from "Transformer" to "UNet/CNN".

7. **Expand wavelength range** in physics fundamentals from 400-700 nm to 500-1100 nm.

### Priority 3 (Medium -- Nice to Have)

8. **Add local dataset** under datasets/benchmark/cars/ with at least a small public-tier
   subset for offline development.

9. **Replace Shepp-Logan fallback** with a spectroscopically meaningful synthetic generator
   (e.g., Lorentzian peak mixture simulator).

10. **Document the RRUFF-to-CARS data pipeline** explicitly, explaining that ground truth is the
    Raman spectrum and the forward model simulates CARS acquisition.

11. **Update maturity from M0** to M1 or M2 given the existing leaderboard with 3-tier results.

12. **Populate reference_psnr** and expected_psnr_range in the config based on current leaderboard
    data.

---

## 6. Action Items

| # | Priority | Item | Owner | Status |
|---|----------|------|-------|--------|
| 1 | P1 | Implement KK-transform solver in pwm_core.recon and add to leaderboard | Benchmark team | TODO |
| 2 | P1 | Implement MEM solver and add to leaderboard | Benchmark team | TODO |
| 3 | P1 | Rewrite 02_forward_model.md with CARS-specific signal equation (chi^(3) model) | Learn content | TODO |
| 4 | P1 | Remove "Coded Mask" from 01_physics_fundamentals.md hardware chain | Learn content | TODO |
| 5 | P2 | Add consistency metric to local config or remove from page scoring formula | Config / Platform | TODO |
| 6 | P2 | Align mismatch ranges (page vs. config); fix NRB sign convention | Config / Platform | TODO |
| 7 | P2 | Fix Cascade-UNet type label from "Transformer" to "UNet" in catalog | Catalog | TODO |
| 8 | P2 | Update wavelength range to 500-1100 nm | Learn content | TODO |
| 9 | P3 | Create datasets/benchmark/cars/ with public-tier HDF5 subset | Data pipeline | TODO |
| 10 | P3 | Replace Shepp-Logan synthetic generator with spectral phantom | Forward model | TODO |
| 11 | P3 | Document RRUFF -> CARS forward simulation pipeline | Learn content | TODO |
| 12 | P3 | Update maturity to M1+ and populate reference_psnr | Config | TODO |
| 13 | P3 | Add VECTOR and GAN-based NRB removal methods from 2024 literature | Benchmark team | TODO |

---

## Appendix: Key References

1. **Yao et al.** (2024). "Recent Progress in Deep Learning for Improving Coherent Anti-Stokes Raman Scattering Microscopy." *Laser & Photonics Reviews* 18(11). https://onlinelibrary.wiley.com/doi/full/10.1002/lpor.202400562

2. **Super-resolved CARS by coherent image scanning** (2024). *Nature Communications*. https://www.nature.com/articles/s41467-024-54429-1

3. **Deep Learning-Assisted Dynamic Mode Decomposition for NRB Removal in CARS Spectroscopy** (2024). *Springer Nature*. https://link.springer.com/chapter/10.1007/978-3-032-12840-9_4

4. **Machine learning empowered coherent Raman imaging and analysis for biomedical applications** (2025). *Communications Engineering*. https://www.nature.com/articles/s44172-025-00345-1

5. **Hyperspectral microscopy with Computational field-resolved CARS** (2025). *CLEO*. https://opg.optica.org/abstract.cfm?uri=CLEO_SI-2025-SS143_6

6. **Coherent Stokes Raman scattering microscopy (CSRS)** (2023). *Nature Communications*. https://www.nature.com/articles/s41467-023-38941-4

7. **RRUFF Raman Database** -- https://rruff.info/ (ground truth source)

8. **Liu et al.** (2009). "Broadband CARS spectral phase retrieval using a time-domain Kramers-Kronig transform." *Opt. Lett.* 34, 1363.

9. **Vartiainen et al.** (2006). "Direct extraction of Raman line-shapes from congested CARS spectra." *Opt. Express* 14, 3622.

10. **Camp et al.** (2016). "Quantitative, comparable coherent anti-Stokes Raman scattering (CARS) spectroscopy: correcting errors in phase retrieval." *J. Raman Spectrosc.* 47, 408.

---

*Comprehensive 6-point review on 2026-03-03. Benchmark page fetched from https://pwm.platformai.org/benchmark/cars. Local codebase inspected at /home/spiritai/abraham/pwm/production/Physics_World_Model. Web search performed for 2024-2025 CARS reconstruction literature. 4 HIGH, 5 MEDIUM, and 3 LOW severity issues identified. 13 action items prioritized across 3 tiers.*