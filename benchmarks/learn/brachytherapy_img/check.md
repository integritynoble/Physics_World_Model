# Comprehensive Benchmark QA Check — Brachytherapy Imaging

**URL:** https://pwm.platformai.org/benchmark/brachytherapy_img
**HTTP Status:** 200
**Check Date:** 2026-03-03 (comprehensive 6-point review)
**Reviewer:** Local server (automated + manual deep analysis)

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
| HIGH     | 6     |
| MEDIUM   | 7     |
| LOW      | 5     |

### HIGH Severity

**H1. Config x_shape [64, 64] contradicts expanded config standard size [512, 512]**
- `brachytherapy_img.yaml` declares x_shape [64, 64] and y_shape [64, 64]
- `brachytherapy_img_expanded.yaml` declares standard x_shape [512, 512] and small x_shape [256, 256]
- The physics fundamentals also state image shape [64, 64] and measurement shape [64, 64]
- These are irreconcilable: 64x64 is too small for clinical brachytherapy imaging
**Fix:** Decide on canonical resolution. Standard brachytherapy CT is typically 512x512. Update all configs to match.

**H2. PSNR_norm undefined in scoring formula**
- Webpage scoring formula: 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - ||y - Hx_hat|| / ||y||)
- PSNR_norm normalization method not defined (min-max? reference baseline? dynamic range?)
- Without a definition, scores are not reproducible
**Fix:** Define PSNR_norm = (PSNR - PSNR_min) / (PSNR_max - PSNR_min) with explicit bounds, or use PSNR / PSNR_ref.

**H3. Mismatch ranges on webpage differ from local config YAML**
- Webpage: source_position_error [-0.4, 0.8] mm (Public), [-0.48, 0.72] (Dev), [-0.28, 0.92] (Hidden)
- Local config: source_position_error [-2.0, 2.0] mm (global)
- Webpage: attenuation_coefficient [0.19, 0.22] 1/cm (Public)
- Local config: attenuation_coefficient [0.15, 0.25] 1/cm (global)
- Similar discrepancies for detector_gain_drift and scatter_fraction
- Webpage ranges are asymmetric and tier-specific; config has symmetric global ranges
**Fix:** Either (a) update config to have per-tier ranges matching webpage, or (b) sync webpage to match config. Per-tier ranges are preferred for the PWM mismatch philosophy.

**H4. Mismatch ranges do not monotonically increase in difficulty across tiers**
- source_position_error: Public [-0.4, 0.8] span=1.2, Dev [-0.48, 0.72] span=1.2, Hidden [-0.28, 0.92] span=1.2
- All three tiers have equal span (1.2 mm) -- difficulty does NOT increase
- attenuation_coefficient: Public [0.19, 0.22] span=0.03, Dev [0.188, 0.218] span=0.03, Hidden [0.193, 0.223] span=0.03
- Again equal spans across all tiers
- This violates the PWM mismatch philosophy: Public < Dev < Hidden
**Fix:** Redesign mismatch ranges so hidden tier has strictly wider (harder) ranges than dev, and dev wider than public.

**H5. Expanded config mismatch_params is empty list**
- `brachytherapy_img_expanded.yaml` has `mismatch_params: []` (empty)
- But `brachytherapy_img.yaml` has 4 mismatch parameters defined
- The expanded config should be a superset of the base config
**Fix:** Populate mismatch_params in expanded config with at least the 4 parameters from the base config.

**H6. Dataset source references AAPM Low-Dose CT Grand Challenge but modality is brachytherapy, not CT**
- Webpage references "AAPM Low-Dose CT Grand Challenge (McCollough et al., Med. Phys. 2017)"
- Brachytherapy imaging is fundamentally different from diagnostic CT: it involves radioactive source placement within/near tissue for therapy, not external X-ray beam CT
- Using CT phantoms for brachytherapy imaging is a domain mismatch
**Fix:** Use brachytherapy-specific phantoms or datasets (e.g., TG-43 dose distributions, Monte Carlo simulated brachytherapy fields, or clinical brachytherapy planning CTs).

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Webpage shows 3 scenes per tier, but expanded config shows B1=12, B2=40, B3=40, B4=40 (total 132 cases). Sample count is inconsistent. |
| M2 | Default solver `tg43_dose` not listed in solvers section of config (only FBP and DL-Recon are listed). Missing solver definition. |
| M3 | Forward model type `nonlinear_operator` with category_module `medical_ct_radon` -- Radon transform is linear, not nonlinear. Brachytherapy dose calculation (TG-43) IS nonlinear but the physics engine label is wrong. |
| M4 | Webpage leaderboard shows "Learned Primal-Dual + gradient" and "FBPConvNet + gradient" but config only has FBP and DL-Recon. Missing solver registrations for leaderboard algorithms. |
| M5 | Noise model described as Poisson (photon counting) in physics fundamentals, but no SNR or photon count (I0) specified anywhere. |
| M6 | Maturity level M0 (no mismatch - perfect forward model) but benchmark explicitly tests mismatch. Maturity should be at least M2 or M3. |
| M7 | Wavelength/energy range listed as "0 -- 0 nm" in physics fundamentals. Brachytherapy uses Ir-192 (mean ~380 keV) or Cs-137 (662 keV) or I-125 (27-35 keV). |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/brachytherapy_img/compete, /benchmark/brachytherapy_img/contribute |
| L2 | No alt-text on gallery images |
| L3 | SSIM window size and data_range not specified in scoring section |
| L4 | Data source fields (dataset_id, dataset_url, citation, license) are all empty in config |
| L5 | Spec DAG description "Pi --> D" is minimally documented; no mapping of Pi (Projection) and D (Detector) to physical brachytherapy components |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Status |
|------|------|--------|
| Public | `datasets/benchmark/brachytherapy_img/public/` | **NOT FOUND** |
| Dev | `datasets/benchmark/brachytherapy_img/dev/` | **NOT FOUND** |
| Hidden | `datasets/benchmark/brachytherapy_img/hidden/` | **NOT FOUND** |

**No local dataset exists.** The directory `datasets/benchmark/brachytherapy_img` does not exist at all.

### Expected Schema (from config)

| Key | Expected Shape | Dtype | Description |
|-----|---------------|-------|-------------|
| x_true | (64, 64) or (512, 512) | float32 | Ground-truth dose map or attenuation image |
| y_measured | (64, 64) or (512, 512) | float32 | Measured projections with mismatch |
| y_ideal | (64, 64) or (512, 512) | float32 | Ideal projections without mismatch |
| spec_ranges | JSON | -- | Per-tier mismatch parameter ranges |
| true_spec | JSON | -- | Actual mismatch values for each sample |

### Data Source (from config)

| Property | Value |
|----------|-------|
| Dataset ID | (empty) |
| Dataset URL | (empty) |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | (empty) |
| License | (empty) |

The data source relies entirely on synthetic generation using Shepp-Logan phantom. No real or published dataset is configured.

### Dataset Integrity Assessment: **FAIL -- No data exists**

---

## 3. Public Dataset Source Assessment

### Current Source: Shepp-Logan Phantom (Generated) -- **POOR**

The benchmark currently uses procedurally generated Shepp-Logan phantom data. This is a generic CT test phantom, not specific to brachytherapy imaging.

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | POOR | Shepp-Logan is well-known for CT, but irrelevant for brachytherapy dose reconstruction |
| Public: Accepted by professors? | POOR | No brachytherapy researcher would consider a Shepp-Logan phantom as a credible brachytherapy benchmark |
| Public: Domain-appropriate? | FAIL | Brachytherapy involves radioactive source dose deposition, not external beam CT projection |
| Dev: Protected? | N/A | No data exists |
| Hidden: Protected? | N/A | No data exists |

### Recommended Brachytherapy-Specific Sources

1. **Monte Carlo simulated dose distributions** (TG-186 compliant):
   - Use Geant4, MCNP, or EGSnrc to simulate dose from Ir-192/Cs-137/I-125 sources
   - Heterogeneous tissue phantoms with realistic anatomy
   - Ground truth from high-statistics MC runs (>10^9 histories)

2. **AAPM TG-43/TG-186 reference datasets**:
   - Standardized dose calculation parameters for brachytherapy sources
   - Published dose rate constants, radial dose functions, anisotropy functions
   - Widely used in clinical brachytherapy treatment planning

3. **Clinical brachytherapy planning CTs with contours**:
   - Cervical cancer HDR brachytherapy CT images (most common application)
   - Prostate LDR seed implant CTs
   - Available from institutional review-board approved repositories

4. **RapidBrachyDL dataset** (Correa-Alfonso et al., IJROBP 2021):
   - Deep learning dataset for rapid brachytherapy dose calculation
   - Monte Carlo ground truth dose distributions
   - Published and citable

5. **Personalized brachytherapy dose reconstruction dataset** (Akhavanallaf et al., Comput. Biol. Med. 2021):
   - Deep learning-based personalized dose reconstruction
   - MC-derived ground truth, CT-based density maps as input

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard (Webpage)

| # | Algorithm | Type | Overall | Public PSNR/SSIM | Dev PSNR/SSIM | Hidden PSNR/SSIM |
|---|-----------|------|---------|------------------|---------------|------------------|
| 1 | Learned Primal-Dual + gradient | Unrolled optimization | 0.753 | 34.8 / 0.966 | 30.5 / 0.924 | 28.6 / 0.893 |
| 2 | FBPConvNet + gradient | CNN post-processing | 0.726 | 33.5 / 0.957 | 29.2 / 0.904 | 26.35 / 0.842 |
| 3 | PnP-ADMM + gradient | Plug-and-play | 0.660 | 30.0 / 0.917 | 25.24 / 0.810 | 22.81 / 0.724 |
| 4 | FBP + gradient | Analytical | 0.639 | 25.87 / 0.828 | 24.67 / 0.792 | 23.89 / 0.765 |

### Registered Solvers (Config YAML)

| Solver | Module | Function | On Leaderboard? |
|--------|--------|----------|-----------------|
| FBP | pwm_core.recon.fbp | run_fbp | Yes (rank 4) |
| DL-Recon | pwm_core.recon.dl_recon | dl_reconstruct | Unclear mapping |

**Gap:** Leaderboard has 4 algorithms but config only registers 2 solvers. Learned Primal-Dual, FBPConvNet, and PnP-ADMM are on the leaderboard but NOT in the solver registry.

### Missing Famous/Recent Algorithms

| Priority | Algorithm | Year | Why Important |
|----------|-----------|------|---------------|
| **CRITICAL** | RapidBrachyDL | 2021 | Purpose-built DL for brachytherapy dose calculation (Correa-Alfonso et al., IJROBP ~150 citations) |
| **CRITICAL** | TG-43 analytical | 1995/2004 | Gold standard clinical dose calculation protocol (Rivard et al., Med. Phys., ~3,000 citations) |
| **CRITICAL** | Monte Carlo (Geant4/EGSnrc) | -- | Ground truth method, essential reference baseline |
| **HIGH** | Personalized DL dose reconstruction | 2021 | Akhavanallaf et al., MC-to-DL dose prediction (Comput. Biol. Med.) |
| **HIGH** | 3D U-Net dose engine | 2023 | Automatic brachytherapy planning with DCNN dose engine (Li et al., Med. Phys.) |
| **HIGH** | pix2pix GAN for needle/seed segmentation | 2025 | Two-phase DL for HDR catheter localization (CTRO) |
| **MEDIUM** | SART/SIRT | Classical | Standard iterative algebraic reconstruction |
| **MEDIUM** | DiffusionMBIR | 2023 | Diffusion + model-based inverse reconstruction (Chung et al.) |
| **MEDIUM** | Fast MC (DL-accelerated) | 2023 | DL-accelerated MC dose calculation for LDR brachytherapy |
| **LOW** | TV-ADMM | Classical | Total variation regularized iterative |
| **LOW** | Direct Inversion + gradient | -- | Listed on original automated check but absent from current leaderboard |

### Algorithm Observations

- All 4 leaderboard algorithms use "+ gradient" suffix, indicating gradient-based spec correction. This is good for the PWM mismatch challenge framework.
- Performance degradation from Public to Hidden is modest (34.8 -> 28.6 dB for rank 1), suggesting mismatch ranges are not severe enough. This aligns with finding H4 (equal span across tiers).
- FBP baseline (rank 4) has relatively competitive hidden-tier performance (23.89 dB) vs learned methods (28.6 dB), suggesting the inverse problem may not be very challenging.
- The leaderboard methods (Learned Primal-Dual, FBPConvNet, PnP-ADMM) are generic CT reconstruction algorithms, not brachytherapy-specific methods.

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Clinical brachytherapy (TG-43) | 0 | TG-43, TG-186 | 2 methods |
| Monte Carlo reference | 0 | Geant4/EGSnrc MC | 1 method |
| Brachytherapy-specific DL | 0 | RapidBrachyDL, personalized DL dose | 2 methods |
| Unrolled optimization | Learned Primal-Dual | -- | 0 methods |
| CNN post-processing | FBPConvNet | 3D U-Net dose engine | 1 method |
| Plug-and-play | PnP-ADMM | PnP-DRUNet | 1 method |
| Analytical baseline | FBP | SART/SIRT | 1-2 methods |
| Generative/diffusion | 0 | DiffusionMBIR | 1 method |
| Seed/needle localization | 0 | pix2pix GAN, HRNet | 2 methods |

**Total gap: 10+ algorithms missing. Zero brachytherapy-specific algorithms present.**

---

## 5. Improvement Suggestions

### 5.1 Dataset (CRITICAL)

1. **Create brachytherapy-specific dataset** -- replace Shepp-Logan phantom entirely
   - Generate Monte Carlo dose distributions using Geant4/EGSnrc for Ir-192 HDR source
   - Include heterogeneous tissue phantoms (bone, soft tissue, air, applicator)
   - Provide both dose-to-medium (D_m,m) and dose-to-water-in-medium (D_w,m) as ground truth

2. **Public tier: Use published brachytherapy data**
   - RapidBrachyDL dataset (if publicly available)
   - Or generate from TG-43 reference data with clinical geometries
   - Minimum 15-20 samples with diverse source configurations

3. **Dev/Hidden tier: Realistic clinical scenarios**
   - Cervical cancer (tandem-and-ovoid applicator) for dev
   - Prostate seed implant with metallic seeds for hidden
   - Include tissue heterogeneity, inter-seed attenuation, organ motion

4. **Fix image resolution ambiguity**
   - Decide between 64x64 (too coarse) and 512x512 (realistic)
   - Clinical brachytherapy CT is typically 512x512 at 0.5-1.0 mm pixel size
   - 3D dose grids are often 256x256x256 at 1 mm resolution

5. **Populate all empty data source fields** (dataset_id, citation, license, URL)

### 5.2 Algorithms

6. **Add TG-43 analytical baseline immediately** -- this is THE standard in brachytherapy
   - Every brachytherapy physicist knows TG-43
   - It should be the primary baseline, not FBP

7. **Add Monte Carlo reference** as ground truth / upper bound
   - Geant4 or EGSnrc with sufficient history count
   - This defines the achievable accuracy ceiling

8. **Add RapidBrachyDL** -- purpose-built brachytherapy DL
   - Published, citable, brachytherapy-specific
   - Demonstrates 300x speedup over MC with <1% error

9. **Register all leaderboard algorithms in solver_registry**
   - Learned Primal-Dual, FBPConvNet, PnP-ADMM are missing from config
   - Add module paths and function references

10. **Add SART/SIRT iterative baseline**
    - Standard algebraic reconstruction for projection-based problems
    - Widely expected by the medical imaging community

### 5.3 Forward Model and Physics

11. **Resolve nonlinear_operator vs medical_ct_radon contradiction**
    - If the forward model is Radon transform: it is linear, use `linear_operator`
    - If the forward model is TG-43 dose calculation: it is nonlinear, use a brachytherapy-specific category module
    - Current labeling is internally inconsistent

12. **Define the brachytherapy forward model explicitly**
    - TG-43: D(r, theta) = S_K * Lambda * [G_L(r,theta)/G_L(r_0,theta_0)] * g_L(r) * F(r,theta)
    - Where S_K = air-kerma strength, Lambda = dose rate constant, G_L = geometry function, g_L = radial dose function, F = anisotropy function
    - This should replace the generic "y = A(x) + noise" equation

13. **Redesign mismatch parameters for brachytherapy**
    - Source position error (already present, KEEP)
    - Replace "attenuation coefficient" with tissue heterogeneity correction factor
    - Replace "detector gain drift" with inter-seed attenuation factor
    - Replace "scatter fraction" with applicator shielding correction
    - Ensure difficulty increases monotonically across tiers

14. **Specify source isotope and energy**
    - Ir-192: mean ~380 keV (most common HDR source)
    - Cs-137: 662 keV (some older units)
    - I-125: 27-35 keV (LDR prostate seeds)
    - Currently wavelength/energy is "0 -- 0 nm" (empty)

### 5.4 Infrastructure

15. **Fix maturity level** from M0 to M2+ (benchmark explicitly tests mismatch)
16. **Sync all sample counts** between webpage (3 per tier) and expanded config (12/40/40/40)
17. **Add per-sample metric breakdown** in leaderboard
18. **Fix default solver** -- `tg43_dose` is referenced but not defined in the solvers section

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Create brachytherapy-specific dataset (MC-generated dose distributions) | Dataset team | TODO |
| CRITICAL | Fix image size contradiction (64x64 vs 512x512) | Config team | TODO |
| CRITICAL | Redesign mismatch ranges to increase monotonically across tiers | Config team | TODO |
| CRITICAL | Add TG-43 analytical baseline as primary solver | Algorithm team | TODO |
| CRITICAL | Define PSNR_norm formula explicitly | Main server | TODO |
| CRITICAL | Resolve nonlinear_operator vs medical_ct_radon contradiction | Physics team | TODO |
| HIGH | Add Monte Carlo reference (Geant4/EGSnrc) as accuracy ceiling | Algorithm team | TODO |
| HIGH | Add RapidBrachyDL to leaderboard | Algorithm team | TODO |
| HIGH | Register all leaderboard algorithms (LPD, FBPConvNet, PnP-ADMM) in solver_registry.yaml | Config team | TODO |
| HIGH | Sync webpage mismatch ranges with config YAML | Main server | TODO |
| HIGH | Define brachytherapy-specific forward model (TG-43 equation) on webpage | Physics team | TODO |
| HIGH | Specify source isotope and energy (Ir-192 / I-125 / Cs-137) | Physics team | TODO |
| MEDIUM | Add SART/SIRT iterative baseline | Algorithm team | TODO |
| MEDIUM | Add DiffusionMBIR and 3D U-Net dose engine | Algorithm team | TODO |
| MEDIUM | Fix maturity from M0 to M2+ | Config team | TODO |
| MEDIUM | Populate data source fields (dataset_id, citation, license) | Config team | TODO |
| MEDIUM | Fix default solver tg43_dose -- define or remove from config | Config team | TODO |
| MEDIUM | Sync sample counts (webpage vs expanded config) | Main server | TODO |
| LOW | Add per-sample metric breakdown to leaderboard | Main server | TODO |
| LOW | Fix placeholder links (/compete, /contribute) | Main server | TODO |
| LOW | Specify SSIM window size and data_range | Main server | TODO |
| LOW | Map DAG nodes (Pi, D) to physical brachytherapy components | Physics team | TODO |

---

## Appendix: Key References

- Rivard, M.J., et al. "Update of AAPM Task Group No. 43 Report: A revised AAPM protocol for brachytherapy dose calculations." Med. Phys. 31.3 (2004): 633-674.
- Nath, R., et al. "Dosimetry of interstitial brachytherapy sources: recommendations of the AAPM (TG-43)." Med. Phys. 22.2 (1995): 209-234.
- Beaulieu, L., et al. "Report of the Task Group 186 on model-based dose calculation methods." Med. Phys. 39.10 (2012): 6208-6236.
- Correa-Alfonso, C.M., et al. "RapidBrachyDL: Rapid Radiation Dose Calculations in Brachytherapy Via Deep Learning." IJROBP 108.3 (2020): S103.
- Akhavanallaf, A., et al. "Personalized brachytherapy dose reconstruction using deep learning." Comput. Biol. Med. 136 (2021): 104755.
- Li, Z., et al. "Automatic planning for head and neck seed implant brachytherapy based on deep DCNN dose engine." Med. Phys. 50.10 (2023): 6290-6302.
- Adler, J. and Oktem, O. "Learned Primal-Dual Reconstruction." IEEE TMI 37.6 (2018): 1322-1332.
- Jin, K.H., et al. "Deep Convolutional Neural Network for Inverse Problems in Imaging (FBPConvNet)." IEEE TIP 26.9 (2017): 4509-4522.
- McCollough, C.H., et al. "Low-Dose CT for the Detection and Classification of Metastatic Liver Lesions: AAPM CT Grand Challenge." Med. Phys. 44.10 (2017): e339-e352.
- Chung, H., et al. "Diffusion posterior sampling for general noisy inverse problems." ICLR (2023).

---

*Comprehensive 6-point review on 2026-03-03. Brachytherapy Imaging benchmark has fundamental issues: no local dataset, domain-inappropriate phantoms (Shepp-Logan/CT for brachytherapy), unresolved config contradictions, and zero brachytherapy-specific algorithms. Requires significant rework of dataset, forward model, and algorithm suite.*