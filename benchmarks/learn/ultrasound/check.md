# Comprehensive Benchmark QA Check — Ultrasound

**URL:** https://pwm.platformai.org/benchmark/ultrasound
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
| HIGH     | 5     |
| MEDIUM   | 7     |
| LOW      | 4     |

### HIGH Severity

**H1. Sample count mismatch**
- Webpage: "3 scenes" per tier
- Local H5: 11 public / 20 dev / 20 hidden (51 total)
**Fix:** Update webpage to show actual sample counts.

**H2. Mismatch parameter ranges differ between webpage and local data**
- Webpage element_sensitivity: -5.0 to 10.0% -- local data uses 0.0 to 3.0 dB (public), 0.0 to 8.0 dB (hidden)
- Webpage phase_aberration: -0.3 to 0.6 rad -- local data uses 0.0 to 0.2 rad (public), 0.0 to 0.6 rad (hidden)
- Different units and ranges indicate webpage is out of sync
**Fix:** Sync webpage with HDF5 spec_ranges.

**H3. PSNR_norm undefined in scoring formula**
**Fix:** Define normalization method explicitly.

**H4. Public tier source is SYNTHETIC, not PICMUS**
- README states "PLACEHOLDER: Synthetic PICMUS-style phantoms (PICMUS data not found)"
- metadata.source = "synthetic_picmus_style" for public tier
- Webpage claims "PICMUS Challenge" as source
**Fix:** Either download and integrate real PICMUS data, or update webpage to say "synthetic PICMUS-style".

**H5. Rank inversion on Hidden tier unexplained**
- US-CycleGAN: 1st (public) -> 2nd (dev) -> 3rd (hidden)
- MU-Net: 2nd (public) -> 1st (dev) -> 1st (hidden)
- No discussion of why rankings shift
**Fix:** Add note explaining robustness differences.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Scatterer count varies greatly: 6K (public) vs 64K (dev) vs 120K (hidden) -- makes comparison difficult |
| M2 | Webpage says "Verasonics Vantage 256 / GE LOGIQ E10" but data is purely synthetic simulation |
| M3 | Missing references for leaderboard methods: MU-Net, US-CycleGAN not cited |
| M4 | Webpage forward model DAG is oversimplified vs README's explicit delay-and-sum equation |
| M5 | Public tier has shared true_spec across all samples (same mismatch), inconsistent with "independently randomized" claim |
| M6 | PICMUS probe parameters (128 elements, 0.30mm pitch, 5.208 MHz) not shown on webpage |
| M7 | Hidden tier uses coupled/correlated mismatch but webpage doesn't explain this |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/ultrasound/compete, /benchmark/ultrasound/contribute |
| L2 | No gallery images showing beamformed comparisons |
| L3 | SSIM parameters not specified for ultrasound (log-compressed vs linear domain?) |
| L4 | Element_positions array has 128 elements matching spec, but webpage doesn't document this |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples |
|------|------|------|---------|
| Public | ultrasound_challenge_public.h5 | 120 MB | 11 |
| Dev | ultrasound_challenge_dev.h5 | 245 MB | 20 |
| Hidden | ultrasound_challenge_hidden.h5 | 256 MB | 20 |

### HDF5 Schema Verification

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `y` | (2048, 128, 11) | float32 | RF channel data (time x elements x angles) |
| `x_true_image` | (256, 256) | float32 | Reference beamformed image |
| `x_true_scatterers_pos` | (N, 2) | float64 | Scatterer positions [x_m, z_m] |
| `x_true_scatterers_amp` | (N,) | float64 | Scatterer amplitudes |
| `angles_rad` | (11,) | float32 | Steering angles in radians |
| `element_positions` | (128,) | float32 | Transducer element x-positions |

### Value Range Checks

| Check | Public | Dev | Hidden | Status |
|-------|--------|-----|--------|--------|
| x_true_image in [0,1] | Verified | Verified | Verified | PASS |
| RF data non-trivial | max > 0 | max > 0 | max > 0 | PASS |
| 11 steering angles | PASS | PASS | PASS | PASS |
| 128 elements | PASS | PASS | PASS | PASS |
| Scatterer count | ~6K | ~64K | ~120K | PASS (increases with tier) |

### Spec Range Nesting Verification

| Parameter | Public | Dev | Hidden | Nesting |
|-----------|--------|-----|--------|---------|
| sos (m/s) | [1510, 1570] | [1500, 1580] | [1480, 1600] | PASS |
| attenuation (dB/cm/MHz) | [0.3, 0.6] | [0.3, 0.7] | [0.3, 0.9] | PASS |
| element_sensitivity (dB) | [0, 3] | [0, 4] | [0, 8] | PASS |
| phase_aberration (rad) | [0, 0.2] | [0, 0.3] | [0, 0.6] | PASS |

**All spec ranges properly nest: Public < Dev < Hidden.**

### Scene Diversity

| Tier | Sample Scenes |
|------|---------------|
| Public | resolution_grid, cyst_phantom, tissue_speckle (PICMUS-style) |
| Dev | liver_portal, kidney_detailed, thyroid, breast, carotid, abdominal, MSK, pelvic, vessels, tissue_char, cyst_array, diffuse_disease |
| Hidden | aberrating_fat_slab, reverberation_trap, calcification_storm, heterogeneous_medium, extreme_depth, pathology_dense, bone_shadow, parallel_interfaces, sub_resolution_grid, worst_case_combined |

### Dataset Integrity Assessment: **PASS** (structurally sound, but public uses synthetic fallback)

---

## 3. Public Dataset Source Assessment

### Current Source

**Public tier:** `source = "synthetic_picmus_style"` -- NOT real PICMUS data
- Synthetic scatterer-based phantoms mimicking PICMUS geometry
- README explicitly states "PLACEHOLDER: Synthetic PICMUS-style phantoms"
- Real PICMUS data was not found/downloaded

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Public: Well-known dataset?** | NEEDS IMPROVEMENT | PICMUS is the standard, but data is synthetic fallback |
| **Public: Accepted by professors?** | NEEDS IMPROVEMENT | Synthetic approximation of PICMUS, not the real challenge data |
| **Dev: Protected?** | EXCELLENT | 12 procedural clinical scene types with 50K-150K scatterers |
| **Hidden: Protected?** | EXCELLENT | 10 adversarial scene types with correlated mismatch |

### Recommendations for Public Tier

1. **Download real PICMUS challenge data** (HIGHEST PRIORITY):
   - Source: https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/
   - Also via USTB: https://www.ustb.no/datasets/
   - Contains: resolution phantoms, cyst phantoms, in-vivo carotid
   - Standard benchmark accepted by all ultrasound imaging researchers

2. **Alternative established datasets:**
   - **CUBDL** (Hyun et al., IEEE TUFFC 2020): Deep learning ultrasound beamforming challenge
   - **PICMUS Category I and II**: Simulation + experimental data
   - **CIRS phantom data**: Commercial phantom scans, widely available
   - **Plane Wave Imaging Design Study (PWIDS)**: Standardized PW dataset

3. **For clinical credibility, add in-vivo data:**
   - PICMUS includes in-vivo carotid cross-section
   - CLUST (Challenge on Liver US Tracking): In-vivo liver sequences
   - SYNTHUS (Synthetic Ultrasound): Bridge between simulation and clinical

---

## 4. Algorithm Coverage Assessment

### Currently Tested (Webpage Leaderboard)

| # | Algorithm | Type | Notes |
|---|-----------|------|-------|
| 1 | US-CycleGAN + gradient | GAN-based | Good on public, degrades on hidden |
| 2 | MU-Net + gradient | U-Net variant | Most robust across tiers |
| 3 | PnP-ADMM + gradient | Plug-and-play | Consistent but not top |
| 4 | MV-Beamformer + gradient | Minimum variance | Classical adaptive BF |

### PWM Solver Registry

Only generic solvers (Adjoint, PnP-ADMM) -- no ultrasound-specific solvers registered.

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Why Important |
|----------|-----------|------|---------------|
| **CRITICAL** | DAS (Delay-and-Sum) | Classical | THE baseline beamformer, every US paper compares against it |
| **CRITICAL** | CPWC (Coherent Plane Wave Compounding) | 2009 | Standard multi-angle beamformer, Montaldo et al. (~2,000 citations) |
| **CRITICAL** | DMAS (Delay-Multiply-and-Sum) | 2015 | Superior sidelobe reduction, Matrone et al. (~500 citations) |
| **HIGH** | Capon / MVDR beamformer | 1969/2009 | Adaptive beamforming gold standard for US (MV already on leaderboard as MV-Beamformer) |
| **HIGH** | Eigenspace-based MV (ESBMV) | 2010 | Improved MV with eigenvalue decomposition |
| **HIGH** | ABLE (Adaptive BF by DL) | 2020 | DL adaptive beamforming, Luijten et al. IEEE TUFFC |
| **HIGH** | Deep Coherence Learning (DCL) | 2024 | Unsupervised, single-PW imaging, state-of-the-art contrast |
| **MEDIUM** | iMAP (iterative MAP) | 2017 | Model-based iterative reconstruction for US |
| **MEDIUM** | REFoCUS | 2018 | Retrospective transmit focusing, Ali & Bhatt |
| **MEDIUM** | CapsBeam | 2025 | Capsule network beamformer, 32% contrast improvement |
| **LOW** | Spatial coherence beamformer (SCB) | 2014 | Coherence-based weighting |
| **LOW** | Phase coherence imaging (PCI) | 2012 | Noise rejection via phase coherence |

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Classical DAS/CPWC | MV-BF | DAS, CPWC, DMAS | 3 |
| Adaptive beamforming | MV-BF | ESBMV, SCB, PCI | 3 |
| DL beamforming | MU-Net | ABLE, DCL, CapsBeam | 3 |
| GAN-based | US-CycleGAN | -- | 0 |
| Model-based iterative | -- | iMAP, REFoCUS | 2 |
| Plug-and-play | PnP-ADMM | PnP-BM3D | 1 |

**Total gap: 12 algorithms missing across 5 categories**

---

## 5. Improvement Suggestions

### 5.1 Dataset Improvements

1. **Download real PICMUS data for public tier (CRITICAL)**
   - Current synthetic fallback undermines benchmark credibility
   - PICMUS is the IEEE IUS 2016 standard challenge dataset
   - Both simulation and experimental data available

2. **Add in-vivo data to public tier**
   - PICMUS includes in-vivo carotid cross-section
   - This is essential for clinical relevance

3. **Add CUBDL challenge data**
   - Hyun et al. IEEE TUFFC 2020 deep learning beamforming challenge
   - Provides standardized comparison framework

4. **Increase angle diversity**
   - Currently fixed at 11 angles for all tiers
   - Real applications: 1-angle (ultrafast) to 75-angle (high quality)
   - Suggested: Public 11, Dev 3-11, Hidden 1-7 (fewer = harder)

5. **Add speed-of-sound map (not just scalar)**
   - Current mismatch uses global SoS error
   - Real tissue has spatially varying SoS (fat: 1450, liver: 1570 m/s)
   - Hidden tier should include SoS heterogeneity maps

6. **Add motion simulation for hidden tier**
   - Real-time ultrasound encounters tissue motion, breathing, cardiac
   - Motion between PW transmissions causes phase errors

### 5.2 Algorithm Testing Improvements

7. **Add DAS/CPWC baseline immediately**
   - Every ultrasound paper uses DAS as the reference
   - CPWC is the multi-angle extension, trivial to implement
   - No training required

8. **Add DMAS (Delay-Multiply-and-Sum)**
   - Improved sidelobe suppression, widely adopted
   - Standard alternative to DAS in plane-wave imaging

9. **Add ABLE or Deep Coherence Learning**
   - State-of-the-art DL beamforming approaches
   - Represent the latest paradigm in adaptive beamforming

10. **Run all algorithms on all 3 tiers**
    - Target: 8-10 algorithms x 3 tiers

### 5.3 Infrastructure Improvements

11. **Sync webpage with local data**
    - Fix sample counts
    - Fix mismatch parameter units and ranges
    - Note that public is synthetic fallback, not real PICMUS

12. **Add ultrasound-specific metrics**
    - Contrast-to-Noise Ratio (CNR) for cyst phantoms
    - Lateral/axial resolution (-6dB width) for point targets
    - Speckle SNR for tissue regions
    - These are standard in ultrasound quality assessment

13. **Specify PSNR/SSIM computation domain**
    - Ultrasound images are typically log-compressed for display
    - PSNR/SSIM in log domain vs linear domain gives very different results
    - Must specify: compute on B-mode (log-compressed) or envelope (linear)?

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Download real PICMUS data for public tier | Dataset team | TODO |
| CRITICAL | Add DAS/CPWC baseline to leaderboard | Algorithm team | TODO |
| CRITICAL | Sync webpage sample counts (11/20/20) | Main server | TODO |
| CRITICAL | Sync webpage mismatch parameters with HDF5 | Main server | TODO |
| HIGH | Add DMAS beamformer | Algorithm team | TODO |
| HIGH | Add ABLE or DCL deep beamformer | Algorithm team | TODO |
| HIGH | Define PSNR_norm and computation domain (log vs linear) | Main server | TODO |
| HIGH | Add CUBDL challenge data or PICMUS in-vivo | Dataset team | TODO |
| MEDIUM | Add ultrasound-specific metrics (CNR, resolution) | Metrics team | TODO |
| MEDIUM | Add missing references (MU-Net, US-CycleGAN, ABLE) | Main server | TODO |
| MEDIUM | Add spatially-varying SoS map for hidden tier | Dataset team | TODO |
| MEDIUM | Add REFoCUS and iMAP model-based methods | Algorithm team | TODO |
| LOW | Add variable angle count (1-75) across tiers | Dataset team | TODO |
| LOW | Add motion simulation for hidden tier | Dataset team | TODO |
| LOW | Fix placeholder links (/compete, /contribute) | Main server | TODO |

---

## Appendix: Key References

- Montaldo, G., et al. "Coherent plane-wave compounding for very high frame rate ultrasonography." IEEE TUFFC 56.3 (2009): 489-506.
- Matrone, G., et al. "The Delay Multiply and Sum beamforming algorithm." IEEE TMI 34.4 (2015): 940-949.
- Liebgott, H., et al. "Plane-wave imaging challenge in medical ultrasound." IEEE IUS (2016).
- Jensen, J.A. "Field: A Program for Simulating Ultrasound Systems." Med. Biol. Eng. Comput. 34 (1996).
- Luijten, B., et al. "Adaptive ultrasound beamforming using deep learning." IEEE TMI 39.12 (2020): 3967-3978.
- Hyun, D., et al. "Deep learning for ultrasound image formation (CUBDL)." IEEE TUFFC 68.12 (2021): 3442-3452.
- "Deep coherence learning for single plane wave imaging." Ultrasonics (2024).
- "CapsBeam: Capsule Network Beamformer." arXiv (2025).

---

*Comprehensive 6-point review on 2026-03-03. Covers: page errors, local dataset verification, source quality, algorithm coverage, improvement suggestions, and action items.*
