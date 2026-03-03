# Comprehensive 6-Point Check -- pet

**URL:** https://pwm.platformai.org/benchmark/pet
**Check Date:** 2026-03-03
**Status:** PASS (correct carrier routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Positron Emission Tomography (PET)

**Physical principle:** PET detects pairs of annihilation photons (511 keV gamma rays) emitted when a positron from a radiotracer annihilates with an electron in tissue. The two photons travel in nearly opposite directions and are detected in coincidence by a ring of scintillation detectors. Each coincidence event defines a line of response (LOR). The radiotracer distribution (activity concentration) is reconstructed from the set of LOR measurements.

**Forward model:**
```
y_i = sum_j  a_ij * lambda_j + scatter_i + random_i
```
where y_i = expected counts in LOR i, a_ij = system matrix element (probability of detecting emission from voxel j in LOR i), lambda_j = activity concentration in voxel j.

**Inverse problem:** Reconstruct the 3D radiotracer distribution from noisy coincidence count data. The data is Poisson-distributed, the system matrix is very large (millions of LORs x millions of voxels), and corrections for attenuation, scatter, randoms, and normalization are required.

**Current physics engine:** Emission tomography forward model. The carrier routing `(medical, Gamma) -> particle_imaging` correctly sends PET to the nuclear medicine algorithm pool.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** Pi(LOR) -> Sigma_t -> D(g, eta_3)

**Mismatch sources in PET:**
- Attenuation correction errors (CT-based vs MR-based)
- Scatter estimation inaccuracies
- Random coincidence subtraction errors
- Detector normalization drift
- Dead time and pile-up at high count rates
- Patient motion during scan
- Partial volume effects (limited spatial resolution ~4 mm)
- Time-of-flight (TOF) timing resolution

**Dataset format (GCS):**
- `x_true` -- ground truth activity distribution
- `y` -- sinogram/LOR measurements
- `H_ideal` -- system matrix or projection parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (particle_imaging pool via carrier routing: medical + Gamma -> particle_imaging):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| OSEM | Classical | Hudson & Larkin, IEEE TMI 1994 | CORRECT -- the standard clinical PET reconstruction algorithm |
| MAPEM-RDP | PnP | Nuyts et al., Phys. Med. Biol. 2002 | CORRECT -- MAP-EM with relative difference penalty, used clinically (e.g., GE Q.Clear) |
| DeepPET | Deep Learning | Haggstrom et al., Med. Image Anal. 2019 | CORRECT -- end-to-end deep learning PET reconstruction |
| TransEM | Transformer | Xie et al., 2023 | CORRECT -- transformer-based emission tomography reconstruction |

All 4 algorithms are domain-appropriate for PET. OSEM is the universal clinical standard, MAPEM-RDP is the basis of commercial implementations (GE Q.Clear/BSREM), and DeepPET/TransEM represent modern deep learning approaches.

## 4. Literature & State of the Art (2024--2025)

1. **OSEM** (Hudson & Larkin, 1994): Ordered-Subsets Expectation Maximization -- the workhorse of clinical PET reconstruction for 30 years.
2. **BSREM/Q.Clear** (Ahn et al., 2015): Block sequential regularized EM -- commercial implementation of MAPEM-RDP by GE Healthcare. Already represented by MAPEM-RDP in the pool.
3. **DIP-Recon for PET** (Gong et al., Eur. J. Nucl. Med. 2024): Deep image prior for low-count PET reconstruction.
4. **Total-body PET** (Cherry et al., 2024): uEXPLORER and PennPET systems with 100x sensitivity gain -- drives need for fast reconstruction algorithms.
5. **Score-based PET** (2024): Diffusion model priors for PET image reconstruction.
6. **AI-assisted PET/CT** (2024--2025): Joint PET-CT reconstruction and attenuation correction using deep learning.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `pet_challenge_public.h5` -- present on GCS
- `pet_challenge_dev.h5` -- present on GCS (x_true stripped)
- `pet_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** 24/24 load OK from GCS.

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** `(medical, Gamma)` -> `particle_imaging` pool. This was previously fixed from the generic medical/CT pool (FBP, FBPConvNet) to the correct nuclear medicine pool. The current algorithms (OSEM, MAPEM-RDP, DeepPET, TransEM) are all domain-appropriate.

**Previously fixed:** PET was incorrectly getting CT algorithms via the generic medical category. The carrier-based routing `(medical, Gamma) -> particle_imaging` resolved this.

**No further changes required.** The algorithm assignment is correct.

---
*Comprehensive 6-point check by deep-check pipeline v3*
