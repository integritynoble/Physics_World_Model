# Comprehensive 6-Point Check -- spect

**URL:** https://pwm.platformai.org/benchmark/spect
**Check Date:** 2026-03-03
**Status:** PASS (correct carrier routing, no code changes needed)

---

## 1. Physics & Forward Model

**Modality:** Single-Photon Emission Computed Tomography (SPECT)

**Physical principle:** SPECT detects individual gamma-ray photons emitted by a radiotracer injected into the patient. A rotating gamma camera with a parallel-hole or pinhole collimator acquires projection images at multiple angles. The collimator defines the direction of detected photons (unlike PET which uses coincidence detection). Common radiotracers include Tc-99m (140 keV), I-123 (159 keV), and Tl-201 (68-80 keV).

**Forward model:**
```
y_i = sum_j  c_ij * a_ij * lambda_j + scatter_i + noise_i
```
where y_i = counts in projection bin i, c_ij = collimator-detector response, a_ij = attenuation factor, lambda_j = activity concentration in voxel j, and scatter_i = scattered photon contribution.

**Inverse problem:** Reconstruct the 3D radiotracer distribution from noisy, attenuated, scatter-contaminated projection data. The data is Poisson-distributed with much lower count rates than PET (single-photon detection vs coincidence).

**Current physics engine:** Emission tomography forward model. The carrier routing `(medical, Gamma) -> particle_imaging` correctly sends SPECT to the nuclear medicine algorithm pool (shared with PET).

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** Pi(collimated) -> Sigma_t -> D(g, eta_3)

**Mismatch sources in SPECT:**
- Collimator-dependent resolution (depth-dependent blurring)
- Attenuation correction errors (CT-based attenuation map)
- Scatter estimation and correction
- Detector uniformity and energy window settings
- Patient motion during rotation
- Partial volume effects
- Collimator septal penetration (high-energy isotopes)
- Center-of-rotation misalignment

**Dataset format (GCS):**
- `x_true` -- ground truth activity distribution
- `y` -- projection/sinogram measurements
- `H_ideal` -- system matrix / collimator parameters

**Tier structure:** Public (with x_true), Dev (no x_true), Hidden (blocked).

## 3. Reconstruction Methods & Leaderboard

**Algorithms (particle_imaging pool via carrier routing: medical + Gamma -> particle_imaging):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| OSEM | Classical | Hudson & Larkin, IEEE TMI 1994 | CORRECT -- the standard clinical SPECT reconstruction algorithm |
| MAPEM-RDP | PnP | Nuyts et al., Phys. Med. Biol. 2002 | CORRECT -- MAP-EM with regularization, widely used in SPECT |
| DeepPET | Deep Learning | Haggstrom et al., Med. Image Anal. 2019 | CORRECT -- applicable to emission tomography generally |
| TransEM | Transformer | Xie et al., 2023 | CORRECT -- transformer-based emission tomography reconstruction |

All 4 algorithms are domain-appropriate for SPECT. OSEM is the universal clinical SPECT standard (same as PET), and the other methods apply to emission tomography in general. Sharing the pool with PET is justified since both modalities solve the same class of emission tomography inverse problem.

## 4. Literature & State of the Art (2024--2025)

1. **OSEM + resolution recovery** (standard): OSEM with 3D depth-dependent detector response modeling -- the clinical standard for SPECT (Siemens Flash3D, GE Evolution).
2. **CZT-SPECT** (2024): Cadmium Zinc Telluride detectors (GE NM530c, Spectrum Dynamics) with improved energy resolution and sensitivity.
3. **DL-based SPECT** (2024): CNN and transformer approaches for low-count SPECT reconstruction, denoising, and attenuation correction.
4. **SPECT/CT joint reconstruction** (2024): Simultaneous activity and attenuation estimation.
5. **DaTSCAN quantification** (2024--2025): Deep learning for dopamine transporter SPECT quantification (Parkinson's disease diagnosis).
6. **Total-body SPECT** (2025): Multi-pinhole whole-body SPECT systems with novel reconstruction algorithms.

## 5. Local Dataset & GCS Status

**GCS datasets verified:**
- `spect_challenge_public.h5` -- present on GCS
- `spect_challenge_dev.h5` -- present on GCS (x_true stripped)
- `spect_challenge_hidden.h5` -- present on GCS (blocked from download)

**Gallery images:** 24/24 load OK from GCS (4 scenes x 6 images).

**Learning materials:** Complete 5-module set present (README, physics fundamentals, forward model, reconstruction algorithms, PWM benchmark, hands-on tutorial).

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS -- no code changes needed.

**Routing:** `(medical, Gamma)` -> `particle_imaging` pool. This was previously fixed from the generic medical/CT pool (FBP, FBPConvNet) to the correct nuclear medicine pool. Sharing with PET is appropriate since both are emission tomography modalities.

**Previously fixed:** SPECT was incorrectly getting CT algorithms via the generic medical category. The carrier-based routing `(medical, Gamma) -> particle_imaging` resolved this.

**No further changes required.** The algorithm assignment is correct.

---
*Comprehensive 6-point check by deep-check pipeline v3*
