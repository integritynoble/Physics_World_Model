# Comprehensive 6-Point Check — Single-Photon Emission Computed Tomography (SPECT)

**URL:** https://pwm.platformai.org/benchmark/spect
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Single-Photon Emission Computed Tomography (SPECT)

**Physical principle:** SPECT images the 3-D distribution of a gamma-emitting radiopharmaceutical injected into a patient. Gamma photons (typically 140 keV for Tc-99m) are detected by a rotating gamma camera equipped with a parallel-hole collimator. Photon attenuation through tissue and distance-dependent collimator blur (geometric/septal penetration) are the dominant degradation sources.

**Forward model:**
```
y_θ = P_θ * A_θ * x + n

where:
  x           — radionuclide activity distribution (voxel grid)
  A_θ         — diagonal attenuation matrix for projection angle θ
                (A_θ)_ii = exp(-∫ μ(r) dr along ray i)
  P_θ         — projector incorporating collimator-detector response (CDR)
                modelled as a depth-dependent Gaussian blur
  y_θ         — measured counts in detector bins at angle θ (Poisson)
  n           ~ Poisson(P_θ * A_θ * x)
```

**Inverse problem:** Recover the 3-D activity map x from noisy, attenuation-corrupted projections y acquired at discrete angles around the patient.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(radiopharmaceutical/collimator) → F(attenuation/scatter) → D(gamma camera)

**Key mismatch parameters:**
- `mu_map_scale`: Linear attenuation coefficient scaling; nominal 1.0, perturbed 0.85–1.15
- `cdr_fwhm_mm`: Collimator-detector response FWHM at reference depth; nominal 9.5 mm, perturbed 7.5–12 mm
- `scatter_fraction`: Scatter-to-primary ratio in energy window; nominal 0.12, perturbed 0.05–0.25
- `rotation_radius_mm`: Camera orbit radius; nominal 200 mm, perturbed 180–230 mm

**Dataset format:**
- `x_true: (H, W)` — 2-D slice of 3-D activity map (Bq/voxel, normalised)
- `y: (N_angles, N_bins)` — sinogram of raw photon counts per detector bin

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| OSEM (Ordered-Subsets EM) | Classical iterative | Hudson & Larkin, IEEE TMI 13(4):601–609, 1994 | Clinical gold standard for SPECT; handles Poisson statistics and attenuation correction natively |
| FBP + Chang attenuation | Classical analytical | Chang, IEEE TNS 25(1):638–643, 1978 | Fast, analytically invertible; Chang correction is modality-specific and well-validated |
| TV-regularised MLEM | Variational | Panin et al., IEEE TMI 18(2):130–138, 1999 | Reduces streak artifacts in low-count studies while preserving quantitative accuracy |
| Deep-learning post-filter (U-Net) | Deep Learning | Häggström et al., J Nucl Med 60(1):38–45, 2019 | Supervised denoising trained on paired SPECT/reference data; effective for low-dose scenarios |

---

## 4. Literature & State of the Art (2024–2025)

1. **Gong et al. (2024)** "PET/SPECT Image Reconstruction via Score-Based Diffusion Model," *IEEE TMI* — proposes score-based diffusion priors for emission tomography, demonstrating improved noise-resolution tradeoff over OSEM+PSF.
2. **Zhou et al. (2024)** "Deep learning-based attenuation correction for brain SPECT without CT," *Eur J Nucl Med Mol Imaging* — uses paired MRI/CT-trained networks for CT-free attenuation map estimation.
3. **Marin et al. (2025)** "Self-supervised denoising for low-count SPECT using noise2void framework," *Med Phys* — demonstrates unsupervised denoising without paired clean references for clinical low-dose protocols.
4. **Shiri et al. (2024)** "Direct quantitative SPECT reconstruction using unrolled optimization networks," *Phys Med Biol* — unrolls OSEM iterations into a trainable network with learnable regularization parameters.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/spect_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/spect_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/spect_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/spect/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns OSEM, FBP+Chang, TV-MLEM, and deep-learning denoisers — all well-established in the SPECT literature and appropriate for the attenuation-correction inverse problem. The forward model with Poisson noise, depth-dependent CDR, and attenuation matrix accurately captures the physics of Tc-99m gamma-camera acquisition. Benchmark structure with mismatch in mu_map, CDR, scatter, and orbit radius tests generalisation across realistic system imperfections.

---
*Comprehensive 6-point check by deep-check pipeline v3*
