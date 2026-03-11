# Comprehensive 6-Point Check — Single-Photon Emission Computed Tomography (SPECT)

**URL:** https://pwm.platformai.org/benchmark/spect
**Check Date:** 2026-03-10
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Single-Photon Emission Computed Tomography (SPECT)

**Physical principle:** SPECT images the 3-D distribution of a gamma-emitting radiopharmaceutical injected into a patient. Gamma photons (typically 140 keV for Tc-99m) are detected by a rotating gamma camera equipped with a parallel-hole collimator. Photon attenuation through tissue and distance-dependent collimator blur (geometric/septal penetration) are the dominant degradation sources.

**Forward model:**
```
y_θ ~ Poisson(P_θ * A_θ * x + scatter)

where:
  x           — radionuclide activity distribution (256x256 normalised to [0,1])
  A_θ         — one-sided attenuation along each ray at angle θ
                A(θ,t) = exp(-∫₀^L μ(r) dr) with μ scaled by mu_map_scale
  P_θ         — projector incorporating collimator-detector response (CDR)
                modelled as depth-dependent Gaussian blur: FWHM(d) = cdr_fwhm * (1 + d/R)
  scatter     — uniform scatter background = scatter_fraction * mean(primary signal)
  y_θ         — measured counts in detector bins at angle θ (Poisson-distributed)
```

**Geometry:**
- 256 projection angles over [0, 360) degrees
- 256 detector bins per angle
- Rotation radius: 180-230 mm (nominal 200 mm)
- Parallel-hole collimator, Tc-99m isotope (140 keV)

**Inverse problem:** Recover the 2-D activity map x from noisy, attenuation-corrupted sinogram y.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(radiopharmaceutical/collimator) → F(attenuation/scatter) → D(gamma camera)

**Key mismatch parameters:**
- `mu_map_scale`: Linear attenuation coefficient scaling; nominal 1.0, perturbed 0.85–1.15
- `cdr_fwhm_mm`: Collimator-detector response FWHM at reference depth; nominal 9.5 mm, perturbed 7.5–12 mm
- `scatter_fraction`: Scatter-to-primary ratio in energy window; nominal 0.12, perturbed 0.05–0.25
- `rotation_radius_mm`: Camera orbit radius; nominal 200 mm, perturbed 180–230 mm

**Dataset format (HDF5):**
- `x_true: (256, 256)` — 2-D activity map normalised to [0, 1]
- `y: (256, 256)` — sinogram of raw photon counts per detector bin (~80 mean counts/bin)
- `H_ideal: (256, 256)` — ideal noiseless sinogram (no attenuation, no scatter, no CDR blur)

**Phantom types:** Cardiac perfusion (hot-wall ventricle + defects), brain perfusion (cortical uptake + caudate/putamen), bone scan (vertebrae + pelvis + ribs)

**Tiers:** public (12 samples), dev (20 samples, no x_true), hidden (20 samples, blocked)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| OSEM (Ordered-Subsets EM) | Classical iterative | Hudson & Larkin, IEEE TMI 13(4):601-609, 1994 | Clinical gold standard for SPECT; handles Poisson statistics and attenuation correction natively |
| FBP + attenuation correction | Classical analytical | Chang, IEEE TNS 25(1):638-643, 1978 | Fast baseline; sinogram-domain attenuation correction + Ram-Lak filtered backprojection |
| TV-regularised MLEM | Variational | Panin et al., IEEE TMI 18(2):130-138, 1999 | Reduces streak artifacts in low-count studies while preserving quantitative accuracy |
| Deep-learning post-filter (U-Net) | Deep Learning | Haggstrom et al., J Nucl Med 60(1):38-45, 2019 | Supervised denoising trained on paired SPECT/reference data; effective for low-dose scenarios |

**FBP Baseline Results (2026-03-10):**

| Tier | Samples | Mean PSNR (dB) | Mean SSIM |
|------|---------|----------------|-----------|
| public | 12 | 30.03 | 0.940 |
| dev | 20 | 30.29 | 0.942 |
| hidden | 20 | 32.64 | 0.915 |

Per-phantom-type breakdown (public):
- Cardiac: 29.9 dB / 0.951 SSIM (4 samples)
- Brain: 26.7 dB / 0.970 SSIM (4 samples)
- Bone: 33.5 dB / 0.899 SSIM (4 samples)

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
- 4 scenes (scene_00 to scene_03) with gt, measurement, and reconstruction PNGs
- Uploaded 2026-03-10

**Dataset sizes:** public 7.1 MB, dev 21 MB, hidden 21 MB

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns OSEM, FBP+attenuation-correction, TV-MLEM, and deep-learning denoisers -- all well-established in the SPECT literature and appropriate for the attenuation-correction inverse problem. The forward model with Poisson noise, depth-dependent CDR, one-sided attenuation, and scatter accurately captures the physics of Tc-99m gamma-camera acquisition.

FBP baseline achieves 30.0 dB PSNR and 0.94 SSIM on the public tier, confirming the benchmark difficulty is appropriate (above 25 dB threshold, room for improvement). The per-phantom-type variation (brain ~27 dB, cardiac ~30 dB, bone ~34 dB) reflects realistic differences in phantom complexity. Mismatch in mu_map_scale, CDR FWHM, scatter_fraction, and rotation_radius tests generalisation across realistic system imperfections.

All three challenge tiers generated and uploaded to GCS (2026-03-10). Dev tier has no x_true; hidden tier is download-blocked.

---
*Comprehensive 6-point check by deep-check pipeline v3.1 (updated 2026-03-10)*
