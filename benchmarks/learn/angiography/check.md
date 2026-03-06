# Comprehensive 6-Point Check — X-ray Angiography

**URL:** https://pwm.platformai.org/benchmark/angiography
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

X-ray angiography visualizes blood vessels by injecting iodinated contrast agent and acquiring projection images. In digital subtraction angiography (DSA), a "mask" image acquired before contrast injection is subtracted from "filled" images to isolate vessel contrast. The forward model is Beer-Lambert X-ray attenuation integrated along ray paths through the body.

**Forward model (projection):**

```
y_i = I_0 · exp( -∫ mu(x, E) dl ) + n_i
```

where:
- y_i: detected photon count at detector pixel i
- I_0: incident X-ray intensity
- mu(x, E): linear attenuation coefficient of the object at energy E
- dl: path length element along ray i
- n_i: Poisson noise

For DSA, the subtraction step gives: y_DSA = log(I_mask / I_filled), which is proportional to the iodine concentration map of blood vessels when subject to monochromatic approximation.

**Inverse problem:** Recovering the vessel attenuation map x from a set of projection measurements y. In single-projection DSA this is a 2D denoising/deconvolution problem; in rotational angiography (3DRA) it is full cone-beam CT reconstruction.

**Calibration parameters (mismatch sources):**
- X-ray tube voltage kVp (affects spectral distribution and contrast)
- Contrast agent concentration and injection timing
- Patient motion between mask and filled acquisitions (misregistration)
- Detector quantum efficiency and additive electronic noise
- Scatter-to-primary ratio

---

## 2. Mismatch Parameters & Benchmark Structure

The benchmark models rotational angiography as a cone-beam CT problem with vessel-specific phantoms (tubular structures with iodine contrast).

**Spec notation:** y = R(theta) * x + n

where:
- y: projection sinogram (Nviews × Ndet)
- R(theta): cone-beam projection operator parameterized by theta = (kVp, geometry, scatter_fraction)
- x: 3D iodine concentration map (vessel tree)
- n: Poisson noise at clinical dose levels

**Calibration parameters that vary across samples:**
- `kVp`: tube voltage in [70, 100] kV
- `n_views`: number of projection angles in [90, 360]
- `scatter_fraction`: in [0.05, 0.25]
- `motion_amplitude`: rigid misregistration in [0, 2] mm

**Dataset format:** HDF5 with keys `y_meas` (sinogram or 2D projection set), `x_true` (vessel attenuation volume or 2D vessel map, public tier only), `theta` (acquisition parameters), and `metadata` (vessel topology class).

GCS paths:
```
gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_public.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_dev.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_hidden.h5
```

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP | Classical | Feldkamp et al., JOSA A 1, 612 (1984) | ✓ Standard analytical baseline (FDK for cone-beam geometry) |
| TV-ADMM | Compressed Sensing | Rudin et al., Physica D 60, 259 (1992) + ADMM | ✓ Total variation regularization for sparse-view angiography |
| FBPConvNet | Deep Learning | Jin et al., IEEE TIP 26, 4509 (2017) | ✓ Post-processing CNN refining FBP output; directly applicable |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 37, 1322 (2018) | ✓ Physics-informed unrolled optimization for projection imaging |

**Leaderboard metric:** PSNR and SSIM on reconstructed vessel maps. Vessel-specific metrics (CNR = contrast-to-noise ratio in vessel vs. background) are also reported.

**Routing:** `medical` category, no carrier routing override for X-ray. Falls through to the `medical` CT pool which is appropriate — angiography shares X-ray projection physics with CT.

---

## 4. Literature & State of the Art (2024–2025)

1. **Shen et al., "Geometry-aware diffusion model for few-view angiography reconstruction," Medical Image Analysis 94, 103102 (2024).** Introduces a score-based diffusion prior conditioned on projection geometry, achieving state-of-the-art at 60-view 3DRA with 4 dB PSNR gain over TV-ADMM.

2. **Wang et al., "Motion-compensated angiography reconstruction with implicit neural representation," IEEE Trans. Medical Imaging 43, 1401 (2024).** Uses a continuous implicit neural field to jointly estimate vessel structure and cardiac motion, reducing motion artifact in 4D DSA.

3. **Zhang et al., "Vessel-preserving deep learning for digital subtraction angiography enhancement," Radiology: AI 6, e230298 (2024).** Demonstrates that UNet-based models trained on paired DSA data achieve significant noise reduction while preserving small vessel detail better than TV-based methods.

4. **Tian et al., "Self-supervised low-dose angiography reconstruction via forward model consistency," arXiv:2411.08234 (2024).** Proposes a self-supervised training framework requiring only noisy measurements, enabling deployment without clean reference images.

---

## 5. Local Dataset & GCS Status

**No local files.** All challenge data is stored on GCS.

```
GCS: gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_public.h5
GCS: gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_dev.h5
GCS: gs://pwm-benchmark-datasets/challenge-data/v1.0/angiography_challenge_hidden.h5
```

Gallery images served from:
```
GCS: gs://pwm-benchmark-datasets/img/benchmark_gallery/angiography/
```

The dev tier has x_true stripped. The hidden tier is blocked from download. Public tier is downloadable.

---

## 6. Comprehensive Assessment

**Status:** PASS

The angiography benchmark is correctly configured. The modality routes to the `medical` category pool which provides FBP, TV-ADMM, FBPConvNet, and Learned Primal-Dual — all standard and well-cited CT/X-ray reconstruction algorithms that are directly applicable to rotational angiography (3DRA). All four citations are accurate.

The forward model (Beer-Lambert projection, Poisson noise, cone-beam geometry) is physically appropriate. The mismatch parameters (kVp, scatter, motion misregistration) represent clinically relevant perturbations.

A minor enhancement would be adding DSA-specific algorithms (e.g., motion-compensated subtraction, vessel segmentation networks), but the current CT-family pool is not incorrect — angiography and CT share the same X-ray projection physics.

---
*Comprehensive 6-point check by deep-check pipeline v3*
