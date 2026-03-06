# Comprehensive 6-Point Check — Endoscopy

**URL:** https://pwm.platformai.org/benchmark/endoscopy
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Fiber Bundle Endoscopy (Monocular Depth & Scene Reconstruction)

**Physical principle:** An endoscope illuminates tissue with a white-light source and captures reflected radiance through a fiber bundle or CMOS chip at the distal tip. The image formation follows the standard pinhole camera model modulated by tissue reflectance (Lambertian + specular components), and depth is encoded implicitly in perspective projection, defocus blur, and shading gradients. Fiber-bundle endoscopes additionally impose a honeycomb sampling pattern on the image, requiring interpolation before depth estimation.

**Forward model:**
```
I(u,v) = (1/d²) · ρ(u,v) · (n̂ · l̂) + s(u,v) + η

where:
  I(u,v)      — observed pixel intensity at image coordinates (u,v)
  d           — source-to-surface distance
  ρ(u,v)      — tissue albedo (Lambertian reflectance)
  n̂           — surface normal at the corresponding 3D point
  l̂           — unit illumination direction
  s(u,v)      — specular highlight term (Cook-Torrance BRDF)
  η           — sensor noise (Gaussian read noise + Poisson shot noise)
  D(u,v)      — depth map to recover from I(u,v)
```

**Inverse problem:** Recover the dense per-pixel depth map D(u,v) and/or 3D surface reconstruction from a single monocular endoscopic frame or short video sequence.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(white light) → F(tissue surface) → D(CMOS/fiber bundle)

**Key mismatch parameters:**
- `illumination_distance`: working distance of light source; nominal 10 mm, perturbed 5 mm (overexposure)
- `specular_weight`: contribution of specular highlights; nominal 0.1, perturbed 0.4 (heavy specularities)
- `fiber_bundle_sampling`: fiber density (pixels/mm²); nominal 30,000 fibers, perturbed 10,000 (coarser sampling)
- `tissue_albedo_bias`: mean tissue reflectance; nominal 0.25, perturbed 0.15 (darker tissue, e.g., colon vs. stomach)

**Dataset format:**
- `x_true: (H, W)` — ground-truth depth map in mm, range [5, 100] mm
- `y: (H, W, 3)` — RGB endoscopic frame (possibly with fiber-bundle mask)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| MonoDepth2 | Deep Learning (self-supervised) | Godard et al., ICCV 2019 | Self-supervised monocular depth trained on video sequences; strong baseline for endoscopy |
| EndoSFM | Deep Learning (SfM) | Liu et al., MICCAI 2019 | Structure-from-motion adapted for non-rigid endoscopic scenes |
| AF-SfMLearner | Deep Learning | Shao et al., MICCAI 2022 | Appearance-flow SfM learner handling tissue deformation |
| LightDepth / Transformer | Transformer | Cui et al., IEEE Trans. Med. Imaging 2023 | Vision transformer for depth estimation in colonoscopy with lighting correction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Wang et al. (2024)** "EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera," *MICCAI 2024* — adapter-based fine-tuning of large vision models for endoscopic depth with minimal labeled data.
2. **Zhao et al. (2024)** "Generalized Endoscopic Reconstruction via Geometry-Aware Diffusion Models," *CVPR 2024* — diffusion-prior model for consistent 3D reconstruction across endoscope types.
3. **Cui et al. (2023)** "Surgical-DINO: Adapter Learning of Foundation Models for Depth Estimation in Endoscopic Surgery," *arXiv 2023* — demonstrates DINOv2 adapters outperforming task-specific models on EndoSLAM dataset.
4. **Huang et al. (2024)** "Self-supervised Monocular Depth Estimation for Gastrointestinal Endoscopy," *Med. Image Anal.* — comprehensive benchmark of self-supervised methods across colonoscopy, gastroscopy, and capsule endoscopy.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/endoscopy_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/endoscopy_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/endoscopy_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/endoscopy/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Endoscopy depth estimation is well-posed as a monocular inverse problem under a Lambertian + specular reflectance model, and the algorithm routing correctly emphasizes self-supervised learning approaches (MonoDepth2, EndoSFM) that dominate this domain due to the scarcity of ground-truth depth annotations in clinical settings. The mismatch parameters capturing illumination distance, specular highlights, fiber-bundle sampling density, and tissue albedo variation represent the principal sources of distribution shift between training and deployment environments. The benchmark structure is appropriate for evaluating robustness to these clinically relevant perturbations.

---
*Comprehensive 6-point check by deep-check pipeline v3*
