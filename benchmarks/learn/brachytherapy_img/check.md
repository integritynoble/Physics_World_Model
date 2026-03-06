# Comprehensive 6-Point Check — Brachytherapy Imaging

**URL:** https://pwm.platformai.org/benchmark/brachytherapy_img
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Brachytherapy Imaging

**Physical principle:** Brachytherapy is an internal radiation therapy technique where radioactive seeds or sources (typically I-125, Ir-192, or Pd-103) are placed directly in or adjacent to a tumour. Post-implant imaging using X-ray fluoroscopy or CT verifies seed placement geometry for dose verification. The imaging problem is an X-ray projection/CT reconstruction: the seeds appear as high-attenuation point objects on a background of soft tissue anatomy. Accurate seed localisation (sub-millimetre precision) is required for dose-volume histogram calculation.

**Forward model:**
```
Beer-Lambert projection (monoenergetic approximation):
  g(u,v,θ) = ∫∫∫ μ(x,y,z) · δ(u - x cos θ - z sin θ, v - y) dx dy dz

Multi-view discrete form:
  y = A x + n
  y ∈ R^{N_views × M × N}   — projection images
  x ∈ R^{H × W × D}         — 3D attenuation map (seeds + tissue)
  A                          — X-ray projection operator (Radon transform)
  n                          — quantum + detector noise

Seed localisation:
  Brachytherapy seed model: μ_seed >> μ_tissue
  Point source assumption allows seed detection as local maxima in reconstructed volume
```

**Inverse problem:** Recover the 3D seed position map from limited-angle X-ray projections (fluoroscopy) or CT data, with high sensitivity to sub-mm position errors for dose verification.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** Π(X-ray projection) → D(flat-panel detector)

**Key mismatch parameters:**
- `source_position_error` (s_p): seed centroid localisation uncertainty; nominal 0.0 mm, perturbed 0.4 mm
- `attenuation_coefficient` (a_c): tissue linear attenuation calibration; nominal 0.20/cm, perturbed 0.21/cm
- `detector_gain_drift` (d_g): detector gain temporal drift; nominal 1.0, perturbed 1.01
- `scatter_fraction` (s_f): scattered radiation contamination; nominal 0.15, perturbed 0.17

**Dataset format:**
- `x_true: (H, W)` — 2D seed distribution projection (ground truth seed localisation)
- `y: (N_views, H, W)` — multi-view X-ray projection images
- `H_ideal: (N_views*H*W, H*W)` — ideal X-ray projection operator (Radon matrix)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP | Classical | Kak & Slaney 1988 | Filtered Back Projection; standard CT reconstruction baseline; applicable to brachytherapy CT verification |
| TV-ADMM | Classical/Variational | Rudin et al. 1992; ADMM: Boyd et al. 2011 | Total variation regularisation; reduces CT streak artefacts from high-density seeds |
| PnP-ADMM | Plug-and-Play | Venkatakrishnan et al., IEEE GlobalSIP 2013 | ADMM with learned denoising prior; applicable to limited-view brachytherapy reconstruction |
| FBPConvNet | Deep Learning | Jin et al., IEEE TIP 2017 | CNN post-processing of FBP images; removes reconstruction artefacts around metallic seeds |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018 | Unrolled primal-dual optimisation; directly applicable to CT reconstruction |
| DuDoTrans | Transformer | Wang et al., IEEE TMI 2022 | Dual-domain Transformer for CT reconstruction from sparse projections |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deep learning for brachytherapy seed detection** (Ma et al., Med. Phys. 2022 / extended 2024): 3D CNN for automatic seed segmentation and counting from post-implant CT; achieves 98% detection rate.
2. **Metal artefact reduction for brachytherapy CT** (2024): Transformer-based sinogram interpolation to reduce streak artefacts from I-125 seeds; improves seed localisation accuracy by ~30%.
3. **Limited-angle reconstruction for fluoroscopy-based verification** (2024): Learned primal-dual network adapted to the 3–5 projection geometry of intra-operative fluoroscopy; outperforms FBP for real-time dose verification.
4. **Diffusion model for dose-guided reconstruction** (2025): Score-based posterior sampling conditioned on dose constraints; ensures reconstructed seed positions are consistent with TG-43 dose distribution requirements.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brachytherapy_img_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brachytherapy_img_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/brachytherapy_img_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/brachytherapy_img/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing for carrier `Gamma/X-ray` falls through to the `medical` CT reconstruction pool (13 methods: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN, FBPConvNet, RED-CNN, Learned Primal-Dual, DuDoTrans, CT-ViT, CTFormer, DOLCE, DiffusionCT, Score-CT). Since brachytherapy verification imaging is fundamentally X-ray CT, these CT algorithms are technically correct. The four mismatch parameters — seed position error, tissue attenuation coefficient, detector gain drift, scatter fraction — target the key calibration uncertainties in post-implant dosimetry. Note: the compound carrier `Gamma/X-ray` does not match explicit routing entries but the fallthrough to the medical CT pool is correct for this modality.

---
*Comprehensive 6-point check by deep-check pipeline v3*
