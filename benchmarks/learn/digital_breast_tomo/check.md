# Comprehensive 6-Point Check — Digital Breast Tomosynthesis (DBT)

**URL:** https://pwm.platformai.org/benchmark/digital_breast_tomo
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Digital Breast Tomosynthesis (DBT)

**Physical principle:** Digital breast tomosynthesis is a limited-angle 3D X-ray mammography technique. The X-ray source sweeps through a narrow angular arc (typically 15–50° total) above the compressed breast, acquiring 9–25 low-dose 2D projection images. These projections are reconstructed into a series of in-focus planes (pseudo-3D tomograms) that reduce tissue overlap compared to conventional 2D mammography. The limited angular range causes significant artefacts (elongation along z, residual out-of-focus structures) that require dedicated reconstruction algorithms. DBT has become the clinical standard for breast cancer screening in many countries due to superior lesion detection compared to 2D mammography.

**Forward model:**
```
X-ray Beer-Lambert projection (linearised log model):
  p_i(u,v) = ∫∫∫ μ(x,y,z) · δ(u - f_x(x,z,θ_i), v - y) dx dy dz

DBT discrete form:
  y = A_θ x + n

where:
  x ∈ R^{H × W × D}             — 3D breast attenuation map (ground truth)
  A_θ                            — DBT projection operator (limited-angle geometry)
  θ_i ∈ [-α/2, +α/2]           — projection angles (total arc α ≈ 15–50°)
  N_proj ≈ 9–25                  — number of projections
  y ∈ R^{N_proj × H × W}        — projection image stack
  n                              — quantum noise (Poisson) + detector noise

Limited-angle effect:
  Missing Fourier cone: |k_z/k_xy| > tan(α/2) → elongation artefacts along z
```

**Inverse problem:** Reconstruct the 3D breast attenuation map x from a limited set of low-dose angled projections {y_i}, suppressing out-of-plane artefacts while preserving calcification and mass detectability at minimal radiation dose.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** Π(limited-angle X-ray) → D(flat-panel detector)

**Key mismatch parameters:**
- `angular_range_error` (a_r): total angular sweep deviation; nominal 0.0°, perturbed 0.4°
- `detector_motion_blur` (d_m): detector motion during source sweep; nominal 0.0 px, perturbed 0.1 px
- `scatter_fraction` (s_f): scattered X-ray contamination fraction; nominal 0.30, perturbed 0.36

**Dataset format:**
- `x_true: (H, W)` — 2D slice of the 3D breast reconstruction (ground truth in-plane slice)
- `y: (N_proj, H, W)` — limited-angle projection image stack
- `H_ideal: (N_proj*H*W, H*W)` — ideal limited-angle projection operator (Radon geometry)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP | Classical | Kak & Slaney 1988 | Filtered Back Projection; standard DBT reconstruction baseline (with limited-angle artefacts) |
| TV-ADMM | Classical/Variational | Rudin et al. 1992; ADMM: Boyd et al. 2011 | TV-regularised iterative reconstruction; reduces out-of-plane artefacts vs FBP |
| PnP-ADMM | Plug-and-Play | Venkatakrishnan et al., IEEE GlobalSIP 2013 | ADMM with learned denoising prior; excellent for limited-angle DBT |
| FBPConvNet | Deep Learning | Jin et al., IEEE TIP 2017 | Post-processing CNN applied to FBP output; reduces limited-angle streak artefacts |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018 | End-to-end learned iterative reconstruction; directly applicable to DBT geometry |
| CTFormer | Transformer | Chen et al., Med. Image Anal. 2023 | Transformer-based CT reconstruction; applied to limited-angle DBT |
| DOLCE | Diffusion | Gao et al., ICCV 2023 | Diffusion model for low-dose CT; applicable to DBT dose reduction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deep learning DBT reconstruction** (Sidky et al. / Sanchez et al., Med. Phys. 2023 / 2024): End-to-end deep learning reconstruction outperforms TV-ADMM in clinical DBT reader studies; improved calcification detection sensitivity.
2. **Score-based diffusion for DBT** (2024): Conditional diffusion model posterior sampling for limited-angle DBT reconstruction; provides uncertainty maps for ambiguous lesion interpretation.
3. **Learned primal-dual for DBT dose reduction** (2024): Extension of Adler & Oktem's learned primal-dual to DBT geometry; achieves standard-dose quality from 50% dose reduction projections.
4. **Transformer DBT reconstruction with anatomical priors** (2025): Anatomy-aware Transformer incorporating breast glandular tissue prior from contralateral breast; reduces out-of-plane artefacts near dense glandular-adipose boundaries.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/digital_breast_tomo_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/digital_breast_tomo_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/digital_breast_tomo_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/digital_breast_tomo/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses carrier routing `(medical, X-ray)` → CT reconstruction pool (13 methods: FBP, TV-ADMM, PnP-ADMM, PnP-DnCNN, FBPConvNet, RED-CNN, Learned Primal-Dual, DuDoTrans, CT-ViT, CTFormer, DOLCE, DiffusionCT, Score-CT). Since DBT is fundamentally a limited-angle CT reconstruction problem, the CT algorithm pool is technically correct — all methods (FBP, TV-ADMM, Learned Primal-Dual) are directly applicable to DBT. The three mismatch parameters (angular range error, detector motion blur, scatter fraction) address the key DBT acquisition calibration uncertainties. No code changes are required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
