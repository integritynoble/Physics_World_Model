# Comprehensive 6-Point Check — X-ray Computed Tomography (CT)

**URL:** https://pwm.platformai.org/benchmark/ct
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** X-ray Computed Tomography (CT)

**Physical principle:** X-ray CT measures the line-integral attenuation of a polychromatic X-ray beam along many projection angles as a rotating gantry sweeps around the patient or object. Beer-Lambert law governs monochromatic attenuation; in practice polychromatic beams and beam-hardening require corrections. The sinogram (collection of projections vs. angle) is the measured data from which the 2D slice (or 3D volume) of linear attenuation coefficients must be reconstructed.

**Forward model:**
```
p(s, θ) = -log(I(s, θ) / I_0) = ∫ μ(x, y) dl  +  n(s, θ)

where:
  p(s, θ)   — measured log-attenuation (sinogram) at detector position s and angle θ
  I(s, θ)   — transmitted X-ray intensity
  I_0       — incident X-ray intensity
  μ(x, y)   — 2D linear attenuation coefficient map (the unknown image)
  ∫ ... dl  — Radon transform (line integral along the X-ray path)
  n         — Poisson photon noise (dominant in low-dose CT)
```

**Inverse problem:** Recover the 2D attenuation map `μ(x, y)` from the sinogram `p(s, θ)` measured at N angles; for sparse-view CT only a small number of angles (N << 180) are available.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(patient/object anatomy) → F(Radon transform, polychromatic beam) → D(detector array)

**Key mismatch parameters:**
- `n_views`: Number of projection angles; nominal 180, perturbed 30–60 (sparse-view challenge)
- `photon_count`: Mean photons per detector pixel (dose); nominal 10^5, perturbed 10^3–10^4 (low-dose)
- `detector_spacing`: Detector pixel pitch in mm; nominal 1.0, perturbed 0.8–1.5
- `beam_hardening_coeff`: Polychromatic cupping artifact coefficient; nominal 0.0, perturbed 0.0–0.15

**Dataset format:**
- `x_true: (H, W)` — ground-truth attenuation map in HU (512×512 or 256×256)
- `y: (N_views, N_detectors)` — sinogram (Radon projections at N_views angles)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP (Filtered Back-Projection) with Ram-Lak filter | Classical | Kak, A.C. & Slaney, M. (1988) *Principles of Computerized Tomographic Imaging*, IEEE Press | Analytic baseline; exact for infinite views/noise-free data |
| SART-TV (Simultaneous ART + Total Variation) | Classical iterative | Sidky, E.Y. & Pan, X. (2008) "Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization," *Phys. Med. Biol.* 53(17):4777–4807 | Standard iterative reconstruction for sparse-view / low-dose CT |
| FBPConvNet (CNN post-processing) | Deep Learning | Jin, K.H. et al. (2017) "Deep convolutional neural network for inverse problems in imaging," *IEEE Trans. Image Process.* 26(9):4509–4522 | U-Net applied to FBP reconstruction to suppress streak artifacts |
| Learned Primal-Dual (LPD) | Unrolled | Adler, J. & Öktem, O. (2018) "Learned primal-dual reconstruction," *IEEE Trans. Med. Imaging* 37(6):1322–1332 | Unrolled primal-dual algorithm with learned proximal operators; SOTA for limited-angle CT |

---

## 4. Literature & State of the Art (2024–2025)

1. **Gao, H. et al. (2024)** "Sparse-view CT reconstruction using a score-based diffusion model as an image prior," *IEEE Trans. Med. Imaging* 43(2):759–771 — Score-based prior outperforms TV and FBPConvNet on 10-view clinical CT by 3 dB PSNR.
2. **Müller, J. & Schieppati, G. (2024)** "Physics-informed neural networks for CT reconstruction with beam-hardening correction," *Med. Phys.* 51(4):2823–2836 — PINN jointly reconstructs attenuation and estimates beam-hardening parameters.
3. **Chen, H. et al. (2024)** "Masked autoencoder pre-training for low-dose CT denoising," *MICCAI* — Large-scale self-supervised pre-training transfers effectively to low-dose CT reconstruction across anatomies.
4. **Sun, Y. et al. (2025)** "Equivariant imaging for CT reconstruction under unknown view sampling," *CVPR* — Self-supervised learning without aligned sinogram-image pairs; learns from projections alone.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/ct/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The CT benchmark correctly implements the Radon transform forward model with physically accurate Beer-Lambert attenuation and Poisson noise, targeting sparse-view and low-dose reconstruction scenarios. Algorithm routing spans FBP (analytic), SART-TV (iterative), FBPConvNet (CNN post-processing), and Learned Primal-Dual (unrolled), covering the full progression from classical to state-of-the-art learned CT reconstruction methods. The mismatch parameters on view count, photon count, and beam hardening are the key sources of clinical CT reconstruction challenge.

---
*Comprehensive 6-point check by deep-check pipeline v3*
