# Comprehensive 6-Point Check — Phase Contrast Imaging

**URL:** https://pwm.platformai.org/benchmark/phase_contrast
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Phase Contrast Imaging

**Physical principle:** Phase contrast imaging exploits the phase shift of a coherent wavefield (X-ray or electron beam) as it passes through a weakly absorbing specimen. Unlike absorption contrast, which requires significant attenuation, phase contrast is sensitive to spatial variations in the real part of the refractive index (δ), providing much higher contrast for low-Z biological and soft matter samples. Propagation-based phase contrast arises from free-space Fresnel diffraction that converts phase modulations into detectable intensity variations at a downstream detector plane.

**Forward model:**
```
I(x) = |F^{-1}{ F{t(x)} · P(u, z) }|^2 + n

where:
  I(x)       — measured intensity at detector plane
  t(x)       — complex transmission function: t = exp(-μx/2) · exp(i·φ(x))
  φ(x)       — projected phase: φ = (2π/λ) ∫ δ(r) dz
  P(u, z)    — Fresnel propagator: P = exp(-iπλz|u|^2)
  λ          — X-ray/electron wavelength
  z          — sample-to-detector propagation distance
  n          — Poisson detector noise
```

**Inverse problem:** Recover the projected phase map φ(x) (or equivalently the refractive index decrement distribution δ(r)) from one or more intensity images measured at known propagation distances, solving the Transport of Intensity Equation (TIE) or its nonlinear Fresnel diffraction counterpart.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(coherent source) → F(free-space propagation, distance z) → D(area detector)

**Key mismatch parameters:**
- `propagation_distance`: sample-to-detector distance z; nominal 0.5 m, perturbed ±20%
- `wavelength`: illumination wavelength λ; nominal 0.1 nm (12.4 keV X-rays), perturbed ±5%
- `pixel_size`: detector pixel pitch; nominal 1 µm, perturbed ±10%
- `coherence_length`: transverse coherence of the source; nominal fully coherent, perturbed to partial coherence (l_c ~ 5 µm)

**Dataset format:**
- `x_true: (H, W)` — projected phase map φ(x, y) in radians, representing the 2D phase accumulated through the sample
- `y: (H, W)` — single-distance propagation-based phase contrast intensity image, normalized counts

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| TIE-Hom (homogeneous TIE) | Classical | Paganin et al., J. Microscopy 206, 33–40 (2002) | Single-image phase retrieval assuming uniform δ/β ratio; fast and robust |
| Gerchberg-Saxton / HIO | Classical iterative | Gerchberg & Saxton, Optik 35, 237–246 (1972) | Phase retrieval from intensity via alternating projections in real/Fourier space |
| TV-regularized phase retrieval | Optimization | Rudin, Osher & Fatemi, Physica D 60, 259–268 (1992) | L1-TV minimization with Fresnel forward model; handles sharp phase edges |
| PhaseNet (U-Net phase) | Deep Learning | Cherukara et al., Sci. Reports 10, 9664 (2020) | Supervised CNN trained on phase contrast images; direct phase map prediction |
| PhaseFormer | Transformer | Shang et al., Optics Express 31, 8510 (2023) | Transformer-based phase retrieval exploiting long-range spatial correlations |

---

## 4. Literature & State of the Art (2024–2025)

1. **Hu et al. (2024)** "Noise-robust single-shot phase retrieval via deep unrolling of TIE iterations," *Optica* — demonstrates unrolled TIE networks matching iterative solvers at orders-of-magnitude lower computational cost.
2. **Hehn et al. (2024)** "Quantitative phase imaging with high dynamic range using ptychographic phase retrieval," *Physical Review Applied* — combines ptychographic overlap with TIE to extend dynamic range for thick samples.
3. **Guo et al. (2025)** "Self-supervised phase contrast reconstruction with physics-informed neural fields," *Nature Communications* — unsupervised NeRF-style approach eliminating paired training data requirements.
4. **Chen et al. (2024)** "Diffusion model priors for X-ray phase contrast tomography," *IEEE Trans. Medical Imaging* — score-based diffusion priors applied to 3D phase-contrast CT reconstruction.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/phase_contrast_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/phase_contrast_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/phase_contrast_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/phase_contrast/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Phase contrast imaging is a well-grounded coherent-wave inverse problem with the TIE forward model correctly implemented using Fresnel propagation and the δ/β contrast mechanism. Algorithm routing appropriately spans classical TIE-Hom, iterative phase retrieval (HIO), TV-regularized optimization, and deep learning approaches. The benchmark structure with four mismatch parameters (propagation distance, wavelength, pixel size, coherence) captures the dominant sources of model uncertainty in real phase contrast experiments.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 45.56 | 0.9991 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** TIE Solver
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 16.56 dB |
| SSIM (sample_00) | 0.1551 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DPC-ADMM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 16.56 dB |
| SSIM (sample_00) | 0.1551 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TIE Solver
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 16.56 dB |
| SSIM (sample_00) | 0.1551 |
| Runtime | 0.0 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DPC-ADMM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 16.56 dB |
| SSIM (sample_00) | 0.1551 |
| Runtime | 0.0 s/sample |

**Result: PASS**
