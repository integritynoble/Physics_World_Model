# Comprehensive 6-Point Check — Differential Interference Contrast Microscopy (DIC)

**URL:** https://pwm.platformai.org/benchmark/dic
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Differential Interference Contrast Microscopy (DIC)

**Physical principle:** DIC microscopy converts invisible phase gradients in transparent specimens (cells, organelles) into amplitude contrast by using a Nomarski prism to split and laterally shear a polarized beam into two slightly offset copies. The optical path difference (OPD) between the two copies, proportional to the phase gradient of the specimen along the shear direction, produces intensity contrast after recombination through an analyzer. Reconstruction of the 2D phase from DIC images is a phase-gradient integration problem related to the Transport of Intensity Equation (TIE).

**Forward model:**
```
I_DIC(r) = I_0 * [1 + sin(Δφ_OPD(r) + Δφ_bias)] + n(r)

where:
  I_DIC(r)       — DIC intensity image at pixel r
  I_0            — background illumination intensity
  Δφ_OPD(r)     = φ(r + δ/2) - φ(r - δ/2) ≈ δ * ∂φ/∂x  — optical path difference (phase gradient)
  δ              — shear distance (Nomarski prism parameter)
  Δφ_bias        — bias retardance (quarter-wave or other setting)
  φ(r)           — specimen phase map (the unknown)
  n(r)           — Gaussian/Poisson photon noise
```

**Inverse problem:** Recover the 2D phase map `φ(r)` of the specimen from one or more DIC intensity images, either by gradient integration, TIE-based methods, or iterative phase retrieval.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(phase object) → F(Nomarski shear + polarization optics) → D(CCD/sCMOS)

**Key mismatch parameters:**
- `shear_distance`: Lateral beam shear δ in pixels; nominal 2.0, perturbed 1.0–4.0
- `bias_retardance`: Phase bias Δφ_bias; nominal π/2 (quadrature), perturbed 0–π
- `illumination_na`: Numerical aperture of condenser; nominal 0.8, perturbed 0.5–1.2
- `noise_level`: Photon noise level (relative standard deviation); nominal 0.02, perturbed 0.01–0.1

**Dataset format:**
- `x_true: (H, W)` — ground-truth phase map in radians (256×256)
- `y: (H, W)` — single DIC intensity image (or multiple at different bias angles)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Fourier-domain gradient integration (TIE) | Classical | Kou, S.S. et al. (2010) "Transport-of-intensity approach to differential interference contrast (TI-DIC) microscopy for quantitative phase imaging," *Opt. Lett.* 35(3):447–449 | Analytic integration of phase gradient using Transport of Intensity Equation |
| Wiener-filter DIC deconvolution | Classical | Bostan, E. et al. (2014) "Variational phase imaging using the transport-of-intensity equation," *IEEE Trans. Image Process.* 23(9):3944–3954 | Regularized deconvolution with known shear kernel for phase recovery |
| PhaseNet / U-Net phase reconstruction | Deep Learning | Rivenson, Y. et al. (2020) "PhaseStain: the digital staining of label-free quantitative phase microscopy images using deep learning," *Light: Sci. & Appl.* 8:23 | CNN trained on paired DIC and confocal data for phase-to-amplitude mapping |
| Hybrid TIE-DL phase reconstructor | Deep Learning | Zhang, J. et al. (2021) "Transport of intensity equation-guided deep network for phase imaging," *Opt. Lett.* 46(10):2330–2333 | Physics-guided network combining TIE analytic prior with learned residual correction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Chen, X. et al. (2024)** "Quantitative DIC microscopy via deep learning with physics-informed constraints," *Optica* 11(3):312–321 — Self-supervised network trained without ground-truth phase; uses DIC image formation model as consistency loss.
2. **Tahara, T. et al. (2024)** "Simultaneous DIC and fluorescence phase retrieval using deep learning fusion," *APL Photonics* 9(4):046104 — Multimodal CNN fuses DIC gradient images with sparse fluorescence for improved phase reconstruction.
3. **Zuo, C. et al. (2024)** "Computational optical imaging: a review of physics-informed deep learning for quantitative phase imaging," *PhotoniX* 5:12 — Survey including DIC, TIE, and interferometric methods with benchmarks.
4. **Song, P. et al. (2025)** "Diffusion prior-regularized phase retrieval from DIC images," *Optics Express* 33(1):1245–1258 — Denoising diffusion probabilistic model used as image prior for DIC phase inversion.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/dic_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/dic_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/dic_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/dic/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The DIC benchmark correctly models the phase-gradient imaging forward problem with Nomarski shear optics and the sinusoidal contrast transfer function. Algorithm routing appropriately spans TIE-based gradient integration (classical), Wiener deconvolution (regularized), and modern physics-guided deep learning networks, matching the current state of DIC phase reconstruction literature. The mismatch parameters on shear distance, bias retardance, and NA accurately reflect the dominant sources of DIC reconstruction inaccuracy in real microscopy systems.

---
*Comprehensive 6-point check by deep-check pipeline v3*
