# Comprehensive 6-Point Check — Transmission Electron Microscopy (TEM)

**URL:** https://pwm.platformai.org/benchmark/tem
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Transmission Electron Microscopy (TEM)

**Physical principle:** In TEM, a high-energy (80–300 keV) electron beam passes through a thin specimen (< 100 nm). The exit wavefunction ψ_exit carries amplitude and phase information about the projected specimen potential. Phase-contrast imaging (bright-field TEM) records the intensity |ψ_image|², where the contrast transfer function (CTF) describes the Fourier-space weighting imposed by objective lens aberrations and defocus. At high resolution, atomic-scale phase contrast is formed by interference between unscattered and scattered beams.

**Forward model:**
```
Y(k) = CTF(k) · Ψ_exit(k) + N(k)

CTF(k) = A(k) · exp(iχ(k))
χ(k)   = π·λ·k²·Δf + (π/2)·C_s·λ³·k⁴

where:
  Ψ_exit(k)  — Fourier transform of exit wavefunction
  Δf          — defocus (positive = underfocus)
  C_s         — spherical aberration coefficient
  λ           — electron wavelength (0.00197 nm at 300 keV)
  A(k)        — aperture function (= 0 for k > k_max)
  N(k)        ~ complex Gaussian detector noise
  y(r)        = |F⁻¹[CTF · Ψ_exit]|² — recorded intensity
```

**Inverse problem:** Recover the projected potential (or exit wavefunction phase) from one or more defocus series images, compensating for the CTF oscillations and noise.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(electron gun/condenser) → F(specimen potential/thickness) → D(objective lens CTF/camera)

**Key mismatch parameters:**
- `defocus_nm`: Objective lens defocus; nominal -60 nm (underfocus), perturbed -20 to -200 nm
- `cs_mm`: Spherical aberration of objective lens; nominal 1.2 mm, perturbed 0.001–2.0 mm
- `accelerating_voltage_kV`: Electron energy; nominal 300 kV, perturbed 80–300 kV
- `detector_mtf_cutoff`: Modulation transfer function cutoff of CCD/direct detector; nominal 0.7 Nyquist, perturbed 0.4–1.0

**Dataset format:**
- `x_true: (H, W)` — projected electrostatic potential or exit wavefunction phase
- `y: (N_defocus, H, W)` — focal series of TEM images (or single image for single-image methods)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Wiener CTF correction (single image) | Classical analytical | Frank, Optik 38(5):519–536, 1973 | Frequency-domain CTF inversion with noise regularization; fast single-image method |
| Iterative focal series reconstruction (TrueImage) | Classical iterative | Coene et al., Ultramicroscopy 64:109–135, 1996 | Maximum-likelihood exit-wavefunction reconstruction from through-focal series |
| Maximum entropy image processing | Variational | Gull & Daniell, Nature 272:686–690, 1978 | Entropy prior for TEM phase retrieval; effective for sparse/crystalline specimens |
| PhaseGAN / deep phase-contrast retrieval | Deep Learning | Ede & Beanland, npj Comput Mater 7(1):121, 2021 | GAN-based TEM image enhancement and phase retrieval from single defocus images |

---

## 4. Literature & State of the Art (2024–2025)

1. **Farmer et al. (2024)** "Ptychographic neural networks for exit wavefunction reconstruction in TEM," *Ultramicroscopy* — differentiable ptychography with neural network regularization achieving sub-Ångström phase contrast.
2. **Ede (2024)** "Transformer models for TEM image denoising and super-resolution," *npj Comput Mater* — vision transformer applied to TEM restoration, outperforming CNN-based denoisers at low dose.
3. **Chen et al. (2025)** "Atomic-resolution 3D reconstruction from single TEM projections using diffusion models," *Nat Commun* — diffusion-based single-image 3-D structure recovery exploiting crystallographic priors.
4. **Levin et al. (2024)** "Self-supervised CTF estimation and correction for cryo-TEM using equivariant networks," *J Struct Biol* — unsupervised CTF parameter estimation integrated with image restoration, without paired training data.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/tem_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/tem_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/tem_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/tem/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns Wiener CTF correction, focal-series reconstruction, maximum-entropy, and deep-learning phase retrieval — covering the full range of TEM computational imaging approaches. The forward model with CTF, spherical aberration, defocus, and aperture faithfully represents bright-field TEM phase contrast. Mismatch in defocus, C_s, accelerating voltage, and detector MTF tests algorithm robustness to the lens aberration and detector variability encountered in practice.

---
*Comprehensive 6-point check by deep-check pipeline v3*
