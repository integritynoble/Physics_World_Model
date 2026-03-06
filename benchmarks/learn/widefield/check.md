# Comprehensive 6-Point Check — Widefield Fluorescence Microscopy

**URL:** https://pwm.platformai.org/benchmark/widefield
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Widefield Fluorescence Microscopy (Epifluorescence)

**Physical principle:** Widefield fluorescence microscopy illuminates the entire field of view simultaneously with a mercury arc lamp, LED, or laser. Fluorophores are excited uniformly through the objective; emitted light is collected by the same objective, passed through a dichroic/emission filter, and focused onto a CCD or sCMOS camera. The key limitation is out-of-focus fluorescence, which contributes background blur governed by the 3-D optical transfer function (OTF) of the objective. Computational deconvolution seeks to reverse this blurring.

**Forward model:**
```
y(r) = h_3D(r) ⊛ x(r) + n(r)

h_3D(r) = |F⁻¹[OTF(k)]|  (3-D PSF: Gaussian in xy, defocused Stokes in z)

2-D slice approximation:
  y(x,y) = h_2D(x,y) ⊛ x_focus(x,y) + b_OOF(x,y) + n

where:
  x(r)        — 3-D fluorophore density
  h_3D(r)     — 3-D PSF of the objective
  b_OOF       — out-of-focus blur contribution from z ≠ z_focus
  n           ~ Poisson photon shot noise + Gaussian readout noise (sCMOS)
```

**Inverse problem:** Recover the in-focus fluorophore density (2-D) or full 3-D distribution from the blurred, noisy widefield image, removing out-of-focus contributions and suppressing noise.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(illumination/filter set) → F(fluorophore density/3-D distribution) → D(camera/objective)

**Key mismatch parameters:**
- `psf_fwhm_lateral_nm`: Lateral PSF FWHM; nominal 250 nm (NA 1.4, 488 nm), perturbed 200–500 nm
- `psf_fwhm_axial_nm`: Axial PSF FWHM; nominal 600 nm, perturbed 400–1500 nm
- `oof_z_range_um`: Out-of-focus volume contributing blur; nominal ±5 µm, perturbed ±2–15 µm
- `snr_linear`: Signal-to-noise ratio of in-focus signal; nominal 20, perturbed 5–50

**Dataset format:**
- `x_true: (H, W)` — ground-truth in-focus fluorophore density
- `y: (H, W)` — blurred widefield image with out-of-focus background and noise

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Wiener filter deconvolution | Classical analytical | McNally et al., J Opt Soc Am A 11(4):1056–1067, 1994 | Frequency-domain deconvolution with noise regularization; fast single-image method |
| Richardson-Lucy iterative deconvolution | Classical iterative | Richardson, J Opt Soc Am 62(1):55–59, 1972; Lucy, AJ 79:745, 1974 | Poisson ML deconvolution; industry standard in software like ImageJ/Fiji DeconvolutionLab2 |
| Blind deconvolution via PSF estimation (AutoQuant) | Classical blind | Sarder & Nehorai, IEEE Signal Process Mag 23(3):32–45, 2006 | Jointly estimates PSF and image from widefield data without measured PSF calibration |
| CARE / content-aware fluorescence restoration | Deep Learning | Weigert et al., Nat Methods 15(12):1090–1097, 2018 | Supervised U-Net trained on paired widefield/confocal; restores 3-D structure from widefield |

---

## 4. Literature & State of the Art (2024–2025)

1. **Zhang et al. (2024)** "Virtual confocal microscopy from widefield images using diffusion-based conditional generation," *Nat Commun* — score-based model trained to generate confocal-equivalent sections from widefield z-stacks.
2. **Christensen et al. (2024)** "Self-supervised fluorescence deconvolution without paired training data," *Biomed Opt Express* — blind-spot network deconvolution exploiting noise statistics of sCMOS for unsupervised training.
3. **Qiao et al. (2025)** "Transformer-based 3-D deconvolution for widefield neuron volume imaging," *Light Sci Appl* — ViT with positional encoding in 3-D for joint depth estimation and deconvolution of thick neuronal tissue.
4. **Guo et al. (2024)** "Rapid widefield to super-resolution via flow-matching generative model," *CVPR* — normalizing flow conditioned on widefield input for deterministic SR predictions with calibrated uncertainty.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/widefield_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/widefield_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/widefield_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/widefield/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns Wiener filter, Richardson-Lucy, blind deconvolution, and CARE deep restoration — the canonical pipeline from classical to deep-learning approaches for widefield fluorescence deconvolution. The forward model with 3-D PSF, out-of-focus blur, and mixed Poisson/Gaussian noise faithfully represents widefield epifluorescence physics. Mismatch in PSF size, out-of-focus range, and SNR tests robustness across objectives, wavelengths, and cell/tissue preparations.

---
*Comprehensive 6-point check by deep-check pipeline v3*
