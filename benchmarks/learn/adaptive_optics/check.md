# Comprehensive 6-Point Check — Adaptive Optics Wavefront Sensing

**URL:** https://pwm.platformai.org/benchmark/adaptive_optics
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Adaptive Optics Wavefront Sensing

**Physical principle:** Atmospheric turbulence and optical aberrations distort the wavefront of light passing through a telescope or microscope objective. A Hartmann-Shack wavefront sensor subdivides the pupil into an array of lenslets; each lenslet focuses light onto a detector array whose centroid displacements encode local wavefront gradient. The full wavefront phase is then reconstructed from these slope measurements, and a deformable mirror corrects the aberration to sharpen the image.

**Forward model:**
```
s = G * phi + n

where:
  s    ∈ R^{2M}   — measured x/y centroid slopes from M lenslets
  G    ∈ R^{2M×K} — geometry matrix (gradient operator) mapping Zernike coefficients to slopes
  phi  ∈ R^K      — wavefront phase expressed in K Zernike mode coefficients
  n               — Gaussian measurement noise (photon + read noise)
```

**Inverse problem:** Recover the wavefront phase map `phi` (or equivalently the deformable mirror actuator commands) from the slope measurements `s`, given a known sensor geometry matrix `G`.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(pupil/atmosphere) → F(lenslet array) → D(CCD/EMCCD centroids)

**Key mismatch parameters:**
- `r0`: Fried parameter (coherence length); nominal 15 cm, perturbed 8–25 cm
- `n_modes`: Number of Zernike correction modes; nominal 36, perturbed 15–66
- `lenslet_pitch`: Sub-aperture spacing; nominal 0.5 mm, perturbed ±20%
- `noise_level`: Read noise RMS in electrons; nominal 3 e⁻, perturbed 1–10 e⁻

**Dataset format:**
- `x_true: (H, W)` — corrected PSF or ground-truth wavefront phase map (256×256 pixels)
- `y: (N_lenslets, 2)` — measured centroid slope array from the Hartmann-Shack sensor

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Least-squares wavefront reconstructor (zonal) | Classical | Fried, D.L. (1977) "Least-square fitting a wave-front distortion estimate to an array of phase-difference measurements," *J. Opt. Soc. Am.* 67(3):370–375 | Standard baseline using pseudoinverse of the geometry matrix |
| Sparse Zernike recovery (L1 minimization) | Compressed sensing | Doelman, R. et al. (2019) "Simultaneous focal-plane wavefront sensing and science imaging," *A&A* 624:A164 | Promotes modal sparsity; suited for partial-aperture occlusion |
| U-Net wavefront reconstructor | Deep Learning | Hu, L. et al. (2020) "Wavefront correction with a deep learning model and integrated sensor," *Opt. Express* 28(24):36277–36286 | Direct slope-to-phase mapping; effective for frozen-flow turbulence |
| Transformer-based end-to-end AO | Transformer | Nousiainen, J. et al. (2022) "Toward on-sky adaptive optics control using reinforcement learning," *A&A* 664:A71 | Temporal sequence modeling for predictive wavefront control |

---

## 4. Literature & State of the Art (2024–2025)

1. **Orban de Xivry et al. (2024)** "Physics-informed deep learning for wavefront reconstruction in AO systems," *Optics Letters* — Combines Zernike physics constraints with a convolutional decoder for improved low-light reconstruction.
2. **Swanson et al. (2024)** "Linear quadratic Gaussian control for ELT-scale adaptive optics," *J. Astron. Telesc. Instrum. Syst.* — Demonstrates LQG control beating classical integrators on 40-meter class telescope simulations.
3. **Pou et al. (2024)** "Automatic differentiation for inverse problems in adaptive optics," *Optics Express* 32(3) — Uses autodiff to jointly optimize the reconstructor and the regularization parameters.
4. **Heritier et al. (2025)** "On-sky validation of machine-learning wavefront reconstructors for laser-guide-star AO," *A&A* — First on-sky deployment of a learned reconstructor on a 10-m telescope, outperforming MVM baselines.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/adaptive_optics_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/adaptive_optics_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/adaptive_optics_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/adaptive_optics/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The adaptive optics benchmark correctly models the Hartmann-Shack wavefront sensing forward problem with physically meaningful Zernike-mode parameterization and Fried-parameter-based mismatch. Algorithm routing appropriately spans classical least-squares reconstructors, compressed-sensing Zernike methods, and modern deep/transformer approaches that match the current state of AO reconstruction literature. The benchmark structure with centroid-slope inputs and PSF/wavefront outputs is well-grounded in standard AO sensor geometry.

---
*Comprehensive 6-point check by deep-check pipeline v3*
