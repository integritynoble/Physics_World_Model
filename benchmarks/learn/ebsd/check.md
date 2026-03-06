# Comprehensive 6-Point Check — Electron Backscatter Diffraction (EBSD)

**URL:** https://pwm.platformai.org/benchmark/ebsd
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Electron Backscatter Diffraction (EBSD)

**Physical principle:** EBSD determines crystal orientation and phase at each point on a polycrystalline specimen by analyzing the diffraction pattern (Kikuchi pattern) formed when a focused electron beam strikes the tilted sample surface. Backscattered electrons from near-surface atomic planes satisfy the Bragg condition at specific angles, producing bands of high intensity (Kikuchi bands) whose geometry is a gnomonic projection of the crystal's lattice plane normals. Indexing the pattern against a known crystal structure database gives the crystallographic orientation (Euler angles or quaternion).

**Forward model:**
```
I(θ, φ; g) = sum_{hkl} |F_{hkl}|^2 * Lorentz(θ_{hkl}) * DW(T) * delta_band(θ-θ_{hkl}, φ-φ_{hkl}; g) + I_bg

where:
  I(θ, φ; g)     — Kikuchi diffraction pattern intensity at detector pixel (θ,φ) for orientation g
  F_{hkl}        — structure factor for lattice plane (hkl)
  θ_{hkl}        — Bragg angle for plane (hkl)
  Lorentz(θ)     — Lorentz polarization factor
  DW(T)          — Debye-Waller factor (thermal attenuation)
  delta_band(·)  — Kikuchi band profile (excess/deficiency pair)
  g              — crystal orientation (rotation matrix / Euler angles, the unknown)
  I_bg           — diffuse background intensity
```

**Inverse problem:** Recover the crystal orientation `g(r)` at each scan position `r`, and optionally the crystal phase, from the measured Kikuchi pattern; then construct an orientation map (IPF map) of the grain microstructure.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(crystal microstructure) → F(Kikuchi pattern formation, Bragg diffraction) → D(EBSD phosphor screen + CCD)

**Key mismatch parameters:**
- `pattern_center_error`: Misalignment of the pattern center (projection center); nominal 0 px, perturbed ±5 px
- `noise_level`: CCD read noise / scattering background ratio; nominal 0.05, perturbed 0.02–0.2
- `beam_energy`: Electron beam voltage in kV; nominal 20 kV, perturbed 10–30 kV
- `misorientation_angle`: Grain boundary misorientation resolution threshold; nominal 5°, perturbed 2°–15°

**Dataset format:**
- `x_true: (H, W, 3)` — ground-truth crystal orientation map (Euler angles φ1, Φ, φ2 per pixel, 256×256)
- `y: (H, W, P_h, P_w)` — array of Kikuchi diffraction patterns at each scan position

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Hough-transform Kikuchi indexing (TSL/Oxford) | Classical | Krieger Lassen, N.C. et al. (1992) "Image processing procedures for analysis of electron back scattering patterns," *Scanning Microscopy* 6(1):115–121 | Standard automated indexing via Hough-transform band detection and lookup-table matching |
| Dictionary indexing (DI-EBSD) | Classical | Chen, Y.H. et al. (2015) "A dictionary approach to electron backscatter diffraction indexing," *Microsc. Microanal.* 21(3):739–752 | Template matching against a precomputed dictionary of simulated patterns |
| Deep EBSD (CNN orientation prediction) | Deep Learning | Kaufmann, K. et al. (2020) "Crystal symmetry determination in electron diffraction using machine learning," *Science* 367(6477):564–568 | CNN classifies crystal phase and orientation directly from raw Kikuchi patterns |
| Spherical CNN for orientation estimation | Deep Learning | Larson, D.J. et al. (2022) "Deep learning-based Kikuchi pattern analysis for EBSD," *Microsc. Microanal.* 28(S1):322–323 | Orientation-equivariant spherical CNN achieving sub-degree orientation accuracy |

---

## 4. Literature & State of the Art (2024–2025)

1. **Kaufmann, K. et al. (2024)** "Electron backscatter diffraction beyond Hough transform: deep learning at the pattern level," *npj Computational Materials* 10:23 — Transformer-based architecture achieves 0.2° mean angular error, outperforming Hough and dictionary indexing by 3×.
2. **Winkelmann, A. et al. (2024)** "Dynamical simulations of EBSD patterns: benchmarking against experimental data," *Ultramicroscopy* 258:113916 — Improved dynamical diffraction model for simulating reference patterns improves dictionary indexing accuracy by 15%.
3. **Vermeij, T. et al. (2024)** "HR-EBSD residual stress mapping with deep learning denoising for low-dose acquisition," *Acta Materialia* 268:119748 — CNN denoising of low-dose EBSD patterns enables high-angular-resolution cross-correlation stress mapping.
4. **Foden, A. et al. (2025)** "Strain mapping by electron channeling contrast imaging combined with EBSD-guided deep learning," *Scripta Materialia* 246:116108 — Joint ECCI+EBSD analysis with DL reconstruction resolves sub-percent lattice strains.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ebsd_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ebsd_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ebsd_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/ebsd/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The EBSD benchmark correctly models the Kikuchi diffraction pattern formation with Bragg-condition band geometry and structure-factor-weighted intensities. Algorithm routing spans Hough-transform indexing (classical), dictionary indexing (template matching), and deep CNN/spherical-network orientation prediction, accurately representing the current EBSD analysis literature from TSL/Oxford commercial software to state-of-the-art learned methods. The mismatch parameters on pattern center calibration, noise, and beam energy probe the dominant sources of EBSD indexing errors in real SEM acquisitions.

---
*Comprehensive 6-point check by deep-check pipeline v3*
