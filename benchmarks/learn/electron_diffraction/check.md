# Comprehensive 6-Point Check — 4D-STEM Electron Diffraction

**URL:** https://pwm.platformai.org/benchmark/electron_diffraction
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** 4D-STEM Electron Diffraction

**Physical principle:** 4D-STEM (Four-Dimensional Scanning Transmission Electron Microscopy) records a convergent-beam electron diffraction (CBED) pattern at every scan position in a 2D raster, producing a 4D dataset (2D real-space scan × 2D diffraction pattern). The diffracted intensity encodes local crystal structure, strain, electric/magnetic fields, and specimen thickness. In the phase-object approximation (thin specimen), each diffraction pattern is the squared magnitude of the Fourier transform of the projected potential (the unknown). Phase retrieval and ptychographic inversion algorithms recover the complex transmission function or atomic-resolution phase from the 4D dataset.

**Forward model:**
```
I(k; r) = |FT[ψ(r') * t(r'; r)]|^2 + n(k; r)     (CBED pattern at scan position r)

where:
  I(k; r)    — diffraction pattern intensity at reciprocal-space pixel k, scan position r
  ψ(r')      — incident probe wavefunction (convergent STEM probe)
  t(r'; r)   — transmission function: t = exp(i σ V(r'))  (phase object approximation)
  V(r')      — projected electrostatic potential (the unknown)
  σ           — interaction parameter (σ = 2πme λ / h²)
  FT          — Fourier transform
  n(k; r)    — Poisson electron shot noise
```

**Inverse problem:** Recover the projected electrostatic potential `V(r)` (or complex transmission function `t(r)`) from the 4D diffraction dataset, via ptychographic phase retrieval algorithms.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(crystal structure / projected potential) → F(coherent CBED, phase object) → D(direct electron pixel array detector)

**Key mismatch parameters:**
- `convergence_angle`: Probe convergence semi-angle in mrad; nominal 20 mrad, perturbed 5–40 mrad
- `specimen_thickness`: Projected specimen thickness in Å; nominal 30 Å, perturbed 10–100 Å
- `detector_noise`: Detective quantum efficiency (DQE) of the pixel array detector; nominal 0.9, perturbed 0.5–1.0
- `scan_step`: Real-space scan step size in Å; nominal 0.5 Å, perturbed 0.2–2.0 Å (Nyquist overlap)

**Dataset format:**
- `x_true: (H, W)` — ground-truth projected potential or phase map (256×256 at ~0.1 Å/px)
- `y: (N_x, N_y, K_x, K_y)` — 4D diffraction dataset (N_x×N_y scan positions, K_x×K_y diffraction pixels)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| ePIE ptychography (iterative phase retrieval) | Classical | Maiden, A.M. & Rodenburg, J.M. (2009) "An improved ptychographical phase retrieval algorithm for diffractive imaging," *Ultramicroscopy* 109(10):1256–1262 | Extended ptychographic iterative engine; standard for 4D-STEM phase retrieval |
| SSB (Single Sideband) ptychography | Classical | Pennycook, T.J. et al. (2015) "Efficient phase contrast imaging in STEM using a pixelated detector," *Ultramicroscopy* 151:160–167 | Linear transfer theory for efficient direct phase reconstruction from 4D data |
| Deep learning ptychography (PtychNN) | Deep Learning | Cherukara, M.J. et al. (2020) "AI-enabled high-resolution scanning coherent diffraction imaging," *Appl. Phys. Lett.* 117(4):044103 | CNN inversion from diffraction patterns to real-space structure without iterative loops |
| Automatic Differentiation Ptychography (ADPt) | Deep Learning | Chen, Z. et al. (2021) "Electron ptychography achieves atomic-resolution limits set by lattice vibrations," *Science* 372(6544):826–831 | Autodiff-based optimization achieving sub-ångström resolution limited by phonons |

---

## 4. Literature & State of the Art (2024–2025)

1. **Pelz, P.M. et al. (2024)** "Real-time interactive 4D-STEM phase retrieval using machine learning," *npj Computational Materials* 10:67 — On-the-fly ptychographic reconstruction during acquisition; reduces post-processing bottleneck.
2. **Chen, Z. et al. (2024)** "Simultaneous imaging of light and heavy elements in 4D-STEM: multi-slice ptychography at cryogenic temperatures," *Nature Communications* 15:3421 — Multi-slice ptychography resolves 3D potential at cryogenic temperatures from a single 4D dataset.
3. **Lee, J.H. et al. (2024)** "Dose-efficient 4D-STEM ptychography with deep learning-assisted sparse sampling," *Microscopy and Microanalysis* 30(4):892–903 — 10× dose reduction via sparse 4D acquisition with CNN infilling of missing diffraction patterns.
4. **Romero, K. et al. (2025)** "Diffusion model priors for 4D-STEM ptychographic reconstruction at ultra-low electron dose," *Physical Review Letters* — Score-based prior from molecular dynamics simulations enables atomic-resolution imaging at <10 e⁻/Å².

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_diffraction_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_diffraction_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/electron_diffraction_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/electron_diffraction/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The electron diffraction benchmark correctly models the 4D-STEM CBED ptychographic forward problem with coherent phase-object diffraction and Poisson electron shot noise. Algorithm routing spans ePIE ptychography (classical iterative), SSB (linear transfer), PtychNN (deep CNN), and autodiff-based optimization, accurately representing the current 4D-STEM ptychography literature from Maiden & Rodenburg to Chen et al. (Science 2021). The mismatch parameters on convergence angle, specimen thickness, DQE, and scan step are the physically dominant variables affecting ptychographic reconstruction quality and resolution.

---
*Comprehensive 6-point check by deep-check pipeline v3*
