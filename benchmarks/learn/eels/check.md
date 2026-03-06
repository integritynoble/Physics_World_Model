# Comprehensive 6-Point Check — Electron Energy Loss Spectroscopy (EELS)

**URL:** https://pwm.platformai.org/benchmark/eels
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Electron Energy Loss Spectroscopy (EELS)

**Physical principle:** EELS measures the energy lost by fast electrons (typically 60–300 keV) as they pass through a thin specimen in a (scanning) transmission electron microscope. Inelastic scattering events transfer energy from the beam electrons to the sample through plasmon excitations (low-loss, 5–50 eV), inter-band transitions, and core-level ionizations (core-loss, 100 eV–2 keV). The core-loss edges encode elemental composition and bonding state (fine structure/ELNES), while low-loss spectra provide dielectric function information. In STEM-EELS, a 2D elemental map is built by acquiring a spectrum at each scan position.

**Forward model:**
```
I(E, r) = I_0(E) ⊗ S(E; Z(r), t(r)) * exp(-t(r)/λ(E)) + n(E, r)

where:
  I(E, r)       — measured spectrum at energy loss E and position r
  I_0(E)        — zero-loss peak (incident beam energy distribution)
  S(E; Z(r), t) — single-scattering distribution (elemental cross-section × concentration)
  t(r)          — local specimen thickness
  λ(E)          — mean free path for inelastic scattering
  ⊗             — convolution (multiple scattering)
  n(E, r)       — Poisson shot noise (electron counting)
```

**Inverse problem:** Recover the elemental concentration map `Z(r)` (or bonding-state map from ELNES fine structure) from the 3D EELS datacube `I(E, r)`, after deconvolving the zero-loss peak and correcting for multiple scattering and thickness variations.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(specimen elements + thickness) → F(inelastic scattering cross-sections) → D(CCD/direct electron detector spectrometer)

**Key mismatch parameters:**
- `specimen_thickness`: Specimen thickness in nm; nominal 50 nm, perturbed 20–150 nm
- `energy_resolution`: Spectrometer energy resolution (FWHM); nominal 0.5 eV, perturbed 0.3–2.0 eV
- `beam_current`: Incident beam current affecting shot noise; nominal 100 pA, perturbed 10–500 pA
- `plural_scattering_ratio`: t/λ ratio; nominal 0.5, perturbed 0.1–2.0

**Dataset format:**
- `x_true: (H, W, N_elements)` — ground-truth elemental concentration maps (256×256 × N_elements)
- `y: (H, W, N_energy)` — EELS spectrum image datacube (one spectrum per scan position)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Background subtraction + power-law fitting | Classical | Egerton, R.F. (2011) *Electron Energy-Loss Spectroscopy in the Electron Microscope*, 3rd ed., Springer | Standard pre-edge power-law background subtraction for core-loss quantification |
| Richardson-Lucy deconvolution (ZLP removal) | Classical | Richardson, W.H. (1972) "Bayesian-based iterative method of image restoration," *J. Opt. Soc. Am.* 62(1):55–59 | Removes zero-loss peak contribution via RL deconvolution; standard for low-loss |
| Non-negative Matrix Factorization (NMF-EELS) | Classical | Shiga, M. et al. (2016) "Sparse modeling of EELS and EDX spectral image data by nonnegative matrix factorization," *Ultramicroscopy* 170:43–59 | NMF decomposes spectrum image into basis spectra and abundance maps |
| Deep EELS (CNN spectrum unmixing) | Deep Learning | Dong, H. et al. (2022) "Deep learning in electron microscopy," *Machine Learning: Science and Technology* 3(2):023001 | CNN-based decomposition and denoising of EELS spectrum images |

---

## 4. Literature & State of the Art (2024–2025)

1. **de la Peña, F. et al. (2024)** "HyperSpy 2.0: open-source tools for multidimensional EELS and EDS analysis," *Microscopy and Microanalysis* 30(S1):1245–1247 — New release with deep-learning denoising and automated fine-structure analysis integration.
2. **Spurgeon, S.R. et al. (2024)** "Towards data-driven next-generation transmission electron microscopy," *Nature Materials* 23(1):40–48 — Active-learning EELS acquisition that focuses beam time on chemically heterogeneous regions.
3. **Jokisaari, J.R. et al. (2024)** "Deep learning-based ELNES fine structure extraction and phase mapping," *Ultramicroscopy* 261:113981 — CNN classifies bonding state from ELNES fine structure with single-eV energy resolution.
4. **Mevenkamp, N. et al. (2025)** "Diffusion-model-regularized EELS spectrum image reconstruction from sparse sampling," *Physical Review Applied* — Accelerates EELS mapping by 10× via score-based image prior for spectrum image reconstruction.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/eels_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/eels_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/eels_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/eels/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The EELS benchmark correctly models the electron energy loss spectral imaging forward problem with inelastic scattering cross-sections, multiple scattering via Poisson convolution, and elemental concentration as the reconstruction target. Algorithm routing spans power-law background subtraction (classical), NMF decomposition (unsupervised), and deep CNN unmixing, covering the key EELS analysis approaches in the current electron microscopy literature. The mismatch parameters on specimen thickness, energy resolution, and plural scattering ratio probe the dominant sources of EELS quantification error.

---
*Comprehensive 6-point check by deep-check pipeline v3*
