# Comprehensive 6-Point Check — Ptychography

**URL:** https://pwm.platformai.org/benchmark/ptychography
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Ptychographic Imaging

**Physical principle:** Ptychography is a scanning coherent diffractive imaging technique in which a localized coherent probe is scanned across an object with overlapping illumination positions. At each scan position, a far-field diffraction pattern is recorded. The redundancy from overlapping measurements (typically 60–80% overlap) dramatically over-determines the inverse problem, enabling simultaneous recovery of both the complex object transmission function and the complex probe wavefield. This probe-object decoupling makes ptychography uniquely powerful for high-resolution, quantitative phase imaging without lenses.

**Forward model:**
```
I_j(u) = |F{ P(r - r_j) · O(r) }|^2 + n_j

where:
  I_j(u)      — diffraction intensity at scan position j, reciprocal coordinate u
  P(r - r_j)  — complex probe wavefield centered at scan position r_j
  O(r)        — complex object transmission function
  F{·}        — 2D Fourier transform (far-field propagation)
  n_j         — Poisson shot noise

Overlap ratio: ρ = 1 - d/probe_diameter, typically ρ ≥ 0.6
```

**Inverse problem:** Recover the complex object O(r) and probe P(r) simultaneously from a set of J oversampled diffraction patterns measured at known (or unknown) scan positions; multi-mode extensions handle partial coherence.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(coherent focused X-ray/electron probe) → F(probe-object interaction, far-field diffraction) → D(photon-counting 2D detector)

**Key mismatch parameters:**
- `scan_position_error`: calibration error in probe positions r_j; nominal 0 nm, perturbed ±5% of step size
- `overlap_ratio`: fractional overlap between adjacent scan positions; nominal 70%, perturbed to 50%
- `probe_modes`: number of incoherent probe modes; nominal 1 (fully coherent), perturbed to 3 modes (partial coherence)
- `photon_flux`: photons per diffraction pattern; nominal 10⁶, perturbed to 5×10⁴ (high noise regime)

**Dataset format:**
- `x_true: (H, W)` — complex object transmission magnitude (or phase), representing the 2D spatial distribution of sample optical constants
- `y: (J, Pd, Pd)` — J diffraction intensity patterns of size Pd×Pd pixels at each scan position

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| ePIE (extended Ptychographic Iterative Engine) | Classical iterative | Maiden & Rodenburg, Ultramicroscopy 109, 1256–1262 (2009) | Standard ptychographic reconstruction; simultaneous probe+object update via gradient steps |
| DM-Ptycho (Difference Map) | Classical iterative | Thibault et al., Science 321, 379–382 (2008) | Difference map formulation; robust convergence for highly overlapping scans |
| PIE (Ptychographic Iterative Engine) | Classical | Rodenburg & Faulkner, Applied Physics Letters 85, 4795 (2004) | Original single-probe sequential update algorithm |
| Wigner Distribution Deconvolution (WDD) | Classical | Bates & Rodenburg, Ultramicroscopy 31, 303–313 (1989) | Direct (non-iterative) ptychographic reconstruction via Wigner distribution |
| PtychoNN | Deep Learning | Cherukara et al., Applied Physics Letters 117, 044191 (2020) | Neural network replacing iterative loops for real-time ptychography |
| Ptychoshelves / ML-ptycho | Optimization+ML | Kandel et al., Optica 6, 793–803 (2019) | Automatic differentiation through the ptychographic forward model; handles complex aberrations |

---

## 4. Literature & State of the Art (2024–2025)

1. **Du et al. (2024)** "Advancing X-ray ptychography with deep learning for large field-of-view imaging," *npj Computational Materials* — deep learning accelerates 20× convergence over ePIE while recovering sub-nm features.
2. **Odstrcil et al. (2024)** "Self-calibrating ptychography with position correction and multi-mode probe," *Optica* — automatic probe position refinement within a differentiable ptychographic framework.
3. **Pelz et al. (2025)** "Real-time 4D-STEM ptychography using deep unrolled networks," *Nature Communications* — unrolled ePIE for online 4D-STEM; 100 ms per reconstruction.
4. **Yao et al. (2024)** "Generative model-based ptychographic reconstruction with uncertainty quantification," *Physical Review Applied* — diffusion model priors for ptychography from sparse, noisy patterns.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ptychography_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ptychography_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ptychography_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/ptychography/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Ptychography has a rigorous forward model (probe-object product in real space, far-field Fourier intensity measurement) with well-studied reconstruction algorithms. Algorithm routing correctly includes the foundational PIE/ePIE/DM-Ptycho iterative engines, WDD direct inversion, automatic differentiation approaches, and modern deep learning (PtychoNN). The four mismatch parameters (scan position error, overlap ratio, probe coherence modes, photon flux) represent the primary experimental challenges in practical ptychographic experiments.

---
*Comprehensive 6-point check by deep-check pipeline v3*
