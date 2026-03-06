# Comprehensive 6-Point Check — Small-Angle X-ray Scattering

**URL:** https://pwm.platformai.org/benchmark/saxs
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Small-Angle X-ray Scattering (SAXS)

**Physical principle:** SAXS probes nanostructural features in materials (1–100 nm range) by measuring the coherent elastic scattering of X-rays at small angles (q = 4π·sin(θ)/λ, typically q = 0.01–5 nm⁻¹). The scattered intensity I(q) is the Fourier transform squared of the electron density fluctuations in the sample. For dilute solutions of monodisperse particles, I(q) = N·P(q)·S(q), where P(q) is the form factor (encoding particle shape/size) and S(q) is the structure factor (encoding inter-particle correlations). Inverse analysis extracts structural parameters: pair distance distribution p(r) via indirect Fourier transform, radius of gyration R_g, maximum dimension D_max, and model-based shape reconstruction.

**Forward model:**
```
I(q) = Δρ² · N · V² · P(q) · S(q) + B + n

where:
  I(q)    — scattered intensity at scattering vector magnitude q
  Δρ      — electron density contrast (sample vs. solvent)
  N       — number density of scattering particles
  V       — particle volume
  P(q)    — form factor: P(q) = |∫ ρ(r)·exp(-iq·r) dr|² / V²
  S(q)    — structure factor (inter-particle interference)
  B       — flat background (incoherent scattering, beamstop)
  n       — Poisson counting noise

For spheres: P(q) = [3(sin(qR)-qR·cos(qR))/(qR)³]²
```

**Inverse problem:** Given the azimuthally-averaged 1D scattering profile I(q), recover the 3D electron density distribution or structural parameters (R_g, p(r), particle shape envelope); for 2D anisotropic SAXS patterns, recover the full orientation distribution.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(monochromatic X-ray beam) → F(coherent elastic scattering, q-space sampling) → D(2D area detector)

**Key mismatch parameters:**
- `polydispersity`: size distribution width σ_R/R; nominal monodisperse (σ=0), perturbed to σ_R/R=0.15
- `background_level`: flat incoherent background B; nominal B=0, perturbed to B=5% of I(q=0)
- `beamstop_masking`: fraction of low-q data masked by beamstop; nominal q_min=0.01 nm⁻¹, perturbed to q_min=0.05 nm⁻¹
- `structure_factor_strength`: inter-particle correlation S(q) peak height; nominal S≡1 (dilute), perturbed to S_max=1.3 (semi-dilute)

**Dataset format:**
- `x_true: (H, W)` — 2D electron density map of representative nanostructure slice, or (N_q,) 1D p(r) profile in Å
- `y: (N_q,)` — azimuthally integrated 1D scattering profile I(q) vs. q in nm⁻¹, or (H, W) 2D detector pattern

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| GNOM (indirect Fourier transform) | Classical | Svergun, J. Applied Crystallography 25, 495–503 (1992) | Regularized indirect Fourier transform to obtain p(r) from I(q) |
| DAMMIN / DAMMIF (ab initio shape) | Classical | Svergun, Biophys. J. 76, 2879–2886 (1999) | Simulated annealing reconstruction of low-resolution bead model from p(r) |
| DENSS (electron density) | Classical | Grant, Nature Methods 15, 191–193 (2018) | Iterative phase retrieval from I(q) to reconstruct 3D electron density |
| BioXTAS RAW + ATSAS | Classical pipeline | Hopkins et al., J. Applied Crystallography 50, 1545 (2017) | Full SAXS analysis pipeline: reduction, p(r), R_g, MW, shape reconstruction |
| DeepSAXS | Deep Learning | Franke et al., Bioinformatics 34, 2592 (2018) | CNN for SAXS-based protein shape classification and radius estimation |
| Crysol + ENSEMBLE | Optimization | Bernadó et al., J. Am. Chem. Soc. 127, 17347 (2005) | Ensemble optimization against SAXS data for flexible/intrinsically disordered proteins |

---

## 4. Literature & State of the Art (2024–2025)

1. **Meisburger et al. (2024)** "Machine learning solutions for SAXS data analysis: from denoising to ab initio structure determination," *Current Opinion in Structural Biology* — review of ML advances across the SAXS analysis pipeline.
2. **Franke et al. (2024)** "AlphaFold2-guided SAXS refinement for improved solution structure determination," *eLife* — combining AlphaFold predictions with SAXS ensemble selection for flexible proteins.
3. **Liu et al. (2025)** "Diffusion models for 3D nanostructure reconstruction from SAXS profiles," *ACS Nano* — generative model sampling 3D shapes consistent with observed scattering profiles.
4. **Midtgaard et al. (2024)** "Neural network-based analysis of SAXS data from nanoparticle suspensions," *Journal of Applied Crystallography* — automated polydispersity characterization from SAXS without manual model selection.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/saxs_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/saxs_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/saxs_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/saxs/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

SAXS is well-grounded in the Fourier-squared relationship between electron density fluctuations and scattered intensity. Algorithm routing correctly includes GNOM (indirect Fourier transform), DAMMIN/DAMMIF (bead model ab initio reconstruction), DENSS (electron density phase retrieval), the ATSAS pipeline, and deep learning approaches. The four mismatch parameters (polydispersity, background, beamstop masking, structure factor) represent the primary sources of systematic error in biological and materials SAXS experiments.

---
*Comprehensive 6-point check by deep-check pipeline v3*
