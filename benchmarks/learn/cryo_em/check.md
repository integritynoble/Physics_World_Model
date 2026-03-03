# Comprehensive Benchmark QA Check — Cryo-EM

**URL:** https://pwm.platformai.org/benchmark/cryo_em
**HTTP Status:** 200
**Check Date:** 2026-03-03 (comprehensive 6-point review)
**Reviewer:** Local server (automated + manual deep analysis)

---

## Table of Contents

1. [Benchmark Page Errors](#1-benchmark-page-errors)
2. [Local Dataset Inspection](#2-local-dataset-inspection)
3. [Public Dataset Source Assessment](#3-public-dataset-source-assessment)
4. [Algorithm Coverage Assessment](#4-algorithm-coverage-assessment)
5. [Improvement Suggestions](#5-improvement-suggestions)
6. [Action Items](#6-action-items)

---

## 1. Benchmark Page Errors

### Summary

| Severity | Count |
|----------|-------|
| HIGH     | 5     |
| MEDIUM   | 7     |
| LOW      | 4     |

### HIGH Severity

**H1. Mismatch parameters differ between webpage and local data**
- Webpage shows 4 params: defocus_error, astigmatism, beam_tilt, ice_thickness_variation
- Local HDF5 has 4 params: defocus_error_nm, Cs_error_mm, B_factor_error, ice_thickness_error_nm
- "astigmatism" and "beam_tilt" on webpage DO NOT EXIST in the actual data
- "Cs_error_mm" and "B_factor_error" in data are NOT on the webpage
**Fix:** Sync webpage to match HDF5 spec_ranges: defocus_error_nm, Cs_error_mm, B_factor_error, ice_thickness_error_nm.

**H2. Sample count mismatch**
- Webpage: "5 scenes" per tier (15 total)
- Local HDF5: 20/20/20 samples (60 total)
**Fix:** Update webpage to show 20 samples per tier.

**H3. PSNR_norm undefined in scoring formula**
**Fix:** Define normalization method explicitly.

**H4. Rank inversion on Hidden tier unexplained**
- Public/Dev: DiffractFormer ranks 2nd, Deconv ranks 4th
- Hidden: Deconv jumps to 2nd (0.553), DiffractFormer drops to 4th (0.500)
- PnP-DnCNN drops from 3rd to 3rd but with much worse score
**Fix:** Discuss why DiffractFormer fails on severe mismatch while simple Deconv is robust.

**H5. README says "11 real EMDB structures" for public but data has 20 synthetic samples**
- README: `public/ 11 real EMDB structures (TRPV1, ribosome, proteasome, ...)`
- Actual metadata: source not set, scene="streptavidin" for all 20 samples
- These appear to be procedural phantoms, NOT real EMDB structures
**Fix:** Either use real EMDB projections for public, or update README to match reality.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Webpage forward model pipeline "C -> D" is too simplified. Should show: x -> P(projection) -> CTF(f) -> E(B) -> A(ice) -> D(noise) |
| M2 | Spec ranges on webpage show asymmetric ranges (-70 to 230 nm defocus) but local data is symmetric (-200 to 200 nm for public) |
| M3 | Missing references: CryoDRGN (Zhong et al. 2021), CryoSPARC (Punjani et al. 2017), RELION (Scheres 2012) |
| M4 | Cs=2.0 mm is very high -- modern microscopes have Cs-correctors yielding Cs < 0.01 mm. Should note this is deliberately challenging |
| M5 | 256x256 image size is small by cryo-EM standards (typical: 512-1024). May limit applicability |
| M6 | No particle orientation information -- this is a 2D deconvolution benchmark, not full 3D reconstruction |
| M7 | The 4 mismatch knobs in README (Cs_error_mm, B_factor_error) don't appear on webpage (which shows astigmatism, beam_tilt) |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Placeholder links: /benchmark/cryo_em/compete, /benchmark/cryo_em/contribute |
| L2 | No gallery images showing reconstruction comparisons |
| L3 | "300 keV" wavelength stated as 2.51 pm -- should verify (actual: 1.97 pm for relativistic electrons) |
| L4 | SSIM window size and data_range not specified |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Size | Samples |
|------|------|------|---------|
| Public | cryo_em_challenge_public.h5 | 16.9 MB | 20 |
| Dev | cryo_em_challenge_dev.h5 | 14.5 MB | 20 |
| Hidden | cryo_em_challenge_hidden.h5 | 14.5 MB | 20 |

### HDF5 Schema Verification

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (256, 256) | float32 | Ground-truth projected potential [0, 1] |
| `y_ideal` | (256, 256) | float32 | Ideal CTF-convolved image (no mismatch) |
| `y_measured` | (256, 256) | float32 | Measured image (with mismatch + noise) |
| `ctf_ideal` | (256, 256) | float32 | Ideal CTF in Fourier domain |

**All shapes/dtypes consistent across all 60 samples.**

### Value Range Checks

| Check | Public | Dev | Hidden | Status |
|-------|--------|-----|--------|--------|
| x_true in [0,1] | [0.000, 1.000] | [0.000, 1.000] | [0.000, 1.000] | PASS |
| Samples count | 20 | 20 | 20 | PASS |

### Spec Range Nesting Verification

| Parameter | Public | Dev | Hidden | Nesting |
|-----------|--------|-----|--------|---------|
| defocus_error_nm | [-200, 200] | [-500, 500] | [-1000, 1000] | PASS |
| Cs_error_mm | [-0.3, 0.3] | [-0.5, 0.5] | [-1.0, 1.0] | PASS |
| B_factor_error | [-5, 20] | [-8, 50] | [-10, 100] | PASS |
| ice_thickness_error_nm | [-10, 10] | [-15, 20] | [-25, 30] | PASS |

**Spec ranges properly widen: Public < Dev < Hidden.**

### Metadata Spot-Check

| Tier | Scene (sample 0) | Scene (last) |
|------|-------------------|--------------|
| Public | streptavidin | (same for all) |
| Dev | dev_00_reprojected_blend | (unique per sample) |
| Hidden | hidden_00_crowded_field | (unique per sample) |

### Dataset Integrity Assessment: **PASS** (all structural checks pass, but source authenticity needs review)

---

## 3. Public Dataset Source Assessment

### Current Source

**Public tier (20 samples):** Labeled as "streptavidin" in metadata
- No explicit EMDB accession numbers in metadata
- README claims "11 real EMDB structures (TRPV1, ribosome, proteasome, ...)" but actual data has 20 samples all labeled "streptavidin"
- The phantoms appear to be procedurally generated, NOT real EMDB data

**Dev tier (20 samples):** Procedural -- "reprojected_blend" multi-component particles
**Hidden tier (20 samples):** Adversarial -- "crowded_field", ice contamination, conformational heterogeneity

### Assessment of Source Quality

| Criterion | Rating | Notes |
|-----------|--------|-------|
| **Public: Well-known dataset?** | NEEDS IMPROVEMENT | Claims EMDB origin but data appears synthetic. Should use actual EMDB maps. |
| **Public: Accepted by professors?** | NEEDS IMPROVEMENT | Streptavidin is well-known but 20 samples of same structure lack diversity |
| **Dev: Protected from lookup?** | GOOD | Procedural blending prevents reverse-engineering |
| **Hidden: Protected from lookup?** | EXCELLENT | Adversarial scenarios (crowded, aggregated, ice) are unique |

### Recommendations for Public Tier

1. **Use real EMDB structures** (highest credibility):
   - EMD-3061: beta-galactosidase at 2.2 A (>4,000 citations)
   - EMD-2660: TRPV1 at 3.4 A (Nobel Prize work, Cheng/Julius)
   - EMD-0190: 80S ribosome at 2.7 A (~2,000 citations)
   - EMD-3228: 20S proteasome at 2.8 A (~1,000 citations)
   - EMD-9718: ApoFerritin at 1.54 A (highest resolution benchmark)

2. **Standard cryo-EM benchmarks:**
   - **EMPIAR** (empiar.org): Raw micrograph archives, gold standard
   - EMPIAR-10028: 80S ribosome micrographs (~500 citations)
   - EMPIAR-10025: T20S proteasome micrographs
   - EMPIAR-10061: beta-galactosidase micrographs

3. **Use projected potential maps from EMDB**
   - Download 3D map from EMDB, project along random orientations
   - Apply CTF and noise to get realistic 2D micrograph patches
   - This is the standard approach in cryo-EM benchmarking

---

## 4. Algorithm Coverage Assessment

### Currently Tested (Webpage Leaderboard)

| # | Algorithm | Type | Notes |
|---|-----------|------|-------|
| 1 | ResNet-Calib + gradient | CNN + calibration | Top performer, custom |
| 2 | DiffractFormer + gradient | Transformer | Fails on severe mismatch |
| 3 | PnP-DnCNN + gradient | Plug-and-play denoiser | Middle performer |
| 4 | Deconv + gradient | Classical deconvolution | Robust baseline |

### PWM Solver Registry

| Solver | Function | Status |
|--------|----------|--------|
| Adjoint | `adjoint.run_adjoint` | Registered (traditional_cpu) |
| PnP-ADMM | `pnp_admm.pnp_admm_recon` | Registered (best_quality) |

### Missing Famous/Recent Algorithms (MUST ADD)

| Priority | Algorithm | Year | Why Important |
|----------|-----------|------|---------------|
| **CRITICAL** | Wiener filter (CTF correction) | Classical | THE standard CTF correction, every cryo-EM paper uses it |
| **CRITICAL** | CryoSPARC 2D classification | 2017 | Industry standard, Punjani et al., Nature Methods (~5,000 citations) |
| **CRITICAL** | RELION CTF refinement | 2012 | Gold standard in cryo-EM, Scheres, J. Struct. Biol. (~10,000 citations) |
| **HIGH** | CryoDRGN | 2021 | Deep generative model for heterogeneity, Zhong et al., Nature Methods |
| **HIGH** | CryoFIRE | 2023 | Amortized inference for pose + conformational states |
| **HIGH** | CryoEMNet | 2025 | Symmetry-aware DL, outperforms cryoSPARC and RELION |
| **MEDIUM** | CTFFIND4 + phase-flip | 2015 | Standard CTF estimation, Rohou & Grigorieff |
| **MEDIUM** | Iterative CTF refinement (Bayesian) | 2020 | Per-particle CTF refinement in RELION 3.1 |
| **MEDIUM** | Score-based diffusion for cryo-EM | 2024 | Generative prior approach |
| **LOW** | Total variation deconvolution | Classical | Simple regularized baseline |
| **LOW** | CryoAI | 2023 | Amortized inference for ab initio |

### Algorithm Gap Analysis

| Category | Have | Missing | Gap |
|----------|------|---------|-----|
| Classical CTF correction | Deconv | Wiener filter, phase-flip | 2 |
| Bayesian refinement | -- | RELION CTF refinement | 1 |
| Iterative optimization | -- | CryoSPARC 2D class, CTF refinement | 2 |
| CNN-based | ResNet-Calib | CryoEMNet | 1 |
| Plug-and-play | PnP-DnCNN | -- | 0 |
| Generative/VAE | -- | CryoDRGN, CryoFIRE | 2 |
| Diffusion | -- | Score-based cryo-EM | 1 |
| Transformer | DiffractFormer | CryoAI | 1 |

**Total gap: 10 algorithms missing across 7 categories**

---

## 5. Improvement Suggestions

### 5.1 Dataset Improvements

1. **Use real EMDB projected potentials for public tier**
   - Current "streptavidin" synthetic data lacks credibility
   - Download 5-10 EMDB maps, project along orientations, apply CTF
   - Suggested: EMD-3061, EMD-2660, EMD-0190, EMD-3228, EMD-9718

2. **Add particle diversity to public tier**
   - All 20 samples appear to be the same protein (streptavidin)
   - Should have 5-10 different proteins with varying size, symmetry, shape

3. **Add 3D reconstruction benchmark track**
   - Current benchmark is 2D deconvolution only
   - Full cryo-EM is a 3D reconstruction problem (2D projections -> 3D volume)
   - Would need: multiple projections per particle, orientation estimation

4. **Increase image size to 512x512**
   - 256x256 is small by cryo-EM standards
   - Most real micrograph patches are 256-1024 pixels
   - Larger images test algorithm scalability

5. **Add astigmatic CTF to mismatch**
   - Real microscopes have astigmatism (different defocus along x/y axes)
   - This is a 5th mismatch knob that would increase realism
   - Webpage mentions astigmatism but data doesn't have it

6. **Add beam-induced motion blurring**
   - Real cryo-EM suffers from beam-induced sample movement
   - This is a major source of resolution loss in practice

### 5.2 Algorithm Testing Improvements

7. **Add Wiener filter baseline immediately**
   - This is the most fundamental CTF correction method
   - H_wiener(f) = CTF*(f) / (|CTF(f)|^2 + SNR^-1)
   - Trivial to implement, essential baseline

8. **Add CryoSPARC-style 2D classification**
   - Industry standard, would validate benchmark against real-world tools
   - At minimum, implement the CTF correction + regularized inversion

9. **Add CryoDRGN or CryoFIRE**
   - State-of-the-art deep generative approaches
   - Would demonstrate the benchmark's utility for cutting-edge methods

10. **Run all algorithms on all 3 tiers consistently**
    - Current: 4 algorithms x 3 tiers
    - Target: 8-10 algorithms x 3 tiers

### 5.3 Infrastructure Improvements

11. **Sync webpage with local data**
    - Fix mismatch parameter names (Cs_error not astigmatism)
    - Fix sample counts (20 not 5 per tier)
    - Add actual spec ranges from HDF5

12. **Verify electron wavelength**
    - README states 2.51 pm for 300 keV electrons
    - Relativistic calculation: lambda = h/sqrt(2*m*eV*(1+eV/(2*m*c^2))) ~ 1.97 pm
    - 2.51 pm corresponds to ~200 keV, not 300 keV

13. **Add FSC (Fourier Shell Correlation) metric**
    - Standard resolution metric in cryo-EM
    - More meaningful than PSNR for structural biology applications

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| CRITICAL | Fix webpage mismatch params (Cs_error, B_factor, not astigmatism/beam_tilt) | Main server | TODO |
| CRITICAL | Fix webpage sample counts (20 per tier, not 5) | Main server | TODO |
| CRITICAL | Replace public synthetic data with real EMDB projected potentials | Dataset team | TODO |
| CRITICAL | Add Wiener filter to algorithm suite | Algorithm team | TODO |
| CRITICAL | Verify electron wavelength (2.51 pm may be wrong for 300 keV) | Physics team | TODO |
| HIGH | Add protein diversity to public tier (5-10 different EMDB structures) | Dataset team | TODO |
| HIGH | Add CryoSPARC/RELION-style CTF refinement | Algorithm team | TODO |
| HIGH | Define PSNR_norm formula explicitly | Main server | TODO |
| HIGH | Explain rank inversion on Hidden tier (DiffractFormer failure) | Main server | TODO |
| MEDIUM | Add CryoDRGN or CryoFIRE generative approach | Algorithm team | TODO |
| MEDIUM | Add astigmatic CTF to mismatch model | Dataset team | TODO |
| MEDIUM | Add FSC metric alongside PSNR/SSIM | Metrics team | TODO |
| MEDIUM | Add missing references (CryoSPARC, RELION, CryoDRGN) | Main server | TODO |
| LOW | Consider 3D reconstruction track | Dataset team | TODO |
| LOW | Increase image size to 512x512 | Dataset team | TODO |
| LOW | Add beam-induced motion blurring | Dataset team | TODO |

---

## Appendix: Key References

- Scheres, S.H.W. "RELION: Implementation of a Bayesian approach to cryo-EM structure determination." J. Struct. Biol. 180.3 (2012): 519-530.
- Rohou, A., Grigorieff, N. "CTFFIND4: Fast and accurate defocus estimation from electron micrographs." J. Struct. Biol. 192.2 (2015): 216-221.
- Punjani, A., et al. "cryoSPARC: algorithms for rapid unsupervised cryo-EM structure determination." Nature Methods 14.3 (2017): 290-296.
- Frank, J. "Three-Dimensional Electron Microscopy of Macromolecular Assemblies." Oxford Univ. Press (2006).
- Zhong, E.D., et al. "CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks." Nature Methods 18.2 (2021): 176-185.
- Levy, A., et al. "CryoFIRE: Amortized Inference for Ab-Initio Cryo-EM Reconstruction." NeurIPS (2023).
- CryoEMNet. "Symmetry-aware molecular reconstruction through deep learning." Scientific Reports 15 (2025).

---

*Comprehensive 6-point review on 2026-03-03. Covers: page errors, local dataset verification, source quality, algorithm coverage, improvement suggestions, and action items.*
