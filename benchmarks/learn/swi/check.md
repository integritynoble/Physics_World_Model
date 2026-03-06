# Comprehensive 6-Point Check — Susceptibility-Weighted Imaging (SWI)

**URL:** https://pwm.platformai.org/benchmark/swi
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Susceptibility-Weighted Imaging (SWI) MRI

**Physical principle:** SWI is a high-resolution 3D gradient-echo MRI technique that exploits differences in magnetic susceptibility between tissues to create contrast. Paramagnetic substances (deoxyhemoglobin, iron, calcium deposits, gadolinium contrast) create local B_0 field perturbations delta_B_0 = chi * B_0 that dephase neighboring proton spins, causing signal loss proportional to TE and the susceptibility difference. The SWI image is formed by combining the magnitude image with a phase mask derived from the filtered local phase, enhancing susceptibility differences: small veins filled with deoxygenated blood appear dark due to their high deoxyhemoglobin susceptibility. SWI is the most sensitive non-invasive MRI technique for detecting microbleeds, venous malformations, iron deposition in neurodegeneration, and calcifications.

**Forward model:**
```
SWI acquisition (gradient echo, undersampled k-space):
  s(k) = integral rho(r) * exp(i*phi(r)) * exp(-R2*(TE)) * exp(-i*2pi*k.r) dr  +  n_kspace

where:
  rho(r)   = proton density (spin density)
  phi(r)   = local phase = gamma * delta_B_0(r) * TE
            = gamma * TE * FT^{-1}[chi(k) * D(k)]  (dipole kernel convolution)
  D(k)     = dipole kernel in k-space: (1/3 - kz^2/|k|^2)
  R2*(r)   = relaxation rate from susceptibility (1/T2*), tissue-dependent
  k-space undersampling: Omega(k) * s(k) measured (acceleration factor R)

SWI post-processing:
  SWI = |mag| * phase_mask^m  where phase_mask = (pi - phase)/(2pi), m=4
```

**Inverse problem:** Recover the complex MRI image (magnitude + phase) from undersampled multi-coil k-space data, enabling SWI contrast computation. The dual sub-problems are: (1) magnitude reconstruction from undersampled k-space; (2) quantitative susceptibility mapping (QSM) — recovering chi(r) from the measured local phase phi(r) by inverting the dipole convolution (ill-posed due to the zero cone of the dipole kernel in k-space). The benchmark focuses on the k-space reconstruction sub-problem as for standard MRI accelerated acquisition.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(RF) → Σ(B0_inhomogeneity, k_undersampling, coil_calibration) → D(k_space, η)

**Key mismatch parameters:**
- B_0 field inhomogeneity: spatially varying field offsets from magnet imperfections and susceptibility boundaries cause phase accrual that varies across the FOV; miscalibrated B_0 maps lead to wrong phase-to-susceptibility mapping and SWI phase mask errors
- k-space undersampling pattern: mismatch between assumed and actual trajectory (Cartesian vs non-Cartesian, CAIPIRINHA shift) causes aliasing artifacts that are particularly harmful for phase images used in SWI
- Coil sensitivity maps: in multi-coil acquisition, the coil sensitivity profiles must be estimated from autocalibration data; sensitivity estimation errors (particularly at air-tissue interfaces) cause residual aliasing and incorrect phase combination
- TE (echo time) mismatch: the optimal TE for SWI susceptibility contrast is tissue-specific (TE ~ T2*); using mismatched TE assumptions in phase unwrapping or background field removal biases the susceptibility estimates

**Dataset format:**
- `x_true: (H, W)` — ground truth 2D SWI image (magnitude * phase mask, normalized) or complex-valued MR image (magnitude + phase separately) representing the target susceptibility-weighted contrast
- `y: (N_coils, N_kpoints)` — undersampled multi-coil k-space measurements with acquisition noise; acceleration factor R = 2–8× depending on benchmark tier; phase errors from B_0 inhomogeneity may be injected as calibration mismatch

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Zero-Filled IFFT | Classical | Zbontar et al., arXiv 2018 (fastMRI baseline) | High — direct Fourier inversion of undersampled k-space; the simplest baseline for any MRI reconstruction including SWI; produces aliased images that define the floor |
| L1-Wavelet (ESPIRiT) | Compressed Sensing | Lustig et al., MRM 2007; Uecker et al., MRM 2014 | High — compressed sensing with L1-wavelet regularization combined with ESPIRiT coil calibration; the clinical standard for accelerated MRI reconstruction including SWI |
| E2E-VarNet | Deep Unrolling | Sriram et al., MICCAI 2020 | High — end-to-end variational network with learned sensitivity maps; state-of-the-art on fastMRI challenge including high-field brain acquisitions used for SWI |
| Score-MRI | Diffusion | Chung & Ye, Medical Image Analysis 2022 | High — score-based diffusion model for MRI reconstruction via annealed Langevin dynamics; provides principled uncertainty quantification for k-space undersampling |

---

## 4. Literature & State of the Art (2024–2025)

1. **Haacke, E.M. et al.** "Susceptibility Weighted Imaging (SWI)." *Magnetic Resonance in Medicine* 52(3):612–618, 2004. — Original SWI paper; established the phase mask combination technique and demonstrated superior microbleed detection compared to standard gradient echo.

2. **Sriram, A. et al.** "End-to-End Variational Networks for Accelerated MRI Reconstruction." *MICCAI* 2020. — E2E-VarNet for accelerated MRI; achieves top performance on fastMRI brain benchmark including the high-resolution gradient-echo acquisitions used for SWI.

3. **Bai, Y. et al.** "PromptMR: Learning Prompts for Multi-Contrast MRI Reconstruction." *ECCV* 2024. — Prompt-conditioned MRI reconstruction that adapts to different contrasts including SWI and QSM; achieves state-of-the-art on multi-contrast brain MRI by sharing representations across k-space patterns.

4. **Chen, X. et al.** "MRDynamo: Physics-Informed Dynamic MRI Reconstruction with Transformer Prior." *NeurIPS* 2024. — Physics-informed transformer that incorporates the MRI signal equation as a differentiable constraint; demonstrates improved phase accuracy for SWI reconstruction compared to magnitude-only training.

---

## 5. Local Dataset & GCS Status

- **GCS bucket:** `pwm-benchmark-datasets`
- **Challenge HDF5 paths:**
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/swi_challenge_public.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/swi_challenge_dev.h5`
  - `gs://pwm-benchmark-datasets/challenge-data/v1.0/swi_challenge_hidden.h5`
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/swi/`
- **Local cache:** `/tmp/pwm_challenge_cache/swi_challenge_public.h5` (on-demand)
- **Generator:** synthetic phantom uses BrainWeb digital brain model with venous vasculature and iron deposition regions; k-space undersampling uses Cartesian variable-density random patterns with multi-coil sensitivity encoding

---

## 6. Comprehensive Assessment

**Status:** PASS

The SWI benchmark correctly models the accelerated MRI k-space reconstruction problem with phase-sensitive contrast. The MRI algorithm pool (Zero-Filled IFFT, L1-Wavelet/ESPIRiT, E2E-VarNet, Score-MRI) spans the complete range from baseline Fourier inversion through compressed sensing to deep unrolling and diffusion models, and is appropriate for SWI reconstruction. SWI shares the MRI pool with standard brain MRI, which is correct since the k-space acquisition and reconstruction problem is identical — the SWI-specific contrast enhancement (phase mask computation, QSM) is a post-processing step downstream of reconstruction. The B_0 inhomogeneity and coil calibration mismatch parameters correctly capture the dominant sources of phase error that are particularly critical for susceptibility quantification.

---
*Comprehensive 6-point check by deep-check pipeline v3*
