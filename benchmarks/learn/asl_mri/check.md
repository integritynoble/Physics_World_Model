# Comprehensive 6-Point Check — Arterial Spin Labeling (ASL) MRI

**URL:** https://pwm.platformai.org/benchmark/asl_mri
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Arterial Spin Labeling (ASL) MRI

**Physical principle:** ASL MRI measures cerebral blood flow (CBF) non-invasively by magnetically labelling water protons in arterial blood proximal to the imaging slice. A labelling RF pulse inverts the magnetisation of inflowing blood; after a post-labelling delay (PLD), a control-label image pair is acquired and subtracted to reveal the perfusion signal. The subtracted signal is proportional to CBF modulated by T1 relaxation of labelled blood. The k-space undersampling challenge is the same as standard MRI, but the perfusion contrast introduces ASL-specific kinetic model parameters.

**Forward model:**
```
ASL perfusion signal:
  ΔM(t) = 2 M0 f / λ * α * T1_blood * exp(-t/T1_blood) * [arrival function]

k-space acquisition (undersampled):
  y = U_Ω F C x + n

where:
  x ∈ C^{H×W}    — perfusion-weighted image (ground truth)
  F               — 2D Fourier transform
  C               — coil sensitivity maps
  U_Ω             — undersampling mask
  α               — labelling efficiency (nominal 0.85)
  f               — CBF (mL/100g/min)
  λ               — blood-brain partition coefficient
  T1_blood        — T1 of arterial blood (~1.65 s at 3T)
```

**Inverse problem:** Recover the ASL perfusion image x from under-sampled k-space measurements y, accounting for the kinetic model parameters (labelling efficiency, transit delay, T1_blood) that are imperfectly calibrated.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** M(RF label) → F(k-space) → S(Ω) → D(ADC)

**Key mismatch parameters:**
- `labeling_efficiency` (l_e): fraction of blood magnetisation inverted; nominal 0.85, perturbed 0.87
- `transit_delay` (t_d): arterial transit time from label to imaging plane; nominal 1.5 s, perturbed 1.8 s
- `t1_blood_error` (t_b): T1 of blood estimation error; nominal 0.0, perturbed 2.0 (relative %)

**Dataset format:**
- `x_true: (H, W)` — perfusion-weighted ASL image (ground truth after label-control subtraction)
- `y: (|Ω|, N_c)` — under-sampled multi-coil k-space of the ASL image
- `H_ideal: (|Ω|, N_c, H, W)` — ideal Fourier undersampling + coil sensitivity encoding

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Zero-Filled IFFT | Classical | fastMRI baseline (Zbontar et al. 2018) | Direct inverse FFT; establishes aliasing baseline for accelerated ASL |
| L1-Wavelet (ESPIRiT) | Compressed Sensing | Lustig et al., MRM 2007; Uecker et al., MRM 2014 | Gold-standard CS reconstruction; appropriate for ASL k-space undersampling |
| E2E-VarNet | Deep Unrolling | Sriram et al., MICCAI 2020 | End-to-end variational network; best fastMRI knee/brain results |
| PromptMR | Deep Unrolling | Xin et al., ECCV 2024 | Prompt-based generalizable MRI reconstruction; state-of-art fastMRI 2023 |
| ReconFormer | Transformer | Guo et al., IEEE TMI 2024 | Recurrent Transformer for accelerated MRI reconstruction |
| Score-MRI | Diffusion | Chung & Ye, Med. Image Anal. 2022 | Score-based diffusion posterior sampling for MRI |

---

## 4. Literature & State of the Art (2024–2025)

1. **Accelerated ASL with deep learning** (Tian et al., MRM 2023 / extended 2024): U-Net and VarNet applied specifically to ASL k-space undersampling; demonstrates 4× acceleration without CBF quantification bias.
2. **PromptMR for multi-contrast MRI** (Xin et al., ECCV 2024): Prompt-based approach generalising to ASL, diffusion-weighted, and BOLD contrasts with a single model.
3. **Score-MRI applied to ASL** (2024): Diffusion model posterior sampling conditioned on ASL kinetic model; handles the unique signal characteristics of pulsed and pseudo-continuous ASL.
4. **ASL-specific compressed sensing** (Zhao et al., JMRI 2024): Kinetic-model-constrained compressed sensing for ASL; integrates Buxton model into the regularisation to prevent CBF bias at high acceleration factors.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/asl_mri_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/asl_mri_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/asl_mri_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/asl_mri/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses carrier routing `(medical, Spin/RF)` → `mri` pool (10 methods). All 10 algorithms are real, well-cited MRI reconstruction methods that are directly applicable to ASL k-space reconstruction. The three mismatch parameters (labelling efficiency, transit delay, T1_blood) capture the ASL-specific kinetic model uncertainties on top of standard MRI calibration errors. Note that ASL-specific perfusion quantification methods (Buxton kinetic model fitting) are not in the leaderboard, which is by design — the benchmark focuses on the k-space reconstruction step.

---
*Comprehensive 6-point check by deep-check pipeline v3*
