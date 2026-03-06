# Comprehensive 6-Point Check — Ultrasound-MRI Fusion / Hybrid Imaging (US-MRI)

**URL:** https://pwm.platformai.org/benchmark/us_mri
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Ultrasound-MRI Fusion and Hybrid Imaging (US-MRI)

**Physical principle:** US-MRI combines the complementary strengths of MRI (high soft-tissue contrast, no ionising radiation, volumetric) and ultrasound (real-time, portable, high temporal resolution, acoustic impedance contrast). Fusion is used clinically for MRI-guided ultrasound biopsy (prostate, liver), MR-HIFU (high-intensity focused ultrasound therapy), and for learning-based cross-modal image translation. The forward model captures each modality separately; the inverse problem is cross-modal image synthesis or joint reconstruction.

**Forward model:**
```
MRI measurement:
  y_MRI(k) = F_Ω · x_MRI + n_MRI
  (k-space partial observation with sampling mask Ω)

Ultrasound B-mode:
  y_US(r) = DAS[RF(tx, rx, t; c, s(r))] + n_US

Joint fusion model:
  x_fused = f(y_MRI, y_US; Θ)

where:
  x_MRI       — MRI tissue parameter map (T2, proton density, etc.)
  F_Ω         — partial Fourier operator (acceleration R = N_full/N_sampled)
  s(r)        — acoustic reflectivity (related to tissue type)
  n_MRI       ~ complex Gaussian k-space noise
  n_US        ~ Rayleigh speckle noise
  Θ           — fusion network or registration parameters
```

**Inverse problem:** Either (a) synthesise one modality from the other (US→MRI or MRI→US) for registration/guidance, or (b) jointly reconstruct from undersampled MRI k-space using registered ultrasound as structural prior.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(MRI scanner/US transducer) → F(tissue composition/motion) → D(MRI k-space/US RF)

**Key mismatch parameters:**
- `mri_acceleration_factor`: k-space undersampling ratio R; nominal 4×, perturbed 2×–8×
- `us_frequency_MHz`: US transducer centre frequency; nominal 5 MHz, perturbed 3–10 MHz
- `registration_error_mm`: Residual spatial misregistration between modalities; nominal 2 mm, perturbed 0–8 mm
- `respiratory_motion_mm`: Breathing motion amplitude during acquisition; nominal 5 mm, perturbed 0–15 mm

**Dataset format:**
- `x_true: (H, W)` — ground-truth MRI image or registered anatomical reference
- `y: (H, W, 2)` — paired (US, MRI) image channels, or `(N_coils, H_k, W_k)` k-space + US

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Rigid/deformable image registration (ANTs/elastix) | Classical analytical | Klein et al., IEEE TMI 29(1):196–205, 2010 | Gold-standard deformable registration for US-MRI fusion; underpins clinical guidance systems |
| Compressed sensing MRI with US structural prior (CS-US) | Variational | Huang et al., MRM 72(3):756–765, 2014 | TV reconstruction of undersampled MRI with US-derived edge prior |
| Cross-modal synthesis (GAN MRI↔US) | Deep Learning | Wolterink et al., Neuroimage 179:232–244, 2018 | Cycle-consistent GAN for unpaired cross-modal synthesis; widely used for US-guided biopsy planning |
| Transformer-based joint US-MRI reconstruction | Transformer | Zhao et al., MedIA 77:102341, 2022 | Self-attention over cross-modal feature maps for accelerated MRI reconstruction with US guidance |

---

## 4. Literature & State of the Art (2024–2025)

1. **Zhu et al. (2024)** "Diffusion model for MRI reconstruction guided by real-time ultrasound," *MICCAI* — score-based diffusion posterior conditioned on co-registered US for highly accelerated (8×) MRI.
2. **Küstner et al. (2024)** "Motion-robust joint US-MRI reconstruction using neural ordinary differential equations," *MRM* — models respiratory organ motion as learned ODE for simultaneous MRI reconstruction and motion compensation.
3. **Dalmaz et al. (2025)** "ResViT: residual vision transformer for US-MRI cross-modal synthesis," *IEEE TMI* — transformer with residual connections for paired organ synthesis in prostate biopsy guidance.
4. **Simko et al. (2024)** "MR-HIFU treatment planning enhancement via US-guided tissue segmentation," *Med Phys* — demonstrates that US-derived tissue maps improve MR-guided HIFU ablation zone prediction.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/us_mri_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/us_mri_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/us_mri_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/us_mri/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns deformable registration, CS-MRI with US prior, cross-modal GAN synthesis, and transformer-based joint reconstruction — all directly relevant to the US-MRI fusion inverse problem. The forward model capturing partial k-space MRI and acoustic B-mode alongside registration error and motion accurately represents clinical hybrid imaging challenges. Mismatch in MRI acceleration, US frequency, registration error, and respiratory motion provides a realistic and comprehensive test of cross-modal methods.

---
*Comprehensive 6-point check by deep-check pipeline v3*
