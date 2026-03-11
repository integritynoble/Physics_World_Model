# Comprehensive 6-Point Check — OCT Angiography

**URL:** https://pwm.platformai.org/benchmark/octa
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** OCT Angiography (OCTA)

**Physical principle:** OCT angiography detects blood flow without contrast agents by comparing repeated B-scans acquired at the same position. Moving red blood cells produce temporal decorrelation of the OCT speckle pattern, while static tissue remains correlated. Algorithms computing speckle variance, optical coherence, or complex-signal decorrelation between repeated acquisitions yield a motion-contrast map that highlights retinal and choroidal vasculature with capillary-level resolution.

**Forward model:**
```
For M repeated B-scans at the same position: {A_m(z, x)}_{m=1}^{M}

Speckle variance: SV(z,x) = (1/M) Σ_m |A_m(z,x)|² - |(1/M) Σ_m A_m(z,x)|²

Complex decorrelation (OCDS):
  D(z,x) = 1 - |⟨A_m(z,x) · A_{m+1}*(z,x)⟩| / (⟨|A_m|²⟩ · ⟨|A_{m+1}|²⟩)^{1/2}

where A_m = OCT complex A-scan amplitude at repetition m
      * denotes complex conjugate
      ⟨·⟩ denotes average over repeats
```

**Inverse problem:** Recover the 2D/3D blood flow (angiographic) map from M repeated OCT B-scans, maximizing vessel contrast and capillary detectability while suppressing motion-induced projection artifacts, bulk motion, and noise.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(swept-source / SD-OCT) → F(retinal microvasculature + static tissue) → D(balanced photodetector)

**Key mismatch parameters:**
- `n_repeats`: number of repeated B-scans per position; nominal 4, perturbed 2 (lower contrast)
- `bulk_motion_um`: axial bulk motion amplitude between repeats; nominal 0 µm, perturbed 10–30 µm
- `snr_db`: OCT signal-to-noise ratio for individual A-scans; nominal 35 dB, perturbed 20–25 dB
- `vessel_diameter_um`: characteristic capillary diameter in the scene; nominal 8 µm, perturbed 4–6 µm (marginal capillaries)

**Dataset format:**
- `x_true: (256, 256)` — ground-truth angiographic map (vessel mask or continuous flow map)
- `y: (M, 256, 256)` — stack of M repeated B-scans (magnitude or complex) for motion-contrast computation

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Speckle Variance OCT-A | Classical | Mariampillai et al. (2008) *Opt. Lett.* 33:1530–1532 | Foundational intensity-based motion contrast; computes variance across repeated B-scans |
| OMAG (Optical Microangiography) | Classical | Wang et al. (2007) *Nature Protocols* 2:1212–1219 | Complex-signal-based angiography separating flow from static signals via high-pass filtering |
| SSADA (Split-Spectrum Amplitude-Decorrelation) | Variational | Jia et al. (2012) *Opt. Express* 20:4710–4725 | Splits spectrum into sub-bands to improve decorrelation SNR; basis of most commercial OCTA |
| Deep OCTA Enhancement (IPN-V2 / RV-GAN) | Deep Learning | Ma et al. (2021) *Artif. Intell. Med.* 115:102057; Kamran et al. (2021) *CVPR* | GAN-based vessel enhancement and projection artifact suppression in retinal OCTA |

---

## 4. Literature & State of the Art (2024–2025)

1. **Liu et al. (2024)** "Self-supervised motion artifact correction for wrist-worn OCTA," *Biomedical Optics Express* — contrastive learning approach removes bulk-motion-induced false-positive flow signals, reducing artifact area by 70% in wearable OCTA.
2. **Zhang et al. (2024)** "Diffusion model-based OCTA reconstruction from single B-scan," *Optica* — score-based generative model learns to produce OCTA maps from a single structural B-scan using paired training data.
3. **Zang et al. (2025)** "Transformer-based 3D OCTA vessel segmentation and flow quantification," *IEEE Trans. Medical Imaging* — hierarchical vision transformer for volumetric vessel segmentation with FAZ quantification in diabetic retinopathy screening.
4. **Spaide et al. (2024)** "Review: Clinical interpretation of OCTA artifacts and flow impairment," *Prog. Retinal Eye Res.* — comprehensive review of artifact mechanisms and their impact on clinical OCTA image interpretation.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/octa_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/octa_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/octa_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/octa/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

OCTA is correctly formulated as a motion-contrast extraction problem where repeated B-scan acquisitions are analyzed for temporal decorrelation to reveal blood flow, with the challenge of separating true flow signals from bulk motion, noise, and projection artifacts. The algorithm routing from speckle variance through SSADA to deep GAN-based enhancement appropriately spans the state of the art in clinical OCTA processing. The mismatch parameters (number of repeats, bulk motion, SNR, vessel diameter) are the primary experimental factors affecting capillary detection sensitivity in retinal OCTA.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 16.78 | 0.4326 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
