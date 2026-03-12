# Comprehensive 6-Point Check — Ultrasonic Phased Array Imaging (UT-PA)

**URL:** https://pwm.platformai.org/benchmark/ultrasonic_phased_array
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Ultrasonic Phased Array Non-Destructive Testing (UT-PA)

**Physical principle:** Ultrasonic phased arrays use a linear or matrix array of piezoelectric elements (typically 64–256 elements, 1–10 MHz) to transmit focused acoustic beams and receive reflected echoes from defects in industrial components. Full Matrix Capture (FMC) acquires the complete set of transmit-receive element-pair responses (N² A-scans), enabling offline post-processing with the Total Focusing Method (TFM) or synthetic aperture focusing technique (SAFT) to reconstruct a high-resolution defect image at any depth.

**Forward model:**
```
A(tx, rx, t) = ∫∫ r(x,z) · h(t - τ_tx(x,z) - τ_rx(x,z)) dx dz + n(tx,rx,t)

τ_tx(x,z) = √((x_tx - x)² + z²) / c
τ_rx(x,z) = √((x_rx - x)² + z²) / c

TFM image:
  I(x,z) = |Σ_tx Σ_rx A(tx, rx, t = τ_tx + τ_rx)|²

where:
  A(tx,rx,t)  — FMC A-scan data
  r(x,z)      — reflectivity map (defect/microstructure)
  h(t)        — system impulse response (element + cable + digitizer)
  c           — longitudinal wave speed in material
  n           ~ Gaussian electronic + thermal noise
```

**Inverse problem:** Recover the reflectivity map r(x,z) from the FMC dataset A(tx,rx,t), resolving defect position, geometry, and size beyond the diffraction limit.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(array transducer/frequency) → F(material wave speed/attenuation/anisotropy) → D(receive array/digitizer)

**Key mismatch parameters:**
- `wave_speed_m_s`: Longitudinal wave speed in material; nominal 5920 m/s (steel), perturbed 5500–6400 m/s
- `attenuation_dB_mm_MHz`: Frequency-dependent attenuation; nominal 0.1 dB/mm/MHz, perturbed 0.05–0.5
- `centre_frequency_MHz`: Array centre frequency; nominal 5 MHz, perturbed 2–10 MHz
- `element_pitch_mm`: Array element pitch; nominal 0.6 mm, perturbed 0.4–1.0 mm

**Dataset format:**
- `x_true: (H, W)` — ground-truth defect reflectivity map (or binary defect mask)
- `y: (N_tx, N_rx, N_t)` — FMC A-scan data cube

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Total Focusing Method (TFM) | Classical delay-and-sum | Holmes et al., Insight 47(9):587–595, 2005 | Gold-standard FMC post-processing; coherent summation over all tx-rx pairs at each pixel |
| SAFT (Synthetic Aperture Focusing Technique) | Classical analytical | Nagai et al., Ultrasonics 20(3):117–122, 1982 | F-SAFT frequency-domain implementation; computationally efficient alternative to TFM |
| Adaptive imaging via phase coherence factor | Classical adaptive | Camacho et al., IEEE TUFFC 56(5):958–974, 2009 | Weights TFM by phase coherence to suppress grating lobes and noise |
| Deep learning FMC-to-image (ResNet/U-Net) | Deep Learning | Pyle et al., IEEE TUFFC 68(2):507–520, 2021 | CNN trained to map FMC data directly to high-resolution defect images |

---

## 4. Literature & State of the Art (2024–2025)

1. **Budyn et al. (2024)** "Physics-informed deep learning for ultrasonic full matrix capture reconstruction in anisotropic welds," *NDT & E Int* — PINN embedding wave propagation for accurate TFM in austenitic stainless steel welds with crystallographic texture.
2. **Pyle et al. (2024)** "Unrolled TFM network for model-based defect reconstruction from sparse array data," *IEEE TUFFC* — unrolled iterative algorithm with learned regularization outperforming standard TFM on sparse arrays.
3. **Masson et al. (2025)** "Self-supervised ultrasonic phased array imaging via contrastive learning," *NDT Int* — self-supervised framework for defect detection without annotated FMC training datasets.
4. **Zhang et al. (2024)** "Transformer-based super-resolution TFM for sub-wavelength defect characterisation," *Ultrasonics* — ViT architecture applied to FMC data achieves lateral resolution beyond the diffraction limit.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ultrasonic_phased_array_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ultrasonic_phased_array_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ultrasonic_phased_array_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/ultrasonic_phased_array/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns TFM, SAFT, adaptive phase-coherence imaging, and deep-learning FMC reconstruction — all standard and emerging approaches specific to ultrasonic phased array NDT. The forward model with FMC delay-and-sum, frequency-dependent attenuation, and wave speed accurately represents UT-PA acquisition. Mismatch in wave speed, attenuation, centre frequency, and element pitch tests generalisation across diverse materials (steel, aluminium, composites) and transducer configurations.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 29.60 | 0.6891 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** TFM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.51 dB |
| SSIM (sample_00) | 0.8477 |
| Runtime | 0.6 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SAFT
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.51 dB |
| SSIM (sample_00) | 0.8477 |
| Runtime | 0.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TFM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.51 dB |
| SSIM (sample_00) | 0.8477 |
| Runtime | 0.51 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SAFT
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 34.51 dB |
| SSIM (sample_00) | 0.8477 |
| Runtime | 0.51 s/sample |

**Result: PASS**
