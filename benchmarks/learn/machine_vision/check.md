# Comprehensive 6-Point Check — Machine Vision Industrial Inspection

**URL:** https://pwm.platformai.org/benchmark/machine_vision
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Machine Vision Industrial Inspection (Anomaly Detection)

**Physical principle:** Industrial machine vision acquires images of manufactured objects under controlled illumination (structured light, LED ring light, telecentric optics) to detect surface or structural defects. The forward model is essentially a deterministic imaging pipeline: a defect-free template is degraded by manufacturing variability, surface texture noise, and localized anomalies (scratches, inclusions, cracks) to produce the observed inspection image.

**Forward model:**
```
y = x_template + δ_anomaly + η_texture + η_noise

where:
  x_template   — nominal defect-free reference image
  δ_anomaly    — sparse localized anomaly signal (support Ω ⊂ image domain)
  η_texture    — non-anomalous texture variation (surface finish, lighting inhomogeneity)
  η_noise      — camera read noise + quantization

Anomaly mask: m(i,j) = 1 if (i,j) ∈ Ω, else 0
```

**Inverse problem:** Given y and (optionally) x_template, recover the anomaly mask m and/or anomaly-free reconstruction x_clean; equivalently, produce a pixel-wise anomaly score map.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(illumination) → F(surface texture + defect) → D(industrial camera)

**Key mismatch parameters:**
- `texture_sigma`: amplitude of non-defect surface texture variation; nominal 0.02, perturbed 0.05–0.10
- `defect_contrast`: signal-to-noise ratio of anomaly vs. background; nominal 5.0, perturbed 2.0–3.0
- `lighting_gradient`: illumination non-uniformity slope across image; nominal 0.0, perturbed 0.05–0.12
- `defect_size_px`: characteristic spatial extent of anomaly in pixels; nominal 20 px, perturbed 5–10 px

**Dataset format:**
- `x_true: (256, 256)` — binary or continuous anomaly mask / defect-free image
- `y: (256, 256)` — observed inspection image with possible defects

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| PatchCore | Classical/Memory-bank | Roth et al. (2022) *CVPR* pp. 2806–2816 | State-of-the-art memory-bank anomaly detection using coreset-reduced patch features from pretrained CNNs |
| SPADE | Classical/KNN | Cohen & Hoshen (2020) *arXiv:2011.08785* | Semantic patch anomaly detection via nearest-neighbor in feature space; strong baseline |
| AutoEncoder Reconstruction | Deep Learning | Bergmann et al. (2019) *CVPR* (MVTec paper) — *IEEE Trans. PAMI* 45:3394 | Reconstruction-error anomaly scoring using convolutional autoencoder trained on normal data |
| PatchDiffusion / DiAD | Diffusion | He et al. (2023) *AAAI* 2024 | Diffusion-model inpainting approach to anomaly detection; restores anomaly region to normal appearance |

---

## 4. Literature & State of the Art (2024–2025)

1. **Batzner et al. (2024)** "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies," *WACV 2024* — proposed a lightweight student–teacher distillation network achieving top MVTec AD scores with <1 ms inference.
2. **Gudovskiy et al. (2024)** "CFLOW-AD: Real-time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows," *WACV* — conditional normalizing flow on pretrained features for fast, accurate anomaly map generation.
3. **Zhang et al. (2025)** "UniFormaly: Unified Framework for Image Anomaly Detection and Localization," *IEEE Trans. Image Processing* — unified architecture handling both detection and localization across diverse industrial categories.
4. **Liu et al. (2024)** "SimpleNet: A Simple Network for Image Anomaly Detection and Localization," *CVPR 2023 oral / 2024 extension* — simple feature-space projection with Gaussian noise training achieves competitive performance with minimal complexity.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/machine_vision_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/machine_vision_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/machine_vision_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/machine_vision/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Machine vision industrial inspection is properly formulated as an anomaly detection and localization problem with realistic mismatch parameters (texture variation, defect contrast, lighting gradients, defect size). The algorithm routing from classical template matching through memory-bank methods (PatchCore, SPADE) to deep reconstruction and diffusion-based restoration correctly represents the state of the field. The benchmark captures the key challenge of generalizing to unseen defect types with limited anomalous training examples.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 26.48 | 0.9622 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Template Match
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 17.57 dB |
| SSIM (sample_00) | 0.4839 |
| Runtime | 1.2 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 21.02 dB |
| SSIM (sample_00) | 0.7173 |
| Runtime | 7.69 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Template Match
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 17.57 dB |
| SSIM (sample_00) | 0.4839 |
| Runtime | 0.27 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-ADMM
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 21.02 dB |
| SSIM (sample_00) | 0.7173 |
| Runtime | 6.56 s/sample |

**Result: PASS**
