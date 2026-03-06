# Comprehensive 6-Point Check — Coded Aperture Compressive Temporal Imaging (CACTI)

**URL:** https://pwm.platformai.org/benchmark/cacti
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Coded Aperture Compressive Temporal Imaging (CACTI)

**Physical principle:** CACTI captures high-speed video scenes by encoding multiple temporal frames onto a single 2D detector snapshot. A spatially-varying, time-modulated binary mask (coded aperture) is placed in front of the sensor; as the mask shifts during the detector exposure, each pixel integrates a different linear combination of the underlying video frames. The result is a compressed 2D measurement from which all frames must be jointly reconstructed.

**Forward model:**
```
y = sum_{t=1}^{T} (M_t ⊙ x_t) + n

where:
  y       ∈ R^{H×W}        — single 2D detector snapshot (compressed measurement)
  x_t     ∈ R^{H×W}        — t-th video frame of the scene
  M_t     ∈ {0,1}^{H×W}    — binary mask pattern at time t (known, shifts by 1 row/frame)
  ⊙                         — element-wise multiplication (Hadamard product)
  T                         — number of compressed frames (typically 8–32)
  n                         — Gaussian detector noise
```

**Inverse problem:** Recover the full video sequence `{x_1, ..., x_T}` from the single compressed snapshot `y` and the known mask sequence `{M_t}`, an extremely under-determined reconstruction problem.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(scene/motion) → F(coded mask + optics) → D(CMOS snapshot)

**Key mismatch parameters:**
- `n_frames`: Number of compressed temporal frames; nominal 8, perturbed 4–32
- `mask_shift`: Sub-pixel mask shift accuracy; nominal 1.0 px/frame, perturbed ±0.3 px
- `mask_binarization`: Threshold for binary aperture; nominal 0.5, perturbed 0.3–0.7
- `noise_std`: Detector read noise standard deviation; nominal 0.01, perturbed 0.005–0.05

**Dataset format:**
- `x_true: (T, H, W)` — ground-truth video frames (T frames, 256×256 pixels each)
- `y: (H, W)` — single 2D coded snapshot (compressed measurement)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| GAP-TV (Generalized Alternating Projection with TV) | Classical | Yuan, X. (2016) "Generalized alternating projection based total variation minimization for compressive sensing," *ICIP* | Standard total-variation regularized baseline for CACTI |
| DeSCI (Decompress Snapshots of Compressively Imaged Dynamic Scenes) | Classical/Sparse | Liu, Y. et al. (2018) "Rank minimization for snapshot compressive imaging," *IEEE TPAMI* 41(12):2990–3006 | Exploits low-rank + sparsity structure for video recovery |
| BIRNAT (Bidirectional Recurrent Neural Network for Adaptive Temporal) | Deep Learning | Cheng, Z. et al. (2021) "Recurrent neural networks for snapshot compressive imaging," *IEEE TPAMI* 44(12):9758–9775 | End-to-end learned unrolling with bidirectional RNN |
| STFormer (Spatial-Temporal Transformer for CACTI) | Transformer | Wang, Z. et al. (2022) "Spatial-temporal transformer for video snapshot compressive imaging," *IEEE TPAMI* 45(7):9072–9089 | Attention-based joint spatial-temporal video reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Zheng, S. et al. (2024)** "EfficientSCI: Densely connected network with space-time factorization for large-scale video snapshot compressive imaging," *CVPR* — Achieves real-time reconstruction of 10-megapixel 30-fps video from single snapshots.
2. **Yang, J. et al. (2024)** "Scalable learned video compressive sensing with physics-guided unrolling," *IEEE Trans. Image Process.* — Unrolled ADMM with learned proximal operators scales to 256-frame compression ratios.
3. **Meng, Z. et al. (2024)** "Coded aperture snapshot spectral-temporal imaging: joint design and reconstruction," *Optica* — Extends CACTI to simultaneous spectral-temporal compression in a single optical system.
4. **Chen, X. et al. (2025)** "Diffusion models for snapshot compressive imaging reconstruction," *Optics Express* — Score-based generative model as a prior for high-quality CACTI video recovery under heavy compression.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cacti_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cacti_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cacti_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/cacti/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The CACTI benchmark correctly captures the core compressive video sensing problem with a physically accurate Hadamard-product forward model and binary shifting mask. Algorithm routing spans the canonical literature from GAP-TV through DeSCI, BIRNAT, and transformer-based STFormer, representing the progression of the field from classical sparse recovery to learned temporal sequence models. The benchmark mismatch parameters (mask accuracy, compression ratio, noise) are well-chosen to probe algorithmic robustness to real hardware imperfections.

---
*Comprehensive 6-point check by deep-check pipeline v3*
