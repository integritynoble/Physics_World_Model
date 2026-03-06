# Comprehensive 6-Point Check — Coded Exposure / Flutter Shutter

**URL:** https://pwm.platformai.org/benchmark/coded_exposure
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Coded Exposure / Flutter Shutter Photography

**Physical principle:** Coded exposure (flutter shutter) modulates the camera shutter open/close state pseudo-randomly during a single exposure interval. Unlike a conventional open-shutter exposure that produces an uninvertible rectangular motion blur, the flutter shutter creates a coded motion blur whose frequency response avoids deep nulls — making the point spread function (PSF) well-conditioned for deconvolution. The resulting motion-blurred image can then be deconvolved to recover a sharp, motion-frozen image of a moving scene. The technique was introduced by Raskar et al. (SIGGRAPH 2006) and the forward model is a 1D convolution along the motion direction.

**Forward model:**
```
Coded motion blur:
  y(x,y) = ∫_0^T c(t) · f(x - v_x t, y - v_y t) dt + n(x,y)

Discrete 1D convolution model:
  y = h_code ⊛ x + n

where:
  x          — sharp scene image (ground truth)
  c(t) ∈ {0,1}^T — binary shutter code sequence (known)
  v = (v_x, v_y) — scene velocity (pixels/second)
  h_code     — coded exposure PSF: h_code[k] = c(k/f_s)
  ⊛          — 2D convolution (1D along motion direction)
  n          — sensor readout + shot noise
  y          — observed blurred image

Frequency-domain condition:
  |Ĥ_code(ω)|² >> 0  ∀ω   (well-conditioned if code is pseudo-random)
  vs conventional: |Ĥ_rect(ω)|² has deep nulls at ω = k/T
```

**Inverse problem:** Recover the sharp scene image x from the coded motion-blurred observation y by deconvolving with the known PSF h_code, exploiting the fact that the flutter shutter avoids nulls in the frequency response.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** M(coded shutter) → C(motion blur convolution) → D(sensor)

**Key mismatch parameters:**
- `shutter_code_timing_error` (s_c): imprecision in shutter code timing implementation; nominal 0.0, perturbed 1.0 (code shift)
- `motion_blur_psf_mismatch` (m_b): velocity estimation error causing PSF mismatch; nominal 0.0, perturbed 4.0 (velocity error %)
- `sensor_readout_noise` (s_r): sensor readout noise level; nominal 5.0 e⁻, perturbed 7.0 e⁻

**Dataset format:**
- `x_true: (H, W)` — sharp ground truth image (motion-free scene)
- `y: (H, W)` — coded exposure blurred image (single frame with coded PSF)
- `H_ideal: (H*W, H*W)` — sparse convolution matrix encoding the flutter shutter PSF

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Wiener-Deconv | Classical | Raskar et al., SIGGRAPH 2006 | Wiener deconvolution of coded exposure PSF; THE foundational flutter shutter method |
| Laplacian Pyramid | Classical | Burt & Adelson 1983 | Multi-scale Laplacian pyramid deblurring; classical spatial-domain approach |
| Lucy-Richardson | Classical | Richardson 1972 / Lucy 1974 | Iterative blind deconvolution; applicable when velocity is estimated jointly |
| PnP-FFDNet | Plug-and-Play | Zhang et al., IEEE TIP 2018 | PnP with FFDNet denoiser; handles noise amplification in frequency-domain deconvolution |
| U-Net | Deep Learning | Ronneberger et al., MICCAI 2015 | Encoder-decoder for coded exposure deblurring |
| Uformer | Transformer | Wang et al., CVPR 2022 | Transformer for general image restoration; applicable to coded exposure deblurring |

---

## 4. Literature & State of the Art (2024–2025)

1. **Flutter shutter with event cameras** (2024): Coded exposure combined with event-camera velocity estimation; removes velocity estimation step enabling real-time flutter shutter at 1000+ fps scenes.
2. **Neural coded aperture design** (2024): Differentiable coded exposure PSF optimisation for joint motion estimation and deblurring; learns the optimal shutter code for a given scene class.
3. **Restormer for motion deblurring** (Zamir et al., CVPR 2022 / extended 2024): Transformer outperforms U-Net on non-uniform motion blur at equivalent parameter count; applicable to coded and uncoded motion blur.
4. **Coded exposure video reconstruction** (2025): Temporal extension of flutter shutter to video; compressive temporal sensing with transformer-based reconstruction from single-frame exposures.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/coded_exposure_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/coded_exposure_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/coded_exposure_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/coded_exposure/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the `computational_photography` category pool (14 methods: Wiener-Deconv, Laplacian Pyramid, Lucy-Richardson, PnP-FFDNet, PnP-ADMM, HDR-CNN, U-Net, LaplacianFormer, Uformer, DeblurGaussian, HDRFormer, PhotoFormer, DiffusionPhoto, ScorePhoto). Wiener-Deconv is the canonical flutter shutter deconvolution method (Raskar et al. 2006), confirming domain correctness. The three mismatch parameters address the key coded exposure calibration issues: code timing error, velocity mismatch, and sensor noise. Note that HDR-CNN in the pool (Eilertsen et al., ACM TOG 2017) is domain-mismatched (HDR reconstruction, not motion deblurring) but is a known limitation of the shared computational photography pool. No code changes required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
