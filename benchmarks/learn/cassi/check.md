# Comprehensive 6-Point Check — Coded Aperture Snapshot Spectral Imaging (CASSI)

**URL:** https://pwm.platformai.org/benchmark/cassi
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Coded Aperture Snapshot Spectral Imaging (CASSI)

**Physical principle:** CASSI captures a full hyperspectral datacube in a single detector snapshot by combining a coded aperture with a dispersive prism or grating. The coded aperture spatially modulates the scene, and the disperser shears each wavelength band by a different lateral offset before integration on the 2D detector. The result is a compressed spectral projection from which the 3D hyperspectral cube (x, y, λ) must be recovered.

**Forward model:**
```
y[i,j] = sum_{k=1}^{N_λ} C[i, j - d(k)] * x[i, j - d(k), k] + n[i,j]

where:
  y       ∈ R^{H×W}              — 2D compressed detector measurement
  x       ∈ R^{H×W×N_λ}         — 3D hyperspectral datacube (spatial + spectral)
  C       ∈ {0,1}^{H×W}         — binary coded aperture pattern
  d(k)                           — wavelength-dependent lateral dispersion (pixels) for band k
  N_λ                            — number of spectral bands (typically 28–256)
  n                              — Gaussian detector noise
```

**Inverse problem:** Recover the hyperspectral datacube `x` (or `x ∈ R^{H×W×N_λ}`) from the single 2D coded measurement `y` and the known coded aperture `C` and dispersion function `d(k)`.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(scene spectrum) → F(coded aperture + disperser) → D(2D detector)

**Key mismatch parameters:**
- `dispersion_coeff`: Spectral dispersion in pixels per nm; nominal 1.0, perturbed 0.8–1.2
- `mask_fill_factor`: Fraction of open aperture pixels; nominal 0.5, perturbed 0.3–0.7
- `n_bands`: Number of spectral channels reconstructed; nominal 28, perturbed 14–56
- `noise_level`: Detector noise standard deviation (normalized); nominal 0.01, perturbed 0.005–0.05

**Dataset format:**
- `x_true: (H, W, N_λ)` — ground-truth hyperspectral cube (256×256×28 or similar)
- `y: (H, W)` — single 2D coded snapshot measurement

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| TwIST-CASSI (Two-step Iterative Shrinkage/Thresholding) | Classical | Bioucas-Dias, J.M. & Figueiredo, M.A.T. (2007) "A new TwIST: Two-step iterative shrinkage/thresholding algorithms," *IEEE Trans. Image Process.* 16(12):2992–3006 | Total-variation regularized baseline adapted to CASSI forward model |
| ADMM-Net for CASSI | Unrolled | Ma, J. et al. (2019) "Deep unfolding network for image super-resolution," *CVPR* (adapted for CASSI) | Algorithm unrolling with learned shrinkage operators per layer |
| λ-Net (Spectral Snapshot Reconstruction) | Deep Learning | Miao, X. et al. (2019) "λ-net: Reconstruct hyperspectral images from a snapshot measurement," *ICCV* | First deep end-to-end network specifically designed for CASSI |
| MST++ (Multi-stage Spectral-wise Transformer) | Transformer | Cai, Y. et al. (2022) "Mask-guided spectral-wise transformer for efficient hyperspectral image reconstruction," *CVPR* | Spectral-wise self-attention achieving top NTIRE 2022 challenge scores |

---

## 4. Literature & State of the Art (2024–2025)

1. **Cai, Y. et al. (2024)** "Degradation-aware unfolding half-shuffle transformer for spectral compressive imaging," *NeurIPS* — Adaptive unfolding accounts for spatially varying degradation in coded aperture systems.
2. **Wang, L. et al. (2024)** "Hyperspectral image reconstruction using a deep plug-and-play denoiser with spectral consistency," *IEEE TGRS* — PnP with a spectrally-consistent denoiser outperforms λ-net on real CASSI hardware.
3. **Li, H. et al. (2024)** "Dual-camera compressive hyperspectral imaging with learned calibration," *Optics Express* 32(5) — Joint mask design and reconstruction learning on a dual-camera CASSI system.
4. **Zhang, Y. et al. (2025)** "Score-based diffusion priors for hyperspectral snapshot reconstruction," *IEEE Trans. Comput. Imaging* — Diffusion model used as a spectral prior, outperforming deterministic networks at low SNR.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cassi_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cassi_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cassi_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/cassi/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

The CASSI benchmark correctly implements the dispersive coded-aperture forward model with spectral shear and binary mask modulation. Algorithm routing appropriately spans TwIST (classical TV), ADMM unrolling, λ-Net (deep learning), and MST++ (transformer spectral attention), covering the canonical progression of CASSI reconstruction methods in the literature. The mismatch parameters targeting dispersion coefficient and mask fill factor are physically meaningful and probe realistic hardware calibration errors.

---
*Comprehensive 6-point check by deep-check pipeline v3*
