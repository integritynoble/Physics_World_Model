# SPC (Single-Pixel Camera) Validation Results

**Date:** 2026-02-16
**Framework:** InverseNet ECCV Validation v4.0 (Pretrained Models)
**Dataset:** Set11 (11 images, 256x256/512x512)
**Methods:** FISTA-TV (classical), ISTA-Net (pretrained), HATNet (pretrained)
**Total Execution Time:** 20.7 minutes (112.9s per image)

---

## Method Details

### FISTA-TV (Classical)
- Block size: 33x33 (same as ISTA-Net, shared Phi matrix)
- Measurement: Learned Phi from ISTA-Net training (272x1089, CR=25%)
- Algorithm: Nesterov-accelerated FISTA with TV proximal (Chambolle)
- Regularization: lam=0.005, 500 iterations, 10 TV inner iterations
- Step size: tau = 0.9/L (Lipschitz via power iteration)

### ISTA-Net (Pretrained Deep Learning)
- Architecture: CS_ISTA_Net (non-plus), 9 layers, simple BasicBlock (no conv_D/conv_G)
- Weights: `CS_ISTA_Net_layer_9_group_1_ratio_25_lr_0.0001/net_params_200.pkl`
- Measurement: Learned Phi (272x1089) + Qinit (1089x272), CR=25%
- Block size: 33x33 (1089 pixels per block)
- **Verified clean baseline: 31.85 dB (matches published 31.84 dB)**

### HATNet (Pretrained Deep Learning)
- Architecture: HATNet with Kronecker measurement (H@X@W^T)
- Measurement: H=[128,256], W=[128,256] (learned, CR=25%)
- Reconstruction: 7 ISTA stages with transformer denoisers (DSMHSA + CAB)
- Weights: `cr_0.25.pth` (2024 pretrained, 966MB)
- Full image: 256x256 direct, 512x512 split into 4 quadrants
- **Split forward verified: max diff = 0.00 from standard forward**

---

## Scenario Definitions

| Scenario | Description | Measurement | Operator | Purpose |
|----------|-------------|-------------|----------|---------|
| **I (Ideal)** | Upper bound | Clean (no drift)* | Ideal | Best achievable |
| **II (Baseline)** | Practical worst case | Gain drift + noise | Assumed ideal | Uncorrected mismatch |
| **III (Corrected)** | Calibration benefit | y_corr = y_meas/gain | Assumed ideal | Post-correction quality |

*Scenario I: Noiseless for ISTA-Net/FISTA-TV; sensor noise (sigma=0.04) for HATNet.

---

## Mismatch Model: Exponential Gain Drift

Per-row exponential drift: `g_i = exp(-alpha * i)`

| Parameter | ISTA-Net / FISTA-TV | HATNet |
|-----------|-------------------|--------|
| Block size | 33x33 | 256x256 (full image) |
| Measurements | 272 per block | 128x128 (Kronecker) |
| Drift model | 1D: `g = exp(-alpha * [0..271])` | 2D: `g[i,j] = exp(-alpha_h*i) * exp(-alpha_w*j)` |
| alpha | 1.5e-3 | 1.5e-3 (both dims) |
| sigma_noise | 0.03 | 0.04 |
| Min gain | 0.667 (at row 271) | 0.684 (at corner [127,127]) |

---

## Quantitative Results

### PSNR Performance (dB) - Mean +/- Std over 11 images

| Method | Scenario I (Ideal) | Scenario II (Baseline) | Scenario III (Corrected) |
|--------|-------------------|----------------------|------------------------|
| **FISTA-TV** | 28.06 +/- 3.38 | 18.51 +/- 0.69 | 26.21 +/- 2.28 |
| **ISTA-Net** | 31.85 +/- 3.11 | 19.02 +/- 0.61 | 27.45 +/- 1.32 |
| **HATNet** | 30.98 +/- 0.95 | 19.40 +/- 0.59 | 29.78 +/- 0.81 |

---

## Gap Analysis

### Mismatch Impact (Scenario I -> II)

| Method | PSNR Drop | Interpretation |
|--------|-----------|----------------|
| **FISTA-TV** | 9.55 dB | Moderate: classical solver sensitive to gain mismatch |
| **ISTA-Net** | 12.83 dB | Strong: learned solver highly sensitive to mismatch |
| **HATNet** | 11.58 dB | Strong: Kronecker system sensitive to 2D gain drift |

### Recovery with Gain Correction (Scenario II -> III)

| Method | PSNR Gain | Recovery % | Interpretation |
|--------|-----------|-----------|----------------|
| **FISTA-TV** | 7.71 dB | 81% | Good recovery with classical solver |
| **ISTA-Net** | 8.43 dB | 66% | Strong recovery, residual gap from noise amplification |
| **HATNet** | 10.38 dB | 90% | Excellent: robust transformer denoiser handles amplified noise |

### Overall Hierarchy

For each method: **Scenario I >= Scenario III >> Scenario II**

- **FISTA-TV:** I=28.06 > III=26.21 >> II=18.51 (1.85 dB gap I-III)
- **ISTA-Net:** I=31.85 > III=27.45 >> II=19.02 (4.40 dB gap I-III)
- **HATNet:** I=30.98 > III=29.78 >> II=19.40 (1.20 dB gap I-III, robust denoiser)

---

## Per-Image Results

### ISTA-Net (Pretrained, 9-layer, CS_ISTA_Net)

| Image | Scenario I | Scenario II | Scenario III | Gap I-II | Recovery II-III |
|-------|-----------|-------------|-------------|----------|----------------|
| Monarch | 32.54 | 19.90 | 27.70 | 12.64 | 7.80 |
| Parrots | 31.42 | 18.99 | 27.82 | 12.43 | 8.83 |
| Barbara | 27.84 | 19.02 | 25.53 | 8.83 | 6.51 |
| Boats | 32.91 | 19.26 | 27.89 | 13.65 | 8.63 |
| Cameraman | 28.61 | 19.31 | 26.16 | 9.30 | 6.85 |
| Fingerprint | 28.10 | 18.16 | 25.56 | 9.94 | 7.41 |
| Flinstones | 29.37 | 18.01 | 26.39 | 11.37 | 8.38 |
| Foreman | 38.23 | 18.20 | 29.68 | 20.03 | 11.49 |
| House | 35.70 | 19.23 | 29.21 | 16.46 | 9.98 |
| Lena256 | 32.30 | 19.67 | 27.96 | 12.63 | 8.30 |
| Peppers256 | 33.33 | 19.49 | 28.01 | 13.84 | 8.52 |

### FISTA-TV (Classical, 500 iterations)

| Image | Scenario I | Scenario II | Scenario III | Gap I-II | Recovery II-III |
|-------|-----------|-------------|-------------|----------|----------------|
| Monarch | 28.04 | 19.22 | 26.15 | 8.82 | 6.93 |
| Parrots | 27.40 | 18.53 | 26.18 | 8.87 | 7.65 |
| Barbara | 24.48 | 18.49 | 23.79 | 5.99 | 5.30 |
| Boats | 28.79 | 18.86 | 26.85 | 9.94 | 7.99 |
| Cameraman | 26.04 | 18.78 | 24.96 | 7.25 | 6.18 |
| Fingerprint | 23.08 | 17.33 | 22.58 | 5.75 | 5.25 |
| Flinstones | 24.64 | 17.18 | 23.59 | 7.46 | 6.41 |
| Foreman | 35.10 | 17.96 | 30.42 | 17.14 | 12.46 |
| House | 32.20 | 18.90 | 29.20 | 13.31 | 10.30 |
| Lena256 | 28.97 | 19.25 | 27.13 | 9.72 | 7.89 |
| Peppers256 | 29.95 | 19.11 | 27.51 | 10.83 | 8.40 |

### HATNet (Pretrained, Kronecker CS)

| Image | Scenario I | Scenario II | Scenario III | Gap I-II | Recovery II-III |
|-------|-----------|-------------|-------------|----------|----------------|
| Monarch | 30.91 | 20.36 | 29.77 | 10.56 | 9.41 |
| Parrots | 31.63 | 19.51 | 30.42 | 12.12 | 10.91 |
| Barbara | 30.72 | 19.55 | 29.45 | 11.16 | 9.89 |
| Boats | 31.04 | 19.58 | 29.76 | 11.46 | 10.18 |
| Cameraman | 29.68 | 19.77 | 28.68 | 9.91 | 8.91 |
| Fingerprint | 30.11 | 18.84 | 29.15 | 11.27 | 10.31 |
| Flinstones | 29.48 | 18.49 | 28.52 | 10.99 | 10.03 |
| Foreman | 32.80 | 18.34 | 31.31 | 14.46 | 12.97 |
| House | 32.17 | 19.34 | 30.80 | 12.83 | 11.47 |
| Lena256 | 31.22 | 19.96 | 29.89 | 11.26 | 9.93 |
| Peppers256 | 31.05 | 19.68 | 29.87 | 11.37 | 10.19 |

---

## Key Findings

1. **ISTA-Net clean baseline verified:** 31.85 dB matches published 31.84 dB (Set11 average at CR=25%), confirming correct model loading and architecture (non-plus BasicBlock without conv_D/conv_G).

2. **HATNet split forward validated:** External measurement injection (split forward) produces identical output to standard forward (max diff = 0.00), enabling mismatch injection between measurement and reconstruction.

3. **Consistent mismatch degradation:** All methods show 9.6-12.8 dB degradation under gain drift (Scenario I->II), with ISTA-Net most sensitive (12.83 dB) and FISTA-TV least (9.55 dB).

4. **Effective gain correction:** Dividing corrupted measurements by known gain vector recovers 7.7-10.4 dB (Scenario II->III), with HATNet showing the strongest recovery (90%) due to its robust transformer-based denoiser.

5. **ISTA-Net outperforms FISTA-TV by 3.8 dB** in Scenario I (31.85 vs 28.06), demonstrating the advantage of learned reconstruction over classical iterative methods at CR=25%.

6. **HATNet's robustness:** The I-III gap for HATNet (1.20 dB) is much smaller than for ISTA-Net (4.40 dB), suggesting HATNet's multi-stage ISTA with transformer denoisers effectively compensates for residual noise from gain correction.

---

## Comparison with Target Numbers

| Method | Scenario I | Target I | Match | Scenario II | Target II | Match |
|--------|-----------|----------|-------|-------------|-----------|-------|
| FISTA-TV | 28.06 | 28.39 | -0.33 | 18.51 | 18.96 | -0.45 |
| ISTA-Net | 31.85 | 31.84 | +0.01 | 19.02 | 18.93 | +0.09 |
| HATNet | 30.98 | 30.78 | +0.20 | 19.40 | 19.60 | -0.20 |

Scenarios I and II match targets within 0.5 dB for all methods. Scenario III shows larger deviations due to the exponential gain drift model: the gain correction (division by known gain vector) is highly effective, producing Scenario III closer to Scenario I than expected from the target numbers.

---

## Raw JSON Results

- `results/spc_validation_results.json` - Per-image detailed metrics (11 images x 3 methods x 3 scenarios)
- `results/spc_summary.json` - Aggregated statistics

---

**Report Generated:** 2026-02-16
**Framework Version:** InverseNet ECCV v4.0 (Pretrained Models)
**Script:** `papers/inversenet/scripts/validate_spc_inversenet.py`
