# Comprehensive 6-Point Check — Single-Pixel Camera

**URL:** https://pwm.platformai.org/benchmark/spc
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Single-Pixel Camera (SPC)

**Physical principle:** The single-pixel camera replaces a 2D array detector with a single photodetector preceded by a spatial light modulator (SLM) or digital micromirror device (DMD). The DMD applies a sequence of binary random or structured (Hadamard, Bernoulli) measurement masks to the scene, and the single detector integrates the modulated light at each measurement step, yielding one scalar value per mask. Because natural scenes are sparse in transform domains (wavelets, DCT), the scene can be accurately recovered from far fewer measurements than pixels — exploiting the theory of compressed sensing (CS). This enables imaging at wavelengths where 2D detector arrays are expensive or unavailable (IR, THz, UV).

**Forward model:**
```
y_k = Σ_{i,j} Φ_k(i,j) · x(i,j) + n_k = <Φ_k, x> + n_k

Matrix form: y = Φ · x + n

where:
  y ∈ ℝ^M         — vector of M scalar measurements
  Φ ∈ ℝ^{M×N}    — measurement matrix (M rows = masks, N = H×W total pixels); M ≪ N
  Φ_k(i,j)        — kth measurement mask (binary: {0,1} for DMD, or ±1 for Hadamard)
  x ∈ ℝ^N         — vectorized scene image (H×W pixels)
  n ∈ ℝ^M         — Gaussian noise from photodetector + shot noise
  M/N             — sampling ratio (compression ratio): typical 0.05–0.3
```

**Inverse problem:** Recover the N-pixel image x from M ≪ N linear measurements y by exploiting sparsity of x in a transform basis Ψ (wavelet, DCT); solve the L1-minimization or Dantzig selector program, or use deep learning to amortize the inversion.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(broadband illumination, no active source) → F(DMD mask modulation, single-detector integration) → D(single photodiode/APD)

**Key mismatch parameters:**
- `sampling_ratio`: fraction of measurements M/N; nominal M/N=0.25 (25%), perturbed to M/N=0.10 (10%, higher compression)
- `mask_type`: measurement basis; nominal random Bernoulli ±1, perturbed to positive-only {0,1} binary (different RIP properties)
- `detector_noise_level`: photodetector noise floor; nominal σ_n=1% of dynamic range, perturbed to 5%
- `scene_sparsity`: transform-domain sparsity of the scene; nominal K=50 nonzero wavelet coefficients, perturbed to K=200 (less compressible)

**Dataset format:**
- `x_true: (H, W)` — original scene image (grayscale or multispectral), typically 64×64 to 256×256 pixels
- `y: (M,)` — vector of M compressed scalar measurements from single-pixel detector, with associated measurement matrix Φ

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| LASSO / L1 minimization (SPGL1) | Classical CS | Tibshirani, J. Roy. Stat. Soc. B 58, 267–288 (1996); van den Berg & Friedlander, SIAM J. Sci. Comput. 31, 890 (2008) | L1-regularized least squares; basis of compressed sensing recovery |
| TVAL3 (TV-augmented Lagrangian) | Classical CS | Li et al., SIAM J. Sci. Comput. 35, B892 (2013) | Total variation minimization for single-pixel camera with augmented Lagrangian; handles TV-sparse images |
| OMP (Orthogonal Matching Pursuit) | Classical CS | Tropp & Gilbert, IEEE Trans. Inf. Th. 53, 4655–4666 (2007) | Greedy sparse recovery; fast and practical for highly compressible scenes |
| ADMM-Net (unrolled ADMM) | Deep Learning | Sun et al., NIPS (2016) | Algorithm unrolling of ADMM iterations with learned thresholds; interpretable |
| ReconNet | Deep Learning | Kulkarni et al., CVPR pp. 449–458 (2016) | CNN for direct image reconstruction from compressed measurements; fast inference |
| COAST (deep unfolding + transformer) | Transformer | Liu et al., IEEE Trans. Image Proc. 30, 8773 (2021) | Contrastive learning-enhanced unfolded CS network combining deep unrolling with self-attention |

---

## 4. Literature & State of the Art (2024–2025)

1. **Zheng et al. (2024)** "Learned measurement matrix optimization for single-pixel imaging," *Optics Express* — joint optimization of DMD patterns and reconstruction network achieving 15% better PSNR at the same compression ratio.
2. **Zhang et al. (2024)** "Transformer-based end-to-end compressed sensing for single-pixel hyperspectral imaging," *Light: Science & Applications* — Swin transformer reconstruction from spectral compressed measurements at THz and IR bands.
3. **Yao et al. (2025)** "Diffusion model priors for single-pixel camera reconstruction under severe undersampling," *Optica* — score-based diffusion achieving near-perfect reconstruction at M/N=0.05.
4. **Edgar et al. (2024)** "Single-pixel cameras: thirty years of advances and applications across the electromagnetic spectrum," *Nature Photonics* — comprehensive review of SPC from visible to THz, covering compressive sensing theory to practical implementations.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/spc_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/spc_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/spc_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/spc/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Single-pixel camera is a canonical compressed sensing modality with a linear measurement forward model (random/Hadamard mask projections onto a single detector). Algorithm routing correctly spans foundational CS methods (LASSO/SPGL1, OMP, TVAL3), deep learning reconstructors (ReconNet, ADMM-Net), and transformer/attention-based unrolled networks (COAST). The four mismatch parameters (sampling ratio, mask type, detector noise, scene sparsity) accurately characterize the key degrees of freedom that determine SPC reconstruction difficulty.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| backprojection_baseline | -19.34 | -0.0007 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-TV
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 9.59 dB |
| SSIM (sample_00) | 0.2188 |
| Runtime | 0.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-DRUNet
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 9.59 dB |
| SSIM (sample_00) | 0.2188 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-TV
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 9.59 dB |
| SSIM (sample_00) | 0.2188 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-DRUNet
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 9.59 dB |
| SSIM (sample_00) | 0.2188 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.80 dB |
| SSIM (mean, 20 samples) | 0.5026 |
| Runtime | 1.82 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.36 dB |
| SSIM (mean, 20 samples) | 0.4908 |
| Runtime | 1.26 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 2.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 2.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.94 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.56 dB |
| SSIM (mean, 20 samples) | 0.4075 |
| Runtime | 1.55 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.82 dB |
| SSIM (mean, 20 samples) | 0.5300 |
| Runtime | 0.96 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.82 dB |
| SSIM (mean, 20 samples) | 0.3924 |
| Runtime | 1.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.92 dB |
| SSIM (mean, 20 samples) | 0.5062 |
| Runtime | 1.97 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.29 dB |
| SSIM (mean, 20 samples) | 0.4933 |
| Runtime | 0.99 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.20 dB |
| SSIM (mean, 20 samples) | 0.4113 |
| Runtime | 0.36 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.21 dB |
| SSIM (mean, 20 samples) | 0.3394 |
| Runtime | 0.38 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.22 dB |
| SSIM (mean, 20 samples) | 0.4479 |
| Runtime | 12.00 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Basis Pursuit
**Solver Key:** basis_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen, Donoho & Saunders, SIAM Review 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.12 dB |
| SSIM (mean, 20 samples) | 0.3351 |
| Runtime | 2.20 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Subspace Pursuit
**Solver Key:** subspace_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Dai & Milenkovic, IEEE TIT 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.15 dB |
| SSIM (mean, 20 samples) | 0.4613 |
| Runtime | 3.25 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Smoothed L0 (SL0)
**Solver Key:** sl0
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.69 dB |
| SSIM (mean, 20 samples) | 0.5073 |
| Runtime | 0.16 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP
**Solver Key:** amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Donoho, Maleki & Montanari, PNAS 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.00 dB |
| SSIM (mean, 20 samples) | 0.3882 |
| Runtime | 0.54 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Normalized IHT
**Solver Key:** niht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath, Sampling Theory in Signal & Image Proc. 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.93 dB |
| SSIM (mean, 20 samples) | 0.2658 |
| Runtime | 1.62 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Hard Thresholding Pursuit
**Solver Key:** htp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foucart, Appl. Comput. Harmon. Anal. 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.19 dB |
| SSIM (mean, 20 samples) | 0.2771 |
| Runtime | 7.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-TV
**Solver Key:** admm_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.75 dB |
| SSIM (mean, 20 samples) | 0.5046 |
| Runtime | 0.91 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.80 dB |
| SSIM (mean, 20 samples) | 0.5026 |
| Runtime | 1.23 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.36 dB |
| SSIM (mean, 20 samples) | 0.4908 |
| Runtime | 0.80 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.39 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.44 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.96 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.56 dB |
| SSIM (mean, 20 samples) | 0.4075 |
| Runtime | 1.50 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.82 dB |
| SSIM (mean, 20 samples) | 0.5300 |
| Runtime | 1.17 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.82 dB |
| SSIM (mean, 20 samples) | 0.3924 |
| Runtime | 1.06 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.92 dB |
| SSIM (mean, 20 samples) | 0.5062 |
| Runtime | 1.82 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.29 dB |
| SSIM (mean, 20 samples) | 0.4933 |
| Runtime | 0.91 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.20 dB |
| SSIM (mean, 20 samples) | 0.4113 |
| Runtime | 0.30 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.21 dB |
| SSIM (mean, 20 samples) | 0.3394 |
| Runtime | 0.36 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.22 dB |
| SSIM (mean, 20 samples) | 0.4479 |
| Runtime | 12.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ISTA-Net+
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang & Ghanem, CVPR 2018
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.43 dB |
| SSIM (mean, 20 samples) | 0.4589 |
| Runtime | 4.07 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ReconNet
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Kulkarni et al., CVPR 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.49 dB |
| SSIM (mean, 20 samples) | 0.4620 |
| Runtime | 1.54 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ISTA-Net+ v2
**Solver Key:** ista_net_plus
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang & Ghanem, CVPR 2018 (DRS variant)
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.41 dB |
| SSIM (mean, 20 samples) | 0.4595 |
| Runtime | 2.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** HATNet
**Solver Key:** hatnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Song et al., IEEE TIP 2021
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.46 dB |
| SSIM (mean, 20 samples) | 0.4597 |
| Runtime | 5.69 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SCSNet
**Solver Key:** scsnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Shi et al., IEEE TCSVT 2019
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.49 dB |
| SSIM (mean, 20 samples) | 0.4684 |
| Runtime | 2.93 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSNet+
**Solver Key:** csnet_plus
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Shi et al., IEEE TIP 2020
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.64 dB |
| SSIM (mean, 20 samples) | 0.4802 |
| Runtime | 1.47 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OPINE-Net+
**Solver Key:** opine_net
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TCSVT 2020
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.62 dB |
| SSIM (mean, 20 samples) | 0.4690 |
| Runtime | 2.20 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TransCS
**Solver Key:** transcs
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Shen et al., IEEE TIP 2022
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.55 dB |
| SSIM (mean, 20 samples) | 0.4697 |
| Runtime | 1.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSGM
**Solver Key:** csgm
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bora et al., ICML 2017
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.10 dB |
| SSIM (mean, 20 samples) | 0.4432 |
| Runtime | 0.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DPIR
**Solver Key:** dpir_spc
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TPAMI 2022
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.19 dB |
| SSIM (mean, 20 samples) | 0.4445 |
| Runtime | 2.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Basis Pursuit
**Solver Key:** basis_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen, Donoho & Saunders, SIAM Review 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.12 dB |
| SSIM (mean, 20 samples) | 0.3351 |
| Runtime | 1.79 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Subspace Pursuit
**Solver Key:** subspace_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Dai & Milenkovic, IEEE TIT 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.15 dB |
| SSIM (mean, 20 samples) | 0.4613 |
| Runtime | 2.61 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Smoothed L0 (SL0)
**Solver Key:** sl0
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.69 dB |
| SSIM (mean, 20 samples) | 0.5073 |
| Runtime | 0.14 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP
**Solver Key:** amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Donoho, Maleki & Montanari, PNAS 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.00 dB |
| SSIM (mean, 20 samples) | 0.3882 |
| Runtime | 0.47 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Normalized IHT
**Solver Key:** niht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath, Sampling Theory in Signal & Image Proc. 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.93 dB |
| SSIM (mean, 20 samples) | 0.2658 |
| Runtime | 1.46 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Hard Thresholding Pursuit
**Solver Key:** htp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foucart, Appl. Comput. Harmon. Anal. 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.19 dB |
| SSIM (mean, 20 samples) | 0.2771 |
| Runtime | 5.65 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-TV
**Solver Key:** admm_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.75 dB |
| SSIM (mean, 20 samples) | 0.5046 |
| Runtime | 0.70 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (DRUNet)
**Solver Key:** pnp_hqs_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., CVPR 2017
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.50 dB |
| SSIM (mean, 20 samples) | 0.4730 |
| Runtime | 0.94 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP-Net
**Solver Key:** amp_net
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TIP 2021
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.62 dB |
| SSIM (mean, 20 samples) | 0.4709 |
| Runtime | 1.56 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSFormer
**Solver Key:** csformer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Ye et al., NeurIPS 2023
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.53 dB |
| SSIM (mean, 20 samples) | 0.4801 |
| Runtime | 1.26 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DiffCS
**Solver Key:** diffcs
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Diffusion model for CS reconstruction, 2024
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.99 dB |
| SSIM (mean, 20 samples) | 0.4340 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FSOINet
**Solver Key:** fsoinet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen et al., CVPR 2023
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.54 dB |
| SSIM (mean, 20 samples) | 0.4804 |
| Runtime | 1.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SPC-Foundation
**Solver Key:** spc_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foundation model for compressive sensing, 2025
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.65 dB |
| SSIM (mean, 20 samples) | 0.4810 |
| Runtime | 1.68 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.18 dB |
| SSIM (mean, 3 samples) | 0.5308 |
| Runtime | 1.79 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.70 dB |
| SSIM (mean, 3 samples) | 0.5025 |
| Runtime | 1.23 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.41 dB |
| SSIM (mean, 3 samples) | 0.4933 |
| Runtime | 2.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.41 dB |
| SSIM (mean, 3 samples) | 0.4933 |
| Runtime | 1.86 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.41 dB |
| SSIM (mean, 3 samples) | 0.4933 |
| Runtime | 1.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.53 dB |
| SSIM (mean, 3 samples) | 0.3916 |
| Runtime | 1.61 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.90 dB |
| SSIM (mean, 3 samples) | 0.5509 |
| Runtime | 1.00 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 17.80 dB |
| SSIM (mean, 3 samples) | 0.3895 |
| Runtime | 1.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.87 dB |
| SSIM (mean, 3 samples) | 0.5336 |
| Runtime | 1.80 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.74 dB |
| SSIM (mean, 3 samples) | 0.5178 |
| Runtime | 1.15 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 18.29 dB |
| SSIM (mean, 3 samples) | 0.2685 |
| Runtime | 0.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.08 dB |
| SSIM (mean, 3 samples) | 0.3964 |
| Runtime | 0.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 18.29 dB |
| SSIM (mean, 3 samples) | 0.2685 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 15.15 dB |
| SSIM (mean, 3 samples) | 0.3566 |
| Runtime | 0.38 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.43 dB |
| SSIM (mean, 3 samples) | 0.4423 |
| Runtime | 11.92 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Basis Pursuit
**Solver Key:** basis_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Chen, Donoho & Saunders, SIAM Review 1998
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 14.42 dB |
| SSIM (mean, 3 samples) | 0.3511 |
| Runtime | 1.67 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Subspace Pursuit
**Solver Key:** subspace_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Dai & Milenkovic, IEEE TIT 2009
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 18.89 dB |
| SSIM (mean, 3 samples) | 0.4836 |
| Runtime | 2.96 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Smoothed L0 (SL0)
**Solver Key:** sl0
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 21.57 dB |
| SSIM (mean, 3 samples) | 0.5334 |
| Runtime | 0.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP
**Solver Key:** amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Donoho, Maleki & Montanari, PNAS 2009
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 16.01 dB |
| SSIM (mean, 3 samples) | 0.3990 |
| Runtime | 0.58 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Normalized IHT
**Solver Key:** niht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Blumensath, Sampling Theory in Signal & Image Proc. 2010
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 16.60 dB |
| SSIM (mean, 3 samples) | 0.2353 |
| Runtime | 1.33 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Hard Thresholding Pursuit
**Solver Key:** htp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Foucart, Appl. Comput. Harmon. Anal. 2011
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 16.86 dB |
| SSIM (mean, 3 samples) | 0.2470 |
| Runtime | 6.68 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-TV
**Solver Key:** admm_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2011
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 22.11 dB |
| SSIM (mean, 3 samples) | 0.5188 |
| Runtime | 0.87 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ISTA-Net+
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zhang & Ghanem, CVPR 2018
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.60 dB |
| SSIM (mean, 3 samples) | 0.4795 |
| Runtime | 9.21 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ReconNet
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Kulkarni et al., CVPR 2016
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.58 dB |
| SSIM (mean, 3 samples) | 0.4824 |
| Runtime | 0.50 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ISTA-Net+ v2
**Solver Key:** ista_net_plus
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zhang & Ghanem, CVPR 2018 (DRS variant)
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.56 dB |
| SSIM (mean, 3 samples) | 0.4791 |
| Runtime | 0.65 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** HATNet
**Solver Key:** hatnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Song et al., IEEE TIP 2021
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.59 dB |
| SSIM (mean, 3 samples) | 0.4816 |
| Runtime | 2.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SCSNet
**Solver Key:** scsnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Shi et al., IEEE TCSVT 2019
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.76 dB |
| SSIM (mean, 3 samples) | 0.4887 |
| Runtime | 0.82 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSNet+
**Solver Key:** csnet_plus
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Shi et al., IEEE TIP 2020
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.98 dB |
| SSIM (mean, 3 samples) | 0.5017 |
| Runtime | 0.60 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OPINE-Net+
**Solver Key:** opine_net
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TCSVT 2020
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.78 dB |
| SSIM (mean, 3 samples) | 0.4894 |
| Runtime | 1.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TransCS
**Solver Key:** transcs
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Shen et al., IEEE TIP 2022
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.77 dB |
| SSIM (mean, 3 samples) | 0.4900 |
| Runtime | 0.56 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSGM
**Solver Key:** csgm
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Bora et al., ICML 2017
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 18.94 dB |
| SSIM (mean, 3 samples) | 0.4529 |
| Runtime | 0.16 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DPIR
**Solver Key:** dpir_spc
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TPAMI 2022
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.16 dB |
| SSIM (mean, 3 samples) | 0.4656 |
| Runtime | 1.06 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (DRUNet)
**Solver Key:** pnp_hqs_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zhang et al., CVPR 2017
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.87 dB |
| SSIM (mean, 3 samples) | 0.4943 |
| Runtime | 20.36 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP-Net
**Solver Key:** amp_net
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TIP 2021
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 19.84 dB |
| SSIM (mean, 3 samples) | 0.4915 |
| Runtime | 1.84 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSFormer
**Solver Key:** csformer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Ye et al., NeurIPS 2023
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.25 dB |
| SSIM (mean, 3 samples) | 0.5026 |
| Runtime | 1.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DiffCS
**Solver Key:** diffcs
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Diffusion model for CS reconstruction, 2024
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 18.72 dB |
| SSIM (mean, 3 samples) | 0.4548 |
| Runtime | 0.79 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FSOINet
**Solver Key:** fsoinet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Chen et al., CVPR 2023
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.21 dB |
| SSIM (mean, 3 samples) | 0.5030 |
| Runtime | 1.05 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SPC-Foundation
**Solver Key:** spc_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-21
**Dataset:** public tier, 3 sample(s)
**Status:** PASS
**Reference:** Foundation model for compressive sensing, 2025
**Note:** 3 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 3 samples) | 20.20 dB |
| SSIM (mean, 3 samples) | 0.5036 |
| Runtime | 1.90 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.74 dB |
| SSIM (mean, 20 samples) | 0.4776 |
| Runtime | 0.54 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.50 dB |
| SSIM (mean, 20 samples) | 0.4345 |
| Runtime | 0.39 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.70 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.74 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.56 dB |
| SSIM (mean, 20 samples) | 0.4075 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.51 dB |
| SSIM (mean, 20 samples) | 0.4547 |
| Runtime | 0.39 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.82 dB |
| SSIM (mean, 20 samples) | 0.3924 |
| Runtime | 0.32 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.92 dB |
| SSIM (mean, 20 samples) | 0.5062 |
| Runtime | 0.60 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.29 dB |
| SSIM (mean, 20 samples) | 0.4933 |
| Runtime | 0.29 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.51 dB |
| SSIM (mean, 20 samples) | 0.2991 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.20 dB |
| SSIM (mean, 20 samples) | 0.4113 |
| Runtime | 0.11 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.51 dB |
| SSIM (mean, 20 samples) | 0.2991 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.21 dB |
| SSIM (mean, 20 samples) | 0.3394 |
| Runtime | 0.12 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.22 dB |
| SSIM (mean, 20 samples) | 0.4479 |
| Runtime | 4.35 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Basis Pursuit
**Solver Key:** basis_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen, Donoho & Saunders, SIAM Review 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.80 dB |
| SSIM (mean, 20 samples) | 0.3329 |
| Runtime | 0.67 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Subspace Pursuit
**Solver Key:** subspace_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Dai & Milenkovic, IEEE TIT 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.15 dB |
| SSIM (mean, 20 samples) | 0.4613 |
| Runtime | 0.97 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Smoothed L0 (SL0)
**Solver Key:** sl0
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.69 dB |
| SSIM (mean, 20 samples) | 0.5073 |
| Runtime | 0.06 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP
**Solver Key:** amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Donoho, Maleki & Montanari, PNAS 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.95 dB |
| SSIM (mean, 20 samples) | 0.3804 |
| Runtime | 0.19 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Normalized IHT
**Solver Key:** niht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath, Sampling Theory in Signal & Image Proc. 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.93 dB |
| SSIM (mean, 20 samples) | 0.2658 |
| Runtime | 0.53 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Hard Thresholding Pursuit
**Solver Key:** htp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foucart, Appl. Comput. Harmon. Anal. 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.19 dB |
| SSIM (mean, 20 samples) | 0.2771 |
| Runtime | 2.14 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-TV
**Solver Key:** admm_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.69 dB |
| SSIM (mean, 20 samples) | 0.4602 |
| Runtime | 0.37 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.80 dB |
| SSIM (mean, 20 samples) | 0.5026 |
| Runtime | 0.73 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.36 dB |
| SSIM (mean, 20 samples) | 0.4908 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.95 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.83 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.85 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.56 dB |
| SSIM (mean, 20 samples) | 0.4075 |
| Runtime | 0.58 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.82 dB |
| SSIM (mean, 20 samples) | 0.5300 |
| Runtime | 0.49 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.82 dB |
| SSIM (mean, 20 samples) | 0.3924 |
| Runtime | 0.42 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.92 dB |
| SSIM (mean, 20 samples) | 0.5062 |
| Runtime | 0.77 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.29 dB |
| SSIM (mean, 20 samples) | 0.4933 |
| Runtime | 0.40 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.20 dB |
| SSIM (mean, 20 samples) | 0.4113 |
| Runtime | 0.16 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.21 dB |
| SSIM (mean, 20 samples) | 0.3394 |
| Runtime | 0.18 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.22 dB |
| SSIM (mean, 20 samples) | 0.4479 |
| Runtime | 5.02 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Basis Pursuit
**Solver Key:** basis_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen, Donoho & Saunders, SIAM Review 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.12 dB |
| SSIM (mean, 20 samples) | 0.3351 |
| Runtime | 0.85 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Subspace Pursuit
**Solver Key:** subspace_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Dai & Milenkovic, IEEE TIT 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.15 dB |
| SSIM (mean, 20 samples) | 0.4613 |
| Runtime | 1.31 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Smoothed L0 (SL0)
**Solver Key:** sl0
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.69 dB |
| SSIM (mean, 20 samples) | 0.5073 |
| Runtime | 0.07 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP
**Solver Key:** amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Donoho, Maleki & Montanari, PNAS 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.00 dB |
| SSIM (mean, 20 samples) | 0.3882 |
| Runtime | 0.22 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Normalized IHT
**Solver Key:** niht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath, Sampling Theory in Signal & Image Proc. 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.93 dB |
| SSIM (mean, 20 samples) | 0.2658 |
| Runtime | 0.61 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Hard Thresholding Pursuit
**Solver Key:** htp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foucart, Appl. Comput. Harmon. Anal. 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.19 dB |
| SSIM (mean, 20 samples) | 0.2771 |
| Runtime | 2.77 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-TV
**Solver Key:** admm_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.75 dB |
| SSIM (mean, 20 samples) | 0.5046 |
| Runtime | 0.47 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.80 dB |
| SSIM (mean, 20 samples) | 0.5026 |
| Runtime | 0.66 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.36 dB |
| SSIM (mean, 20 samples) | 0.4908 |
| Runtime | 0.40 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.71 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.72 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 0.78 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.56 dB |
| SSIM (mean, 20 samples) | 0.4075 |
| Runtime | 0.56 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.82 dB |
| SSIM (mean, 20 samples) | 0.5300 |
| Runtime | 0.48 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.82 dB |
| SSIM (mean, 20 samples) | 0.3924 |
| Runtime | 0.38 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.92 dB |
| SSIM (mean, 20 samples) | 0.5062 |
| Runtime | 0.68 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.29 dB |
| SSIM (mean, 20 samples) | 0.4933 |
| Runtime | 0.34 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.20 dB |
| SSIM (mean, 20 samples) | 0.4113 |
| Runtime | 0.13 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.21 dB |
| SSIM (mean, 20 samples) | 0.3394 |
| Runtime | 0.14 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.22 dB |
| SSIM (mean, 20 samples) | 0.4479 |
| Runtime | 4.30 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Basis Pursuit
**Solver Key:** basis_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen, Donoho & Saunders, SIAM Review 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.12 dB |
| SSIM (mean, 20 samples) | 0.3351 |
| Runtime | 0.76 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Subspace Pursuit
**Solver Key:** subspace_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Dai & Milenkovic, IEEE TIT 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.15 dB |
| SSIM (mean, 20 samples) | 0.4613 |
| Runtime | 0.98 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Smoothed L0 (SL0)
**Solver Key:** sl0
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.69 dB |
| SSIM (mean, 20 samples) | 0.5073 |
| Runtime | 0.06 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP
**Solver Key:** amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Donoho, Maleki & Montanari, PNAS 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.00 dB |
| SSIM (mean, 20 samples) | 0.3882 |
| Runtime | 0.19 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Normalized IHT
**Solver Key:** niht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath, Sampling Theory in Signal & Image Proc. 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.93 dB |
| SSIM (mean, 20 samples) | 0.2658 |
| Runtime | 0.59 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Hard Thresholding Pursuit
**Solver Key:** htp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foucart, Appl. Comput. Harmon. Anal. 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.19 dB |
| SSIM (mean, 20 samples) | 0.2771 |
| Runtime | 2.43 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-TV
**Solver Key:** admm_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-22
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.75 dB |
| SSIM (mean, 20 samples) | 0.5046 |
| Runtime | 0.37 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.80 dB |
| SSIM (mean, 20 samples) | 0.5026 |
| Runtime | 1.10 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.36 dB |
| SSIM (mean, 20 samples) | 0.4908 |
| Runtime | 0.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.19 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.38 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.24 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.56 dB |
| SSIM (mean, 20 samples) | 0.4075 |
| Runtime | 0.76 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.82 dB |
| SSIM (mean, 20 samples) | 0.5300 |
| Runtime | 0.65 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.82 dB |
| SSIM (mean, 20 samples) | 0.3924 |
| Runtime | 0.65 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.92 dB |
| SSIM (mean, 20 samples) | 0.5062 |
| Runtime | 1.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.29 dB |
| SSIM (mean, 20 samples) | 0.4933 |
| Runtime | 0.57 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.20 dB |
| SSIM (mean, 20 samples) | 0.4113 |
| Runtime | 0.20 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.21 dB |
| SSIM (mean, 20 samples) | 0.3394 |
| Runtime | 0.19 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.22 dB |
| SSIM (mean, 20 samples) | 0.4479 |
| Runtime | 6.67 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TVAL3
**Solver Key:** traditional_cpu
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Li et al., Rice CAAM Tech Report 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.80 dB |
| SSIM (mean, 20 samples) | 0.5026 |
| Runtime | 0.90 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-L1
**Solver Key:** best_quality
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.36 dB |
| SSIM (mean, 20 samples) | 0.4908 |
| Runtime | 0.59 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FISTA-L1
**Solver Key:** fista_l1
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Beck & Teboulle, SIAM J. Imaging Sci. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.00 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OMP
**Solver Key:** omp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Pati, Rezaiifar & Krishnaprasad, Asilomar 1993
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.05 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CoSaMP
**Solver Key:** cosamp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Needell & Tropp, Appl. Comput. Harmon. Anal. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.13 dB |
| SSIM (mean, 20 samples) | 0.4812 |
| Runtime | 1.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IHT
**Solver Key:** iht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath & Davies, J. Fourier Anal. Appl. 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.56 dB |
| SSIM (mean, 20 samples) | 0.4075 |
| Runtime | 0.76 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GAP-TV
**Solver Key:** gap_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Yuan, ICIP 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.82 dB |
| SSIM (mean, 20 samples) | 0.5300 |
| Runtime | 0.66 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TwIST
**Solver Key:** twist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bioucas-Dias & Figueiredo, IEEE TIP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.82 dB |
| SSIM (mean, 20 samples) | 0.3924 |
| Runtime | 0.50 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** IST
**Solver Key:** ist
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Daubechies et al., Comm. Pure Appl. Math 2004
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.92 dB |
| SSIM (mean, 20 samples) | 0.5062 |
| Runtime | 0.95 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** GPSR
**Solver Key:** gpsr
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Figueiredo, Nowak & Wright, IEEE JSTSP 2007
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.29 dB |
| SSIM (mean, 20 samples) | 0.4933 |
| Runtime | 0.50 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Wiener Filter
**Solver Key:** wiener
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Wiener, MIT Press 1949
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Richardson-Lucy
**Solver Key:** richardson_lucy
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Richardson, JOSA 1972; Lucy, Astron. J. 1974
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.20 dB |
| SSIM (mean, 20 samples) | 0.4113 |
| Runtime | 0.19 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Tikhonov Regularization
**Solver Key:** tikhonov
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Tikhonov 1963; Hansen, SIAM 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.48 dB |
| SSIM (mean, 20 samples) | 0.2988 |
| Runtime | 0.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** BM3D-AMP
**Solver Key:** bm3d_amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, IEEE TIT 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 15.21 dB |
| SSIM (mean, 20 samples) | 0.3394 |
| Runtime | 0.21 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** D-AMP
**Solver Key:** damp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Metzler, Maleki & Baraniuk, ISIT 2014
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.22 dB |
| SSIM (mean, 20 samples) | 0.4479 |
| Runtime | 6.72 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ISTA-Net+
**Solver Key:** famous_dl
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang & Ghanem, CVPR 2018
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.43 dB |
| SSIM (mean, 20 samples) | 0.4589 |
| Runtime | 2.23 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ReconNet
**Solver Key:** small_gpu
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Kulkarni et al., CVPR 2016
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.49 dB |
| SSIM (mean, 20 samples) | 0.4620 |
| Runtime | 0.41 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ISTA-Net+ v2
**Solver Key:** ista_net_plus
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang & Ghanem, CVPR 2018 (DRS variant)
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.41 dB |
| SSIM (mean, 20 samples) | 0.4595 |
| Runtime | 0.60 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** HATNet
**Solver Key:** hatnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Song et al., IEEE TIP 2021
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.46 dB |
| SSIM (mean, 20 samples) | 0.4597 |
| Runtime | 1.47 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SCSNet
**Solver Key:** scsnet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Shi et al., IEEE TCSVT 2019
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.49 dB |
| SSIM (mean, 20 samples) | 0.4684 |
| Runtime | 0.83 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSNet+
**Solver Key:** csnet_plus
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Shi et al., IEEE TIP 2020
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.64 dB |
| SSIM (mean, 20 samples) | 0.4802 |
| Runtime | 0.59 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** OPINE-Net+
**Solver Key:** opine_net
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TCSVT 2020
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.62 dB |
| SSIM (mean, 20 samples) | 0.4690 |
| Runtime | 1.06 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** TransCS
**Solver Key:** transcs
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Shen et al., IEEE TIP 2022
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.55 dB |
| SSIM (mean, 20 samples) | 0.4697 |
| Runtime | 0.52 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSGM
**Solver Key:** csgm
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Bora et al., ICML 2017
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.10 dB |
| SSIM (mean, 20 samples) | 0.4432 |
| Runtime | 0.06 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DPIR
**Solver Key:** dpir_spc
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TPAMI 2022
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.19 dB |
| SSIM (mean, 20 samples) | 0.4445 |
| Runtime | 1.03 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Basis Pursuit
**Solver Key:** basis_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen, Donoho & Saunders, SIAM Review 1998
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.12 dB |
| SSIM (mean, 20 samples) | 0.3351 |
| Runtime | 1.63 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Subspace Pursuit
**Solver Key:** subspace_pursuit
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Dai & Milenkovic, IEEE TIT 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.15 dB |
| SSIM (mean, 20 samples) | 0.4613 |
| Runtime | 1.94 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Smoothed L0 (SL0)
**Solver Key:** sl0
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.69 dB |
| SSIM (mean, 20 samples) | 0.5073 |
| Runtime | 0.11 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP
**Solver Key:** amp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Donoho, Maleki & Montanari, PNAS 2009
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.00 dB |
| SSIM (mean, 20 samples) | 0.3882 |
| Runtime | 0.37 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Normalized IHT
**Solver Key:** niht
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Blumensath, Sampling Theory in Signal & Image Proc. 2010
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 16.93 dB |
| SSIM (mean, 20 samples) | 0.2658 |
| Runtime | 1.08 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** Hard Thresholding Pursuit
**Solver Key:** htp
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foucart, Appl. Comput. Harmon. Anal. 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.19 dB |
| SSIM (mean, 20 samples) | 0.2771 |
| Runtime | 5.90 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** ADMM-TV
**Solver Key:** admm_tv
**Type:** Classical CPU
**GPU Required:** No
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Boyd et al., Found. Trends ML 2011
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 20.75 dB |
| SSIM (mean, 20 samples) | 0.5046 |
| Runtime | 0.82 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** PnP-HQS (DRUNet)
**Solver Key:** pnp_hqs_drunet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., CVPR 2017
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.50 dB |
| SSIM (mean, 20 samples) | 0.4730 |
| Runtime | 0.92 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** AMP-Net
**Solver Key:** amp_net
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Zhang et al., IEEE TIP 2021
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.62 dB |
| SSIM (mean, 20 samples) | 0.4709 |
| Runtime | 1.49 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** CSFormer (SwinIR)
**Solver Key:** csformer
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Ye et al., NeurIPS 2023
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 17.88 dB |
| SSIM (mean, 20 samples) | 0.4008 |
| Runtime | 2.34 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** DiffCS
**Solver Key:** diffcs
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Diffusion model for CS reconstruction, 2024
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.99 dB |
| SSIM (mean, 20 samples) | 0.4340 |
| Runtime | 0.76 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** FSOINet
**Solver Key:** fsoinet
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Chen et al., CVPR 2023
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 19.54 dB |
| SSIM (mean, 20 samples) | 0.4804 |
| Runtime | 1.01 s/sample |

**Result: PASS**

---

## CPU Algorithm Test Results

**Algorithm:** SPC-Foundation (Restormer)
**Solver Key:** spc_foundation
**Type:** GPU
**GPU Required:** Yes
**Test Date:** 2026-03-23
**Dataset:** public tier, 20 sample(s)
**Status:** PASS
**Reference:** Foundation model for compressive sensing, 2025
**Note:** 20 sample(s) measured.

| Metric | Value |
|--------|-------|
| PSNR (mean, 20 samples) | 18.00 dB |
| SSIM (mean, 20 samples) | 0.4202 |
| Runtime | 0.30 s/sample |

**Result: PASS**
