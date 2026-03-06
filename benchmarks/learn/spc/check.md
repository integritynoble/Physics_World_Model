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
