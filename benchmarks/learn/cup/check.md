# Comprehensive 6-Point Check — Compressed Ultrafast Photography (CUP)

**URL:** https://pwm.platformai.org/benchmark/cup
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Compressed Ultrafast Photography (CUP)

**Physical principle:** CUP achieves single-shot 2D+time imaging at 10^10 fps by encoding temporal information spatially using a compressed sensing framework. A digital micromirror device (DMD) applies a pseudo-random spatial code to the scene; a streak camera then temporally disperses the coded image along one axis, integrating the entire event in a single readout. Each pixel in the streak image contains a superposition of different time frames weighted by the spatial code, allowing compressed sensing recovery of the full spatio-temporal datacube. The technique was demonstrated by Gao et al. (Nature 2014) for capturing light-in-flight, shock wave dynamics, and laser ablation at 100 billion fps.

**Forward model:**
```
CUP acquisition:
  E(x,y) = ∫_0^T [C(x,y) · I(x, y - v·t, t)] dt   [streak camera integration]

Discrete compressed sensing model:
  y = M * Σ_t [ D_t ⊗ C · x_t ] + n

where:
  x_t ∈ R^{H × W}     — scene intensity at time t (ground truth per frame)
  C ∈ {0,1}^{H × W}   — random spatial code from DMD
  D_t                  — temporal shear operator (displacement = v·t pixels per frame)
  M                    — spatial undersampling / streak camera mask
  Σ_t                  — temporal summation (integration by streak tube)
  y ∈ R^{H × W_s}     — measured streak camera image (compressed)
  n                    — CCD readout noise
```

**Inverse problem:** Recover the 3D spatio-temporal datacube {x_t}_{t=1..T} from the single 2D compressed streak image y by solving the compressed sensing recovery problem, exploiting sparsity priors or learned neural representations.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** M(DMD code) → Σ(streak camera integration) → D(CCD)

**Key mismatch parameters:**
- `dmd_encoding_error` (d_e): DMD pixel misfiring / code pattern error; nominal 0.0, perturbed 0.4 (relative)
- `streak_sweep_calibration` (s_s): streak camera sweep rate calibration; nominal 0.0, perturbed 1.0 (calibration offset)
- `temporal_spatial_coupling` (t_c): cross-coupling between temporal shear and spatial code; nominal 0.0, perturbed 2.0

**Dataset format:**
- `x_true: (H, W, T)` — ground truth spatio-temporal video cube (T frames at ultrafast rate)
- `y: (H, W_streak)` — compressed streak camera measurement (single 2D image)
- `H_ideal: (H*W_streak, H*W*T)` — CUP forward operator (DMD code + streak dispersion + summation)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| TwIST | Classical CS | Bioucas-Dias & Figueiredo, IEEE TIP 2007 | Two-step iterative shrinkage/thresholding; standard CS solver used in original CUP paper |
| Temporal Filtering | Classical | — | Temporal matched filtering using known streak PSF |
| PnP-FFDNet | Plug-and-Play | Yuan et al., Opt. Lett. 2020 | PnP with FFDNet denoiser for snapshot compressive imaging; directly applied to CUP |
| CUP-Net | Deep Learning | Parker et al., Appl. Phys. Lett. 2021 | Neural network designed specifically for CUP reconstruction |
| AL-DL | Hybrid | Yao et al., Photon. Res. 2021 | Augmented Lagrangian + deep learning; hybrid model-based/data-driven CUP recovery |
| UltraFormer | Transformer | — | Transformer architecture for ultrafast imaging spatio-temporal reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Lossless-CUP** (Li et al., Nat. Commun. 2023 / extended 2024): High-fidelity CUP with lossless encoding; removes DMD diffraction losses; demonstrates 10^11 fps for photoacoustic wave tracking.
2. **T-CUP** (Ndao et al., Light Sci. Appl. 2020 / applied 2024): Trillion-fps CUP imaging; extended theoretical framework with deep learning reconstruction surpassing TwIST at extreme compression ratios.
3. **Learned CUP forward model** (2024): End-to-end differentiable CUP system optimising DMD code design for scene class; co-design of measurement matrix and reconstruction network.
4. **Diffusion model for ultrafast imaging** (2025): Score-based posterior sampling for CUP; handles the severe underdetermination of CUP reconstruction with physically-consistent uncertainty quantification.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cup_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cup_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/cup_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/cup/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the dedicated `ultrafast` category pool (11 methods: TwIST, Temporal Filtering, PnP-FFDNet, PnP-ADMM, CUP-Net, Temporal-U-Net, AL-DL, Unfolded-CUP, UltraFormer, DiffusionUltrafast, ScoreUltrafast). TwIST (Bioucas-Dias & Figueiredo 2007) was used in the original CUP paper (Gao et al., Nature 2014), confirming excellent domain alignment. CUP-Net and AL-DL are specifically designed for CUP reconstruction. The three mismatch parameters (DMD encoding error, streak sweep calibration, temporal-spatial coupling) address the principal CUP system calibration uncertainties. No code changes are required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
