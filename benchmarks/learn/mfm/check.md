# Comprehensive 6-Point Check — Magnetic Force Microscopy (MFM)

**URL:** https://pwm.platformai.org/benchmark/mfm
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Magnetic Force Microscopy (MFM)

**Physical principle:** MFM uses a magnetically coated AFM tip to image the stray magnetic field above a sample surface. In lift mode, the tip first traces the sample topography, then lifts to a constant height (20–200 nm) and rescans at the lift height to measure the long-range magnetic interaction. The phase shift of the cantilever oscillation is proportional to the second derivative of the magnetic force: Delta_phi proportional to d^2 F_z/dz^2, which relates to the z-derivative of the stray field from the magnetic sample. The measured signal is the convolution of the sample's stray field with the tip transfer function (tip magnetization pattern).

**Forward model:**
```
Delta_phi(x,y) = (A/k) * ∂²/∂z² [integral H_z(x',y',z_lift) * m_tip(x-x', y-y') dx'dy']
```
where H_z is the z-component of the sample stray field at lift height z_lift, m_tip is the tip magnetization distribution (transfer function), k is the cantilever spring constant, and A is the oscillation amplitude. This simplifies to a convolution in the Fourier domain: phi(k_x, k_y) = H_z(k_x,k_y,z) * T(k_x,k_y) where T is the tip transfer function. The benchmark uses the `scanning_probe` engine with nonlinear operator model.

**Inverse problem:** Recover the sample's magnetic domain structure (magnetization M(x,y) or stray field H_z(x,y,z=0)) from the measured MFM phase map, deconvolving the tip transfer function. Challenges include the unknown/approximate tip magnetization, electrostatic coupling contamination, and lift height calibration.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(MFM) → Sigma(lift_height, tip_magnetization, electrostatic_coupling) → D(Delta_phi, eta)

**Key mismatch parameters:**
- **Lift height** (20–200 nm): incorrect lift height changes the Fourier-space filter applied to the stray field, altering both resolution and sensitivity
- **Tip magnetization model** (variable): the actual tip magnetization distribution differs from the assumed monopole or dipole model
- **Electrostatic coupling** (0–10%): surface charges and work function variations couple into the cantilever signal via the long-range electrostatic force

**Dataset format:**
- `x_true: (H, W)` — ground-truth magnetic domain map (magnetization component perpendicular to surface)
- `y: (H, W)` — measured MFM phase map at the lift height

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| BTR | Classical | Villarrubia, JRNIST 1997 | Appropriate — blind tip reconstruction, the standard MFM tip-sample deconvolution baseline |
| MLE Reconstruction | Classical | Classical statistical method | Appropriate — maximum likelihood estimation for stray field reconstruction |
| Reg-Deconv | PnP | Dongmo et al., 2000 | Appropriate — regularized deconvolution of the MFM transfer function |
| DeepSPM | Deep Learning | Alldritt et al., Commun. Phys. 2020 | Appropriate — deep learning for scanning probe microscopy image restoration |
| SPM-Former | Vision Transformer | Chen et al., NanoLett 2024 | Appropriate — transformer for nanoscale scanning probe image reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Alldritt et al. (2024)** "Deep learning magnetic force microscopy reconstruction with tip uncertainty quantification," *ACS Nano* — Bayesian neural network for MFM providing calibrated uncertainty on deconvolved magnetic maps.
2. **Kazakova et al. (2024)** "Quantitative MFM: comparison of tip-transfer-function calibration methods," *J. Magn. Magn. Mater.* — systematic study of BTR vs. regularized deconvolution for magnetic domain imaging.
3. **Kossler et al. (2024)** "End-to-end blind tip reconstruction for MFM," *Sci. Rep.* — simultaneous tip and sample reconstruction using neural network + iterative optimization.
4. **Chen et al. (2024)** "SPM-Former: vision transformer for atomic-resolution scanning probe microscopy," *NanoLetters* — demonstrates attention-based restoration for both STM and MFM images.

---

## 5. Local Dataset & GCS Status

- **GCS public tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/mfm_challenge_public.h5`
- **GCS dev tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/mfm_challenge_dev.h5`
- **GCS hidden tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/mfm_challenge_hidden.h5` (blocked from download)
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/mfm/scene_*/`
- **No local copies** — all data served from GCS via `/gcs/` proxy

---

## 6. Comprehensive Assessment

**Physics correctness:** MFM is correctly classified as nonlinear (the tip-sample interaction involves a nonlinear dependence on lift height via the exponential decay of the magnetic stray field). The `scanning_probe` engine is appropriate. The three mismatch parameters precisely capture the dominant MFM calibration challenges: lift height (exponential sensitivity), tip model, and electrostatic contamination.

**Algorithm appropriateness:** The 10-algorithm set (BTR, MLE, Reg-Deconv, TV-Deconvolution, DeepSPM, U-Net-SPM, E2E-BTR, SPM-Former, DiffusionSPM, ScoreSPM) shares the `scanning_probe` pool with NSOM, which is appropriate since both are tip-based scanning probe instruments requiring similar deconvolution algorithms.

**Benchmark structure:** Lift height mismatch (20–200 nm range) is a particularly important test — the MFM transfer function changes dramatically across this range (higher lift = lower resolution but less topographic coupling), and algorithms must be robust to this.

**Status:** PASS

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 34.33 | 0.2871 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
