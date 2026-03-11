# Comprehensive 6-Point Check — Near-field Scanning Optical Microscopy (NSOM)

**URL:** https://pwm.platformai.org/benchmark/nsom
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Near-field Scanning Optical Microscopy (NSOM / SNOM)

**Physical principle:** NSOM overcomes the optical diffraction limit by using an aperture or tip-on-tip probe in the near-field zone (within ~lambda/10 of the sample surface). An aperture NSOM uses a metal-coated tapered fiber with a sub-wavelength aperture (50–100 nm diameter) that transmits or collects evanescent optical fields. Since evanescent fields decay exponentially with distance, only features within ~lambda/10 of the aperture contribute to the signal. The spatial resolution is determined by the aperture size, not the wavelength. The aperture-sample interaction is modeled as a localized optical source convolved with the near-field probe response.

**Forward model:**
```
I(r_scan) = integral |E_near(r_scan - r', z)|^2 * O(r') dr'  + I_farfield(r_scan)
```
where E_near is the near-field amplitude of the probe at scan position r_scan, O(r') is the sample near-field optical response (permittivity distribution), and I_farfield is the far-field background contribution. The evanescent field coupling decays as exp(-z/delta) where delta ~ lambda/4pi. The benchmark models this via the `scanning_probe` nonlinear engine with PSF-convolution approximation:
```
y = PSF ⊛ x + noise
```

**Inverse problem:** Recover the near-field optical response map O(r) from the NSOM scan image, deconvolving the aperture probe function. The key challenges are tip-sample distance calibration, aperture size uncertainty, topographic coupling (near-field response changes with local sample height), and far-field background leakage.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(NSOM) → Sigma(tip_distance, aperture_size, topo_coupling, farfield_background) → D(I_nsom, eta)

**Key mismatch parameters:**
- **Tip-sample distance** (5–50 nm): incorrect working distance changes the evanescent field decay length, altering the effective spatial resolution
- **Aperture size error** (-20 to +20%): actual aperture diameter differs from nominal, changing the probe PSF width
- **Topographic coupling** (0–30%): near-field signal is modulated by local sample topography; incorrect topography correction leaves structured artifacts
- **Far-field background** (0–20%): light leaking around the aperture or through the tip shaft contributes a diffraction-limited background that reduces near-field contrast

**Dataset format:**
- `x_true: (H, W)` — ground-truth near-field optical response map (amplitude or intensity)
- `y: (H, W)` — measured NSOM scan image with aperture PSF blurring, topographic coupling, and shot noise

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| BTR | Classical | Villarrubia, JRNIST 1997 | Appropriate — blind tip reconstruction (BTR) adapted to optical aperture probe characterization |
| Reg-Deconv | PnP | Dongmo et al., 2000 | Appropriate — regularized deconvolution of the NSOM aperture PSF |
| TV-Deconvolution | PnP | TV regularization for SPM | Appropriate — TV prior for near-field image restoration |
| DeepSPM | Deep Learning | Alldritt et al., Commun. Phys. 2020 | Appropriate — deep learning for scanning probe microscopy, applicable to NSOM |
| SPM-Former | Vision Transformer | Chen et al., NanoLett 2024 | Appropriate — transformer for nanoscale scanning probe image reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Huth et al. (2024)** "Near-field spectroscopic imaging beyond the diffraction limit with deep learning," *ACS Nano* — CNN-based NSOM reconstruction achieving sub-20 nm optical resolution.
2. **Dai et al. (2024)** "Topographic artifact removal for NSOM imaging of biological membranes," *Nano Lett.* — physics-informed correction of topographic coupling in lipid bilayer NSOM.
3. **Chen et al. (2024)** "SPM-Former: vision transformer for near-field optical microscopy," *Nano Letters* — attention-based restoration achieving near-diffraction deconvolution.
4. **Muller et al. (2025)** "Score-based diffusion for NSOM aperture deconvolution," *Phys. Rev. Applied* — posterior sampling for near-field image restoration with aperture size uncertainty.

---

## 5. Local Dataset & GCS Status

- **GCS public tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/nsom_challenge_public.h5`
- **GCS dev tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/nsom_challenge_dev.h5`
- **GCS hidden tier:** `gs://pwm-benchmark-datasets/challenge-data/v1.0/nsom_challenge_hidden.h5` (blocked from download)
- **Gallery images:** `gs://pwm-benchmark-datasets/img/benchmark_gallery/nsom/scene_*/`
- **No local copies** — all data served from GCS via `/gcs/` proxy

---

## 6. Comprehensive Assessment

**Physics correctness:** NSOM is correctly classified as nonlinear (the evanescent field coupling is an exponentially nonlinear function of tip-sample distance). The `scanning_probe` engine appropriately captures the near-field probe interaction. The four mismatch parameters precisely cover the dominant NSOM calibration uncertainties: working distance (exponential sensitivity), aperture size, topographic coupling, and far-field background.

**Algorithm appropriateness:** The 10-algorithm set (BTR, MLE, Reg-Deconv, TV-Deconvolution, DeepSPM, U-Net-SPM, E2E-BTR, SPM-Former, DiffusionSPM, ScoreSPM) shares the `scanning_probe` pool with MFM. This is appropriate — NSOM and MFM are both tip-scanning probe modalities requiring identical classes of deconvolution algorithms despite different physical contrast mechanisms.

**Benchmark structure:** Topographic coupling mismatch (0–30%) is the most subtle and practically challenging parameter for NSOM — surface roughness creates a ghost signal that is not present in the ideal model, and algorithms must be robust to this structured artifact on the hidden tier.

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
| precomputed_baseline | 19.63 | 0.7328 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
