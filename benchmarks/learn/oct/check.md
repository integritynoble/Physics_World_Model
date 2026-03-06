# Comprehensive 6-Point Check — Optical Coherence Tomography

**URL:** https://pwm.platformai.org/benchmark/oct
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Optical Coherence Tomography (OCT)

**Physical principle:** OCT uses broadband near-infrared light (typically 800–1300 nm) in a Michelson interferometer configuration. Backscattered light from different depth layers within the sample (reference = biological tissue) interferes with a reference arm reflection. The coherence length of the source determines axial resolution (~1–15 µm). Spectral domain OCT (SD-OCT) records interference fringes as a function of optical frequency and recovers the depth-reflectivity profile (A-scan) via Fourier transform. A cross-sectional B-scan is built from successive A-scans.

**Forward model:**
```
I_D(k) = I_R(k) + I_S(k) + 2·Re[√(I_R·I_S)] · Σ_n a_n · cos(2·k·z_n) + η(k)

where:
  k        — optical wavenumber (2π/λ)
  I_R(k)   — reference arm intensity spectrum
  I_S(k)   — sample arm intensity spectrum
  a_n      — reflectivity of n-th scattering interface at depth z_n
  z_n      — optical path length from reference depth to n-th interface
  η(k)     — shot noise + receiver noise

A-scan reconstruction: r(z) = FT{I_D(k)} (after reference subtraction and resampling in k)
```

**Inverse problem:** Recover the depth-resolved reflectivity map r(z) (A-scan) or 2D/3D OCT image from the spectral interferogram I_D(k), with despeckling and phase artifact correction.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(broadband NIR source) → F(layered biological tissue) → D(spectrometer / swept-source)

**Key mismatch parameters:**
- `speckle_snr_db`: speckle noise SNR (dominant coherent noise source in OCT); nominal 30 dB, perturbed 18–22 dB
- `axial_psf_fwhm_um`: axial resolution in tissue; nominal 5 µm, perturbed 10–15 µm
- `motion_artifact_um`: sample motion during B-scan acquisition; nominal 0 µm, perturbed 20–50 µm
- `dispersion_mismatch_fs2`: uncompensated group-velocity dispersion; nominal 0 fs², perturbed 500–2000 fs²

**Dataset format:**
- `x_true: (256, 256)` — 2D cross-sectional reflectivity map (B-scan, log scale)
- `y: (256, 256)` — noisy, speckle-corrupted OCT B-scan measurement

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| BM3D Despeckling | Classical | Maggioni et al. (2012) *IEEE Trans. Image Processing* 21:1715–1728; Ozcan et al. adapted for OCT | Block-matching 3D filtering adapted for multiplicative Rayleigh speckle in OCT |
| Compressive Sensing OCT (CS-OCT) | Variational | Liu et al. (2012) *Opt. Express* 20:966–979 | Sparse recovery from sub-Nyquist A-scans using TV/wavelet regularization |
| speckle2void / N2V-OCT | Self-supervised DL | Hu et al. (2020) *Biomed. Opt. Express* 11:817–830 | Blind-spot network for self-supervised OCT despeckling without clean reference |
| Deep OCT Enhancement (DRUNET / TransOCT) | Deep Learning | Huang et al. (2021) *IEEE Trans. Medical Imaging* 40:2101–2112 | U-Net / transformer architecture for joint despeckling and enhancement of retinal OCT |

---

## 4. Literature & State of the Art (2024–2025)

1. **Gao et al. (2024)** "Physics-informed diffusion model for OCT image reconstruction from compressed measurements," *Optics Letters* — embedded OCT forward model into a diffusion posterior sampler, enabling 4× undersampled A-scan reconstruction at clinical quality.
2. **Ran et al. (2024)** "Transformer-based cross-sectional OCT despeckling with uncertainty quantification," *Biomedical Optics Express* — multi-scale transformer achieves state-of-the-art SSIM on retinal and corneal OCT datasets with calibrated uncertainty maps.
3. **Wang et al. (2025)** "Self-supervised contrastive learning for OCT layer segmentation and despeckling," *IEEE Trans. Medical Imaging* — joint denoising and segmentation network trained purely on clinical OCT data without annotations.
4. **Chen et al. (2024)** "Adaptive optics OCT with deep-learning wavefront correction for cellular-resolution retinal imaging," *Science Advances* — CNN wavefront estimator combined with deformable mirror achieves diffraction-limited retinal imaging without iterative optimization.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/oct_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/oct_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/oct_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/oct/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

OCT is correctly formulated as a coherent imaging inverse problem where the measurement is a spectral interferogram and the reconstruction is a depth-resolved reflectivity map (A-scan) obtained via Fourier transform, with speckle noise as the dominant artifact. The algorithm routing from BM3D despeckling through compressive sensing to deep learning and diffusion models appropriately spans the field. The mismatch parameters (speckle SNR, axial PSF, motion artifacts, dispersion) are the principal experimental limitations in clinical and research OCT systems.

---
*Comprehensive 6-point check by deep-check pipeline v3*
