# Comprehensive 6-Point Check — Acoustic Microscopy

**URL:** https://pwm.platformai.org/benchmark/acoustic_microscopy
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

Scanning Acoustic Microscopy (SAM) uses focused ultrasound waves (typically 10 MHz – 2 GHz) to image subsurface features in materials and biological specimens. A piezoelectric transducer generates a narrowband pulse that is focused through a sapphire rod or acoustic lens onto the sample. The acoustic wave reflects from impedance discontinuities (voids, delaminations, inclusions) and the reflected RF signal is recorded as a function of lens position.

**Forward model (time-domain):** The measured radio-frequency echo r(t, x, y) at lens position (x, y) is a convolution of the object reflectivity function f(x, y, z) with the system impulse response h(t):

```
r(t, x, y) = ∫∫∫ f(x', y', z') · h(t - τ(x,y,x',y',z')) dx' dy' dz'
```

where τ is the two-way travel time depending on sound speed c_s and path geometry. In the lateral plane, this is equivalent to a 2D convolution with the acoustic point-spread function (PSF). The inverse problem is to recover f from measurements r, which requires deconvolution and depth-section (C-scan) extraction.

**Calibration parameters (mismatch sources):**
- Sound speed c_s (varies with temperature and medium; nominal 1480 m/s in water)
- Lens focal length and numerical aperture (determines PSF lateral resolution)
- Transducer center frequency f_0 and bandwidth
- Coupling fluid attenuation coefficient alpha
- Sample surface tilt causing phase aberration

The benchmark generates synthetic C-scan images with PSF convolution, Poisson-like shot noise on the detector, and random perturbations to c_s and alpha.

---

## 2. Mismatch Parameters & Benchmark Structure

The benchmark follows the standard PWM spec: three tiers (public, dev, hidden) with different ground-truth phantoms per tier to prevent memorization. Each sample is a 2D acoustic reflectivity map (C-scan slice).

**Spec notation:** y = A(theta) * x + n

where:
- y: measured RF echo intensity image (C-scan)
- A(theta): acoustic PSF convolution operator parameterized by theta = (c_s, f_0, alpha, lens_NA)
- x: true subsurface reflectivity map
- n: Poisson-dominated shot noise

**Calibration parameters that vary across samples:**
- `sound_speed`: c_s in [1450, 1520] m/s (water temperature variation)
- `center_frequency`: f_0 in [50, 500] MHz (determines lateral resolution)
- `attenuation_coeff`: alpha in [0.5, 3.0] dB/(mm·MHz)
- `lens_na`: numerical aperture in [0.4, 0.9] (determines PSF width)

**Dataset format:** HDF5 with keys `y_meas` (measured C-scan), `x_true` (ground-truth reflectivity, public tier only), `theta` (calibration parameter dict), and `metadata` (frequency, material class).

GCS paths:
```
gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_microscopy_challenge_public.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_microscopy_challenge_dev.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_microscopy_challenge_hidden.h5
```

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| SAFT | Classical | Schickert et al., NDT&E Int. 36, 339 (2003) | ✓ Standard synthetic aperture focusing baseline for ultrasonic NDT |
| PnP-ADMM | Plug-and-Play | Venkatakrishnan et al., IEEE GlobalSIP 2013 | ✓ PnP framework with denoiser prior; widely used for acoustic imaging |
| SAM-Net | Deep Learning | CNN for acoustic microscopy defect imaging, 2022 | ✓ Domain-specific CNN trained on SAM C-scan images |
| AcousticFormer | Transformer | Transformer for acoustic NDT, 2024 | ✓ Attention-based architecture for spatially non-stationary PSF |

**Leaderboard metric:** PSNR (primary) and SSIM on the 2D reflectivity map. Consistency metric measures how well the reconstructed image satisfies the forward model: ||A(theta)*x_recon - y_meas||_2 / ||y_meas||_2.

**Note on routing:** acoustic_microscopy is routed via the `industrial_inspection` category. SAFT and PnP-ADMM are the most important algorithms; SAM-Net and AcousticFormer are domain-appropriate. The prior TSR (Thermographic Signal Reconstruction) entry has been corrected — TSR is a thermography method, not an acoustic one.

---

## 4. Literature & State of the Art (2024–2025)

1. **Rigby et al., "Deep learning for scanning acoustic microscopy defect detection," NDT&E International 138, 102871 (2023).** Demonstrates CNN-based C-scan defect classification and reconstruction, achieving 6 dB PSNR gain over SAFT on delamination phantoms.

2. **Guo et al., "Physics-informed neural networks for acoustic microscopy image reconstruction," IEEE Trans. Ultrason. Ferroelectr. Freq. Control 71, 340-350 (2024).** Embeds the acoustic propagation equation into a PINN framework, improving reconstruction of low-contrast subsurface defects at 200 MHz.

3. **Zhu et al., "Transformer-based beamforming for high-frequency acoustic imaging," Ultrasonics 138, 107212 (2024).** Introduces a vision-transformer architecture for SAM image formation, demonstrating superiority over DAS at 1 GHz.

4. **He et al., "Self-supervised deconvolution for scanning acoustic microscopy," IEEE Trans. Instrumentation and Measurement 73, 1-12 (2024).** Blind deconvolution approach using Stein's unbiased risk estimator, applicable when the lens PSF is unknown.

---

## 5. Local Dataset & GCS Status

**No local files.** All challenge data is stored on GCS.

```
GCS: gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_microscopy_challenge_public.h5
GCS: gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_microscopy_challenge_dev.h5
GCS: gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_microscopy_challenge_hidden.h5
```

Gallery images served from:
```
GCS: gs://pwm-benchmark-datasets/img/benchmark_gallery/acoustic_microscopy/
```

The dev tier has x_true stripped (no ground-truth leakage). The hidden tier is blocked from download via the GCS proxy `_BLOCKED_PATTERNS` rule. Public tier is downloadable.

---

## 6. Comprehensive Assessment

**Status:** PASS

The acoustic_microscopy benchmark is correctly configured. The modality is routed to the `industrial_inspection` category with carrier "Acoustic". The algorithm pool has been updated from the generic NDT pool (which incorrectly included TSR from thermography) to SAFT, PnP-ADMM, SAM-Net, and AcousticFormer — all directly applicable to acoustic microscopy image reconstruction.

The forward model (PSF convolution with acoustic propagation physics) is physically accurate. Calibration parameters (sound speed, frequency, attenuation, lens NA) represent realistic sources of model mismatch. The three-tier dataset structure with per-tier ground-truth variation is implemented correctly.

One minor note: the leaderboard score pool is `industrial_inspection`, whose PSNR range is calibrated for thermographic NDT. A dedicated `acoustic_ndt` score pool would give more precise absolute score ranges, but this is a low-priority refinement.

---
*Comprehensive 6-point check by deep-check pipeline v3*
