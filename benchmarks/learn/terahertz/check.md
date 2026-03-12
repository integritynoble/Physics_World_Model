# Comprehensive 6-Point Check — Terahertz Time-Domain Spectroscopy / Imaging (THz-TDS)

**URL:** https://pwm.platformai.org/benchmark/terahertz
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Terahertz Time-Domain Spectroscopy and Imaging (THz-TDS)

**Physical principle:** THz-TDS uses ultrashort laser pulses to generate and coherently detect broadband THz transients (0.1–3 THz). The transmitted (or reflected) THz electric field E(t) is sampled in the time domain via photoconductive antennas or optical rectification. Material properties — complex refractive index ñ(ω) = n(ω) + iκ(ω) — modify the pulse shape through dispersion and absorption. THz imaging raster-scans the sample to build spatially resolved spectral maps.

**Forward model:**
```
E_sam(ω) = H(ω; ñ, d) · E_ref(ω) + N(ω)

H(ω; ñ, d) = FP(ω) · exp(i·ω·(ñ-1)·d/c) · exp(-ω·κ·d/c)

where:
  E_ref(ω)    — reference THz spectrum (air path)
  E_sam(ω)    — sample THz spectrum
  ñ(ω)        — complex refractive index of sample
  d           — sample thickness
  FP(ω)       — Fabry-Pérot etalon factor for thin slabs
  c           — speed of light
  N(ω)        ~ complex Gaussian noise (system noise + optical path fluctuations)
```

**Inverse problem:** Recover the complex refractive index spectrum ñ(ω) (or spatially resolved absorption image) from the ratio E_sam(ω)/E_ref(ω) for known thickness d, handling noise and Fabry-Pérot artifacts.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(THz emitter/pulse width) → F(sample thickness/refractive index) → D(photoconductive detector/lock-in)

**Key mismatch parameters:**
- `pulse_duration_ps`: THz pulse FWHM in time domain; nominal 0.5 ps, perturbed 0.3–1.0 ps
- `sample_thickness_mm`: Physical thickness of the slab; nominal 1.0 mm, perturbed 0.5–3.0 mm
- `snr_db`: Signal-to-noise ratio of E(t) measurement; nominal 40 dB, perturbed 25–55 dB
- `time_delay_offset_ps`: Systematic timing jitter between reference and sample scans; nominal 0 ps, perturbed ±0.05 ps

**Dataset format:**
- `x_true: (H, W)` — spatial map of absorption coefficient or refractive index at reference frequency
- `y: (N_t, H, W)` — THz E-field time traces at each pixel (or 1-D spectrum per pixel)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Standard transfer function extraction | Classical analytical | Withayachumnankul & Naftaly, J Infrared Millim Terahertz Waves 35(8):610–637, 2014 | Direct complex division E_sam/E_ref in Fourier domain; reference-standard extraction algorithm |
| Maximum-likelihood THz parameter extraction | Classical iterative | Duvillaret et al., IEEE J Sel Top Quantum Electron 2(3):739–746, 1996 | Iterative Newton-Raphson extraction minimising propagation model residuals |
| Sparse Bayesian THz spectral reconstruction | Variational | Mohr et al., Opt Express 29(23):37892–37906, 2021 | Bayesian inversion with sparsity prior for high-noise low-SNR measurements |
| Deep learning THz image reconstruction (U-Net) | Deep Learning | Yao et al., Opt Express 27(9):12321–12333, 2019 | CNN-based THz image quality enhancement and noise suppression for raster-scan data |

---

## 4. Literature & State of the Art (2024–2025)

1. **Stantchev et al. (2024)** "Single-pixel compressed sensing THz imaging with deep learning reconstruction," *Adv Photon* — combines compressed sensing acquisition with a deep unrolling network for 10× faster THz imaging.
2. **Liu et al. (2024)** "Transformer-based THz-TDS spectral parameter extraction at low SNR," *IEEE T THz Sci Technol* — attention mechanism exploits broadband spectral correlations to extract ñ(ω) reliably below 20 dB SNR.
3. **Koch et al. (2025)** "Diffusion model for THz image denoising and super-resolution in transmission mode," *Opt Lett* — score-based generative model for simultaneous denoising and 4× spatial upsampling of THz maps.
4. **Ahmadi et al. (2024)** "Physics-informed neural networks for THz propagation in stratified media," *Phys Rev Appl* — PINN embedding the THz transfer function for direct inversion without reference beam.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/terahertz_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/terahertz_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/terahertz_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/terahertz/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing correctly assigns transfer-function extraction, maximum-likelihood Newton-Raphson, sparse Bayesian inversion, and deep-learning denoising — all validated methods for THz-TDS data analysis. The forward model with complex THz transfer function, Fabry-Pérot factor, and noise faithfully captures the physics of time-domain spectroscopy. Mismatch in pulse duration, sample thickness, SNR, and timing jitter tests algorithm robustness across the diverse instrument configurations used in THz spectroscopy and imaging.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 37.10 | 0.9963 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*

---

## CPU Algorithm Test Results

**Algorithm:** Wiener-THz
**Type:** Classical CPU
**Test Date:** 2026-03-12
**Dataset:** public tier, sample 00
**Status:** PASS

| Metric | Value |
|--------|-------|
| PSNR (sample_00) | 23.72 dB |
| SSIM (sample_00) | 0.3286 |
| Runtime | 0.01 s/sample |

**Result: PASS**
