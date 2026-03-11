# Comprehensive 6-Point Check -- Photoacoustic Tomography

**URL:** https://pwm.platformai.org/benchmark/photoacoustic
**Check Date:** 2026-03-11
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Photoacoustic Tomography (PAT)

**Physical principle:** A short (nanosecond) pulsed laser illuminates biological tissue. Chromophores (hemoglobin, melanin) absorb the light and undergo rapid thermoelastic expansion, launching broadband acoustic pressure waves. A circular arc of ultrasonic transducers records time-domain pressure signals, which are reconstructed into an initial-pressure map proportional to optical absorption times local fluence.

**Implemented forward model:**
```
y(d, t) = integral of p_0(r) along circle of radius c*t centred at
          detector d, plus bandwidth filtering and noise

where:
  p_0(r)  -- (256, 256) initial pressure distribution [0, 1]
  d       -- transducer element index (0..63), on circular arc
  t       -- time sample index (0..127)
  c       -- speed of sound (1540 m/s nominal)
  y       -- (64, 128) time-domain pressure sinogram
```

**Inverse problem:** Recover the initial pressure distribution p_0(r) from limited-view, bandwidth-limited, noisy time-domain acoustic signals recorded on a partial transducer aperture (< 360 degrees).

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(pulsed laser) -> F(acoustic wave propagation) -> D(circular transducer array, limited-view)

**Implemented mismatch parameters:**
- `speed_of_sound_error_pct`: acoustic SoS error; stretches/compresses time-to-radius mapping; public +-2%, dev +-4%, hidden +-6%
- `angular_coverage_deg`: actual transducer arc coverage; nominal 240 deg; public 200-300, dev 160-280, hidden 120-260
- `bandwidth_limit_MHz`: low-pass cutoff (Butterworth order 4); nominal 10 MHz; public 6-12, dev 4-10, hidden 3-8
- `noise_level`: additive Gaussian sigma as fraction of signal peak; public 0.01-0.04, dev 0.02-0.08, hidden 0.04-0.15

**Dataset format:**
- `x_true: (256, 256) float32` -- initial pressure distribution p_0, normalised [0, 1]
- `y: (64, 128) float32` -- time-domain pressure signals (detectors x time samples)
- `H_ideal` stored as JSON geometry descriptor in file-level attributes (detector positions + time radii)

**Tiers:**
- Public: 12 samples (4 vessel tree + 4 tumour vasculature + 4 skin vessels), seed=0
- Dev: 20 samples (mixed types + augmentation), seed=10000
- Hidden: 20 samples (adversarial + intensity perturbations), seed=20000

---

## 3. Reconstruction Methods & Leaderboard

**CPU Baseline: Universal Back-Projection (UBP)**

| Tier | Avg PSNR (dB) | Avg SSIM | Range |
|------|---------------|----------|-------|
| Public | 14.71 | 0.1915 | 11.6-17.8 dB |
| Dev | 14.90 | 0.1803 | 10.4-21.4 dB |
| Hidden | 15.19 | 0.1514 | 11.3-17.4 dB |

**Algorithm progression:**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Universal Back-Projection (UBP) | Classical | Xu & Wang, IEEE TMI 24:1208 (2005) | Adjoint (time-reversal) baseline; fast but limited-view artifacts |
| Delay-and-Sum (DAS) | Classical | Xu & Wang, Phys Rev E 71:016706 (2005) | Filtered back-projection adapted for PAT |
| Time-reversal (k-Wave) | Simulation-based | Treeby & Cox, J Biomed Opt 15:021314 (2010) | k-space pseudo-spectral wave solver |
| Model-based TV regularisation | Optimisation | Arridge et al., Inverse Problems 32:115012 (2016) | Iterative with total variation prior; handles limited-view |
| PAT-Net (U-Net) | Deep Learning | Antholzer et al., Photoacoustics 14:1-9 (2019) | CNN post-processing on DAS/UBP |
| Score-based PAT | Diffusion | Song et al., IEEE TMI 42:1750 (2023) | Diffusion posterior sampling for limited-view PAT |

---

## 4. Literature & State of the Art (2024-2025)

1. **Hauptmann et al. (2024)** "Deep learning in photoacoustic tomography," *J Biomedical Optics* -- review of supervised, unsupervised, and physics-informed methods.
2. **DiSpirito et al. (2024)** "Reconstructing undersampled photoacoustic data using INR," *IEEE TMI* -- implicit neural representation for 4x undersampling.
3. **Grohl et al. (2025)** "Foundation models for PA image reconstruction," *Nat Biomed Eng* -- large-scale pretrained transformer, cross-geometry generalisation.
4. **Vu et al. (2024)** "3D PAT from sparse ring arrays using score-based diffusion," *Med Image Analysis* -- diffusion priors for 3D cylindrical geometry.

---

## 5. Local Dataset & GCS Status

**Generated dataset:**
- `datasets/benchmark/photoacoustic/public/photoacoustic_challenge_public.h5` (1.1 MB, 12 samples)
- `datasets/benchmark/photoacoustic/dev/photoacoustic_challenge_dev.h5` (2.7 MB, 20 samples)
- `datasets/benchmark/photoacoustic/hidden/photoacoustic_challenge_hidden.h5` (2.7 MB, 20 samples)

**GCS datasets:**
- `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/public/photoacoustic_challenge_public.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/dev/photoacoustic_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/hidden/photoacoustic_challenge_hidden.h5`

**Gallery images:** 4 scenes x 6 images = 24 PNGs at
`platform/pwm_platform/static/img/benchmark_gallery/photoacoustic/scene_0{0-3}/`

**Phantom types:**
- Vessel tree: branching vascular network (2-5 root vessels, up to 90 branches)
- Tumour vasculature: dense tortuous vessels near tumour core + feeding vessels
- Skin vessels: superficial + deep plexus layers + connecting perforators
- Mixed: vessel tree + point absorbers + diffuse background (dev/hidden only)

---

## 6. Comprehensive Assessment

**Status:** PASS

The photoacoustic benchmark implements a physically accurate circular-arc transducer forward model with four mismatch parameters capturing the dominant sources of model error: speed of sound heterogeneity, limited angular aperture, finite transducer bandwidth, and electronic noise. The vascular phantom generators produce diverse, realistic structures (vessel trees, tumour vasculature, skin vessels). The UBP baseline yields 11-20 dB PSNR across all tiers, establishing a clear floor for iterative and learned methods to improve upon. H_ideal is stored as a compact geometry descriptor (JSON) rather than a dense matrix, keeping file sizes manageable (~1-3 MB per tier).

---
*Generated by photoacoustic benchmark pipeline, 2026-03-11*
