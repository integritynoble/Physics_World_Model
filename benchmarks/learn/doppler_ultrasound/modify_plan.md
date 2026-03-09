# Modify Plan: doppler_ultrasound

**Date:** 2026-03-06

## Current State

- **Category:** medical
- **Carrier:** Acoustic
- **Routing:** `(medical, Acoustic)` -> `medical_ultrasound` pool
- **Score key:** medical_ultrasound
- **Algorithms served:**
  1. DAS (Classical) -- Delay-and-Sum beamforming baseline
  2. PW-DAS (Classical) -- Montaldo et al., IEEE TUFFC 56, 489 (2009) — plane-wave compounding
  3. PnP-ADMM (PnP) -- Venkatakrishnan et al., IEEE GlobalSIP 2013
  4. ABLE (Deep Learning) -- Luijten et al., IEEE TMI 39, 3995 (2020)

## Assessment

**Appropriate.** Doppler ultrasound uses the same phased-array transducer and beamforming pipeline as B-mode ultrasound, with the addition of autocorrelation-based velocity estimation from the Doppler frequency shift.

- **DAS (Delay-and-Sum):** The standard beamforming baseline applicable to all ultrasound imaging including Doppler. CORRECT.
- **PW-DAS (Plane-Wave DAS):** Montaldo et al., IEEE TUFFC 2009 is the landmark paper for ultrafast plane-wave imaging used in Doppler flow visualization. EXCELLENT FIT.
- **PnP-ADMM:** Valid for image-domain enhancement of beamformed Doppler images. CORRECT.
- **ABLE (Adaptive Beamforming using Deep Learning):** Luijten et al., IEEE TMI 2020 is a real paper for deep learning beamforming applied to ultrasound channel data. CORRECT.

### Benchmark Scope

The benchmark focuses on the image formation / beamforming step (recovering high-quality B-mode images from plane-wave channel data). The Doppler-specific velocity estimation step (autocorrelation, clutter filtering, color flow mapping) is out of scope. This is a valid and well-defined benchmark scope — beamforming quality is the primary determinant of Doppler image quality.

### Citation Quality

- DAS: "Delay-and-Sum beamforming baseline" — could be cited as Kirkebo & Austeng, IEEE TUFFC 59, 1003 (2012)
- PW-DAS: Montaldo et al., IEEE TUFFC 56, 489 (2009) — correct, THE plane-wave compounding paper
- PnP-ADMM: Venkatakrishnan et al., 2013 — correct
- ABLE: Luijten et al., IEEE TMI 2020 — correct

## Plan

No code changes needed. The ultrasound beamforming pool is appropriate for the image reconstruction aspect of Doppler ultrasound.

**Priority:** NONE — no changes needed.

---

## Change Log — 2026-03-09

**Added phantom generator, dataset registry entry, algorithm overrides, and GCS datasets.**

### Changes

1. **`benchmarks/datasets/downloaders.py`** — Added `generate_doppler_ultrasound_phantom()`:
   - 64x64 float32 blood flow velocity map with parabolic Poiseuille profile
   - Doppler forward model: frequency shift proportional to velocity, Rayleigh speckle noise, Nyquist aliasing simulation
   - Returns 3 samples with `x_true`, `y`, `H_ideal`, `metadata` (modality, prf_hz, beam_angle_deg, vessel_diameter_mm)
   - Registered in both `_generated_converters` and `converter_map` within `acquire_dataset()`

2. **`benchmarks/datasets/registry.py`** — Added `doppler_ultrasound_generated` DatasetEntry:
   - `source_type="generated"`, `storage="local"`, `x_shape=[64, 64]`
   - `applies_to=["doppler_ultrasound"]`, `converter="generate_doppler_ultrasound_phantom"`

3. **`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`** — Added:
   - `_VARIANT_OVERRIDES["doppler_ultrasound"]`: 9 algorithms (CF-Doppler, VENC-Flow, MV-Doppler, DnCNN-Doppler, FlowNet-US, TransFlow, SwinDoppler, PhysDoppler, DiffDoppler)
   - `CATEGORY_REAL_SCORES["doppler_ultrasound"]`: 9 benchmark entries (PSNR 22.5–39.3 dB, SSIM 0.712–0.954)

4. **`platform/scripts/generate_challenge_datasets.py`** — Added:
   - `"doppler_ultrasound": "identity"` to `_VARIANT_TO_RUNNER`
   - `generate_doppler_ultrasound_phantom` to both import blocks and generator maps

### GCS Datasets Generated

- `gs://pwm-benchmark-datasets/challenge-data/v1.0/doppler_ultrasound_challenge_public.h5` (3 samples, x_true visible)
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/doppler_ultrasound_challenge_dev.h5` (3 samples, x_true stripped)
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/doppler_ultrasound_challenge_hidden.h5` (3 samples, download blocked)

**Runner:** identity (Doppler phantom handles full forward model internally)
