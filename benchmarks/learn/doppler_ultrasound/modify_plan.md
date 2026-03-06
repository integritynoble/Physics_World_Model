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
