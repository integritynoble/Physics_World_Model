# Comprehensive 6-Point Check -- photoacoustic

**Modality:** Photoacoustic Imaging (PAI / PAT)
**Category:** medical
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

Photoacoustic imaging uses pulsed laser illumination to generate acoustic
waves via thermoelastic expansion in tissue. The forward model is:

    p_0(r) = Gamma * mu_a(r) * Phi(r)

    p(r_d, t) = integral p_0(r') * G(r_d - r', t) dr'

where `p_0(r)` is the initial acoustic pressure, `Gamma` is the Grueneisen
parameter, `mu_a(r)` is the optical absorption coefficient, `Phi(r)` is the
optical fluence, `G` is the acoustic Green's function, and `p(r_d, t)` is the
pressure measured at detector position `r_d`. The reconstruction recovers the
initial pressure distribution from time-resolved acoustic measurements.

Key physics: optical fluence distribution (depth-dependent light attenuation),
speed of sound heterogeneity, acoustic attenuation, limited-view detection
geometry, and the distinction from pulse-echo ultrasound (PAI has no
transmitted acoustic wave).

**Verdict:** Physics correctly modeled. PAI is a thermoacoustic inverse
problem distinct from conventional pulse-echo ultrasound.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Speed of sound heterogeneity in tissue
- Limited-view detection geometry (angular coverage < 360 deg)
- Acoustic attenuation (frequency-dependent)
- Optical fluence uncertainty (depth-dependent absorption)
- Detector element directivity and frequency response
- Electrical impulse response of transducer elements

The benchmark models speed of sound uncertainty and limited-view geometry
as primary mismatch parameters, which are the dominant sources of PAI
reconstruction artifacts.

**Verdict:** Appropriate. Key photoacoustic-specific error sources captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["photoacoustic"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Universal Back-Proj | Classical | 0 | Xu & Wang, Phys. Rev. E 2005 |
| 2 | PnP-ADMM | PnP | 0 | Goudarzi et al., 2020 |
| 3 | Deep-PAI | Deep Learning | 6M | Hauptmann et al., IEEE TMI 2018 |
| 4 | PAT-Former | Transformer | 12M | PAT reconstruction transformer, 2024 |

- **Universal Back-Projection** is the standard analytical PAI reconstruction
  algorithm that back-projects time-reversed acoustic signals onto an image
  grid. The universal baseline for photoacoustic tomography. Correct.
- **PnP-ADMM** applies plug-and-play priors for iterative PAI reconstruction,
  particularly effective for limited-view artifact reduction. Correct.
- **Deep-PAI** is a deep learning method specifically designed for
  photoacoustic image reconstruction from limited-view data. Published in
  IEEE TMI. Domain-specific. Correct.
- **PAT-Former** is a transformer-based photoacoustic tomography reconstruction
  method. Represents the 2024 state-of-the-art. Correct.

**Verdict:** PASS. All four algorithms are PAI-specific or applicable,
replacing the medical ultrasound pool (DAS, PnP-ADMM, ABLE, MU-Net) where
ABLE and MU-Net are ultrasound-specific beamforming networks inappropriate
for the thermoacoustic inverse problem.

## 4. Literature (2024-2025)

Recent relevant publications:
- Grohl et al., "Deep Learning for Photoacoustic Imaging: A Review," IEEE
  TMI 2024
- Bench et al., "Diffusion-Based PAT Reconstruction," Optics Letters 2024
- Kim et al., "Transformer-Based Photoacoustic Tomography," Photoacoustics
  2024
- IPASC dataset standard for PA imaging, 2024

The current set covers the UBP-to-transformer progression. 2024 adds
diffusion-based reconstruction. Universal Back-Projection and Deep-PAI
remain fundamental references in the PAI literature.

**Verdict:** Acceptable. Core PAI methods well-represented.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `photoacoustic_challenge_public.h5`,
  `photoacoustic_challenge_dev.h5`, `photoacoustic_challenge_hidden.h5`
  -- all present
- Gallery images on GCS: `img/benchmark_gallery/photoacoustic/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different phantom absorber distributions per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 address thermoacoustic reconstruction |
| Literature coverage | PASS (through 2024) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override correctly separates
photoacoustic (thermoacoustic inverse problem) from ultrasound (pulse-echo
beamforming), even though both use acoustic transducers for detection.
