# Modify Plan: photoacoustic

## Current State (Updated 2026-03-03)

- **Category:** medical
- **Carrier:** Acoustic
- **Score key:** medical_ultrasound (via carrier routing)
- **Variant override:** Yes -- `_VARIANT_OVERRIDES["photoacoustic"]` in `_algorithm_catalog.py`
- **Algorithms assigned (via override):**
  1. Universal Back-Proj (Classical) -- Xu & Wang, Phys. Rev. E 2005
  2. PnP-ADMM (PnP) -- Goudarzi et al., 2020
  3. Deep-PAI (Deep Learning) -- Hauptmann et al., IEEE TMI 2018
  4. PAT-Former (Transformer) -- PAT reconstruction transformer, 2024

## Assessment

**PASS -- domain-specific override applied and verified.**

The variant override replaces the medical ultrasound pool (DAS, PnP-ADMM,
ABLE, MU-Net) with photoacoustic-specific reconstruction algorithms. While PAI
shares acoustic detection hardware with ultrasound, the physics is fundamentally
different: PAI reconstructs initial pressure distributions from laser-generated
acoustic waves (thermoacoustic inverse problem), whereas ultrasound performs
pulse-echo beamforming. ABLE and MU-Net are ultrasound beamforming networks
not designed for PAI.

## Changes Applied

- Added `_VARIANT_OVERRIDES["photoacoustic"]` with four PAI-specific algorithms
- Universal Back-Projection: standard analytical PAI reconstruction
- PnP-ADMM: iterative reconstruction with learned denoisers for limited-view PAI
- Deep-PAI: deep learning for photoacoustic image reconstruction
- PAT-Former: transformer-based photoacoustic tomography reconstruction

## Remaining Items

None. No further code changes needed.

### Files modified:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
