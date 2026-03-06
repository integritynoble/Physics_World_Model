# Modify Plan -- ivus

## Current State

- **Category:** medical
- **Carrier:** Acoustic
- **Score key:** medical_ultrasound (routed via `_CARRIER_ROUTING[("medical", "Acoustic")]` -> `"medical_ultrasound"`)
- **Algorithms:**
  1. DAS (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Goudarzi et al., 2020
  3. ABLE (Deep Learning) -- Luijten et al., IEEE TMI 2020
  4. MU-Net (Deep Learning) -- Hyun et al., IEEE TUFFC 2022

## Assessment

**Appropriate.** IVUS (Intravascular Ultrasound) is correctly routed via the `_CARRIER_ROUTING` system from `("medical", "Acoustic")` to the `medical_ultrasound` algorithm pool. The algorithms are well-suited:

- **DAS (Delay-and-Sum)** is the standard beamforming algorithm for ultrasound, including IVUS. Directly applicable.
- **PnP-ADMM** (Goudarzi et al., 2020) is a published plug-and-play method for ultrasound image reconstruction. Appropriate.
- **ABLE** (Luijten et al., IEEE TMI 2020) is a published adaptive beamforming and learning method for ultrasound. Appropriate.
- **MU-Net** (Hyun et al., IEEE TUFFC 2022) is a published deep learning method for ultrasound beamforming. Appropriate.

The carrier-based routing correctly identifies IVUS as an ultrasound modality and assigns ultrasound-specific algorithms rather than generic medical (CT-oriented) algorithms. This is a good example of the routing system working as intended.

Note: Both ABLE and MU-Net have type "Deep Learning" (no Transformer class), which means 2 of the 4 are deep learning. This is a minor non-uniformity but acceptable since these are the real published methods in the field.

## Current Algorithm Count (updated 2026-03-06)

Full pool (14 algorithms): DAS, DAS-CF, PW-DAS, PnP-ADMM, PnP-TV, ABLE, MU-Net, Phase-ADMM-Net, UltrasoundFormer, BeamFormer, AttentionBeam, BeamDATA, DiffUS, ScoreUS.

**Status:** PASS — check.md written 2026-03-06

## Recommendation

No code changes needed.
