# Modify Plan -- ceus

**Date:** 2026-03-03
**Category:** medical | **Carrier:** Acoustic | **Score key:** medical_ultrasound

## Current Algorithms (from catalog)

| # | Algorithm | Type          | Source                          |
|---|-----------|---------------|---------------------------------|
| 1 | DAS       | Classical     | Analytical baseline             |
| 2 | PnP-ADMM | PnP           | Goudarzi et al., 2020           |
| 3 | ABLE      | Deep Learning | Luijten et al., IEEE TMI 2020   |
| 4 | MU-Net    | Deep Learning | Hyun et al., IEEE TUFFC 2022    |

## Assessment

### Are algorithms domain-appropriate?
YES. CEUS (Contrast-Enhanced Ultrasound) is an ultrasound technique, and the carrier-based routing correctly sends (medical, Acoustic) to the `medical_ultrasound` pool. All four algorithms are real, published ultrasound beamforming/reconstruction methods:

- DAS (Delay-and-Sum): THE standard ultrasound beamforming algorithm. Correct and essential baseline.
- PnP-ADMM: Goudarzi et al., 2020 -- PnP approach for ultrasound image reconstruction. Correct.
- ABLE (Adaptive Beamforming using deep LEarning): Luijten et al., IEEE TMI 2020 -- deep learning beamformer for ultrasound. Correct and well-cited.
- MU-Net: Hyun et al., IEEE TUFFC 2022 -- U-Net for ultrasound image quality improvement. Correct.

The CEUS-specific aspect (microbubble contrast agent detection, super-resolution localization microscopy) is a downstream task beyond beamforming. The benchmark correctly focuses on the image reconstruction step.

### Are citations correct?
YES. All four citations are accurate:
- DAS: "Analytical baseline" -- standard label for the foundational beamforming method
- PnP-ADMM: Goudarzi et al., 2020 -- correct reference for PnP in ultrasound
- ABLE: Luijten et al., IEEE TMI 2020 -- correct, well-known paper
- MU-Net: Hyun et al., IEEE TUFFC 2022 -- correct

### Other issues
- check.md reports US-Transformer and PnP-DRUNet but the actual catalog shows MU-Net and PnP-ADMM. The check.md is stale.
- Two algorithms (ABLE and MU-Net) are both typed as "Deep Learning" -- there is no Transformer-type algorithm in this pool. This is a minor diversity gap but acceptable given the ultrasound domain.

## Plan

No code changes needed. The medical_ultrasound pool provides four well-established, correctly-cited ultrasound reconstruction algorithms. The carrier-based routing (medical, Acoustic) -> medical_ultrasound works correctly for CEUS.
