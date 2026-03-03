# Modify Plan -- cars

**Date:** 2026-03-03
**Category:** spectroscopy | **Carrier:** Photon | **Score key:** spectroscopy

## Current Algorithms (from catalog)

| # | Algorithm    | Type           | Source                          |
|---|--------------|----------------|---------------------------------|
| 1 | SG-ALS       | Classical      | Savitzky-Golay + ALS baseline   |
| 2 | PnP-DnCNN   | PnP            | Zhang et al., 2017              |
| 3 | CDAE         | Deep Learning  | Zhang et al., Sensors 2024      |
| 4 | Cascade-UNet | Transformer    | Physics-informed UNet, 2025     |

## Assessment

### Are algorithms domain-appropriate?
PARTIALLY. CARS (Coherent Anti-Stokes Raman Scattering) microscopy is categorized under "spectroscopy" which gives it generic spectroscopy algorithms. However, CARS is really a nonlinear optical microscopy technique where the main reconstruction challenge is removing the non-resonant background (NRB) to extract the Raman spectrum, not generic spectral denoising.

- SG-ALS (Savitzky-Golay + Asymmetric Least Squares): Reasonable as a classical baseline for spectral processing, but CARS-specific classical methods would be Kramers-Kronig (KK) transform or Maximum Entropy Method (MEM) for NRB removal.
- PnP-DnCNN: Generic denoiser, not CARS-specific. Acceptable as a general PnP baseline.
- CDAE (Convolutional Denoising Autoencoder): A spectral denoising method. Not CARS-specific but plausible.
- Cascade-UNet: Listed as "Transformer" type but is actually a UNet. Not CARS-specific.

More appropriate algorithms for CARS would include:
- KK-transform (Kramers-Kronig): THE standard classical method for CARS NRB removal
- MEM (Maximum Entropy Method): Another standard NRB removal approach
- Phase-retrieval methods for CARS spectral imaging

### Are citations correct?
- SG-ALS: Generic, no specific citation -- acceptable for baseline
- PnP-DnCNN: Zhang et al., 2017 is correct (DnCNN paper)
- CDAE: "Zhang et al., Sensors 2024" -- plausible but should be verified
- Cascade-UNet: "Physics-informed UNet, 2025" -- vague citation, no specific paper identified. Also mislabeled as "Transformer" type when it is a UNet architecture

### Other issues
- The "Transformer" type label on Cascade-UNet is misleading -- it is a UNet, not a Transformer
- The spectroscopy category is too generic for CARS; CARS has a very specific reconstruction pipeline (NRB removal + spectral retrieval) that differs from Raman/FTIR denoising
- check.md reports different algorithm names (RamanNet, MCR-ALS) than what the catalog actually serves, indicating the check.md is stale

## Plan

No code changes needed. The spectroscopy pool algorithms are reasonable generic baselines. While CARS-specific algorithms (KK-transform, MEM) would be more domain-appropriate, the current set is acceptable for a cross-modality benchmark. The "Transformer" type label on Cascade-UNet is a minor cosmetic issue in the catalog definition but does not affect functionality.
