# Modify Plan: ocean_color

## Current State (After Fix)
- **Category:** remote_sensing
- **Sub-category pool:** remote_sensing (ocean color override)
- **Algorithms:** [Gordon AC, MUMM, OC-Net, AquaFormer]

## Assessment
Algorithms are now domain-appropriate.

The previous pool routed remote_sensing + Photon to the generic computational pool, which was acceptable, but the automated QA check showed that in practice the ocean_color page was displaying SAR-specific algorithms (Matched Filter, SAR-DRN, SAR-ViT) that are entirely inappropriate for passive optical ocean color remote sensing. The replacement algorithms are ocean-color-native:
- **Gordon AC** — Gordon and Wang dark pixel atmospheric correction, the canonical ocean color algorithm (Gordon & Wang, Appl. Opt. 1994); uses NIR dark pixel assumption to estimate aerosol contribution
- **MUMM** — MUMM (Management Unit of the North Sea Mathematical Models) iterative AC for turbid Case-2 coastal waters where the dark pixel assumption fails (Ruddick et al., RSE 2000)
- **OC-Net** — neural network atmospheric correction and bio-optical inversion (Fan et al., RSE 2021; OC-SMART framework)
- **AquaFormer** — transformer-based multispectral ocean color retrieval spanning multiple satellite sensors

## Verdict
No further code changes needed.
