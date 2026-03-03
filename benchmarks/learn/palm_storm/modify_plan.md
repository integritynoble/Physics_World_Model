# Modify Plan: palm_storm

## Current State
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

PALM/STORM is a single-molecule localization microscopy (SMLM) technique. The reconstruction task is fundamentally different from conventional deconvolution microscopy: SMLM requires **localization** of individual fluorophore blinking events from sparse, stochastic frames, not deconvolution of a blurred image. The current algorithms (Richardson-Lucy, CARE, Restormer) are deconvolution/denoising methods that do not perform single-molecule localization at all.

The check.md leaderboard methods (DECODE, ANNA-PALM, SPARCOM, SOFI) are correct domain-specific algorithms for SMLM, but the algorithm catalog returns generic microscopy deconvolution algorithms instead. This is a **mismatch** -- the catalog algorithms are inappropriate.

**Appropriate PALM/STORM algorithms would be:**
- ThunderSTORM (Classical) -- Ovesny et al., Bioinformatics 2014
- DECODE (Deep Learning) -- Speiser et al., Nat. Methods 2021
- ANNA-PALM (Deep Learning) -- Ouyang et al., Nat. Biotechnol. 2018
- Deep-STORM (Deep Learning) -- Nehme et al., Optica 2018

## Required Changes

Add a carrier routing rule or variant override in `_algorithm_catalog.py` for `palm_storm` to use SMLM-specific algorithms instead of generic microscopy deconvolution. Recommended approach: add a `_VARIANT_OVERRIDES["palm_storm"]` entry with localization-specific algorithms (ThunderSTORM, DECODE, ANNA-PALM, Deep-STORM or SPARCOM).

### Files to modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` -- add variant override for `palm_storm`
