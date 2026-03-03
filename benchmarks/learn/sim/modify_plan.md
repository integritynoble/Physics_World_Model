# Modify Plan: sim

## Current State (After Fix)
- **Category:** microscopy
- **Sub-category pool:** fluorescence_micro (SIM-specific)
- **Algorithms:** [Wiener-SIM, PnP-SIM, DL-SIM, SIMformer]

## Assessment
Algorithms are now domain-appropriate.

The previous generic microscopy pool (Richardson-Lucy, PnP-FISTA, CARE, Restormer) was replaced with four SIM-specific algorithms that correctly model the structured illumination reconstruction problem:
- **Wiener-SIM** — canonical frequency-domain Wiener filter with OTF extension (Gustafsson 2000)
- **PnP-SIM** — plug-and-play ADMM exploiting structured illumination geometry
- **DL-SIM** — end-to-end deep learning on multi-frame SIM stacks (Christensen et al., Nat. Methods 2021)
- **SIMformer** — vision transformer adapted for SIM pattern separation

## Verdict
No further code changes needed.
