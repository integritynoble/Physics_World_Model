# Modify Plan: shearography

## Current State (After Fix)

- **Category:** industrial_inspection
- **Sub-category pool:** interferometric_ndt (shearography-specific override)
- **Algorithms:** Goldstein MCF, PnP-Phase, ShearNet, PhaseFormer

## Assessment

Algorithms are now domain-appropriate.

The previous pool (TSR, PnP-ADMM, DefectNet, LSTM-NDT) was drawn from the generic `industrial_inspection` category. TSR (Thermographic Signal Reconstruction) was particularly mismatched — it is a technique for pulsed thermography time-domain analysis (polynomial fitting of IR decay curves) with no relevance to optical interference fringe analysis.

The new pool is fully specific to speckle-shearing interferometry:
- **Goldstein MCF** (1988): The canonical branch-cut minimum cost flow algorithm for 2D phase unwrapping — used in every production shearography system.
- **PnP-Phase**: Plug-and-play with a phase-aware denoiser prior, adapted for complex interferometric measurements.
- **ShearNet**: CNN trained end-to-end on shearographic fringe images for defect detection (Feng et al., 2019).
- **PhaseFormer**: Vision transformer operating on phase sequences for temporal phase-stepping shearography.

## Verdict

No further code changes needed.
