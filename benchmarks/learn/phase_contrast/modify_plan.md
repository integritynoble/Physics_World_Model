# Modify Plan: phase_contrast

## Current State (After Fix)
- **Category:** microscopy
- **Sub-category pool:** optical_medical (phase imaging)
- **Algorithms:** [TIE Solver, DPC-ADMM, QPI-Net, PhaseFormer]

## Assessment
Algorithms are now domain-appropriate.

The previous generic microscopy pool (Richardson-Lucy, PnP-FISTA, CARE, Restormer) was replaced with four quantitative phase imaging algorithms that address the phase retrieval inverse problem:
- **TIE Solver** — Transport of Intensity Equation, canonical analytical phase recovery (Teague 1983)
- **DPC-ADMM** — Differential Phase Contrast reconstruction via ADMM optimization (Tian & Waller, Optica 2015)
- **QPI-Net** — deep learning quantitative phase imaging (Rivenson et al., Light Sci. Appl. 2019)
- **PhaseFormer** — transformer-based phase retrieval from intensity measurements

## Verdict
No further code changes needed.
