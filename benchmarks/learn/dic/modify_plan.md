# Modify Plan: dic

## Current State (After Fix)
- **Category:** microscopy
- **Sub-category pool:** functional_micro (DIC-specific phase gradient)
- **Algorithms:** [Fourier Integration, DIC-Tikhonov, DIC-Net, PhaseFormer]

## Assessment
Algorithms are now domain-appropriate.

The previous generic microscopy pool (Richardson-Lucy, PnP-FISTA, CARE, Restormer) was replaced with four DIC-specific algorithms that address the gradient-integration phase retrieval inverse problem:
- **Fourier Integration** — canonical integration of phase gradient in Fourier space (Arnison et al., J. Microsc. 2004)
- **DIC-Tikhonov** — regularized gradient inversion accounting for noise in phase derivative measurements (Bostan et al., Opt. Lett. 2014)
- **DIC-Net** — end-to-end CNN learning DIC-to-phase mapping including artifact correction (Li et al., Opt. Express 2018)
- **PhaseFormer** — transformer-based phase recovery from single or multi-orientation DIC images

## Verdict
No further code changes needed.
