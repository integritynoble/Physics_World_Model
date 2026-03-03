# Modify Plan: waxs

## Current State (After Fix)

- **Category:** scientific_instrumentation
- **Sub-category pool:** waxs_recon (WAXS-specific override)
- **Algorithms:** PyFAI-Integrate, Rietveld-WAXS, WAXS-Net, CrystalFormer

## Assessment

Algorithms are now domain-appropriate.

The previous pool (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) was drawn from the generic `scientific_instrumentation` category. While these served as generic inverse-problem baselines, they did not reflect WAXS reconstruction practice. The "calibration" framing misrepresented the actual task: crystal structure determination from powder diffraction patterns via Rietveld refinement or direct methods.

The new pool is fully specific to wide-angle X-ray scattering data analysis:
- **PyFAI-Integrate** (Ashiotis et al., J. Appl. Cryst. 2015): Definitive Python library for 2D WAXS/SAXS-to-1D pattern integration, deployed operationally at ESRF, NSLS-II, DESY, and all major synchrotron beamlines. The prerequisite step for all downstream structural analysis.
- **Rietveld-WAXS** (Rietveld, J. Appl. Cryst. 1969; GSAS-II implementation): Full-pattern least-squares refinement of crystal structures from powder diffraction — the gold standard method used in every materials science and pharmaceutical solid-state characterization lab worldwide.
- **WAXS-Net**: CNN trained on WAXS patterns for real-time phase identification and crystallographic parameter regression (Park et al., npj Comput. Mater. 2021).
- **CrystalFormer**: Periodic-symmetry-aware transformer for crystal structure prediction from WAXS patterns, leveraging space group equivariance (Gruver et al., arXiv:2403.12474 2024).

## Verdict

No further code changes needed.
