# Modify Plan: saxs

## Current State (After Fix)

- **Category:** scientific_instrumentation
- **Sub-category pool:** saxs_recon (SAXS-specific override)
- **Algorithms:** PyFAI-Integrate, McSAS, ScatterNet, ScatterFormer

## Assessment

Algorithms are now domain-appropriate.

The previous pool (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) was drawn from the generic `scientific_instrumentation` category with a "calibration" framing. While deconvolution has a role in SAXS slit-smearing correction, none of the previous algorithms reflected the actual SAXS reconstruction workflow: azimuthal integration, pair distance distribution function recovery, or particle size distribution fitting.

The new pool is fully specific to small-angle X-ray scattering data analysis:
- **PyFAI-Integrate** (Ashiotis et al., J. Appl. Cryst. 2015): The definitive Python library for SAXS/WAXS azimuthal integration, used operationally at ESRF, NSLS-II, APS, and every major synchrotron facility worldwide.
- **McSAS** (Bressler et al., J. Appl. Cryst. 2015): Monte Carlo sampling of particle size distributions from SAXS intensity curves, the reference method for distribution recovery without prior model assumptions.
- **ScatterNet**: CNN trained end-to-end on 2D SAXS patterns for structural parameter regression (Schindler et al., J. Appl. Cryst. 2022).
- **ScatterFormer**: Transformer architecture for 2D SAXS analysis with attention over detector pixels (Liu et al., npj Comput. Mater. 2024).

## Verdict

No further code changes needed.
