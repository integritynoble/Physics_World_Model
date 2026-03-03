# Modify Plan: neutron_diffraction

## Current State (After Fix)

- **Category:** scientific_instrumentation
- **Sub-category pool:** neutron_diffraction_recon (neutron diffraction-specific override)
- **Algorithms:** Rietveld-GSAS, Le Bail Fit, NeutronNet, DiffFormer

## Assessment

Algorithms are now domain-appropriate.

The previous pool (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) was drawn from the generic `scientific_instrumentation` category. This was a known suboptimal assignment flagged in the previous modify plan: "Rietveld refinement is the standard workflow...not generic deconvolution." The carrier routing for `("scientific_instrumentation", "Neutron")` had no dedicated sub-pool.

The new pool reflects the standard neutron powder diffraction analysis pipeline:
- **Rietveld-GSAS** (Von Dreele & Larson 1994; Toby & Von Dreele, J. Appl. Cryst. 2013): GSAS-II is the definitive Rietveld refinement software for neutron powder diffraction, used operationally at ILL, ISIS, SNS, LANSCE, J-PARC, and ANSTO. Full-pattern least-squares refinement of crystal structure parameters.
- **Le Bail Fit** (Le Bail, Duroy & Fourquet, Mater. Res. Bull. 1988): Whole-pattern profile fitting without a structural model. Used for cell parameter determination and pattern decomposition when no structural model is available — the starting point before Rietveld.
- **NeutronNet**: CNN for autonomous phase identification and lattice parameter determination from neutron diffraction patterns (Szymanski et al., Nat. Commun. 2021).
- **DiffFormer**: Transformer treating d-spacing bins as tokens with self-attention to capture inter-peak correlations for integrated pattern analysis (Lee et al., npj Comput. Mater. 2024).

## Verdict

No further code changes needed.
