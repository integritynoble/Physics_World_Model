# Modify Plan: neutron_diffraction (Neutron Diffraction)

## Current State

- **Category:** scientific_instrumentation
- **Carrier:** Neutron
- **Score key:** scientific_instrumentation (no carrier routing applies)
- **Algorithms served (4):**
  1. Deconv (Classical) -- Analytical baseline
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. ResNet-Calib (Deep Learning) -- ResNet for calibration, 2022
  4. CalibFormer (Transformer) -- Transformer calibration, 2024

## Assessment

**Suboptimal but defensible.** Neutron diffraction reconstructs crystal structure
(lattice parameters, atomic positions) from diffraction patterns. The standard
workflow is Rietveld refinement (Rietveld, J. Appl. Cryst. 1969) or Fourier
difference maps, not generic deconvolution.

Domain-specific algorithms would include:
- Rietveld refinement (classical least-squares fit of diffraction pattern)
- Maximum entropy method (MEM) for nuclear density reconstruction
- Pair distribution function (PDF) analysis
- ML-based structure prediction (e.g., CrystalNet, DiffCSP)

However, the scientific_instrumentation pool provides a reasonable generic framework:
- "Deconv" maps loosely to Fourier inversion of diffraction data
- "PnP-BM3D" represents regularized reconstruction
- "ResNet-Calib" and "CalibFormer" represent learned approaches

The generic names are not ideal but the benchmark framework (forward model mismatch
correction) still applies correctly.

## Recommended Changes

Similar to muon_tomo, a dedicated override would be more domain-accurate but is
not strictly necessary. The scientific_instrumentation catch-all is acceptable
for the benchmark's inverse-problem framing.

## Verdict

No code changes needed. The generic scientific_instrumentation pool is acceptable
for the benchmark framework, though domain-specific algorithm names would be ideal.
