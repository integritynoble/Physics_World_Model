# Modify Plan: muon_tomo (Muon Tomography)

## Current State

- **Category:** scientific_instrumentation
- **Carrier:** Muon
- **Score key:** scientific_instrumentation (no carrier routing applies)
- **Algorithms served (4):**
  1. Deconv (Classical) -- Analytical baseline
  2. PnP-BM3D (PnP) -- Danielyan et al., 2012
  3. ResNet-Calib (Deep Learning) -- ResNet for calibration, 2022
  4. CalibFormer (Transformer) -- Transformer calibration, 2024

## Assessment

**Acceptable.** Muon tomography is a tomographic imaging modality that reconstructs
3D density/atomic-number maps from multiple Coulomb scattering of cosmic-ray muons.
While domain-specific algorithms exist (PoCA, MLSD), the generic scientific_instrumentation
pool is acceptable for the benchmark framework:

- Deconv provides a classical analytical baseline
- PnP-BM3D provides a plug-and-play regularization approach
- ResNet-Calib and CalibFormer provide learned reconstruction methods
- The solver-class progression (classical -> PnP -> DL -> transformer) is maintained
- The algorithms are generic but not incorrect for the inverse-problem framing

The generic names describe algorithmic approaches rather than domain-specific
implementations, which is consistent across the scientific_instrumentation category.

## Verdict

**PASS -- no code changes needed.** The scientific_instrumentation category pool
provides a valid set of algorithms covering all solver classes. The generic approach
is consistent with how the benchmark handles diverse instrumentation modalities.

## Recommended Changes

None required. Optional future enhancement: add a variant-specific override with
PoCA/MLSD for improved domain specificity, but this is not necessary for correct
benchmark operation.
