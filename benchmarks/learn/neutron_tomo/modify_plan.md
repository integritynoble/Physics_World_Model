# Modify Plan: neutron_tomo (Neutron Radiography / Tomography)

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

**Acceptable.** Neutron tomography uses the same mathematical framework as X-ray CT
(Beer-Lambert attenuation + Radon transform), so CT-specific algorithms would also
be appropriate. However, the generic scientific_instrumentation pool is acceptable:

- Deconv provides a classical analytical baseline for tomographic data
- PnP-BM3D provides plug-and-play regularization
- ResNet-Calib and CalibFormer provide learned reconstruction
- The solver-class progression is correctly maintained
- A carrier route for `("scientific_instrumentation", "Neutron")` would be unsafe
  because it would also affect neutron_diffraction (not tomographic)

The current assignment is the safer choice given the shared carrier with
neutron_diffraction.

## Verdict

**PASS -- no code changes needed.** The scientific_instrumentation pool is
acceptable for the benchmark framework. The generic algorithms correctly test
inverse-problem solving from projection data.

## Recommended Changes

None required. If a future variant-level override is desired for improved domain
specificity, it should be a `_VARIANT_OVERRIDES["neutron_tomo"]` entry (not a
carrier route) to avoid affecting neutron_diffraction.
