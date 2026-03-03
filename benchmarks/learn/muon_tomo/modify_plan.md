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

**Suboptimal.** Muon tomography is a tomographic imaging modality (DAG: Pi -> D)
that reconstructs 3D density/atomic-number maps from multiple Coulomb scattering
of cosmic-ray muons. The reconstruction problem is closer to CT than to generic
"calibration":

- The classical approach should be PoCA (Point of Closest Approach) or MLSD
  (Maximum Likelihood Scattering with Displacement), not generic deconvolution.
- Published algorithms include PoCA (Schultz, NIM-A 2003), MLSD/MLM (Anghel et al., 2015),
  and filtered back-projection adapted to scattering data.
- "ResNet-Calib" and "CalibFormer" are generic placeholder names that do not correspond
  to published muon tomography methods.
- More appropriate DL methods: Muon-ResNet (Joshi et al., 2023) or scattering-angle
  CNN approaches.

However, the scientific_instrumentation pool is a catch-all for diverse instruments
(atom probe, mass spec, etc.) and no single algorithm set works perfectly for all.
The generic names are not factually wrong -- they just lack domain specificity.

## Recommended Changes

**Option A (ideal):** Add a carrier routing for `("scientific_instrumentation", "Muon")`
pointing to a new `muon_tomo_pool` or add a variant override:
```python
"muon_tomo": [
    {"name": "PoCA",        "type": "Classical",     ...},
    {"name": "MLSD",        "type": "PnP",           ...},
    {"name": "Muon-CNN",    "type": "Deep Learning",  ...},
    {"name": "ScatterFormer","type": "Transformer",   ...},
]
```

**Option B (minimal):** Leave as-is. The generic scientific_instrumentation pool
is a defensible catch-all and the benchmark still tests the inverse-problem framework
correctly.

## Verdict

Changes would improve domain accuracy but are not strictly required.
Current algorithms are generic but not incorrect for the benchmark framework.
