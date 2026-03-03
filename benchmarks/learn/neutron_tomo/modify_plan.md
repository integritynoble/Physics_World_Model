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

**Suboptimal.** Neutron tomography (DAG: Pi -> D) is a projection-based tomographic
modality very similar to X-ray CT. The forward model is Beer-Lambert attenuation
along ray paths, and reconstruction uses the same algorithms as CT:

- FBP (Filtered Back-Projection) is the standard classical baseline
- Iterative methods (SIRT, CGLS, TV-regularized) are widely used
- The problem is fundamentally the same as CT reconstruction with different contrast
  mechanisms (neutron attenuation cross-sections vs. X-ray attenuation)

The current algorithms (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) are generic
scientific instrumentation methods that do not reflect the tomographic nature of
neutron imaging. More appropriate algorithms would be:

- FBP or SIRT (classical tomographic reconstruction)
- PnP-ADMM (plug-and-play with tomographic forward model)
- FBPConvNet or Learned Primal-Dual (CT-domain deep learning)

The `medical` category pool (FBP, PnP-ADMM, FBPConvNet, Learned Primal-Dual) or
even the `ct` variant override would be much more appropriate.

## Recommended Changes

**Option A (recommended):** Add carrier routing for neutron tomography to the
CT-like algorithm pool:
```python
# In _CARRIER_ROUTING or as a variant override:
("scientific_instrumentation", "Neutron"): "medical",  # CT-like reconstruction
```
Or better, add a variant override:
```python
"neutron_tomo": [
    {"name": "FBP",                  "type": "Classical",      ...},
    {"name": "PnP-ADMM",            "type": "PnP",            ...},
    {"name": "FBPConvNet",           "type": "Deep Learning",  ...},
    {"name": "Learned Primal-Dual",  "type": "Deep Unrolling", ...},
]
```

**Caution:** A blanket `("scientific_instrumentation", "Neutron")` routing would
also affect `neutron_diffraction`, which is NOT tomographic. A variant override
is safer than carrier routing here.

**Option B (leave as-is):** The generic pool works for the benchmark framework
but is misleading for a tomographic modality.

## Verdict

Changes would improve domain accuracy. A variant-level override for neutron_tomo
specifically (not a blanket carrier route) is recommended to avoid affecting
neutron_diffraction.
