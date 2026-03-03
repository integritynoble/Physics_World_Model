# Modify Plan: dna_paint (DNA-PAINT Super-Resolution)

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms served:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

**Significant mismatch.** DNA-PAINT is a single-molecule localization microscopy
(SMLM) technique. The reconstruction problem is fundamentally different from
standard microscopy deconvolution:

- The input is a stack of frames with sparse blinking emitters.
- Reconstruction involves **localizing individual molecules** to sub-pixel
  precision, then rendering a super-resolved image from the localization list.
- The key algorithms are localization-based: ThunderSTORM (Ovesny et al. 2014),
  DECODE (Speiser et al. Nat. Methods 2021), Deep-STORM (Nehme et al. Optica
  2018), and FALCON (Min et al. 2014).
- Richardson-Lucy and CARE are denoising/deconvolution methods that do NOT
  perform single-molecule localization.

The leaderboard on the live site actually shows domain-appropriate names
(FP-INR, Deep-STORM, PnP-HQS, ThunderSTORM from the check.md), suggesting
the leaderboard display may have been customized, but the `get_algorithms()`
catalog still returns the generic microscopy pool.

DNA-PAINT should ideally use a "super_resolution" or "smlm" sub-category with
algorithms like: ThunderSTORM (Classical), FALCON (Sparse), Deep-STORM (DL),
DECODE (DL), FP-INR (Neural field).

## Recommended Changes

Add a carrier-routing or variant-override entry in `_algorithm_catalog.py` for
SMLM modalities (dna_paint, palm_storm, sted, minflux, etc.) that maps to a
localization-specific pool:

```python
# In _VARIANT_OVERRIDES or a new _SMLM_POOL:
"dna_paint": [
    {"name": "ThunderSTORM",  "type": "Classical",     ...},
    {"name": "FALCON",        "type": "Sparse",        ...},
    {"name": "Deep-STORM",    "type": "Deep Learning", ...},
    {"name": "DECODE",        "type": "Deep Learning", ...},
]
```

**File to modify:** `/home/spiritai/pwm/Physics_World_Model/platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`

## Verdict

Code changes recommended. The generic microscopy deconvolution pool is
inappropriate for SMLM modalities. The reconstruction task (molecule
localization from blinking frames) is fundamentally different from
deconvolution, and all four current algorithms are mismatched.
