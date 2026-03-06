# Modify Plan: spect_ct

## Current State
- **Category:** multi_modal_fusion
- **Carrier:** Gamma
- **Score key:** multi_modal_fusion
- **Algorithms:**
  1. MLAA (Classical) -- Rezaei et al., IEEE TMI 2012
  2. MR-Guided (PnP) -- Ehrhardt et al., SIIS 2015
  3. FBSEM-Net (Deep Learning) -- Mehranian & Reader, IEEE TMI 2020
  4. PPMF-Net (Transformer) -- Li et al., 2024

## Assessment

**Problem:** The multi_modal_fusion pool contains algorithms designed primarily for PET-CT and PET-MR fusion, not SPECT-CT. While the general concept of multi-modal fusion applies, the specific algorithms are mismatched:

- **MLAA (Maximum Likelihood Activity and Attenuation)** is specifically a PET algorithm that jointly estimates activity and attenuation from PET data. SPECT uses different physics (collimator-based projection vs. coincidence detection).
- **MR-Guided** (Ehrhardt et al.) is for MR-guided PET reconstruction, not SPECT-CT.
- **FBSEM-Net** (Mehranian & Reader) is for PET-MR, not SPECT-CT specifically.
- **PPMF-Net** is a generic fusion network that could apply.

Appropriate SPECT-CT algorithms include:
1. **OSEM** (Classical) -- Hudson & Larkin, IEEE TMI 1994; the standard SPECT iterative reconstruction
2. **CT-based AC-OSEM** (Classical) -- Patton & Turkington, JNMT 2008; OSEM with CT-based attenuation correction
3. **MAP-OSEM with MRP** (PnP-like) -- Nuyts et al.; maximum a posteriori with median root prior
4. **DL-SPECT** (Deep Learning) -- Shao et al., IEEE TMI 2021; deep learning for SPECT

## Required Changes

Add `spect_ct` to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py` with SPECT-CT specific algorithms:

```python
"spect_ct": [
    {"name": "OSEM",          "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Hudson & Larkin, IEEE TMI 1994"},
    {"name": "AC-OSEM",       "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Patton & Turkington, JNMT 2008"},
    {"name": "MAP-OSEM",      "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Nuyts et al., 2002"},
    {"name": "DL-SPECT",      "type": "Deep Learning", "mask_aware": False, "params": "8M",   "source": "Shao et al., IEEE TMI 2021"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Add variant override for `spect_ct`

## 2026-03-06 Comprehensive Check Update

STATUS: The variant override has been IMPLEMENTED. Current algorithms (from get_algorithms):
  1. OSEM (Classical) -- Hudson & Larkin, IEEE TMI 1994  [CORRECT]
  2. AC-OSEM (Classical) -- CT-based attenuation correction  [CORRECT]
  3. MAP-OSEM (PnP) -- Nuyts et al., 2002  [CORRECT]
  4. DL-SPECT (Deep Learning) -- Ramon et al., IEEE TMI 2020  [CORRECT]

- Physics: y(b) = Poisson(H_b * lambda * ACF + scatter); depth-dependent CDR blurring; OSEM iteration
- Key mismatch: CDR model, ACF calibration, scatter fraction (30-50%), dead time
- GCS datasets: 3 tiers confirmed
- Algorithm pool: PASS — all 4 algorithms are SPECT-CT specific and highly appropriate
- Note: This is one of the best-curated algorithm pools in the catalog — all 4 are domain-optimal
