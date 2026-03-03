# Modify Plan: xray_ndt

## Current State

- **Category:** industrial_inspection
- **Carrier:** X-ray
- **Score key:** industrial_inspection
- **Algorithms assigned:**

| Name       | Type           | Source                  |
|------------|----------------|-------------------------|
| TSR        | Classical      | Shepard et al., 2003    |
| PnP-ADMM  | PnP            | ADMM + denoiser prior   |
| DefectNet  | Deep Learning  | U-Net for NDT, 2021     |
| LSTM-NDT   | Recurrent      | Fang et al., 2022       |

## Assessment

**Partially appropriate -- needs improvement.**

The `industrial_inspection` pool was designed around *active thermography* NDT (note TSR = Thermographic Signal Reconstruction, a thermal-specific method from Shepard et al. 2003). X-ray NDT radiography is a fundamentally different inverse problem: it is a projection/transmission imaging modality (Beer-Lambert attenuation), not a thermal diffusion problem. The current algorithms are a poor fit:

1. **TSR** (Thermographic Signal Reconstruction) is a thermography-specific algorithm that processes thermal decay curves. It has no relevance to X-ray projection imaging. This is the most significant mismatch.
2. **PnP-ADMM** is a generic framework and is reasonable for any inverse problem -- acceptable.
3. **DefectNet** is described as "U-Net for NDT, 2021" which is vague. Real X-ray NDT uses defect-detection CNNs trained on radiographic images, so the name is broadly acceptable but the reference is imprecise.
4. **LSTM-NDT** (Fang et al., 2022) is an LSTM-based method for thermographic sequence analysis, not X-ray radiography. This is a mismatch.

X-ray NDT radiography should instead use algorithms from the X-ray/CT reconstruction literature adapted for single-projection or limited-angle settings, e.g.:
- **Classical:** FBP or Tikhonov (backprojection/deconvolution for scatter correction)
- **PnP:** PnP-ADMM (already present -- keep)
- **Deep Learning:** A radiograph-enhancement CNN such as U-Net scatter correction (Maier et al., Med. Phys. 2019) or ASTRA-based DL
- **Domain-specific:** GAN-based scatter estimation (Li et al., PMB 2021) or a dual-energy decomposition network

## Proposed Changes

Add a carrier-routing rule or a variant override in `_algorithm_catalog.py`:

```python
# In _CARRIER_ROUTING or as a variant override for xray_ndt:
# Option A: Add to _CARRIER_ROUTING
("industrial_inspection", "X-ray"): "xray_ndt_pool",  # separate from thermal NDT

# Option B: Add variant override (preferred -- more targeted)
_VARIANT_OVERRIDES["xray_ndt"] = [
    {"name": "FBP",           "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Analytical baseline"},
    {"name": "PnP-ADMM",     "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Venkatakrishnan et al., 2013"},
    {"name": "Scatter-Net",   "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "Maier et al., Med. Phys. 2019"},
    {"name": "DR-GAN",        "type": "GAN",           "mask_aware": False, "params": "8M",   "source": "Li et al., PMB 2021"},
]
```

### Files to modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` -- add variant override for `xray_ndt`

### Risk
- Changing algorithms changes the leaderboard display. No functional risk since scores are synthetic.
- Other industrial_inspection modalities with non-X-ray carriers (thermography, eddy current) remain unaffected.
