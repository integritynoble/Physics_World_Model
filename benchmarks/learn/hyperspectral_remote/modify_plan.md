# Modify Plan -- hyperspectral_remote

## Current State

- **Category:** remote_sensing
- **Carrier:** Photon
- **Score key:** computational (routed via `_CARRIER_ROUTING[("remote_sensing", "Photon")]` -> `"computational"`)
- **Algorithms:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. Deep Image Prior (Deep Learning) -- Ulyanov et al., CVPR 2018
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Problem:** The routing correctly avoids SAR-specific algorithms (since hyperspectral remote sensing uses photon/optical carrier, not RF), but the fallback to the generic `computational` pool gives algorithms that are not specific to hyperspectral imaging. Hyperspectral remote sensing reconstruction has a well-established literature with dedicated algorithms.

**Appropriate domain algorithms would include:**
- Classical: HySure (Simoes et al., IEEE TGRS 2015) or CNMF (Yokoya et al., IEEE TGRS 2012) for spectral unmixing/fusion
- PnP: LTTR (Low-Tensor-Train-Rank, He et al., IEEE TGRS 2020) or PnP-based HSI denoising
- Deep Learning: DBIN (Dong et al., CVPR 2021) or SSPSR (Jiang et al., IEEE TGRS 2020)
- Transformer: MST++ (Cai et al., CVPRW 2022) or Spectral Transformer (Li et al., IEEE TGRS 2023)

## Recommendation

**Code changes needed** -- add a sub-category routing or variant override for `hyperspectral_remote` in `_algorithm_catalog.py` to use hyperspectral-imaging-specific algorithms instead of the generic computational pool.

### Proposed change in `_algorithm_catalog.py`:

1. Add a `_CARRIER_ROUTING` entry or a `_VARIANT_OVERRIDES` entry:
   ```python
   "hyperspectral_remote": [
       {"name": "CNMF",   "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Yokoya et al., IEEE TGRS 2012"},
       {"name": "PnP-LTTR","type": "PnP",           "mask_aware": True,  "params": "0",   "source": "He et al., IEEE TGRS 2020"},
       {"name": "DBIN",    "type": "Deep Learning", "mask_aware": False, "params": "3.2M","source": "Dong et al., CVPR 2021"},
       {"name": "MST++",   "type": "Transformer",   "mask_aware": True,  "params": "8M",  "source": "Cai et al., CVPRW 2022"},
   ],
   ```
2. Add corresponding real scores in `CATEGORY_REAL_SCORES["hyperspectral_remote"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
