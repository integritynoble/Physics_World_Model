# Modify Plan: gravitational_wave

## Current State

- **Category:** experimental_science
- **Carrier:** Gravitational
- **Score key:** experimental_science
- **Algorithms assigned:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Problematic -- generic algorithms; GW-specific methods exist and are well-known**

Gravitational wave detection/signal extraction has a very specific set of
published algorithms. The current generic experimental_science pool (Tikhonov,
PnP-RED, ResUNet, SwinIR) is inappropriate because:

- **Tikhonov**: GW signal extraction is NOT a standard Tikhonov regularization
  problem. The classical method is matched filtering against template banks.
- **PnP-RED, SwinIR**: These are image restoration methods. GW data is 1D
  time-series strain data, not 2D images.
- **ResUNet**: Generic, not GW-specific.

Well-known GW-specific algorithms:
- **Matched Filtering** (Abbott et al., LIGO/Virgo standard pipeline)
- **BayesWave** (Cornish & Littenberg, CQG 2015) -- Bayesian wavelet-based
- **MLy** / **GW-CNN** (Gebhard et al., PRD 2019) -- deep learning for GW
- **WaveFormer** (Zhao et al., 2023) -- transformer-based GW detection

## Code Changes Needed

**Add GW-specific variant override in `_algorithm_catalog.py`:**

```python
"gravitational_wave": [
    {"name": "Matched Filter (Template Bank)", "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Allen et al., PRD 2012"},
    {"name": "BayesWave",                      "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "Cornish & Littenberg, CQG 2015"},
    {"name": "GW-CNN",                         "type": "Deep Learning", "mask_aware": False, "params": "2M",   "source": "Gebhard et al., PRD 2019"},
    {"name": "WaveFormer",                     "type": "Transformer",   "mask_aware": True,  "params": "8M",   "source": "Zhao et al., arXiv 2023"},
],
```
