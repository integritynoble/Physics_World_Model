# Modify Plan: sonar

## Current State
- **Category:** remote_sensing
- **Carrier:** Acoustic
- **Score key:** experimental_science (routed via `_CARRIER_ROUTING`)
- **Algorithms:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Problem:** Sonar imaging is routed from `remote_sensing` via carrier `Acoustic` to the `experimental_science` pool, which gives completely generic algorithms (Tikhonov, PnP-RED, ResUNet, SwinIR). These have no domain relevance to sonar/acoustic imaging.

Sonar reconstruction is a beamforming and matched-filter problem. Appropriate algorithms include:

1. **DAS (Delay-and-Sum)** -- classical beamforming baseline for sonar
2. **MVDR / Capon** -- Capon, Proc. IEEE 1969; minimum variance distortionless response beamformer
3. **MUSIC** -- Schmidt, IEEE TAP 1986; subspace-based high-resolution direction finding
4. **Deep Beamforming / U-Net-BF** -- DL-based beamforming, e.g., Luo et al., JASA 2020

The current routing `("remote_sensing", "Acoustic"): "experimental_science"` is too generic.

## Required Changes

Add `sonar` to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py` with sonar-specific algorithms:

```python
"sonar": [
    {"name": "DAS",         "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Delay-and-Sum baseline"},
    {"name": "MVDR/Capon",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Capon, Proc. IEEE 1969"},
    {"name": "PnP-ADMM",    "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "ADMM + denoiser prior"},
    {"name": "SonarNet",    "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "DL beamforming, 2022"},
],
```

Also add a `CATEGORY_REAL_SCORES` entry for sonar if desired, since the current `experimental_science` scores are generic.

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Add variant override for `sonar`
