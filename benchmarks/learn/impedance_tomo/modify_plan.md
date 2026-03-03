# Modify Plan -- impedance_tomo

## Current State

- **Category:** experimental_science
- **Carrier:** Electric
- **Score key:** experimental_science
- **Algorithms:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Problem:** Electrical Impedance Tomography (EIT) is a well-established inverse problem with dedicated reconstruction algorithms. The generic `experimental_science` pool (Tikhonov, PnP-RED, ResUNet, SwinIR) is too generic. EIT reconstruction has unique challenges (highly ill-posed, nonlinear forward model, conductivity mapping) that warrant domain-specific methods.

**Appropriate domain algorithms would include:**
- Classical: GREIT (Graz consensus Reconstruction for EIT, Adler et al., Physiol. Meas. 2009) or one-step Gauss-Newton
- PnP: TV-regularized Gauss-Newton (total variation on conductivity maps)
- Deep Learning: D-bar CNN (Hamilton & Hauptmann, IEEE TMI 2018) -- combines D-bar method with CNN post-processing
- Transformer: EIT-TransNet or similar learned direct inverse maps

## Recommendation

**Code changes needed** -- add a `_VARIANT_OVERRIDES` entry for `impedance_tomo` in `_algorithm_catalog.py` with EIT-specific algorithms.

### Proposed change in `_algorithm_catalog.py`:

```python
"impedance_tomo": [
    {"name": "Gauss-Newton",  "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Cheney et al., Int J Imaging Syst Technol 1999"},
    {"name": "TV-GN",         "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Borsic et al., Physiol. Meas. 2010"},
    {"name": "D-bar CNN",     "type": "Deep Learning", "mask_aware": False, "params": "2.5M","source": "Hamilton & Hauptmann, IEEE TMI 2018"},
    {"name": "EIT-UFormer",   "type": "Transformer",   "mask_aware": True,  "params": "6M",  "source": "Li et al., IEEE TIM 2024"},
],
```

Add corresponding `CATEGORY_REAL_SCORES["impedance_tomo"]`.

### Files to modify:
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
