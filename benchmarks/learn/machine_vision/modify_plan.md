# Modify Plan — machine_vision

## Current State

- **Category:** industrial_inspection
- **Carrier:** Photon
- **Score key:** industrial_inspection
- **Algorithms (from catalog):**
  1. TSR (Classical) -- Shepard et al., 2003
  2. PnP-ADMM (PnP) -- ADMM + denoiser prior
  3. DefectNet (Deep Learning) -- U-Net for NDT, 2021
  4. LSTM-NDT (Recurrent) -- Fang et al., 2022
- **Leaderboard (live):** TSR, PnP-ADMM, DefectNet, LSTM-NDT (4 entries)

## Assessment

The algorithms are **partially appropriate** but have some domain mismatch.

- **TSR (Thermographic Signal Reconstruction)** is specifically a thermography-based NDT technique (Shepard et al., 2003) for analyzing thermal decay curves. Machine vision / Automated Optical Inspection (AOI) uses visible-light cameras, not thermography. TSR is not applicable to machine vision.
- **PnP-ADMM** is a generic reconstruction framework -- acceptable as a general PnP baseline for image enhancement/denoising.
- **DefectNet** as a U-Net for NDT is reasonable but generic. For machine vision AOI specifically, defect detection nets trained on surface inspection data (e.g., DAGM textures) would be more specific.
- **LSTM-NDT** (Fang et al., 2022) is a temporal/sequential NDT method. Machine vision AOI typically works on single images or multi-view stills, not temporal sequences. This is a poor fit.

The category "industrial_inspection" lumps together thermography, eddy current, machine vision, and other NDT modalities. The algorithms lean heavily toward thermal NDT rather than optical inspection.

## Recommended Changes

1. **Add a variant override** for `machine_vision` in `_algorithm_catalog.py`:
   - Classical: **Matched Filter / Template Matching** (classical defect detection baseline)
   - PnP: **PnP-ADMM** (keep -- general purpose)
   - Deep Learning: **PatchCore** (Roth et al., CVPR 2022) or **EfficientAD** (anomaly detection)
   - Transformer: **AnomalyGPT** or **UniAD** (You et al., NeurIPS 2022) -- transformer-based anomaly detection

2. Alternatively, add `("industrial_inspection", "Photon")` to `_CARRIER_ROUTING` to separate optical inspection from thermal/acoustic NDT.

## Verdict

**Changes recommended** -- TSR and LSTM-NDT are thermography/temporal NDT methods, not optical machine vision methods. The modality needs a variant override or carrier-based routing.
