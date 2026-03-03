# Modify Plan: streak_camera

## Current State
- **Category:** ultrafast
- **Carrier:** Photon
- **Score key:** ultrafast
- **Algorithms:**
  1. TwIST (Classical) -- Bioucas-Dias & Figueiredo, IEEE TIP 2007
  2. PnP-FFDNet (PnP) -- Yuan et al., 2020
  3. CUP-Net (Deep Learning) -- Parker et al., 2021
  4. AL-DL (Hybrid) -- Yao et al., Photon. Res. 2021

## Assessment

The algorithms are appropriate for streak camera imaging. Streak cameras perform temporal-to-spatial mapping, and compressed ultrafast photography (CUP) is a major application that encodes spatiotemporal scenes into a single 2D snapshot using a streak camera.

- **TwIST** (Two-Step Iterative Shrinkage/Thresholding) is a standard compressive reconstruction algorithm used as a baseline for CUP data.
- **PnP-FFDNet** has been specifically applied to CUP reconstruction (Yuan et al., 2020), combining the CUP forward model with FFDNet denoising.
- **CUP-Net** is explicitly designed for compressed ultrafast photography reconstruction using streak cameras.
- **AL-DL** (Alternating Learning with Deep Learning) is a hybrid method for ultrafast imaging reconstruction.

All four algorithms directly target the streak camera / CUP reconstruction problem. The mismatch parameters (DMD mask registration, streak sweep rate, shearing angle) correctly capture the domain-specific calibration challenges.

No code changes needed.

## Files to Modify
None.
