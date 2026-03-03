# Modify Plan: ocean_acoustic_tomo (Ocean Acoustic Tomography)

## Current State

- **Category:** experimental_science
- **Carrier:** Acoustic
- **Score key:** experimental_science (no carrier routing applies)
- **Algorithms served (4):**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Acceptable.** Ocean acoustic tomography reconstructs ocean sound-speed (temperature)
fields from acoustic travel-time measurements between source-receiver pairs. This is
a linear inverse problem (travel time = integral of slowness along ray path), making
it structurally similar to CT.

The experimental_science pool algorithms are reasonable:
- **Tikhonov** regularization is the standard approach for travel-time inversion
  in ocean acoustics (Munk et al., 1995).
- **PnP-RED** is applicable as a regularized inversion with learned prior.
- **ResUNet** and **SwinIR** are generic but reasonable deep learning baselines.

More domain-specific algorithms would include:
- SIRT/ART (iterative tomographic methods adapted for irregular ray coverage)
- Matched-field processing (for range-dependent environments)
- Neural-network-based travel-time tomography (e.g., NeuralOAT)

But the current generic pool is a defensible choice since ocean acoustic tomography
is a relatively niche field without a large published benchmark ecosystem.

## Verdict

No code changes needed.
