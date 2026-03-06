# Modify Plan: ocean_acoustic_tomo (Ocean Acoustic Tomography)

## Current State

- **Category:** experimental_science
- **Carrier:** Acoustic
- **Score key:** experimental_science (no carrier routing applies)
- **Algorithms served (11 in full catalog, 4 highlighted):**
  1. Tikhonov (Classical) -- Tikhonov, Doklady 1963
  2. Wiener Filter (Classical) -- Wiener filtering baseline
  3. Matched Filter (Classical) -- Optimal linear filter
  4. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  5. PnP-ADMM (PnP) -- ADMM + denoiser prior
  6. ResUNet (Deep Learning) -- Residual U-Net baseline
  7. Domain-Adapted-CNN (Deep Learning) -- Domain adaptation CNN
  8. SwinIR (Vision Transformer) -- Liang et al., ICCVW 2021
  9. ExpFormer (Vision Transformer) -- Experimental science transformer, 2024
  10. DiffusionExperimental (Diffusion) -- Zhang et al., 2024
  11. ScoreExperimental (Score-based) -- Wei et al., 2025

## Assessment

**Acceptable.** Ocean acoustic tomography reconstructs ocean sound-speed (temperature)
fields from acoustic travel-time measurements between source-receiver pairs. This is
a linear inverse problem (travel time = integral of slowness along ray path), making
it structurally similar to CT and seismic tomography.

The experimental_science pool algorithms are appropriate:
- **Tikhonov** regularization is the historical standard for travel-time inversion
  in ocean acoustics (Munk & Wunsch, Deep-Sea Research 1979).
- **PnP-RED** is applicable as regularized inversion with learned prior over structured
  ocean mesoscale fields.
- **ResUNet** and **ExpFormer** are reasonable deep learning baselines for end-to-end
  inversion from sparse ray-path measurements.
- More domain-specific algorithms would include SIRT/ART (iterative tomographic methods
  for irregular ray coverage) and matched-field processing (for range-dependent environments).
  The current generic pool is a defensible choice for this niche field.

## 2026-03-06 Comprehensive Check Update

- Physics correctly models linearized travel-time Radon integral
- Calibration mismatches: background sound-speed profile, clock drift, ambient noise
- GCS datasets: 3 tiers confirmed in challenge-data/v1.0/
- Algorithm pool: PASS — spans classical Tikhonov through score-based diffusion

## Verdict

No code changes needed.
