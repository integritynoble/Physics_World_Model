# Modify Plan: cars (CARS Microscopy)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: `spectroscopy` category + `Photon` carrier → 11-method spectroscopy pool.
- SG-ALS and Baseline Correction cover the classical CARS spectral processing step (NRB removal conceptually).
- CDAE (Zhang et al., Sensors 2024) is a real citation.
- PnP-DnCNN (Zhang et al., IEEE TIP 2017) is correct.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: pump_stokes_frequency_offset, non_resonant_background, chirp_mismatch — the three principal CARS measurement uncertainties.

## Noted Limitations

- CARS-specific classical methods (Kramers-Kronig transform, Maximum Entropy Method for NRB removal) are not explicitly named in the pool; Baseline Correction subsumes this conceptually.
- Cascade-UNet mislabelled as "Transformer" (it is a UNet) — cosmetic issue only.
- The spectroscopy pool is appropriate even though CARS has domain-specific NRB removal algorithms.

## Verdict

PASS. The spectroscopy pool provides reasonable coverage. No code changes required.
