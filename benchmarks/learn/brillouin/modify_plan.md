# Modify Plan: brillouin (Brillouin Microscopy)

**Updated:** 2026-03-06
**Status:** PASS — minor cosmetic issue only

## Current State

- Algorithm routing: `spectroscopy` category + `Photon` carrier → 11-method spectroscopy pool.
- SG-ALS and Baseline Correction are standard spectral preprocessing methods for Brillouin data.
- PnP-DnCNN citation (Zhang et al., IEEE TIP 2017) is correct.
- CDAE (Zhang et al., Sensors 2024) is a real citation.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: brillouin_shift_calibration, vipa_fsr_error, elastic_scattering_leakage — the three main VIPA Brillouin artefact sources.

## Noted Issues

- **Cascade-UNet type label**: listed as "Transformer" in the catalog but is a UNet architecture. Cosmetic issue, no functional impact.
- Brillouin-specific algorithms (Lorentzian peak fitting, VIPA calibration) are not in the pool but the spectroscopy pool is an appropriate generalization.

## Verdict

PASS. No functional code changes required. The Cascade-UNet type label mislabelling is cosmetic only.
