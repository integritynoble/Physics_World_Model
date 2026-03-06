# Modify Plan: desi (DESI Mass Spectrometry Imaging)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: `spectroscopy` category + `Ion` carrier → 11-method spectroscopy pool.
- SG-ALS and Baseline Correction are standard mass spectral preprocessing methods.
- PnP-DnCNN (Zhang et al., IEEE TIP 2017) is real and applicable to spatial MSI denoising.
- CDAE (Zhang et al., Sensors 2024) is a real citation.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: spray_angle_error, solvent_flow_variation, ion_suppression_matrix_effect, spatial_resolution_degradation — four DESI-specific calibration uncertainties.

## Noted Limitations

- Domain-specific MSI methods (MCR-ALS for multivariate unmixing, NMF for spectral decomposition, msImpute for missing values) are absent; the spectroscopy pool covers the spectral denoising aspect but not full MSI analysis workflow.
- Cascade-UNet mislabelled as "Transformer" (it is a UNet) — cosmetic issue.

## Verdict

PASS. Spectroscopy pool provides adequate coverage for the spectral reconstruction benchmark task. No code changes required.
