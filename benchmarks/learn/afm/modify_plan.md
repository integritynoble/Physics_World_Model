# Modify Plan: afm (Atomic Force Microscopy)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: dedicated `scanning_probe` category pool → 10 methods (BTR, MLE Reconstruction, Reg-Deconv, TV-Deconvolution, DeepSPM, U-Net-SPM, E2E-BTR, SPM-Former, DiffusionSPM, ScoreSPM).
- BTR (Villarrubia 1997) and E2E-BTR (Kossler 2022) are real published AFM-specific algorithms — excellent domain alignment.
- DeepSPM (Alldritt 2020) is a real published paper on deep learning for scanning probe microscopy.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: tip_shape_convolution, piezo_nonlinearity, thermal_drift, scanner_hysteresis — all four address principal AFM artefact sources.

## Verdict

PASS. The scanning_probe pool is purpose-built for AFM/STM modalities and contains real, well-cited methods. No code changes required.
