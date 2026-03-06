# Modify Plan: atom_probe (Atom Probe Tomography)

**Updated:** 2026-03-06
**Status:** PASS (with noted limitations in generic citations)

## Current State

- Algorithm routing: `scientific_instrumentation` category pool → 11 methods (Deconv, Calibration-Lookup, Peak Fitting, PnP-BM3D, PnP-NLM, ResNet-Calib, Instrument-CNN, CalibFormer, MassSpecFormer, DiffusionInstrumentation, ScoreInstrumentation).
- Deconv (Bas protocol) and Calibration-Lookup (Geiser protocol) are genuine APT reconstruction methods.
- ResNet-Calib and CalibFormer have generic citations ("ResNet for calibration, 2022", "Transformer calibration, 2024") — these are archetypes rather than specific published papers.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: flight_path_error, voltage_calibration, detection_efficiency, tip_radius_error — all physically grounded in APT practice.

## Noted Limitations

- ResNet-Calib and CalibFormer citations are generic placeholders; specific APT ML papers (e.g., Wei et al., Ultramicroscopy 2019) would improve citation quality.
- The scientific_instrumentation pool is a shared catch-all; APT-specific variant override would give more targeted algorithms, but is not required for benchmark functionality.

## Verdict

PASS. Classical algorithms (Bas/Geiser) are domain-correct. Generic DL citations are a known limitation of the shared pool. No functional code changes required.
