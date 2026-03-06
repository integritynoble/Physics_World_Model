# Modify Plan: brachytherapy_img (Brachytherapy Imaging)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: carrier `Gamma/X-ray` does not match explicit routing entries; falls through to `medical` category pool → CT reconstruction pool (13 methods).
- CT algorithms (FBP, TV-ADMM, PnP-ADMM, FBPConvNet, Learned Primal-Dual, DuDoTrans, CTFormer, DOLCE, DiffusionCT, Score-CT, etc.) are technically correct — brachytherapy verification imaging IS X-ray CT.
- All main citations are real (Jin 2017 IEEE TIP, Adler & Oktem 2018 IEEE TMI, Venkatakrishnan 2013 IEEE GlobalSIP).
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: source_position_error, attenuation_coefficient, detector_gain_drift, scatter_fraction — all physically grounded.

## Noted Limitations

- Compound carrier `Gamma/X-ray` does not trigger explicit routing — falls through to medical category. Functionally correct but should be documented.
- No brachytherapy-specific seed localisation algorithms; CT reconstruction pool is the correct abstraction for the benchmark.

## Verdict

PASS. CT algorithms are appropriate for brachytherapy X-ray verification imaging. No code changes required.
