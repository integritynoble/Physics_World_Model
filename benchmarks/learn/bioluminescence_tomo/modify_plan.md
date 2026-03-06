# Modify Plan: bioluminescence_tomo (Bioluminescence Tomography)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: `experimental_science` category + `Photon` carrier → 11-method experimental science pool.
- Tikhonov is the standard BLT baseline (Lv et al., PMB 2006 is the canonical reference).
- PnP-RED (Romano et al., IEEE TIP 2017) is real and appropriate.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: optical_property_error, source_depth_ambiguity, autofluorescence_background — the three dominant BLT model uncertainties.

## Noted Limitations

- SwinIR in the pool is a 2D image restoration transformer applied to what is fundamentally a 3D volumetric source reconstruction; acceptable for 2D projection benchmark but noted.
- ResUNet source citation is generic; BLT-specific DL citation (Gao et al. 2018) would be preferable.

## Verdict

PASS. Tikhonov and PnP-RED are domain-correct. No code changes required.
