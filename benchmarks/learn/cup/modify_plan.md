# Modify Plan: cup (Compressed Ultrafast Photography)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: dedicated `ultrafast` category pool → 11 methods (TwIST, Temporal Filtering, PnP-FFDNet, PnP-ADMM, CUP-Net, Temporal-U-Net, AL-DL, Unfolded-CUP, UltraFormer, DiffusionUltrafast, ScoreUltrafast).
- TwIST (Bioucas-Dias & Figueiredo, IEEE TIP 2007) was the CS solver used in the original CUP paper (Gao et al., Nature 2014) — confirms domain correctness.
- PnP-FFDNet (Yuan et al., 2020) is a real PnP paper for snapshot compressive imaging.
- CUP-Net and AL-DL are CUP-specific algorithms.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: dmd_encoding_error, streak_sweep_calibration, temporal_spatial_coupling — three principal CUP system calibration uncertainties.

## Verdict

PASS. Dedicated ultrafast pool with domain-specific algorithms. No code changes required.
