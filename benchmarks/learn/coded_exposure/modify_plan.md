# Modify Plan: coded_exposure (Coded Exposure / Flutter Shutter)

**Updated:** 2026-03-06
**Status:** PASS — no code changes required

## Current State

- Algorithm routing: `computational_photography` category + `Photon` carrier → 14-method pool.
- Wiener-Deconv is THE canonical flutter shutter deconvolution method (Raskar et al., SIGGRAPH 2006) — confirms domain correctness.
- PnP-FFDNet (Zhang et al., IEEE TIP 2018) is real and well-cited for image deblurring.
- Uformer (Wang et al., CVPR 2022) is real and well-cited for image restoration.
- Challenge datasets on GCS for all three tiers.
- Mismatch parameters: shutter_code_timing_error, motion_blur_psf_mismatch, sensor_readout_noise — the key coded exposure calibration uncertainties.

## Noted Limitations

- HDR-CNN (Eilertsen et al., ACM TOG 2017) in the pool is domain-mismatched for coded exposure (it is for HDR reconstruction, not motion deblurring) — known limitation of the shared computational_photography pool. No functional impact.

## Verdict

PASS. Wiener-Deconv baseline is correct. No code changes required.
