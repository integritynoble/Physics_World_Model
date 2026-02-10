# PWM Weekly Challenge: 2026-W10

## CASSI Spectral Reconstruction

## Modality

CASSI (see `contrib/casepacks/cassi_spectral_imaging_v1.json`)

## Objective

Reconstruct a 64x64x8 hyperspectral cube from a single coded-aperture snapshot
measurement. The forward model applies a binary mask and spectral dispersion
before summing onto a 2D detector.

## Rules

1. **Input data**: Generate using the provided `generate_data.py` script.
2. **Submission format**: A valid RunBundle (v0.3.0) packaged as a `.zip` file.
3. **Required artifacts**: `x_gt`, `y`, `x_hat` (minimum).
4. **Required metrics**: `psnr_db`, `ssim`, `runtime_s` in `runbundle_manifest.json`.
5. **Hash verification**: All artifact hashes must match file contents.
6. **Solver constraints**: Any solver allowed. Must run in under 300 seconds on CPU.
7. **Deadline**: Sunday 2026-03-08 23:59 UTC.

## Evaluation Criteria

1. Primary metric: PSNR (dB) -- higher is better
2. Secondary metric: SSIM
3. Runtime (lower is better)

## Data Generation

```bash
cd community/challenges/2026-W10
python generate_data.py --output ./data
```

## Submission

```bash
python community/validate.py my_submission.zip
python community/leaderboard.py --week 2026-W10
```

See `community/CONTRIBUTING_CHALLENGE.md` for full participation guide.

## Reference Performance

Baseline GAP-TV solver achieves ~30.6 dB PSNR. See `expected.json` for details.

## Notes

This challenge is adapted from the CASSI benchmark in `packages/pwm_core/benchmarks/`.
Focus areas: spectral unmixing, coded aperture optimization, PnP denoisers.
