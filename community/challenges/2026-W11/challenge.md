# PWM Weekly Challenge: 2026-W11

## Single-Pixel Camera Reconstruction

## Modality

SPC (see `contrib/casepacks/spc_low_sampling_poisson_v1.json`)

## Objective

Reconstruct a 64x64 image from compressive single-pixel measurements at 25%
sampling ratio using Hadamard patterns.

## Rules

1. **Input data**: Generate using the provided `generate_data.py` script.
2. **Submission format**: A valid RunBundle (v0.3.0) packaged as a `.zip` file.
3. **Required artifacts**: `x_gt`, `y`, `x_hat` (minimum).
4. **Required metrics**: `psnr_db`, `ssim`, `runtime_s` in `runbundle_manifest.json`.
5. **Hash verification**: All artifact hashes must match file contents.
6. **Solver constraints**: Any solver allowed. Must run in under 120 seconds on CPU.
7. **Deadline**: Sunday 2026-03-15 23:59 UTC.

## Evaluation Criteria

1. Primary metric: PSNR (dB) -- higher is better
2. Secondary metric: SSIM
3. Runtime (lower is better)

## Data Generation

```bash
cd community/challenges/2026-W11
python generate_data.py --output ./data
```

## Submission

```bash
python community/validate.py my_submission.zip
python community/leaderboard.py --week 2026-W11
```

See `community/CONTRIBUTING_CHALLENGE.md` for full participation guide.

## Reference Performance

Baseline PnP-FISTA achieves ~30.9 dB PSNR. See `expected.json` for details.

## Notes

This challenge tests compressive sensing reconstruction at low sampling ratios.
Focus areas: sparsity-based methods, learned denoisers, measurement optimization.
