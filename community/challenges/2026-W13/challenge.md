# PWM Weekly Challenge: 2026-W13

## Widefield Deconvolution Under Low Photon Budget

## Modality

Widefield (see `contrib/casepacks/widefield_lowdose_highbg_v1.json`)

## Objective

Reconstruct a 128x128 fluorescence microscopy image from a low-photon-count
widefield measurement with high background and Poisson + Gaussian noise.

## Rules

1. **Input data**: Generate using the provided `generate_data.py` script.
2. **Submission format**: A valid RunBundle (v0.3.0) packaged as a `.zip` file.
3. **Required artifacts**: `x_gt`, `y`, `x_hat` (minimum).
4. **Required metrics**: `psnr_db`, `ssim`, `runtime_s` in `runbundle_manifest.json`.
5. **Hash verification**: All artifact hashes must match file contents.
6. **Solver constraints**: Any solver allowed. Must run in under 60 seconds on CPU.
7. **Deadline**: Sunday 2026-03-29 23:59 UTC.

## Evaluation Criteria

1. Primary metric: PSNR (dB) -- higher is better
2. Secondary metric: SSIM
3. Runtime (lower is better)

## Data Generation

```bash
cd community/challenges/2026-W13
python generate_data.py --output ./data
```

## Submission

```bash
python community/validate.py my_submission.zip
python community/leaderboard.py --week 2026-W13
```

See `community/CONTRIBUTING_CHALLENGE.md` for full participation guide.

## Reference Performance

Baseline PnP denoiser achieves ~27.8 dB PSNR. See `expected.json` for details.

## Notes

This challenge tests deconvolution under extreme photon-limited conditions.
Focus areas: Plug-and-Play denoisers, Richardson-Lucy, Wiener filtering, deep priors.
