# PWM Weekly Challenge: {{WEEK_ID}}

## Challenge Title

{{TITLE}}

## Modality

{{MODALITY}} (see `contrib/casepacks/{{CASEPACK_ID}}.json`)

## Objective

{{OBJECTIVE_DESCRIPTION}}

## Rules

1. **Input data**: Generate using the provided `generate_data.py` script.
2. **Submission format**: A valid RunBundle (v0.3.0) packaged as a `.zip` file.
3. **Required artifacts**: `x_gt`, `y`, `x_hat` (minimum). Additional artifacts welcome.
4. **Required metrics**: `psnr_db`, `ssim`, `runtime_s` in `runbundle_manifest.json`.
5. **Hash verification**: All artifact hashes must match file contents.
6. **Solver constraints**: {{SOLVER_CONSTRAINTS}}
7. **Deadline**: Sunday 23:59 UTC of challenge week.

## Evaluation Criteria

Submissions are ranked by **primary metric** (see `expected.json`), with ties broken by:

1. Primary metric: {{PRIMARY_METRIC}} (higher is better)
2. Secondary metric: {{SECONDARY_METRIC}}
3. Runtime (lower is better)

## Data Generation

```bash
cd community/challenges/{{WEEK_ID}}
python generate_data.py --output ./data
```

This generates a small dataset slice from existing PWM benchmark data.
No large files are committed to the repository.

## Submission

```bash
# Validate your submission
python community/validate.py my_submission.zip

# Check expected metrics
python community/leaderboard.py --week {{WEEK_ID}}
```

See `community/CONTRIBUTING_CHALLENGE.md` for full participation guide.

## Reference Performance

See `expected.json` for baseline metrics from the default solver pipeline.

## Notes

{{NOTES}}
