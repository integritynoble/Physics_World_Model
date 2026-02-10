# Contributing to PWM Weekly Challenges

This guide explains how to participate in the Physics World Model (PWM) weekly
reconstruction challenges.

## Overview

Each week, a new computational imaging challenge is posted in
`community/challenges/<YYYY-Www>/`. Participants reconstruct an image from
simulated measurements and submit a RunBundle for scoring.

## Quick Start

```bash
# 1. Check the current challenge
ls community/challenges/

# 2. Read the challenge description
cat community/challenges/2026-W10/challenge.md

# 3. Generate the challenge dataset
cd community/challenges/2026-W10
python generate_data.py --output ./data

# 4. Write your reconstruction (produces x_hat.npy)
# ... your code here ...

# 5. Package as a RunBundle and validate
python community/validate.py my_submission.zip

# 6. Check the leaderboard
python community/leaderboard.py --week 2026-W10
```

## RunBundle Format

Your submission must be a `.zip` file containing a valid RunBundle v0.3.0.
The zip must include a `runbundle_manifest.json` at the root (or one level down).

### Required manifest fields

```json
{
  "version": "0.3.0",
  "spec_id": "your_unique_run_id",
  "timestamp": "2026-03-01T12:00:00Z",
  "provenance": {
    "git_hash": "abc1234",
    "seeds": [42],
    "platform": "linux-x86_64",
    "pwm_version": "0.3.0"
  },
  "metrics": {
    "psnr_db": 30.5,
    "ssim": 0.92,
    "runtime_s": 12.3
  },
  "artifacts": {
    "x_gt": "data/x_gt.npy",
    "y": "data/y.npy",
    "x_hat": "results/x_hat.npy"
  },
  "hashes": {
    "x_gt": "sha256:<hex>",
    "y": "sha256:<hex>",
    "x_hat": "sha256:<hex>"
  }
}
```

See `docs/contracts/runbundle_schema.md` for the full specification.

## Validation

Before submitting, always validate your RunBundle:

```bash
python community/validate.py my_submission.zip
```

The validator checks:
- Manifest JSON is well-formed
- All required fields are present
- Version is `"0.3.0"`
- Metrics are finite floats (no NaN, no Inf)
- Timestamp is valid ISO 8601
- All artifact files exist
- All SHA256 hashes match file contents

## Scoring

Submissions are scored against the `expected.json` reference metrics for each
challenge week. The scoring process:

1. **Threshold check**: Submissions below minimum PSNR/SSIM thresholds or
   above maximum runtime are marked INVALID.
2. **Primary ranking**: By primary metric (usually PSNR) -- higher is better.
3. **Tiebreaker**: Secondary metric (usually SSIM), then runtime (lower is better).

## Leaderboard

Generate or view the leaderboard:

```bash
python community/leaderboard.py --week 2026-W10
```

This produces `community/challenges/2026-W10/leaderboard.md`.

## Creating a New Challenge

If you want to propose a new weekly challenge:

1. Copy `community/challenges/template/` to a new week directory.
2. Fill in `challenge.md` with the problem description.
3. Set reference metrics in `expected.json`.
4. Write `generate_data.py` that creates data from NumPy operations only --
   do NOT commit large binary files.
5. Submit a PR with the new challenge directory.

### Data policy

- Challenges must NOT include large binary files (`.npy`, `.npz`, `.h5`).
- All data must be generated on-the-fly by `generate_data.py`.
- Scripts should use deterministic seeds for reproducibility.
- Keep generated datasets small (e.g., 64x64 or 128x128 images).

## Tips for Good Submissions

- Use deterministic seeds and record them in `provenance.seeds`.
- Include your git commit hash in `provenance.git_hash`.
- Compute honest runtime measurements (wall-clock, not CPU time).
- Consider both reconstruction quality AND computational efficiency.
- Additional metrics (e.g., `theta_error_rmse`, `sam`) are welcome as extras.

## Code of Conduct

- Submit only your own work.
- Do not hard-code ground truth into your reconstruction.
- Do not reverse-engineer the scoring system to inflate metrics.
- Be respectful in discussions and issue reports.

## Questions?

Open an issue on GitHub with the `challenge` label, or check existing challenge
discussions in the repository.
