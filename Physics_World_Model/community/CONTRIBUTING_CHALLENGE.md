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

---

## Contributing Beyond Challenges: 4-Level Contribution Guide

PWM accepts contributions at 4 levels, from easy to advanced.

### Level 1: New Solver (Easiest)

**Who**: Any ML researcher, PhD student, or imaging lab.
**Time**: 1 day to first result.

```bash
# 1. Scaffold
pwm scaffold solver my_solver

# 2. Implement (edit contrib/solvers/my_solver/solver.py)
#    Your function: run_my_solver(y, physics, cfg) -> (x_hat, info)

# 3. Test locally
python contrib/solvers/my_solver/test_local.py

# 4. Run sandbox evaluation
pwm evaluate --sandbox --modality widefield --solver my_solver

# 5. Validate for PR
pwm contrib check my_solver

# 6. Submit PR (auto-labeled: fast-lane, auto-merge in 48h if CI passes)
```

**Your solver never knows what modality it's solving.** Write once, compete on all 64+ modalities.

**Paper**: "Our method achieves rho=0.85 across 20 modalities on LIP-Arena"

### Level 2: New Calibrator (Medium)

**Who**: Self-calibration, blind deconvolution, operator learning researchers.

```bash
pwm scaffold calibrator my_calibrator
# Implement: calibrate_my_method(y, H_nom, budget) -> (H_hat, info)
# H_nom exposes: get_theta(), set_theta(), forward(), adjoint()
```

**Paper**: "Our blind calibrator reduces oracle gap from 12 dB to 2 dB"

### Level 3: New Modality (Medium-Hard)

**Who**: Domain experts with a modality PWM doesn't cover.

```bash
pwm scaffold modality my_modality
# Fill in: graph.yaml, mismatch.yaml, photon.yaml, metrics.yaml, meta.yaml
pwm evaluate --sandbox --modality my_modality
```

Requires entries in all 5 YAML registries. See `docs/modality_pack_spec.md`.

**Paper**: "We formalize 4D-STEM as an OperatorGraph and benchmark 10 solvers"

### Level 4: New Primitive (Hardest -- RFC Process)

**Who**: Physics experts willing to implement a new atomic operator.

1. Open RFC issue with physics justification + adjoint proof
2. Implement `PrimitiveOp` (or use `contrib/templates/tier2_wrapper.py`)
3. Pass adjoint correctness tests
4. Community + steward review
5. Merge into `PRIMITIVE_REGISTRY`

**Paper**: "Our full-wave primitive improves fidelity by 3 dB across 5 modalities"

### Contribution without Code: Submit Path

Don't want to submit a PR? Just compete:

```bash
# Run locally, submit results only
pwm evaluate --modality cassi --solver my_solver --output ./results
pwm submit ./results/runbundle.zip
# Score appears on leaderboard. No fork, no PR needed.
```

See `docs/GOVERNANCE.md` for merge authority rules and `docs/IP_POLICY.md` for licensing.

---

## Questions?

Open an issue on GitHub with the `challenge` label, or check existing challenge
discussions in the repository.
