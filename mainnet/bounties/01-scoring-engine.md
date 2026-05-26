# Bounty 1 — PWM Scoring Engine

- **Amount:** 200,000 PWM
- **Opens:** when `agent-scoring` reference merges to `main` (~Day 17)
- **Reference implementation:** `infrastructure/agent-scoring/pwm_scoring/`
- **Mandatory interface:** `coordination/agent-coord/interfaces/scoring_api.py`
- **Acceptance harness:** `infrastructure/agent-scoring/tests/`

## What you build

A Python package named `pwm_scoring` that takes a benchmark manifest plus a
solver's raw output and returns a verified `SolutionScore` plus a signed
certificate payload ready for submission to `PWMCertificate.submit()`. Every
downstream component (CLI, mining client) consumes this package as a library.

## Interface contract

```python
# from scoring_api.py — drop-in replacement required
from pwm_scoring import score_solution, SolutionScore

result: SolutionScore = score_solution(
    benchmark_manifest = dict,     # parsed YAML from benchmark spec
    instance_dir       = Path,     # directory of benchmark instances
    solver_output      = np.ndarray,  # raw solver output, per-instance stacked
    omega_params       = dict,     # Ω tier parameters
)

# result fields (all required):
#   psnr_per_instance:   list[float]
#   ssim_per_instance:   list[float]
#   Q:                   float in [0.0, 1.0]
#   track_a_pass:        bool   (stratified worst-case)
#   track_b_pass:        bool   (aggregate quality)
#   track_c_pass:        bool   (physics-consistency)
#   gate_verdicts:       dict {"S1": "pass"|"fail", "S2": ..., "S3": ..., "S4": ...}
#   cert_payload:        dict — must validate against cert_schema.json
```

Read the full dataclass in `interfaces/scoring_api.py`. Changing field names
or types is a rejection at triage.

## What must pass

Your submission must pass **every test** in `infrastructure/agent-scoring/tests/`
verbatim:

1. **Track A — stratified worst-case.** For each instance bucket (easy/med/hard),
   worst-case PSNR must exceed the floor declared in the benchmark manifest.
   See pwm_overview1.md §Track A.

2. **Track B — aggregate quality.** Mean PSNR (or benchmark-declared metric)
   exceeds declared threshold.

3. **Track C — physics-consistency.** Forward-model residuals pass the
   manifest's epsilon_fn check. For imaging benchmarks this is a reconstruction
   residual; for PDE benchmarks it is a conservation-law residual.

4. **S1–S4 gates.** Four orthogonal integrity checks:
   - S1: no solver contact with ground truth
   - S2: solver runtime within declared compute budget
   - S3: output shape and dtype match declared schema
   - S4: no hard-coded per-instance hacks (structural similarity of solve path)

5. **Q_p combination.** The final `Q` scalar must match the reference impl's
   value to 4 decimal places. Formula in pwm_overview1.md §Q_p Formula.

6. **cert_payload validity.** The produced `cert_payload` must validate
   against `interfaces/cert_schema.json` and must be acceptable to
   `PWMCertificate.submit()` on Sepolia testnet.

## What you may change

- Internal algorithms (GPU vectorization, alternate S1 heuristic)
- Package layout under `pwm_scoring/`
- Dependencies (so long as they are MIT/Apache-2/BSD; no paid SaaS)
- Language for non-interface components (e.g., Rust S3 gate implementation),
  provided the public API is still a Python importable module

## What you may not change

- The public interface (`score_solution` signature, `SolutionScore` dataclass)
- The `cert_payload` schema
- The Q_p formula or the gate definitions

## Shadow run

Accepted submissions run for 30 days as a parallel scorer on Sepolia testnet.
We compare your Q score against the reference on every submitted L4 certificate.
Divergence > 0.01 on any instance is a regression event; 3 regression events
void the award. Divergence data is published to `reviews/bounty-1.md`.

## Payment

- Funded from Reserve (2.1M PWM).
- Paid once shadow run completes cleanly.
- Paid to the wallet listed in your PR description; confirm on-chain before
  opening the PR (Sepolia receive tx is sufficient proof).
- Mainnet PWM swap: bounty winners receive mainnet PWM at launch, 1:1 with
  testnet PWM held at Phase 2 sign-off.
