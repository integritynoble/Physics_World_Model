# CorrectionResult Schema Contract

**Status:** FROZEN — Task C must implement this exact schema in `agents/contracts.py`.

## Schema Definition

```python
class CorrectionResult(StrictBaseModel):
    """Result of operator parameter correction with uncertainty."""

    theta_corrected: dict[str, float]
    """Corrected operator parameters. Keys are parameter names."""

    theta_uncertainty: dict[str, tuple[float, float]]
    """95% confidence interval per parameter. Keys match theta_corrected.
    Each value is (lower_bound, upper_bound)."""

    improvement_db: float
    """PSNR improvement in dB (corrected - uncorrected)."""

    n_evaluations: int
    """Total number of forward model evaluations during correction."""

    convergence_curve: list[float]
    """PSNR (or loss) at each major iteration. Length = number of iterations."""

    bootstrap_seeds: list[int]
    """RNG seeds used for each bootstrap resample. Length = K (typically 20).
    Stored for deterministic reproducibility."""

    resampling_indices: list[list[int]]
    """Bootstrap resampling indices. resampling_indices[k] is the list of
    sample indices used in bootstrap resample k. Stored in RunBundle."""
```

## Field Requirements

| Field | Type | Required | Constraint |
|-------|------|----------|------------|
| `theta_corrected` | dict[str, float] | YES | No NaN/Inf values |
| `theta_uncertainty` | dict[str, tuple[float, float]] | YES | Keys must match `theta_corrected`; lower <= upper |
| `improvement_db` | float | YES | Finite float |
| `n_evaluations` | int | YES | > 0 |
| `convergence_curve` | list[float] | YES | Non-empty; all finite |
| `bootstrap_seeds` | list[int] | YES | Length = K (number of resamples) |
| `resampling_indices` | list[list[int]] | YES | Length = K; each sublist has valid sample indices |

## Bootstrap Protocol

1. Given calibration data of N samples, resample with replacement K=20 times.
2. For each resample k:
   - Use `bootstrap_seeds[k]` as the RNG seed for index generation.
   - Store the indices in `resampling_indices[k]`.
   - Run the correction function on the resampled data.
   - Record corrected theta.
3. Compute 95% CI per parameter from the K theta estimates (percentile method: 2.5th and 97.5th percentiles).
4. Report `theta_uncertainty[param] = (ci_low, ci_high)`.

## Validation Rules

1. `theta_uncertainty` keys must be a superset of `theta_corrected` keys.
2. For each parameter: `ci_low <= theta_corrected[param] <= ci_high`.
3. `len(bootstrap_seeds) == len(resampling_indices)`.
4. All floats must be finite (StrictBaseModel enforces this).

## Compatibility

This extends the existing `CorrectionResult`-like data in the codebase. The new fields (`theta_uncertainty`, `convergence_curve`, `bootstrap_seeds`, `resampling_indices`) are additions — existing correction code paths can populate them as empty/default until bootstrap is wired in.
