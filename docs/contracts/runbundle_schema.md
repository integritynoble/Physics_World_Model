# RunBundle Schema Contract (v0.3.0)

**Status:** FROZEN — All parallel tasks must produce RunBundles conforming to this schema.

## Minimal Required Fields

Every RunBundle directory must contain a `runbundle_manifest.json` at its root:

```json
{
  "version": "0.3.0",
  "spec_id": "<string: experiment/run identifier>",
  "timestamp": "<string: ISO 8601, e.g. 2026-02-09T14:30:00Z>",
  "provenance": {
    "git_hash": "<string: 7+ char commit hash>",
    "seeds": [42],
    "platform": "<string: e.g. linux-x86_64>",
    "pwm_version": "<string: e.g. 0.3.0>"
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
    "x_gt": "sha256:abc123...",
    "y": "sha256:def456...",
    "x_hat": "sha256:789ghi..."
  }
}
```

## Field Specifications

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | YES | Must be `"0.3.0"` |
| `spec_id` | string | YES | Unique run identifier |
| `timestamp` | string | YES | ISO 8601 datetime with timezone |
| `provenance` | object | YES | See below |
| `metrics` | object | YES | See below |
| `artifacts` | object | YES | Relative paths to stored arrays |
| `hashes` | object | YES | SHA256 hex digests for all artifacts |

### `provenance` object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `git_hash` | string | YES | Git commit hash at generation time |
| `seeds` | list[int] | YES | All RNG seeds used |
| `platform` | string | YES | OS + arch identifier |
| `pwm_version` | string | YES | PWM package version |

### `metrics` object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `psnr_db` | float | YES | Peak signal-to-noise ratio in dB |
| `ssim` | float | YES | Structural similarity index |
| `runtime_s` | float | YES | Wall-clock runtime in seconds |

Additional metrics (e.g., `sam`, `theta_error_rmse`) are allowed as extra keys.

### `artifacts` object

All values are **relative paths** from the RunBundle root directory.
Minimum required artifact keys: `x_gt`, `y`, `x_hat`.
Additional keys (e.g., `mask`, `psf`, `theta_corrected`) are allowed.

### `hashes` object

Every key in `artifacts` must have a corresponding key in `hashes` with value `"sha256:<hex_digest>"`.

## Validation Rules

1. `version` must equal `"0.3.0"` (reject older versions).
2. All artifact paths must exist as files relative to the RunBundle directory.
3. All hashes must match the actual file contents (verified by `pwm validate`).
4. `psnr_db`, `ssim`, `runtime_s` must be finite floats (no NaN, no Inf).
5. `timestamp` must parse as valid ISO 8601.
6. `seeds` must be non-empty.

## Compatibility with Existing RunBundle Format

This schema is a **simplified manifest** that can coexist with the full RunBundle format (v0.2.x `bundle.json`). The `runbundle_manifest.json` is the contract-level interface; the full `bundle.json` with `spec/`, `provenance/`, `data/`, etc. remains valid for detailed provenance.

Tasks producing RunBundles MUST write `runbundle_manifest.json`. They MAY also write the full v0.2.x structure.
