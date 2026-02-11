# PWM RunBundle Format (v0.2.x)

A **RunBundle** is PWM’s reproducible, portable output artifact for every run
(simulation, reconstruction, operator fitting, or design sweep).

RunBundles are designed to:
- preserve **reproducibility** (spec + seeds + code + internal state),
- support **big data** without copying (reference mode),
- support **sharing & inspection** (viewer + report),
- enable **AI_Scientist/agents** to programmatically iterate via `suggested_actions`.

---

## 1. Directory layout

A RunBundle is a folder with a stable internal structure:

```text
<runbundle>/
  bundle.json                   # top-level manifest (pointer index)
  spec/
    spec_input.json             # user-provided spec (as received)
    spec_resolved.json          # resolved + normalized + validated spec
    validation_report.json      # validation results + warnings/errors
    auto_repair_patch.json      # optional; patch applied to fix invalid input
  provenance/
    versions.json               # library versions, git hash, CUDA, OS
    environment.txt             # pip freeze/conda export (best effort)
    seeds.json                  # random seeds used
    command.txt                 # CLI command used
  data/
    data_manifest.json          # lists all inputs/outputs, copy/reference, checksums
    inputs/                     # optional; present when copying small data
    outputs/                    # optional; present when exporting derived arrays/images
  internal_state/
    perturbations.pt            # exact perturbations used (noise masks, drift vectors, etc.)
    operator_fit/               # present for y+A or θ-fit runs
      candidates.json           # candidate θ list
      scores.json               # scoring per candidate
      trajectory.json           # refinement trajectory for selected candidates
      theta_best.json           # best-fit θ
      residual_evidence.json    # whiteness/fourier/etc. evidence
  results/
    recon/
      xhat.pt                   # reconstructed latent estimate (if saved)
      xhat_preview.png          # optional preview
      uncertainty.pt            # optional uncertainty outputs
    metrics.json                # PSNR/SSIM/etc. (when ground truth available)
    analysis.json               # structured diagnosis results + advisor actions
    report.md                   # human-readable report
    report.json                 # machine-readable report (DiagnosisResult)
  exports/
    code/
      simulate.py               # generated script (artifact-loaded)
      reconstruct.py
      fit_operator.py           # optional
      requirements.txt          # best-effort pinned env
    notebook/
      reproduce.ipynb           # optional
  viewer/
    snapshot.png                # optional static snapshot
```

---

## 2. bundle.json (top-level index)

`bundle.json` acts as an index and compatibility anchor:

```json
{
  "runbundle_version": "0.2.1",
  "created_at": "2026-02-01T12:34:56Z",
  "id": "cassi_measured_fit__20260201_123456",
  "paths": {
    "spec_input": "spec/spec_input.json",
    "spec_resolved": "spec/spec_resolved.json",
    "validation_report": "spec/validation_report.json",
    "auto_repair_patch": "spec/auto_repair_patch.json",
    "data_manifest": "data/data_manifest.json",
    "analysis": "results/analysis.json",
    "report_md": "results/report.md",
    "report_json": "results/report.json",
    "metrics": "results/metrics.json",
    "xhat": "results/recon/xhat.pt"
  }
}
```

Notes:
- Some fields may be absent depending on task kind (e.g., no `xhat` for pure simulation-only runs).
- All paths are relative to the RunBundle root.

---

## 3. data_manifest.json (copy vs reference)

The `data_manifest.json` lists all **inputs and outputs**, describing whether they are stored inside the bundle or referenced externally.

### 3.1 Schema overview

```json
{
  "policy": {
    "mode": "auto",
    "copy_threshold_mb": 100
  },
  "entries": [
    {
      "role": "input",
      "name": "measured_y",
      "type": "tensor",
      "mode": "reference",
      "uri": "file:///mnt/lab_server/data/huge_stack_y.pt",
      "relative_path": null,
      "checksum": "sha256:...",
      "size_bytes": 512345678901,
      "shape": [1024, 1024],
      "dtype": "float32",
      "format": "pt"
    },
    {
      "role": "output",
      "name": "xhat",
      "type": "tensor",
      "mode": "copy",
      "uri": null,
      "relative_path": "results/recon/xhat.pt",
      "checksum": "sha256:...",
      "size_bytes": 12345678,
      "shape": [256, 256, 31],
      "dtype": "float32",
      "format": "pt"
    }
  ]
}
```

### 3.2 Fields
- `role`: `"input"` or `"output"`
- `name`: short semantic name (e.g., `"measured_y"`, `"A_matrix"`, `"xhat"`)
- `type`: `"tensor" | "matrix" | "image" | "json" | "text" | "binary"`
- `mode`: `"copy"` or `"reference"`
- `uri`: required if mode is `reference` (e.g., `file://`, `s3://`, `gs://`)
- `relative_path`: required if mode is `copy` (path inside bundle)
- `checksum`: recommended for both modes
- `size_bytes`: recommended for both modes
- `shape`, `dtype`, `format`: optional but recommended

---

## 4. spec/ artifacts

### 4.1 spec_input.json
The user-provided spec as received (after parsing), preserved verbatim when possible.

### 4.2 spec_resolved.json
The validated, normalized, default-filled final spec used for execution.
- Units normalized
- Defaults filled
- Hard constraints applied

### 4.3 validation_report.json
Contains:
- errors (blocking)
- warnings (non-blocking)
- inferred assumptions
- value clamps and normalizations

### 4.4 auto_repair_patch.json (optional)
If PWM auto-repairs invalid values, it records a patch:
- which fields changed
- old value → new value
- reason

---

## 5. internal_state/

This folder ensures **bit-exact reproducibility** even when simulation involves randomness or complex mismatch logic.

### 5.1 perturbations.pt
Should store exact realization of:
- Poisson/gaussian noise seeds or sampled noise masks (if stored)
- drift vectors, motion paths
- aberration maps, phase screens
- calibration perturbations

If storing full noise masks is too large, store:
- seeds + PRNG algorithm version
- plus any non-deterministic runtime values

### 5.2 operator_fit/
Present when task involves fitting θ or correcting operator A.

Recommended files:
- `candidates.json`: list of θ candidates evaluated
- `scores.json`: scoring breakdown per candidate
- `trajectory.json`: local refine steps for top-K candidates
- `theta_best.json`: best θ
- `residual_evidence.json`: residual tests evidence (whiteness, Fourier structure, etc.)

---

## 6. results/

### 6.1 results/analysis.json
A machine-readable diagnosis summary.
Should contain:
- `verdict` (e.g., `"Dose-limited"`, `"Drift-limited"`, `"Model-mismatch"`),
- `confidence`,
- `evidence` (metrics + test statistics),
- `suggested_actions` (structured knob operations).

### 6.2 results/report.md
Human-readable report for sharing.

### 6.3 results/report.json
Machine-readable report, usually mirroring `analysis.json` but may include additional details.

### 6.4 results/recon/
Reconstruction outputs (optional depending on task).

---

## 7. exports/

PWM can generate artifact-loaded scripts to reproduce results:

- `simulate.py`: reproduces y generation using stored internal_state.
- `reconstruct.py`: reconstructs x̂ using resolved spec + stored operator fit results.
- `fit_operator.py`: repeats operator fitting, using stored candidates/seeds.

Key requirement:
> Generated scripts MUST load internal_state artifacts rather than “re-guessing” them.

---

## 8. Reference mode and big data

In `auto` mode, PWM uses a threshold (default `copy_threshold_mb`) to decide:
- copy small inputs into the bundle,
- reference large inputs via URI.

For scientific reproducibility, reference entries should include:
- checksum
- size
- acquisition metadata (when possible)

---

## 9. Compatibility rules

- A RunBundle should be readable by **any** PWM version that supports its `runbundle_version` (0.2.x).
- Breaking changes must bump major version.
- `bundle.json` is the compatibility anchor: tools should rely on it first.

---

## 10. Minimal RunBundle example (operator correction)

A minimal operator-correction run will typically contain:

- `spec/spec_resolved.json`
- `data/data_manifest.json` (with `measured_y` and `A_matrix`)
- `internal_state/operator_fit/theta_best.json`
- `results/recon/xhat.pt`
- `results/report.md` + `results/report.json`

---

## 11. Notes for contributors

When adding new outputs:
- register them in `bundle.json` paths if they’re core artifacts
- add them to `data_manifest.json` entries (especially if large)
- prefer small previews (PNG/JPEG) for quick viewer rendering

