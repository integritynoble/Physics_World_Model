# ExperimentSpec v0.2.1 — PWM Specification

This document defines **ExperimentSpec v0.2.1**, the core configuration model used by **PWM (Physics World Model)**.

ExperimentSpec is designed to be:
- **Modality-agnostic**: covers microscopy, computational photography, tomography, spectral imaging, lensless, etc.
- **Reproducible**: every run is deterministic when seeds are fixed.
- **Diagnosable**: errors can be attributed to explicit “world states.”
- **Extensible**: new modalities and operators can be added via plugins and CasePacks.

---

## 1. Overview

PWM runs a pipeline:

**Prompt/Spec → Resolve & Validate → Build PhysicsTrue/PhysicsModel → (Simulate or Load) y → (Fit operator θ) → Reconstruct x̂ → Diagnose → Recommend actions → RunBundle**

There are two primary operating modes:

1) **Simulation Mode**: the spec defines a forward model and PWM synthesizes measurements `y`.
2) **Measured Mode**: the user provides measurements `y` (and optionally an operator `A`), and PWM reconstructs/diagnoses.
3) **Operator-Correction Mode** (subset of Measured Mode): the user provides `y` + an imperfect operator/matrix `A` (or a parametric operator family), and PWM **fits/corrects** the forward model before reconstructing.

---

## 2. Top-level schema

At a high level:

```yaml
version: "0.2.1"
id: "run-uuid-or-user-label"

input:
  mode: "simulate" | "measured"
  data:
    # simulate: where x comes from; measured: where y comes from
  operator:   # optional, typically used in measured/operator-correction mode
    kind: "matrix" | "callable" | "parametric"
    ...

states:
  physics:        # required
  budget:         # optional but recommended
  calibration:    # optional but recommended
  environment:    # optional
  sample:         # optional
  sensor:         # optional
  compute:        # optional
  task:           # optional

recon:
  portfolio:       # solver set + auto-tuning policy
  outputs:         # what to export (xhat, uncertainty, logs, etc.)

analysis:
  metrics:
  residual_tests:
  advisor:

export:
  runbundle:
    path:
    data_policy:
    codegen:
```

PWM supports JSON and YAML. The canonical internal representation is JSON.

---

## 3. Versioning rules

- `version` is required and must be `"0.2.1"` for this spec.
- Minor changes (0.2.x) preserve field meanings; new optional fields may be added.
- Major changes (0.x → 1.x) may change field meanings.

---

## 4. Input block

### 4.1 input.mode
- `"simulate"`: PWM will generate `y` by applying PhysicsTrue to synthetic `x`.
- `"measured"`: PWM will load `y` from `input.data`.

### 4.2 input.data

#### (A) Simulate mode
Defines how to generate or load ground truth `x`.

```yaml
input:
  mode: simulate
  data:
    x_source:
      kind: "dataset" | "synthetic" | "file"
      dataset_id: "cave"                 # example
      split: "train" | "val" | "test"
      index: 0
      transforms:
        - type: "crop"
          size: [256, 256]
```

#### (B) Measured mode
Defines how to load measurements `y`.

```yaml
input:
  mode: measured
  data:
    y_source:
      kind: "file" | "dataset"
      path: "data/measured_y.pt"
      format: "pt" | "npz" | "mat" | "tif" | "h5" | "zarr"
      keys:
        y: "y"              # if container format
      meta:
        units: "photons"    # optional
        shape_hint: [H, W, C]  # optional
```

---

## 5. Operator input (optional)

Operator input is optional because many modalities can build operators from PhysicsState directly.

It becomes important in **measured mode** when:
- the user provides an explicit **matrix A**, or
- the user provides callable forward/adjoint functions, or
- the user specifies a parametric operator family and wants PWM to fit θ.

### 5.1 operator.kind = "matrix"

```yaml
input:
  operator:
    kind: matrix
    matrix:
      path: "data/A_matrix.npz"
      format: "npz" | "pt"
      keys:
        A: "A"
      shape: [M, N]
      storage: "dense" | "csr" | "csc"
```

### 5.2 operator.kind = "callable"
Uses a plugin ID registered in `pwm_core/registry.py`:

```yaml
input:
  operator:
    kind: callable
    callable:
      operator_id: "my_lab.forward_operator_v1"
      params:
        some_param: 1.0
```

### 5.3 operator.kind = "parametric"
Uses a named operator family and a θ search space:

```yaml
input:
  operator:
    kind: parametric
    parametric:
      operator_id: "cassi"
      theta_space:
        dx: {min: -2.0, max: 2.0}
        dy: {min: -2.0, max: 2.0}
        disp_poly: {coeffs_min: [-0.02, -0.01], coeffs_max: [0.02, 0.01]}
        psf_sigma: {min: 0.8, max: 2.2}
```

---

## 6. States block

### 6.1 PhysicsState (required)
Defines the forward model family and its core geometry.

Minimal:

```yaml
states:
  physics:
    modality: "cassi" | "sci_video" | "spc" | "widefield" | "confocal" | "sim" | "lightsheet" | "ct" | "mri" | "ptychography" | "holography" | "nerf"
    dims:
      x: [H, W, C]            # latent dimensions (may be 2D/3D)
      y: [M]                  # measurement dims (optional if inferred)
    operator:
      model: "linear" | "nonlinear"
      notes: "optional"
```

CASSI example:

```yaml
states:
  physics:
    modality: cassi
    dims:
      x: [256, 256, 31]
    cassi:
      mask:
        type: "binary"
        pattern_id: "mask_v1"
      dispersion:
        model: "poly"
        order: 2
      sensor:
        integration: "sum"
```

### 6.2 BudgetState
Controls dose and sampling.

```yaml
states:
  budget:
    photon_budget:
      max_photons: 1500.0
      exposure_time: 1.0
      bleaching:
        enabled: false
    measurement_budget:
      sampling_rate: 0.1         # e.g., SPC/MRI
      num_frames: 9              # e.g., SIM/SCI-video
      scan_speed: null
```

### 6.3 CalibrationState (θ)
Parameters that are often wrong in real systems.

```yaml
states:
  calibration:
    alignment:
      dx: 0.0
      dy: 0.0
      rotation_deg: 0.0
    psf:
      sigma: 1.3
      depth_variant: false
    dispersion:
      poly_coeffs: [0.0, 0.0]
    gain:
      scale: 1.0
      bias: 0.0
    timing:
      jitter_sigma: 0.0
```

### 6.4 EnvironmentState
Background/scattering/attenuation.

```yaml
states:
  environment:
    background_level: 0.0
    autofluorescence: 0.0
    attenuation:
      enabled: false
      model: "exp"
      k: 0.0
    scatter:
      enabled: false
      strength: 0.0
```

### 6.5 SampleState
Motion/drift/dynamics.

```yaml
states:
  sample:
    motion:
      enabled: true
      drift_px_per_frame: 0.2
      random_walk: false
    fluorescence:
      blinking:
        enabled: false
      labeling_density: null
```

### 6.6 SensorState
Read noise, quantization, saturation, FPN, nonlinearity.

```yaml
states:
  sensor:
    read_noise_sigma: 0.01
    shot_noise:
      enabled: true
      model: "poisson"
    quantization_bits: 12
    saturation_level: 4095
    fixed_pattern_noise:
      enabled: false
      sigma: 0.0
    nonlinearity:
      gamma: 1.0
```

### 6.7 ComputeState (optional)
Guides algorithm choices by resource limits.

```yaml
states:
  compute:
    device: "cuda" | "cpu"
    max_seconds: 60
    max_memory_gb: 8
    streaming:
      enabled: false
      chunk_size: 64
```

### 6.8 TaskState (optional)
What the pipeline should do.

```yaml
states:
  task:
    kind: "simulate_recon_analyze" | "reconstruct_only" | "fit_operator" | "calibrate_and_reconstruct" | "design_of_experiments" | "qc_report"
    targets:
      objective: "psnr" | "ssim" | "task_specific"
```

---

## 7. PhysicsTrue vs PhysicsModel

PWM can instantiate two physics objects:

- **PhysicsTrue**: used to simulate measurements `y` (in simulate mode)
- **PhysicsModel**: used in recon/fit loops (often imperfect)

This enables realistic mismatch studies:

```yaml
states:
  calibration:
    alignment: {dx: 0.0, dy: 0.0}

mismatch:
  enabled: true
  true_minus_model:
    alignment:
      dx: {dist: "uniform", min: -2.0, max: 2.0}
      dy: {dist: "uniform", min: -2.0, max: 2.0}
```

---

## 8. Reconstruction block

PWM supports a solver portfolio with auto-tuning:

```yaml
recon:
  portfolio:
    solvers:
      - id: "tv_fista"
        params:
          lam: 0.02
          iters: 200
      - id: "pnp_deepinv"
        params:
          denoiser: "drunet_gray"
          sigma: 0.02
          iters: 50
    selection:
      policy: "best_score"
      score: "residual_whiteness+psnr_proxy"
  outputs:
    save_xhat: true
    save_uncertainty: false
```

---

## 9. Operator-fit / calibration settings

Used for `task.kind` = `fit_operator` or `calibrate_and_reconstruct`.

```yaml
mismatch:
  fit_operator:
    enabled: true
    theta_space_ref: "input.operator.parametric.theta_space"   # or inline
    search:
      candidates: 12
      refine_top_k: 3
      refine_steps: 8
      strategy: "coarse_grid+local"
    proxy_recon:
      solver_id: "tv_fista"
      budget:
        iters: 40
    scoring:
      terms:
        - name: "data_fidelity"
          weight: 1.0
        - name: "residual_whiteness"
          weight: 0.5
        - name: "theta_prior"
          weight: 0.2
    stop:
      max_evals: 20
      plateau_delta: 1e-3
      verify_required: true
```

---

## 10. Analysis block

```yaml
analysis:
  metrics:
    - psnr
    - ssim
  residual_tests:
    - whiteness
    - fourier_structure
  advisor:
    enabled: true
    knobs_to_sweep:
      - "states.budget.photon_budget.max_photons"
      - "states.budget.measurement_budget.sampling_rate"
    max_candidates: 10
```

The **advisor** must output structured actions:

```yaml
analysis:
  advisor:
    enabled: true
    output_actions: true
```

---

## 11. Export block

```yaml
export:
  runbundle:
    path: "runs/"
    name: "latest"
    data_policy:
      mode: "auto"         # auto/copy/reference
      copy_threshold_mb: 100
    codegen:
      enabled: true
      include_internal_state: true
    viewer:
      enabled: true
```

---

## 12. Minimal examples

### 12.1 CASSI simulation + recon + analyze
```yaml
version: "0.2.1"
id: "cassi_sim_demo"
input:
  mode: simulate
  data:
    x_source:
      kind: dataset
      dataset_id: cave
      split: test
      index: 0

states:
  physics:
    modality: cassi
    dims: {x: [256,256,31]}
  budget:
    photon_budget: {max_photons: 1200.0}
    measurement_budget: {sampling_rate: 1.0}
  calibration:
    alignment: {dx: 0.0, dy: 0.0}
    dispersion: {poly_coeffs: [0.0, 0.0]}
  sensor:
    shot_noise: {enabled: true, model: poisson}
    read_noise_sigma: 0.01
  task:
    kind: simulate_recon_analyze

recon:
  portfolio:
    solvers:
      - id: tv_fista
        params: {lam: 0.02, iters: 200}

analysis:
  metrics: [psnr, ssim]
  residual_tests: [whiteness]

export:
  runbundle:
    path: runs/
    name: latest
    data_policy: {mode: auto, copy_threshold_mb: 100}
    codegen: {enabled: true}
    viewer: {enabled: true}
```

### 12.2 Measured y + parametric operator-fit + recon
```yaml
version: "0.2.1"
id: "cassi_measured_fit"
input:
  mode: measured
  data:
    y_source:
      kind: file
      path: data/measured_y.pt
      format: pt
  operator:
    kind: parametric
    parametric:
      operator_id: cassi
      theta_space:
        dx: {min: -2.0, max: 2.0}
        dy: {min: -2.0, max: 2.0}
        disp_poly: {coeffs_min: [-0.02,-0.01], coeffs_max: [0.02,0.01]}
        psf_sigma: {min: 0.8, max: 2.2}

states:
  physics:
    modality: cassi
    dims: {x: [256,256,31]}
  task:
    kind: calibrate_and_reconstruct

mismatch:
  fit_operator:
    enabled: true
    search: {candidates: 12, refine_top_k: 3, refine_steps: 8}
    proxy_recon: {solver_id: tv_fista, budget: {iters: 40}}

recon:
  portfolio:
    solvers:
      - id: pnp_deepinv
        params: {denoiser: drunet_gray, sigma: 0.02, iters: 50}

export:
  runbundle:
    path: runs/
    name: latest
    data_policy: {mode: auto}
```

---

## 13. Notes for implementers

- Treat all specs as **untrusted input**: validate, clamp, and auto-repair.
- Always emit a `ValidationReport` and, if used, an `AutoRepairPatch`.
- Prefer bounded search; avoid unbounded LLM-driven parameter hallucination.
- Keep operator-fit deterministic and reproducible: store candidate sets and scores.

---

## Appendix: Field stability and deprecations

- In v0.2.1, `states.physics.modality` is mandatory.
- In future versions, `input.operator` may move under `states.physics.operator` for tighter coupling; v0.2.1 keeps it under `input` to support user-provided matrices easily.

