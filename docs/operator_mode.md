# Operator Mode (y + A) — PWM v0.2.1

This document describes PWM’s **Operator Mode**: users provide **measured data**
`y` and a forward operator `A` (or an operator family), and PWM performs:

- **fit/correct forward model** (optional but recommended),
- **reconstruct** the latent `x̂`,
- **diagnose** failure modes,
- **recommend** actionable improvements,
- **export** a reproducible RunBundle (including operator-fit artifacts).

Operator Mode is the foundation for workflows like:
- CASSI / SCI-spectral “operator correction” (dispersion, mask shift, PSF mismatch)
- SPC / SCI-video calibration from measurement residuals
- Generic linear inverse problems with user-provided matrices

---

## 1. When to use Operator Mode

Use Operator Mode when:

1) You have **real measured data** `y`.
2) You have an operator or model `A` but suspect it is **imperfect**.
3) You want PWM to:
   - automatically “repair” the forward model within safe bounds,
   - then reconstruct robustly,
   - and explain the bottleneck (dose vs mismatch vs motion vs sensor).

Typical signatures:
- recon is unstable or looks “structured wrong,”
- residuals show strong non-white structure,
- small perturbations to assumed parameters change outputs dramatically.

---

## 2. Supported operator inputs

PWM supports three kinds of operator input (see `docs/spec_v0.2.1.md`):

### 2.1 Matrix operator (`A`)
User provides an explicit matrix:

- dense (small problems)
- sparse CSR/CSC (preferred for large linear systems)
- optionally a `LinearOperator` wrapper for implicit multiplication

### 2.2 Callable operator (plugin)
User provides forward/adjoint via a plugin entrypoint:

- `forward(x) -> y`
- `adjoint(y) -> x`
- optional JVP/VJP hooks if available

### 2.3 Parametric operator family (`A(θ)`)
User provides:
- an operator family ID (e.g., `cassi`)
- a θ search space / safe bounds

PWM fits θ using bounded search + local refinement:
- coarse candidates → score via proxy recon + residual tests → refine top-K

---

## 3. The Operator Mode pipeline

Operator Mode adds an **operator-fit stage** before reconstruction:

**(Load y, A/model) → Resolve & Validate → Fit operator θ (optional) → Reconstruct → Diagnose → Recommend → Export**

### 3.1 Two-physics interpretation (mental model)
Even in measured mode, PWM uses a conceptual separation:

- **PhysicsTrue** is unknown in real experiments.
- **PhysicsModel** is your provided `A` or `A(θ0)`.

Operator fitting tries to find θ such that **PhysicsModel ≈ PhysicsTrue** under evidence from `y`.

---

## 4. Calibration/fit objectives (what “fit operator” means)

PWM can fit θ by optimizing a composite objective:

### 4.1 Data fidelity
- For linear models: `|| A(θ) x̂(θ) - y ||`
- In practice, we use a proxy reconstruction `x̂(θ)` with a fast solver.

### 4.2 Residual “whiteness” / structure tests
Good θ tends to produce residuals that look noise-like:
- whiteness test statistics
- Fourier-domain structure checks
- correlation with known patterns (e.g., SIM moiré, CASSI dispersion)

### 4.3 θ priors / regularizers
Prevent overfitting:
- small shift preference
- smooth dispersion polynomial penalty
- physically plausible ranges (hard clamps)

---

## 5. Bounded auto-refine (recommended)

Pure unbounded optimization can overfit noise or be too slow.

PWM defaults to **bounded refinement**:

- Evaluate ≤ *N* candidates (e.g., 12)
- Refine top-*K* (e.g., 3) with small local search (e.g., 8 steps)
- Stop early on plateau

This keeps operator mode:
- reproducible,
- stable,
- fast enough for real labs.

---

## 6. CLI usage

### 6.1 Fit operator only
This produces `theta_best.json` and fit diagnostics.

```bash
pwm fit-operator \
  --y data/measured_y.pt \
  --A data/A_matrix.npz \
  --casepack generic_matrix_yA_fit_gain_shift_v1 \
  --out runs/
```

### 6.2 Calibrate + reconstruct
This runs operator fit and then reconstruction.

```bash
pwm calib-recon \
  --y data/measured_y.pt \
  --casepack cassi_measured_y_fit_theta_v1 \
  --out runs/
pwm view runs/latest
```

---

## 7. CasePacks for Operator Mode

Operator Mode is driven by **CasePacks** that define:
- what θ means,
- safe bounds,
- candidate generation strategy,
- proxy recon solver and scoring.

Recommended starter packs:
- `cassi_measured_y_fit_theta_v1`
- `generic_matrix_yA_fit_gain_shift_v1`

These live under:
- `packages/pwm_core/contrib/casepacks/`

---

## 8. What gets exported in RunBundle

Operator Mode RunBundles include additional artifacts:

```text
internal_state/operator_fit/
  candidates.json
  scores.json
  trajectory.json
  theta_best.json
  residual_evidence.json
```

And typical outputs:
- `results/recon/xhat.pt`
- `results/report.md` + `results/report.json`
- `data/data_manifest.json` containing `y` and `A` (copy or reference)

See `docs/runbundle_format.md`.

---

## 9. CASSI-specific notes (SCI-spectral)

In CASSI, small mismatch often dominates:
- mask shifts (dx, dy)
- dispersion polynomial mismatch
- blur/PSF width mismatch
- throughput/gain drift

A good CASSI operator-fit θ parameterization usually includes:
- `dx, dy` (integer or subpixel)
- `disp_poly` coefficients (low-order)
- `psf_sigma` (Gaussian approx)
- optional gain/bias

PWM can fit these within bounded ranges and then reconstruct with:
- TV-FISTA, ADMM-TV, primal-dual, or PnP (DeepInv)

---

## 10. Safety and validity rules

Operator Mode must be **safe** and **physically plausible**:

- Validate dimensions: `A` must map `x` → `y`
- Clamp θ to physically plausible bounds
- Reject dangerous nonsense (e.g., invalid units, negative exposure)
- Record any auto-repair patches

Operator fitting should:
- store candidate sets and scores for reproducibility,
- avoid unbounded LLM parameter invention.

---

## 11. Programmatic API endpoints (pwm_core)

Typical endpoints:

- `fit_operator(spec, y, operator)` → `CalibResult`
- `calibrate_recon(spec, y, operator)` → `CalibReconResult`
- `reconstruct(spec, y, operator)` → `ReconResult`
- `analyze(spec, y, xhat, operator_fit)` → `DiagnosisResult`

Expected return objects:
- `DiagnosisResult` includes structured `suggested_actions` suitable for agents.

---

## 12. Suggested actions (closing the loop)

Operator Mode is most useful when it produces concrete next steps:
- increase photon budget
- change sampling rate / number of frames
- re-run calibration packet
- tighten alignment / re-estimate dispersion
- switch solver recipe (PnP vs TV)

PWM expresses these as structured actions in `report.json`:

```json
{
  "verdict": "model-mismatch",
  "confidence": 0.92,
  "suggested_actions": [
    {"knob": "states.calibration.alignment.dx", "op": "optimize", "val": null},
    {"knob": "states.budget.photon_budget.max_photons", "op": "multiply", "val": 2.0}
  ]
}
```

This enables AI_Scientist or other agents to iterate specs safely.

---

## 13. Minimal Example Spec (Operator Mode)

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
      - id: tv_fista
        params: {lam: 0.02, iters: 200}

export:
  runbundle:
    path: runs/
    name: latest
```

---

## 14. Contributor notes: adding a new operator-fit workflow

To add a new operator-fit workflow:
1) Implement an operator wrapper (matrix/callable/parametric) in `physics/adapters/`
2) Define θ parameterization in `mismatch/parameterizations.py`
3) Implement scoring components in `mismatch/scoring.py`
4) Create a CasePack in `contrib/casepacks/`
5) Add a unit test in `tests/`

Templates:
- `contrib/templates/new_operator_template.py`
- `contrib/templates/new_calibrator_template.py`

