# PWM v2 Plan (OperatorGraph-First, Two Modes, Viral → Papers → Revenue)

**Priorities (explicit):**
1) **Make PWM go viral** (fast "wow", easy sharing, community flywheel)
2) **Publish top papers** (Paper 1 system/paradigm, Paper 2 benchmark/dataset, Paper 3 method)
3) **Earn revenue** (open-core compatible; does not slow OSS + papers)

**Non‑negotiable foundation:** **OperatorGraph-first rule** — *every* forward model is `OperatorGraphSpec`, and *all* reconstruction/calibration runs through `GraphOperator`.

---

## 0) The Two Modes (first-class)

| Mode | Input | What PWM does | Output |
|------|-------|---------------|--------|
| **Mode 1: Prompt-driven simulation + reconstruction** | Natural language / ExperimentSpec | Agents → build OperatorGraph → simulate `y` → reconstruct `x̂` → diagnose | RunBundle (+ SharePack) |
| **Mode 2: Operator correction (measured y + A)** | Measured `y` (+ optional operator A / metadata) | Agents still run → compile nominal OperatorGraph → fit/correct θ (mismatch/noise/system) → reconstruct with corrected operator → diagnose | RunBundle (+ calibrated θ + uncertainty) |

**Key insight:** Mode 2 **does not skip agents**. Even with measured `y`, PWM uses **Plan/Photon/Mismatch** agents to infer the imaging system + noise regime + calibration targets before fitting θ.

---

## 1) Success Criteria (measurable gates)

| # | Criterion | Gate |
|---|----------|------|
| 1 | **Viral: 60-second wow** | `pwm demo cassi` outputs a SharePack (teaser image + 5–10s video + summary.md) in <60s |
| 2 | **Viral: gallery** | Public gallery shows 26+ modalities with "reproduce" buttons (one command) |
| 3 | **Viral: weekly challenge** | "Calibration Friday" challenge + leaderboard running continuously |
| 4 | **Mode 1 works end-to-end** | `pwm run --prompt "CASSI 28 bands"` → OperatorGraph → simulate → reconstruct → RunBundle |
| 5 | **Mode 2 works end-to-end** | `pwm calib-recon --y measurement.npy --operator cassi` → agents → OperatorGraph → fit θ → reconstruct → RunBundle |
| 6 | **Paper 1 (Flagship)** | Two modes + OperatorGraph IR spec + agents + depth (SPC/CACTI/CASSI) + breadth (CT/Widefield) + "26 compile" evidence |
| 7 | **Paper 2 (InverseNet)** | Dataset + tasks + baselines + leaderboard: **SPC(Set11)**, **CACTI(6 gray)**, **CASSI(10 scenes)** |
| 8 | **Paper 3 (PWMI-CASSI)** | Algorithm 1 + 2 on **10 CASSI scenes**, mismatch families/severities, bootstrap CI, capture advisor |
| 9 | **OperatorGraph-first enforced** | All solvers/calibration call `GraphOperator` only; no modality-specific forward code in the hot path |
| 10 | **Reproducibility** | Every table/figure backed by RunBundle with SHA256 hashes + seeds + git hash |

---

## 2) OperatorGraph-First Rule (foundation of PWM)

### 2.1 The rule (applies to BOTH modes)

> **All forward models are represented as `OperatorGraphSpec`.**
> **All reconstruction and calibration pipelines consume a `GraphOperator`.**
> Layer B templates/CasePacks only supply **defaults** (graph_template_id, priors, presets, datasets), never bespoke physics.

### 2.2 "No bypass" enforcement

- `pwm run`, `pwm demo`, `pwm calib-recon`, all benchmark scripts must go through:
  1) `GraphCompiler.compile(OperatorGraphSpec) → GraphOperator`
  2) `GraphOperator.forward()` / `GraphOperator.adjoint()`
  3) `SolverRunner(GraphOperator, y, config)`
- Any legacy per-modality forward operators are allowed **only** as reference implementations for equivalence tests during migration.

---

## 3) OperatorGraph IR Spec (formal semantics reviewers can trust)

This section makes OperatorGraph a **research contribution**, not just an engineering choice.

### 3.1 NodeSpec: tags + contracts

Each node MUST declare:

- `tag_linear: bool`
- `tag_nonlinear: bool` (mutually exclusive with linear except for piecewise-linear approximations)
- `tag_stochastic: bool` (noise / random processes)
- `tag_differentiable: bool` (supports autograd/JVP/VJP)
- `tag_stateful: bool` (depends on internal state, e.g., hysteresis)

Each node provides:

- `forward(inputs, params, rng=None) → outputs`
- `adjoint(dy, cache) → dx` **only if** `tag_linear=True`
- `serialize()` and deterministic hash inputs (for RunBundle reproducibility)

### 3.2 EdgeSpec: TensorSpec (type-checked graphs)

Every edge carries a `TensorSpec`:

- `shape`: static or symbolic (e.g., `[H,W,C]`, `[N_views,H,W]`)
- `dtype`: float32/float16/etc.
- `unit`: counts, intensity, phase, k-space, projection, wavelength-index, etc.
- `domain`: semantic domain label (e.g., `image_space`, `sensor_space`, `k_space`, `spectral_cube`)

GraphCompiler must validate TensorSpec consistency (shape/dtype/unit/domain).

### 3.3 ParameterSpec (θ for system+mismatch, ϕ for noise)

Every parameter MUST define:

- `bounds`: [min,max] or discrete set
- `prior_family`: e.g., Normal, Uniform, Laplace, LogNormal, SpikeSlab
- `parameterization`: direct / log / softplus / angle-wrap
- `identifiability_hint`: {strong, moderate, weak} + reason (e.g., "weak at low photons")

### 3.4 Adjoint policy

- `check_adjoint()` is required **only** for graphs where **all nodes are linear**.
- If any node is nonlinear/stochastic/stateful, the graph is marked `adjoint_undefined` and uses:
  - forward-only solvers, or
  - local linearization/JVP/VJP where differentiable, or
  - surrogate calibration objectives.

### 3.5 Noise policy (likelihood-aware)

Noise nodes are `tag_stochastic=True`. Scoring/calibration MUST prefer likelihood-aware objectives:

- Poisson(-Gaussian) → Negative Log Likelihood (NLL) or variance-stabilized loss
- Read noise / background → heteroscedastic likelihood or robust loss
- Residual L2 remains a baseline, but **paper experiments must report NLL-based scoring** when noise is present.

---

## 4) Layer A vs Layer B (B is built on A)

### 4.1 Layer A (truth / foundation)
- OperatorGraph IR + primitives
- GraphCompiler → GraphOperator
- SolverRunner (all reconstruction)
- Calibration loops (Mode 2)
- RunBundle/SharePack exporters
- CI enforcement and reproducibility rules

### 4.2 Layer B (applications / onboarding / evidence)
- 26 modality templates are **specific applications of Layer A**:
  - `graph_template_id`
  - default ImagingSystemSpec (elements)
  - default priors + mismatch families
  - solver portfolio + metric sets
  - "what to upload / what to simulate" guidance
- Layer B is used to:
  - onboard new users fast
  - provide a public "26 modalities" gallery
  - provide breadth evidence in Paper 1
- Layer B must never bypass Layer A physics.

---

## 5) Mode 1 Pipeline (Prompt-driven simulation + reconstruction)

**Entry point:**
```bash
pwm run --prompt "CASSI 28 bands, low-light, mild mask shift" --out runs/
```

**Pipeline:**
1) **PlanAgent** parses intent and chooses `graph_template_id`
2) **PhotonAgent** estimates photon budget + noise params (ϕ), sets noise nodes
3) **MismatchAgent** selects mismatch family + targets θ (calibration knobs)
4) **Recoverability/CompressedAgent** maps sampling/compression to feasibility priors
5) **GraphCompiler** builds two operators:
   - `op_ideal` (nominal)
   - `op_real` (mismatched + noise)
6) **Simulation:** `y = op_real.forward(x_gt)` (+ noise)
7) **Reconstruction:** `x̂ = SolverRunner(op_nominal or op_corrected, y)`
8) **Diagnosis:** compare predicted feasibility vs realized metrics
9) **Export:** RunBundle + SharePack

---

## 6) Mode 2 Pipeline (Operator correction with agents)

**Entry point:**
```bash
pwm calib-recon --y measurement.npy --operator cassi --out runs/
```

### 6.1 Inputs
- Required: measured `y`
- Optional: explicit `A` (matrix/operator), calibration metadata (mask, PSF, geometry), `y_cal` captures

### 6.2 Pipeline steps (OperatorGraph-first)
1) **PlanAgent**
   - selects `graph_template_id` (or compiles from provided operator identity)
2) **PhotonAgent**
   - estimates noise regime from `y` (variance structure, saturation, background), sets ϕ
3) **MismatchAgent**
   - proposes calibration targets θ (which parameters to fit) + mismatch family prior
4) Compile **nominal operator**: `op_nominal = compile(template, θ_nominal, ϕ)`
5) **Calibration loop (Algorithm 1 or 2):**
   - For candidate θ':
     - `op_θ' = compile(template, θ', ϕ)`
     - `x̂ = SolverRunner(op_θ', y, config)`
     - **score(θ') = NLL(y | op_θ'(x̂), ϕ)** (baseline L2 allowed)
   - Choose θ*; produce `op_corrected = op_θ*`
6) **Reconstruction + diagnosis** using `op_corrected`
7) **Uncertainty** via bootstrap / CI (task-dependent)
8) **Export:** RunBundle + `OperatorSpec_calib.json` + `identifiability_report.json`

### 6.3 Stop criteria + failure modes
- stop when:
  - score improvement < ε for K steps, OR
  - confidence threshold reached, OR
  - budget exhausted (time/evals)
- failure reasons recorded (must be explicit):
  - unidentifiable params under photon/compression regime
  - degeneracy/multiple equivalent solutions
  - insufficient calibration captures
  - model out-of-scope (missing primitive)

---

## 7) Identifiability Guardrail (mandatory for calibration papers)

Before searching parameter θᵢ:
1) run a cheap sensitivity probe on the compiled graph:
   - perturb θᵢ by ±δ; measure effect on score
2) if influence < ε under current photon/sampling:
   - freeze θᵢ or mark "weakly identifiable"
3) export `identifiability_report.json` into RunBundle:
   - sensitivity, frozen params, recommended extra captures (if needed)

This prevents wasted search and provides paper-grade explanations.

---

## 8) Adding mismatch (OperatorGraph-first)

Mismatch is applied by modifying **GraphNode parameters** in `OperatorGraphSpec` before compilation.

Example:
```python
spec = load_graph_template("cassi_graph_v1")
spec.nodes["modulate"].params["mask_dx"] = nominal_dx + delta_dx
spec.nodes["modulate"].params["mask_dy"] = nominal_dy + delta_dy
spec.nodes["disperse"].params["disp_step"] = nominal_disp + delta_disp
op_mismatched = GraphCompiler().compile(spec)
y = op_mismatched.forward(x_gt)  # + noise via noise nodes
```

**Principle:** mismatch families are defined in YAML as parameter perturbations + priors; calibration is defined as searching those same parameters to minimize the calibration objective.

---

## 9) Solver API Unification (so *all* reconstruction is OperatorGraph-based)

All solvers must consume the same operator interface:

```python
class LinearLikeOperator(Protocol):
    def forward(self, x): ...
    def adjoint(self, y): ...
    def shape(self): ...  # (x_shape, y_shape)
```

Rules:
- classical solvers call `forward/adjoint` only
- PnP solvers call `forward/adjoint` + denoiser
- learning-based solvers receive `op` when needed
- if `adjoint_undefined`, solver must declare compatibility (forward-only, differentiable, etc.)

---

## 10) Virality Flywheel (Phase 0 must remove friction)

### 10.1 "first run always works"
- `pip install pwm-core`
- `pwm doctor` prints a green/red dependency checklist
- minimal CPU demo path that never downloads large datasets

### 10.2 SharePack exporter (viral artifact)
One command exports:
- `teaser.png`, `teaser.mp4` (5–10s)
- `summary.md` (what modality, what mismatch, what improvement)
- `runbundle.zip` with hashes + seeds

### 10.3 Gallery website
- static site built from SharePacks
- each tile has:
  - preview
  - reproduce command
  - RunBundle link
  - metrics snapshot

### 10.4 Community loop
- Weekly challenge ("Calibration Friday")
- automated submissions
- leaderboard + badges
- contributor pipeline (good first issues)

---

## 11) Papers (3-paper pipeline with explicit reuse)

### Paper 1 (Flagship / paradigm): "Physics World Model for Imaging"
**Depth:** SPC + CACTI + CASSI (full Mode 1 + Mode 2)
- feasibility prediction: photon × compression × mismatch (agents) vs realized outcome
- ablations: remove each agent and show degradation
- operator correction: wrong θ → calibrated θ (Algorithm 1/2 where relevant)

**Breadth:** CT + Widefield (and optional Holography)
Minimal checklist:
- compile to OperatorGraph
- serialize + reproduce
- adjoint check (if linear)
- one mismatch + one correction improvement

**Universality evidence:** "All 26 templates compile via Layer A" + reproducibility.

### Paper 2 (InverseNet): benchmark + dataset + leaderboard
**Modalities + datasets (fixed):**
- SPC: **Set11**
- CACTI: **6 grayscale benchmark videos**
- CASSI: **10 scenes**
Tasks:
- T1 parameter estimation
- T2 mismatch identification
- T3 calibration
- T4 reconstruction under mismatch
All baselines run via OperatorGraph-first pipeline.

### Paper 3 (PWMI-CASSI): method paper
- CASSI calibration on **10 scenes**
- **Algorithm 1 (UPWMI beam search + agents)**
- **Algorithm 2 (GPU differentiable GAP-TV + grid/refinement)**
- mismatch families × severities
- bootstrap CI + capture advisor (what extra data reduces uncertainty)

**Reuse rule (reduces total workload):**
- Paper 3 produces the calibration figures/stats
- Paper 1 imports those results as the "Mode 2 depth" chapter for CASSI
- Paper 2 reuses the same mismatch families + scoring + operator correction infrastructure

---

## 12) Dataset Hygiene (prevent submission blockers)

For every dataset (Paper 2/3):
- provenance + license/terms (as known)
- retrieval scripts (download/symlink)
- **CI slices** committed (tiny samples) so tests never depend on downloads
- `DATASET_CARD.md` for:
  - InverseNet package
  - PWMI-CASSI evaluation set

---

## 13) Compute Budget + Caching (prevent silent explosions)

Add a small "cost model" table (fill with measured numbers):
- runtime per scene per candidate θ eval
- K bootstrap replicates and total time
- GPU vs CPU deltas

Caching rules:
- cache compiled graphs by (template_id, θ, ϕ, assets_hash)
- warm-start recon across θ candidates (reuse x̂)
- store intermediate scores for beam/grid search

Parallelization:
- by modality
- by scene
- by mismatch family/severity

---

## 14) CI Enforcement (OperatorGraph-first cannot regress)

CI must fail if:
- any solver/calibration path calls modality-specific forward code outside GraphCompiler
- a template cannot compile
- serialize → reload → rerun changes output beyond tolerance
- adjoint check fails for linear-tagged graphs

Core CI tests (fast):
- compile test for all templates
- serialize roundtrip
- quick recon on CI slice
- operator correction sanity (1 tiny case)

Nightly tests:
- full 26-suite
- operator correction suite
- leaderboard baseline regeneration

---

## 15) Revenue (open-core boundary; third priority)

Keep the core open:
- OperatorGraph IR, compiler, templates, benchmarks, SharePack, RunBundle

Optional paid offerings (later, non-blocking):
- "Calibration sprint" service (you bring measured data, PWM produces correction + report)
- hosted gallery + hosted RunBundle viewer + private leaderboards
- hardware adapter packs (vendor-specific metadata, integration)

---

## 16) Milestones (aggressive, parallel)

**Phase 0 (viral):** SharePack + `pwm demo` + `pwm doctor` + gallery skeleton + weekly challenge
**Phase 1 (foundation):** IR spec + primitives registry + graph_templates for 26 + CI enforcement
**Phase 2 (Mode 2 hardening):** likelihood-aware scoring + identifiability guardrail + y_cal support
**Phase 3 (papers):** run depth + breadth experiments, then write Paper 1/2/3 with reuse rule
**Phase 4 (community + revenue option):** leaderboard + open-core boundary + calibration sprint beta

---

## Appendix A: Concrete pointers (implementation alignment)

- Mode 2 calibration for CASSI uses Algorithm 1 + Algorithm 2 logic already present in your benchmark code paths.
- InverseNet mismatch injection MUST be "parameter perturbation in OperatorGraphSpec → compile → forward" (never bespoke forward).
