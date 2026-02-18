# PWM Agent System — Plan v4.1 (Viral-First + Publishable + Monetizable)
**Priority order:** (1) Viral adoption → (2) Top papers → (3) Revenue  
**Core thesis:** *Universal within PhysicsScope* via **OperatorGraph**, with **26 modality templates** as examples + regression suite.

---

## 0) Executive Summary (Why this plan hits all 3 goals)

### Viral (first)
PWM goes viral when it ships **shareable, reproducible “wow” artifacts** with one command:
- **RunBundle + Viewer** (replayable) + **SharePack** (1 image + 30s video + summary)
- A **gallery** of impressive examples and a **community loop** (weekly challenges + PR-able templates)

### Publishable (second)
PWM is publishable because it formalizes and validates:
- a universal **OperatorGraph intermediate representation**
- a multi-factor **feasibility + bottleneck** model (Photon × Recoverability × Mismatch × Solver Fit)
- a robust **calibration / operator correction** loop with uncertainty and reproducibility

### Revenue (third)
PWM monetizes *after mindshare* through:
- calibration/design sprints, enterprise reproducibility, hosted runs, and proprietary primitive packs

---

## 1) Success Criteria (Product + Research)

PWM is “successful” if it can, for systems **within PhysicsScope**:
1. **Represent** the system as a composable, auditable forward model (OperatorGraph + metadata).
2. **Predict feasibility** (pre-flight): photon budget, noise regime, recoverability, mismatch sensitivity.
3. **Design**: propose system variants that satisfy constraints (resolution, SNR, cost, exposure, CR).
4. **Calibrate**: infer operator parameters θ from datasets and produce corrected operator + uncertainty.
5. **Reconstruct**: run solvers reproducibly and export RunBundles (full provenance + viewer replay).

> **Reality check:** photons + compression ratio are dominant but not sufficient.  
> Feasibility is a **multi-factor verdict**: Photon × Recoverability × Mismatch × Solver Fit.

---

## 2) Two Layers (Keep Both)

### Layer A — Universal Representation (truth)
- **OperatorGraphSpec + PhysicsScope primitives**
- Everything compiles to a GraphOperator (forward/adjoint/serialize/check)

### Layer B — Human-facing templates (examples)
- **26 modality templates** (CASSI, SPC, CACTI, CT, MRI, …)
- Templates are *thin wrappers* that reference IDs:
  - default ImagingSystemSpec (elements)
  - default OperatorGraphSpec skeleton (graph_template_id)
  - default priors & mismatch families
  - default solver families & metrics
  - guidance: “what to upload / what to simulate”

**Rule:** Templates never gate new systems.  
If it compiles to primitives, PWM runs—even if template match is “custom_graph” or multi-tag hybrid.

---

## 3) The Viral Unit (What People Share)

### 3.1 One-command demo that produces a SharePack
CLI goal:
```bash
pwm demo cassi --preset tissue --run --open-viewer --export-sharepack
```
Outputs:
- `runbundle/` (graph + blobs + hashes + metrics + provenance)
- `viewer/` (HTML/Streamlit artifact; “what-if” sliders)
- `sharepack/`:
  - `teaser.png` (before/after calibration)
  - `teaser.mp4` (30s auto-animated slider)
  - `summary.md` (3 bullets: problem → fix → result)

**Why this is viral:** it creates a “look what PWM did” artifact in <1 minute of human effort.

### 3.2 PWM Gallery (public)
A simple static site/docs page listing curated SharePacks:
- modality tag(s) + system diagram + 3-bullet story
- reproduce button + command
- metrics and failure modes

### 3.3 Weekly community loop (“Calibration Friday”)
Every week you publish:
- a small dataset slice + mismatch scenario + expected evaluation
Community submits:
- RunBundle zip + screenshot
You publish:
- leaderboard + a short analysis post

This drives:
- new contributors
- benchmarking pressure
- continuous content for social distribution

---

## 4) OperatorGraphSpec (Universal IR)

### 4.1 Graph schema (Pydantic)
- Nodes = primitive operators
- Edges = tensor flow (axis metadata)
- Parameters θ attached to nodes
- Noise/readout explicit nodes or distributions

(Keep the v4 schema; ensure strict `extra="forbid"` everywhere.)

### 4.2 Graph semantics
- **Adjoint correctness** for linear graphs
- **Serialization** of graph + blobs (mask/PSF/trajectory) + SHA256 hashes
- **Deterministic introspection**: explain a graph without LLM

---

## 5) PhysicsScope & Primitive Library (What “Any system in scope” means)

**PhysicsScope = the set of implemented primitive operators.**  
If a system needs a missing primitive → add it (primitive_id + forward/adjoint + tests).

Starter families (expand over time):
- propagation (ray/Fresnel/angular spectrum)
- PSF/convolution (2D/3D)
- modulation (mask/DMD)
- warp/dispersion
- sampling/readout (including CT/MRI operators)
- nonlinearities (magnitude-square, saturation, log)
- noise (Poisson, read, background, FPN)
- temporal (integration, motion warp)

**Scope discipline (critical):**
- Every primitive must ship with: unit tests, adjoint check (if linear), serialization contract.

---

## 6) GraphCompiler → GraphOperator (The Engine)

Compilation pipeline:
1. Validate graph (DAG, primitives exist, axis compatibility)
2. Bind primitives (instantiate with θ)
3. Build forward plan
4. Build adjoint plan (if linear)
5. Export RunBundle artifacts (blobs + hashes)

GraphOperator implements:
- `forward(x)`
- `adjoint(y)` (when valid)
- `serialize()`
- `check_adjoint()`

---

## 7) Agents (Deterministic Physics First; LLM Optional)

### 7.1 PlanAgent (routing + intent)
Inputs: goal (design/calibrate/reconstruct), constraints, assets  
Outputs (IDs only):
- modality template label (optional)
- graph_template_id OR custom graph skeleton
- priors/solvers/metrics IDs
- PreFlightReport request

### 7.2 System Discern Agent (recommended v4.1 addition)
Converts:
- user description + metadata + calibration snippets
into:
- ImagingSystemSpec (with confidence bands)
- candidate graphs (top-K) if ambiguous

### 7.3 PhotonAgent (feasibility)
Deterministic chain-throughput computation:
- photons/pixel, SNR, dominant noise term classification
- outputs: uncertainty and “increase photons” levers

### 7.4 MismatchAgent (what can go wrong)
- mismatch family IDs + severity + suggested calibration captures
- outputs: “minimum viable calibration recipe” for the system

### 7.5 RecoverabilityAgent (undersampling/compression feasibility)
Inputs:
- operator diversity proxies, CR, noise regime, prior class
Outputs:
- recoverability score + expected metric band
- backed by **empirical tables** with provenance (not fake CS formulas)

### 7.6 AnalysisAgent (bottleneck diagnosis)
Combines:
- Photon × Recoverability × Mismatch × Solver Fit
Outputs:
- primary bottleneck + ranked next actions
- “what to change first” with expected gain

---

## 8) Calibration & Operator Correction (Digital Twin Loop)

Calibration is θ-inference:
**min_θ  D( y, A_θ(x_cal) ) + R(θ)**

Two correction modes:
- **Gradient-based** (when primitives support autodiff)
- **UPWMI beam search** (derivative-free, registry-guided perturbations)

Outputs:
- corrected operator + uncertainty bands
- calibration report + “next capture suggestion” if underdetermined

---

## 9) Registries (YAML) v4.1

```text
packages/pwm_core/contrib/
├── primitives.yaml              # primitive operators + parameter schema
├── graph_templates.yaml         # reusable skeleton graphs
├── modalities.yaml              # 26 example templates (thin wrappers)
├── mismatch_db.yaml
├── photon_db.yaml
├── recoverability_db.yaml       # empirical tables + provenance + uncertainty
├── solver_registry.yaml
├── metrics_db.yaml
└── casepacks/
```

**Hard rule:** LLM outputs must be **registry IDs only** (mechanically enforced).

---

## 10) The 26 Modalities: Example Suite (Demos + Regression + Papers)

Each modality entry must include:
- `graph_template_id`
- default ImagingSystemSpec
- default photon model ID
- mismatch families and calibration recipe
- recoverability table group
- solver + metrics set
- a **CasePack** and a **SharePack**

**Why this helps all goals:**
- Viral: 26 ready demos + screenshots
- Papers: standardized benchmarks/ablations
- Revenue: “we support your family; here’s the playbook”

---

## 11) Paper Track (Second Priority, Designed In)

### Paper A (Systems): “OperatorGraph IR + Reproducible Digital Twin Loop”
Core claims:
- universal representation within PhysicsScope
- reproducible compilation + provenance
- feasibility verdict and bottleneck analysis
Experiments:
- 3 flagship modalities (CASSI, CACTI, MRI/CT or SPC)
- ablations: remove PhotonAgent / remove Recoverability / remove calibration loop

### Paper B (Benchmark): “Forward-model mismatch & calibration benchmark”
Deliver:
- controlled mismatch families, calibration data protocols
- standardized evaluation and baseline scripts

### Paper C (Method): “UPWMI operator correction + uncertainty”
Deliver:
- fast correction loop, caching, early stopping
- uncertainty bands; robust across mismatch families

**Note:** These papers are “born from the same codebase” and share RunBundle artifacts.

---

## 12) Revenue Track (Third Priority, Doesn’t Fight OSS)

### 12.1 Open-core boundary
Free:
- primitives + graph compiler + templates + demos + viewer

Paid:
- enterprise viewer (team sharing, access control, audit trails)
- hosted “PWM Runs” (GPU backend + job queue + RunBundle registry)
- premium primitive packs (vendor-specific / advanced physics)
- calibration/design sprints (2–4 weeks)

### 12.2 First revenue product (fastest)
**PWM Calibration Sprint**
Deliverables:
- calibrated operator + uncertainty
- RunBundle suite + viewer
- recommended capture protocol
- “design suggestions” (what to change to improve SNR/recoverability)

---

## 13) Implementation Phases (Aligned to 3 Goals)

### Phase 0 (Weeks 1–2): Viral MVP plumbing
- CLI demo path + SharePack exporter
- Viewer minimal replay
- 3 hero demos stubbed

### Phase 1 (Weeks 3–6): Core engine + 3 flagship modalities
- primitives + GraphCompiler + serialization + adjoint checks
- CASSI + CACTI + (MRI/CT or SPC) end-to-end
- PreFlightReport + bottleneck output

### Phase 2 (Weeks 7–10): Calibration loop + recoverability tables
- mismatch_db + correction optimizer
- recoverability_db tables for 3 modalities (with provenance)

### Phase 3 (Weeks 11–14): Community scaling + 26-suite expansion
- turn each modality into PR-able unit (template + CasePack + SharePack)
- publish Gallery + start Calibration Friday

### Phase 4 (Weeks 15+): Paper submissions + first paid engagements
- lock experimental protocol + run ablations
- package enterprise/hosted options behind a clean boundary

---

## 14) Testing Strategy (Prevents OSS regressions)

- registry integrity & ID cross-refs
- adjoint checks for every linear graph
- golden runs with fixed seeds (flagship)
- calibration regression (θ recovery)
- performance budgets; heavy runs behind permit gates

---

## 15) Final Checklist (If you do only 10 things)

1. One-command demo → RunBundle + Viewer + SharePack
2. Three hero modalities end-to-end (CASSI/CACTI/SPC or MRI/CT)
3. PreFlightReport with clear bottleneck recommendations
4. Calibration loop for at least one mismatch family per hero modality
5. Recoverability tables with provenance (not theory-only)
6. Gallery page with 10+ SharePacks
7. Weekly Calibration Friday challenge
8. Template PR workflow (how to add a modality)
9. Paper A experimental protocol + ablations
10. “Calibration Sprint” service offer (simple landing page)

---

## Summary

- **Layer A** gives universal representation within PhysicsScope (OperatorGraph).
- **Layer B** (26 templates) makes PWM easy to adopt, demo, test, and grow.
- v4.1 adds a **viral unit**, **community loop**, **paper track**, and **revenue boundary** so the plan directly supports all three goals.
