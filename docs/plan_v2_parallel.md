# PWM v2 — Parallel Execution Plan

**Condition:** OperatorGraph-First Rule is the single prerequisite gate.
Once Gate 0 (foundation) is solid, **5 independent tracks** launch in parallel.
Paper experiments form a second wave; paper writing forms a third.

**Every track, task, and exit gate is tagged with the Success Criteria (SC-1..SC-10) it must satisfy. No criterion is left unassigned.**

---

## Success Criteria (10 measurable gates)

| SC | Criterion | Gate | Satisfied by |
|----|-----------|------|--------------|
| **SC-1** | **Viral: 60-second wow** | `pwm demo cassi` outputs SharePack (teaser image + 5-10s video + summary.md) in <60s | Track A (A.2, A.3) |
| **SC-2** | **Viral: gallery** | Public gallery shows 26+ modalities with "reproduce" buttons (one command) | Track A (A.4) |
| **SC-3** | **Viral: weekly challenge** | "Calibration Friday" challenge + leaderboard running continuously | Track A (A.5) |
| **SC-4** | **Mode 1 works end-to-end** | `pwm run --prompt "CASSI 28 bands"` → OperatorGraph → simulate → reconstruct → RunBundle | Track B (B.5) |
| **SC-5** | **Mode 2 works end-to-end** | `pwm calib-recon --y measurement.npy --operator cassi` → agents → OperatorGraph → fit θ → reconstruct → RunBundle | Track C (C.8) |
| **SC-6** | **Paper 1 (Flagship)** | Two modes + OperatorGraph IR spec + agents + depth (SPC/CACTI/CASSI) + breadth (CT/Widefield) + "26 compile" evidence | Gate 0 + B + C + H + I |
| **SC-7** | **Paper 2 (InverseNet)** | Dataset + tasks + baselines + leaderboard: SPC(Set11), CACTI(6 gray), CASSI(10 scenes) | C + D + G + I |
| **SC-8** | **Paper 3 (PWMI-CASSI)** | Algorithm 1 + 2 on 10 CASSI scenes, mismatch families/severities, bootstrap CI, capture advisor | C + D + F + I |
| **SC-9** | **OperatorGraph-first enforced** | All solvers/calibration call `GraphOperator` only; no modality-specific forward code in the hot path | Gate 0 (G0.2-G0.4) + E |
| **SC-10** | **Reproducibility** | Every table/figure backed by RunBundle with SHA256 hashes + seeds + git hash | Gate 0 (G0.1) + B (B.6) + C (C.8) + E (E.2) + F/G/H |

---

## Dependency Diagram

```
                        ┌─────────────────────┐
                        │   GATE 0: Foundation │
                        │  (OperatorGraph-First│
                        │   IR + Solver API +  │
                        │   CI skeleton + 26   │
                        │   template validation)│
                        └─────────┬────────────┘
                                  │
              ┌───────────┬───────┼───────┬────────────┐
              ▼           ▼       ▼       ▼            ▼
        ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
        │ Track A  │ │Track B │ │Track C │ │Track D │ │Track E │
        │  Viral   │ │ Mode 1 │ │ Mode 2 │ │Dataset │ │CI+Infra│
        │  MVP     │ │Pipeline│ │Pipeline│ │  Prep  │ │  Ops   │
        └────┬─────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
             │           │         │           │           │
             │           │    ┌────┴────┐      │           │
             │           │    │         │      │           │
             │      ┌────┴────┴──┐  ┌───┴──────┴──┐       │
             │      │  Track F   │  │  Track G    │       │
             │      │  Paper 3   │  │  Paper 2    │       │
             │      │ PWMI-CASSI │  │  InverseNet │       │
             │      │ Experiments│  │  Experiments│       │
             │      └─────┬──────┘  └──────┬──────┘       │
             │            │                │              │
             │      ┌─────┴────────────────┴──┐           │
             │      │       Track H           │           │
             │      │       Paper 1           │           │
             │      │  Flagship Experiments   │           │
             │      │  (imports F for CASSI)  │           │
             │      └──────────┬──────────────┘           │
             │                 │                          │
        ┌────┴─────────────────┴──────────────────────────┘
        │
        ▼
  ┌──────────────────────────┐      ┌──────────┐
  │  Track I: Paper Writing  │      │ Track J  │
  │  (3 manuscripts parallel)│      │ Revenue  │
  └──────────────────────────┘      │(anytime) │
                                    └──────────┘
```

---

## Gate 0: Foundation (sequential prerequisite)

**Everything depends on this. Must complete before any track launches.**
**Criteria addressed: SC-6 (OperatorGraph IR spec), SC-9 (enforcement), SC-10 (reproducibility via hashes)**

The OperatorGraph-First Rule requires a solid Layer A before any Mode, experiment, or viral feature is built on top.

| Task | What | Deliverable | Criteria |
|------|------|-------------|----------|
| G0.1 | **IR formal types** — implement `NodeSpec` (tags: linear, nonlinear, stochastic, differentiable, stateful), `TensorSpec` (shape, dtype, unit, domain), `ParameterSpec` (bounds, prior, parameterization, identifiability_hint) as Pydantic models | `graph/ir_types.py` | SC-6, SC-9, SC-10 |
| G0.2 | **Solver API unification** — enforce `LinearLikeOperator` protocol; all registered solvers consume `GraphOperator` | Updated solver registry + protocol | SC-9 |
| G0.3 | **26-template validation** — all templates compile with new IR types, NodeTags present on every node, TensorSpec edges validated, adjoint check passes for linear graphs | CI test: `test_26_compile.py` | SC-6, SC-9 |
| G0.4 | **CI enforcement skeleton** — grep-based lint (no modality-specific forward outside GraphOperator), serialize roundtrip gate, adjoint gate | CI config + gate scripts | SC-9, SC-10 |

**Exit gate:** `pytest tests/test_26_compile.py` passes; all 26 templates have NodeTags + TensorSpec; solver protocol enforced.
- SC-9 verified: no modality-specific forward code outside GraphOperator
- SC-10 verified: serialize roundtrip produces deterministic hashes

**Estimated effort:** ~3 days

---

## Wave 1: Five Parallel Tracks (launch simultaneously after Gate 0)

### Track A: Viral MVP

**Depends on:** Gate 0 only
**No dependency on:** Mode 1, Mode 2, datasets, paper experiments
**Criteria addressed: SC-1, SC-2, SC-3**

| Task | What | Deliverable | Criteria |
|------|------|-------------|----------|
| A.1 | `pwm doctor` — green/red environment checklist | `cli/doctor.py` | SC-1 (friction-free first run) |
| A.2 | SharePack exporter — teaser.png, teaser.mp4, summary.md, reproduce.sh | `export/sharepack.py` | **SC-1** (SharePack output) |
| A.3 | `pwm demo` command + 3 hero presets (CASSI, CACTI, SPC) | `cli/demo.py` + CasePack updates | **SC-1** (<60s demo) |
| A.4 | Gallery static site — 26 modality cards with images + reproduce buttons | `docs/gallery/` | **SC-2** (26+ modality gallery) |
| A.5 | Calibration Friday — publish 4 challenges, submission + scoring pipeline | `community/challenges/` | **SC-3** (weekly challenge + leaderboard) |
| A.6 | QuickStart doc + Colab notebook | `docs/quickstart/` | SC-1 (zero-friction onboarding) |

**Exit gate:**
- **SC-1:** `pwm demo cassi --export-sharepack` produces teaser.png + summary.md in <60s
- **SC-2:** Gallery page renders 26 modality cards with images and one-command reproduce buttons
- **SC-3:** `pwm calib-recon --challenge 2026-W11` works end-to-end; leaderboard auto-updates

**Estimated effort:** ~5 days

---

### Track B: Mode 1 Pipeline (Prompt-driven)

**Depends on:** Gate 0 only
**No dependency on:** Mode 2, datasets, viral layer
**Criteria addressed: SC-4, SC-6, SC-9, SC-10**

| Task | What | Deliverable | Criteria |
|------|------|-------------|----------|
| B.1 | Agent chain integration — PlanAgent → PhotonAgent → MismatchAgent → RecoverabilityAgent wired end-to-end | Pipeline orchestrator | SC-4 (agents in Mode 1), SC-6 (agents for Paper 1) |
| B.2 | Simulation path — agents select template → GraphCompiler → `op_real.forward(x_gt)` → noisy y | Simulation module | SC-4, **SC-9** (OperatorGraph-only path) |
| B.3 | Reconstruction path — SolverRunner(GraphOperator, y, config) → x̂ | Solver integration | SC-4, **SC-9** (solver via GraphOperator) |
| B.4 | Diagnosis — compare predicted feasibility vs realized metrics | AnalysisAgent output | SC-4, SC-6 (pre-flight feasibility) |
| B.5 | `pwm run` CLI — `pwm run --prompt "CASSI 28 bands"` end-to-end | CLI command | **SC-4** (Mode 1 end-to-end) |
| B.6 | RunBundle export for Mode 1 — SHA256 hashes + seeds + git hash | Export integration | **SC-10** (reproducibility) |

**Exit gate:**
- **SC-4:** `pwm run --prompt "CASSI 28 bands"` → OperatorGraph → simulate → reconstruct → RunBundle
- **SC-9:** Entire path goes through GraphCompiler → GraphOperator → SolverRunner (no bypass)
- **SC-10:** RunBundle contains SHA256 hashes + seeds + git hash

**Estimated effort:** ~5 days

---

### Track C: Mode 2 Pipeline + Calibration

**Depends on:** Gate 0 only
**No dependency on:** Mode 1, viral layer, datasets
**Criteria addressed: SC-5, SC-6, SC-7, SC-8, SC-9, SC-10**

| Task | What | Deliverable | Criteria |
|------|------|-------------|----------|
| C.1 | **Likelihood-aware scoring** — Poisson NLL, Gaussian NLL, mixed Poisson-Gaussian, L2 fallback; selected by PhotonAgent noise estimate | `graph/scoring.py` | SC-5, SC-8 (Paper 3 scoring) |
| C.2 | **Identifiability guardrail** — sensitivity probe per θᵢ on compiled graph → freeze unidentifiable params → `identifiability_report.json` | `graph/identifiability.py` | SC-5, SC-6, SC-8 |
| C.3 | **Mismatch via GraphNode params** — modify OperatorGraphSpec nodes before compilation; YAML-defined perturbation families | Mismatch module | **SC-9** (mismatch via OperatorGraph), SC-7, SC-8 |
| C.4 | **Calibration loop** — integrate Algorithm 1 (beam search) + Algorithm 2 (GPU differentiable) through GraphOperator compile-per-candidate pattern | Calibration runner | **SC-5** (fit θ), **SC-8** (Alg 1+2), **SC-9** |
| C.5 | **Stop criteria + failure modes** — convergence detection, budget exhaustion, flat landscape, divergence; structured `calibration_status.json` | Stop logic + failure codes | SC-5, SC-10 (documented failures) |
| C.6 | **Bootstrap CI + capture advisor** — K=20 resamples, 95% CI, capture recommendations when CI is wide | Uncertainty module | **SC-8** (bootstrap CI + capture advisor) |
| C.7 | **Agent involvement in Mode 2** — PlanAgent, PhotonAgent, MismatchAgent, RecoverabilityAgent, AnalysisAgent, CaptureAdvisor all participate | Agent wiring | **SC-5** (agents in Mode 2), **SC-6** (agents for Paper 1) |
| C.8 | `pwm calib-recon` CLI end-to-end — RunBundle + calibrated θ + identifiability_report.json | CLI command | **SC-5**, **SC-10** |

**Exit gate:**
- **SC-5:** `pwm calib-recon --y data.npy --operator cassi` → agents → OperatorGraph → fit θ → reconstruct → RunBundle + calibrated θ + identifiability_report.json
- **SC-8:** Algorithm 1 + Algorithm 2 both operational through GraphOperator; bootstrap CI produces 95% intervals; capture advisor recommends additional captures when CI is wide
- **SC-9:** Calibration loop compiles GraphOperator per candidate θ (no bypass)
- **SC-10:** RunBundle includes SHA256 hashes + seeds + git hash + calibration_status.json

**Estimated effort:** ~7 days

---

### Track D: Dataset Infrastructure

**Depends on:** Gate 0 only (needs GraphOperator to generate samples)
**No dependency on:** Mode 1, Mode 2, viral layer
**Criteria addressed: SC-7, SC-8, SC-9, SC-10**

| Task | What | Deliverable | Criteria |
|------|------|-------------|----------|
| D.1 | **Provenance + licensing** — document sources, licenses, citations for Set11, CACTI 6-gray, TSA_simu_data | Provenance docs | SC-7 (Paper 2 dataset), SC-10 |
| D.2 | **Retrieval scripts** — `download_set11.sh`, `download_cacti6.sh`, `download_tsa.sh` with SHA256 checksum verification | `scripts/` | SC-7, SC-10 (checksum verification) |
| D.3 | **CI slice fixtures** — tiny crops committed to repo (cameraman 32x32, kobe 32x32x4, scene01 32x32x8) | `tests/fixtures/` | SC-9 (tests use GraphOperator) |
| D.4 | **Dataset cards** — `DATASET_CARD_inversenet.md`, `DATASET_CARD_pwmi_cassi.md` | `datasets/` | SC-7, SC-8, SC-10 |
| D.5 | **InverseNet generation infrastructure** — gen scripts for SPC (Set11 x sweeps), CACTI (6 gray x sweeps), CASSI (10 scenes x sweeps); all use OperatorGraph compile → forward | `experiments/inversenet/gen_*.py` | **SC-7** (1,728 samples), **SC-9** (generation via GraphOperator) |
| D.6 | **Manifest validation** — schema + integrity checks for generated samples | Validation script | **SC-10** (SHA256 stable across re-generation) |

**Exit gate:**
- **SC-7:** Gen scripts produce valid InverseNet samples (SPC/CACTI/CASSI) on smoke run; dataset cards complete
- **SC-8:** CASSI generation covers 10 scenes x 3 families x 3 severities
- **SC-9:** All sample generation uses OperatorGraph compile → forward (no bespoke physics)
- **SC-10:** SHA256 hashes stable across re-generation with fixed seeds

**Estimated effort:** ~4 days

---

### Track E: CI + Compute Infrastructure

**Depends on:** Gate 0 only
**No dependency on:** any other track
**Criteria addressed: SC-9, SC-10**

| Task | What | Deliverable | Criteria |
|------|------|-------------|----------|
| E.1 | **No-bypass lint gate** — grep CI check: no `Operator.forward(` in benchmarks/recon outside `graph_operator.py` | CI gate script | **SC-9** (continuous enforcement) |
| E.2 | **Serialize roundtrip gate** — compile → serialize → reload → re-compile → forward produces identical output for all 26 templates | Roundtrip test | **SC-9**, **SC-10** (deterministic reproduction) |
| E.3 | **Adjoint regression gate** — `check_adjoint()` passes for all linear-tagged graphs | Adjoint test | **SC-9** (GraphOperator contract) |
| E.4 | **Compute budget benchmarks** — measure per-sample runtimes (compile, forward, recon, calibration) for cost model table | Benchmark script + results | SC-6 (Paper 1 evidence) |
| E.5 | **Caching layer** — compiled graph cache by (template_id, θ_hash), recon warm-start, intermediate score storage | Cache implementation | SC-10 (deterministic cache keys) |
| E.6 | **Template stability test** — NodeTags present on every node, TensorSpec edges valid, schema validation on `graph_templates.yaml` | Schema validation test | **SC-9** (all 26 templates valid) |

**Exit gate:**
- **SC-9:** No-bypass lint passes; all 26 templates compile + adjoint check + serialize roundtrip; no regressions possible
- **SC-10:** Serialize → reload → rerun produces bit-identical output; cache keys include SHA256 hashes

**Estimated effort:** ~3 days

---

## Wave 2: Paper Experiments (launch as tracks complete)

### Track F: Paper 3 Experiments — PWMI-CASSI

**Depends on:** Track C (Mode 2 pipeline) + Track D (CASSI dataset)
**Can start as soon as C + D finish; does NOT wait for A, B, or E.**
**Criteria addressed: SC-8, SC-9, SC-10**

| Task | What | Data | Criteria |
|------|------|------|----------|
| F.1 | Family sweep (F1) — 3 families x 3 severities x 10 scenes x 5 trials | TSA_simu_data (10 scenes) | **SC-8** (mismatch families/severities) |
| F.2 | 5-baseline comparison (F2) — no-calib, grid, gradient, Alg1, Alg2 | 10 scenes | **SC-8** (Algorithm 1 + 2 on 10 scenes) |
| F.3 | Statistical analysis (F3) — paired t-tests, Cohen's d, CI coverage | Results from F.1/F.2 | **SC-8** (bootstrap CI) |
| F.4 | Calibration budget sweep (F4) — 1/3/5/10 captures x 10 scenes | 10 scenes | **SC-8** (capture advisor) |
| F.5 | Photon regime sweep (F5) — low/medium/high x 3 families x 10 scenes | 10 scenes | SC-8 |
| F.6 | Agent ablation (F6) — remove agents → measure degradation | 10 scenes | SC-6, SC-8 |

**Internal parallelism:** F.1, F.2, F.4, F.5, F.6 are independent (different experimental axes). Run by family or by scene.

**Exit gate:**
- **SC-8:** Algorithm 1 + 2 on 10 CASSI scenes with 3 mismatch families x 3 severities; bootstrap CI coverage >= 90%; capture advisor results for budget sweep; statistical significance (p < 0.05 paired t-test UPWMI vs baselines)
- **SC-9:** All experiments compile GraphOperator per candidate θ (no bypass)
- **SC-10:** Every run produces RunBundle with SHA256 hashes + seeds + git hash

**Estimated effort:** ~40 GPU-hours (F.1) + ~50 GPU-hours (F.2) + CPU analysis

---

### Track G: Paper 2 Experiments — InverseNet

**Depends on:** Track C (Mode 2 pipeline) + Track D (dataset generation infrastructure)
**Parallel with Track F** (different modality focus: SPC + CACTI + CASSI vs CASSI-only).
**Criteria addressed: SC-7, SC-9, SC-10**

| Task | What | Samples | Criteria |
|------|------|---------|----------|
| G.1 | Full dataset generation — SPC (594) + CACTI (324) + CASSI (810) = 1,728 samples | All 3 modalities | **SC-7** (SPC Set11 + CACTI 6 gray + CASSI 10 scenes) |
| G.2 | T1: Parameter estimation baselines | 1,728 | **SC-7** (tasks) |
| G.3 | T2: Mismatch identification baselines | 1,728 | **SC-7** (tasks) |
| G.4 | T3: Calibration baselines (grid/gradient/UPWMI + likelihood-aware scoring) | 1,728 | **SC-7** (baselines), SC-9 |
| G.5 | T4: Reconstruction under mismatch (oracle vs wrong vs calibrated) | 1,728 | **SC-7** (baselines), SC-9 |
| G.6 | Leaderboard tables + dataset packaging | All results | **SC-7** (leaderboard) |

**Internal parallelism:** G.1 parallelizes by modality (3 jobs). G.2-G.5 parallelize by task x modality (12 jobs).

**Exit gate:**
- **SC-7:** 1,728 samples validated across SPC(Set11), CACTI(6 gray), CASSI(10 scenes); baseline tables for all 4 tasks (T1-T4); leaderboard JSON generated
- **SC-9:** All baselines run through OperatorGraph-first pipeline (no modality-specific forward)
- **SC-10:** Every sample + baseline run backed by RunBundle with SHA256 hashes + seeds

**Estimated effort:** ~4 hours (generation) + ~100 hours (baselines, parallelizable)

---

### Track H: Paper 1 Experiments — Flagship

**Depends on:** Track B (Mode 1) + Track C (Mode 2) + Track D (datasets) + **Track F results** (CASSI Mode 2 imported, not re-run)
**Criteria addressed: SC-6, SC-9, SC-10**

| Task | What | Modalities | Criteria |
|------|------|-----------|----------|
| H.1 | Mode 1 sweeps — photon x CR x mismatch x solver portfolio | SPC, CACTI, CASSI | **SC-6** (depth: 3 modalities Mode 1) |
| H.2 | Mode 2 correction — SPC (Set11, 11 images x 3 severities) | SPC | **SC-6** (depth: SPC Mode 2) |
| H.3 | Mode 2 correction — CACTI (6 videos x 2 families x 3 severities) | CACTI | **SC-6** (depth: CACTI Mode 2) |
| H.4 | **Import Track F results** — CASSI Mode 2 (no re-run) | CASSI | **SC-6** (depth: CASSI Mode 2) |
| H.5 | Pre-flight validation — predicted vs actual feasibility | SPC, CACTI, CASSI | **SC-6** (agents predict outcome) |
| H.6 | Agent + identifiability ablations — remove each agent → measure degradation | CASSI (representative) | **SC-6** (ablation evidence) |
| H.7 | Breadth modalities — CT + Widefield (compile + adjoint + mismatch + correction) | CT, Widefield | **SC-6** (breadth evidence) |
| H.8 | 26-suite compilation evidence — all templates compile with NodeTags + TensorSpecs | All 26 | **SC-6** ("26 compile"), **SC-9** |

**Internal parallelism:** H.1 by modality (3 jobs). H.2 and H.3 in parallel (different modalities). H.7 independent.

**Note:** H.1/H.2/H.3/H.7 can start before Track F finishes. Only H.4/H.6 wait for F.

**Exit gate:**
- **SC-6:** Two modes demonstrated on 3 depth modalities (SPC/CACTI/CASSI) + 2 breadth (CT/Widefield); OperatorGraph IR spec as formal contribution; agents predict feasibility; ablations show each agent matters; 26-suite compilation evidence
- **SC-9:** All experiments use GraphOperator exclusively; 26-suite compile validates every template
- **SC-10:** Every experiment backed by RunBundle with SHA256 hashes + seeds + git hash

**Estimated effort:** ~8 hours (Mode 1 sweeps) + ~12 hours (Mode 2 SPC+CACTI) + reuse from F

---

## Wave 3: Paper Writing (launch as experiments complete)

### Track I: Three Manuscripts (parallel)

**Criteria addressed: SC-6, SC-7, SC-8, SC-10**

| Paper | Depends on | Content | Criteria |
|-------|-----------|---------|----------|
| **Paper 3 (PWMI-CASSI)** | Track F done | Algorithm 1+2, bootstrap CI, capture advisor, 10-scene results | **SC-8** (final gate) |
| **Paper 2 (InverseNet)** | Track G done | Dataset, T1-T4 tasks, baselines, leaderboard | **SC-7** (final gate) |
| **Paper 1 (Flagship)** | Track H done (which includes F) | Two modes, OperatorGraph IR spec, agents, depth+breadth, 26-suite | **SC-6** (final gate) |

**Internal parallelism:** Paper 3 manuscript can start as soon as Track F finishes, even if G and H are still running. Paper 2 starts after G. Paper 1 starts after H.

All three manuscripts are independent writing tasks once their experiments are complete.

**Exit gate (all papers):**
- **SC-6:** Paper 1 manuscript submitted with two modes + IR spec + agents + depth + breadth + 26-suite evidence
- **SC-7:** Paper 2 manuscript submitted with dataset + T1-T4 + baselines + leaderboard for SPC/CACTI/CASSI
- **SC-8:** Paper 3 manuscript submitted with Alg 1+2 + 10 scenes + families/severities + bootstrap CI + capture advisor
- **SC-10:** Every table/figure in every paper backed by RunBundle with SHA256 hashes + seeds + git hash + `reproduce.sh`

---

## Track J: Revenue (independent, lowest priority)

**Depends on:** Nothing (can run anytime, does not block anything)

| Task | What |
|------|------|
| J.1 | Define open-core boundary (what's MIT, what's paid) |
| J.2 | Calibration sprint landing page + intake form |
| J.3 | Hosted infrastructure planning (GPU backend, job queue) |

---

## Parallel Execution Timeline

```
Week 1          Week 2          Week 3          Week 4          Week 5+
─────────────── ─────────────── ─────────────── ─────────────── ───────
[Gate 0: 3d]
    ├──→ [Track A: Viral ~~~~~~~~~ 5d]
    ├──→ [Track B: Mode 1 ~~~~~~~~ 5d]
    ├──→ [Track C: Mode 2 ~~~~~~~~~~~~~ 7d]
    ├──→ [Track D: Datasets ~~~~~~ 4d]
    ├──→ [Track E: CI+Infra ~~~ 3d]
    │                                │
    │                    ┌───────────┤ (C+D done)
    │                    ▼           ▼
    │              [Track F: Paper 3 experiments ~~~~~~~ GPU]
    │              [Track G: Paper 2 experiments ~~~~~~~ GPU]
    │                    │           │
    │                    │     ┌─────┤ (B+C+D+F done)
    │                    │     ▼     │
    │                    │ [Track H: Paper 1 experiments ~~~~]
    │                    │     │     │
    │                    ▼     ▼     ▼
    │              [Track I: Paper writing (3 parallel) ~~~~~]
    │
    └──→ [Track J: Revenue ··· anytime, background ··········]
```

---

## Criteria Traceability Matrix

Every criterion must be fully satisfied. This matrix shows where each criterion is **built**, **tested**, and **delivered**.

| SC | Criterion | Built by | Tested by | Delivered by |
|----|-----------|----------|-----------|-------------|
| **SC-1** | Viral: 60-second wow | A.2 (SharePack), A.3 (pwm demo) | A exit gate: <60s timed test | Track A complete |
| **SC-2** | Viral: gallery | A.4 (Gallery site) | A exit gate: 26 cards render | Track A complete |
| **SC-3** | Viral: weekly challenge | A.5 (Calibration Friday) | A exit gate: end-to-end challenge pipeline | Track A complete |
| **SC-4** | Mode 1 end-to-end | B.1-B.5 (agent chain → recon → CLI) | B exit gate: `pwm run` produces RunBundle | Track B complete |
| **SC-5** | Mode 2 end-to-end | C.1-C.8 (scoring → guardrail → calib → CLI) | C exit gate: `pwm calib-recon` produces RunBundle + θ | Track C complete |
| **SC-6** | Paper 1 (Flagship) | Gate 0 (IR spec) + B (Mode 1) + C (Mode 2) + H (experiments) | H exit gate: depth + breadth + 26-suite + ablations | Track I (Paper 1 manuscript) |
| **SC-7** | Paper 2 (InverseNet) | D (dataset infra) + G (generation + baselines) | G exit gate: 1,728 samples + T1-T4 tables + leaderboard | Track I (Paper 2 manuscript) |
| **SC-8** | Paper 3 (PWMI-CASSI) | C (Alg 1+2, bootstrap CI) + F (10-scene experiments) | F exit gate: p < 0.05 + CI coverage >= 90% | Track I (Paper 3 manuscript) |
| **SC-9** | OperatorGraph-first enforced | Gate 0 (G0.2-G0.4) + E (CI gates) | E exit gate: lint + roundtrip + adjoint + stability | Continuous (CI enforced on every commit) |
| **SC-10** | Reproducibility | Gate 0 (G0.1 hashes) + B (B.6) + C (C.8) + E (E.2) | E exit gate: serialize roundtrip; all RunBundles have SHA256 | Every track (RunBundle on all outputs) |

### Criteria completion checkpoints

```
Gate 0 done  → SC-9 (foundation laid), SC-10 (hash infrastructure)
Track A done → SC-1, SC-2, SC-3 COMPLETE
Track B done → SC-4 COMPLETE
Track C done → SC-5 COMPLETE
Track E done → SC-9 COMPLETE (continuous enforcement active)
Track F done → SC-8 (experiments done, awaiting manuscript)
Track G done → SC-7 (experiments done, awaiting manuscript)
Track H done → SC-6 (experiments done, awaiting manuscript)
Track I done → SC-6, SC-7, SC-8 COMPLETE (manuscripts submitted)
All tracks    → SC-10 COMPLETE (every artifact has RunBundle)
```

---

## Summary: Maximum Parallelism from OperatorGraph-First

| Property | Benefit |
|----------|---------|
| **Single IR** | All tracks build on the same GraphOperator interface — no cross-track API negotiation |
| **Compile-time validation** | NodeTags + TensorSpec catch integration bugs at Gate 0, not during experiments |
| **Template-driven** | Viral (Track A) uses the same templates as experiments (F/G/H) — no divergence |
| **Solver API unified** | Mode 1 (Track B) and Mode 2 (Track C) share SolverRunner — no duplication |
| **Mismatch = param perturbation** | Dataset generation (Track D) and calibration (Track C) use identical GraphNode modification — write once |
| **Paper reuse** | Track F results flow into Track H without re-running — OperatorGraph ensures identical operator semantics |

**Maximum concurrent tracks:** 5 (Wave 1) + 2-3 (Wave 2) = up to 7 parallel workstreams.

**Critical path:** Gate 0 → Track C (longest in Wave 1) → Track F → Track H → Paper 1 manuscript.

**All 10 success criteria are covered.** No criterion depends on a single track — each has redundant coverage across build, test, and delivery phases.
