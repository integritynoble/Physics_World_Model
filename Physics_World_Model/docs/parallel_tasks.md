# PWM Parallel Task Division (v2) — Depth/Breadth Paper Strategy + Safer Merges

**Goal:** Maximum parallelism across separate folders (one Claude Code process each), while keeping merges safe.

**Paper strategy baked in:**
- **Depth (Paper 1 core):** SPC + CACTI + CASSI (full design → preflight → calibration → reconstruction → ablations)
- **Breadth anchors (Paper 1 support):** CT + Widefield (+ optional Holography) using minimal checklist:
  - compile → serialize/reproduce → adjoint check (if linear) → 1 mismatch + 1 calibration improvement

---

## Success Criteria

PWM is "successful" when all of the following are met:

1. **Viral (Task A):** A new user runs `pwm demo cassi --preset tissue --export-sharepack` and gets a shareable result (teaser.png + summary.md) in <60 seconds.
2. **Viral (Task A):** The PWM Gallery has 26+ curated modality cards with teaser images, metrics, and reproduce commands.
3. **Viral (Task E):** "Calibration Friday" infrastructure is live — `python community/validate.py submission.zip` and `python community/leaderboard.py --week 2026-W10` work end-to-end, with first 4 weekly challenges published.
4. **Universal IR (Task B):** All 26 modality templates compile to OperatorGraph, validate against schema, serialize with SHA256 hashes, and pass adjoint checks (where linear). `GraphCompiler.compile(cassi_spec).forward(x)` matches `CassiOperator.forward(x)`. CT + Widefield breadth anchors compile and pass adjoint checks.
5. **Calibration (Task C):** `CorrectionResult` includes 95% CI (`theta_uncertainty`) with bootstrap coverage ≥90% on synthetic tests. Capture advisor produces actionable next-capture suggestions when CI is wide. SystemDiscernAgent converts text description → ImagingSystemSpec.
6. **Paper 2 — InverseNet (Task D):** Ships as a public benchmark for SPC/CACTI/CASSI with controlled sweeps (3 CR × 3 photon × 3+ mismatch × 3 severity), manifest.jsonl per modality, `run_baselines.py --smoke` passes, all 4 task baselines have error bars, and dataset_card.md committed.
7. **Paper 3 — PWMI-CASSI (Task F):** UPWMI shows statistically significant improvement (p < 0.05) over grid search and gradient descent. Bootstrap 95% CI covers true θ in ≥90% of trials. "Next capture" advisor measurably reduces uncertainty. All results in RunBundles with reproducibility hashes.
8. **Paper 1 — Flagship PWM (Task G):** 3 depth modalities (SPC + CACTI + CASSI) demonstrate complete design → preflight → calibration → reconstruction loop with paper-ready RunBundles. CT + Widefield breadth checklist passes (compile + adjoint + 1 mismatch/cal). 26/26 templates compile to OperatorGraph. 4 mandatory ablations show measurable degradation (>0.5 dB or metric drop) on all 3 depth modalities.
9. **Revenue (Task E):** Calibration sprint landing page live with service description, intake form, and one anonymized example report.

**Depth vs Breadth rule (Paper 1):** Deep results on SPC/CACTI/CASSI; breadth-only checklist on CT + Widefield (+ Holography optional).

---

## Round 0 (1-2 hours) — Interface Freeze

**Owner:** You (main branch, before copying folders)
**Why:** Prevents merge chaos across A/B/C/D/E by locking shared contracts first.

### Deliverables

Create `docs/contracts/` with frozen interface definitions:

**1. `runbundle_schema.md`** — RunBundle required fields
```yaml
# Every RunBundle must contain:
runbundle_manifest.json:
  version: "0.3.0"
  spec_id: str
  timestamp: str (ISO 8601)
  provenance:
    git_hash: str
    seeds: list[int]
    platform: str
    pwm_version: str
  metrics: {psnr_db: float, ssim: float, runtime_s: float}
  artifacts: {x_gt: path, y: path, x_hat: path}
  hashes: {artifact_name: sha256_hex}
```

**2. `registry_conventions.md`** — Registry ID naming + versioning
```
Format:  <domain>_<name>_v<N>    (e.g. cassi_gap_tv_v1)
Rules:   lowercase, underscores, no spaces, monotonic version numbers
Lookup:  registry.get(id) raises KeyError on miss (never silent fallback)
```

**3. `correction_result_schema.md`** — CorrectionResult fields (Task C must implement this)
```python
class CorrectionResult(StrictBaseModel):
    theta_corrected: dict[str, float]
    theta_uncertainty: dict[str, tuple[float, float]]  # 95% CI per param
    improvement_db: float
    n_evaluations: int
    convergence_curve: list[float]
    bootstrap_seeds: list[int]           # deterministic reproducibility
    resampling_indices: list[list[int]]  # stored in RunBundle
```

**4. `cli_conventions.md`** — CLI subcommand signatures
```
pwm demo <modality> [--preset NAME] [--run] [--open-viewer] [--export-sharepack]
pwm validate <runbundle_dir>
pwm gallery build [--output-dir DIR]
```

**5. `Makefile`** — Add `make check` target
```makefile
check:
	python -m pytest packages/pwm_core/tests/ -x -q
	python -m pytest packages/pwm_core/benchmarks/test_operator_correction.py -x -q
	python packages/pwm_core/pwm_core/agents/_generate_literals.py --check
	@echo "All checks passed."
```

**Exit criteria:**
- `docs/contracts/` committed to main with all 4 schema files.
- `make check` exists and passes on current main.
- All 5 task folders are copied/worktree'd from this frozen commit.

---

## Overview: 3 Rounds, 7 Tasks

```
ROUND 0 — Interface Freeze (you, main branch, 1-2 hours)
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ ROUND 1 — All 5 start simultaneously (no dependencies)             │
│                                                                     │
│  Folder A          Folder B          Folder C          Folder D     │
│  Viral MVP         OperatorGraph IR  Calibration Enh.  InverseNet   │
│  (sharepack,       (graph/ module,   (uncertainty,     (dataset     │
│   demo, gallery)    primitives.yaml,  capture_advisor,  generation,  │
│                     graph_templates,  system_discern)   baselines,   │
│                     +CT/WF prims)                       manifests)   │
│                                                                     │
│  Folder E                                                           │
│  Community +                                                        │
│  Revenue                                                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                     merge gate: make check + smoke tests
                              │
                              v
┌─────────────────────────────────────────────────────────────────────┐
│ ROUND 2 — Starts after merging Round 1                             │
│                                                                     │
│  Folder F                                                           │
│  PWMI-CASSI (Paper 3)                                              │
│  (needs: uncertainty from C + InverseNet CASSI data from D)        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                     merge gate: make check
                              │
                              v
┌─────────────────────────────────────────────────────────────────────┐
│ ROUND 3 — Starts after merging Round 2                             │
│                                                                     │
│  Folder G                                                           │
│  Flagship PWM (Paper 1)                                            │
│  (needs: OperatorGraph from B + InverseNet from D + PWMI from F)   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Round 1 — 5 parallel tasks (start after Round 0 freeze)

### Task A: Viral MVP
**Folder:** `pwm_task_a_viral/`
**Branch:** `task/a-viral`
**Owner:** Claude process 1
**Est.:** ~1,200 lines

**Files to create/modify:**
```
packages/pwm_core/pwm_core/export/sharepack.py       # NEW ~250 lines
packages/pwm_core/pwm_core/cli/demo.py               # NEW ~200 lines
packages/pwm_core/pwm_core/cli/main.py               # EDIT: add demo subcommand
packages/pwm_core/contrib/casepacks/                  # EDIT: add preset tags
docs/gallery/index.html                               # NEW ~300 lines
docs/gallery/generate_gallery.py                      # NEW ~200 lines
docs/gallery/assets/                                  # NEW: generated teaser images
packages/pwm_core/tests/test_sharepack.py             # NEW
packages/pwm_core/tests/test_demo_command.py          # NEW
packages/pwm_core/tests/test_performance_budget.py    # NEW
```

**Prompt for Claude:**
> Implement the Viral MVP from docs/plan1.md Section 3 + Phase 0. Read docs/contracts/ for CLI conventions and RunBundle schema.
>
> 1. Create `export/sharepack.py`:
>    - `generate_teaser_image(x_gt, y, x_hat, modality) -> PIL.Image` — side-by-side PNG.
>    - `generate_teaser_video(x_gt, x_hat, duration=30) -> Path | None` — mp4 via matplotlib.animation + ffmpeg. **Must have pure-python fallback:** if ffmpeg/encoder is unavailable, generate animated GIF via Pillow or print "mp4 skipped; encoder unavailable" and continue without error.
>    - `generate_summary(metrics, modality, mismatch_info) -> str` — 3-bullet markdown.
>    - `export_sharepack(runbundle_dir, output_dir) -> Path` — orchestrator.
> 2. Create `cli/demo.py`: `pwm demo <modality> [--preset] [--run] [--open-viewer] [--export-sharepack]`.
> 3. Wire demo subcommand into `cli/main.py`.
> 4. Build `docs/gallery/`:
>    - `generate_gallery.py` reads RunBundle index (if available) OR `benchmark_results.json` as fallback.
>    - Outputs static HTML site with 26 modality cards (teaser image, 3-bullet summary, PSNR/SSIM, reproduce command).
> 5. Write tests: test_sharepack.py, test_demo_command.py, test_performance_budget.py (demo <5min).
>
> Exit criteria: `pwm demo cassi --preset tissue --export-sharepack` creates teaser.png + summary.md + (teaser.mp4 or prints skip message).

**Dependencies:** None. Uses existing CasePacks, RunBundle, benchmark_results.json.

**Files this task MUST NOT touch:** anything in `graph/`, `mismatch/uncertainty.py`, `mismatch/capture_advisor.py`, `agents/contracts.py`, `agents/system_discern.py`, `analysis/bottleneck.py`, `experiments/`, `community/`, `CONTRIBUTING.md`.

---

### Task B: OperatorGraph IR (+ pulled-left CT/Widefield breadth primitives)
**Folder:** `pwm_task_b_opergraph/`
**Branch:** `task/b-opergraph`
**Owner:** Claude process 2
**Est.:** ~2,400 lines

**Files to create:**
```
packages/pwm_core/pwm_core/graph/__init__.py          # NEW
packages/pwm_core/pwm_core/graph/primitives.py        # NEW ~300 lines
packages/pwm_core/pwm_core/graph/graph_spec.py        # NEW ~150 lines
packages/pwm_core/pwm_core/graph/compiler.py          # NEW ~250 lines
packages/pwm_core/pwm_core/graph/graph_operator.py    # NEW ~200 lines
packages/pwm_core/pwm_core/graph/introspection.py     # NEW ~100 lines
packages/pwm_core/contrib/primitives.yaml             # NEW ~400 lines
packages/pwm_core/contrib/graph_templates.yaml        # NEW ~600 lines
tools/gen_graph_templates_from_casepacks.py            # NEW ~150 lines
packages/pwm_core/tests/test_graph_adjoint.py         # NEW
packages/pwm_core/tests/test_graph_compiler.py        # NEW
packages/pwm_core/tests/test_graph_equivalence.py     # NEW
packages/pwm_core/tests/test_golden_runs.py           # NEW
```

**Prompt for Claude:**
> Implement the OperatorGraph IR from docs/plan1.md Sections 2 + 4 + Phase 1. Read docs/contracts/ for RunBundle and registry conventions.
>
> 1. Create `graph/primitives.py` — PrimitiveOp protocol + ~30 implementations wrapping existing operators (see primitives table in plan1.md Section 4.1). **Ensure CT primitives (`ct_radon`) and Widefield primitives (`conv2d`, `deconv_rl`) are included** — these are needed for Paper 1 breadth anchors.
> 2. Create `graph/graph_spec.py` — OperatorGraphSpec, GraphNode, GraphEdge Pydantic models with StrictBaseModel, extra="forbid".
> 3. Create `graph/compiler.py` — GraphCompiler: validate DAG → bind primitives → build forward/adjoint plans.
> 4. Create `graph/graph_operator.py` — GraphOperator with forward(), adjoint(), serialize() (includes SHA256 hashes), check_adjoint(), explain().
> 5. Create `graph/introspection.py` — deterministic graph explanation without LLM.
> 6. Create `contrib/primitives.yaml` — all ~30 primitives with parameter schemas and adjoint flag.
> 7. Create `contrib/graph_templates.yaml` — skeleton graphs for all 26 modalities. **CT + Widefield templates must exist and compile** (breadth anchors for Paper 1).
> 8. Create `tools/gen_graph_templates_from_casepacks.py` — reads existing operators/casepacks to scaffold graph_templates.yaml entries.
> 9. Write tests:
>    - test_graph_adjoint.py: adjoint checks for all 26 graphs that declare linear=True.
>    - test_graph_compiler.py: compilation + serialization round-trip for all 26 templates.
>    - test_graph_equivalence.py: GraphOperator.forward(x) matches existing operator.forward(x) for SPC, CACTI, CASSI, CT, Widefield.
>    - test_golden_runs.py: same seed → bit-identical output.
>
> Exit criteria:
> - GraphCompiler.compile(cassi_spec).forward(x) matches CassiOperator.forward(x).
> - CT + Widefield templates compile and pass adjoint checks.
> - All 26 templates validate against schema.
> - check_adjoint() passes for all linear graphs.

**Dependencies:** None. Reads existing operators in physics/ for reference but does not modify them.

**Files this task MUST NOT touch:** anything in `cli/`, `export/`, `mismatch/`, `agents/`, `analysis/`, `experiments/`, `community/`, `docs/gallery/`, `docs/calibration-sprint/`, `CONTRIBUTING.md`.

---

### Task C: Calibration Enhancement
**Folder:** `pwm_task_c_calibration/`
**Branch:** `task/c-calibration`
**Owner:** Claude process 3
**Est.:** ~780 lines

**Files to create/modify:**
```
packages/pwm_core/pwm_core/mismatch/uncertainty.py    # NEW ~200 lines
packages/pwm_core/pwm_core/mismatch/capture_advisor.py # NEW ~150 lines
packages/pwm_core/pwm_core/agents/system_discern.py   # NEW ~200 lines
packages/pwm_core/pwm_core/agents/contracts.py        # EDIT: add CorrectionResult CI fields
packages/pwm_core/pwm_core/analysis/bottleneck.py     # EDIT: enhanced bottleneck output
packages/pwm_core/tests/test_bootstrap_ci.py          # NEW
packages/pwm_core/tests/test_capture_advisor.py       # NEW
```

**Prompt for Claude:**
> Implement Calibration Enhancement from docs/plan1.md Sections 5 + 6 + Phase 3 infrastructure. Read docs/contracts/correction_result_schema.md for the exact CorrectionResult fields to implement.
>
> 1. Create `mismatch/uncertainty.py`:
>    - `bootstrap_correction(correction_fn, data, K=20, seed=42) -> CorrectionResult`
>    - Resamples calibration data K times, runs correction_fn on each subsample, reports 95% CI per parameter.
>    - **Must store deterministic seeds and resampling indices** in the result (for RunBundle reproducibility).
>    - Must work with the existing correction loop in test_operator_correction.py.
> 2. Create `mismatch/capture_advisor.py`:
>    - `suggest_next_capture(correction_result, system_spec) -> CaptureAdvice`
>    - When uncertainty bands are wide, returns: what additional measurements to capture, recommended geometry/parameters, expected uncertainty reduction (estimated from Fisher information or bootstrap variance).
>    - Returns **actionable** suggestions, not vague guidance.
> 3. Create `agents/system_discern.py` — SystemDiscernAgent: user text description → ImagingSystemSpec + candidate graph_template_ids. Uses LLMClient, outputs registry IDs only.
> 4. Update `agents/contracts.py` — add theta_uncertainty, convergence_curve, bootstrap_seeds, resampling_indices fields to CorrectionResult per docs/contracts/correction_result_schema.md.
> 5. Update `analysis/bottleneck.py` — output ranked Photon × Recoverability × Mismatch × Solver Fit with expected gain per factor and "what to change first" recommendation.
> 6. Write tests:
>    - test_bootstrap_ci.py: synthetic test where true θ is known; verify coverage ≥90% over 100 trials.
>    - test_capture_advisor.py: verify advisor returns non-empty suggestions when CI is wide, and suggestions are actionable (specific capture parameters).
>
> Exit criteria:
> - CorrectionResult includes theta_uncertainty (95% CI) + convergence_curve + bootstrap_seeds.
> - Bootstrap coverage ≥90% on synthetic test.
> - Capture advisor produces actionable next-capture suggestions.

**Dependencies:** None. Uses existing correction loop, mismatch/, agents/ infrastructure.

**Files this task MUST NOT touch:** anything in `cli/`, `export/`, `graph/`, `experiments/`, `community/`, `docs/gallery/`, `docs/calibration-sprint/`, `contrib/primitives.yaml`, `contrib/graph_templates.yaml`, `CONTRIBUTING.md`.

---

### Task D: InverseNet Dataset + Baselines (Paper 2)
**Folder:** `pwm_task_d_inversenet/`
**Branch:** `task/d-inversenet`
**Owner:** Claude process 4
**Est.:** ~2,050 lines

**Files to create:**
```
experiments/inversenet/__init__.py                    # NEW
experiments/inversenet/gen_spc.py                     # NEW ~300 lines
experiments/inversenet/gen_cacti.py                   # NEW ~300 lines
experiments/inversenet/gen_cassi.py                   # NEW ~300 lines
experiments/inversenet/mismatch_sweep.py              # NEW ~200 lines
experiments/inversenet/run_baselines.py               # NEW ~400 lines
experiments/inversenet/leaderboard.py                 # NEW ~200 lines
experiments/inversenet/package.py                     # NEW ~150 lines
experiments/inversenet/dataset_card.md                # NEW: version, license, recipe
experiments/inversenet/manifest_schema.py             # NEW ~50 lines
packages/pwm_core/tests/test_inversenet_generation.py # NEW
papers/inversenet/README.md                           # NEW: manuscript skeleton
```

**Prompt for Claude:**
> Implement InverseNet (Paper 2) from docs/plan1.md Section 8.1 + Phase 2. Read docs/contracts/ for RunBundle schema and registry conventions.
>
> 1. Create `experiments/inversenet/gen_spc.py` — generate SPC dataset:
>    - Sweep: CR ∈ {10%,25%,50%} × photon ∈ {1e3,1e4,1e5} × mismatch ∈ {gain, mask_error} × severity ∈ {mild, moderate, severe}.
>    - Each sample: x (ground truth), y (measurement), operator params θ, mismatch Δθ + noise φ (labeled), calibration captures y_cal. Output as RunBundle.
> 2. Create `gen_cacti.py` — same for CACTI: frames ∈ {4,8,16} × photon × mismatch ∈ {mask_shift, temporal_jitter}.
> 3. Create `gen_cassi.py` — same for CASSI: bands ∈ {8,16,28} × photon × mismatch ∈ {disp_step, mask_shift, PSF_blur}.
> 4. Create `mismatch_sweep.py` — parameterized mismatch injection (mild/moderate/severe per family). Reuse existing mismatch/ code.
> 5. Create `run_baselines.py` — run all 4 InverseNet tasks:
>    - T1: operator parameter estimation (θ-error RMSE)
>    - T2: mismatch identification (accuracy, F1)
>    - T3: calibration (θ-error ↓, CI coverage)
>    - T4: reconstruction under mismatch (PSNR, SSIM, SAM)
>    - Baselines: oracle/wrong operator, grid/gradient/UPWMI calibration, GAP-TV/PnP/LISTA reconstruction.
>    - Add `--smoke` flag for quick validation run (1 sample per modality).
> 6. Create `leaderboard.py` — score + rank submissions per task. Output markdown table.
> 7. Create `package.py` — package dataset for HuggingFace/Zenodo (tar.gz + checksums).
> 8. Create `manifest_schema.py` — Pydantic model for `manifest.jsonl` (one line per sample):
>    - Fields: seed, photon_level, compression_ratio, mismatch_family, severity, theta, delta_theta, paths.
> 9. Create `dataset_card.md` — version, license (CC-BY-4.0), generation recipe, citation.
> 10. Write test_inversenet_generation.py: verify manifest completeness + RunBundle integrity.
>
> **Use EXISTING operators** (SpcOperator, CactiOperator, CassiOperator) directly. Do NOT depend on the new graph/ module. OperatorGraph specs will be attached as metadata post-merge.
>
> Exit criteria:
> - 3 modalities × 3 CR × 3 photon × 3 mismatch × 3 severity = 81+ RunBundles (or subset with manifest).
> - `run_baselines.py --smoke` completes without error.
> - All 4 task baselines have error bars saved in results/.
> - manifest.jsonl written per modality.
> - dataset_card.md committed.

**Dependencies:** None. Uses existing operators, solvers, mismatch code. OperatorGraph metadata added post-merge.

**Files this task MUST NOT touch:** anything in `cli/`, `export/`, `graph/`, `mismatch/uncertainty.py`, `mismatch/capture_advisor.py`, `agents/`, `analysis/`, `community/`, `docs/gallery/`, `docs/calibration-sprint/`, `contrib/primitives.yaml`, `contrib/graph_templates.yaml`, `CONTRIBUTING.md`.

---

### Task E: Community + Revenue
**Folder:** `pwm_task_e_community/`
**Branch:** `task/e-community`
**Owner:** Claude process 5
**Est.:** ~1,750 lines

**Files to create:**
```
community/challenges/template/challenge.md            # NEW
community/challenges/template/expected.json           # NEW
community/leaderboard.py                              # NEW ~200 lines
community/validate.py                                 # NEW ~150 lines
community/CONTRIBUTING_CHALLENGE.md                   # NEW ~100 lines
community/challenges/2026-W10/                        # NEW (4 weekly challenges)
community/challenges/2026-W11/
community/challenges/2026-W12/
community/challenges/2026-W13/
docs/calibration-sprint/index.html                    # NEW ~200 lines
docs/calibration-sprint/intake-form.html              # NEW ~100 lines
docs/calibration-sprint/example-report/               # NEW: anonymized example
packages/pwm_core/tests/test_challenge_scoring.py     # NEW
CONTRIBUTING.md                                       # EDIT: add template PR workflow
```

**Prompt for Claude:**
> Implement Community + Revenue from docs/plan1.md Sections 3.4 + 9 + Phase 5. Read docs/contracts/ for RunBundle schema.
>
> 1. Create `community/leaderboard.py`:
>    - Score submitted RunBundles against expected.json (PSNR, SSIM, θ-error).
>    - `python community/leaderboard.py --week 2026-W10` outputs leaderboard.md.
> 2. Create `community/validate.py`:
>    - Validate RunBundle submissions: schema check, hash verification, required fields per docs/contracts/runbundle_schema.md.
>    - `python community/validate.py submission.zip` returns pass/fail + error details.
> 3. Create 4 weekly challenge templates under `community/challenges/`:
>    - Each has: challenge.md (problem description + rules), expected.json (reference metrics).
>    - **Avoid repo bloat:** challenges store metadata + a small Python script to generate/fetch dataset slices from existing benchmark data. Do NOT commit large .npz files. Use Git LFS or generate-on-demand.
> 4. Create `docs/calibration-sprint/`:
>    - `index.html` — service description, what's included, pricing placeholder.
>    - `intake-form.html` — what customer provides (system description, measurement data, constraints).
>    - `example-report/` — anonymized example deliverable showing RunBundle output + recommendations.
> 5. Update `CONTRIBUTING.md` — add section on template PR workflow (how to add a modality: operator + CasePack + solver entry + tests).
> 6. Write test_challenge_scoring.py: mock submission → validate → score → leaderboard.
>
> Exit criteria:
> - `python community/validate.py submission.zip` works end-to-end.
> - `python community/leaderboard.py --week 2026-W10` produces leaderboard.md.
> - Landing page is clean, responsive HTML.
> - No large binary files committed (all data generated on demand or via LFS).

**Dependencies:** None. Uses existing RunBundle schema, benchmark data.

**Files this task MUST NOT touch:** anything in `cli/`, `export/`, `graph/`, `mismatch/`, `agents/`, `analysis/`, `experiments/`, `docs/gallery/`, `contrib/primitives.yaml`, `contrib/graph_templates.yaml`.

---

## Round 1 Merge Gate (required before Round 2)

After merging branches A-E into main, run:

```bash
# 1. Integration check
make check

# 2. Viral smoke test
pwm demo cassi --preset tissue --export-sharepack

# 3. InverseNet smoke test
python experiments/inversenet/run_baselines.py --smoke

# 4. Graph compilation smoke test
python -c "
from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec
import yaml
with open('packages/pwm_core/contrib/graph_templates.yaml') as f:
    templates = yaml.safe_load(f)
compiler = GraphCompiler()
for name in ['cassi_sd', 'spc_random', 'cacti_sci']:
    op = compiler.compile(OperatorGraphSpec(**templates[name]))
    print(f'{name}: compiled, adjoint={op.check_adjoint()}')
"

# 5. Bootstrap CI smoke test
python -m pytest packages/pwm_core/tests/test_bootstrap_ci.py -x -q
```

**All 5 must pass before starting Round 2.**

---

## Round 2 — Task F: PWMI-CASSI (Paper 3)

**Branch:** `task/f-pwmi-cassi` (from merged main)
**Owner:** Claude process 6
**Est.:** ~1,330 lines

**Prerequisite merge:** Task C (uncertainty.py, capture_advisor.py) + Task D (InverseNet CASSI data)

**Files to create:**
```
experiments/pwmi_cassi/__init__.py                    # NEW
experiments/pwmi_cassi/run_families.py                # NEW ~300 lines
experiments/pwmi_cassi/cal_budget.py                  # NEW ~200 lines
experiments/pwmi_cassi/comparisons.py                 # NEW ~300 lines
experiments/pwmi_cassi/stats.py                       # NEW ~150 lines
papers/pwmi_cassi/README.md                           # NEW: manuscript skeleton
```

**Prompt for Claude:**
> Implement PWMI-CASSI (Paper 3) from docs/plan1.md Section 8.2 + Phase 3.
>
> 1. Create `experiments/pwmi_cassi/run_families.py`:
>    - Run CASSI calibration across all mismatch families (disp_step, mask_shift/rotation, PSF_blur, wavelength response) on InverseNet CASSI splits from `experiments/inversenet/`.
>    - Record per-family: θ-error (RMSE per param), PSNR/SSIM/SAM improvement, runtime.
>    - Use `mismatch/uncertainty.py` bootstrap for CI on all results.
> 2. Create `cal_budget.py`:
>    - Sweep calibration budget (1/3/5/10 captures) per mismatch family.
>    - Measure: θ-error vs cal budget, uncertainty band width vs cal budget.
>    - Show "next capture" advisor value: uncertainty reduction after each suggested capture.
> 3. Create `comparisons.py`:
>    - Run 5 baselines on identical data:
>      - No calibration (wrong operator)
>      - Grid search (brute-force)
>      - Gradient descent (differentiable GAP-TV)
>      - UPWMI (ours, derivative-free)
>      - UPWMI + gradient refinement (ours, full pipeline)
>    - Record same metrics for all. Output comparison tables.
> 4. Create `stats.py`:
>    - Paired t-tests (UPWMI vs each baseline) per mismatch family.
>    - Effect sizes (Cohen's d).
>    - CI coverage analysis: does 95% bootstrap interval cover true θ in ≥90% of trials?
>    - Output summary table suitable for paper.
>
> All results exported as RunBundles with full provenance.
>
> Exit criteria:
> - UPWMI shows statistically significant improvement (p < 0.05) over grid search and gradient descent.
> - Bootstrap 95% CI covers true θ in ≥90% of trials.
> - "Next capture" advisor measurably reduces uncertainty.
> - All results in RunBundles with reproducibility hashes.

**Dependencies:** Task C (mismatch/uncertainty.py, capture_advisor.py) + Task D (InverseNet CASSI data).

**Files this task MUST NOT touch:** anything in `cli/`, `export/`, `graph/`, `agents/`, `community/`, `docs/`.

---

## Round 2 Merge Gate

```bash
make check
python -m pytest experiments/pwmi_cassi/ -x -q 2>/dev/null || true
# Verify all RunBundles have provenance hashes
python -c "
import json, glob
for rb in glob.glob('experiments/pwmi_cassi/results/**/runbundle_manifest.json', recursive=True):
    m = json.load(open(rb))
    assert 'hashes' in m.get('provenance', m), f'Missing hashes in {rb}'
print('All PWMI-CASSI RunBundles have provenance.')
"
```

---

## Round 3 — Task G: Flagship PWM (Paper 1)

**Branch:** `task/g-flagship` (from merged main)
**Owner:** Claude process 7
**Est.:** ~2,250 lines

**Prerequisite merge:** Task B (OperatorGraph) + Task D (InverseNet) + Task F (PWMI-CASSI)

**Files to create:**
```
experiments/pwm_flagship/__init__.py                  # NEW
experiments/pwm_flagship/spc_loop.py                  # NEW ~300 lines
experiments/pwm_flagship/cacti_loop.py                # NEW ~300 lines
experiments/pwm_flagship/cassi_loop.py                # NEW ~200 lines
experiments/pwm_flagship/breadth_ct.py                # NEW ~150 lines
experiments/pwm_flagship/breadth_wf.py                # NEW ~150 lines
experiments/pwm_flagship/breadth_holo.py              # NEW ~150 lines
experiments/pwm_flagship/universality.py              # NEW ~200 lines
experiments/pwm_flagship/ablations.py                 # NEW ~400 lines
packages/pwm_core/tests/test_universality_26.py       # NEW
packages/pwm_core/tests/test_ablations.py             # NEW
papers/pwm_flagship/README.md                         # NEW: manuscript skeleton
```

**Prompt for Claude:**
> Implement Flagship PWM (Paper 1) from docs/plan1.md Section 8.3 + Phase 4.
>
> **Depth experiments (full loop) — SPC + CACTI + CASSI:**
> 1. Create `spc_loop.py`, `cacti_loop.py`, `cassi_loop.py` — each demonstrates the complete pipeline on InverseNet data:
>    - **Design:** propose system variants under constraints (photons, CR, resolution, exposure).
>    - **Pre-flight:** Photon × Recoverability × Mismatch × Solver Fit predicts success/failure bands.
>    - **Calibration:** correct at least one major mismatch family with uncertainty bands (using mismatch/uncertainty.py) + "next capture" suggestions (using mismatch/capture_advisor.py).
>    - **Reconstruction:** run solvers with full provenance, compare oracle/wrong/calibrated operators.
>    - For CASSI: reference PWMI-CASSI results from experiments/pwmi_cassi/.
>
> **Breadth experiments (minimal checklist) — CT + Widefield (+ optional Holography):**
> 2. Create `breadth_ct.py`, `breadth_wf.py`, `breadth_holo.py` — each must:
>    - Compile to OperatorGraph (using graph/ module from Task B).
>    - Serialize + reproduce (RunBundle + hashes).
>    - Pass adjoint check (if linear).
>    - Demonstrate one mismatch + one calibration improvement (small but real).
>
> **Universality:**
> 3. Create `universality.py` — compile all 26 graph templates from `contrib/graph_templates.yaml`, validate schema, serialize, and run adjoint check where applicable. Output pass/fail table.
>
> **Ablations:**
> 4. Create `ablations.py` — for each of the 3 depth modalities, run 4 ablations:
>    - Remove PhotonAgent → show feasibility predictions degrade.
>    - Remove Recoverability tables → show compression feasibility claims collapse.
>    - Remove mismatch priors → show calibration becomes unstable.
>    - Remove RunBundle discipline → show reproducibility breaks (different seeds/no hashes).
>    - Output: degradation in dB or metric per ablation, with error bars.
>
> 5. Write tests:
>    - test_universality_26.py: all 26 templates compile + validate + serialize.
>    - test_ablations.py: each ablation produces measurable degradation (>0.5 dB or metric drop).
>
> All results exported as RunBundles.
>
> Exit criteria:
> - 3 depth modalities demonstrate design → preflight → calibration → reconstruction loop with paper-ready RunBundles.
> - CT + Widefield breadth checklist passes (compile + adjoint + 1 mismatch/cal).
> - 26/26 templates compile to OperatorGraph.
> - 4 ablations show measurable degradation on all 3 depth modalities.

**Dependencies:** Task B (OperatorGraph IR) + Task D (InverseNet data) + Task F (PWMI-CASSI results).

---

## Git Worktree Setup (recommended over folder copies)

```bash
# From the main repo directory, after Round 0 freeze:
git worktree add ../pwm_task_a_viral     -b task/a-viral
git worktree add ../pwm_task_b_opergraph -b task/b-opergraph
git worktree add ../pwm_task_c_calibration -b task/c-calibration
git worktree add ../pwm_task_d_inversenet -b task/d-inversenet
git worktree add ../pwm_task_e_community -b task/e-community

# Each Claude process works in its own worktree.
# After Round 1, merge all branches:
git checkout main
git merge task/a-viral task/b-opergraph task/c-calibration task/d-inversenet task/e-community

# Round 2:
git worktree add ../pwm_task_f_pwmi -b task/f-pwmi-cassi
# ... work ... then merge

# Round 3:
git worktree add ../pwm_task_g_flagship -b task/g-flagship
# ... work ... then merge
```

**Why worktrees > folder copies:**
- Shared .git → smaller disk footprint.
- `git merge` handles conflicts properly.
- Each branch tracks exactly what changed → easy code review.
- `git log --all --graph` shows the full picture.

---

## Conflict Zones (files touched by multiple tasks)

| File | Touched by | Resolution |
|------|-----------|------------|
| `cli/main.py` | A only | No conflict |
| `agents/contracts.py` | C only | No conflict |
| `agents/__init__.py` | C only (adds system_discern import) | No conflict |
| `analysis/bottleneck.py` | C only | No conflict |
| `CONTRIBUTING.md` | E only | No conflict |

**All other files are NEW and task-exclusive → zero conflicts.**

---

## Summary Table

| Task | Round | Parallel? | Est. lines | Focus | Branch |
|------|-------|-----------|-----------|-------|--------|
| **Round 0: Interface Freeze** | 0 | Pre-req | ~200 | contracts, Makefile | main |
| **A: Viral MVP** | 1 | 5-way | ~1,200 | sharepack, demo, gallery | task/a-viral |
| **B: OperatorGraph IR** | 1 | 5-way | ~2,400 | graph/, YAML registries, CT/WF prims | task/b-opergraph |
| **C: Calibration Enhancement** | 1 | 5-way | ~780 | uncertainty, capture_advisor, system_discern | task/c-calibration |
| **D: InverseNet (Paper 2)** | 1 | 5-way | ~2,050 | dataset gen, baselines, manifests | task/d-inversenet |
| **E: Community + Revenue** | 1 | 5-way | ~1,750 | challenges, leaderboard, landing page | task/e-community |
| **F: PWMI-CASSI (Paper 3)** | 2 | Solo | ~1,330 | calibration experiments, comparisons | task/f-pwmi-cassi |
| **G: Flagship PWM (Paper 1)** | 3 | Solo | ~2,250 | depth/breadth experiments, ablations | task/g-flagship |
| **Total** | | | **~11,960** | | |

**Parallelism gain:** Round 1 runs 5 tasks simultaneously (~8,180 lines, 68% of total) → ~5x speedup on the bulk of the work.
