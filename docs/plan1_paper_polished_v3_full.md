# PWM Development Plan v4.1 — Viral-First + Publishable + Monetizable

**Priority order:** (1) Viral adoption → (2) Top papers → (3) Revenue
**Date:** 2026-02-09
**Baseline:** Plan v3 fully implemented — 26 modalities, 45+ solvers, 17 agents, 6 YAML registries, 16 operator correction tests, all passing.

---

## 0) Executive Summary

### What we have (solid foundation)
- 26 imaging modalities benchmarked end-to-end with reference-quality PSNR
- 45+ reconstruction solvers (traditional, DL, hybrid)
- 17 deterministic-first agents with optional LLM enhancement
- 6 YAML registries (modalities, solvers, compression, mismatch, photon, metrics)
- 23 CasePacks (validated modality experiment templates)
- RunBundle export with full provenance (git hash, seeds, SHA256 hashes)
- Streamlit viewer for interactive exploration
- CLI: `pwm run`, `pwm fit-operator`, `pwm calib-recon`, `pwm view`
- Operator correction tested on 16 modalities (cross-validation metric)
- Strict Pydantic contracts (`extra="forbid"`, NaN/Inf rejection)

### What's missing (the gaps this plan closes)

| Gap | Priority | Why it matters |
|-----|----------|----------------|
| **SharePack exporter** | Viral | One-command shareable artifact (image + video + summary) |
| **`pwm demo` command + presets** | Viral | Zero-config "wow" in 60 seconds |
| **Gallery website** | Viral | Public showcase of 26 modalities |
| **OperatorGraph IR + GraphCompiler** | Paper | Universal representation claim |
| **primitives.yaml + graph_templates.yaml** | Paper | Composable primitive library |
| **Recoverability tables with full provenance** | Paper | Empirical, not theoretical |
| **Community infrastructure** | Viral | Weekly challenges, leaderboard, contributor pipeline |
| **Open-core boundary + landing page** | Revenue | Calibration sprint service |

---

## 1) Success Criteria

PWM is "successful" when:

1. **Viral:** A new user runs `pwm demo cassi` and gets a shareable result in <60 seconds.
2. **Viral:** The PWM Gallery has 26+ curated demos with reproduce buttons.
3. **Viral:** "Calibration Friday" runs weekly with community submissions.
4. **Paper 1 (Flagship):** **"Physics World Model for Imaging"** paper has a complete, reproducible system story (OperatorGraph → feasibility → design → calibration → reconstruction) validated across 26 modalities and deeply on **SPC + CACTI + CASSI**.
5. **Paper 2 (Dataset/Benchmark):** **InverseNet** ships as a public benchmark for **SPC/CACTI/CASSI** with parameter sweeps (photon levels, compression ratios, mismatch severities) and RunBundle-grounded reproducibility.
6. **Paper 3 (Method/Applied):** **PWMI-CASSI** shows statistically significant, robust calibration gains (operator parameter error ↓, reconstruction quality ↑) with uncertainty bands and a practical capture protocol.
7. **Revenue:** First paid calibration sprint delivered.

**Depth vs Breadth rule (Paper 1):** Deep results on SPC/CACTI/CASSI; breadth-only checklist on CT + Widefield (+ Holography optional).

---

## 2) Architecture: Two Layers

### Layer A — OperatorGraph (Universal IR)
Everything compiles to a `GraphOperator` with `forward()`, `adjoint()`, `serialize()`, `check_adjoint()`.

**New files needed:**
```
packages/pwm_core/pwm_core/graph/
├── primitives.py          # PrimitiveOp protocol + implementations
├── graph_spec.py          # OperatorGraphSpec (Pydantic, DAG of primitives)
├── compiler.py            # GraphCompiler: validate → bind → plan → export
├── graph_operator.py      # GraphOperator: forward/adjoint over compiled graph
└── introspection.py       # Explain graph structure without LLM
```

**New registries:**
```
packages/pwm_core/contrib/
├── primitives.yaml        # ~30 primitive operators with parameter schemas
└── graph_templates.yaml   # 26 skeleton graphs (one per modality)
```

### Layer B — 26 Modality Templates (Human-Facing)
Already implemented. Each modality has a CasePack + operator + solver suite.
Templates reference `graph_template_id` once Layer A is built.

**Rule:** Templates never gate new systems. If it compiles to primitives, PWM runs.

---

## 3) The Viral Unit (What People Share)

### 3.1 SharePack: One-command shareable artifact

**CLI goal:**
```bash
pwm demo cassi --preset tissue --run --open-viewer --export-sharepack
```

**Outputs:**
```
sharepack/
├── teaser.png          # Before/after side-by-side (ground truth | measurement | reconstruction)
├── teaser.mp4          # 30s auto-animated slider (matplotlib + ffmpeg)
├── summary.md          # 3 bullets: problem → approach → result
├── metrics.json        # PSNR, SSIM, runtime
└── reproduce.sh        # One-line command to reproduce
```

**Implementation (new file):**
```
packages/pwm_core/pwm_core/export/sharepack.py  (~250 lines)
```

Key functions:
- `generate_teaser_image(x_gt, y, x_hat, modality) -> PIL.Image`
- `generate_teaser_video(x_gt, x_hat, duration=30) -> Path`
- `generate_summary(metrics, modality, mismatch_info) -> str`
- `export_sharepack(runbundle_dir, output_dir) -> Path`

**Dependencies:** matplotlib, Pillow, ffmpeg (optional for video).

### 3.2 `pwm demo` Command + Presets

**New CLI subcommand** in `cli/main.py`:
```bash
pwm demo <modality> [--preset <name>] [--run] [--open-viewer] [--export-sharepack]
```

**Preset system** (new file):
```
packages/pwm_core/pwm_core/cli/demo.py  (~200 lines)
```

Each modality gets 1-3 presets stored in existing CasePacks:
- `cassi: tissue, satellite, urban`
- `cacti: moving_disk, rotating_bar`
- `mri: brain_t1, knee_pd`
- etc.

The `demo` command:
1. Loads CasePack by modality + preset name
2. Runs simulation → reconstruction pipeline
3. Exports RunBundle + SharePack
4. Optionally opens viewer

### 3.3 PWM Gallery (Public Static Site)

**Implementation:**
```
docs/gallery/
├── index.html             # Static page listing all 26 modalities
├── assets/                # Teaser images per modality
├── _template.html         # Card template for each modality
└── generate_gallery.py    # Script: reads benchmark_results.json → renders gallery
```

Each gallery card shows:
- Modality name + icon
- 3-bullet story (from SharePack summary.md)
- Before/after teaser image
- PSNR / SSIM metrics
- "Reproduce" button with CLI command
- Link to CasePack + RunBundle

**Tech:** Plain HTML + CSS (no framework). Can be hosted on GitHub Pages.

### 3.4 Weekly Community Loop ("Calibration Friday")

**Infrastructure:**
```
community/
├── challenges/
│   └── 2026-W07/
│       ├── challenge.md       # Problem description + rules
│       ├── dataset_slice.npz  # Small test data
│       └── expected.json      # Reference metrics
├── submissions/               # Community RunBundle zips
├── leaderboard.py             # Score + rank submissions
└── CONTRIBUTING_CHALLENGE.md  # How to participate
```

**Workflow:**
1. Publish challenge (modality + mismatch scenario + small dataset)
2. Community runs `pwm calib-recon --challenge 2026-W07 --export-sharepack`
3. Submit RunBundle zip
4. Automated scoring → leaderboard update
5. Weekly post: top submissions + analysis

---

## 4) OperatorGraph IR (The Universal Representation)

### 4.1 Primitive Library

Each primitive implements:
```python
class PrimitiveOp(Protocol):
    primitive_id: str
    def forward(self, x: np.ndarray, **params) -> np.ndarray: ...
    def adjoint(self, y: np.ndarray, **params) -> np.ndarray: ...
    def serialize(self) -> dict: ...
```

**Starter primitives (mapped to existing operators):**

| Family | Primitives | Maps to existing |
|--------|-----------|-----------------|
| Propagation | `fresnel_prop`, `angular_spectrum`, `ray_trace` | holography, ptychography |
| PSF/Convolution | `conv2d`, `conv3d`, `deconv_rl` | widefield, confocal |
| Modulation | `coded_mask`, `dmd_pattern`, `sim_pattern` | CASSI, SPC, SIM |
| Warp/Dispersion | `spectral_dispersion`, `chromatic_warp` | CASSI |
| Sampling | `random_mask`, `ct_radon`, `mri_kspace`, `temporal_mask` | SPC, CT, MRI, CACTI |
| Nonlinearity | `magnitude_sq`, `saturation`, `log_compress` | holography, OCT |
| Noise | `poisson`, `gaussian`, `poisson_gaussian`, `fpn` | all |
| Temporal | `frame_integration`, `motion_warp` | CACTI, light-sheet |
| Readout | `quantize`, `adc_clip` | all |

**Registry:** `primitives.yaml` stores `primitive_id` + parameter schema + adjoint flag.

### 4.2 Graph Schema (Pydantic)

```python
class OperatorGraphSpec(StrictBaseModel):
    """DAG of primitive operators defining a forward model."""
    graph_id: str
    nodes: list[GraphNode]         # Each node = primitive_id + params
    edges: list[GraphEdge]         # Tensor flow with axis metadata
    noise_model: NoiseSpec | None  # Explicit noise node
    metadata: dict[str, Any]       # Modality tags, references

class GraphNode(StrictBaseModel):
    node_id: str
    primitive_id: str              # Must exist in primitives.yaml
    params: dict[str, Any]         # Bound at compile time
    learnable: list[str] = []      # Parameters eligible for calibration

class GraphEdge(StrictBaseModel):
    source: str                    # node_id
    target: str                    # node_id
    axes: list[str] = []           # Axis metadata for compatibility checking
```

### 4.3 GraphCompiler

Compilation pipeline:
1. **Validate:** DAG check, all primitive_ids exist, axis compatibility
2. **Bind:** Instantiate primitives with parameters θ
3. **Plan forward:** Topological sort → sequential execution plan
4. **Plan adjoint:** Reverse topological order (for linear graphs)
5. **Export:** RunBundle artifacts (blobs + hashes)

```python
class GraphCompiler:
    def compile(self, spec: OperatorGraphSpec) -> GraphOperator: ...

class GraphOperator:
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def adjoint(self, y: np.ndarray) -> np.ndarray: ...
    def serialize(self) -> dict: ...
    def check_adjoint(self, rtol: float = 1e-4) -> bool: ...
    def explain(self) -> str: ...  # Deterministic introspection
```

### 4.4 Graph Templates

`graph_templates.yaml` provides skeleton graphs for each modality:

```yaml
cassi_sd:
  description: "Single-disperser CASSI"
  nodes:
    - {node_id: modulate, primitive_id: coded_mask}
    - {node_id: disperse, primitive_id: spectral_dispersion}
    - {node_id: integrate, primitive_id: frame_integration}
    - {node_id: noise, primitive_id: poisson_gaussian}
  edges:
    - {source: modulate, target: disperse}
    - {source: disperse, target: integrate}
    - {source: integrate, target: noise}
  learnable: [modulate.mask_shift, disperse.dispersion_coeff]
```

---

## 5) Agents (Existing + Enhancements)

### Already implemented (17 agents)
- PlanAgent, PhotonAgent, MismatchAgent, RecoverabilityAgent
- AnalysisAgent, ContinuityChecker, Negotiator
- PhysicsStageVisualizer, PreFlightReport
- UPWMIAgent, SelfImprovementLoop
- WhatIfPrecomputer, AssetManager, HybridModalityAgent
- LLMClient (multi-LLM fallback), RegistryBuilder, Contracts

### New: SystemDiscernAgent (v4.1 addition)

Converts user description + metadata → ImagingSystemSpec + candidate graphs.

```
packages/pwm_core/pwm_core/agents/system_discern.py  (~200 lines)
```

Inputs:
- User natural-language description of their system
- Optional: metadata files, calibration snippets, photos

Outputs:
- `ImagingSystemSpec` with confidence bands
- Top-K candidate `graph_template_id`s if ambiguous
- "What additional info would help" suggestions

Uses LLM for semantic understanding, but all outputs are registry IDs.

### Enhancement: AnalysisAgent bottleneck output

Extend existing `analysis/bottleneck.py` to output:
- Ranked: Photon × Recoverability × Mismatch × Solver Fit
- Primary bottleneck + expected gain from fixing each factor
- "What to change first" recommendation

---

## 6) Calibration & Operator Correction (Existing + Enhancement)

### Already implemented
- 16-modality operator correction test suite (all passing)
- Cross-validation calibration metric (avoids circular reprojection)
- UPWMI beam search (derivative-free, registry-guided perturbations)
- GPU differentiable GAP-TV (50x speedup)

### Enhancement: Uncertainty bands

Add confidence intervals to corrected parameters:

```python
class CorrectionResult(StrictBaseModel):
    theta_corrected: dict[str, float]
    theta_uncertainty: dict[str, tuple[float, float]]  # 95% CI
    improvement_db: float
    n_evaluations: int
    convergence_curve: list[float]
```

Implementation: Bootstrap resampling of calibration data (run correction K times on subsampled data, report parameter spread).

### Enhancement: "Next capture suggestion"

When calibration is underdetermined (wide uncertainty bands), suggest:
- What additional measurements to capture
- Recommended capture geometry/parameters
- Expected uncertainty reduction

---

## 7) Registries (Existing + New)

### Existing (6 files, all populated)
| Registry | Lines | Status |
|----------|-------|--------|
| `modalities.yaml` | 2,300 | 26 modalities |
| `solver_registry.yaml` | 710 | 43+ solvers |
| `compression_db.yaml` | 1,186 | Compression tables |
| `mismatch_db.yaml` | 706 | Mismatch families |
| `photon_db.yaml` | 254 | Photon models |
| `metrics_db.yaml` | 48 | Quality metrics |

### New (2 files)
| Registry | Est. lines | Purpose |
|----------|-----------|---------|
| `primitives.yaml` | ~400 | ~30 primitive operators + parameter schemas |
| `graph_templates.yaml` | ~600 | 26 skeleton graphs (one per modality) |

### Enhancement: Recoverability tables with provenance

Current `compression_db.yaml` has recoverability data but needs:
- `dataset_id`, `seed_set`, `software_versions`, `measurement_date`
- Empirical confidence intervals (not just point estimates)
- Clear distinction: measured vs interpolated entries

---

## 8) Paper Track (Re-centered to Your 3-Paper Goal)

You want three tightly-coupled papers that share one codebase, one artifact standard (RunBundle), and one narrative arc:

1) **Flagship paradigm paper**: *Physics World Model for Imaging* (the “system + paradigm” paper)  
2) **Benchmark paper**: *InverseNet* (the dataset/benchmark that makes the flagship paper un-ignorable)  
3) **Applied method paper**: *PWMI-CASSI* (a focused calibration result that proves the digital-twin loop works on a hard real modality)

> **Design principle:** Paper 2 (InverseNet) is not “extra work.” It is the *evidence engine* that powers Paper 1’s claims and Paper 3’s evaluation.

---

### Paper 1 (Flagship / Paradigm): **“Physics World Model for Imaging”**
**Goal:** A cross-disciplinary, high-impact “paradigm + system” paper.  
**Title you want:** *Physics World Model for Imaging*

#### Core claims (make them concrete and testable)
1. **Executable digital twin** for computational imaging within a declared **PhysicsScope**: any in-scope system compiles to an **OperatorGraph** and runs end-to-end.
2. **Pre-flight feasibility** is predictable (before expensive reconstruction) by combining:
   - photon budget + noise regime (PhotonAgent),
   - recoverability / compression feasibility (RecoverabilityAgent with empirical tables),
   - mismatch sensitivity + calibration identifiability (MismatchAgent),
   - solver/prior fit.
3. **Design + calibration + reconstruction** are a single reproducible loop:
   - design: propose hardware/code/exposure changes to hit targets,
   - calibrate: infer θ with uncertainty + “next capture” suggestions,
   - reconstruct: run solvers with full provenance.
4. **Reproducibility as a first-class contribution**: every figure and table is backed by **RunBundles** (hashes, seeds, code versions, artifacts).

#### Experiments (Depth vs Breadth: designed for Nature/Science)

**Depth (full loop, strong claims): SPC + CACTI + CASSI**  
These three are the flagship modalities. For each one, PWM must demonstrate the complete pipeline:

1) **Design (synthesis):** propose system variants under constraints (photons, CR, resolution, exposure, cost).  
2) **Pre-flight feasibility prediction:** Photon × Recoverability × Mismatch × Solver Fit predicts success/failure bands *before* expensive runs.  
3) **Calibration (θ inference):** correct at least one major mismatch family with uncertainty bands + “next capture” suggestions.  
4) **Reconstruction:** reproducible reconstruction with baselines, robustness checks, and ablations.  

**Breadth (minimal checklist, universality anchors): CT + Widefield (+ Holography optional)**  
These are included to support the paradigm/universality claim without scope explosion. For each breadth modality, require only:

- **Compile** to OperatorGraph  
- **Serialize + reproduce** (RunBundle + hashes)  
- **Adjoint check** (if linear)  
- **One mismatch + one calibration improvement** (small but real)  

**Universality evidence (26 templates):** all 26 modality templates compile to OperatorGraph, validate, serialize, and (where applicable) pass adjoint checks.  
> The 26-suite is the broad evidence layer; the depth trio provides the hard, deep results.

**Ablations (mandatory):**
- remove PhotonAgent → feasibility predictions fail under low-photon regimes  
- remove Recoverability tables → compression feasibility claims collapse  
- remove mismatch priors → calibration becomes unstable / non-identifiable  
- remove RunBundle discipline → reproducibility breaks  


#### “What reviewers will look for” (build these in)
- Not just “we can reconstruct”: **we can predict failure and explain bottlenecks**.
- Not just “we have agents”: **the agents output verifiable quantities** and drive decisions.
- Not just “we’re universal”: **PhysicsScope is explicit, extensible, and tested**.

---

### Paper 2 (Dataset/Benchmark): **“InverseNet: A Benchmark for Operator Mismatch, Calibration, and Reconstruction”**
**Scope (your choice):** start with **SPC + CACTI + CASSI** only—deep, not broad.

#### What makes InverseNet publishable
- It defines a **standard task suite** where the unknown is not only x (scene), but also **θ (operator parameters)** and mismatch.
- It provides **controlled sweeps** that researchers can’t easily reproduce:
  - photon levels,
  - compression ratios,
  - mismatch severities and families,
  - calibration data regimes (how much calibration is enough).

#### Dataset structure (what you release)
For each modality (SPC, CACTI, CASSI), each sample includes:
- `x` (ground truth), `y` (measurement)
- `A_θ` graph spec + θ (true operator parameters)
- mismatch `Δθ` + noise `ϕ` (with labels)
- calibration captures (subset) + recommended protocol tags
- RunBundle pointer (or packaged RunBundle) so every sample is reproducible

#### Tasks (leaderboard-ready)
1. **Operator parameter estimation** (recover θ / Δθ)
2. **Mismatch identification/classification** (which family? which severity?)
3. **Calibration** (produce corrected operator with uncertainty)
4. **Reconstruction under mismatch** (quality vs calibration budget)

#### Baselines (reuse your existing strengths)
- “oracle operator” recon vs “wrong operator” recon
- calibration: grid / gradient / UPWMI
- reconstruction: GAP-TV, PnP-FISTA+DRUNet, unrolled nets, diffusion-based recon (optional track)

#### Why Paper 2 helps Paper 1 and 3
- Paper 1 needs a *public*, repeatable way to prove universality + feasibility.
- Paper 3 needs a *controlled* evaluation bed for CASSI calibration claims.

---

### Paper 3 (Applied / Method): **“PWMI-CASSI: Practical Calibration of CASSI via Physics World Model”**
**Goal:** show PWM solves a hard real-world pain point: CASSI calibration.

#### Contributions
1. A **CASSI OperatorGraph** that exposes learnable parameters (dispersion step, mask shift, PSF blur, spectral response).
2. A **practical calibration protocol** (what to capture, in what order, under what constraints).
3. A robust **operator correction pipeline**:
   - UPWMI (derivative-free) + optional differentiable refinement,
   - uncertainty bands (bootstrap),
   - “next capture” advisor when underdetermined.

#### Experiments (make them “obviously real”)
- mismatch families: dispersion-step error, mask shift/rotation, PSF blur, wavelength-dependent response
- regimes: low-photon vs high-photon, low-CR vs high-CR
- outcomes:
  - θ error ↓ (calibration accuracy)
  - PSNR/SSIM/SAM ↑ (reconstruction quality)
  - uncertainty bands narrow after recommended captures
  - runtime and robustness vs baselines

#### Artifacts
- `pwm pwmi-cassi` end-to-end CLI command
- evaluation scripts + RunBundles for all reported results
- optional: a small real calibration dataset (even if the first release is synthetic + “real-ready” protocol)

---

### Paper packaging rule (very important)
- **One repo**, three paper folders, one shared artifact standard:
  - `papers/pwm_flagship/`
  - `papers/inversenet/`
  - `papers/pwmi_cassi/`
- Every figure must be reproducible from an “experiments/” folder with RunBundles.


## 9) Revenue Track (Third Priority, Doesn't Fight OSS)

### 9.1 Open-core boundary

**Free (MIT/Apache):**
- All primitives + GraphCompiler + templates + demos
- Viewer (Streamlit)
- CLI + SharePack exporter
- 26 CasePacks + all YAML registries
- Community challenge infrastructure

**Paid:**
- Enterprise viewer (team sharing, access control, audit trails, SSO)
- Hosted "PWM Runs" (GPU backend + job queue + RunBundle registry)
- Premium primitive packs (vendor-specific optics, proprietary sensor models)
- Calibration/design sprints (2-4 week engagements)

### 9.2 First revenue product: PWM Calibration Sprint

**Service package:**
- Customer provides: measurement data + system description
- PWM delivers:
  1. Calibrated operator with uncertainty bands
  2. RunBundle suite + viewer access
  3. Recommended capture protocol
  4. Design suggestions (what to change to improve SNR/recoverability)
- Duration: 2-4 weeks
- Deliverable: RunBundle zip + written report

**Landing page:**
```
docs/calibration-sprint/
├── index.html          # Service description + pricing
├── intake-form.html    # What to provide
└── example-report/     # Anonymized example deliverable
```

---

## 10) Implementation Phases

### Phase 0 (Weeks 1-2): Viral MVP — SharePack + Demo + Gallery

**Goal:** A new user gets a shareable "wow" artifact in 60 seconds.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| SharePack exporter | `export/sharepack.py` | 250 | matplotlib, Pillow |
| Teaser image generator | (in sharepack.py) | - | RunBundle artifacts |
| Teaser video generator | (in sharepack.py) | - | ffmpeg |
| Summary markdown generator | (in sharepack.py) | - | metrics.json |
| `pwm demo` CLI command | `cli/demo.py` | 200 | CasePacks |
| Demo presets (3 hero modalities) | CasePack updates | 50 | - |
| Gallery static site | `docs/gallery/` | 500 | SharePacks for 26 modalities |
| Gallery generator script | `docs/gallery/generate_gallery.py` | 200 | benchmark_results.json |

**Exit criteria:**
- `pwm demo cassi --preset tissue --export-sharepack` produces teaser.png + summary.md
- Gallery page renders 26 modality cards with images and reproduce commands
- Three hero demos (CASSI, CACTI, MRI) run end-to-end in <5 minutes each

### Phase 1 (Weeks 3-6): OperatorGraph IR + Primitives

**Goal:** All 26 modalities compile to a common graph representation.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| PrimitiveOp protocol | `graph/primitives.py` | 300 | existing operators |
| OperatorGraphSpec schema | `graph/graph_spec.py` | 150 | Pydantic |
| GraphCompiler | `graph/compiler.py` | 250 | primitives.py |
| GraphOperator | `graph/graph_operator.py` | 200 | compiler.py |
| Graph introspection | `graph/introspection.py` | 100 | graph_operator.py |
| primitives.yaml | `contrib/primitives.yaml` | 400 | existing operator code |
| graph_templates.yaml | `contrib/graph_templates.yaml` | 600 | modalities.yaml |
| Adjoint checks for all graphs | `tests/test_graph_adjoint.py` | 200 | GraphOperator |
| Serialization + SHA256 | (in graph_operator.py) | - | RunBundle |

**Exit criteria:**
- `GraphCompiler.compile(cassi_graph_spec).forward(x)` produces same output as existing `CassiOperator.forward(x)`
- `check_adjoint()` passes for all linear graphs
- All 26 graph templates validate against schema

### Phase 2 (Weeks 7-10): Calibration Enhancement + Recoverability Tables

**Goal:** Calibration results include uncertainty; recoverability tables have full provenance.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| Bootstrap uncertainty for correction | `mismatch/uncertainty.py` | 200 | correction loop |
| CorrectionResult with CI | `agents/contracts.py` update | 30 | uncertainty.py |
| "Next capture" suggestion | `mismatch/capture_advisor.py` | 150 | uncertainty analysis |
| Recoverability provenance fields | `compression_db.yaml` update | 100 | - |
| SystemDiscernAgent | `agents/system_discern.py` | 200 | LLMClient |
| Enhanced AnalysisAgent bottleneck | `analysis/bottleneck.py` update | 100 | all agents |

**Exit criteria:**
- `CorrectionResult` includes 95% CI for all corrected parameters
- `compression_db.yaml` entries have `dataset_id`, `seed_set`, `measurement_date`
- SystemDiscernAgent converts text description → ImagingSystemSpec

### Phase 3 (Weeks 11-14): Community Scaling + 26-Suite SharePacks
- Add **CT + Widefield** breadth anchors (compile/serialize/adjoint/mismatch+calibration) and optionally a **Holography** nonlinear stress test.

**Goal:** Every modality has a SharePack; community challenge infrastructure live.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| Generate SharePacks for all 26 modalities | script | 100 | sharepack.py |
| Challenge infrastructure | `community/` | 500 | RunBundle |
| Leaderboard scoring | `community/leaderboard.py` | 200 | metrics |
| Challenge submission validator | `community/validate.py` | 150 | RunBundle schema |
| CONTRIBUTING_CHALLENGE.md | `community/` | 100 | - |
| Template PR workflow docs | `CONTRIBUTING.md` update | 50 | - |
| Publish first 4 weekly challenges | `community/challenges/` | 200 | datasets |

**Exit criteria:**
- Gallery shows 26 SharePacks with teaser images
- `pwm calib-recon --challenge 2026-W11` works end-to-end
- Leaderboard auto-updates from submitted RunBundles

### Phase 4 (Weeks 15+): Paper Submissions + First Revenue

**Goal:** Paper 1/2/3 ready (PWM / InverseNet / PWMI-CASSI); first calibration sprint delivered.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| Paper 1 (PWM): flagship experiments + ablations | `experiments/paper_a/` | 500 | OperatorGraph |
| Paper 1 (PWM): manuscript ("Physics World Model for Imaging") | `papers/paper_a/` | - | experiments |
| Paper 2 (InverseNet): dataset generation + sweeps | `experiments/paper_b/` | 300 | mismatch_db |
| Paper 2 (InverseNet): protocol + stats + leaderboard scripts | `experiments/paper_b/` | 200 | benchmark data |
| Paper 3 (PWMI-CASSI): method comparisons | `experiments/paper_c/` | 400 | correction results |
| Paper 3 (PWMI-CASSI): uncertainty + profiling | `experiments/paper_c/` | 200 | uncertainty.py |
| Calibration sprint landing page | `docs/calibration-sprint/` | 300 | - |
| Intake form + example report | `docs/calibration-sprint/` | 200 | RunBundle |

**Exit criteria:**
- Paper 1/2/3 manuscripts ready for submission (PWM / InverseNet / PWMI-CASSI)
- Landing page live with pricing + intake form
- One example calibration sprint report (anonymized)

---

## 11) Testing Strategy

### Existing (maintained)
- 45 unit tests (all passing)
- 16 operator correction tests (all passing, cross-validation metric)
- Registry integrity validation
- Adjoint checks on all linear operators

### New tests for this plan

| Test suite | What it validates | Phase |
|-----------|-------------------|-------|
| `test_sharepack.py` | SharePack generation (image, summary, reproduce.sh) | 0 |
| `test_demo_command.py` | `pwm demo` CLI with all presets | 0 |
| `test_graph_adjoint.py` | Adjoint correctness for all 26 graph templates | 1 |
| `test_graph_compiler.py` | Graph compilation + serialization round-trip | 1 |
| `test_graph_equivalence.py` | GraphOperator output matches existing operator output | 1 |
| `test_bootstrap_ci.py` | Uncertainty bands are calibrated (coverage test) | 2 |
| `test_challenge_scoring.py` | Challenge submission validation + scoring | 3 |
| `test_golden_runs.py` | Reproducibility: same seed → bit-identical output | 1 |
| `test_performance_budget.py` | Runtime budgets (demo <5min, correction <30min) | 0 |

---

## 12) File Structure (New Files Only)

```
packages/pwm_core/
├── pwm_core/
│   ├── graph/                         # NEW: OperatorGraph IR
│   │   ├── __init__.py
│   │   ├── primitives.py              # PrimitiveOp implementations
│   │   ├── graph_spec.py              # OperatorGraphSpec Pydantic model
│   │   ├── compiler.py                # GraphCompiler
│   │   ├── graph_operator.py          # GraphOperator
│   │   └── introspection.py           # Deterministic graph explanation
│   ├── cli/
│   │   └── demo.py                    # NEW: pwm demo command
│   ├── export/
│   │   └── sharepack.py               # NEW: SharePack exporter
│   ├── mismatch/
│   │   ├── uncertainty.py             # NEW: Bootstrap uncertainty
│   │   └── capture_advisor.py         # NEW: Next capture suggestion
│   └── agents/
│       └── system_discern.py          # NEW: SystemDiscernAgent
├── contrib/
│   ├── primitives.yaml                # NEW: Primitive operator registry
│   └── graph_templates.yaml           # NEW: 26 graph skeletons
└── tests/
    ├── test_sharepack.py              # NEW
    ├── test_demo_command.py           # NEW
    ├── test_graph_adjoint.py          # NEW
    ├── test_graph_compiler.py         # NEW
    ├── test_graph_equivalence.py      # NEW
    └── test_bootstrap_ci.py           # NEW

docs/
├── gallery/                           # NEW: Static gallery site
│   ├── index.html
│   ├── assets/
│   └── generate_gallery.py
└── calibration-sprint/                # NEW: Revenue landing page
    ├── index.html
    └── intake-form.html

community/                             # NEW: Challenge infrastructure
├── challenges/
├── leaderboard.py
├── validate.py
└── CONTRIBUTING_CHALLENGE.md
```

---

## 13) Effort Estimates

| Phase | New code (est.) | Calendar | Critical path |
|-------|----------------|----------|---------------|
| Phase 0: Viral MVP | ~1,200 lines | Weeks 1-2 | SharePack + demo command |
| Phase 1: OperatorGraph | ~2,200 lines | Weeks 3-6 | primitives.py + compiler.py |
| Phase 2: Calibration++ | ~780 lines | Weeks 7-10 | bootstrap uncertainty |
| Phase 3: Community | ~1,300 lines | Weeks 11-14 | challenge infrastructure |
| Phase 4: Papers + Revenue | ~2,100 lines | Weeks 15+ | manuscripts |
| **Total** | **~7,580 lines** | **~18 weeks** | |

---

## 14) Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| OperatorGraph adds complexity without adoption | Paper claim weakens | Build incrementally; existing operators work without graphs |
| Video generation requires ffmpeg | SharePack incomplete | Fallback: static GIF via Pillow; video is optional |
| Community challenges get no submissions | Viral loop stalls | Seed with internal submissions; make challenges very easy |
| Bootstrap uncertainty is slow | Correction runtime balloons | Limit to K=20 bootstrap; parallelize on GPU |
| Paper reviewers want more modalities | Scope creep | Already have 26; focus on depth, not breadth |

---

## 15) Final Checklist (If You Do Only 10 Things)

1. **SharePack exporter** → one-command shareable artifact
2. **`pwm demo` command** with 3 hero presets (CASSI, CACTI, MRI)
3. **Gallery page** with 26 modality cards + reproduce buttons
4. **OperatorGraph IR** compiling 3 flagship modalities
5. **primitives.yaml + graph_templates.yaml** for 26 modalities
6. **Bootstrap uncertainty** on correction results
7. **Weekly Calibration Friday** with first 4 challenges
8. **Paper 1 (PWM)** experimental protocol + ablation scripts (SPC/CACTI/CASSI deep studies)
9. **Template PR workflow** (how to add a modality)
10. **Calibration Sprint landing page** with intake form

---

## Summary

**Foundation is strong.** Plan v3 delivered 26 modalities, 45+ solvers, 17 agents, 6 registries — all tested and passing.

**This plan adds three layers:**
1. **Viral layer** (Phase 0): SharePack + `pwm demo` + Gallery → "look what PWM did" in 60 seconds
2. **Paper layer** (Phase 1-2): OperatorGraph IR + uncertainty → universal representation claim + 3 manuscripts
3. **Revenue layer** (Phase 3-4): Community loop + calibration sprint → mindshare first, then paid engagements

**Priority order is non-negotiable:** Ship the viral unit first. Papers and revenue follow naturally from a tool people already use.
