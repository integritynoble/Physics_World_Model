# PWM Development Plan v4.1 — Viral-First + Publishable + Monetizable

**Priority order:** (1) Viral adoption → (2) Top papers → (3) Revenue
**Date:** 2026-02-09
**Baseline:** Plan v3 fully implemented — 26 modalities, 45+ solvers, 17 agents, 6 YAML registries, 16 operator correction tests, all passing.

**Paper development order:** Paper 2 (InverseNet) → Paper 3 (PWMI-CASSI) → Paper 1 (Flagship PWM)
> InverseNet creates the benchmark that Paper 3 evaluates against and Paper 1 cites as evidence.
> PWMI-CASSI proves the calibration method on a hard modality; Paper 1 subsumes it as one chapter in the full paradigm story.

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
| **InverseNet benchmark dataset** | Paper 2 | Controlled sweeps for SPC/CACTI/CASSI |
| **Bootstrap uncertainty + capture advisor** | Paper 3 | PWMI-CASSI calibration claims |
| **Recoverability tables with full provenance** | Paper 1 | Empirical, not theoretical |
| **Community infrastructure** | Viral | Weekly challenges, leaderboard, contributor pipeline |
| **Open-core boundary + landing page** | Revenue | Calibration sprint service |

---

## 1) Success Criteria

PWM is "successful" when:

1. **Viral:** A new user runs `pwm demo cassi` and gets a shareable result in <60 seconds.
2. **Viral:** The PWM Gallery has 26+ curated demos with reproduce buttons.
3. **Viral:** "Calibration Friday" runs weekly with community submissions.
4. **Paper 2 (InverseNet):** Ships as a public benchmark for **SPC/CACTI/CASSI** with parameter sweeps (photon levels, compression ratios, mismatch severities) and RunBundle-grounded reproducibility.
5. **Paper 3 (PWMI-CASSI):** Shows statistically significant, robust calibration gains (operator parameter error ↓, reconstruction quality ↑) with uncertainty bands and a practical capture protocol.
6. **Paper 1 (Flagship):** **"Physics World Model for Imaging"** has a complete, reproducible system story (OperatorGraph → feasibility → design → calibration → reconstruction) validated across 26 modalities and deeply on **SPC + CACTI + CASSI**.
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

## 8) Paper Track — Temporal Development Order

Three tightly-coupled papers sharing one codebase, one artifact standard (RunBundle), and one narrative arc.

> **Development order:** Paper 2 → Paper 3 → Paper 1.
> InverseNet (Paper 2) creates the controlled benchmark. PWMI-CASSI (Paper 3) proves calibration on the hardest modality. The Flagship (Paper 1) subsumes both as evidence for the full paradigm.

### Paper packaging rule
- **One repo**, three paper folders, one shared artifact standard:
  - `papers/inversenet/`
  - `papers/pwmi_cassi/`
  - `papers/pwm_flagship/`
- Every figure must be reproducible from `experiments/` with RunBundles.

---

### 8.1 Paper 2 (FIRST to develop): **"InverseNet: A Benchmark for Operator Mismatch, Calibration, and Reconstruction"**

**Goal:** A public, citable benchmark where the unknown is not only the scene x, but also the operator parameters θ and mismatch Δθ.
**Scope:** SPC + CACTI + CASSI only — deep, not broad.

#### What makes InverseNet publishable
- Defines a **standard task suite** where the unknowns include x (scene), θ (operator parameters), and Δθ (mismatch).
- Provides **controlled sweeps** that researchers cannot easily reproduce on their own:
  - photon levels (low / medium / high SNR),
  - compression ratios (10% / 25% / 50%),
  - mismatch severities (mild / moderate / severe) per family,
  - calibration data regimes (how much calibration is enough?).

#### Dataset structure (what gets released)
For each modality (SPC, CACTI, CASSI), each sample includes:

| Field | Description |
|-------|-------------|
| `x` | Ground-truth scene (hyperspectral cube / video / image) |
| `y` | Measurement under true operator + noise |
| `A_θ` | OperatorGraph spec + true θ |
| `Δθ`, `φ` | Mismatch parameters + noise parameters (labeled) |
| `y_cal` | Calibration captures (subset) + protocol tags |
| `runbundle/` | Full RunBundle so every sample is reproducible |

#### Tasks (leaderboard-ready)

| Task | Input | Output | Metric |
|------|-------|--------|--------|
| **T1: Operator parameter estimation** | y, A_θ_wrong | θ̂ | θ-error (RMSE, per-param) |
| **T2: Mismatch identification** | y, A_θ_wrong | family label + severity | accuracy, F1 |
| **T3: Calibration** | y, y_cal, A_θ_wrong | A_θ_corrected + uncertainty | θ-error ↓, CI coverage |
| **T4: Reconstruction under mismatch** | y, A_θ_wrong (±cal) | x̂ | PSNR, SSIM, SAM |

#### Baselines (reuse existing code)
- **Oracle vs wrong:** recon with true operator vs mismatched operator
- **Calibration:** grid search / gradient descent / UPWMI
- **Reconstruction:** GAP-TV, PnP-FISTA+DRUNet, unrolled nets (LISTA/MoDL), diffusion posterior (optional track)

#### Sweep axes (what makes tables publishable)

```
SPC:   CR ∈ {10%, 25%, 50%} × photon ∈ {1e3, 1e4, 1e5} × mismatch ∈ {gain, mask_error}
CACTI: frames ∈ {4, 8, 16} × photon ∈ {1e3, 1e4, 1e5} × mismatch ∈ {mask_shift, temporal_jitter}
CASSI: bands ∈ {8, 16, 28} × photon ∈ {1e3, 1e4, 1e5} × mismatch ∈ {disp_step, mask_shift, PSF_blur}
```

#### Deliverables
- `inversenet/` dataset package (HuggingFace or Zenodo)
- `experiments/inversenet/generate_dataset.py` — deterministic sweep generation
- `experiments/inversenet/run_baselines.py` — all baselines
- `experiments/inversenet/leaderboard.py` — scoring + ranking
- `papers/inversenet/` — LaTeX manuscript

#### Why Paper 2 must come first
- Paper 3 (PWMI-CASSI) evaluates on InverseNet CASSI splits → needs the dataset to exist.
- Paper 1 (Flagship) cites InverseNet as the evidence engine → needs the benchmark to be citable.

---

### 8.2 Paper 3 (SECOND to develop): **"PWMI-CASSI: Practical Calibration of CASSI via Physics World Model"**

**Goal:** Prove PWM solves a hard real-world pain point — CASSI calibration — with a focused, publishable method paper.

#### Contributions
1. A **CASSI OperatorGraph** that exposes learnable parameters (dispersion step, mask shift dx/dy, rotation θ, PSF blur σ, spectral response).
2. A **practical calibration protocol**: what to capture, in what order, under what constraints, and how much is enough.
3. A robust **operator correction pipeline**:
   - UPWMI beam search (derivative-free) + optional differentiable refinement,
   - bootstrap uncertainty bands (K=20 resamples),
   - "next capture" advisor when calibration is underdetermined.

#### Experiments (make them "obviously real")

| Axis | Levels | What it tests |
|------|--------|---------------|
| **Mismatch family** | dispersion-step error, mask shift/rotation, PSF blur, wavelength-dependent response | Robustness across failure modes |
| **Photon regime** | low (1e3) / medium (1e4) / high (1e5) | Noise sensitivity |
| **Compression ratio** | 8 / 16 / 28 bands | Undersampling stress |
| **Calibration budget** | 1 / 3 / 5 / 10 calibration captures | How much cal data is enough? |

#### Outcome metrics (what the tables show)

| Metric | Description |
|--------|-------------|
| θ-error ↓ | Per-parameter RMSE before vs after correction |
| PSNR/SSIM/SAM ↑ | Reconstruction quality improvement |
| CI coverage | 95% bootstrap interval covers true θ in ≥90% of trials |
| Runtime | Wall-clock seconds per correction |
| "Next capture" value | Uncertainty reduction after suggested additional capture |

#### Comparison baselines
- No calibration (wrong operator)
- Grid search (brute-force)
- Gradient descent (differentiable GAP-TV)
- UPWMI (ours)
- UPWMI + gradient refinement (ours, full pipeline)

#### Artifacts
- `pwm pwmi-cassi` end-to-end CLI command
- Evaluation scripts + RunBundles for all reported results
- `experiments/pwmi_cassi/` — all experiment scripts
- `papers/pwmi_cassi/` — LaTeX manuscript
- Optional: small real calibration dataset (even if first release is synthetic + "real-ready" protocol)

#### Why Paper 3 depends on Paper 2
- All CASSI evaluation runs on InverseNet CASSI splits (controlled mismatch + sweeps).
- Comparison baselines are the same baselines defined in InverseNet → no redundant work.

---

### 8.3 Paper 1 (THIRD to develop): **"Physics World Model for Imaging"**

**Goal:** A cross-disciplinary, high-impact "paradigm + system" paper.

#### Core claims (concrete and testable)
1. **Executable digital twin** for computational imaging within a declared **PhysicsScope**: any in-scope system compiles to an **OperatorGraph** and runs end-to-end.
2. **Pre-flight feasibility** is predictable (before expensive reconstruction) by combining:
   - photon budget + noise regime (PhotonAgent),
   - recoverability / compression feasibility (RecoverabilityAgent with empirical tables),
   - mismatch sensitivity + calibration identifiability (MismatchAgent),
   - solver/prior fit.
3. **Design + calibration + reconstruction** are a single reproducible loop:
   - design: propose hardware/code/exposure changes to hit targets,
   - calibrate: infer θ with uncertainty + "next capture" suggestions,
   - reconstruct: run solvers with full provenance.
4. **Reproducibility as a first-class contribution**: every figure and table is backed by **RunBundles** (hashes, seeds, code versions, artifacts).

#### Experiments (Depth vs Breadth: designed for Nature/Science)

**Depth (full loop, strong claims): SPC + CACTI + CASSI**
These three are the flagship modalities. For each one, PWM demonstrates the complete pipeline:

1) **Design (synthesis):** propose system variants under constraints (photons, CR, resolution, exposure, cost).
2) **Pre-flight feasibility prediction:** Photon × Recoverability × Mismatch × Solver Fit predicts success/failure bands *before* expensive runs.
3) **Calibration (θ inference):** correct at least one major mismatch family with uncertainty bands + "next capture" suggestions.
4) **Reconstruction:** reproducible reconstruction with baselines, robustness checks, and ablations.

**Breadth (minimal checklist, universality anchors): CT + Widefield (+ Holography optional)**
These support the paradigm/universality claim without scope explosion. For each breadth modality, require only:

- **Compile** to OperatorGraph
- **Serialize + reproduce** (RunBundle + hashes)
- **Adjoint check** (if linear)
- **One mismatch + one calibration improvement** (small but real)

**Universality evidence (26 templates):** all 26 modality templates compile to OperatorGraph, validate, serialize, and (where applicable) pass adjoint checks.
> The 26-suite is the broad evidence layer; the depth trio provides the hard, deep results.

#### Ablations (mandatory)
- remove PhotonAgent → feasibility predictions fail under low-photon regimes
- remove Recoverability tables → compression feasibility claims collapse
- remove mismatch priors → calibration becomes unstable / non-identifiable
- remove RunBundle discipline → reproducibility breaks

#### What reviewers will look for (build these in)
- Not just "we can reconstruct": **we can predict failure and explain bottlenecks**.
- Not just "we have agents": **the agents output verifiable quantities** and drive decisions.
- Not just "we're universal": **PhysicsScope is explicit, extensible, and tested**.

#### Why Paper 1 comes last
- Cites InverseNet (Paper 2) as its evaluation bed → benchmark must exist and be citable.
- Subsumes PWMI-CASSI (Paper 3) results as the "deep calibration" chapter → method must be validated.
- Requires the full OperatorGraph IR + breadth anchors → most infrastructure of all three papers.

---

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

## 10) Implementation Phases (Paper 2 → Paper 3 → Paper 1)

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
- Three hero demos (CASSI, CACTI, SPC) run end-to-end in <5 minutes each

---

### Phase 1 (Weeks 3-6): OperatorGraph IR + Primitives (shared foundation for all papers)

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

---

### Phase 2 (Weeks 7-10): Paper 2 — InverseNet Dataset + Benchmark

**Goal:** InverseNet benchmark generated, baselines run, paper draft complete.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| Dataset generation: SPC sweeps | `experiments/inversenet/gen_spc.py` | 300 | SPC operator + OperatorGraph |
| Dataset generation: CACTI sweeps | `experiments/inversenet/gen_cacti.py` | 300 | CACTI operator + OperatorGraph |
| Dataset generation: CASSI sweeps | `experiments/inversenet/gen_cassi.py` | 300 | CASSI operator + OperatorGraph |
| Mismatch injection (parameterized) | `experiments/inversenet/mismatch_sweep.py` | 200 | mismatch_db.yaml |
| Baseline runner (all 4 tasks) | `experiments/inversenet/run_baselines.py` | 400 | existing solvers + correction |
| Leaderboard scorer | `experiments/inversenet/leaderboard.py` | 200 | metrics_db.yaml |
| Dataset packaging (HuggingFace/Zenodo) | `experiments/inversenet/package.py` | 150 | RunBundle export |
| InverseNet manuscript draft | `papers/inversenet/` | - | all experiments |

**Exit criteria:**
- 3 modalities × 3 CR × 3 photon × 3 mismatch = 81+ RunBundles generated
- All 4 task baselines evaluated with statistical error bars
- Manuscript draft with all tables/figures reproducible from RunBundles
- Dataset downloadable as a single archive with README

---

### Phase 3 (Weeks 11-14): Paper 3 — PWMI-CASSI Calibration + Uncertainty

**Goal:** CASSI calibration method validated with uncertainty, manuscript draft complete.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| Bootstrap uncertainty for correction | `mismatch/uncertainty.py` | 200 | correction loop |
| CorrectionResult with CI | `agents/contracts.py` update | 30 | uncertainty.py |
| "Next capture" advisor | `mismatch/capture_advisor.py` | 150 | uncertainty analysis |
| CASSI mismatch family experiments | `experiments/pwmi_cassi/run_families.py` | 300 | InverseNet CASSI data |
| Calibration budget sweep (1/3/5/10) | `experiments/pwmi_cassi/cal_budget.py` | 200 | bootstrap uncertainty |
| Comparison baselines (grid/gradient/UPWMI) | `experiments/pwmi_cassi/comparisons.py` | 300 | differentiable GAP-TV |
| Statistical significance tests | `experiments/pwmi_cassi/stats.py` | 150 | all experiment results |
| PWMI-CASSI manuscript draft | `papers/pwmi_cassi/` | - | all experiments |

**Exit criteria:**
- UPWMI shows statistically significant improvement over grid search and gradient descent
- Bootstrap 95% CI covers true θ in ≥90% of trials
- "Next capture" advisor reduces uncertainty by measurable margin
- Manuscript draft with all tables/figures reproducible from RunBundles

---

### Phase 4 (Weeks 15-18): Paper 1 — Flagship PWM Experiments + Breadth Anchors

**Goal:** Full paradigm story validated; flagship manuscript draft complete.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| Depth experiments: SPC full loop | `experiments/pwm_flagship/spc_loop.py` | 300 | InverseNet SPC data |
| Depth experiments: CACTI full loop | `experiments/pwm_flagship/cacti_loop.py` | 300 | InverseNet CACTI data |
| Depth experiments: CASSI full loop | `experiments/pwm_flagship/cassi_loop.py` | 200 | PWMI-CASSI results |
| Breadth: CT compile+serialize+adjoint+cal | `experiments/pwm_flagship/breadth_ct.py` | 150 | OperatorGraph |
| Breadth: Widefield compile+serialize+adjoint+cal | `experiments/pwm_flagship/breadth_wf.py` | 150 | OperatorGraph |
| Breadth: Holography (optional) | `experiments/pwm_flagship/breadth_holo.py` | 150 | OperatorGraph |
| 26-template universality check | `experiments/pwm_flagship/universality.py` | 200 | all graph templates |
| Ablation scripts (4 ablations) | `experiments/pwm_flagship/ablations.py` | 400 | all agents |
| Recoverability provenance fields | `compression_db.yaml` update | 100 | - |
| SystemDiscernAgent | `agents/system_discern.py` | 200 | LLMClient |
| Enhanced AnalysisAgent bottleneck | `analysis/bottleneck.py` update | 100 | all agents |
| Flagship manuscript draft | `papers/pwm_flagship/` | - | all experiments |

**Exit criteria:**
- 3 flagship modalities demonstrate design → preflight → calibration → reconstruction loop
- 2-3 breadth modalities pass compile/serialize/adjoint/calibration checklist
- 26/26 templates compile to OperatorGraph
- 4 ablations show each agent/component is necessary
- Manuscript draft with all tables/figures reproducible from RunBundles

---

### Phase 5 (Weeks 19+): Community Scaling + Revenue + Manuscript Polish

**Goal:** All 3 manuscripts submission-ready; community live; first revenue.

| Task | File(s) | Est. lines | Depends on |
|------|---------|-----------|------------|
| Generate SharePacks for all 26 modalities | script | 100 | sharepack.py |
| Challenge infrastructure | `community/` | 500 | RunBundle |
| Leaderboard scoring | `community/leaderboard.py` | 200 | metrics |
| Challenge submission validator | `community/validate.py` | 150 | RunBundle schema |
| CONTRIBUTING_CHALLENGE.md | `community/` | 100 | - |
| Publish first 4 weekly challenges | `community/challenges/` | 200 | InverseNet data |
| Calibration sprint landing page | `docs/calibration-sprint/` | 300 | - |
| Intake form + example report | `docs/calibration-sprint/` | 200 | RunBundle |
| All 3 manuscripts final polish | `papers/*/` | - | reviewer prep |

**Exit criteria:**
- Gallery shows 26 SharePacks with teaser images
- Community challenge infrastructure live with first 4 challenges
- 3 manuscripts ready for submission
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
| `test_inversenet_generation.py` | InverseNet dataset completeness + RunBundle integrity | 2 |
| `test_bootstrap_ci.py` | Uncertainty bands are calibrated (coverage ≥90%) | 3 |
| `test_capture_advisor.py` | Next-capture advisor reduces uncertainty | 3 |
| `test_universality_26.py` | All 26 templates compile + validate + serialize | 4 |
| `test_ablations.py` | Each ablation produces measurable degradation | 4 |
| `test_challenge_scoring.py` | Challenge submission validation + scoring | 5 |
| `test_golden_runs.py` | Reproducibility: same seed → bit-identical output | 1 |
| `test_performance_budget.py` | Runtime budgets (demo <5min, correction <30min) | 0 |

---

## 12) File Structure (New Files Only)

```
packages/pwm_core/
├── pwm_core/
│   ├── graph/                         # NEW: OperatorGraph IR (Phase 1)
│   │   ├── __init__.py
│   │   ├── primitives.py
│   │   ├── graph_spec.py
│   │   ├── compiler.py
│   │   ├── graph_operator.py
│   │   └── introspection.py
│   ├── cli/
│   │   └── demo.py                    # NEW: pwm demo command (Phase 0)
│   ├── export/
│   │   └── sharepack.py               # NEW: SharePack exporter (Phase 0)
│   ├── mismatch/
│   │   ├── uncertainty.py             # NEW: Bootstrap uncertainty (Phase 3)
│   │   └── capture_advisor.py         # NEW: Next capture suggestion (Phase 3)
│   └── agents/
│       └── system_discern.py          # NEW: SystemDiscernAgent (Phase 4)
├── contrib/
│   ├── primitives.yaml                # NEW (Phase 1)
│   └── graph_templates.yaml           # NEW (Phase 1)
└── tests/
    ├── test_sharepack.py              # Phase 0
    ├── test_demo_command.py           # Phase 0
    ├── test_graph_adjoint.py          # Phase 1
    ├── test_graph_compiler.py         # Phase 1
    ├── test_graph_equivalence.py      # Phase 1
    ├── test_inversenet_generation.py  # Phase 2
    ├── test_bootstrap_ci.py           # Phase 3
    ├── test_capture_advisor.py        # Phase 3
    ├── test_universality_26.py        # Phase 4
    └── test_ablations.py              # Phase 4

experiments/
├── inversenet/                        # Paper 2 (Phase 2)
│   ├── gen_spc.py
│   ├── gen_cacti.py
│   ├── gen_cassi.py
│   ├── mismatch_sweep.py
│   ├── run_baselines.py
│   ├── leaderboard.py
│   └── package.py
├── pwmi_cassi/                        # Paper 3 (Phase 3)
│   ├── run_families.py
│   ├── cal_budget.py
│   ├── comparisons.py
│   └── stats.py
└── pwm_flagship/                      # Paper 1 (Phase 4)
    ├── spc_loop.py
    ├── cacti_loop.py
    ├── cassi_loop.py
    ├── breadth_ct.py
    ├── breadth_wf.py
    ├── breadth_holo.py
    ├── universality.py
    └── ablations.py

papers/
├── inversenet/                        # Paper 2 manuscript
├── pwmi_cassi/                        # Paper 3 manuscript
└── pwm_flagship/                      # Paper 1 manuscript

docs/
├── gallery/                           # Viral (Phase 0)
│   ├── index.html
│   ├── assets/
│   └── generate_gallery.py
└── calibration-sprint/                # Revenue (Phase 5)
    ├── index.html
    └── intake-form.html

community/                             # Phase 5
├── challenges/
├── leaderboard.py
├── validate.py
└── CONTRIBUTING_CHALLENGE.md
```

---

## 13) Effort Estimates

| Phase | Focus | New code (est.) | Calendar | Critical path |
|-------|-------|----------------|----------|---------------|
| Phase 0 | Viral MVP | ~1,200 lines | Weeks 1-2 | SharePack + demo command |
| Phase 1 | OperatorGraph IR | ~2,200 lines | Weeks 3-6 | primitives.py + compiler.py |
| Phase 2 | **Paper 2: InverseNet** | ~1,850 lines | Weeks 7-10 | dataset generation + baselines |
| Phase 3 | **Paper 3: PWMI-CASSI** | ~1,330 lines | Weeks 11-14 | bootstrap uncertainty + experiments |
| Phase 4 | **Paper 1: Flagship PWM** | ~2,250 lines | Weeks 15-18 | ablations + breadth anchors |
| Phase 5 | Community + Revenue | ~1,750 lines | Weeks 19+ | manuscripts + landing page |
| **Total** | | **~10,580 lines** | **~22 weeks** | |

---

## 14) Dependency Graph

```
Phase 0 (Viral)
    │
    v
Phase 1 (OperatorGraph IR)
    │
    ├──────────────────────────┐
    v                          v
Phase 2 (Paper 2: InverseNet) │
    │                          │
    v                          │
Phase 3 (Paper 3: PWMI-CASSI) │
    │                          │
    ├──────────────────────────┘
    v
Phase 4 (Paper 1: Flagship PWM)
    │
    v
Phase 5 (Community + Revenue + Manuscript Polish)
```

- Phase 2 depends on Phase 1 (InverseNet needs OperatorGraph specs for dataset samples).
- Phase 3 depends on Phase 2 (PWMI-CASSI evaluates on InverseNet CASSI splits).
- Phase 4 depends on Phase 2 + Phase 3 (Flagship cites InverseNet and subsumes PWMI-CASSI).
- Phase 5 can partially overlap with Phase 4 (community infra is independent of manuscripts).

---

## 15) Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| OperatorGraph adds complexity without adoption | Paper 1 claim weakens | Build incrementally; existing operators work without graphs |
| InverseNet dataset too large to host | Paper 2 blocked | Release "mini" splits first; full dataset on Zenodo/HuggingFace |
| Bootstrap uncertainty is slow (K=20 × correction) | Paper 3 runtime balloons | Parallelize on GPU; limit to 20 bootstrap; cache intermediates |
| CASSI calibration gains not statistically significant | Paper 3 claim weakens | Increase sample size; report effect sizes + confidence intervals |
| Paper 1 reviewers want more breadth modalities | Scope creep | Already have 26 templates; focus on depth trio + 2-3 breadth anchors |
| Video generation requires ffmpeg | SharePack incomplete | Fallback: static GIF via Pillow; video is optional |
| Community challenges get no submissions | Viral loop stalls | Seed with internal submissions; make challenges very easy |

---

## 16) Final Checklist (If You Do Only 12 Things)

1. **SharePack exporter** → one-command shareable artifact
2. **`pwm demo` command** with 3 hero presets (CASSI, CACTI, SPC)
3. **Gallery page** with 26 modality cards + reproduce buttons
4. **OperatorGraph IR** compiling all 26 modalities
5. **InverseNet dataset** for SPC/CACTI/CASSI with sweeps (Paper 2)
6. **InverseNet baselines** for all 4 tasks (Paper 2)
7. **Bootstrap uncertainty** on CASSI correction results (Paper 3)
8. **PWMI-CASSI comparison** vs grid/gradient/UPWMI (Paper 3)
9. **Flagship depth experiments** on SPC/CACTI/CASSI (Paper 1)
10. **Flagship ablations** (remove each agent → measure degradation) (Paper 1)
11. **Weekly Calibration Friday** with first 4 challenges
12. **Calibration Sprint landing page** with intake form

---

## Summary

**Foundation is strong.** Plan v3 delivered 26 modalities, 45+ solvers, 17 agents, 6 registries — all tested and passing.

**This plan adds four layers in order:**
1. **Viral layer** (Phase 0): SharePack + `pwm demo` + Gallery → "look what PWM did" in 60 seconds
2. **Benchmark layer** (Phase 1-2): OperatorGraph IR + InverseNet → controlled evidence engine (Paper 2)
3. **Method layer** (Phase 3): PWMI-CASSI + uncertainty → focused calibration proof (Paper 3)
4. **Paradigm layer** (Phase 4-5): Flagship PWM + ablations + community + revenue → the full story (Paper 1)

**Paper development order is deliberate:**
- Paper 2 (InverseNet) first — creates the benchmark that makes everything else evaluable.
- Paper 3 (PWMI-CASSI) second — proves calibration works on the hardest modality.
- Paper 1 (Flagship PWM) last — subsumes both as chapters in the universal paradigm story.

**Priority order is non-negotiable:** Ship the viral unit first. Papers and revenue follow naturally from a tool people already use.
