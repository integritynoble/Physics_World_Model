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

**Depth vs Breadth rule (for Paper 1):** Deep results on SPC/CACTI/CASSI; breadth-only checklist on CT + Widefield (+ Holography optional).

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

#### Experiments (aligned to your strength: SPC + CACTI + CASSI)
#### Experiments (Depth vs Breadth: designed for Nature/Science)

**Depth (full loop, strong claims): SPC + CACTI + CASSI**  
These three are your *flagship* modalities. For each one, PWM must demonstrate the complete pipeline:

1) **Design** (synthesis): propose system variants under constraints (photons, CR, resolution, exposure).  
2) **Pre-flight feasibility prediction**: Photon × Recoverability × Mismatch × Solver Fit predicts success/failure bands.  
3) **Calibration** (θ inference): correct at least one major mismatch family with uncertainty bands.  
4) **Reconstruction**: reproducible recon results with baselines and ablations.  

**Breadth (minimal checklist, universality anchors): CT + Widefield (+ Holography optional)**  
These are included to support the paradigm/universality claim without scope explosion. For each breadth modality, require only:

- **Compile** to OperatorGraph  
- **Serialize + reproduce** (RunBundle + hashes)  
- **Adjoint check** (if linear)  
- **One mismatch + one calibration improvement** (small but real)  

**Universality proof (26 templates):** all 26 modality templates must compile to OperatorGraph, validate, serialize, and (where applicable) pass adjoint checks.  
> The 26-suite is the *evidence layer* that PWM is a general system, while the depth trio provides the *hard, deep* results.

#### Ablations (mandatory)
- remove PhotonAgent → feasibility predictions fail under low-photon regimes  
- remove Recoverability tables → compression feasibility claims collapse  
- remove mismatch priors → calibration becomes unstable / non-identifiable  
- remove RunBundle discipline → reproducibility breaks  


---

### Paper 2