# PWM System-to-Solver Co-Design Benchmark (PWM-SyS)

**Author:** Chengshuai Shi
**Date:** 2026-03-07
**Version:** 0.2 (revised after internal review)
**Status:** PROPOSAL

---

## 1. Motivation

Existing computational imaging benchmarks — including PWM v1.0 — evaluate **reconstruction quality**: given measurements y, how well does algorithm A recover ground truth x? This answers a narrow question: *"Which algorithm wins on this dataset?"*

Researchers, system designers, and lab PIs face a harder question:

> **"Given my application requirements and constraints, which imaging system + solver combination delivers the most value?"**

A system achieving 40 dB PSNR but costing $2M and requiring a PhD to operate may be inferior to one achieving 32 dB at $50K with push-button operation — depending on whether the task is clinical screening or nanoscale defect characterization.

**PWM-SyS answers this question** by defining a benchmark protocol for **purpose-conditioned system-solver selection** under explicit constraints. It extends reconstruction-centric benchmarking to a system-to-solver evaluation layer for computational imaging.

### 1.1 What PWM-SyS Is Not

PWM-SyS is not a universal cross-modality ranking ("TEM is better than ultrasound"). Such rankings are scientifically meaningless because different modalities serve different physical observables. Instead, PWM-SyS evaluates **how well a system-solver pair satisfies a stated purpose**, where the purpose includes hard constraints and soft objectives.

---

## 2. Architecture: Three Layers

The benchmark is explicitly organized into three separable layers:

```
┌─────────────────────────────────────────────┐
│  Layer C: Visualization & Product           │
│  Radar charts, Pareto viewers, UI pages     │
├─────────────────────────────────────────────┤
│  Layer B: Evaluation Protocol               │
│  Benchmark tasks, feasibility gates,        │
│  constraint checking, Pareto ranking        │
├─────────────────────────────────────────────┤
│  Layer A: System Descriptor Catalog         │
│  Neutral schema for 168 imaging systems     │
│  No value judgments, just measured facts     │
└─────────────────────────────────────────────┘
```

### Layer A — System Descriptor (the catalog)

A neutral, factual schema describing each imaging system. No scores, no rankings — just measured or documented properties.

```python
@dataclass
class SystemDescriptor:
    id: str                            # e.g., "cacti"
    name: str                          # "CACTI"
    full_name: str                     # "Coded Aperture Compressive Temporal Imaging"

    # Physical chain
    source: str                        # "flash lamp", "X-ray tube", "laser 532nm"
    carrier: str                       # "Photon", "Electron", "Acoustic", "RF"
    encoding: str                      # "coded aperture", "k-space undersampling"
    optics: str                        # "4f relay + binary mask", "fan-beam geometry"
    detector: str                      # "CMOS 512x512", "PSD + ToF"
    modulation_type: str               # "spatial", "spectral", "temporal", "none"

    # Acquisition properties (measured, not scored)
    shots_per_datacube: int            # 1 for single-shot, 10000 for raster scan
    max_frame_rate_fps: float          # effective fps of full datacube
    spatial_resolution_um: float       # finest feature, micrometers
    spectral_channels: int             # 1 for grayscale, 31 for hyperspectral
    temporal_frames: int               # frames per datacube (1 for static)
    output_dimensionality: str         # "2D", "3D(x,y,t)", "3D(x,y,z)", "4D(x,y,z,t)"
    observable: str                    # "reflectivity", "fluorescence", "attenuation"

    # Solver properties (for default/recommended algorithm)
    solver_name: str                   # "EfficientSCI", "FBP", "E2E-VarNet"
    solver_type: str                   # "classical", "PnP", "deep_learning", "diffusion"
    solver_latency_s: float            # seconds per reconstruction
    solver_gpu_required: bool
    solver_training_data: str          # "none", "self-supervised", "paired_100", "paired_10K"
    solver_psnr_on_own_benchmark: float  # PSNR on PWM modality benchmark (modality-specific)
    solver_robustness_drop_pct: float  # % PSNR drop under standard perturbation suite

    # Cost & operations (documented, not scored)
    capital_cost_usd: tuple[int, int]  # (low, high) estimate
    compute_cost_per_recon_usd: float  # GPU cost per reconstruction
    calibration_frequency: str         # "daily", "weekly", "per-session", "once"
    operator_skill: str                # "untrained", "technician", "expert", "specialist"
    sample_prep: str                   # "none", "staining", "sectioning", "destructive"
    sample_contact: bool               # True for AFM, False for optical
    in_vivo_capable: bool

    # Links
    references: list[str]
    related_pwm_variants: list[str]    # ["cacti", "cup", "coded_exposure"]
    year_introduced: int
```

**Key design choice:** Layer A contains only measured/documented facts. No subjective scores. A reviewer can verify every field against published datasheets or papers.

### Layer B — Evaluation Protocol (the benchmark)

The actual benchmark. Defined as three explicit tasks (see Section 4).

### Layer C — Visualization & Product (the platform)

Interactive tools for exploring results: radar charts, Pareto frontiers, recommendation engine. Described in Section 7.

---

## 3. Task-Normalized Adequacy Scoring

### 3.1 Why Not Absolute Scales?

The v0.1 draft used absolute axis scales (e.g., "sub-angstrom = 10, > 1mm = 0"). This creates distortions:

- TEM dominates spatial resolution but is irrelevant for clinical screening
- CUP dominates temporal resolution but cannot image below surface
- A 5D datacube is not inherently better than 2D if the task only needs 2D inspection

**Absolute capability ≠ task value.** A system that exceeds the requirement by 100× scores the same as one that barely meets it — both are sufficient.

### 3.2 Task-Normalized Adequacy (TNA)

Instead of "how capable is this system?", we ask "how well does this system satisfy **this specific task's requirements**?"

For each task query Q with requirement r on dimension d, the adequacy score is:

```
TNA_d(system, task) = {
    0.0    if system.d < r_min           (hard fail — infeasible)
    5.0    if system.d ≈ r_target        (meets requirement)
    10.0   if system.d ≥ r_comfort       (exceeds with margin)
    linear interpolation otherwise
}
```

Where:
- `r_min` = hard constraint (system MUST meet this)
- `r_target` = desired specification
- `r_comfort` = comfortable margin (diminishing returns beyond)

**Example:** Task requires temporal resolution ≥ 1 Mfps.

| System | Actual fps | TNA_time |
|--------|-----------|----------|
| CACTI | 100 Mfps | 10.0 (far exceeds) |
| CUP | 10 Gfps | 10.0 (far exceeds — no bonus for overkill) |
| High-speed CMOS | 10 kfps | 0.0 (hard fail) |
| Streak camera | 1 Tfps | 10.0 (exceeds) |

Notice: CUP and streak camera score the same as CACTI despite being 100–10,000× faster, because the task only needs 1 Mfps. Overkill is not rewarded.

### 3.3 Observable Sufficiency (replaces A_info)

Instead of "5D > 4D > 3D > 2D", the question is:

> **Does the acquired measurement include the degrees of freedom needed for the task?**

```
Observable_Sufficiency(system, task) = {
    0    if system cannot observe the required physical quantity
    5    if system observes required quantity but at insufficient sampling
    8    if system meets sampling requirements
    10   if system provides redundant/complementary observables
}
```

**Example:** Task requires spectral identification of two materials.

| System | Observables | Score |
|--------|------------|-------|
| CASSI | (x, y, λ) hyperspectral | 10 — full spectral cube |
| RGB camera | (x, y, 3-channel) | 5 — partial spectral info |
| Grayscale camera | (x, y) intensity only | 0 — cannot distinguish materials |

### 3.4 Dimension Catalog

The evaluation protocol uses **8 adequacy dimensions**, evaluated per-task:

| Dimension | What it measures | Anchored to |
|-----------|-----------------|-------------|
| **D1: Acquisition Feasibility** | Can the system acquire data under task constraints (single-shot, field conditions, etc.)? | Task's acquisition requirements |
| **D2: Temporal Adequacy** | Does system frame rate / temporal resolution meet task needs? | Task's temporal spec |
| **D3: Spatial Adequacy** | Does spatial resolution suffice for task's feature size? | Task's smallest feature |
| **D4: Observable Sufficiency** | Does measurement capture the physical quantity of interest? | Task's required observables |
| **D5: Output Recovery Quality** | How well does the solver recover the target from measurements? | Modality-normalized (see §3.5) |
| **D6: Budget Feasibility** | Does total cost (capital + compute + operation) fit budget? | Task's budget ceiling |
| **D7: Deployment Burden** | Can the intended operator acquire + reconstruct reliably? | Task's operator skill ceiling |
| **D8: Sample Compatibility** | Is the measurement compatible with sample constraints? | Task's sample constraints |

### 3.5 Output Recovery Quality — Solver Utility (D5)

D5 replaces the old "A_recon" and is decomposed into **4 sub-dimensions**:

#### D5a: Primary Reconstruction Utility (PRU)

**Not raw PSNR.** Instead, modality-normalized quality using one of:

| Metric type | Used when | Example |
|------------|-----------|---------|
| Percentile rank | Modality has established benchmark | "Top 5% on PWM-CT benchmark" |
| Regret-to-best | Comparing solvers within same modality | "Within 2 dB of SOTA" |
| Task-specific utility | Detection/classification task | "AUC = 0.95 for defect detection" |
| Perceptual adequacy | Visual quality matters | "Diagnostic quality per radiologist rating" |

A 36 dB MRI reconstruction and a 36 dB hyperspectral reconstruction are NOT directly compared. Each is evaluated relative to its own modality's best-known results.

#### D5b: Solver Speed

Time from raw measurement to usable output:

| Score | Latency | Practical meaning |
|-------|---------|-------------------|
| 10 | < 1 ms | Real-time display (≥ 1000 fps pipeline) |
| 8 | 1–100 ms | Interactive (single-pass CNN on GPU) |
| 6 | 0.1–10 s | Near-real-time (unrolled network) |
| 4 | 10 s – 5 min | Batch processing (iterative PnP) |
| 2 | 5–60 min | Offline (diffusion model, 1000 steps) |
| 0 | > 1 hour | Research only (NeRF training, cryo-EM refinement) |

#### D5c: Solver Robustness

**Measured empirically, not described philosophically.**

Robustness = performance degradation under a standardized perturbation suite:

| Perturbation | Method | Severity |
|-------------|--------|----------|
| Calibration shift | Perturb forward operator parameters by ±10% | Mild / Moderate / Severe |
| PSF mismatch | Replace ideal PSF with measured PSF or shifted PSF | 3 levels |
| Noise increase | Increase measurement noise by 2×, 5×, 10× | 3 levels |
| Source drift | Intensity variation ±5%, ±15% | 2 levels |
| Spectral misalignment | Shift spectral response by ±2 nm | Where applicable |

```
Robustness_score = 1 - (avg_PSNR_drop / max_tolerable_drop)
```

Scaled to [0, 10]. A solver that loses 0.5 dB under all perturbations scores ~9. One that collapses under mild calibration shift scores ~2.

**This is PWM-native:** PWM already models mismatch parameters (PSF error, noise level, calibration uncertainty) for every modality. The perturbation suite is built from the existing mismatch parameter definitions.

#### D5d: Deployment Burden (separated from D5)

Moved to its own top-level dimension **D7**, because training data requirements are an operational/deployment concern, not a reconstruction quality attribute:

| Component | What it captures |
|-----------|-----------------|
| Training data requirements | None (analytical) → self-supervised → paired 100s → paired 10K+ |
| Compute infrastructure | CPU-only → single GPU → multi-GPU → cluster |
| Calibration/maintenance | One-time → per-session → continuous |
| Reproducibility | Deterministic → stochastic with CI → non-reproducible |

#### Identity Recovery for Direct-Readout Systems

Systems with no inverse problem (e.g., high-speed CMOS direct capture) are handled as a degenerate case:

```
D5 for direct readout:
  D5a = detector_SNR_adequacy (does sensor quality meet task needs?)
  D5b = 10 (instantaneous — identity transform)
  D5c = 10 (no model to be mismatched)
  D5d = 10 (no solver to deploy)
```

This makes direct-readout systems competitive on D5, which correctly captures that their advantage is zero reconstruction uncertainty — but they may score poorly on D1 (acquisition mode) or D4 (observable sufficiency) because they cannot do computational multiplexing.

---

## 4. Benchmark Tasks — Unified as SpecLab Mode 1

PWM-SyS defines **three benchmark tasks**, all unified under a single interactive workflow in [SpecLab](https://pwm.platformai.org/speclab) as **Mode 1: Prompt → Spec → Simulate**:

```
User Prompt (natural language requirements)
    ↓
┌──────────────────────────────┐
│ Task 1: System Retrieval     │ ← Feasibility gate + Pareto ranking
│ Task 2: Solver Recommendation│ ← Operating-point comparison
│ Task 3: Co-Design Proposal   │ ← Forward to LLM with catalog context
└──────────────────────────────┘
    ↓
System + Solver Recommendation
    ↓
Physics Simulation (existing SpecLab pipeline)
```

The user types a single prompt; the system automatically determines which task(s) apply and returns a unified response with feasibility analysis, ranked recommendations, and a "Simulate" button that bridges to the existing physics simulation pipeline.

**Implementation:** `platform/pwm_platform/services/system_recommender.py` (feasibility gate, TNA scoring, Pareto ranking) integrated into `routers/spec_chat.py`. Live at `/benchmark/system-design` and interactive in `/speclab`.

### Task 1: Constrained System Retrieval

**Question:** "Given my requirements, which existing systems are feasible and how do they rank?"

```yaml
Input:
  purpose: "Image subsurface delaminations in CFRP panel"
  hard_constraints:
    spatial_resolution_um: <= 50
    sample_contact: false
    in_vivo_capable: false     # not required
    budget_usd: <= 100000
    operator_skill: <= "technician"
    acquisition_mode: "non-destructive"
  soft_objectives:
    temporal_resolution: "faster is better"
    depth_penetration_mm: ">= 5"
    observable: "defect_contrast_map"

Output:
  ranked_list:
    - system: active_thermography
      feasibility: PASS (all constraints met)
      adequacy_scores: {D1: 9, D2: 7, D3: 6, D4: 8, D5: 7, D6: 8, D7: 9, D8: 10}
      recommended_solver: ThermoFormer
    - system: acoustic_microscopy
      feasibility: PASS
      adequacy_scores: {D1: 4, D2: 5, D3: 8, D4: 7, D5: 7, D6: 7, D7: 7, D8: 8}
      recommended_solver: AcousticFormer
    - system: industrial_ct
      feasibility: PASS
      adequacy_scores: {D1: 6, D2: 5, D3: 7, D4: 9, D5: 8, D6: 5, D7: 7, D8: 8}
      recommended_solver: FBP+DL
    - system: tem
      feasibility: FAIL (budget exceeded, specialist required)
    - system: atom_probe
      feasibility: FAIL (destructive, specialist required, budget exceeded)

Evaluation:
  - Feasibility pass rate: correct classification of feasible/infeasible
  - Rank quality: Kendall τ against expert panel ranking
  - Utility regret: gap between recommended system and expert-optimal choice
```

### Task 2: System + Solver Recommendation

**Question:** "For my application, which system + algorithm pair is optimal at my latency budget?"

```yaml
Input:
  purpose: "Capture non-repeatable transient combustion event"
  hard_constraints:
    acquisition_mode: "single-shot"
    temporal_resolution_fps: >= 10_000_000   # 10 Mfps
    spatial_resolution_um: <= 50
    budget_usd: <= 50000
    sample_contact: false
  soft_objectives:
    spectral_channels: ">= 1"    # grayscale acceptable
    reconstruction_latency: "< 60 s acceptable (offline OK)"

Output:
  recommendations:
    - rank: 1
      system: cacti
      solver: EfficientSCI
      rationale: "Single-shot 10+ Mfps, 256×256×8 datacube, offline recon ~2s on GPU"
      tradeoff: "Moderate spatial-temporal tradeoff at compression ratio B=8"
      adequacy: {D1: 10, D2: 10, D3: 7, D4: 7, D5: 7.2, D6: 8, D7: 5, D8: 10}
      solver_detail:
        D5a_pru: 7.5 (top 15% on PWM-CACTI benchmark)
        D5b_speed: 8 (2.1 s on RTX 4090)
        D5c_robustness: 6 (1.8 dB drop under mask miscalibration)

    - rank: 2
      system: cup
      solver: PnP-ADMM
      rationale: "Single-shot, 10 Gfps, but lower spatial resolution"
      tradeoff: "Higher temporal resolution than needed; spatial limited to ~100 um"
      adequacy: {D1: 10, D2: 10, D3: 5, D4: 6, D5: 5.5, D6: 6, D7: 4, D8: 10}

    - rank: 3
      system: streak_camera
      solver: FBP
      rationale: "Fastest temporal (Tfps) but 1D only — needs mechanical scanning"
      tradeoff: "Cannot do single-shot 2D; only feasible for line-scan geometry"
      adequacy: {D1: 3, D2: 10, D3: 5, D4: 3, D5: 8, D6: 4, D7: 4, D8: 8}
      note: "FAILS D1 hard constraint (not truly single-shot for 2D)"

  # Solver comparison at different latency budgets (for winning system CACTI):
  solver_operating_points:
    - latency: "< 100 ms (real-time)"
      solver: BIRNAT
      D5a: 6.5
      D5b: 9
      D5c: 5

    - latency: "< 10 s (interactive)"
      solver: EfficientSCI
      D5a: 7.5
      D5b: 8
      D5c: 6

    - latency: "< 5 min (offline)"
      solver: DiffusionSCI
      D5a: 9.0
      D5b: 3
      D5c: 7

    - latency: "unlimited (best-effort)"
      solver: DeSCI
      D5a: 5.5
      D5b: 1
      D5c: 5
      note: "Classical method; no training data; highest robustness guarantee"

Evaluation:
  - Recommendation quality: proximity to expert Pareto frontier
  - Constraint satisfaction: 0 violations of hard constraints in top-3
  - Solver selection quality: PSNR within 1 dB of best feasible solver at stated latency
```

### Task 3: Co-Design Proposal

**Question:** "If I could build a new system from scratch for this purpose, what should I design?"

```yaml
Input:
  purpose: "In-vivo retinal imaging with cellular resolution"
  hard_constraints:
    in_vivo_capable: true
    sample_contact: false
    spatial_resolution_um: <= 5
    acquisition_time_s: <= 0.1
    budget_usd: <= 500000
    safety: "ANSI Z136 compliant laser exposure"
  soft_objectives:
    depth_penetration_um: ">= 300"
    spectral_channels: ">= 1"
    volumetric: "preferred"

Output:
  proposed_design:
    source: "SLD 840 nm, 50 nm bandwidth"
    carrier: "Photon (near-IR)"
    encoding: "spectral-domain interferometry + AO correction"
    optics: "Michelson interferometer + Shack-Hartmann WFS + deformable mirror"
    detector: "InGaAs line camera, 2048 px, 70 kHz"
    solver: "AO-OCT DL reconstruction"
    estimated_performance:
      spatial_resolution: "3 um lateral, 5 um axial"
      temporal: "70,000 A-lines/s → 500x500 volume in 3.6 s"
      depth: "2 mm in retina"
    estimated_cost: "$350K"
    operator: "ophthalmic technician (after training)"

  comparison_to_existing:
    - system: oct (standard SD-OCT)
      gap: "No AO → 15 um lateral vs 3 um; misses cellular features"
    - system: adaptive_optics (standalone AO)
      gap: "No depth sectioning; surface-only imaging"
    - system: fundus (fundus photography)
      gap: "No depth; 10 um resolution; no volumetric"

Evaluation:
  - Expert review: feasibility assessment by domain specialists
  - Simulated performance: PWM forward model simulation of proposed system
  - Pareto quality: does proposed system extend or match existing Pareto frontier?
```

---

## 5. Evaluation Stages

For Tasks 1 and 2, evaluation proceeds in three stages:

```
┌──────────────────────────────┐
│  Stage 0: Feasibility Gate   │
│  Hard constraint checking    │
│  Binary: PASS / FAIL         │
│  Reject infeasible systems   │
└──────────┬───────────────────┘
           │ feasible systems only
           ▼
┌──────────────────────────────┐
│  Stage 1: Pareto Ranking     │
│  Multi-objective dominance   │
│  Identify Pareto frontier    │
│  No scalar aggregation       │
└──────────┬───────────────────┘
           │ Pareto-optimal set
           ▼
┌──────────────────────────────┐
│  Stage 2: Preference Ranking │
│  Application-weighted score  │
│  Used only as tie-breaker    │
│  within Pareto-optimal set   │
└──────────────────────────────┘
```

### Stage 0: Feasibility Gate

Every task query defines hard constraints. A system that violates **any** hard constraint is rejected before scoring:

```python
def feasibility_gate(system: SystemDescriptor, query: TaskQuery) -> bool:
    for constraint in query.hard_constraints:
        if not constraint.satisfied_by(system):
            return False  # hard fail — system is infeasible
    return True
```

Common hard constraints:
- `budget_usd <= X`
- `spatial_resolution_um <= X`
- `temporal_resolution_fps >= X`
- `acquisition_mode in ["single-shot", "few-shot"]`
- `sample_contact == False`
- `in_vivo_capable == True`
- `operator_skill <= "technician"`
- `non_destructive == True`

### Stage 1: Pareto Ranking

Among feasible systems, identify the **Pareto frontier** — systems where no other feasible system is strictly better on all dimensions simultaneously.

No scalar aggregation is needed. A system is Pareto-dominated if another feasible system scores ≥ on all 8 dimensions and strictly > on at least one.

### Stage 2: Preference Ranking (tie-breaking only)

Within the Pareto-optimal set, use application-specific weights **only as a tie-breaker**:

```
S_pref = Σ_d  w_d × TNA_d(system, task)
```

Weights are user-specified or drawn from predefined application profiles. But the key insight is: **weighted scores never override Pareto dominance.** A Pareto-dominated system cannot beat a Pareto-optimal system regardless of weights.

---

## 6. Pilot Benchmark (Phase 1 Scope)

### 6.1 Rationale for Small Initial Scope

Scoring all 168 modalities requires validated data for each. Phase 1 proves the protocol with a focused pilot before scaling.

### 6.2 Pilot Systems (12 systems)

Selected to span the space of acquisition modes, costs, and application domains:

| # | System | Domain | Why included |
|---|--------|--------|-------------|
| 1 | CACTI | Compressive temporal | Single-shot video; strong PWM benchmark data |
| 2 | CASSI | Compressive spectral | Single-shot hyperspectral; complementary to CACTI |
| 3 | SPC | Compressive spatial | Single-pixel; extreme cost-performance trade-off |
| 4 | High-speed CMOS | Direct capture | Direct readout baseline (no inverse problem) |
| 5 | CT | Medical X-ray | Gold standard 3D medical; mature solver ecosystem |
| 6 | MRI | Medical RF | Undersampled k-space; rich solver landscape |
| 7 | Ultrasound | Medical acoustic | Real-time, portable, lowest-cost clinical modality |
| 8 | OCT | Medical optical | Interferometric; depth sectioning; clinical + research |
| 9 | Confocal 3D | Microscopy | Scanning; high-resolution; established deconvolution |
| 10 | Widefield | Microscopy | Snapshot; low-cost; denoising-dominant problem |
| 11 | Active thermography | NDT | Non-contact inspection; industry-relevant |
| 12 | SAR | Remote sensing | Coherent; non-optical carrier; large-scale |

### 6.3 Pilot Application Profiles (3 profiles)

| Profile | Description | Key constraints | Key soft objectives |
|---------|-------------|----------------|---------------------|
| **Clinical screening** | Hospital deployment for patient triage | budget ≤ $500K, operator ≤ technician, in-vivo, non-contact, real-time recon | high D5a, high D8 |
| **High-speed research** | Lab capture of non-repeatable transient | single-shot, ≥ 1 Mfps, budget ≤ $100K, offline recon OK | max D2, max D4 |
| **Industrial NDT** | Factory-floor defect detection | non-destructive, operator ≤ technician, budget ≤ $50K, throughput ≥ 1 part/min | high D3, high D7 |

### 6.4 Pilot Case Study: CACTI System-Solver Selection

**Query:**
```yaml
purpose: "Capture non-repeatable transient combustion event in lab"
hard_constraints:
  acquisition_mode: "single-shot"
  temporal_resolution_fps: >= 10_000_000
  budget_usd: <= 50000
  sample_contact: false
  offline_recon_acceptable: true
  max_recon_time_s: 60
```

**Stage 0 — Feasibility Gate:**

| System | Budget | Single-shot? | ≥ 10 Mfps? | Verdict |
|--------|--------|-------------|------------|---------|
| CACTI | $15K | Yes | Yes (100M) | **PASS** |
| CUP | $80K | Yes | Yes (10G) | **FAIL** (budget) |
| High-speed CMOS | $40K | No (multi-frame) | Yes (1M only at full res) | **FAIL** (not single-shot) |
| Streak camera | $200K | No (1D line) | Yes (1T) | **FAIL** (budget, not 2D single-shot) |
| SPC | $5K | No (sequential) | No | **FAIL** (not single-shot, too slow) |

Only **CACTI passes** all hard constraints for this query.

**Stage 1 — Solver Selection at Feasible System (CACTI):**

| Solver | Year | D5a (PRU) | D5b (Speed) | D5c (Robust) | D5 combined | Meets latency? |
|--------|------|-----------|-------------|--------------|-------------|----------------|
| GAP-TV | 2014 | 4.0 (bottom 40%) | 4 (45 s) | 7 (analytical) | 4.8 | Yes |
| PnP-FFDNet | 2020 | 6.0 (top 40%) | 4 (30 s) | 6 (1.5 dB drop) | 5.4 | Yes |
| BIRNAT | 2022 | 7.0 (top 25%) | 9 (0.05 s) | 5 (2.5 dB drop) | 7.0 | Yes |
| EfficientSCI | 2023 | 7.5 (top 15%) | 8 (2.1 s) | 6 (1.8 dB drop) | 7.2 | Yes |
| DiffusionSCI | 2024 | 9.0 (top 3%) | 2 (180 s) | 7 (1.0 dB drop) | 6.2 | **FAIL** (> 60s) |

**Recommendation:** CACTI + EfficientSCI

**Explanation:** DiffusionSCI achieves highest raw quality but **fails the latency constraint** (180 s > 60 s limit). EfficientSCI offers the best feasible trade-off: top-15% quality with 2.1 s reconstruction and acceptable robustness. BIRNAT is the alternative if real-time display is needed during acquisition.

---

## 7. Cost Structure (Expanded)

Cost is not a single number. The system descriptor separates:

| Cost component | What it covers | Example |
|---------------|---------------|---------|
| **Capital cost** | Hardware purchase: source + optics + detector + housing | CACTI: $15K (CMOS + mask + stage + relay lens) |
| **Compute cost** | Per-reconstruction GPU/cloud cost | EfficientSCI: $0.002/recon (2.1s on RTX 4090) |
| **Operating cost** | Consumables, maintenance, power per year | CT: $50K/yr (tube replacement, QA, power) |
| **Calibration cost** | Time and expertise for calibration | CACTI: 2 hrs/session (mask alignment) |

Budget feasibility (D6) checks total cost of ownership over a specified time horizon:

```
TCO(T_years) = capital + T × operating + N_recons × compute + N_sessions × calibration
```

---

## 8. Operational Complexity (Expanded)

Deployment burden (D7) separates three distinct skill requirements:

| Component | Question | Scale |
|-----------|----------|-------|
| **Acquisition complexity** | How hard is it to acquire valid raw data? | 0 (specialist alignment) → 10 (press button) |
| **Calibration complexity** | How often and how expertly must the system be calibrated? | 0 (daily by specialist) → 10 (factory-calibrated, never) |
| **Solver deployment** | How hard is it to run the reconstruction and trust the output? | 0 (custom code + cluster) → 10 (built-in, deterministic) |

```
D7 = min(acquisition, calibration, solver_deployment)
```

We use **min** (not average) because the bottleneck determines real-world deployability. A system with push-button acquisition but requiring daily calibration by a PhD student is limited by the calibration step.

---

## 9. Full System Catalog — All 168 PWM Modalities

The complete catalog of Layer A descriptors for all 168 PWM imaging modalities. Organized by domain. Absolute capability values are listed (not task-normalized scores, which are computed per-query).

### 9.1 Compressive / Computational Imaging (9 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Solver latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------------|-------------|----------|
| 1 | cacti | CACTI | 1 | 100M | 5 | 3D(x,y,t) | EfficientSCI | 2.1 s | 15 | expert |
| 2 | cassi | CASSI | 1 | 30 | 5 | 3D(x,y,λ) | DGSMP | 3.5 s | 20 | expert |
| 3 | spc | Single-Pixel Camera | 1000+ | 1 | 100 | 2D | PnP-DnCNN | 0.5 s | 2 | technician |
| 4 | cup | CUP | 1 | 10G | 100 | 3D(x,y,t) | PnP-ADMM | 30 s | 80 | specialist |
| 5 | coded_exposure | Coded Exposure | 1 | 100 | 5 | 2D+blur | Wiener | 0.01 s | 3 | technician |
| 6 | lensless | Lensless Camera | 1 | 30 | 50 | 2D/3D | U-Net | 0.1 s | 0.5 | untrained |
| 7 | ghost_imaging | Ghost Imaging | 1000+ | 0.1 | 100 | 2D | DGI-Net | 0.5 s | 10 | expert |
| 8 | entangled_photon | Entangled Photon | 1000+ | 0.01 | 50 | 2D | Coincidence-CNN | 5 s | 100 | specialist |
| 9 | quantum_illumination | Quantum Illumination | 1000+ | 0.001 | 1000 | 2D | Bayesian | 1 s | 200 | specialist |

### 9.2 Optical Microscopy (21 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 10 | widefield | Widefield Fluorescence | 1 | 100 | 0.3 | 2D | CARE | 0.05 s | 30 | technician |
| 11 | widefield_lowdose | Widefield Low-Dose | 1 | 100 | 0.3 | 2D | Noise2Void | 0.05 s | 30 | technician |
| 12 | confocal_3d | Confocal 3D | 500+ | 0.5 | 0.2 | 3D(x,y,z) | RL-Deconv | 5 s | 200 | expert |
| 13 | confocal_livecell | Confocal Live-Cell | 50+ | 5 | 0.2 | 3D+t | CSBDeep | 0.5 s | 200 | expert |
| 14 | confocal_endomicroscopy | Confocal Endomicroscopy | 1 | 30 | 1 | 2D | pix2pix | 0.1 s | 100 | technician |
| 15 | spinning_disk | Spinning Disk Confocal | 100+ | 10 | 0.25 | 3D(x,y,z) | RCAN | 1 s | 250 | expert |
| 16 | two_photon | Two-Photon | 500+ | 1 | 0.3 | 3D(x,y,z) | DeepCAD | 2 s | 400 | expert |
| 17 | three_photon | Three-Photon | 500+ | 0.5 | 0.3 | 3D(x,y,z) | Self2Self | 5 s | 600 | specialist |
| 18 | lightsheet | Light-Sheet (SPIM) | 200+ | 5 | 0.4 | 3D(x,y,z) | pN2V | 2 s | 150 | expert |
| 19 | lattice_lightsheet | Lattice Light-Sheet | 200+ | 10 | 0.2 | 3D(x,y,z)+t | CARE | 2 s | 500 | specialist |
| 20 | sim | SIM | 9–15 | 10 | 0.1 | 2D/3D | DL-SIM | 0.5 s | 200 | expert |
| 21 | palm_storm | SMLM (PALM/STORM) | 10K+ | 0.001 | 0.02 | 2D/3D | DECODE | 30 s | 150 | expert |
| 22 | sted | STED | 500+ | 1 | 0.05 | 2D/3D | Richardson-Lucy | 2 s | 500 | specialist |
| 23 | minflux | MINFLUX | 10K+ | 0.001 | 0.002 | 3D | MLE | 60 s | 800 | specialist |
| 24 | dna_paint | DNA-PAINT | 50K+ | 0.0001 | 0.01 | 2D/3D | PICASSO | 120 s | 150 | specialist |
| 25 | expansion | Expansion Microscopy | 1 | 1 | 0.07 | 3D | Deconv+Reg | 10 s | 50 | expert |
| 26 | tirf | TIRF | 1 | 100 | 0.2 | 2D (surface) | eSRRF | 1 s | 100 | expert |
| 27 | ism | ISM | 100+ | 5 | 0.15 | 2D | ISM-deconv | 1 s | 200 | expert |
| 28 | flim | FLIM | 100+ | 0.1 | 0.3 | 2D+τ | Phasor-Net | 2 s | 150 | expert |
| 29 | shg | SHG | 500+ | 1 | 0.3 | 2D | DnCNN | 1 s | 300 | expert |
| 30 | srs | SRS | 500+ | 1 | 0.3 | 2D+λ | SRS-Net | 1 s | 300 | expert |

### 9.3 Coherent / Phase Imaging (12 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 31 | holography | Digital Holographic Microscopy | 1 | 100 | 0.3 | 2D+phase | PhaseGAN | 0.1 s | 50 | expert |
| 32 | phase_contrast | QPI | 1 | 100 | 0.3 | 2D+phase | QPI-Net | 0.1 s | 60 | expert |
| 33 | dic | DIC | 1 | 100 | 0.2 | 2D+gradient | DIC-Net | 0.1 s | 80 | technician |
| 34 | odt | Optical Diffraction Tomography | 50+ | 1 | 0.2 | 3D(n) | ODT-UNet | 5 s | 100 | specialist |
| 35 | phase_retrieval | Phase Retrieval | 1–10 | 30 | 0.3 | 2D+phase | prDeep | 1 s | 30 | expert |
| 36 | ptychography | Ptychography | 100+ | 0.1 | 0.01 | 2D+phase | PtychoNN | 10 s | 200 | specialist |
| 37 | fpm | Fourier Ptychographic Microscopy | 100+ | 0.5 | 0.1 | 2D+phase | FPM-INR | 5 s | 20 | expert |
| 38 | dark_field | Dark-Field | 1 | 30 | 0.5 | 2D | U-Net | 0.05 s | 50 | technician |
| 39 | polarization | Polarization Imaging | 1 | 30 | 5 | 2D+Stokes | Mueller-Net | 0.5 s | 20 | technician |
| 40 | talbot_lau | Talbot-Lau Interferometry | 3–5 | 5 | 10 | 2D(abs,dpc,df) | Phase Stepping | 0.01 s | 100 | expert |
| 41 | integral | Integral Field Spectroscopy | 1 | 10 | 10 | 3D(x,y,λ) | IFS-Recon | 2 s | 200 | specialist |
| 42 | matrix | Reflection Matrix | 100+ | 0.1 | 0.3 | 3D(x,y,z) | Matrix-SVD | 10 s | 100 | specialist |

### 9.4 Medical — X-ray & CT (10 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 43 | ct | Clinical CT | 1000+ | 0.5 | 300 | 3D(x,y,z) | DOLCE | 5 s | 1000 | technician |
| 44 | cbct | Cone-Beam CT | 300+ | 0.2 | 200 | 3D(x,y,z) | FDK+DL | 5 s | 200 | technician |
| 45 | spectral_ct | Spectral CT | 1000+ | 0.3 | 300 | 3D+E | Butterfly-Net | 10 s | 2000 | technician |
| 46 | industrial_ct | Industrial Micro-CT | 1000+ | 0.01 | 5 | 3D(x,y,z) | ASTRA-TV | 30 s | 300 | expert |
| 47 | xray_radiography | X-ray Radiography | 1 | 30 | 100 | 2D | DRR-Net | 0.05 s | 50 | technician |
| 48 | mammography | Mammography | 1 | 2 | 70 | 2D | INbreast-Net | 0.1 s | 200 | technician |
| 49 | digital_breast_tomo | Digital Breast Tomo | 15–25 | 1 | 100 | 3D | DBT-Recon | 5 s | 400 | technician |
| 50 | fluoroscopy | Fluoroscopy | 1 | 30 | 200 | 2D(t) | Temporal-CNN | 0.03 s | 300 | technician |
| 51 | angiography | X-ray Angiography / DSA | 10+ | 15 | 200 | 2D(t) | DSA-Net | 0.1 s | 500 | technician |
| 52 | dexa | DEXA | 1 | 1 | 500 | 2D(dual-E) | DEXA-Recon | 0.1 s | 50 | technician |

### 9.5 Medical — MRI (10 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 53 | MRI | Clinical MRI (3T) | 100+ | 0.1 | 500 | 3D+contrast | E2E-VarNet | 2 s | 2000 | technician |
| 54 | asl_mri | ASL MRI | 100+ | 0.05 | 2000 | 2D+perfusion | ASL-Net | 5 s | 2000 | expert |
| 55 | cest_mri | CEST MRI | 100+ | 0.02 | 1000 | 2D+Z-spec | CEST-CNN | 10 s | 2000 | specialist |
| 56 | diffusion_mri | Diffusion MRI | 100+ | 0.05 | 1500 | 3D+diffusion | q-DL | 10 s | 2000 | expert |
| 57 | fmri | Functional MRI | 100+ | 0.5 | 2000 | 3D+t(BOLD) | fMRI-DL | 2 s | 2000 | expert |
| 58 | mr_elastography | MR Elastography | 100+ | 0.1 | 2000 | 3D+stiffness | MRE-PINN | 10 s | 2000 | specialist |
| 59 | mr_fingerprinting | MR Fingerprinting | 1000+ | 0.01 | 1000 | 2D+multi-param | MRF-DL | 5 s | 2000 | specialist |
| 60 | mra | MR Angiography | 100+ | 0.1 | 500 | 3D(vessels) | CS-MRA | 5 s | 2000 | technician |
| 61 | mrs | MR Spectroscopy | 100+ | 0.01 | 10000 | 1D+chem | DL-MRS | 2 s | 2000 | specialist |
| 62 | swi | Susceptibility-Weighted Imaging | 100+ | 0.1 | 500 | 3D+suscept | QSM-Net | 5 s | 2000 | expert |

### 9.6 Medical — Ultrasound (6 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 63 | ultrasound | B-mode Ultrasound | 1 | 100 | 300 | 2D | DAS-UNet | 0.001 s | 30 | technician |
| 64 | doppler_ultrasound | Doppler Ultrasound | 1 | 50 | 300 | 2D+velocity | Power-Doppler-Net | 0.005 s | 50 | technician |
| 65 | elastography | US Elastography | 2+ | 20 | 500 | 2D+stiffness | Elas-Net | 0.1 s | 80 | technician |
| 66 | ceus | Contrast-Enhanced US | 1 | 30 | 300 | 2D+perfusion | CEUS-DL | 0.05 s | 80 | expert |
| 67 | ivus | Intravascular US | 1 | 30 | 100 | 2D(cross-section) | IVUS-Net | 0.01 s | 200 | specialist |
| 68 | us_mri | US-MRI Fusion | 1 | 10 | 500 | 3D(fused) | Reg-Fusion | 1 s | 100 | expert |

### 9.7 Medical — Nuclear (5 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 69 | pet | PET | 10M+ | 0.001 | 4000 | 3D | DeepPET | 30 s | 2000 | technician |
| 70 | pet_ct | PET/CT | 10M+ | 0.001 | 2000 | 3D+anat | MAPEM-DL | 30 s | 3000 | technician |
| 71 | pet_mr | PET/MR | 10M+ | 0.001 | 2000 | 3D+multi | ML-EM-MR | 60 s | 5000 | specialist |
| 72 | spect | SPECT | 10M+ | 0.001 | 8000 | 3D | DuDoSS | 30 s | 500 | technician |
| 73 | spect_ct | SPECT/CT | 10M+ | 0.001 | 5000 | 3D+anat | SPECT-DL | 30 s | 1000 | technician |

### 9.8 Medical — Optical (8 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 74 | oct | OCT (SD-OCT) | 1 | 100K A/s | 5 | 3D(x,y,z) | OCT-DL | 1 s | 80 | technician |
| 75 | octa | OCT Angiography | 2+ | 50K A/s | 10 | 3D(vessels) | OCTA-Net | 2 s | 100 | technician |
| 76 | fundus | Fundus Photography | 1 | 10 | 10 | 2D | RetinalNet | 0.05 s | 20 | technician |
| 77 | endoscopy | Endoscopy | 1 | 30 | 50 | 2D | Endo-Deblur | 0.03 s | 30 | technician |
| 78 | photoacoustic | Photoacoustic Tomography | 100+ | 5 | 50 | 3D | PAT-UNet | 5 s | 200 | expert |
| 79 | dot | Diffuse Optical Tomography | 100+ | 1 | 5000 | 3D | DOT-DL | 10 s | 100 | expert |
| 80 | nirs_brain | fNIRS Brain Imaging | 1 | 10 | 10000 | 2D(cortex) | GLM-HRF | 1 s | 30 | technician |
| 81 | bioluminescence_tomo | Bioluminescence Tomography | 1 | 0.1 | 1000 | 3D | BLT-PINN | 30 s | 100 | specialist |

### 9.9 Medical — Radiotherapy (4 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 82 | brachytherapy_img | Brachytherapy Imaging | 1+ | 1 | 500 | 3D | Brachy-DL | 5 s | 500 | specialist |
| 83 | portal_imaging | Portal Imaging (EPID) | 1 | 15 | 400 | 2D | EPID-Net | 0.1 s | 200 | technician |
| 84 | proton_radiography | Proton Radiography | 10K+ | 0.01 | 500 | 2D(RSP) | pCT-Recon | 30 s | 10000 | specialist |
| 85 | proton_therapy_img | Proton Therapy Imaging | 100+ | 0.1 | 1000 | 3D | Range-DL | 10 s | 10000 | specialist |
| 86 | ct_fluorescence | CT Fluorescence (XFCT) | 1000+ | 0.01 | 200 | 3D+element | XFCT-DL | 30 s | 1000 | specialist |
| 87 | magnetic_particle | Magnetic Particle Imaging | 100+ | 40 | 1000 | 3D | MPI-DL | 2 s | 500 | expert |

### 9.10 Electron Microscopy (9 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 88 | sem | SEM | 10K+ | 0.01 | 0.001 | 2D | SEM-DL | 1 s | 200 | expert |
| 89 | tem | TEM (aberr.-corrected) | 1 | 10 | 0.00005 | 2D | CTF-Correct | 5 s | 3000 | specialist |
| 90 | stem | STEM | 10K+ | 0.001 | 0.00008 | 2D | STEM-DL | 2 s | 2000 | specialist |
| 91 | cryo_em | Cryo-EM (SPA) | 100K+ | 0.0001 | 0.0003 | 3D | cryoSPARC | 3600 s | 3000 | specialist |
| 92 | cryo_et | Cryo-Electron Tomography | 60+ | 0.001 | 0.002 | 3D | IsoNet | 1800 s | 3000 | specialist |
| 93 | eels | EELS | 100+ | 0.01 | 0.0001 | 1D+spec | EELS-DL | 5 s | 2000 | specialist |
| 94 | electron_diffraction | Electron Diffraction | 1 | 1 | 0.0001 | 2D(recip) | 4D-STEM | 10 s | 2000 | specialist |
| 95 | electron_holography | Electron Holography | 1 | 1 | 0.0005 | 2D+phase | Phase-Unwrap | 5 s | 2000 | specialist |
| 96 | electron_tomography | Electron Tomography | 100+ | 0.001 | 0.001 | 3D | SIRT | 600 s | 2000 | specialist |

### 9.11 Ion / Mass Spectroscopy Imaging (5 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 97 | atom_probe | Atom Probe Tomography | 10M+ | 0.0001 | 0.0003 | 3D+chem | Bas Protocol | 300 s | 2000 | specialist |
| 98 | sims | SIMS | 10K+ | 0.01 | 0.05 | 2D+mass | SIMS-DL | 10 s | 1000 | specialist |
| 99 | maldi_msi | MALDI-MSI | 10K+ | 0.01 | 10 | 2D+mass | MALDI-Net | 10 s | 500 | specialist |
| 100 | desi | DESI Imaging | 10K+ | 0.1 | 100 | 2D+mass | DESI-DL | 5 s | 300 | expert |
| 101 | libs | LIBS | 10K+ | 10 | 50 | 2D+element | LIBS-Net | 1 s | 100 | expert |

### 9.12 Scanning Probe Microscopy (6 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 102 | afm | AFM (tapping mode) | 100K+ | 0.001 | 0.0001 | 2D(topo) | DeepAFM | 5 s | 100 | expert |
| 103 | stm | STM | 100K+ | 0.0001 | 0.00001 | 2D(LDOS) | STM-DL | 5 s | 200 | specialist |
| 104 | nsom | NSOM | 100K+ | 0.001 | 0.05 | 2D+optical | NSOM-Deconv | 5 s | 150 | specialist |
| 105 | mfm | MFM | 100K+ | 0.001 | 0.03 | 2D(magnetic) | MFM-DL | 5 s | 120 | expert |
| 106 | ebsd | EBSD | 10K+ | 0.1 | 0.05 | 2D(orient) | EBSD-DL | 5 s | 200 | expert |
| 107 | fib_sem | FIB-SEM | 1000+ | 0.0001 | 0.005 | 3D | FIB-SEM-3D | 600 s | 1000 | specialist |

### 9.13 NDT / Industrial Inspection (7 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 108 | active_thermography | Pulsed IR Thermography | 1 | 100 | 500 | 2D+t(IR) | ThermoFormer | 2 s | 30 | technician |
| 109 | acoustic_microscopy | SAM (C-scan) | 100K+ | 0.01 | 1 | 2D | AcousticFormer | 2 s | 80 | expert |
| 110 | acoustic_emission | Acoustic Emission | 1 | 1M | 5000 | 2D(source) | PINN-AE | 5 s | 10 | technician |
| 111 | eddy_current | Eddy Current | 100+ | 10 | 500 | 2D | EC-Net | 0.1 s | 15 | technician |
| 112 | xray_ndt | X-ray NDT | 1 | 10 | 100 | 2D | DR-DL | 0.1 s | 50 | technician |
| 113 | shearography | Shearography | 1 | 30 | 100 | 2D(strain) | Shear-Net | 0.5 s | 40 | technician |
| 114 | ultrasonic_phased_array | Ultrasonic Phased Array | 1 | 100 | 300 | 2D/3D | TFM | 0.5 s | 30 | technician |

### 9.14 Spectroscopy / Hyperspectral (8 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 115 | raman_imaging | Raman Microscopy | 10K+ | 0.01 | 0.5 | 2D+spec | Raman-DL | 10 s | 200 | expert |
| 116 | cars | CARS Microscopy | 1 | 30 | 0.3 | 2D+spec | CARS-Net | 0.5 s | 300 | expert |
| 117 | brillouin | Brillouin Microscopy | 10K+ | 0.01 | 0.3 | 2D+mech | Brillouin-DL | 10 s | 200 | specialist |
| 118 | ftir_imaging | FTIR Imaging | 100+ | 0.1 | 5 | 2D+IR spec | FTIR-Net | 5 s | 100 | expert |
| 119 | cathodoluminescence | Cathodoluminescence | 10K+ | 0.01 | 0.01 | 2D+spec | CL-DL | 5 s | 500 | specialist |
| 120 | edx_mapping | EDX Mapping | 10K+ | 0.01 | 0.01 | 2D+element | EDX-DL | 5 s | 300 | expert |
| 121 | xrf_imaging | XRF Imaging | 10K+ | 0.1 | 20 | 2D+element | XRF-DL | 2 s | 100 | expert |
| 122 | xrf_tomo | XRF Tomography | 100K+ | 0.001 | 50 | 3D+element | XRF-CT | 60 s | 500 | specialist |

### 9.15 Remote Sensing / Geophysical (9 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 123 | sar | SAR | 1 | 0.01 | 1e6 | 2D(complex) | SAR-DL | 10 s | 10000 | specialist |
| 124 | polsar | Polarimetric SAR | 1 | 0.01 | 1e6 | 2D+pol | PolSAR-DL | 15 s | 15000 | specialist |
| 125 | insar | InSAR | 2+ | 0.001 | 1e6 | 2D(deform) | InSAR-DL | 30 s | 15000 | specialist |
| 126 | lidar | LiDAR | 100K+ | 10 | 10000 | 3D(point cloud) | PointNet++ | 0.5 s | 50 | technician |
| 127 | flash_lidar | Flash LiDAR | 1 | 30 | 10000 | 3D(depth) | SPAD-Net | 0.05 s | 20 | untrained |
| 128 | hyperspectral_remote | Hyperspectral Remote | 1 | 0.1 | 5e5 | 2D+λ | HSI-DL | 5 s | 500 | expert |
| 129 | multispectral_sat | Multispectral Satellite | 1 | 0.01 | 1e6 | 2D+bands | SRResNet | 2 s | 5000 | specialist |
| 130 | ocean_color | Ocean Color | 1 | 0.01 | 1e7 | 2D+bands | OC-DL | 2 s | 3000 | specialist |
| 131 | passive_microwave | Passive Microwave | 1 | 0.01 | 1e8 | 2D+freq | MW-DL | 2 s | 5000 | specialist |

### 9.16 Geophysics / Subsurface (5 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 132 | fwi | Full Waveform Inversion | 100+ | 0.0001 | 1e7 | 3D(velocity) | InversionNet | 3600 s | 5000 | specialist |
| 133 | seismic_tomo | Seismic Tomography | 100+ | 0.0001 | 1e8 | 3D(velocity) | PhaseNet | 3600 s | 10000 | specialist |
| 134 | gpr | Ground-Penetrating Radar | 100+ | 1 | 1e5 | 2D/3D | GPR-DL | 5 s | 20 | technician |
| 135 | impedance_tomo | EIT | 16–256 | 10 | 5e4 | 2D | D-bar | 1 s | 10 | technician |
| 136 | muon_tomo | Muon Tomography | 10M+ | 0.0001 | 1e6 | 3D | Muon-DL | 3600 s | 500 | specialist |

### 9.17 Astronomy / Space (7 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 137 | adaptive_optics | Adaptive Optics | 1 | 1000 | 0.5 | 2D | AO-ViT | 0.001 s | 1000 | specialist |
| 138 | coronagraphy | Coronagraphy | 100+ | 0.01 | 5 | 2D | VIP | 60 s | 5000 | specialist |
| 139 | lucky_imaging | Lucky Imaging | 1000+ | 100 | 0.3 | 2D | Drizzle | 30 s | 50 | expert |
| 140 | solar_imaging | Solar Imaging | 1 | 10 | 100 | 2D/3D | Solar-DL | 5 s | 2000 | specialist |
| 141 | radio_astronomy | Radio Astronomy | 10K+ | 0.001 | 1e6 | 2D | CLEAN | 60 s | 50000 | specialist |
| 142 | radio_interferometry | Radio Interferometry (VLBI) | 10K+ | 0.0001 | 1e4 | 2D | eht-imaging | 3600 s | 100000 | specialist |
| 143 | eht_imaging | Event Horizon Telescope | 10K+ | 0.00001 | 1e4 | 2D | THEMIS | 86400 s | 500000 | specialist |

### 9.18 X-ray Scattering / Crystallography (5 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 144 | saxs | SAXS | 1 | 10 | 1e4 (nm-scale struct) | 1D/2D(recip) | SAS-DL | 2 s | 500 | expert |
| 145 | waxs | WAXS | 1 | 10 | 1e3 (A-scale struct) | 1D/2D(recip) | WAXS-DL | 2 s | 500 | expert |
| 146 | xray_crystallography | X-ray Crystallography | 100+ | 0.1 | 0.0001 | 3D(electron density) | Direct Methods | 60 s | 500 | specialist |
| 147 | xfel_sfx | XFEL Serial Crystallography | 1 | 120 | 0.0001 | 3D(electron density) | CrystFEL | 3600 s | 500000 | specialist |
| 148 | neutron_diffraction | Neutron Diffraction | 100+ | 0.01 | 0.001 | 3D(nuclear density) | FullProf | 300 s | 100000 | specialist |

### 9.19 Neutron & Particle (3 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 149 | neutron_tomo | Neutron Tomography | 100+ | 0.01 | 50 | 3D | FBP-TV | 60 s | 100000 | specialist |
| 150 | particle_calorimetry | Particle Calorimetry | 1 | 40M | 1e5 | 3D(energy) | Calo-GNN | 0.001 s | 100000 | specialist |
| 151 | gravitational_wave | Gravitational Wave | 1 | 16000 | N/A | 1D(strain) | cWB | 1 s | 1000000 | specialist |

### 9.20 Acoustic / Weather (3 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 152 | sonar | Sonar Imaging | 100+ | 1 | 1e5 | 2D/3D | MFP | 2 s | 50 | technician |
| 153 | ocean_acoustic_tomo | Ocean Acoustic Tomography | 100+ | 0.001 | 1e8 | 3D(sound speed) | OAT-DL | 300 s | 5000 | specialist |
| 154 | weather_radar | Weather Radar | 1 | 0.2 | 1e6 | 3D(reflectivity) | Dual-Pol | 0.5 s | 1000 | technician |

### 9.21 Terahertz (1 system)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 155 | terahertz | Terahertz Imaging | 100+ | 0.1 | 200 | 2D+spec | THz-DL | 5 s | 100 | expert |

### 9.22 Computational Photography / 3D (8 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 156 | nerf | Neural Radiance Fields | 50+ | 0.001 | 10 | 3D(radiance) | 3DGS | 300 s train | 1 | untrained |
| 157 | gaussian_splatting | 3D Gaussian Splatting | 50+ | 30 render | 10 | 3D(radiance) | 3DGS | 120 s train | 1 | untrained |
| 158 | light_field | Light-Field Camera | 1 | 30 | 10 | 4D(x,y,u,v) | LF-DNet | 0.5 s | 5 | untrained |
| 159 | hdr_imaging | HDR Imaging | 3+ | 10 | 5 | 2D(HDR) | HDR-CNN | 0.1 s | 1 | untrained |
| 160 | panorama | Panoramic Imaging | 10+ | 5 | 5 | 2D(360) | APAP | 1 s | 0.5 | untrained |
| 161 | photometric_stereo | Photometric Stereo | 4+ | 5 | 10 | 2D(normal) | PS-FCN | 0.5 s | 2 | technician |
| 162 | structured_light | Structured Light 3D | 5+ | 10 | 50 | 3D(depth) | SL-DL | 0.5 s | 5 | technician |
| 163 | tof_camera | Time-of-Flight Camera | 1 | 30 | 1000 | 3D(depth) | ToF-DL | 0.01 s | 2 | untrained |

### 9.23 Event / Ultrafast (4 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 164 | event_camera | Event Camera (DVS) | 1 | 1M event/s | 10 | 2D(events) | E2VID | 0.01 s | 5 | technician |
| 165 | streak_camera | Streak Camera | 1 | 1T | 50 | 1D+t | FBP | 0.01 s | 200 | specialist |
| 166 | pump_probe | Pump-Probe | 1000+ | fs resolution | 10 | 2D+t | SVD | 10 s | 300 | specialist |

### 9.24 Other (2 systems)

| # | ID | System | Shots | Max fps | Res (um) | Dims | Solver | Latency | Capital ($K) | Operator |
|---|-----|--------|-------|---------|---------|------|--------|---------|-------------|----------|
| 167 | machine_vision | Machine Vision (inspection) | 1 | 100 | 10 | 2D | YOLO | 0.01 s | 5 | untrained |
| 168 | clem | CLEM | 10K+ | 0.0001 | 0.01 | 2D(correlated) | BigWarp | 300 s | 3000 | specialist |

---

## 10. Visualization & Platform Integration (Layer C)

### 10.1 New Page: `/benchmark/system-design`

- **System catalog**: Searchable/filterable table of all 168 systems with Layer A descriptors
- **Task query builder**: User specifies hard constraints + soft objectives → system retrieval
- **Radar chart comparison**: Select 2–5 systems, show 8-dimension adequacy profiles
- **Pareto frontier viewer**: Interactive 2D projections; highlight dominated vs non-dominated
- **Solver operating-point selector**: For a given system, compare algorithms at different latency budgets

### 10.2 Per-Modality Integration

Each existing `/benchmark/{variant}` page adds a **System Context** sidebar:
- Layer A descriptor card for this modality's system
- "Compare to alternatives" link → pre-filtered system-design page
- Solver operating-point table (which algorithm at which latency?)

### 10.3 Community Contributions

- Submit new system evaluations with required evidence (paper or datasheet)
- Propose corrections to existing descriptors (with citations)
- Submit new benchmark task queries for community evaluation

---

## 11. Relation to PWM v1.0

| Aspect | PWM v1.0 (current) | PWM-SyS (proposed) |
|--------|--------------------|--------------------|
| Evaluates | Algorithm quality (PSNR/SSIM) | System+solver under task constraints |
| Scope | Single modality | Cross-modality (task-conditioned) |
| Primary metric | PSNR leaderboard | Feasibility gate + Pareto ranking |
| User | Algorithm researcher | System designer, lab PI, purchasing |
| Novelty of submission | New algorithm | New system, new task query, new co-design |

PWM-SyS does **not replace** PWM v1.0. It is an additional layer. PWM v1.0 PSNR results feed into D5a (Primary Reconstruction Utility) as modality-normalized quality inputs.

---

## 12. Roadmap

### Phase 1 (v2.0): Pilot Benchmark
- [ ] Finalize Layer A schema and validate against datasheets for 12 pilot systems
- [ ] Define perturbation suite for D5c (robustness) using existing PWM mismatch parameters
- [ ] Implement feasibility gate + Pareto ranking for Tasks 1 and 2
- [ ] Build 3 application profiles with expert-validated rankings
- [ ] Publish pilot results on 12 systems × 3 profiles
- [ ] Full CACTI case study (query → feasibility → solver selection → recommendation)

### Phase 2 (v2.1): Full Catalog
- [ ] Extend Layer A descriptors to all 168 systems with literature validation
- [ ] Add 5 more application profiles (materials science, field/portable, ultra-resolution, etc.)
- [ ] Build `/benchmark/system-design` page with interactive tools
- [ ] Task 3 (co-design) with expert panel evaluation

### Phase 3 (v2.2): Community & Automation
- [ ] Community submission portal for systems, tasks, and corrections
- [ ] Automatic D5a update from PWM v1.0 leaderboard changes
- [ ] Cost tracking with historical price curves
- [ ] System design recommendation engine

---

## 13. References

1. Llull, P. et al. "Coded aperture compressive temporal imaging." *Optics Express* 21(9):10526, 2013.
2. Wagadarikar, A. et al. "Single disperser design for coded aperture snapshot spectral imaging." *Applied Optics* 47(10):B44, 2008.
3. Duarte, M. et al. "Single-pixel imaging via compressive sampling." *IEEE Signal Processing Magazine* 25(2):83, 2008.
4. Gao, L. et al. "Single-shot compressed ultrafast photography at one hundred billion frames per second." *Nature* 516:74, 2014.
5. Lustig, M. et al. "Compressed sensing MRI." *IEEE Signal Processing Magazine* 25(2):72, 2008.
6. Yuan, X. et al. "Snapshot compressive imaging: Theory, algorithms, and applications." *IEEE TPAMI* 44(4):2191, 2021.
7. Noll, R. "Zernike polynomials and atmospheric turbulence." *JOSA* 66:207, 1976.
8. Maldague, X.P.V. *Theory and Practice of Infrared Technology for NDE*. Wiley, 2001.
9. Adler, J. & Oktem, O. "Learned primal-dual reconstruction." *IEEE TMI* 37(6):1322, 2018.
10. Ongie, G. et al. "Deep learning techniques for inverse problems in imaging." *IEEE JSTSP* 14(6):1062, 2020.

---

*PWM-SyS extends reconstruction-centric benchmarking to purpose-conditioned system-solver evaluation — a standardized framework for answering "which system should I build?" rather than "which algorithm wins on this dataset?"*
