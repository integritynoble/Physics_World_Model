# PWM — Physics World Model for Imaging System Autonomy

**Live Platform: [pwm.platformai.org](https://pwm.platformai.org)**

[![Discussions](https://img.shields.io/github/discussions/integritynoble/Physics_World_Model?label=Discussions&logo=github)](https://github.com/integritynoble/Physics_World_Model/discussions)
[![Good First Issues](https://img.shields.io/github/issues/integritynoble/Physics_World_Model/good%20first%20issue?label=Good%20First%20Issues&color=7057ff)](https://github.com/integritynoble/Physics_World_Model/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
[![Contributing](https://img.shields.io/badge/Contributing-Guide-blue)](CONTRIBUTING.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/integritynoble/Physics_World_Model/blob/master/examples/PWM_Quickstart.ipynb)

PWM is the **evaluation harness + current best methods** for computational imaging -- an open, reproducible toolkit that aims to make any imaging system **self-specifying, self-diagnosing, and self-correcting**.

It turns **either**:
- a **natural-language prompt** ("SIM live-cell, low dose, 9 frames...") **or**
- a structured **ExperimentSpec JSON/YAML** **or**
- **measured data** `y` + an imperfect operator/matrix `A` (**operator-correction mode**)

into a fully reproducible run:

**Prompt/Spec -> OperatorGraph -> (Sim or Load) y -> (Fit operator theta) -> Reconstruct x_hat -> TriadReport -> RunBundle**

PWM is designed to be:
- **Public and extensible** (plugins, CasePacks, dataset adapters)
- **Deterministic by default** (bounded search, reproducible seeds)
- **Audit-grade** (every run produces a RunBundle with TriadReport, decision records, and uncertainty estimates)
- **Embeddable** into agent systems like **AI_Scientist** (via `pwm_AI_Scientist`)
- **Includes LIP-Arena** (Live Imaging Physics Arena) -- PWM's built-in prospective, blinded, adversarial evaluation harness; run `pwm evaluate` locally (see `docs/targeting_system.md`)

---

## Table of Contents

**Core Concepts**
- [PWM = Harness + Best Methods](#pwm--harness--best-methods) — the rail vs the trains
- [The Rails: SolveEverything Implementation](#the-rails-solveeverything-implementation) — 10-gear framework + LIP-Arena targeting system
- [Physics Fidelity Ladder](#physics-fidelity-ladder) — 4-tier operator hierarchy
- [Theoretical Foundations](#theoretical-foundations-flagship-paper) — FPB Theorem (10 canonical primitives), Triad Decomposition, Extension Protocol
- [4-Scenario Evaluation Protocol](#4-scenario-evaluation-protocol) — how methods are scored

**Getting Started**
- [Install](#install)
- [Quickstart](#quickstart)

**Capabilities (detailed)**
- [What PWM can do](#what-pwm-can-do-harness--current-best-methods) — two modes, four tracks, 64 modalities
- [The ExperimentSpec model](#the-experimentspec-model) — 8 state groups
- [Operator correction mode](#operator-correction-mode-measured-y--a---fitcorrect-operator---reconstruct) — calibration pipeline
- [DeepInv integration](#deepinv-integration)

**Data & Benchmarks**
- [Modality Coverage](#modality-coverage) — 64-modality catalog
- [Benchmark Results](#benchmark-results-26-modalities-with-psnr-table) — PSNR/SSIM tables
- [Datasets & Model Weights](DATA.md) — **all heavy data is stored in GCS** (`gs://pwm-benchmark-datasets/`); see DATA.md for download instructions

**Clinical Medical Physics**
- [CT QC Copilot](#ct-qc-copilot) — metric-first QA for diagnostic CT
- [Clinical Architecture](#clinical-architecture) — CasePack-driven, threshold-resolved, audit-grade

**Community**
- [Community & Contributing](#community--contributing) — 4 contribution levels, weekly challenges, calibration sprints
- [Documentation Index](#documentation-index)

**Reference**
- [Repository layout](#repository-layout)
- [YAML Registries](#yaml-registries)
- [Embedding into AI_Scientist](#embedding-into-ai_scientist)
- [License](#license) / [Citation](#citation)

---

## PWM = Harness + Best Methods

One repo, one install -- you get both the evaluation infrastructure and the methods that currently win on it.

| Component | Role | Durability |
|-----------|------|------------|
| **Harness** (OperatorGraph IR, 10 canonical primitives, Triad Decomposition, 4-scenario protocol, LIP-Arena) | Defines *how* methods are tested | Durable -- the railroad |
| **Current best methods** (GAP-TV, MST-L, Alg 1/2, HDNet, EfficientSCI, ...) | The methods that currently score highest | Replaceable -- the trains |

**Evaluate any method on the harness:**

```bash
# Score a method against a modality
pwm evaluate --method my_solver --modality cassi --track correct

# Compare two methods head-to-head
pwm evaluate --method my_solver --method mst_l --modality cassi

# Run the full 4-scenario protocol
pwm evaluate --method my_solver --modality cassi --scenarios I,II,III,IV
```

**Submit a better method:** Implement the `ReconSolver` protocol, register in `contrib/solver_registry.yaml`, beat the current default on the harness, and open a PR. See `docs/targeting_system.md` for details.

---

## The Rails: SolveEverything Implementation

PWM is the first repository that implements all 10 gears of the [SolveEverything.org](https://solveeverything.org/) abundance engine as a concrete, runnable reference for **computational imaging**: 64 modalities, 99 implementation primitives mapped to 10 canonical types, 43+ solvers, a built-in adversarial evaluation harness (LIP-Arena), and a complete audit trail (RunBundle + DR-IS). See [`rails/`](rails/) for the complete mapping:

| Gear | Status | PWM Implementation |
|------|--------|--------------------|
| 1. [Targeting System](rails/gear01_targeting_system.md) | **BUILT** | LIP-Arena, 4-scenario protocol, `pwm evaluate` |
| 2. [Outcome Contracts](rails/gear02_outcome_contracts.md) | **PARTIAL** | Recovery ratio, oracle gap, RoIC metrics |
| 3. [Compute Escrow](rails/gear03_compute_escrow.md) | **PARTIAL** | BudgetState, 2x enforcement |
| 4. [Action Networks](rails/gear04_action_networks.md) | **PLANNED** | Software actuation for 16 modalities |
| 5. [Data Trusts](rails/gear05_data_trusts.md) | **FOUNDATION** | Dataset registry, synthetic-first |
| 6. [Decision Logs](rails/gear06_decision_logs.md) | **BUILT** | DR-IS, RunBundle v0.3.0, SHA-256 |
| 7. [Two-Source Rule](rails/gear07_two_source_rule.md) | **PARTIAL** | Multi-solver portfolio, safety brakes |
| 8. [Compute + Energy](rails/gear08_compute_energy.md) | **OUT OF SCOPE** | RoIC makes compute measurable |
| 9. [Fairness Targets](rails/gear09_fairness_targets.md) | **PARTIAL** | Tail-risk scoring, anti-Goodhart |
| 10. [Literacy](rails/gear10_literacy.md) | **PARTIAL** | 26 working-process docs, quickstart |

Also in `rails/`: [maturity levels (L0-L5)](rails/maturity_levels.md) and [industrial stack (9 layers)](rails/industrial_stack.md).

### Gear 1: The Targeting System (LIP-Arena)

The targeting system is the foundation gear -- everything else depends on it. PWM's implementation is **LIP-Arena** (Live Imaging Physics Arena), a built-in adversarial evaluation harness that ships with PWM itself. It is not a separate benchmark or a static dataset.

**Core protocol -- Commit-Measure-Score:**

1. **Commit**: Teams submit containerized pipelines + declared compute budgets before the measurement deadline.
2. **Measure**: New measurement sets generated *after* the commit deadline (sealed-simulator + live-lab).
3. **Execute**: All submissions run in a sealed environment with no ground truth access.
4. **Score**: Fully automated scoring; all RunBundles and methodologies published.

**4 Evaluation Tracks:**

| Track | Goal | Key Metric |
|-------|------|------------|
| Track 1: Correct | Infer and correct operator mismatch | Recovery ratio $\rho$ |
| Track 2: Diagnose | Attribute failure to Triad gate | Gate attribution accuracy |
| Track 3: No-GT | Correct without ground truth | Self-consistency + invariants |
| Track 4: Design | Specify robust imaging systems | Constraint satisfaction + robustness |

**Anti-Goodhart scoring:** Prospective score dominates (70% weight). Gaming is penalized: wrong diagnosis, overconfident uncertainty, or missing artifacts result in rank loss.

**Red Team module:** Dedicated adversarial layer injecting novel mismatch types, compound failures, out-of-family physics, gate-flip scenarios, and compute traps every round.

**Safety brakes:** 5 pre-committed thresholds ($\rho$ < 0.30, uncertainty miscalibration, compute excess, etc.) that automatically block deployment.

**Current maturity:** PWM is transitioning from **L1 (Measurable)** to **L2 (Repeatable)**. See [`rails/maturity_levels.md`](rails/maturity_levels.md) for the full L0-L5 framework.

See [`rails/gear01_targeting_system.md`](rails/gear01_targeting_system.md) for the full targeting system specification and [`docs/targeting_system.md`](docs/targeting_system.md) for the LIP-Arena protocol.

---

## Physics Fidelity Ladder

PWM is not "one physics model per modality."
Every modality is compiled into a canonical **OperatorGraph**, and **each node can run at a different physics tier** depending on budget and accuracy needs. The same four-tier ladder applies across all physical carriers — photons, electrons, spins, acoustic waves, and particles.

| Tier | Code label | Physics regime | Carriers & examples |
|------|-----------|----------------|---------------------|
| 0 | `tier0_geometry` | **Ray / ballistic** — geometric optics, projection, coordinate transforms | Photon rays (CT, X-ray), electron beam geometry (SEM/TEM), acoustic ray tracing, scan trajectories |
| 1 | `tier1_approx` | **Wave / field approximations** — Fourier optics, paraxial propagation, linearized transport | Fresnel / angular spectrum (photon), Bloch equations (spin/MRI), Born approximation (acoustic/DOT), paraxial electron optics |
| 2 | `tier2_full` | **Full transport / scattering** — Maxwell, wave equation, Monte Carlo, quantum corrections | Full-wave EM (photon), electron–matter scattering (EELS, diffraction), acoustic FWI, spin dynamics (diffusion MRI), particle Monte Carlo (neutron/muon) |
| 3 | `tier3_learned` | **Learned surrogates with uncertainty** — neural operators trained to emulate Tier 2 | NeRF / 3DGS (photon), learned scattering kernels, diffusion priors; must provide calibrated error bars |

**Rule:** Tier is selected **per node**, not globally. This keeps PWM universal while allowing realistic accuracy when needed.

**How it maps to code:**
Every primitive carries a `_physics_tier` class attribute (e.g. `FresnelProp._physics_tier = "tier1_approx"`).
The graph compiler copies this into `NodeSpec.tags["physics_tier"]` so the runner can enforce a `TierPolicy` — selecting the cheapest tier that meets the requested accuracy and compute budget.
See `pwm_core/graph/tier_policy.py` and `tests/test_tier_policy.py`.

**Tie to execution modes:**
- **Mode S** (simulate) and **Mode I** (infer) default to Tier 0/1 for fast turnaround.
- **Mode C** (calibrate) starts at Tier 0/1, then validates the corrected operator at a higher tier when budget allows.

---

## Theoretical Foundations (Flagship Paper)

PWM's theoretical core is established in the flagship paper: *"Ten Primitives and Three Gates: The Universal Structure of Computational Imaging"* (Yang & Yuan, 2026). Two main results underpin the entire framework.

### Finite Primitive Basis Theorem

**Theorem (FPB).** Every imaging forward model in the Tier-2 operator class admits an $\varepsilon$-approximate representation as a typed DAG over exactly **10 canonical primitives**:

| # | Primitive | Notation | Physical action | Physics-stage family |
|---|-----------|----------|-----------------|---------------------|
| 1 | Propagate | $P(d,\lambda)$ | Free-space wave propagation | Propagation |
| 2 | Modulate | $M(\mathbf{m})$ | Element-wise multiplication (mask, coil, absorption) | Interaction |
| 3 | Project | $\Pi(\theta)$ | Radon line-integral projection | Encoding-Projection |
| 4 | Encode | $F(\mathbf{k})$ | Fourier-domain encoding ($k$-space) | Encoding-Projection |
| 5 | Convolve | $C(\mathbf{h})$ | Spatial convolution (PSF) | Propagation |
| 6 | Accumulate | $\Sigma$ | Summation over spectral/temporal axis | Detection-Readout |
| 7 | Detect | $D(g,\eta)$ | Detector response (5 canonical families) | Detection-Readout |
| 8 | Sample | $S(\Omega)$ | Sub-sampling on index set $\Omega$ | Detection-Readout |
| 9 | Disperse | $W(\alpha,a)$ | Wavelength-dependent spatial shift | Detection-Readout |
| 10 | Scatter | $R(\sigma,\Delta\varepsilon)$ | Direction change and/or energy shift | Interaction |

The 10 primitives are organized into **4 physics-stage families**: Propagation → {P, C}; Interaction → {M, R}; Encoding-Projection → {Π, F}; Detection-Readout → {Σ, S, W, D}. The Detect nonlinearity is restricted to **5 canonical families**: linear-intensity, logarithmic, sigmoid, Poisson-rate, and coherent-field.

**Basis-growth saturation.** Plotting distinct primitives $K$ vs. registered modalities $N$ reveals clear saturation: $K = 10$ at $N = 31$, with no new primitive required for the most recent 19 modalities. New modalities compose existing primitives rather than requiring new ones.

**Implementation.** The 10 canonical types are implemented as `CanonicalPrimitive` enums in `pwm_core/graph/ir_types.py`, with all 99 implementation primitives mapped to their canonical type via `CANONICAL_REGISTRY` in `pwm_core/graph/primitives.py`. The 31-modality canonical decomposition registry is in `pwm_core/graph/canonical_decompositions.py`.

### Triad Decomposition

Every reconstruction failure decomposes into three root causes (gates):

| Gate | Name | Physical origin |
|------|------|-----------------|
| Gate 1 | Recoverability | Null-space loss — measurement encodes insufficient information |
| Gate 2 | Carrier Budget | SNR floor — photon/electron/spin/acoustic noise dominates |
| Gate 3 | Operator Mismatch | $H_{\text{nom}} \neq H_{\text{true}}$ — solver targets the wrong inverse problem |

**Key finding: Gate 3 dominates** across all validated modalities. In CASSI, a sub-pixel mask shift degrades MST-L by 13.98 dB; in MRI, a 5% coil mismatch produces 6.94 dB degradation. Autonomous correction recovers +0.8 to +10.7 dB without retraining the solver.

The two results are complementary: the FPB provides a universal representation (every forward model is a DAG over 10 primitives); the Triad provides a universal diagnostic law over that representation. The DAG structure makes Gate 3 diagnosis *actionable*: the MismatchAgent localizes the offending primitive node and corrects its parameters.

### Extension Protocol

A new primitive is warranted only when no DAG over the existing 10 achieves $\varepsilon_{\text{tier2}} \leq \varepsilon$. The formal 5-step process requires: (1) validated forward/adjoint, (2) demonstrated representation gap, (3) error reduction below $\varepsilon$, (4) need by ≥2 modalities, (5) backward-compatible closure re-test. See `pwm_core/graph/extension_protocol.py`.

---

## 4-Scenario Evaluation Protocol

Every validated modality is tested under 4 scenarios that isolate the effect of operator mismatch:

| Scenario | Measurement | Reconstruction Operator | Purpose |
|----------|-------------|------------------------|---------|
| I (Ideal) | True H | True H | Oracle upper bound |
| II (Assumed) | True H | Nominal H_nom | Mismatch impact baseline |
| III (Corrected) | True H | Calibrated H_hat | Calibration benefit |
| IV (Oracle Mask) | True H | Partial oracle | Partial upper bound |

**Key metric:** Recovery ratio $\rho$ = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II) — how much of the oracle gap does calibration close?

See [`docs/targeting_system.md`](docs/targeting_system.md) for the full LIP-Arena specification and scoring details.

---

## Install

### Requirements
- Python 3.10+ recommended
- PyTorch (CPU or CUDA)
- Optional: `deepinv`, `streamlit`, `opencv-python`, `scikit-image`

### Workspace install (editable)

```bash
pip install -U pip
pip install -e packages/pwm_core
pip install -e packages/pwm_AI_Scientist
```

If you want the viewer:

```bash
pip install -e "packages/pwm_core[viewer]"
```

> Tip: If you use CUDA, install PyTorch first using the official selector for your CUDA version.

---

## Quickstart

### A) Prompt -> auto CasePack -> simulate -> reconstruct -> analyze -> view

```bash
# Microscopy examples
pwm run --prompt "widefield deconvolution, low dose, PSF mismatch"
pwm run --prompt "SIM structured illumination, 3 angles, 3 phases, live cell"
pwm run --prompt "confocal 3D stack, depth attenuation, z-drift"

# Compressive imaging examples
pwm run --prompt "CASSI spectral imaging, 28 bands, coded aperture"
pwm run --prompt "single pixel camera, 25% sampling, Hadamard patterns"
pwm run --prompt "CACTI video, 8 frames compressed to 1 snapshot"

# Medical imaging examples
pwm run --prompt "CT sparse view, 90 angles, low dose"
pwm run --prompt "MRI accelerated, 4x undersampling, parallel imaging"

# New modalities
pwm run --prompt "OCT retinal scan, dispersion compensation"
pwm run --prompt "light field microscopy, 5x5 lenslet array"
pwm run --prompt "photoacoustic imaging, circular transducer array"
pwm run --prompt "FLIM, two-component decay, IRF deconvolution"
pwm run --prompt "FPM, LED array, synthetic aperture"

# Neural rendering examples
pwm run --prompt "NeRF from 30 views, synthetic scene"
pwm run --prompt "3D Gaussian splatting, multi-view reconstruction"

# View results
pwm view runs/latest
```

PWM will:
1) select a CasePack from the 64 validated modalities,
2) compile a draft spec,
3) validate/repair,
4) simulate measurement `y`,
5) reconstruct `x_hat` using solver portfolio,
6) diagnose failure modes,
7) export a RunBundle.

### B) Spec -> run

```bash
# Run with custom spec file
pwm run --spec my_experiment.json
pwm view runs/latest
```

### C) Python API

```python
from pwm_core.api import endpoints

# Option 1: Run from prompt (auto-selects casepack from 64 modalities)
result = endpoints.run(prompt="widefield deconvolution, low dose")
print(f"RunBundle: {result['runbundle_path']}")
print(f"Verdict: {result['diagnosis']['verdict']}")
print(f"PSNR: {result['recon'][0]['metrics'].get('psnr', 'N/A')}")

# Option 2: Run from spec dict
spec = {
    "id": "my_cassi_experiment",
    "input": {"mode": "simulate"},
    "states": {
        "physics": {"modality": "cassi"},
        "budget": {"measurement_budget": {"num_bands": 28}}
    }
}
result = endpoints.run(spec=spec, out_dir="runs/")

# Option 3: Compile prompt first, inspect casepack, then run
compile_result = endpoints.compile_prompt("MRI accelerated imaging")
print(f"Selected casepack: {compile_result.casepack_id}")
print(f"Modality: {compile_result.draft_spec['states']['physics']['modality']}")

# Run with the compiled spec
result = endpoints.run(spec=compile_result.draft_spec, out_dir="runs/")
```

### D) Run benchmarks directly

```bash
# Navigate to project directory
cd packages/pwm_core

# Run ALL 64 modalities (~28 min)
python benchmarks/run_all.py --all

# Run specific modality
python benchmarks/run_all.py --modality mri
python benchmarks/run_all.py --modality oct
python benchmarks/run_all.py --modality flim
python benchmarks/run_all.py --modality photoacoustic

# Run operator correction tests (16 tests, ~63 min)
python -m pytest benchmarks/test_operator_correction.py -v

# Run unit tests (3985 tests)
python -m pytest tests/ -v         # 3743 core + 32 canonical tests
python -m pytest tests/clinical/ -v  # 210 clinical tests
```

### E) Evaluate a method on the harness

```bash
# Score a method against the built-in harness
pwm evaluate --method my_solver --modality cassi --track correct

# Run the full 4-scenario protocol
pwm evaluate --method my_solver --modality cassi --scenarios I,II,III,IV

# Compare your method against the current default
pwm evaluate --method my_solver --method mst_l --modality cassi
```

---

## What PWM can do (harness + current best methods)

PWM operates in two modes. Each mode hosts two built-in evaluation tracks (see `docs/targeting_system.md`):

| Mode | Input | Evaluation Tracks (built-in) | ISA Capability |
|------|-------|-----------------|----------------|
| **1. Prompt-driven simulation + reconstruction** | Natural-language prompt or ExperimentSpec | **Track 4 (Design):** requirements -> robust OperatorGraph | Self-specify |
| | | **Track 2 (Diagnose):** Triad gate attribution under shift | Self-diagnose |
| **2. Operator correction** | Measured `y` + operator/matrix `A` | **Track 1 (Correct):** infer $\hat{H}$, correct mismatch, reconstruct | Self-correct |
| | | **Track 3 (No-GT):** correct without ground truth via consistency + invariants | Self-correct (blind) |

Every run in both modes produces the four mandatory ISA artifacts:
- **Reconstruction** $\hat{x}$
- **Operator estimate** $\hat{\theta} \pm \sigma_\theta$ with identifiability flags
- **TriadReport** (dominant gate attribution + evidence + recommended action)
- **RunBundle** (full audit trail with DR-IS decision records)

---

### 1) Prompt-driven simulation + reconstruction (Harness Tracks 2 + 4)

**Track 4 (Design):** Given requirements (prompt or spec), PWM selects the optimal modality, compiles an OperatorGraph, and predicts performance bounds -- scored on constraint satisfaction, Pareto efficiency, robustness margin, and calibration cost.

**Track 2 (Diagnose):** After reconstruction, PWM attributes failure to the dominant Triad gate (sampling, noise, or operator mismatch) -- scored on attribution accuracy, evidence quality, and robustness under gate-flip scenarios.

PWM supports **64 validated imaging modalities** with prompt-driven workflows:

**Microscopy:**
- `widefield` - Richardson-Lucy deconvolution (27.31 dB)
- `widefield_lowdose` - BM3D+RL for low photon counts (32.88 dB)
- `confocal_livecell` - Live-cell confocal with CARE (30.04 dB)
- `confocal_3d` - 3D stack with CARE 3D (39.17 dB)
- `sim` - Structured Illumination Microscopy, 2x resolution (27.48 dB)
- `lightsheet` - Stripe artifact removal (28.05 dB)

**Compressive Imaging:**
- `spc` - Single-Pixel Camera with PnP-FISTA (32.17 dB @ 25%)
- `cassi` - Hyperspectral imaging, 4 solvers: HDNet (35.06 dB), MST-L (34.99 dB), MST-S (34.09 dB), GAP-TV (14.92 dB)
- `cacti` - Video snapshot compressive imaging, 4 solvers: EfficientSCI (36.28 dB), ELP (33.94 dB), PnP-FFDNet (29.36 dB), GAP-TV (26.62 dB)
- `lensless` - DiffuserCam with FlatNet (33.89 dB)

**Medical Imaging:**
- `ct` - Computed Tomography with RED-CNN (26.77 dB)
- `mri` - MRI with PnP-ADMM (44.97 dB)

**Coherent Imaging:**
- `ptychography` - Phase retrieval with Neural Network (59.41 dB)
- `holography` - Off-axis holography with Angular Spectrum (46.54 dB)
- `phase_retrieval` - CDI with Hybrid Input-Output (30.66 dB)
- `fpm` - Fourier Ptychographic Microscopy with Gradient Descent (34.61 dB)

**Optical Imaging:**
- `oct` - Optical Coherence Tomography with FFT (64.84 dB)
- `light_field` - Light Field with LFBM5D (35.28 dB)
- `integral` - Integral Imaging with DIBR (28.14 dB)
- `flim` - Fluorescence Lifetime with MLE Fit (48.11 dB)

**Diffuse / Acoustic Imaging:**
- `dot` - Diffuse Optical Tomography with Born/Tikhonov (32.06 dB)
- `photoacoustic` - Photoacoustic with Time Reversal (50.54 dB)

**Neural Rendering:**
- `nerf` - Neural Radiance Fields with SIREN (61.35 dB)
- `gaussian_splatting` - 3D Gaussian Splatting (30.89 dB)

**General:**
- `matrix` - Generic linear inverse problem with FISTA-TV (33.86 dB)
- `panorama_multifocal` - Multi-view focus fusion with Neural Network (27.90 dB)

Each modality includes:
- **Forward model simulation** with dose/compression/mismatch/sensor pipeline
- **Solver portfolio** (classical + PnP + neural methods)
- **Diagnosis + actionable recommendations**

### 2) Operator correction mode (measured `y` + operator/matrix `A`) (Harness Tracks 1 + 3)

**Track 1 (Correct):** Given measurement `y` and nominal operator `H_nom` with unknown mismatch, PWM infers the effective operator, corrects mismatch parameters, and reconstructs -- scored on recovery ratio, parameter recovery, uncertainty calibration, tail-risk performance, and compute efficiency (RoIC).

**Track 3 (No-GT):** When ground truth is unavailable (the realistic deployment case), PWM corrects using self-consistency (re-projection error), physical invariants (energy conservation, spectral smoothness), and held-out measurement channels -- ensuring correction works in real laboratories, not just simulations.

For real experiments where the forward model is imperfect, PWM can:
- **fit/correct** forward-model parameters (theta) with a bounded calibration loop
- reconstruct with the corrected operator
- output a **TriadReport** attributing failure to sampling, noise, or operator mismatch
- export a reproducible **RunBundle** including calibration trajectory, DR-IS decision records, and uncertainty estimates

**16 modalities support operator correction**, all verified with >0.5 dB improvement:

| Modality | Mismatch Parameter | Calibration Method |
|----------|--------------------|--------------------|
| Matrix/SPC | gain/bias | Cross-validation grid search |
| CT | center of rotation | Reprojection error |
| CACTI | temporal shift | Reprojection error |
| Lensless | PSF shift | Reprojection error |
| MRI | k-space mask | ACS-based estimation |
| SPC | gain/bias | Reprojection error |
| CASSI | dx, dy, theta, phi_d | UPWMI beam search (+4.8 dB) |
| Ptychography | probe position | Sharpness metric |
| OCT | dispersion coefficients | Reprojection error |
| Light Field | disparity | Sharpness metric |
| DOT | scattering coefficient | Regularized least-squares |
| Photoacoustic | speed of sound | FBP sharpness |
| FLIM | IRF width | Method-of-moments |
| CDI | support mask | Reprojection error |
| Integral | baseline | Reprojection error |
| FPM | pupil radius | Gradient descent |

### 3) Multi-agent system

PWM includes a **multi-agent orchestration system** with 17 agent modules:

| Agent | Role |
|-------|------|
| **PlanAgent** | Orchestrator (registry-ID-only LLM output) |
| **PhotonAgent** | Variance-dominance noise model + LLM narrative |
| **MismatchAgent** | Deterministic mismatch analysis + LLM prior selection |
| **RecoverabilityAgent** | Calibration table lookup with interpolation + confidence |
| **AnalysisAgent** | Bottleneck scoring + suggestions |
| **Negotiator** | Agent veto/negotiation logic |
| **ContinuityChecker** | Physical continuity validation |
| **PreFlight** | Pre-flight report + CLI modes (--auto-proceed, --force) |
| **PhysicsStageVisualizer** | Deterministic before/after element visualization |
| **UPWMI** | Unified scoring, caching, budget control |
| **SelfImprovement** | Design alternative advisor loop |
| **WhatIfPrecomputer** | Sensitivity curves for parameter sweeps |
| **AssetManager** | Illustration stage + licensing |
| **HybridModalityManager** | Hybrid modality fusion support |

**Key design principle:** All agents run **deterministically without LLM**. LLM is an optional enhancement that returns only **registry IDs** (mechanically enforced).

### 4) RunBundle export + Viewer

Every run exports a **RunBundle** with full reproducibility:

```
run_{spec_id}_{uuid}/
  artifacts/
    x_hat.npy             # Reconstructed signal
    y.npy                 # Measurements
    x_true.npy            # Ground truth (if available)
    metrics.json          # PSNR, SSIM, runtime
    images/               # PNG visualizations
  internal_state/
    diagnosis.json        # Diagnosis result
    recon_info.json       # Reconstruction metadata
  agents/                 # Agent report snapshots
  logs/                   # Run logs
```

The **Streamlit viewer** (`pwm view`) provides:
- Split-view: ground truth vs reconstruction
- Metrics dashboard over solver portfolio
- Residual diagnostics and artifact analysis
- Interactive report with recommended actions

---

## The ExperimentSpec model

PWM organizes the world into:

1. **PhysicsState** *(required)* -- forward operator family
2. **BudgetState** -- dose, sampling rate, #frames/views
3. **CalibrationState (theta)** -- alignment/PSF/dispersion/gain drift/timing jitter
4. **EnvironmentState** -- background, scattering/attenuation, autofluorescence
5. **SampleState** -- motion/drift, blinking/kinetics, dynamics
6. **SensorState** -- saturation, quantization, read noise, FPN, nonlinearity
7. **ComputeState** *(optional)* -- runtime/memory/streaming constraints
8. **TaskState** *(optional)* -- recon vs calibration vs DOE vs QC report

This structure makes it possible to (a) simulate realistic data, (b) diagnose failure modes, and (c) recommend concrete improvements.

See: `docs/spec_v0.2.1.md`.

---

## Operator correction mode: measured y + A -> fit/correct operator -> reconstruct

This mode is for real experiments where the forward model is imperfect. It implements the core ISA loop:

1. **Infer** the effective operator $\hat{H}$ from measurements
2. **Diagnose** the dominant Triad gate (sampling / noise / operator mismatch)
3. **Correct** via minimal feasible intervention
4. **Verify** via re-projection consistency and physical invariants
5. **Export** RunBundle with full decision trail

**4-Scenario Evaluation Protocol** (used for all validated modalities):

| Scenario | Measurement | Reconstruction Operator | Purpose |
|----------|-------------|------------------------|---------|
| I (Ideal) | True H | True H | Oracle upper bound |
| II (Assumed) | True H | Nominal H_nom | Mismatch impact baseline |
| III (Corrected) | True H | Calibrated H_hat | Calibration benefit |
| IV (Oracle Mask) | True H | Partial oracle | Partial upper bound |

**Key metric:** Recovery ratio $\rho$ = (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II)

### Supported modalities for calibration (16 tested)

| Modality | Calibration Parameters | Typical Improvement |
|----------|------------------------|--------------------|
| Widefield | background/gain | +4.5 dB |
| CASSI | dx, dy, theta, phi_d | +4.8 dB (UPWMI beam search) |
| CT | center of rotation | +13.0 dB |
| MRI | coil sensitivities | +48.3 dB |
| CACTI | mask timing | +12.6 dB |
| SPC | gain/bias | +24 dB |
| Lensless | PSF shift | +10.2 dB |
| Ptychography | position offset | +7.1 dB |
| OCT | dispersion coefficients | +50.5 dB |
| Light Field | disparity | +6.9 dB |
| DOT | scattering coefficient | +0.8 dB |
| Photoacoustic | speed of sound | +9.9 dB |
| FLIM | IRF width | +15.4 dB |
| CDI | support mask | +1.4 dB |
| Integral | PSF sigma | +21.0 dB |
| FPM | pupil radius | +8.9 dB |
| Matrix | gain/bias | +1.7 dB |

**Note:** Improvement depends on mismatch severity and calibration search quality. Results above use grid search with benchmark-quality reconstruction algorithms. SPC and CASSI tests require extended runtime (~50 min each).

### CLI Examples

```bash
# CASSI hyperspectral calibration
pwm calib-recon \
  --y data/cassi_measurement.npy \
  --operator cassi \
  --out-dir runs/cassi_calib

# Generic matrix operator calibration
pwm calib-recon \
  --y data/measured_y.npy \
  --operator matrix \
  --out-dir runs/matrix_calib

# OCT dispersion calibration
pwm calib-recon \
  --y data/oct_scan.npy \
  --operator oct \
  --out-dir runs/oct_calib

# View results
pwm view runs/cassi_calib
```

### Python API

```python
from pwm_core.api import endpoints
from pwm_core.api.types import (
    ExperimentSpec, ExperimentInput, ExperimentStates,
    InputMode, PhysicsState, TaskState, TaskKind,
    MismatchSpec, MismatchFitOperator
)

# Build spec for CASSI calibration + reconstruction
spec = ExperimentSpec(
    id="cassi_calib_001",
    input=ExperimentInput(
        mode=InputMode.measured,
        y_source="data/cassi_measurement.npy",
    ),
    states=ExperimentStates(
        physics=PhysicsState(modality="cassi"),
        task=TaskState(kind=TaskKind.calibrate_and_reconstruct),
    ),
    mismatch=MismatchSpec(
        enabled=True,
        fit_operator=MismatchFitOperator(
            enabled=True,
            search={"method": "random", "max_evals": 50},
        ),
    ),
)

# Run calibration + reconstruction
result = endpoints.calibrate_recon(spec, out_dir="runs/")

print(f"Best-fit params: {result.calib.theta_best}")
print(f"Recon solver: {result.recon[0].solver_id}")
```

### Testing Operator Correction

```bash
# Run all 16 calibration tests via pytest
cd packages/pwm_core
python -m pytest benchmarks/test_operator_correction.py -v

# Run specific modality via script
python benchmarks/test_operator_correction.py --modality ct
python benchmarks/test_operator_correction.py --modality cassi
python benchmarks/test_operator_correction.py --modality oct
python benchmarks/test_operator_correction.py --modality flim
```

### What gets saved

```
run_{spec_id}_{uuid}/
  artifacts/
    x_hat.npy               # Final reconstruction
    y.npy                   # Measurements
    metrics.json            # PSNR, SSIM
    images/                 # PNG visualizations
  internal_state/
    diagnosis.json          # Diagnosis + calibration results
    recon_info.json         # Reconstruction metadata
  agents/                   # Agent report snapshots
  logs/                     # Run logs
```

See: `docs/operator_mode.md`.

---

## DeepInv integration

PWM supports solver portfolios, including:
- **DeepInv** PnP / unrolled methods / diffusion adapters (optional)
- classical solvers (TV-FISTA, ADMM-TV, primal-dual, RL)

`pwm_core/recon/deepinv_adapter.py` provides a stub adapter that passes through existing DeepInv physics objects; custom PWM-to-DeepInv operator wrapping is not yet implemented.

---

## Modality Coverage

PWM's registry contains **64 imaging modalities** spanning microscopy, medical imaging, coherent/computational optics, electron microscopy, remote sensing, and more.

- **64** modalities in `contrib/modalities.yaml` with forward-model templates and solver portfolios
- **26** modalities with quantitative PSNR benchmark results (table below)
- **16** modalities with operator-correction calibration tests (see [Operator correction mode](#operator-correction-mode-measured-y--a---fitcorrect-operator---reconstruct))

## Modality Catalog (64)

<details>
<summary>All 64 modalities grouped by execution tier (click to expand)</summary>

*Catalog generated from `contrib/modalities.yaml`. Tier groupings follow `docs/PLAN_v4_report_contract.md` §5.2.*

**Tier 1 — Core compressive (5)**
`spc` · `cassi` · `cacti` · `ct` · `mri`

**Tier 2 — Microscopy fundamentals (8)**
`widefield` · `widefield_lowdose` · `confocal_livecell` · `confocal_3d` · `sim` · `lensless` · `lightsheet` · `flim`

**Tier 3 — Coherent imaging (5)**
`ptychography` · `holography` · `phase_retrieval` · `fpm` · `oct`

**Tier 4 — Medical imaging (10)**
`xray_radiography` · `ultrasound` · `photoacoustic` · `dot` · `pet` · `spect` · `fluoroscopy` · `mammography` · `dexa` · `cbct`

**Tier 5 — Neural rendering + computational (6)**
`nerf` · `gaussian_splatting` · `matrix` · `panorama` · `light_field` · `integral`

**Tier 6 — Electron microscopy (7)**
`sem` · `tem` · `stem` · `electron_tomography` · `electron_diffraction` · `ebsd` · `eels`

**Tier 7 — Advanced medical (6)**
`angiography` · `doppler_ultrasound` · `elastography` · `fmri` · `mrs` · `diffusion_mri`

**Tier 8 — Advanced microscopy (5)**
`two_photon` · `sted` · `palm_storm` · `tirf` · `polarization`

**Tier 9 — Clinical optics + depth (6)**
`endoscopy` · `fundus` · `octa` · `tof_camera` · `lidar` · `structured_light`

**Tier 10 — Remote sensing + exotic (6)**
`sar` · `sonar` · `electron_holography` · `neutron_tomo` · `proton_radiography` · `muon_tomo`

See `docs/PLAN_v4_report_contract.md` for full per-modality reports.

</details>

## Benchmark Results (26 modalities with PSNR table)

| # | Modality | Best Solver | PSNR (dB) | Ref (dB) | Status |
|---|----------|-------------|-----------|----------|--------|
| 1 | Widefield | Richardson-Lucy | 27.31 | 28.0 | Pass |
| 2 | Widefield Low-Dose | BM3D+RL | 32.88 | 30.0 | Pass |
| 3 | Confocal Live-Cell | CARE | 30.04 | 26.0 | Pass |
| 4 | Confocal 3D | CARE 3D | 39.17 | 26.0 | Pass |
| 5 | SIM | Wiener | 27.48 | 28.0 | Pass |
| 6 | CASSI | HDNet | 35.06 | 34.71 | Pass |
| 7 | SPC (25%) | PnP-FISTA | 32.17 | 32.0 | Pass |
| 8 | CACTI | EfficientSCI | 36.28 | 26.5 | Pass |
| 9 | Lensless | FlatNet | 33.89 | 24.0 | Pass |
| 10 | Light-Sheet | Stripe Removal | 28.05 | 25.0 | Pass |
| 11 | CT | RED-CNN | 26.77 | 28.0 | Pass |
| 12 | MRI | PnP-ADMM | 44.97 | 34.2 | Pass |
| 13 | Ptychography | Neural | 59.41 | 35.0 | Pass |
| 14 | Holography | Angular Spectrum | 46.54 | 35.0 | Pass |
| 15 | NeRF | SIREN | 61.35 | 32.0 | Pass |
| 16 | 3D Gaussian Splatting | 2D Gaussian Opt | 30.89 | 30.0 | Pass |
| 17 | Matrix | FISTA-TV | 33.86 | 25.0 | Pass |
| 18 | Panorama Multifocal | Neural Fusion | 27.90 | 28.0 | Pass |
| 19 | Light Field | LFBM5D | 35.28 | 28.0 | Pass |
| 20 | Integral | DIBR | 28.14 | 27.0 | Pass |
| 21 | Phase Retrieval | HIO | 30.66 | 30.0 | Pass |
| 22 | FLIM | MLE Fit | 48.11 | 25.0 | Pass |
| 23 | Photoacoustic | Time Reversal | 50.54 | 32.0 | Pass |
| 24 | OCT | FFT Recon | 64.84 | 36.0 | Pass |
| 25 | FPM | Gradient Descent | 34.61 | 34.0 | Pass |
| 26 | DOT | Born/Tikhonov | 32.06 | 25.0 | Pass |

### CASSI Real-Data Benchmark (10 scenes, 4 solvers)

TSA simulation benchmark: 10 hyperspectral scenes (256×256×28, step=2 dispersion), evaluated with GAP-TV (classical), HDNet, MST-S, MST-L (CVPR 2022 deep spectral transformers).

**PSNR (dB)**

| Scene | GAP-TV | HDNet | MST-S | MST-L |
|-------|--------|-------|-------|-------|
| scene01 | 15.41 | 35.17 | 34.78 | 35.43 |
| scene02 | 15.33 | 35.73 | 34.42 | 35.90 |
| scene03 | 14.42 | 36.13 | 33.82 | 34.91 |
| scene04 | 15.86 | 42.78 | 42.10 | 42.23 |
| scene05 | 14.53 | 32.72 | 31.79 | 32.51 |
| scene06 | 14.77 | 34.53 | 33.74 | 34.75 |
| scene07 | 14.41 | 33.70 | 32.38 | 33.44 |
| scene08 | 15.07 | 32.49 | 31.88 | 32.91 |
| scene09 | 14.42 | 34.93 | 34.11 | 35.04 |
| scene10 | 15.02 | 32.39 | 31.88 | 32.75 |
| **Average** | **14.92** | **35.06** | **34.09** | **34.99** |

**SSIM**

| Scene | GAP-TV | HDNet | MST-S | MST-L |
|-------|--------|-------|-------|-------|
| scene01 | 0.1917 | 0.9358 | 0.9295 | 0.9419 |
| scene02 | 0.1844 | 0.9421 | 0.9233 | 0.9452 |
| scene03 | 0.1711 | 0.9421 | 0.9271 | 0.9480 |
| scene04 | 0.2389 | 0.9764 | 0.9692 | 0.9750 |
| scene05 | 0.1793 | 0.9457 | 0.9271 | 0.9448 |
| scene06 | 0.2131 | 0.9542 | 0.9407 | 0.9541 |
| scene07 | 0.1685 | 0.9232 | 0.9056 | 0.9222 |
| scene08 | 0.2224 | 0.9467 | 0.9362 | 0.9511 |
| scene09 | 0.1658 | 0.9409 | 0.9272 | 0.9375 |
| scene10 | 0.2107 | 0.9441 | 0.9287 | 0.9460 |
| **Average** | **0.1946** | **0.9451** | **0.9315** | **0.9466** |

GAP-TV's low 14.92 dB reflects the extreme 28:1 spectral compression ratio. HDNet leads at 35.06 dB, with MST-L comparable at 34.99 dB. W2 mask-shift correction recovers exact 2px injected shift (NLL decrease 100.0%).

```bash
PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cassi_benchmark.py
```

### CACTI Real-Data Benchmark (6 scenes, 4 solvers)

Grayscale SCI video benchmark: 6 scenes (256×256, 8:1 temporal compression), evaluated with GAP-TV (classical), PnP-FFDNet (plug-and-play), ELP-Unfolding (ECCV 2022), EfficientSCI (CVPR 2023).

**PSNR (dB)**

| Scene | GAP-TV | PnP-FFDNet | ELP-Unfolding | EfficientSCI |
|-------|--------|------------|---------------|-------------|
| kobe32 | 24.00 | 30.33 | 34.08 | 35.76 |
| crash32 | 25.40 | 24.69 | 29.39 | 31.12 |
| aerial32 | 26.13 | 24.36 | 30.54 | 31.50 |
| traffic48 | 21.06 | 23.88 | 31.34 | 32.29 |
| runner40 | 28.70 | 32.97 | 38.17 | 41.89 |
| drop40 | 34.42 | 39.91 | 40.09 | 45.10 |
| **Average** | **26.62** | **29.36** | **33.94** | **36.28** |

**SSIM**

| Scene | GAP-TV | PnP-FFDNet | ELP-Unfolding | EfficientSCI |
|-------|--------|------------|---------------|-------------|
| kobe32 | 0.7461 | 0.9253 | 0.9644 | 0.9758 |
| crash32 | 0.8649 | 0.8332 | 0.9537 | 0.9726 |
| aerial32 | 0.8510 | 0.8200 | 0.9398 | 0.9542 |
| traffic48 | 0.7063 | 0.8299 | 0.9623 | 0.9691 |
| runner40 | 0.8908 | 0.9357 | 0.9744 | 0.9868 |
| drop40 | 0.9654 | 0.9863 | 0.9798 | 0.9950 |
| **Average** | **0.8374** | **0.8884** | **0.9624** | **0.9756** |

EfficientSCI leads at 36.28 dB average, followed by ELP-Unfolding at 33.94 dB. W2 mask-shift correction recovers exact 2px injected shift (+8.24 dB PSNR gain).

```bash
PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_benchmark.py
```

### Running the Benchmarks

```bash
cd packages/pwm_core

# Run ALL 64 modalities (~28 min)
python benchmarks/run_all.py --all

# Run core modalities only (faster)
python benchmarks/run_all.py --core

# Run a specific modality
python benchmarks/run_all.py --modality oct
python benchmarks/run_all.py --modality flim
python benchmarks/run_all.py --modality photoacoustic
python benchmarks/run_all.py --modality fpm
```

### Test Suite

```bash
cd packages/pwm_core

# Unit tests (3743 core + 32 canonical + 210 clinical = 3985 tests)
python -m pytest tests/ -v

# Operator correction tests (16 tests, ~63 min)
python -m pytest benchmarks/test_operator_correction.py -v
```

### Benchmark Output

Results are saved to `packages/pwm_core/benchmarks/results/`:
- `benchmark_results.json` - Raw metrics for all modalities
- `benchmark_report.md` - Formatted report with detailed analysis

### Dataset Preparation

Most benchmarks use **synthetic data by default** (no download required). For real datasets:

```bash
# CASSI uses TSA_simu_data (10 scenes, 256x256x28)
# Symlinked at packages/pwm_core/datasets/TSA_simu_data
```

For large datasets (LoDoPaB-CT, fastMRI, KAIST), see `docs/plan.md` for details.

---

## CT QC Copilot

PWM extends from research computational imaging into **clinical diagnostic medical physics**. The CT QC Copilot is a metric-first quality assurance module for CT scanners, implementing ACR CT accreditation standards (ACR CT 464 phantom), AAPM TG-233 performance metrics, and Western Electric SPC rules for drift detection.

**Design philosophy:** Autopilot for QC, Digital Twin for troubleshooting. PWM provides the targeting system (which scanner needs attention), outcome contracts (pass/fail against published thresholds with full evidence), and decision logs (immutable QC records). The qualified medical physicist provides clinical judgment, sign-off, and regulatory accountability.

### 12 ACR CT Phantom Metrics

| # | Metric | ACR Criterion | Method |
|---|--------|---------------|--------|
| 1 | CT Number (Water) | 0 +/- 5 HU | Central circular ROI mean |
| 2 | CT Number (Bone) | 850--970 HU | Insert ROI with rotation correction |
| 3 | CT Number (Polyethylene) | -107 to -84 HU | Insert ROI with rotation correction |
| 4 | CT Number (Acrylic) | 110--135 HU | Insert ROI with rotation correction |
| 5 | CT Number (Air) | -1005 to -970 HU | Insert ROI with rotation correction |
| 6 | Geometric Accuracy | +/- 2 mm of 200 mm | Bounding box extent with fence-post correction |
| 7 | Slice Thickness | +/- 1.5 mm of nominal | Wire-ramp FWHM with pixel spacing |
| 8 | Uniformity | < 5 HU center-to-edge | 4 peripheral ROIs (12/3/6/9 o'clock) |
| 9 | Noise (Std Dev) | Site-specific | Central water ROI sample std |
| 10 | Low-Contrast Detectability | >= 4 targets visible | CNR-based detection at 5 target sizes |
| 11 | Artifact Evaluation | Score 0--3 | Radial profile peak-to-valley analysis |
| 12 | Spatial Resolution | Site-specific lp/cm | Bar pattern MTF analysis |

### Clinical Architecture

```
DICOM Files -> DICOMIngester (PHI-safe) -> CTScanBundle
    -> compute_all_metrics() -> QAMetricsReport
    -> ThresholdResolver (4-layer) -> ThresholdResults
    -> DiagnosisEngine (scored root-cause) -> DiagnosisReport
    -> DriftDetector (SPC/Western Electric) -> DriftReport
    -> BaselineManager (versioned, SHA-256 signed) -> BaselineComparison
    -> ReportGenerator -> physicist_report.json + PDF + evidence/
```

**Key components:**

| Component | Description |
|-----------|-------------|
| **DICOMIngester** | PHI-safe DICOM loading with phantom-only validation, CasePack-driven series selection, and canonical resampling |
| **QA Metrics** | 12 ACR CT phantom metrics with automatic phantom center detection and rotation correction |
| **ThresholdResolver** | 4-layer cascade: standard_default -> scanner_model -> protocol -> site_override |
| **DiagnosisEngine** | Scored root-cause diagnosis using mismatch library YAML with 6 artifact features |
| **DriftDetector** | 5 Western Electric SPC rules on Shewhart control charts with baseline-anchored limits |
| **BaselineManager** | Immutable, SHA-256 signed, version-chained CommissioningBundles |
| **CTOperatorGraph** | Tier 1/2 CT forward model for troubleshoot-mode Triad diagnosis |
| **ReportGenerator** | Triple-output: JSON (tamper-evident SHA-256), PDF, and evidence directory |

**CasePack extensibility:** Each phantom/test combination is a versioned CasePack YAML containing ROI definitions, metric sets, thresholds, and report templates. Adding PET/CT or SPECT requires a new CasePack, not new code. Scaffold directories for `pet_ct/` and `spect_ct/` are in place.

### Clinical Quickstart

```python
from pwm_core.clinical.ct.dicom_ingester import DICOMIngester
from pwm_core.clinical.ct.qa_metrics import compute_all_metrics
from pwm_core.clinical.common.threshold_resolver import ThresholdResolver
from pwm_core.clinical.ct.report_generator import ReportGenerator

# 1. Ingest DICOM (PHI-safe, phantom-only)
ingester = DICOMIngester(phi_strict=True)
scan_bundle = ingester.ingest(Path("acr_phantom_scan/"), casepack=casepack)

# 2. Compute all 12 metrics
metrics_report = compute_all_metrics(scan_bundle, casepack)

# 3. Resolve thresholds (4-layer cascade)
resolver = ThresholdResolver(threshold_config)
threshold_results = resolver.evaluate(metrics_report)

# 4. Generate audit-grade report
gen = ReportGenerator()
gen.generate(config, metrics_report, threshold_results)
# -> physicist_report.json + physicist_report.pdf + evidence/
```

```bash
# Run clinical test suite (210 tests)
cd packages/pwm_core
python -m pytest tests/clinical/ -v
```

---

## Repository layout

```text
pwm/
  README.md
  LICENSE
  rails/                   # SolveEverything 10-gear implementation map
    README.md              # Trail overview + status table
    gear01_targeting_system.md .. gear10_literacy.md
    maturity_levels.md     # L0-L5 maturation framework
    industrial_stack.md    # 9-layer stack reference
  docs/
    purpose.md            # Stage 1 purpose: Imaging System Autonomy
    targeting_system.md   # LIP-Arena: PWM's built-in evaluation harness
    plan.md               # Master plan v3 (hardened, fully implemented)
    spec_v0.2.1.md
    runbundle_format.md
    operator_mode.md
  examples/
    prompt_to_casepack.py
    yA_calibrate_recon_cassi.py
    yA_calibrate_recon_generic.py
  pyproject.toml

  packages/
    pwm_core/              # public core library (no AI_Scientist deps)
      pwm_core/
        graph/             # OperatorGraph IR + canonical framework
          ir_types.py      # CanonicalPrimitive, PhysicsStageFamily, DetectFamily enums
          primitives.py    # 99 primitives + CANONICAL_REGISTRY (10 canonical types)
          canonical_decompositions.py  # 31-modality canonical DAG registry
          fidelity.py      # Operator-norm, pointwise, mean fidelity metrics
          extension_protocol.py  # 5-step formal extension process
          graph_operator.py  # to_canonical(), canonical_dag_string()
          compiler.py      # Compiles specs to graphs with canonical tags
        agents/            # 17 agent modules + contracts + registry
        physics/           # 64 modality operators
        analysis/          # Metrics, bottleneck, uncertainty
        core/              # Runner, RunBundle, simulator
        api/               # Pydantic types, endpoints
        clinical/          # Clinical medical physics QA modules
          ct/              # CT QC Copilot (12 metrics, diagnosis, drift, reports)
          common/          # Shared: threshold resolver, PHI filter, scanner registry
          casepacks/       # CasePack loader + YAML configs (acr_ct.yaml)
          pet_ct/          # PET/CT QC (scaffold)
          spect_ct/        # SPECT/CT QC (scaffold)
      contrib/
        modalities.yaml    # 64-modality source of truth
        mismatch_db.yaml   # Mismatch parameters per modality
        photon_db.yaml     # Photon models
        compression_db.yaml # Recoverability calibration tables
        metrics_db.yaml    # Per-modality metric sets
        solver_registry.yaml # 43+ solvers
      benchmarks/
        run_all.py         # 64-modality benchmark suite
        test_operator_correction.py  # 16 calibration tests
      tests/               # 3985 unit tests (incl. 32 canonical + 210 clinical)
    pwm_AI_Scientist/      # AI_Scientist adapter (thin)
```

---

## YAML Registries

PWM uses **6 YAML registries** as the source of truth for all modalities, solvers, and parameters:

| Registry | Entries | Purpose |
|----------|---------|---------|
| `modalities.yaml` | 64 modalities | Forward model families + upload templates |
| `mismatch_db.yaml` | Per-modality | Mismatch parameters and ranges |
| `photon_db.yaml` | Per-modality | Photon/noise models (model_id, not formulas) |
| `compression_db.yaml` | Calibration tables | Recoverability with provenance fields |
| `metrics_db.yaml` | Per-modality | Metric sets (phase_rmse, SAM, CNR, etc.) |
| `solver_registry.yaml` | 43+ solvers | Solver parameters and tier classification |

All registries are validated by Pydantic schemas with cross-reference integrity tests.

---

## Embedding into AI_Scientist

PWM exposes stable endpoints:
- `compile(prompt)` -> draft spec
- `resolve_validate(spec)` -> safe spec + auto-repair
- `simulate(spec)` / `reconstruct(spec, y)` / `analyze(...)`
- `fit_operator(...)` / `calibrate_recon(...)`
- `export(runbundle)` / `view(runbundle)`

Use `packages/pwm_AI_Scientist/` as the thin adapter layer.

> You do **not** need AG2/LangGraph to run PWM.
> If you want autonomy loops later (planner<->reviewer, tool-using multi-step agents), implement them in `pwm_AI_Scientist` without changing `pwm_core`.

---

## Community & Contributing

**New here?** Start with the [Colab Quickstart](https://colab.research.google.com/github/integritynoble/Physics_World_Model/blob/master/examples/PWM_Quickstart.ipynb) (5-minute demo, no install). Then check out [Good First Issues](https://github.com/integritynoble/Physics_World_Model/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for entry points, or ask questions in [GitHub Discussions](https://github.com/integritynoble/Physics_World_Model/discussions).

PWM is intended to be extended by the community. All algorithms are open source under the [PWM Noncommercial Share-Alike License v1.0](LICENSE) (free for academic research, teaching, and personal projects; commercial use requires a separate license). No algorithm is paywalled. See [`community/OPEN_CORE_BOUNDARY.md`](community/OPEN_CORE_BOUNDARY.md) for the full open-core policy.

### 4 Levels of Contribution

| Level | What You Add | Difficulty | Time to First Result | Merge Lane |
|-------|-------------|------------|---------------------|------------|
| **Solver** | A new `ReconSolver` for an existing modality | Easy | ~1 day | Fast (48h) |
| **Calibrator** | A new calibration method for operator correction | Medium | ~3 days | Fast (48h) |
| **Modality** | A full modality (operator + CasePack + solver + tests) | Hard | ~1 week | Review (7d) |
| **Primitive** | A new OperatorGraph node type | Expert | ~2 weeks | Governance (90d RFC) |

#### Level 1: New Solver (Easiest)

**Who**: Any ML researcher, PhD student, or imaging lab. **Your solver never knows what modality it's solving** -- write once, compete on all 64+ modalities.

```bash
# 1. Scaffold
pwm scaffold solver my_solver

# 2. Implement (edit contrib/solvers/my_solver/solver.py)
#    Your function: run_my_solver(y, physics, cfg) -> (x_hat, info)

# 3. Test locally
python contrib/solvers/my_solver/test_local.py

# 4. Run sandbox evaluation
pwm evaluate --sandbox --modality widefield --solver my_solver

# 5. Validate for PR
pwm contrib check my_solver

# 6. Submit PR (auto-labeled: fast-lane, auto-merge in 48h if CI passes)
```

**Paper**: "Our method achieves $\rho$=0.85 across 20 modalities on LIP-Arena"

#### Level 2: New Calibrator (Medium)

**Who**: Self-calibration, blind deconvolution, operator learning researchers.

```bash
pwm scaffold calibrator my_calibrator
# Implement: calibrate_my_method(y, H_nom, budget) -> (H_hat, info)
# H_nom exposes: get_theta(), set_theta(), forward(), adjoint()
```

**Paper**: "Our blind calibrator reduces oracle gap from 12 dB to 2 dB"

#### Level 3: New Modality (Medium-Hard)

**Who**: Domain experts with a modality PWM doesn't cover.

```bash
pwm scaffold modality my_modality
# Fill in: graph.yaml, mismatch.yaml, photon.yaml, metrics.yaml, meta.yaml
pwm evaluate --sandbox --modality my_modality
```

Requires entries in all 5 YAML registries. Full checklist:
1) Create a new operator in `pwm_core/physics/<modality>/`
2) Add YAML entries to all 6 registries (`modalities.yaml`, `mismatch_db.yaml`, `photon_db.yaml`, `compression_db.yaml`, `metrics_db.yaml`, `solver_registry.yaml`)
3) Add a benchmark in `benchmarks/run_all.py`
4) Add an operator correction test in `benchmarks/test_operator_correction.py`
5) Run `python -m pytest tests/test_registry_integrity.py` to verify no orphan keys

Templates:
- `pwm_core/contrib/templates/new_operator_template.py`
- `pwm_core/contrib/templates/new_calibrator_template.py`

**Paper**: "We formalize 4D-STEM as an OperatorGraph and benchmark 10 solvers"

#### Level 4: New Primitive (Hardest -- RFC + Extension Protocol)

**Who**: Physics experts willing to implement a new atomic operator. Must follow the formal [Extension Protocol](packages/pwm_core/pwm_core/graph/extension_protocol.py) from the FPB Theorem:

1) Demonstrate representation gap: no DAG over existing 10 canonical primitives achieves $\varepsilon_{\text{tier2}} \leq \varepsilon$
2) Open RFC issue with physics justification + validated forward/adjoint
3) Show error reduction below $\varepsilon$ with the new primitive
4) Demonstrate need by ≥2 modalities
5) Pass backward-compatible closure re-test (all existing decompositions preserved)
6) Community + steward review
7) Merge into `CANONICAL_REGISTRY` and update `canonical_decompositions.py`

**Paper**: "Our full-wave primitive improves fidelity by 3 dB across 5 modalities"

### Compete Without a PR

Don't want to fork or submit code? Just compete on the leaderboard:

```bash
# Run locally, submit results only
pwm evaluate --modality cassi --solver my_solver --output ./results
pwm submit ./results/runbundle.zip
# Score appears on leaderboard. No fork, no PR needed.
```

### Three-Speed Merge

| Lane | Scope | Timeline | Human Veto? |
|------|-------|----------|-------------|
| **Fast** | Solvers, calibrators, config tweaks | Auto-merge within **48 hours** of CI pass | **Not allowed** -- solvers are trains, blocking a solver is blocking science |
| **Review** | Modalities, metrics, track tweaks | **7 days**, 2 reviewers (1 maintainer + 1 domain expert) | Must provide written rationale |
| **Governance** | Rail changes (scoring, protocol, frozen specs) | **90-day RFC**, unanimous steward vote | Required -- major version bump |

See [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) for full merge authority rules, steward board, and dispute resolution.

### Weekly Challenges

Every week a new reconstruction challenge is posted in [`community/challenges/`](community/challenges/). Reconstruct from simulated measurements, submit a RunBundle, compete on the leaderboard.

```bash
# 1. Check the current challenge
ls community/challenges/

# 2. Read the challenge description
cat community/challenges/2026-W10/challenge.md

# 3. Generate the challenge dataset
cd community/challenges/2026-W10
python generate_data.py --output ./data

# 4. Write your reconstruction (produces x_hat.npy)
# ... your code here ...

# 5. Package as a RunBundle and validate
python community/validate.py my_submission.zip

# 6. Check the leaderboard
python community/leaderboard.py --week 2026-W10
```

See [`community/CONTRIBUTING_CHALLENGE.md`](community/CONTRIBUTING_CHALLENGE.md) for full participation details including RunBundle format, scoring rules, and tips.

### Calibration Sprint Service

Need expert help calibrating your imaging system? The [Calibration Sprint](community/calibration_sprint/) is a focused 1-2 week engagement: characterize mismatch, calibrate operator parameters, validate with bootstrap confidence intervals, and deliver a calibrated operator + RunBundle. All tools are open source -- the Sprint provides expert guidance and GPU compute. See [`community/calibration_sprint/README.md`](community/calibration_sprint/README.md).

### Submit a better method
1) Implement the `ReconSolver` protocol (accept `y`, `H`, return `x_hat` with uncertainty)
2) Register your solver in `contrib/solver_registry.yaml`
3) Run `pwm evaluate --method my_solver --modality <target>` and beat the current default
4) Open a PR with your method code and RunBundle artifacts demonstrating the improvement

When your method scores higher on the harness, it becomes PWM's new shipped default.

### Add a dataset adapter
- Implement loader in `pwm_core/io/datasets.py` and format handler in `io/formats.py`
- Add an example under `examples/`
- Prefer reference-mode support for large datasets

See also: [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contribution guide.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| **Architecture** | |
| [`rails/README.md`](rails/README.md) | SolveEverything 10-gear framework + status table |
| [`docs/purpose.md`](docs/purpose.md) | Imaging System Autonomy (ISA) discipline |
| [`docs/spec_v0.2.1.md`](docs/spec_v0.2.1.md) | ExperimentSpec data model (8 state groups) |
| **Theoretical Foundations** | |
| [`papers/pwm_flagship/main.tex`](papers/pwm_flagship/main.tex) | Flagship paper: FPB Theorem + Triad Decomposition |
| [`pwm_core/graph/canonical_decompositions.py`](packages/pwm_core/pwm_core/graph/canonical_decompositions.py) | 31-modality canonical DAG registry |
| [`pwm_core/graph/extension_protocol.py`](packages/pwm_core/pwm_core/graph/extension_protocol.py) | 5-step extension protocol for new primitives |
| **Specifications** | |
| [`docs/targeting_system.md`](docs/targeting_system.md) | LIP-Arena: 4-scenario protocol, scoring, tracks |
| [`docs/operator_mode.md`](docs/operator_mode.md) | Operator correction pipeline + 16 calibration modalities |
| [`docs/quickstart/README.md`](docs/quickstart/README.md) | Getting started guide |
| **Modalities & Data** | |
| [`packages/pwm_core/contrib/modalities.yaml`](packages/pwm_core/contrib/modalities.yaml) | 64-modality registry (source of truth) |
| [`packages/pwm_core/contrib/solver_registry.yaml`](packages/pwm_core/contrib/solver_registry.yaml) | 43+ solver registry |
| [`docs/benchmark_results_26_modalities.md`](docs/benchmark_results_26_modalities.md) | Benchmark results (26 modalities with PSNR) |
| **Governance** | |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guide (modalities, solvers, datasets) |
| [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md) | Three-speed merge, steward board, dispute resolution |
| [`community/CONTRIBUTING_CHALLENGE.md`](community/CONTRIBUTING_CHALLENGE.md) | Weekly challenge participation guide |
| **Clinical Medical Physics** | |
| [`packages/pwm_core/pwm_core/clinical/`](packages/pwm_core/pwm_core/clinical/) | CT QC Copilot source modules |
| [`packages/pwm_core/tests/clinical/`](packages/pwm_core/tests/clinical/) | 210 clinical tests |
| [`packages/pwm_core/pwm_core/clinical/casepacks/acr_ct.yaml`](packages/pwm_core/pwm_core/clinical/casepacks/acr_ct.yaml) | ACR CT phantom CasePack |

---

## License

See `LICENSE`.

---

## Citation

If you use PWM in academic work, please cite the flagship paper and link to this repository:

```bibtex
@article{yang2026pwm,
  title   = {Ten Primitives and Three Gates: The Universal Structure
             of Computational Imaging},
  author  = {Yang, Chengshuai and Yuan, Xin},
  journal = {Under review},
  year    = {2026},
  note    = {Flagship paper for the Physics World Model (PWM) framework}
}
```
